"""CPU simulation with a bounded accelerator lifetime for PPO updates."""
from contextlib import contextmanager
import ctypes
import sys
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to("cpu" if key == "step" else device)


@contextmanager
def cuda_update_slot(device):
    """Share a Windows GPU between training processes, including after a crash.

    The kernel releases an abandoned mutex when its owner dies. CPU collection
    remains independent; only model transfer, update and cleanup hold the slot.
    """
    started = time.perf_counter()
    if device.type != "cuda" or sys.platform != "win32":
        yield 0.0
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.ReleaseMutex.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateMutexW(None, False, f"Local\\mist_sim_cuda_update_{device.index or 0}")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result not in (0, 0x80):  # acquired, or previous owner terminated
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = True
        yield time.perf_counter() - started
    finally:
        try:
            if acquired and not kernel.ReleaseMutex(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)


@contextmanager
def update_device(policy, optimizer, device):
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Unsupported update device")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA update requested but CUDA is unavailable")
    original = next(policy.parameters()).device
    if original.type != "cpu":
        raise ValueError("Sampling policy must reside on CPU before an update")
    report = dict(sampling_device="cpu", update_device=str(selected))
    with cuda_update_slot(selected) as wait_seconds:
        report["gpu_slot_wait_seconds"] = wait_seconds
        started = time.perf_counter()
        try:
            policy.to(selected)
            move_optimizer(optimizer, selected)
            synchronize(selected)
            report["model_to_device_seconds"] = time.perf_counter() - started
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
            yield report
        finally:
            synchronize(selected)
            if selected.type == "cuda":
                report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(selected) / 2**20
                report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(selected) / 2**20
            started = time.perf_counter()
            policy.to(original)
            move_optimizer(optimizer, original)
            synchronize(selected)
            report["return_to_cpu_seconds"] = time.perf_counter() - started
            if selected.type == "cuda":
                # Once per complete update, after all live policy/Adam tensors
                # return to CPU; never flush the allocator per minibatch.
                torch.cuda.empty_cache()
                report["idle_reserved_mib"] = torch.cuda.memory_reserved(selected) / 2**20
