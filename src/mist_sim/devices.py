"""CPU simulation with a bounded accelerator lifetime for PPO updates."""
from contextlib import contextmanager
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
