"""Opt-in Windows process scheduling; no global power-plan changes."""
import ctypes
import os
import sys


class PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32),
                ("StateMask", ctypes.c_uint32)]


def configure_process(windows_performance=False):
    report = dict(pid=os.getpid(), requested=windows_performance, status="unchanged")
    if not windows_performance:
        return report
    if sys.platform != "win32":
        raise RuntimeError("runtime.windows_performance requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetPriorityClass.restype = ctypes.c_int
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel.GetPriorityClass.restype = ctypes.c_uint32
    for name in ("SetProcessInformation", "GetProcessInformation"):
        function = getattr(kernel, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        function.restype = ctypes.c_int
    handle = kernel.GetCurrentProcess()
    # ABOVE_NORMAL_PRIORITY_CLASS, not High/Realtime. Each worker opts in too.
    if not kernel.SetPriorityClass(handle, 0x8000):
        raise ctypes.WinError(ctypes.get_last_error())
    # Explicit HighQoS: control EXECUTION_SPEED (bit 1), leave it off.
    requested = PowerThrottlingState(1, 1, 0)
    if not kernel.SetProcessInformation(handle, 4, ctypes.byref(requested), ctypes.sizeof(requested)):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = PowerThrottlingState(1, 0, 0)
    if not kernel.GetProcessInformation(handle, 4, ctypes.byref(observed), ctypes.sizeof(observed)):
        raise ctypes.WinError(ctypes.get_last_error())
    priority = kernel.GetPriorityClass(handle)
    if priority != 0x8000 or not observed.ControlMask & 1 or observed.StateMask & 1:
        raise RuntimeError("Windows did not retain requested AboveNormal/HighQoS process settings")
    report.update(status="applied", priority_class="AboveNormal", high_qos=True,
                  power_control_mask=observed.ControlMask, power_state_mask=observed.StateMask)
    return report
