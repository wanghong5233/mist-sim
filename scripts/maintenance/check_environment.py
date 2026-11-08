import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check package import and optional CUDA readiness; no training.")
    parser.add_argument("--cuda", action="store_true", help="Run one tiny CUDA tensor sum.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import mist_sim
    import numpy as np
    import torch

    package_path = Path(mist_sim.__file__).resolve()
    expected_path = Path(__file__).resolve().parents[2] / "src/mist_sim/__init__.py"
    if package_path != expected_path:
        raise RuntimeError(f"Wrong package imported: {package_path}; expected {expected_path}")
    numpy_sum = float(np.ones(2).sum())
    if numpy_sum != 2.0:
        raise RuntimeError(f"Unexpected NumPy smoke result: {numpy_sum}")
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "package_path": str(package_path),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mist-sim", "torch", "numpy", "scipy", "setuptools", "pip")
        },
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_smoke_requested": args.cuda,
        "numpy_sum": numpy_sum,
    }
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        properties = torch.cuda.get_device_properties(0)
        result = torch.ones(2, device="cuda").sum().item()
        torch.cuda.synchronize()
        if result != 2.0:
            raise RuntimeError(f"Unexpected CUDA smoke result: {result}")
        report.update(gpu=properties.name, gpu_total_bytes=properties.total_memory, cuda_sum=result)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
