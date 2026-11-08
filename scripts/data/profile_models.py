"""Prepare official assets, profile models serially, and preserve failures."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from mist_sim.profiling import environment_metadata, prepare_assets, profile_model, validate_spec
from mist_sim.storage import RunStore, write_json, digest_file


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", choices=["resnet50", "resnet152", "vit_b_32"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config or root / "configs/profiling.json"
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    validate_spec(spec)
    if args.worker:
        assets = json.loads((args.run / "assets.json").read_text(encoding="utf-8"))
        profile_model(root, args.worker, spec, assets, args.run / f"traces/{args.worker}.json")
        return
    store = RunStore(root, dict(spec, profiles={}), " ".join(sys.argv), spec["seed"])
    store.manifest.update(device=spec["devices"], budget={
        "kind": "offline inference profiling; no training",
        "per_model_timeout_seconds": spec["per_model_timeout_seconds"],
        "total_measurement_limit_seconds": spec["total_measurement_limit_seconds"],
        "asset_preparation_included": False})
    store.write_json("manifest.json", store.manifest)
    try:
        started = time.perf_counter()
        assets = prepare_assets(root, spec)
        store.write_json("assets.json", assets)
        store.write_json("environment.json", environment_metadata())
        store.write_json("asset_preparation.json", {"measured_wall_seconds": time.perf_counter() - started,
                                                    "included_in_measurement_budget": False})
        write_json(root / "data/raw/assets.json", assets)
        completed = []
        computation_started = time.perf_counter()
        if not args.prepare_only:
            for name in spec["models"]:
                remaining = spec["total_measurement_limit_seconds"] - (time.perf_counter() - computation_started)
                timeout = min(spec["per_model_timeout_seconds"], remaining)
                if timeout <= 0:
                    raise TimeoutError("Total measurement budget exhausted")
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                           "--worker", name, "--run", str(store.path)]
                with (store.path / f"{name}_stdout.log").open("w", encoding="utf-8") as stdout, (store.path / f"{name}_stderr.log").open("w", encoding="utf-8") as stderr:
                    process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, timeout=timeout)
                if process.returncode:
                    raise RuntimeError(f"{name} failed: inspect {store.path / (name + '_stderr.log')}")
                completed.append(name)
                store.write_json("progress.json", {"completed": completed})
                print(f"COMPLETED {name}: {store.path}", flush=True)
        store.write_json("summary.json", {"models_completed": completed, "prepare_only": args.prepare_only,
                                          "measurement_wall_seconds": time.perf_counter() - computation_started})
        store.finish()
        write_json(root / "results/profiling_run.json", {"run_id": store.path.name, "prepare_only": args.prepare_only,
                                                        "manifest_sha256": digest_file(store.path / "manifest.json")})
        print(f"FINISHED {store.path}", flush=True)
    except BaseException as error:
        store.finish(error)
        raise


if __name__ == "__main__":
    main()
