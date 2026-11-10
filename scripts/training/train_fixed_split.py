"""Fixed-Split workflow: select cuts, freeze them, then independently train PPO.

Optional dependency runs reserve at most two simultaneous PPO training jobs.
The development selector is CPU-only and never consumes a GPU training slot.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mist_sim.config import default_project_root, for_method, load_config
from mist_sim.storage import digest_file
from scripts.evaluation.select_fixed_split import candidate_cuts
from scripts.training.overnight_queue import (
    cycles, keep_awake, process_alive, read_json, source_hash, terminal_command,
    utc_now, write_json,
)


def selected_training_config(root, selection_run, suite, seed, source_compatibility=None):
    directory = root / "logs/runs" / selection_run
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "completed" or manifest.get("exit_code") != 0:
        raise ValueError("Cut selection has not completed successfully")
    recorded_source = {name.replace("\\", "/"): value
                       for name, value in manifest.get("source", {}).get("files", {}).items()
                       if name.replace("\\", "/").startswith("src/") or name == "pyproject.toml"}
    current_source = {path.relative_to(root).as_posix(): digest_file(path)
                      for path in (root / "src").rglob("*.py")}
    if (root / "pyproject.toml").exists():
        current_source["pyproject.toml"] = digest_file(root / "pyproject.toml")
    if current_source != recorded_source:
        from scripts.maintenance.optimize_training_runtime import verify_selection_compatibility
        verify_selection_compatibility(root, recorded_source, current_source, source_compatibility)
    config = load_config(root / "configs/conditional.json", [
        root / "configs/algorithms/hpgi_finite_horizon.json",
        root / "configs/algorithms/cuda_update.json",
        root / "configs/training/fixed_300.json",
        root / "configs/runtime/windows_performance.json"])
    result = root / "results" / selection_run
    selection = read_json(result / "selection.json")
    overlay = result / "selected_overlay.json"
    if digest_file(overlay) != selection["overlay_sha256"]:
        raise ValueError("Selected cuts changed after scoring")
    if digest_file(suite / "manifest.json") != selection["suite_sha256"]:
        raise ValueError("Development inputs changed after selection")
    expected = candidate_cuts(config)
    if selection["candidates"] != expected or len(selection["ranking"]) != len(expected):
        raise ValueError("Incomplete or incompatible cut search")
    # Frozen cuts are reused across training seeds, budgets and PPO CPU/GPU use.
    # Physical sensitivity experiments (capacity/bandwidth/workload/objective or
    # model profiles) require a new development selection for that scenario.
    # Reject stale cuts rather than silently treating them as scenario-optimal.
    # Check physical scenario/target independently of training/runtime settings.
    from scripts.evaluation.evaluate_greedy import scenario
    if read_json(suite / "manifest.json")["scenario"] != scenario(config):
        raise ValueError("Development and training scenarios differ")
    if selection["selected"] != {name: selection["ranking"][0][name] for name in expected[0]}:
        raise ValueError("Chosen cuts do not match the top-ranked candidate")
    config = load_config(root / "configs/conditional.json", [
        root / "configs/algorithms/hpgi_finite_horizon.json",
        root / "configs/algorithms/cuda_update.json",
        root / "configs/training/fixed_300.json",
        root / "configs/runtime/windows_performance.json", overlay],
        [f"seeds.train={seed}"])
    return for_method(config, "fixed_split"), selection


def available_dependency(root, runs):
    for run in runs:
        folder = root / "logs/runs" / run
        manifest = read_json(folder / "manifest.json")
        pid = read_json(folder / "process_policy.json")["pid"]
        if process_alive(pid):
            continue
        if manifest["status"] != "completed" or manifest.get("exit_code") != 0:
            raise RuntimeError(f"Dependency exited unsuccessfully: {run}")
        history = read_json(folder / "cycle_history.json")
        if [row["stage"] for row in history] != [f"cycle_{index:03d}" for index in range(300)]:
            raise RuntimeError(f"Dependency did not finish 300 iterations: {run}")
        return run
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_project_root())
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wait-run", action="append", default=[])
    parser.add_argument("--seed", type=int, default=4200001)
    parser.add_argument("--source-compatibility", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or not 0 < args.max_hours <= 24:
        parser.error("Positive polling and at most 24 hours required")
    for name in [args.selection_run, args.run_id, *args.wait_run]:
        if not name or Path(name).name != name or name in (".", ".."):
            parser.error("Single-component run identifiers required")
    root = args.root.resolve()
    suite = args.suite.resolve()
    directory = root / "logs/queues" / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    if (root / "logs/runs" / args.run_id).exists() or (root / "results" / args.run_id).exists():
        raise FileExistsError(args.run_id)
    expected_source = source_hash(root)
    state = dict(status="waiting_for_cut_selection", pid=os.getpid(), created_utc=utc_now(),
                 selection_run=args.selection_run, run_id=args.run_id, dependencies=args.wait_run,
                 source_sha256=expected_source, seed=args.seed)
    deadline = time.monotonic() + args.max_hours * 3600

    def save():
        state["updated_utc"] = utc_now()
        write_json(directory / "status.json", state)

    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError("Workflow monitoring limit reached; existing training is left running")
        if source_hash(root) != expected_source:
            raise RuntimeError("Simulator source changed; do not launch with stale selection")

    keep_awake(True)
    try:
        save()
        selection_manifest = root / "logs/runs" / args.selection_run / "manifest.json"
        if not selection_manifest.exists():
            command = [sys.executable, "-m", "scripts.evaluation.select_fixed_split",
                       "--root", str(root), "--config", str(root / "configs/conditional.json"),
                       "--overlay", str(root / "configs/runtime/windows_performance.json"),
                       "--suite", str(suite), "--run-id", args.selection_run]
            with (directory / "selection_stdout.log").open("w", encoding="utf-8") as out, \
                 (directory / "selection_stderr.log").open("w", encoding="utf-8") as err:
                selection_process = subprocess.Popen(command, cwd=root, stdout=out, stderr=err)
            state["selection_pid"] = selection_process.pid
            save()
            while not selection_manifest.exists():
                guard()
                if selection_process.poll() is not None:
                    raise RuntimeError("Cut selector exited before creating its manifest")
                time.sleep(min(args.poll_seconds, 1))
        while True:
            guard()
            manifest = read_json(selection_manifest)
            if manifest["status"] == "completed":
                break
            if manifest["status"] != "running":
                raise RuntimeError("Cut selection failed; no PPO launched")
            time.sleep(args.poll_seconds)
        config, selection = selected_training_config(root, args.selection_run, suite, args.seed, args.source_compatibility)
        config_path = directory / "frozen_training.json"
        write_json(config_path, config)
        config_hash = digest_file(config_path)
        state.update(status="waiting_for_training_slot", selected=selection["selected"],
                     config=str(config_path), config_sha256=config_hash,
                     selection_sha256=digest_file(root / "results" / args.selection_run / "selection.json"))
        save()
        released = None
        while args.wait_run:
            guard()
            released = available_dependency(root, args.wait_run)
            if released is not None:
                break
            time.sleep(args.poll_seconds)
        guard()
        if digest_file(config_path) != config_hash:
            raise RuntimeError("Frozen training configuration changed")
        # Recheck selection/input hashes at launch, including long queue waits.
        selected_training_config(root, args.selection_run, suite, args.seed, args.source_compatibility)
        receipt = directory / "exit.json"
        command = terminal_command(root, sys.executable, config_path, args.run_id, args.seed,
                                   receipt, "Fixed-Split PPO | selected cuts | Iteration 1-300 | CUDA")
        write_json(directory / "launch.json", dict(command=command, selection_run=args.selection_run,
                                                   released_dependency=released))
        process = subprocess.Popen(command, cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE)
        state.update(status="training", terminal_pid=process.pid, released_dependency=released)
        save()
        while not receipt.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Monitoring limit reached; training is left running")
            if process.poll() is not None:
                raise RuntimeError("Training terminal closed without an exit receipt")
            time.sleep(args.poll_seconds)
        result = read_json(receipt)
        if result["exit_code"] != 0 or len(cycles(root, args.run_id)) != 300:
            raise RuntimeError("Fixed-Split did not successfully complete 300 iterations")
        manifest = read_json(root / "logs/runs" / args.run_id / "manifest.json")
        if manifest["status"] != "completed":
            raise RuntimeError("Fixed-Split completion manifest is inconsistent")
        state.update(status="completed", exit_code=0)
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
