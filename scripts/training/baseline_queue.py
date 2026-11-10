"""Completion-driven continuation of distinct baselines, with at most two jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from mist_sim.config import load_config
from scripts.training.overnight_queue import (
    keep_awake, process_alive, read_json, sha256, source_hash, terminal_command, utc_now, write_json,
)


def completed(root, run):
    directory = root / "logs/runs" / run
    if not directory.exists():
        return False
    policy = directory / "process_policy.json"
    if not policy.exists() or process_alive(read_json(policy)["pid"]):
        return False
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "completed" or manifest.get("exit_code") != 0:
        raise RuntimeError(f"Training process exited without successful completion: {run}")
    history = read_json(directory / "cycle_history.json")
    if [row["stage"] for row in history] != [f"cycle_{index:03d}" for index in range(300)]:
        raise RuntimeError(f"Training did not finish 300 iterations: {run}")
    return True


def fill_slots(active, pending):
    return pending[:max(0, 2 - len(active))]


def execute(root, spec_path, queue_id, poll_seconds=30, max_hours=24):
    spec = read_json(spec_path)
    require_runs = spec["active"] + [job["run_id"] for job in spec["pending"]]
    if len(spec["active"]) > 2 or len(set(require_runs)) != len(require_runs):
        raise ValueError("At most two distinct active jobs; no duplicate runs")
    for run in require_runs:
        if not run or Path(run).name != run or run in (".", ".."):
            raise ValueError("Invalid run identifier")
    frozen_source = source_hash(root)
    spec_hash = sha256(spec_path)
    for job in spec["pending"]:
        config = load_config(job["config"])
        if (root / "logs/runs" / job["run_id"]).exists():
            raise FileExistsError(job["run_id"])
        if config["method"] not in ("fixed_split", "fixed_window") or config["training"]["cycles"] != 300:
            raise ValueError("Queue is limited to the two pending 300-iteration baselines")
        if config["runtime"]["windows_performance"] or config["runtime"]["collection_workers"] != 4:
            raise ValueError("Expected ordinary-priority four-worker configuration")
        if sha256(job["config"]) != job["config_sha256"]:
            raise ValueError("Frozen configuration changed")
    directory = root / "logs/queues" / queue_id
    directory.mkdir(parents=True, exist_ok=False)
    active, pending = list(spec["active"]), list(spec["pending"])
    children, completed_runs = {}, []
    state = dict(status="waiting_for_training_slot", pid=os.getpid(), created_utc=utc_now(),
                 source_sha256=frozen_source, spec=str(spec_path), spec_sha256=spec_hash, jobs=[])

    def save():
        state.update(updated_utc=utc_now(), active=active, pending=[job["run_id"] for job in pending], completed=completed_runs)
        write_json(directory / "status.json", state)

    deadline = time.monotonic() + max_hours * 3600
    keep_awake(True)
    state["keep_awake"] = True
    try:
        save()
        while active or pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("24-hour monitoring limit; existing jobs left running")
            if source_hash(root) != frozen_source or sha256(spec_path) != spec_hash:
                raise RuntimeError("Frozen source/specification changed; no further jobs launched")
            for run in list(active):
                if run in children:
                    process, receipt, launched_at = children[run]
                    if receipt.exists() and read_json(receipt)["exit_code"] != 0:
                        raise RuntimeError(f"Training terminal reported failure: {run}")
                    policy = root / "logs/runs" / run / "process_policy.json"
                    if not policy.exists() and (process.poll() is not None or time.monotonic() - launched_at > 300):
                        raise RuntimeError(f"Training failed to start: {run}")
                if completed(root, run):
                    active.remove(run)
                    completed_runs.append(run)
            for job in fill_slots(active, pending):
                if sha256(job["config"]) != job["config_sha256"]:
                    raise RuntimeError("Frozen configuration changed")
                if job.get("selection"):
                    from scripts.training.train_fixed_split import selected_training_config
                    selected, _ = selected_training_config(root, job["selection"], Path(job["suite"]),
                                                           job["seed"], Path(job["compatibility"]))
                    selected["runtime"]["windows_performance"] = False
                    if selected != load_config(job["config"]):
                        raise RuntimeError("Frozen Fixed-Split selection changed")
                receipt = directory / f"{job['run_id']}-exit.json"
                command = terminal_command(root, sys.executable, Path(job["config"]), job["run_id"], job["seed"],
                                           receipt, f"MIST {job['method']} | Iteration 1-300 | CUDA shared | 4 workers")
                process = subprocess.Popen(command, cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE)
                children[job["run_id"]] = (process, receipt, time.monotonic())
                active.append(job["run_id"])
                pending.remove(job)
                state["jobs"].append(dict(run_id=job["run_id"], terminal_pid=process.pid, config=job["config"],
                                           receipt=str(receipt), started_utc=utc_now()))
                state["status"] = "training"
                save()
            save()
            if active or pending:
                time.sleep(poll_seconds)
        state["status"] = "completed"
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        keep_awake(False)
        state["keep_awake"] = False
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    args = parser.parse_args()
    if not args.queue_id or Path(args.queue_id).name != args.queue_id or args.queue_id in (".", ".."):
        parser.error("Invalid queue identifier")
    execute(args.root.resolve(), args.spec.resolve(), args.queue_id)


if __name__ == "__main__":
    main()
