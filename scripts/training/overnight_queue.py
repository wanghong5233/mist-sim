"""One-shot local training dependency queue; never changes simulator source."""
import argparse
import base64
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    digest.update((root / "pyproject.toml").read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def cycles(root, run):
    with (root / "results" / run / "cycles.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(row["cycle_index"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Non-contiguous complete iterations: {run}")
    return rows


def completed_score(root, run, expected_cycles=None):
    directory = root / "logs" / "runs" / run
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] == "running":
        return None
    if manifest["status"] != "completed" or manifest.get("exit_code") != 0:
        raise RuntimeError(f"Candidate did not complete successfully: {run}")
    rows = cycles(root, run)
    if expected_cycles is not None and len(rows) != expected_cycles:
        raise ValueError(f"Expected {expected_cycles} iterations, found {len(rows)}: {run}")
    if len(rows) < 40:
        raise ValueError("Selection requires at least two complete 20-iteration windows")
    cost = [float(row["cost_per_slot"]) for row in rows]
    service = [float(row["service"]) for row in rows]
    if not all(math.isfinite(value) for value in cost + service):
        raise ValueError("Non-finite candidate metrics")
    tail = statistics.mean(cost[-20:])
    previous = statistics.mean(cost[-40:-20])
    return dict(run=run, iterations=len(rows), tail20_cost=tail,
                tail20_service=statistics.mean(service[-20:]),
                previous20_cost=previous, tail_relative_change=(tail - previous) / max(abs(previous), 1e-12),
                tail20_seconds=statistics.mean(float(row["phase_seconds"]) for row in rows[-20:]),
                selection_basis="training tail-20 mean raw cost; preliminary, not held-out comparison")


def choose(scores):
    return min(scores, key=lambda item: (item["tail20_cost"], item["tail20_seconds"], item["run"]))


def new_seed_config(original, seed):
    config = json.loads(json.dumps(original))
    if type(seed) is not int or seed < 0 or seed in config["seeds"].values():
        raise ValueError("New seed must differ from original train, validation and test seeds")
    config["seeds"]["train"] = seed
    config["training"]["cycles"] = 200
    config["convergence"]["min_cycles"] = 200
    config["runtime"]["windows_performance"] = True
    return config


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def terminal_command(root, python, config, run_id, seed, receipt, title, checkpoint=None, curve_recovery=None):
    # Native stdout stays attached to the new console, preserving tqdm behavior.
    code = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "Set-Location -LiteralPath " + ps_quote(root),
        "$Host.UI.RawUI.WindowTitle = " + ps_quote(title),
        "$env:PYTHONPATH = " + ps_quote(root / "src"),
        "$trainingExit = 1",
        "try {",
        "  & " + ps_quote(python) + " -m mist_sim train --config " + ps_quote(config)
        + " --seed " + ps_quote(seed) + " --run-id " + ps_quote(run_id)
        + (" --checkpoint " + ps_quote(checkpoint) if checkpoint else ""),
        "  $trainingExit = $LASTEXITCODE",
        *( ["  if ($trainingExit -eq 0) {",
            "    & " + ps_quote(python) + " -m scripts.maintenance.optimize_training_runtime --root " + ps_quote(root)
            + " --output " + ps_quote(curve_recovery) + " --merge-run " + ps_quote(run_id),
            "    $trainingExit = $LASTEXITCODE",
            "  }"] if curve_recovery else []),
        "} catch { Write-Host $_ -ForegroundColor Red }",
        "$receiptText = @{exit_code=$trainingExit;finished_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json",
        "[IO.File]::WriteAllText(" + ps_quote(str(receipt) + ".partial") + ", $receiptText)",
        "Move-Item -LiteralPath " + ps_quote(str(receipt) + ".partial") + " -Destination " + ps_quote(receipt),
        "Write-Host ('Training exited with code ' + $trainingExit)",
    ])
    encoded = base64.b64encode(code.encode("utf-16le")).decode("ascii")
    return ["powershell.exe", "-NoProfile", "-NoExit", "-EncodedCommand", encoded]


def keep_awake(enable):
    if os.name != "nt":
        raise RuntimeError("Visible training queue requires Windows")
    api = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    api.argtypes = [ctypes.c_uint32]
    api.restype = ctypes.c_uint32
    if not api(0x80000001 if enable else 0x80000000):
        raise ctypes.WinError(ctypes.get_last_error())


def process_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # exited PID
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 258
    finally:
        kernel.CloseHandle(handle)


def check_frozen(root, snapshots, expected_source):
    if source_hash(root) != expected_source:
        raise RuntimeError("Simulator source changed while queue was waiting; no training launched")
    for run, frozen in snapshots.items():
        config_path = root / "logs" / "runs" / run / "resolved_config.json"
        if sha256(config_path) != frozen["config_sha256"]:
            raise RuntimeError(f"Candidate configuration changed: {run}")


def execute(args):
    root = args.root.resolve()
    if len(set(args.seed)) != 2 or len(args.candidate) != 2 or len(set(args.candidate)) != 2:
        raise ValueError("Exactly two distinct new seeds and two distinct candidates required")
    if args.wait_run not in args.candidate:
        raise ValueError("The dependency run must be a candidate")
    # mkdir(exist_ok=False) is the single-use lock; reruns never launch duplicates.
    directory = root / "logs" / "queues" / args.queue_id
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(status="preparing", pid=os.getpid(), created_utc=utc_now(),
                 dependency=args.wait_run, candidate_runs=args.candidate, seeds=args.seed,
                 source_sha256=source_hash(root), jobs=[], keep_awake=False)

    def save():
        state["updated_utc"] = utc_now()
        write_json(directory / "status.json", state)

    deadline = time.monotonic() + args.max_hours * 3600
    awake = False
    children = []
    try:
        snapshots = {}
        for run in args.candidate:
            candidate_dir = root / "logs" / "runs" / run
            config = read_json(candidate_dir / "resolved_config.json")
            manifest = read_json(candidate_dir / "manifest.json")
            for seed in args.seed:
                new_seed_config(config, seed)
            snapshots[run] = dict(config=config, config_sha256=sha256(candidate_dir / "resolved_config.json"),
                                  source_commit=manifest["source"]["commit"])
        write_json(directory / "candidates.json", snapshots)
        dependency_pid = read_json(root / "logs" / "runs" / args.wait_run / "process_policy.json")["pid"]
        keep_awake(True)
        awake = True
        state.update(status="waiting", keep_awake=True, dependency_pid=dependency_pid)
        save()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Queue exceeded its 24-hour authorization window")
            check_frozen(root, snapshots, state["source_sha256"])
            score = completed_score(root, args.wait_run, expected_cycles=200)
            if score is not None and not process_alive(dependency_pid):
                break
            if score is None and not process_alive(dependency_pid):
                raise RuntimeError("Dependency process disappeared without successful completion")
            save()
            time.sleep(args.poll_seconds)
        scores = [completed_score(root, run) for run in args.candidate]
        if any(score is None for score in scores):
            raise RuntimeError("Another selection candidate is still running")
        selected = choose(scores)
        state.update(status="selected", scores=scores, selected=selected)
        write_json(directory / "selection.json", dict(selected=selected, candidates=scores))
        save()
        config = snapshots[selected["run"]]["config"]
        for seed in args.seed:
            check_frozen(root, snapshots, state["source_sha256"])
            run_id = f"{config['method']}-finite-horizon-gpu-b128-w4-highqos-200-seed{seed}-{args.queue_id}"
            if (root / "logs" / "runs" / run_id).exists() or (root / "results" / run_id).exists():
                raise FileExistsError(run_id)
            config_path = directory / f"seed{seed}.json"
            write_json(config_path, new_seed_config(config, seed))
            receipt = directory / f"seed{seed}-exit.json"
            job = dict(seed=seed, run_id=run_id, status="launching", config=str(config_path), receipt=str(receipt))
            state["jobs"].append(job)
            save()
            try:
                process = subprocess.Popen(terminal_command(root, args.python, config_path, run_id, seed, receipt,
                    f"MIST {config['method']} | seed {seed} | 200 iterations"), cwd=root,
                    creationflags=subprocess.CREATE_NEW_CONSOLE)
            except OSError as error:
                job.update(status="failed", error=f"Terminal launch failed: {error}")
                save()
                continue
            children.append((job, process))
            job.update(status="running", terminal_pid=process.pid)
            save()
        state["status"] = "training"
        save()
        while any(job["status"] == "running" for job, _ in children):
            if time.monotonic() >= deadline:
                state["status"] = "monitor_timeout_jobs_left_running"
                save()
                return 1
            for job, process in children:
                if job["status"] != "running":
                    continue
                receipt = Path(job["receipt"])
                if receipt.exists():
                    result = read_json(receipt)
                    job["exit_code"] = result["exit_code"]
                    if result["exit_code"] == 0:
                        try:
                            if completed_score(root, job["run_id"], expected_cycles=200) is None:
                                raise RuntimeError("Exit code zero but manifest still running")
                            job["status"] = "completed"
                        except Exception as error:
                            job.update(status="failed", error=str(error))
                    else:
                        job["status"] = "failed"
                elif process.poll() is not None:
                    job.update(status="failed", error="Terminal closed without training exit receipt")
            save()
            if any(job["status"] == "running" for job, _ in children):
                time.sleep(args.poll_seconds)
        state["status"] = "completed" if all(job["status"] == "completed" for job in state["jobs"]) else "completed_with_failures"
        save()
        if state["status"] == "completed":
            label = "HPGI" if config["method"] == "hpgi" else "Joint PPO"
            command = [args.python, str(root / "scripts" / "plotting" / "plot_convergence.py")]
            for run in [selected["run"]] + [job["run_id"] for job, _ in children]:
                command.extend(["--curve", f"{label}={root / 'results' / run}"])
            command.extend(["--output", str(root / "results" / "comparisons" / args.queue_id / "figures" / "convergence")])
            with (directory / "plot.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            state["plot_exit_code"] = result.returncode
            if result.returncode != 0:
                state["status"] = "training_completed_plot_failed"
            save()
        return 0 if state["status"] == "completed" else 1
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if awake:
            keep_awake(False)
            state["keep_awake"] = False
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--wait-run", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    for value in [args.queue_id, args.wait_run] + args.candidate:
        if not value or Path(value).name != value or value in (".", ".."):
            parser.error("Queue/run identifiers must be single path components")
    if args.poll_seconds <= 0 or not 0 < args.max_hours <= 24:
        parser.error("Positive polling interval and at most 24 hours required")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
