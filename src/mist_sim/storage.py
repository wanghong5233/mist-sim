"""Immutable run creation, provenance and durable failure records."""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import tempfile
import traceback
import uuid
import numpy as np
import scipy
import torch


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_trace_path(run, stage, kind):
    nested = Path(run) / "traces" / "phases" / stage / (kind + ".json")
    if nested.exists():
        return nested
    suffix = "" if kind == "physical" else "_" + kind
    return Path(run) / "traces" / (stage + suffix + ".json")


def validation_trace_path(run, stage):
    nested = Path(run) / "traces" / "validation" / (stage + ".json")
    if nested.exists():
        return nested
    name = "validation.json" if stage == "final" else stage + "_validation.json"
    return Path(run) / "traces" / name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_identity(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        return result.stdout if result.returncode == 0 else b""
    files = (sorted((root / "src").rglob("*.py")) + sorted((root / "configs").rglob("*.json")) +
             sorted((root / "tests").rglob("*.py")) + sorted((root / "scripts").rglob("*.py")) + [root / "pyproject.toml"])
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty_patch_sha256": hashlib.sha256(git("diff", "HEAD", "--", str(root))).hexdigest(),
            "files": {str(path.relative_to(root)): digest_file(path) for path in files}}


def readable_run_id(config, command, seed):
    if "profile_models" in command:
        purpose = "dnn-profiling"
    elif "check_inputs" in command:
        purpose = "input-check"
    elif "verification" in command or "verify_commands" in command:
        purpose = "code-validation"
    else:
        match = re.search(r"(?:^|\s)(train|evaluate|check|replay|summarize|reference)(?:\s|$)", command)
        purpose = match.group(1) if match else "run"
    method = re.sub(r"[^a-zA-Z0-9_-]", "-", config.get("method", "models"))
    return f"{purpose}-{method}-seed{seed}__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]


@contextmanager
def validation_workspace(keep_artifacts=None):
    if keep_artifacts is not None:
        directory = Path(keep_artifacts).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        yield directory
    else:
        with tempfile.TemporaryDirectory(prefix="mist-validation-") as directory:
            yield Path(directory)


class RunStore:
    def __init__(self, root, config, command, seed, run_id=None, output_dir=None):
        self.root = Path(root).resolve()
        identifier = run_id or readable_run_id(config, command, seed)
        if Path(identifier).name != identifier or identifier in (".", ".."):
            raise ValueError("Invalid run id")
        self.path = (Path(output_dir).resolve() if output_dir is not None else self.root / "logs" / "runs") / identifier
        self.result_path = self.root / "results" / identifier if output_dir is None else self.path / "results"
        if self.result_path.exists():
            raise FileExistsError(self.result_path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.monitor = None
        self.trace_retention = config.get("trace_retention", {})
        self.trimmed_traces = set()
        self.monitoring = config.get("monitoring", {"tensorboard": False, "console": True})
        self.manifest = {"run_id": identifier, "command": command, "seed": seed,
                         "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                        "source": source_identity(self.root), "device": config.get("device", "cpu"), "sampling_device": "cpu",
                         "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                                          "scipy": scipy.__version__, "torch": torch.__version__},
                         "budget": None, "trace_retention": self.trace_retention}
        self.write_json("resolved_config.json", config)
        self.manifest["config_sha256"] = digest_file(self.path / "resolved_config.json")
        self.manifest["input_profile_sha256"] = hashlib.sha256(json.dumps(config["profiles"], sort_keys=True).encode()).hexdigest()
        self.write_json("manifest.json", self.manifest)
        (self.path / "stdout.log").write_text(command + "\n", encoding="utf-8")
        (self.path / "stderr.log").touch()

    def write_json(self, name, value):
        write_json(self.path / name, value)

    def record_input_seed(self, index, seed):
        with (self.path / "input_seeds.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(index=index, seed=seed)) + "\n")
        keep = self.trace_retention.get("inputs")
        if keep is not None:
            for path in sorted((self.path / "traces" / "inputs").glob("input_*.json"))[:-keep]:
                path.unlink()

    def retain_traces(self, kind, completed):
        """Bound completed diagnostics only; scalar CSV and input seed ledger survive."""
        if kind not in ("phases", "validations"):
            raise ValueError("Unknown trace kind")
        keep = self.trace_retention.get(kind)
        if keep is None:
            return
        root = self.path / "traces" / ("validation" if kind == "validations" else kind)
        for stage in completed[:-keep]:
            if (kind, stage) in self.trimmed_traces:
                continue
            if not re.fullmatch(r"initial|pretrain(?:_\d+)?|cycle_\d+(?:_upper|_lower|_joint)?", stage):
                raise ValueError("Invalid trace stage")
            paths = [root / stage] if kind == "phases" else [root / (stage + suffix + ".json") for suffix in ("", "_sampled")]
            for path in paths:
                if not path.resolve().is_relative_to((self.path / "traces").resolve()):
                    raise ValueError("Trace escaped run directory")
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            self.trimmed_traces.add((kind, stage))

    def append_metrics(self, metrics):
        from .results import iteration_row
        self.append_result("iterations.csv", iteration_row(metrics))
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
        with (self.path / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": metrics["stage"], "samples": metrics["samples"], "total_slots": metrics["total_slots"]}) + "\n")

    def append_result(self, name, row):
        from .results import append_csv
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("Result tables must be CSV filenames within the run result directory")
        append_csv(self.result_path / name, dict(run_id=self.manifest["run_id"], seed=self.manifest["seed"], **row))

    def record_episode(self, **fields):
        self.append_result("episodes.csv", fields)
        self.event("episode_completed", **fields)

    def record_validation(self, **fields):
        self.append_result("validation.csv", fields)
        self.event("validation_completed", **fields)

    def record_evaluation(self, metrics, **identity):
        from .results import evaluation_row
        self.append_result("evaluation.csv", dict(identity, **evaluation_row(metrics)))

    def event(self, event, **fields):
        from .monitoring import ConsoleProgress, TrainingMonitor, format_event
        if event in ("training_started", "phase_started") and self.monitor is None:
            self.monitor = TrainingMonitor(self.path / "tensorboard", self.monitoring["tensorboard"])
        record = dict(event=event, elapsed_seconds=time.perf_counter() - self.started, **fields)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (self.path / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        message = format_event(event, fields)
        if message:
            with (self.path / "stdout.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        if self.monitoring["console"]:
            if not hasattr(self, "console_progress"):
                self.console_progress = ConsoleProgress()
            self.console_progress.record(event, fields)
        if self.monitor is not None:
            self.monitor.record(event, fields)

    def finish(self, error=None, stopped=None):
        if hasattr(self, "console_progress"):
            self.console_progress.close()
        if self.monitor is not None:
            self.monitor.close()
        self.manifest.update(status="failed" if error else "stopped" if stopped else "completed", exit_code=1 if error else 0,
                             finished_utc=datetime.now(timezone.utc).isoformat(),
                             measured_wall_seconds=time.perf_counter() - self.started)
        if stopped:
            self.manifest["stop_reason"] = stopped
        if error:
            (self.path / "stderr.log").write_text("".join(traceback.format_exception(type(error), error, error.__traceback__)), encoding="utf-8")
        self.manifest["artifacts"] = {str(path.relative_to(self.path)): digest_file(path)
                                      for path in self.path.rglob("*") if path.is_file() and path.name != "manifest.json"}
        self.write_json("manifest.json", self.manifest)
        if self.result_path.exists():
            write_json(self.result_path / "resolved_config.json",
                       json.loads((self.path / "resolved_config.json").read_text(encoding="utf-8")))
            result_manifest = dict(self.manifest)
            result_manifest["logs_path"] = str(self.path)
            result_manifest["artifacts"] = {str(path.relative_to(self.result_path)): digest_file(path)
                                             for path in self.result_path.rglob("*")
                                             if path.is_file() and path.name != "manifest.json"}
            write_json(self.result_path / "manifest.json", result_manifest)
