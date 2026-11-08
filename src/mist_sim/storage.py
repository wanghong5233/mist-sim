"""Immutable run creation, provenance and durable failure records."""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time
import tempfile
import traceback
import uuid
import numpy as np
import scipy
import torch


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


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
        self.path = (Path(output_dir).resolve() if output_dir is not None else self.root / "runs") / identifier
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.monitor = None
        self.monitoring = config.get("monitoring", {"tensorboard": False, "console": True})
        self.manifest = {"run_id": identifier, "command": command, "seed": seed,
                         "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                         "source": source_identity(self.root), "device": "cpu",
                         "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                                          "scipy": scipy.__version__, "torch": torch.__version__},
                         "budget": None}
        self.write_json("resolved_config.json", config)
        self.manifest["config_sha256"] = digest_file(self.path / "resolved_config.json")
        self.manifest["input_profile_sha256"] = hashlib.sha256(json.dumps(config["profiles"], sort_keys=True).encode()).hexdigest()
        self.write_json("manifest.json", self.manifest)
        (self.path / "stdout.log").write_text(command + "\n", encoding="utf-8")
        (self.path / "stderr.log").touch()

    def write_json(self, name, value):
        write_json(self.path / name, value)

    def append_metrics(self, metrics):
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
        with (self.path / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": metrics["stage"], "samples": metrics["samples"], "total_slots": metrics["total_slots"]}) + "\n")

    def event(self, event, **fields):
        from .monitoring import TrainingMonitor, format_event
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
                print(message, flush=True)
        if self.monitor is not None:
            self.monitor.record(event, fields)

    def finish(self, error=None, stopped=None):
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
