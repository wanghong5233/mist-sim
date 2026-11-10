import csv
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from mist_sim.config import for_method, load_config
from mist_sim.evaluation import assert_close
from mist_sim.schedule import describe_schedule
from mist_sim.storage import write_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/evaluate_greedy.py"


def test_parallel_greedy_matches_serial_and_keeps_raw_evidence(config, tmp_path):
    config["horizon"] = 4
    path = tmp_path / "config.json"
    write_json(path, config)
    overlay = tmp_path / "runtime.json"
    write_json(overlay, {"runtime": {"windows_performance": sys.platform == "win32"}})
    suite = tmp_path / "suite"
    runs = []
    for workers in (1, 2):
        name = f"workers{workers}"
        command = [sys.executable, str(SCRIPT), "--config", str(path), "--overlay", str(overlay), "--suite", str(suite),
                   "--first-seed", "91", "--trajectories", "2", "--workers", str(workers),
                   "--run-id", name, "--output-dir", str(tmp_path / "outputs")]
        subprocess.run(command, cwd=ROOT, capture_output=True, check=True, timeout=90)
        run = tmp_path / "outputs" / name
        runs.append(run)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["budget"]["training_updates"] == 0
        runtime = json.loads((run / "summaries/seed91.json").read_text(encoding="utf-8"))["runtime"]
        assert runtime["status"] == ("applied" if sys.platform == "win32" else "unchanged")
        assert not (run / "checkpoints").exists()
        with (run / "results/evaluation.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert int(rows[-1]["arrivals"]) == sum(int(row["arrivals"]) for row in rows[:-1])
        assert float(rows[-1]["on_time_rate"]) == pytest.approx(
            sum(int(row["completed"]) for row in rows[:-1]) / int(rows[-1]["arrivals"]))
        assert float(rows[-1]["cost_per_slot"]) == pytest.approx(float(rows[-1]["conservative_cost"]) / 8)
    for seed in (91, 92):
        first, second = [json.loads((run / f"traces/evaluation/seed{seed}.json").read_text(encoding="utf-8")) for run in runs]
        for records in (first, second):
            for record in records:
                for key in ("measured_policy_seconds", "measured_allocation_seconds"):
                    record.pop(key, None)
        assert_close(first, second)
    module = runpy.run_path(str(SCRIPT))
    freeze = module["freeze_suite"]
    with pytest.raises(ValueError, match="unique"):
        freeze(tmp_path, suite, config, [91, 91])
    (suite / "seed91.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        freeze(tmp_path, suite, config, [91, 92])


def test_always_on_overlay_has_equal_hpgi_physical_budget():
    config = load_config(ROOT / "configs/smoke.json", [
        ROOT / "configs/algorithms/always_on_finite_horizon.json",
        ROOT / "configs/training/fixed_200.json"], ["horizon=128", "training.pretrain_slots=8192"])
    schedule = describe_schedule(for_method(config, config["method"]))
    assert schedule["planned_training_slots"] == 417792
    assert schedule["cycles"] == 200
    assert schedule["pretrain_rollouts"] == 4
    assert set(schedule["ppo"]) == {"lower"}
    assert schedule["value_model"] == "finite_horizon"
    assert schedule["value_target"] == "sampled_return"
