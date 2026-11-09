import copy
import csv
import json
from pathlib import Path

import pytest

from mist_sim.convergence import DEFAULT_CONVERGENCE, reward_stability
from mist_sim.storage import RunStore
from mist_sim.training import Trainer


def settings():
    return dict(DEFAULT_CONVERGENCE, enabled=True, min_cycles=4, window=2, patience=2, tail_cycles=2)


def add_cycle(history, cost=100., sampled_cost=None):
    count = sum(row["stage"].startswith("cycle_") for row in history)
    record = dict(stage=f"cycle_{count:03d}", metrics=dict(conservative_cost=cost), trajectories=[{}])
    if sampled_cost is not None:
        record["sampled_metrics"] = dict(conservative_cost=sampled_cost)
    history.append(record)
    record["convergence"] = reward_stability(history, settings(), 4)
    return record["convergence"]


def test_stability_requires_windows_patience_tail_and_resets():
    history = []
    for count in range(1, 8):
        state = add_cycle(history)
        assert state["stop"] == (count == 7)
    assert state["modes"]["metrics"]["reward_mean"] == -25.
    state = add_cycle(history, cost=200.)
    assert not state["stable"] and state["detected_cycle"] is None
    assert state["stable_checks"] == 0


@pytest.mark.parametrize("mode", ["trend", "noise", "sampled"])
def test_nonstable_reward_does_not_stop(mode):
    history = []
    for index in range(30):
        cost = 100 + index if mode == "trend" else 100 + 20 * (index % 2) if mode == "noise" else 100
        state = add_cycle(history, cost, 100 + index if mode == "sampled" else None)
        assert not state["stop"]


def test_pretrain_ignored_and_resume_preserves_streak():
    history = [dict(stage="initial"), dict(stage="pretrain")]
    for index in range(5):
        state = add_cycle(history)
    assert state["detected_cycle"] == 5
    restored = json.loads(json.dumps(history))
    assert add_cycle(restored) == add_cycle(history)
    assert add_cycle(restored)["stop"]


def test_empty_request_diagnostics_are_defined(config):
    from mist_sim.diagnostics import policy_input_diagnostics
    config["arrival_probability"] = 0.
    trainer = Trainer(config, 91)
    buffer, _ = trainer.collect("upper", 1)
    diagnostics = policy_input_diagnostics(trainer.controller.upper, buffer.samples)
    assert any(value is None for value in diagnostics["tanh_saturation"].values())
    assert diagnostics["features"]["item"]["rows"] == 0


def test_minimal_training_stops_saves_infrequently_and_keeps_csv(config, tmp_path, monkeypatch):
    config["horizon"] = 4
    config["training"].update(pretrain_slots=2, fast_slots=2, slow_intervals=1, cycles=20, calibration_episodes=1)
    config["ppo"]["epochs"] = 1
    config["monitoring"] = dict(tensorboard=False, console=False)
    config["checkpoint_policy"].update(mode="minimal", cycle_interval=3)
    config["trace_retention"] = dict(phases=2, validations=2, inputs=2)
    config["convergence"] = settings()
    original = Trainer.evaluate_validation

    def constant_evaluation(self):
        records, metrics, trajectories = original(self)
        metrics["conservative_cost"] = 100.
        return records, metrics, trajectories

    monkeypatch.setattr(Trainer, "evaluate_validation", constant_evaluation)
    saved_stages = []
    save = Trainer.save

    def record_save(self, path):
        saved_stages.append((Path(path).name, self.validation_history[-1]["stage"] if self.validation_history else None))
        save(self, path)

    monkeypatch.setattr(Trainer, "save", record_save)
    store = RunStore(Path(__file__).resolve().parents[1], config, "train", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    trainer.run()
    store.finish()
    assert trainer.stop_reason == "reward_stable"
    assert len([row for row in trainer.validation_history if row["stage"].startswith("cycle_")]) == 7
    assert saved_stages == [("last.pt", None), ("last.pt", "pretrain"), ("last.pt", "cycle_002"),
                            ("last.pt", "cycle_005"), ("last.pt", "cycle_006")]
    assert [path.name for path in (store.path / "checkpoints").iterdir()] == ["last.pt"]
    assert trainer.best_checkpoint_path is None
    assert json.loads((store.path / "selection.json").read_text())["status"] == "metrics_only"
    assert len(list((store.path / "traces/phases").iterdir())) == 2
    assert len(list((store.path / "traces/inputs").iterdir())) == 2
    assert len((store.path / "input_seeds.jsonl").read_text().splitlines()) == trainer.trace_counter
    assert {path.stem for path in (store.path / "traces/validation").iterdir()} == {"cycle_005", "cycle_006", "final"}
    with (store.result_path / "convergence.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 and rows[-1]["stop"] == "True"
    with (store.result_path / "iterations.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 15
    restored = Trainer.load(store.path / "checkpoints/last.pt")
    before = copy.deepcopy(restored.versions)
    restored.run(extra_cycles=1)
    assert restored.versions == before and restored.stop_reason == "reward_stable"


def test_validation_retention_keeps_pairs_and_rejects_paths(config, tmp_path):
    config["trace_retention"] = dict(phases=1, validations=1, inputs=1)
    store = RunStore(Path(__file__).resolve().parents[1], config, "train", 91, output_dir=tmp_path)
    for stage in ("cycle_000", "cycle_001"):
        for suffix in ("", "_sampled"):
            store.write_json(f"traces/validation/{stage}{suffix}.json", [])
    store.retain_traces("validations", ["cycle_000", "cycle_001"])
    assert {path.name for path in (store.path / "traces/validation").iterdir()} == {"cycle_001.json", "cycle_001_sampled.json"}
    with pytest.raises(ValueError, match="Invalid trace stage"):
        store.retain_traces("phases", ["../../escape", "cycle_001"])
    store.finish()
