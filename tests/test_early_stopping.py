import json
from pathlib import Path

import pytest

from mist_sim.cli import main
from mist_sim.schedule import plateau_status
from mist_sim.storage import write_json
from mist_sim.training import Trainer


def test_plateau_threshold_and_warmup():
    settings = dict(enabled=True, patience=2, min_slots=10, min_relative_improvement=.01)
    history = []
    for index, (slots, cost) in enumerate(((1, 100), (10, 100), (20, 99.5), (30, 98), (40, 98), (50, 98))):
        history.append(dict(stage=f"cycle_{index:03d}", total_slots=slots, eligible=True,
                            metrics={"conservative_cost": cost}))
        assert plateau_status(history, settings)["stop"] == (index == 5)
    settings["enabled"] = False
    assert not plateau_status(history, settings)["stop"]


def test_sampled_improvement_prevents_deterministic_plateau_stop():
    settings = dict(enabled=True, patience=2, min_slots=0, min_relative_improvement=.001)
    history = []
    for index, cost in enumerate((100., 95., 90., 90., 90.)):
        history.append(dict(stage=f"cycle_{index:03d}", total_slots=index * 10, eligible=True,
                            metrics={"conservative_cost": 100.},
                            sampled_metrics={"conservative_cost": cost}, sampled_eligible=True))
        status = plateau_status(history, settings)
        assert status["stop"] == (index == 4)
    assert status["monitored_modes"] == "deterministic_and_sampled"


def test_sampled_validation_is_repeatable_rng_isolated_and_saved(config, tmp_path):
    import csv
    import random
    import numpy as np
    import torch
    from mist_sim.storage import RunStore
    config["validation"]["sampled"] = True
    config["horizon"] = 4
    config["training"].update(pretrain_slots=2, fast_slots=2, slow_intervals=1, cycles=1, calibration_episodes=1)
    config["monitoring"] = {"tensorboard": False, "console": False}
    root = Path(__file__).resolve().parents[1]
    store = RunStore(root, config, "train", 91, output_dir=tmp_path / "runs")
    trainer = Trainer(config, 91, store)
    try:
        state = torch.get_rng_state().clone()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        first = trainer.evaluate_sampled_validation()
        second = trainer.evaluate_sampled_validation()
        assert torch.equal(state, torch.get_rng_state())
        assert random.getstate() == python_state
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert first[1] == second[1]
        assert [row["active"] for row in first[0]] == [row["active"] for row in second[0]]
        trainer.validate_cycle("cycle_000")
        record = trainer.validation_history[-1]
        assert record["sampled_metrics"] == first[1]
        with (store.result_path / "validation.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert int(row["cycle_index"]) == 1
        assert float(row["sampled_cost_per_slot"]) == pytest.approx(first[1]["conservative_cost"] / 4)
        assert float(row["sampled_return_per_episode"]) == pytest.approx(sum(item["reward"] for item in first[0]))
        loaded = Trainer.load(store.path / "checkpoints/last.pt")
        assert loaded.validation_history[-1]["sampled_metrics"] == first[1]
        assert (store.result_path / "validation_sampled_trajectories.csv").exists()
    finally:
        store.finish()


def test_five_epoch_update_reports_activation_probabilities(config):
    config["ppo"].update(epochs=5, minibatch=4, target_kl=100.)
    trainer = Trainer(config, 91)
    metrics = trainer.phase("cycle_000_upper", "upper", 8)
    assert metrics["epochs_completed"] == 5
    assert metrics["optimizer_steps"] == 10
    probability = metrics["activation_probability"]
    assert 0 <= probability["min"] <= probability["mean"] <= probability["max"] <= 1


def test_cli_stops_and_resume_keeps_patience(config, tmp_path, monkeypatch):
    config["training"].update(pretrain_slots=2, fast_slots=2, slow_intervals=1, cycles=10, calibration_episodes=1)
    config["horizon"] = 4
    config["ppo"]["epochs"] = 1
    config["stopping"]["plateau"] = dict(enabled=True, patience=2, min_slots=0, min_relative_improvement=.001)
    original = Trainer.evaluate_validation

    def constant_validation(self):
        records, metrics, trajectories = original(self)
        metrics["conservative_cost"] = 100.
        return records, metrics, trajectories

    monkeypatch.setattr(Trainer, "evaluate_validation", constant_validation)
    filename = tmp_path / "config.json"
    write_json(filename, config)
    run = main(["train", "--config", str(filename), "--output-dir", str(tmp_path / "runs")])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["stop_reason"] == "validation_plateau"
    assert summary["total_slots"] == 20
    assert (run / "checkpoints/final.pt").exists()
    assert (run / "checkpoints/best.pt").exists()
    resumed = main(["train", "--checkpoint", str(run / "checkpoints/last.pt"),
                    "--output-dir", str(tmp_path / "runs"), "--extra-cycles", "10"])
    assert json.loads((resumed / "summary.json").read_text())["total_slots"] == 20


@pytest.mark.parametrize("key,value", [("patience", 0), ("min_slots", -1), ("min_relative_improvement", 1)])
def test_invalid_plateau(config, key, value):
    from mist_sim.config import validate
    config["stopping"]["plateau"][key] = value
    with pytest.raises(ValueError, match="plateau"):
        validate(config)
