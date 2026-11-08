import json

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
