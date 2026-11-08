"""Read actual event files and verify stopping/resume against uninterrupted PPO."""
import copy
import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mist_sim.cli import main
from mist_sim.diagnostics import inspect_run
from mist_sim.schedule import describe_schedule, rollout_slots
from mist_sim.storage import RunStore, digest_file, write_json
from mist_sim.training import Trainer, state_hash

ROOT = Path(__file__).resolve().parents[1]


def small(config):
    config["horizon"] = 5
    config["k"] = 1
    config["training"].update(pretrain_slots=7, fast_slots=7, slow_intervals=3, cycles=1, calibration_episodes=1)
    config["ppo"].update(epochs=2, minibatch=4, target_kl=100.)
    config["monitoring"] = {"tensorboard": True, "console": False}
    return config


def test_tensorboard_matches_raw_logs_and_does_not_change_learning(config, tmp_path):
    config = small(config)
    store = RunStore(ROOT, config, "train", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    try:
        trainer.run()
    finally:
        store.finish()
    events = [json.loads(line) for line in (store.path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    epochs = [event for event in events if event["event"] == "ppo_epoch"]
    accumulator = EventAccumulator(str(store.path / "tensorboard"))
    accumulator.Reload()
    for layer in ("upper", "lower"):
        selected = [event for event in epochs if event["layer"] == layer]
        scalars = accumulator.Scalars("ppo/" + layer + "/loss")
        assert [scalar.step for scalar in scalars] == list(range(1, len(selected) + 1))
        assert [scalar.value for scalar in scalars] == pytest.approx([event["loss"] for event in selected], rel=1e-6)
        for event in selected:
            audit = json.loads((store.path / "traces/phases" / event["stage"] / "update.json").read_text(encoding="utf-8"))
            batches = [batch for batch in audit["minibatches"] if batch["epoch"] == event["epoch"] - 1]
            expected = sum(batch["total_loss"] * len(batch["indices"]) for batch in batches) / sum(len(batch["indices"]) for batch in batches)
            assert event["loss"] == pytest.approx(expected)
    for layer in ("upper", "lower"):
        episodes = [event for event in events if event["event"] == "episode_completed" and event["layer"] == layer]
        scalars = accumulator.Scalars("train/" + layer + "/episode/raw_return")
        assert [scalar.step for scalar in scalars] == [event["total_slots"] for event in episodes]
        assert [scalar.value for scalar in scalars] == pytest.approx([event["raw_return"] for event in episodes])
    validation = [event for event in events if event["event"] == "validation_completed"]
    assert [scalar.value for scalar in accumulator.Scalars("validation/return_per_episode")] == pytest.approx(
        [event["return_per_episode"] for event in validation])
    checked = inspect_run(store.path)
    assert len(checked["updates"]) == 3 and len(checked["physics"]) == 7
    manifest = json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts"].items():
        assert digest_file(store.path / name) == digest
    unmonitored = Trainer(copy.deepcopy(config), 91)
    unmonitored.run()
    for layer in ("upper", "lower"):
        assert state_hash(getattr(trainer.controller, layer)) == state_hash(getattr(unmonitored.controller, layer))
    assert trainer.total_slots == describe_schedule(config)["planned_training_slots"] == 23
    assert trainer.training_episodes == 3


def test_cli_interaction_stop_and_exact_resume(config, tmp_path):
    config = small(config)
    config["monitoring"]["tensorboard"] = False
    filename = tmp_path / "config.json"
    write_json(filename, config)
    output = tmp_path / "runs"
    run = main(["train", "--config", str(filename), "--output-dir", str(output), "--run-id", "bounded",
                "--max-slots", "8"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped" and manifest["exit_code"] == 0
    assert summary["total_slots"] == summary["resume_slots"] == 7
    assert summary["discarded_incomplete_phase_slots"] == 0
    resumed = main(["train", "--checkpoint", str(run / "checkpoints/last.pt"), "--output-dir", str(output),
                    "--run-id", "resumed"])
    restored = Trainer.load(resumed / "checkpoints/final.pt")
    uninterrupted = Trainer(config, config["seeds"]["train"])
    uninterrupted.run()
    assert restored.total_slots == 23 and len(restored.completed_stages) == 3
    assert restored.epoch_counts == uninterrupted.epoch_counts
    for layer in ("upper", "lower"):
        assert state_hash(getattr(restored.controller, layer)) == state_hash(getattr(uninterrupted.controller, layer))
    assert torch.equal(torch.load(resumed / "checkpoints/final.pt", weights_only=False)["rng"]["torch"],
                       torch.get_rng_state())


def test_time_limit_during_update_keeps_resumable_phase_start(config, tmp_path, monkeypatch):
    from mist_sim.schedule import TrainingLimit
    config = small(config)
    config["monitoring"]["tensorboard"] = False
    filename = tmp_path / "config.json"
    write_json(filename, config)
    original = Trainer.event

    def interrupt_after_first_epoch(self, event, **fields):
        original(self, event, **fields)
        if event == "ppo_epoch":
            raise TrainingLimit("wall_time_limit")

    monkeypatch.setattr(Trainer, "event", interrupt_after_first_epoch)
    run = main(["train", "--config", str(filename), "--output-dir", str(tmp_path / "runs")])
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["resume_slots"] == 0 and summary["discarded_incomplete_phase_slots"] == 7
    assert Trainer.load(run / "checkpoints/last.pt").total_slots == 0
    with pytest.raises(ValueError, match="diagnostic only"):
        Trainer.load(run / "checkpoints/interrupted.pt")
    monkeypatch.setattr(Trainer, "event", original)
    stopped = main(["train", "--config", str(filename), "--output-dir", str(tmp_path / "runs"),
                    "--max-seconds", "0.000000001"])
    assert json.loads((stopped / "manifest.json").read_text(encoding="utf-8"))["status"] == "stopped"
    assert Trainer.load(stopped / "checkpoints/last.pt").total_slots == 0


def test_dry_run_does_not_collect_or_write(config, tmp_path, monkeypatch):
    config = small(config)
    filename = tmp_path / "config.json"
    write_json(filename, config)
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not initialize a trainer")
    monkeypatch.setattr(Trainer, "__init__", forbidden)
    output = tmp_path / "must-not-exist"
    result = main(["train", "--config", str(filename), "--dry-run", "--max-slots", "11", "--output-dir", str(output)])
    assert result["planned_training_slots"] == 23 and result["stopping"]["max_slots"] == 11
    assert not output.exists()
    config["horizon"] = 17
    assert rollout_slots(config, "upper", 8) == 29


@pytest.mark.parametrize("field,value", [("max_slots", 0), ("max_slots", 1.5), ("max_seconds", -1)])
def test_invalid_stopping_configuration(config, field, value):
    from mist_sim.config import validate
    config["stopping"][field] = value
    with pytest.raises(ValueError):
        validate(config)
