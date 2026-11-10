import csv
from pathlib import Path

import pytest

from mist_sim.storage import RunStore
from mist_sim.training import Trainer


def test_training_curve_without_any_extra_evaluation(config, tmp_path, monkeypatch):
    config["validation"]["enabled"] = False
    config["training"].update(pretrain_slots=2, fast_slots=2, slow_intervals=1, cycles=2, calibration_episodes=1)
    config["horizon"] = 4
    config["ppo"]["epochs"] = 1
    config["monitoring"] = dict(tensorboard=False, console=False)
    config["diagnostics"] = dict(phase_interval=10)
    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not launch evaluation")
    monkeypatch.setattr(Trainer, "evaluate_validation", forbidden)
    monkeypatch.setattr(Trainer, "evaluate_sampled_validation", forbidden)
    store = RunStore(Path(__file__).resolve().parents[1], config, "train", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    try:
        trainer.run()
        assert trainer.validation_history == []
        assert not (store.path / "traces/validation").exists()
        assert not (store.result_path / "validation.csv").exists()
        assert {path.name for path in (store.path / "traces/phases").iterdir()} == {"pretrain", "cycle_000_upper"}
        with (store.result_path / "cycles.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        for row in rows:
            phases = [phase for phase in trainer.history if phase["stage"].startswith(row["stage"] + "_")]
            assert float(row["reward_per_slot"]) == pytest.approx(sum(phase["raw_reward_sum"] for phase in phases) /
                                                                   sum(phase["physical_slots"] for phase in phases))
        restored = Trainer.load(store.path / "checkpoints/final.pt")
        restored.run(extra_cycles=1)
        assert len(restored.cycle_history) == 3
    finally:
        store.finish()


def test_minimal_periodic_checkpoint_survives_later_phase_failure(config, tmp_path, monkeypatch):
    from mist_sim.config import for_method
    config = for_method(config, "always_on")
    config["validation"]["enabled"] = False
    config["training"].update(pretrain_slots=2, fast_slots=2, cycles=3, calibration_episodes=1)
    config["horizon"] = 4
    config["ppo"]["epochs"] = 1
    config["monitoring"] = dict(tensorboard=False, console=False)
    config["checkpoint_policy"] = dict(mode="minimal", stage_interval=0, keep_stages=0, cycle_interval=2)
    store = RunStore(Path(__file__).resolve().parents[1], config, "train", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    original = trainer.collect
    def interrupt(layer, count, pretrain=False):
        if len(trainer.cycle_history) == 2:
            raise RuntimeError("injected third cycle failure")
        return original(layer, count, pretrain)
    monkeypatch.setattr(trainer, "collect", interrupt)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            trainer.run()
        restored = Trainer.load(store.path / "checkpoints/last.pt")
        assert len(restored.cycle_history) == 2
        assert restored.completed_stages[-1] == "cycle_001_lower"
        assert restored.versions["lower"] == 3
        restored.run()
        assert [r["stage"] for r in restored.cycle_history] == ["cycle_000", "cycle_001", "cycle_002"]
        assert restored.versions["lower"] == 4
    finally:
        store.finish()
