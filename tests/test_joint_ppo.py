import copy
import csv
import io
import json

import numpy as np
import pytest
import torch

from mist_sim.buffers import EventRolloutBuffer
from mist_sim.diagnostics import check_physics, check_update
from mist_sim.evaluation import assert_close, evaluate
from mist_sim.anonymity import generate_trace
from mist_sim.policies.batching import replay_chunks
from mist_sim.schedule import describe_schedule, event_count
from mist_sim.storage import RunStore
from mist_sim.traces import read_behavior
from mist_sim.training import Trainer


def test_joint_event_buffer_connects_zero_duration_phase(config):
    buffer = EventRolloutBuffer({"upper": 0, "lower": 0})
    frame = {"items": [], "servers": [], "context": [0] * 7}
    buffer.append(dict(phase="upper", value=1.0, frame=frame, steps=[], physical_rewards=[]),
                  0.0, 2.0, False, False, duration=0, next_observation=frame)
    buffer.append(dict(phase="lower", value=2.0, frame=frame, steps=[], physical_rewards=[-3.0]),
                  -3.0, 0.0, True, False, duration=1, next_observation=None)
    advantages, targets = buffer.advantages(1.0, .95)
    np.testing.assert_allclose(advantages, [-4.0, -5.0])
    np.testing.assert_allclose(targets, [-3.0, -3.0])


def test_joint_ppo_runs_and_updates_both_components(config):
    config["method"] = "joint_ppo"
    config["training"].update(pretrain_slots=8, fast_slots=16, slow_intervals=4, cycles=1, calibration_episodes=1)
    config["k"] = 1
    config["deadline_range"] = [.95, 1.0]
    config["validation"]["enabled"] = False
    config["runtime"]["collection_workers"] = 0
    config["ppo"]["epochs"] = 1
    trainer = Trainer(config, 91)
    trainer.run()
    try:
        assert trainer.completed_stages == ["pretrain", "cycle_000_joint"]
        assert trainer.versions == {"upper": 1, "lower": 2}
        row = trainer.history[-1]
        assert row["layer"] == "joint"
        assert row["event_counts"] == {"upper": 4, "lower": 16}
        assert row["parameters_changed"]
        assert row["components"]["upper"]["parameter_delta_l2"]["actor"] > 0
        assert row["components"]["lower"]["parameter_delta_l2"]["critic"] > 0
        assert row["components"]["lower"]["parameter_delta_l2"]["actor"] > 0
    finally:
        trainer.close()


@pytest.mark.parametrize("count", [4, 6, 19])
def test_joint_boundaries_masks_and_physical_accounting(config, count):
    config["method"] = "joint_ppo"
    config["horizon"] = 17
    config["k"] = 1
    trainer = Trainer(config, 323)
    trainer.calibrate()
    buffer, records = trainer.collect("joint", count)
    assert len(records) == count
    assert len(buffer.samples) == event_count(config, count)
    assert sum(sample["duration"] for sample in buffer.samples) == count
    assert sum(sum(sample["physical_rewards"]) for sample in buffer.samples) == pytest.approx(sum(row["reward"] for row in records))
    check_physics(config, records)
    for index, sample in enumerate(buffer.samples[:-1]):
        following = buffer.samples[index + 1]
        if sample["terminated"]:
            assert sample["next_value"] == 0
            assert following["episode_id"] != sample["episode_id"]
        else:
            assert sample["next_phase"] == following["phase"]
            assert sample["next_observation"] == following["frame"]
            assert sample["next_value"] == following["value"]
            assert following["slot"] == sample["slot"] + sample["duration"]
        for step in sample["steps"]:
            assert step["mask"][step["action"]]
    with torch.no_grad():
        actual = replay_chunks(trainer.joint_policy, buffer.samples, 4)
    torch.testing.assert_close(actual, torch.tensor([sample["logprob"] for sample in buffer.samples]), atol=1e-5, rtol=0)
    last = buffer.samples[-1]
    assert last["truncated"]
    assert last["next_phase"] == ("upper" if count % 17 % 4 == 0 else "lower")
    expected = np.zeros(len(buffer.samples))
    for start in range(len(buffer.samples)):
        coefficient = 1.0
        for sample in buffer.samples[start:]:
            delta = sample["reward"] - sample["value"] + .9 ** sample["duration"] * sample["next_value"] * (not sample["terminated"])
            expected[start] += coefficient * delta
            if sample["terminated"] or sample["truncated"]:
                break
            coefficient *= (.9 * .7) ** sample["duration"]
    np.testing.assert_allclose(buffer.advantages(.9, .7)[0], expected)
    snapshot = buffer.behavior_version
    with pytest.raises(RuntimeError, match="Stale"):
        buffer.consume((snapshot[0] + 1, snapshot[1]), snapshot)


def test_joint_parallel_snapshot_and_restore(config, tmp_path):
    config["method"] = "joint_ppo"
    config["horizon"] = 5
    config["runtime"]["collection_workers"] = 2
    config["reward_normalization"]["enabled"] = True
    trainer = Trainer(config, 91)
    trainer.calibrate()
    checkpoint = tmp_path / "joint.pt"
    trainer.save(checkpoint)
    restored = Trainer.load(checkpoint)
    restored.config["runtime"]["collection_workers"] = 1
    try:
        buffer, records = trainer.collect("joint", 13)
        other, other_records = restored.collect("joint", 13)
        assert_close(buffer.samples, other.samples)
        clean = lambda rows: [{key: value for key, value in row.items() if key != "measured_allocation_seconds"} for row in rows]
        assert_close(clean(records), clean(other_records))
        assert trainer.total_slots == 13
        assert trainer.training_episodes == 2
        assert len(buffer.samples) == 18
    finally:
        trainer.close()
        restored.close()


def test_joint_saved_audit_results_and_resume(config, tmp_path):
    config["method"] = "joint_ppo"
    config["horizon"] = 5
    config["k"] = 1
    config["training"].update(pretrain_slots=5, fast_slots=8, cycles=1, calibration_episodes=1)
    config["validation"]["enabled"] = False
    config["monitoring"]["console"] = False
    config["reward_normalization"]["enabled"] = True
    config["ppo"].update(epochs=2, minibatch=4)
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    store = RunStore(root, config, "joint-test", 91, output_dir=tmp_path / "run")
    trainer = Trainer(config, 91, store)
    trainer.run()
    store.finish()
    samples = read_behavior(store.path / "traces/phases/cycle_000_joint/behavior.json")
    audit = json.loads((store.path / "traces/phases/cycle_000_joint/update.json").read_text(encoding="utf-8"))
    check_update(samples, audit)
    with (store.result_path / "cycles.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert int(rows[0]["physical_slots"]) == 8
    assert float(rows[0]["reward_per_slot"]) == trainer.history[-1]["raw_reward_sum"] / 8
    assert not (store.result_path / "validation.csv").exists()
    assert trainer.reward_normalizer.state_dict()["count"] == 18
    restored = Trainer.load(store.path / "checkpoints/final.pt")
    trace = generate_trace(config, config["seeds"]["test"])
    _, metrics = evaluate(trainer.controller, config, trace)
    _, actual = evaluate(restored.controller, config, trace)
    assert_close(metrics, actual)
    restored.run(extra_cycles=1)
    expected = copy.deepcopy(restored.controller.state_dict())
    repeat = Trainer.load(store.path / "checkpoints/final.pt")
    repeat.run(extra_cycles=1)
    for phase in expected:
        for name in expected[phase]:
            torch.testing.assert_close(repeat.controller.state_dict()[phase][name], expected[phase][name], rtol=0, atol=0)
    assert repeat.total_slots == 21


def test_joint_sparse_diagnostics_survive_first_joint_eviction(config, tmp_path):
    from pathlib import Path
    config["method"] = "joint_ppo"
    config["horizon"] = 4
    config["k"] = 1
    config["training"].update(pretrain_slots=32, fast_slots=4, cycles=23, calibration_episodes=1)
    config["validation"]["enabled"] = False
    config["monitoring"] = dict(tensorboard=False, console=False)
    config["runtime"]["collection_workers"] = 0
    config["ppo"].update(epochs=1, minibatch=16)
    config["checkpoint_policy"].update(mode="minimal", cycle_interval=250)
    config["trace_retention"] = dict(phases=2, validations=2, inputs=2)
    config["diagnostics"]["phase_interval"] = 10
    store = RunStore(Path(__file__).resolve().parents[1], config, "joint-retention-test", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    trainer.run()
    store.finish()
    assert len(trainer.cycle_history) == 23
    assert trainer.total_slots == 32 + 23 * 4
    assert {path.name for path in (store.path / "traces/phases").iterdir()} == {"cycle_012_joint", "cycle_022_joint"}
    with (store.result_path / "cycles.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 23
    with (store.path / "metrics.jsonl").open(encoding="utf-8") as stream:
        assert len(stream.readlines()) == 31
    restored = Trainer.load(store.path / "checkpoints/last.pt")
    restored.run(extra_cycles=1)
    assert len(restored.cycle_history) == 24
    assert restored.total_slots == trainer.total_slots + 4


def test_joint_schedule_and_console(config):
    from mist_sim.monitoring import ConsoleProgress
    config["method"] = "joint_ppo"
    config["horizon"] = 5
    config["training"]["fast_slots"] = 13
    config["validation"]["enabled"] = False
    schedule = describe_schedule(config)
    assert schedule["joint_event_transitions_per_rollout"] == 18
    assert schedule["planned_training_slots"] == config["training"]["pretrain_slots"] + config["training"]["cycles"] * 13
    assert list(schedule["ppo"]) == ["joint"]
    stream = io.StringIO()
    progress = ConsoleProgress(stream)
    progress.record("training_started", dict(schedule=schedule))
    progress.record("phase_started", dict(stage="cycle_000_joint"))
    assert "[1/1]" in stream.getvalue()
    assert "联合事件PPO" in stream.getvalue()


def test_paired_methods_share_initialization_and_budget(config, tmp_path):
    from pathlib import Path
    from mist_sim.comparison import run_pair
    config["horizon"] = 4
    config["k"] = 1
    config["monitoring"]["console"] = False
    config["training"].update(pretrain_slots=4, fast_slots=4, slow_intervals=1, calibration_episodes=1)
    config["ppo"].update(epochs=1, minibatch=4)
    root = Path(__file__).resolve().parents[1]
    reports = run_pair(config, 91, 8, root, output_dir=tmp_path, evaluation_trajectories=1)
    assert [report["method"] for report in reports] == ["hpgi", "joint_ppo"]
    assert [report["post_pretrain_slots"] for report in reports] == [8, 8]
    assert [report["iterations"] for report in reports] == [1, 2]
    assert [report["training_transitions"] for report in reports] == [5, 10]
    assert [report["sample_presentations"] for report in reports] == [5, 10]
    assert [report["optimizer_steps"] for report in reports] == [2, 4]
    assert all(report["collector_start"] == "cold" for report in reports)
    assert reports[0]["common_checkpoint_sha256"] == reports[1]["common_checkpoint_sha256"]
    assert all(np.isfinite(report["final_development_reward_per_slot"]) for report in reports)
