import copy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mist_sim.config import for_method, layer_settings
from mist_sim.diagnostics import check_update, inspect_run
from mist_sim.ppo import update
from mist_sim.storage import RunStore
from mist_sim.training import Trainer


@pytest.mark.parametrize("method,workers", [("hpgi", 0), ("joint_ppo", 0),
                                           ("single_timescale", 2), ("upper_ppo_greedy", 0)])
def test_disabled_exports_keep_training_buffer_results_and_resume(config, tmp_path, monkeypatch, method, workers):
    import csv
    import mist_sim.diagnostics as diagnostics
    from mist_sim.config import for_method
    config = for_method(config, method)
    config["diagnostics"]["phase_interval"] = 0
    config["runtime"]["collection_workers"] = workers
    config["horizon"] = 4
    config["k"] = 1
    config["training"].update(pretrain_slots=4, fast_slots=4, slow_intervals=4, cycles=1, calibration_episodes=1)
    config["validation"]["enabled"] = False
    config["monitoring"] = dict(console=False, tensorboard=False)
    config["ppo"].update(epochs=1, minibatch=8)
    config["checkpoint_policy"].update(mode="minimal", cycle_interval=10)
    store = RunStore(Path(__file__).resolve().parents[1], config, "exports-off", 91, output_dir=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Detailed debug export was executed")
    monkeypatch.setattr(diagnostics, "policy_input_diagnostics", forbidden)
    monkeypatch.setattr("mist_sim.training.pack_behavior", forbidden)
    monkeypatch.setattr("mist_sim.training_joint.pack_behavior", forbidden)
    trainer = Trainer(config, 91, store)
    buffers = []
    collect = trainer.collect
    def tracked(*args, **kwargs):
        buffer, records = collect(*args, **kwargs)
        buffers.append(buffer)
        assert buffer.samples
        return buffer, records
    monkeypatch.setattr(trainer, "collect", tracked)
    try:
        trainer.run()
        assert all(buffer.consumed and not buffer.samples for buffer in buffers)
        assert all(row["parameters_changed"] and not row["detailed_trace"] for row in trainer.history)
        assert not list((store.path / "traces").rglob("*.json"))
        assert (store.path / "input_seeds.jsonl").stat().st_size > 0
        assert (store.path / "metrics.jsonl").stat().st_size > 0
        with (store.result_path / "cycles.csv").open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 1
        restored = Trainer.load(store.path / "checkpoints/last.pt")
        assert len(restored.cycle_history) == 1
        restored.close()
    finally:
        trainer.close()
        store.finish()


@pytest.mark.parametrize("layer", ["upper", "lower"])
def test_hpgi_detailed_inputs_follow_diagnostic_interval(config, monkeypatch, layer):
    import mist_sim.diagnostics as diagnostics
    config["diagnostics"]["phase_interval"] = 2
    config["ppo"]["epochs"] = 1
    describe = Mock(wraps=diagnostics.policy_input_diagnostics)
    monkeypatch.setattr(diagnostics, "policy_input_diagnostics", describe)
    trainer = Trainer(config, 91)
    rows = [trainer.phase(f"probe_{index}", layer, 2) for index in range(3)]
    assert describe.call_count == 2
    assert [bool(row["input_diagnostics"]) for row in rows] == [True, False, True]
    assert all(layer in row["rollout"]["learning_signal"] for row in rows)
    assert all(row["frozen_unchanged"] and row["behavior_logprob_error"] < 1e-5 for row in rows)


def test_hpgi_diagnostic_sampling_does_not_skip_one_component(config):
    config["diagnostics"]["phase_interval"] = 2
    config["ppo"]["epochs"] = 1
    trainer = Trainer(config, 91)
    rows = [trainer.phase(f"probe_{index}_{layer}", layer, 2)
            for index in range(3) for layer in ("upper", "lower")]
    for layer in ("upper", "lower"):
        assert [row["detailed_trace"] for row in rows if row["layer"] == layer] == [True, False, True]


@pytest.mark.parametrize("layer", ["upper", "lower"])
def test_hpgi_phase_normalization_is_already_layer_local(config, layer):
    trainer = Trainer(config, 91)
    buffer, _ = trainer.collect(layer, 5)
    samples = list(buffer.samples)
    reports, weights = [], []
    for normalization in ("global", "phase"):
        policy = copy.deepcopy(getattr(trainer.controller, layer))
        settings = dict(layer_settings(config, layer), epochs=1, advantage_normalization=normalization)
        optimizer = torch.optim.Adam(policy.parameters(), lr=settings["learning_rate"], eps=settings["adam_epsilon"])
        audit = {}
        torch.manual_seed(71)
        update(policy, optimizer, copy.deepcopy(buffer), settings, 0, 0, audit)
        check_update(samples, audit)
        reports.append(audit)
        weights.append(copy.deepcopy(policy.state_dict()))
    for key in ("advantages", "normalized_advantages", "returns"):
        np.testing.assert_array_equal(reports[0][key], reports[1][key])
    for name, value in weights[0].items():
        torch.testing.assert_close(value, weights[1][name], atol=0, rtol=0)


def test_comparison_accepts_shared_acceleration_overlay(monkeypatch):
    from mist_sim import comparison
    run_pair = Mock()
    monkeypatch.setattr(comparison, "run_pair", run_pair)
    monkeypatch.setattr("sys.argv", ["compare", "--config", "configs/smoke.json", "--slots", "128",
        "--overlay", "configs/algorithms/cuda_update.json"])
    comparison.main()
    config = run_pair.call_args.args[0]
    assert config["method"] == "hpgi"
    assert config["device"] == "cuda"
    assert config["runtime"]["collection_workers"] == 4
    assert config["ppo"]["minibatch"] == 128


@pytest.mark.parametrize("workers", [0, 2])
def test_hpgi_finite_value_tail_discount_freezing_and_resume(config, tmp_path, monkeypatch, workers):
    config["device"] = "cpu"
    config["runtime"]["collection_workers"] = workers
    config["horizon"], config["window"], config["k"] = 5, 4, 1
    config["reward_normalization"]["enabled"] = True
    config["validation"]["enabled"] = False
    config["ppo"].update(value_model="finite_horizon", value_target="sampled_return",
                         gamma=.97, epochs=1, minibatch=8)
    config["training"].update(pretrain_slots=6, fast_slots=6, slow_intervals=3,
                               cycles=2, calibration_episodes=1)
    config["trace_retention"].update(phases=None, inputs=None)
    store = RunStore(Path(__file__).resolve().parents[1], config, "check hpgi finite", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    batches = []
    original_collect = trainer.collect

    def collect(layer, count, pretrain=False):
        buffer, records = original_collect(layer, count, pretrain)
        batches.append((layer, buffer.behavior_version, buffer.opponent_version, copy.deepcopy(buffer.samples)))
        # The interval reward must contain each physical reward exactly once,
        # with its physical-slot discount and this batch's frozen reward units.
        for sample in buffer.samples:
            rewards = np.asarray(sample["physical_rewards"]) * sample["reward_scale"]
            if sample["reward_clip"] is not None:
                rewards = np.clip(rewards, -sample["reward_clip"], sample["reward_clip"])
            expected = sum(config["ppo"]["gamma"] ** i * reward for i, reward in enumerate(rewards))
            assert sample["reward"] == pytest.approx(expected)
            assert len(rewards) == sample["duration"]
        return buffer, records

    monkeypatch.setattr(trainer, "collect", collect)
    restored = None
    try:
        trainer.run()
        assert trainer.total_slots == 36
        assert trainer.versions == {"upper": 2, "lower": 3}
        assert [(layer, behavior, opponent) for layer, behavior, opponent, _ in batches] == [
            ("lower", 0, 0), ("upper", 0, 1), ("lower", 1, 1),
            ("upper", 1, 2), ("lower", 2, 2)]
        for layer, _, _, samples in batches:
            if layer == "upper":
                assert [sample["duration"] for sample in samples] == [4, 1, 4]
            assert samples[-1]["truncated"] and not samples[-1]["terminated"]
            assert samples[-1]["next_value"] < 0  # truncated fragments retain bootstrap
            assert all(sample["next_value"] == 0 for sample in samples if sample["terminated"])
            assert all(sample["value"] < 0 for sample in samples)  # past negative reward-rate baseline
        assert all(row["frozen_unchanged"] and row["parameters_changed"] for row in trainer.history)
        assert all(row["behavior_logprob_error"] < 1e-5 for row in trainer.history)
        assert all(row["value_target"] == "sampled_return" for row in trainer.history)
        assert len(inspect_run(store.path)["updates"]) == 5  # independent physical/GAE/loss arithmetic
        checkpoint = tmp_path / "finite_hpgi.pt"
        trainer.save(checkpoint)
        restored = Trainer.load(checkpoint)
        assert restored.reward_normalizer.state_dict() == trainer.reward_normalizer.state_dict()
        for layer in ("upper", "lower"):
            for key, value in getattr(trainer.controller, layer).state_dict().items():
                torch.testing.assert_close(value, getattr(restored.controller, layer).state_dict()[key])
        restored.run(extra_cycles=1)
        assert restored.total_slots == 51
        assert restored.versions == {"upper": 3, "lower": 4}
    finally:
        trainer.close()
        if restored is not None:
            restored.close()
        store.finish()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("method", ["hpgi", "harp"])
def test_hierarchical_cuda_four_workers_freezing_fresh_buffers_and_resume(config, tmp_path, monkeypatch, method):
    config = for_method(config, method)
    config["device"] = "cuda"
    config["runtime"]["collection_workers"] = 4
    config["horizon"] = 5
    config["k"] = 1
    config["reward_normalization"]["enabled"] = True
    config["validation"]["enabled"] = False
    config["ppo"].update(epochs=1, minibatch=8)
    config["training"].update(pretrain_slots=20, fast_slots=20, slow_intervals=8, cycles=1, calibration_episodes=1)
    config["checkpoint_policy"]["mode"] = "minimal"
    store = RunStore(Path(__file__).resolve().parents[1], config, "check hpgi cuda", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    retained = []
    original_collect = trainer.collect

    def collect(*args, **kwargs):
        buffer, records = original_collect(*args, **kwargs)
        retained.append(buffer)
        return buffer, records

    monkeypatch.setattr(trainer, "collect", collect)
    restored = None
    try:
        trainer.run()
        assert trainer.total_slots == 60
        assert trainer.versions == {"upper": 1, "lower": 2}
        assert all(row["frozen_unchanged"] and row["parameters_changed"] for row in trainer.history)
        assert all(row["device_metrics"]["update_device"] == "cuda" for row in trainer.history)
        assert all(buffer.consumed and not buffer.samples for buffer in retained)
        assert [(buffer.behavior_version, buffer.opponent_version) for buffer in retained] == [(0, 0), (0, 1), (1, 1)]
        with pytest.raises(RuntimeError, match="Stale"):
            retained[1].consume(1, 2)
        report = inspect_run(store.path)
        assert len(report["updates"]) == 3
        assert len(trainer.collection_timings["jobs"]) == 4
        restored = Trainer.load(store.path / "checkpoints/last.pt")
        restored.run(extra_cycles=1)
        assert restored.total_slots == 100
        assert restored.versions == {"upper": 2, "lower": 3}
        for policy in (restored.controller.upper, restored.controller.lower):
            assert next(policy.parameters()).device.type == "cpu"
        for optimizer in restored.optimizers.values():
            assert all(value.device.type == "cpu" for state in optimizer.state.values()
                       for value in state.values() if isinstance(value, torch.Tensor))
    finally:
        trainer.close()
        if restored is not None:
            restored.close()
        store.finish()
