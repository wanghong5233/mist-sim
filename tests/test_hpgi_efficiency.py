import copy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mist_sim.config import layer_settings
from mist_sim.diagnostics import check_update, inspect_run
from mist_sim.ppo import update
from mist_sim.storage import RunStore
from mist_sim.training import Trainer


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hpgi_cuda_four_workers_freezing_fresh_buffers_and_resume(config, tmp_path, monkeypatch):
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
