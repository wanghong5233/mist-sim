from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.environment import Environment
from mist_sim.policies.networks import SetPolicy, lower_frame
from mist_sim.policies import batching
from mist_sim.training import Trainer
from mist_sim.allocation import solve_p3


def test_cached_request_encoding_does_not_repack_candidates(config, monkeypatch):
    environment = Environment(config, generate_trace(config, 91))
    frame = lower_frame(environment.begin_slot(config["initial_active"]))
    policy = SetPolicy(16)
    original = batching.tensor
    calls = Mock(wraps=original)
    monkeypatch.setattr(batching, "tensor", calls)
    cache = {}
    with torch.no_grad():
        expected = policy.encode(frame, request_cache=cache)
        first = calls.call_count
        calls.reset_mock()
        actual = policy.encode(frame, request_cache=cache)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert calls.call_count < first


def test_environment_reuses_validated_profiles(config, monkeypatch):
    environment = Environment(config, generate_trace(config, 91))
    import mist_sim.environment as module
    monkeypatch.setattr(module, "Profiles", Mock(side_effect=AssertionError("Repeated profile validation")))
    problem = environment.begin_slot(config["initial_active"])
    environment.execute([0] * len(problem.public))
    environment.begin_slot(config["initial_active"])


def test_supplied_trace_not_regenerated(config, monkeypatch):
    trainer = Trainer(config, 91)
    trace = generate_trace(config, 91)
    trainer.collection_trace = trace
    import mist_sim.training as module
    monkeypatch.setattr(module, "generate_trace", Mock(side_effect=AssertionError("Repeated trace generation")))
    environment = trainer.fresh_environment()
    assert environment._trace == trace
    assert trainer.collection_trace is None


def test_p3_wide_scales_satisfy_capacity_and_kkt():
    generator = np.random.default_rng(177)
    for _ in range(80):
        count = int(generator.integers(2, 13))
        alpha = 10. ** generator.uniform(-6, -1, count)
        beta = 10. ** generator.uniform(-1, 3, count)
        minimum = 10. ** generator.uniform(-1, 1, count)
        unconstrained = np.maximum(minimum, np.cbrt(beta / (2 * alpha)))
        capacity = float(minimum.sum() + .4 * (unconstrained.sum() - minimum.sum()))
        rates = solve_p3(alpha, beta, minimum, capacity)
        assert rates.sum() <= capacity + 1e-9
        assert np.all(rates >= minimum)
        free = rates > minimum + 1e-8
        multipliers = beta / rates ** 2 - 2 * alpha * rates
        if free.any():
            np.testing.assert_allclose(multipliers[free], np.mean(multipliers[free]), rtol=1e-8, atol=1e-8)
            assert np.all(multipliers[~free] <= np.mean(multipliers[free]) + 1e-8)


def test_profile_bound_cache_is_private(config):
    from mist_sim.profiles import Profiles
    profiles = Profiles(config)
    model = next(iter(config["profiles"]))
    sizes = profiles.sizes(model)
    first = profiles.bound(model, [min(sizes), max(sizes)], None, "robust")
    expected = first["local_work"]
    first["local_work"] = -1
    assert profiles.bound(model, [min(sizes), max(sizes)], None, "robust")["local_work"] == expected


def test_learning_signal_reports_preupdate_time_bias(config):
    from mist_sim.buffers import EventRolloutBuffer
    from mist_sim.diagnostics import rollout_diagnostics
    buffer = EventRolloutBuffer(dict(upper=0, lower=0))
    frame = dict(items=[], servers=[], context=[0.] * 7)
    for slot, value in enumerate((0., -1., -2.)):
        sample = dict(phase="lower", slot=slot, value=value, frame=frame, steps=[], physical_rewards=[-1.])
        buffer.append(sample, -1., 0., True, False)
    signal = rollout_diagnostics(buffer, config["ppo"])["learning_signal"]["lower"]
    assert signal["advantage_slot_correlation"] == 1.
    assert signal["positive_fraction"] == 1 / 3
    assert signal["pre_update_explained_variance"] is None


def test_monitor_receives_collection_and_learning_signal():
    from mist_sim.monitoring import TrainingMonitor
    monitor = TrainingMonitor(None, False)
    monitor.scalars = Mock()
    monitor.record("rollout_ready", dict(total_slots=16, rollout=dict(learning_signal={"upper": {"advantage_std": .1}})))
    assert monitor.scalars.call_args.args == ("signal/upper", {"advantage_std": .1}, 16)
    monitor.record("collection_timing", dict(total_slots=16, layer="joint", preparation_seconds=1., pool_seconds=3.,
        jobs=[dict(worker_seconds=2., worker_cpu_seconds=1., allocation_seconds=.3, return_seconds=.5)]))
    assert monitor.scalars.call_args.args[1]["worker_cpu_seconds_max"] == 1.


def test_phase_normalization_removes_component_offset_without_changing_targets():
    from mist_sim.buffers import normalize_advantages
    samples = [{"phase": "upper"}, {"phase": "lower"}, {"phase": "upper"}, {"phase": "lower"}]
    raw = np.asarray([-101., 49., -99., 51.])
    expected = np.asarray([-1., -1., 1., 1.])
    np.testing.assert_array_equal(normalize_advantages(raw, samples, "phase"), expected)
    np.testing.assert_array_equal(raw, [-101., 49., -99., 51.])
    np.testing.assert_allclose(normalize_advantages(raw, samples), (raw - raw.mean()) / raw.std())


def test_phase_normalization_joint_update_independently_recomputes(config):
    from mist_sim.diagnostics import check_update
    from mist_sim.ppo import update
    config["method"] = "joint_ppo"
    config["ppo"]["advantage_normalization"] = "phase"
    config["ppo"]["epochs"] = 1
    trainer = Trainer(config, 91)
    buffer, _ = trainer.collect("joint", config["horizon"])
    samples = list(buffer.samples)
    audit = {}
    snapshot = (0, 0)
    update(trainer.joint_policy, trainer.optimizers["joint"], buffer, config["ppo"], snapshot, snapshot, audit)
    check_update(samples, audit)
    phases = np.asarray([sample["phase"] for sample in samples])
    normalized = np.asarray(audit["normalized_advantages"])
    for phase in ("upper", "lower"):
        assert abs(normalized[phases == phase].mean()) < 1e-10
        np.testing.assert_allclose(normalized[phases == phase].std(), 1.)


def test_normalization_configuration_and_schedule(config):
    from mist_sim.config import load_config, validate
    from mist_sim.schedule import describe_schedule
    loaded = load_config("configs/smoke.json", overlays=["configs/algorithms/joint_phase_normalized.json"])
    assert loaded["method"] == "joint_ppo"
    assert loaded["ppo"]["advantage_normalization"] == "phase"
    assert describe_schedule(loaded)["ppo"]["joint"]["advantage_normalization"] == "phase"
    assert config["ppo"]["advantage_normalization"] == "global"
    config["ppo"]["advantage_normalization"] = "unknown"
    with pytest.raises(ValueError, match="advantage normalization"):
        validate(config)


def test_phase_normalization_small_groups():
    from mist_sim.buffers import normalize_advantages
    samples = [{"phase": "upper"}, {"phase": "lower"}, {"phase": "lower"}]
    np.testing.assert_array_equal(normalize_advantages([3., 2., 2.], samples, "phase"), [3., 0., 0.])
