import copy
import json
from unittest.mock import Mock
import pytest
import torch
from mist_sim.anonymity import generate_trace
from mist_sim.environment import Environment
from mist_sim.training import Trainer
from mist_sim.policies.batching import RolloutTensors, replay_batch
from mist_sim.traces import pack_behavior, read_behavior
from mist_sim.storage import write_json


@pytest.mark.parametrize("rates", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_action_only_preserves_actions_rng_and_skips_value(config, rates, deterministic):
    config["learned_rates"] = rates
    config["k"] = 1
    trainer = Trainer(config, 37)
    env = Environment(config, generate_trace(config, 41))
    observation = env.upper_observation()
    for policy, method, argument in [(trainer.controller.upper, "upper_action", observation),
                                     (trainer.controller.lower, "lower_action", env.begin_slot([1, 1]))]:
        rng = torch.get_rng_state()
        full = getattr(policy, method)(argument, deterministic)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        original = policy.value
        policy.value = Mock(side_effect=AssertionError("Unused critic"))
        light = getattr(policy, method)(argument, deterministic, record=False)
        policy.value = original
        assert full[:-1] == light[:-1]
        assert light[-1] is None
        assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("layer,count", [("lower", 19), ("upper", 6)])
def test_successor_value_computed_once(config, layer, count):
    config["horizon"] = 17
    trainer = Trainer(config, 93)
    policy = getattr(trainer.controller, layer)
    opponent = getattr(trainer.controller, "upper" if layer == "lower" else "lower")
    policy.value = Mock(wraps=policy.value)
    opponent.value = Mock(side_effect=AssertionError("Frozen critic"))
    buffer, _ = trainer.collect(layer, count, pretrain=layer == "lower")
    assert policy.value.call_count == count + int(buffer.samples[-1]["truncated"])
    for current, following in zip(buffer.samples, buffer.samples[1:]):
        if not current["terminated"]:
            assert current["next_observation"] is following["frame"]
            assert current["next_value"] == following["value"]


def test_cached_inputs_refresh_activations_after_parameter_change(config):
    trainer = Trainer(config, 100)
    buffer, _ = trainer.collect("upper", 3)
    policy = trainer.controller.upper
    cache = RolloutTensors(policy, buffer.samples)
    for _ in range(2):
        plain = replay_batch(policy, buffer.samples)
        prepared = replay_batch(policy, buffer.samples, cache=cache)
        for first, second in zip(plain, prepared):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        policy.zero_grad(set_to_none=True)
        sum(value.sum() for value in plain).backward()
        gradients = [p.grad.clone() if p.grad is not None else None for p in policy.parameters()]
        policy.zero_grad(set_to_none=True)
        sum(value.sum() for value in prepared).backward()
        for parameter, gradient in zip(policy.parameters(), gradients):
            if gradient is not None:
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(.001)


def test_shared_trace_roundtrip_and_legacy(config, tmp_path):
    trainer = Trainer(config, 202)
    buffer, _ = trainer.collect("upper", 4)
    original = copy.deepcopy(buffer.samples)
    path = tmp_path / "behavior.json"
    packed = pack_behavior(buffer.samples)
    write_json(path, packed)
    restored = read_behavior(path)
    assert restored == original
    assert len(json.dumps(packed)) < len(json.dumps(original))
    assert restored[0]["frame"]["items"] is restored[0]["steps"][0]["prefix_frame"]["items"]
    write_json(path, original)
    assert read_behavior(path) == original


def test_stream_json_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "atomic.json"
    write_json(path, {"valid": True})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}
    assert not path.with_name("atomic.json.partial").exists()


def test_checkpoint_history_omits_full_diagnostics(config, tmp_path):
    trainer = Trainer(config, 109)
    metrics = trainer.phase("history", "lower", 3, pretrain=True)
    assert "input_diagnostics" in metrics and "examples" in metrics["rollout"]
    checkpoint = tmp_path / "last.pt"
    trainer.save(checkpoint)
    restored = Trainer.load(checkpoint)
    row = restored.history[0]
    assert "input_diagnostics" not in row and "durations" not in row
    assert "examples" not in row["rollout"]
    assert row["samples"] == 3 and row["rollout"]["behavior_version"] == 0
