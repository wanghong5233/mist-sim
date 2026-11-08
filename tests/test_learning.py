import copy
import math
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mist_sim.anonymity import generate_trace
from mist_sim.buffers import RolloutBuffer
from mist_sim.config import METHODS, for_method
from mist_sim.environment import Environment
from mist_sim.evaluation import evaluate, assert_close
from mist_sim.policies.hpgi import Controller
from mist_sim.policies.networks import upper_frame
from mist_sim.training import Trainer, parameter_hash


def test_gae_terminal_truncated_partial():
    buffer = RolloutBuffer("upper", 0, 0)
    buffer.append({"value": 1}, 2, 4, False, False, 4)
    buffer.append({"value": 4}, 3, 0, True, False, 1)
    buffer.append({"value": 2}, 1, 7, False, True, 2)
    advantages, returns = buffer.advantages(.9, .95)
    last = 1 + .9 ** 2 * 7 - 2
    middle = 3 - 4
    first = 2 + .9 ** 4 * 4 - 1 + .9 ** 4 * .95 * middle
    np.testing.assert_allclose(advantages, [first, middle, last])
    with pytest.raises(RuntimeError, match="Stale"):
        buffer.consume(0, 1)
    buffer.consume(0, 0)
    with pytest.raises(RuntimeError, match="Stale"):
        buffer.consume(0, 0)


def test_joint_ratio_masks_and_no_resampling(config):
    config["k"] = 1
    torch.set_num_threads(1)
    torch.manual_seed(4)
    controller = Controller(config)
    environment = Environment(config, generate_trace(config, 45))
    _, upper = controller.slow(environment.upper_observation())
    assert float(controller.upper.replay(upper)[0]) == pytest.approx(upper["logprob"], abs=1e-6)
    problem = environment.begin_slot([1, 1])
    _, _, sample = controller.fast(problem)
    logprob, _, _ = controller.lower.replay(sample)
    assert math.exp(float(logprob) - sample["logprob"]) == pytest.approx(1.0)
    hand = 0.0
    encoding = controller.lower.encode(sample["frame"])
    for step in sample["steps"]:
        conditional, servers = controller.lower.encode(step["prefix_frame"], with_servers=True)
        distribution, _ = controller.lower.distribution(conditional, step["features"], step["prefix"],
                                                        step["mask"], servers, step["targets"])
        hand += math.log(float(distribution.probs[step["action"]]))
        for legal, probability in zip(step["mask"], distribution.probs):
            if not legal:
                assert float(probability) == 0
    assert float(logprob) == pytest.approx(hand, abs=1e-6)
    features = [[0.0] * 19, [1.0] * 19]
    distribution, _ = controller.lower.distribution(encoding, features, [0.0] * 7, [True, False])
    assert distribution.probs.tolist() == [1, 0]


def test_empty_actor_critic_sample(config):
    config["arrival_probability"] = 0
    trainer = Trainer(config, 1)
    trainer.calibrate()
    actor_before = {name: tensor.clone() for name, tensor in trainer.controller.lower.named_parameters()
                    if not name.startswith(("critic", "value"))}
    metrics = trainer.phase("empty", "lower", 3, pretrain=True)
    assert metrics["parameters_changed"] and metrics["frozen_unchanged"]
    for name, tensor in trainer.controller.lower.named_parameters():
        if name in actor_before:
            torch.testing.assert_close(tensor, actor_before[name], rtol=0, atol=0)


def test_slow_interval_discount_and_frozen_normalization(config):
    config["ppo"]["gamma"] = .9
    trainer = Trainer(config, 31)
    trainer.calibrate()
    normalizer = trainer.controller.lower.item_norm.mean.clone()
    buffer, records = trainer.collect("upper", 1)
    expected = sum(.9 ** offset * record["reward"] for offset, record in enumerate(records))
    assert buffer.samples[0]["reward"] == pytest.approx(expected)
    assert buffer.samples[0]["duration"] == 4
    assert buffer.samples[0]["truncated"]
    torch.testing.assert_close(normalizer, trainer.controller.lower.item_norm.mean, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="frozen"):
        trainer.controller.lower.item_norm.fit([[0] * 19])


def test_behavior_trace_serialization(config):
    trainer = Trainer(config, 1000)
    trainer.calibrate()
    buffer, _ = trainer.collect("lower", 32, pretrain=True)
    encoded = json.dumps(buffer.samples, allow_nan=False)
    assert len(json.loads(encoded)) == 32


def test_learned_rate_joint_density(config):
    from mist_sim.policies.networks import lower_frame, item_features
    from tests.test_physics import request
    config = for_method(config, "fixed_window")
    config["k"] = 1
    config["horizon"] = 1
    config["initial_active"] = [1, 1]
    environment = Environment(config, [[request()]])
    problem = environment.begin_slot([1, 1])
    controller = Controller(config)
    features = [item_features(choice, config) for choice in problem.choices[0]]
    action = next(index for index, choice in enumerate(problem.choices[0]) if choice["kind"] == "edge")
    frame = lower_frame(problem)
    sample = {"frame": frame, "steps": [{"features": features, "prefix": [0] * 7,
              "mask": [True] * len(features), "action": action, "rate_latent": .3}]}
    logprob, _, _ = controller.lower.replay(sample)
    distribution, combined = controller.lower.distribution(controller.lower.encode(frame), features, [0] * 7, [True] * len(features))
    mean = controller.lower.rate_head(combined[action]).squeeze()
    expected = distribution.log_prob(torch.tensor(action)) + torch.distributions.Normal(mean, controller.lower.rate_log_std.exp()).log_prob(torch.tensor(.3))
    torch.testing.assert_close(logprob, expected)
    logprob.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in controller.lower.rate_head.parameters())


def test_partial_interval_and_checkpoint_resume(config, tmp_path):
    config["horizon"] = 17
    trainer = Trainer(config, 7)
    trainer.calibrate()
    buffer, _ = trainer.collect("upper", 6)
    assert [sample["duration"] for sample in buffer.samples] == [4, 4, 4, 4, 1, 4]
    assert buffer.samples[4]["terminated"] and buffer.samples[-1]["truncated"]
    trainer.phase("pretrain", "lower", 4, pretrain=True)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save(checkpoint)
    trace = generate_trace(config, 3000)
    before, metrics_before = evaluate(trainer.controller, config, trace)
    trainer.phase("next", "upper", 2)
    continued_state = copy.deepcopy(trainer.controller.state_dict())
    resumed = Trainer.load(checkpoint)
    after, metrics_after = evaluate(resumed.controller, config, trace)
    assert_close(metrics_before, metrics_after)
    assert [record["actions"] for record in before] == [record["actions"] for record in after]
    resumed.phase("next", "upper", 2)
    for layer in ("upper", "lower"):
        for name, value in continued_state[layer].items():
            torch.testing.assert_close(value, resumed.controller.state_dict()[layer][name], rtol=0, atol=0)
    modified = copy.deepcopy(config)
    modified["rejection_cost"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.load(checkpoint, modified)


@pytest.mark.parametrize("method", METHODS)
def test_method_registry(config, method):
    configured = for_method(config, method)
    controller = Controller(configured)
    assert controller.learn_lower == (method not in ("upper_ppo_greedy", "greedy_consolidation"))
    assert controller.learn_upper == (method not in ("always_on", "greedy_consolidation"))
    if method == "single_timescale":
        assert configured["window"] == 1
    if method == "fixed_window":
        assert controller.lower.learned_rates and not configured["capacity_mask"]
    if method == "nominal":
        assert configured["information"] == "nominal"
    if method == "full_information":
        assert configured["information"] == "full"
    with pytest.raises(ValueError):
        for_method(config, "unimplemented")
