import copy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from mist_sim.anonymity import generate_trace
from mist_sim.config import load_config, for_method
from mist_sim.diagnostics import check_physics, check_update, policy_input_diagnostics
from mist_sim.environment import Environment
from mist_sim.policies.networks import Normalizer, SetPolicy, lower_frame, CONTEXT_DIM
from mist_sim.ppo import update
from mist_sim.training import Trainer, state_hash


def test_bounded_features_constant_and_out_of_range():
    normalizer = Normalizer(2)
    normalizer.fit([[1., 3.], [1., 5.]])
    values = torch.tensor([[1., -1e6], [100., 4.], [1., 1e6]])
    outputs = normalizer(values)
    assert torch.isfinite(outputs).all()
    assert outputs.abs().max() <= 1
    assert outputs[0, 1] < outputs[1, 1] < outputs[2, 1]
    assert outputs[1, 1] == 0
    assert normalizer(values).equal(outputs)
    with pytest.raises(ValueError, match="Nonfinite"):
        normalizer(torch.tensor([float("nan"), 1.]))


def test_masked_logits_have_zero_gradient_and_exact_joint_sum(config):
    trainer = Trainer(config, 94)
    environment = Environment(config, generate_trace(config, 95))
    problem = environment.begin_slot([1] * len(config["servers"]))
    policy = trainer.controller.lower
    encoding = policy.encode(lower_frame(problem))
    captured = []
    def capture(module, inputs, output):
        output.retain_grad()
        captured.append(output)
    handle = policy.score.register_forward_hook(capture)
    try:
        distribution, _ = policy.distribution(encoding, [[0.] * 19, [1.] * 19, [2.] * 19], [0.] * 7, [True, False, True])
        assert distribution.probs[1] == 0
        (-distribution.log_prob(torch.tensor(0))).backward()
        assert captured[-1].grad[1] == 0
        assert captured[-1].grad[0].abs() > 0
        assert captured[-1].grad[2].abs() > 0
    finally:
        handle.remove()
    _, sample = trainer.controller.upper.upper_action(environment.upper_observation())
    logprob, _, _ = trainer.controller.upper.replay(sample)
    terms = []
    for step in sample["steps"]:
        encoding, servers = trainer.controller.upper.encode(step["prefix_frame"], with_servers=True)
        distribution, _ = trainer.controller.upper.distribution(
            encoding, step["features"], step["prefix"], step["mask"], servers, step["targets"])
        terms.append(float(distribution.log_prob(torch.tensor(step["action"]))))
    assert float(logprob) == pytest.approx(sum(terms), abs=1e-6)
    assert math.exp(float(logprob) - sample["logprob"]) == pytest.approx(1., abs=1e-6)


def test_control_phase_exposed_and_clock_not_advanced(config):
    environment = Environment(config, generate_trace(config, 90))
    for slot in range(5):
        problem = environment.begin_slot([1] * len(config["servers"]))
        frame = lower_frame(problem)
        assert len(frame["context"]) == CONTEXT_DIM == 7
        assert frame["context"][-1] == (slot % config["window"]) / config["window"]
        trainer = Trainer(config, 91)
        actions, predicted, sample = trainer.controller.fast(problem)
        trainer.controller.lower.replay(sample)
        assert environment.slot == slot
        environment.execute(actions, predicted)
        assert environment.slot == slot + 1
    check_physics(config, environment.records)


def test_nonfinite_update_does_not_step_or_change_weights(config):
    trainer = Trainer(config, 93)
    buffer, _ = trainer.collect("lower", 3, pretrain=True)
    buffer.samples[0]["reward"] = float("nan")
    before = state_hash(trainer.controller.lower)
    with pytest.raises(FloatingPointError):
        update(trainer.controller.lower, trainer.optimizers["lower"], buffer, config["ppo"], 0, 0)
    assert state_hash(trainer.controller.lower) == before
    assert not trainer.optimizers["lower"].state


@pytest.mark.parametrize("method", ["hpgi", "fixed_window", "nominal", "full_information"])
def test_scaled_pipeline_against_physical_ledger(config, method):
    config = for_method(config, method)
    config["reward_normalization"]["enabled"] = True
    config["horizon"] = 5
    config["ppo"]["epochs"] = 1
    trainer = Trainer(config, 96)
    trainer.calibrate()
    for layer in ("upper", "lower"):
        opponent = "lower" if layer == "upper" else "upper"
        frozen = state_hash(getattr(trainer.controller, opponent))
        buffer, records = trainer.collect(layer, 3)
        samples = copy.deepcopy(buffer.samples)
        audit = {}
        result = update(getattr(trainer.controller, layer), trainer.optimizers[layer], buffer,
                        config["ppo"], 0, 0, audit)
        check_update(samples, audit)
        check_physics(config, records)
        assert result["critic_after"] <= .500001
        assert result["actor_after"] <= .500001
        assert result["parameter_delta_l2"]["critic"] > 0
        assert state_hash(getattr(trainer.controller, opponent)) == frozen
        report = policy_input_diagnostics(getattr(trainer.controller, layer), samples)
        assert all(max(value.get("network_max_abs", [0])) <= 1 for value in report["features"].values())
        assert not buffer.samples


def test_tensor_widths_match_actual_network(config):
    trainer = Trainer(config, 97)
    for layer in ("upper", "lower"):
        buffer, _ = trainer.collect(layer, 2, pretrain=layer == "lower")
        policy = getattr(trainer.controller, layer)
        report = policy_input_diagnostics(policy, buffer.samples)
        tensors = report["tensors"]
        assert tensors["frame_items_width"] == 19
        assert tensors["servers"] == [len(config["servers"]), 13]
        assert tensors["context"] == [CONTEXT_DIM] and tensors["prefix"] == [7]
        hidden = config["ppo"]["hidden"]
        assert report["linear_layers"]["score.0"]["weight"] == [hidden, 4 * hidden + CONTEXT_DIM + 7]
        assert report["linear_layers"]["value_head.0"]["weight"] == [hidden, 2 * hidden + CONTEXT_DIM]
        logprob, entropy, value = policy.replay(buffer.samples[0])
        assert logprob.shape == entropy.shape == value.shape == torch.Size([])
