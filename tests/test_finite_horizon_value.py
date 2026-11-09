import copy

import numpy as np
import pytest
import torch

from mist_sim.buffers import EventRolloutBuffer, RolloutBuffer
from mist_sim.config import load_config, validate
from mist_sim.diagnostics import check_update
from mist_sim.policies.batching import replay_batch, values_batch, RolloutTensors
from mist_sim.policies.networks import SetPolicy
from mist_sim.ppo import update
from mist_sim.training import Trainer


def test_sampled_returns_use_real_rewards_and_only_boundary_bootstrap():
    buffer = RolloutBuffer("upper", 0, 0)
    buffer.append(dict(value=100.), 2., 300., False, False, 4)
    buffer.append(dict(value=-100.), 3., 999., True, False, 1)
    buffer.append(dict(value=700.), 1., 7., False, True, 2)
    np.testing.assert_allclose(buffer.sampled_returns(.9), [2 + .9 ** 4 * 3, 3., 1 + .9 ** 2 * 7])
    buffer.samples[-1]["truncated"] = False
    assert buffer.sampled_returns(.9)[-1] == pytest.approx(1 + .9 ** 2 * 7)


def test_sampled_returns_count_zero_duration_reward_once():
    buffer = EventRolloutBuffer(dict(upper=0, lower=0))
    buffer.append(dict(phase="upper", value=50.), 0., 9., False, False, 0)
    buffer.append(dict(phase="lower", value=-50.), -2., 9., False, False, 1)
    buffer.append(dict(phase="lower", value=50.), -3., 100., True, False, 1)
    np.testing.assert_allclose(buffer.sampled_returns(.9), [-2 - .9 * 3, -2 - .9 * 3, -3])


@pytest.mark.parametrize("gamma", [0., 1e-12, .9, 1 - 1e-9, 1.])
def test_value_has_physical_clock_and_nonzero_last_reward(gamma):
    policy = SetPolicy(8, value_model="finite_horizon", horizon=4, gamma=gamma)
    policy.set_value_baseline(-2.)
    encoding = torch.randn(3, 23)  # 2*hidden + context
    context = torch.zeros(3, 7)
    context[:, 0] = torch.tensor([1., .25, 0.])
    actual = policy.value_from_encoding(encoding, context)
    whole = sum(gamma ** n for n in range(4))
    torch.testing.assert_close(actual, torch.tensor([-2 * whole, -2., 0.]))
    with torch.no_grad():
        policy.value_head[-1].bias.fill_(4.)
    actual = policy.value_from_encoding(encoding, context)
    assert actual[1].detach().item() == pytest.approx(-2 + 4 / whole)
    assert actual[2] == 0.
    policy.value_scale.mul_(.1)
    policy.set_value_baseline(-.2)
    torch.testing.assert_close(policy.value_from_encoding(encoding, context), actual * .1)


def test_near_undiscounted_clock_preserves_remaining_returns_and_gradient():
    gamma, horizon = 1 - 1e-9, 128
    policy = SetPolicy(8, value_model="finite_horizon", horizon=horizon, gamma=gamma)
    policy.set_value_baseline(-2.)
    remaining = [128, 64, 1, 0]
    context = torch.zeros(len(remaining), 7)
    context[:, 0] = torch.tensor(remaining) / horizon
    encoding = torch.randn(len(remaining), 23)
    actual = policy.value_from_encoding(encoding, context)
    # Independent finite sum in double precision, including the final reward.
    discounted = torch.tensor([sum(gamma ** i for i in range(n)) for n in remaining])
    torch.testing.assert_close(actual, -2 * discounted)
    whole = sum(gamma ** i for i in range(horizon))
    assert policy.discounted_horizon == pytest.approx(whole, rel=1e-12)
    for index, expected in enumerate(discounted / whole):
        gradient, = torch.autograd.grad(actual[index], policy.value_head[-1].bias, retain_graph=True)
        torch.testing.assert_close(gradient.squeeze(), expected)


def finite_config(config):
    config = copy.deepcopy(config)
    config["method"] = "joint_ppo"
    config["ppo"].update(value_model="finite_horizon", value_target="sampled_return", advantage_normalization="phase")
    config["runtime"]["collection_workers"] = 0
    config["k"] = 1
    return config


def test_sampled_return_target_does_not_replace_actor_gae(config):
    config = finite_config(config)
    trainer = Trainer(config, 31)
    trainer.calibrate()
    buffer, records = trainer.collect("joint", 6)
    samples = copy.deepcopy(buffer.samples)
    expected_advantage = buffer.advantages(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])[0]
    expected_returns = buffer.sampled_returns(config["ppo"]["gamma"])
    audit = {}
    metrics = update(trainer.joint_policy, trainer.optimizers["joint"], buffer, config["ppo"],
                     buffer.behavior_version, buffer.opponent_version, audit)
    np.testing.assert_allclose(audit["advantages"], expected_advantage)
    np.testing.assert_allclose(audit["returns"], expected_returns)
    assert metrics["value_target"] == "sampled_return"
    assert check_update(samples, audit)["forward_gae_and_loss"]
    assert metrics["critic_prediction"]["terminal_count"] == 0  # 6-slot truncated collection
    trainer.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_value_serial_batch_cache_and_gradients_match(config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = finite_config(config)
    trainer = Trainer(config, 31)
    trainer.calibrate()
    buffer, _ = trainer.collect("joint", 6)
    samples = [row for row in buffer.samples if row["phase"] == "lower"]
    policy = trainer.controller.lower
    serial = torch.stack([policy.value(sample["frame"]) for sample in samples])
    policy.to(device)
    cache = RolloutTensors(policy, samples)
    batch = values_batch(policy, samples, cache)
    torch.testing.assert_close(batch.cpu(), serial, atol=3e-6, rtol=2e-5)
    batch.sum().backward()
    assert policy.value_head[-1].weight.grad.norm() > 0
    policy.to("cpu")
    trainer.close()


def test_finite_training_parallel_tail_save_resume(config, tmp_path):
    config = finite_config(config)
    config["horizon"] = 5
    config["runtime"]["collection_workers"] = 2
    config["training"].update(pretrain_slots=10, fast_slots=10, slow_intervals=4, cycles=2)
    config["validation"]["enabled"] = False
    trainer = Trainer(config, 31)
    try:
        trainer.run()
        checkpoint = tmp_path / "finite.pt"
        trainer.save(checkpoint)
        restored = Trainer.load(checkpoint)
        try:
            assert restored.reward_normalizer.state_dict() == trainer.reward_normalizer.state_dict()
            for phase in ("upper", "lower"):
                for name, value in getattr(trainer.controller, phase).state_dict().items():
                    torch.testing.assert_close(value, getattr(restored.controller, phase).state_dict()[name])
            restored.run(extra_cycles=1)
            assert restored.total_slots > trainer.total_slots
            assert restored.history[-1]["critic_prediction"]["terminal_count"] == 2
            assert restored.history[-1]["behavior_logprob_error"] < 1e-5
        finally:
            restored.close()
    finally:
        trainer.close()


def test_value_config_is_explicit_and_rejects_unknown_modes(config):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    configured = load_config(root / "configs/smoke.json", [root / "configs/algorithms/joint_finite_horizon.json"])
    assert configured["ppo"]["value_model"] == "finite_horizon"
    assert configured["ppo"]["value_target"] == "sampled_return"
    config["ppo"]["value_model"] = "invented"
    with pytest.raises(ValueError, match="value model"):
        validate(config)
