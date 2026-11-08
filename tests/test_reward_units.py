import copy
import numpy as np
import pytest
import torch
from mist_sim.rewards import RunningRewardNormalizer, training_reward
from mist_sim.training import Trainer, state_hash
from mist_sim.diagnostics import check_update
from mist_sim.evaluation import evaluate
from mist_sim.anonymity import generate_trace
from mist_sim.ppo import clip_policy_gradients
from mist_sim.policies.networks import SetPolicy


@pytest.mark.parametrize("gamma", [1., .9])
def test_running_statistics_independent_numpy(config, gamma):
    config["reward_normalization"]["enabled"] = True
    config["ppo"]["gamma"] = gamma
    normalizer = RunningRewardNormalizer(config)
    records = [dict(slot=slot, reward=reward) for slot, reward in [(0,-2),(1,-4),(2,-1),(0,-3),(1,-5)]]
    expected = [-2, -2 * gamma - 4, (-2 * gamma - 4) * gamma - 1, -3, -3 * gamma - 5]
    normalizer.observe(records[:2])
    normalizer.observe(records[2:])
    assert normalizer.mean == pytest.approx(np.mean(expected))
    assert normalizer.units()["variance"] == pytest.approx(np.var(expected))
    assert normalizer.units()["scale"] == pytest.approx(1 / np.std(expected))
    restored = RunningRewardNormalizer(config)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.units() == normalizer.units()
    for instance in (normalizer, restored):
        instance.observe([dict(slot=2, reward=-9)])
    assert restored.state_dict() == normalizer.state_dict()


def test_clipping_no_shift_and_disabled(config):
    config["reward_normalization"]["enabled"] = True
    normalizer = RunningRewardNormalizer(config)
    units = normalizer.units()
    assert [training_reward(dict(reward=value), units) for value in [-100, -2, 0, 2, 100]] == [-10,-2,0,2,10]
    assert units["shift"] == 0
    normalizer.observe([dict(slot=0, reward=0), dict(slot=1, reward=0)])
    assert np.isfinite(normalizer.units()["scale"])
    with pytest.raises(ValueError, match="Nonfinite"):
        training_reward(dict(reward=float("nan")), units)
    config["reward_normalization"]["enabled"] = False
    assert training_reward(dict(reward=-100), RunningRewardNormalizer(config).units()) == -100


@pytest.mark.parametrize("gamma", [1., .9])
def test_phase_freezes_scale_and_recomputes_loss(config, gamma):
    config["reward_normalization"]["enabled"] = True
    config["horizon"] = 5
    config["ppo"]["gamma"] = gamma
    trainer = Trainer(config, 71)
    trainer.calibrate()
    statistics = copy.deepcopy(trainer.reward_normalizer.state_dict())
    for layer in ("upper", "lower"):
        buffer, _ = trainer.collect(layer, 3, pretrain=layer == "lower")
        assert trainer.reward_normalizer.state_dict() == statistics
        units = trainer.reward_normalizer.units()
        assert len({sample["reward_scale"] for sample in buffer.samples}) == 1
        for sample in buffer.samples:
            expected = sum(gamma ** offset * training_reward(dict(reward=reward), units)
                           for offset, reward in enumerate(sample["physical_rewards"]))
            assert sample["reward"] == pytest.approx(expected)
        advantages, returns = buffer.advantages(gamma, .95)
        audit = dict(settings=config["ppo"], advantages=advantages, returns=returns,
                     normalized_advantages=(advantages-advantages.mean())/max(advantages.std(),1e-8),
                     td_errors=[sample["reward"] + gamma**sample["duration"]*sample["next_value"]*
                                (not sample["terminated"]) - sample["value"] for sample in buffer.samples], minibatches=[])
        check_update(buffer.samples, audit)


def test_value_unit_change_preserves_actor_and_physical_value(config):
    config["reward_normalization"]["enabled"] = True
    trainer = Trainer(config, 72)
    trainer.calibrate()
    buffer, _ = trainer.collect("lower", 2, pretrain=True)
    sample = buffer.samples[0]
    policy = trainer.controller.lower
    old_probability, _, old_value = policy.replay(sample)
    old_scale = trainer.value_reward_scales["lower"]
    trainer.reward_normalizer.observe([dict(slot=0, reward=-1000),dict(slot=1,reward=-2000)])
    opponent = state_hash(trainer.controller.upper)
    trainer.collect("lower", 2, pretrain=True)
    probability, _, value = policy.replay(sample)
    torch.testing.assert_close(probability, old_probability, rtol=0, atol=0)
    assert float(value) / trainer.value_reward_scales["lower"] == pytest.approx(float(old_value) / old_scale, rel=1e-5)
    assert state_hash(trainer.controller.upper) == opponent


def test_statistics_update_only_training_and_resume(config, tmp_path):
    config["reward_normalization"]["enabled"] = True
    trainer = Trainer(config, 73)
    trainer.calibrate()
    before = copy.deepcopy(trainer.reward_normalizer.state_dict())
    evaluate(trainer.controller, config, generate_trace(config, config["seeds"]["validation"]))
    assert trainer.reward_normalizer.state_dict() == before
    trainer.phase("test", "lower", 3, pretrain=True)
    assert trainer.reward_normalizer.count == before["count"] + 3
    path = tmp_path / "state.pt"
    trainer.save(path)
    trainer.phase("next", "upper", 2)
    expected = copy.deepcopy(trainer.controller.state_dict())
    statistics = trainer.reward_normalizer.state_dict()
    restored = Trainer.load(path)
    restored.phase("next", "upper", 2)
    assert restored.reward_normalizer.state_dict() == statistics
    assert restored.value_reward_scales == trainer.value_reward_scales
    for layer in ("upper", "lower"):
        for key, value in expected[layer].items():
            torch.testing.assert_close(value, restored.controller.state_dict()[layer][key], rtol=0, atol=0)


def test_critic_magnitude_does_not_shrink_actor():
    policy = SetPolicy(hidden=4)
    policy.score[-1].bias.grad = torch.full_like(policy.score[-1].bias, .1)
    policy.value_head[-1].bias.grad = torch.full_like(policy.value_head[-1].bias, 100000.)
    result = clip_policy_gradients(policy, .5)
    assert result["actor_after"] == pytest.approx(.1)
    assert result["critic_after"] <= .500001
