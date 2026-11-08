import copy
import numpy as np
import pytest
import torch
from mist_sim.buffers import RolloutBuffer
from mist_sim.diagnostics import rollout_diagnostics
from mist_sim.training import Trainer


def test_pretraining_chunks_resume_without_reusing_samples(config, tmp_path):
    config["horizon"] = 5
    config["training"].update(pretrain_slots=10, fast_slots=4, slow_intervals=1, cycles=1)
    trainer = Trainer(config, 321)
    trainer.calibrate()
    trainer.validate_cycle("initial")
    trainer.phase("pretrain_000", "lower", 4, pretrain=True)
    checkpoint = tmp_path / "chunk.pt"
    trainer.save(checkpoint)
    trainer.run()
    expected = copy.deepcopy(trainer.controller.state_dict())
    resumed = Trainer.load(checkpoint)
    resumed.run()
    assert resumed.completed_stages == ["pretrain_000", "pretrain_001", "pretrain",
                                        "cycle_000_upper", "cycle_000_lower"]
    assert [row["samples"] for row in resumed.history] == [4, 4, 2, 1, 4]
    assert resumed.total_slots == 18
    assert [row["rollout"]["behavior_version"] for row in resumed.history[:3]] == [0, 1, 2]
    for layer in ("upper", "lower"):
        for name, tensor in expected[layer].items():
            torch.testing.assert_close(tensor, resumed.controller.state_dict()[layer][name], rtol=0, atol=0)
    assert resumed.reward_normalizer.state_dict() == trainer.reward_normalizer.state_dict()


def test_rollout_return_diagnostics_distinguish_lambda_and_segment_returns(config):
    buffer = RolloutBuffer("lower", 0, 0)
    frame = {"items": [], "servers": [], "context": [0] * 11}
    for index, (reward, value, next_value, terminated, truncated) in enumerate(
            [(2, 1, 3, False, False), (4, 3, 0, True, False), (5, 7, 11, False, True)]):
        buffer.append(dict(value=value, frame=frame, steps=[], physical_rewards=[reward],
                           episode_id=int(index == 2), slot=index % 2),
                      reward, next_value, terminated, truncated, next_observation=frame)
    report = rollout_diagnostics(buffer, dict(gamma=.9, gae_lambda=.5, minibatch=2, epochs=2))
    assert report["episode_segments"] == 2
    assert report["minibatches_per_epoch"] == 2
    examples = report["examples"]
    assert examples[0]["segment_return_with_bootstrap"] == pytest.approx(5.6)
    assert examples[0]["gae_value_target"] == pytest.approx(5.15)
    assert examples[-1]["segment_return_with_bootstrap"] == pytest.approx(14.9)
    assert examples[-1]["gae_value_target"] == pytest.approx(14.9)
    assert report["normalized_advantage_mean"] == pytest.approx(0, abs=1e-14)
    assert report["normalized_advantage_std"] == pytest.approx(1)


@pytest.mark.parametrize("layer,count,durations", [("lower", 19, [1] * 19),
                                                   ("upper", 6, [4, 4, 4, 4, 1, 4])])
def test_collected_transition_identity_and_boundaries(config, layer, count, durations):
    config["horizon"] = 17
    trainer = Trainer(config, 323)
    trainer.calibrate()
    buffer, records = trainer.collect(layer, count, pretrain=layer == "lower")
    assert [sample["duration"] for sample in buffer.samples] == durations
    assert len(records) == sum(durations)
    for index, sample in enumerate(buffer.samples[:-1]):
        following = buffer.samples[index + 1]
        if sample["terminated"]:
            assert following["slot"] == 0
            assert sample["episode_id"] != following["episode_id"]
            assert sample["next_value"] == 0
        else:
            assert sample["episode_id"] == following["episode_id"]
            assert following["slot"] == sample["slot"] + sample["duration"]
            assert sample["next_observation"] == following["frame"]
            assert sample["next_value"] == pytest.approx(following["value"], abs=1e-6)
    assert buffer.samples[-1]["truncated"]
    report = rollout_diagnostics(buffer, config["ppo"])
    assert report["terminated"] == 1
    assert report["episode_segments"] == 2
    assert np.isfinite(report["advantage_std"])
