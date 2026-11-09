import copy
import math
from multiprocessing import get_context

import numpy as np
import pytest
import torch

from mist_sim.buffers import RolloutBuffer
from mist_sim.collection import initialize_collector, collect_episode
from mist_sim.evaluation import assert_close, replay
from mist_sim.policies.batching import replay_chunks
from mist_sim.training import Trainer


def strip_timing(records):
    return [{key: value for key, value in record.items() if key != "measured_allocation_seconds"}
            for record in records]


@pytest.mark.parametrize("layer,count", [("lower", 13), ("upper", 5)])
def test_parallel_episode_sampling_matches_serial_reference(config, layer, count, tmp_path):
    config["horizon"] = 5
    config["runtime"]["collection_workers"] = 2
    trainer = Trainer(config, 91)
    trainer.calibrate()
    checkpoint = tmp_path / "state.pt"
    trainer.save(checkpoint)
    try:
        start_index = trainer.trace_counter
        buffer, records = trainer.collect(layer, count)
        expected_state = {layer_name: {name: tensor.detach().numpy().copy() for name, tensor in values.items()}
                          for layer_name, values in trainer.controller.state_dict().items()}
        initialize_collector(config, 91, get_context("spawn").Event())
        per_episode = config["horizon"] if layer == "lower" else math.ceil(config["horizon"] / config["window"])
        expected_samples, expected_records = [], []
        episode_advantages = []
        for episode, offset in enumerate(range(0, count, per_episode)):
            samples, physical = collect_episode(expected_state, trainer.reward_normalizer.state_dict(),
                trainer.versions, start_index + episode, layer, min(per_episode, count - offset), False, None)
            expected_samples.extend(samples)
            expected_records.extend(physical)
            episode_buffer = RolloutBuffer(layer, 0, 0)
            episode_buffer.samples = samples
            episode_advantages.extend(episode_buffer.advantages(1., .95)[0])
            if len(physical) == config["horizon"]:
                replay(config, physical)
        assert_close(buffer.samples, expected_samples)
        assert_close(strip_timing(records), strip_timing(expected_records))
        np.testing.assert_array_equal(buffer.advantages(1., .95)[0], episode_advantages)
        with torch.no_grad():
            probabilities = replay_chunks(getattr(trainer.controller, layer), buffer.samples, 4)
        torch.testing.assert_close(probabilities, torch.tensor([sample["logprob"] for sample in buffer.samples]), atol=1e-5, rtol=0)
        assert len(buffer.samples) == count
        assert buffer.samples[-1]["truncated"]
        assert trainer.training_episodes == 2
        assert trainer.total_slots == len(records)
        restored = Trainer.load(checkpoint)
        try:
            restored_buffer, restored_records = restored.collect(layer, count)
            assert_close(restored_buffer.samples, buffer.samples)
            assert_close(strip_timing(restored_records), strip_timing(records))
        finally:
            restored.close()
    finally:
        trainer.close()


def test_parallel_pool_updates_policy_snapshot_and_runs_ppo(config):
    config["runtime"]["collection_workers"] = 2
    config["horizon"] = 4
    config["training"].update(pretrain_slots=4, fast_slots=4, slow_intervals=2, cycles=1, calibration_episodes=1)
    config["validation"]["enabled"] = False
    config["ppo"]["epochs"] = 1
    trainer = Trainer(config, 91)
    trainer.run()
    assert trainer.collectors is None
    assert trainer.versions == dict(upper=1, lower=2)
    assert all(row["behavior_logprob_error"] < 1e-5 and row["frozen_unchanged"] for row in trainer.history)
    assert all(row["parameters_changed"] for row in trainer.history)
