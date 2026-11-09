import copy

import pytest
import torch

from mist_sim.config import validate
from mist_sim.devices import update_device
from mist_sim.policies.batching import RolloutTensors, replay_batch
from mist_sim.ppo import update
from mist_sim.training import Trainer


def test_device_validation_and_cpu_worker(config, monkeypatch):
    config["device"] = "cuda"
    validate(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        Trainer(config, 31)
    from mist_sim import collection
    collection.initialize_collector(config, 31, None)
    assert collection.collector.config["device"] == "cpu"
    assert next(collection.collector.controller.lower.parameters()).device.type == "cpu"
    for invalid in ("gpu", "auto", "cuda:bad", None):
        config["device"] = invalid
        with pytest.raises(ValueError, match="device"):
            validate(config)


def test_rollout_float_views_share_one_backing_tensor(config):
    config["method"] = "joint_ppo"
    trainer = Trainer(config, 31)
    buffer, _ = trainer.collect("joint", 6)
    cache = RolloutTensors(trainer.joint_policy, buffer.samples)
    pointer = cache.storage[torch.float32].untyped_storage().data_ptr()
    assert all(values.untyped_storage().data_ptr() == pointer for _, values in cache.entries.values())
    assert cache.host_to_device_copies == 0
    assert cache.storage_bytes > 0
    for actual, expected in zip(replay_batch(trainer.joint_policy, buffer.samples, cache=cache),
                                replay_batch(trainer.joint_policy, buffer.samples)):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_cuda_update_matches_cpu_and_returns_optimizer(config, tmp_path):
    config["method"] = "joint_ppo"
    config["k"] = 1
    config["reward_normalization"]["enabled"] = True
    config["ppo"].update(epochs=1, minibatch=4, target_kl=100.)
    trainer = Trainer(config, 31)
    trainer.calibrate()
    buffer, _ = trainer.collect("joint", 6)
    original_buffer = copy.deepcopy(buffer)
    original_weights = copy.deepcopy(trainer.controller.state_dict())
    torch.manual_seed(101)
    expected = update(trainer.joint_policy, trainer.optimizers["joint"], buffer, config["ppo"],
                      buffer.behavior_version, buffer.opponent_version)
    expected_weights = copy.deepcopy(trainer.controller.state_dict())
    accelerated = Trainer(dict(config, device="cuda"), 31)
    accelerated.controller.load_state_dict(original_weights)
    seen = []

    def observe(summary):
        assert next(accelerated.joint_policy.parameters()).device.type == "cuda"
        for state in accelerated.optimizers["joint"].state.values():
            assert state["exp_avg"].device.type == "cuda"
            assert state["step"].device.type == "cpu"
        seen.append(summary)

    torch.manual_seed(101)
    actual = update(accelerated.joint_policy, accelerated.optimizers["joint"], original_buffer, config["ppo"],
                    original_buffer.behavior_version, original_buffer.opponent_version, on_epoch=observe, device="cuda")
    assert len(seen) == 1
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-5)
    assert actual["rollout_tensor_transfers"] == 4
    assert actual["device_metrics"]["peak_allocated_mib"] > 0
    assert next(accelerated.joint_policy.parameters()).device.type == "cpu"
    for state in accelerated.optimizers["joint"].state.values():
        assert all(value.device.type == "cpu" for value in state.values() if isinstance(value, torch.Tensor))
    for phase, weights in accelerated.controller.state_dict().items():
        for name, value in weights.items():
            torch.testing.assert_close(value, expected_weights[phase][name], atol=3e-5, rtol=1e-3)
    checkpoint = tmp_path / "gpu.pt"
    accelerated.save(checkpoint)
    restored = Trainer.load(checkpoint, device="cpu")
    assert restored.config["device"] == "cpu"
    for phase, weights in restored.controller.state_dict().items():
        for name, value in weights.items():
            torch.testing.assert_close(value, accelerated.controller.state_dict()[phase][name], atol=0, rtol=0)
    modified = copy.deepcopy(config)
    modified["rejection_cost"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.load(checkpoint, modified, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_exception_restores_cpu_policy(config):
    trainer = Trainer(config, 31)
    policy, optimizer = trainer.controller.lower, trainer.optimizers["lower"]
    parameters = list(policy.parameters())
    with pytest.raises(RuntimeError, match="injected"):
        with update_device(policy, optimizer, "cuda"):
            assert next(policy.parameters()).is_cuda
            raise RuntimeError("injected failure")
    assert next(policy.parameters()).device.type == "cpu"
    assert all(before is after for before, after in zip(parameters, policy.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_pretrain_joint_and_resume(config, tmp_path):
    config["method"] = "joint_ppo"
    config["device"] = "cuda"
    config["k"] = 1
    config["validation"]["enabled"] = False
    config["training"].update(pretrain_slots=4, fast_slots=4, cycles=1, calibration_episodes=1)
    config["ppo"].update(epochs=1, minibatch=4)
    trainer = Trainer(config, 31)
    trainer.run()
    assert trainer.total_slots == 8
    assert all(row["device_metrics"]["update_device"] == "cuda" for row in trainer.history)
    checkpoint = tmp_path / "completed.pt"
    trainer.save(checkpoint)
    restored = Trainer.load(checkpoint)
    restored.run(extra_cycles=1)
    assert restored.total_slots == 12
    assert next(restored.controller.lower.parameters()).device.type == "cpu"
