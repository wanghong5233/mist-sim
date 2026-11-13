import copy
from pathlib import Path

import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.config import load_config, validate
from mist_sim.environment import Environment
from mist_sim.policies.batching import RolloutTensors, replay_batch, check_behavior_probability
from mist_sim.policies.base import BaseController
from mist_sim.training import Trainer
from mist_sim.transfer import initialize_transfer


def development_config(config):
    result = copy.deepcopy(config)
    result.update(horizon=4, k=1, arrival_probability=1.)
    result["ppo"].update(hidden=16, epochs=1, minibatch=4)
    result["training"].update(pretrain_slots=4, fast_slots=8, slow_intervals=4, cycles=2, calibration_episodes=1)
    result["validation"]["enabled"] = False
    result["diagnostics"]["phase_interval"] = 0
    result["monitoring"] = dict(console=False, tensorboard=False)
    result["runtime"]["collection_workers"] = 0
    result["ppo_layers"] = dict(upper=dict(sampling_temperature=2., final_sampling_temperature=1.5, temperature_anneal_cycles=2),
                                lower=dict(sampling_temperature=.7, final_sampling_temperature=.5, temperature_anneal_cycles=2))
    return result


@pytest.mark.parametrize("layer", ["upper", "lower"])
def test_tempered_sampling_replay_and_deterministic_actions(config, layer):
    torch.set_num_threads(1)
    torch.manual_seed(92)
    config = development_config(config)
    controller = BaseController(config)
    policy = getattr(controller, layer)
    environment = Environment(config, generate_trace(config, 91))
    argument = environment.upper_observation() if layer == "upper" else environment.begin_slot([1] * len(config["servers"]))
    action = policy.upper_action if layer == "upper" else policy.lower_action
    selected, *_, sample = action(argument)
    assert sample["sampling_temperature"] == policy.sampling_temperature
    first_step = sample["steps"][0]
    encoding, servers = policy.encode(first_step["prefix_frame"], with_servers=True)
    ordinary, _ = policy.distribution(encoding, first_step["features"], first_step["prefix"], first_step["mask"],
                                     servers, first_step["targets"], temperature=1.)
    tempered, _ = policy.distribution(encoding, first_step["features"], first_step["prefix"], first_step["mask"],
                                     servers, first_step["targets"])
    expected = ordinary.probs.pow(1 / policy.sampling_temperature)
    torch.testing.assert_close(tempered.probs, expected / expected.sum())
    deterministic = action(argument, deterministic=True)[0]
    policy.sampling_temperature = 1.
    assert action(argument, deterministic=True)[0] == deterministic
    serial = policy.replay(sample)
    cached = replay_batch(policy, [sample], cache=RolloutTensors(policy, [sample]))
    assert float(serial[0]) == pytest.approx(sample["logprob"], abs=1e-5)
    for expected_value, actual in zip(serial, cached):
        torch.testing.assert_close(expected_value, actual[0], atol=1e-5, rtol=2e-5)
    assert check_behavior_probability(policy, [sample])["max_error"] < 1e-5
    policy.zero_grad()
    (-cached[0].sum()).backward()
    assert sum(float(parameter.grad.square().sum()) for parameter in policy.parameters() if parameter.grad is not None) > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_transfer_temperature_workers_and_resume(config, tmp_path, device):
    config = development_config(config)
    parent_config = copy.deepcopy(config)
    parent_config["ppo_layers"] = dict(upper={}, lower={})
    parent = Trainer(parent_config, 91)
    parent.calibrate()
    parent_path = tmp_path / "parent.pt"
    parent.save(parent_path)
    config["device"] = device
    config["runtime"]["collection_workers"] = 2 if device == "cuda" else 0
    trainer = initialize_transfer(parent_path, config, 92)
    trainer.run()
    assert trainer.controller.sampling_temperatures() == dict(upper=1.75, lower=.6)
    assert all(row["parameters_changed"] and row["frozen_unchanged"] for row in trainer.history)
    assert all(row["behavior_logprob_error"] < 1e-5 for row in trainer.history)
    assert trainer.history[-1]["sampling_temperatures"] == dict(upper=1.75, lower=.6)
    saved = tmp_path / "last.pt"
    trainer.save(saved)
    restored = Trainer.load(saved)
    assert restored.controller.sampling_temperatures() == trainer.controller.sampling_temperatures()
    restored.run(extra_cycles=1)
    expected_state = copy.deepcopy(restored.controller.state_dict())
    repeated = Trainer.load(saved)
    repeated.run(extra_cycles=1)
    assert repeated.controller.sampling_temperatures() == dict(upper=1.5, lower=.5)
    for layer, parameters in expected_state.items():
        for name, expected in parameters.items():
            torch.testing.assert_close(repeated.controller.state_dict()[layer][name], expected,
                                       rtol=2e-6 if device == "cuda" else 0, atol=2e-7 if device == "cuda" else 0)
    changed = copy.deepcopy(config)
    changed["ppo_layers"]["lower"]["sampling_temperature"] = .8
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.load(saved, changed)


@pytest.mark.parametrize("settings", [dict(sampling_temperature=0), dict(sampling_temperature=float("nan")),
    dict(sampling_temperature=True), dict(temperature_anneal_cycles=-1), dict(final_sampling_temperature=.5)])
def test_invalid_exploration_rejected(config, settings):
    config["ppo_layers"]["lower"] = settings
    with pytest.raises(ValueError, match="temperature|Temperature|Nonfinite"):
        validate(config)


def test_exploration_overlay_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/smoke.json", [root / "configs/algorithms/hpgi_exploration.json"])
    assert config["ppo_layers"]["lower"]["sampling_temperature"] == .7
