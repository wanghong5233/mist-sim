import copy

import numpy as np
import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.environment import Environment
from mist_sim.evaluation import assert_close
from mist_sim.policies.batching import replay_batch, values_batch, RolloutTensors
from mist_sim.policies.networks import upper_frame, lower_frame, lower_prefix_frame, item_features
from mist_sim.ppo import loss_terms, update
from mist_sim.training import Trainer


def test_mixed_events_match_scalar_loss_and_every_parameter_gradient(config):
    config["method"] = "joint_ppo"
    config["k"] = 1
    config["deadline_range"] = [.95, 1.0]
    trainer = Trainer(config, 323)
    trainer.calibrate()
    buffer, _ = trainer.collect("joint", 6)
    samples = buffer.samples
    policy = trainer.joint_policy
    serial = tuple(torch.stack(values) for values in zip(*(
        policy.phase_policies[sample["phase"]].replay(sample) for sample in samples)))
    batched = replay_batch(policy, samples, cache=RolloutTensors(policy, samples))
    for actual, expected in zip(batched, serial):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batched[0], torch.tensor([sample["logprob"] for sample in samples]), atol=2e-6, rtol=0)
    old = serial[0].detach() + torch.tensor([.3, -.3, .02, -.02, 0., .4, -.4, .1])
    advantage = torch.tensor([-.5, .5, 1., -1., 2., -2., .2, -.2])
    target = serial[2].detach() + .2
    valid = torch.tensor([bool(sample["steps"]) for sample in samples])

    def objective(outputs):
        logprob, entropy, value = outputs
        return loss_terms(logprob, old, advantage, value, target, entropy, valid, config["ppo"])[0]

    expected, actual = objective(serial), objective(batched)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                 for name, parameter in policy.named_parameters()}
    policy.zero_grad(set_to_none=True)
    actual.backward()
    for name, parameter in policy.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None
        else:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=1e-3)


def test_joint_masks_block_score_gradients_and_reject_unknown_phase(config):
    config["method"] = "joint_ppo"
    policy = Trainer(config, 12).joint_policy
    frame = dict(items=[], servers=[[0.] * 13], context=[0.] * 7)
    samples = [dict(phase=phase, frame=frame, steps=[dict(features=[[0.] * 19, [1.] * 19, [2.] * 19],
        prefix=[0.] * 7, mask=[True, False, True], action=0)]) for phase in ("lower", "upper")]
    scores, handles = {}, []
    for phase, branch in policy.phase_policies.items():
        def retain(module, inputs, output, phase=phase):
            output.retain_grad()
            scores[phase] = output
        handles.append(branch.score.register_forward_hook(retain))
    try:
        (-replay_batch(policy, samples)[0].sum()).backward()
        for output in scores.values():
            assert output.grad[1].item() == 0
            assert output.grad[0].abs().item() > 0
    finally:
        for handle in handles:
            handle.remove()
    for operation in (values_batch, replay_batch):
        with pytest.raises(ValueError, match="Unknown joint"):
            operation(policy, [dict(samples[0], phase="invalid")])


def test_joint_actions_reconstruct_same_environment_and_prefixes(config):
    config["method"] = "joint_ppo"
    config["horizon"] = 5
    config["k"] = 1
    config["deadline_range"] = [.95, 1.0]
    trainer = Trainer(config, 323)
    trainer.calibrate()
    buffer, records = trainer.collect("joint", 13)
    for method in ("hpgi", "joint_ppo"):
        replay_config = dict(config, method=method)
        episode, record_index = None, 0
        for sample in buffer.samples:
            if sample["episode_id"] != episode:
                episode = sample["episode_id"]
                trace = generate_trace(config, trainer.seed + 10000 * (episode - 1))
                environment = Environment(replay_config, trace)
            if sample["phase"] == "upper":
                assert_close(upper_frame(environment.upper_observation()), sample["frame"])
                counterexample = copy.deepcopy(environment)
                counterexample._trace[environment.slot] = []
                assert_close(upper_frame(counterexample.upper_observation()), sample["frame"])
                active = sample["actions"]
                continue
            problem = environment.begin_slot(active)
            frame = lower_frame(problem)
            assert_close(frame, sample["frame"])
            usage, allocated, owners = problem.empty_usage(), [], []
            assert sample["order"] == [request["id"] for request in problem.public]
            for index, (step, options) in enumerate(zip(sample["steps"], problem.choices)):
                prefix = lower_prefix_frame(frame, problem, usage, index, allocated, owners)
                assert_close(prefix, step["prefix_frame"])
                assert step["mask"] == problem.mask(options, usage, capacity=True)
                assert step["targets"] == [choice["server"] for choice in options]
                assert step["mask"][step["action"]]
                for choice, features in zip(options, step["features"]):
                    expected = item_features(choice, config)
                    if choice["kind"] == "edge":
                        expected[-3:] = prefix["remaining"][choice["server"]]
                    np.testing.assert_allclose(features, expected)
                choice = options[step["action"]]
                problem.reserve(choice, usage)
                if choice["kind"] == "edge":
                    allocated.append(item_features(choice, config))
                    owners.append(choice["server"])
            actual = environment.execute(sample["actions"], sample["predicted_rates"])
            expected = records[record_index]
            assert_close({key: value for key, value in actual.items() if key != "measured_allocation_seconds"},
                         {key: value for key, value in expected.items() if key != "measured_allocation_seconds"})
            record_index += 1
        assert record_index == len(records)


def test_joint_empty_slots_update_critic_without_lower_actor(config):
    config["method"] = "joint_ppo"
    config["arrival_probability"] = 0
    trainer = Trainer(config, 13)
    trainer.calibrate()
    buffer, records = trainer.collect("joint", 4)
    assert len(records) == 4
    assert all(not sample["steps"] for sample in buffer.samples if sample["phase"] == "lower")
    snapshot = buffer.behavior_version
    report = update(trainer.joint_policy, trainer.optimizers["joint"], buffer,
                    dict(config["ppo"], epochs=1), snapshot, snapshot)
    assert report["components"]["lower"]["parameter_delta_l2"]["actor"] == 0
    assert report["components"]["lower"]["parameter_delta_l2"]["critic"] > 0
    assert len(buffer.samples) == 0


def test_gradient_maximum_does_not_require_detailed_audit(config, monkeypatch):
    from mist_sim import ppo
    trainer = Trainer(config, 13)
    buffer, _ = trainer.collect("lower", 4, pretrain=True)
    actual_clip = ppo.clip_policy_gradients
    observed = []

    def decreasing_clip(policy, limit):
        groups = actual_clip(policy, limit)
        with torch.no_grad():
            for parameter in policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(.01 ** len(observed))
        observed.append(float(torch.sqrt(sum(parameter.grad.square().sum()
            for parameter in policy.parameters() if parameter.grad is not None))))
        return groups

    monkeypatch.setattr(ppo, "clip_policy_gradients", decreasing_clip)
    report = update(trainer.controller.lower, trainer.optimizers["lower"], buffer,
                    dict(config["ppo"], epochs=1, minibatch=1), 0, 0)
    assert observed[0] > observed[-1]
    assert report["gradient_after_clip_max"] == max(observed)
