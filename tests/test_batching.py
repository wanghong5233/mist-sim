import copy
import pytest
import torch
from mist_sim.policies.networks import SetPolicy, CONTEXT_DIM
from mist_sim.policies.batching import replay_batch, replay_chunks
from mist_sim.ppo import loss_terms
from mist_sim.config import validate


@pytest.mark.parametrize("upper,rates", [(False, False), (True, False), (False, True)])
def test_ragged_replay_loss_and_gradients(config, upper, rates):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    policy = SetPolicy(16, learned_rates=rates)
    samples = []
    for n in (0, 1, 4, 2):
        frame = dict(items=torch.randn(n, 19).tolist(), servers=torch.randn(3, 13).tolist(),
                     context=torch.randn(CONTEXT_DIM).tolist())
        steps = []
        for j in range(n):
            step = dict(features=torch.randn(j + 2, 19).tolist(), prefix=torch.randn(7).tolist(),
                        mask=[True] + [False] * j + [True], action=j + 1)
            if upper:
                step["prefix_frame"] = copy.deepcopy(frame)
                step["prefix_frame"]["servers"][0][6] = j
            if rates and j % 2 == 0:
                step["rate_latent"] = .3
            steps.append(step)
        samples.append(dict(frame=frame, steps=steps))
    serial = tuple(torch.stack(t) for t in zip(*(policy.replay(s) for s in samples)))
    batched = replay_batch(policy, samples)
    for a, b in zip(serial, batched):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(replay_chunks(policy, samples, 2), serial[0], rtol=2e-5, atol=2e-6)
    old = serial[0].detach() + .02
    adv = torch.tensor([-.5, .5, 1., -1.])
    target = torch.tensor([.5, 1., -.5, 0.])
    valid = torch.tensor([False, True, True, True])
    def loss(values):
        lp, entropy, value = values
        return loss_terms(lp, old, adv, value, target, entropy, valid, config["ppo"])[0]
    a, b = loss(serial), loss(batched)
    torch.testing.assert_close(a, b)
    a.backward()
    gradients = {n: p.grad.clone() if p.grad is not None else None for n, p in policy.named_parameters()}
    policy.zero_grad(set_to_none=True)
    b.backward()
    for n, p in policy.named_parameters():
        if gradients[n] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[n], rtol=5e-4, atol=2e-6)


def test_thread_setting_validation(config):
    config["runtime"] = {"torch_threads": 2}
    validate(config)
    for bad in (0, -1, True, 1.5):
        config["runtime"]["torch_threads"] = bad
        with pytest.raises(ValueError, match="torch_threads"):
            validate(config)


def test_batched_mask_zero_gradient():
    torch.set_num_threads(1)
    policy = SetPolicy(16)
    frame = dict(items=[], servers=[[0.] * 13], context=[0.] * CONTEXT_DIM)
    sample = dict(frame=frame, steps=[dict(features=[[0.] * 19, [1.] * 19, [2.] * 19],
                  prefix=[0.] * 7, mask=[True, False, True], action=0)])
    outputs = []
    def retain(module, inputs, output):
        output.retain_grad()
        outputs.append(output)
    handle = policy.score.register_forward_hook(retain)
    try:
        (-replay_batch(policy, [sample])[0].sum()).backward()
        assert outputs[0].grad[1].item() == 0
        assert outputs[0].grad[0].abs().item() > 0
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["roundoff", "old_policy", "bad_batch", "nan", "joint"])
def test_behavior_guard_distinguishes_roundoff_from_policy_drift(monkeypatch, kind):
    from types import SimpleNamespace
    from mist_sim.policies import batching
    torch.set_num_threads(1)
    torch.manual_seed(73)
    policy = SetPolicy(16)
    frame = dict(items=[], servers=[[0.] * 13], context=[0.] * CONTEXT_DIM)
    sample = dict(frame=frame, steps=[dict(features=[[0.] * 19, [1.] * 19],
                  prefix=[0.] * 7, mask=[True, True], action=0) for _ in range(19)])
    with torch.no_grad():
        original = policy.replay(sample)[0].detach()
    sample["logprob"] = float(original) + (.01 if kind == "old_policy" else 0.)
    replayed = original.reshape(1) + (.01 if kind == "bad_batch" else 1.1444091796875e-5)
    if kind == "nan":
        replayed[:] = float("nan")
    monkeypatch.setattr(batching, "replay_chunks", lambda *args: replayed)
    if kind == "joint":
        sample["phase"] = "lower"
        policy = SimpleNamespace(phase_policies={"lower": policy})
    if kind in ("old_policy", "bad_batch"):
        with pytest.raises(AssertionError, match="Behavior probability changed"):
            batching.check_behavior_probability(policy, [sample])
    elif kind == "nan":
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            batching.check_behavior_probability(policy, [sample])
    else:
        result = batching.check_behavior_probability(policy, [sample])
        assert result["serial_fallback_rows"] == 1
        assert result["serial_max_error"] == 0.
