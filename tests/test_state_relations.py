"""Decision-relevant grouping, resource binding and anonymity counterexamples."""
import itertools
import torch
from mist_sim.environment import PublicProblem, Environment
from mist_sim.anonymity import generate_trace
from mist_sim.policies.networks import SetPolicy, Normalizer, lower_frame, lower_prefix_frame
from mist_sim.policies.batching import replay_batch, RolloutTensors
from mist_sim.training import Trainer


def example_choice(kind, cost, split=-1, alpha=.01):
    return dict(kind=kind, server=0 if kind == "edge" else -1, split=split,
        minimum=1. if kind == "edge" else 0., memory=1. if kind == "edge" else 0.,
        bandwidth=1. if kind == "edge" else 0., ap=0 if kind == "edge" else -1,
        fixed_energy=cost if kind == "edge" else 0., delay_fixed=0.,
        alpha=alpha if kind == "edge" else 0., beta=0.,
        edge_work=1. if kind == "edge" else 0., local_work=0., bits=0.,
        cost_fixed=cost, deadline=1., weight=1.)


def grouped_problem(config, groups):
    problem = PublicProblem(config, [], [1] * len(config["servers"]), [1] * len(config["servers"]), 1, [])
    problem.public = [dict(id=str(index)) for index in range(len(groups))]
    problem.choices = groups
    return problem


def test_request_grouping_changes_optimal_action_and_encoded_state(config):
    config["servers"] = config["servers"][:1]
    config["servers"][0]["capacity"] = 2.
    reject, current = example_choice("reject", 10), example_choice("edge", 4, 1)
    a, b = example_choice("edge", 1, 2), example_choice("edge", 1, 3)
    problems = [grouped_problem(config, [[reject, current], [reject, a, b], [reject]]),
                grouped_problem(config, [[reject, current], [reject, a], [reject, b]])]
    frames = [lower_frame(problem) for problem in problems]
    assert sorted(map(tuple, frames[0]["items"])) == sorted(map(tuple, frames[1]["items"]))
    best_first = []
    for problem in problems:
        feasible = []
        for actions in itertools.product(*[range(len(options)) for options in problem.choices]):
            try:
                feasible.append((problem.scheduling_cost(actions), actions))
            except ValueError:
                pass
        best_first.append(min(feasible)[1][0])
    assert best_first == [1, 0]
    torch.manual_seed(71)
    policy = SetPolicy(16)
    for critic in (False, True):
        first, second = [policy.encode(frame, critic)[0] for frame in frames]
        assert not torch.allclose(first, second, atol=1e-6)


def test_prior_assignments_bind_costs_to_servers(config):
    torch.manual_seed(73)
    policy = SetPolicy(16)
    reject, task = example_choice("reject", 10), example_choice("edge", 1)
    problem = grouped_problem(config, [[reject, task]])
    frame = lower_frame(problem)
    usage = problem.empty_usage()
    low = lower_frame(grouped_problem(config, [[example_choice("edge", 1, alpha=.01)]]))["items"][0]
    high = lower_frame(grouped_problem(config, [[example_choice("edge", 1, alpha=1.)]]))["items"][0]
    first = lower_prefix_frame(frame, problem, usage, 0, [low, high], [0, 1])
    second = lower_prefix_frame(frame, problem, usage, 0, [low, high], [1, 0])
    _, a = policy.encode(first, with_servers=True)
    _, b = policy.encode(second, with_servers=True)
    assert not torch.allclose(a[0], b[0], atol=1e-6)
    prefix = [0.] * 7
    features = frame["items"]
    pa, _ = policy.distribution(policy.encode(first), features, prefix, [True, True], a, [-1, 0])
    pb, _ = policy.distribution(policy.encode(second), features, prefix, [True, True], b, [-1, 0])
    assert not torch.equal(pa.logits, pb.logits)


def test_grouped_real_samples_batch_probability_and_gradient(config):
    config["k"] = 1
    config["initial_active"] = [1] * len(config["servers"])
    trainer = Trainer(config, 99)
    for layer, count in (("upper", 2), ("lower", 4)):
        buffer, _ = trainer.collect(layer, count, pretrain=layer == "lower")
        policy = getattr(trainer.controller, layer)
        samples = buffer.samples
        serial = tuple(torch.stack(values) for values in zip(*(policy.replay(sample) for sample in samples)))
        batched = replay_batch(policy, samples, cache=RolloutTensors(policy, samples))
        torch.testing.assert_close(batched[0], torch.tensor([sample["logprob"] for sample in samples]), atol=2e-5, rtol=2e-5)
        for first, second in zip(serial, batched):
            torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
        sum(value.sum() for value in serial).backward()
        gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                     for name, parameter in policy.named_parameters()}
        policy.zero_grad(set_to_none=True)
        sum(value.sum() for value in batched).backward()
        for name, parameter in policy.named_parameters():
            if gradients[name] is not None:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=5e-4)


def test_small_physical_coefficients_are_not_flattened():
    normalizer = Normalizer(1)
    normalizer.fit([[1e-5], [2e-5], [3e-5]])
    values = normalizer(torch.tensor([[1e-5], [3e-5]]))
    assert values[0].item() < -.5 and values[1].item() > .5


def test_request_cache_does_not_encode_again_within_one_action(config):
    from unittest.mock import Mock
    torch.set_num_threads(1)
    config["k"] = 1
    config["initial_active"] = [1] * len(config["servers"])
    env = Environment(config, generate_trace(config, 21))
    policy = SetPolicy(16)
    policy.actor_relations.forward = Mock(wraps=policy.actor_relations.forward)
    problem = env.begin_slot([1] * len(config["servers"]))
    policy.lower_action(problem)
    assert policy.actor_relations.forward.call_count == int(bool(problem.public))


def test_structured_observations_survive_checkpoint_and_trace_roundtrip(config, tmp_path):
    from mist_sim.traces import pack_behavior, read_behavior
    from mist_sim.storage import write_json
    trainer = Trainer(config, 101)
    trainer.calibrate()
    buffer, _ = trainer.collect("lower", 4, pretrain=True)
    trace_path, checkpoint = tmp_path / "trace.json", tmp_path / "state.pt"
    write_json(trace_path, pack_behavior(buffer.samples))
    restored_samples = read_behavior(trace_path)
    assert restored_samples == buffer.samples
    trainer.save(checkpoint)
    restored = Trainer.load(checkpoint)
    for first, second in zip(replay_batch(trainer.controller.lower, buffer.samples),
                             replay_batch(restored.controller.lower, restored_samples)):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert trainer.total_slots == 4 and restored.total_slots == 4
    assert trainer.history == [] and restored.history == []
