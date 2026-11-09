"""Bounded learning-signal probes; no persistent training or evaluation artifacts."""
import argparse
import copy
import json

import numpy as np
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.environment import Environment
from mist_sim.buffers import RolloutBuffer, normalize_advantages
from mist_sim.policies.baselines.heuristics import greedy_fast
from mist_sim.policies.batching import replay_batch
from mist_sim.ppo import parameter_group, update
from mist_sim.training import Trainer
from mist_sim.training_joint import joint_phase


def joint_probe(payload, count, lambdas, workers, normalization):
    for gae_lambda in lambdas:
        config = copy.deepcopy(payload["config"])
        config["ppo"]["gae_lambda"] = gae_lambda
        config["ppo"]["advantage_normalization"] = normalization
        if workers is not None:
            config["runtime"]["collection_workers"] = workers
        trainer = Trainer(config, payload["seed"])
        trainer.controller.load_state_dict(payload["controller"])
        trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
        trainer.value_reward_scales = copy.deepcopy(payload["value_reward_scales"])
        trainer.versions = copy.deepcopy(payload["versions"])
        trainer.trace_counter = payload["trace_counter"]
        for phase in ("upper", "lower"):
            trainer.optimizers[phase].load_state_dict(payload["optimizers"][phase])
        try:
            trainer.trace_counter = payload["trace_counter"] + 1000
            buffer, records = trainer.collect("joint", config["horizon"] * 2)
            before = float(np.mean([record["reward"] for record in records]))
            trainer.trace_counter = payload["trace_counter"]
            del buffer, records
            for iteration in range(count):
                row = joint_phase(trainer, f"cycle_{iteration:03d}_joint", config["training"]["fast_slots"])
                print(json.dumps(dict(probe="joint_lambda", gae_lambda=gae_lambda, normalization=normalization, iteration=iteration + 1,
                    reward=row["raw_reward_per_slot"], service=row["on_time_rate"],
                    collection_seconds=row["collection_seconds"], update_seconds=row["update_seconds"],
                    behavior_logprob_error=row["behavior_logprob_error"],
                    learning_signal=row["rollout"]["learning_signal"])), flush=True)
            trainer.trace_counter = payload["trace_counter"] + 1000
            buffer, records = trainer.collect("joint", config["horizon"] * 2)
            print(json.dumps(dict(probe="joint_lambda_fixed_diagnostic_seeds", gae_lambda=gae_lambda, normalization=normalization,
                slots=len(records), before=before, mean_reward=float(np.mean([record["reward"] for record in records])))), flush=True)
            del buffer, records
        finally:
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--joint-updates", type=int, default=0)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[.95, .8])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--normalization", choices=["global", "phase"], default="global")
    args = parser.parse_args()
    if args.samples < 2 or args.updates < 1:
        parser.error("At least two samples and one update required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.joint_updates:
        if args.joint_updates < 1 or any(not 0 <= value <= 1 for value in args.lambdas):
            parser.error("Positive joint updates and lambdas in [0,1] required")
        if args.workers is not None and not 0 <= args.workers <= 32:
            parser.error("Workers must be in [0,32]")
        joint_probe(payload, args.joint_updates, args.lambdas, args.workers, args.normalization)
        return
    config = copy.deepcopy(payload["config"])
    config["device"] = "cpu"
    config["ppo"]["advantage_normalization"] = args.normalization
    config["runtime"]["collection_workers"] = 0
    trainer = Trainer(config, payload["seed"])
    trainer.controller.load_state_dict(payload["controller"])
    trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
    trainer.value_reward_scales = payload["value_reward_scales"]
    trainer.trace_counter = payload["trace_counter"]
    buffer, records = trainer.collect("joint", config["horizon"])
    advantage = buffer.advantages(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])[0]
    normalized = normalize_advantages(advantage, buffer.samples, args.normalization)
    for phase in ("upper", "lower"):
        indices = [index for index, sample in enumerate(buffer.samples) if sample["phase"] == phase][:32]
        samples = [buffer.samples[index] for index in indices]
        policy = getattr(trainer.controller, phase)
        parameters = [parameter for name, parameter in policy.named_parameters() if parameter_group(name) == "actor"]
        logprob, entropy, _ = replay_batch(policy, samples, with_value=False)
        old = torch.tensor([sample["logprob"] for sample in samples])
        surrogate = -((logprob - old).exp() * torch.tensor(normalized[indices], dtype=torch.float32)).mean()
        actor_gradient = torch.autograd.grad(surrogate, parameters, retain_graph=True, allow_unused=True)
        entropy_gradient = torch.autograd.grad(-config["ppo"]["entropy_coefficient"] * entropy.mean(),
                                               parameters, allow_unused=True)
        def flatten(gradients):
            return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).flatten()
                              for parameter, gradient in zip(parameters, gradients)])
        actor_gradient, entropy_gradient = flatten(actor_gradient), flatten(entropy_gradient)
        print(json.dumps(dict(probe="rollout_gradient", phase=phase, samples=len(indices),
            surrogate_norm=float(actor_gradient.norm()), entropy_norm=float(entropy_gradient.norm()),
            cosine=float(torch.nn.functional.cosine_similarity(actor_gradient, entropy_gradient, dim=0)),
            advantage_std=float(advantage[indices].std()))), flush=True)
    del buffer
    environment = Environment(config, generate_trace(config, payload["seed"] + 9000000))
    active = [1] * len(config["servers"])
    first = environment.begin_slot(active)
    environment.execute([0] * len(first.choices))
    problem = environment.begin_slot(active)
    policy = trainer.controller.lower
    initial_policy = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["ppo"]["learning_rate"], eps=config["ppo"]["adam_epsilon"])
    greedy_actions, _, _ = greedy_fast(problem)
    greedy_cost = problem.scheduling_cost(greedy_actions)
    def costs(model, seed, count, record=False):
        torch.manual_seed(seed)
        samples, values = [], []
        for index in range(count):
            actions, _, sample = model.lower_action(problem, record=record)
            values.append(problem.scheduling_cost(actions))
            samples.append(sample)
        return np.asarray(values), samples
    before, _ = costs(policy, 717, args.samples * 2)
    for iteration in range(args.updates):
        values, samples = costs(policy, 1700 + iteration, args.samples, True)
        buffer = RolloutBuffer("lower", 0, 0)
        for sample, cost in zip(samples, values):
            buffer.append(dict(sample, value=-float(values.mean())), -cost, 0., True, False)
        settings = dict(config["ppo"], value_coefficient=0.)
        update(policy, optimizer, buffer, settings, 0, 0)
        print(json.dumps(dict(probe="fixed_public_state", update=iteration + 1, sampled_cost=float(values.mean()))), flush=True)
    after, _ = costs(policy, 717, args.samples * 2)
    print(json.dumps(dict(probe="fixed_public_state_summary", before=float(before.mean()), after=float(after.mean()),
        paired_difference=float((after - before).mean()), paired_standard_error=float((after - before).std(ddof=1) / np.sqrt(len(after))),
        greedy_cost=greedy_cost, sampled_cost_std=float(before.std()),
        parameters_changed=any(not torch.equal(value, initial_policy.state_dict()[key]) for key, value in policy.state_dict().items()),
        scope="one fixed public state and immediate cost; not full-task convergence")), flush=True)


if __name__ == "__main__":
    main()
