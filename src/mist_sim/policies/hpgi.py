"""Method registry and shared HPGI components; no silent fallback."""
import torch
from ..config import METHODS, layer_settings
from .networks import SetPolicy
from .baselines.heuristics import greedy_fast, greedy_slow


class Controller:
    def __init__(self, config):
        if config["method"] not in METHODS:
            raise ValueError("Unimplemented method")
        self.config = config
        self.method = config["method"]
        self.learn_upper = self.method not in ("always_on", "greedy_consolidation")
        self.learn_lower = self.method not in ("upper_ppo_greedy", "greedy_consolidation")
        def policy(layer, rates=False):
            settings = layer_settings(config, layer)
            return SetPolicy(settings["hidden"], rates, settings.get("value_model", "plain"),
                             config["horizon"], settings["gamma"])
        self.upper = policy("upper")
        self.lower = policy("lower", config["learned_rates"])

    def slow(self, observation, deterministic=False, pretrain=False, record=True, cached=None):
        if pretrain or self.method == "always_on":
            return [1] * len(self.config["servers"]), None
        if self.method == "greedy_consolidation":
            return greedy_slow(observation), None
        return self.upper.upper_action(observation, deterministic, record=record, cached=cached)

    def fast(self, problem, deterministic=False, record=True, cached=None):
        if not self.learn_lower:
            return greedy_fast(problem)
        if self.config.get("split_solver", "policy") == "fixed_proposal":
            from .split_solver import proposal_problem, map_actions, refine_splits
            proposal = proposal_problem(problem)
            actions, predicted, sample = self.lower.lower_action(proposal, deterministic, record=record, cached=cached)
            mapped = map_actions(proposal, problem, actions)
            passes = self.config.get("split_refinement_passes", 1)
            if passes:
                executed, detail = refine_splits(problem, mapped, passes)
            else:
                executed, detail = mapped, dict(changes=0, p3_solves=0)
            if sample is not None:
                # PPO scores the sampled latent proposal. The solver is part of
                # its transition kernel; executed actions are never relabeled.
                sample["executed_actions"] = executed
                sample["split_refinement"] = detail
            return executed, predicted, sample
        return self.lower.lower_action(problem, deterministic, record=record, cached=cached)

    def state_dict(self):
        return {"upper": self.upper.state_dict(), "lower": self.lower.state_dict()}

    def load_state_dict(self, state):
        self.upper.load_state_dict(state["upper"])
        self.lower.load_state_dict(state["lower"])
