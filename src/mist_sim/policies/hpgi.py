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
        self.upper = SetPolicy(layer_settings(config, "upper")["hidden"])
        self.lower = SetPolicy(layer_settings(config, "lower")["hidden"], config["learned_rates"])

    def slow(self, observation, deterministic=False, pretrain=False):
        if pretrain or self.method == "always_on":
            return [1] * len(self.config["servers"]), None
        if self.method == "greedy_consolidation":
            return greedy_slow(observation), None
        return self.upper.upper_action(observation, deterministic)

    def fast(self, problem, deterministic=False):
        if not self.learn_lower:
            return greedy_fast(problem)
        return self.lower.lower_action(problem, deterministic)

    def state_dict(self):
        return {"upper": self.upper.state_dict(), "lower": self.lower.state_dict()}

    def load_state_dict(self, state):
        self.upper.load_state_dict(state["upper"])
        self.lower.load_state_dict(state["lower"])
