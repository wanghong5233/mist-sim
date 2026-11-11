"""Shared networks, slow control and checkpoint layout for simulator methods."""
from ..config import layer_settings
from .networks import SetPolicy
from .baselines.heuristics import greedy_fast, greedy_slow


class BaseController:
    def __init__(self, config):
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
        return self.lower.lower_action(problem, deterministic, record=record, cached=cached)

    def state_dict(self):
        return {"upper": self.upper.state_dict(), "lower": self.lower.state_dict()}

    def load_state_dict(self, state):
        self.upper.load_state_dict(state["upper"])
        self.lower.load_state_dict(state["lower"])
