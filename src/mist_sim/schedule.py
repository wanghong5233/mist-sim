"""Training work units and deterministic interaction budgets."""
import math


class TrainingLimit(TimeoutError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def rollout_slots(config, layer, count):
    if layer == "lower":
        return count
    intervals = math.ceil(config["horizon"] / config["window"])
    episodes, tail = divmod(count, intervals)
    return episodes * config["horizon"] + min(tail * config["window"], config["horizon"])


def describe_schedule(config, extra_cycles=0):
    lower = config["method"] not in ("upper_ppo_greedy", "greedy_consolidation")
    upper = config["method"] not in ("always_on", "greedy_consolidation")
    training = config["training"]
    cycles = training["cycles"] + extra_cycles
    pretrain = training["pretrain_slots"] if lower else 0
    fast = training["fast_slots"] if lower else 0
    slow = rollout_slots(config, "upper", training["slow_intervals"]) if upper else 0
    return {"method": config["method"], "episode_slots": config["horizon"], "control_window": config["window"],
            "pretrain_slots": pretrain, "pretrain_rollouts": math.ceil(pretrain / training["fast_slots"]),
            "cycles": cycles, "fast_transitions_per_rollout": fast,
            "slow_transitions_per_rollout": training["slow_intervals"] if upper else 0,
            "slow_physical_slots_per_rollout": slow, "planned_training_slots": pretrain + cycles * (fast + slow),
            "calibration_slots": training["calibration_episodes"] * config["horizon"],
            "validation_trajectories_per_check": config.get("validation", {}).get("trajectories", 1),
            "stopping": config.get("stopping", {"max_slots": None, "max_seconds": None}),
            "ppo": {layer: {"epochs": settings["epochs"], "minibatch": settings["minibatch"],
                              "minibatches_per_epoch": math.ceil(count / settings["minibatch"]),
                              "target_kl": settings["target_kl"]}
                    for layer, count, settings in _layers(config, lower, upper)},
            "convergence_rule": "No loss-based stopping; inspect fixed-validation return, cost and service"}


def _layers(config, lower, upper):
    from .config import layer_settings
    for layer, enabled, key in (("lower", lower, "fast_slots"), ("upper", upper, "slow_intervals")):
        if enabled:
            yield layer, config["training"][key], layer_settings(config, layer)
