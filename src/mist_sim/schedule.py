"""Training work units and deterministic interaction budgets."""
import math


class TrainingLimit(TimeoutError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def rollout_slots(config, layer, count):
    if layer in ("lower", "joint"):
        return count
    intervals = math.ceil(config["horizon"] / config["window"])
    episodes, tail = divmod(count, intervals)
    return episodes * config["horizon"] + min(tail * config["window"], config["horizon"])


def describe_schedule(config, extra_cycles=0):
    joint = config["method"] == "joint_ppo"
    lower = config["method"] not in ("upper_ppo_greedy", "greedy_consolidation")
    upper = config["method"] not in ("always_on", "greedy_consolidation")
    training = config["training"]
    cycles = training["cycles"] + extra_cycles
    pretrain = training["pretrain_slots"] if lower else 0
    fast = training["fast_slots"] if lower else 0
    slow = rollout_slots(config, "upper", training["slow_intervals"]) if upper else 0
    joint_upper = event_count(config, fast) - fast if joint else 0
    return {"method": config["method"], "episode_slots": config["horizon"], "control_window": config["window"],
            "sampling_device": "cpu", "update_device": config["device"],
            "collection_workers": config.get("runtime", {}).get("collection_workers", 0),
            "pretrain_slots": pretrain, "pretrain_rollouts": math.ceil(pretrain / training["fast_slots"]),
            "cycles": cycles, "fast_transitions_per_rollout": fast,
            "slow_transitions_per_rollout": joint_upper if joint else training["slow_intervals"] if upper else 0,
            "slow_physical_slots_per_rollout": fast if joint else slow,
            "joint_event_transitions_per_rollout": fast + joint_upper if joint else 0,
            "planned_training_slots": pretrain + cycles * (fast if joint else fast + slow),
            "calibration_slots": training["calibration_episodes"] * config["horizon"],
            "validation_enabled": config.get("validation", {}).get("enabled", True),
            "validation_trajectories_per_check": config.get("validation", {}).get("trajectories", 1)
                if config.get("validation", {}).get("enabled", True) else 0,
            "stopping": config.get("stopping", {"max_slots": None, "max_seconds": None}),
            "ppo": {layer: {"epochs": settings["epochs"], "minibatch": settings["minibatch"],
                              "learning_rate": settings["learning_rate"], "learning_rate_schedule": "constant",
                              "learning_rate_warmup": False,
                              "advantage_normalization": settings.get("advantage_normalization", "global"),
                              "minibatches_per_epoch": math.ceil(count / settings["minibatch"]),
                              "target_kl": settings["target_kl"]}
                    for layer, count, settings in _layers(config, lower, upper)},
            "convergence_rule": config.get("convergence", {"enabled": False}),
            "checkpoint_policy": config.get("checkpoint_policy", {})}


def _layers(config, lower, upper):
    from .config import layer_settings
    if config["method"] == "joint_ppo":
        yield "joint", event_count(config, config["training"]["fast_slots"]), config["ppo"]
        return
    for layer, enabled, key in (("lower", lower, "fast_slots"), ("upper", upper, "slow_intervals")):
        if enabled:
            yield layer, config["training"][key], layer_settings(config, layer)


def event_count(config, slots):
    episodes, tail = divmod(slots, config["horizon"])
    return slots + episodes * math.ceil(config["horizon"] / config["window"]) + math.ceil(tail / config["window"])
