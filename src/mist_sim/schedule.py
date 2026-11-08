"""Training work units and deterministic interaction budgets."""
import math


class TrainingLimit(TimeoutError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def plateau_status(history, settings):
    """Reconstruct patience from checkpointed validation history, never training loss."""
    best, bad, stopped = None, 0, False
    sampled_best = None
    for record in history:
        if not record["stage"].startswith("cycle_") or record["total_slots"] < settings["min_slots"]:
            continue
        cost = record["metrics"]["conservative_cost"]
        if not record["eligible"] or not math.isfinite(cost):
            best, bad = None, 0
            continue
        improved = best is None or best - cost > settings["min_relative_improvement"] * max(abs(best), 1e-12)
        if improved:
            best = cost
        sampled = record.get("sampled_metrics", {}).get("conservative_cost")
        if sampled is not None:
            if not math.isfinite(sampled) or not record.get("sampled_eligible", True):
                bad, sampled_best = 0, None
                continue
            if sampled_best is None or sampled_best - sampled > settings["min_relative_improvement"] * max(abs(sampled_best), 1e-12):
                sampled_best, improved = sampled, True
        if improved:
            bad = 0
        else:
            bad += 1
        if settings["enabled"] and bad >= settings["patience"]:
            stopped = True
            break
    return {"stop": stopped, "bad_checks": bad, "reference_cost": best,
            "sampled_reference_cost": sampled_best,
            "monitored_modes": "deterministic_and_sampled" if sampled_best is not None else "deterministic"}


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
                              "learning_rate": settings["learning_rate"], "learning_rate_schedule": "constant",
                              "learning_rate_warmup": False,
                              "minibatches_per_epoch": math.ceil(count / settings["minibatch"]),
                              "target_kl": settings["target_kl"]}
                    for layer, count, settings in _layers(config, lower, upper)},
            "convergence_rule": "No loss-based stopping; inspect fixed-validation return, cost and service"}


def _layers(config, lower, upper):
    from .config import layer_settings
    for layer, enabled, key in (("lower", lower, "fast_slots"), ("upper", upper, "slow_intervals")):
        if enabled:
            yield layer, config["training"][key], layer_settings(config, layer)
