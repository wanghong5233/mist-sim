"""Read-only learning diagnostics; these quantities never enter PPO targets."""
from collections import Counter

import numpy as np

from .buffers import EventRolloutBuffer
from .environment import PublicProblem
from .profiles import Profiles


def advantage_quality(samples, advantages, gamma):
    """Expose time-dependent value bias that aggregate explained variance hides."""
    advantage = np.asarray(advantages, dtype=np.float64)
    if not len(samples) or len(samples) != len(advantage):
        raise ValueError("Nonempty matching samples and advantages required")
    reward = np.asarray([sample["reward"] for sample in samples])
    bootstrap = np.asarray([gamma ** sample["duration"] * sample["next_value"] *
                            (not sample["terminated"]) - sample["value"] for sample in samples])
    reward_std = float(reward.std())
    result = dict(reward_std=reward_std, bootstrap_delta_std=float(bootstrap.std()),
        bootstrap_to_reward_std=float(bootstrap.std() / reward_std) if reward_std > 1e-12 else None,
        advantage_reward_correlation=float(np.corrcoef(advantage, reward)[0, 1])
            if advantage.std() > 1e-12 and reward_std > 1e-12 else None)
    # Only repeated positions from distinct episodes support this descriptive
    # decomposition. Time effects can be legitimate; this is not proof of bias.
    groups = {}
    for index, sample in enumerate(samples):
        if sample.get("slot") is not None and sample.get("episode_id") is not None:
            groups.setdefault(sample["slot"], []).append(index)
    selected, residual = [], []
    for indices in groups.values():
        if len({samples[index]["episode_id"] for index in indices}) > 1:
            values = advantage[indices]
            selected.extend(indices)
            residual.extend(values - values.mean())
    variance = float(np.var(advantage[selected])) if selected else 0.
    result["repeated_position_samples"] = len(selected)
    result["position_mean_variance_fraction"] = float(1 - np.var(residual) / variance) if variance > 1e-12 else None
    terminal = [sample for sample in samples if sample["terminated"]]
    result["terminal"] = dict(count=len(terminal),
        value_mean=float(np.mean([sample["value"] for sample in terminal])) if terminal else None,
        reward_mean=float(np.mean([sample["reward"] for sample in terminal])) if terminal else None,
        td_mean=float(np.mean([sample["reward"] - sample["value"] for sample in terminal])) if terminal else None)
    return result


def clock_baseline_probe(samples, horizon, settings):
    """Offline counterfactual GAE with a leave-one-episode-out clock baseline.

    Applicable only to complete, undiscounted joint episodes. This deliberately
    simple baseline diagnoses critic calibration; it is not a training option.
    """
    if settings["gamma"] != 1 or not samples or any(sample.get("phase") not in ("upper", "lower") for sample in samples):
        return dict(applicable=False, reason="Requires undiscounted joint episodes")
    episodes = {}
    for sample in samples:
        episodes.setdefault(sample.get("episode_id"), []).append(sample)
    if None in episodes or len(episodes) < 2:
        return dict(applicable=False, reason="Requires at least two identified episodes")
    totals, counts = {}, {}
    for episode, rows in episodes.items():
        lower = [row for row in rows if row["phase"] == "lower"]
        if ([row.get("slot") for row in lower] != list(range(horizon)) or
                not lower[-1]["terminated"] or any(row["truncated"] for row in rows) or
                any(row["duration"] != (row["phase"] == "lower") for row in rows)):
            return dict(applicable=False, reason="Requires complete consecutive joint episodes")
        if any(row.get("reward_scale", 0) <= 0 or len(row.get("physical_rewards", [])) != 1 or
               not np.isclose(row["reward"], row["physical_rewards"][0] * row["reward_scale"]) for row in lower):
            return dict(applicable=False, reason="Requires unclipped physical reward and positive scale")
        totals[episode] = sum(row["physical_rewards"][0] for row in lower)
        counts[episode] = len(lower)
    total, count = sum(totals.values()), sum(counts.values())
    means = {episode: (total - totals[episode]) / (count - counts[episode]) for episode in episodes}
    original = EventRolloutBuffer(dict(upper=0, lower=0))
    original.samples = samples
    changed = EventRolloutBuffer(dict(upper=0, lower=0))
    changed.samples = [dict(row,
        value=(horizon - row["slot"]) * means[row["episode_id"]] * row["reward_scale"],
        next_value=(horizon - row["slot"] - row["duration"]) * means[row["episode_id"]] * row["reward_scale"])
        for row in samples]
    result = dict(applicable=True, episodes=len(episodes), scope="Offline GAE only; no policy update", phases={})
    for label, buffer in (("original", original), ("clock", changed)):
        advantages, _ = buffer.advantages(settings["gamma"], settings["gae_lambda"])
        for phase in ("upper", "lower"):
            indices = [index for index, row in enumerate(samples) if row["phase"] == phase]
            if indices:
                result["phases"].setdefault(phase, {})[label] = dict(
                    advantage_std=float(advantages[indices].std()),
                    **advantage_quality([buffer.samples[index] for index in indices], advantages[indices], settings["gamma"]))
    return result


def public_action_quality(config, records):
    """Rebuild legal choices and conservative costs from saved public records."""
    if not records:
        raise ValueError("Physical records required")
    if config["learned_rates"]:
        raise ValueError("This per-slot P3 saving bound requires solver-allocated rates")
    profiles, counts = Profiles(config), Counter()
    savings = Counter()
    utilization = []
    for record in records:
        problem = PublicProblem(config, record["public"], record["active"], record["ready"], record["slot"], [], profiles)
        usage = problem.empty_usage()
        actions = record["executed_actions"]
        if len(actions) != len(problem.choices):
            raise ValueError("Physical action count mismatch")
        _, rates, _, _ = problem.allocate(actions)
        for index, (options, action) in enumerate(zip(problem.choices, actions)):
            choice, legal = options[action], problem.mask(options, usage)
            counts["requests"] += 1
            counts["chosen_" + choice["kind"]] += 1
            if choice["kind"] == "reject":
                reason = ("no_service_candidate" if len(options) == 1 else
                          "prefix_resources_exhausted" if not any(legal[1:]) else "legal_service_available")
                counts["reject_" + reason] += 1
            local = next((option for option in options if option["kind"] == "local"), None)
            if local is not None and choice["kind"] != "local":
                gain = problem.choice_cost(choice, rates[index]) - local["cost_fixed"]
                if gain > 1e-9:
                    counts["dominated_" + choice["kind"] + "_by_local"] += 1
                    savings[choice["kind"]] += gain
            problem.reserve(choice, usage)
        utilization.append([usage["bandwidth"][index] / ap["bandwidth"]
                            for index, ap in enumerate(config["access_points"])])
    slots = len(records)
    return dict(slots=slots, counts=dict(counts),
        guaranteed_local_saving_per_slot={key: value / slots for key, value in savings.items()},
        ap_bandwidth_full_fraction=(np.asarray(utilization) > .999).mean(0).tolist(),
        cost_per_slot=dict(conservative_energy=sum(row["conservative_energy"] for row in records) / slots,
            weighted_delay=config["lambda_time"] * sum(row["weighted_conservative_delay"] for row in records) / slots,
            rejection=sum(row["rejection_penalty"] for row in records) / slots))
