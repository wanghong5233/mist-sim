"""Reward stability on complete joint-policy evaluations, not PPO loss."""
import numpy as np


DEFAULT_CONVERGENCE = dict(enabled=False, min_cycles=100, window=20,
    relative_mean_tolerance=.005, relative_trend_tolerance=.01,
    relative_std_tolerance=.02, patience=5, tail_cycles=20)


def reward_stability(history, settings, horizon):
    cycles = [row for row in history if row["stage"].startswith("cycle_")]
    count, window = len(cycles), settings["window"]
    result = dict(enabled=settings["enabled"], cycles=count, ready=False,
                  stable=False, stable_checks=0, detected_cycle=None,
                  tail_remaining=None, stop=False, modes={})
    if not settings["enabled"] or count < max(settings["min_cycles"], 2 * window):
        return result
    rows = cycles[-2 * window:]
    modes = ["metrics"]
    if any("sampled_metrics" in row for row in rows):
        if not all("sampled_metrics" in row for row in rows):
            raise ValueError("Sampled evaluation coverage changed within stability window")
        modes.append("sampled_metrics")
    for mode in modes:
        values = np.asarray([-row[mode]["conservative_cost"] /
                             (len(row["trajectories"]) * horizon) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite convergence reward")
        previous, recent = values[:window], values[window:]
        scale = max(abs(float(previous.mean())), 1.)
        mean_change = abs(float(recent.mean() - previous.mean())) / scale
        offsets = np.arange(len(values), dtype=float)
        offsets -= offsets.mean()
        trend = abs(float(np.dot(offsets, values - values.mean()) / np.dot(offsets, offsets))) * (len(values) - 1) / scale
        spread = float(recent.std()) / scale
        result["modes"][mode] = dict(previous_reward_mean=float(previous.mean()),
            reward_mean=float(recent.mean()), relative_mean_change=mean_change,
            relative_trend=trend, relative_std=spread,
            stable=(mean_change <= settings["relative_mean_tolerance"] and
                    trend <= settings["relative_trend_tolerance"] and
                    spread <= settings["relative_std_tolerance"]))
    result["ready"] = True
    result["stable"] = all(mode["stable"] for mode in result["modes"].values())
    previous = cycles[-2].get("convergence", {}) if count > 1 else {}
    if result["stable"]:
        result["stable_checks"] = previous.get("stable_checks", 0) + 1
        result["detected_cycle"] = previous.get("detected_cycle")
        if result["detected_cycle"] is None and result["stable_checks"] >= settings["patience"]:
            result["detected_cycle"] = count
        if result["detected_cycle"] is not None:
            result["tail_remaining"] = max(0, settings["tail_cycles"] - (count - result["detected_cycle"]))
            result["stop"] = result["tail_remaining"] == 0
    return result
