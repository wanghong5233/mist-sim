"""Training-only running return statistics, frozen per PPO phase."""
import math
import numpy as np


class RunningRewardNormalizer:
    def __init__(self, config):
        self.enabled = config.get("reward_normalization", {}).get("enabled", False)
        self.gamma = config["ppo"]["gamma"]
        self.clip = config.get("reward_normalization", {}).get("max_abs_slot_reward", 10.)
        self.count, self.mean, self.m2 = 0, 0., 0.
        self.discounted_return = 0.
        self.slot_count, self.slot_mean = 0, 0.

    def observe(self, records):
        for record in records:
            self.slot_count += 1
            self.slot_mean += (record["reward"] - self.slot_mean) / self.slot_count
            if record["slot"] == 0:
                self.discounted_return = 0.
            self.discounted_return = self.gamma * self.discounted_return + record["reward"]
            if not math.isfinite(self.discounted_return):
                raise ValueError("Nonfinite running return")
            self.count += 1
            delta = self.discounted_return - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (self.discounted_return - self.mean)

    def units(self):
        variance = max(0., self.m2 / self.count) if self.count else 0.
        scale = 1 / math.sqrt(max(variance, 1e-8)) if self.enabled and self.count > 1 else 1.
        return dict(mode="running_return_std" if self.enabled else "disabled", scale=scale, shift=0.,
                    clip=self.clip if self.enabled else None, count=self.count, mean=self.mean, variance=variance)

    def state_dict(self):
        return dict(count=self.count, mean=self.mean, m2=self.m2, discounted_return=self.discounted_return,
                    slot_count=self.slot_count, slot_mean=self.slot_mean)

    def load_state_dict(self, state):
        for key in ("count", "mean", "m2", "discounted_return"):
            setattr(self, key, state[key])
        self.slot_count, self.slot_mean = state.get("slot_count", 0), state.get("slot_mean", 0.)


def training_reward(record, units):
    value = record["reward"] * units["scale"]
    if not math.isfinite(value):
        raise ValueError("Nonfinite training reward")
    return float(np.clip(value, -units["clip"], units["clip"])) if units["clip"] is not None else value
