"""On-policy buffers. Each item is a slot or an actual-length interval."""
import numpy as np


def normalize_advantages(advantages, samples, mode="global"):
    if mode not in ("global", "phase"):
        raise ValueError("Invalid advantage normalization")
    result = np.asarray(advantages, dtype=np.float64).copy()
    if len(result) != len(samples):
        raise ValueError("Advantage/sample count mismatch")
    groups = np.asarray([sample.get("phase", "single") for sample in samples]) if mode == "phase" else np.zeros(len(samples))
    for group in np.unique(groups):
        selected = groups == group
        values = result[selected]
        if len(values) > 1:
            result[selected] = (values - values.mean()) / max(values.std(), 1e-8)
    return result


class RolloutBuffer:
    def __init__(self, layer, behavior_version, opponent_version):
        self.layer = layer
        self.behavior_version = behavior_version
        self.opponent_version = opponent_version
        self.samples = []
        self.consumed = False

    def append(self, sample, reward, next_value, terminated, truncated, duration=1, next_observation=None):
        if self.consumed or duration < 1:
            raise RuntimeError("Invalid buffer lifecycle/duration")
        self.samples.append(dict(sample, reward=float(reward), next_value=float(next_value),
                                 terminated=bool(terminated), truncated=bool(truncated),
                                 duration=duration, next_observation=next_observation,
                                 behavior_version=self.behavior_version, opponent_version=self.opponent_version))

    def advantages(self, gamma, gae_lambda):
        advantage = 0.0
        values = np.zeros(len(self.samples), dtype=np.float64)
        for index in reversed(range(len(self.samples))):
            sample = self.samples[index]
            discount = gamma ** sample["duration"]
            continuation = not sample["terminated"]
            delta = sample["reward"] + discount * sample["next_value"] * continuation - sample["value"]
            recurse = continuation and not sample["truncated"] and index + 1 < len(self.samples)
            advantage = delta + discount * gae_lambda * advantage * recurse
            values[index] = advantage
        return values, values + np.array([sample["value"] for sample in self.samples])

    def consume(self, behavior_version, opponent_version):
        if self.consumed or behavior_version != self.behavior_version or opponent_version != self.opponent_version:
            raise RuntimeError("Stale on-policy buffer")
        if any(sample["behavior_version"] != behavior_version or sample["opponent_version"] != opponent_version
               for sample in self.samples):
            raise RuntimeError("Mixed behavior/opponent versions")
        self.consumed = True

    def clear(self):
        self.samples.clear()


class EventRolloutBuffer(RolloutBuffer):
    def __init__(self, versions):
        snapshot = tuple(versions[layer] for layer in ("upper", "lower"))
        super().__init__("joint", snapshot, snapshot)

    def append(self, sample, reward, next_value, terminated, truncated, duration=1, next_observation=None):
        expected = {"upper": 0, "lower": 1}.get(sample.get("phase"))
        if self.consumed or expected is None or duration != expected:
            raise RuntimeError("Invalid joint event lifecycle/duration")
        if sample["phase"] == "upper" and (reward != 0 or terminated or truncated):
            raise ValueError("A server decision must lead to its scheduling event")
        self.samples.append(dict(sample, reward=float(reward), next_value=float(next_value),
            terminated=bool(terminated), truncated=bool(truncated), duration=duration,
            next_observation=next_observation, behavior_version=self.behavior_version,
            opponent_version=self.opponent_version))

    def advantages(self, gamma, gae_lambda):
        advantage = 0.0
        values = np.zeros(len(self.samples), dtype=np.float64)
        for index in reversed(range(len(self.samples))):
            sample = self.samples[index]
            delta = sample["reward"] - sample["value"]
            if not sample["terminated"]:
                delta += gamma ** sample["duration"] * sample["next_value"]
            recurse = not (sample["terminated"] or sample["truncated"]) and index + 1 < len(self.samples)
            advantage = delta + (gamma * gae_lambda) ** sample["duration"] * advantage * recurse
            values[index] = advantage
        return values, values + np.array([sample["value"] for sample in self.samples])
