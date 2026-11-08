"""On-policy buffers. Each item is a slot or an actual-length interval."""
import numpy as np


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
