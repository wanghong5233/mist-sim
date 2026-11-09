"""One phase-conditioned policy with the existing two action/value networks."""
from torch import nn


class JointPolicy(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.phase_policies = nn.ModuleDict({"upper": controller.upper, "lower": controller.lower})

    @property
    def value_scale(self):
        return self.phase_policies["upper"].value_scale

    @property
    def context_norm(self):
        return self.phase_policies["upper"].context_norm
