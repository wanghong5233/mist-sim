"""Compatibility for historical hpgi+fixed_proposal experiments, not HPGI."""
from ..config import validate
from .base import BaseController
from .harp import HARPController
from .split_solver import refine_splits


class LegacyRefinedController(HARPController):
    def __init__(self, config):
        if config["method"] != "hpgi" or config.get("split_solver") != "fixed_proposal":
            raise ValueError("Legacy refinement requires its original hpgi configuration")
        validate(config)
        BaseController.__init__(self, config)

    def _refine(self, problem, mapped):
        passes = self.config.get("split_refinement_passes", 1)
        if not passes:
            return mapped, dict(changes=0, p3_solves=0)
        if self.config.get("split_refinement_scope", "splits") == "assignment":
            return super()._refine(problem, mapped)
        return refine_splits(problem, mapped, passes)
