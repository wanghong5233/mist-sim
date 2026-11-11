"""HARP: Hierarchical Assignment Refinement with PPO.

PPO samples a fixed-cut proposal; public feasible assignment refinement uses
P3 to evaluate dynamic execution choices. PPO always scores the proposal.
"""
from ..config import validate
from .base import BaseController
from .split_solver import proposal_problem, map_actions
from .assignment_solver import refine_assignments


class HARPController(BaseController):
    def __init__(self, config):
        if config["method"] != "harp":
            raise ValueError("HARP requires method=harp")
        validate(config)
        super().__init__(config)

    def _refine(self, problem, mapped):
        return refine_assignments(problem, mapped, self.config.get("split_refinement_passes", 1))

    def fast(self, problem, deterministic=False, record=True, cached=None):
        proposal = proposal_problem(problem)
        actions, predicted, sample = self.lower.lower_action(
            proposal, deterministic, record=record, cached=cached)
        mapped = map_actions(proposal, problem, actions)
        executed, detail = self._refine(problem, mapped)
        if sample is not None:
            # The deterministic solver belongs to the transition kernel.
            # Never replace the proposal actions/logprob used by PPO.
            sample["executed_actions"] = executed
            sample["split_refinement"] = detail
        return executed, predicted, sample
