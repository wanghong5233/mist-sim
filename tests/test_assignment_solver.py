import itertools
from pathlib import Path

import pytest

from mist_sim.environment import PublicProblem
from mist_sim.config import load_config
from mist_sim.policies.assignment_solver import _Costs, refine_assignments


def test_opt_in_presets_and_overrides_preserve_old_configuration_signatures():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/smoke.json"
    old = load_config(path)
    assert not (set(old) & {"split_solver", "split_refinement_passes", "split_refinement_scope"})
    selected = load_config(path, [root / "configs/algorithms/hpgi_assignment_refined.json"])
    assert selected["split_refinement_scope"] == "assignment"
    overridden = load_config(path, overrides=['dynamic_split=true', 'split_solver="fixed_proposal"', 'split_refinement_scope="assignment"',
                                              'split_refinement_passes=1'])
    assert selected == overridden
    assert load_config(path) == old
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path, [root / "configs/algorithms/hpgi_assignment_refined.json"], ['unknown_solver=true'])


def problem_with_bandwidth_bottleneck():
    problem = PublicProblem.__new__(PublicProblem)
    problem.config = dict(servers=[dict(capacity=10., memory=10.)], access_points=[dict(bandwidth=1.)])
    problem.active = [1]
    reject = dict(kind="reject", cost_fixed=20.)
    local = dict(kind="local", cost_fixed=2.)
    edge = dict(kind="edge", server=0, ap=0, split=2, minimum=1., memory=1., bandwidth=1.,
                cost_fixed=0., alpha=1., beta=0.)
    problem.choices = [[reject, local, edge], [reject, edge]]
    return problem


def test_exchange_finds_improvement_when_no_single_move_can_improve():
    problem = problem_with_bandwidth_bottleneck()
    proposal = [2, 0]
    assert problem.scheduling_cost(proposal) == 21.
    # Moving the first task to local alone costs more; admitting the other alone
    # exceeds AP bandwidth. Their simultaneous exchange costs 2 + 1 = 3.
    assert problem.scheduling_cost([1, 0]) == 22.
    with pytest.raises(ValueError, match="capacity"):
        problem.allocate([2, 1])
    result, detail = refine_assignments(problem, proposal)
    assert result == [1, 1] and proposal == [2, 0]
    assert detail["exchanges"] == 1
    assert detail["refined_cost"] == problem.scheduling_cost(result) == 3.


def test_cached_costs_match_common_allocator_for_every_small_assignment():
    problem = problem_with_bandwidth_bottleneck()
    problem.choices[0][2].update(alpha=.1, beta=20., minimum=.5)
    problem.choices[1][1] = dict(problem.choices[0][2], beta=10.)
    costs = _Costs(problem)
    for actions in itertools.product(range(3), range(2)):
        try:
            expected = problem.scheduling_cost(actions)
        except ValueError:
            expected = float("inf")
        assert costs(actions) == pytest.approx(expected)
    # A lower-bound cut can discard a trial only if it cannot beat the ceiling.
    assert costs([1, 1], 0.) == float("inf")
    assert costs.pruned > 0


def test_refinement_checks_input_and_preserves_admission_count():
    problem = problem_with_bandwidth_bottleneck()
    for actions in itertools.product(range(3), range(2)):
        try:
            initial = problem.scheduling_cost(actions)
        except ValueError:
            with pytest.raises(ValueError):
                refine_assignments(problem, actions)
            continue
        result, detail = refine_assignments(problem, actions, 2)
        accepted = lambda values: sum(problem.choices[i][a]["kind"] != "reject" for i, a in enumerate(values))
        assert accepted(result) >= accepted(actions)
        assert problem.scheduling_cost(result) <= initial + 1e-9
        assert detail["refined_cost"] == pytest.approx(problem.scheduling_cost(result))
