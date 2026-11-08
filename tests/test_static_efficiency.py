"""Operation-count checks: no training or timing-sensitive assertions."""
import hashlib
from pathlib import Path
from unittest.mock import Mock
from mist_sim.anonymity import publish
from mist_sim.environment import PublicProblem, candidates
from mist_sim.profiles import Profiles
from mist_sim.policies.baselines.heuristics import greedy_fast
from mist_sim.storage import digest_file
from tests.test_physics import request


def test_profile_bounds_once_per_split_and_request(config):
    config["k"] = 1
    public = publish([request()], config)[0]
    profiles = Profiles(config)
    profiles.bound = Mock(wraps=profiles.bound)
    choices = candidates(config, profiles, public, [1, 1])
    calls = [call.args[2] for call in profiles.bound.call_args_list]
    assert len(calls) == len(set(calls))
    assert set(calls) == {None, *config["profiles"]["toy"]["legal_splits"]}
    expected = candidates(config, Profiles(config), public, [1, 0])
    assert [choice for choice in choices if choice["server"] != 1] == expected


def test_allocator_only_checks_chosen_candidates(config, monkeypatch):
    config["k"] = 1
    problem = PublicProblem(config, publish([request()], config), [1, 1], [1, 1], 0, [])
    mask = Mock(wraps=problem.mask)
    monkeypatch.setattr(problem, "mask", mask)
    import mist_sim.environment as environment
    solver = Mock(wraps=environment.solve_p3)
    monkeypatch.setattr(environment, "solve_p3", solver)
    problem.allocate([0])
    assert all(len(call.args[0]) == 1 for call in mask.call_args_list)
    assert solver.call_count == 0


def test_greedy_cost_reuse_preserves_actions(config):
    config["k"] = 1
    public = publish([request("a"), request("b")], config)
    problem = PublicProblem(config, public, [1, 1], [1, 1], 0, [])
    actions = [0] * len(problem.choices)
    usage = problem.empty_usage()
    for index, options in enumerate(problem.choices):
        best, cost = 0, problem.scheduling_cost(actions)
        for action, legal in enumerate(problem.mask(options, usage)):
            if not action or not legal:
                continue
            trial = list(actions)
            trial[index] = action
            trial_cost = problem.scheduling_cost(trial)
            if trial_cost < cost - 1e-12:
                best, cost = action, trial_cost
        actions[index] = best
        problem.reserve(options[best], usage)
    assert greedy_fast(problem)[0] == actions


def test_digest_streams_large_files(tmp_path, monkeypatch):
    contents = b"bounded-memory-hash" * 200000
    path = tmp_path / "trace.bin"
    path.write_bytes(contents)
    def forbidden(*args, **kwargs):
        raise AssertionError("Whole-file read")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert digest_file(path) == hashlib.sha256(contents).hexdigest()
