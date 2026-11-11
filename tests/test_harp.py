"""Algorithm identity, proposal/execution separation and explicit transfer."""
import copy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.config import METHODS, OPTIONAL_SOLVER_KEYS, for_method, load_config, validate
from mist_sim.environment import Environment
from mist_sim.evaluation import replay
from mist_sim.policies import harp
from mist_sim.policies.hpgi import HPGIController, Controller
from mist_sim.policies.legacy_refined import LegacyRefinedController
from mist_sim.policies.registry import make_controller
from mist_sim.policies.batching import replay_batch
from mist_sim.training import Trainer
from mist_sim.transfer import initialize_transfer


ROOT = Path(__file__).resolve().parents[1]


def test_method_identity_and_switching_does_not_leak_solver(config):
    original = copy.deepcopy(config)
    target = for_method(config, "harp")
    assert type(make_controller(target)) is harp.HARPController
    assert type(Controller(for_method(config, "hpgi"))) is HPGIController
    assert config == original
    for method in METHODS:
        if method != "harp":
            switched = for_method(target, method)
            expected = for_method(config, method)
            assert switched == expected
            assert not (OPTIONAL_SOLVER_KEYS & switched.keys())
    legacy = dict(config, split_solver="fixed_proposal")
    assert type(make_controller(legacy)) is LegacyRefinedController
    assert legacy == for_method(legacy, "hpgi")
    for name in ("harp.json", "harp_finite_horizon.json"):
        loaded = load_config(ROOT / "configs/smoke.json", [ROOT / "configs/algorithms" / name])
        assert type(make_controller(loaded)) is harp.HARPController


@pytest.mark.parametrize("change", [dict(split_solver="policy", split_refinement_passes=0),
    dict(split_refinement_passes=0), dict(split_refinement_scope="splits"),
    dict(dynamic_split=False), dict(capacity_mask=False), dict(learned_rates=True),
    dict(information="full")])
def test_harp_cannot_silently_become_another_algorithm(config, change):
    with pytest.raises(ValueError):
        make_controller(dict(for_method(config, "harp"), **change))


def test_original_hpgi_never_calls_harp_solver(config, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("HPGI must not refine assignments"))
    monkeypatch.setattr(harp, "refine_assignments", forbidden)
    hpgi = for_method(config, "hpgi")
    env = Environment(hpgi, generate_trace(hpgi, 91))
    make_controller(hpgi).fast(env.begin_slot([1] * len(hpgi["servers"])))
    forbidden.assert_not_called()


def test_harp_matches_legacy_execution_and_scores_original_proposal(config, monkeypatch):
    torch.set_num_threads(1)
    target = for_method(config, "harp")
    target.update(horizon=5, k=1, arrival_probability=1.)
    legacy = dict(target, method="hpgi")
    validate(legacy)
    current, previous = make_controller(target), make_controller(legacy)
    previous.load_state_dict(current.state_dict())
    solver = Mock(wraps=harp.refine_assignments)
    monkeypatch.setattr(harp, "refine_assignments", solver)
    trace = generate_trace(target, 91)
    left, right = Environment(target, trace), Environment(legacy, trace)
    while not left.done:
        active, _ = current.slow(left.upper_observation(), deterministic=True)
        assert previous.slow(right.upper_observation(), deterministic=True)[0] == active
        a, b = left.begin_slot(active), right.begin_slot(active)
        rng = torch.get_rng_state()
        actions, _, sample = current.fast(a)
        torch.set_rng_state(rng)
        older, _, old_sample = previous.fast(b)
        assert sample == old_sample and actions == older
        assert sample["executed_actions"] == actions
        assert sample["actions"] == [step["action"] for step in sample["steps"]]
        assert float(current.lower.replay(sample)[0]) == pytest.approx(sample["logprob"], abs=1e-5)
        assert float(replay_batch(current.lower, [sample])[0][0]) == pytest.approx(sample["logprob"], abs=1e-5)
        assert not a.allocate(actions)[3]
        assert left.execute(actions)["tasks"] == right.execute(older)["tasks"]
    assert solver.call_count == 2 * target["horizon"]
    replay(target, left.records)
    replay(legacy, right.records)


def test_transfer_is_explicit_and_not_implicit_initialization(config, tmp_path):
    config["ppo"]["hidden"] = 16
    config["training"]["calibration_episodes"] = 1
    parent = Trainer(for_method(config, "fixed_split"), 91)
    parent.calibrate()
    checkpoint = tmp_path / "parent.pt"
    parent.save(checkpoint)
    target = for_method(parent.config, "harp")
    fresh = Trainer(target, 92)
    assert fresh.transfer_provenance is None and not fresh.completed_stages
    assert any(not torch.equal(value, fresh.controller.state_dict()[layer][name])
               for layer, weights in parent.controller.state_dict().items()
               for name, value in weights.items())
    transferred = initialize_transfer(checkpoint, target, 92)
    assert transferred.total_slots == 0 and transferred.completed_stages == ["pretrain"]
    assert transferred.transfer_provenance["parent"]["config_signature"]
    assert all(not optimizer.state for optimizer in transferred.optimizers.values())
    for layer, weights in parent.controller.state_dict().items():
        for name, value in weights.items():
            torch.testing.assert_close(value, transferred.controller.state_dict()[layer][name])
    changed = copy.deepcopy(target)
    changed["rejection_cost"] *= 2
    with pytest.raises(ValueError, match="contract"):
        initialize_transfer(checkpoint, changed, 92)
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.load(checkpoint, target)
