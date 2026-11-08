"""No-learning feasibility and constrained-optimum evidence."""
import copy
import json
from pathlib import Path
import pytest
from mist_sim.config import load_config
from mist_sim.preflight import physical_envelope, feasibility_witness, small_optimality_audit
from mist_sim.reference_solvers import enumerate_trajectory


def conditional():
    root = Path(__file__).resolve().parents[1]
    environment = json.loads((root / "configs/environments/conditional.json").read_text(encoding="utf-8"))
    if not (root / "configs/environments" / environment["profile_file"]).is_file():
        pytest.skip("Private measured profiles must be restored")
    return load_config(root / "configs/conditional.json")


def test_worst_public_intervals_have_service_and_old_setting_does_not():
    config = conditional()
    envelope = physical_envelope(config)
    assert all(row["edge_feasible"] for row in envelope.values())
    assert all(row["fastest_edge_seconds"] < config["deadline_range"][0] for row in envelope.values())
    old = copy.deepcopy(config)
    old.update(deadline_range=[.4, 1.], reserved_bandwidth=5e6)
    assert not physical_envelope(old)["resnet152"]["edge_feasible"]


def test_real_profiles_executable_witness_and_replay():
    config = conditional()
    config["horizon"] = 8
    report = feasibility_witness(config, [4100017])
    assert report["passed"]
    modes = report["trajectories"][0]["modes"]
    assert all(modes["all_on"]["completed_by_model"][model] > 0 for model in config["profiles"])
    assert modes["consolidated_local_first"]["cost"] < modes["sleep"]["cost"]


def test_exact_grid_includes_best_fixed_and_no_gain_cases():
    result = small_optimality_audit(conditional())
    assert result["complete"]
    assert all(gap >= -1e-8 for values in result["gaps"].values() for gap in values)
    assert all(row["comparisons"][mode]["completed"] == row["requests"]
               for row in result["cases"] for mode in ("dynamic", "always_on", "fixed_split"))
    best = result["best_fixed_by_model"]
    assert best["resnet152"]["gap"] > 0
    assert best["resnet50"]["gap"] == pytest.approx(0)
    assert best["vit_b_32"]["gap"] == pytest.approx(0)
    limited = small_optimality_audit(conditional(), max_nodes=1)
    assert not limited["complete"] and limited["strict_improvement_cases"] is None
    assert limited["gaps"] is None and limited["best_fixed_by_model"] is None


def test_all_methods_execute_same_real_trace_without_updates():
    import torch
    from mist_sim.config import METHODS, for_method
    from mist_sim.anonymity import generate_trace
    from mist_sim.evaluation import evaluate, replay
    from mist_sim.diagnostics import check_physics
    from mist_sim.policies.hpgi import Controller
    torch.set_num_threads(1)
    config = conditional()
    config["horizon"] = 5
    trace = generate_trace(config, 4100017)
    original = copy.deepcopy(trace)
    for method in METHODS:
        torch.manual_seed(127)
        scenario = for_method(config, method)
        controller = Controller(scenario)
        before = copy.deepcopy(controller.state_dict())
        records, metrics = evaluate(controller, scenario, trace)
        check_physics(scenario, records)
        assert replay(scenario, records) == metrics
        assert metrics["arrivals"] == sum(len(batch) for batch in trace)
        for layer, state in controller.state_dict().items():
            for name, value in state.items():
                torch.testing.assert_close(value, before[layer][name], rtol=0, atol=0)
    assert trace == original


def test_offline_reference_respects_control_interval(config):
    config["servers"] = config["servers"][:1]
    config["initial_active"] = [1]
    config.update(horizon=2, window=2, k=1)
    from tests.test_physics import request
    result = enumerate_trajectory(config, [[request()], []], max_nodes=1000)
    assert result["optimal"]
    assert result["records"][0]["active"] == result["records"][1]["active"]
