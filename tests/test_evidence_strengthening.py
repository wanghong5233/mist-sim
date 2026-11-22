import copy
import time
import pytest
from mist_sim.anonymity import generate_trace
from mist_sim.evaluation import evaluate, replay
from mist_sim.diagnostics import check_physics
from scripts.evaluation import evaluate_frozen as f
from scripts.evaluation.strengthen_evidence import StaticController, server_vectors, static_method, select


def test_static_wakes_once_and_cannot_serve_edge_in_wakeup_slot(config):
    config = copy.deepcopy(config)
    config["initial_active"] = [0] * len(config["servers"])
    active = [1] * len(config["servers"])
    records, metrics = evaluate(StaticController(active), config, generate_trace(config, 17))
    assert all(r["active"] == active for r in records)
    assert records[0]["ready"] == [0] * len(active)
    assert all(t["action"] != "edge" for t in records[0]["tasks"])
    assert metrics["wake_events"] == metrics["switches"] == len(active)
    assert check_physics(config, records)["constraints_and_kkt"]
    assert replay(config, records) == metrics


def test_server_reduction_requires_identical_full_definition(config):
    server = config["servers"][0]
    cfg = {**config, "servers": [copy.deepcopy(server) for _ in range(2)]}
    assert server_vectors(cfg) == [[0, 0], [1, 0], [1, 1]]
    cfg["servers"][1]["wake_energy"] += 1
    assert len(server_vectors(cfg)) == 4


def test_fixed_candidates_and_reserve_do_not_change_raw_input(config):
    cfg = copy.deepcopy(config)
    cfg["dynamic_split"] = False
    cfg["fixed_split"] = {m: p["legal_splits"][0] for m, p in cfg["profiles"].items()}
    trace = generate_trace(cfg, 19)
    before = copy.deepcopy(trace)
    for reserve in (.01, .06, .08):
        changed = f.evaluation_config(cfg, cfg["arrival_probability"], {"control_seconds": reserve})
        assert {k for k in cfg if cfg[k] != changed[k]} <= {"control_seconds"}
        records, _ = evaluate(StaticController([1] * len(cfg["servers"])), changed, trace)
        assert check_physics(changed, records)["constraints_and_kkt"]
        for record in records:
            models = {r["id"]: r["model"] for r in record["raw"]}
            assert all(t["split"] == cfg["fixed_split"][models[t["id"]]] for t in record["tasks"] if t["action"] == "edge")
    assert trace == before
    with pytest.raises(ValueError, match="within"):
        f.evaluation_config(cfg, .5, {"control_seconds": -1})


def test_selection_uses_complete_cost_and_slot_denominator():
    rows = [{"method": "a", "slots": 1, "metrics": {"conservative_cost": 1}},
            {"method": "a", "slots": 9, "metrics": {"conservative_cost": 90}},
            {"method": "b", "slots": 10, "metrics": {"conservative_cost": 80}}]
    winner, scores = select(rows)
    assert winner == "b" and scores["a"] == pytest.approx(9.1)
