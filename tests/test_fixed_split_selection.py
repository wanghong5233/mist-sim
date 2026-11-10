"""The fixed-cut selector compares complete paired workloads and the real cost."""
import copy
import json
from pathlib import Path
import shutil

import pytest
from mist_sim.anonymity import generate_trace
from mist_sim.config import load_config
from mist_sim.storage import digest_file, write_json
from scripts.evaluation.select_fixed_split import candidate_cuts, evaluate_candidate, rank_candidates, selection_config
from scripts.evaluation.evaluate_greedy import scenario
from scripts.training.train_fixed_split import selected_training_config


def test_all_combinations_and_fixed_environment():
    config = load_config("configs/smoke.json")
    candidates = candidate_cuts(config)
    expected = 1
    for model in config["profiles"].values():
        expected *= len(model["legal_splits"])
    assert len(candidates) == expected
    assert len({json.dumps(row, sort_keys=True) for row in candidates}) == expected
    resolved = selection_config(config, candidates[-1])
    assert resolved["dynamic_split"] is False
    assert resolved["fixed_split"] == candidates[-1]
    assert resolved["information"] == "robust"
    assert resolved["servers"] == config["servers"]


def test_rank_pooled_cost_and_require_paired_inputs():
    candidates = [{"model": 1}, {"model": 2}]
    rows = [dict(candidate=i, trajectory_seed=seed, physical_slots=slots,
                 cost=cost, energy=cost, arrivals=2, completed=1, physics_passed=True)
            for i, seed, slots, cost in [(0, 10, 1, 8), (0, 20, 3, 0), (1, 10, 1, 0), (1, 20, 3, 9)]]
    ranked = rank_candidates(candidates, rows, [10, 20])
    # Pool by slots, not an unweighted average of the two per-trajectory means.
    assert ranked[0]["model"] == 1
    assert ranked[0]["cost_per_slot"] == 2
    with pytest.raises(ValueError, match="same complete"):
        rank_candidates(candidates, rows[:-1], [10, 20])
    invalid = copy.deepcopy(rows)
    invalid[0]["cost"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        rank_candidates(candidates, invalid, [10, 20])


def test_real_environment_uses_declared_cuts_and_preserves_inputs(tmp_path):
    config = load_config("configs/smoke.json", overrides=["horizon=4", "k=1", "terminals=3"])
    original = copy.deepcopy(config)
    path = tmp_path / "input.json"
    write_json(path, dict(seed=12345, trace=generate_trace(config, 12345)))
    before = digest_file(path)
    candidates = candidate_cuts(config)
    rows = [evaluate_candidate(config, index, cuts, path, before) for index, cuts in enumerate(candidates)]
    ranked = rank_candidates(candidates, rows, [12345])
    assert ranked[0]["cost_per_slot"] == min(row["cost_per_slot"] for row in rows)
    assert all(row["physical_slots"] == 4 and row["physics_passed"] for row in rows)
    assert digest_file(path) == before
    assert config == original


def test_selected_cuts_reach_actual_training_and_tampering_fails(tmp_path):
    base = load_config("configs/smoke.json")
    write_json(tmp_path / "configs/conditional.json", base)
    for name in ["algorithms/hpgi_finite_horizon.json", "algorithms/cuda_update.json",
                 "training/fixed_300.json", "runtime/windows_performance.json"]:
        target = tmp_path / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path("configs") / name, target)
    candidates = candidate_cuts(base)
    selected = candidates[-1]
    suite = tmp_path / "suite"
    write_json(suite / "manifest.json", dict(scenario=scenario(base)))
    name = "selection-test"
    write_json(tmp_path / "logs/runs" / name / "manifest.json", dict(status="completed", exit_code=0))
    result = tmp_path / "results" / name
    write_json(result / "selected_overlay.json", dict(fixed_split=selected))
    write_json(result / "selection.json", dict(
        candidates=candidates, selected=selected,
        ranking=[dict(candidate=index, **cuts) for index, cuts in reversed(list(enumerate(candidates)))],
        overlay_sha256=digest_file(result / "selected_overlay.json"),
        suite_sha256=digest_file(suite / "manifest.json")))
    config, _ = selected_training_config(tmp_path, name, suite, 123456)
    assert config["method"] == "fixed_split"
    assert config["fixed_split"] == selected
    assert config["training"]["cycles"] == config["convergence"]["min_cycles"] == 300
    assert config["seeds"]["train"] == 123456
    assert config["device"] == "cuda" and config["runtime"]["collection_workers"] == 2
    write_json(result / "selected_overlay.json", dict(fixed_split=candidates[0], device="cpu"))
    with pytest.raises(ValueError, match="changed after scoring"):
        selected_training_config(tmp_path, name, suite, 123456)
