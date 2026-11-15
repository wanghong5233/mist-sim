import copy
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pytest

from mist_sim.anonymity import generate_trace
from mist_sim.config import for_method
from mist_sim.training import Trainer, implementation_signature
from scripts.evaluation import evaluate_frozen as frozen


def fixture_inputs(config, tmp_path):
    config = for_method(copy.deepcopy(config), "hpgi")
    config["horizon"] = 4
    trainer = Trainer(config, 91)
    checkpoint = tmp_path / "last.pt"
    trainer.save(checkpoint)
    method = {"config": config, "checkpoint": str(checkpoint),
              "checkpoint_sha256": frozen.digest(checkpoint),
              "implementation_signature": implementation_signature()}
    path = tmp_path / "input.json"
    frozen.write_json(path, {"seed": 981, "trace": generate_trace(config, 981)})
    task = {"id": "test", "method": "hpgi", "load": "middle", "probability": config["arrival_probability"],
            "action_seed": 10981, "deterministic": False,
            "input": {"seed": 981, "path": str(path), "sha256": frozen.digest(path)}}
    return method, task


def scientific_output(result):
    result = copy.deepcopy(result)
    for name in ("runtime", "wall_seconds", "audit_sha256"):
        result.pop(name)
    return result


def test_frozen_serial_parallel_replay_and_seed_reset(config, tmp_path):
    method, task = fixture_inputs(config, tmp_path)
    frozen.initialize_worker(method, time.time() + 90)
    first = frozen.evaluate_task(task)
    frozen.evaluate_task({**task, "action_seed": 12345})
    replay_task = {**task, "audit_path": str(tmp_path / "audit.json")}
    repeated = frozen.evaluate_task(replay_task)
    assert scientific_output(first) == scientific_output(repeated)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
            initializer=frozen.initialize_worker, initargs=(method, time.time() + 90)) as pool:
        parallel = list(pool.map(frozen.evaluate_task, [task, task]))
    assert all(scientific_output(row) == scientific_output(first) for row in parallel)
    assert frozen.digest(method["checkpoint"]) == method["checkpoint_sha256"]
    assert first["frozen_state_verified"] and first["physics"]["constraints_and_kkt"]
    metrics = first["metrics"]
    assert metrics["arrivals"] == sum(metrics[name] for name in
        ("completed", "suppressed", "rejected", "timeout", "resource_failed"))
    assert len(first["completed_delays"]) == metrics["completed"]


def test_input_source_and_checkpoint_tampering_are_rejected(config, tmp_path):
    method, task = fixture_inputs(config, tmp_path)
    with pytest.raises(ValueError, match="wrong source"):
        frozen.initialize_worker({**method, "implementation_signature": "wrong"}, time.time() + 90)
    with pytest.raises(ValueError, match="Checkpoint checksum"):
        frozen.initialize_worker({**method, "checkpoint_sha256": "wrong"}, time.time() + 90)
    frozen.initialize_worker(method, time.time() + 90)
    task["input"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="input checksum"):
        frozen.evaluate_task(task)


def test_load_override_preserves_every_other_setting(config):
    original = copy.deepcopy(config)
    changed = frozen.evaluation_config(config, .25)
    assert config == original
    assert {key for key in config if changed[key] != config[key]} == {"arrival_probability"}
    with pytest.raises(ValueError, match="outside"):
        frozen.evaluation_config(config, 2.)


def test_pooling_weights_requests_and_uses_raw_delay_quantiles(config, tmp_path):
    method, task = fixture_inputs(config, tmp_path)
    frozen.initialize_worker(method, time.time() + 90)
    first = frozen.evaluate_task(task)
    second = copy.deepcopy(first)
    first["metrics"].update(arrivals=10, completed=1, suppressed=0)
    second["metrics"].update(arrivals=100, completed=90, suppressed=0)
    first["completed_delays"] = [.1]
    second["completed_delays"] = [.9] * 90
    pooled = frozen.aggregate_group([first, second])
    assert pooled["on_time_rate"] == pytest.approx(91 / 110)
    assert pooled["completion_delay_mean"] == pytest.approx(81.1 / 91)
    assert pooled["completion_delay_p95"] == .9
    equal_pairs = frozen.paired_intervals([first, second], [first, second])
    assert all(row["hpgi_minus_comparison"] == row["ci95_low"] == row["ci95_high"] == 0 for row in equal_pairs)
    changed = copy.deepcopy(second)
    changed["input_sha256"] = "different"
    with pytest.raises(ValueError, match="identical"):
        frozen.paired_intervals([first, second], [first, changed])
