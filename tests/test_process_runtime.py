import concurrent.futures
import copy
import multiprocessing
import sys

import pytest

from mist_sim.config import load_config, validate
from mist_sim.convergence import reward_stability
from mist_sim.process_runtime import configure_process
from mist_sim.schedule import describe_schedule


def windows_probe(_):
    return configure_process(True)


def test_default_process_policy_has_no_os_mutation():
    result = configure_process()
    assert result["status"] == "unchanged" and not result["requested"]


def test_windows_policy_rejects_unsupported_platform(monkeypatch):
    import mist_sim.process_runtime as runtime
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires Windows"):
        configure_process(True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process API")
def test_spawned_process_reads_back_priority_and_explicit_high_qos():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        reports = list(pool.map(windows_probe, range(2)))
    for report in reports:
        assert report["status"] == "applied" and report["priority_class"] == "AboveNormal"
        assert report["high_qos"] and report["power_control_mask"] & 1
        assert not report["power_state_mask"] & 1


def test_hpgi_200_budget_and_no_premature_stability_stop():
    config = load_config("configs/smoke.json", ["configs/algorithms/cuda_update.json",
        "configs/algorithms/hpgi_finite_horizon.json", "configs/training/fixed_200.json",
        "configs/runtime/windows_performance.json"])
    schedule = describe_schedule(config)
    assert schedule["cycles"] == 200
    assert config["method"] == "hpgi" and config["runtime"]["collection_workers"] == 4
    assert config["ppo"]["value_model"] == "finite_horizon" and config["ppo"]["value_target"] == "sampled_return"
    history = [dict(stage=f"cycle_{i:03d}", physical_slots=2048,
                    metrics=dict(conservative_cost=204800)) for i in range(199)]
    assert not reward_stability(history, config["convergence"], config["horizon"])["stop"]
    bad = copy.deepcopy(config)
    bad["runtime"]["windows_performance"] = "yes"
    with pytest.raises(ValueError, match="must be boolean"):
        validate(bad)


def test_trainer_applies_runtime_flag(monkeypatch):
    import mist_sim.training as training
    calls = []
    monkeypatch.setattr(training, "configure_process", lambda enabled: calls.append(enabled) or {"requested": enabled})
    config = load_config("configs/smoke.json")
    config["runtime"]["windows_performance"] = True
    trainer = training.Trainer(config, config["seeds"]["train"])
    assert calls == [True] and trainer.process_policy["requested"]
