import copy
import csv
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.cli import main
from mist_sim.config import METHODS, for_method, load_config
from mist_sim.diagnostics import inspect_run
from mist_sim.environment import Environment
from mist_sim.evaluation import assert_close
from mist_sim.monitoring import ConsoleProgress
from mist_sim.policies.baselines import heuristics
from mist_sim.schedule import describe_schedule
from mist_sim.storage import digest_file, write_json
from mist_sim.training import Trainer
from mist_sim.verification import independent_totals


ROOT = Path(__file__).resolve().parents[1]
LEARNING_BASELINES = [method for method in METHODS if method not in
                      ("hpgi", "joint_ppo", "greedy_consolidation")]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def baseline_config(config, device="cpu"):
    config.update(horizon=5, k=1, device=device, arrival_probability=1.)
    config["ppo"].update(hidden=16, epochs=1, minibatch=4)
    config["training"].update(pretrain_slots=7, fast_slots=7, slow_intervals=3,
                              cycles=1, calibration_episodes=1)
    config["runtime"]["collection_workers"] = 2 if device == "cuda" else 0
    config["validation"]["enabled"] = False
    config["reward_normalization"]["enabled"] = True
    config["monitoring"] = {"console": False, "tensorboard": False}
    config["checkpoint_policy"]["mode"] = "minimal"
    return config


def test_layer_overrides_use_same_validation_as_overlays():
    config = load_config(ROOT / "configs/smoke.json", overrides=[
        "ppo_layers.upper.minibatch=8", "ppo_layers.lower.learning_rate=0.0001"])
    assert config["ppo_layers"] == {"upper": {"minibatch": 8}, "lower": {"learning_rate": .0001}}
    for invalid in ("ppo_layers.upper.minbatch=8", "ppo_layers.upper.minibatch=0",
                    "ppo_layers.upper.gamma=0.9", "ppo_layer.upper.minibatch=8"):
        with pytest.raises(ValueError):
            load_config(ROOT / "configs/smoke.json", overrides=[invalid])


@pytest.mark.parametrize("method", ["always_on", "upper_ppo_greedy", "hpgi", "joint_ppo"])
@pytest.mark.parametrize("validation", [False, True])
def test_console_counts_actual_components(config, method, validation):
    config = for_method(config, method)
    config["validation"]["enabled"] = validation
    schedule = describe_schedule(config)
    layers = [layer for layer in ("joint", "upper", "lower") if layer in schedule["ppo"]]
    stream = io.StringIO()
    console = ConsoleProgress(stream)
    console.record("training_started", {"schedule": schedule})
    for layer in layers:
        console.record("phase_started", {"stage": "cycle_000_" + layer})
    console.record("cycle_completed", dict(cycle_index=1, reward_per_slot=-2.,
        upper_loss=.1 if "upper" in layers else None, lower_loss=.2 if "lower" in layers else None,
        collection_seconds=1., update_seconds=2., phase_seconds=3.))
    output = stream.getvalue()
    for step in range(1, len(layers) + 1):
        assert f"Iteration 1 [{step}/{len(layers) + int(validation)}]" in output
    assert "None" not in output
    if method == "always_on":
        assert "服务器loss" not in output
    if method == "upper_ppo_greedy":
        assert "调度loss" not in output


def test_greedy_history_reuses_profiles_without_changing_actions(config, monkeypatch):
    environment = Environment(config, generate_trace(config, 91))
    while environment.slot < config["window"]:
        problem = environment.begin_slot([1] * len(config["servers"]))
        environment.execute(heuristics.greedy_fast(problem)[0])
    observation = environment.upper_observation()
    constructor = Mock(wraps=heuristics.Profiles)
    monkeypatch.setattr(heuristics, "Profiles", constructor)
    cached = heuristics.greedy_slow(observation)
    assert constructor.call_count == 1
    original = heuristics.PublicProblem
    uncached = Mock(side_effect=lambda *args, **kwargs: original(*args))
    monkeypatch.setattr(heuristics, "PublicProblem", uncached)
    assert heuristics.greedy_slow(observation) == cached
    assert uncached.call_count > 1


@pytest.mark.parametrize("envelope", [False, True])
def test_fixed_window_check_accepts_saved_inputs(config, tmp_path, envelope):
    config = baseline_config(config)
    filename, tracefile = tmp_path / "config.json", tmp_path / "trace.json"
    write_json(filename, config)
    trace = generate_trace(config, 91)
    write_json(tracefile, {"seed": 91, "trace": trace} if envelope else trace)
    run = main(["check", "--config", str(filename), "--method", "fixed_window",
                "--trace", str(tracefile), "--output-dir", str(tmp_path / "outputs")])
    assert read_json(run / "manifest.json")["status"] == "completed"
    assert read_json(run / "traces/input.json") == trace
    assert read_json(run / "manifest.json")["trace_input"]["sha256"] == digest_file(tracefile)
    assert not (run / "results").exists()


@pytest.mark.parametrize("payload", [{"seed": 91}, {"trace": {}}, [{}], {"trace": [], "seed": -1}])
def test_invalid_trace_envelope_fails(config, tmp_path, payload):
    filename, tracefile = tmp_path / "config.json", tmp_path / "trace.json"
    write_json(filename, baseline_config(config))
    write_json(tracefile, payload)
    with pytest.raises(ValueError, match="Trace"):
        main(["check", "--config", str(filename), "--trace", str(tracefile),
              "--output-dir", str(tmp_path / "outputs")])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
@pytest.mark.parametrize("method", LEARNING_BASELINES)
def test_learning_baseline_cli_roundtrip(config, tmp_path, method, device):
    config = baseline_config(config, device)
    filename, tracefile = tmp_path / "config.json", tmp_path / "trace.json"
    write_json(filename, config)
    write_json(tracefile, {"seed": 91, "trace": generate_trace(config, 91)})
    outputs = tmp_path / "outputs"
    run = main(["train", "--config", str(filename), "--method", method,
                "--output-dir", str(outputs), "--run-id", "train"])
    checkpoint = run / "checkpoints/last.pt"
    trainer = Trainer.load(checkpoint)
    resolved = trainer.config
    upper = method != "always_on"
    lower = method != "upper_ppo_greedy"
    assert trainer.versions == {"upper": int(upper), "lower": 2 * int(lower)}
    assert trainer.total_slots == describe_schedule(resolved)["planned_training_slots"]
    assert all(row["parameters_changed"] and row["frozen_unchanged"] for row in trainer.history)
    assert all(row["behavior_logprob_error"] < 1e-5 for row in trainer.history)
    assert all(row["device_metrics"]["update_device"] == device for row in trainer.history)
    audit = inspect_run(run)
    assert len(audit["updates"]) == int(upper) + 2 * int(lower)
    cycles = read_csv(run / "results/cycles.csv")
    phases = [row for row in trainer.history if row["stage"].startswith("cycle_")]
    expected_reward = sum(row["raw_reward_sum"] for row in phases) / sum(row["physical_slots"] for row in phases)
    assert len(cycles) == 1 and float(cycles[0]["reward_per_slot"]) == pytest.approx(expected_reward)
    evaluated = main(["evaluate", "--checkpoint", str(checkpoint), "--trace", str(tracefile),
                      "--output-dir", str(outputs), "--run-id", "evaluate"])
    metrics = read_json(evaluated / "summary.json")
    records_path = evaluated / "traces/evaluation.json"
    records = read_json(records_path)
    for key, value in independent_totals(resolved, records).items():
        assert_close(value, metrics[key], key)
    evaluation_rows = read_csv(evaluated / "results/evaluation.csv")
    assert len(evaluation_rows) == 1
    assert float(evaluation_rows[0]["conservative_cost"]) == pytest.approx(metrics["conservative_cost"])
    assert evaluation_rows[0]["trajectory_seed"] == "91"
    for command in ("replay", "summarize"):
        checked = main([command, "--checkpoint", str(checkpoint), "--records", str(records_path),
                        "--output-dir", str(outputs), "--run-id", command])
        assert_close(read_json(checked / "summary.json"), metrics)
    if method == "always_on":
        assert all(record["active"] == [1] * len(config["servers"]) for record in records)
    if method == "single_timescale":
        assert resolved["window"] == 1
    if method == "fixed_split":
        environment = Environment(resolved, read_json(tracefile)["trace"])
        choices = environment.begin_slot([1] * len(config["servers"])).choices
        assert all(choice["split"] == resolved["fixed_split"] for options in choices
                   for choice in options if choice["kind"] == "edge")
    if method in ("nominal", "full_information"):
        assert resolved["information"] == ("nominal" if method == "nominal" else "full")
    with pytest.raises(ValueError, match="relabel"):
        main(["evaluate", "--checkpoint", str(checkpoint), "--method", "hpgi"])
    with pytest.raises(ValueError, match="own checkpoint"):
        main(["evaluate", "--config", str(filename), "--method", method,
              "--output-dir", str(outputs), "--run-id", "untrained"])
    try:
        trainer.run(extra_cycles=1)
        expected_state = copy.deepcopy(trainer.controller.state_dict())
    finally:
        trainer.close()
    resumed = main(["train", "--checkpoint", str(checkpoint), "--extra-cycles", "1",
                    "--output-dir", str(outputs), "--run-id", "resume"])
    restored = Trainer.load(resumed / "checkpoints/last.pt")
    assert restored.versions == {"upper": 2 * int(upper), "lower": 3 * int(lower)}
    assert len(inspect_run(resumed)["updates"]) == int(upper) + int(lower)
    for layer in ("upper", "lower"):
        for name, value in restored.controller.state_dict()[layer].items():
            torch.testing.assert_close(value, expected_state[layer][name], rtol=2e-5, atol=2e-7)
    schedule = main(["train", "--checkpoint", str(resumed / "checkpoints/last.pt"),
                     "--extra-cycles", "1", "--dry-run"])
    assert schedule["cycles"] == 3
    assert schedule["remaining_training_slots"] == sum(row["physical_slots"] for row in phases)


def test_heuristic_cli_needs_no_training(config, tmp_path, monkeypatch):
    filename = tmp_path / "config.json"
    write_json(filename, baseline_config(config))
    monkeypatch.setattr(Trainer, "__init__", Mock(side_effect=AssertionError("Heuristic must not train")))
    outputs = tmp_path / "outputs"
    run = main(["evaluate", "--config", str(filename), "--method", "greedy_consolidation",
                "--output-dir", str(outputs), "--run-id", "greedy"])
    assert not (run / "checkpoints").exists()
    records_path = run / "traces/evaluation.json"
    metrics = read_json(run / "summary.json")
    for command in ("replay", "summarize"):
        checked = main([command, "--config", str(filename), "--method", "greedy_consolidation",
                        "--records", str(records_path), "--output-dir", str(outputs), "--run-id", command])
        assert_close(read_json(checked / "summary.json"), metrics)
    assert len(read_csv(run / "results/evaluation.csv")) == 1
    with pytest.raises(ValueError, match="uses evaluate"):
        main(["train", "--config", str(filename), "--method", "greedy_consolidation",
              "--output-dir", str(outputs), "--run-id", "must-not-exist"])
    assert not (outputs / "must-not-exist").exists()


def test_verification_uses_actual_minimal_checkpoint(config, tmp_path):
    from mist_sim.verification import verify_one
    config = for_method(baseline_config(config), "always_on")
    report, run = verify_one(ROOT, config, 91, tmp_path)
    assert report["reload_equal"] and report["independent_totals_equal"]
    assert (run / "checkpoints/last.pt").exists()
    assert not (run / "checkpoints/final.pt").exists()


@pytest.mark.parametrize("method,unused", [("always_on", "upper"), ("upper_ppo_greedy", "lower")])
def test_single_layer_calibration_respects_baseline_policy(config, method, unused, monkeypatch):
    config = for_method(baseline_config(config), method)
    config["training"]["calibration_episodes"] = 2
    trainer = Trainer(config, 91)
    policy = getattr(trainer.controller, unused)
    monkeypatch.setattr(policy, "upper_action" if unused == "upper" else "lower_action",
                        Mock(side_effect=AssertionError("Unused baseline network must not act")))
    observed = []
    original = trainer.reward_normalizer.observe

    def observe(records):
        observed.extend(records)
        original(records)

    monkeypatch.setattr(trainer.reward_normalizer, "observe", observe)
    trainer.calibrate()
    assert trainer.calibrated and len(observed) == 10
    if method == "always_on":
        assert all(record["active"] == [1] * len(config["servers"]) for record in observed)
    else:
        for record in observed:
            environment = Environment(dict(config, horizon=1, initial_active=record["previous_active"]), [record["raw"]])
            problem = environment.begin_slot(record["active"])
            assert heuristics.greedy_fast(problem)[0] == record["actions"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"))])
def test_fixed_window_projection_preserves_latent_probability_and_gradient(config, monkeypatch, device):
    from mist_sim.policies.batching import RolloutTensors, replay_batch
    from mist_sim.policies.hpgi import Controller
    from tests.test_physics import request

    config = for_method(config, "fixed_window")
    config.update(horizon=1, k=1, initial_active=[1, 1])
    config["access_points"][0]["bandwidth"] = config["reserved_bandwidth"]
    environment = Environment(config, [[request("first"), request("second")]])
    problem = environment.begin_slot([1, 1])
    edge = next(index for index, choice in enumerate(problem.choices[0]) if choice["kind"] == "edge")
    monkeypatch.setattr(torch.distributions.Categorical, "sample", lambda self: torch.tensor(edge))
    monkeypatch.setattr(torch.distributions.Normal, "sample", lambda self: torch.tensor(-10.))
    policy = Controller(config).lower
    actions, predicted, sample = policy.lower_action(problem)
    saved_sample = copy.deepcopy(sample)
    record = environment.execute(actions, predicted)
    assert record["actions"] == [edge, edge]
    assert record["executed_actions"][1] == 0
    assert {violation["reason"] for violation in record["projection_violations"]} == {"joint_capacity", "rate_capacity"}
    assert sample == saved_sample
    assert [step["rate_latent"] for step in sample["steps"]] == [-10., -10.]
    serial = policy.replay(sample)[0]
    assert serial.item() == pytest.approx(sample["logprob"], abs=1e-5)
    serial.backward()
    expected_gradients = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()
                          if parameter.grad is not None}
    batched_policy = copy.deepcopy(policy).to(device)
    batched_policy.zero_grad(set_to_none=True)
    cache = RolloutTensors(batched_policy, [sample])
    batched = replay_batch(batched_policy, [sample], cache=cache)[0].sum()
    torch.testing.assert_close(batched.cpu(), serial, rtol=2e-5, atol=2e-6)
    batched.backward()
    assert any(parameter.grad.abs().sum() > 0 for parameter in batched_policy.rate_head.parameters())
    for name, parameter in batched_policy.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad.cpu(), expected_gradients[name], rtol=5e-4, atol=2e-5)
