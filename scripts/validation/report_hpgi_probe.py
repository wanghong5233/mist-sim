"""Independently check a bounded HPGI development run and estimate training cost."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from mist_sim.config import load_config
from mist_sim.diagnostics import inspect_run
from mist_sim.evaluation import replay, summarize, evaluate, assert_close
from mist_sim.anonymity import generate_trace
from mist_sim.storage import digest_file, write_json, phase_trace_path, validation_trace_path
from mist_sim.training import Trainer
from mist_sim.ppo import loss_terms


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initial_gradient_balance(run):
    trainer = Trainer.load(run / "checkpoints" / "initial.pt")
    results = {}
    for stage, layer in (("pretrain", "lower"), ("cycle_000_upper", "upper")):
        update_path = phase_trace_path(run, stage, "update")
        if not update_path.exists():
            continue
        audit = read_json(update_path)
        samples = read_json(phase_trace_path(run, stage, "behavior"))
        indices = audit["minibatches"][0]["indices"]
        policy = getattr(trainer.controller, layer)
        replayed = [policy.replay(samples[index]) for index in indices]
        loss, *_ = loss_terms(torch.stack([item[0] for item in replayed]),
            torch.tensor([samples[index]["logprob"] for index in indices]),
            torch.tensor([audit["normalized_advantages"][index] for index in indices], dtype=torch.float32),
            torch.stack([item[2] for item in replayed]),
            torch.tensor([audit["returns"][index] for index in indices], dtype=torch.float32),
            torch.stack([item[1] for item in replayed]),
            torch.tensor([bool(samples[index]["steps"]) for index in indices]), audit["settings"])
        policy.zero_grad()
        loss.backward()
        squared = {"actor": 0., "critic": 0.}
        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:
                group = "critic" if name.startswith(("critic_", "value_head")) else "actor"
                squared[group] += float(parameter.grad.square().sum())
        norm = sum(squared.values()) ** .5
        factor = min(1., audit["settings"]["max_grad_norm"] / (norm + 1e-6))
        results[layer] = dict(actor_norm=squared["actor"] ** .5, critic_norm=squared["critic"] ** .5,
                              common_clip_factor=factor, actor_norm_after_clip=squared["actor"] ** .5 * factor,
                              first_minibatch_loss=float(loss.detach()),
                              recorded_loss=audit["minibatches"][0]["total_loss"])
        assert_close(results[layer]["first_minibatch_loss"], results[layer]["recorded_loss"])
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/hpgi_probe_summary.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = read_json(args.run / "manifest.json")
    config = read_json(args.run / "resolved_config.json")
    checked = 0
    for name, expected in manifest["artifacts"].items():
        if digest_file(args.run / name) != expected:
            raise AssertionError(f"Artifact changed: {name}")
        checked += 1
    audit = inspect_run(args.run)
    gradient_balance = initial_gradient_balance(args.run)
    history = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in history:
        records = read_json(phase_trace_path(args.run, row["stage"], "physical"))
        for key, value in summarize(records).items():
            assert_close(value, row[key], key)
    validation = read_json(args.run / "validation_history.json")
    for row in validation:
        records = read_json(validation_trace_path(args.run, row["stage"]))
        assert_close(replay(config, records), row["metrics"])
    checkpoint = args.run / "checkpoints" / ("final.pt" if manifest["status"] == "completed" else "last.pt")
    trainer = Trainer.load(checkpoint)
    reload_equal = None
    if manifest["status"] == "completed":
        original = read_json(validation_trace_path(args.run, "final"))[:config["horizon"]]
        restored, _ = evaluate(trainer.controller, config, generate_trace(config, config["seeds"]["validation"]))
        for expected, actual in zip(original, restored):
            for key in ("active", "actions", "rates", "energy", "cost"):
                assert_close(expected[key], actual[key], key)
        reload_equal = True
    events = [json.loads(line) for line in (args.run / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    phase_seconds = {}
    for row in history:
        begin = next(event for event in events if event["event"] == "phase_started" and event["stage"] == row["stage"])
        end = next(event for event in events if event["event"] == "phase_completed" and event["stage"] == row["stage"])
        phase_seconds[row["stage"]] = end["elapsed_seconds"] - begin["elapsed_seconds"]
    estimate = None
    if {"pretrain", "cycle_000_upper", "cycle_000_lower"} <= phase_seconds.keys():
        target = load_config(Path("configs/conditional.json"))
        training = target["training"]
        validation_seconds = sum(row["measured_seconds"] for row in validation) / len(validation)
        estimate = (phase_seconds["pretrain"] * training["pretrain_slots"] / config["training"]["pretrain_slots"]
                    + training["cycles"] * (
                        phase_seconds["cycle_000_upper"] * training["slow_intervals"] / config["training"]["slow_intervals"]
                        + phase_seconds["cycle_000_lower"] * training["fast_slots"] / config["training"]["fast_slots"])
                    + (training["cycles"] + 3) * validation_seconds)
    report = dict(run_path=str(args.run), source=manifest["source"], status=manifest["status"],
                  measured_run_seconds=manifest["measured_wall_seconds"], artifact_hashes_checked=checked,
                  update_and_physics_audit=audit, updates=history, validation=validation,
                  initial_actor_critic_gradients=gradient_balance,
                  checkpoint=str(checkpoint), checkpoint_sha256=digest_file(checkpoint),
                  reloaded_actions_equal=reload_equal, total_training_slots=trainer.total_slots,
                  calibration_slots=config["training"]["calibration_episodes"] * config["horizon"],
                  selection=read_json(args.run / "selection.json"),
                  estimated_65536_training_seconds=estimate,
                  estimate_scope="Linear extrapolation; not measured full training, excludes calibration and post-run reproduction; large rollouts may scale nonlinearly",
                  formal_performance_evidence=False, measured_verification_seconds=time.perf_counter() - started)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "measured_run_seconds", "total_training_slots",
                                                  "reloaded_actions_equal", "estimated_65536_training_seconds",
                                                  "measured_verification_seconds")}, indent=2))


if __name__ == "__main__":
    main()
