"""Paired training from one pretraining snapshot under equal interaction budgets."""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from .anonymity import generate_trace
from .config import for_method, load_config, default_project_root, validate
from .evaluation import evaluate, summarize
from .schedule import describe_schedule, rollout_slots
from .storage import RunStore, digest_file, validation_workspace
from .training import Trainer, CHECKPOINT_SCHEMA, implementation_signature


def initialize_common(trainer, payload):
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["implementation_signature"] != implementation_signature():
        raise ValueError("Common initialization must use the current implementation")
    if not payload["resumable"] or payload["completed_stages"][-1] != "pretrain":
        raise ValueError("Common initialization requires completed pretraining")
    original, current = copy.deepcopy(payload["config"]), copy.deepcopy(trainer.config)
    for config in (original, current):
        config.pop("method")
        config["training"].pop("cycles")
    if original != current:
        raise ValueError("Common initialization changed environment, inputs or learning settings")
    trainer.controller.load_state_dict(payload["controller"])
    trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
    trainer.value_reward_scales = copy.deepcopy(payload["value_reward_scales"])
    for phase in ("upper", "lower"):
        trainer.optimizers[phase].load_state_dict(copy.deepcopy(payload["optimizers"][phase]))
    for key in ("versions", "trace_counter", "completed_stages", "history", "total_slots", "calibrated",
                "training_episodes"):
        setattr(trainer, key, copy.deepcopy(payload[key]))
    trainer.epoch_counts.update(payload["epoch_counts"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])


def run_pair(config, seed, slots, root, output_dir=None, evaluation_trajectories=0):
    if slots < 1 or evaluation_trajectories < 0:
        raise ValueError("Positive interaction budget and nonnegative evaluation count required")
    config = copy.deepcopy(config)
    config["validation"]["enabled"] = False
    budgets = dict(hpgi=config["training"]["fast_slots"] + rollout_slots(config, "upper", config["training"]["slow_intervals"]),
                   joint_ppo=config["training"]["fast_slots"])
    if any(slots % batch for batch in budgets.values()):
        raise ValueError(f"Budget must be a multiple of both iteration sizes: {budgets}")
    configurations = {method: for_method(config, method) for method in budgets}
    for method, settings in configurations.items():
        settings["training"]["cycles"] = slots // budgets[method]
        validate(settings)
    first_store = RunStore(root, configurations["hpgi"], f"paired train slots={slots}", seed, output_dir=output_dir)
    first = Trainer(configurations["hpgi"], seed, first_store)
    common_started = time.perf_counter()
    try:
        first.event("training_started", schedule=describe_schedule(first.config))
        first.calibrate()
        first.event("calibration_completed", episodes=first.config["training"]["calibration_episodes"])
        total, chunk = first.config["training"]["pretrain_slots"], first.config["training"]["fast_slots"]
        for index, offset in enumerate(range(0, total, chunk)):
            count = min(chunk, total - offset)
            stage = "pretrain" if offset + count == total else f"pretrain_{index:03d}"
            first.phase(stage, "lower", count, pretrain=True)
        common_path = first_store.path / "checkpoints/common_pretrain.pt"
        first.save(common_path)
        payload = torch.load(common_path, map_location="cpu", weights_only=False)
    except BaseException as error:
        first.save(first_store.path / "checkpoints/failed_last.pt")
        first_store.finish(error)
        first.close()
        raise
    common_seconds = time.perf_counter() - common_started
    first.close()
    provenance = dict(checkpoint=str(common_path), sha256=digest_file(common_path),
                      physical_slots=first.total_slots, seconds=common_seconds)
    reports = []
    for method in ("hpgi", "joint_ppo"):
        if method == "hpgi":
            trainer, store = first, first_store
        else:
            store = RunStore(root, configurations[method], f"paired train slots={slots}", seed, output_dir=output_dir)
            trainer = Trainer(configurations[method], seed, store)
            initialize_common(trainer, payload)
        store.manifest["comparison"] = dict(common_initialization=provenance, requested_post_pretrain_slots=slots,
            methods=list(budgets), development_evaluation_trajectories=evaluation_trajectories,
            collector_start="cold")
        store.write_json("manifest.json", store.manifest)
        started = time.perf_counter()
        try:
            trainer.run()
            training_seconds = time.perf_counter() - started
            phases = [row for row in trainer.history if row["stage"].startswith("cycle_")]
            actual_slots = sum(row["physical_slots"] for row in phases)
            report = dict(method=method, run_id=store.path.name, seed=seed,
                results=str(store.result_path), common_checkpoint_sha256=provenance["sha256"],
                post_pretrain_slots=actual_slots, training_seconds=training_seconds,
                training_transitions=sum(row["samples"] for row in phases),
                sample_presentations=sum(row["samples"] * row["epochs_completed"] for row in phases),
                optimizer_steps=sum(row["optimizer_steps"] for row in phases),
                atomic_decisions=sum(row["rollout"]["decisions"] for row in phases),
                collection_seconds=sum(row["collection_seconds"] for row in phases),
                update_seconds=sum(row["update_seconds"] for row in phases),
                probability_check_seconds=sum(row["probability_check_seconds"] for row in phases),
                trace_write_seconds=sum(row["trace_write_seconds"] for row in phases),
                collector_start="cold",
                iterations=len(trainer.cycle_history), stop_reason=trainer.stop_reason,
                training_reward_per_slot=sum(row["raw_reward_sum"] for row in phases) / actual_slots,
                training_service=sum(row["completed"] for row in phases) / max(1, sum(row["arrivals"] for row in phases)),
                final_development_reward_per_slot=None, final_development_service=None)
            if evaluation_trajectories:
                records = []
                for index in range(evaluation_trajectories):
                    trace_seed = config["seeds"]["validation"] + 10000 * index
                    physical, metrics = evaluate(trainer.controller, trainer.config, generate_trace(config, trace_seed))
                    records.extend(physical)
                    store.append_result("comparison_trajectories.csv", dict(trajectory_seed=trace_seed,
                        physical_slots=len(physical), raw_reward_sum=sum(row["reward"] for row in physical),
                        **{key: value for key, value in metrics.items() if key != "energy"},
                        **{"energy_" + key: value for key, value in metrics["energy"].items()}))
                metrics = summarize(records)
                report.update(final_development_reward_per_slot=sum(row["reward"] for row in records) / len(records),
                              final_development_service=metrics["on_time_rate"])
            reports.append(report)
            store.append_result("comparison.csv", {key: value for key, value in report.items() if key not in ("run_id", "seed")})
            store.write_json("summary.json", report)
            store.finish()
            print(f"{method}: {report}", flush=True)
        except BaseException as error:
            trainer.save(store.path / "checkpoints/failed_last.pt")
            store.finish(error)
            raise
        finally:
            trainer.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conditional.json")
    parser.add_argument("--slots", type=int, required=True, help="Post-pretraining physical slots per method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--evaluation-trajectories", type=int, default=0, help="One final paired development check, never per iteration")
    arguments = parser.parse_args()
    config = load_config(arguments.config, overrides=arguments.overrides)
    seed = arguments.seed if arguments.seed is not None else config["seeds"]["train"]
    root = default_project_root()
    if config["fixture_kind"] == "synthetic_hand_calculation":
        with validation_workspace() as temporary:
            run_pair(config, seed, arguments.slots, root, temporary, arguments.evaluation_trajectories)
    else:
        run_pair(config, seed, arguments.slots, root, evaluation_trajectories=arguments.evaluation_trajectories)
