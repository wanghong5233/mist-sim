"""One bounded transferred HPGI trial, paired scalar development evaluations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import torch

from mist_sim.anonymity import generate_trace
from mist_sim.cli import main as cli
from mist_sim.config import load_config
from mist_sim.evaluation import evaluate
from mist_sim.policies.hpgi import Controller
from mist_sim.process_runtime import configure_process
from mist_sim.storage import digest_file, write_json
from mist_sim.training import Trainer, signature
from mist_sim.transfer import read_parent, initialize_transfer


def compare_one(job):
    config, weights, seed, label, deterministic, temperatures = job
    torch.set_num_threads(1)
    configure_process(config["runtime"].get("windows_performance", False))
    controller = Controller(config)
    controller.load_state_dict(weights)
    for layer, temperature in (temperatures or {}).items():
        getattr(controller, layer).sampling_temperature = temperature
    action_seed = seed + 4600000
    torch.manual_seed(action_seed)
    trace = generate_trace(config, seed)
    started = time.perf_counter()
    records, metrics = evaluate(controller, config, trace, deterministic=deterministic)
    metrics.pop("energy")
    metrics.update(label=label, seed=seed, slots=len(records), cost_per_slot=metrics["conservative_cost"] / len(records),
        mode="deterministic" if deterministic else "sampled", action_seed=action_seed,
        reward_per_slot=-metrics["conservative_cost"] / len(records),
        rejection_cost_per_slot=sum(record["rejection_penalty"] for record in records) / len(records),
        conservative_energy_per_slot=sum(record["conservative_energy"] for record in records) / len(records),
        weighted_delay_cost_per_slot=config["lambda_time"] * sum(record["weighted_conservative_delay"] for record in records) / len(records),
        actual_energy_per_slot=metrics["total_energy"] / len(records),
        upper_temperature=controller.upper.sampling_temperature, lower_temperature=controller.lower.sampling_temperature,
        elapsed_seconds=time.perf_counter() - started,
        input_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest())
    return metrics


def compare(policies, seeds, both_modes=False, temperatures=None):
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(compare_one, [(config, weights, seed, label, deterministic, temperatures)
            for label, config, weights in policies for seed in seeds for deterministic in ((True, False) if both_modes else (True,))]))


def configured_trial(args, parent, receipt, logs):
    """Bounded exploration trial; public train entry point and scalar evaluation only."""
    root = Path.cwd()
    config = load_config(args.config)
    if config["method"] != "hpgi" or config.get("split_solver", "policy") != "policy":
        raise ValueError("Configured trial requires original HPGI without search")
    if config["training"]["cycles"] != args.cycles or args.cycles > 10:
        raise ValueError("Configured development trial must have the requested 1-10 cycles")
    if config["diagnostics"]["phase_interval"] or config["validation"]["enabled"]:
        raise ValueError("Use scalar before/after evaluation, not detailed trajectory exports")
    trainer = initialize_transfer(args.parent, config, args.seed, source_revision=args.parent_revision)
    write_json(logs / "config.json", config)
    output = root / "results/comparisons" / args.run_id
    if output.exists():
        raise FileExistsError(output)
    rows = compare([("initial", config, trainer.controller.state_dict())], args.eval_seeds, both_modes=True,
                   temperatures=trainer.controller.sampling_temperatures())
    output.mkdir(parents=True, exist_ok=False)

    def save_rows():
        with (output / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_rows()
    write_json(output / "provenance.json", dict(parent=receipt, config_signature=signature(config),
        seeds=args.eval_seeds, action_seed_offset=4600000, training_seed=args.seed, training_cycles=args.cycles,
        purpose="development_transfer_not_from_scratch", trajectories_exported=False,
        script_sha256=digest_file(Path(__file__)), baseline_sampling="unit_temperature",
        stopping="Stop at configured cycles; no automatic long training"))
    print(json.dumps([{k: row[k] for k in ("label", "mode", "seed", "cost_per_slot", "on_time_rate")}
                      for row in rows]), flush=True)
    cli(["train", "--config", str(logs / "config.json"), "--transfer-from", str(args.parent),
         "--parent-source-revision", args.parent_revision, "--seed", str(args.seed), "--run-id", args.run_id])
    trained = Trainer.load(root / "logs/runs" / args.run_id / "checkpoints/last.pt")
    if len(trained.cycle_history) != args.cycles:
        raise RuntimeError("Development stopped before the configured full iterations")
    rows.extend(compare([("initial_final_temperature", config, parent["controller"]),
                         ("trained", config, trained.controller.state_dict())], args.eval_seeds, both_modes=True,
                        temperatures=trained.controller.sampling_temperatures()))
    save_rows()
    if digest_file(args.parent) != receipt["sha256"]:
        raise RuntimeError("Immutable parent changed")
    print(json.dumps([{k: row[k] for k in ("label", "mode", "seed", "cost_per_slot", "on_time_rate")}
                      for row in rows]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--config", type=Path, help="Explicit original-HPGI exploration trial; up to 10 cycles")
    parser.add_argument("--passes", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    logs = root / "logs/launchers" / args.run_id
    logs.mkdir(parents=True, exist_ok=False)
    parent, receipt = read_parent(args.parent, args.parent_revision)
    if args.config:
        return configured_trial(args, parent, receipt, logs)
    if args.cycles > 5:
        raise ValueError("Historical refinement development remains limited to 5 cycles")
    config = copy.deepcopy(parent["config"])
    config.update(method="hpgi", dynamic_split=True, split_solver="fixed_proposal", split_refinement_passes=args.passes,
                  device="cuda")
    config["training"]["cycles"] = args.cycles
    config["ppo"].update(learning_rate=3e-5, entropy_coefficient=.001)
    config["runtime"].update(collection_workers=4, torch_threads=1, windows_performance=True)
    config["diagnostics"]["phase_interval"] = 0
    config["checkpoint_policy"].update(mode="minimal", stage_interval=0, cycle_interval=10)
    config["validation"]["enabled"] = False
    config["monitoring"] = dict(console=True, tensorboard=False)
    config["stopping"] = dict(max_slots=None, max_seconds=None)
    config["convergence"]["min_cycles"] = 300
    filename = logs / "config.json"
    write_json(filename, config)
    trainer = initialize_transfer(args.parent, config, args.seed, source_revision=args.parent_revision)
    trainer.save(logs / "initial.pt")
    reproduction = dict(config, split_refinement_passes=0)
    rows = compare([( "parent_fixed", parent["config"], parent["controller"]),
                    ("restricted_reproduction", reproduction, parent["controller"]),
                    ("hybrid_initial", config, trainer.controller.state_dict())], args.eval_seeds)
    for seed in args.eval_seeds:
        left, right = [next(r for r in rows if r["label"] == label and r["seed"] == seed)
                       for label in ("parent_fixed", "restricted_reproduction")]
        if left["input_sha256"] != right["input_sha256"] or abs(left["conservative_cost"] - right["conservative_cost"]) > 1e-7:
            raise AssertionError("Restricted policy failed to reproduce mature Fixed-Split")
    print(json.dumps([{k: row[k] for k in ("label", "seed", "cost_per_slot", "on_time_rate", "elapsed_seconds")}
                      for row in rows]), flush=True)
    output = root / "results/comparisons" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    def save_rows():
        with (output / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_rows()
    write_json(output / "provenance.json", dict(parent=receipt, config_signature=signature(config),
        seeds=args.eval_seeds, purpose="development_only_not_independent_from_scratch", trajectories_exported=False,
        script_sha256=digest_file(Path(__file__)), training_cycles=args.cycles))
    cli(["train", "--config", str(filename), "--transfer-from", str(args.parent), "--parent-source-revision", args.parent_revision,
         "--seed", str(args.seed), "--run-id", args.run_id])
    trained = Trainer.load(root / "logs/runs" / args.run_id / "checkpoints/last.pt")
    rows.extend(compare([(f"hybrid_{args.cycles}", config, trained.controller.state_dict())], args.eval_seeds))
    save_rows()
    print(json.dumps([{k: row[k] for k in ("label", "seed", "cost_per_slot", "on_time_rate", "elapsed_seconds")}
                      for row in rows[-len(args.eval_seeds):]]), flush=True)


if __name__ == "__main__":
    main()
