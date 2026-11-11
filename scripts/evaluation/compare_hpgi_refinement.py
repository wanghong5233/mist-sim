"""Paired, frozen-weight comparison of the public HPGI refinement mechanisms."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.config import validate
from mist_sim.evaluation import evaluate
from mist_sim.policies.hpgi import Controller
from mist_sim.process_runtime import configure_process
from mist_sim.storage import digest_file, write_json
from mist_sim.training import implementation_signature, signature
from mist_sim.transfer import read_parent


def run_one(job):
    label, config, weights, seed = job
    torch.set_num_threads(1)
    configure_process(config["runtime"].get("windows_performance", False))
    controller = Controller(config)
    controller.load_state_dict(weights)
    trace = generate_trace(config, seed)
    started = time.perf_counter()
    records, metrics = evaluate(controller, config, trace)
    metrics.pop("energy")
    policy_times = [r["measured_policy_seconds"] for r in records]
    metrics.update(label=label, seed=seed, slots=len(records),
        cost_per_slot=metrics["conservative_cost"] / len(records),
        energy_per_slot=metrics["total_energy"] / len(records),
        elapsed_seconds=time.perf_counter() - started,
        policy_seconds_mean=float(np.mean(policy_times)), policy_seconds_p95=float(np.quantile(policy_times, .95)),
        input_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Independent trajectory seeds must not be duplicated")
    parent, receipt = read_parent(args.parent, args.parent_revision)
    if parent["config"]["method"] != "fixed_split":
        parser.error("This comparison requires a mature Fixed-Split parent")
    policies = [("fixed_split", copy.deepcopy(parent["config"]))]
    for scope in ("splits", "assignment"):
        config = copy.deepcopy(parent["config"])
        config.update(method="hpgi", dynamic_split=True, split_solver="fixed_proposal",
                      split_refinement_scope=scope, split_refinement_passes=1)
        validate(config)
        policies.append((scope, config))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", dict(parent=receipt, seeds=args.seeds,
        configs={label: config for label, config in policies}, config_hashes={label: signature(c) for label, c in policies},
        implementation_signature=implementation_signature(), script_sha256=digest_file(Path(__file__)),
        purpose="development frozen-parent comparison, not independent from-scratch training",
        measured_latency_in_reward=False, detailed_trajectories=False, workers=4))
    rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(run_one, (label, config, parent["controller"], seed))
                for seed in args.seeds for label, config in policies]
        for job in as_completed(jobs):
            row = job.result()
            rows.append(row)
            with (args.output / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (r["seed"], r["label"])))
            print(json.dumps({k: row[k] for k in ("label", "seed", "cost_per_slot", "on_time_rate", "elapsed_seconds")}), flush=True)
    for seed in args.seeds:
        assert len({r["input_sha256"] for r in rows if r["seed"] == seed}) == 1
    assert digest_file(args.parent) == receipt["sha256"]
    write_json(args.output / "audit.json", dict(paired_input_hashes=True, parent_unchanged=True,
        rows=len(rows), result_sha256=digest_file(args.output / "evaluations.csv"),
        implementation_signature=implementation_signature()))
    for label, _ in policies:
        selected = [row for row in rows if row["label"] == label]
        print(json.dumps(dict(label=label,
            cost_per_slot=sum(r["conservative_cost"] for r in selected) / sum(r["slots"] for r in selected),
            on_time_rate=sum(r["completed"] for r in selected) / sum(r["arrivals"] for r in selected))), flush=True)


if __name__ == "__main__":
    main()
