"""Freeze shared request inputs and evaluate the causal, nonlearning greedy baseline."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import time

import torch
from mist_sim.anonymity import generate_trace
from mist_sim.config import default_project_root, for_method, load_config
from mist_sim.diagnostics import check_physics
from mist_sim.evaluation import evaluate, summarize
from mist_sim.policies.baselines.heuristics import greedy_fast, greedy_slow
from mist_sim.process_runtime import configure_process
from mist_sim.storage import RunStore, digest_file, write_json


class GreedyController:
    """Adapter to the common evaluator; no network or optimizer is constructed."""
    def slow(self, observation, **kwargs):
        return greedy_slow(observation), None

    def fast(self, problem, **kwargs):
        return greedy_fast(problem)


def scenario(config):
    # Physical inputs and request generation, independent of learning settings.
    names = ("slot_seconds", "horizon", "terminals", "k", "arrival_probability",
             "position_range", "deadline_range", "fading_range", "generalization_divisions",
             "local_rate", "local_kappa", "tx_power", "noise_density", "interference",
             "gain_reference", "reserved_bandwidth", "anonymization_seconds", "control_seconds",
             "backhaul_seconds", "lambda_time", "rejection_cost", "initial_active",
             "servers", "access_points", "profiles", "units")
    return {name: config[name] for name in names}


def freeze_suite(root, directory, config, seeds):
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Trajectory seeds must be unique nonnegative integers")
    used = set()
    for ledger in (root / "logs/runs").glob("*/input_seeds.jsonl"):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            used.add(json.loads(line)["seed"])
    if used.intersection(seeds):
        raise ValueError("Comparison inputs overlap recorded training inputs")
    physical = scenario(config)
    signature = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["scenario_sha256"] != signature or [row["seed"] for row in manifest["inputs"]] != seeds:
            raise ValueError("Frozen suite scenario/seeds cannot change")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        inputs = []
        for seed in seeds:
            path = directory / f"seed{seed}.json"
            write_json(path, {"seed": seed, "trace": generate_trace(config, seed)})
            inputs.append({"seed": seed, "file": path.name, "sha256": digest_file(path)})
        manifest = {"purpose": "shared_comparison_inputs_not_final_blind_test",
                    "scenario_sha256": signature, "scenario": physical, "inputs": inputs}
        write_json(manifest_path, manifest)
    for row in manifest["inputs"]:
        if digest_file(directory / row["file"]) != row["sha256"]:
            raise ValueError("Frozen input checksum mismatch")
    return manifest


def evaluate_one(config, path, expected_hash, records_path):
    torch.set_num_threads(1)
    runtime = configure_process(config["runtime"]["windows_performance"])
    if digest_file(path) != expected_hash:
        raise ValueError("Evaluation input checksum mismatch")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    started = time.perf_counter()
    records, metrics = evaluate(GreedyController(), config, payload["trace"])
    write_json(records_path, records)
    # Checks recorded arithmetic and capacities; does not rerun the controller.
    physics = check_physics(config, records)
    return {"seed": payload["seed"], "metrics": metrics, "physics": physics,
            "wall_seconds": time.perf_counter() - started, "records_sha256": digest_file(records_path), "runtime": runtime}


def aggregate(records, metrics, slots):
    pooled = summarize(records)  # In particular, pooled P95 is not an average of P95s.
    names = ("conservative_cost", "on_time_rate", "total_energy", "completion_delay_mean")
    distribution = {}
    for name in names:
        values = [row[name] / slots if name in ("conservative_cost", "total_energy") else row[name]
                  for row in metrics if row[name] is not None]
        distribution[name] = {"mean": statistics.mean(values) if values else None,
                              "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    return pooled, distribution


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_project_root())
    parser.add_argument("--config", type=Path, default=Path("configs/conditional.json"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=4400101)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.trajectories < 1 or not 1 <= args.workers <= 32:
        parser.error("Positive trajectory count and 1..32 workers required")
    config = for_method(load_config(args.config, args.overlay), "greedy_consolidation")
    seeds = list(range(args.first_seed, args.first_seed + args.trajectories))
    suite = freeze_suite(args.root, args.suite, config, seeds)
    store = RunStore(args.root, config, "evaluate greedy shared-suite " + " ".join(sys.argv[1:]),
                     seeds[0], args.run_id, output_dir=args.output_dir)
    try:
        store.write_json("runtime.json", configure_process(config["runtime"]["windows_performance"]))
        store.manifest["evaluation_suite"] = {"path": str(args.suite.resolve()),
            "sha256": digest_file(args.suite / "manifest.json"), "inputs": suite["inputs"]}
        store.manifest["budget"] = {"trajectories": len(seeds), "physical_slots": len(seeds) * config["horizon"],
                                    "workers": args.workers, "training_updates": 0}
        store.write_json("manifest.json", store.manifest)
        print(f"Greedy | {len(seeds)} trajectories x {config['horizon']} slots | {args.workers} CPU workers | no training", flush=True)
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_one, config, args.suite / row["file"], row["sha256"],
                                  store.path / "traces" / "evaluation" / row["file"]): row for row in suite["inputs"]}
            for future in as_completed(futures):
                result = future.result()
                seed, metrics = result["seed"], result["metrics"]
                results[seed] = result
                store.record_evaluation(metrics, trajectory_seed=seed, physical_slots=config["horizon"],
                    cost_per_slot=metrics["conservative_cost"] / config["horizon"], wall_seconds=result["wall_seconds"])
                store.write_json(f"summaries/seed{seed}.json", result)
                message = (f"Greedy {len(results)}/{len(seeds)} | seed={seed} | cost/slot="
                           f"{metrics['conservative_cost'] / config['horizon']:.4f} | on-time={metrics['on_time_rate']:.2%}"
                           f" | {result['wall_seconds']:.1f}s")
                print(message, flush=True)
                with (store.path / "stdout.log").open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
        records = []
        for seed in seeds:
            records.extend(json.loads((store.path / "traces/evaluation" / f"seed{seed}.json").read_text(encoding="utf-8")))
        metrics = [results[seed]["metrics"] for seed in seeds]
        pooled, distribution = aggregate(records, metrics, config["horizon"])
        store.write_json("summary.json", {"pooled": pooled, "across_trajectories": distribution,
                                          "physical_checks": [results[seed]["physics"] for seed in seeds]})
        store.record_evaluation(pooled, trajectory_seed="pooled", physical_slots=len(records),
            cost_per_slot=pooled["conservative_cost"] / len(records), wall_seconds=time.perf_counter() - store.started)
        for name, values in distribution.items():
            store.append_result("aggregate.csv", {"metric": name, "trajectories": len(seeds), **values})
        store.finish()
        print(f"Completed | cost/slot={pooled['conservative_cost'] / len(records):.4f} | on-time={pooled['on_time_rate']:.2%}", flush=True)
        print(store.result_path, flush=True)
    except BaseException as error:
        store.finish(error)
        raise


if __name__ == "__main__":
    main()
