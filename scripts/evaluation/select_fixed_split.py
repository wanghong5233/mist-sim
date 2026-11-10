"""Select static DNN cuts on shared development inputs, then export a PPO overlay.

This is controller-based hyperparameter selection, not a certificate of the best
possible trained Fixed-Split policy. No HPGI scores or final test inputs are read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import torch
from mist_sim.config import default_project_root, for_method, load_config, validate
from mist_sim.diagnostics import check_physics
from mist_sim.evaluation import evaluate
from mist_sim.process_runtime import configure_process
from mist_sim.storage import RunStore, digest_file, write_json
from scripts.evaluation.evaluate_greedy import GreedyController, freeze_suite


def candidate_cuts(config):
    models = sorted(config["profiles"])
    return [dict(zip(models, values)) for values in itertools.product(
        *(sorted(config["profiles"][name]["legal_splits"]) for name in models))]


def selection_config(config, cuts):
    result = for_method(config, "greedy_consolidation")
    result.update(fixed_split=dict(cuts), dynamic_split=False, device="cpu")
    validate(result)
    return result


def evaluate_candidate(config, index, cuts, input_path, expected_hash):
    torch.set_num_threads(1)
    configure_process(config["runtime"]["windows_performance"])
    if digest_file(input_path) != expected_hash:
        raise ValueError("Development input checksum mismatch")
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    resolved = selection_config(config, cuts)
    started = time.perf_counter()
    records, metrics = evaluate(GreedyController(), resolved, payload["trace"])
    check_physics(resolved, records)
    # Check the executed cut, not just the configuration label.
    for record in records:
        models = {request["id"]: request["model"] for request in record["public"]}
        for task in record["tasks"]:
            if task["action"] == "edge" and task["split"] != cuts[models[task["id"]]]:
                raise AssertionError("Fixed cut was not used during execution")
    return dict(candidate=index, trajectory_seed=payload["seed"], **cuts,
                physical_slots=len(records), cost=metrics["conservative_cost"],
                cost_per_slot=metrics["conservative_cost"] / len(records),
                energy=metrics["total_energy"], arrivals=metrics["arrivals"],
                completed=metrics["completed"], service=metrics["on_time_rate"],
                rejected=metrics["rejected"], suppressed=metrics["suppressed"],
                timeout=metrics["timeout"], resource_failed=metrics["resource_failed"],
                wall_seconds=time.perf_counter() - started, physics_passed=True)


def rank_candidates(candidates, rows, seeds):
    ranked = []
    for index, cuts in enumerate(candidates):
        group = [row for row in rows if row["candidate"] == index]
        if sorted(row["trajectory_seed"] for row in group) != sorted(seeds):
            raise ValueError("Every candidate requires the same complete development inputs")
        slots = sum(row["physical_slots"] for row in group)
        arrivals = sum(row["arrivals"] for row in group)
        if slots <= 0 or any(not row["physics_passed"] for row in group):
            raise ValueError("Invalid candidate evaluation")
        if any(not math.isfinite(row[key]) for row in group for key in ("cost", "energy")):
            raise ValueError("Non-finite candidate evaluation")
        ranked.append(dict(candidate=index, **cuts, physical_slots=slots,
            cost_per_slot=sum(row["cost"] for row in group) / slots,
            energy_per_slot=sum(row["energy"] for row in group) / slots,
            service=sum(row["completed"] for row in group) / arrivals if arrivals else None))
    # The same scalar objective as PPO; no invented service threshold or weights.
    ranked.sort(key=lambda row: (row["cost_per_slot"], tuple(row[name] for name in sorted(candidates[0]))))
    return [dict(rank=index + 1, **row) for index, row in enumerate(ranked)]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_project_root())
    parser.add_argument("--config", type=Path, default=Path("configs/conditional.json"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=4300201)
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.trajectories < 1 or not 1 <= args.workers <= 32:
        parser.error("Positive trajectory count and 1..32 workers required")
    config = load_config(args.config, args.overlay)
    candidates = candidate_cuts(config)
    config = selection_config(config, candidates[0])
    seeds = list(range(args.first_seed, args.first_seed + args.trajectories))
    # Freeze common mixed-model workloads before any candidate scores are seen.
    suite = freeze_suite(args.root, args.suite, config, seeds)
    store = RunStore(args.root, config, "select fixed split " + " ".join(sys.argv[1:]),
                     seeds[0], args.run_id, output_dir=args.output_dir)
    try:
        store.manifest["selection"] = dict(
            purpose="fixed_split_development_only_not_final_test",
            controller="common causal greedy consolidation + capacity mask + P3",
            objective="minimum pooled conservative cost per physical slot; same as negative PPO reward",
            candidates=candidates, trajectory_seeds=seeds,
            suite_sha256=digest_file(args.suite / "manifest.json"),
            budget_physical_slots=len(candidates) * len(seeds) * config["horizon"],
            training_updates=0,
            scope="Best of all legal cut combinations under the fixed development controller; not optimal trained PPO")
        store.write_json("manifest.json", store.manifest)
        store.write_json("runtime.json", configure_process(config["runtime"]["windows_performance"]))

        def report(message):
            print(message, flush=True)
            with (store.path / "stdout.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

        total = len(candidates) * len(seeds)
        report(f"Fixed-Split selection | {len(candidates)} combinations x {len(seeds)} inputs x {config['horizon']} slots | {args.workers} CPU workers | no GPU")
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(evaluate_candidate, config, index, cuts, args.suite / item["file"], item["sha256"])
                       for index, cuts in enumerate(candidates) for item in suite["inputs"]]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                store.append_result("candidates.csv", row)
                report(f"Selection {len(rows)}/{total} | cuts=" + "/".join(str(row[name]) for name in sorted(candidates[0])) +
                       f" | seed={row['trajectory_seed']} | cost/slot={row['cost_per_slot']:.4f} | service={row['service']:.2%} | {row['wall_seconds']:.1f}s")
        ranked = rank_candidates(candidates, rows, seeds)
        for row in ranked:
            store.append_result("ranking.csv", row)
        winner = {name: ranked[0][name] for name in candidates[0]}
        write_json(store.result_path / "selected_overlay.json", {"fixed_split": winner})
        write_json(store.result_path / "selection.json", dict(
            store.manifest["selection"], selected=winner, ranking=ranked,
            config_sha256=store.manifest["config_sha256"],
            overlay_sha256=digest_file(store.result_path / "selected_overlay.json")))
        report("Selected " + json.dumps(winner, sort_keys=True) + "; freeze these cuts before independent Fixed-Split PPO training.")
        store.finish()
    except BaseException as error:
        store.finish(error)
        raise


if __name__ == "__main__":
    main()
