"""Evaluate immutable final checkpoints on shared, independently seeded requests."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
PHYSICS = ("environment.py", "anonymity.py", "allocation.py", "profiles.py", "evaluation.py")
LOADS = {"low": .25, "middle": .5, "high": .75}
SEEDS = {"middle": list(range(5400101, 5400121)),
         "low": list(range(5400201, 5400221)), "high": list(range(5400301, 5400321))}
DEVELOPMENT_SEEDS = [5400401, 5400402]
WORKER = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_tree(revision):
    archive = subprocess.check_output(["git", "archive", revision, "src/mist_sim"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        return {member.name: stream.extractfile(member).read()
                for member in stream.getmembers() if member.isfile() and member.name.endswith(".py")}


def source_signature(files):
    result = hashlib.sha256()
    for name, content in sorted(files.items()):
        result.update(name.removeprefix("src/mist_sim/").encode())
        result.update(content.replace(b"\r\n", b"\n"))
    return result.hexdigest()


def extract_source(files, directory):
    directory = Path(directory).resolve()
    for name, content in files.items():
        destination = (directory / name).resolve()
        if not destination.is_relative_to(directory / "src/mist_sim"):
            raise ValueError("Source archive escapes package")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def check_unused_seeds(candidates):
    collisions = []
    checked = 0
    roots = [ROOT / "logs/runs", ROOT / "results", ROOT / "data/processed/evaluation"]
    expression = re.compile(r"(?<!\d)(54\d{5})(?!\d)")
    for directory in roots:
        for path in directory.rglob("*"):
            if not path.is_file() or not (path.suffix == ".csv" or path.name in
                    ("manifest.json", "input_seeds.jsonl", "resolved_config.json")):
                continue
            checked += 1
            with path.open(encoding="utf-8-sig") as stream:
                overlap = {int(match) for line in stream for match in expression.findall(line)} & candidates
            if overlap:
                collisions.append({"file": str(path), "seeds": sorted(overlap)})
    if collisions:
        raise ValueError(f"Evaluation seed reuse: {collisions}")
    return {"checked_files": checked, "overlap": [], "candidate_seeds": sorted(candidates)}


def select_methods():
    import torch
    from scripts.evaluation.evaluate_greedy import scenario
    methods = {}
    sources = {}
    provenance_hashes = {}
    reference = None
    reference_physics = None
    for figure in ("learning-baselines", "information-baselines"):
        provenance = ROOT / f"results/comparisons/{figure}/figures/convergence/provenance.json"
        provenance_hashes[str(provenance)] = digest(provenance)
        for row in read_json(provenance)["sources"]:
            if row["completed_iterations"] != 300:
                raise ValueError("Only the confirmed complete 300-iteration curves are allowed")
            source_path = Path(row["path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if digest(source_path) != row["sha256"]:
                raise ValueError("Confirmed figure source CSV changed")
            run_id = (row["final_segment_manifest"]["run_id"] if "final_segment_manifest" in row
                      else source_path.parent.name)
            run = ROOT / "logs/runs" / run_id
            checkpoint = run / "checkpoints/last.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = payload["config"]
            method = config["method"]
            if method in methods:
                if methods[method]["checkpoint_sha256"] != digest(checkpoint):
                    raise ValueError("Conflicting final checkpoint")
                continue
            manifest = read_json(run / "manifest.json")
            if manifest["status"] != "completed" or manifest["exit_code"] != 0:
                raise ValueError("Training did not finish successfully")
            if config != read_json(run / "resolved_config.json"):
                raise ValueError("Resolved training configuration differs from checkpoint")
            revision = manifest["source"]["commit"]
            if revision not in sources:
                sources[revision] = source_tree(revision)
            files = sources[revision]
            signature = source_signature(files)
            if signature != payload["implementation_signature"]:
                raise ValueError("Git source does not reproduce checkpoint implementation")
            physical = scenario(config)
            physical_code = {name: hashlib.sha256(files["src/mist_sim/" + name].replace(b"\r\n", b"\n")).hexdigest()
                             for name in PHYSICS}
            if reference is None:
                reference, reference_physics = physical, physical_code
            if reference != physical or reference_physics != physical_code:
                raise ValueError("Cross-method physical scenario or implementation differs")
            methods[method] = {"run_id": run_id, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint), "config": config,
                "config_sha256": object_digest(config), "revision": revision,
                "implementation_signature": signature, "total_training_slots": payload["total_slots"],
                "sampling_temperatures": payload.get("sampling_temperatures", {}),
                "manifest_sha256": digest(run / "manifest.json")}
    if set(methods) != {"hpgi", "fixed_split", "single_timescale", "always_on", "fixed_window",
                       "upper_ppo_greedy", "nominal", "full_information"}:
        raise ValueError("The two confirmed figures must identify exactly eight methods")
    from mist_sim.config import for_method
    methods["greedy_consolidation"] = {**methods["hpgi"], "run_id": None, "checkpoint": None,
        "checkpoint_sha256": None, "total_training_slots": 0, "sampling_temperatures": {},
        "config": for_method(copy.deepcopy(methods["hpgi"]["config"]), "greedy_consolidation")}
    methods["greedy_consolidation"]["config_sha256"] = object_digest(methods["greedy_consolidation"]["config"])
    return methods, provenance_hashes, reference_physics


def load_probe(config):
    from mist_sim.anonymity import generate_trace, publish
    rows = []
    for label, probability in LOADS.items():
        settings = {**config, "arrival_probability": probability}
        arrivals = published = 0
        models = Counter()
        for seed in DEVELOPMENT_SEEDS:
            for batch in generate_trace(settings, seed):
                arrivals += len(batch)
                published += len(publish(batch, settings))
                models.update(item["model"] for item in batch)
        slots = len(DEVELOPMENT_SEEDS) * config["horizon"]
        rows.append({"load": label, "arrival_probability": probability, "slots": slots,
                     "arrivals_per_slot": arrivals / slots, "published_per_slot": published / slots,
                     "suppression_rate": 1 - published / arrivals, "model_arrivals": dict(models)})
    return rows


def prepare(directory):
    from mist_sim.anonymity import generate_trace
    if directory.exists():
        raise FileExistsError("Frozen evaluation specification already exists; do not regenerate inputs")
    candidates = set(DEVELOPMENT_SEEDS).union(*SEEDS.values())
    seed_audit = check_unused_seeds(candidates)
    methods, provenance, physics = select_methods()
    base = methods["hpgi"]["config"]
    probe = load_probe(base)
    suite = ROOT / "data/processed/evaluation" / directory.name
    if suite.exists():
        raise FileExistsError(suite)
    inputs = {}
    for label, probability in LOADS.items():
        inputs[label] = []
        config = {**base, "arrival_probability": probability}
        for seed in SEEDS[label] + (DEVELOPMENT_SEEDS if label == "middle" else DEVELOPMENT_SEEDS[:1]):
            path = suite / label / f"seed{seed}.json"
            write_json(path, {"seed": seed, "trace": generate_trace(config, seed)})
            inputs[label].append({"seed": seed, "path": str(path), "sha256": digest(path)})
    tasks = []
    for method in methods:
        modes = [True] if method == "greedy_consolidation" else [True, False]
        for deterministic in modes:
            for entry in inputs["middle"]:
                tasks.append({"batch": "preflight" if entry["seed"] in DEVELOPMENT_SEEDS else "A",
                    "method": method, "load": "middle", "probability": .5,
                    "deterministic": deterministic, "input": entry})
        if method not in ("nominal", "full_information"):
            for label in ("low", "high"):
                for entry in inputs[label]:
                    tasks.append({"batch": "preflight" if entry["seed"] in DEVELOPMENT_SEEDS else "B",
                        "method": method, "load": label,
                        "probability": LOADS[label], "deterministic": True, "input": entry})
    for task in tasks:
        task["action_seed"] = task["input"]["seed"] + 10000000
        task["id"] = f"{task['load']}-{task['method']}-{'det' if task['deterministic'] else 'sample'}-{task['input']['seed']}"
        task["audit"] = (task["batch"] == "A" and task["method"] in ("hpgi", "fixed_window")
                         and task["deterministic"] and task["input"]["seed"] == SEEDS["middle"][0])
    specification = {"schema": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "frozen_final_policy_independent_test_and_load_transfer", "training_updates": 0,
        "methods": methods, "figure_provenance": provenance, "shared_physics_sha256": physics,
        "seed_audit": seed_audit, "load_probe": probe, "tasks": tasks,
        "workers_per_terminal": 4, "max_concurrent_terminals": 2,
        "budget_seconds": {"preflight": 600, "A": 7200, "B": 7200},
        "primary_decode": "deterministic", "bootstrap_replicates": 5000, "bootstrap_seed": 0,
        "interpretation": "Single training seed; uncertainty across paired request trajectories only. Low load also changes anonymity suppression."}
    write_json(directory / "spec.json", specification)
    print(json.dumps({"spec": str(directory / "spec.json"), "load_probe": probe,
                      "task_counts": dict(Counter(task["batch"] for task in tasks))}), flush=True)


def evaluation_config(config, probability):
    from mist_sim.config import validate
    if not 0 < probability <= 1:
        raise ValueError("Arrival probability outside (0, 1]")
    result = copy.deepcopy(config)
    result["arrival_probability"] = probability
    validate(result)
    return result


def frozen_state(trainer):
    from mist_sim.training import state_hash
    return {"networks": {layer: state_hash(getattr(trainer.controller, layer)) for layer in ("upper", "lower")},
            "reward_normalizer": copy.deepcopy(trainer.reward_normalizer.state_dict()),
            "temperatures": {layer: getattr(getattr(trainer.controller, layer), "sampling_temperature", 1.)
                             for layer in ("upper", "lower")},
            "value_reward_scales": copy.deepcopy(trainer.value_reward_scales),
            "total_slots": trainer.total_slots, "versions": copy.deepcopy(trainer.versions),
            "optimizer_steps": {layer: [float(state["step"]) for state in optimizer.state.values()]
                                for layer, optimizer in trainer.optimizers.items()}}


def initialize_worker(method, deadline):
    import torch
    from mist_sim.process_runtime import configure_process
    from mist_sim.training import Trainer, implementation_signature
    from scripts.evaluation.evaluate_greedy import GreedyController
    torch.set_num_threads(1)
    if implementation_signature() != method["implementation_signature"]:
        raise ValueError("Evaluation imported the wrong source tree")
    trainer = None
    if method["checkpoint"]:
        if digest(method["checkpoint"]) != method["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum changed")
        trainer = Trainer.load(method["checkpoint"], device="cpu")
        if {**trainer.config, "device": method["config"]["device"]} != method["config"]:
            raise ValueError("Frozen configuration differs from loaded checkpoint")
        controller = trainer.controller
        controller.upper.eval()
        controller.lower.eval()
    else:
        controller = GreedyController()
    WORKER.update(method=method, trainer=trainer, controller=controller, deadline=deadline,
                  before=frozen_state(trainer) if trainer else None,
                  runtime=configure_process(method["config"]["runtime"]["windows_performance"]))


def mechanism_metrics(records, config):
    models = {name: Counter() for name in config["profiles"]}
    splits = Counter()
    allocated = reserved = bandwidth = 0.
    edge_counts = []
    for record in records:
        lookup = {item["id"]: item["model"] for item in record["raw"]}
        for model in lookup.values():
            models[model]["arrivals"] += 1
        for identity in record["suppressed"]:
            models[lookup[identity]]["suppressed"] += 1
        for task in record["tasks"]:
            model = lookup[task["id"]]
            models[model]["published"] += 1
            models[model]["completed"] += int(task["on_time"])
            models[model][task["action"]] += 1
            models[model]["timeout"] += int(task["reason"] == "deadline")
            models[model]["resource_failed"] += int(task["reason"] == "execution_memory")
            if task["action"] == "edge":
                splits[f"{model}:{task['split']}"] += 1
                allocated += task["rate"]
        reserved += sum(record["usage"]["compute"])
        bandwidth += sum(record["usage"]["bandwidth"])
        edge_counts.append(sum(task["action"] == "edge" for task in record["tasks"]))
    return {"models": {name: dict(counts) for name, counts in models.items()}, "splits": dict(splits),
            "ready_server_slots": sum(sum(record["ready"]) for record in records),
            "allocated_compute": allocated, "reserved_compute": reserved, "reserved_bandwidth": bandwidth,
            "bandwidth_saturated_slots": sum(value * config["reserved_bandwidth"] >=
                sum(access["bandwidth"] for access in config["access_points"]) - 1e-8 for value in edge_counts)}


def evaluate_task(task):
    import numpy as np
    import torch
    from mist_sim.diagnostics import check_physics
    from mist_sim.evaluation import evaluate, replay
    method, trainer, controller = WORKER["method"], WORKER["trainer"], WORKER["controller"]
    def check_budget():
        if time.time() >= WORKER["deadline"]:
            raise TimeoutError("Registered evaluation budget exhausted")
    check_budget()
    path = task["input"]["path"]
    if digest(path) != task["input"]["sha256"]:
        raise ValueError("Shared input checksum changed")
    config = evaluation_config(method["config"], task["probability"])
    if trainer:
        controller.config = config
    random.seed(task["action_seed"])
    np.random.seed(task["action_seed"])
    torch.manual_seed(task["action_seed"])
    started = time.perf_counter()
    with torch.no_grad():
        records, metrics = evaluate(controller, config, read_json(path)["trace"],
            check_budget=check_budget, deterministic=task["deterministic"])
    physics = check_physics(config, records)
    if len(records) != config["horizon"]:
        raise AssertionError("Incomplete evaluation episode")
    if trainer and frozen_state(trainer) != WORKER["before"]:
        raise AssertionError("Evaluation mutated training state or normalization")
    audit_hash = None
    if task.get("audit_path"):
        replay(config, records)
        write_json(task["audit_path"], records)
        audit_hash = digest(task["audit_path"])
    result = {"id": task["id"], "method": task["method"], "load": task["load"],
        "seed": task["input"]["seed"], "action_seed": task["action_seed"],
        "decode": "deterministic" if task["deterministic"] else "stochastic",
        "information": config["information"], "slots": len(records), "metrics": metrics,
        "mechanism": mechanism_metrics(records, config),
        "completed_delays": [item["delay"] for record in records for item in record["tasks"] if item["on_time"]],
        "physics": physics, "frozen_state_verified": True, "runtime": WORKER["runtime"],
        "wall_seconds": time.perf_counter() - started, "input_sha256": task["input"]["sha256"],
        "checkpoint_sha256": method["checkpoint_sha256"], "audit_sha256": audit_hash}
    return result


def scalar_row(result):
    metrics = result["metrics"]
    row = {key: result[key] for key in ("id", "method", "load", "seed", "action_seed", "decode", "information", "slots", "wall_seconds")}
    row.update({key: value for key, value in metrics.items() if key != "energy"})
    row.update({"energy_" + key: value for key, value in metrics["energy"].items()})
    row["energy_per_slot"] = metrics["total_energy"] / result["slots"]
    row["cost_per_slot"] = metrics["conservative_cost"] / result["slots"]
    published = metrics["arrivals"] - metrics["suppressed"]
    row["on_time_rate_published"] = metrics["completed"] / published if published else None
    row["input_sha256"] = result["input_sha256"]
    row["checkpoint_sha256"] = result["checkpoint_sha256"]
    return row


def materialize_csv(path, results):
    rows = [scalar_row(result) for result in sorted(results, key=lambda item: item["id"])]
    write_csv(path, rows)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_group(rows):
    import numpy as np
    slots = sum(row["slots"] for row in rows)
    totals = {name: sum(row["metrics"][name] for row in rows) for name in
              ("arrivals", "completed", "suppressed", "rejected", "timeout", "resource_failed",
               "total_energy", "conservative_cost", "active_server_slots", "switches", "wake_events",
               "projection_violations")}
    arrivals, published = totals["arrivals"], totals["arrivals"] - totals["suppressed"]
    delays = [value for row in rows for value in row["completed_delays"]]
    result = {"method": rows[0]["method"], "load": rows[0]["load"], "decode": rows[0]["decode"],
              "information": rows[0]["information"], "episodes": len(rows), "slots": slots, **totals,
              "arrivals_per_slot": arrivals / slots, "energy_per_slot": totals["total_energy"] / slots,
              "cost_per_slot": totals["conservative_cost"] / slots,
              "on_time_rate": totals["completed"] / arrivals if arrivals else None,
              "on_time_rate_published": totals["completed"] / published if published else None,
              "completion_delay_mean": float(np.mean(delays)) if delays else None,
              "completion_delay_p95": float(np.quantile(delays, .95)) if delays else None}
    for name in ("suppressed", "rejected", "timeout", "resource_failed"):
        result[name + "_rate"] = totals[name] / arrivals if arrivals else None
    for name in ("static", "wake", "local", "transmit", "edge"):
        result["energy_" + name + "_per_slot"] = sum(row["metrics"]["energy"][name] for row in rows) / slots
    return result


def paired_intervals(hpgi, comparison, replicates=5000, seed=0):
    import numpy as np
    hpgi = sorted(hpgi, key=lambda row: row["seed"])
    comparison = sorted(comparison, key=lambda row: row["seed"])
    if [(row["seed"], row["input_sha256"]) for row in hpgi] != [
            (row["seed"], row["input_sha256"]) for row in comparison]:
        raise ValueError("Paired inference requires identical request trajectories")
    indices = np.random.default_rng(seed).integers(0, len(hpgi), (replicates, len(hpgi)))
    estimates = []
    for metric, numerator, denominator in (("energy_per_slot", "total_energy", "slots"),
                                           ("on_time_rate", "completed", "arrivals")):
        samples = []
        point = []
        for rows in (hpgi, comparison):
            numerators = np.array([row["metrics"][numerator] for row in rows], dtype=float)
            denominators = np.array([row["slots"] if denominator == "slots" else
                                     row["metrics"][denominator] for row in rows], dtype=float)
            if not np.all(denominators > 0):
                raise ValueError("Bootstrap ratio has zero denominator")
            samples.append(numerators[indices].sum(1) / denominators[indices].sum(1))
            point.append(numerators.sum() / denominators.sum())
        lower, upper = np.quantile(samples[0] - samples[1], [.025, .975])
        estimates.append({"load": hpgi[0]["load"], "decode": hpgi[0]["decode"],
                          "comparison": comparison[0]["method"], "metric": metric,
                          "hpgi_minus_comparison": point[0] - point[1], "ci95_low": float(lower),
                          "ci95_high": float(upper), "paired_trajectories": len(hpgi),
                          "bootstrap_replicates": replicates, "bootstrap_seed": seed})
    return estimates


def summarize_batches(spec_path, output):
    specification = read_json(spec_path)
    groups = {}
    sources = {}
    for batch in ("A", "B"):
        run_id = f"{spec_path.parent.name}-{batch}"
        status = read_json(ROOT / "logs/runs" / run_id / "status.json")
        if status["status"] != "completed" or status["spec_sha256"] != digest(spec_path):
            raise ValueError("Both registered batches must finish before pooling")
        path = ROOT / "results" / run_id / "observations.jsonl"
        sources[str(path)] = digest(path)
        if sources[str(path)] != status["observations_sha256"]:
            raise ValueError("Completed results checksum changed")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            groups.setdefault((row["load"], row["decode"], row["method"]), []).append(row)
    summaries = [aggregate_group(rows) for rows in groups.values()]
    pairs = []
    mechanisms = {}
    for (load, decode, method), rows in groups.items():
        if method != "hpgi":
            pairs.extend(paired_intervals(groups[load, decode, "hpgi"], rows,
                specification["bootstrap_replicates"], specification["bootstrap_seed"]))
        models, splits = {}, Counter()
        for row in rows:
            splits.update(row["mechanism"]["splits"])
            for model, counts in row["mechanism"]["models"].items():
                models.setdefault(model, Counter()).update(counts)
        mechanisms[f"{load}:{decode}:{method}"] = {"models": models, "splits": splits,
            **{name: sum(row["mechanism"][name] for row in rows) for name in
               ("ready_server_slots", "allocated_compute", "reserved_compute", "reserved_bandwidth", "bandwidth_saturated_slots")}}
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_differences.csv", pairs)
    write_json(output / "mechanisms.json", mechanisms)
    write_json(output / "provenance.json", {"spec": str(spec_path), "spec_sha256": digest(spec_path),
        "sources": sources, "aggregation_source_sha256": digest(__file__), "training_updates": 0,
        "uncertainty": specification["interpretation"],
        "result_sha256": {name: digest(output / name) for name in
                          ("summary.csv", "paired_differences.csv", "mechanisms.json")}})
    print(json.dumps({"output": str(output), "groups": len(groups),
                      "episodes": sum(len(rows) for rows in groups.values())}), flush=True)


def run_cohort(spec_path, batch, method_name, output, log, workers, deadline):
    specification = read_json(spec_path)
    tasks = [task for task in specification["tasks"] if task["batch"] == batch and task["method"] == method_name]
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "observations.jsonl"
    results = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    finished = {row["id"] for row in results}
    tasks = [task for task in tasks if task["id"] not in finished]
    for task in tasks:
        if task["audit"]:
            task["audit_path"] = str(log / "audit" / (task["id"] + ".json"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(specification["methods"][method_name], deadline)) as pool:
        futures = {pool.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            results.append(result)
            materialize_csv(output / "evaluation.csv", results)
            message = (f"{batch} {len(results)} | {result['id']} | energy/slot="
                       f"{result['metrics']['total_energy'] / result['slots']:.3f} | on-time="
                       f"{result['metrics']['on_time_rate']:.2%} | {result['wall_seconds']:.2f}s | pid={result['runtime']['pid']}")
            print(message, flush=True)
    method = specification["methods"][method_name]
    if method["checkpoint"] and digest(method["checkpoint"]) != method["checkpoint_sha256"]:
        raise AssertionError("Checkpoint file changed during evaluation")


def run_batch(spec_path, batch, output=None, log=None):
    specification = read_json(spec_path)
    if batch != "preflight":
        preflight = read_json(spec_path.parent / "preflight.json")
        if preflight["status"] != "completed" or preflight["spec_sha256"] != digest(spec_path):
            raise ValueError("This exact evaluation specification has not passed preflight")
    run_id = f"{spec_path.parent.name}-{batch}"
    output = output or ROOT / "results" / run_id
    log = log or ROOT / "logs/runs" / run_id
    log.mkdir(parents=True, exist_ok=True)
    if (log / "status.json").exists() and read_json(log / "status.json")["spec_sha256"] != digest(spec_path):
        raise ValueError("Existing run belongs to a different specification")
    lock = log / "active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    status = {"status": "running", "pid": os.getpid(), "spec_sha256": digest(spec_path),
              "batch": batch, "started": time.time(), "training_updates": 0, "output": str(output),
              "evaluation_script_sha256": digest(__file__)}
    status_path = log / "status.json"
    write_json(status_path, status)
    try:
        deadline = time.time() + specification["budget_seconds"][batch]
        methods = list(dict.fromkeys(task["method"] for task in specification["tasks"] if task["batch"] == batch))
        for name in methods:
            method = specification["methods"][name]
            files = source_tree(method["revision"])
            if source_signature(files) != method["implementation_signature"]:
                raise ValueError("Frozen source signature mismatch")
            with tempfile.TemporaryDirectory(prefix="mist-frozen-source-") as temporary:
                extract_source(files, temporary)
                environment = dict(os.environ, PYTHONPATH=str(Path(temporary) / "src"))
                command = [sys.executable, "-u", "-m", "scripts.evaluation.evaluate_frozen", "cohort",
                    "--spec", str(spec_path), "--batch", batch, "--method", name,
                    "--output", str(output), "--log", str(log), "--workers", str(specification["workers_per_terminal"]),
                    "--deadline", str(deadline)]
                with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        with (log / "stdout.log").open("a", encoding="utf-8") as stream:
                            stream.write(line)
                    if process.wait() != 0:
                        raise RuntimeError(f"{name} evaluation failed; see {log / 'stdout.log'}")
        rows = [json.loads(line) for line in (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = {task["id"] for task in specification["tasks"] if task["batch"] == batch}
        if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
            raise AssertionError("Missing or duplicate evaluation cells")
        materialize_csv(output / "evaluation.csv", rows)
        status.update(status="completed", episodes=len(rows), wall_seconds=time.time() - status["started"],
                      observations_sha256=digest(output / "observations.jsonl"), csv_sha256=digest(output / "evaluation.csv"))
        if batch == "preflight":
            status["timing_by_method_decode"] = {}
            for row in rows:
                key = row["method"] + ":" + row["decode"]
                status["timing_by_method_decode"].setdefault(key, []).append(row["wall_seconds"])
            status["physical_checks_passed"] = all(row["physics"]["constraints_and_kkt"] for row in rows)
            status["frozen_state_checks_passed"] = all(row["frozen_state_verified"] for row in rows)
            write_json(spec_path.parent / "preflight.json", status)
    except BaseException:
        status.update(status="failed", error=traceback.format_exc(), wall_seconds=time.time() - status["started"])
        if (log / "stdout.log").exists():
            status["error_log_tail"] = (log / "stdout.log").read_text(encoding="utf-8").splitlines()[-80:]
        raise
    finally:
        write_json(status_path, status)
        if batch == "preflight":
            write_json(spec_path.parent / "preflight.json", status)
        lock.unlink()
    print(json.dumps(status), flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "cohort", "summarize"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--batch", choices=("preflight", "A", "B"))
    parser.add_argument("--method")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    spec = args.spec.resolve()
    if args.command == "prepare":
        prepare(spec.parent)
    elif args.command == "preflight":
        with tempfile.TemporaryDirectory(prefix="mist-frozen-preflight-") as temporary:
            run_batch(spec, "preflight", Path(temporary) / "results", Path(temporary) / "logs")
    elif args.command == "run":
        if args.batch not in ("A", "B"):
            parser.error("run requires --batch A or B")
        run_batch(spec, args.batch)
    elif args.command == "summarize":
        summarize_batches(spec, args.output or ROOT / "results/comparisons" / spec.parent.name)
    else:
        run_cohort(spec, args.batch, args.method, args.output, args.log, args.workers, args.deadline)


if __name__ == "__main__":
    main()
