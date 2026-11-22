"""Bounded static-policy selection and latency-reserve checks on frozen physics.

Only experiment orchestration and controller adapters live here. Candidate
generation, greedy scheduling, P3, execution and accounting use the original
checkpoint's public simulator implementation.
"""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from scripts.evaluation import evaluate_frozen as f

ROOT = f.ROOT
NAME = "paper-strengthening-20261123"
SPEC = ROOT / "logs/launchers" / NAME / "spec.json"
RESULT = ROOT / "results" / NAME
LOG = ROOT / "logs/runs" / NAME


class StaticController:
    """Hold a declared server vector; delegate fast actions to public greedy."""
    def __init__(self, active, lower=None):
        self.active = list(active)
        self.lower = lower

    def slow(self, observation, **kwargs):
        return list(self.active), None

    def fast(self, problem, **kwargs):
        if self.lower is not None:
            return self.lower.fast(problem, **kwargs)
        from mist_sim.policies.baselines.heuristics import greedy_fast
        return greedy_fast(problem)


def server_vectors(config):
    # Exchangeability is verified by ALL server fields (including AP/model).
    groups = {}
    for i, server in enumerate(config["servers"]):
        groups.setdefault(json.dumps(server, sort_keys=True), []).append(i)
    vectors = []
    for counts in itertools.product(*(range(len(group) + 1) for group in groups.values())):
        active = [0] * len(config["servers"])
        for count, group in zip(counts, groups.values()):
            for i in group[:count]:
                active[i] = 1
        vectors.append(active)
    return sorted(vectors)


def initialize(method, deadline):
    f.initialize_worker(method, deadline)
    if method.get("active") is not None:
        lower = f.WORKER["controller"] if method.get("learned_lower") else None
        f.WORKER["controller"] = StaticController(method["active"], lower)


def static_method(base, active, cuts=None, learned_lower=False):
    method = copy.deepcopy(base)
    method["active"] = list(active)
    method["learned_lower"] = learned_lower
    if not learned_lower:
        method.update(checkpoint=None, checkpoint_sha256=None, total_training_slots=0)
    if cuts is not None:
        method["config"]["dynamic_split"] = False
        method["config"]["fixed_split"] = cuts
    return method


def task(method, entry, load="middle", probability=.5, reserve=.01):
    return {"id": f"{method}-{load}-{reserve}-{entry['seed']}", "method": method,
            "load": load, "probability": probability, "deterministic": True,
            "action_seed": entry["seed"] + 10000000, "input": entry,
            "overrides": {"control_seconds": reserve}, "reserve": reserve}


def evaluate_group(payload):
    method, tasks, deadline = payload
    initialize(method, deadline)
    return [f.evaluate_task(t) for t in tasks]


def prepare():
    from mist_sim.anonymity import generate_trace
    if SPEC.exists():
        raise FileExistsError("Specification already frozen")
    base = f.read_json(ROOT / "logs/launchers/paper-completion-v1/spec.json")
    methods = {key: base["methods"][key] for key in ("hpgi", "single_timescale", "fixed_split")}
    for method in methods.values():
        files = f.source_tree(method["revision"])
        if f.source_signature(files) != method["implementation_signature"]:
            raise ValueError("Source signature differs")
        if f.digest(method["checkpoint"]) != method["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs")
    seeds = {"development": list(range(5600101, 5600105)),
             "low": list(range(5600201, 5600221)),
             "middle": list(range(5600301, 5600321)),
             "high": list(range(5600401, 5600421))}
    audit = f.check_unused_seeds(set().union(*map(set, seeds.values())))
    inputs = {}
    for label, values in seeds.items():
        probability = {"low": .25, "high": .75}.get(label, .5)
        inputs[label] = []
        config = {**methods["hpgi"]["config"], "arrival_probability": probability}
        for seed in values:
            path = ROOT / "data/processed/evaluation" / NAME / label / f"seed{seed}.json"
            if path.exists():
                raise FileExistsError(path)
            f.write_json(path, {"seed": seed, "trace": generate_trace(config, seed)})
            inputs[label].append({"seed": seed, "path": str(path), "sha256": f.digest(path)})
    old = f.read_json(ROOT / "logs/launchers/paper-evaluation-v1/spec.json")
    timing_inputs = {label: [t["input"] for t in old["tasks"] if t["method"] == "hpgi"
        and t["deterministic"] and t["load"] == label and t["batch"] in ("A", "B")]
        for label in ("middle", "high")}
    cfg = methods["hpgi"]["config"]
    vectors = server_vectors(cfg)
    cuts = [dict(zip(sorted(cfg["profiles"]), values)) for values in itertools.product(
        *(cfg["profiles"][m]["legal_splits"] for m in sorted(cfg["profiles"])))]
    spec = {"methods": methods, "inputs": inputs, "timing_inputs": timing_inputs,
        "server_vectors": vectors, "split_combinations": cuts, "seed_audit": audit,
        "training_updates": 0, "workers": 4, "budget_seconds": {"preflight": 600, "static": 14400, "latency": 7200},
        "selection": "81 server sets with adaptive greedy; 64 fixed cuts at selected set; 81 sets at selected cuts. Four development episodes. No test-based reselection.",
        "diagnostic": "If static beats HPGI, compare frozen HPGI lower under selected servers and greedy under HPGI modal set, middle load only; interventions are not trained baselines.",
        "reserve_seconds": [.01, .06, .08], "source_script_sha256": f.digest(__file__),
        "frozen_evaluator_sha256": f.digest(f.__file__),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    f.write_json(SPEC, spec)
    print(json.dumps({"spec": str(SPEC), "sets": len(vectors), "cuts": len(cuts), "seed_audit": audit}), flush=True)


def stage(name, groups, deadline):
    """Resume only identical task specifications; append completed atomic groups."""
    path = RESULT / name / "observations.jsonl"
    manifest = {"spec_sha256": f.digest(SPEC), "groups_sha256": f.object_digest(groups)}
    saved = LOG / name / "manifest.json"
    if saved.exists() and f.read_json(saved) != manifest:
        raise ValueError("Resume task identity differs")
    f.write_json(saved, manifest)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {r["id"] for r in rows}
    pending = [(method, [t for t in tasks if t["id"] not in done], deadline) for method, tasks in groups]
    pending = [g for g in pending if g[1]]
    if pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(evaluate_group, item) for item in pending]
            for future in as_completed(futures):
                batch = future.result()
                with path.open("a", encoding="utf-8") as stream:
                    for row in batch:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.extend(batch)
                f.materialize_csv(path.parent / "evaluation.csv", rows)
                print(f"{name}: {len(rows)} episodes; {batch[0]['method']}; "
                      f"cost={sum(r['metrics']['conservative_cost'] for r in batch)/sum(r['slots'] for r in batch):.4f}", flush=True)
    expected = {t["id"] for _, tasks in groups for t in tasks}
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise AssertionError("Missing/duplicate episodes")
    if not all(r["physics"]["constraints_and_kkt"] and r["frozen_state_verified"] for r in rows):
        raise AssertionError("Physics/freeze check failed")
    return rows


def select(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    scores = {key: sum(r["metrics"]["conservative_cost"] for r in group)/sum(r["slots"] for r in group)
              for key, group in groups.items()}
    winner = min(scores, key=lambda key: (scores[key], key))
    return winner, scores


def run_static(spec, deadline):
    base, dev = spec["methods"]["hpgi"], spec["inputs"]["development"]
    vectors, cuts = spec["server_vectors"], spec["split_combinations"]
    groups = [(static_method(base, active), [task(f"server{i:03d}", e) for e in dev]) for i, active in enumerate(vectors)]
    rows = stage("dev_servers", groups, deadline)
    winner, scores = select(rows)
    active = vectors[int(winner.removeprefix("server"))]
    f.write_json(LOG / "selection_servers.json", {"winner": winner, "active": active, "scores": scores})
    groups = [(static_method(base, active, cut), [task(f"cut{i:03d}", e) for e in dev]) for i, cut in enumerate(cuts)]
    rows = stage("dev_cuts", groups, deadline)
    winner_cut, scores_cut = select(rows)
    chosen_cuts = cuts[int(winner_cut.removeprefix("cut"))]
    groups = [(static_method(base, vector, chosen_cuts), [task(f"fixedserver{i:03d}", e) for e in dev]) for i, vector in enumerate(vectors)]
    rows = stage("dev_fixed_servers", groups, deadline)
    winner_fixed, scores_fixed = select(rows)
    fixed_active = vectors[int(winner_fixed.removeprefix("fixedserver"))]
    selection = {"adaptive_active": active, "fixed_active": fixed_active, "fixed_cuts": chosen_cuts,
        "server_scores": scores, "cut_scores": scores_cut, "fixed_server_scores": scores_fixed,
        "development_inputs": dev, "spec_sha256": f.digest(SPEC)}
    frozen = LOG / "selection.json"
    if frozen.exists() and f.read_json(frozen) != selection:
        raise ValueError("Frozen selection changed")
    f.write_json(frozen, selection)
    methods = {"hpgi": base, "static_adaptive": static_method(base, active),
               "static_fixed": static_method(base, fixed_active, chosen_cuts)}
    groups = [(method, [task(name, e, label, p) for e in spec["inputs"][label]])
              for name, method in methods.items() for label, p in (("low", .25), ("middle", .5), ("high", .75))]
    stage("test", groups, deadline)


def run_latency(spec, deadline, method_name):
    method = spec["methods"][method_name]
    groups = [(method, [task(method_name, e, label, p, reserve) for e in entries])
        for label, entries in spec["timing_inputs"].items() for reserve in (.06, .08)
        for p in ([.5] if label == "middle" else [.75])]
    stage("latency_" + method_name, groups, deadline)


def launch(mode):
    spec = f.read_json(SPEC)
    if spec["source_script_sha256"] != f.digest(__file__) or spec["frozen_evaluator_sha256"] != f.digest(f.__file__):
        raise ValueError("Experiment orchestration differs from frozen specification")
    if mode != "preflight":
        preflight = f.read_json(LOG / "preflight_status.json")
        if preflight["status"] != "completed" or preflight["spec_sha256"] != f.digest(SPEC):
            raise ValueError("Exact specification has not passed preflight")
    lock = LOG / (mode + ".lock")
    LOG.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    started = time.time()
    status = {"status": "running", "pid": os.getpid(), "started": started,
              "spec_sha256": f.digest(SPEC), "training_updates": 0}
    status_path = LOG / (mode + "_status.json")
    f.write_json(status_path, status)
    try:
        names = list(spec["methods"]) if mode == "latency" else ["hpgi"]
        for name in names:
            method = spec["methods"][name]
            files = f.source_tree(method["revision"])
            if f.source_signature(files) != method["implementation_signature"]:
                raise ValueError("Historical source mismatch")
            with tempfile.TemporaryDirectory(prefix="mist-evidence-source-") as directory:
                f.extract_source(files, directory)
                env = dict(os.environ, PYTHONPATH=str(Path(directory) / "src"))
                command = [sys.executable, "-u", "-m", "scripts.evaluation.strengthen_evidence", "worker",
                           "--mode", mode, "--method", name, "--deadline", str(started + spec["budget_seconds"][mode])]
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      encoding="utf-8", errors="replace") as child:
                    with (LOG / (mode + ".log")).open("a", encoding="utf-8") as stream:
                        for line in child.stdout:
                            print(line, end="", flush=True)
                            stream.write(line)
                            stream.flush()
                    if child.wait():
                        raise RuntimeError(f"{name} failed; see {mode}.log")
        status["status"] = "completed"
    except BaseException:
        import traceback
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status["wall_seconds"] = time.time() - started
        f.write_json(status_path, status)
        lock.unlink()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "static", "latency", "worker"))
    parser.add_argument("--mode", choices=("preflight", "static", "latency"))
    parser.add_argument("--method")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        spec = f.read_json(SPEC)
        if args.mode == "preflight":
            base, entry = spec["methods"]["hpgi"], spec["inputs"]["development"][0]
            groups = [(static_method(base, active, cuts), [task(f"probe{i}", entry)])
                for i, (active, cuts) in enumerate(((spec["server_vectors"][0], None),
                    (spec["server_vectors"][-1], None), (spec["server_vectors"][-1], spec["split_combinations"][0])))]
            groups += [(base, [task("hpgi", entry, reserve=reserve)]) for reserve in (.01, .06, .08)]
            stage("preflight", groups, args.deadline)
        elif args.mode == "static":
            run_static(spec, args.deadline)
        else:
            run_latency(spec, args.deadline, args.method)
    else:
        launch(args.command)


if __name__ == "__main__":
    main()
