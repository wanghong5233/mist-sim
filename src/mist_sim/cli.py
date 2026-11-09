"""Unified commands: check, train, evaluate, replay, summarize, reference."""
import argparse
import json
from pathlib import Path
import sys
import torch
from .config import load_config, for_method, METHODS, default_project_root
from .anonymity import generate_trace
from .environment import Environment
from .evaluation import evaluate, replay, summarize
from .policies.hpgi import Controller
from .reference_solvers import enumerate_slot, enumerate_trajectory
from .storage import RunStore, write_json, digest_file, validation_workspace
from .training import Trainer
from .schedule import TrainingLimit, describe_schedule


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "evaluate", "replay", "summarize", "reference"))
    root = default_project_root()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="PPO update device; environment and sampling stay on CPU")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, help="Explicit run output directory; source root stays unchanged")
    parser.add_argument("--extra-cycles", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, help="Training wall-time limit, including calibration and validation")
    parser.add_argument("--max-slots", type=int, help="Cumulative training-slot cap, checked before each complete rollout")
    parser.add_argument("--dry-run", action="store_true", help="Show training counts without sampling or creating a run")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--measured-latency", action="store_true")
    args = parser.parse_args(argv)
    args.invocation = " ".join(sys.argv if argv is None else argv)
    if (args.dry_run or args.max_slots is not None or args.max_seconds is not None) and args.command != "train":
        parser.error("--dry-run/--max-slots/--max-seconds require train")
    if args.extra_cycles < 0 or (args.max_slots is not None and args.max_slots <= 0):
        parser.error("extra-cycles must be nonnegative; max-slots must be positive")
    if args.max_seconds is not None:
        import math
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error("max-seconds must be positive and finite")
    if args.trace and args.command not in ("check", "evaluate", "reference"):
        raise ValueError("--trace is only supported by check/evaluate/reference; training uses configured seeded generation")
    if args.checkpoint and (args.overlay or args.overrides) and not args.config:
        raise ValueError("Checkpoint overrides require explicit --config for compatibility checking")
    if args.command == "check" and args.output_dir is None:
        with validation_workspace() as workspace:
            args.output_dir = workspace / "logs/runs"
            execute(args)
        print("Check passed; temporary artifacts removed")
        return None
    return execute(args)


def execute(args):
    torch.set_num_threads(1)
    if args.checkpoint:
        trainer = Trainer.load(args.checkpoint, device=args.device)
        config = trainer.config
        if args.config:
            config = load_config(args.config, args.overlay, args.overrides)
            config = for_method(config, args.method or config["method"])
            trainer = Trainer.load(args.checkpoint, config, device=args.device)
            config = trainer.config
        elif args.method and args.method != config["method"]:
            raise ValueError("Cannot relabel checkpoint method")
    else:
        config = load_config(args.config or args.root / "configs" / "smoke.json", args.overlay, args.overrides)
        config = for_method(config, args.method or config["method"])
        if args.device is not None:
            config["device"] = args.device
        trainer = None
    if args.command == "train" and config["method"] == "greedy_consolidation":
        raise ValueError("Heuristic baseline uses evaluate, not train")
    if args.dry_run:
        schedule = describe_schedule(config, args.extra_cycles, trainer.completed_stages if trainer else ())
        schedule["stopping"] = dict(config["stopping"])
        for name in ("max_slots", "max_seconds"):
            if getattr(args, name) is not None:
                schedule["stopping"][name] = getattr(args, name)
        schedule["completed_training_slots"] = trainer.total_slots if trainer else 0
        schedule["remaining_training_slots"] = schedule["planned_training_slots"] - schedule["completed_training_slots"]
        print(json.dumps(schedule, ensure_ascii=False, indent=2))
        return schedule
    if config["fixture_kind"] == "synthetic_hand_calculation" and args.output_dir is None:
        with validation_workspace() as workspace:
            args.output_dir = workspace / "logs/runs"
            run_command(args, config, trainer)
        print("Synthetic check finished; temporary artifacts removed")
        return None
    return run_command(args, config, trainer)


def run_command(args, config, trainer):
    seed = args.seed if args.seed is not None else config["seeds"]["train" if args.command == "train" else "test"]
    if trainer is not None and args.command == "train" and args.seed is None:
        seed = trainer.seed
    if args.extra_cycles < 0:
        raise ValueError("extra-cycles must be nonnegative")
    store = RunStore(args.root, config, args.invocation, seed, args.run_id, output_dir=args.output_dir)
    try:
        if args.checkpoint:
            store.manifest["checkpoint"] = {"path": str(args.checkpoint.resolve()), "sha256": digest_file(args.checkpoint)}
        if args.command == "train":
            store.manifest["budget"] = {"purpose": "policy_training",
                "wall_seconds_limit": args.max_seconds if args.max_seconds is not None else config["stopping"]["max_seconds"],
                "training_slots_limit": args.max_slots if args.max_slots is not None else config["stopping"]["max_slots"]}
            store.write_json("manifest.json", store.manifest)
            if trainer is None:
                trainer = Trainer(config, seed, store)
            else:
                if args.seed is not None and args.seed != trainer.seed:
                    raise ValueError("Resume seed cannot change")
                trainer.store = store
            trainer.run(args.extra_cycles, max_seconds=args.max_seconds, max_slots=args.max_slots)
            store.write_json("summary.json", {"stages": trainer.completed_stages, "total_slots": trainer.total_slots,
                                              "stop_reason": trainer.stop_reason})
            store.manifest["stop_reason"] = trainer.stop_reason
        elif args.command in ("replay", "summarize"):
            if not args.records:
                raise ValueError("--records required")
            records = json.loads(args.records.read_text(encoding="utf-8"))
            store.manifest["records_input"] = {"path": str(args.records.resolve()), "sha256": digest_file(args.records)}
            result = replay(config, records) if args.command == "replay" else summarize(records)
            store.write_json("summary.json", result)
        else:
            trace_seed = seed
            if args.trace:
                payload = json.loads(args.trace.read_text(encoding="utf-8"))
                trace = payload.get("trace") if isinstance(payload, dict) else payload
                trace_seed = payload.get("seed") if isinstance(payload, dict) else None
                if not isinstance(trace, list) or any(not isinstance(batch, list) or
                        any(not isinstance(item, dict) for item in batch) for batch in trace):
                    raise ValueError("Trace must be a list of request batches or a {seed, trace} envelope")
                if trace_seed is not None and (type(trace_seed) is not int or trace_seed < 0):
                    raise ValueError("Trace seed must be a nonnegative integer or null")
                store.manifest["trace_input"] = {"path": str(args.trace.resolve()),
                    "sha256": digest_file(args.trace), "seed": trace_seed}
            else:
                trace = generate_trace(config, seed)
            store.write_json("traces/input.json", trace)
            if args.command == "reference":
                if args.offline:
                    result = enumerate_trajectory(config, trace, args.max_nodes, args.timeout)
                else:
                    environment = Environment(config, trace)
                    problem = environment.begin_slot(config["initial_active"])
                    result = enumerate_slot(problem, args.max_nodes, args.timeout)
                store.write_json("reference.json", result)
            else:
                if args.command == "evaluate" and trainer is None and config["method"] != "greedy_consolidation":
                    raise ValueError("Learned methods require their own checkpoint")
                controller = trainer.controller if trainer else Controller(for_method(config, "greedy_consolidation"))
                if args.command == "check" and config["learned_rates"] and trainer is None:
                    controller = KernelCheckController(config)
                records, metrics = evaluate(controller, config, trace, args.measured_latency)
                store.write_json("traces/evaluation.json", records)
                store.write_json("summary.json", metrics)
                if args.command == "evaluate":
                    store.record_evaluation(metrics, trajectory_seed=trace_seed)
                if args.command == "check":
                    replay(config, records)
        store.finish()
    except TrainingLimit as error:
        trainer.event("training_stopped", reason=error.reason)
        if trainer.phase_in_progress:
            trainer.save(store.path / "checkpoints" / "interrupted.pt")
        checkpoint = store.path / "checkpoints" / "last.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        store.write_json("summary.json", {"stages": trainer.completed_stages, "total_slots": trainer.total_slots,
            "stop_reason": error.reason, "resume_checkpoint": str(checkpoint),
            "resume_slots": saved["total_slots"], "discarded_incomplete_phase_slots": trainer.total_slots - saved["total_slots"]})
        store.finish(stopped=error.reason)
    except BaseException as error:
        if trainer is not None:
            trainer.save(store.path / "checkpoints" / "failed_last.pt")
        store.finish(error)
        raise
    print(str(store.path))
    return store.path


class KernelCheckController:
    """Explicit hand-driven kernel check, not an untrained learned baseline."""
    def __init__(self, config):
        self.config = config

    def slow(self, observation, deterministic=False, record=False):
        return [1] * len(self.config["servers"]), None

    def fast(self, problem, deterministic=False, record=False):
        from .policies.baselines.heuristics import greedy_fast
        actions, _, _ = greedy_fast(problem)
        _, rates, _, _ = problem.allocate(actions)
        return actions, rates.tolist(), None
