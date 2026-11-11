"""Bounded engineering validation, never a paper performance experiment."""
import copy
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from .anonymity import generate_trace
from .config import METHODS, for_method, load_config
from .evaluation import evaluate, replay, assert_close
from .policies.registry import make_controller
from .storage import RunStore, digest_file, write_json, validation_workspace
from .training import Trainer, parameter_hash


def independent_totals(config, records):
    total_arrivals, total_completed, total_rejected, total_suppressed, total_timeout, total_resource = (0,) * 6
    energy = dict(static=0.0, wake=0.0, local=0.0, transmit=0.0, edge=0.0)
    cost = 0.0
    for record in records:
        total_arrivals += len(record["raw"])
        total_suppressed += len(record["suppressed"])
        for index, server in enumerate(config["servers"]):
            energy["static"] += config["slot_seconds"] * (server["idle_power"] if record["active"][index] else server["sleep_power"])
            energy["wake"] += server["wake_energy"] * max(0, record["active"][index] - record["previous_active"][index])
        for task in record["tasks"]:
            total_completed += int(task["on_time"])
            total_rejected += int(task["action"] == "reject")
            total_timeout += int(task["reason"] == "deadline")
            total_resource += int(task["reason"] == "execution_memory")
            for key in ("local", "transmit", "edge"):
                energy[key] += task["energy"][key]
        cost += record["conservative_energy"] + config["lambda_time"] * record["weighted_conservative_delay"] + record["rejection_penalty"]
    if total_arrivals != total_completed + total_rejected + total_suppressed + total_timeout + total_resource:
        raise AssertionError("Independent denominator mismatch")
    return {"arrivals": total_arrivals, "completed": total_completed, "rejected": total_rejected,
            "suppressed": total_suppressed, "timeout": total_timeout, "resource_failed": total_resource,
            "energy": energy, "total_energy": sum(energy.values()), "conservative_cost": cost}


def verify_one(root, config, seed, output_dir):
    store = RunStore(root, config, "python -m mist_sim.verification: method short validation", seed,
                     output_dir=output_dir)
    trainer = None
    try:
        if config["method"] == "greedy_consolidation":
            controller = make_controller(config)
        else:
            trainer = Trainer(config, seed, store)
            trainer.run()
            controller = trainer.controller
        test_seed = config["seeds"]["test"] + (config["horizon"] - 16)
        trace = generate_trace(config, test_seed)
        store.write_json("traces/test_input.json", {"seed": test_seed, "trace": trace})
        records, metrics = evaluate(controller, config, trace)
        store.write_json("traces/test.json", records)
        replayed = replay(config, records)
        assert_close(metrics, replayed)
        independent = independent_totals(config, records)
        for key, value in independent.items():
            assert_close(value, metrics[key], key)
        restored_equal = None
        if trainer:
            checkpoint = "last.pt" if trainer.minimal_checkpoints else "final.pt"
            reloaded = Trainer.load(store.path / "checkpoints" / checkpoint)
            restored_records, restored_metrics = evaluate(reloaded.controller, config, trace)
            assert_close(metrics, restored_metrics)
            ignored = ("measured_policy_seconds", "measured_allocation_seconds")
            assert_close([{key: value for key, value in record.items() if key not in ignored} for record in records],
                         [{key: value for key, value in record.items() if key not in ignored} for record in restored_records])
            restored_equal = True
        report = {"method": config["method"], "horizon": config["horizon"], "seed": seed,
                  "run_id": store.path.name, "stages": trainer.completed_stages if trainer else [],
                  "training_slots": trainer.total_slots if trainer else 0,
                  "test_slots": len(records), "reload_equal": restored_equal,
                  "replay_equal": True, "independent_totals_equal": True,
                  "metrics": metrics, "not_performance_evidence": True}
        store.write_json("summary.json", report)
        store.finish()
        return report, store.path
    except BaseException as error:
        if trainer:
            trainer.save(store.path / "checkpoints" / "failed_last.pt")
        store.finish(error)
        raise
    finally:
        if trainer:
            trainer.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from .config import default_project_root
    root = default_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-artifacts", type=Path, help="Explicit new debug directory; default uses disposable OS temp")
    parser.add_argument("--skip-unit-tests", action="store_true", help="Use only after a separate successful pytest run")
    parser.add_argument("--summary", type=Path, help="Persist compact evidence only; temporary weights and traces are deleted")
    args = parser.parse_args()
    with validation_workspace(args.keep_artifacts) as workspace:
        summary = run_suite(root, workspace, run_tests=not args.skip_unit_tests)
        if args.summary:
            write_json(args.summary, summary)
    print("Validation finished; " + (f"debug artifacts: {workspace}" if args.keep_artifacts else "temporary artifacts removed"))


def run_suite(root, workspace, run_tests=True):
    config = load_config(root / "configs" / "smoke.json")
    config["reward_normalization"]["enabled"] = True
    output_dir = workspace / "logs/runs"
    suite = RunStore(root, config, "python -m mist_sim.verification", config["seeds"]["train"], output_dir=output_dir)
    started = time.perf_counter()
    reports = []
    try:
        test_exit_code = None
        if run_tests:
            tested = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=root,
                                    capture_output=True, timeout=180)
            (suite.path / "unit_tests_stdout.log").write_bytes(tested.stdout)
            (suite.path / "unit_tests_stderr.log").write_bytes(tested.stderr)
            test_exit_code = tested.returncode
            if test_exit_code:
                print(tested.stdout.decode("utf-8", errors="replace"), flush=True)
                print(tested.stderr.decode("utf-8", errors="replace"), flush=True)
                raise RuntimeError("Unit tests failed; diagnostics printed before temporary cleanup")
        for horizon in (16, 17):
            for method in METHODS:
                if time.perf_counter() - started > 3600:
                    raise TimeoutError("Engineering verification exceeded 3600 seconds")
                case = for_method(copy.deepcopy(config), method)
                case["horizon"] = horizon
                report, run_path = verify_one(root, case, config["seeds"]["train"] + horizon - 16, output_dir)
                reports.append(report)
                suite.write_json("cases.json", reports)
                print(json.dumps({"method": method, "horizon": horizon, "run_id": run_path.name,
                                  "training_slots": report["training_slots"], "status": "passed"}), flush=True)
        original = next(report for report in reports if report["method"] == "hpgi" and report["horizon"] == 16)
        checkpoint = output_dir / original["run_id"] / "checkpoints" / "final.pt"
        resumed_store = RunStore(root, config, "resume HPGI for one bounded cycle", config["seeds"]["train"], output_dir=output_dir)
        try:
            resumed = Trainer.load(checkpoint, store=resumed_store)
            before_slots = resumed.total_slots
            before_parameters = parameter_hash(resumed.controller.upper)
            resumed.run(extra_cycles=1)
            if before_parameters == parameter_hash(resumed.controller.upper):
                raise AssertionError("Resume did not update")
            resumed_store.manifest["checkpoint"] = {"path": str(checkpoint), "sha256": digest_file(checkpoint)}
            resumed_store.write_json("summary.json", {"additional_slots": resumed.total_slots - before_slots,
                                                     "versions": resumed.versions, "stages": resumed.completed_stages})
            resumed_store.finish()
        except BaseException as error:
            resumed_store.finish(error)
            raise
        report = {"scope": "Synthetic correctness / trainability only; not paper results",
                  "suite_run_id": suite.path.name, "cases": reports, "resume_run_id": resumed_store.path.name,
                  "resume_additional_slots": resumed.total_slots - before_slots,
                  "unit_test_exit_code": test_exit_code, "unit_tests_skipped": not run_tests,
                  "source": suite.manifest["source"],
                  "wall_seconds": time.perf_counter() - started}
        suite.write_json("summary.json", report)
        suite.finish()
        write_json(workspace / "reports" / "smoke_verification.json", report)
        for script, arguments in (
            ("verify_commands.py", ["--workspace", str(workspace)]),
            ("inspect_logs.py", ["--suite", str(workspace / "reports/smoke_verification.json"),
                                 "--output", str(workspace / "reports/chain_diagnostics.json")]),
            ("build_verification_index.py", ["--workspace", str(workspace)]),
        ):
            subprocess.run([sys.executable, str(root / "scripts/validation" / script), *arguments],
                           cwd=root, check=True, timeout=600)
        print(f"VERIFIED {suite.path}", flush=True)
        diagnostics = json.loads((workspace / "reports/chain_diagnostics.json").read_text(encoding="utf-8"))
        artifacts = json.loads((workspace / "reports/artifact_index.json").read_text(encoding="utf-8"))
        commands = json.loads((workspace / "reports/cli_verification.json").read_text(encoding="utf-8"))
        return {"scope": report["scope"], "source": report["source"],
                "unit_test_exit_code": test_exit_code, "unit_tests_skipped": not run_tests,
                "measured_wall_seconds": time.perf_counter() - started,
                "cases": [{key: value for key, value in case.items() if key not in ("run_id", "metrics")}
                          for case in reports],
                "resume_additional_slots": report["resume_additional_slots"],
                "diagnostics": {key: value for key, value in diagnostics.items() if key != "runs"},
                "cli_checks": len(commands["checks"]),
                "actor_update_checks": artifacts["actor_update_checks"],
                "all_listed_assets_checked": artifacts["all_listed_assets_checked"],
                "paper_sha256": artifacts["paper_sha256"],
                "raw_artifact_retention": "Temporary unless --keep-artifacts explicitly supplied"}
    except BaseException as error:
        suite.write_json("partial_cases.json", reports)
        suite.finish(error)
        raise


if __name__ == "__main__":
    main()
