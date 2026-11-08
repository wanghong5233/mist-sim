"""Bounded real-profile RL correctness check; no performance or convergence claim."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
from mist_sim.anonymity import generate_trace
from mist_sim.config import load_config
from mist_sim.diagnostics import inspect_run
from mist_sim.evaluation import evaluate, replay, assert_close
from mist_sim.storage import RunStore, digest_file, validation_workspace, write_json
from mist_sim.training import Trainer
from mist_sim.verification import independent_totals


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/conditional.json")
    parser.add_argument("--summary", type=Path, default=root / "results/profile_chain_check.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["method"] != "hpgi":
        raise ValueError("This bounded check targets HPGI")
    config["training"].update(pretrain_slots=16, fast_slots=16, slow_intervals=4, cycles=1)
    config["ppo"]["epochs"] = 1
    config["ppo_layers"] = {"upper": {}, "lower": {}}
    config["validation"]["trajectories"] = 1
    started = time.perf_counter()
    with validation_workspace() as workspace:
        store = RunStore(root, config, "check_profile_chain correctness", config["seeds"]["train"],
                         output_dir=workspace / "runs")
        trainer = Trainer(config, config["seeds"]["train"], store)
        try:
            trainer.run(max_seconds=180)
            trained_slots = trainer.total_slots
            trace = generate_trace(config, config["seeds"]["validation"])
            records, metrics = evaluate(trainer.controller, config, trace)
            assert_close(metrics, replay(config, records))
            for key, value in independent_totals(config, records).items():
                assert_close(value, metrics[key], key)
            checkpoint = store.path / "checkpoints/final.pt"
            trainer.phase("resume_check", "upper", 2)
            continued = copy.deepcopy(trainer.controller.state_dict())
            restored = Trainer.load(checkpoint)
            restored_records, restored_metrics = evaluate(restored.controller, config, trace)
            assert_close(metrics, restored_metrics)
            ignored = ("measured_policy_seconds", "measured_allocation_seconds")
            assert_close([{key: value for key, value in record.items() if key not in ignored} for record in records],
                         [{key: value for key, value in record.items() if key not in ignored} for record in restored_records])
            restored.phase("resume_check", "upper", 2)
            for layer in ("upper", "lower"):
                for name, value in continued[layer].items():
                    torch.testing.assert_close(value, restored.controller.state_dict()[layer][name], rtol=0, atol=0)
            for layer in ("upper", "lower"):
                original_state = trainer.optimizers[layer].state_dict()
                restored_state = restored.optimizers[layer].state_dict()
                assert original_state["param_groups"] == restored_state["param_groups"]
                for parameter, fields in original_state["state"].items():
                    for name, value in fields.items():
                        torch.testing.assert_close(value, restored_state["state"][parameter][name], rtol=0, atol=0)
            diagnostics = inspect_run(store.path)
            store.finish()
            for name, expected in store.manifest["artifacts"].items():
                assert digest_file(store.path / name) == expected
            fields = ("stage", "layer", "samples", "training_reward_range", "return_range", "advantage_range",
                      "normalized_advantage", "reward_units", "reward_clipped_slots",
                      "training_reward_signs", "physical_cost_components",
                      "actor_before", "actor_after", "critic_before", "critic_after", "parameter_delta_l2",
                      "behavior_logprob_error", "frozen_unchanged", "post_update_joint_kl", "input_diagnostics",
                      "arrivals", "completed", "on_time_rate", "conservative_cost")
            report = dict(scope="Real DNN profiles with conditional system parameters; correctness only",
                source=store.manifest["source"], config=config,
                measured_wall_seconds=time.perf_counter() - started,
                calibration_slots=config["training"]["calibration_episodes"] * config["horizon"],
                training_slots=trained_slots, resume_slots=trainer.total_slots - trained_slots,
                repeated_resume_slots=restored.total_slots - trained_slots,
                deterministic_reload_equal=True, resumed_parameters_and_optimizers_equal=True,
                replay_and_independent_totals_equal=True, manifest_hashes_passed=True,
                update_samples=sum(item["samples"] for item in diagnostics["updates"]),
                minibatches=sum(item["minibatches"] for item in diagnostics["updates"]),
                physical_records=sum(item["slots"] for item in diagnostics["physics"]),
                edge_assignments_checked=sum(item["edge_assignments"] for item in diagnostics["physics"]),
                stages=[{key: row[key] for key in fields} for row in trainer.history],
                validation=[{key: row[key] for key in ("stage", "total_slots", "metrics", "service_threshold")}
                            for row in trainer.validation_history],
                final_deterministic_metrics=metrics,
                raw_artifact_retention="Disposable temporary directory; no retained weights or traces")
        except BaseException as error:
            store.finish(error)
            write_json(args.summary, dict(status="failed", error=repr(error), source=store.manifest["source"],
                                         measured_wall_seconds=time.perf_counter() - started))
            raise
    write_json(args.summary, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("source", "config", "stages")}, indent=2))


if __name__ == "__main__":
    main()
