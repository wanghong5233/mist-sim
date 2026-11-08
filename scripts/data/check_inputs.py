"""Validate the complete conditional input without optimizing any policy."""
import json
import argparse
from pathlib import Path

from mist_sim.config import METHODS, for_method, load_config
from mist_sim.anonymity import generate_trace
from mist_sim.evaluation import evaluate, replay
from mist_sim.diagnostics import check_physics
from mist_sim.storage import RunStore, write_json, validation_workspace


class InputCheckController:
    def slow(self, observation, deterministic=False):
        return [1] * len(observation["config"]["servers"]), None

    def fast(self, problem, deterministic=False):
        usage = problem.empty_usage()
        actions = []
        for options in problem.choices:
            legal = problem.mask(options, usage)
            selected = next((index for index, choice in enumerate(options)
                             if legal[index] and choice["kind"] == "edge"), None)
            if selected is None:
                selected = next((index for index, choice in enumerate(options)
                                 if legal[index] and choice["kind"] == "local"), 0)
            actions.append(selected)
            problem.reserve(options[selected], usage)
        return actions, None, None


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-artifacts", type=Path, help="Explicit new debug directory; default cleans up after checking")
    args = parser.parse_args()
    with validation_workspace(args.keep_artifacts) as workspace:
        check_inputs(root, workspace)
    print("Input check finished; " + (f"debug artifacts: {workspace}" if args.keep_artifacts else "temporary artifacts removed"))


def check_inputs(root, workspace):
    config = load_config(root / "configs/conditional.json")
    for method in METHODS:
        for_method(config, method)
    seed = 4100017
    store = RunStore(root, config, "scripts/data/check_inputs.py; no training", seed, output_dir=workspace / "logs/runs")
    store.manifest["budget"] = "One fixed 128-slot input check; no policy optimization"
    try:
        trace = generate_trace(config, seed)
        if trace != generate_trace(config, seed):
            raise AssertionError("Trace not deterministic")
        store.write_json("traces/input.json", trace)
        checking = for_method(config, "greedy_consolidation")
        controller = InputCheckController()
        records, metrics = evaluate(controller, checking, trace)
        check_physics(checking, records)
        if replay(checking, records) != metrics:
            raise AssertionError("Replay metrics differ")
        store.write_json("traces/evaluation.json", records)
        store.write_json("summary.json", metrics)
        store.finish()
        write_json(workspace / "reports/input_readiness.json", {
            "run_id": store.manifest["run_id"], "run_path": store.path.relative_to(workspace).as_posix(), "source": store.manifest["source"],
            "config": "configs/conditional.json", "all_methods_load": list(METHODS),
            "deterministic_generation": True, "physics_and_replay_passed": True,
            "input_ready_for_conditional_simulation": True,
            "hardware_calibrated": False, "policy_training_started": False,
            "scope": "224 FP32 batch1 DNN structural data plus explicit conditional system assumptions",
            "metrics_are_pipeline_checks_not_performance_evidence": True,
            "metrics": metrics})
        print(json.dumps({"run_id": store.path.name, "input_ready": True, "trained": False}))
    except BaseException as error:
        store.finish(error)
        raise


if __name__ == "__main__":
    main()
