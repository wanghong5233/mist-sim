"""No-update inspection of historical rewards, network inputs and feasible actions."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from mist_sim.config import load_config
from mist_sim.diagnostics import check_physics
from mist_sim.environment import Environment
from mist_sim.evaluation import summarize
from mist_sim.policies.networks import SetPolicy, ITEM_DIM, SERVER_DIM, CONTEXT_DIM, PREFIX_DIM
from mist_sim.rewards import training_reward
from mist_sim.storage import digest_file, write_json
from mist_sim.training import Trainer


def statistics(values):
    values = np.asarray(values, dtype=float)
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()),
                negative=int((values < 0).sum()), positive=int((values > 0).sum()), count=len(values))


def inspect(config, run):
    started = time.perf_counter()
    checkpoint_path = run / "checkpoints" / "initial.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    units = dict(scale=1 / 128, shift=0., clip=None, mode="historical_fixed_units_7ebbe68")
    stages = {}
    for stage, layer in (("pretrain", "lower"), ("cycle_000_upper", "upper"), ("cycle_000_lower", "lower")):
        samples = json.loads((run / "traces" / (stage + "_behavior.json")).read_text(encoding="utf-8"))
        audit = json.loads((run / "traces" / (stage + "_update.json")).read_text(encoding="utf-8"))
        records = json.loads((run / "traces" / (stage + ".json")).read_text(encoding="utf-8"))
        raw_rewards = [record["reward"] for record in records]
        scaled = [training_reward(record, units) for record in records]
        state = payload["controller"][layer]
        feature_ranges = {}
        for component, rows in (
            ("context", [sample["frame"]["context"] for sample in samples]),
            ("server", [row for sample in samples for row in sample["frame"]["servers"]]),
            ("item", [row for sample in samples for row in sample["frame"]["items"]]),
            ("prefix", [step["prefix"] for sample in samples for step in sample["steps"]])):
            mean, scale = state[component + "_norm.mean"].numpy(), state[component + "_norm.scale"].numpy()
            normalized = (np.asarray(rows) - mean) / scale
            feature_ranges[component] = dict(max_abs_by_feature=np.max(np.abs(normalized), axis=0).tolist(),
                                            fraction_abs_over_10=float(np.mean(np.abs(normalized) > 10)))
        stages[stage] = dict(physical_slots=len(records), buffer_samples=len(samples),
            raw_slot_rewards=statistics(raw_rewards), scaled_slot_rewards=statistics(scaled),
            raw_interval_rewards=statistics([sample["reward"] for sample in samples]),
            old_critic_targets=statistics(audit["returns"]),
            scale_only_historical_targets_excludes_shift=statistics(np.asarray(audit["returns"]) * units["scale"]),
            normalized_advantages=statistics(audit["normalized_advantages"]),
            feature_ranges=feature_ranges)
    network = SetPolicy(config["ppo"]["hidden"])
    groups = {"actor": 0, "critic": 0}
    for name, parameter in network.named_parameters():
        groups["critic" if name.startswith(("critic_", "value_head")) else "actor"] += parameter.numel()
    trace_records = json.loads((run / "traces" / "pretrain.json").read_text(encoding="utf-8"))
    small = copy.deepcopy(config)
    small["horizon"] = 16
    trace = [record["raw"] for record in trace_records[:16]]
    comparisons = {}
    for name, active, serve in (("sleep_reject", 0, False), ("awake_reject", 1, False),
                                ("sleep_first_legal", 0, True), ("awake_first_legal", 1, True)):
        environment = Environment(small, trace)
        for _ in trace:
            problem = environment.begin_slot([active] * len(config["servers"]))
            usage, actions = problem.empty_usage(), []
            for options in problem.choices:
                valid = problem.mask(options, usage)
                action = next((index for index in range(1, len(options)) if valid[index]), 0) if serve else 0
                actions.append(action)
                problem.reserve(options[action], usage)
            record = environment.execute(actions)
            training_reward(record, units)
        check_physics(small, environment.records)
        comparisons[name] = dict(metrics=summarize(environment.records),
            raw_return=sum(record["reward"] for record in environment.records),
            scaled_return=sum(training_reward(record, units) for record in environment.records),
            raw_slot_rewards=statistics([record["reward"] for record in environment.records]),
            ready_sequence=[record["ready"][0] for record in environment.records[:4]])
    assert comparisons["awake_first_legal"]["metrics"]["completed"] > 0
    assert comparisons["awake_first_legal"]["raw_return"] > comparisons["awake_reject"]["raw_return"]
    for row in comparisons.values():
        assert np.isclose(row["scaled_return"], row["raw_return"] * units["scale"] + len(trace) * units["shift"])
    return dict(no_optimizer_steps=True, config_reward_units=units, stages=stages,
        historical_features="Read with the historical checkpoint's own normalization statistics; not current bounded inputs",
        network=dict(input_dimensions=dict(candidate=ITEM_DIM, server=SERVER_DIM, context=CONTEXT_DIM, prefix=PREFIX_DIM),
                     hidden=config["ppo"]["hidden"], parameters_per_layer=groups,
                     two_policy_total_parameters=2 * sum(groups.values())),
        same_trace_witnesses=comparisons,
        witness_scope="First 16 chronological slots of old training input, deterministic legal actions; not an optimized policy or general capacity claim",
        source_checkpoint_sha256=digest_file(checkpoint_path),
        measured_seconds=time.perf_counter() - started)


def check_calibration(config, run):
    trainer = Trainer(config, config["seeds"]["train"])
    before = {layer: {name: parameter.detach().clone() for name, parameter in getattr(trainer.controller, layer).named_parameters()}
              for layer in ("upper", "lower")}
    trainer.calibrate()
    result = {}
    for stage, layer in (("cycle_000_upper", "upper"), ("cycle_000_lower", "lower")):
        samples = json.loads((run / "traces" / (stage + "_behavior.json")).read_text(encoding="utf-8"))
        policy = getattr(trainer.controller, layer)
        completed = torch.tensor([sample["frame"]["context"][4] for sample in samples])
        standardized = (completed - policy.context_norm.mean[4]) / policy.context_norm.scale[4]
        normalized = standardized / (1 + standardized.abs())
        result[layer] = dict(history_completed_mean=float(policy.context_norm.mean[4]),
                            history_completed_scale=float(policy.context_norm.scale[4]),
                            historical_completed_max_abs=float(normalized.abs().max()))
        for name, parameter in policy.named_parameters():
            assert torch.equal(parameter, before[layer][name])
    return dict(calibration_physical_slots=config["horizon"] * config["training"]["calibration_episodes"],
                optimizer_steps=0, network_weights_unchanged=True, comparison=result)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/hpgi-development-cost-probe-seed4210002"))
    args = parser.parse_args()
    config = load_config(Path("configs/conditional.json"))
    report = inspect(config, args.run)
    started = time.perf_counter()
    report["corrected_calibration"] = check_calibration(config, args.run)
    report["corrected_calibration_seconds"] = time.perf_counter() - started
    write_json(Path("results/diagnostics/training_design_check.json"), report)
    print(json.dumps(dict(reward_units=report["config_reward_units"], network=report["network"],
                         witnesses=report["same_trace_witnesses"], seconds=report["measured_seconds"]), indent=2))


if __name__ == "__main__":
    main()
