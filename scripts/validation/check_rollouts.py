"""Inspect full-size on-policy rollouts without optimizing or retaining trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from mist_sim.config import load_config, layer_settings
from mist_sim.diagnostics import check_physics, close, policy_input_diagnostics, rollout_diagnostics
from mist_sim.storage import digest_file, write_json
from mist_sim.training import Trainer, implementation_signature


def verify_returns(buffer, settings):
    samples = buffer.samples
    gamma, gae_lambda = settings["gamma"], settings["gae_lambda"]
    advantages, _ = buffer.advantages(gamma, gae_lambda)
    for start, sample in enumerate(samples):
        expected, weight = 0.0, 1.0
        for following in samples[start:]:
            delta = following["reward"] - following["value"]
            if not following["terminated"]:
                delta += gamma ** following["duration"] * following["next_value"]
            expected += weight * delta
            if following["terminated"] or following["truncated"]:
                break
            weight *= gamma ** following["duration"] * gae_lambda
        close(advantages[start], expected, "independent forward GAE")
        physical = np.asarray(sample["physical_rewards"]) * sample["reward_scale"]
        if sample["reward_clip"] is not None:
            physical = np.clip(physical, -sample["reward_clip"], sample["reward_clip"])
        close(sample["reward"], sum(gamma ** offset * reward for offset, reward in enumerate(physical)),
              "discounted interval reward")
        if sample["terminated"]:
            assert sample["next_value"] == 0
        elif start + 1 < len(samples):
            following = samples[start + 1]
            assert sample["episode_id"] == following["episode_id"]
            assert sample["slot"] + sample["duration"] == following["slot"]
            assert sample["next_observation"] == following["frame"]
            close(sample["next_value"], following["value"], "successor value")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/conditional.json")
    parser.add_argument("--summary", type=Path, help="Optional diagnostic output; default prints only")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    report = dict(scope="Sampling and arithmetic only; no optimizer steps or performance claims",
                  implementation_signature=implementation_signature(), config=config,
                  input_hashes={str(path.relative_to(root)): digest_file(path)
                                for path in sorted((root / "data/processed").rglob("*.json"))},
                  layers={})
    trainer = Trainer(config, config["seeds"]["train"])
    trainer.calibrate()
    frozen_statistics = trainer.reward_normalizer.state_dict()
    for layer, count in (("lower", config["training"]["fast_slots"]),
                         ("upper", config["training"]["slow_intervals"])):
        layer_started = time.perf_counter()
        buffer, records = trainer.collect(layer, count, pretrain=layer == "lower")
        verify_returns(buffer, layer_settings(config, layer))
        physics = check_physics(config, records)
        with torch.no_grad():
            error = max(abs(float(getattr(trainer.controller, layer).replay(sample)[0]) - sample["logprob"])
                        for sample in buffer.samples)
        assert error < 1e-5
        assert trainer.reward_normalizer.state_dict() == frozen_statistics
        summary = rollout_diagnostics(buffer, layer_settings(config, layer))
        summary.update(measured_seconds=time.perf_counter() - layer_started,
                       behavior_logprob_error=error, physics=physics,
                       independent_forward_gae_passed=True, successor_observations_equal=True,
                       input_diagnostics=policy_input_diagnostics(getattr(trainer.controller, layer), buffer.samples))
        report["layers"][layer] = summary
        print(json.dumps({"layer": layer, **{key: value for key, value in summary.items()
                                           if key != "input_diagnostics"}}, ensure_ascii=False), flush=True)
        buffer.clear()
        del records
    report.update(measured_wall_seconds=time.perf_counter() - started, optimizer_steps=0,
                  calibration_slots=config["training"]["calibration_episodes"] * config["horizon"],
                  collected_physical_slots=trainer.total_slots, retained_weights_or_raw_trajectories=False)
    if args.summary:
        write_json(args.summary, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("config", "layers", "input_hashes")}))


if __name__ == "__main__":
    main()
