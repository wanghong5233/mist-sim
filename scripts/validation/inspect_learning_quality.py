"""Explain saved reward, avoidable actions and value bias without running training."""
import argparse
import json
from pathlib import Path

from mist_sim.learning_analysis import clock_baseline_probe, public_action_quality
from mist_sim.storage import digest_file
from mist_sim.traces import read_behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--stage", required=True, help="Saved detailed stage, e.g. cycle_022_joint")
    args = parser.parse_args()
    phase = args.run / "traces" / "phases" / args.stage
    config = json.loads((args.run / "resolved_config.json").read_text(encoding="utf-8"))
    paths = {name: phase / (name + ".json") for name in ("physical", "behavior")}
    records = json.loads(paths["physical"].read_text(encoding="utf-8"))
    samples = read_behavior(paths["behavior"])
    result = dict(run=str(args.run.resolve()), stage=args.stage,
        sha256={name: digest_file(path) for name, path in paths.items()},
        public_actions=public_action_quality(config, records),
        clock_baseline=clock_baseline_probe(samples, config["horizon"], config["ppo"]))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
