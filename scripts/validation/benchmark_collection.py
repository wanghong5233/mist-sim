"""Time real training collection without optimizer updates or extra evaluation."""
import argparse
import copy
import hashlib
import json
import math
import time

import torch

from mist_sim.config import load_config
from mist_sim.training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conditional.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--layer", choices=["upper", "lower"], default="lower")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    original, current = copy.deepcopy(payload["config"]), copy.deepcopy(config)
    for key in ("runtime", "validation", "diagnostics", "convergence", "trace_retention", "checkpoint_policy"):
        original.pop(key, None)
        current.pop(key, None)
    if original != current:
        raise ValueError("Benchmark model/environment configuration differs")
    if arguments.episodes < 1 or any(workers < 1 for workers in arguments.workers):
        raise ValueError("Positive episodes/workers required")
    expected = None
    for workers in arguments.workers:
        config["runtime"]["collection_workers"] = workers
        trainer = Trainer(config, payload["seed"])
        trainer.controller.load_state_dict(payload["controller"])
        trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
        trainer.value_reward_scales = copy.deepcopy(payload["value_reward_scales"])
        trainer.trace_counter = payload["trace_counter"]
        trainer.versions = copy.deepcopy(payload["versions"])
        count = arguments.episodes * (config["horizon"] if arguments.layer == "lower" else math.ceil(config["horizon"] / config["window"]))
        try:
            started = time.perf_counter()
            buffer, records = trainer.collect(arguments.layer, count)
            elapsed = time.perf_counter() - started
            identity = [{key: value for key, value in record.items() if key != "measured_allocation_seconds"} for record in records]
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if expected is not None and digest != expected:
                raise AssertionError("Worker count changed sampled actions or physical outcomes")
            expected = digest
            print(json.dumps(dict(workers=workers, seconds_including_startup=elapsed,
                physical_slots=len(records), transitions=len(buffer.samples), same_sample_hash=digest)), flush=True)
        finally:
            trainer.close()


if __name__ == "__main__":
    main()
