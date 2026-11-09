"""Compare persistent collector counts using identical frozen policies and traces."""
import argparse
import copy
import hashlib
import json
import math
import statistics
import time

import torch

from mist_sim.training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--layer", choices=["upper", "lower", "joint"], default="joint")
    args = parser.parse_args()
    if args.repeats < 1 or args.episodes < 1 or any(worker < 1 or worker > 32 for worker in args.workers):
        parser.error("Positive workers/repeats/slots required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    per_episode = (math.ceil(payload["config"]["horizon"] / payload["config"]["window"])
                   if args.layer == "upper" else payload["config"]["horizon"])
    expected = None
    results = []
    for workers in args.workers:
        config = copy.deepcopy(payload["config"])
        config["device"] = "cpu"
        config["runtime"]["collection_workers"] = workers
        trainer = Trainer(config, payload["seed"])
        trainer.controller.load_state_dict(payload["controller"])
        trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
        trainer.value_reward_scales = copy.deepcopy(payload["value_reward_scales"])
        trainer.versions = copy.deepcopy(payload["versions"])
        durations = []
        try:
            for repeat in range(args.repeats + 1):
                trainer.trace_counter = payload["trace_counter"]
                started = time.perf_counter()
                buffer, records = trainer.collect(args.layer, args.episodes * per_episode)
                elapsed = time.perf_counter() - started
                content = [(sample.get("phase", args.layer), sample["actions"], sample["logprob"], sample["reward"])
                           for sample in buffer.samples]
                content.append([{key: value for key, value in record.items() if key != "measured_allocation_seconds"}
                                for record in records])
                digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
                if expected is None:
                    expected = digest
                if expected != digest:
                    raise AssertionError("Worker count changed sampled actions/probabilities/rewards")
                if repeat:
                    durations.append(elapsed)
                print(json.dumps(dict(workers=workers, repeat=repeat, cold=repeat == 0,
                    seconds=elapsed, slots=len(records), events=len(buffer.samples), digest=digest,
                    timing=trainer.collection_timings)), flush=True)
                del buffer, records, content
        finally:
            trainer.close()
        results.append(dict(workers=workers, seconds=durations, median_seconds=statistics.median(durations)))
    print(json.dumps(dict(summary=results, selected=min(results, key=lambda row: row["median_seconds"]),
        checkpoint_role="frozen diagnostic weights, not resumed training")), flush=True)


if __name__ == "__main__":
    main()
