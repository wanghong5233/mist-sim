"""Read-only replay benchmark, no optimizer steps or persistent outputs.

Run from the project root; --behavior accepts a saved behavior.json.
Without private inputs this uses variable-size synthetic frames.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from mist_sim.policies.networks import SetPolicy
from mist_sim.policies.batching import replay_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(71)
    policy = SetPolicy(64)
    if args.behavior:
        samples = json.loads(args.behavior.read_text(encoding="utf-8"))[:args.samples]
    else:
        samples = []
        for i in range(args.samples):
            frame = dict(items=torch.randn(80 + i % 10, 19).tolist(),
                         servers=torch.randn(8, 13).tolist(), context=torch.randn(11).tolist())
            steps = [dict(features=torch.randn(2 + j % 16, 19).tolist(),
                          prefix=torch.randn(7).tolist(), mask=[True] * (2 + j % 16), action=0)
                     for j in range(16 + i % 8)]
            samples.append(dict(frame=frame, steps=steps))
    def run(serial):
        policy.zero_grad(set_to_none=True)
        result = (tuple(torch.stack(v) for v in zip(*(policy.replay(s) for s in samples)))
                  if serial else replay_batch(policy, samples))
        sum(v.sum() for v in result).backward()
        return tuple(v.detach() for v in result)
    timings = {"serial": [], "batch": []}
    old, new = run(True), run(False)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-5)
    for repeat in range(3):
        for serial in ((True, False) if repeat % 2 == 0 else (False, True)):
            start = time.perf_counter()
            run(serial)
            timings["serial" if serial else "batch"].append(time.perf_counter() - start)
    seconds = {key: statistics.median(values) for key, values in timings.items()}
    print(json.dumps(dict(samples=len(samples), threads=args.threads, seconds=seconds,
                         speedup=seconds["serial"] / seconds["batch"],
                         max_errors=[float((a-b).abs().max()) for a,b in zip(old,new)])))


if __name__ == "__main__":
    main()
