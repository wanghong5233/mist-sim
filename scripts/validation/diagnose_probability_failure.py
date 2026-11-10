"""Reproduce a pre-update failure using its recorded, matching source revision.

Run before patching source (or in that exact source checkout). Reproduction never
updates weights and refuses mismatched source; the saved evidence can then be
checked by a separately audited recovery tool after applying the fix.
"""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from mist_sim.training import Trainer, implementation_signature
from mist_sim.collection import EpisodeCollectors
from mist_sim.policies.batching import replay_chunks
from mist_sim.storage import write_json, digest_file
from mist_sim.traces import pack_behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    path = run / "checkpoints/failed_last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["implementation_signature"] == implementation_signature()
    events = [json.loads(line) for line in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    last_start = max(i for i, event in enumerate(events) if event["event"] == "phase_started")
    tail = events[last_start:]
    assert tail[-1]["event"] == "rollout_ready" and not any(event["event"] == "update_started" for event in tail)
    assert "Behavior probability changed before update" in (run / "stderr.log").read_text(encoding="utf-8")
    # A temporary diagnostic copy only; never publish it as a resumable checkpoint.
    with tempfile.TemporaryDirectory(prefix="mist-probability-") as directory:
        payload["resumable"] = True
        diagnostic = Path(directory) / "diagnostic.pt"
        torch.save(payload, diagnostic)
        trainer = Trainer.load(diagnostic)
    count = trainer.config["training"]["fast_slots"]
    horizon = trainer.config["horizon"]
    jobs = [(index, horizon) for index in range(trainer.trace_counter - count // horizon, trainer.trace_counter)]
    pool = EpisodeCollectors(trainer.config, trainer.seed, 4)
    try:
        outputs = pool.collect(trainer, jobs, "lower", False)
    finally:
        pool.close()
    samples = [sample for batch, _ in outputs for sample in batch]
    policy = trainer.controller.lower
    with torch.no_grad():
        batch = replay_chunks(policy, samples, trainer.config["ppo"]["minibatch"])
    old = torch.tensor([sample["logprob"] for sample in samples])
    differences = (batch - old).abs()
    indices = torch.argsort(differences, descending=True)[:8].tolist()
    worst = []
    for index in indices:
        with torch.no_grad():
            serial = float(policy.replay(samples[index])[0])
        worst.append(dict(index=index, decisions=len(samples[index]["steps"]), old=float(old[index]),
                          batch=float(batch[index]), serial=serial, error=float(differences[index])))
    evidence = run / "failure_diagnosis"
    write_json(evidence / "behavior.json", pack_behavior(samples))
    write_json(evidence / "physical.json", [record for _, records in outputs for record in records])
    report = dict(checkpoint_sha256=digest_file(path), stage=tail[0]["stage"],
                  no_update_started=True, trace_indices=[index for index, _ in jobs],
                  max_error=float(differences.max()), count_over_1e5=int((differences > 1e-5).sum()), worst=worst)
    write_json(evidence / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
