"""Audited migration for one identified pre-update float32 failure, not a general override.

Requires the exact incident checkpoint, original source tree and reviewed fix hashes.
It preserves all learning state, discards the unapplied rollout and records its cost.
Any other failure or future source revision requires a fresh audit.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from mist_sim.policies.hpgi import Controller
from mist_sim.policies.batching import check_behavior_probability
from mist_sim.storage import digest_file, write_json
from mist_sim.traces import read_behavior
from mist_sim.training import Trainer, implementation_signature, signature


BASE = "0d3aea50405de3efaf74416e83f2d35142b1ea41"
CHECKPOINT = "0950277fbc928d5ce23978cd88d0b7fb7b4bcbb67d9556cf748b34c0babfec23"
REVIEWED = {
    "training.py": "ff7ce23e6cf9e8d0594eb77c310c43bd887820bca34d0d045b4b82916a653cc8",
    "training_joint.py": "fe0520f143eb2b88e4bd99b860da7607021562802444925ca77d29c07d1098ae",
    "policies/batching.py": "8b53b800c374c4aa638961f9d3e8328e3c1fae19d3b157357f0ffef090bdce85",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(root, payload):
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "src/mist_sim"], cwd=root).decode().splitlines()
    files = sorted(Path(name) for name in files if name.endswith(".py"))
    current = sorted((root / "src/mist_sim").rglob("*.py"))
    require([path.relative_to(root) for path in current] == files, "Source file set changed")
    digest = hashlib.sha256()
    changed = {}
    for path in files:
        relative = path.relative_to("src/mist_sim").as_posix()
        original = subprocess.check_output(["git", "show", f"{BASE}:{path.as_posix()}"], cwd=root).replace(b"\r\n", b"\n")
        digest.update(relative.encode())
        digest.update(original)
        now = (root / path).read_bytes().replace(b"\r\n", b"\n")
        if now != original:
            changed[relative] = hashlib.sha256(now).hexdigest()
    require(digest.hexdigest() == payload["implementation_signature"], "Original source signature mismatch")
    require(changed == REVIEWED, "Source edits differ from the reviewed probability-only patch")
    return changed


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(item) for item in value)
    return True


def merge_continuation(root, run, recovery, continuation, restart=None):
    """Create a derived plotting CSV; original segment files are never modified."""
    require(Path(continuation).name == continuation, "Invalid continuation run id")
    provenance = json.loads((recovery / "provenance.json").read_text(encoding="utf-8"))
    original = root / "results" / run.name / "cycles.csv"
    require(str(original) == provenance["original_csv"] and
            digest_file(original) == provenance["original_csv_sha256"], "Original curve changed")
    log = root / "logs/runs" / continuation
    manifest = json.loads((log / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["checkpoint"]["sha256"] == provenance["checkpoint_sha256"], "Different continuation checkpoint")
    require(manifest["seed"] == 4200001, "Different training seed")
    config = json.loads((log / "resolved_config.json").read_text(encoding="utf-8"))
    expected = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    expected["checkpoint_policy"]["cycle_interval"] = 10
    require(config == expected, "Learning configuration changed")
    sources = [original, root / "results" / continuation / "cycles.csv"]
    cutoff, rollback_slots, restart_identity = None, 0, None
    active_run = continuation
    if restart:
        require(Path(restart).name == restart, "Invalid restart run id")
        restart_log = root / "logs/runs" / restart
        restarted = json.loads((restart_log / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = log / "checkpoints/last.pt"
        require(Path(restarted["checkpoint"]["path"]).resolve() == checkpoint.resolve(), "Different restart parent")
        require(restarted["checkpoint"]["sha256"] == digest_file(checkpoint), "Restart checkpoint changed")
        require(restarted["seed"] == manifest["seed"], "Restart seed changed")
        restarted_config = json.loads((restart_log / "resolved_config.json").read_text(encoding="utf-8"))
        require(restarted_config == config, "Restart learning configuration changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(state["resumable"] and state["config"] == config, "Invalid completed restart checkpoint")
        require(state["implementation_signature"] == implementation_signature(), "Restart source changed")
        cutoff = len(state["cycle_history"])
        require(139 < cutoff < 200, "Unexpected restart boundary")
        events = [json.loads(line) for line in (log / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
        rollback_slots = max(event["total_slots"] for event in events) - state["total_slots"]
        require(rollback_slots >= 0, "Negative restart accounting")
        restart_identity = dict(run=restart, checkpoint=str(checkpoint), sha256=digest_file(checkpoint),
                                retained_through_iteration=cutoff, rollback_slots=rollback_slots)
        sources.append(root / "results" / restart / "cycles.csv")
        active_run, manifest = restart, restarted
    rows, segments, fields = [], [], None
    for path in sources:
        content = path.read_bytes()
        require(content.endswith(b"\n"), "CSV has an incomplete live row; retry after iteration completes")
        import io
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        require(fields is None or fields == reader.fieldnames, "CSV columns differ")
        fields = reader.fieldnames
        part = list(reader)
        require(bool(part) and all(None not in row and None not in row.values() for row in part), "Malformed CSV")
        require(all(int(row["seed"]) == 4200001 for row in part), "Mixed seeds")
        segment = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), rows=len(part),
                       first=int(part[0]["cycle_index"]), last=int(part[-1]["cycle_index"]))
        if cutoff is not None and path == sources[1]:
            segment["excluded_superseded_iterations"] = [int(row["cycle_index"]) for row in part if int(row["cycle_index"]) > cutoff]
            part = [row for row in part if int(row["cycle_index"]) <= cutoff]
        segments.append(segment)
        for row in part:
            rows.append(dict(row, source_run_id=row["run_id"], run_id=active_run))
    require([int(row["cycle_index"]) for row in rows] == list(range(1, len(rows) + 1)), "Iteration gap or overlap")
    require(140 <= len(rows) <= 200, "Unexpected combined curve length")
    slots = [int(row["total_slots"]) for row in rows]
    require(all(b > a for a, b in zip(slots, slots[1:])), "Nonmonotonic physical budget")
    destination = root / "results" / "comparisons" / "always-on-recovered-seed4200001"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "cycles.csv.partial"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["source_run_id"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination / "cycles.csv")
    write_json(destination / "provenance.json", dict(segments=segments, recovery=str(recovery / "provenance.json"),
        rows=len(rows), target=200, continuation_status=manifest["status"],
        discarded_unapplied_slots=2048, discarded_rollback_slots=rollback_slots, restart=restart_identity,
        actual_training_slots_including_retries=slots[-1] + rollback_slots,
        combined_csv_sha256=digest_file(destination / "cycles.csv"),
        identity="Same seed and learning state; run_id unified for plotting, source_run_id retains segment identity"))
    print(json.dumps(dict(combined_csv=str(destination / "cycles.csv"), iterations=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-continuation", help="Refresh a derived curve from this audited recovery's actual CSV segments")
    parser.add_argument("--restart-run", help="Normal checkpoint restart after the first continuation; exclude its superseded tail")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run, output = args.run.resolve(), args.output.resolve()
    if args.merge_continuation:
        return merge_continuation(root, run, output, args.merge_continuation, args.restart_run)
    if args.restart_run:
        parser.error("--restart-run requires --merge-continuation")
    require(not output.exists(), "Recovery destination already exists")
    checkpoint = run / "checkpoints/failed_last.pt"
    require(digest_file(checkpoint) == CHECKPOINT, "Not the audited incident checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_changes = check_source(root, payload)
    require(payload["schema"] == 2 and not payload["resumable"], "Unexpected snapshot status")
    require(payload["config"]["method"] == "always_on" and payload["seed"] == 4200001, "Wrong run")
    require(payload["signature"] == signature(payload["config"]), "Original config signature mismatch")
    require(len(payload["cycle_history"]) == 139 and len(payload["history"]) == 143, "Incomplete history")
    require(payload["versions"] == dict(upper=0, lower=143), "Policy update count mismatch")
    require([r["stage"] for r in payload["cycle_history"]] == [f"cycle_{i:03d}" for i in range(139)], "Cycle gap")
    require(payload["completed_stages"][-1] == "cycle_138_lower", "Unexpected last completed phase")
    events_path = run / "progress.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    start = max(i for i, e in enumerate(events) if e["event"] == "phase_started")
    tail = events[start:]
    require(tail[0]["stage"] == "cycle_139_lower" and tail[-1]["event"] == "rollout_ready", "Wrong failure boundary")
    require(not any(e["event"] in ("update_started", "ppo_epoch", "phase_completed") for e in tail), "Update may have started")
    require("AssertionError: Behavior probability changed before update" in (run / "stderr.log").read_text(encoding="utf-8"), "Wrong failure")
    steps = sum(row["optimizer_steps"] for row in payload["history"])
    adam_steps = {float(state["step"]) for state in payload["optimizers"]["lower"]["state"].values()}
    require(adam_steps == {steps} and steps == 10400, "Adam progressed beyond completed history")
    require(not payload["optimizers"]["upper"]["state"], "Unused upper policy was updated")
    require(finite_state(payload["controller"]) and finite_state(payload["optimizers"]), "Nonfinite learning state")
    result_csv = root / "results" / run.name / "cycles.csv"
    with result_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([int(row["cycle_index"]) for row in rows] == list(range(1, 140)), "Original CSV is not intact")
    require(payload["total_slots"] == 294912 and payload["trace_counter"] == 2306, "Unexpected sampler counters")
    discarded = payload["total_slots"] - payload["history"][-1]["total_slots"]
    require(discarded == 2048, "Unexpected unapplied rollout size")
    evidence = run / "failure_diagnosis"
    report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
    require(report["checkpoint_sha256"] == CHECKPOINT and report["no_update_started"], "Unmatched diagnosis")
    torch.set_num_threads(1)
    policy = Controller(payload["config"])
    policy.load_state_dict(payload["controller"])
    samples = read_behavior(evidence / "behavior.json")
    require(len(samples) == discarded, "Incomplete failure samples")
    checked = check_behavior_probability(policy.lower, samples, payload["config"]["ppo"]["minibatch"])
    require(checked["serial_fallback_rows"] == 1 and checked["serial_max_error"] == 0., "Different numerical failure")
    # Only metadata/config protection changes; all tensors, Adam, RNG, statistics
    # and consumed input counters are preserved. The failed rollout is NOT learned.
    payload["config"]["checkpoint_policy"]["cycle_interval"] = 10
    payload["signature"] = signature(payload["config"])
    payload["implementation_signature"] = implementation_signature()
    payload["resumable"] = True
    output.mkdir(parents=True)
    target = output / "resume.pt"
    temporary = target.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(target)
    loaded = Trainer.load(target)
    require(len(loaded.cycle_history) == 139 and loaded.trace_counter == 2306, "Recovery readback mismatch")
    loaded.close()
    write_json(output / "provenance.json", dict(
        source_checkpoint=str(checkpoint), source_sha256=CHECKPOINT, base_commit=BASE,
        approved_source_changes=source_changes, implementation_signature=payload["implementation_signature"],
        checkpoint=str(target), checkpoint_sha256=digest_file(target),
        original_csv=str(result_csv), original_csv_sha256=digest_file(result_csv),
        events_sha256=digest_file(events_path), diagnosis_sha256=digest_file(evidence / "report.json"),
        completed_cycles=139, next_iteration=140, target_cycles=200, optimizer_steps=steps,
        discarded_unapplied_slots=discarded, total_slots_including_discarded=payload["total_slots"],
        trace_counter=payload["trace_counter"], probability_check=checked,
        changes=["serial-confirmed roundoff guard", "checkpoint cycle interval 250 -> 10", "pre-update snapshot marked resumable after audit"],
        preserved=["weights", "Adam", "reward statistics", "value scales", "RNG", "completed stages", "cycle history", "sampling counters"]
    ))
    print(json.dumps(dict(checkpoint=str(target), completed_cycles=139, next_iteration=140,
                         discarded_slots=discarded, probability_check=checked)), flush=True)


if __name__ == "__main__":
    main()
