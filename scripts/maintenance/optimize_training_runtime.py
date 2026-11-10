"""Audited 2026-11-10 runtime-only migration; never relax Trainer.load checks.

Only the reviewed export switch and equivalent greedy allocation cache are
accepted. Learning state, RNG, input streams and original files are preserved.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from mist_sim.config import validate
from mist_sim.storage import digest_file, write_json
from mist_sim.training import Trainer, implementation_signature, signature


BASE = "2791fcb0fa8ea92461091ab33e3da2e9be85a75e"
OLD_IMPLEMENTATION = "a4ff7d4481cf0042a97317a36dc3a45904ac76c19882e85b48a9e4b15fb8d6a6"
REVIEWED = {
    "config.py": "c978d3b394b066b44b870102f71c775337493a7819ebf082dee31a1ebbc12233",
    "training.py": "70495a897c0b35d52a1f70b6a4d42ab192fb184f00439ec930bb0d1421fbeaba",
    "training_joint.py": "344096d8aab01d63491aea7f0c23e74ec115e6ad46c1eb6abab778a486663ea8",
    "policies/baselines/heuristics.py": "178c89ca5396abde74fe73488bbfab095dd5c5b38ffb5b202fc6f3c9c729e631",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reviewed_source(root):
    paths = sorted(path.relative_to(root).as_posix() for path in (root / "src/mist_sim").rglob("*.py"))
    original_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, "src/mist_sim"], cwd=root).decode().splitlines()
    require(paths == sorted(name for name in original_paths if name.endswith(".py")), "Source file set changed")
    old_digest = hashlib.sha256()
    changed, originals = {}, {}
    for name in paths:
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=root).replace(b"\r\n", b"\n")
        relative = name.removeprefix("src/mist_sim/")
        old_digest.update(relative.encode())
        old_digest.update(original)
        current = (root / name).read_bytes().replace(b"\r\n", b"\n")
        if current != original:
            changed[relative] = hashlib.sha256(current).hexdigest()
        originals[name] = original
    require(old_digest.hexdigest() == OLD_IMPLEMENTATION, "Original implementation mismatch")
    require(changed == REVIEWED, "Source differs from the reviewed runtime optimization")
    originals["pyproject.toml"] = subprocess.check_output(["git", "show", f"{BASE}:pyproject.toml"], cwd=root).replace(b"\r\n", b"\n")
    require((root / "pyproject.toml").read_bytes().replace(b"\r\n", b"\n") == originals["pyproject.toml"], "Package configuration changed")
    return originals


def check_original_hashes(originals, recorded):
    require(set(originals) == set(recorded), "Selection source file set changed")
    for name, contents in originals.items():
        variants = (contents, contents.replace(b"\n", b"\r\n"))
        require(recorded[name] in {hashlib.sha256(value).hexdigest() for value in variants},
                f"Selection was not executed with the reviewed original source: {name}")


def verify_selection_compatibility(root, recorded, current, receipt):
    require(receipt is not None, "Simulator source changed after cut selection")
    proof = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if proof.get("kind") == "reviewed_watchdog_recovery":
        from scripts.maintenance.recover_watchdog import verify_selection_compatibility as verify
        return verify(root, recorded, current, receipt)
    require(proof["kind"] == "reviewed_runtime_optimization", "Invalid compatibility receipt")
    check_original_hashes(reviewed_source(root), recorded)
    require(proof["original_source"] == recorded and proof["current_source"] == current,
            "Compatibility receipt/source mismatch")
    require(proof["implementation_signature"] == implementation_signature(), "Stale compatibility receipt")


def migrate_payload(payload):
    require(payload["schema"] == 2 and payload.get("resumable"), "Completed-stage checkpoint required")
    require(payload["implementation_signature"] == OLD_IMPLEMENTATION, "Unreviewed checkpoint source")
    require(payload["signature"] == signature(payload["config"]), "Checkpoint configuration hash mismatch")
    config = copy.deepcopy(payload["config"])
    require(config["method"] in ("single_timescale", "upper_ppo_greedy"), "Unreviewed training method")
    require(config["runtime"]["collection_workers"] == 2 and config["diagnostics"]["phase_interval"] == 10,
            "Unexpected runtime settings")
    require(config["training"]["cycles"] == 300 and config["convergence"]["min_cycles"] == 300,
            "Expected fixed 300-cycle training")
    config["runtime"]["collection_workers"] = 4
    config["diagnostics"]["phase_interval"] = 0
    validate(config)
    # A shallow copy deliberately preserves every tensor/state object unchanged.
    result = dict(payload, config=config, signature=signature(config),
                  implementation_signature=implementation_signature())
    return result


def migrate(root, run, output):
    reviewed_source(root)
    directory = root / "logs/runs" / run
    checkpoint = directory / "checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = migrate_payload(payload)
    cutoff = len(payload["cycle_history"])
    require(cutoff > 0 and payload["completed_stages"][-1].startswith(f"cycle_{cutoff - 1:03d}_"),
            "Checkpoint must finish a full iteration")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "resume.pt"
    torch.save(result, target)
    # Loading through the strict public API verifies networks, Adam and scales.
    restored = Trainer.load(target)
    require(restored.total_slots == payload["total_slots"], "Recovery slot count changed")
    restored.close()
    original_csv = root / "results" / run / "cycles.csv"
    with original_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    prefix = [row for row in rows if int(row["cycle_index"]) <= cutoff]
    require([int(row["cycle_index"]) for row in prefix] == list(range(1, cutoff + 1)), "Incomplete original curve prefix")
    with (output / "cycles_prefix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prefix)
    write_json(output / "resolved_config.json", result["config"])
    proof = dict(kind="reviewed_runtime_optimization", original_run=run,
        original_checkpoint=str(checkpoint), original_checkpoint_sha256=digest_file(checkpoint),
        checkpoint=str(target), checkpoint_sha256=digest_file(target), resumed_cycles=cutoff,
        original_csv=str(original_csv), original_csv_sha256=digest_file(original_csv),
        prefix_csv_sha256=digest_file(output / "cycles_prefix.csv"),
        superseded_completed_cycles=len(rows) - cutoff,
        original_implementation_signature=OLD_IMPLEMENTATION,
        implementation_signature=implementation_signature(), reviewed_changes=REVIEWED,
        runtime_changes={"runtime.collection_workers": [2, 4], "diagnostics.phase_interval": [10, 0]},
        learning_state_unchanged=True)
    write_json(output / "provenance.json", proof)
    return proof


def combine_curve(root, recovery, continuation):
    """Derived full curve, excluding the superseded tail; raw segments survive."""
    require(Path(continuation).name == continuation, "Invalid continuation run")
    proof = json.loads((recovery / "provenance.json").read_text(encoding="utf-8"))
    log = root / "logs/runs" / continuation
    manifest = json.loads((log / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "completed" and manifest["exit_code"] == 0, "Continuation did not complete")
    require(manifest["checkpoint"]["sha256"] == proof["checkpoint_sha256"], "Different recovery checkpoint")
    require(digest_file(recovery / "resume.pt") == proof["checkpoint_sha256"], "Recovery checkpoint changed")
    require(digest_file(recovery / "cycles_prefix.csv") == proof["prefix_csv_sha256"], "Curve prefix changed")
    require(digest_file(Path(proof["original_csv"])) == proof["original_csv_sha256"], "Original curve changed")
    for source in proof.get("sources", []):
        require(digest_file(Path(source["path"])) == source["sha256"], "Earlier curve source changed")
    sources = [recovery / "cycles_prefix.csv", root / "results" / continuation / "cycles.csv"]
    rows, fields = [], None
    for path in sources:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            require(fields is None or fields == reader.fieldnames, "CSV schema changed")
            fields = reader.fieldnames
            rows.extend(reader)
    require([int(row["cycle_index"]) for row in rows] == list(range(1, 301)), "Expected 300 contiguous iterations")
    destination = root / "results" / continuation / "combined"
    destination.mkdir(exist_ok=False)
    with (destination / "cycles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "provenance.json", dict(
        derivation="checkpoint-prefix plus continuation; superseded original tail excluded",
        sources=[dict(path=str(path), sha256=digest_file(path)) for path in sources],
        migration=str(recovery / "provenance.json"), original_run=proof["original_run"],
        continuation=continuation, resumed_cycles=proof["resumed_cycles"],
        superseded_completed_cycles=proof["superseded_completed_cycles"],
        cycles_csv_sha256=digest_file(destination / "cycles.csv")))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-run")
    parser.add_argument("--merge-run")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.merge_run:
        print(str(combine_curve(root, args.output.resolve(), args.merge_run)))
        return
    require(args.run is not None, "--run is required for migration")
    require(Path(args.run).name == args.run, "Invalid run identifier")
    proof = migrate(root, args.run, args.output.resolve())
    if args.selection_run:
        require(Path(args.selection_run).name == args.selection_run, "Invalid selection identifier")
        manifest = json.loads((root / "logs/runs" / args.selection_run / "manifest.json").read_text(encoding="utf-8"))
        recorded = {name.replace("\\", "/"): value for name, value in manifest["source"]["files"].items()
                    if name.replace("\\", "/").startswith("src/") or name == "pyproject.toml"}
        check_original_hashes(reviewed_source(root), recorded)
        current = {name: digest_file(root / name) for name in recorded}
        write_json(args.output / "selection_compatibility.json", dict(
            kind="reviewed_runtime_optimization", original_source=recorded, current_source=current,
            implementation_signature=implementation_signature(), reviewed_changes=REVIEWED))
    print(json.dumps(proof, ensure_ascii=True))


if __name__ == "__main__":
    main()
