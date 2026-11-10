"""Audited recovery of the 2026-11-11 Windows 0x101 interruption.

Only GPU scheduling/cache lifetime changes; keep background performance enabled.
Learning tensors, optimizer state, input counters and RNG remain untouched.
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
from scripts.maintenance.optimize_training_runtime import require, check_original_hashes


BASE = "a2ce5d0"
OLD_IMPLEMENTATION = "14c701047087c564b11f53049bec8675d71999b687856adf9e77251ced81ca72"
REVIEWED = {"devices.py": "18ded93c3790c5b8da14a9d361b73ae5ba04a9f4f4a085148fe6874f0e61d132"}


def reviewed_source(root):
    names = sorted(path.relative_to(root).as_posix() for path in (root / "src/mist_sim").rglob("*.py"))
    original_names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE, "src/mist_sim"], cwd=root).decode().splitlines()
    require(names == sorted(name for name in original_names if name.endswith(".py")), "Source file set changed")
    digest, changes, originals = hashlib.sha256(), {}, {}
    for name in names:
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=root).replace(b"\r\n", b"\n")
        relative = name.removeprefix("src/mist_sim/")
        digest.update(relative.encode())
        digest.update(original)
        current = (root / name).read_bytes().replace(b"\r\n", b"\n")
        if current != original:
            changes[relative] = hashlib.sha256(current).hexdigest()
        originals[name] = original
    require(digest.hexdigest() == OLD_IMPLEMENTATION, "Original implementation mismatch")
    require(changes == REVIEWED, "Source differs from reviewed GPU scheduling recovery")
    originals["pyproject.toml"] = subprocess.check_output(["git", "show", f"{BASE}:pyproject.toml"], cwd=root).replace(b"\r\n", b"\n")
    require((root / "pyproject.toml").read_bytes().replace(b"\r\n", b"\n") == originals["pyproject.toml"], "Package configuration changed")
    return originals


def migrate_payload(payload):
    require(payload["schema"] == 2 and payload.get("resumable"), "Completed-stage checkpoint required")
    require(payload["implementation_signature"] == OLD_IMPLEMENTATION, "Unreviewed checkpoint source")
    require(payload["signature"] == signature(payload["config"]), "Checkpoint configuration hash mismatch")
    config = copy.deepcopy(payload["config"])
    require(config["method"] in ("single_timescale", "upper_ppo_greedy"), "Unreviewed training method")
    require(config["runtime"]["collection_workers"] == 4 and config["diagnostics"]["phase_interval"] == 0,
            "Unexpected sampling/export settings")
    require(config["runtime"]["windows_performance"] is True, "Unexpected process scheduling")
    require(config["training"]["cycles"] == config["convergence"]["min_cycles"] == 300, "Expected 300-cycle training")
    validate(config)
    return dict(payload, config=config, signature=signature(config), implementation_signature=implementation_signature())


def read_curve(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def curve_prefix(root, run, earlier, cutoff):
    """Retain two actual segments through the new checkpoint, never its tail."""
    previous = json.loads((earlier / "provenance.json").read_text(encoding="utf-8"))
    log = root / "logs/runs" / run
    manifest = json.loads((log / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["checkpoint"]["sha256"] == previous["checkpoint_sha256"], "Wrong earlier continuation")
    frozen = [(earlier / "resume.pt", previous["checkpoint_sha256"]),
              (earlier / "cycles_prefix.csv", previous["prefix_csv_sha256"]),
              (Path(previous["original_csv"]), previous["original_csv_sha256"])]
    for path, digest in frozen:
        require(digest_file(path) == digest, "Earlier recovery source changed")
    fields, prefix = read_curve(earlier / "cycles_prefix.csv")
    following_fields, following = read_curve(root / "results" / run / "cycles.csv")
    require(fields == following_fields, "CSV schema changed")
    start = previous["resumed_cycles"]
    require([int(row["cycle_index"]) for row in following] == list(range(start + 1, start + len(following) + 1)),
            "Non-contiguous interrupted continuation")
    prefix += [row for row in following if int(row["cycle_index"]) <= cutoff]
    require([int(row["cycle_index"]) for row in prefix] == list(range(1, cutoff + 1)), "Incomplete checkpoint curve")
    return fields, prefix, len(following) + start - cutoff


def selection_originals(root):
    # Recheck the earlier runtime-only optimization as well as this new patch.
    from scripts.maintenance.optimize_training_runtime import BASE as first_base, REVIEWED as first_changes
    base = reviewed_source(root)
    older, changes = {}, {}
    for name, contents in base.items():
        original = subprocess.check_output(["git", "show", f"{first_base}:{name}"], cwd=root).replace(b"\r\n", b"\n")
        older[name] = original
        if original != contents:
            changes[name.removeprefix("src/mist_sim/")] = hashlib.sha256(contents).hexdigest()
    require(changes == first_changes, "Earlier runtime optimization changed")
    return older


def verify_selection_compatibility(root, recorded, current, receipt):
    require(receipt is not None, "Selection compatibility receipt required")
    proof = json.loads(Path(receipt).read_text(encoding="utf-8"))
    require(proof["kind"] == "reviewed_watchdog_recovery", "Invalid recovery compatibility receipt")
    check_original_hashes(selection_originals(root), recorded)
    require(proof["original_source"] == recorded and proof["current_source"] == current, "Compatibility source mismatch")
    require(proof["implementation_signature"] == implementation_signature(), "Stale compatibility receipt")


def migrate(root, run, earlier, output, selection_run=None):
    reviewed_source(root)
    directory = root / "logs/runs" / run
    checkpoint = directory / "checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = migrate_payload(payload)
    cutoff = len(payload["cycle_history"])
    require(cutoff > 0 and [row["stage"] for row in payload["cycle_history"]] ==
            [f"cycle_{index:03d}" for index in range(cutoff)], "Full checkpoint iterations required")
    require(payload["completed_stages"][-1].startswith(f"cycle_{cutoff - 1:03d}_"), "Incomplete checkpoint phase")
    fields, prefix, superseded = curve_prefix(root, run, earlier, cutoff)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "resume.pt"
    torch.save(result, target)
    restored = Trainer.load(target)
    require(restored.total_slots == payload["total_slots"], "Recovery slots changed")
    restored.close()
    with (output / "cycles_prefix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prefix)
    write_json(output / "resolved_config.json", result["config"])
    original_csv = root / "results" / run / "cycles.csv"
    sources = [earlier / "provenance.json", earlier / "cycles_prefix.csv", original_csv]
    proof = dict(kind="reviewed_watchdog_recovery", original_run=run,
        original_checkpoint=str(checkpoint), original_checkpoint_sha256=digest_file(checkpoint),
        checkpoint=str(target), checkpoint_sha256=digest_file(target), resumed_cycles=cutoff,
        original_csv=str(original_csv), original_csv_sha256=digest_file(original_csv),
        prefix_csv_sha256=digest_file(output / "cycles_prefix.csv"), superseded_completed_cycles=superseded,
        sources=[dict(path=str(path), sha256=digest_file(path)) for path in sources],
        original_implementation_signature=OLD_IMPLEMENTATION, implementation_signature=implementation_signature(),
        reviewed_changes=REVIEWED, runtime_changes={},
        learning_state_unchanged=True, interruption="Windows bugcheck 0x101; root cause undetermined")
    write_json(output / "provenance.json", proof)
    if selection_run:
        manifest = json.loads((root / "logs/runs" / selection_run / "manifest.json").read_text(encoding="utf-8"))
        recorded = {name.replace("\\", "/"): value for name, value in manifest["source"]["files"].items()
                    if name.replace("\\", "/").startswith("src/") or name == "pyproject.toml"}
        check_original_hashes(selection_originals(root), recorded)
        write_json(output / "selection_compatibility.json", dict(kind="reviewed_watchdog_recovery",
            original_source=recorded, current_source={name: digest_file(root / name) for name in recorded},
            implementation_signature=implementation_signature(), reviewed_changes=REVIEWED))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-run")
    args = parser.parse_args()
    for name in [args.run, args.selection_run]:
        if name is not None:
            require(Path(name).name == name and name not in (".", ".."), "Invalid run identifier")
    print(json.dumps(migrate(args.root.resolve(), args.run, args.earlier.resolve(), args.output.resolve(), args.selection_run)))


if __name__ == "__main__":
    main()
