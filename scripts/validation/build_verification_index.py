"""Verify manifests against local assets and export a portable checksum index."""
import json
import argparse
from pathlib import Path
import sys
import torch
from mist_sim.storage import digest_file, write_json, source_identity


def paper_checksums(directory):
    if directory is None:
        return {}
    return {name: digest_file(Path(directory) / name) for name in ("main.tex", "main.pdf")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, help="Optional local manuscript directory; never uploaded")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    smoke = json.loads((workspace / "results/smoke_verification.json").read_text(encoding="utf-8"))
    commands = json.loads((workspace / "results/cli_verification.json").read_text(encoding="utf-8"))
    identifiers = [smoke["suite_run_id"], smoke["resume_run_id"], commands["run_id"]]
    identifiers += [case["run_id"] for case in smoke["cases"]]
    identifiers += [case["run_id"] for case in commands["checks"]]
    entries = []
    current_files = source_identity(root)["files"]
    for identifier in dict.fromkeys(identifiers):
        directory = workspace / "runs" / identifier
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] not in ("completed", "failed"):
            raise AssertionError(f"Unfinished run: {identifier}")
        for name, expected in manifest["source"]["files"].items():
            if current_files.get(name) != expected:
                raise AssertionError(f"Source changed since run: {name}")
        for relative, expected in manifest["artifacts"].items():
            if digest_file(directory / relative) != expected:
                raise AssertionError(f"Artifact changed: {identifier}/{relative}")
        entries.append({"run_id": identifier, "status": manifest["status"],
                        "exit_code": manifest["exit_code"], "source_commit": manifest["source"]["commit"],
                        "manifest_sha256": digest_file(directory / "manifest.json"),
                        "measured_wall_seconds": manifest["measured_wall_seconds"],
                        "assets": {relative: digest for relative, digest in manifest["artifacts"].items()
                                   if relative.startswith("checkpoints/") or relative in
                                   ("traces/test_input.json", "traces/test.json", "traces/evaluation.json",
                                    "summary.json", "unit_tests_stdout.log")}})
    actor_checks = []
    for case in smoke["cases"]:
        if case["method"] == "greedy_consolidation":
            continue
        checkpoints = workspace / "runs" / case["run_id"] / "checkpoints"
        initial = torch.load(checkpoints / "initial.pt", map_location="cpu", weights_only=False)["controller"]
        final = torch.load(checkpoints / "final.pt", map_location="cpu", weights_only=False)["controller"]
        check = {"method": case["method"], "horizon": case["horizon"]}
        for layer in ("upper", "lower"):
            changed = sum(not torch.equal(value, final[layer][name]) for name, value in initial[layer].items()
                          if name.startswith(("actor_", "score.", "rate_")))
            expected = case["method"] != ("always_on" if layer == "upper" else "upper_ppo_greedy")
            if bool(changed) != expected:
                raise AssertionError(f"Unexpected actor update: {case['method']} {layer}")
            check[layer + "_changed_tensors"] = changed
        actor_checks.append(check)
    result = {"source_commit": smoke["source"]["commit"], "local_runs_root": str(workspace / "runs"),
              "rebuild_commands": ["python -m mist_sim.verification --keep-artifacts NEW_DEBUG_DIRECTORY"],
              "training_slots_main_cases": sum(case["training_slots"] for case in smoke["cases"]),
              "test_slots_main_cases": sum(case["test_slots"] for case in smoke["cases"]),
              "paper_sha256": paper_checksums(args.paper_dir),
              "paper_check": "checked" if args.paper_dir else "not_requested",
              "all_listed_assets_checked": True, "actor_update_checks": actor_checks, "runs": entries}
    write_json(workspace / "results/artifact_index.json", result)
    print(json.dumps({"runs": len(entries), "training_slots": result["training_slots_main_cases"],
                      "test_slots": result["test_slots_main_cases"], "checksums": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
