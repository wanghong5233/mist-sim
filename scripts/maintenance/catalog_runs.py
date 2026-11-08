"""Index existing run metadata and disk use; never load checkpoints or modify runs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "results/maintenance/run_catalog.csv")
    args = parser.parse_args()
    rows = []
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {"status": "missing_or_invalid_manifest"}
        command = manifest.get("command", "")
        kind = "profiling" if "profile_models" in command else "engineering_validation_or_other"
        if "check_inputs" in command:
            kind = "input_validation"
        files = [path for path in directory.rglob("*") if path.is_file()]
        checkpoints = [path for path in files if path.suffix == ".pt"]
        rows.append({"run_id": manifest.get("run_id", directory.name), "kind": kind, "status": manifest.get("status", "unknown"),
                     "source_commit": manifest.get("source", {}).get("commit", ""),
                     "disk_bytes": sum(path.stat().st_size for path in files),
                     "checkpoint_count": len(checkpoints),
                     "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
                     "path": directory.relative_to(root).as_posix(), "command": command})
    fields = ["run_id", "kind", "status", "source_commit", "disk_bytes", "checkpoint_count", "checkpoint_bytes", "path", "command"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "bytes": sum(row["disk_bytes"] for row in rows),
                      "checkpoints": sum(row["checkpoint_count"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
