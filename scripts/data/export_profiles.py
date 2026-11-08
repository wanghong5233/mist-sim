"""Independently validate saved profiling records and export data components."""
import argparse
import csv
import json
import sys
from pathlib import Path

from mist_sim.profile_export import export_profiles
from mist_sim.storage import digest_file, source_identity, write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", type=Path)
    selection.add_argument("--verify-saved", action="store_true")
    args = parser.parse_args()
    if args.verify_saved:
        index = json.loads((root / "results/data_index.json").read_text(encoding="utf-8"))
        for label in ("processed", "summary", "observations"):
            if digest_file(root / index[label + "_path"]) != index[label + "_sha256"]:
                raise AssertionError(f"Changed exported asset: {label}")
        records = json.loads((root / index["observations_path"]).read_text(encoding="utf-8"))
        components, rows, checks = export_profiles(records, index["source_run"])
        if components != json.loads((root / index["processed_path"]).read_text(encoding="utf-8")):
            raise AssertionError("Recomputed components differ")
        with (root / index["summary_path"]).open(encoding="utf-8", newline="") as stream:
            saved_rows = list(csv.DictReader(stream))
        if saved_rows != [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]:
            raise AssertionError("Recomputed CSV differs")
        print(json.dumps({"source_run": index["source_run"], **checks, "saved_data_recomputed": True}, indent=2))
        return
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Only completed measurement runs can be exported")
    for relative, expected in manifest["artifacts"].items():
        if digest_file(run / relative.replace("\\", "/")) != expected:
            raise AssertionError(f"Changed measurement asset: {relative}")
    spec = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    records = [json.loads((run / "traces" / (name + ".json")).read_text(encoding="utf-8")) for name in spec["models"]]
    run_id = manifest["run_id"]
    components, rows, checks = export_profiles(records, run_id)
    output = root / "data/processed" / ("dnn_profiles_" + run_id + ".json")
    write_json(output, components)
    table = root / "results/dnn_profile_summary.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assets = json.loads((run / "assets.json").read_text(encoding="utf-8"))
    for measured in records:
        for device in measured["devices"].values():
            for split in device["splits"]:
                split.pop("prefix_graph", None)
                split.pop("suffix_graph", None)
    observations = root / "results/profile_observations.json"
    write_json(observations, records)
    index = {"source_run": run_id, "run_path": run.relative_to(root).as_posix(), "source_commit": manifest["source"]["commit"],
             "export_source": source_identity(root),
             "source_files": manifest["source"]["files"], "raw_manifest_sha256": digest_file(run / "manifest.json"),
             "raw_measurements": {(Path("runs") / run.name / "traces" / (name + ".json")).as_posix(): digest_file(run / "traces" / (name + ".json")) for name in spec["models"]},
             "processed_path": output.relative_to(root).as_posix(), "processed_sha256": digest_file(output),
             "summary_path": table.relative_to(root).as_posix(), "summary_sha256": digest_file(table),
             "observations_path": observations.relative_to(root).as_posix(), "observations_sha256": digest_file(observations),
             "environment": json.loads((run / "environment.json").read_text(encoding="utf-8")),
             "measurement_summary": json.loads((run / "summary.json").read_text(encoding="utf-8")),
             "measurement_config": spec,
             "assets": assets, "verification": checks, "constant_rate_diagnostics": components["constant_rate_diagnostics"],
             "training_ready": False, "missing_calibrations": components["missing_calibrations"],
             "rebuild_saved_data": f"python scripts/data/export_profiles.py --run runs/{run.name}",
             "verify_without_hardware_or_original_assets": "python scripts/data/export_profiles.py --verify-saved",
             "remeasure": ["python scripts/data/profile_models.py", "python scripts/data/export_profiles.py --run runs/<new_run_id>"],
             "reproduction_note": "Structural values and outputs are reproducible; timings/memory are device/runtime dependent, not bitwise targets"}
    write_json(root / "results/data_index.json", index)
    print(json.dumps({"processed": str(output), **checks, "training_ready": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
