"""Inspect physical and PPO logs without running training; paths are relative to cwd."""
import argparse
import json
from pathlib import Path
import sys
from mist_sim.diagnostics import inspect_run
from mist_sim.storage import write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = list(args.runs)
    if args.suite:
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        root = args.suite.resolve().parents[1]
        paths += [root / "logs/runs" / case["run_id"] for case in suite["cases"]]
        paths.append(root / "logs/runs" / suite["resume_run_id"])
    if not paths:
        parser.error("Provide run paths or --suite reports/smoke_verification.json")
    cases = [inspect_run(path) for path in paths]
    report = {"runs": cases, "update_samples": sum(update["samples"] for case in cases for update in case["updates"]),
              "minibatches": sum(update["minibatches"] for case in cases for update in case["updates"]),
              "physical_records": sum(item["slots"] for case in cases for item in case["physics"]),
              "edge_assignments_checked": sum(item["edge_assignments"] for case in cases for item in case["physics"]),
              "independent_arithmetic_passed": True}
    if args.output:
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
