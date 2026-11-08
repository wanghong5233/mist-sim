"""Exercise installed CLI paths against the completed short-run checkpoints."""
import json
import argparse
from pathlib import Path
import subprocess
import sys

from mist_sim.storage import RunStore, write_json
from mist_sim.config import load_config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    workspace = parser.parse_args().workspace.resolve()
    output_dir = workspace / "runs"
    report = json.loads((workspace / "results" / "smoke_verification.json").read_text(encoding="utf-8"))
    case = next(case for case in report["cases"] if case["method"] == "hpgi" and case["horizon"] == 16)
    checkpoint = output_dir / case["run_id"] / "checkpoints" / "final.pt"
    config = load_config(root / "configs" / "smoke.json")
    store = RunStore(root, config, "python scripts/validation/verify_commands.py", 3000, output_dir=output_dir)
    checks = []

    def command(arguments, expected=0):
        before = set(output_dir.iterdir())
        result = subprocess.run([sys.executable, "-m", "mist_sim", *map(str, arguments), "--output-dir", str(output_dir)],
                                cwd=root.parent, capture_output=True, timeout=180)
        new_paths = set(output_dir.iterdir()) - before
        if len(new_paths) != 1:
            raise AssertionError("Command did not create exactly one run")
        path = new_paths.pop()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if result.returncode != expected or manifest["exit_code"] != expected:
            raise AssertionError(result.stderr.decode(errors="replace"))
        index = len(checks)
        (store.path / f"command_{index}_stdout.log").write_bytes(result.stdout)
        (store.path / f"command_{index}_stderr.log").write_bytes(result.stderr)
        checks.append({"arguments": list(map(str, arguments)), "expected_exit": expected,
                       "actual_exit": result.returncode, "run_id": path.name})
        store.write_json("checks.json", checks)
        return path

    try:
        command(["check"])
        command(["check", "--method", "fixed_window"])
        command(["check", "--set", "horizon=17", "--set", "ppo.hidden=32"])
        command(["train", "--seed", "1091", "--set", "training.cycles=1", "--set", "training.pretrain_slots=2",
                 "--set", "training.fast_slots=2", "--set", "training.slow_intervals=1", "--set", "ppo.epochs=1"])
        evaluated = command(["evaluate", "--checkpoint", checkpoint, "--seed", "3000"])
        records = evaluated / "traces" / "evaluation.json"
        replayed = command(["replay", "--checkpoint", checkpoint, "--records", records])
        summarized = command(["summarize", "--records", records])
        summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8"))
                     for path in (evaluated, replayed, summarized)]
        if not summaries[0] == summaries[1] == summaries[2]:
            raise AssertionError("CLI summaries differ")
        command(["train", "--checkpoint", checkpoint, "--extra-cycles", "1"])
        command(["evaluate", "--method", "greedy_consolidation"])
        command(["reference", "--max-nodes", "30", "--timeout", "1"])
        command(["reference", "--offline", "--max-nodes", "30", "--timeout", "1"])
        failed = command(["evaluate", "--method", "hpgi"], expected=1)
        if not (failed / "stderr.log").read_text(encoding="utf-8"):
            raise AssertionError("Expected failure was not retained")
        result = {"run_id": store.path.name, "checks": checks, "summary_equality": True,
                  "expected_failure_retained": True}
        store.write_json("summary.json", result)
        store.finish()
        write_json(workspace / "results" / "cli_verification.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        store.finish(error)
        raise


if __name__ == "__main__":
    main()
