"""Write a small reproducible development feasibility report; no policy training."""
import argparse
from pathlib import Path
import json
import sys
from mist_sim.config import load_config
from mist_sim.preflight import service_ceiling, feasibility_witness, small_optimality_audit
from mist_sim.storage import source_identity, write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/conditional.json"))
    parser.add_argument("--output", type=Path, help="Optional diagnostic output; default prints only")
    parser.add_argument("--witness", action="store_true", help="Construct and replay feasible schedules; no learning")
    parser.add_argument("--exact", action="store_true", help="Bounded reduced-grid optima and best fixed splits")
    parser.add_argument("--seed", type=int, action="append", help="Explicit diagnostic seeds; repeat for multiple inputs")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seed or [config["seeds"]["validation"] + 10000 * index for index in range(config["validation"]["trajectories"])]
    result = service_ceiling(config, seeds)
    result["full_information_reference"] = service_ceiling(dict(config, information="full"), seeds)
    if args.witness:
        result["witness"] = feasibility_witness(config, seeds)
    if args.exact:
        result["exact"] = small_optimality_audit(config)
    result.update(config=config, source=source_identity(Path(__file__).resolve().parents[2]))
    if args.output:
        write_json(args.output, result)
    summary = {key: result[key] for key in ("models", "arrivals", "optimistic_service_ceiling",
                                           "threshold_ruled_out", "measured_seconds")}
    if args.witness:
        witness = result["witness"]
        summary["witness"] = dict(passed=witness["passed"], scope=witness["scope"],
            trajectories=[dict(seed=row["seed"], trace_sha256=row["trace_sha256"],
                modes={name: dict(cost=value["cost"], on_time_rate=value["metrics"]["on_time_rate"],
                                  completed_by_model=value["completed_by_model"])
                       for name, value in row["modes"].items()}) for row in witness["trajectories"]])
    if args.exact:
        summary["exact"] = {key: result["exact"][key] for key in
                            ("scope", "complete", "strict_improvement_cases", "best_fixed_by_model")}
    print(json.dumps(summary, indent=2))
    if args.witness and not result["witness"]["passed"]:
        raise SystemExit("Feasible improvement witness not established")
    if args.exact and not result["exact"]["complete"]:
        raise SystemExit("Exact search incomplete; no optimality claim")


if __name__ == "__main__":
    main()
