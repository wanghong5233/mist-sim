"""Write a small reproducible development feasibility report; no policy training."""
import argparse
from pathlib import Path
import json
import sys
from mist_sim.config import load_config
from mist_sim.preflight import service_ceiling
from mist_sim.storage import source_identity, write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/conditional.json"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/hpgi_preflight.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [config["seeds"]["validation"] + 10000 * index for index in range(config["validation"]["trajectories"])]
    result = service_ceiling(config, seeds)
    result["full_information_reference"] = service_ceiling(dict(config, information="full"), seeds)
    result.update(config=config, source=source_identity(Path(__file__).resolve().parents[2]))
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("models", "arrivals", "optimistic_service_ceiling",
                                                 "threshold_ruled_out", "measured_seconds")}, indent=2))


if __name__ == "__main__":
    main()
