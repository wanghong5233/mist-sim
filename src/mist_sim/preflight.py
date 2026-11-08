"""Policy-independent optimistic service ceiling on fixed development trajectories."""
import hashlib
import json
import time
from .anonymity import generate_trace, publish
from .environment import candidates
from .profiles import Profiles


def service_ceiling(config, seeds):
    started = time.perf_counter()
    profiles = Profiles(config)
    ready = [1] * len(config["servers"])
    rows = []
    for seed in seeds:
        trace = generate_trace(config, seed)
        counts = {model: dict(arrivals=0, published=0, local=0, edge=0, either=0)
                  for model in config["profiles"]}
        for batch in trace:
            for request in batch:
                counts[request["model"]]["arrivals"] += 1
            for request in publish(batch, config, config["information"]):
                options = candidates(config, profiles, request, ready)
                kinds = {choice["kind"] for choice in options}
                row = counts[request["model"]]
                row["published"] += 1
                row["local"] += "local" in kinds
                row["edge"] += "edge" in kinds
                row["either"] += bool(kinds - {"reject"})
        rows.append(dict(seed=seed, models=counts,
                         trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()))
    totals = {model: {key: sum(row["models"][model][key] for row in rows)
                      for key in ("arrivals", "published", "local", "edge", "either")}
              for model in config["profiles"]}
    arrivals = sum(row["arrivals"] for row in totals.values())
    possible = sum(row["either"] for row in totals.values())
    ceiling = possible / arrivals if arrivals else None
    threshold = config["validation"]["min_service_rate"]
    return dict(scope="Fixed development trajectories only; all servers ready; joint capacities relaxed",
                upper_bound_not_achievable_policy=True, models=totals, trajectories=rows,
                arrivals=arrivals, individually_serviceable=possible, optimistic_service_ceiling=ceiling,
                threshold=threshold,
                threshold_ruled_out=threshold is not None and ceiling is not None and ceiling < threshold,
                measured_seconds=time.perf_counter() - started)
