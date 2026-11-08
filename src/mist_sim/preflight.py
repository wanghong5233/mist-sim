"""Policy-independent optimistic service ceiling on fixed development trajectories."""
import hashlib
import json
import time
import copy
import itertools
from collections import Counter
from .anonymity import generate_trace, publish
from .environment import candidates, Environment, radio_rate
from .profiles import Profiles
from .diagnostics import check_physics
from .evaluation import summarize, replay


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


def physical_envelope(config):
    """Worst public intervals, no resource competition; a bound, not a policy."""
    profiles = Profiles(config)
    report = {}
    for model in config["profiles"]:
        sizes = profiles.sizes(model)
        request = dict(model=model, size=[min(sizes), max(sizes)],
                       position=config["position_range"], fading=config["fading_range"],
                       deadline=config["deadline_range"], weight=1.)
        choices = candidates(config, profiles, request, [1] * len(config["servers"]))
        rows = []
        for server_index, server in enumerate(config["servers"]):
            if model not in server["models"]:
                continue
            position = max(request["position"], key=lambda value: abs(value - config["access_points"][server["ap"]]["position"]))
            rate = radio_rate(config, server, position, request["fading"][0])
            for split in profiles.models[model]["legal_splits"]:
                profile = profiles.bound(model, request["size"], split, "robust")
                local = profile["local_work"] / config["local_rate"]
                transmission = profile["bits"] / rate
                fixed = config["anonymization_seconds"] + config["control_seconds"] + config["backhaul_seconds"] + local + transmission
                rows.append(dict(server=server_index, split=split, local_seconds=local,
                    transmission_seconds=transmission, fixed_seconds=fixed,
                    fastest_total_seconds=fixed + profile["edge_work"] / server["capacity"],
                    memory_fits=profile["memory"] <= server["memory"],
                    bandwidth_fits=config["reserved_bandwidth"] <= config["access_points"][server["ap"]]["bandwidth"]))
        report[model] = dict(local_feasible=any(choice["kind"] == "local" for choice in choices),
            edge_feasible=any(choice["kind"] == "edge" for choice in choices),
            fastest_edge_seconds=min((row["fastest_total_seconds"] for row in rows
                                      if row["memory_fits"] and row["bandwidth_fits"]), default=None),
            candidates=rows)
    return report


def constructive_actions(problem, prefer_local=False):
    """Fixed public-order feasible witness; not an optimal solver or learned policy.

    Rank by isolated P3 cost, reserve minimum resources, then execute joint P3.
    This intentionally does not search trajectories or tune scenario parameters.
    """
    usage, actions = problem.empty_usage(), []
    for options in problem.choices:
        scored = []
        for index, (choice, valid) in enumerate(zip(options, problem.mask(options, usage))):
            if not valid:
                continue
            rate = 0.
            if choice["kind"] == "edge":
                capacity = problem.config["servers"][choice["server"]]["capacity"]
                stationary = ((choice["beta"] / (2 * choice["alpha"])) ** (1 / 3)
                              if choice["alpha"] > 0 else capacity)
                rate = min(capacity, max(choice["minimum"], stationary))
            priority = -1 if prefer_local and choice["kind"] == "local" else 0
            scored.append((priority, problem.choice_cost(choice, rate), index))
        action = min(scored)[2]
        actions.append(action)
        problem.reserve(options[action], usage)
    return actions


def feasibility_witness(config, seeds):
    """Construct executable schedules, independently replay, retain scalar evidence."""
    profiles = Profiles(config)
    consolidated = [0] * len(config["servers"])
    for ap in range(len(config["access_points"])):
        indices = [index for index, server in enumerate(config["servers"]) if server["ap"] == ap]
        if indices:
            chosen = max(indices, key=lambda index: (config["servers"][index]["capacity"] /
                         config["servers"][index]["idle_power"], config["servers"][index]["capacity"], -index))
            consolidated[chosen] = 1
    vectors = {"sleep": [0] * len(consolidated), "all_on": [1] * len(consolidated), "consolidated": consolidated,
               "all_on_local_first": [1] * len(consolidated), "consolidated_local_first": consolidated}
    results = []
    for seed in seeds:
        trace = generate_trace(config, seed)
        modes = {}
        for name, active in vectors.items():
            environment = Environment(config, trace)
            for _ in trace:
                problem = environment.begin_slot(active)
                environment.execute(constructive_actions(problem, prefer_local=name.endswith("local_first")))
            check_physics(config, environment.records)
            metrics = summarize(environment.records)
            if replay(config, environment.records) != metrics:
                raise AssertionError("Witness replay mismatch")
            completed = Counter()
            edge_completed = Counter()
            for record in environment.records:
                models = {request["id"]: request["model"] for request in record["public"]}
                for task in record["tasks"]:
                    if task["on_time"]:
                        completed[models[task["id"]]] += 1
                        edge_completed[models[task["id"]]] += task["action"] == "edge"
            modes[name] = dict(active=active, cost=sum(row["cost"] for row in environment.records),
                               metrics=metrics, completed_by_model=dict(completed), edge_completed_by_model=dict(edge_completed))
        results.append(dict(seed=seed, trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(), modes=modes))
    envelope = physical_envelope(config)
    passed = bool(results) and all(row["edge_feasible"] for row in envelope.values()) and all(
        all(row["modes"]["all_on"]["completed_by_model"].get(model, 0) > 0 for model in profiles.models) and
        min(row["modes"][name]["cost"] for name in vectors if name != "sleep") < row["modes"]["sleep"]["cost"]
        for row in results)
    return dict(scope="Fixed, public-information constructive witnesses; not optimality or HPGI performance",
                passed=passed, physical_envelope=envelope, trajectories=results)


def small_optimality_audit(config, max_nodes=20000, timeout_seconds=5):
    """Predeclared reduced grid; exact results only when every search completes.

    Two representative servers, one/two requests, k=1, already awake.
    It isolates switching/splitting opportunity, not eight-server performance.
    No cases are selected or discarded according to the direction of the gap.
    """
    from .reference_solvers import enumerate_slot
    results = []
    representatives = sorted(range(len(config["servers"])), key=lambda index: config["servers"][index]["capacity"])
    selected = list(dict.fromkeys([representatives[0], representatives[-1]]))
    small = copy.deepcopy(config)
    small.update(servers=[copy.deepcopy(config["servers"][index]) for index in selected],
                 horizon=1, k=1, terminals=2, initial_active=[1] * len(selected),
                 learned_rates=False, capacity_mask=True, information="robust")
    positions = [config["access_points"][0]["position"], config["position_range"][1]]
    for model, position, count in itertools.product(sorted(config["profiles"]), positions, (1, 2)):
        size = min(Profiles(small).sizes(model))
        batch = [dict(id=str(index), model=model, **{"class": "standard"}, weight=1.,
                      size=size, position=position, deadline=config["deadline_range"][0],
                      fading=config["fading_range"][0], tie=float(index)) for index in range(count)]
        comparisons = {}
        fixed_modes = {f"fixed_{split}": split for split in config["profiles"][model]["legal_splits"]}
        for mode in ("dynamic", "always_on", "fixed_split", *fixed_modes):
            scenario = copy.deepcopy(small)
            scenario["method"] = "fixed_split" if mode == "fixed_split" or mode in fixed_modes else "hpgi"
            if mode in fixed_modes:
                scenario["fixed_split"] = {name: fixed_modes[mode] if name == model else value["legal_splits"][0]
                                           for name, value in scenario["profiles"].items()}
            controls = ([tuple([1] * len(selected))] if mode == "always_on" else
                        itertools.product((0, 1), repeat=len(selected)))
            best = None
            complete = True
            for control in controls:
                environment = Environment(scenario, [batch])
                problem = environment.begin_slot(list(control))
                solved = enumerate_slot(problem, max_nodes, timeout_seconds)
                complete &= solved["optimal"]
                record = environment.execute(solved["actions"])
                if best is None or record["cost"] < best["cost"]:
                    best = dict(cost=record["cost"], completed=record["completed"], active=list(control),
                                actions=solved["actions"], splits=[task["split"] for task in record["tasks"]])
            comparisons[mode] = dict(best, optimal=complete)
        results.append(dict(model=model, position=position, requests=count, comparisons=comparisons))
    complete = all(value["optimal"] for row in results for value in row["comparisons"].values())
    gaps = {mode: [row["comparisons"][mode]["cost"] - row["comparisons"]["dynamic"]["cost"] for row in results]
            for mode in ("always_on", "fixed_split")}
    best_fixed = {}
    for model in sorted(config["profiles"]):
        cases = [row for row in results if row["model"] == model]
        costs = {split: sum(row["comparisons"][f"fixed_{split}"]["cost"] for row in cases)
                 for split in config["profiles"][model]["legal_splits"]}
        split = min(costs, key=costs.get)
        dynamic_cost = sum(row["comparisons"]["dynamic"]["cost"] for row in cases)
        best_fixed[model] = dict(split=split, cost=costs[split], dynamic_cost=dynamic_cost,
                                 gap=costs[split] - dynamic_cost, all_fixed_costs=costs)
    return dict(scope="Reduced two-server k=1 exact grid; constrained optima, not trained baselines or HPGI", complete=complete,
                cases=results, gaps=gaps if complete else None, best_fixed_by_model=best_fixed if complete else None,
                strict_improvement_cases={mode: sum(gap > 1e-8 for gap in values)
                                                                for mode, values in gaps.items()} if complete else None)
