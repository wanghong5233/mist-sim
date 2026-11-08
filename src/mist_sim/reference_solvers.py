"""Bounded exact references. Only exhaustive completion implies optimality."""
import copy
import itertools
import time
from .environment import Environment


def enumerate_slot(problem, max_nodes=10000, timeout_seconds=5):
    start = time.perf_counter()
    nodes = 0
    best_actions = [0] * len(problem.choices)
    best_cost = problem.scheduling_cost(best_actions)
    status = "complete"

    def visit(prefix, usage):
        nonlocal nodes, best_actions, best_cost, status
        if nodes >= max_nodes:
            status = "search_limit"
            return
        if time.perf_counter() - start >= timeout_seconds:
            status = "timeout"
            return
        nodes += 1
        if len(prefix) == len(problem.choices):
            cost = problem.scheduling_cost(prefix)
            if cost < best_cost:
                best_actions, best_cost = list(prefix), cost
            return
        options = problem.choices[len(prefix)]
        for action, legal in enumerate(problem.mask(options, usage)):
            if legal and status == "complete":
                next_usage = {key: values.copy() for key, values in usage.items()}
                problem.reserve(options[action], next_usage)
                visit(prefix + [action], next_usage)

    visit([], problem.empty_usage())
    return {"status": status, "nodes": nodes, "cost": best_cost, "actions": best_actions,
            "optimal": status == "complete", "lower_bound": best_cost if status == "complete" else 0.0,
            "elapsed_seconds": time.perf_counter() - start}


def enumerate_trajectory(config, trace, max_nodes=10000, timeout_seconds=5):
    config = copy.deepcopy(config)
    config["learned_rates"] = False
    config["capacity_mask"] = True
    start = time.perf_counter()
    nodes, status = 0, "complete"
    best_cost, best_records = float("inf"), None
    initial = Environment(config, trace)

    def visit(environment, cost):
        nonlocal nodes, status, best_cost, best_records
        if nodes >= max_nodes:
            status = "search_limit"
            return
        if time.perf_counter() - start >= timeout_seconds:
            status = "timeout"
            return
        nodes += 1
        if environment.done:
            if cost < best_cost:
                best_cost, best_records = cost, copy.deepcopy(environment.records)
            return
        if cost >= best_cost:
            return
        for active in itertools.product((0, 1), repeat=len(config["servers"])):
            if status != "complete":
                return
            pending = copy.deepcopy(environment)
            problem = pending.begin_slot(list(active))
            for actions in itertools.product(*(range(len(options)) for options in problem.choices)):
                if nodes >= max_nodes or time.perf_counter() - start >= timeout_seconds:
                    status = "search_limit" if nodes >= max_nodes else "timeout"
                    return
                nodes += 1
                try:
                    problem.allocate(actions)
                except ValueError:
                    continue
                following = copy.deepcopy(pending)
                record = following.execute(list(actions))
                visit(following, cost + record["cost"])
    visit(initial, 0.0)
    return {"status": status, "nodes": nodes, "cost": best_cost if best_records is not None else None,
            "records": best_records, "optimal": status == "complete",
            "lower_bound": best_cost if status == "complete" else 0.0,
            "elapsed_seconds": time.perf_counter() - start}
