"""Metrics rebuilt from raw task records rather than trusting summaries."""
import math
import time
import numpy as np
from .environment import Environment


def summarize(records):
    arrivals = sum(len(record["raw"]) for record in records)
    tasks = [task for record in records for task in record["tasks"]]
    completed = [task for task in tasks if task["on_time"]]
    energy = {key: sum(record["energy"][key] for record in records)
              for key in ("static", "wake", "local", "transmit", "edge")}
    rejected = sum(task["action"] == "reject" for task in tasks)
    timeout = sum(task["reason"] == "deadline" for task in tasks)
    resource_failed = sum(task["reason"] == "execution_memory" for task in tasks)
    suppressed = sum(len(record["suppressed"]) for record in records)
    if arrivals != len(completed) + rejected + timeout + suppressed + resource_failed:
        raise AssertionError("Request conservation")
    return {"arrivals": arrivals, "completed": len(completed), "suppressed": suppressed,
            "rejected": rejected, "timeout": timeout, "resource_failed": resource_failed,
            "on_time_rate": len(completed) / arrivals if arrivals else None,
            "completion_delay_mean": sum(task["delay"] for task in completed) / len(completed) if completed else None,
            "completion_delay_p95": float(np.quantile([task["delay"] for task in completed], .95)) if completed else None,
            "accepted": len(tasks) - rejected,
            "acceptance_rate_all_arrivals": (len(tasks) - rejected) / arrivals if arrivals else None,
            "published_rate": len(tasks) / arrivals if arrivals else None,
            "energy_per_completed": sum(energy.values()) / len(completed) if completed else None,
            "energy": energy, "total_energy": sum(energy.values()),
            "conservative_cost": sum(record["cost"] for record in records),
            "active_server_slots": sum(sum(record["active"]) for record in records),
            "switches": sum(sum(old != new for old, new in zip(record["previous_active"], record["active"])) for record in records),
            "wake_events": sum(sum(new > old for old, new in zip(record["previous_active"], record["active"])) for record in records),
            "controller_seconds": sum(record["controller_seconds"] for record in records),
            "control_overruns": sum(record["control_overrun"] for record in records),
            "projection_violations": sum(len(record["projection_violations"]) for record in records)}


def evaluate(controller, config, trace, measured_latency=False, check_budget=None):
    environment = Environment(config, trace)
    active = list(config["initial_active"])
    while not environment.done:
        if check_budget is not None:
            check_budget()
        started = time.perf_counter()
        if environment.slot % config["window"] == 0:
            active, _ = controller.slow(environment.upper_observation(), deterministic=True, record=False)
        problem = environment.begin_slot(active)
        actions, predicted, _ = controller.fast(problem, deterministic=True, record=False)
        elapsed = time.perf_counter() - started
        record = environment.execute(actions, predicted, elapsed if measured_latency else 0.0, measured_latency)
        record["measured_policy_seconds"] = elapsed
    return environment.records, summarize(environment.records)


def assert_close(left, right, path="root"):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {path}")
        for key in left:
            assert_close(left[key], right[key], path + "." + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"Lengths differ: {path}")
        for index, (first, second) in enumerate(zip(left, right)):
            assert_close(first, second, f"{path}[{index}]")
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        if not math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-9):
            raise AssertionError(f"Values differ: {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"Values differ: {path}")


def replay(config, records):
    if len(records) > config["horizon"]:
        if len(records) % config["horizon"]:
            raise ValueError("Multi-trajectory replay requires complete episodes")
        for offset in range(0, len(records), config["horizon"]):
            replay(config, records[offset:offset + config["horizon"]])
        return summarize(records)
    environment = Environment(config, [record["raw"] for record in records])
    for original in records:
        environment.begin_slot(original["active"])
        result = environment.execute(original["actions"], original["predicted_rates"], original["controller_seconds"])
        ignored = ("measured_policy_seconds", "measured_allocation_seconds")
        expected = {key: value for key, value in original.items() if key not in ignored}
        result = {key: value for key, value in result.items() if key not in ignored}
        assert_close(expected, result)
    return summarize(environment.records)
