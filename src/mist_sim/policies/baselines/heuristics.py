"""Causal greedy baselines, using only public problems/history."""
from ...environment import PublicProblem


def greedy_fast(problem):
    actions = [0] * len(problem.choices)
    usage = problem.empty_usage()
    for index, options in enumerate(problem.choices):
        legal = problem.mask(options, usage)
        best_action, best_cost = 0, problem.scheduling_cost(actions)
        for action in range(1, len(options)):
            if not legal[action]:
                continue
            trial = list(actions)
            trial[index] = action
            cost = problem.scheduling_cost(trial)
            if cost < best_cost - 1e-12:
                best_action, best_cost = action, cost
        actions[index] = best_action
        problem.reserve(options[best_action], usage)
    return actions, None, None


def greedy_slow(observation):
    config = observation["config"]
    history = observation["history"][-config["window"]:]
    if not history:
        return [1] * len(config["servers"])

    def estimate(active):
        cost = 0.0
        previous = observation["active"]
        for offset in range(min(config["window"], config["horizon"] - observation["slot"])):
            public = history[offset % len(history)]["public"]
            ready = [old * new for old, new in zip(previous, active)]
            problem = PublicProblem(config, public, active, ready, observation["slot"], history)
            actions, _, _ = greedy_fast(problem)
            cost += problem.scheduling_cost(actions)
            cost += sum(config["slot_seconds"] * (server["idle_power"] if enabled else server["sleep_power"])
                        + server["wake_energy"] * max(0, enabled - old)
                        for server, enabled, old in zip(config["servers"], active, previous))
            previous = active
        return cost

    active = [1] * len(config["servers"])
    cost = estimate(active)
    for index in sorted(range(len(active)), key=lambda index: -config["servers"][index]["idle_power"]):
        trial = list(active)
        trial[index] = 0
        trial_cost = estimate(trial)
        if trial_cost < cost - 1e-12:
            active, cost = trial, trial_cost
    return active
