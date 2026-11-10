"""Causal greedy baselines, using only public problems/history."""
from ...environment import PublicProblem
from ...allocation import solve_p3
from ...profiles import Profiles


def greedy_fast(problem):
    actions = [0] * len(problem.choices)
    usage = problem.empty_usage()
    # Future requests remain rejected. Adding a request can only alter the P3
    # rates of its target server; preserve all other per-request costs.
    costs = [problem.choice_cost(options[0], 0.0) for options in problem.choices]
    assignments = [[] for _ in problem.config["servers"]]
    current_cost = sum(costs)
    for index, options in enumerate(problem.choices):
        legal = problem.mask(options, usage)
        best_action, best_cost = 0, current_cost
        best_costs = costs
        for action in range(1, len(options)):
            if not legal[action]:
                continue
            choice = options[action]
            trial_costs = list(costs)
            if choice["kind"] == "edge":
                indices = assignments[choice["server"]] + [index]
                selected = [problem.choices[other][actions[other]] for other in indices[:-1]] + [choice]
                rates = solve_p3([item["alpha"] for item in selected],
                                 [item["beta"] for item in selected],
                                 [item["minimum"] for item in selected],
                                 problem.config["servers"][choice["server"]]["capacity"])
                for other, item, rate in zip(indices, selected, rates):
                    trial_costs[other] = problem.choice_cost(item, rate)
            else:
                trial_costs[index] = problem.choice_cost(choice, 0.0)
            # Use the original request-order sum, including for close ties.
            cost = sum(trial_costs)
            if cost < best_cost - 1e-12:
                best_action, best_cost = action, cost
                best_costs = trial_costs
        actions[index] = best_action
        current_cost = best_cost
        costs = best_costs
        choice = options[best_action]
        if choice["kind"] == "edge":
            assignments[choice["server"]].append(index)
        problem.reserve(options[best_action], usage)
    return actions, None, None


def greedy_slow(observation):
    config = observation["config"]
    history = observation["history"][-config["window"]:]
    if not history:
        return [1] * len(config["servers"])
    profiles = Profiles(config)

    def estimate(active):
        cost = 0.0
        previous = observation["active"]
        for offset in range(min(config["window"], config["horizon"] - observation["slot"])):
            public = history[offset % len(history)]["public"]
            ready = [old * new for old, new in zip(previous, active)]
            problem = PublicProblem(config, public, active, ready, observation["slot"], history, profiles=profiles)
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
