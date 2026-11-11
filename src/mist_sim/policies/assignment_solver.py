"""Bounded public-information improvement of a feasible scheduling proposal."""
import math

from ..allocation import solve_p3


class _Costs:
    """Slot-local exact server costs, with admissible unconstrained bounds."""

    def __init__(self, problem):
        self.problem = problem
        self.cache = {}
        self.solves = self.pruned = 0
        self.bounds = []
        for options in problem.choices:
            bounds = []
            for choice in options:
                if choice["kind"] == "edge":
                    rate = max(choice["minimum"], (choice["beta"] / (2 * choice["alpha"])) ** (1 / 3))
                    bounds.append(problem.choice_cost(choice, rate))
                else:
                    bounds.append(problem.choice_cost(choice, 0.))
            self.bounds.append(bounds)

    def __call__(self, actions, ceiling=math.inf):
        problem = self.problem
        compute = [0.] * len(problem.active)
        memory = [0.] * len(problem.active)
        bandwidth = [0.] * len(problem.config["access_points"])
        groups = [[] for _ in problem.active]
        lower = cost = 0.
        for index, (options, action) in enumerate(zip(problem.choices, actions)):
            choice = options[action]
            lower += self.bounds[index][action]
            if choice["kind"] != "edge":
                cost += choice["cost_fixed"]
                continue
            server_index, ap = choice["server"], choice["ap"]
            compute[server_index] += choice["minimum"]
            memory[server_index] += choice["memory"]
            bandwidth[ap] += choice["bandwidth"]
            server = problem.config["servers"][server_index]
            if (compute[server_index] > server["capacity"] + 1e-10 or
                    memory[server_index] > server["memory"] + 1e-9 or
                    bandwidth[ap] > problem.config["access_points"][ap]["bandwidth"] + 1e-9):
                return math.inf
            groups[server_index].append((index, action))
        if lower >= ceiling - 1e-10:
            self.pruned += 1
            return math.inf
        for server_index, group in enumerate(groups):
            if not group:
                continue
            key = server_index, tuple(group)
            if key not in self.cache:
                selected = [problem.choices[i][a] for i, a in group]
                rates = solve_p3([c["alpha"] for c in selected], [c["beta"] for c in selected],
                                 [c["minimum"] for c in selected],
                                 problem.config["servers"][server_index]["capacity"])
                self.cache[key] = sum(problem.choice_cost(c, rate) for c, rate in zip(selected, rates))
                self.solves += 1
            cost += self.cache[key]
        return cost


def refine_assignments(problem, actions, passes=1):
    """Coordinate moves plus admission/relief exchanges; never reduce admissions.

    A relief move sends one existing edge task to its best local/reject option,
    allowing a rejected task to use the freed AP/server. Neither move need
    improve by itself; their jointly feasible P3 cost must strictly decrease.
    Server activation stays with PPO. No hidden requests, future or raw values.
    """
    # Validate the proposal using the common allocator, never project it.
    problem.allocate(actions)
    actions = list(actions)
    costs = _Costs(problem)
    current = initial = costs(actions)
    changes = exchanges = 0
    for _ in range(passes):
        changed = False
        for index, options in enumerate(problem.choices):
            old = actions[index]
            best, best_cost = old, current
            for action in sorted(range(len(options)), key=lambda a: (costs.bounds[index][a], a)):
                if action == old or (options[old]["kind"] != "reject" and options[action]["kind"] == "reject"):
                    continue
                trial = list(actions)
                trial[index] = action
                value = costs(trial, best_cost)
                if value < best_cost - 1e-10:
                    best, best_cost = action, value
            if best != old:
                actions[index], current = best, best_cost
                changes += 1
                changed = True

        for index, options in enumerate(problem.choices):
            if options[actions[index]]["kind"] != "reject":
                continue
            best_actions, best_cost = None, current
            for action in sorted(range(len(options)), key=lambda a: (costs.bounds[index][a], a)):
                choice = options[action]
                if choice["kind"] != "edge":
                    continue
                for other, other_options in enumerate(problem.choices):
                    old = other_options[actions[other]]
                    if (other == index or old["kind"] != "edge" or
                            (old["ap"] != choice["ap"] and old["server"] != choice["server"])):
                        continue
                    relief = min((a for a, c in enumerate(other_options) if c["kind"] != "edge"),
                                 key=lambda a: (other_options[a]["cost_fixed"], a))
                    trial = list(actions)
                    trial[index], trial[other] = action, relief
                    value = costs(trial, best_cost)
                    if value < best_cost - 1e-10:
                        best_actions, best_cost = trial, value
            if best_actions is not None:
                actions, current = best_actions, best_cost
                exchanges += 1
                changed = True
        if not changed:
            break
    return actions, dict(changes=changes, exchanges=exchanges, p3_solves=costs.solves,
                         bound_prunes=costs.pruned, proposal_cost=initial, refined_cost=current)
