"""Public-information split optimization after a sampled fixed-cut proposal."""
import copy
import math

from ..allocation import solve_p3
from ..config import fixed_split_for


def proposal_config(config):
    if config.get("split_solver", "policy") != "fixed_proposal":
        return config
    return dict(config, dynamic_split=False, split_solver="policy")


def proposal_problem(problem):
    """Preserve the exact Fixed-Split feature/action set, including bootstrap."""
    if problem.config.get("split_solver", "policy") != "fixed_proposal":
        return problem
    if not hasattr(problem, "_fixed_proposal"):
        proposal = copy.copy(problem)
        proposal.config = proposal_config(problem.config)
        proposal.choices = [[choice for choice in options if choice["kind"] != "edge" or
                             choice["split"] == fixed_split_for(problem.config, request["model"])]
                            for request, options in zip(problem.public, problem.choices)]
        problem._fixed_proposal = proposal
    return problem._fixed_proposal


def map_actions(proposal, problem, actions):
    mapped = []
    for source, target, action in zip(proposal.choices, problem.choices, actions):
        choice = source[action]
        key = (choice["kind"], choice["server"], choice["split"])
        mapped.append(next(index for index, item in enumerate(target)
                           if (item["kind"], item["server"], item["split"]) == key))
    return mapped


def refine_splits(problem, actions, passes=1):
    """Feasible coordinate descent; keep admission/server fixed, recompute P3.

    Only the affected server needs a new solve. Strict decreases break ties in
    favor of the proposal. This is a local per-slot guarantee, not global or
    closed-loop dominance. No raw requests, future inputs or realized costs.
    """
    actions = list(actions)
    _, rates, usage, _ = problem.allocate(actions)
    costs = [problem.choice_cost(options[action], rate)
             for options, action, rate in zip(problem.choices, actions, rates)]
    assignments = [[i for i, (options, action) in enumerate(zip(problem.choices, actions))
                    if options[action]["kind"] == "edge" and options[action]["server"] == server]
                   for server in range(len(problem.active))]
    initial = sum(costs)
    changes = solves = 0
    for _ in range(passes):
        changed = False
        for index, options in enumerate(problem.choices):
            old = options[actions[index]]
            if old["kind"] != "edge":
                continue
            server_index = old["server"]
            server = problem.config["servers"][server_index]
            indices = assignments[server_index]
            best, best_cost = actions[index], sum(costs[i] for i in indices)
            best_costs = None
            for action, choice in enumerate(options):
                if action == actions[index] or choice["kind"] != "edge" or choice["server"] != server_index:
                    continue
                if (usage["compute"][server_index] - old["minimum"] + choice["minimum"] > server["capacity"] + 1e-9 or
                    usage["memory"][server_index] - old["memory"] + choice["memory"] > server["memory"] + 1e-9):
                    continue
                selected = [choice if i == index else problem.choices[i][actions[i]] for i in indices]
                values = solve_p3([c["alpha"] for c in selected], [c["beta"] for c in selected],
                                  [c["minimum"] for c in selected], server["capacity"])
                solves += 1
                trial = [problem.choice_cost(c, rate) for c, rate in zip(selected, values)]
                cost = sum(trial)
                if math.isfinite(cost) and cost < best_cost - 1e-10:
                    best, best_cost, best_costs = action, cost, trial
            if best != actions[index]:
                choice = options[best]
                usage["compute"][server_index] += choice["minimum"] - old["minimum"]
                usage["memory"][server_index] += choice["memory"] - old["memory"]
                actions[index] = best
                for i, cost in zip(indices, best_costs):
                    costs[i] = cost
                changes += 1
                changed = True
        if not changed:
            break
    return actions, dict(changes=changes, p3_solves=solves, proposal_cost=initial, refined_cost=sum(costs))
