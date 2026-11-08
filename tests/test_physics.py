import copy
import numpy as np
import pytest
from scipy.optimize import minimize
from mist_sim.allocation import solve_p3, project_rates
from mist_sim.anonymity import generate_trace, publish
from mist_sim.config import for_method
from mist_sim.environment import Environment, PublicProblem, candidates
from mist_sim.evaluation import replay, summarize
from mist_sim.profiles import Profiles
from mist_sim.reference_solvers import enumerate_slot, enumerate_trajectory


def request(identifier="0:0", size=1, deadline=1.0, position=1.0):
    return {"id": identifier, "model": "toy", "class": "standard", "weight": 1.0,
            "size": size, "deadline": deadline, "position": position, "fading": 1.0, "tie": 0.1}


@pytest.mark.parametrize("alpha,beta,minimum,capacity", [
    ([1, 2], [3, 5], [.1, .2], 10),
    ([.01, .02], [5, 9], [1, 1], 3),
    ([1, 2], [0, 0], [1, 2], 3),
    ([.1, .1], [10, 0], [1, 1], 3),
    ([2], [1], [.8], .8)])
def test_p3_scipy(alpha, beta, minimum, capacity):
    alpha, beta, minimum = map(np.asarray, (alpha, beta, minimum))
    rates = solve_p3(alpha, beta, minimum, capacity)
    objective = lambda values: float(np.sum(alpha * values ** 2 + beta / values))
    result = minimize(objective, minimum + (capacity - minimum.sum()) / len(minimum) * .2,
                      method="SLSQP", bounds=[(bound, capacity) for bound in minimum],
                      constraints=[{"type": "ineq", "fun": lambda values: capacity - values.sum()}],
                      options={"ftol": 1e-11, "maxiter": 1000})
    assert result.success
    assert objective(rates) == pytest.approx(result.fun, rel=1e-7, abs=1e-8)
    assert rates.sum() <= capacity + 1e-8
    assert np.all(rates >= minimum - 1e-9)


def test_p3_empty_infeasible_projection():
    assert solve_p3([], [], [], 1).size == 0
    with pytest.raises(ValueError, match="Infeasible"):
        solve_p3([1, 1], [1, 1], [2, 2], 3)
    np.testing.assert_allclose(project_rates([10, 1], [1, 2], 5), [3, 2])


def test_readiness_and_wake(config):
    config["horizon"] = 4
    environment = Environment(config, [[] for _ in range(4)])
    ready = []
    for _ in range(4):
        problem = environment.begin_slot([1, 0])
        ready.append(problem.ready[0])
        environment.execute([])
    assert ready == [0, 1, 1, 1]
    assert sum(record["energy"]["wake"] for record in environment.records) == config["servers"][0]["wake_energy"]
    assert summarize(environment.records)["on_time_rate"] is None
    assert summarize(environment.records)["completion_delay_mean"] is None
    replay(config, environment.records)


def test_local_no_resources_and_deadline(config):
    config = for_method(config, "full_information")
    config["k"] = 1
    config["horizon"] = 1
    environment = Environment(config, [[request()]])
    problem = environment.begin_slot([0, 0])
    action = next(index for index, choice in enumerate(problem.choices[0]) if choice["kind"] == "local")
    record = environment.execute([action])
    assert all(sum(values) == 0 for values in record["usage"].values())
    assert record["completed"] == 1
    profiles = Profiles(config)
    public = publish([request()], config, "full")[0]
    threshold = config["anonymization_seconds"] + config["control_seconds"] + 4 / config["local_rate"]
    public["deadline"] = [threshold, threshold]
    assert any(choice["kind"] == "local" for choice in candidates(config, profiles, public, [0, 0]))
    public["deadline"] = [threshold - 1e-6] * 2
    assert all(choice["kind"] != "local" for choice in candidates(config, profiles, public, [0, 0]))


def test_generalization_coverage_suppression(config):
    batch = [request("0:0"), request("0:1", size=3, position=2, deadline=.65)]
    released = publish(batch, config)
    assert len(released) == 2
    assert released[0]["position"] == released[1]["position"]
    for item in released:
        original = next(value for value in batch if value["id"] == item["id"])
        for key in ("position", "size", "deadline"):
            assert item[key][0] <= original[key] <= item[key][1]
        assert item["group_size"] >= 2
    assert publish(batch[:1], config) == []


def test_information_boundary(config):
    config["generalization_divisions"] = [1]
    first = [request("0:0", position=.1), request("0:1", position=.2)]
    second = [request("0:0", position=1.1), request("0:1", position=1.2)]
    assert publish(first, config) == publish(second, config)
    config["horizon"] = 1
    env_first, env_second = Environment(config, [first]), Environment(config, [second])
    assert env_first.upper_observation() == env_second.upper_observation()
    public_first, public_second = env_first.begin_slot([1, 1]), env_second.begin_slot([1, 1])
    assert public_first.as_dict() == public_second.as_dict()
    from mist_sim.policies.networks import lower_frame
    assert lower_frame(public_first) == lower_frame(public_second)


def test_shared_ap_and_projection(config):
    config = for_method(config, "full_information")
    config["initial_active"] = [1, 1]
    config["horizon"] = 1
    config["access_points"][0]["bandwidth"] = config["reserved_bandwidth"]
    batch = [request("0:0"), request("0:1")]
    environment = Environment(config, [batch])
    problem = environment.begin_slot([1, 1])
    actions = [next(index for index, choice in enumerate(options) if choice["server"] == server)
               for options, server in zip(problem.choices, (0, 1))]
    with pytest.raises(ValueError, match="capacity"):
        problem.allocate(actions)
    executed, rates, usage, violations = problem.allocate(actions, [100, 100], allow_projection=True)
    assert executed[1] == 0 and rates[1] == 0
    assert violations and usage["bandwidth"][0] <= config["reserved_bandwidth"]


def test_bound_records_and_reference(config):
    config["horizon"] = 2
    config["window"] = 1
    config["k"] = 1
    config["initial_active"] = [1, 1]
    trace = [[request()], []]
    environment = Environment(config, trace)
    problem = environment.begin_slot([1, 1])
    result = enumerate_slot(problem, max_nodes=100)
    assert result["optimal"] and result["status"] == "complete"
    environment.execute(result["actions"])
    environment.begin_slot([0, 0])
    environment.execute([])
    replay(config, environment.records)
    assert enumerate_slot(problem, max_nodes=1)["status"] == "search_limit"
    assert enumerate_slot(problem, timeout_seconds=0)["status"] == "timeout"
    offline = enumerate_trajectory(config, trace, max_nodes=1000)
    assert offline["optimal"]
    assert offline["cost"] <= sum(record["cost"] for record in environment.records) + 1e-9


def test_nonmonotone_profile_bounds(config):
    config["profiles"]["toy"]["sizes"]["2"]["splits"]["1"]["crossing_tensors_bits"] = [99999, 1]
    profile = Profiles(config).bound("toy", [1, 3], 1, "robust")
    assert profile["bits"] == 100000


def test_numeric_profile_keys(config):
    rows = config["profiles"]["toy"]["sizes"]
    rows["1.0"] = rows.pop("1")
    assert Profiles(config).row("toy", 1)["full_work"] == 4
    rows["1"] = rows["1.0"]
    with pytest.raises(ValueError, match="Duplicate"):
        Profiles(config)


def test_nominal_actual_violations_not_hidden(config):
    config = for_method(config, "nominal")
    config["horizon"] = 1
    config["generalization_divisions"] = [1]
    environment = Environment(config, [[request("0:0", 3, .65), request("0:1", 3, .65)]])
    problem = environment.begin_slot([0, 0])
    actions = [next(index for index, choice in enumerate(options) if choice["kind"] == "local")
               for options in problem.choices]
    record = environment.execute(actions)
    assert summarize([record])["timeout"] == 2
    for task in record["tasks"]:
        assert task["elapsed"] == pytest.approx(.65)
        assert task["energy"]["local"] == pytest.approx((.65 - .02) * config["local_kappa"] * config["local_rate"] ** 3)
        assert task["delay"] is None


def test_actual_memory_failure(config):
    config = for_method(config, "nominal")
    config["horizon"] = 1
    config["initial_active"] = [1, 1]
    config["generalization_divisions"] = [1]
    config["servers"][0]["memory"] = 2500000
    config["servers"][0]["capacity"] = 100
    environment = Environment(config, [[request("0:0", 3, 1), request("0:1", 3, 1)]])
    problem = environment.begin_slot([1, 1])
    action = next(index for index, choice in enumerate(problem.choices[0]) if choice["kind"] == "edge" and choice["server"] == 0 and choice["split"] == 1)
    record = environment.execute([action, 0])
    assert record["tasks"][0]["reason"] == "execution_memory"
    assert record["tasks"][0]["energy"]["edge"] == 0
    assert summarize([record])["resource_failed"] == 1


def test_anonymous_observation_same_actual_execution_different(config):
    config["horizon"] = 1
    config["generalization_divisions"] = [1]
    config["initial_active"] = [1, 1]
    config["gain_reference"] = 1.0
    config["servers"][0]["capacity"] = 100.0
    first = [request("0:0", 1), request("0:1", 1)]
    second = [request("0:0", 2), request("0:1", 2)]
    environments = [Environment(config, [batch]) for batch in (first, second)]
    problems = [environment.begin_slot([1, 1]) for environment in environments]
    assert problems[0].as_dict() == problems[1].as_dict()
    action = next(index for index, choice in enumerate(problems[0].choices[0]) if choice["kind"] == "edge")
    records = [environment.execute([action, 0]) for environment in environments]
    assert records[0]["tasks"][0]["delay"] != records[1]["tasks"][0]["delay"]
    assert records[0]["cost"] == records[1]["cost"]


def test_identifiers_are_ephemeral_and_true_terminal_private(config):
    config["arrival_probability"] = 1
    trace = generate_trace(config, 42)
    suffixes = [request["id"].split(":")[1] for batch in trace for request in batch]
    assert len(suffixes) == len(set(suffixes))
    assert all("terminal" not in item for batch in trace for item in publish(batch, config))
