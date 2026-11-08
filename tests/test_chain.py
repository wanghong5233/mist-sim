"""Independent numerical oracles and cross-module regression cases."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from mist_sim.allocation import solve_p3, project_rates
from mist_sim.anonymity import publish, generate_trace
from mist_sim.config import load_config, for_method, validate, layer_settings
from mist_sim.diagnostics import check_physics, check_update
from mist_sim.environment import Environment
from mist_sim.policies.networks import upper_frame
from mist_sim.policies.baselines.heuristics import greedy_fast, greedy_slow
from mist_sim.ppo import loss_terms
from mist_sim.storage import RunStore
from mist_sim.training import Trainer
from tests.test_physics import request


def test_loss_values_and_gradients_independent_numpy(config):
    settings = config["ppo"]
    advantages = np.array([2., -3., 4., -2.])
    ratios = np.array([1.5, .5, .9, 1.1])
    active = np.array([True, True, False, True])
    values = np.array([1., 3., 2., -1.])
    targets = np.array([3., 2., 5., 0.])
    entropies = np.array([.1, .2, .9, .3])
    logprob = torch.tensor(np.log(ratios), requires_grad=True)
    predicted = torch.tensor(values, requires_grad=True)
    loss, actor, value, entropy, _, _ = loss_terms(
        logprob, torch.zeros(4, dtype=torch.float64), torch.tensor(advantages), predicted,
        torch.tensor(targets), torch.tensor(entropies), torch.tensor(active), settings)

    def oracle(logs):
        importance = np.exp(logs)
        unclipped = importance * advantages
        clipped = np.clip(importance, .8, 1.2) * advantages
        actor_cost = -sum(min(raw, cut) if enabled else 0 for raw, cut, enabled in zip(unclipped, clipped, active)) / 4
        return actor_cost + settings["value_coefficient"] * np.mean((values - targets) ** 2) / 2 - settings["entropy_coefficient"] * np.mean(np.where(active, entropies, 0))
    assert float(loss) == pytest.approx(oracle(np.log(ratios)))
    loss.backward()
    finite_difference = []
    for index in range(4):
        plus, minus = np.log(ratios), np.log(ratios)
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference.append((oracle(plus) - oracle(minus)) / 2e-6)
    np.testing.assert_allclose(logprob.grad.numpy(), finite_difference, atol=1e-8)
    np.testing.assert_allclose(predicted.grad.numpy(), settings["value_coefficient"] * (values - targets) / 4)
    assert logprob.grad[2] == 0


def test_random_p3_projection_against_scipy():
    generator = np.random.default_rng(84)
    for _ in range(12):
        alpha = generator.uniform(.1, 2, 3)
        beta = generator.uniform(0, 8, 3)
        minimum = generator.uniform(.1, .5, 3)
        capacity = float(minimum.sum() + generator.uniform(.1, 2))
        rates = solve_p3(alpha, beta, minimum, capacity)
        objective = lambda vector: np.sum(alpha * vector ** 2 + beta / vector)
        solution = minimize(objective, minimum, bounds=[(bound, capacity) for bound in minimum],
                            constraints=[{"type": "ineq", "fun": lambda vector: capacity - sum(vector)}],
                            method="SLSQP", options={"ftol": 1e-10, "maxiter": 200})
        assert solution.success
        assert objective(rates) == pytest.approx(solution.fun, rel=2e-6)
        predictions = generator.uniform(-2, 4, 3)
        projected = project_rates(predictions, minimum, capacity)
        projection = minimize(lambda vector: np.sum((vector - predictions) ** 2), minimum,
                              bounds=[(bound, capacity) for bound in minimum],
                              constraints=[{"type": "ineq", "fun": lambda vector: capacity - sum(vector)}],
                              method="SLSQP", options={"ftol": 1e-10})
        assert projection.success
        np.testing.assert_allclose(projected, projection.x, atol=3e-5)


def test_upper_observes_public_demand_and_actual_allocation(config):
    config["k"] = 1
    config["generalization_divisions"] = [4, 2, 1]
    environment = Environment(config, generate_trace(config, 10))
    observation = environment.upper_observation()
    history = {"public": publish([request(size=1)], config), "cost": 1., "completed": 1,
               "released": 1, "compute": [2., 0.], "allocated_compute": [8., 0.],
               "accepted": 1, "active": [1, 1]}
    observation["history"] = [history]
    first = upper_frame(observation)
    assert first["servers"][0][10] == pytest.approx(8 / config["servers"][0]["capacity"])
    history["public"] = publish([request(size=3)], config)
    second = upper_frame(observation)
    assert first["items"] != second["items"]
    observation["config"]["long_term_feedback"] = False
    assert upper_frame(observation)["servers"][0][10] == 0


def test_full_information_same_published_request_set(config):
    batches = [[request()], [request("a"), request("b", 3, .65, 2)]]
    for batch in batches:
        anonymous = publish(batch, config, "robust")
        ideal = publish(batch, config, "full")
        assert {item["id"] for item in anonymous} == {item["id"] for item in ideal}
        for item in ideal:
            real = next(value for value in batch if value["id"] == item["id"])
            assert item["size"] == [real["size"]] * 2


def test_profiles_do_not_impose_unstated_additivity(config):
    from mist_sim.profiles import Profiles
    config["profiles"]["toy"]["sizes"]["1"]["splits"]["1"]["edge_work"] = 3.2
    assert Profiles(config).bound("toy", [1, 1], 1, "robust")["edge_work"] == 3.2


def test_single_request_entire_radio_delay_energy_reward_by_hand(config):
    config = for_method(config, "full_information")
    config["horizon"], config["k"] = 1, 1
    config["initial_active"] = [1, 0]
    raw = request(size=1, deadline=1)
    raw["weight"] = 2.0
    environment = Environment(config, [[raw]])
    problem = environment.begin_slot([1, 0])
    action = next(index for index, choice in enumerate(problem.choices[0])
                  if choice["kind"] == "edge" and choice["split"] == 1)
    server = config["servers"][0]
    radio = 100000 * np.log2(1 + .1 * .01 / (1e-8 * 100000 + 1e-5))
    overhead = .01 + .01 + 1 / 10 + 12000 / radio + .005
    minimum = 3 / (1 - overhead)
    optimal = max(minimum, (2 / (2 * server["kappa"])) ** (1 / 3))
    record = environment.execute([action])
    task = record["tasks"][0]
    expected_energy = .001 * 1 * 10 ** 2 + .1 * 12000 / radio + server["kappa"] * 3 * optimal ** 2
    expected_delay = overhead + 3 / optimal
    assert task["rate"] == pytest.approx(optimal)
    assert sum(task["energy"].values()) == pytest.approx(expected_energy)
    assert task["delay"] == pytest.approx(expected_delay)
    assert record["cost"] == pytest.approx(.2 + .01 + expected_energy + 2 * expected_delay)
    assert record["reward"] == pytest.approx(-record["cost"])
    check_physics(config, [record])


def test_config_composition_overrides_and_guards(config, tmp_path):
    root = Path(__file__).resolve().parents[1]
    combined = load_config(root / "configs/smoke.json")
    assert combined == config
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"ppo_layers": {"upper": {"learning_rate": .001, "hidden": 32}},
                                   "training": {"cycles": 1}}), encoding="utf-8")
    modified = load_config(root / "configs/smoke.json", [overlay], ["horizon=17", "ppo.minibatch=4"])
    assert layer_settings(modified, "upper")["hidden"] == 32
    assert layer_settings(modified, "lower")["hidden"] == 64
    trainer = Trainer(modified, 41)
    assert trainer.optimizers["upper"].param_groups[0]["lr"] == .001
    assert trainer.optimizers["lower"].param_groups[0]["lr"] == .0003
    with pytest.raises(ValueError, match="Unknown"):
        load_config(root / "configs/smoke.json", overrides=["ppo.lerning_rate=0.1"])
    for key, bad in (("minibatch", 0), ("gae_lambda", 2), ("learning_rate", -1), ("epochs", 1.5)):
        wrong = copy.deepcopy(config)
        wrong["ppo"][key] = bad
        with pytest.raises(ValueError):
            validate(wrong)
    wrong = copy.deepcopy(config)
    wrong["ppo_layers"]["upper"] = {"gamma": .9}
    with pytest.raises(ValueError, match="discount"):
        validate(wrong)


def test_config_scales_models_servers_aps_and_update(config):
    config["horizon"], config["terminals"], config["k"] = 5, 7, 1
    config["window"], config["history_slots"] = 2, 2
    config["arrival_probability"] = 1
    config["initial_active"] = [1, 1, 1]
    config["servers"].append(copy.deepcopy(config["servers"][0]))
    config["access_points"].append({"position": 1.8, "bandwidth": 200000})
    config["servers"][2]["ap"] = 1
    config["profiles"]["other"] = copy.deepcopy(config["profiles"]["toy"])
    config["servers"][2]["models"] = ["other"]
    config["ppo_layers"]["upper"] = {"hidden": 32}
    config["ppo_layers"]["lower"] = {"hidden": 48}
    config["ppo"]["epochs"] = 1
    validate(config)
    trainer = Trainer(config, 51)
    trainer.calibrate()
    for layer in ("upper", "lower"):
        buffer, records = trainer.collect(layer, 2)
        assert len(buffer.samples) == 2
        check_physics(config, records)
    trainer.phase("scaled", "lower", 2)
    assert trainer.history[-1]["optimizer_steps"] == 1


def test_update_logs_forward_gae_physics_and_tampering(config, tmp_path):
    config["horizon"] = 5
    config["k"] = 1
    config["ppo"]["epochs"] = 1
    root = Path(__file__).resolve().parents[1]
    store = RunStore(root, config, "pytest detailed-chain check", 91, output_dir=tmp_path)
    trainer = Trainer(config, 91, store)
    try:
        trainer.calibrate()
        for stage, layer, count in (("upper_check", "upper", 3), ("lower_check", "lower", 7)):
            trainer.phase(stage, layer, count)
            samples = json.loads((store.path / "traces" / (stage + "_behavior.json")).read_text(encoding="utf-8"))
            audit = json.loads((store.path / "traces" / (stage + "_update.json")).read_text(encoding="utf-8"))
            records = json.loads((store.path / "traces" / (stage + ".json")).read_text(encoding="utf-8"))
            check_update(samples, audit)
            check_physics(config, records)
            audit["minibatches"][0]["total_loss"] += 10
            with pytest.raises(AssertionError, match="total_loss"):
                check_update(samples, audit)
        store.finish()
    except BaseException as error:
        store.finish(error)
        raise


def test_service_validation_selection_and_failure(config, tmp_path):
    config["arrival_probability"] = 0
    config["validation"]["min_service_rate"] = .5
    trainer = Trainer(config, 23)
    trainer.calibrate()
    trainer.validate_cycle("zero")
    assert trainer.best_stage is None
    assert not trainer.validation_history[-1]["eligible"]
    config["validation"]["min_service_rate"] = None
    trainer = Trainer(config, 23)
    trainer.calibrate()
    trainer.validate_cycle("first")
    assert trainer.best_stage == "first"
    trainer.validate_cycle("equal")
    assert trainer.best_stage == "first"
    checkpoint = tmp_path / "selected.pt"
    trainer.save(checkpoint)
    restored = Trainer.load(checkpoint)
    assert restored.validation_history == trainer.validation_history


def test_zero_advantage_single_sample_update(config):
    from mist_sim.buffers import RolloutBuffer
    from mist_sim.ppo import update
    trainer = Trainer(config, 82)
    trainer.calibrate()
    sample = trainer.controller.upper.upper_action(Environment(config, generate_trace(config, 1)).upper_observation())[1]
    buffer = RolloutBuffer("upper", 0, 0)
    buffer.append(sample, sample["value"] + 1, 0, True, False)
    buffer.samples[0]["physical_rewards"] = [sample["value"] + 1]
    audit = {}
    settings = dict(config["ppo"], epochs=1)
    update(trainer.controller.upper, trainer.optimizers["upper"], buffer, settings, 0, 0, audit)
    assert audit["normalized_advantages"] == pytest.approx([1])


def test_greedy_reads_previous_complete_window(config):
    config = for_method(config, "greedy_consolidation")
    config["window"], config["history_slots"] = 4, 1
    environment = Environment(config, generate_trace(config, 88))
    for _ in range(4):
        problem = environment.begin_slot([1, 1])
        actions, rates, _ = greedy_fast(problem)
        environment.execute(actions, rates)
    assert len(environment.upper_observation()["history"]) == 4
    assert len(greedy_slow(environment.upper_observation())) == 2


def test_resume_preserves_selected_weights_reference(config, tmp_path):
    from types import SimpleNamespace
    from mist_sim.storage import write_json, digest_file
    config["arrival_probability"] = 0
    config["validation"]["min_service_rate"] = None
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_store = SimpleNamespace(path=first, write_json=lambda name, value: write_json(first / name, value), event=lambda *args, **kwargs: None)
    second_store = SimpleNamespace(path=second, write_json=lambda name, value: write_json(second / name, value), event=lambda *args, **kwargs: None)
    trainer = Trainer(config, 23, first_store)
    trainer.calibrate()
    trainer.validate_cycle("cycle_000")
    selected = Path(trainer.best_checkpoint_path)
    expected_hash = digest_file(selected)
    restored = Trainer.load(first / "checkpoints/last.pt", store=second_store)
    restored.validate_cycle("cycle_001")
    selection = json.loads((second / "selection.json").read_text(encoding="utf-8"))
    assert selection["best_stage"] == "cycle_000"
    assert selection["checkpoint_available"]
    assert selection["checkpoint_path"] == str(selected)
    assert selection["checkpoint_sha256"] == expected_hash
    selected_trainer = Trainer.load(selection["checkpoint_path"])
    assert selected_trainer.best_stage == "cycle_000"
    for name, value in trainer.controller.upper.state_dict().items():
        assert torch.equal(value, selected_trainer.controller.upper.state_dict()[name])
