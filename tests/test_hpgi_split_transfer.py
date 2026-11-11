import copy

import pytest
import torch

from mist_sim.anonymity import generate_trace
from mist_sim.cli import main
from mist_sim.config import for_method, validate
from mist_sim.environment import Environment
from mist_sim.evaluation import evaluate, replay
from mist_sim.policies.hpgi import Controller
from mist_sim.policies.networks import lower_frame, upper_frame
from mist_sim.policies.batching import replay_batch
from mist_sim.policies.baselines.heuristics import greedy_fast
from mist_sim.policies.split_solver import proposal_problem, map_actions, refine_splits
from mist_sim.storage import write_json
from mist_sim.training import Trainer
from mist_sim.schedule import describe_schedule
from mist_sim.transfer import initialize_transfer, read_parent


def small(config):
    config = copy.deepcopy(config)
    config.update(horizon=5, k=1, arrival_probability=1.)
    config["ppo"].update(hidden=16, epochs=1, minibatch=4)
    config["training"].update(pretrain_slots=5, fast_slots=5, slow_intervals=2, cycles=1, calibration_episodes=1)
    config["validation"]["enabled"] = False
    config["runtime"]["collection_workers"] = 0
    config["monitoring"] = dict(console=False, tensorboard=False)
    config["diagnostics"]["phase_interval"] = 0
    return config


def hybrid(config, passes=1):
    return dict(for_method(config, "hpgi"), split_solver="fixed_proposal", split_refinement_passes=passes, dynamic_split=True)


def test_restricted_features_bootstrap_and_complete_policy_reproduce_fixed(config):
    torch.set_num_threads(1)
    fixed = for_method(small(config), "fixed_split")
    refined = hybrid(fixed, 0)
    first, second = Controller(fixed), Controller(refined)
    second.load_state_dict(first.state_dict())
    trace = generate_trace(fixed, 91)
    left, right = Environment(fixed, trace), Environment(refined, trace)
    while not left.done:
        assert upper_frame(left.upper_observation()) == upper_frame(right.upper_observation())
        active, _ = first.slow(left.upper_observation(), deterministic=True)
        assert second.slow(right.upper_observation(), deterministic=True)[0] == active
        a, b = left.begin_slot(active), right.begin_slot(active)
        assert lower_frame(a) == lower_frame(b)
        cached = lower_frame(b), float(second.lower.value(lower_frame(b)))
        rng = torch.get_rng_state()
        acts, _, sample = first.fast(a)
        torch.set_rng_state(rng)
        mapped, _, other = second.fast(b, cached=cached)
        assert sample["logprob"] == other["logprob"]
        assert sample["value"] == other["value"]
        assert sample["steps"] == other["steps"]
        ra, rb = left.execute(acts), right.execute(mapped)
        assert ra["cost"] == rb["cost"]
        assert ra["tasks"] == rb["tasks"]


@pytest.mark.parametrize("scope", ["splits", "assignment"])
def test_refinement_is_feasible_and_keeps_latent_probability(config, scope):
    config = hybrid(small(config))
    config["split_refinement_scope"] = scope
    env = Environment(config, generate_trace(config, 91))
    controller = Controller(config)
    while not env.done:
        problem = env.begin_slot([1] * len(config["servers"]))
        proposal = proposal_problem(problem)
        actions, _, _ = greedy_fast(proposal)
        mapped = map_actions(proposal, problem, actions)
        refined, detail = refine_splits(problem, mapped, 2)
        _, _, _, violations = problem.allocate(refined)
        assert not violations
        assert problem.scheduling_cost(refined) <= problem.scheduling_cost(mapped) + 1e-8
        assert detail["refined_cost"] == pytest.approx(problem.scheduling_cost(refined))
        for options, old, new in zip(problem.choices, mapped, refined):
            assert (options[old]["kind"], options[old]["server"]) == (options[new]["kind"], options[new]["server"])
        executed, _, sample = controller.fast(problem)
        assert sample["executed_actions"] == executed
        assert sample["actions"] == [step["action"] for step in sample["steps"]]
        assert float(controller.lower.replay(sample)[0]) == pytest.approx(sample["logprob"], abs=1e-5)
        assert float(replay_batch(controller.lower, [sample])[0][0]) == pytest.approx(sample["logprob"], abs=1e-5)
        env.execute(executed)
    replay(config, env.records)


def test_assignment_from_scratch_includes_calibration_pretraining_and_both_layers(config):
    torch.set_num_threads(1)
    target = hybrid(small(config))
    target["split_refinement_scope"] = "assignment"
    trainer = Trainer(target, 91)
    trainer.run()
    assert trainer.calibrated and "pretrain" in trainer.completed_stages
    assert trainer.transfer_provenance is None
    assert trainer.versions["upper"] >= 1 and trainer.versions["lower"] >= 2
    assert all(row["parameters_changed"] and row["frozen_unchanged"] for row in trainer.history)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_transfer_fresh_update_parallel_bootstrap_and_strict_resume(config, tmp_path, device):
    config = for_method(small(config), "fixed_split")
    parent = Trainer(config, 91)
    parent.run()
    path = tmp_path / "parent.pt"
    parent.save(path)
    target = hybrid(config)
    target["split_refinement_scope"] = "assignment"
    target["device"] = device
    target["runtime"]["collection_workers"] = 2 if device == "cuda" else 0
    new = initialize_transfer(path, target, 92)
    assert new.total_slots == 0 and new.versions == dict(upper=0, lower=0)
    assert new.completed_stages == ["pretrain"] and new.calibrated
    assert describe_schedule(target, inherited_pretraining=True)["pretrain_slots"] == 0
    assert describe_schedule(target, completed_stages=["pretrain"])["pretrain_slots"] == 5
    assert all(not optimizer.state for optimizer in new.optimizers.values())
    assert new.transfer_provenance["parent"]["inherited_training_slots"] == parent.total_slots
    filename = tmp_path / "config.json"
    write_json(filename, target)
    run = main(["train", "--config", str(filename), "--transfer-from", str(path), "--seed", "92",
                "--output-dir", str(tmp_path / "outputs"), "--run-id", "transfer"])
    loaded = Trainer.load(run / "checkpoints/last.pt")
    assert loaded.total_slots == 10
    assert loaded.transfer_provenance == new.transfer_provenance
    assert all(row["parameters_changed"] and row["frozen_unchanged"] for row in loaded.history)
    records, _ = evaluate(loaded.controller, loaded.config, generate_trace(loaded.config, 93))
    replay(loaded.config, records)
    altered = copy.deepcopy(target)
    altered["ppo"]["learning_rate"] *= .5
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.load(run / "checkpoints/last.pt", altered)


def test_transfer_rejects_physics_source_corruption_and_substream_reuse(config, tmp_path):
    config = for_method(small(config), "fixed_split")
    trainer = Trainer(config, 91)
    trainer.calibrate()
    path = tmp_path / "parent.pt"
    trainer.save(path)
    target = hybrid(config)
    with pytest.raises(ValueError, match="substream"):
        initialize_transfer(path, target, 10091)
    changed = copy.deepcopy(target)
    changed["local_rate"] *= 2
    with pytest.raises(ValueError, match="contract"):
        initialize_transfer(path, changed, 92)
    payload = torch.load(path, weights_only=False)
    payload["implementation_signature"] = "unknown"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="audited"):
        read_parent(path, "made-up")
    with pytest.raises(ValueError, match="split solver"):
        validate(dict(target, split_refinement_passes=3))
