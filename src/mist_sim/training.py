"""Three-stage training; all samples are fresh after an opponent update."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .anonymity import generate_trace
from .environment import Environment
from .buffers import RolloutBuffer
from .evaluation import evaluate, summarize
from .policies.hpgi import Controller
from .policies.networks import upper_frame, lower_frame
from .ppo import update
from .traces import pack_behavior
from .policies.batching import replay_chunks
from .config import validate, layer_settings
from .storage import digest_file
from .checkpoints import trim_snapshots
from .rewards import training_reward, RunningRewardNormalizer
from .schedule import TrainingLimit, rollout_slots, describe_schedule

CHECKPOINT_SCHEMA = 2


def implementation_signature():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def state_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()


class Trainer:
    def __init__(self, config, seed, store=None):
        validate(config)
        if type(seed) is not int or seed < 0 or seed in (config["seeds"]["validation"], config["seeds"]["test"]):
            raise ValueError("Training seed must differ from validation/test seeds")
        self.config, self.seed, self.store = copy.deepcopy(config), seed, store
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(config.get("runtime", {}).get("torch_threads", 1))
        self.controller = Controller(config)
        self.reward_normalizer = RunningRewardNormalizer(config)
        self.value_reward_scales = {"upper": None, "lower": None}
        self.optimizers = {layer: torch.optim.Adam(getattr(self.controller, layer).parameters(),
                                                 lr=layer_settings(config, layer)["learning_rate"],
                                                 eps=layer_settings(config, layer)["adam_epsilon"])
                           for layer in ("upper", "lower")}
        self.versions = {"upper": 0, "lower": 0}
        self.trace_counter = 0
        self.completed_stages = []
        self.history = []
        self.total_slots = 0
        self.calibrated = False
        self.phase_in_progress = False
        self.validation_history = []
        self.best_validation_cost = None
        self.best_stage = None
        self.best_checkpoint_path = None
        self.deadline = None
        self.max_slots = None
        self.epoch_counts = {"upper": 0, "lower": 0}
        self.training_episodes = 0

    def check_budget(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TrainingLimit("wall_time_limit")

    def event(self, event, **fields):
        if self.store:
            values = {"total_slots": self.total_slots} | fields
            if event == "episode_completed":
                self.store.record_episode(**values)
            elif event == "validation_completed":
                self.store.record_validation(**values)
            else:
                self.store.event(event, **values)

    def fresh_environment(self):
        trace_seed = self.seed + 10000 * self.trace_counter
        if trace_seed in self.validation_seeds() or trace_seed == self.config["seeds"]["test"]:
            raise ValueError("Training substream collides with validation/test seed")
        self.trace_counter += 1
        trace = generate_trace(self.config, trace_seed)
        if self.store:
            self.store.write_json(f"traces/inputs/input_{self.trace_counter:05d}.json", {"seed": trace_seed, "trace": trace})
        return Environment(self.config, trace)

    def validation_seeds(self):
        return [self.config["seeds"]["validation"] + 10000 * index
                for index in range(self.config.get("validation", {}).get("trajectories", 1))]

    def evaluate_validation(self):
        records, per_trajectory = [], []
        for seed in self.validation_seeds():
            self.check_budget()
            trace = generate_trace(self.config, seed)
            trajectory, metrics = evaluate(self.controller, self.config, trace, check_budget=self.check_budget)
            records.extend(trajectory)
            per_trajectory.append(dict(seed=seed, metrics=metrics))
            self.event("validation_progress", completed=len(per_trajectory), total=len(self.validation_seeds()))
        return records, summarize(records), per_trajectory

    def evaluate_sampled_validation(self):
        """Paired requests and fixed action seeds; never advance training RNG."""
        records, per_trajectory = [], []
        seeds = self.validation_seeds()
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                self.check_budget()
                torch.manual_seed(seed)
                trace = generate_trace(self.config, seed)
                trajectory, metrics = evaluate(self.controller, self.config, trace,
                                               check_budget=self.check_budget, deterministic=False)
                records.extend(trajectory)
                per_trajectory.append(dict(seed=seed, action_seed=seed, metrics=metrics))
                self.event("validation_progress", mode="sampled", completed=len(per_trajectory), total=len(seeds))
        return records, summarize(records), per_trajectory

    def calibrate(self):
        if self.calibrated:
            return
        upper_frames, lower_frames = [], []
        upper_steps, lower_steps = [], []
        for episode in range(self.config["training"]["calibration_episodes"]):
            environment = self.fresh_environment()
            active = [1] * len(self.config["servers"])
            while not environment.done:
                self.check_budget()
                upper_frames.append(upper_frame(environment.upper_observation()))
                proposed, upper_sample = self.controller.upper.upper_action(environment.upper_observation())
                upper_steps.extend(upper_sample["steps"])
                if episode % 2 and environment.slot % self.config["window"] == 0:
                    active = proposed
                problem = environment.begin_slot(active)
                lower_frames.append(lower_frame(problem))
                actions, predicted, lower_sample = self.controller.lower.lower_action(problem)
                lower_steps.extend(lower_sample["steps"])
                environment.execute(actions, predicted)
            self.reward_normalizer.observe(environment.records)
        self.controller.upper.fit_normalizers(upper_frames, upper_steps)
        self.controller.lower.fit_normalizers(lower_frames, lower_steps)
        self.calibrated = True

    def collect(self, layer, count, pretrain=False):
        units = self.reward_normalizer.units()
        previous_scale = self.value_reward_scales[layer]
        if previous_scale is not None:
            getattr(self.controller, layer).value_scale.mul_(units["scale"] / previous_scale)
        self.value_reward_scales[layer] = units["scale"]
        opponent = "lower" if layer == "upper" else "upper"
        buffer = RolloutBuffer(layer, self.versions[layer], self.versions[opponent])
        records = []
        environment = self.fresh_environment()
        active = list(self.config["initial_active"])
        problem = None
        cached = None
        while len(buffer.samples) < count:
            self.check_budget()
            if environment.done:
                environment = self.fresh_environment()
                problem = None
                cached = None
            episode_id, start_slot = self.trace_counter, environment.slot
            if layer == "upper":
                observation = environment.upper_observation()
                active, sample = self.controller.slow(observation, pretrain=pretrain, cached=cached)
                if sample is None:
                    raise RuntimeError("No learned upper policy")
                interval_records = []
                for _ in range(min(self.config["window"], self.config["horizon"] - environment.slot)):
                    self.check_budget()
                    problem = environment.begin_slot(active)
                    actions, predicted, _ = self.controller.fast(problem, record=False)
                    interval_records.append(environment.execute(actions, predicted))
                    self.total_slots += 1
                duration = len(interval_records)
                reward = sum(self.config["ppo"]["gamma"] ** offset * training_reward(record, units)
                             for offset, record in enumerate(interval_records))
                next_frame = None if environment.done else upper_frame(environment.upper_observation())
                with torch.no_grad():
                    next_value = 0.0 if environment.done else float(self.controller.upper.value(next_frame))
                buffer.append(sample, reward, next_value, environment.done,
                              not environment.done and len(buffer.samples) + 1 == count, duration, next_frame)
                buffer.samples[-1]["physical_rewards"] = [record["reward"] for record in interval_records]
                records.extend(interval_records)
            else:
                if problem is None:
                    if environment.slot % self.config["window"] == 0:
                        active, _ = self.controller.slow(environment.upper_observation(), pretrain=pretrain, record=False)
                    problem = environment.begin_slot(active)
                actions, predicted, sample = self.controller.fast(problem, cached=cached)
                if sample is None:
                    raise RuntimeError("No learned lower policy")
                record = environment.execute(actions, predicted)
                self.total_slots += 1
                records.append(record)
                if environment.done:
                    next_frame, next_value = None, 0.0
                    problem = None
                else:
                    if environment.slot % self.config["window"] == 0:
                        active, _ = self.controller.slow(environment.upper_observation(), pretrain=pretrain, record=False)
                    problem = environment.begin_slot(active)
                    next_frame = lower_frame(problem)
                    with torch.no_grad():
                        next_value = float(self.controller.lower.value(next_frame))
                buffer.append(sample, training_reward(record, units), next_value, environment.done,
                              not environment.done and len(buffer.samples) + 1 == count, 1, next_frame)
                buffer.samples[-1]["physical_rewards"] = [record["reward"]]
            cached = None if environment.done else (next_frame, next_value)
            buffer.samples[-1]["reward_scale"] = units["scale"]
            buffer.samples[-1]["reward_clip"] = units["clip"]
            buffer.samples[-1]["episode_id"] = episode_id
            buffer.samples[-1]["slot"] = start_slot
            if environment.done:
                self.training_episodes += 1
                episode_metrics = summarize(environment.records)
                self.event("episode_completed", layer=layer, episode_id=episode_id,
                           episode_count=self.training_episodes, length=len(environment.records),
                           raw_return=sum(record["reward"] for record in environment.records),
                           energy=episode_metrics["total_energy"], on_time_rate=episode_metrics["on_time_rate"])
            if len(buffer.samples) % 64 == 0:
                self.event("collect", layer=layer, samples=len(buffer.samples), target_samples=count,
                           collected_slots=len(records))
        return buffer, records

    def phase(self, name, layer, count, pretrain=False):
        if self.phase_in_progress:
            raise RuntimeError("Recover from a completed-stage checkpoint after an interrupted phase")
        if self.max_slots is not None and self.total_slots + rollout_slots(self.config, layer, count) > self.max_slots:
            raise TrainingLimit("interaction_limit_before_next_rollout")
        self.check_budget()
        if self.store:
            self.save(self.store.path / "checkpoints" / "last.pt")
        self.phase_in_progress = True
        opponent = "lower" if layer == "upper" else "upper"
        frozen = getattr(self.controller, opponent)
        policy = getattr(self.controller, layer)
        before = parameter_hash(policy)
        frozen_before = state_hash(frozen)
        units = self.reward_normalizer.units()
        started = time.perf_counter()
        self.event("phase_started", stage=name, layer=layer, target_samples=count)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        try:
            buffer, records = self.collect(layer, count, pretrain)
            collected = time.perf_counter()
            from .diagnostics import policy_input_diagnostics, rollout_diagnostics
            diagnostics_started = time.perf_counter()
            input_diagnostics = policy_input_diagnostics(policy, buffer.samples)
            rollout = rollout_diagnostics(buffer, layer_settings(self.config, layer))
            diagnostics_seconds = time.perf_counter() - diagnostics_started
            trace_started = time.perf_counter()
            if self.store:
                self.store.write_json(f"traces/phases/{name}/behavior.json", pack_behavior(buffer.samples))
                self.store.write_json(f"traces/phases/{name}/inputs.json", input_diagnostics)
                self.store.write_json(f"traces/phases/{name}/rollout.json", rollout)
            self.event("rollout_ready", stage=name, layer=layer, rollout=rollout)
            trace_write_seconds = time.perf_counter() - trace_started
            probability_started = time.perf_counter()
            with torch.no_grad():
                differences = (replay_chunks(policy, buffer.samples, layer_settings(self.config, layer)["minibatch"]) -
                               torch.tensor([sample["logprob"] for sample in buffer.samples])).abs().tolist()
            probability_check_seconds = time.perf_counter() - probability_started
            if max(differences, default=0) > 1e-5:
                raise AssertionError("Behavior probability changed before update")
            audit = {}
            update_started = time.perf_counter()
            self.event("update_started", stage=name, samples=len(buffer.samples))
            def report_epoch(summary):
                self.epoch_counts[layer] += 1
                self.event("ppo_epoch", stage=name, layer=layer, layer_epoch=self.epoch_counts[layer], **summary)
            diagnostics = update(policy, self.optimizers[layer], buffer, layer_settings(self.config, layer),
                                 self.versions[layer], self.versions[opponent], audit, self.check_budget, report_epoch)
            updated = time.perf_counter()
            if self.store:
                self.store.write_json(f"traces/phases/{name}/update.json", audit)
            self.versions[layer] += 1
            if state_hash(frozen) != frozen_before:
                raise AssertionError("Frozen opponent changed")
        finally:
            for parameter in frozen.parameters():
                parameter.requires_grad_(True)
        metrics = dict(iteration=len(self.history) + 1, stage=name, layer=layer,
                       physical_slots=len(records), raw_reward_sum=sum(record["reward"] for record in records),
                       raw_reward_per_slot=sum(record["reward"] for record in records) / len(records),
                       behavior_logprob_error=max(differences, default=0),
                       reward_units=units,
                       reward_clipped_slots=sum(abs(record["reward"] * units["scale"]) > units["clip"]
                                                for record in records) if units["clip"] is not None else 0,
                       physical_cost_components={"conservative_energy": sum(record["conservative_energy"] for record in records),
                           "weighted_delay_cost": self.config["lambda_time"] * sum(record["weighted_conservative_delay"] for record in records),
                           "rejection_cost": sum(record["rejection_penalty"] for record in records)},
                       input_diagnostics=input_diagnostics,
                       rollout=rollout,
                       diagnostics_seconds=diagnostics_seconds, trace_write_seconds=trace_write_seconds,
                       probability_check_seconds=probability_check_seconds,
                       collection_seconds=collected - started, update_seconds=updated - update_started,
                       phase_seconds=time.perf_counter() - started,
                       local_completed=sum(task["on_time"] and task["action"] == "local" for record in records for task in record["tasks"]),
                       edge_completed=sum(task["on_time"] and task["action"] == "edge" for record in records for task in record["tasks"]),
                       frozen_unchanged=True, parameters_changed=parameter_hash(policy) != before, versions=dict(self.versions),
                       total_slots=self.total_slots, **diagnostics, **summarize(records))
        self.completed_stages.append(name)
        self.history.append({key: value for key, value in metrics.items()
                             if key not in ("input_diagnostics", "rollout", "durations")} |
                            {"rollout": {key: value for key, value in rollout.items() if key != "examples"}})
        self.reward_normalizer.observe(records)
        self.phase_in_progress = False
        if self.store:
            self.store.write_json(f"traces/phases/{name}/physical.json", records)
            self.store.append_metrics(metrics)
            self.save(self.store.path / "checkpoints" / "last.pt")
            policy = self.config.get("checkpoint_policy", {"stage_interval": 0, "keep_stages": 2})
            if policy["stage_interval"] and len(self.completed_stages) % policy["stage_interval"] == 0:
                directory = self.store.path / "checkpoints"
                self.save(directory / f"snapshot_{name}.pt")
                trim_snapshots(directory, "snapshot_*.pt", policy["keep_stages"])
        self.event("phase_completed", stage=name, layer=layer, service=metrics["on_time_rate"],
                   loss=metrics["loss"],
                   activation_probability=metrics.get("activation_probability"),
                   energy=metrics["total_energy"], cost=metrics["conservative_cost"],
                   joint_kl=metrics["post_update_joint_kl"], frozen_unchanged=True,
                   raw_reward_per_slot=sum(record["reward"] for record in records) / len(records),
                   training_reward_mean=metrics["training_reward_mean"], explained_variance=metrics["explained_variance"],
                   reward_scale=units["scale"], reward_clipped_slots=metrics["reward_clipped_slots"],
                   collection_seconds=metrics["collection_seconds"], update_seconds=metrics["update_seconds"])
        return metrics

    def run(self, extra_cycles=0, max_seconds=None, max_slots=None):
        stopping = self.config.get("stopping", {})
        max_seconds = max_seconds if max_seconds is not None else stopping.get("max_seconds")
        max_slots = max_slots if max_slots is not None else stopping.get("max_slots")
        if max_slots is not None and (type(max_slots) is not int or max_slots <= 0):
            raise ValueError("max_slots must be a positive integer")
        self.max_slots = max_slots
        self.deadline = None
        if max_seconds is not None:
            if not np.isfinite(max_seconds) or max_seconds <= 0:
                raise ValueError("Training wall-time limit must be positive and finite")
            self.deadline = time.perf_counter() + max_seconds
        if type(extra_cycles) is not int or extra_cycles < 0:
            raise ValueError("extra_cycles must be a nonnegative integer")
        if not (self.controller.learn_lower or self.controller.learn_upper):
            raise ValueError("Heuristic baseline uses evaluate, not train")
        self.event("training_started", schedule=describe_schedule(self.config, extra_cycles),
                   resumed_stages=len(self.completed_stages), max_seconds=max_seconds, max_slots=max_slots)
        if self.store:
            self.save(self.store.path / "checkpoints" / "last.pt")
        self.calibrate()
        self.event("calibration_completed", episodes=self.config["training"]["calibration_episodes"])
        if self.store and not self.completed_stages:
            self.save(self.store.path / "checkpoints" / "initial.pt")
        if not self.validation_history:
            self.validate_cycle("initial")
        if self.controller.learn_lower and "pretrain" not in self.completed_stages:
            total = self.config["training"]["pretrain_slots"]
            chunk_size = self.config["training"]["fast_slots"]
            for chunk, offset in enumerate(range(0, total, chunk_size)):
                count = min(chunk_size, total - offset)
                name = "pretrain" if offset + count == total else f"pretrain_{chunk:03d}"
                if name not in self.completed_stages:
                    self.phase(name, "lower", count, pretrain=True)
        if self.controller.learn_lower and "pretrain" not in [item["stage"] for item in self.validation_history]:
            self.validate_cycle("pretrain")
        finished_cycles = sum(item["stage"].startswith("cycle_") for item in self.validation_history)
        target = max(self.config["training"]["cycles"], finished_cycles) + extra_cycles
        for cycle in range(target):
            for layer, enabled, count in (
                ("upper", self.controller.learn_upper, self.config["training"]["slow_intervals"]),
                ("lower", self.controller.learn_lower, self.config["training"]["fast_slots"])):
                name = f"cycle_{cycle:03d}_{layer}"
                if enabled and name not in self.completed_stages:
                    self.phase(name, layer, count)
            validation_stage = f"cycle_{cycle:03d}"
            if validation_stage not in [item["stage"] for item in self.validation_history]:
                self.validate_cycle(validation_stage)
        records, metrics, per_trajectory = self.evaluate_validation()
        if self.store:
            self.store.write_json("traces/validation/final.json", records)
            self.store.write_json("validation_metrics.json", metrics)
            self.store.write_json("validation_trajectories.json", per_trajectory)
            self.save(self.store.path / "checkpoints" / "final.pt")
        self.event("training_completed", reason="configured_cycles_completed")
        return self.history

    def validate_cycle(self, stage):
        from .diagnostics import check_physics
        self.event("validation_started", stage=stage, trajectories=len(self.validation_seeds()))
        started = time.perf_counter()
        records, metrics, per_trajectory = self.evaluate_validation()
        check_physics(self.config, records)
        minimum = self.config.get("validation", {}).get("min_service_rate")
        service_ok = minimum is None or (metrics["on_time_rate"] is not None and metrics["on_time_rate"] >= minimum)
        improved = service_ok and (self.best_validation_cost is None or metrics["conservative_cost"] < self.best_validation_cost)
        record = {"stage": stage, "metrics": metrics, "hard_constraints_valid": True,
                  "total_slots": self.total_slots, "trajectories": per_trajectory,
                  "measured_seconds": time.perf_counter() - started,
                  "service_threshold": minimum, "service_requirement_specified": minimum is not None,
                  "eligible": service_ok, "selected": improved}
        sampled_fields = {}
        if self.config["validation"].get("sampled", False):
            sampled_records, sampled_metrics, sampled_trajectories = self.evaluate_sampled_validation()
            check_physics(self.config, sampled_records)
            sampled_ok = minimum is None or (sampled_metrics["on_time_rate"] is not None
                                              and sampled_metrics["on_time_rate"] >= minimum)
            record.update(sampled_metrics=sampled_metrics, sampled_eligible=sampled_ok,
                          sampled_trajectories=sampled_trajectories)
            sampled_fields = dict(sampled_cost_per_slot=sampled_metrics["conservative_cost"] / len(sampled_records),
                                  sampled_return_per_episode=sum(r["reward"] for r in sampled_records) / len(sampled_trajectories),
                                  sampled_service=sampled_metrics["on_time_rate"],
                                  sampled_active_servers=sampled_metrics["active_server_slots"] / len(sampled_records))
            if self.store:
                from .results import evaluation_row
                self.store.write_json(f"traces/validation/{stage}_sampled.json", sampled_records)
                for trajectory in sampled_trajectories:
                    self.store.append_result("validation_sampled_trajectories.csv", dict(
                        stage=stage, total_slots=self.total_slots, trajectory_seed=trajectory["seed"],
                        action_seed=trajectory["action_seed"], **evaluation_row(trajectory["metrics"])))
        record["measured_seconds"] = time.perf_counter() - started
        if improved:
            self.best_validation_cost, self.best_stage = metrics["conservative_cost"], stage
            self.best_checkpoint_path = str((self.store.path / "checkpoints" / f"selected_{stage}.pt").resolve()) if self.store else None
        self.validation_history.append(record)
        if self.store:
            self.store.write_json(f"traces/validation/{stage}.json", records)
            self.store.write_json("validation_history.json", self.validation_history)
            from .results import evaluation_row
            for trajectory in per_trajectory:
                self.store.append_result("validation_trajectories.csv", dict(
                    iteration=len(self.history), stage=stage, total_slots=self.total_slots,
                    trajectory_seed=trajectory["seed"], **evaluation_row(trajectory["metrics"])))
            if improved:
                self.save(self.best_checkpoint_path)
                self.save(self.store.path / "checkpoints" / "best.pt")
                directory = self.store.path / "checkpoints"
                protected = [self.best_checkpoint_path]
                for snapshot in directory.glob("snapshot_*.pt"):
                    selected = torch.load(snapshot, map_location="cpu", weights_only=False).get("best_checkpoint_path")
                    if selected and Path(selected).parent.resolve() == directory.resolve():
                        protected.append(selected)
                trim_snapshots(directory, "selected_*.pt", len(set(protected)), protected)
            selected_path = Path(self.best_checkpoint_path) if self.best_checkpoint_path else None
            self.store.write_json("selection.json", {"best_stage": self.best_stage,
                                  "status": "selected" if self.best_stage else "no_eligible_checkpoint",
                                  "checkpoint_path": self.best_checkpoint_path,
                                  "checkpoint_available": selected_path is not None and selected_path.is_file(),
                                  "checkpoint_sha256": digest_file(selected_path) if selected_path is not None and selected_path.is_file() else None,
                                  "service_requirement_specified": minimum is not None})
            self.save(self.store.path / "checkpoints" / "last.pt")
        self.event("validation_completed", iteration=len(self.history), stage=stage, service=metrics["on_time_rate"],
                   cycle_index=int(stage.split("_")[1]) + 1 if stage.startswith("cycle_") else 0,
                   cost=metrics["conservative_cost"], eligible=service_ok, selected=improved,
                   cost_per_slot=metrics["conservative_cost"] / len(records),
                   return_per_episode=sum(record["reward"] for record in records) / len(per_trajectory),
                   energy_per_slot=metrics["total_energy"] / len(records),
                   validation_seconds=record["measured_seconds"], **sampled_fields)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": CHECKPOINT_SCHEMA, "config": self.config, "signature": signature(self.config),
                   "implementation_signature": implementation_signature(), "resumable": not self.phase_in_progress,
                   "seed": self.seed, "controller": self.controller.state_dict(),
                   "reward_normalizer": self.reward_normalizer.state_dict(),
                   "value_reward_scales": self.value_reward_scales,
                   "optimizers": {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()},
                   "versions": self.versions, "trace_counter": self.trace_counter,
                   "completed_stages": self.completed_stages, "history": self.history,
                   "total_slots": self.total_slots, "calibrated": self.calibrated,
                   "epoch_counts": self.epoch_counts, "training_episodes": self.training_episodes,
                   "validation_history": self.validation_history, "best_validation_cost": self.best_validation_cost,
                   "best_stage": self.best_stage, "best_checkpoint_path": self.best_checkpoint_path,
                   "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
        temporary = path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, config=None, store=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError("Incompatible checkpoint schema")
        if payload.get("implementation_signature") != implementation_signature():
            raise ValueError("Checkpoint implementation changed; use its recorded source version")
        if not payload.get("resumable", False):
            raise ValueError("Interrupted phase snapshot is diagnostic only; load last completed or initial checkpoint")
        expected = config if config is not None else payload["config"]
        if signature(expected) != payload["signature"]:
            raise ValueError("Checkpoint/config/input mismatch")
        trainer = cls(expected, payload["seed"], store)
        trainer.controller.load_state_dict(payload["controller"])
        trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
        trainer.value_reward_scales = payload["value_reward_scales"]
        for key, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][key])
        for key in ("versions", "trace_counter", "completed_stages", "history", "total_slots", "calibrated", "epoch_counts", "training_episodes",
                    "validation_history", "best_validation_cost", "best_stage", "best_checkpoint_path"):
            setattr(trainer, key, payload[key])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        return trainer
