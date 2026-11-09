"""Phase-conditioned on-policy PPO on the ordered system decision trajectory."""
import time
import copy

import torch

from .buffers import EventRolloutBuffer
from .diagnostics import policy_input_diagnostics, rollout_diagnostics
from .evaluation import summarize
from .policies.batching import replay_chunks
from .policies.networks import upper_frame, lower_frame
from .ppo import update
from .rewards import training_reward
from .schedule import TrainingLimit
from .traces import pack_behavior


def collect_events(trainer, count):
    buffer = EventRolloutBuffer(trainer.versions)
    records = []
    units = trainer.reward_normalizer.units()
    environment = trainer.fresh_environment()
    active = list(trainer.config["initial_active"])

    def append(sample, phase, reward, duration, terminated=False):
        if sample is None:
            raise RuntimeError(f"Joint {phase} policy did not return a training action")
        sample = dict(sample, phase=phase, episode_id=trainer.trace_counter,
                      slot=environment.slot - duration, reward_scale=units["scale"], reward_clip=units["clip"],
                      physical_rewards=[] if duration == 0 else [records[-1]["reward"]], next_phase=None)
        if buffer.samples and not buffer.samples[-1]["terminated"]:
            previous = buffer.samples[-1]
            previous.update(next_value=sample["value"], next_observation=sample["frame"], next_phase=phase)
        buffer.append(sample, reward, 0.0, terminated, False, duration)

    while len(records) < count:
        trainer.check_budget()
        if environment.done:
            environment = trainer.fresh_environment()
        if environment.slot % trainer.config["window"] == 0:
            active, sample = trainer.controller.slow(environment.upper_observation())
            append(sample, "upper", 0.0, 0)
        problem = environment.begin_slot(active)
        actions, predicted, sample = trainer.controller.fast(problem)
        record = environment.execute(actions, predicted)
        records.append(record)
        trainer.total_slots += 1
        append(sample, "lower", training_reward(record, units), 1, environment.done)
        if environment.done:
            trainer.training_episodes += 1
            metrics = summarize(environment.records)
            trainer.event("episode_completed", layer="joint", episode_id=trainer.trace_counter,
                episode_count=trainer.training_episodes, length=len(environment.records),
                raw_return=sum(row["reward"] for row in environment.records),
                energy=metrics["total_energy"], on_time_rate=metrics["on_time_rate"])
        if len(records) % 64 == 0:
            trainer.event("collect", layer="joint", samples=len(records), target_samples=count,
                          event_transitions=len(buffer.samples), collected_slots=len(records))
    if not environment.done:
        phase = "upper" if environment.slot % trainer.config["window"] == 0 else "lower"
        frame = (upper_frame(environment.upper_observation()) if phase == "upper" else
                 lower_frame(environment.begin_slot(active)))
        with torch.no_grad():
            value = float(getattr(trainer.controller, phase).value(frame))
        buffer.samples[-1].update(next_value=value, next_observation=frame, next_phase=phase, truncated=True)
    return buffer, records


def joint_phase(trainer, name, count):
    if trainer.phase_in_progress:
        raise RuntimeError("Recover from a completed-stage checkpoint after an interrupted phase")
    if trainer.max_slots is not None and trainer.total_slots + count > trainer.max_slots:
        raise TrainingLimit("interaction_limit_before_next_rollout")
    trainer.check_budget()
    if trainer.store and not trainer.minimal_checkpoints:
        trainer.save(trainer.store.path / "checkpoints" / "last.pt")
    trainer.phase_in_progress = True
    if not any(stage.endswith("_joint") for stage in trainer.completed_stages):
        for parameter, state in trainer.optimizers["lower"].state.items():
            trainer.optimizers["joint"].state[parameter] = copy.deepcopy(state)
    settings = trainer.config["ppo"]
    units = trainer.reward_normalizer.units()
    detailed = len(trainer.completed_stages) % trainer.config.get("diagnostics", {}).get("phase_interval", 1) == 0
    started = time.perf_counter()
    trainer.event("phase_started", stage=name, layer="joint", target_samples=count)
    buffer, records = trainer.collect("joint", count)
    collected = time.perf_counter()
    diagnostic_started = time.perf_counter()
    rollout = rollout_diagnostics(buffer, settings)
    inputs = {phase: policy_input_diagnostics(policy, [sample for sample in buffer.samples if sample["phase"] == phase])
              for phase, policy in trainer.joint_policy.phase_policies.items()} if detailed else {}
    diagnostics_seconds = time.perf_counter() - diagnostic_started
    trace_started = time.perf_counter()
    if trainer.store and detailed:
        for filename, content in (("behavior", pack_behavior(buffer.samples)), ("inputs", inputs), ("rollout", rollout)):
            trainer.store.write_json(f"traces/phases/{name}/{filename}.json", content)
    trainer.event("rollout_ready", stage=name, layer="joint", rollout=rollout)
    trace_seconds = time.perf_counter() - trace_started
    probability_started = time.perf_counter()
    with torch.no_grad():
        difference = float((replay_chunks(trainer.joint_policy, buffer.samples, settings["minibatch"]) -
                            torch.tensor([sample["logprob"] for sample in buffer.samples])).abs().max())
    if difference > 1e-5:
        raise AssertionError("Joint behavior probability changed before update")
    probability_seconds = time.perf_counter() - probability_started
    audit = {} if detailed else None
    event_counts = {phase: sum(sample["phase"] == phase for sample in buffer.samples) for phase in ("upper", "lower")}
    update_started = time.perf_counter()
    trainer.event("update_started", stage=name, samples=len(buffer.samples))

    def report_epoch(summary):
        trainer.epoch_counts["joint"] += 1
        trainer.event("ppo_epoch", stage=name, layer="joint", layer_epoch=trainer.epoch_counts["joint"], **summary)

    snapshot = tuple(trainer.versions[phase] for phase in ("upper", "lower"))
    diagnostics = update(trainer.joint_policy, trainer.optimizers["joint"], buffer, settings,
        snapshot, snapshot, audit, trainer.check_budget, report_epoch, device=trainer.config["device"])
    diagnostics["training_event_reward_mean"] = diagnostics["training_reward_mean"]
    diagnostics["training_reward_mean"] = sum(training_reward(record, units) for record in records) / len(records)
    updated = time.perf_counter()
    if trainer.store and detailed:
        trainer.store.write_json(f"traces/phases/{name}/update.json", audit)
    for phase in ("upper", "lower"):
        trainer.versions[phase] += 1
    raw_reward = sum(record["reward"] for record in records)
    metrics = dict(iteration=len(trainer.history) + 1, stage=name, layer="joint", event_counts=event_counts,
        physical_slots=len(records), raw_reward_sum=raw_reward, raw_reward_per_slot=raw_reward / len(records),
        detailed_trace=detailed, behavior_logprob_error=difference, reward_units=units,
        reward_clipped_slots=sum(abs(record["reward"] * units["scale"]) > units["clip"] for record in records)
            if units["clip"] is not None else 0,
        physical_cost_components={"conservative_energy": sum(record["conservative_energy"] for record in records),
            "weighted_delay_cost": trainer.config["lambda_time"] * sum(record["weighted_conservative_delay"] for record in records),
            "rejection_cost": sum(record["rejection_penalty"] for record in records)},
        input_diagnostics=inputs, rollout=rollout, diagnostics_seconds=diagnostics_seconds,
        trace_write_seconds=trace_seconds, probability_check_seconds=probability_seconds,
        collection_seconds=collected - started, update_seconds=updated - update_started,
        phase_seconds=time.perf_counter() - started,
        local_completed=sum(task["on_time"] and task["action"] == "local" for record in records for task in record["tasks"]),
        edge_completed=sum(task["on_time"] and task["action"] == "edge" for record in records for task in record["tasks"]),
        frozen_unchanged=None, parameters_changed=any(
            any(value > 0 for value in component["parameter_delta_l2"].values())
            for component in diagnostics["components"].values()), versions=dict(trainer.versions),
        total_slots=trainer.total_slots, **diagnostics, **summarize(records))
    return trainer.finish_phase(name, "joint", metrics, records, units)
