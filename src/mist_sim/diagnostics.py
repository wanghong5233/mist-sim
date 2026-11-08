"""Independent arithmetic checks of saved physical and PPO intermediate values."""
import json
import math
from pathlib import Path
import numpy as np


def close(actual, expected, name, atol=2e-5):
    if not np.allclose(actual, expected, rtol=2e-5, atol=atol, equal_nan=False):
        raise AssertionError(f"{name}: {actual} != {expected}")


def rollout_diagnostics(buffer, settings):
    samples = buffer.samples
    if not samples:
        raise ValueError("Cannot describe an empty rollout")
    gamma = settings["gamma"]
    advantages, targets = buffer.advantages(gamma, settings["gae_lambda"])
    normalized = ((advantages - advantages.mean()) / max(advantages.std(), 1e-8)
                  if len(samples) > 1 else advantages.copy())
    segment_returns = np.zeros(len(samples), dtype=np.float64)
    for index in reversed(range(len(samples))):
        sample = samples[index]
        if sample["terminated"]:
            continuation = 0.0
        elif sample["truncated"] or index == len(samples) - 1:
            continuation = sample["next_value"]
        else:
            continuation = segment_returns[index + 1]
        segment_returns[index] = sample["reward"] + gamma ** sample["duration"] * continuation

    def shapes(frame):
        if frame is None:
            return None
        return {"items": [len(frame["items"]), 19], "servers": [len(frame["servers"]), 13],
                "context": [len(frame["context"])]}

    indices = {0, len(samples) - 1}
    for predicate in (lambda sample: sample["terminated"], lambda sample: sample["truncated"],
                      lambda sample: bool(sample["steps"])):
        match = next((index for index, sample in enumerate(samples) if predicate(sample)), None)
        if match is not None:
            indices.add(match)
    examples = []
    for index in sorted(indices):
        sample = samples[index]
        examples.append(dict(index=index, episode_id=sample.get("episode_id"), slot=sample.get("slot"),
            observation_shapes=shapes(sample["frame"]), next_observation_shapes=shapes(sample["next_observation"]),
            context=sample["frame"]["context"],
            first_item=sample["frame"]["items"][0] if sample["frame"]["items"] else None,
            next_context=sample["next_observation"]["context"] if sample["next_observation"] is not None else None,
            actions=[step["action"] for step in sample["steps"]],
            legal_choices=[sum(step["mask"]) for step in sample["steps"]],
            physical_rewards=sample["physical_rewards"], reward=sample["reward"],
            value=sample["value"], next_value=sample["next_value"], duration=sample["duration"],
            terminated=sample["terminated"], truncated=sample["truncated"],
            td_error=sample["reward"] + gamma ** sample["duration"] * sample["next_value"] *
                     (not sample["terminated"]) - sample["value"],
            advantage=float(advantages[index]), normalized_advantage=float(normalized[index]),
            gae_value_target=float(targets[index]), segment_return_with_bootstrap=float(segment_returns[index])))
    return dict(transitions=len(samples), physical_slots=sum(sample["duration"] for sample in samples),
        episode_segments=len({sample.get("episode_id") for sample in samples}),
        terminated=sum(sample["terminated"] for sample in samples),
        truncated=sum(sample["truncated"] for sample in samples),
        duration_range=[min(sample["duration"] for sample in samples), max(sample["duration"] for sample in samples)],
        decisions=sum(len(sample["steps"]) for sample in samples),
        empty_action_transitions=sum(not sample["steps"] for sample in samples),
        minibatches_per_epoch=math.ceil(len(samples) / settings["minibatch"]),
        max_optimizer_steps=math.ceil(len(samples) / settings["minibatch"]) * settings["epochs"],
        behavior_version=buffer.behavior_version, opponent_version=buffer.opponent_version,
        advantage_mean=float(advantages.mean()), advantage_std=float(advantages.std()),
        normalized_advantage_mean=float(normalized.mean()), normalized_advantage_std=float(normalized.std()),
        examples=examples)


def policy_input_diagnostics(policy, samples, limit=16):
    import torch
    selected = [samples[index] for index in np.linspace(0, len(samples) - 1, min(limit, len(samples)), dtype=int)]
    rows = {
        "item": [row for sample in selected for row in sample["frame"]["items"]] +
                [row for sample in selected for step in sample["steps"] for row in step["features"]],
        "server": [row for sample in selected for row in sample["frame"]["servers"]] +
                  [row for sample in selected for step in sample["steps"] if "prefix_frame" in step for row in step["prefix_frame"]["servers"]],
        "context": [sample["frame"]["context"] for sample in selected],
        "prefix": [step["prefix"] for sample in selected for step in sample["steps"]]}
    features = {}
    with torch.no_grad():
        for name, values in rows.items():
            if not values:
                features[name] = {"rows": 0}
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            normalizer = getattr(policy, name + "_norm")
            standardized = (tensor - normalizer.mean) / normalizer.scale
            normalized = normalizer(tensor)
            features[name] = dict(rows=len(values), raw_min=tensor.min(0).values.tolist(),
                raw_max=tensor.max(0).values.tolist(), standardized_max_abs=standardized.abs().max(0).values.tolist(),
                network_max_abs=normalized.abs().max(0).values.tolist(),
                near_boundary_fraction=float((normalized.abs() > .95).float().mean()),
                mean=normalizer.mean.tolist(), scale=normalizer.scale.tolist())
    activations = {}
    handles = []
    def hook(name):
        def record(module, inputs, output):
            counts = activations.setdefault(name, [0, 0])
            counts[0] += output.numel()
            counts[1] += int((output.detach().abs() > .95).sum())
        return record
    try:
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Tanh):
                handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            for sample in selected:
                policy.replay(sample)
    finally:
        for handle in handles:
            handle.remove()
    decisions = [step for sample in samples for step in sample["steps"]]
    item_counts = [len(sample["frame"]["items"]) for sample in samples]
    candidate_counts = [len(step["features"]) for step in decisions]
    linear_layers = {name: {"weight": list(module.weight.shape), "bias": list(module.bias.shape),
                            "input_width": module.in_features, "output_width": module.out_features}
                     for name, module in policy.named_modules() if isinstance(module, torch.nn.Linear)}
    return dict(sampled_frames=len(selected), total_frames=len(samples), features=features,
        tensors={"dtype": "float32", "device": "cpu", "frame_items_rows": [min(item_counts), max(item_counts)],
                 "frame_items_width": policy.item_norm.mean.numel(),
                 "servers": [len(samples[0]["frame"]["servers"]), policy.server_norm.mean.numel()],
                 "context": [policy.context_norm.mean.numel()], "prefix": [policy.prefix_norm.mean.numel()],
                 "decision_candidate_rows": [min(candidate_counts), max(candidate_counts)] if candidate_counts else [0, 0]},
        linear_layers=linear_layers, parameter_count=sum(parameter.numel() for parameter in policy.parameters()),
        tanh_saturation={name: counts[1] / counts[0] for name, counts in activations.items()},
        decisions=len(decisions), forced_decisions=sum(sum(step["mask"]) == 1 for step in decisions),
        legal_candidates=sum(sum(step["mask"]) for step in decisions),
        masked_candidates=sum(len(step["mask"]) - sum(step["mask"]) for step in decisions),
        empty_physical_samples=sum(not sample["steps"] for sample in samples))


def check_update(samples, audit):
    settings = audit["settings"]
    gamma, lam = settings["gamma"], settings["gae_lambda"]
    targets, deltas = [], []
    for sample in samples:
        scaled = np.asarray(sample["physical_rewards"]) * sample.get("reward_scale", 1.0)
        if sample.get("reward_clip") is not None:
            scaled = np.clip(scaled, -sample["reward_clip"], sample["reward_clip"])
        expected_reward = sum(gamma ** offset * reward for offset, reward in enumerate(scaled))
        close(sample["reward"], expected_reward, "physical interval reward")
        if len(sample["physical_rewards"]) != sample["duration"]:
            raise AssertionError("Duration/reward mismatch")
        delta = sample["reward"] - sample["value"]
        if not sample["terminated"]:
            delta += gamma ** sample["duration"] * sample["next_value"]
        deltas.append(delta)
    for start, sample in enumerate(samples):
        total, coefficient = 0.0, 1.0
        for index in range(start, len(samples)):
            total += coefficient * deltas[index]
            if samples[index]["terminated"] or samples[index]["truncated"]:
                break
            coefficient *= gamma ** samples[index]["duration"] * lam
        targets.append(total)
    close(audit["td_errors"], deltas, "TD")
    close(audit["advantages"], targets, "forward-sum GAE")
    returns = np.asarray(targets) + [sample["value"] for sample in samples]
    close(audit["returns"], returns, "returns")
    normalized = np.asarray(targets)
    if len(normalized) > 1:
        normalized = (normalized - normalized.mean()) / max(normalized.std(), 1e-8)
    close(audit["normalized_advantages"], normalized, "normalized advantages")
    for batch in audit["minibatches"]:
        indices = batch["indices"]
        ratio = np.exp(np.asarray(batch["new_logprob"]) - batch["old_logprob"])
        advantage = normalized[indices]
        valid = np.asarray(batch["actor_valid"])
        actor = -np.where(valid, np.minimum(ratio * advantage,
                         np.clip(ratio, 1 - settings["clip"], 1 + settings["clip"]) * advantage), 0).mean()
        value = .5 * np.mean((np.asarray(batch["value"]) - returns[indices]) ** 2)
        entropy = np.where(valid, batch["entropy"], 0).mean()
        total = actor + settings["value_coefficient"] * value - settings["entropy_coefficient"] * entropy
        for name, expected in (("actor_loss", actor), ("value_loss", value),
                               ("entropy_bonus", entropy), ("total_loss", total)):
            close(batch[name], expected, name)
        if "gradient_groups" in batch:
            if any(batch["gradient_groups"][group + "_after"] > settings["max_grad_norm"] + 1e-5 for group in ("actor", "critic")):
                raise AssertionError("Per-network gradient norm exceeds configured clipping")
        elif batch["gradient_after_clip"] > settings["max_grad_norm"] + 1e-5:
            raise AssertionError("Gradient norm exceeds configured clipping")
    return {"samples": len(samples), "minibatches": len(audit["minibatches"]), "forward_gae_and_loss": True}


def check_physics(config, records):
    edge_count = 0
    for record in records:
        server_count = len(config["servers"])
        compute, memory = np.zeros(server_count), np.zeros(server_count)
        bandwidth = np.zeros(len(config["access_points"]))
        expected_static, expected_wake, task_cost = 0.0, 0.0, 0.0
        for index, server in enumerate(config["servers"]):
            close(record["ready"][index], record["active"][index] * record["previous_active"][index], "readiness", 0)
            expected_static += config["slot_seconds"] * (server["idle_power"] if record["active"][index] else server["sleep_power"])
            expected_wake += server["wake_energy"] * max(0, record["active"][index] - record["previous_active"][index])
        for task in record["tasks"]:
            choice = task["candidate"]
            if (config["information"] in ("robust", "full") and not record["control_overrun"]
                    and task["action"] != "reject" and not task["on_time"]):
                raise AssertionError("Covered robust/full candidate failed execution")
            task_cost += task["conservative_cost"]
            if choice["kind"] == "edge":
                edge_count += 1
                server_index, rate = choice["server"], task["rate"]
                server = config["servers"][server_index]
                if not record["ready"][server_index] or rate + 1e-8 < choice["minimum"]:
                    raise AssertionError("Unready server or deficient rate")
                close(choice["minimum"], choice["edge_work"] / (choice["deadline"] - choice["delay_fixed"]), "minimum rate")
                close(choice["alpha"], server["kappa"] * choice["edge_work"], "P3 alpha")
                close(choice["beta"], config["lambda_time"] * choice["weight"] * choice["edge_work"], "P3 beta")
                close(task["conservative_cost"], choice["cost_fixed"] + choice["alpha"] * rate ** 2 + choice["beta"] / rate, "task cost")
                compute[server_index] += rate
                memory[server_index] += choice["memory"]
                bandwidth[choice["ap"]] += choice["bandwidth"]
            if task["action"] != "reject" and task["elapsed"] > task["deadline"] + 1e-8:
                raise AssertionError("Execution extends past deadline")
        for index, server in enumerate(config["servers"]):
            if compute[index] > server["capacity"] + 1e-8 or memory[index] > server["memory"] + 1e-8:
                raise AssertionError("Server resource violation")
            if not config["learned_rates"]:
                selected = [task for task in record["tasks"] if task["server"] == index]
                if selected:
                    free = [task for task in selected if task["rate"] > task["minimum"] + 1e-7]
                    derivatives = [2 * task["candidate"]["alpha"] * task["rate"] -
                                   task["candidate"]["beta"] / task["rate"] ** 2 for task in free]
                    multiplier = max(0.0, -float(np.mean(derivatives))) if derivatives else max(
                        [0.0] + [task["candidate"]["beta"] / task["rate"] ** 2 -
                                 2 * task["candidate"]["alpha"] * task["rate"] for task in selected])
                    if compute[index] < server["capacity"] - 1e-7:
                        close(multiplier, 0, "nonbinding dual")
                    for task in selected:
                        stationarity = 2 * task["candidate"]["alpha"] * task["rate"] - task["candidate"]["beta"] / task["rate"] ** 2 + multiplier
                        if task in free:
                            close(stationarity, 0, "P3 free stationarity")
                        elif stationarity < -2e-5:
                            raise AssertionError("P3 lower-bound stationarity")
        if any(used > access["bandwidth"] + 1e-8 for used, access in zip(bandwidth, config["access_points"])):
            raise AssertionError("Shared AP bandwidth violation")
        close(record["energy"]["static"], expected_static, "static energy")
        close(record["energy"]["wake"], expected_wake, "wake energy")
        for key in ("local", "transmit", "edge"):
            close(record["energy"][key], sum(task["energy"][key] for task in record["tasks"]), key + " energy")
        close(record["cost"], expected_static + expected_wake + task_cost, "slot cost")
        close(record["reward"], -record["cost"], "reward")
    return {"slots": len(records), "edge_assignments": edge_count, "constraints_and_kkt": True}


def inspect_run(path):
    path = Path(path)
    config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    result = {"run_id": path.name, "updates": [], "physics": []}
    from .storage import phase_trace_path
    updates = list((path / "traces").glob("*_update.json")) + list((path / "traces/phases").glob("*/update.json"))
    for update_path in sorted(updates):
        stem = update_path.parent.name if update_path.name == "update.json" else update_path.name.removesuffix("_update.json")
        samples = json.loads(phase_trace_path(path, stem, "behavior").read_text(encoding="utf-8"))
        audit = json.loads(update_path.read_text(encoding="utf-8"))
        result["updates"].append(dict(stage=stem, **check_update(samples, audit)))
    for records_path in sorted((path / "traces").rglob("*.json")):
        if records_path.name.endswith(("_behavior.json", "_update.json")) or records_path.name.startswith("input_"):
            continue
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if isinstance(records, list) and records and isinstance(records[0], dict) and "tasks" in records[0]:
            result["physics"].append(dict(file=records_path.relative_to(path).as_posix(), **check_physics(config, records)))
    return result
