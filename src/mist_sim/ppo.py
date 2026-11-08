"""PPO joint-action clipped loss; general math independently implemented."""
import numpy as np
import torch
import time
from .policies.batching import replay_batch, replay_chunks, values_batch, RolloutTensors


def clip_policy_gradients(policy, limit):
    groups = {"actor": [], "critic": []}
    for name, parameter in policy.named_parameters():
        if parameter.grad is not None:
            group = "critic" if name.startswith(("critic_", "value_head")) else "actor"
            groups[group].append(parameter)
    result = {}
    for group, parameters in groups.items():
        before = torch.nn.utils.clip_grad_norm_(parameters, limit)
        if not torch.isfinite(before):
            raise FloatingPointError("Nonfinite gradient")
        result[group + "_before"] = float(before)
        result[group + "_after"] = float(torch.sqrt(sum(parameter.grad.square().sum() for parameter in parameters))) if parameters else 0.0
    return result


def loss_terms(logprob, old, advantage, values, target, entropy, actor_valid, config):
    logratio = logprob - old
    ratio = logratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage)
    actor_loss = -torch.where(actor_valid, surrogate, torch.zeros_like(surrogate)).mean()
    value_loss = 0.5 * (values - target).square().mean()
    entropy_bonus = torch.where(actor_valid, entropy, torch.zeros_like(entropy)).mean()
    total = actor_loss + config["value_coefficient"] * value_loss - config["entropy_coefficient"] * entropy_bonus
    return total, actor_loss, value_loss, entropy_bonus, ratio, logratio


def update(policy, optimizer, buffer, config, behavior_version, opponent_version, audit=None, check_budget=None, on_epoch=None):
    buffer.consume(behavior_version, opponent_version)
    if not buffer.samples:
        raise ValueError("Empty rollout")
    advantages, returns = buffer.advantages(config["gamma"], config["gae_lambda"])
    raw_advantages = advantages.copy()
    initial_parameters = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
    if audit is not None:
        audit.update(settings=dict(config), advantages=raw_advantages.tolist(), normalized_advantages=advantages.tolist(),
                     returns=returns.tolist(), minibatches=[], epoch_checks=[],
                     td_errors=[sample["reward"] + config["gamma"] ** sample["duration"] * sample["next_value"] *
                                (not sample["terminated"]) - sample["value"] for sample in buffer.samples])
    input_cache = RolloutTensors(policy, buffer.samples)
    old_all = torch.tensor([sample["logprob"] for sample in buffer.samples])
    advantages_all = torch.tensor(advantages, dtype=torch.float32)
    targets_all = torch.tensor(returns, dtype=torch.float32)
    valid_all = torch.tensor([bool(sample["steps"]) for sample in buffer.samples])
    diagnostics = []
    timings = dict(forward_loss_seconds=0.0, backward_step_seconds=0.0, kl_check_seconds=0.0)
    sample_count = len(buffer.samples)
    if not sample_count:
        raise ValueError("Empty rollout")
    for epoch in range(config["epochs"]):
        epoch_started = time.perf_counter()
        epoch_offset = len(diagnostics)
        permutation = torch.randperm(sample_count).tolist()
        for offset in range(0, sample_count, config["minibatch"]):
            if check_budget is not None:
                check_budget()
            indices = permutation[offset:offset + config["minibatch"]]
            forward_started = time.perf_counter()
            logprob, entropy, values = replay_batch(policy, [buffer.samples[index] for index in indices], cache=input_cache)
            old = old_all[indices]
            advantage = advantages_all[indices]
            target = targets_all[indices]
            actor_valid = valid_all[indices]
            loss, actor_loss, value_loss, entropy_loss, ratio, logratio = loss_terms(
                logprob, old, advantage, values, target, entropy, actor_valid, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad()
            timings["forward_loss_seconds"] += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            loss.backward()
            gradient_groups = clip_policy_gradients(policy, config["max_grad_norm"])
            gradient = np.hypot(gradient_groups["actor_before"], gradient_groups["critic_before"])
            clipped_gradient = float(torch.sqrt(sum(parameter.grad.square().sum() for parameter in policy.parameters()
                                                     if parameter.grad is not None)))
            if audit is not None:
                audit["minibatches"].append({"epoch": epoch, "indices": indices, "old_logprob": old.tolist(),
                    "new_logprob": logprob.detach().tolist(), "value": values.detach().tolist(),
                    "entropy": entropy.detach().tolist(), "actor_valid": actor_valid.tolist(),
                    "actor_loss": float(actor_loss), "value_loss": float(value_loss),
                    "entropy_bonus": float(entropy_loss), "total_loss": float(loss),
                    "gradient_before_clip": float(gradient), "gradient_after_clip": clipped_gradient,
                    "gradient_groups": gradient_groups})
            optimizer.step()
            timings["backward_step_seconds"] += time.perf_counter() - backward_started
            with torch.no_grad():
                kl = (torch.expm1(logratio) - logratio).mean()
                diagnostics.append({"epoch": epoch, "batch_size": len(indices), "loss": float(loss), "actor_loss": float(actor_loss),
                                    "value_loss": float(value_loss), "entropy": float(entropy_loss),
                                    "joint_kl": float(kl), "clip_fraction": float(((ratio - 1).abs() > config["clip"]).float().mean()),
                                    "gradient_norm": float(gradient), **gradient_groups})
        kl_started = time.perf_counter()
        with torch.no_grad():
            updated_logprob = replay_chunks(policy, buffer.samples, config["minibatch"], input_cache)
            old_logprob = old_all
            joint_logratio = updated_logprob - old_logprob
            post_kl = float((torch.expm1(joint_logratio.double()) - joint_logratio.double()).mean())
        timings["kl_check_seconds"] += time.perf_counter() - kl_started
        if not np.isfinite(post_kl):
            raise FloatingPointError("Nonfinite post-update KL")
        if audit is not None:
            audit["epoch_checks"].append({"epoch": epoch, "post_update_joint_kl": post_kl,
                                          "early_stop": post_kl > config["target_kl"]})
        if on_epoch is not None:
            epoch_rows = diagnostics[epoch_offset:]
            weights = [row["batch_size"] for row in epoch_rows]
            summary = {key: float(np.average([row[key] for row in epoch_rows], weights=weights))
                       for key in epoch_rows[0] if key not in ("epoch", "batch_size")}
            on_epoch(dict(summary, epoch=epoch + 1, epoch_limit=config["epochs"], minibatches=len(epoch_rows),
                          post_update_joint_kl=post_kl, early_stop=post_kl > config["target_kl"],
                          seconds=time.perf_counter() - epoch_started))
        if post_kl > config["target_kl"]:
            break
    with torch.no_grad():
        final_values = torch.cat([values_batch(policy, buffer.samples[start:start + config["minibatch"]], input_cache)
                                  for start in range(0, sample_count, config["minibatch"])]).cpu().numpy()
    variance = float(np.var(returns))
    explained = None if variance <= 1e-12 else float(1 - np.var(returns - final_values) / variance)
    output = {key: float(np.average([item[key] for item in diagnostics], weights=[item["batch_size"] for item in diagnostics]))
              for key in diagnostics[0] if key not in ("epoch", "batch_size")}
    output.update(samples=sample_count, optimizer_steps=len(diagnostics), explained_variance=explained,
                  update_timings=timings,
                  post_update_joint_kl=post_kl, gradient_after_clip_max=max(
                      [item["gradient_after_clip"] for item in audit["minibatches"]]) if audit else clipped_gradient,
                  durations=[sample["duration"] for sample in buffer.samples],
                  terminated=sum(sample["terminated"] for sample in buffer.samples),
                  truncated=sum(sample["truncated"] for sample in buffer.samples))
    changes = {"actor": 0.0, "critic": 0.0}
    for name, parameter in policy.named_parameters():
        group = "critic" if name.startswith(("critic_", "value_head")) else "actor"
        changes[group] += float((parameter.detach() - initial_parameters[name]).double().square().sum())
    output["parameter_delta_l2"] = {name: value ** .5 for name, value in changes.items()}
    output["training_reward_range"] = [min(sample["reward"] for sample in buffer.samples), max(sample["reward"] for sample in buffer.samples)]
    output["training_reward_mean"] = float(np.mean([sample["reward"] for sample in buffer.samples]))
    output["epochs_completed"] = epoch + 1
    output["epoch_stop_reason"] = "target_kl" if post_kl > config["target_kl"] else "epoch_limit"
    output["return_range"] = [float(returns.min()), float(returns.max())]
    output["advantage_range"] = [float(raw_advantages.min()), float(raw_advantages.max())]
    output["normalized_advantage"] = {"mean": float(advantages.mean()), "std": float(advantages.std()),
                                      "min": float(advantages.min()), "max": float(advantages.max())}
    output["training_reward_signs"] = {"positive": sum(sample["reward"] > 0 for sample in buffer.samples),
                                       "negative": sum(sample["reward"] < 0 for sample in buffer.samples),
                                       "zero": sum(sample["reward"] == 0 for sample in buffer.samples)}
    buffer.clear()
    return output
