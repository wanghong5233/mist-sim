"""Ragged set/decision packing; neural computation is batched, not per request.

Python loops here only assemble immutable trajectory metadata. No padding is
encoded by an MLP; segment means preserve empty-set and variable-size semantics.
"""
import numpy as np
import torch
from torch.distributions import Categorical, Normal


def tensor(policy, values, dtype=torch.float32):
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=policy.value_scale.device)


class RolloutTensors:
    """One detached input tensor per shared list, scoped to one PPO update.

    Only raw inputs are cached: neural activations must be recomputed after every
    optimizer step. Keep source references so Python cannot recycle identity keys.
    """
    def __init__(self, policy, samples):
        self.entries = {}
        self.decisions = {}
        def add(values, width):
            if id(values) not in self.entries:
                self.entries[id(values)] = (values, tensor(policy, values).reshape(-1, width))
        def frame(value):
            for key, width in (("items", 19), ("servers", 13), ("context", 11)):
                add(value[key], width)
        for sample in samples:
            frame(sample["frame"])
            for step in sample["steps"]:
                self.decisions[id(step)] = dict(
                    mask=tensor(policy, step["mask"], torch.bool),
                    action=tensor(policy, [step["action"]], torch.long),
                    latent=tensor(policy, [step["rate_latent"]]) if step.get("rate_latent") is not None else None)
                if "prefix_frame" in step:
                    frame(step["prefix_frame"])
                add(step["features"], 19)
                add(step["prefix"], 7)

    def rows(self, values):
        return self.entries[id(values)][1]


def encode_frames(policy, frames, critic=False, cache=None):
    encoders = ((policy.critic_items, policy.critic_servers) if critic else
                (policy.actor_items, policy.actor_servers))
    pools = []
    for key, width, normalizer, encoder in zip(
            ("items", "servers"), (19, 13), (policy.item_norm, policy.server_norm), encoders):
        unique, inverse, seen = [], [], {}
        for frame in frames:
            identity = id(frame[key])
            if identity not in seen:
                seen[identity] = len(unique)
                unique.append(frame[key])
            inverse.append(seen[identity])
        lengths = np.asarray([len(rows) for rows in unique])
        values = (torch.cat([cache.rows(group) for group in unique]) if cache is not None else
                  tensor(policy, [row for group in unique for row in group]).reshape(-1, width))
        pooled = values.new_zeros((len(unique), encoder[-1].out_features))
        if len(values):
            owners = tensor(policy, np.repeat(np.arange(len(unique)), lengths), torch.long)
            pooled = pooled.index_add(0, owners, encoder(normalizer(values)))
        pooled = pooled / tensor(policy, np.maximum(lengths, 1))[:, None]
        pools.append(pooled[tensor(policy, inverse, torch.long)])
    context = policy.context_norm(torch.cat([cache.rows(frame["context"]) for frame in frames])
                                  if cache is not None else tensor(policy, [frame["context"] for frame in frames]))
    return torch.cat((*pools, context), dim=1)


def values_batch(policy, samples, cache=None):
    return policy.value_head(encode_frames(policy, [sample["frame"] for sample in samples], True, cache)).flatten() * policy.value_scale


def replay_batch(policy, samples, with_value=True, cache=None):
    if not samples:
        raise ValueError("Empty replay batch")
    steps, owners, frames, frame_ids, seen = [], [], [], [], {}
    for owner, sample in enumerate(samples):
        for step in sample["steps"]:
            frame = step.get("prefix_frame", sample["frame"])
            identity = id(frame)
            if identity not in seen:
                seen[identity] = len(frames)
                frames.append(frame)
            steps.append(step)
            owners.append(owner)
            frame_ids.append(seen[identity])
    joint = policy.value_scale.new_zeros(len(samples))
    entropy = joint.clone()
    if steps:
        lengths = np.asarray([len(step["features"]) for step in steps])
        if np.any(lengths == 0) or any(not any(step["mask"]) for step in steps):
            raise ValueError("No legal action")
        decision_ids = tensor(policy, np.repeat(np.arange(len(steps)), lengths), torch.long)
        columns = tensor(policy, np.concatenate([np.arange(length) for length in lengths]), torch.long)
        features = (torch.cat([cache.rows(step["features"]) for step in steps]) if cache is not None else
                    tensor(policy, [row for step in steps for row in step["features"]]))
        prefixes = (torch.cat([cache.rows(step["prefix"]) for step in steps]) if cache is not None else
                    tensor(policy, [step["prefix"] for step in steps]))
        shared = torch.cat((encode_frames(policy, frames, cache=cache)[tensor(policy, frame_ids, torch.long)],
                            policy.prefix_norm(prefixes)), dim=1)
        combined = torch.cat((policy.actor_items(policy.item_norm(features)), shared[decision_ids]), dim=1)
        scores = policy.score(combined).flatten()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite policy logits")
        mask = (torch.cat([cache.decisions[id(step)]["mask"] for step in steps]) if cache is not None else
                tensor(policy, [bool(valid) for step in steps for valid in step["mask"]], torch.bool))
        logits = scores.new_full((len(steps), int(lengths.max())), -torch.inf)
        logits[decision_ids, columns] = scores.masked_fill(~mask, -torch.inf)
        distribution = Categorical(logits=logits)
        actions = (torch.cat([cache.decisions[id(step)]["action"] for step in steps]) if cache is not None else
                   tensor(policy, [step["action"] for step in steps], torch.long))
        logprob, ent = distribution.log_prob(actions), distribution.entropy()
        rate_indices = [index for index, step in enumerate(steps) if step.get("rate_latent") is not None]
        if rate_indices:
            indices = tensor(policy, rate_indices, torch.long)
            offsets = tensor(policy, np.r_[0, np.cumsum(lengths)[:-1]], torch.long)
            mean = policy.rate_head(combined[(offsets + actions)[indices]]).flatten()
            normal = Normal(mean, policy.rate_log_std.exp())
            latent = (torch.cat([cache.decisions[id(steps[index])]["latent"] for index in rate_indices])
                      if cache is not None else tensor(policy, [steps[index]["rate_latent"] for index in rate_indices]))
            logprob = logprob.index_add(0, indices, normal.log_prob(latent))
            ent = ent.index_add(0, indices, normal.entropy())
        owner_ids = tensor(policy, owners, torch.long)
        joint = joint.index_add(0, owner_ids, logprob)
        entropy = entropy.index_add(0, owner_ids, ent)
    return joint, entropy, values_batch(policy, samples, cache) if with_value else None


def replay_chunks(policy, samples, size=64, cache=None):
    """Bound temporary tensors for full-rollout diagnostics; skip unused critic."""
    return torch.cat([replay_batch(policy, samples[start:start + size], False, cache)[0]
                      for start in range(0, len(samples), size)])
