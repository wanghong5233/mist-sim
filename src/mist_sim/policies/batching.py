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
            for key, width in (("items", 19), ("servers", 13), ("context", policy.context_norm.mean.numel())):
                add(value[key], width)
            if "allocated" in value:
                add(value["allocated"], 19)
            if "item_servers" in value:
                add(value["item_servers"], 13)
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


def encode_frames(policy, frames, critic=False, cache=None, with_servers=False, request_cache=None):
    item_encoder = policy.critic_items if critic else policy.actor_items
    request_encoder = policy.critic_requests if critic else policy.actor_requests
    relation_encoder = policy.critic_relations if critic else policy.actor_relations
    server_encoder = policy.critic_servers if critic else policy.actor_servers
    resource_encoder = policy.critic_resources if critic else policy.actor_resources
    hidden = item_encoder[-1].out_features
    unique, inverse, seen, group_lengths, item_server_groups = [], [], {}, [], []
    for frame in frames:
        identity = (id(frame["items"]), id(frame.get("request_sizes")), id(frame.get("item_servers")))
        if identity not in seen:
            lengths = frame.get("request_sizes", [1] * len(frame["items"]))
            if sum(lengths) != len(frame["items"]) or any(length <= 0 for length in lengths):
                raise ValueError("Invalid request grouping")
            seen[identity] = len(unique)
            unique.append(frame["items"])
            group_lengths.append(lengths)
            item_server_groups.append(frame.get("item_servers", [[0.] * 13] * len(frame["items"])))
        inverse.append(seen[identity])
    values = (torch.cat([cache.rows(rows) for rows in unique]) if cache is not None else
              tensor(policy, [row for rows in unique for row in rows]).reshape(-1, 19))
    lengths = np.asarray([length for group in group_lengths for length in group], dtype=int)
    cache_key = (critic, tuple(seen))
    if request_cache is not None and torch.is_grad_enabled():
        raise ValueError("Sampling activation cache cannot be used during gradient computation")
    if request_cache is not None and cache_key in request_cache:
        request_values = request_cache[cache_key]
    else:
        request_values = values.new_zeros((len(lengths), hidden))
        if len(values):
            owners = tensor(policy, np.repeat(np.arange(len(lengths)), lengths), torch.long)
            target_rows = tensor(policy, [row for group in item_server_groups for row in group]).reshape(-1, 13)
            if len(target_rows) != len(values):
                raise ValueError("Candidate/server association mismatch")
            target_values = server_encoder(policy.server_norm(target_rows)) * (values[:, 2:3] > 0)
            relations = relation_encoder(torch.cat((item_encoder(policy.item_norm(values)), target_values), dim=1))
            request_values = request_values.index_add(0, owners, relations)
            request_values = request_values / tensor(policy, lengths)[:, None]
        request_values = request_encoder(torch.cat((request_values, tensor(policy, np.log1p(lengths))[:, None],
                                                   tensor(policy, lengths > 1)[:, None]), dim=1))
        if request_cache is not None:
            request_cache[cache_key] = request_values
    offsets = np.r_[0, np.cumsum([len(group) for group in group_lengths])]
    request_indices, request_owners = [], []
    for frame_index, (frame, source) in enumerate(zip(frames, inverse)):
        pending = frame.get("request_pending", [1.] * len(group_lengths[source]))
        if len(pending) != len(group_lengths[source]):
            raise ValueError("Invalid pending request mask")
        for request_index, valid in enumerate(pending):
            if valid:
                request_indices.append(int(offsets[source] + request_index))
                request_owners.append(frame_index)
    request_pool = values.new_zeros((len(frames), hidden))
    if request_indices:
        owner_ids = tensor(policy, request_owners, torch.long)
        request_pool = request_pool.index_add(0, owner_ids, request_values[tensor(policy, request_indices, torch.long)])
        request_pool = request_pool / torch.bincount(owner_ids, minlength=len(frames)).clamp(min=1)[:, None]
    server_lengths = np.asarray([len(frame["servers"]) for frame in frames])
    server_offsets = np.r_[0, np.cumsum(server_lengths)]
    server_rows = (torch.cat([cache.rows(frame["servers"]) for frame in frames]) if cache is not None else
                   tensor(policy, [row for frame in frames for row in frame["servers"]]).reshape(-1, 13))
    allocated_rows, allocated_owners = [], []
    for frame_index, frame in enumerate(frames):
        rows, owners = frame.get("allocated", []), frame.get("allocation_servers", [])
        if len(rows) != len(owners) or any(owner < 0 or owner >= server_lengths[frame_index] for owner in owners):
            raise ValueError("Invalid server assignment")
        allocated_rows.extend(rows)
        allocated_owners.extend(int(server_offsets[frame_index] + owner) for owner in owners)
    allocated_pool = values.new_zeros((len(server_rows), hidden))
    allocated_count = values.new_zeros(len(server_rows))
    if allocated_rows:
        owner_ids = tensor(policy, allocated_owners, torch.long)
        encoded = item_encoder(policy.item_norm(tensor(policy, allocated_rows)))
        allocated_pool = allocated_pool.index_add(0, owner_ids, encoded)
        allocated_count = torch.bincount(owner_ids, minlength=len(server_rows)).to(values.dtype)
        allocated_pool = allocated_pool / allocated_count.clamp(min=1)[:, None]
    remaining = tensor(policy, [row for frame in frames for row in
                               frame.get("remaining", [[1., 1., 1.]] * len(frame["servers"]))]).reshape(-1, 3)
    remaining = remaining / (1 + remaining.abs())
    servers = resource_encoder(torch.cat((server_encoder(policy.server_norm(server_rows)), allocated_pool,
                                         remaining, allocated_count.log1p()[:, None]), dim=1))
    server_owners = tensor(policy, np.repeat(np.arange(len(frames)), server_lengths), torch.long)
    server_pool = values.new_zeros((len(frames), hidden)).index_add(0, server_owners, servers)
    server_pool = server_pool / tensor(policy, np.maximum(server_lengths, 1))[:, None]
    context = policy.context_norm(torch.cat([cache.rows(frame["context"]) for frame in frames])
                                  if cache is not None else tensor(policy, [frame["context"] for frame in frames]))
    global_encoding = torch.cat((request_pool, server_pool, context), dim=1)
    return (global_encoding, servers) if with_servers else global_encoding


def values_batch(policy, samples, cache=None):
    if hasattr(policy, "phase_policies"):
        if any(sample.get("phase") not in policy.phase_policies for sample in samples):
            raise ValueError("Unknown joint decision phase")
        values = policy.value_scale.new_zeros(len(samples))
        for phase, subpolicy in policy.phase_policies.items():
            indices = [index for index, sample in enumerate(samples) if sample["phase"] == phase]
            if indices:
                values = values.index_copy(0, tensor(policy, indices, torch.long),
                    values_batch(subpolicy, [samples[index] for index in indices], cache))
        return values
    return policy.value_head(encode_frames(policy, [sample["frame"] for sample in samples], True, cache)).flatten() * policy.value_scale


def replay_batch(policy, samples, with_value=True, cache=None):
    if not samples:
        raise ValueError("Empty replay batch")
    if hasattr(policy, "phase_policies"):
        if any(sample.get("phase") not in policy.phase_policies for sample in samples):
            raise ValueError("Unknown joint decision phase")
        joint, entropy, values = [policy.value_scale.new_zeros(len(samples)) for _ in range(3)]
        for phase, subpolicy in policy.phase_policies.items():
            indices = [index for index, sample in enumerate(samples) if sample["phase"] == phase]
            if indices:
                logprob, ent, value = replay_batch(subpolicy, [samples[index] for index in indices], with_value, cache)
                owners = tensor(policy, indices, torch.long)
                joint = joint.index_copy(0, owners, logprob)
                entropy = entropy.index_copy(0, owners, ent)
                if with_value:
                    values = values.index_copy(0, owners, value)
        return joint, entropy, values if with_value else None
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
        encoded_frames, servers = encode_frames(policy, frames, cache=cache, with_servers=True)
        shared = torch.cat((encoded_frames[tensor(policy, frame_ids, torch.long)],
                            policy.prefix_norm(prefixes)), dim=1)
        target_encoding = features.new_zeros((len(features), policy.actor_items[-1].out_features))
        server_offsets = np.r_[0, np.cumsum([len(frame["servers"]) for frame in frames])]
        targets = np.asarray([target if target < 0 else int(server_offsets[frame_index] + target)
                              for step, frame_index in zip(steps, frame_ids)
                              for target in step.get("targets", [-1] * len(step["features"]))])
        valid_targets = tensor(policy, targets >= 0, torch.bool)
        target_encoding[valid_targets] = servers[tensor(policy, targets[targets >= 0], torch.long)]
        combined = torch.cat((policy.actor_actions(policy.action_norm(features)), shared[decision_ids], target_encoding), dim=1)
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
