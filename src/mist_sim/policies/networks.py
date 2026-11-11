"""Autoregressive set policies with independent actor and critic encoders."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from ..profiles import Profiles
from ..environment import candidates

ITEM_DIM, SERVER_DIM, CONTEXT_DIM, PREFIX_DIM = 19, 13, 7, 7


def mlp(input_size, output_size, hidden=64, output_gain=1.0):
    network = nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(),
                            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_size))
    for module in network:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(network[-1].weight, output_gain)
    return network


class Normalizer(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("scale", torch.ones(dimensions))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, rows):
        if bool(self.fitted):
            raise RuntimeError("Normalization already frozen")
        values = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite normalization input")
        if len(values):
            self.mean.copy_(values.mean(0))
            deviation = values.std(0, unbiased=False)
            self.scale.copy_(torch.where(deviation > 1e-8, deviation, values.mean(0).abs().clamp(min=1.0)))
        self.fitted.fill_(True)

    def forward(self, values):
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite policy features")
        standardized = (values - self.mean) / self.scale
        if not torch.isfinite(standardized).all():
            raise ValueError("Nonfinite standardized features")
        return standardized / (1 + standardized.abs())


def server_features(config, active, ready, history=()):
    return [[server["capacity"], server["memory"] / 1e6, server["idle_power"], server["sleep_power"],
             server["wake_energy"], server["kappa"] * 1000, float(active[index]), float(ready[index]),
             config["access_points"][server["ap"]]["bandwidth"] / 1e5,
             server["ap"] / max(1, len(config["access_points"])),
             float(np.mean([item.get("allocated_compute", item["compute"])[index] / server["capacity"] for item in history])) if history else 0,
             float(np.mean([item["compute"][index] / server["capacity"] for item in history])) if history else 0,
             float(np.mean([item.get("accepted", 0) / max(1, item["released"]) for item in history])) if history else 0]
            for index, server in enumerate(config["servers"])]


def context_features(config, slot, history, count):
    history = history if config["long_term_feedback"] else []
    return [(config["horizon"] - slot) / config["horizon"], count,
            float(np.mean([item["released"] for item in history])) if history else 0,
            float(np.mean([item["cost"] for item in history])) if history else 0,
            float(np.mean([item["completed"] for item in history])) if history else 0,
            float(bool(history)),
            (slot % config["window"]) / config["window"]]


def item_features(choice, config):
    return [float(choice["kind"] == "reject"), float(choice["kind"] == "local"),
            float(choice["kind"] == "edge"), choice["bits"] / 1e6,
            choice["deadline"] - choice["delay_fixed"], choice["minimum"], choice["memory"] / 1e6,
            choice["bandwidth"] / 1e5, choice["cost_fixed"], choice["alpha"],
            choice["beta"], choice["delay_fixed"], choice["deadline"], choice["edge_work"],
            choice["local_work"], choice["weight"], 0.0, 0.0, 0.0]


def upper_choices(observation):
    config = observation["config"]
    active = observation["active"]
    items = []
    for index, server in enumerate(config["servers"]):
        for enabled in (0, 1):
            items.append([0, 0, enabled, active[index], enabled * active[index], server["capacity"],
                          server["memory"] / 1e6, 0, server["idle_power"] if enabled else server["sleep_power"],
                          server["kappa"], server["wake_energy"] * max(0, enabled - active[index]),
                          0, 0, 0, 0, 1, 0, 0, 0])
    return items


def upper_frame(observation):
    config, active = observation["config"], observation["active"]
    from .split_solver import proposal_config
    config = proposal_config(config)
    history = observation["history"] if config["long_term_feedback"] else []
    profiles = Profiles(config)
    server_rows = server_features(config, active, active, history)
    items, request_sizes, item_servers = [], [], []
    for batch in history:
        for request in batch["public"]:
            choices = candidates(config, profiles, request, [1] * len(active))
            items.extend(item_features(choice, config) for choice in choices)
            item_servers.extend(server_rows[choice["server"]] if choice["server"] >= 0 else [0.] * SERVER_DIM
                                for choice in choices)
            request_sizes.append(len(choices))
    return {"servers": server_rows, "items": items, "item_servers": item_servers,
            "request_sizes": request_sizes,
            "context": context_features(config, observation["slot"], observation["history"], 0)}


def lower_frame(problem):
    from .split_solver import proposal_problem
    problem = proposal_problem(problem)
    server_rows = server_features(problem.config, problem.active, problem.ready, problem.history)
    return {"servers": server_rows,
            "items": [item_features(choice, problem.config) for options in problem.choices for choice in options],
            "item_servers": [server_rows[choice["server"]] if choice["server"] >= 0 else [0.] * SERVER_DIM
                             for options in problem.choices for choice in options],
            "request_sizes": [len(options) for options in problem.choices],
            "context": context_features(problem.config, problem.slot, problem.history, len(problem.public))}


def lower_prefix_frame(frame, problem, usage, index, allocated, owners):
    remaining = [[(server["capacity"] - usage["compute"][server_index]) / server["capacity"],
                  (server["memory"] - usage["memory"][server_index]) / server["memory"],
                  (problem.config["access_points"][server["ap"]]["bandwidth"] - usage["bandwidth"][server["ap"]]) /
                  problem.config["access_points"][server["ap"]]["bandwidth"]]
                 for server_index, server in enumerate(problem.config["servers"])]
    return dict(frame, remaining=remaining, allocated=list(allocated), allocation_servers=list(owners),
                request_pending=[float(request_index >= index) for request_index in range(len(problem.public))])


class SetPolicy(nn.Module):
    def __init__(self, hidden=64, learned_rates=False, value_model="plain", horizon=1, gamma=1.):
        super().__init__()
        self.learned_rates = learned_rates
        self.value_model, self.value_horizon, self.value_gamma = value_model, horizon, gamma
        if gamma == 1:
            self.discounted_horizon = horizon
        elif gamma == 0:
            self.discounted_horizon = 1.
        else:
            self.discounted_horizon = -math.expm1(horizon * math.log(gamma)) / (1 - gamma)
        self.register_buffer("value_scale", torch.tensor(1.))
        self.register_buffer("value_baseline", torch.tensor(0.))
        self.item_norm, self.server_norm, self.context_norm = Normalizer(ITEM_DIM), Normalizer(SERVER_DIM), Normalizer(CONTEXT_DIM)
        self.action_norm = Normalizer(ITEM_DIM)
        self.prefix_norm = Normalizer(PREFIX_DIM)
        self.actor_items, self.actor_servers = mlp(ITEM_DIM, hidden, hidden), mlp(SERVER_DIM, hidden, hidden)
        self.critic_items, self.critic_servers = mlp(ITEM_DIM, hidden, hidden), mlp(SERVER_DIM, hidden, hidden)
        self.actor_requests, self.critic_requests = mlp(hidden + 2, hidden, hidden), mlp(hidden + 2, hidden, hidden)
        self.actor_relations, self.critic_relations = mlp(hidden * 2, hidden, hidden), mlp(hidden * 2, hidden, hidden)
        self.actor_resources, self.critic_resources = mlp(hidden * 2 + 4, hidden, hidden), mlp(hidden * 2 + 4, hidden, hidden)
        self.actor_actions = mlp(ITEM_DIM, hidden, hidden)
        combined = hidden * 4 + CONTEXT_DIM + PREFIX_DIM
        self.score = mlp(combined, 1, hidden, 0.01)
        self.value_head = mlp(hidden * 2 + CONTEXT_DIM, 1, hidden)
        if value_model == "finite_horizon":
            # Start from the measured past reward rate. The network learns the
            # state-dependent residual, instead of rediscovering the clock.
            nn.init.zeros_(self.value_head[-1].weight)
            nn.init.zeros_(self.value_head[-1].bias)
        if learned_rates:
            self.rate_head = mlp(combined, 1, hidden, 0.01)
            self.rate_log_std = nn.Parameter(torch.zeros(()))

    def fit_normalizers(self, frames, steps=()):
        frames = list(frames) + [step["prefix_frame"] for step in steps if "prefix_frame" in step]
        def unique_rows(key):
            seen, rows = set(), []
            for frame in frames:
                if key not in frame:
                    continue
                values = frame[key]
                if id(values) not in seen:
                    seen.add(id(values))
                    rows.extend(values)
            return rows
        self.item_norm.fit(unique_rows("items") + unique_rows("allocated"))
        self.action_norm.fit([row for step in steps for row in step["features"]])
        self.server_norm.fit(unique_rows("servers"))
        contexts = {id(frame["context"]): frame["context"] for frame in frames}
        self.context_norm.fit(list(contexts.values()))
        self.prefix_norm.fit([step["prefix"] for step in steps])

    def encode(self, frame, critic=False, with_servers=False, request_cache=None):
        from .batching import encode_frames
        encoded, servers = encode_frames(self, [frame], critic, with_servers=True, request_cache=request_cache)
        hidden = self.actor_items[-1].out_features
        parts = tuple(encoded[0].split([hidden, hidden, CONTEXT_DIM]))
        return (parts, servers) if with_servers else parts

    def value_from_encoding(self, encoding, context):
        residual = self.value_head(encoding).squeeze(-1)
        if self.value_model == "plain":
            return residual * self.value_scale
        remaining = (context[..., 0] * self.value_horizon).round().clamp(0, self.value_horizon)
        if self.value_gamma == 1:
            steps = remaining
        elif self.value_gamma == 0:
            steps = (remaining > 0).to(remaining.dtype)
        else:
            # Raising a float32 gamma close to 1 rounds the base to 1 and
            # cancels the entire clock (and residual gradient). Keep log(gamma)
            # in Python precision and use expm1 for the small difference.
            steps = -torch.expm1(remaining * math.log(self.value_gamma)) / (1 - self.value_gamma)
        return self.value_scale * (self.value_baseline * steps +
                                   residual * (steps / self.discounted_horizon))

    @torch.no_grad()
    def set_value_baseline(self, reward_per_slot):
        if self.value_model == "finite_horizon":
            self.value_baseline.copy_(self.value_scale.new_tensor(reward_per_slot) / self.value_scale)

    def value(self, frame):
        encoded = torch.cat(self.encode(frame, critic=True))
        return self.value_from_encoding(encoded, encoded.new_tensor(frame["context"]))

    def distribution(self, frame_encoding, features, prefix, mask, servers=None, targets=None):
        encoded_items = self.actor_actions(self.action_norm(torch.tensor(features, dtype=torch.float32)))
        shared = torch.cat((*frame_encoding, self.prefix_norm(torch.tensor(prefix, dtype=torch.float32))))
        target_encoding = torch.zeros_like(encoded_items)
        if targets is not None:
            indices = torch.tensor(targets, dtype=torch.long)
            valid_targets = indices >= 0
            target_encoding[valid_targets] = servers[indices[valid_targets]]
        combined = torch.cat((encoded_items, shared.expand(len(features), -1), target_encoding), dim=1)
        logits = self.score(combined).flatten()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite policy logits")
        valid = torch.tensor([bool(value) for value in mask], dtype=torch.bool)
        if not valid.any():
            raise ValueError("No legal action")
        logits = logits.masked_fill(~valid, -torch.inf)
        return Categorical(logits=logits), combined

    def replay(self, sample):
        encoding, servers = self.encode(sample["frame"], with_servers=True)
        logprob = torch.zeros(())
        entropy = torch.zeros(())
        for step in sample["steps"]:
            conditional_encoding, conditional_servers = (self.encode(step["prefix_frame"], with_servers=True)
                                                        if "prefix_frame" in step else (encoding, servers))
            distribution, combined = self.distribution(conditional_encoding, step["features"], step["prefix"], step["mask"],
                                                      conditional_servers, step.get("targets"))
            selected = torch.tensor(step["action"])
            logprob = logprob + distribution.log_prob(selected)
            entropy = entropy + distribution.entropy()
            if step.get("rate_latent") is not None:
                mean = self.rate_head(combined[step["action"]]).squeeze()
                normal = Normal(mean, self.rate_log_std.exp())
                logprob = logprob + normal.log_prob(torch.tensor(step["rate_latent"]))
                entropy = entropy + normal.entropy()
        return logprob, entropy, self.value(sample["frame"])

    @torch.no_grad()
    def upper_action(self, observation, deterministic=False, record=True, cached=None):
        frame = cached[0] if cached is not None else upper_frame(observation)
        choices = upper_choices(observation)
        request_cache = {}
        logprob = torch.zeros(())
        actions, steps, activation_probabilities = [], [], []
        for index in range(len(observation["active"])):
            prefix_frame = dict(frame, servers=[list(row) for row in frame["servers"]])
            for previous_index, enabled in enumerate(actions):
                prefix_frame["servers"][previous_index][6] = enabled
                prefix_frame["servers"][previous_index][7] = enabled * observation["active"][previous_index]
            encoding, servers = self.encode(prefix_frame, with_servers=True, request_cache=request_cache)
            features = choices[2 * index:2 * index + 2]
            prefix = [index, sum(actions), sum(actions) / max(1, index), 0, 0, 0, 0]
            distribution, _ = self.distribution(encoding, features, prefix, [True, True], servers, [index, index])
            action = int(distribution.probs.argmax() if deterministic else distribution.sample())
            if record:
                activation_probabilities.append(float(distribution.probs[1]))
                steps.append({"features": features, "prefix": prefix, "mask": [True, True], "action": action,
                              "prefix_frame": prefix_frame, "targets": [index, index]})
            actions.append(action)
            if record:
                logprob = logprob + distribution.log_prob(torch.tensor(action))
        if not record:
            return actions, None
        sample = {"frame": frame, "steps": steps, "actions": actions,
                  "activation_probabilities": activation_probabilities}
        value = cached[1] if cached is not None else self.value(frame)
        sample.update(logprob=float(logprob), value=float(value))
        return actions, sample

    @torch.no_grad()
    def lower_action(self, problem, deterministic=False, record=True, cached=None):
        from .split_solver import proposal_problem
        problem = proposal_problem(problem)
        frame = cached[0] if cached is not None else lower_frame(problem)
        usage = problem.empty_usage()
        allocated, owners = [], []
        request_cache = {}
        actions, steps, predicted = [], [], []
        edge_count, local_count = 0, 0
        logprob = torch.zeros(())
        for index, options in enumerate(problem.choices):
            prefix_frame = lower_prefix_frame(frame, problem, usage, index, allocated, owners)
            encoding, servers = self.encode(prefix_frame, with_servers=True, request_cache=request_cache)
            features = [item_features(choice, problem.config) for choice in options]
            for feature, choice in zip(features, options):
                if choice["kind"] == "edge":
                    server = problem.config["servers"][choice["server"]]
                    feature[-3:] = [(server["capacity"] - usage["compute"][choice["server"]]) / server["capacity"],
                                    (server["memory"] - usage["memory"][choice["server"]]) / server["memory"],
                                    (problem.config["access_points"][choice["ap"]]["bandwidth"] - usage["bandwidth"][choice["ap"]]) /
                                    problem.config["access_points"][choice["ap"]]["bandwidth"]]
            prefix = [index, float(usage["compute"].sum()), float(usage["memory"].sum()) / 1e6,
                      float(usage["bandwidth"].sum()) / 1e5, edge_count, local_count,
                      (len(problem.public) - index) / max(1, len(problem.public))]
            mask = problem.mask(options, usage, capacity=problem.config["capacity_mask"])
            targets = [choice["server"] for choice in options]
            distribution, combined = self.distribution(encoding, features, prefix, mask, servers, targets)
            action = int(distribution.probs.argmax() if deterministic else distribution.sample())
            choice = options[action]
            if record:
                logprob = logprob + distribution.log_prob(torch.tensor(action))
            step = {"features": features, "prefix": prefix, "mask": mask, "action": action,
                    "prefix_frame": prefix_frame, "targets": targets} if record else None
            rate = 0.0
            if self.learned_rates and choice["kind"] == "edge":
                mean = self.rate_head(combined[action]).squeeze()
                normal = Normal(mean, self.rate_log_std.exp())
                latent = mean if deterministic else normal.sample()
                if record:
                    step["rate_latent"] = float(latent)
                    logprob = logprob + normal.log_prob(latent)
                rate = float(torch.sigmoid(latent)) * problem.config["servers"][choice["server"]]["capacity"]
            predicted.append(rate)
            actions.append(action)
            if record:
                steps.append(step)
            problem.reserve(choice, usage)
            if choice["kind"] == "edge":
                allocated.append(item_features(choice, problem.config))
                owners.append(choice["server"])
            edge_count += choice["kind"] == "edge"
            local_count += choice["kind"] == "local"
        if not record:
            return actions, predicted if self.learned_rates else None, None
        sample = {"frame": frame, "steps": steps, "actions": actions,
                  "order": [item["id"] for item in problem.public], "predicted_rates": predicted if self.learned_rates else None}
        value = cached[1] if cached is not None else self.value(frame)
        sample.update(logprob=float(logprob), value=float(value))
        return actions, sample["predicted_rates"], sample
