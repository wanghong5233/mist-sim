"""Autoregressive set policies with independent actor and critic encoders."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from ..profiles import Profiles
from ..environment import candidates

ITEM_DIM, SERVER_DIM, CONTEXT_DIM, PREFIX_DIM = 19, 13, 11, 7


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
            self.scale.copy_(values.std(0, unbiased=False).clamp(min=0.1))
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
            config["slot_seconds"], config["lambda_time"], config["rejection_cost"],
            config["window"] / config["horizon"], float(bool(history)),
            (slot % config["window"]) / config["window"]]


def item_features(choice, config):
    return [float(choice["kind"] == "reject"), float(choice["kind"] == "local"),
            float(choice["kind"] == "edge"), choice["server"] / max(1, len(config["servers"])),
            choice["split"] / 10, choice["minimum"], choice["memory"] / 1e6,
            choice["bandwidth"] / 1e5, choice["cost_fixed"], choice["alpha"],
            choice["beta"], choice["delay_fixed"], choice["deadline"], choice["edge_work"],
            choice["local_work"], choice["weight"], 0.0, 0.0, 0.0]


def upper_frame(observation):
    config = observation["config"]
    active = observation["active"]
    items = []
    for index, server in enumerate(config["servers"]):
        for enabled in (0, 1):
            items.append([0, 0, enabled, index / len(active), 0, server["capacity"],
                          server["memory"] / 1e6, 0, server["idle_power"] if enabled else server["sleep_power"],
                          server["kappa"], server["wake_energy"] * max(0, enabled - active[index]),
                          0, 0, 0, 0, 1, 0, 0, 0])
    history = observation["history"] if config["long_term_feedback"] else []
    profiles = Profiles(config)
    models = sorted(config["profiles"])
    for batch in history:
        for request in batch["public"]:
            work = profiles.bound(request["model"], request["size"], None, config["information"])["local_work"]
            minimum_rates = [choice["minimum"] for choice in candidates(config, profiles, request, [1] * len(active))
                             if choice["kind"] == "edge"]
            items.append([0, 0, 0, models.index(request["model"]) / len(models),
                          sum(request["position"]) / 2, min(minimum_rates, default=0.0),
                          request["size"][0], request["size"][1], work,
                          request["deadline"][0], request["deadline"][1],
                          request["position"][0], request["position"][1],
                          work, 0, request["weight"], request["group_size"],
                          float(np.quantile(minimum_rates, .5)) if minimum_rates else 0.0, len(minimum_rates)])
    return {"servers": server_features(config, active, active, history), "items": items,
            "context": context_features(config, observation["slot"], observation["history"], 0)}


def lower_frame(problem):
    return {"servers": server_features(problem.config, problem.active, problem.ready, problem.history),
            "items": [item_features(choice, problem.config) for options in problem.choices for choice in options],
            "context": context_features(problem.config, problem.slot, problem.history, len(problem.public))}


class SetPolicy(nn.Module):
    def __init__(self, hidden=64, learned_rates=False):
        super().__init__()
        self.learned_rates = learned_rates
        self.register_buffer("value_scale", torch.tensor(1.))
        self.item_norm, self.server_norm, self.context_norm = Normalizer(ITEM_DIM), Normalizer(SERVER_DIM), Normalizer(CONTEXT_DIM)
        self.prefix_norm = Normalizer(PREFIX_DIM)
        self.actor_items, self.actor_servers = mlp(ITEM_DIM, hidden, hidden), mlp(SERVER_DIM, hidden, hidden)
        self.critic_items, self.critic_servers = mlp(ITEM_DIM, hidden, hidden), mlp(SERVER_DIM, hidden, hidden)
        combined = hidden * 3 + CONTEXT_DIM + PREFIX_DIM
        self.score = mlp(combined, 1, hidden, 0.01)
        self.value_head = mlp(hidden * 2 + CONTEXT_DIM, 1, hidden)
        if learned_rates:
            self.rate_head = mlp(combined, 1, hidden, 0.01)
            self.rate_log_std = nn.Parameter(torch.zeros(()))

    def fit_normalizers(self, frames, steps=()):
        frames = list(frames) + [step["prefix_frame"] for step in steps if "prefix_frame" in step]
        self.item_norm.fit([row for frame in frames for row in frame["items"]] +
                           [row for step in steps for row in step["features"]])
        self.server_norm.fit([row for frame in frames for row in frame["servers"]])
        self.context_norm.fit([frame["context"] for frame in frames])
        self.prefix_norm.fit([step["prefix"] for step in steps])

    def encode(self, frame, critic=False):
        items = torch.tensor(frame["items"], dtype=torch.float32).reshape(-1, ITEM_DIM)
        servers = torch.tensor(frame["servers"], dtype=torch.float32).reshape(-1, SERVER_DIM)
        context = self.context_norm(torch.tensor(frame["context"], dtype=torch.float32))
        item_encoder = self.critic_items if critic else self.actor_items
        server_encoder = self.critic_servers if critic else self.actor_servers
        pooled_items = item_encoder(self.item_norm(items)).mean(0) if len(items) else torch.zeros(server_encoder[-1].out_features)
        pooled_servers = server_encoder(self.server_norm(servers)).mean(0)
        return pooled_items, pooled_servers, context

    def value(self, frame):
        return self.value_head(torch.cat(self.encode(frame, critic=True))).squeeze() * self.value_scale

    def distribution(self, frame_encoding, features, prefix, mask):
        encoded_items = self.actor_items(self.item_norm(torch.tensor(features, dtype=torch.float32)))
        shared = torch.cat((*frame_encoding, self.prefix_norm(torch.tensor(prefix, dtype=torch.float32))))
        combined = torch.cat((encoded_items, shared.expand(len(features), -1)), dim=1)
        logits = self.score(combined).flatten()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite policy logits")
        valid = torch.tensor([bool(value) for value in mask], dtype=torch.bool)
        if not valid.any():
            raise ValueError("No legal action")
        logits = logits.masked_fill(~valid, -torch.inf)
        return Categorical(logits=logits), combined

    def replay(self, sample):
        encoding = self.encode(sample["frame"])
        logprob = torch.zeros(())
        entropy = torch.zeros(())
        for step in sample["steps"]:
            conditional_encoding = self.encode(step["prefix_frame"]) if "prefix_frame" in step else encoding
            distribution, combined = self.distribution(conditional_encoding, step["features"], step["prefix"], step["mask"])
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
    def upper_action(self, observation, deterministic=False):
        frame = upper_frame(observation)
        pooled_items, _, context = self.encode(frame)
        logprob = torch.zeros(())
        actions, steps = [], []
        for index in range(len(observation["active"])):
            prefix_frame = dict(frame, servers=[list(row) for row in frame["servers"]])
            for previous_index, enabled in enumerate(actions):
                prefix_frame["servers"][previous_index][6] = enabled
                prefix_frame["servers"][previous_index][7] = enabled * observation["active"][previous_index]
            servers = torch.tensor(prefix_frame["servers"], dtype=torch.float32)
            encoding = pooled_items, self.actor_servers(self.server_norm(servers)).mean(0), context
            features = frame["items"][2 * index:2 * index + 2]
            prefix = [index, sum(actions), sum(actions) / max(1, index), 0, 0, 0, 0]
            distribution, _ = self.distribution(encoding, features, prefix, [True, True])
            action = int(distribution.probs.argmax() if deterministic else distribution.sample())
            steps.append({"features": features, "prefix": prefix, "mask": [True, True], "action": action,
                          "prefix_frame": prefix_frame})
            actions.append(action)
            logprob = logprob + distribution.log_prob(torch.tensor(action))
        sample = {"frame": frame, "steps": steps, "actions": actions}
        value = self.value(frame)
        sample.update(logprob=float(logprob), value=float(value))
        return actions, sample

    @torch.no_grad()
    def lower_action(self, problem, deterministic=False):
        frame = lower_frame(problem)
        encoding = self.encode(frame)
        usage = problem.empty_usage()
        actions, steps, predicted = [], [], []
        edge_count, local_count = 0, 0
        logprob = torch.zeros(())
        for index, options in enumerate(problem.choices):
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
            distribution, combined = self.distribution(encoding, features, prefix, mask)
            action = int(distribution.probs.argmax() if deterministic else distribution.sample())
            choice = options[action]
            logprob = logprob + distribution.log_prob(torch.tensor(action))
            step = {"features": features, "prefix": prefix, "mask": mask, "action": action}
            rate = 0.0
            if self.learned_rates and choice["kind"] == "edge":
                mean = self.rate_head(combined[action]).squeeze()
                normal = Normal(mean, self.rate_log_std.exp())
                latent = mean if deterministic else normal.sample()
                step["rate_latent"] = float(latent)
                logprob = logprob + normal.log_prob(latent)
                rate = float(torch.sigmoid(latent)) * problem.config["servers"][choice["server"]]["capacity"]
            predicted.append(rate)
            actions.append(action)
            steps.append(step)
            problem.reserve(choice, usage)
            edge_count += choice["kind"] == "edge"
            local_count += choice["kind"] == "local"
        sample = {"frame": frame, "steps": steps, "actions": actions,
                  "order": [item["id"] for item in problem.public], "predicted_rates": predicted if self.learned_rates else None}
        value = self.value(frame)
        sample.update(logprob=float(logprob), value=float(value))
        return actions, sample["predicted_rates"], sample
