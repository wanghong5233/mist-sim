"""Gateway-only true records and fixed interval generalization."""
import numpy as np
from .profiles import Profiles


def generate_trace(config, seed):
    generator = np.random.default_rng(seed)
    profiles = Profiles(config)
    models = sorted(config["profiles"])
    trace = []
    for slot in range(config["horizon"]):
        batch = []
        for terminal in range(config["terminals"]):
            if generator.random() >= config["arrival_probability"]:
                continue
            model = models[int(generator.integers(len(models)))]
            batch.append({"id": f"{slot}:{int(generator.integers(0, 2**63))}", "terminal": terminal,
                          "model": model, "class": "standard",
                          "weight": 1.0, "size": float(generator.choice(profiles.sizes(model))),
                          "position": float(generator.uniform(*config["position_range"])),
                          "deadline": float(generator.uniform(*config["deadline_range"])),
                          "fading": float(generator.uniform(*config["fading_range"])),
                          "tie": float(generator.random())})
        trace.append(batch)
    return trace


def publish(batch, config, mode="robust"):
    if mode == "full":
        released_ids = {item["id"] for item in publish(batch, config, "robust")}
        released = [{"id": item["id"], "model": item["model"], "class": item["class"],
                 "weight": item["weight"], "size": [item["size"]] * 2,
                 "position": [item["position"]] * 2, "deadline": [item["deadline"]] * 2,
                 "fading": [item["fading"]] * 2, "tie": item["tie"], "group_size": 1}
                for item in batch if item["id"] in released_ids]
        return sorted(released, key=lambda item: (-item["weight"], item["deadline"][0], item["tie"]))
    remaining = list(batch)
    released = []
    sizes = sorted({float(size) for model in config["profiles"].values() for size in model["sizes"]})
    limits = [config["position_range"], [min(sizes), max(sizes)], config["deadline_range"]]
    for divisions in config["generalization_divisions"]:
        groups = {}
        for item in remaining:
            coordinates = [item["position"], item["size"], item["deadline"]]
            indices = tuple(min(divisions - 1, int((value - bounds[0]) / max(bounds[1] - bounds[0], 1e-12) * divisions))
                            for value, bounds in zip(coordinates, limits))
            key = (item["model"], item["class"], item["weight"], indices)
            groups.setdefault(key, []).append(item)
        next_remaining = []
        for key, members in groups.items():
            if len(members) < config["k"]:
                next_remaining.extend(members)
                continue
            intervals = [[bounds[0] + (bounds[1] - bounds[0]) * index / divisions,
                          bounds[0] + (bounds[1] - bounds[0]) * (index + 1) / divisions]
                         for bounds, index in zip(limits, key[3])]
            for item in members:
                released.append({"id": item["id"], "model": item["model"], "class": item["class"],
                                 "weight": item["weight"], "position": intervals[0],
                                 "size": intervals[1], "deadline": intervals[2],
                                 "fading": list(config["fading_range"]), "tie": item["tie"],
                                 "group_size": len(members)})
        remaining = next_remaining
    return sorted(released, key=lambda item: (-item["weight"], item["deadline"][0], item["tie"]))
