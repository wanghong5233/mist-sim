"""Profiles use compatible work/s, bits, bytes, joules and seconds."""
import math


class Profiles:
    def __init__(self, config):
        self.models = config["profiles"]
        self._size_keys = {}
        self._bounds = {}
        self.units = config["units"]
        expected = {"work": "reference_work", "rate": "reference_work/s",
                    "tensor": "bit", "memory": "byte", "energy": "J", "time": "s",
                    "bandwidth": "Hz", "power": "W"}
        if self.units != expected:
            raise ValueError("Explicit compatible units required")
        for name, model in self.models.items():
            self._size_keys[name] = {float(size): size for size in model["sizes"]}
            if len(self._size_keys[name]) != len(model["sizes"]):
                raise ValueError("Duplicate numeric input specifications")
            if not model["source"] or not model["legal_splits"]:
                raise ValueError(f"Missing provenance/splits: {name}")
            if not model["sizes"] or type(model["blocks"]) is not int or model["blocks"] < 2:
                raise ValueError("Invalid model blocks/sizes")
            if len(set(model["legal_splits"])) != len(model["legal_splits"]) or any(type(split) is not int for split in model["legal_splits"]):
                raise ValueError("Unique integer split IDs required")
            for size, row in model["sizes"].items():
                if not math.isfinite(float(size)) or float(size) <= 0 or row["full_work"] <= 0:
                    raise ValueError("Invalid size/work")
                for split in model["legal_splits"]:
                    if not 1 <= split < model["blocks"]:
                        raise ValueError("Only internal splits permitted")
                    entry = row["splits"][str(split)]
                    if not all(math.isfinite(value) for value in [row["full_work"], entry["local_work"],
                               entry["edge_work"], entry["memory_bytes"], *entry["crossing_tensors_bits"]]):
                        raise ValueError("Nonfinite profile")
                    if entry["local_work"] <= 0 or entry["edge_work"] <= 0:
                        raise ValueError("Positive work required")
                    if not entry["crossing_tensors_bits"] or min(entry["crossing_tensors_bits"]) < 0:
                        raise ValueError("All crossing tensors must be declared")
                    if entry["memory_bytes"] <= 0:
                        raise ValueError("Positive memory required")

    def sizes(self, model):
        return sorted(float(value) for value in self.models[model]["sizes"])

    def row(self, model, size):
        return self.models[model]["sizes"][self._size_keys[model][float(size)]]

    def bound(self, model, interval, split, mode):
        key = model, tuple(interval), split, mode
        if key not in self._bounds:
            if len(self._bounds) >= 4096:
                self._bounds.clear()
            self._bounds[key] = self._bound(model, interval, split, mode)
        return dict(self._bounds[key])

    def _bound(self, model, interval, split, mode):
        sizes = [size for size in self.sizes(model) if interval[0] <= size <= interval[1]]
        if not sizes:
            raise ValueError("Interval contains no profiled size")
        if mode == "nominal":
            midpoint = sum(interval) / 2
            sizes = [min(sizes, key=lambda size: (abs(size - midpoint), size))]
        rows = [self.row(model, size) for size in sizes]
        if split is None:
            return {"local_work": max(row["full_work"] for row in rows),
                    "edge_work": 0.0, "bits": 0.0, "memory": 0.0}
        entries = [row["splits"][str(split)] for row in rows]
        return {"local_work": max(entry["local_work"] for entry in entries),
                "edge_work": max(entry["edge_work"] for entry in entries),
                "bits": max(sum(entry["crossing_tensors_bits"]) for entry in entries),
                "memory": max(entry["memory_bytes"] for entry in entries)}
