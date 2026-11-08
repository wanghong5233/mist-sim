"""One physical slot per transition; controllers only receive PublicProblem."""
import copy
import math
import time
import numpy as np
from .allocation import solve_p3, project_rates
from .anonymity import publish
from .profiles import Profiles
from .config import fixed_split_for


def radio_rate(config, server, position, fading):
    access = config["access_points"][server["ap"]]
    gain = config["gain_reference"] * fading / (1 + (position - access["position"]) ** 2)
    bandwidth = config["reserved_bandwidth"]
    return bandwidth * math.log2(1 + config["tx_power"] * gain /
                                 (config["noise_density"] * bandwidth + config["interference"]))


def execute_request(config, profile, real, choice, rate, controller_seconds, memory_failed):
    overhead = config["anonymization_seconds"] + max(config["control_seconds"], controller_seconds)
    local_time = profile["local_work"] / config["local_rate"]
    transmit_time, backhaul_time, edge_time = 0.0, 0.0, 0.0
    edge_power = 0.0
    if choice["kind"] == "edge":
        server = config["servers"][choice["server"]]
        radio = radio_rate(config, server, real["position"], real["fading"])
        transmit_time = profile["bits"] / radio
        backhaul_time = config["backhaul_seconds"]
        edge_time = profile["edge_work"] / rate
        edge_power = server["kappa"] * rate ** 3
    planned = overhead + local_time + transmit_time + backhaul_time + edge_time
    elapsed = min(overhead, real["deadline"])
    energy = {"local": 0.0, "transmit": 0.0, "edge": 0.0}
    if overhead > real["deadline"] + 1e-9:
        return energy, None, elapsed, planned, "deadline"
    phases = [("local", local_time, config["local_kappa"] * config["local_rate"] ** 3),
              ("transmit", transmit_time, config["tx_power"]),
              ("backhaul", backhaul_time, 0.0), ("edge", edge_time, edge_power)]
    for phase, duration, power in phases:
        if phase == "edge" and memory_failed:
            return energy, None, elapsed, planned, "execution_memory"
        available = max(0.0, real["deadline"] - elapsed)
        spent = min(duration, available)
        if phase in energy:
            energy[phase] += spent * power
        elapsed += spent
        if duration > available + 1e-9:
            return energy, None, elapsed, planned, "deadline"
    return energy, planned, elapsed, planned, None


def candidates(config, profiles, request, ready):
    mode = config["information"]
    deadline = sum(request["deadline"]) / 2 if mode == "nominal" else request["deadline"][0]
    overhead = config["anonymization_seconds"] + config["control_seconds"]
    weight = request["weight"]
    choices = [{"kind": "reject", "server": -1, "split": -1, "minimum": 0.0, "memory": 0.0,
                "bandwidth": 0.0, "ap": -1, "fixed_energy": 0.0, "delay_fixed": 0.0,
                "alpha": 0.0, "beta": 0.0, "edge_work": 0.0, "local_work": 0.0, "bits": 0.0,
                "cost_fixed": config["rejection_cost"] * weight, "deadline": deadline, "weight": weight}]
    local = profiles.bound(request["model"], request["size"], None, mode)
    local_delay = overhead + local["local_work"] / config["local_rate"]
    local_energy = config["local_kappa"] * local["local_work"] * config["local_rate"] ** 2
    if local_delay <= deadline + 1e-12:
        choices.append(dict(choices[0], kind="local", split=profiles.models[request["model"]]["blocks"],
                            local_work=local["local_work"], fixed_energy=local_energy,
                            delay_fixed=local_delay,
                            cost_fixed=local_energy + config["lambda_time"] * weight * local_delay))
    for server_index, server in enumerate(config["servers"]):
        if not ready[server_index] or request["model"] not in server["models"]:
            continue
        position_bounds = request["position"]
        if mode == "nominal":
            position = sum(position_bounds) / 2
            fading = sum(request["fading"]) / 2
        else:
            position = max(position_bounds, key=lambda value: abs(value - config["access_points"][server["ap"]]["position"]))
            fading = request["fading"][0]
        rate = radio_rate(config, server, position, fading)
        splits = profiles.models[request["model"]]["legal_splits"]
        if config["method"] == "fixed_split" or config.get("dynamic_split") is False:
            splits = [fixed_split_for(config, request["model"])]
        for split in splits:
            profile = profiles.bound(request["model"], request["size"], split, mode)
            delay = overhead + profile["local_work"] / config["local_rate"] + profile["bits"] / rate + config["backhaul_seconds"]
            slack = deadline - delay
            if slack <= 0:
                continue
            minimum = profile["edge_work"] / slack
            if minimum > server["capacity"] + 1e-10 or profile["memory"] > server["memory"]:
                continue
            if config["reserved_bandwidth"] > config["access_points"][server["ap"]]["bandwidth"]:
                continue
            energy = config["local_kappa"] * profile["local_work"] * config["local_rate"] ** 2 + config["tx_power"] * profile["bits"] / rate
            choices.append(dict(choices[0], kind="edge", server=server_index, split=split,
                                minimum=minimum, memory=profile["memory"], bandwidth=config["reserved_bandwidth"],
                                ap=server["ap"], fixed_energy=energy, delay_fixed=delay,
                                alpha=server["kappa"] * profile["edge_work"],
                                beta=config["lambda_time"] * weight * profile["edge_work"],
                                edge_work=profile["edge_work"], local_work=profile["local_work"], bits=profile["bits"],
                                cost_fixed=energy + config["lambda_time"] * weight * delay))
    return choices


class PublicProblem:
    def __init__(self, config, public, active, ready, slot, history):
        self.config = config
        self.public = copy.deepcopy(public)
        self.active, self.ready = list(active), list(ready)
        self.slot = slot
        self.history = copy.deepcopy(history)
        profiles = Profiles(config)
        self.choices = [candidates(config, profiles, item, ready) for item in self.public]

    def empty_usage(self):
        return {"compute": np.zeros(len(self.active)), "memory": np.zeros(len(self.active)),
                "bandwidth": np.zeros(len(self.config["access_points"]))}

    def mask(self, choices, usage, capacity=True):
        return [bool(choice["kind"] != "edge" or not capacity or
                (usage["compute"][choice["server"]] + choice["minimum"] <= self.config["servers"][choice["server"]]["capacity"] + 1e-9 and
                 usage["memory"][choice["server"]] + choice["memory"] <= self.config["servers"][choice["server"]]["memory"] + 1e-9 and
                 usage["bandwidth"][choice["ap"]] + choice["bandwidth"] <= self.config["access_points"][choice["ap"]]["bandwidth"] + 1e-9))
                for choice in choices]

    @staticmethod
    def reserve(choice, usage):
        if choice["kind"] == "edge":
            usage["compute"][choice["server"]] += choice["minimum"]
            usage["memory"][choice["server"]] += choice["memory"]
            usage["bandwidth"][choice["ap"]] += choice["bandwidth"]

    def allocate(self, actions, predicted=None, allow_projection=False):
        if len(actions) != len(self.choices):
            raise ValueError("One decision per published request")
        if predicted is not None and len(predicted) != len(actions):
            raise ValueError("Rate/action size mismatch")
        if predicted is not None and not np.isfinite(predicted).all():
            raise ValueError("Nonfinite predicted rates")
        usage = self.empty_usage()
        executed, violations = [], []
        for index, (options, action) in enumerate(zip(self.choices, actions)):
            if not isinstance(action, (int, np.integer)) or not 0 <= action < len(options):
                raise ValueError("Invalid action index")
            if not self.mask(options, usage)[action]:
                if not allow_projection:
                    raise ValueError("Joint capacity violation")
                violations.append({"request": index, "reason": "joint_capacity", "action": int(action)})
                action = 0
            executed.append(int(action))
            self.reserve(options[action], usage)
        rates = np.zeros(len(actions))
        for server_index, server in enumerate(self.config["servers"]):
            indices = [index for index, action in enumerate(executed)
                       if self.choices[index][action]["server"] == server_index]
            selected = [self.choices[index][executed[index]] for index in indices]
            minimum = [choice["minimum"] for choice in selected]
            if predicted is None:
                values = solve_p3([choice["alpha"] for choice in selected],
                                  [choice["beta"] for choice in selected], minimum, server["capacity"])
            else:
                guesses = [predicted[index] for index in indices]
                if any(guess < bound for guess, bound in zip(guesses, minimum)) or sum(guesses) > server["capacity"]:
                    violations.append({"server": server_index, "reason": "rate_capacity", "predicted": guesses})
                values = project_rates(guesses, minimum, server["capacity"])
            rates[indices] = values
        return executed, rates, usage, violations

    def scheduling_cost(self, actions):
        executed, rates, _, _ = self.allocate(actions)
        return sum(self.choice_cost(self.choices[index][action], rates[index])
                   for index, action in enumerate(executed))

    @staticmethod
    def choice_cost(choice, rate):
        return choice["cost_fixed"] + (choice["alpha"] * rate ** 2 + choice["beta"] / rate
                                       if choice["kind"] == "edge" else 0.0)

    def as_dict(self):
        return {"config": self.config, "public": self.public, "active": self.active, "ready": self.ready,
                "slot": self.slot, "history": self.history}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class Environment:
    def __init__(self, config, trace):
        self.config = copy.deepcopy(config)
        self._trace = copy.deepcopy(trace)
        if len(trace) != config["horizon"]:
            raise ValueError("Trace horizon mismatch")
        self.profiles = Profiles(config)
        for batch in trace:
            if len(batch) > config["terminals"] or len({item["id"] for item in batch}) != len(batch):
                raise ValueError("Invalid request batch")
            terminals = [item["terminal"] for item in batch if "terminal" in item]
            if len(set(terminals)) != len(terminals):
                raise ValueError("More than one request per terminal per slot")
            for item in batch:
                if not isinstance(item["id"], str) or not isinstance(item["class"], str):
                    raise ValueError("String request ID and public class required")
                if not math.isfinite(item["weight"]) or item["weight"] <= 0 or not math.isfinite(item["tie"]):
                    raise ValueError("Invalid public weight/tie key")
                if item["size"] not in self.profiles.sizes(item["model"]):
                    raise ValueError("Unprofiled input")
                for field, bounds in (("position", config["position_range"]), ("deadline", config["deadline_range"]),
                                      ("fading", config["fading_range"])):
                    if not bounds[0] <= item[field] <= bounds[1]:
                        raise ValueError(f"Out-of-domain {field}")
        self.slot = 0
        self.active = list(config["initial_active"])
        self.history = []
        self.records = []
        self._pending = None

    @property
    def done(self):
        return self.slot == self.config["horizon"]

    def upper_observation(self):
        lookback = max(self.config["history_slots"], self.config["window"]) if self.config["method"] == "greedy_consolidation" else self.config["history_slots"]
        history = self.history[-lookback:] if self.config["long_term_feedback"] else []
        return {"active": list(self.active), "slot": self.slot, "history": copy.deepcopy(history),
                "servers": copy.deepcopy(self.config["servers"]),
                "config": self.config}

    def begin_slot(self, active):
        if self.done or self._pending is not None:
            raise RuntimeError("Invalid slot lifecycle")
        if len(active) != len(self.active) or any(value not in (0, 1) for value in active):
            raise ValueError("Binary server vector required")
        previous = list(self.active)
        self.active = list(active)
        ready = [old * new for old, new in zip(previous, active)]
        public = publish(self._trace[self.slot], self.config, self.config["information"])
        problem = PublicProblem(self.config, public, active, ready, self.slot,
                                self.history[-self.config["history_slots"]:] if self.config["long_term_feedback"] else [])
        self._pending = (previous, problem)
        return problem

    def execute(self, actions, predicted=None, controller_seconds=0.0, measured_latency=False):
        if self._pending is None:
            raise RuntimeError("begin_slot required")
        if not math.isfinite(controller_seconds) or controller_seconds < 0:
            raise ValueError("Invalid control time")
        previous, problem = self._pending
        projection = self.config["learned_rates"] or not self.config["capacity_mask"]
        allocation_started = time.perf_counter()
        executed, rates, usage, violations = problem.allocate(actions, predicted, projection)
        allocation_seconds = time.perf_counter() - allocation_started
        if measured_latency:
            controller_seconds += allocation_seconds
        proposed_usage = problem.empty_usage()
        proposed_rates = np.zeros(len(self.active))
        for index, (options, action) in enumerate(zip(problem.choices, actions)):
            choice = options[action]
            problem.reserve(choice, proposed_usage)
            if choice["kind"] == "edge" and predicted is not None:
                proposed_rates[choice["server"]] += predicted[index]
        if self.config["learned_rates"] and predicted is None:
            raise ValueError("Fixed-window requires learned rates")
        raw = self._trace[self.slot]
        lookup = {item["id"]: item for item in raw}
        actual_memory = np.zeros(len(self.active))
        for request, options, action in zip(problem.public, problem.choices, executed):
            choice = options[action]
            if choice["kind"] == "edge":
                real = lookup[request["id"]]
                profile = self.profiles.bound(real["model"], [real["size"]] * 2, choice["split"], "full")
                actual_memory[choice["server"]] += profile["memory"]
        static = sum(self.config["slot_seconds"] * (server["idle_power"] if active else server["sleep_power"])
                     for server, active in zip(self.config["servers"], self.active))
        wake = sum(server["wake_energy"] * max(0, new - old)
                   for server, old, new in zip(self.config["servers"], previous, self.active))
        energy = {"static": static, "wake": wake, "local": 0.0, "transmit": 0.0, "edge": 0.0}
        conservative_energy = static + wake
        conservative_delay = 0.0
        penalty = 0.0
        tasks = []
        for index, (request, options, action) in enumerate(zip(problem.public, problem.choices, executed)):
            real = lookup[request["id"]]
            choice, rate = options[action], float(rates[index])
            task = {"id": request["id"], "action": choice["kind"], "server": choice["server"],
                    "candidate": copy.deepcopy(choice),
                    "split": choice["split"], "rate": rate, "minimum": choice["minimum"],
                    "weight": request["weight"], "deadline": real["deadline"], "delay": None,
                    "energy": {"local": 0.0, "transmit": 0.0, "edge": 0.0}, "on_time": False,
                    "conservative_cost": problem.choice_cost(choice, rate), "reason": None}
            if choice["kind"] == "reject":
                penalty += choice["cost_fixed"]
                task["reason"] = "projection" if action != actions[index] else "policy_reject"
            else:
                profile = self.profiles.bound(real["model"], [real["size"]] * 2,
                                              choice["split"] if choice["kind"] == "edge" else None, "full")
                memory_failed = choice["kind"] == "edge" and actual_memory[choice["server"]] > self.config["servers"][choice["server"]]["memory"] + 1e-9
                task["energy"], task["delay"], task["elapsed"], task["planned_delay"], task["reason"] = execute_request(
                    self.config, profile, real, choice, rate, controller_seconds, memory_failed)
                task["on_time"] = task["reason"] is None
                conservative_energy += choice["fixed_energy"] + choice["alpha"] * rate ** 2
                conservative_delay += request["weight"] * (choice["delay_fixed"] + (choice["edge_work"] / rate if rate else 0))
                for key, value in task["energy"].items():
                    energy[key] += value
            tasks.append(task)
        released_ids = {item["id"] for item in problem.public}
        suppressed = [item["id"] for item in raw if item["id"] not in released_ids]
        cost = conservative_energy + self.config["lambda_time"] * conservative_delay + penalty
        record = {"slot": self.slot, "raw": raw, "public": problem.public, "previous_active": previous,
                  "active": list(self.active), "ready": problem.ready, "actions": list(actions),
                  "executed_actions": executed, "predicted_rates": predicted, "rates": rates.tolist(),
                  "usage": {key: value.tolist() for key, value in usage.items()},
                  "actual_memory_bytes": actual_memory.tolist(),
                  "proposed_usage": {key: value.tolist() for key, value in proposed_usage.items()},
                  "proposed_rate_totals": proposed_rates.tolist(),
                  "projection_violations": violations, "tasks": tasks, "suppressed": suppressed,
                  "energy": energy, "conservative_energy": conservative_energy,
                  "weighted_conservative_delay": conservative_delay, "rejection_penalty": penalty,
                  "cost": cost, "reward": -cost, "controller_seconds": controller_seconds,
                  "measured_allocation_seconds": allocation_seconds,
                  "control_overrun": controller_seconds > self.config["control_seconds"],
                  "arrivals": len(raw), "completed": sum(task["on_time"] for task in tasks)}
        allocated = [sum(float(rates[index]) for index, action in enumerate(executed)
                         if problem.choices[index][action]["server"] == server_index)
                     for server_index in range(len(self.active))]
        history = {"public": problem.public, "cost": cost, "completed": record["completed"],
                   "accepted": sum(task["action"] != "reject" for task in tasks), "allocated_compute": allocated,
                   "released": len(problem.public), "compute": usage["compute"].tolist(),
                   "active": list(self.active)}
        self.history.append(history)
        self.records.append(record)
        self.slot += 1
        self._pending = None
        return record
