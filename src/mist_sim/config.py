"""Validated, versioned configuration for the V6 simulator."""
import copy
import hashlib
import json
import math
from pathlib import Path

METHODS = ("hpgi", "joint_ppo", "always_on", "fixed_split", "single_timescale", "fixed_window",
           "upper_ppo_greedy", "greedy_consolidation", "nominal", "full_information")


def default_project_root():
    current = Path.cwd()
    source = Path(__file__).resolve().parents[2]
    for candidate in (current, source):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs/smoke.json").is_file():
            return candidate
    return current


def merge_config(base, overlay, path=""):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key not in result:
            if path not in ("ppo_layers.upper.", "ppo_layers.lower."):
                raise ValueError(f"Unknown configuration key: {path}{key}")
            result[key] = copy.deepcopy(value)
            continue
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value, path + key + ".")
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, overlays=(), overrides=()):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    profile_base = path.parent
    if "environment" in config or "algorithm" in config:
        if set(config) != {"environment", "algorithm"}:
            raise ValueError("Composition requires exactly environment and algorithm paths")
        environment_path = path.parent / config["environment"]
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        algorithm = json.loads((path.parent / config["algorithm"]).read_text(encoding="utf-8"))
        if environment.keys() & algorithm.keys():
            raise ValueError("Environment/algorithm keys overlap")
        config = dict(environment, **algorithm)
        profile_base = environment_path.parent
    if "profile_file" in config:
        profile_path = (profile_base / config.pop("profile_file")).resolve()
        expected = config.pop("profile_sha256")
        if hashlib.sha256(profile_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Profile checksum mismatch")
        component = json.loads(profile_path.read_text(encoding="utf-8"))
        if "profiles" in config or "units" in config:
            raise ValueError("External profiles cannot override embedded profiles/units")
        config.update(profiles=component["profiles"], units=component["units"])
    config.setdefault("ppo_layers", {"upper": {}, "lower": {}})
    config.setdefault("validation", {"min_service_rate": None, "selection": "best_feasible_cost"})
    config["validation"].setdefault("trajectories", 1)
    config["validation"].setdefault("sampled", False)
    config["validation"].setdefault("enabled", True)
    config.setdefault("diagnostics", {"phase_interval": 1})
    config.setdefault("runtime", {"torch_threads": 1})
    config["runtime"].setdefault("collection_workers", 0)
    config.setdefault("checkpoint_policy", {"stage_interval": 0, "keep_stages": 2})
    config["checkpoint_policy"].setdefault("mode", "full")
    config["checkpoint_policy"].setdefault("cycle_interval", 250)
    from .convergence import DEFAULT_CONVERGENCE
    config.setdefault("convergence", copy.deepcopy(DEFAULT_CONVERGENCE))
    config.setdefault("trace_retention", {"phases": None, "validations": None, "inputs": None})
    config.setdefault("reward_normalization", {"enabled": False, "max_abs_slot_reward": 10.0})
    config.setdefault("monitoring", {"tensorboard": False, "console": True})
    config.setdefault("stopping", {"max_slots": None, "max_seconds": None})
    for overlay in overlays:
        values = json.loads(Path(overlay).read_text(encoding="utf-8"))
        config = merge_config(config, values)
    for expression in overrides:
        key, separator, value = expression.partition("=")
        if not separator:
            raise ValueError("Overrides require key=JSON-value")
        branch = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(branch, dict) or part not in branch:
                raise ValueError(f"Unknown override: {key}")
            branch = branch[part]
        if not isinstance(branch, dict) or parts[-1] not in branch:
            raise ValueError(f"Unknown override: {key}")
        branch[parts[-1]] = json.loads(value)
    validate(config)
    return config


def layer_settings(config, layer):
    settings = copy.deepcopy(config["ppo"])
    overrides = config.get("ppo_layers", {}).get(layer, {})
    if "gamma" in overrides:
        raise ValueError("Both layers must share the physical-slot discount gamma")
    return merge_config(settings, overrides, f"ppo_layers.{layer}.")


def for_method(config, method):
    result = copy.deepcopy(config)
    result["method"] = method
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    result["information"] = {"nominal": "nominal", "full_information": "full"}.get(method, "robust")
    if method == "single_timescale":
        result["window"] = 1
    result["learned_rates"] = method == "fixed_window"
    if method == "fixed_window":
        result["capacity_mask"] = False
    validate(result)
    return result


def fixed_split_for(config, model):
    value = config["fixed_split"]
    return value[model] if isinstance(value, dict) else value


def validate(config):
    allowed = {"schema", "fixture_kind", "source", "method", "information", "slot_seconds", "horizon",
               "window", "terminals", "k", "arrival_probability", "position_range", "deadline_range",
               "fading_range", "generalization_divisions", "local_rate", "local_kappa", "tx_power",
               "noise_density", "interference", "gain_reference", "reserved_bandwidth", "anonymization_seconds",
               "control_seconds", "backhaul_seconds", "lambda_time", "rejection_cost", "capacity_mask",
               "learned_rates", "fixed_split", "long_term_feedback", "dynamic_split", "initial_active",
               "history_slots", "servers", "access_points", "units", "profiles", "ppo", "ppo_layers",
               "training", "seeds", "device", "validation", "checkpoint_policy", "reward_normalization", "monitoring", "stopping", "runtime", "convergence", "trace_retention", "diagnostics"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {set(config) - allowed}")
    monitoring = config.get("monitoring", {"tensorboard": False, "console": True})
    if set(monitoring) != {"tensorboard", "console"} or any(type(value) is not bool for value in monitoring.values()):
        raise ValueError("Invalid monitoring settings")
    stopping = config.get("stopping", {"max_slots": None, "max_seconds": None})
    if set(stopping) - {"max_slots", "max_seconds"}:
        raise ValueError("Invalid stopping settings")
    if stopping["max_slots"] is not None and (type(stopping["max_slots"]) is not int or stopping["max_slots"] < 1):
        raise ValueError("max_slots must be a positive integer or null")
    if stopping["max_seconds"] is not None and (type(stopping["max_seconds"]) not in (int, float) or stopping["max_seconds"] <= 0):
        raise ValueError("max_seconds must be positive or null")
    policy = config.get("checkpoint_policy", {"stage_interval": 0, "keep_stages": 2})
    if set(policy) - {"stage_interval", "keep_stages", "mode", "cycle_interval"} or any(type(policy[key]) is not int or policy[key] < 0 for key in ("stage_interval", "keep_stages")):
        raise ValueError("Invalid checkpoint retention policy")
    if policy.get("mode", "full") not in ("full", "minimal"):
        raise ValueError("Invalid checkpoint mode")
    if type(policy.get("cycle_interval", 250)) is not int or policy.get("cycle_interval", 250) < 1:
        raise ValueError("Checkpoint cycle interval must be positive")
    if policy.get("mode") == "minimal" and policy["stage_interval"]:
        raise ValueError("Minimal checkpoints do not retain stage snapshots")
    if policy["stage_interval"] and policy["keep_stages"] < 1:
        raise ValueError("Stage retention requires a positive limit")
    def finite(value):
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite(config):
        raise ValueError("Nonfinite configuration")
    from .convergence import DEFAULT_CONVERGENCE
    convergence = config.get("convergence", DEFAULT_CONVERGENCE)
    if set(convergence) != set(DEFAULT_CONVERGENCE) or type(convergence["enabled"]) is not bool:
        raise ValueError("Invalid convergence settings")
    for key in ("min_cycles", "window", "patience", "tail_cycles"):
        if type(convergence[key]) is not int or convergence[key] < 1:
            raise ValueError(f"Invalid convergence {key}")
    if convergence["window"] < 2 or convergence["min_cycles"] < 2 * convergence["window"]:
        raise ValueError("Convergence requires two complete windows before eligibility")
    for key in ("relative_mean_tolerance", "relative_trend_tolerance", "relative_std_tolerance"):
        if type(convergence[key]) not in (int, float) or not 0 < convergence[key] < 1:
            raise ValueError(f"Invalid convergence {key}")
    retention = config.get("trace_retention", {"phases": None, "validations": None, "inputs": None})
    if set(retention) != {"phases", "validations", "inputs"} or any(
            value is not None and (type(value) is not int or value < 1) for value in retention.values()):
        raise ValueError("Trace retention requires positive counts or null")
    normalization = config.get("reward_normalization", {"enabled": False, "max_abs_slot_reward": 10.0})
    if set(normalization) != {"enabled", "max_abs_slot_reward"}:
        raise ValueError("Invalid reward normalization settings")
    if type(normalization["enabled"]) is not bool or normalization["max_abs_slot_reward"] <= 0:
        raise ValueError("Invalid reward normalization bounds")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; sampling always uses CPU")
    for key in ("horizon", "window", "terminals", "k", "history_slots"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key)
    if not 0 <= config["arrival_probability"] <= 1 or config["slot_seconds"] <= 0:
        raise ValueError("Invalid arrival/slot")
    if config["method"] not in METHODS:
        raise ValueError("Unknown method")
    if config["information"] not in ("robust", "nominal", "full"):
        raise ValueError("Unknown information mode")
    if config["deadline_range"][0] <= 0 or config["deadline_range"][1] > config["slot_seconds"]:
        raise ValueError("Deadlines must be positive and within slot")
    if config["deadline_range"][0] > config["deadline_range"][1]:
        raise ValueError("Deadline interval")
    if config["lambda_time"] < 0 or config["rejection_cost"] < 0:
        raise ValueError("Negative cost")
    if not config["servers"] or not config["access_points"]:
        raise ValueError("Empty infrastructure")
    if len(config["initial_active"]) != len(config["servers"]) or any(value not in (0, 1) for value in config["initial_active"]):
        raise ValueError("Invalid initial server states")
    for key in ("local_rate", "local_kappa", "tx_power", "noise_density", "gain_reference", "reserved_bandwidth"):
        if config[key] <= 0:
            raise ValueError(f"Positive {key} required")
    for key in ("interference", "anonymization_seconds", "control_seconds", "backhaul_seconds"):
        if config[key] < 0:
            raise ValueError(f"Nonnegative {key} required")
    if config["fading_range"][0] <= 0 or config["fading_range"][1] < config["fading_range"][0]:
        raise ValueError("Invalid fading range")
    if config["position_range"][0] >= config["position_range"][1]:
        raise ValueError("Invalid position range")
    levels = config["generalization_divisions"]
    if not levels or levels[-1] != 1 or any(not isinstance(level, int) or level < 1 for level in levels):
        raise ValueError("Generalization must end at whole-domain level")
    if any(previous <= following or previous % following for previous, following in zip(levels, levels[1:])):
        raise ValueError("Generalization must use nested decreasing divisions")
    for server in config["servers"]:
        if type(server["ap"]) is not int or set(server["models"]) - config["profiles"].keys():
            raise ValueError("Invalid AP index or unsupported model name")
        if min(server["capacity"], server["memory"], server["kappa"]) <= 0:
            raise ValueError("Invalid server")
        if not 0 <= server["sleep_power"] <= server["idle_power"] or server["wake_energy"] < 0:
            raise ValueError("Invalid server energy")
        if not 0 <= server["ap"] < len(config["access_points"]):
            raise ValueError("Unknown AP")
    if any(access["bandwidth"] <= 0 for access in config["access_points"]):
        raise ValueError("Invalid AP bandwidth")
    if config["method"] == "fixed_split" or config.get("dynamic_split") is False:
        if isinstance(config["fixed_split"], dict) and set(config["fixed_split"]) != set(config["profiles"]):
            raise ValueError("Fixed split mapping must cover every model")
        if any(fixed_split_for(config, name) not in model["legal_splits"] for name, model in config["profiles"].items()):
            raise ValueError("Fixed split must be legal for every configured model")
    threads = config.get("runtime", {}).get("torch_threads", 1)
    workers = config.get("runtime", {}).get("collection_workers", 0)
    if type(workers) is not int or not 0 <= workers <= 32:
        raise ValueError("runtime.collection_workers must be an integer from 0 to 32")
    if type(threads) is not int or threads < 1:
        raise ValueError("runtime.torch_threads must be a positive integer")
    if config["ppo"]["epochs"] < 1 or not 0 <= config["ppo"]["gamma"] <= 1:
        raise ValueError("Invalid PPO settings")
    if any(config["training"][key] < 1 for key in ("pretrain_slots", "fast_slots", "slow_intervals", "cycles", "calibration_episodes")):
        raise ValueError("Training counts must be positive")
    for layer in ("upper", "lower"):
        settings = layer_settings(config, layer)
        expected = {"hidden", "learning_rate", "adam_epsilon", "clip", "gae_lambda", "gamma",
                    "max_grad_norm", "value_coefficient", "entropy_coefficient", "epochs", "minibatch", "target_kl"}
        if set(settings) != expected:
            raise ValueError("Unknown/missing PPO setting")
        for key in ("hidden", "epochs", "minibatch"):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"Invalid PPO {key}")
        for key in ("learning_rate", "adam_epsilon", "max_grad_norm", "target_kl"):
            if settings[key] <= 0:
                raise ValueError(f"Invalid PPO {key}")
        if not 0 < settings["clip"] < 1 or not 0 <= settings["gae_lambda"] <= 1:
            raise ValueError("Invalid PPO clipping/GAE")
        if min(settings["value_coefficient"], settings["entropy_coefficient"]) < 0:
            raise ValueError("Invalid loss weight")
    if set(config.get("ppo_layers", {})) - {"upper", "lower"}:
        raise ValueError("Unknown PPO layer")
    if config["method"] == "joint_ppo" and any(
            layer_settings(config, layer) != config["ppo"] for layer in ("upper", "lower")):
        raise ValueError("joint_ppo uses common ppo settings; remove conflicting ppo_layers overrides")
    if len(set(config["seeds"].values())) != len(config["seeds"]):
        raise ValueError("Training/validation/test seeds must be distinct")
    if any(type(value) is not int or value < 0 for value in config["seeds"].values()):
        raise ValueError("Seeds must be nonnegative integers")
    for key, value in config["training"].items():
        if key not in ("pretrain_slots", "fast_slots", "slow_intervals", "cycles", "calibration_episodes") or type(value) is not int:
            raise ValueError("Unknown/noninteger training count")
    validation = config.get("validation", {"min_service_rate": None, "selection": "best_feasible_cost"})
    if set(validation) - {"min_service_rate", "selection", "trajectories", "sampled", "enabled"} or validation["selection"] != "best_feasible_cost":
        raise ValueError("Invalid validation selection")
    if type(validation.get("sampled", False)) is not bool:
        raise ValueError("validation.sampled must be boolean")
    if type(validation.get("enabled", True)) is not bool:
        raise ValueError("validation.enabled must be boolean")
    diagnostics = config.get("diagnostics", {"phase_interval": 1})
    if set(diagnostics) != {"phase_interval"} or type(diagnostics["phase_interval"]) is not int or diagnostics["phase_interval"] < 1:
        raise ValueError("diagnostics.phase_interval must be a positive integer")
    if type(validation.get("trajectories", 1)) is not int or validation.get("trajectories", 1) < 1:
        raise ValueError("Validation trajectory count must be a positive integer")
    validation_seeds = [config["seeds"]["validation"] + 10000 * index for index in range(validation.get("trajectories", 1))]
    if config["seeds"]["test"] in validation_seeds:
        raise ValueError("Validation substream collides with test seed")
    if validation["min_service_rate"] is not None and not 0 <= validation["min_service_rate"] <= 1:
        raise ValueError("Invalid validation service threshold")
    from .profiles import Profiles
    Profiles(config)
    if config["fixture_kind"] != "synthetic_hand_calculation":
        if not config.get("source"):
            raise ValueError("Measured inputs require source")
