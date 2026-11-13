"""Explicit new experiments from audited parent weights; never a resume bypass."""
import copy
import hashlib
from pathlib import Path
import subprocess

import torch

from .config import validate, EXPLORATION_KEYS
from .storage import digest_file
from .training import Trainer, CHECKPOINT_SCHEMA, implementation_signature, signature


# Reviewed legacy inference: the two revisions have identical network, physics,
# reward, and HPGI inference code. Changes concern diagnostics/runtime and greedy
# acceleration (neither parent uses greedy). New modes are opt-in and tested.
AUDITED_PARENTS = {
    "3e44aea8db2c5edbe999142a3453cef0c8991a386d2ca8dc928118cc11a252e4": "b2191a39302b0a43338eb90247493af214028b1e",
    "33fc2708b828baac29468aab06caadbf687a90ba916861d026d243f934e88d71": "bcfb8db27af2fe4a4f23a09bbb38b74ac4cfd5d4",
}


def read_parent(path, source_revision=None):
    """Validate an immutable parent. Legacy sources require an exact audited pin."""
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA or not payload.get("resumable") or not payload.get("calibrated"):
        raise ValueError("Transfer requires a calibrated, completed-phase parent")
    if signature(payload["config"]) != payload.get("signature"):
        raise ValueError("Parent configuration signature mismatch")
    parent_signature = payload.get("implementation_signature")
    if parent_signature != implementation_signature():
        revision = AUDITED_PARENTS.get(parent_signature)
        if revision is None or source_revision != revision:
            raise ValueError("Parent requires its exact audited source revision")
        root = Path(__file__).resolve().parents[2]
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "src/mist_sim"], cwd=root).decode().splitlines()
        digest = hashlib.sha256()
        for name in sorted(name for name in paths if name.endswith(".py")):
            digest.update(name.removeprefix("src/mist_sim/").encode())
            digest.update(subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root).replace(b"\r\n", b"\n"))
        if digest.hexdigest() != parent_signature:
            raise ValueError("Recorded source does not reproduce parent implementation signature")
    validate(payload["config"])
    if payload["config"]["method"] not in ("hpgi", "harp", "fixed_split"):
        raise ValueError("Audited transfer supports HPGI/HARP/Fixed-Split parents only")
    return payload, dict(path=str(path), sha256=digest_file(path),
        implementation_signature=parent_signature, source_revision=source_revision,
        config_signature=payload["signature"], seed=payload["seed"],
        inherited_training_slots=payload["total_slots"], inherited_cycles=len(payload.get("cycle_history", [])))


def initialize_transfer(path, config, seed, store=None, source_revision=None):
    """Copy networks/statistics, reset Adam/RNG/counters, collect fresh PPO data."""
    payload, receipt = read_parent(path, source_revision)
    previous = payload["config"]
    validate(config)
    mutable = {"method", "dynamic_split", "split_solver", "split_refinement_passes", "split_refinement_scope", "training", "device", "runtime", "diagnostics",
               "monitoring", "stopping", "convergence", "checkpoint_policy", "validation", "trace_retention", "seeds"}
    before, after = copy.deepcopy(previous), copy.deepcopy(config)
    for key in mutable:
        before.pop(key, None)
        after.pop(key, None)
    # Keep architecture, discount, reward units and every physical input fixed.
    for section in ("ppo", "ppo_layers"):
        def strip(settings):
            controls = {"learning_rate", "entropy_coefficient"} | EXPLORATION_KEYS
            if section == "ppo":
                return {k: v for k, v in settings.items() if k not in controls}
            stripped = {k: {a: b for a, b in v.items() if a not in controls} for k, v in settings.items()}
            return {k: v for k, v in stripped.items() if v}
        before[section], after[section] = strip(before.get(section, {})), strip(after.get(section, {}))
    if before != after or config["method"] not in ("hpgi", "harp"):
        raise ValueError("Transfer changed a noncompatible physical/representation/learning contract")
    if seed % 10000 == payload["seed"] % 10000:
        raise ValueError("Transfer must use a fresh training substream")
    trainer = Trainer(config, seed, store)
    trainer.controller.load_state_dict(payload["controller"])
    trainer.reward_normalizer.load_state_dict(payload["reward_normalizer"])
    trainer.value_reward_scales = copy.deepcopy(payload["value_reward_scales"])
    trainer.calibrated = True
    trainer.completed_stages = ["pretrain"]
    trainer.transfer_provenance = dict(parent=receipt, parent_config=previous,
        destination_config_signature=signature(config), optimizer="reset", rng="new_seed",
        counters="new_experiment", pretraining="inherited", fresh_on_policy_rollouts=True)
    if store:
        store.manifest["transfer"] = trainer.transfer_provenance
        store.write_json("manifest.json", store.manifest)
    return trainer
