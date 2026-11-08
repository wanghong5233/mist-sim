import json
from pathlib import Path

import pytest

from mist_sim.config import METHODS, for_method, load_config
from mist_sim.checkpoints import trim_snapshots
from mist_sim.environment import candidates
from mist_sim.profiles import Profiles
from mist_sim.storage import write_json
from mist_sim.training import Trainer


ROOT = Path(__file__).resolve().parents[1]


def test_complete_conditional_inputs_and_per_model_split():
    environment = json.loads((ROOT / "configs/environments/conditional.json").read_text(encoding="utf-8"))
    profile = ROOT / "configs/environments" / environment["profile_file"]
    if not profile.is_file():
        pytest.skip("Private measured profile is not distributed; restore local data to run this integration test")
    config = load_config(ROOT / "configs/conditional.json")
    assert "toy" not in config["profiles"]
    assert len(config["profiles"]) == 3
    for method in METHODS:
        for_method(config, method)
    fixed = for_method(config, "fixed_split")
    for name in fixed["profiles"]:
        request = {"model": name, "size": [224, 224], "deadline": [1, 1],
                   "weight": 1, "position": [25, 25], "fading": [1, 1]}
        choices = candidates(fixed, Profiles(fixed), request, [1] * 8)
        assert all(choice["split"] == fixed["fixed_split"][name] for choice in choices if choice["kind"] == "edge")
    train = config["seeds"]["train"]
    assert all((seed - train) % 10000 for name, seed in config["seeds"].items() if name != "train")


def test_external_profile_tamper_rejected(tmp_path):
    component = tmp_path / "profile.json"
    component.write_text('{}', encoding="utf-8")
    config = tmp_path / "config.json"
    write_json(config, {"profile_file": "profile.json", "profile_sha256": "invalid"})
    with pytest.raises(ValueError, match="checksum"):
        load_config(config)


def test_snapshot_retention_preserves_other_files(tmp_path):
    for number in range(8):
        (tmp_path / f"snapshot_{number}.pt").write_bytes(b"test")
    (tmp_path / "final.pt").write_bytes(b"final")
    trim_snapshots(tmp_path, "snapshot_*.pt", 2)
    assert len(list(tmp_path.glob("snapshot_*.pt"))) == 2
    assert (tmp_path / "final.pt").read_bytes() == b"final"


def test_selected_retention_and_resume_without_training(config, tmp_path):
    config["arrival_probability"] = 0
    from mist_sim.storage import RunStore
    store = RunStore(Path(__file__).resolve().parents[1], config, "check", 23, run_id="retention", output_dir=tmp_path)
    trainer = Trainer(config, 23, store)
    for number in range(4):
        trainer.best_validation_cost = float("inf")
        trainer.validate_cycle(f"cycle_{number:03}")
    assert len(list((store.path / "checkpoints").glob("selected_*.pt"))) == 1
    selected = json.loads((store.path / "selection.json").read_text(encoding="utf-8"))
    restored = Trainer.load(selected["checkpoint_path"])
    assert restored.best_stage == "cycle_003"
