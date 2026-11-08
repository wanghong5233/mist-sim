import json
import re

import pytest

from mist_sim.storage import RunStore, readable_run_id, validation_workspace


def test_names_explain_purpose_and_preserve_uniqueness():
    config = {"method": "hpgi"}
    first = readable_run_id(config, "python -m mist_sim train", 42)
    second = readable_run_id(config, "python -m mist_sim train", 42)
    assert first != second
    assert re.fullmatch(r"train-hpgi-seed42__\d{8}T\d{6}_[a-f0-9]{10}", first)
    assert readable_run_id({}, "scripts/data/profile_models.py", 7).startswith("dnn-profiling-models-seed7__")
    assert readable_run_id(config, "scripts/data/check_inputs.py", 7).startswith("input-check-hpgi-seed7__")


def test_explicit_name_and_nonoverwrite(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    config = {"profiles": {}, "method": "hpgi"}
    store = RunStore(tmp_path, config, "check", 7, "input-check-test")
    store.finish()
    assert store.path == tmp_path / "logs/runs/input-check-test"
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "results").exists()
    assert json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "input-check-test"
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, config, "check", 7, "input-check-test")


@pytest.mark.parametrize("fail", [False, True])
def test_validation_workspace_cleans_success_and_failure(fail):
    directory = None
    try:
        with validation_workspace() as directory:
            (directory / "checkpoint.pt").write_bytes(b"disposable test artifact")
            if fail:
                raise RuntimeError("expected test failure")
    except RuntimeError:
        assert fail
    assert directory is not None and not directory.exists()


def test_debug_retention_is_explicit_and_does_not_overwrite(tmp_path):
    requested = tmp_path / "debug"
    with validation_workspace(requested) as directory:
        (directory / "trace.json").write_text("{}", encoding="utf-8")
    assert (requested / "trace.json").is_file()
    with pytest.raises(FileExistsError):
        with validation_workspace(requested):
            pass


def test_output_directory_keeps_source_identity_separate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    output = tmp_path / "disposable"
    store = RunStore(source, {"profiles": {}}, "check", 1, output_dir=output)
    store.finish()
    assert store.path.parent == output
    assert not (source / "logs").exists()
    assert "pyproject.toml" in store.manifest["source"]["files"]


def test_check_cli_uses_disposable_output(monkeypatch):
    from mist_sim import cli
    captured = []

    def inspect_output(args):
        assert args.output_dir.parents[1].is_dir()
        captured.append(args.output_dir.parents[1])

    monkeypatch.setattr(cli, "execute", inspect_output)
    cli.main(["check"])
    assert len(captured) == 1 and not captured[0].exists()


@pytest.mark.parametrize("command", ["train", "evaluate", "reference"])
def test_synthetic_cli_does_not_persist_by_default(monkeypatch, command):
    from mist_sim import cli
    captured = []

    def inspect_output(args, config, trainer):
        assert config["fixture_kind"] == "synthetic_hand_calculation"
        assert args.output_dir.parents[1].is_dir()
        captured.append(args.output_dir.parents[1])

    monkeypatch.setattr(cli, "run_command", inspect_output)
    cli.main([command])
    assert len(captured) == 1 and not captured[0].exists()
