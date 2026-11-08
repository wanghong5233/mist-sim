from pathlib import Path
import runpy
import subprocess
import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISH = runpy.run_path(str(ROOT / "scripts/maintenance/check_publish.py"))
INDEX = runpy.run_path(str(ROOT / "scripts/validation/build_verification_index.py"))


def test_private_assets_rejected_and_source_allowed():
    for name in ("results/curve.csv", "runs/one/weights.pt", "data/processed/profile.json", "SETUP.md",
                 ".env", "private.key", "references/MERA/README.md", "EXPERIMENT_PROTOCOL.md"):
        assert PUBLISH["private_path"](name)
    for name in ("README.md", "src/mist_sim/training.py", "configs/smoke.json", "data/README.md",
                 "references/README.md", ".env.example", "scripts/data/profile_models.py"):
        assert not PUBLISH["private_path"](name)


def test_publish_guard_detects_already_committed_private_files(tmp_path):
    def git(*arguments):
        subprocess.run(["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True)
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@users.noreply.github.com")
    (tmp_path / "README.md").write_text("code", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/private.csv").write_text("private sample", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "fixture")
    git("rm", "results/private.csv")
    git("commit", "-m", "remove current file")
    report = PUBLISH["inspect_repository"](tmp_path)
    assert not report["passed"]
    assert any(issue["file"] == "results/private.csv" for issue in report["issues"])


def test_manuscript_is_an_explicit_optional_dependency(tmp_path):
    assert INDEX["paper_checksums"](None) == {}
    with pytest.raises(FileNotFoundError):
        INDEX["paper_checksums"](tmp_path)
    for name in ("main.tex", "main.pdf"):
        (tmp_path / name).write_bytes(b"test")
    checksums = INDEX["paper_checksums"](tmp_path)
    assert len(checksums["main.tex"]) == 64
    assert checksums["main.tex"] == checksums["main.pdf"]
