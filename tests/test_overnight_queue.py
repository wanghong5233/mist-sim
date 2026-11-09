"""Queue safety and dependency flow without launching real training."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


spec = importlib.util.spec_from_file_location("overnight_queue", Path(__file__).resolve().parents[1] /
                                           "scripts/training/overnight_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fake_run(root, name, cost=100, status="completed", count=200):
    directory = root / "logs/runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(method="hpgi" if name == "hpgi" else "joint_ppo",
                  seeds=dict(train=4200001, validation=4300017, test=4400019),
                  training=dict(cycles=3000), convergence=dict(min_cycles=100),
                  runtime=dict(windows_performance=False))
    queue.write_json(directory / "manifest.json", dict(status=status, exit_code=0 if status == "completed" else 1,
                                                      source=dict(commit="source")))
    queue.write_json(directory / "resolved_config.json", config)
    queue.write_json(directory / "process_policy.json", dict(pid=1234))
    results = root / "results" / name
    results.mkdir(parents=True, exist_ok=True)
    with (results / "cycles.csv").open("w", encoding="utf-8") as stream:
        stream.write("cycle_index,cost_per_slot,service,phase_seconds\n")
        for index in range(count):
            stream.write(f"{index+1},{cost},0.7,50\n")
    return config


def args(root):
    return SimpleNamespace(root=root, seed=[4200002, 4200003], candidate=["hpgi", "joint"],
                           wait_run="hpgi", queue_id="overnight", python="python.exe", max_hours=24,
                           poll_seconds=0.01)


def test_running_and_failed_candidates_never_selected(tmp_path):
    fake_run(tmp_path, "hpgi", status="running")
    assert queue.completed_score(tmp_path, "hpgi") is None
    fake_run(tmp_path, "hpgi", status="failed")
    with pytest.raises(RuntimeError, match="did not complete"):
        queue.completed_score(tmp_path, "hpgi")


def test_real_iteration_count_and_nonfinite_rejected(tmp_path):
    fake_run(tmp_path, "hpgi", count=199)
    with pytest.raises(ValueError, match="Expected 200"):
        queue.completed_score(tmp_path, "hpgi", expected_cycles=200)
    assert queue.completed_score(tmp_path, "hpgi")["iterations"] == 199
    fake_run(tmp_path, "hpgi", cost="nan")
    with pytest.raises(ValueError, match="Non-finite"):
        queue.completed_score(tmp_path, "hpgi")


def test_selection_minimizes_cost_not_sign_of_reward(tmp_path):
    fake_run(tmp_path, "hpgi", cost=170)
    fake_run(tmp_path, "joint", cost=157, count=199)
    scores = [queue.completed_score(tmp_path, name) for name in ("hpgi", "joint")]
    assert queue.choose(scores)["run"] == "joint"


def test_new_seed_changes_only_seed_budget_runtime(tmp_path):
    config = fake_run(tmp_path, "hpgi")
    previous = copy.deepcopy(config)
    result = queue.new_seed_config(config, 4200002)
    assert config == previous
    assert result["training"]["cycles"] == result["convergence"]["min_cycles"] == 200
    assert result["seeds"]["train"] == 4200002
    assert result["runtime"]["windows_performance"] is True
    for seed in previous["seeds"].values():
        with pytest.raises(ValueError):
            queue.new_seed_config(config, seed)


def test_visible_terminal_command_preserves_unicode_quotes_and_stdout(tmp_path):
    command = queue.terminal_command(tmp_path / "中文'路径", "python.exe", tmp_path / "config.json",
                                     "example", 4200002, tmp_path / "exit.json", "Training")
    text = base64.b64decode(command[-1]).decode("utf-16le")
    assert command[:3] == ["powershell.exe", "-NoExit", "-EncodedCommand"]
    assert "中文''路径" in text and "--seed '4200002'" in text
    assert "$LASTEXITCODE" in text and "WriteAllText" in text
    assert "Tee-Object" not in text and "RedirectStandardOutput" not in text


def test_disappeared_dependency_stops_without_launch(tmp_path, monkeypatch):
    fake_run(tmp_path, "hpgi", status="running")
    fake_run(tmp_path, "joint")
    monkeypatch.setattr(queue, "source_hash", lambda root: "frozen")
    awake = []
    monkeypatch.setattr(queue, "keep_awake", awake.append)
    monkeypatch.setattr(queue, "process_alive", lambda pid: False)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="disappeared"):
        queue.execute(args(tmp_path))
    assert awake == [True, False]
    assert queue.read_json(tmp_path / "logs/queues/overnight/status.json")["status"] == "failed"


def test_frozen_source_guard_and_duplicate_queue(tmp_path, monkeypatch):
    fake_run(tmp_path, "hpgi")
    fake_run(tmp_path, "joint")
    values = iter(["before", "after"])
    monkeypatch.setattr(queue, "source_hash", lambda root: next(values))
    monkeypatch.setattr(queue, "keep_awake", lambda enabled: None)
    with pytest.raises(RuntimeError, match="source changed"):
        queue.execute(args(tmp_path))
    with pytest.raises(FileExistsError):
        queue.execute(args(tmp_path))


def test_wait_then_launch_exactly_two_independent_jobs_and_plot(tmp_path, monkeypatch):
    fake_run(tmp_path, "hpgi", cost=160, status="running")
    fake_run(tmp_path, "joint", cost=157, count=199)
    monkeypatch.setattr(queue, "source_hash", lambda root: "frozen")
    monkeypatch.setattr(queue, "keep_awake", lambda enabled: None)
    monkeypatch.setattr(queue, "process_alive", lambda pid: False if sleeps else True)
    sleeps = []
    launches = []

    def finish_dependency(seconds):
        assert not launches
        sleeps.append(seconds)
        fake_run(tmp_path, "hpgi", cost=160)

    monkeypatch.setattr(queue.time, "sleep", finish_dependency)

    def launch(command, **kwargs):
        text = base64.b64decode(command[-1]).decode("utf-16le")
        seed = 4200002 + len(launches)
        run = f"joint_ppo-finite-horizon-gpu-b128-w4-highqos-200-seed{seed}-overnight"
        assert run in text and str(seed) in text
        fake_run(tmp_path, run)
        queue.write_json(tmp_path / f"logs/queues/overnight/seed{seed}-exit.json", dict(exit_code=0))
        launches.append(command)
        return SimpleNamespace(pid=1000+len(launches), poll=lambda: None)

    monkeypatch.setattr(queue.subprocess, "CREATE_NEW_CONSOLE", 16, raising=False)
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    plots = []
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs:
                        (plots.append(command) or SimpleNamespace(returncode=0)))
    assert queue.execute(args(tmp_path)) == 0
    status = queue.read_json(tmp_path / "logs/queues/overnight/status.json")
    assert len(sleeps) == 1 and len(launches) == 2
    assert status["selected"]["run"] == "joint"
    assert status["status"] == "completed" and status["keep_awake"] is False
    assert [job["seed"] for job in status["jobs"]] == [4200002, 4200003]
    assert len(plots) == 1 and plots[0].count("--curve") == 3
