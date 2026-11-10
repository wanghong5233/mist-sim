import copy

import pytest

from mist_sim.config import for_method
from mist_sim.storage import write_json
from scripts.training import baseline_queue as queue


def prepare(tmp_path, config):
    jobs = []
    for method in ("fixed_split", "fixed_window"):
        settings = for_method(copy.deepcopy(config), method)
        settings["runtime"].update(collection_workers=4, windows_performance=True)
        settings["training"]["cycles"] = 300
        path = tmp_path / f"{method}.json"
        write_json(path, settings)
        jobs.append(dict(method=method, run_id=method, config=str(path), config_sha256=queue.sha256(path), seed=31))
    spec = tmp_path / "spec.json"
    write_json(spec, dict(active=["single", "upper"], pending=jobs))
    for run, pid in [("single", 1), ("upper", 2)]:
        folder = tmp_path / "logs/runs" / run
        write_json(folder / "process_policy.json", dict(pid=pid))
        write_json(folder / "manifest.json", dict(status="completed", exit_code=0))
        write_json(folder / "cycle_history.json", [dict(stage=f"cycle_{i:03d}") for i in range(300)])
    return spec


def test_completion_refills_only_two_slots_and_preserves_order(tmp_path, config, monkeypatch):
    spec = prepare(tmp_path, config)
    alive = {1: False, 2: True}
    launched, awake = [], []
    monkeypatch.setattr(queue, "source_hash", lambda root: "frozen")
    monkeypatch.setattr(queue, "process_alive", lambda pid: alive[pid])
    monkeypatch.setattr(queue, "keep_awake", awake.append)
    monkeypatch.setattr(queue, "terminal_command", lambda root, python, config, run, *args: [run])

    class Console:
        def __init__(self, command, **kwargs):
            self.pid = 3 + len(launched)
            assert sum(alive.values()) < 2
            alive[self.pid] = True
            run = command[0]
            launched.append(run)
            folder = tmp_path / "logs/runs" / run
            write_json(folder / "process_policy.json", dict(pid=self.pid))
            write_json(folder / "manifest.json", dict(status="completed", exit_code=0))
            write_json(folder / "cycle_history.json", [dict(stage=f"cycle_{i:03d}") for i in range(300)])

        def poll(self):
            return None

    ticks = []

    def tick(seconds):
        ticks.append(seconds)
        # First initial job is already done. The other completes while the
        # first queued baseline runs, then both queued jobs complete.
        for pid in ([2] if len(ticks) == 1 else [3, 4]):
            alive[pid] = False

    monkeypatch.setattr(queue.subprocess, "Popen", Console)
    monkeypatch.setattr(queue.time, "sleep", tick)
    queue.execute(tmp_path, spec, "queue")
    assert launched == ["fixed_split", "fixed_window"]
    assert awake == [True, False]
    state = queue.read_json(tmp_path / "logs/queues/queue/status.json")
    assert state["status"] == "completed"
    assert state["active"] == state["pending"] == []


def test_failed_dependency_never_launches_pending_job(tmp_path, config, monkeypatch):
    spec = prepare(tmp_path, config)
    write_json(tmp_path / "logs/runs/single/manifest.json", dict(status="failed", exit_code=1))
    monkeypatch.setattr(queue, "source_hash", lambda root: "frozen")
    monkeypatch.setattr(queue, "process_alive", lambda pid: False)
    monkeypatch.setattr(queue, "keep_awake", lambda enable: None)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch after failure"))
    with pytest.raises(RuntimeError, match="without successful"):
        queue.execute(tmp_path, spec, "failed-queue")
    assert queue.read_json(tmp_path / "logs/queues/failed-queue/status.json")["status"] == "failed"
