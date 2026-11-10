import copy
import csv
import multiprocessing
import sys

import pytest
import torch

from mist_sim.config import for_method
from mist_sim.devices import cuda_update_slot
from mist_sim.storage import digest_file, write_json
from mist_sim.training import signature
from scripts.maintenance.recover_watchdog import OLD_IMPLEMENTATION, curve_prefix, migrate_payload


def _enter_slot(ready, entered):
    ready.set()
    with cuda_update_slot(torch.device("cuda")):
        entered.set()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel mutex")
def test_separate_process_waits_and_exception_releases_gpu_slot():
    context = multiprocessing.get_context("spawn")
    ready, entered = context.Event(), context.Event()
    child = context.Process(target=_enter_slot, args=(ready, entered))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with cuda_update_slot(torch.device("cuda")):
                child.start()
                assert ready.wait(20)
                assert not entered.wait(.25)
                raise RuntimeError("injected")
        assert entered.wait(20)
        child.join(20)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)


@pytest.mark.parametrize("method", ["single_timescale", "upper_ppo_greedy"])
def test_migration_keeps_background_performance_and_learning_state(config, method):
    config = for_method(config, method)
    config["runtime"].update(collection_workers=4, windows_performance=True)
    config["diagnostics"]["phase_interval"] = 0
    config["training"]["cycles"] = config["convergence"]["min_cycles"] = 300
    original = copy.deepcopy(config)
    payload = dict(schema=2, resumable=True, implementation_signature=OLD_IMPLEMENTATION,
                   config=config, signature=signature(config), controller={"weight": torch.ones(2)},
                   optimizers={"state": torch.arange(3)}, rng={"torch": torch.get_rng_state()},
                   history=[{"reward": -1}], total_slots=8192)
    result = migrate_payload(payload)
    for key in payload.keys() - {"config", "signature", "implementation_signature"}:
        assert result[key] is payload[key]
    assert result["config"] == original
    assert result["config"]["runtime"]["windows_performance"] is True
    assert payload["config"] == original
    payload["resumable"] = False
    with pytest.raises(ValueError, match="Completed-stage"):
        migrate_payload(payload)


def test_repeated_recovery_excludes_both_superseded_tails(tmp_path):
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    fields = ["cycle_index", "reward_per_slot"]

    def save(path, indices):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(cycle_index=i, reward_per_slot=-i) for i in indices)

    original = tmp_path / "original.csv"
    save(original, range(1, 47))
    save(earlier / "cycles_prefix.csv", range(1, 41))
    (earlier / "resume.pt").write_bytes(b"checkpoint")
    proof = dict(checkpoint_sha256=digest_file(earlier / "resume.pt"),
                 prefix_csv_sha256=digest_file(earlier / "cycles_prefix.csv"),
                 original_csv=str(original), original_csv_sha256=digest_file(original), resumed_cycles=40)
    write_json(earlier / "provenance.json", proof)
    write_json(tmp_path / "logs/runs/crashed/manifest.json",
               dict(checkpoint=dict(sha256=proof["checkpoint_sha256"])))
    save(tmp_path / "results/crashed/cycles.csv", range(41, 56))
    schema, rows, superseded = curve_prefix(tmp_path, "crashed", earlier, 50)
    assert schema == fields
    assert [int(row["cycle_index"]) for row in rows] == list(range(1, 51))
    assert superseded == 5
    assert len(original.read_text().splitlines()) == 47
    save(earlier / "cycles_prefix.csv", range(1, 40))
    with pytest.raises(ValueError, match="source changed"):
        curve_prefix(tmp_path, "crashed", earlier, 50)
