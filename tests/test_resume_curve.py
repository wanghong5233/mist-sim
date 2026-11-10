"""A restarted checkpoint must not count its superseded CSV tail twice."""
import csv
import json
from pathlib import Path
import runpy

import pytest
import torch

from mist_sim.storage import digest_file, write_json
from mist_sim.training import implementation_signature


@pytest.mark.parametrize("new_cycles", [[141, 142], [140, 141], [142]])
def test_restart_curve_keeps_checkpoint_prefix_and_real_new_suffix(tmp_path, new_cycles):
    merge = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "scripts/maintenance/recover_always_on_roundoff.py"))["merge_continuation"]
    root, original, first, restarted = tmp_path, "original", "first", "restart"
    config = dict(method="always_on", checkpoint_policy=dict(cycle_interval=10))
    old_config = dict(method="always_on", checkpoint_policy=dict(cycle_interval=250))
    paths = {}
    for name, cycles in ((original, range(1, 140)), (first, range(140, 145)), (restarted, new_cycles)):
        path = root / "results" / name / "cycles.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run_id", "seed", "cycle_index", "total_slots", "reward_per_slot"])
            writer.writeheader()
            for cycle in cycles:
                writer.writerow(dict(run_id=name, seed=4200001, cycle_index=cycle,
                    total_slots=8192 + cycle * 2048 + (2048 if name != original else 0), reward_per_slot=-200.))
        paths[name] = path
        write_json(root / "logs/runs" / name / "resolved_config.json", old_config if name == original else config)
    before = {name: digest_file(path) for name, path in paths.items()}
    checkpoint = root / "logs/runs" / first / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    torch.save(dict(resumable=True, config=config, cycle_history=[{}] * 140,
                    total_slots=296960, implementation_signature=implementation_signature()), checkpoint)
    write_json(root / "logs/runs" / first / "manifest.json",
               dict(seed=4200001, status="stopped", checkpoint=dict(sha256="migration")))
    write_json(root / "logs/runs" / restarted / "manifest.json", dict(seed=4200001, status="running",
               checkpoint=dict(path=str(checkpoint), sha256=digest_file(checkpoint))))
    (root / "logs/runs" / first / "progress.jsonl").write_text(json.dumps(dict(total_slots=307200)) + "\n", encoding="utf-8")
    recovery = root / "logs/recovery"
    write_json(recovery / "provenance.json", dict(original_csv=str(paths[original]),
        original_csv_sha256=before[original], checkpoint_sha256="migration"))
    if new_cycles != [141, 142]:
        with pytest.raises(ValueError, match="gap or overlap"):
            merge(root, root / "logs/runs" / original, recovery, first, restarted)
    else:
        merge(root, root / "logs/runs" / original, recovery, first, restarted)
        destination = root / "results/comparisons/always-on-recovered-seed4200001"
        with (destination / "cycles.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["cycle_index"]) for row in rows] == list(range(1, 143))
        assert rows[139]["source_run_id"] == first
        assert rows[140]["source_run_id"] == restarted
        assert {row["run_id"] for row in rows} == {restarted}
        provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["segments"][1]["excluded_superseded_iterations"] == [141, 142, 143, 144]
        assert provenance["discarded_rollback_slots"] == 10240
        assert provenance["actual_training_slots_including_retries"] == int(rows[-1]["total_slots"]) + 10240
    assert before == {name: digest_file(path) for name, path in paths.items()}
