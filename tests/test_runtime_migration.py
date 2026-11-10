"""Runtime migration must not edit learning, RNG or curve history."""
import copy
import pytest
import torch
from mist_sim.config import for_method
from mist_sim.training import signature
from scripts.maintenance.optimize_training_runtime import migrate_payload, OLD_IMPLEMENTATION


@pytest.mark.parametrize("method", ["single_timescale", "upper_ppo_greedy"])
def test_runtime_migration_preserves_every_learning_state(config, method):
    config = for_method(config, method)
    config["runtime"]["collection_workers"] = 2
    config["diagnostics"]["phase_interval"] = 10
    config["training"]["cycles"] = config["convergence"]["min_cycles"] = 300
    original = copy.deepcopy(config)
    payload = dict(schema=2, resumable=True, implementation_signature=OLD_IMPLEMENTATION,
                   config=config, signature=signature(config), controller={"weight": torch.ones(2)},
                   optimizers={"state": torch.arange(3)}, history=[{"reward": -123.0}],
                   rng={"torch": torch.get_rng_state()}, total_slots=8192, trace_counter=64)
    result = migrate_payload(payload)
    for key in payload.keys() - {"config", "signature", "implementation_signature"}:
        assert result[key] is payload[key]
    assert payload["config"] == original
    assert result["config"]["runtime"]["collection_workers"] == 4
    assert result["config"]["diagnostics"]["phase_interval"] == 0
    expected = copy.deepcopy(original)
    expected["runtime"]["collection_workers"] = 4
    expected["diagnostics"]["phase_interval"] = 0
    assert result["config"] == expected
    payload["resumable"] = False
    with pytest.raises(ValueError, match="Completed-stage"):
        migrate_payload(payload)
    payload["resumable"] = True
    payload["config"]["ppo"]["gamma"] = .9
    with pytest.raises(ValueError, match="configuration hash"):
        migrate_payload(payload)


def test_derived_curve_excludes_superseded_tail(tmp_path):
    import csv
    from mist_sim.storage import digest_file, write_json
    from scripts.maintenance.optimize_training_runtime import combine_curve
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    (recovery / "resume.pt").write_bytes(b"checkpoint")
    original = tmp_path / "original.csv"
    fields = ["cycle_index", "reward_per_slot"]
    def save(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    prefix = [dict(cycle_index=i, reward_per_slot=-i) for i in range(1, 41)]
    save(original, prefix + [dict(cycle_index=41, reward_per_slot=-999)])
    save(recovery / "cycles_prefix.csv", prefix)
    run = "continued"
    suffix = tmp_path / "results" / run / "cycles.csv"
    save(suffix, [dict(cycle_index=i, reward_per_slot=-i) for i in range(41, 301)])
    write_json(recovery / "provenance.json", dict(checkpoint_sha256=digest_file(recovery / "resume.pt"),
        prefix_csv_sha256=digest_file(recovery / "cycles_prefix.csv"), original_csv=str(original),
        original_csv_sha256=digest_file(original), original_run="original", resumed_cycles=40,
        superseded_completed_cycles=1))
    write_json(tmp_path / "logs/runs" / run / "manifest.json", dict(status="completed", exit_code=0,
        checkpoint=dict(sha256=digest_file(recovery / "resume.pt"))))
    target = combine_curve(tmp_path, recovery, run)
    with (target / "cycles.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    assert rows[40] == {"cycle_index": "41", "reward_per_slot": "-41"}
    assert "-999" in original.read_text(encoding="utf-8")
