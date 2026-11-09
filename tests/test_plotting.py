import json

import numpy as np
import pytest

from mist_sim.plotting import aggregate, group_data, moving_average, read_cycles, plot_curves


def test_trailing_average_keeps_real_initial_point_and_partial_window():
    np.testing.assert_allclose(moving_average([0, 3, 6, 9], 3), [0, 1.5, 3, 6])
    with pytest.raises(ValueError, match="positive integer"):
        moving_average([1], 0)


def test_seed_band_is_seed_variation_and_does_not_extend_short_seed():
    curves = [(np.array([1, 2, 3]), np.array([0, 10, 20])),
              (np.array([1, 2]), np.array([2, 12]))]
    x, mean, low, high = aggregate(curves, 2, "std")
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_allclose(mean, [1, 6])
    np.testing.assert_allclose(high - mean, np.sqrt(2))
    np.testing.assert_allclose(mean - low, np.sqrt(2))


@pytest.mark.parametrize("band", ["std", "ci95"])
def test_one_seed_never_gets_uncertainty_band(band):
    _, _, low, high = aggregate([(np.array([1, 2]), np.array([-3, -2]))], 1, band)
    assert low is None and high is None


def test_seed_bootstrap_is_repeatable_and_uses_seed_units():
    curves = [(np.array([1, 2]), np.array([v, v * 2])) for v in [1, 3, 9]]
    first = aggregate(curves, 1, "ci95", 1000, 17)
    second = aggregate(curves, 1, "ci95", 1000, 17)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(first[2][1], 2 * first[2][0])
    np.testing.assert_allclose(first[3][1], 2 * first[3][0])


def test_repeated_seed_is_not_independent_replication():
    data = [dict(label="PPO", run_id=run, seed=3, cycle_index=1, metric="reward_per_slot", value=-1)
            for run in ("first", "second")]
    with pytest.raises(ValueError, match="Duplicate training seed"):
        group_data(data, "reward_per_slot", "cycle_index")


def test_live_csv_snapshot_ignores_only_incomplete_last_line(tmp_path):
    path = tmp_path / "cycles.csv"
    path.write_bytes(b"run_id,seed,cycle_index,total_slots,reward_per_slot\nrun,3,1,10,-2\nrun,3,2,20,")
    rows, source = read_cycles(path, "PPO", ["reward_per_slot"])
    assert len(rows) == 1 and rows[0]["value"] == -2
    assert source["completed_iterations"] == 1
    assert source["ignored_incomplete_bytes"] > 0
    path.write_bytes(path.read_bytes() + b"NaN\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        read_cycles(path, "PPO", ["reward_per_slot"])


def test_exported_snapshot_replots_after_source_changes_and_rejects_tampering(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scienceplots")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/plotting/convergence.json").read_text(encoding="utf-8"))
    path = tmp_path / "cycles.csv"
    path.write_text("run_id,seed,cycle_index,total_slots,reward_per_slot\nrun,3,1,10,-2\nrun,3,2,20,-1\n")
    settings.update(curves=[dict(label="PPO", runs=[str(path)])], output=str(tmp_path / "figures"),
                    metrics=["reward_per_slot"], languages=["en"], formats=["png"])
    first = plot_curves(settings, tmp_path)
    assert first["summary"][0]["common_points"] == 2
    output = Path(first["output"])
    before = (output / "aggregate.csv").read_bytes()
    frozen = json.loads((output / "plot_config.json").read_text(encoding="utf-8"))
    path.write_text(path.read_text() + "run,3,3,30,1000\n")
    second = plot_curves(frozen, tmp_path)
    assert second["summary"][0]["common_points"] == 2
    assert (output / "aggregate.csv").read_bytes() == before
    assert (output / "reward_per_slot_en.png").is_file()
    # A copied project can replay its frozen data without the original CSV.
    import shutil
    relocated = tmp_path / "relocated"
    shutil.copytree(output, relocated / "figures")
    moved = plot_curves(frozen, relocated)
    assert moved["summary"][0]["common_points"] == 2
    assert (relocated / "figures" / "aggregate.csv").read_bytes() == before
    snapshot = output / "source_data.csv"
    snapshot.write_bytes(snapshot.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen plot data changed"):
        plot_curves(frozen, tmp_path)
