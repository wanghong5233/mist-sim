"""Publication aggregation checks: pairing and physical denominators."""
import copy

import numpy as np
import pytest

from scripts.plotting.plot_paper_results import interval, paired_cost_interval, paired_groups


def observation(seed, arrivals, completed, suppressed, rejected, timeout, published, widths):
    return {"seed": seed, "input_sha256": str(seed), "axis": "anonymity", "axis_value": 4,
            "method": "hpgi", "decode": "deterministic", "slots": 128,
            "metrics": {"arrivals": arrivals, "completed": completed, "suppressed": suppressed,
                        "rejected": rejected, "timeout": timeout},
            "anonymity": {"published": published, "normalized_position_width_sum": widths,
                          "normalized_deadline_width_sum": widths}}


def test_publication_ratios_pool_counts_and_keep_failure_denominators():
    rows = [observation(1, 10, 4, 2, 3, 1, 8, 2), observation(2, 30, 10, 10, 8, 2, 20, 10)]
    assert interval(rows, "on_time_rate")[0] == pytest.approx(14 / 40)
    assert interval(rows, "on_time_rate_published")[0] == pytest.approx(14 / 28)
    assert interval(rows, "timeout_accepted")[0] == pytest.approx(3 / 17)
    assert interval(rows, "suppression_rate")[0] == pytest.approx(12 / 40)
    assert interval(rows, "information_width")[0] == pytest.approx(12 / 28)
    np.testing.assert_equal(interval(rows, "information_width"), interval(list(reversed(rows)), "information_width"))


def test_twenty_paired_traces_reject_duplicates_and_mismatched_inputs():
    rows = [observation(seed, 10, 7, 1, 1, 1, 9, 3) for seed in range(20)]
    comparison = [{**copy.deepcopy(row), "method": "nominal"} for row in rows]
    methods = {"hpgi": "HPGI", "nominal": "Nominal"}
    assert len(paired_groups(rows + comparison, methods, [4], "anonymity")) == 2
    comparison[0]["input_sha256"] = "different-input"
    with pytest.raises(ValueError, match="paired"):
        paired_groups(rows + comparison, methods, [4], "anonymity")
    with pytest.raises(ValueError, match="paired"):
        paired_groups(rows + rows[:1], {"hpgi": "HPGI"}, [4], "anonymity")
    with pytest.raises(ValueError, match="distinct"):
        paired_groups(rows[:-1] + rows[:1], {"hpgi": "HPGI"}, [4], "anonymity")


def test_bandwidth_utilization_counts_reserved_capacity_not_link_rate():
    row = observation(1, 10, 8, 0, 2, 0, 10, 4)
    row.update(axis="bandwidth", axis_value=20, mechanism={"splits": {"resnet152:3": 128}})
    assert interval([row], "edge_per_slot")[0] == 1
    assert interval([row], "bandwidth_utilization")[0] == .25
    row["axis_value"] = 40
    assert interval([row], "bandwidth_utilization")[0] == .125


def test_paired_cost_keeps_sign_and_weights_slots():
    first = observation(1, 10, 8, 0, 2, 0, 10, 4)
    second = observation(2, 10, 8, 0, 2, 0, 10, 4)
    first["metrics"]["conservative_cost"] = 128
    second["metrics"]["conservative_cost"] = 256
    comparison = copy.deepcopy([first, second])
    comparison[0]["metrics"]["conservative_cost"] = 256
    comparison[1]["metrics"]["conservative_cost"] = 384
    assert paired_cost_interval([first, second], comparison) == (-1., -1., -1.)
    comparison[0]["input_sha256"] = "not-paired"
    with pytest.raises(ValueError, match="paired"):
        paired_cost_interval([first, second], comparison)
