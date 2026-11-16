import pytest

from scripts.plotting.plot_frozen_load import pooled_interval


def test_interval_pools_counts_and_conditions_on_published_requests():
    rows = [
        {"seed": 1, "slots": 2, "metrics": {"arrivals": 10, "suppressed": 5,
            "completed": 4, "total_energy": 20, "conservative_cost": 40}},
        {"seed": 2, "slots": 2, "metrics": {"arrivals": 100, "suppressed": 20,
            "completed": 40, "total_energy": 60, "conservative_cost": 80}},
    ]
    expected = {"on_time_rate": 44 / 110, "on_time_rate_published": 44 / 85,
                "energy_per_slot": 20, "cost_per_slot": 30}
    for metric, estimate in expected.items():
        result = pooled_interval(rows, metric, 5000, 0)
        assert result[0] == pytest.approx(estimate)
        assert result[1] <= result[0] <= result[2]
        assert result == pooled_interval(list(reversed(rows)), metric, 5000, 0)
    rows[0]["metrics"]["suppressed"] = 10
    with pytest.raises(ValueError, match="empty trajectory"):
        pooled_interval(rows, "on_time_rate_published", 5000, 0)
