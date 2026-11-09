import copy
from unittest.mock import Mock

import pytest

from mist_sim.learning_analysis import advantage_quality, clock_baseline_probe, public_action_quality
from mist_sim.environment import Environment, PublicProblem
from mist_sim.anonymity import generate_trace


def test_nested_quality_reaches_tensorboard_without_nonfinite_values():
    from mist_sim.monitoring import TrainingMonitor
    monitor = TrainingMonitor(None, False)
    monitor.writer = Mock()
    monitor.record("rollout_ready", dict(total_slots=1024, rollout=dict(learning_signal=dict(lower=dict(
        advantage_std=.16, quality=dict(position_mean_variance_fraction=.89,
            terminal=dict(value_mean=-.46, reward_mean=-.025, missing=None, invalid=float("nan"))))))))
    calls = {call.args[0]: call.args[1:] for call in monitor.writer.add_scalar.call_args_list}
    assert calls == {"signal/lower/advantage_std": (.16, 1024),
        "signal/lower/quality/position_mean_variance_fraction": (.89, 1024),
        "signal/lower/quality/terminal/value_mean": (-.46, 1024),
        "signal/lower/quality/terminal/reward_mean": (-.025, 1024)}
    monitor.writer.flush.assert_called_once()


def clock_samples():
    rows = []
    for episode in range(2):
        for slot in range(2):
            rows.append(dict(phase="lower", episode_id=episode, slot=slot, value=-10., next_value=-10.,
                reward=-1., reward_scale=1., physical_rewards=[-1.], duration=1,
                terminated=slot == 1, truncated=False))
    return rows


def test_value_quality_exposes_terminal_and_nonlinear_time_bias():
    rows = [dict(row, slot=index % 3, episode_id=index // 3) for index, row in enumerate(clock_samples() * 2)]
    # Pearson correlation with position is zero, although all variation is by position.
    rows = rows[:6]
    quality = advantage_quality(rows, [1., -2., 1.] * 2, 1.)
    assert quality["position_mean_variance_fraction"] == pytest.approx(1.)
    assert quality["terminal"]["td_mean"] == pytest.approx(9.)
    assert quality["advantage_reward_correlation"] is None
    assert advantage_quality(rows[:1], [0.], 1.)["position_mean_variance_fraction"] is None


def test_clock_probe_is_independent_of_held_episode_rewards_and_preserves_inputs():
    rows = clock_samples()
    original = copy.deepcopy(rows)
    report = clock_baseline_probe(rows, 2, dict(gamma=1., gae_lambda=.95))
    assert report["applicable"]
    lower = report["phases"]["lower"]
    assert lower["original"]["terminal"]["value_mean"] == -10.
    assert lower["clock"]["terminal"]["value_mean"] == -1.
    assert lower["clock"]["advantage_std"] == 0.
    assert rows == original
    rows[0].update(reward=-3., physical_rewards=[-3.])
    changed = clock_baseline_probe(rows, 2, dict(gamma=1., gae_lambda=1.))
    # At the terminal slots the other-episode means are -1 and -2, respectively.
    assert changed["phases"]["lower"]["clock"]["terminal"]["value_mean"] == -1.5


def test_clock_probe_keeps_zero_duration_server_events():
    rows = []
    for row in clock_samples():
        rows.extend([dict(row, phase="upper", reward=0., physical_rewards=[], duration=0, terminated=False), row])
    report = clock_baseline_probe(rows, 2, dict(gamma=1., gae_lambda=.95))
    assert report["applicable"]
    assert set(report["phases"]) == {"upper", "lower"}
    for phase in report["phases"].values():
        assert phase["clock"]["advantage_std"] == 0.
    assert report["phases"]["upper"]["clock"]["reward_std"] == 0.


@pytest.mark.parametrize("change", ["truncated", "clipped", "missing_slot"])
def test_clock_probe_declines_inapplicable_inputs(change):
    rows = clock_samples()
    if change == "truncated":
        rows[-1]["truncated"] = True
    elif change == "clipped":
        rows[0]["reward"] = -.5
    else:
        rows[0]["slot"] = 1
    assert not clock_baseline_probe(rows, 2, dict(gamma=1., gae_lambda=.95))["applicable"]


def test_public_action_quality_recomputes_savings_without_trusting_task_costs(config):
    config = copy.deepcopy(config)
    config["local_rate"] = 100.
    config["local_kappa"] = 1e-8
    config["rejection_cost"] = 100.
    env = Environment(config, generate_trace(config, 91))
    while not env.done:
        p = env.begin_slot(config["initial_active"])
        env.execute([0] * len(p.choices))
    original = copy.deepcopy(env.records)
    report = public_action_quality(config, env.records)
    assert report["counts"]["dominated_reject_by_local"] > 0
    expected = 0.
    for row in env.records:
        problem = PublicProblem(config, row["public"], row["active"], row["ready"], row["slot"], [])
        for choices in problem.choices:
            local = next((choice for choice in choices if choice["kind"] == "local"), None)
            if local:
                expected += max(0., choices[0]["cost_fixed"] - local["cost_fixed"])
        for task in row["tasks"]:
            task["conservative_cost"] = -999.
    assert public_action_quality(config, env.records) == report
    assert report["guaranteed_local_saving_per_slot"]["reject"] == pytest.approx(expected / len(env.records))
    assert report["cost_per_slot"]["rejection"] == pytest.approx(sum(r["rejection_penalty"] for r in original) / len(original))
