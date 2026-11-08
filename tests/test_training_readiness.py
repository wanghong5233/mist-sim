import time
import pytest
from mist_sim.preflight import service_ceiling
from mist_sim.training import Trainer
from mist_sim.evaluation import replay
from mist_sim.config import validate


def test_ceiling_empty_and_impossible(config):
    config["arrival_probability"] = 0
    assert service_ceiling(config, [7])["optimistic_service_ceiling"] is None
    config["arrival_probability"] = 1
    config["deadline_range"] = [.00001, .00001]
    config["validation"]["min_service_rate"] = .8
    report = service_ceiling(config, [7])
    assert report["arrivals"] == config["horizon"] * config["terminals"]
    assert report["optimistic_service_ceiling"] == 0
    assert report["threshold_ruled_out"]


def test_multitrajectory_selection_and_replay(config):
    config["validation"]["trajectories"] = 2
    trainer = Trainer(config, 91)
    trainer.calibrate()
    records, metrics, rows = trainer.evaluate_validation()
    assert len(rows) == 2
    assert metrics["arrivals"] == sum(row["metrics"]["arrivals"] for row in rows)
    assert metrics["on_time_rate"] == pytest.approx(sum(row["metrics"]["completed"] for row in rows) / metrics["arrivals"])
    assert replay(config, records) == metrics
    trainer.validate_cycle("initial")
    assert trainer.validation_history[0]["total_slots"] == 0


def test_validation_substream_collision(config):
    config["validation"]["trajectories"] = 2
    config["seeds"]["test"] = config["seeds"]["validation"] + 10000
    with pytest.raises(ValueError, match="substream"):
        validate(config)
    config["seeds"]["test"] += 1
    trainer = Trainer(config, config["seeds"]["validation"] + 10000)
    with pytest.raises(ValueError, match="substream"):
        trainer.fresh_environment()


def test_budget_stops_before_calibration(config):
    trainer = Trainer(config, 92)
    trainer.deadline = time.perf_counter() - 1
    with pytest.raises(TimeoutError):
        trainer.calibrate()
    assert trainer.total_slots == 0
    assert not trainer.completed_stages
