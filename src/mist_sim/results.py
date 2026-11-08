"""Unsmoothened numerical results, separate from diagnostic logs."""
import csv
from pathlib import Path


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != list(row):
                raise ValueError(f"CSV schema changed: {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def iteration_row(metrics):
    names = ("iteration", "stage", "layer", "total_slots", "samples", "physical_slots",
             "raw_reward_sum", "raw_reward_per_slot", "training_reward_mean", "epochs_completed",
             "optimizer_steps", "loss", "actor_loss", "value_loss", "entropy", "post_update_joint_kl",
             "clip_fraction", "explained_variance", "arrivals", "completed", "suppressed", "rejected",
             "timeout", "on_time_rate", "total_energy", "conservative_cost", "collection_seconds",
             "update_seconds", "phase_seconds")
    row = {name: metrics[name] for name in names}
    row.update({"energy_" + name: value for name, value in metrics["energy"].items()})
    return row


def evaluation_row(metrics):
    row = {name: value for name, value in metrics.items() if name != "energy"}
    row.update({"energy_" + name: value for name, value in metrics["energy"].items()})
    return row
