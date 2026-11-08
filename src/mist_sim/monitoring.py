"""Human-readable progress and local TensorBoard scalars with explicit axes."""
import math


def format_event(event, fields):
    prefix = f"[{fields.get('stage', event)}] slots={fields.get('total_slots', 0)}"
    if event == "ppo_epoch":
        return (f"{prefix} {fields['layer']} PPO epoch {fields['epoch']}/{fields['epoch_limit']} "
                f"batches={fields['minibatches']} loss={fields['loss']:.4g} "
                f"actor={fields['actor_loss']:.4g} value={fields['value_loss']:.4g} "
                f"entropy={fields['entropy']:.4g} KL={fields['post_update_joint_kl']:.4g} "
                f"clip={fields['clip_fraction']:.3f} time={fields['seconds']:.2f}s "
                f"stop={'KL' if fields['early_stop'] else '-'}")
    if event == "rollout_ready":
        rollout = fields["rollout"]
        return (f"{prefix} {fields['layer']} rollout transitions={rollout['transitions']} "
                f"physical_slots={rollout['physical_slots']} episodes={rollout['terminated']} "
                f"truncated={rollout['truncated']} batches/epoch={rollout['minibatches_per_epoch']}")
    if event == "validation_completed":
        return (f"{prefix} validation cost/slot={fields['cost_per_slot']:.4g} "
                f"return/episode={fields['return_per_episode']:.4g} service={fields['service']} "
                f"time={fields['validation_seconds']:.2f}s selected={fields['selected']}")
    if event == "phase_completed":
        return (f"{prefix} {fields['layer']} rollout complete reward/slot={fields['raw_reward_per_slot']:.4g} "
                f"service={fields['service']} sample={fields['collection_seconds']:.2f}s "
                f"update={fields['update_seconds']:.2f}s")
    if event == "collect":
        return f"{prefix} {fields['layer']} sampling {fields['samples']}/{fields['target_samples']}"
    if event == "training_stopped":
        return f"{prefix} stopped: {fields['reason']}"
    if event in ("phase_started", "validation_started", "training_started", "training_completed", "calibration_completed"):
        return f"{prefix} {event}"
    return None


class TrainingMonitor:
    def __init__(self, path, enabled):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError("TensorBoard requested but unavailable; install mist-sim[monitoring]") from error
            self.writer = SummaryWriter(log_dir=str(path), flush_secs=10)

    def scalars(self, prefix, values, step):
        if self.writer is None:
            return
        for name, value in values.items():
            if value is not None and math.isfinite(float(value)):
                self.writer.add_scalar(prefix + "/" + name, float(value), step)
        self.writer.flush()

    def record(self, event, fields):
        if event == "ppo_epoch":
            names = ("loss", "actor_loss", "value_loss", "entropy", "post_update_joint_kl", "clip_fraction",
                     "actor_before", "critic_before", "actor_after", "critic_after", "seconds", "total_slots")
            self.scalars("ppo/" + fields["layer"], {name: fields[name] for name in names}, fields["layer_epoch"])
        elif event == "episode_completed":
            self.scalars("train/" + fields["layer"] + "/episode",
                         {name: fields[name] for name in ("raw_return", "energy", "on_time_rate", "length")},
                         fields["total_slots"])
        elif event == "phase_completed":
            names = ("raw_reward_per_slot", "training_reward_mean", "service", "energy", "cost", "joint_kl",
                     "explained_variance", "reward_scale", "reward_clipped_slots", "collection_seconds", "update_seconds")
            self.scalars("train/" + fields["layer"], {name: fields[name] for name in names}, fields["total_slots"])
        elif event == "validation_completed":
            names = ("cost_per_slot", "return_per_episode", "service", "energy_per_slot", "validation_seconds")
            self.scalars("validation", {name: fields[name] for name in names}, fields["total_slots"])

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
