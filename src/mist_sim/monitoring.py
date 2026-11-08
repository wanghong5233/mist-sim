"""Human-readable progress and local TensorBoard scalars with explicit axes."""
import math
import sys
import time


class ConsoleProgress:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.bar = None
        self.label = ""
        self.cycle = None
        self.started = None
        self.losses = {}

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def write(self, message):
        self.close()
        print(message, file=self.stream, flush=True)

    def progress(self, label, current, total):
        if not self.stream.isatty():
            return
        from tqdm import tqdm
        if self.bar is None or self.bar.desc != label:
            self.close()
            self.bar = tqdm(total=total, desc=label, file=self.stream, leave=False,
                            dynamic_ncols=True, mininterval=0.2)
        self.bar.update(current - self.bar.n)

    def record(self, event, fields):
        stage = fields.get("stage", "")
        if event == "phase_started":
            if stage.startswith("cycle_"):
                cycle = int(stage.split("_")[1]) + 1
                if cycle != self.cycle:
                    self.cycle, self.started, self.losses = cycle, time.perf_counter(), {}
                    self.write(f"\nIteration {cycle}")
                step, network = (1, "服务器开关网络") if stage.endswith("upper") else (2, "任务调度网络")
                self.label = f"Iteration {cycle} [{step}/3] {network}"
            else:
                self.label = (f"预训练第 {int(stage.split('_')[1]) + 1} 次更新"
                              if stage.startswith("pretrain_") else "预训练最后一次更新")
            self.write(self.label + "：采样 → PPO 更新")
        elif event == "collect":
            self.progress(self.label + " / 采样", fields["samples"], fields["target_samples"])
        elif event == "rollout_ready":
            self.close()
        elif event == "ppo_epoch":
            self.progress(self.label + " / PPO epoch", fields["epoch"], fields["epoch_limit"])
        elif event == "phase_completed":
            self.losses[fields["layer"]] = fields.get("loss")
            loss = fields.get("loss")
            loss_text = f"{loss:.5g}" if loss is not None else "未记录"
            activation = fields.get("activation_probability")
            probability = (f" | 采样时开启概率均值={activation['mean']:.5f}"
                           f" 范围=[{activation['min']:.5f},{activation['max']:.5f}]" if activation else "")
            self.write(f"{self.label} 完成 | 平均loss={loss_text} | KL={fields['joint_kl']:.4g}"
                       f" | 采样={fields['collection_seconds']:.2f}s | PPO={fields['update_seconds']:.2f}s" + probability)
        elif event == "validation_started":
            label = f"Iteration {int(stage.split('_')[1]) + 1} [3/3]" if stage.startswith("cycle_") else stage
            self.label = label + " / 联合评价"
            self.write(f"{label} 联合策略评价中（不更新参数）")
        elif event == "validation_progress":
            self.progress(self.label + (" / 概率采样" if fields.get("mode") == "sampled" else " / 确定性"),
                          fields["completed"], fields["total"])
        elif event == "validation_completed":
            label = f"Iteration {int(stage.split('_')[1]) + 1} 完成" if stage.startswith("cycle_") else stage + " 评价完成"
            service = fields["service"]
            service_text = "N/A" if service is None else f"{service:.2%}"
            losses = " | ".join(f"{name}平均loss={self.losses[layer]:.5g}" for layer, name in
                                (("upper", "服务器"), ("lower", "调度")) if self.losses.get(layer) is not None)
            timing = f" | 本轮耗时={time.perf_counter() - self.started:.2f}s" if stage.startswith("cycle_") and self.started is not None else ""
            sampled = (f" | 采样reward/episode={fields['sampled_return_per_episode']:.6g}"
                       f" | 采样成本/时隙={fields['sampled_cost_per_slot']:.6g}"
                       if "sampled_return_per_episode" in fields else "")
            self.write(f"{label} | reward/episode={fields['return_per_episode']:.6g}"
                       f" | 成本/时隙={fields['cost_per_slot']:.6g} | 按时率={service_text}"
                       + sampled + (" | " + losses if losses else "") + timing)
        elif event in ("training_started", "calibration_completed", "training_stopped", "training_completed"):
            self.write(format_event(event, fields))


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
            if fields.get("activation_probability"):
                self.scalars("train/upper/activation_probability", fields["activation_probability"], fields["total_slots"])
        elif event == "validation_completed":
            names = ("cost_per_slot", "return_per_episode", "service", "energy_per_slot", "validation_seconds")
            self.scalars("validation", {name: fields[name] for name in names}, fields["total_slots"])
            self.scalars("validation_sampled", {name: value for name, value in fields.items()
                                                if name.startswith("sampled_")}, fields["total_slots"])
            if fields.get("stage", "").startswith("cycle_") or fields.get("stage") == "pretrain":
                self.scalars("iteration/validation", {name: fields[name] for name in names}, fields["cycle_index"])
                self.scalars("iteration/validation_sampled", {name: value for name, value in fields.items()
                                                              if name.startswith("sampled_")}, fields["cycle_index"])

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
