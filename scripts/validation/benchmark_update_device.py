"""Isolated CPU/CUDA update probe on a private copy of a saved policy."""
import argparse
import copy
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mist_sim.policies.batching import replay_batch, RolloutTensors
from mist_sim.ppo import loss_terms, clip_policy_gradients
from mist_sim.training import Trainer
from mist_sim.ppo import update


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(base, samples, advantage, target, settings, device, repeats):
    started = time.perf_counter()
    policy = copy.deepcopy(base).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings["learning_rate"], eps=settings["adam_epsilon"])
    synchronize(device)
    model_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cache = RolloutTensors(policy, samples)
    old = torch.tensor([sample["logprob"] for sample in samples], device=device)
    advantages = torch.tensor(advantage, dtype=torch.float32, device=device)
    targets = torch.tensor(target, dtype=torch.float32, device=device)
    valid = torch.tensor([bool(sample["steps"]) for sample in samples], device=device)
    synchronize(device)
    input_seconds = time.perf_counter() - started

    def forward():
        logprob, entropy, values = replay_batch(policy, samples, cache=cache)
        loss = loss_terms(logprob, old, advantages, values, targets, entropy, valid, settings)[0]
        return loss, (logprob, entropy, values)

    optimizer.zero_grad(set_to_none=True)
    initial_loss, initial_outputs = forward()
    initial_loss.backward()
    initial = dict(loss=float(initial_loss.detach()), outputs=[value.detach().cpu().numpy() for value in initial_outputs],
                   gradients={name: parameter.grad.detach().cpu().numpy().copy()
                              for name, parameter in policy.named_parameters() if parameter.grad is not None})
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(repeats + 1):
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward()
        loss.backward()
        clip_policy_gradients(policy, settings["max_grad_norm"])
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        if repeat:
            timings.append(elapsed)
    synchronize(device)
    started = time.perf_counter()
    cpu_weights = {name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items()}
    synchronize(device)
    return_seconds = time.perf_counter() - started
    report = dict(device=str(device), model_setup_seconds=model_seconds, input_pack_transfer_seconds=input_seconds,
                  step_seconds=timings, median_step_seconds=statistics.median(timings),
                  return_weights_seconds=return_seconds,
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                  state_bytes=sum(value.numel() * value.element_size() for value in cpu_weights.values()))
    del policy, optimizer, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, initial


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full-rollout", action="store_true", help="Compare full updates: CPU64 and CUDA64/128/256")
    parser.add_argument("--snapshot-only", action="store_true", help="Import a completed older snapshot for isolated profiling, never resume a run")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 or size > 64 for size in args.sizes):
        parser.error("Use 2..64 events and at least one repeat")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    payload = args.checkpoint.read_bytes()
    print(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, threads=1)), flush=True)
    del payload
    if args.snapshot_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not saved.get("resumable"):
            raise ValueError("Only completed snapshots may initialize this probe")
        trainer = Trainer(saved["config"], saved["seed"])
        trainer.controller.load_state_dict(saved["controller"])
        trainer.reward_normalizer.load_state_dict(saved["reward_normalizer"])
        trainer.value_reward_scales = saved["value_reward_scales"]
        trainer.trace_counter = saved["trace_counter"]
        for phase, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][phase])
        print(json.dumps(dict(snapshot_only=True, source_signature=saved["implementation_signature"],
                              completed_stages=saved["completed_stages"])), flush=True)
        del saved
    else:
        trainer = Trainer.load(args.checkpoint)
    if trainer.config["method"] != "joint_ppo":
        raise ValueError("This probe expects a joint_ppo checkpoint")
    if args.full_rollout:
        full_rollout(trainer)
        return
    trainer.config["runtime"]["collection_workers"] = 0
    started = time.perf_counter()
    buffer, _ = trainer.collect("joint", max(args.sizes))
    print(json.dumps(dict(probe_sampling_seconds=time.perf_counter() - started,
                          events=len(buffer.samples), writes_results=False)), flush=True)
    advantage, target = buffer.advantages(trainer.config["ppo"]["gamma"], trainer.config["ppo"]["gae_lambda"])
    advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
    for size in args.sizes:
        selected = buffer.samples[:size]
        reports, initials = [], []
        for device in (torch.device("cpu"), torch.device("cuda")):
            report, initial = measure(trainer.joint_policy, selected, advantage[:size], target[:size],
                                      trainer.config["ppo"], device, args.repeats)
            reports.append(report)
            initials.append(initial)
        cpu, gpu = initials
        for expected, actual in zip(cpu["outputs"], gpu["outputs"]):
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        assert cpu["gradients"].keys() == gpu["gradients"].keys()
        for name, expected in cpu["gradients"].items():
            np.testing.assert_allclose(gpu["gradients"][name], expected, atol=1e-5, rtol=2e-3)
        print(json.dumps(dict(events=size, phases={phase: sum(sample["phase"] == phase for sample in selected)
            for phase in ("upper", "lower")}, candidate_rows=sum(len(step["features"]) for sample in selected for step in sample["steps"]),
            measurements=reports, speedup=reports[0]["median_step_seconds"] / reports[1]["median_step_seconds"],
            initial_loss=cpu["loss"], loss_abs_error=abs(cpu["loss"] - gpu["loss"]),
            gradient_max_abs_error=max(float(np.max(np.abs(expected - gpu["gradients"][name])))
                for name, expected in cpu["gradients"].items()))), flush=True)
    trainer.close()


def full_rollout(trainer):
    started = time.perf_counter()
    buffer, records = trainer.collect("joint", trainer.config["training"]["fast_slots"])
    sampling_seconds = time.perf_counter() - started
    trainer.close()
    print(json.dumps(dict(physical_slots=len(records), events=len(buffer.samples), sampling_seconds=sampling_seconds)), flush=True)
    if not trainer.optimizers["joint"].state:
        for parameter, state in trainer.optimizers["lower"].state.items():
            trainer.optimizers["joint"].state[parameter] = copy.deepcopy(state)
    results = []
    for device, batch in (("cpu", 64), ("cuda", 64), ("cuda", 128), ("cuda", 256)):
        policy = copy.deepcopy(trainer.joint_policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=trainer.config["ppo"]["learning_rate"],
                                     eps=trainer.config["ppo"]["adam_epsilon"])
        optimizer.load_state_dict(copy.deepcopy(trainer.optimizers["joint"].state_dict()))
        replay_buffer = copy.copy(buffer)
        replay_buffer.samples = list(buffer.samples)
        settings = dict(trainer.config["ppo"], minibatch=batch, target_kl=1e6)
        torch.manual_seed(71)
        started = time.perf_counter()
        metrics = update(policy, optimizer, replay_buffer, settings, buffer.behavior_version, buffer.opponent_version,
                         device=device)
        elapsed = time.perf_counter() - started
        report = dict(device=device, minibatch=batch, physical_slots=len(records), events=len(buffer.samples),
            forced_full_epochs=True, epochs=metrics["epochs_completed"], optimizer_steps=metrics["optimizer_steps"],
            full_update_seconds=elapsed, sampling_seconds=sampling_seconds, combined_seconds=elapsed + sampling_seconds,
            input_pack_transfer_seconds=metrics["input_pack_transfer_seconds"],
            rollout_tensor_bytes=metrics["rollout_tensor_bytes"], rollout_tensor_transfers=metrics["rollout_tensor_transfers"],
            device_metrics=metrics["device_metrics"], loss=metrics["loss"], post_update_joint_kl=metrics["post_update_joint_kl"])
        print(json.dumps(report), flush=True)
        results.append(report)
        del policy, optimizer, replay_buffer
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(dict(update_speedups={str(row["minibatch"]): results[0]["full_update_seconds"] / row["full_update_seconds"]
                                          for row in results[1:]})), flush=True)


if __name__ == "__main__":
    main()
