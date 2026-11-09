"""Persistent episode workers; a rollout uses one frozen pair of policies."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import multiprocessing
import random

import numpy as np
import torch


def initialize_collector(config, seed, stop):
    global collector
    from .training import Trainer
    worker_config = copy.deepcopy(config)
    worker_config["runtime"]["torch_threads"] = 1
    worker_config["device"] = "cpu"
    collector = Trainer(worker_config, seed)
    collector.collection_stop = stop


def collect_episode(state, normalization, versions, trace_index, layer, count, pretrain, deadline):
    collector.controller.load_state_dict({layer_name: {name: torch.from_numpy(value) for name, value in tensors.items()}
                                          for layer_name, tensors in state.items()})
    collector.reward_normalizer.load_state_dict(normalization)
    collector.versions = versions
    collector.trace_counter = trace_index
    collector.total_slots = 0
    collector.training_episodes = 0
    collector.deadline = deadline
    action_seed = (collector.seed + 10000 * trace_index + 7919) % (2 ** 32)
    torch.manual_seed(action_seed)
    random.seed(action_seed)
    np.random.seed(action_seed)
    buffer, records = collector._collect_serial(layer, count, pretrain)
    return buffer.samples, records


class EpisodeCollectors:
    def __init__(self, config, seed, workers):
        context = multiprocessing.get_context("spawn")
        self.stop = context.Event()
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=initialize_collector, initargs=(config, seed, self.stop))

    def collect(self, trainer, jobs, layer, pretrain):
        state = {layer_name: {name: tensor.detach().cpu().numpy().copy() for name, tensor in tensors.items()}
                 for layer_name, tensors in trainer.controller.state_dict().items()}
        normalization = trainer.reward_normalizer.state_dict()
        futures = {self.pool.submit(collect_episode, state, normalization, dict(trainer.versions),
                                    trace_index, layer, count, pretrain, trainer.deadline): index
                   for index, (trace_index, count) in enumerate(jobs)}
        pending, outputs = set(futures), {}
        samples_done, slots_done = 0, 0
        try:
            while pending:
                trainer.check_budget()
                finished, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in finished:
                    samples, records = future.result()
                    outputs[futures[future]] = samples, records
                    samples_done += len(records) if layer == "joint" else len(samples)
                    slots_done += len(records)
                    trainer.event("collect", layer=layer, samples=samples_done,
                                  target_samples=sum(count for _, count in jobs), collected_slots=slots_done)
        except BaseException:
            self.stop.set()
            for future in futures:
                future.cancel()
            raise
        return [outputs[index] for index in range(len(jobs))]

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
