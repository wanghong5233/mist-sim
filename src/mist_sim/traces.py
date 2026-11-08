"""Lossless shared-frame representation for behavior traces."""
import json
from pathlib import Path


def pack_behavior(samples):
    frames, groups, frame_ids, group_ids = [], [], {}, {}
    def group_id(rows):
        identity = id(rows)
        if identity not in group_ids:
            group_ids[identity] = len(groups)
            groups.append(rows)
        return group_ids[identity]
    def frame_id(frame):
        if frame is None:
            return None
        identity = id(frame)
        if identity not in frame_ids:
            frame_ids[identity] = len(frames)
            frames.append({key: group_id(rows) for key, rows in frame.items()})
        return frame_ids[identity]
    packed = []
    for sample in samples:
        result = dict(sample, frame=frame_id(sample["frame"]),
                      next_observation=frame_id(sample.get("next_observation")))
        result["steps"] = [dict(step, prefix_frame=frame_id(step["prefix_frame"]))
                           if "prefix_frame" in step else step for step in sample["steps"]]
        packed.append(result)
    return dict(schema="mist.behavior.shared.v1", groups=groups, frames=frames, samples=packed)


def read_behavior(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, list):
        return value
    if value.get("schema") != "mist.behavior.shared.v1":
        raise ValueError("Unknown behavior trace schema")
    frames = [{key: value["groups"][index] for key, index in frame.items()} for frame in value["frames"]]
    for sample in value["samples"]:
        sample["frame"] = frames[sample["frame"]]
        if sample["next_observation"] is not None:
            sample["next_observation"] = frames[sample["next_observation"]]
        for step in sample["steps"]:
            if "prefix_frame" in step:
                step["prefix_frame"] = frames[step["prefix_frame"]]
    return value["samples"]
