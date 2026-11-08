"""Rebuild numerical profile components from immutable measurement records."""
import math
from pathlib import Path

import numpy as np

from .profiles import Profiles


UNITS = {"work": "reference_work", "rate": "reference_work/s", "tensor": "bit", "memory": "byte",
         "energy": "J", "time": "s", "bandwidth": "Hz", "power": "W"}


def recompute_timing(record):
    calibration = np.asarray(record["raw_seconds"]["calibration"], dtype=np.float64)
    validation = np.asarray(record["raw_seconds"]["validation"], dtype=np.float64)
    for label, values in (("calibration_seconds", calibration), ("validation_seconds", validation)):
        if len(values) == 0 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Invalid raw timings")
        expected = {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "p95": float(np.quantile(values, .95)), "min": float(values.min()), "max": float(values.max())}
        for key, value in expected.items():
            if not math.isclose(value, record[label][key], rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(f"Timing summary mismatch: {label}/{key}")
    bound = float(calibration.max()) * (1 + record["margin"])
    if not math.isclose(bound, record["empirical_upper_seconds"], rel_tol=1e-12):
        raise AssertionError("Empirical bound changed")
    if not math.isclose(float(np.mean(validation <= bound)), record["held_out_coverage"], abs_tol=1e-12):
        raise AssertionError("Held-out coverage mismatch")


def rate_check(stages):
    calibration_rates = [record["operations"]["flops"] / 1e9 / record["calibration_seconds"]["median"] for record in stages]
    fitted_rate = float(np.median(calibration_rates))
    errors = []
    for record in stages:
        predicted = record["operations"]["flops"] / 1e9 / fitted_rate
        errors.extend(abs(predicted - actual) / actual for actual in record["raw_seconds"]["validation"])
    return {"fitted_counted_gflop_per_second": fitted_rate, "fit": "median calibration stage throughput",
            "held_out_median_relative_error": float(np.median(errors)),
            "held_out_max_relative_error": max(errors), "tolerance": .2,
            "within_20_percent_for_all_heldout_samples": max(errors) <= .2,
            "scope": "one constant effective rate for all measured stages; no concurrency/fractional-rate validation"}


def export_profiles(records, run_id):
    profiles, rows, rate_checks = {}, [], {}
    total_equivalence, cross_device_equivalence = 0, 0
    for measured in records:
        if measured["status"] != "completed" or not measured["no_gradients"]:
            raise ValueError("Incomplete profiling result")
        if measured["initial_state_sha256"] != measured["final_state_sha256"]:
            raise AssertionError("Model changed during profiling")
        if not measured["cpu_prefix_cuda_suffix"] or not all(item["allclose"] and item["top1_equal"] for item in measured["cpu_prefix_cuda_suffix"]):
            raise AssertionError("Cross-device split verification incomplete")
        cross_device_equivalence += len(measured["cpu_prefix_cuda_suffix"])
        name = measured["model"]
        if set(measured["devices"]) != {"cpu", "cuda"}:
            raise ValueError("Both CPU and CUDA needed for current export")
        gpu = measured["devices"]["cuda"]
        row = {"full_work": gpu["full"]["operations"]["flops"] / 1e9, "splits": {}}
        splits = []
        for device, result in measured["devices"].items():
            all_stages = [result["full"]]
            recompute_timing(result["full"])
            for split in result["splits"]:
                for stage in ("prefix", "suffix"):
                    recompute_timing(split[stage])
                    all_stages.append(split[stage])
                if split["prefix"]["operations"]["flops"] + split["suffix"]["operations"]["flops"] != result["full"]["operations"]["flops"]:
                    raise AssertionError("Arithmetic operation partition mismatch")
                for tensor in split["crossing_tensors"]:
                    elements = math.prod(tensor["shape"])
                    if tensor["elements"] != elements or tensor["bytes"] != elements * 4 or tensor["bits"] != elements * 32:
                        raise AssertionError("Tensor byte/bit mismatch")
                if not all(item["allclose"] and item["top1_equal"] for item in split["equivalence"]):
                    raise AssertionError("Output equivalence failed")
                total_equivalence += len(split["equivalence"])
                rows.append({"model": name, "device": device, "boundary": split["boundary"],
                             "prefix_counted_gflop": split["prefix"]["operations"]["flops"] / 1e9,
                             "suffix_counted_gflop": split["suffix"]["operations"]["flops"] / 1e9,
                             "transmission_bytes": sum(tensor["bytes"] for tensor in split["crossing_tensors"]),
                             "prefix_calibration_median_ms": split["prefix"]["calibration_seconds"]["median"] * 1000,
                             "suffix_calibration_median_ms": split["suffix"]["calibration_seconds"]["median"] * 1000,
                             "prefix_validation_p95_ms": split["prefix"]["validation_seconds"]["p95"] * 1000,
                             "suffix_validation_p95_ms": split["suffix"]["validation_seconds"]["p95"] * 1000,
                             "prefix_empirical_coverage": split["prefix"]["held_out_coverage"],
                             "suffix_empirical_coverage": split["suffix"]["held_out_coverage"],
                             "suffix_incremental_peak_bytes": split["suffix"]["memory"].get("incremental_peak_bytes"),
                             "suffix_resident_weight_buffer_bytes": split["suffix_resident_bytes"]})
            rate_checks[name + "/" + device] = rate_check(all_stages)
        cpu_splits = {item["boundary"]: item for item in measured["devices"]["cpu"]["splits"]}
        for split in gpu["splits"]:
            boundary = split["boundary"]
            if name.startswith("resnet"):
                split_id, blocks = int(boundary.removeprefix("layer")) + 1, 6
            else:
                split_id, blocks = int(boundary.rsplit("_", 1)[-1]) + 2, 14
            for stage in ("prefix", "suffix"):
                if split[stage]["operations"]["flops"] != cpu_splits[boundary][stage]["operations"]["flops"]:
                    raise AssertionError("Operation counts differ between devices")
            splits.append(split_id)
            input_bytes = sum(tensor["bytes"] for tensor in split["crossing_tensors"])
            row["splits"][str(split_id)] = {"local_work": split["prefix"]["operations"]["flops"] / 1e9,
                                          "edge_work": split["suffix"]["operations"]["flops"] / 1e9,
                                          "crossing_tensors_bits": [tensor["bits"] for tensor in split["crossing_tensors"]],
                                          "memory_bytes": input_bytes + split["suffix"]["memory"]["incremental_peak_bytes"]}
        profiles[name] = {"source": f"Measured offline run {run_id}; official {measured['weights']['weight_enum']}; counted GFLOP convention; memory excludes preloaded weights",
                          "blocks": blocks, "legal_splits": splits, "sizes": {"224": row}}
    components = {"schema": "mist_offline_profile_components_1", "units": UNITS, "profiles": profiles,
                  "reference_work_definition": "1 reference_work = 1e9 PyTorch registered-operator FLOPs; not a hardware throughput guarantee",
                  "memory_definition": "measured suffix input bytes plus incremental PyTorch CUDA allocator peak; preloaded model weights and driver memory excluded; serial batch1 only",
                  "training_ready": False, "source_run": run_id,
                  "missing_calibrations": ["effective resource rates under sharing/concurrency", "dynamic energy coefficients", "server idle/sleep/wake energy", "scenario/service/channel inputs"],
                  "constant_rate_diagnostics": rate_checks}
    Profiles(components)
    return components, rows, {"output_equivalence_checks": total_equivalence, "cross_device_equivalence_checks": cross_device_equivalence, "summary_rows": len(rows),
                               "raw_arithmetic_recomputed": True, "profiles_interface_validated": True}
