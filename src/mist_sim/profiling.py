"""Offline profiling of official models; no optimizer or policy training."""
import hashlib
import copy
import inspect
import json
import math
import pickle
import platform
import shutil
import statistics
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import fx
from torch.utils.benchmark import Timer
from torch.utils.flop_counter import FlopCounterMode

from .storage import digest_file, write_json


def validate_spec(spec):
    if spec["schema"] != 1 or spec["batch_size"] != 1 or spec["dtype"] != "float32":
        raise ValueError("Only schema 1, batch 1 FP32 profiling supported")
    for key in ("cpu_threads", "warmup", "calibration_repeats", "validation_repeats", "per_model_timeout_seconds"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"Invalid positive count: {key}")
    if set(spec["models"]) - {"resnet50", "resnet152", "vit_b_32"}:
        raise ValueError("Unsupported model")
    if set(spec["devices"]) - {"cpu", "cuda"}:
        raise ValueError("Unsupported device")
    calibration = spec["dataset"]["calibration_indices"]
    validation = spec["dataset"]["validation_indices"]
    if not calibration or not validation or set(calibration) & set(validation):
        raise ValueError("Calibration and held-out inputs must be nonempty and disjoint")
    if any(type(index) is not int or not 0 <= index < 10000 for index in calibration + validation):
        raise ValueError("Invalid CIFAR-10 test index")
    if not math.isfinite(spec["empirical_bound_margin"]) or spec["empirical_bound_margin"] < 0:
        raise ValueError("Invalid empirical margin")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_verified(url, path, expected, algorithm="sha256", prefix=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid(candidate):
        actual = file_hash(candidate, algorithm)
        return actual.startswith(expected) if prefix else actual == expected

    if path.exists():
        if not valid(path):
            raise ValueError(f"Existing asset checksum mismatch: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    if shutil.which("curl.exe"):
        subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-", "--connect-timeout", "20",
                        "--max-time", "300", "--output", str(temporary), url], check=True, timeout=310)
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "MIST-offline-profiling/1.0"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(65536), b""):
                if time.perf_counter() - started > 300:
                    raise TimeoutError("Asset download exceeded 300 seconds")
                output.write(block)
    if not valid(temporary):
        raise ValueError(f"Downloaded asset checksum mismatch: {url}")
    temporary.replace(path)
    return path


def prepare_assets(root, spec):
    validate_spec(spec)
    assets = {}
    dataset = spec["dataset"]
    path = fetch_verified(dataset.get("download_url", dataset["url"]), root / "data/raw/cifar-10-python.tar.gz", dataset["md5"], "md5")
    assets["dataset"] = {"path": str(path), "url": dataset["url"], "sha256": file_hash(path),
                         "md5": dataset["md5"], "download_url": dataset.get("download_url", dataset["url"]),
                         "license": "Not specified on source page; no redistribution",
                         "source_page": "https://www.cs.toronto.edu/~kriz/cifar.html"}
    print("ASSET dataset verified", flush=True)
    for name in spec["models"]:
        weight = torchvision.models.get_model_weights(name)[spec["weights"][name]]
        filename = weight.url.rsplit("/", 1)[-1]
        prefix = filename.rsplit("-", 1)[-1].split(".")[0]
        path = fetch_verified(weight.url, root / "data/raw/weights" / filename, prefix, prefix=True)
        assets[name] = {"path": str(path), "url": weight.url, "sha256": file_hash(path),
                        "weight_enum": str(weight), "transform": str(weight.transforms()),
                        "source": "torchvision official ImageNet pretrained weights; upstream terms apply"}
        print(f"ASSET {name} verified", flush=True)
    return assets


def load_inputs(spec, assets, name):
    with tarfile.open(assets["dataset"]["path"], "r:gz") as archive:
        member = archive.extractfile(spec["dataset"]["member"])
        with member:
            payload = pickle.load(member, encoding="bytes")
    images = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    weight = torchvision.models.get_model_weights(name)[spec["weights"][name]]
    transform = weight.transforms()
    outputs, metadata = {}, []
    for group in ("calibration", "validation"):
        outputs[group] = []
        for index in spec["dataset"][group + "_indices"]:
            array = images[index].copy()
            tensor = transform(Image.fromarray(array)).unsqueeze(0).contiguous()
            outputs[group].append(tensor)
            metadata.append({"group": group, "index": index, "input_shape": list(tensor.shape),
                             "raw_image_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                             "tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
    return outputs, metadata


def split_graph(model, boundary):
    traced = fx.symbolic_trace(model.eval())
    prefix_name = "encoder.layers." + boundary if boundary.startswith("encoder_layer_") else boundary
    nodes = list(traced.graph.nodes)
    matches = [node for node in nodes if any(key == prefix_name or key.startswith(prefix_name + ".")
               for key in node.meta.get("nn_module_stack", {}))]
    if not matches:
        matches = [node for node in nodes if node.op == "call_module" and str(node.target) == boundary]
    if not matches:
        raise ValueError(f"No complete graph boundary: {boundary}")
    cut = nodes.index(matches[-1])
    before, after = nodes[:cut + 1], nodes[cut + 1:]
    after_set = set(after)
    crossing = [node for node in before if node.op != "get_attr" and any(user in after_set for user in node.users)]
    if not crossing:
        raise ValueError("Empty crossing interface")
    prefix_graph, prefix_map = fx.Graph(), {}
    for node in before:
        prefix_map[node] = prefix_graph.node_copy(node, lambda dependency: prefix_map[dependency])
    prefix_graph.output(tuple(prefix_map[node] for node in crossing))
    suffix_graph, suffix_map = fx.Graph(), {}
    for node in crossing:
        suffix_map[node] = suffix_graph.placeholder(node.name)
    for node in before:
        if node.op == "get_attr" and any(user in after_set for user in node.users):
            suffix_map[node] = suffix_graph.node_copy(node, lambda dependency: suffix_map[dependency])
    for node in after:
        suffix_map[node] = suffix_graph.node_copy(node, lambda dependency: suffix_map[dependency])
    prefix = fx.GraphModule(traced, prefix_graph).eval()
    suffix = fx.GraphModule(traced, suffix_graph).eval()
    prefix.graph.eliminate_dead_code()
    suffix.graph.eliminate_dead_code()
    prefix.recompile()
    suffix.recompile()
    prefix.graph.lint()
    suffix.graph.lint()
    graph_text = str(traced.graph)
    interface = {"boundary": boundary, "cut_node": matches[-1].name,
                 "crossing_nodes": [node.name for node in crossing],
                 "graph_sha256": hashlib.sha256(graph_text.encode()).hexdigest(),
                 "prefix_graph": str(prefix.graph), "suffix_graph": str(suffix.graph)}
    return prefix, suffix, interface


def tensor_specs(tensors):
    return [{"shape": list(tensor.shape), "dtype": str(tensor.dtype),
             "elements": tensor.numel(), "bytes": tensor.numel() * tensor.element_size(),
             "bits": tensor.numel() * tensor.element_size() * 8} for tensor in tensors]


def module_bytes(module):
    tensors = list(module.parameters()) + list(module.buffers())
    unique = {tensor.data_ptr(): tensor.numel() * tensor.element_size() for tensor in tensors}
    return sum(unique.values())


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def count_operations(module, arguments):
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode(False), torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), FlopCounterMode(display=False) as counter:
            module(*arguments)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    counts = counter.get_flop_counts().get("Global", {})
    return {"flops": counter.get_total_flops(), "operators": {str(operator): count for operator, count in counts.items()},
            "convention": "PyTorch FlopCounterMode registered operators; multiply-add is two FLOPs; not all elementwise operations counted; MHA fastpath disabled and SDPA math backend selected for counting only"}


def statistics_seconds(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing samples must be positive and finite")
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": float(np.quantile(values, .95)), "min": min(values), "max": max(values)}


def summarize_timing(calibration, validation, margin):
    calibration_stats = statistics_seconds(calibration)
    bound = calibration_stats["max"] * (1 + margin)
    return {"calibration_seconds": calibration_stats, "validation_seconds": statistics_seconds(validation),
            "empirical_upper_seconds": bound, "margin": margin,
            "held_out_coverage": sum(value <= bound for value in validation) / len(validation),
            "bound_kind": "calibration max plus fixed margin; not a deterministic guarantee"}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Offline model profiling computation limit reached")


def measure(module, batches, spec, device, deadline):
    samples = {}
    with torch.inference_mode():
        for group in ("calibration", "validation"):
            prepared = batches[group]
            for _ in range(spec["warmup"]):
                check_deadline(deadline)
                module(*prepared[0])
            synchronize(device)
            values = []
            for index in range(spec[group + "_repeats"]):
                check_deadline(deadline)
                arguments = prepared[index % len(prepared)]
                timer = Timer(stmt="module(*arguments)", globals={"module": module, "arguments": arguments},
                              num_threads=spec["cpu_threads"])
                measurement = timer.timeit(number=1)
                values.append(measurement.median)
            samples[group] = values
    return {"raw_seconds": samples, **summarize_timing(samples["calibration"], samples["validation"],
                                                       spec["empirical_bound_margin"])}


def gpu_memory(module, arguments, device, spec):
    if device != "cuda":
        return {"supported": False, "reason": "CPU peak allocator not measured; do not substitute parameter bytes"}
    synchronize(device)
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = module(*arguments)
        synchronize(device)
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    if peak > spec["gpu_allocated_limit_bytes"]:
        raise MemoryError("GPU allocated memory exceeded profiling cap")
    del output
    return {"supported": True, "baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak,
            "incremental_peak_bytes": peak - baseline, "peak_reserved_bytes": reserved,
            "scope": "PyTorch allocator; full model and prepared inputs already resident; excludes driver/non-PyTorch allocations"}


def compare_outputs(actual, expected, spec):
    torch.testing.assert_close(actual, expected, rtol=spec["equivalence_rtol"], atol=spec["equivalence_atol"])
    if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
        raise AssertionError("Split output top-1 differs from complete model")
    return {"max_abs_error": float((actual - expected).abs().max()),
            "top1_equal": True, "allclose": True}


def profile_model(root, name, spec, assets, output):
    validate_spec(spec)
    torch.manual_seed(spec["seed"])
    torch.set_num_threads(spec["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    deadline = started + spec["per_model_timeout_seconds"]
    model = torchvision.models.get_model(name, weights=None)
    state = torch.load(assets[name]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    initial_fingerprint = state_fingerprint(model)
    inputs, input_metadata = load_inputs(spec, assets, name)
    result = {"model": name, "weights": assets[name], "inputs": input_metadata,
              "initial_state_sha256": initial_fingerprint,
              "input_resolution": 224, "batch": 1, "dtype": "float32", "devices": {},
              "model_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
              "status": "running"}
    write_json(output, result)
    try:
        for device in spec["devices"]:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA profiling unavailable")
            check_deadline(deadline)
            model.to(device)
            prepared = {group: [(value.to(device),) for value in values] for group, values in inputs.items()}
            reference = {}
            with torch.inference_mode():
                for group, batches in prepared.items():
                    reference[group] = [model(*arguments).detach().cpu() for arguments in batches]
            full = measure(model, prepared, spec, device, deadline)
            full["operations"] = count_operations(model, prepared["calibration"][0])
            full["memory"] = gpu_memory(model, prepared["calibration"][0], device, spec)
            device_result = {"full": full, "splits": []}
            result["devices"][device] = device_result
            for boundary in spec["boundaries"][name]:
                check_deadline(deadline)
                prefix, suffix, interface = split_graph(model, boundary)
                crossing, checks = {}, []
                with torch.inference_mode():
                    for group, batches in prepared.items():
                        crossing[group] = []
                        for index, arguments in enumerate(batches):
                            features = prefix(*arguments)
                            if not all(isinstance(value, torch.Tensor) for value in features):
                                raise TypeError("Boundary must expose every crossing tensor")
                            crossing[group].append(features)
                            actual = suffix(*features).cpu()
                            checks.append({"group": group, "index": index,
                                           **compare_outputs(actual, reference[group][index], spec)})
                local = measure(prefix, prepared, spec, device, deadline)
                edge = measure(suffix, crossing, spec, device, deadline)
                local["operations"] = count_operations(prefix, prepared["calibration"][0])
                edge["operations"] = count_operations(suffix, crossing["calibration"][0])
                if local["operations"]["flops"] + edge["operations"]["flops"] != full["operations"]["flops"]:
                    raise AssertionError("Counted split operations do not sum to full graph")
                local["memory"] = gpu_memory(prefix, prepared["calibration"][0], device, spec)
                edge["memory"] = gpu_memory(suffix, crossing["calibration"][0], device, spec)
                split = {**interface, "crossing_tensors": tensor_specs(crossing["calibration"][0]),
                         "prefix_resident_bytes": module_bytes(prefix), "suffix_resident_bytes": module_bytes(suffix),
                         "prefix": local, "suffix": edge, "equivalence": checks}
                device_result["splits"].append(split)
                write_json(output, result)
                print(f"PROFILE {name} {device} {boundary} equivalent; prefix={local['calibration_seconds']['median']:.6f}s suffix={edge['calibration_seconds']['median']:.6f}s", flush=True)
                del prefix, suffix, crossing, features, actual
            del prepared, reference
            model.cpu()
            if device == "cuda":
                torch.cuda.empty_cache()
        result["cpu_prefix_cuda_suffix"] = []
        if set(spec["devices"]) == {"cpu", "cuda"}:
            model.cpu()
            with torch.inference_mode():
                expected_outputs = [model(value).cpu() for value in inputs["validation"]]
            for boundary in spec["boundaries"][name]:
                check_deadline(deadline)
                model.cpu()
                prefix, suffix, interface = split_graph(model, boundary)
                suffix = copy.deepcopy(suffix).cuda()
                with torch.inference_mode():
                    for index, value in enumerate(inputs["validation"]):
                        tensors = prefix(value)
                        copied = tuple(torch.from_numpy(tensor.numpy().copy()).cuda() for tensor in tensors)
                        actual = suffix(*copied).cpu()
                        tolerances = {"equivalence_rtol": spec["cross_device_rtol"], "equivalence_atol": spec["cross_device_atol"]}
                        result["cpu_prefix_cuda_suffix"].append({"boundary": boundary, "index": index,
                                                                 **compare_outputs(actual, expected_outputs[index], tolerances)})
                del prefix, suffix, tensors, copied, actual
            model.cpu()
            torch.cuda.empty_cache()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("Profiling produced gradients")
        result["final_state_sha256"] = state_fingerprint(model)
        if result["final_state_sha256"] != initial_fingerprint:
            raise AssertionError("Profiling changed model weights or buffers")
        result["status"] = "completed"
        result["no_gradients"] = True
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["measured_wall_seconds"] = time.perf_counter() - started
        write_json(output, result)
    return result


def environment_metadata():
    result = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
              "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor(),
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "torchvision_license": "BSD-3-Clause for code; weight/data terms separate",
              "sources": {name: {"path": inspect.getfile(inspect.unwrap(torchvision.models.get_model_builder(name))),
                                  "sha256": digest_file(inspect.getfile(inspect.unwrap(torchvision.models.get_model_builder(name))))}
                          for name in ("resnet50", "resnet152", "vit_b_32")}}
    if platform.system() == "Windows":
        hardware = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
                                  capture_output=True, text=True, timeout=20)
        result["cpu_hardware"] = hardware.stdout.strip()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": properties.name, "total_bytes": properties.total_memory}
        measured = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total,power.limit", "--format=csv"],
                                  capture_output=True, text=True, timeout=15)
        result["nvidia_smi"] = measured.stdout
    return result
