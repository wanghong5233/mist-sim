import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from mist_sim.profiling import (compare_outputs, count_operations, fetch_verified, split_graph, summarize_timing,
                                tensor_specs, validate_spec)


def test_split_preserves_all_live_residual_tensors():
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cut = torch.nn.Linear(4, 4)
            self.tail = torch.nn.Linear(4, 4)

        def forward(self, inputs):
            middle = self.cut(inputs)
            return self.tail(middle) + inputs

    model = Residual().eval()
    inputs = torch.rand(1, 4)
    prefix, suffix, interface = split_graph(model, "cut")
    with torch.inference_mode():
        tensors = prefix(inputs)
        assert len(tensors) == 2
        torch.testing.assert_close(suffix(*tensors), model(inputs))
    assert len(interface["crossing_nodes"]) == 2
    assert sum(item["bits"] for item in tensor_specs(tensors)) == 2 * 4 * 4 * 8


@pytest.mark.parametrize("family", ["resnet", "vit"])
def test_official_graph_boundary_equivalence(family):
    torch.set_num_threads(2)
    if family == "resnet":
        model = torchvision.models.resnet18(weights=None).eval()
        inputs = torch.randn(1, 3, 32, 32)
        boundary = "layer2"
    else:
        model = torchvision.models.vision_transformer.VisionTransformer(
            image_size=32, patch_size=8, num_layers=3, num_heads=2, hidden_dim=32, mlp_dim=64).eval()
        torch.nn.init.normal_(model.heads.head.weight)
        inputs = torch.randn(1, 3, 32, 32)
        boundary = "encoder_layer_0"
    prefix, suffix, interface = split_graph(model, boundary)
    with torch.inference_mode():
        crossing = prefix(inputs)
        compare_outputs(suffix(*crossing), model(inputs), {"equivalence_rtol": 1e-5, "equivalence_atol": 1e-6})
    assert len(interface["crossing_nodes"]) == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_timing_bound_does_not_use_heldout_to_widen():
    result = summarize_timing([1., 2., 3.], [2., 4.], .1)
    assert result["empirical_upper_seconds"] == pytest.approx(3.3)
    assert result["held_out_coverage"] == .5
    assert result["calibration_seconds"]["mean"] == 2
    assert result["validation_seconds"]["std"] == pytest.approx(np.std([2., 4.], ddof=1))
    with pytest.raises(ValueError):
        summarize_timing([float("nan")], [1.], .1)


def test_download_checksum_rejects_existing_corruption(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert fetch_verified("unused", path, digest) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fetch_verified("unused", path, digest)


def test_profiling_spec_disjoint_input_split():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/profiling.json").read_text(encoding="utf-8"))
    validate_spec(spec)
    wrong = copy.deepcopy(spec)
    wrong["dataset"]["validation_indices"] = [0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_spec(wrong)


def test_official_flop_counter_units():
    layer = torch.nn.Linear(4, 3, bias=False).eval()
    counted = count_operations(layer, (torch.ones(2, 4),))
    assert counted["flops"] == 2 * 2 * 4 * 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_arithmetic_includes_both_matrix_products(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class Attention(torch.nn.Module):
        def forward(self, queries, keys, values):
            return torch.nn.functional.scaled_dot_product_attention(queries, keys, values)

    arguments = tuple(torch.ones(1, 2, 5, 4, device=device) for _ in range(3))
    counted = count_operations(Attention(), arguments)
    assert counted["flops"] == 2 * (2 * 2 * 5 * 5 * 4)


def test_export_recomputes_timings_and_detects_tampering():
    from mist_sim.profile_export import recompute_timing
    calibration, validation = [.01, .02, .03], [.02, .04]
    record = {"raw_seconds": {"calibration": calibration, "validation": validation},
              **summarize_timing(calibration, validation, .1)}
    recompute_timing(record)
    record["held_out_coverage"] = 1
    with pytest.raises(AssertionError, match="coverage"):
        recompute_timing(record)
