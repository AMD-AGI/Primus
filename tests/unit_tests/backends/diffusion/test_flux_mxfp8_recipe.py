from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest
import torch


def _load_mxfp4(monkeypatch):
    module_name = "primus.backends.diffusion.models.flux.mxfp4"
    if module_name in sys.modules:
        return sys.modules[module_name]
    backend = types.ModuleType("primus_turbo.pytorch.core.backend")
    backend.BackendType = types.SimpleNamespace(
        HIPBLASLT=types.SimpleNamespace(value="hipblaslt"),
        FLYDSL=types.SimpleNamespace(value="flydsl"),
    )
    low_precision = types.ModuleType("primus_turbo.pytorch.core.low_precision")
    low_precision.ScalingGranularity = types.SimpleNamespace(
        MX_BLOCKWISE=types.SimpleNamespace(value="mx")
    )
    stubs = {
        "primus_turbo.pytorch.core.quantized_tensor": {},
        "primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl": {"gemm_fp4_impl": lambda *a, **k: None},
        "primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl": {"gemm_fp8_impl": lambda *a, **k: None},
        "primus_turbo.pytorch.ops.quantization": {"dequantize_fp8": lambda *a, **k: None},
    }
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, low_precision.__name__, low_precision)
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("profile", "recipe", "float8", "mxfp4", "fp8_ag", "mxfp8_ag", "sdma"),
    [
        ("config_4n_gbs1024.sh", "fp8", "tensorwise", "", "1", "0", "rccl"),
        ("config_4n_gbs1024_mxfp8.sh", "mxfp8", "", "custom", "0", "1", "rccl"),
        (
            "config_4n_gbs1024_mxfp8_sdma.sh",
            "mxfp8",
            "",
            "custom",
            "0",
            "1",
            "rccl_sdma",
        ),
    ],
)
def test_flux_recipe_profiles(profile, recipe, float8, mxfp4, fp8_ag, mxfp8_ag, sdma):
    command = (
        f"source examples/mlperf/flux1/{profile}; "
        'printf "%s\\n" "$FLUX_TRAINING_RECIPE" "$FLUX_FLOAT8_RECIPE" '
        '"$FLUX_MXFP4_RECIPE" "$FLUX_FP8_ALL_GATHER" '
        '"$FLUX_MXFP8_ALL_GATHER" "$FSDP_ALL_GATHER_BACKEND"'
    )
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [recipe, float8, mxfp4, fp8_ag, mxfp8_ag, sdma]


def test_unknown_flux_recipe_fails():
    env = os.environ.copy()
    env["FLUX_TRAINING_RECIPE"] = "unknown"
    result = subprocess.run(
        ["bash", "-c", "source examples/mlperf/flux1/config_common.sh"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Unsupported FLUX_TRAINING_RECIPE" in result.stderr


def test_mxfp8_fsdp_hooks_transport_bytes(monkeypatch):
    mxfp4 = _load_mxfp4(monkeypatch)
    weight = torch.randn(128, 128)
    wrapped = mxfp4.MXFP8FSDPWeight(weight, 128)
    monkeypatch.setattr(
        mxfp4,
        "_quantize_mxfp8_rowwise",
        lambda value, use_2d: (
            torch.zeros(128, 128, dtype=torch.float8_e4m3fn),
            torch.zeros(128, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        ),
    )
    payloads, metadata = wrapped.fsdp_pre_all_gather(object())
    assert [tensor.dtype for tensor in payloads] == [torch.uint8, torch.uint8]
    assert [tuple(tensor.shape) for tensor in payloads] == [(128, 128), (128, 4)]
    assert metadata == (128, 128, 4)


def test_sdma_attachment_is_recipe_selective(monkeypatch):
    from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    attached = []
    process_group = object()

    class FakeAllGather:
        def __init__(self, group, backend):
            assert group is process_group
            assert backend == "NCCL"

    class FakeModule:
        def _get_fsdp_state(self):
            group = type("Group", (), {"_all_gather_process_group": process_group})()
            return type("State", (), {"_fsdp_param_groups": [group]})()

        def set_custom_all_gather(self, all_gather):
            attached.append(all_gather)

    monkeypatch.setattr(_fsdp_collectives, "SymmMemAllGather", FakeAllGather, raising=False)
    monkeypatch.setenv("FSDP_ALL_GATHER_BACKEND", "rccl_sdma")
    monkeypatch.setenv("FSDP_SDMA_MODULES", "single_blocks")
    trainer = FSDP2Trainer.__new__(FSDP2Trainer)
    trainer.rank = 1
    trainer._maybe_attach_sdma_all_gather(FakeModule(), "double_blocks.0")
    trainer._maybe_attach_sdma_all_gather(FakeModule(), "single_blocks.0")
    assert len(attached) == 1
