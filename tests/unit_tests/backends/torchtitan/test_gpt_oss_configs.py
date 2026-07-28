###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the GPT-OSS TorchTitan configs added for the v0.2.2 upgrade.

CPU-only: these only parse YAML, so they run in CI without GPU or torchtitan.

Covers:
    - gpt_oss model configs (BF16 + FP8) resolve to name=gpt_oss with the right
      flavor and converters (BF16 -> [], FP8 -> [quantize.linear.float8]).
    - gpt_oss example configs (20B/120B x BF16/FP8 x MI300X/MI325X/MI355X)
      default to Primus-Turbo on (sink attention) and wire the fp8 switch only
      for the FP8 variant.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_CFG_DIR = REPO_ROOT / "primus" / "configs" / "models" / "torchtitan"
EXAMPLE_DIR = REPO_ROOT / "examples" / "torchtitan" / "configs"
MACHINES = ["MI300X", "MI325X", "MI355X"]


def _load(path: Path) -> dict:
    assert path.exists(), f"missing config: {path}"
    with open(path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Model configs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["20b", "120b"])
def test_gpt_oss_bf16_model_config(flavor):
    cfg = _load(MODEL_CFG_DIR / f"gpt_oss_{flavor}.yaml")
    model = cfg["model"]
    assert model["name"] == "gpt_oss"
    assert model["flavor"] == flavor
    # BF16 uses no model converter (sink attention is a setup patch, and the
    # module-form primus_turbo converter is incompatible with GPT-OSS FlexAttn).
    assert model["converters"] == []


@pytest.mark.parametrize("flavor", ["20b", "120b"])
def test_gpt_oss_fp8_model_config(flavor):
    cfg = _load(MODEL_CFG_DIR / f"gpt_oss_{flavor}-fp8.yaml")
    model = cfg["model"]
    assert model["name"] == "gpt_oss"
    assert model["flavor"] == flavor
    # FP8 enables fp8 dense linears through the quantize.linear.float8 converter.
    assert model["converters"] == ["quantize.linear.float8"]


# -----------------------------------------------------------------------------
# Example pretrain configs
# -----------------------------------------------------------------------------


def _example(machine: str, name: str) -> dict:
    return _load(EXAMPLE_DIR / machine / f"{name}-pretrain.yaml")


@pytest.mark.parametrize("machine", MACHINES)
@pytest.mark.parametrize("size", ["20B", "120B"])
@pytest.mark.parametrize("precision", ["BF16", "FP8"])
def test_gpt_oss_example_defaults_turbo_on(machine, size, precision):
    cfg = _example(machine, f"gpt_oss_{size}-{precision}")
    pre = cfg["modules"]["pre_trainer"]
    assert pre["framework"] == "torchtitan"

    expected_model = (
        f"gpt_oss_{size.lower()}.yaml" if precision == "BF16" else f"gpt_oss_{size.lower()}-fp8.yaml"
    )
    assert pre["model"] == expected_model

    turbo = pre["overrides"]["primus_turbo"]
    # Turbo + sink attention are on by default for every gpt_oss example.
    assert turbo["enable_primus_turbo"] is True
    assert turbo["use_turbo_attention"] is True

    if precision == "FP8":
        assert turbo["use_turbo_float8_linear"] is True
    else:
        assert turbo["use_turbo_float8_linear"] is False

    # MoE experts use GPT-OSS's own grouped-mm path; the shared moe fp8 switch
    # must stay off so it is not silently assumed to apply.
    assert turbo["use_moe_fp8"] is False


@pytest.mark.parametrize("machine", MACHINES)
@pytest.mark.parametrize("size", ["20B", "120B"])
@pytest.mark.parametrize("precision", ["BF16", "FP8"])
def test_gpt_oss_example_uses_expert_parallel(machine, size, precision):
    cfg = _example(machine, f"gpt_oss_{size}-{precision}")
    par = cfg["modules"]["pre_trainer"]["overrides"]["parallelism"]
    # GPT-OSS is MoE: experts are sharded with Expert Parallel.
    assert par["expert_parallel_degree"] == 8
    assert par["data_parallel_shard_degree"] == -1
