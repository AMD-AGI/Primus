# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the Wan layer spec.

Covers two invariants that are easy to break silently:

- The q/k norms must be RMSNorm, not LayerNorm. Wan's block norms are
  LayerNorm, so ``config.normalization`` says ``"LayerNorm"`` and the norm
  modules read that field rather than a constructor flag. The spec therefore
  hands the q/k norms a dedicated RMSNorm config clone. Getting this wrong
  changes the math (LayerNorm re-centers) and adds ``.bias`` entries that no
  Wan checkpoint contains.
- The backbone's state dict keys must equal the checkpoint converter's
  analytic key table, since the loader rejects missing or unexpected keys.
"""

import argparse

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
    expected_mcore_keys,
)
from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig
from primus.backends.megatron.core.models.diffusion.wan.model import WanTransformer3D

BACKENDS = ["local", "transformer_engine"]
VARIANTS = [
    ("wan2_1_t2v_1_3b", WanConfig.wan2_1_t2v_1_3b),
    ("wan2_1_t2v_14b", WanConfig.wan2_1_t2v_14b),
    ("wan2_2_ti2v_5b", WanConfig.wan2_2_ti2v_5b),
    ("wan2_2_t2v_a14b", WanConfig.wan2_2_t2v_a14b),
]


def _build_on_meta(ctor, transformer_impl):
    """Instantiate a backbone on the meta device (structure only, no weights)."""
    from megatron.training.global_vars import set_args

    # PrimusTurboLocalAttention reads Megatron's global args at construction.
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))

    config = ctor()
    config.transformer_impl = transformer_impl
    # Both would try to materialize real weights, which meta tensors cannot back.
    config.use_cpu_initialization = False
    config.perform_initialization = False

    with torch.device("meta"):
        return WanTransformer3D(config), config


@pytest.mark.parametrize("transformer_impl", BACKENDS)
def test_qk_norm_is_rms_norm_without_bias(transformer_impl):
    """q/k norms are RMSNorm and carry a weight but no bias."""
    model, _ = _build_on_meta(WanConfig.wan2_1_t2v_1_3b, transformer_impl)

    for attn_name in ("attn1", "attn2"):
        attn = getattr(model.blocks[0], attn_name)
        for norm in (attn.q_layernorm, attn.k_layernorm):
            assert type(norm).__name__ == "RMSNorm", f"{attn_name}: got {type(norm).__name__}"
            params = {name for name, _ in norm.named_parameters()}
            assert params == {"weight"}, f"{attn_name}: unexpected params {params}"


@pytest.mark.parametrize("transformer_impl", BACKENDS)
def test_qk_norm_uses_backend_native_kernel(transformer_impl):
    """The TE backend uses TE's RMSNorm; the local backend uses torch's."""
    model, _ = _build_on_meta(WanConfig.wan2_1_t2v_1_3b, transformer_impl)
    origin = type(model.blocks[0].attn1.q_layernorm).__module__

    if transformer_impl == "transformer_engine":
        assert origin.startswith("transformer_engine"), origin
    else:
        assert origin.startswith("torch."), origin


@pytest.mark.parametrize("transformer_impl", BACKENDS)
@pytest.mark.parametrize("variant_name,ctor", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_state_dict_matches_converter_key_table(variant_name, ctor, transformer_impl):
    """The backbone's keys equal the converter's analytic mcore key table."""
    model, config = _build_on_meta(ctor, transformer_impl)

    # TE linears carry _extra_state entries, which are not weights.
    actual = {key for key in model.state_dict() if not key.endswith("_extra_state")}
    expected = expected_mcore_keys(config)

    assert actual == expected, (
        f"{variant_name}/{transformer_impl}: "
        f"missing={sorted(expected - actual)[:8]} unexpected={sorted(actual - expected)[:8]}"
    )
