###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlexAttention backend against eager, on one Inkling core-attention call."""

from __future__ import annotations

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    inkling_config_from_model_yaml,
)
from primus.backends.megatron.core.transformer.inkling.attention import (
    InklingCoreAttention,
    flex_attention,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import preset

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU"),
    pytest.mark.skipif(flex_attention is None, reason="torch has no flex_attention"),
]

SEQ = 160  # past the sliding window (32), the global bias extent (64) and the log-scaling floor (64)
BATCH = 2


def _inputs(config, layer_idx, dtype, seed=0):
    geo = config.attention_geometry(layer_idx)
    h, g, d = geo["num_attention_heads"], geo["num_query_groups"], geo["kv_channels"]
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, std=1.0):
        return (std * torch.randn(*shape, generator=gen, device="cuda")).to(dtype).requires_grad_()

    return (
        rand(SEQ, BATCH, h, d),
        rand(SEQ, BATCH, g, d),
        rand(SEQ, BATCH, g, d),
        rand(SEQ, BATCH, h, config.d_rel),
        rand(config.d_rel, geo["rel_extent"], std=0.5),
    )


def _run(attention, backend, inputs, attention_mask=None):
    attention.backend = backend
    out = attention(*inputs, attention_mask=attention_mask)
    grads = torch.autograd.grad(out, inputs, torch.ones_like(out) / out.numel() ** 0.5)
    return out, grads


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "global"])
@pytest.mark.parametrize("padding", [False, True], ids=["no_mask", "padding_mask"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_flex_matches_eager(layer_idx, padding, dtype):
    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"), params_dtype=dtype)
    attention = InklingCoreAttention(
        config.attention_config(layer_idx), layer_number=layer_idx + 1, attn_mask_type=AttnMaskType.causal
    ).cuda()
    assert attention.is_sliding == (layer_idx == 0) and attention.log_scaling == (layer_idx == 1)
    inputs = _inputs(config, layer_idx, dtype)

    mask = None
    if padding:
        # The second row hides keys 10-29 from every later query; no row is fully masked.
        mask = torch.zeros(BATCH, 1, SEQ, SEQ, dtype=torch.bool, device="cuda")
        mask[1, :, 30:, 10:30] = True

    eager, eager_grads = _run(attention, "eager", inputs, mask)
    flex, flex_grads = _run(attention, "flex", inputs, mask)
    if padding:
        assert not torch.allclose(
            eager, _run(attention, "eager", inputs)[0]
        ), "the padding mask changed nothing"

    tol = 2e-5 if dtype == torch.float32 else 2e-2
    scale = eager.float().abs().max()
    assert ((flex.float() - eager.float()).abs().max() / scale) < tol
    for name, a, b in zip(("query", "key", "value", "rel_states", "rel_proj"), flex_grads, eager_grads):
        err = ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12)).item()
        assert err < tol, f"d{name}: max |diff| / max |eager| = {err:.2e}"


def test_flex_rejects_attention_dropout():
    config = inkling_config_from_model_yaml(
        preset("inkling_small_tiny"), params_dtype=torch.float32, attention_dropout=0.1
    )
    attention = (
        InklingCoreAttention(config, layer_number=2, attn_mask_type=AttnMaskType.causal).cuda().train()
    )
    attention.backend = "flex"
    with pytest.raises(NotImplementedError, match="dropout"):
        attention(*_inputs(config, 1, torch.float32))
