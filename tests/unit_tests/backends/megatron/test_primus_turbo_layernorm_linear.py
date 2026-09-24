###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for PrimusTurboLayerNormColumnParallelLinear's zero-centered gamma.

Regression guard, sibling of ``test_primus_turbo_rmsnorm.py``. Under
``use_turbo_gemm`` this class computes the input norm fused into ``linear_qkv``
(and a dense MLP's ``linear_fc1``). It reads TE's ``layer_norm_weight`` -- which
``layernorm_zero_centered_gamma`` initialises to *zeros* -- and used to pass it
to the norm verbatim, dropping the ``1 +`` TE applies, so every output was zero.

Parity is checked against ``TELayerNormColumnParallelLinear``, the upstream
class this one stands in for, rather than against a formula alone.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("primus_turbo")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="turbo GEMM needs a GPU")

IN, OUT, EPS = 128, 256, 1.0e-6


@pytest.fixture(autouse=True)
def _megatron_state(init_parallel_state, monkeypatch):
    """TP=1 parallel state plus the few global args the turbo class reads."""
    import primus.backends.megatron.core.extensions.primus_turbo as turbo

    args = SimpleNamespace(
        offload=False,
        offload_ops=[],
        turbo_gemm_backend="default",
        patch_primus_pipeline=False,
        pp_algorithm=None,
        patch_zero_bubble=False,
        enable_zero_bubble=False,
    )
    monkeypatch.setattr(turbo, "get_args", lambda: args)


def _config(normalization, zero_centered):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=IN,
        num_attention_heads=4,
        normalization=normalization,
        layernorm_epsilon=EPS,
        layernorm_zero_centered_gamma=zero_centered,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _build(cls, config):
    return cls(
        IN,
        OUT,
        config=config,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()


def _turbo(config):
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLayerNormColumnParallelLinear,
    )

    return _build(PrimusTurboLayerNormColumnParallelLinear, config)


def _output(module, x):
    out = module(x)
    return out[0] if isinstance(out, tuple) else out


def _input():
    return torch.randn(16, 2, IN, device="cuda", dtype=torch.bfloat16)


def test_zero_centered_output_is_not_zero_at_init():
    """The bug in one assertion: at init a zero-centred gamma is all zeros, and the
    norm must still pass the normalised activations through."""
    module = _turbo(_config("RMSNorm", zero_centered=True))
    assert module.zero_centered_gamma
    assert module.layer_norm_weight.detach().abs().sum().item() == 0.0

    assert _output(module, _input()).abs().sum().item() > 0.0


@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_matches_transformer_engine(normalization, zero_centered):
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )

    config = _config(normalization, zero_centered)
    turbo = _turbo(config)
    reference = _build(TELayerNormColumnParallelLinear, config)

    with torch.no_grad():
        reference.weight.copy_(turbo.weight)
        gamma = torch.randn(IN, device="cuda") * 0.1
        turbo.layer_norm_weight.copy_(gamma)
        reference.layer_norm_weight.copy_(gamma)
        if normalization == "LayerNorm":
            beta = torch.randn(IN, device="cuda") * 0.1
            turbo.layer_norm_bias.copy_(beta)
            reference.layer_norm_bias.copy_(beta)

    x = _input()
    torch.testing.assert_close(_output(turbo, x), _output(reference, x), rtol=2e-2, atol=2e-2)
