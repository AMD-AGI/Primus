###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MinimaxSparseAttention with msa_backend=flydsl against msa_backend=eager.

Same weights, same input: the output, every gradient (input, projections,
indexer) and the recorded sparse indexer loss must agree. This is the module
the runtime builds, from real TE specs, so it also covers the wiring --
autograd, the slot-mass target, the loss autoscaler -- not just the kernels.
"""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="the flydsl MSA kernels target gfx950",
)

S, BATCH, GROUPS, HIDDEN = 600, 2, 2, 512


@pytest.fixture(autouse=True)
def _parallel(init_parallel_state):
    pass


def _snr_db(ref, x):
    ref, x = ref.double(), x.double()
    return (10 * torch.log10(ref.norm() ** 2 / ((ref - x).norm() ** 2 + 1e-12))).item()


def _build(backend):
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        MinimaxSparseAttention,
    )
    from primus.backends.megatron.patches.minimax_m3_patches import _widen_submodules

    config = MSATransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=16 * GROUPS,
        num_query_groups=GROUPS,
        kv_channels=128,
        normalization="RMSNorm",
        activation_func=quick_gelu,
        gated_linear_unit=True,
        add_bias_linear=False,
        sparse_num_index_heads=GROUPS,
        sparse_index_dim=64,
        sparse_block_size=128,
        sparse_topk_blocks=3,
        sparse_indexer_loss_coeff=1.0e-2,
        msa_backend=backend,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    submodules = _widen_submodules(spec.submodules.self_attention.submodules)
    return config, MinimaxSparseAttention(config, submodules, layer_number=1).cuda()


def _run(module, hidden, rope, dout):
    from megatron.core.transformer.moe.moe_utils import (
        get_moe_layer_wise_logging_tracker,
    )

    from primus.backends.megatron.core.transformer.minimax_m3 import (
        MSA_INDEXER_LOSS_NAME,
    )

    tracker = get_moe_layer_wise_logging_tracker()
    tracker.pop(MSA_INDEXER_LOSS_NAME, None)
    module.train()
    module.zero_grad(set_to_none=True)
    x = hidden.clone().requires_grad_(True)
    out, _ = module(x, rotary_pos_emb=rope)
    out.backward(dout)
    loss = tracker.pop(MSA_INDEXER_LOSS_NAME)["values"][0].item()
    grads = {name: p.grad.float() for name, p in module.named_parameters() if p.grad is not None}
    return out.float(), x.grad.float(), grads, loss


def test_flydsl_module_matches_eager():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    torch.manual_seed(0)
    _, eager = _build("eager")
    _, fly = _build("flydsl")
    with torch.no_grad():
        for p in eager.parameters():
            p.normal_(0.0, 0.05)
        fly.load_state_dict(eager.state_dict())

    gen = torch.Generator(device="cuda").manual_seed(1)
    hidden = torch.randn(S, BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
    rope = RotaryEmbedding(kv_channels=128, rotary_percent=0.5)(S).cuda()

    out_e, dx_e, grads_e, loss_e = _run(eager, hidden, rope, dout)
    out_f, dx_f, grads_f, loss_f = _run(fly, hidden, rope, dout)

    assert _snr_db(out_e, out_f) >= 40.0, f"output SNR {_snr_db(out_e, out_f):.1f} dB"
    assert _snr_db(dx_e, dx_f) >= 40.0, f"input grad SNR {_snr_db(dx_e, dx_f):.1f} dB"
    assert grads_e.keys() == grads_f.keys()
    assert any("indexer" in name for name in grads_f), "the indexer received no gradient"
    for name in grads_e:
        snr = _snr_db(grads_e[name], grads_f[name])
        assert snr >= 40.0, f"{name}: grad SNR {snr:.1f} dB"
    assert loss_e > 0
    assert abs(loss_f - loss_e) <= 1e-3 * loss_e, f"indexer loss eager {loss_e:.6g} vs flydsl {loss_f:.6g}"


def test_flydsl_rejects_a_padding_mask():
    _, fly = _build("flydsl")
    hidden = torch.randn(256, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 256, 256, device="cuda", dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="padding mask"):
        fly(hidden, attention_mask=mask)
