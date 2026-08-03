###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 CSA parity suite, split into two kinds of assertion.

The two failure modes these guard against are genuinely different, and mixing
them is how a wiring bug hides behind green maths:

**Group 1 -- numerical parity.** Rebuild the CSA forward from the module's own
parameters using plain matmuls, independent of the production helpers, and
compare element-wise. This is where a wrong RoPE position, a missing score
scale, a dropped Hadamard rotation or an absent output de-rotation shows up.
Two steps are reused rather than rewritten: the joint local-SWA + sparse softmax
(``eager_v4_csa_attention``) and the compressor pooling. Both are shared
contracts with their own dedicated tests -- every CSA backend is pinned against
the former, and the pooling has its own parity coverage -- so duplicating them
here would test a copy instead of the contract. Everything the recent fixes
touched (RoPE positions, the indexer's operand preparation, the output
de-rotation, the softmax temperature) is rebuilt from scratch.

**Group 2 -- gradient topology.** Assert which parameters each objective may and
may not reach. Scale and detach bugs are invisible to Group 1 -- the forward is
bit-identical either way -- and equally invisible to a loss-value assertion. The
only thing that catches them is asking where the gradient went.

Everything here is CPU-friendly: the attention is built with the ``eager``
backends and no process group.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

mla_module = pytest.importorskip(
    "megatron.core.transformer.multi_latent_attention",
    reason="MLA base module not importable in this environment",
)

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: E402
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import (  # noqa: E402
    DualRoPE,
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.hadamard_rotation import (  # noqa: E402
    rotate_activation,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._eager.reference import (  # noqa: E402
    eager_v4_csa_attention,
)

_DTYPE = torch.float32

_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = 16
_ROTARY_DIM = 8
_RATIO = 4
_INDEX_TOPK = 2
_INDEX_HEAD_DIM = 16
_INDEX_N_HEADS = 2


def _make_csa_attention(coeff: float = 0.0) -> DeepseekV4Attention:
    config = DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_query_groups=1,
        kv_channels=_HEAD_DIM,
        qk_pos_emb_head_dim=_ROTARY_DIM,
        qk_head_dim=_ROTARY_DIM,
        v_head_dim=_HEAD_DIM,
        kv_lora_rank=_HEAD_DIM,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sliding_window=0,
        attn_sink=True,
        compress_ratios=None,
        compress_rope_theta=40000.0,
        use_v4_attention_backend="eager",
        use_v4_csa_attention_backend="eager",
        layernorm_epsilon=1e-6,
        norm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        v4_indexer_distill_loss_coeff=coeff,
    )
    config.index_topk = _INDEX_TOPK
    config.index_head_dim = _INDEX_HEAD_DIM
    config.index_n_heads = _INDEX_N_HEADS

    rope = DualRoPE(
        rotary_dim=_ROTARY_DIM,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
        yarn_factor=1.0,
        original_max_position_embeddings=config.original_max_position_embeddings,
    )
    attn = DeepseekV4Attention(config, rope=rope, compress_ratio=_RATIO, submodules=None, layer_number=1)
    return attn.to(_DTYPE)


def _rms_per_head(x: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


# ---------------------------------------------------------------------------
# Group 1: numerical parity
# ---------------------------------------------------------------------------


def _reference_csa_forward(attn: DeepseekV4Attention, hidden, position_ids):
    """Independent CSA reference assembled from the module's parameters."""
    B, S, _ = hidden.shape
    H, Dh, rd, eps = attn.num_heads, attn.head_dim, attn.rotary_dim, attn.norm_eps
    compress_rope = attn.rope.get_rope(compress_ratio=_RATIO)

    # --- Q / KV branches -------------------------------------------------
    q_c = hidden @ attn.linear_q_down_proj.weight.t()
    q_c = _rms_per_head(q_c, eps) * attn.q_layernorm.weight
    q = _rms_per_head((q_c @ attn.linear_q_up_proj.weight.t()).view(B, S, H, Dh), eps)

    kv = hidden @ attn.linear_kv.weight.t()
    kv = (_rms_per_head(kv, eps) * attn.kv_layernorm.weight).view(B, S, 1, Dh)

    q = attn.rope.apply_rope(q, position_ids=position_ids, compress_ratio=_RATIO)
    kv = attn.rope.apply_rope(kv, position_ids=position_ids, compress_ratio=_RATIO)

    # --- compressed pool, rotated at s * ratio ---------------------------
    pool = attn.compressor(hidden)  # [B, P, Dh]
    P = pool.shape[1]
    comp_pos = torch.arange(P, device=hidden.device) * _RATIO
    cos, sin = compress_rope(comp_pos)
    pool_rot = apply_interleaved_partial_rope(
        pool.unsqueeze(2),
        cos.unsqueeze(0).expand(B, -1, -1),
        sin.unsqueeze(0).expand(B, -1, -1),
        rotary_dim=rd,
    ).squeeze(2)

    # --- indexer: RoPE + Hadamard on both operands, then the scaled score -
    idx = attn.indexer
    k_icomp = idx.indexer_compressor(hidden)  # [B, P, Hd_i]
    k_icomp = rotate_activation(
        apply_interleaved_partial_rope(
            k_icomp.unsqueeze(2),
            cos.unsqueeze(0).expand(B, -1, -1),
            sin.unsqueeze(0).expand(B, -1, -1),
            rotary_dim=rd,
        ).squeeze(2)
    )

    if idx._fuse_qw_proj:
        dqw = idx.w_dq_w(hidden)
        q_q, w_i = dqw[..., : idx.dq_rank], dqw[..., idx.dq_rank :]
    else:
        q_q, w_i = idx.w_dq(hidden), idx.w_w(hidden)
    q_i = idx.w_iuq(q_q).view(B, S, _INDEX_N_HEADS, _INDEX_HEAD_DIM)
    icos, isin = compress_rope(position_ids)
    if tuple(icos.shape[:-1]) != (B, S):
        icos, isin = icos.expand(B, S, -1), isin.expand(B, S, -1)
    q_i = rotate_activation(apply_interleaved_partial_rope(q_i, icos, isin, rotary_dim=rd))
    w_i = w_i * ((_INDEX_N_HEADS**-0.5) * (_INDEX_HEAD_DIM**-0.5))

    scores = (F.relu(torch.einsum("bshd,bpd->bshp", q_i, k_icomp)) * w_i.unsqueeze(-1)).sum(dim=2)
    t_idx = torch.arange(S, device=hidden.device).unsqueeze(1)
    s_end = (torch.arange(P, device=hidden.device).unsqueeze(0) + 1) * _RATIO - 1
    scores = scores + torch.where(s_end <= t_idx, 0.0, float("-inf")).to(scores.dtype)

    topk_eff = min(_INDEX_TOPK, P)
    topk_scores, topk_idxs = scores.topk(topk_eff, dim=-1)
    topk_idxs = torch.where(torch.isneginf(topk_scores), -1, topk_idxs)

    # --- joint softmax over local SWA + the selected compressed entries --
    gathered = pool_rot[torch.arange(B).view(B, 1, 1), topk_idxs.clamp_min(0)]
    sparse_mask = torch.where(topk_idxs >= 0, 0.0, float("-inf")).to(hidden.dtype)
    out_bh = eager_v4_csa_attention(
        q.transpose(1, 2),
        kv.expand(B, S, H, Dh).transpose(1, 2),
        kv.expand(B, S, H, Dh).transpose(1, 2),
        gathered,
        sink=attn.attn_sink,
        swa_window=int(attn.attn_sliding_window),
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=1.0 / math.sqrt(Dh),
    )
    out = out_bh.transpose(1, 2).contiguous()

    # --- de-rotate at -t, then the grouped O projection ------------------
    dcos, dsin = compress_rope(position_ids)
    if tuple(dcos.shape[:-1]) != (B, S):
        dcos, dsin = dcos.expand(B, S, -1), dsin.expand(B, S, -1)
    out = apply_interleaved_partial_rope(out, dcos, -dsin, rotary_dim=rd)

    G, r = attn.o_groups, attn.o_lora_rank
    o = torch.einsum(
        "bsgd,grd->bsgr",
        out.reshape(B, S, G, (H * Dh) // G),
        attn.linear_o_a.weight.view(G, r, (H * Dh) // G),
    )
    return o.flatten(2) @ attn.linear_o_b.weight.t()


def test_csa_forward_matches_the_independent_reference():
    """End-to-end CSA parity against a reference built from scratch."""
    torch.manual_seed(0)
    attn = _make_csa_attention()
    attn.eval()

    B, S = 2, 16
    hidden = torch.randn(B, S, _HIDDEN, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    with torch.no_grad():
        ours = attn(hidden, position_ids)
        ref = _reference_csa_forward(attn, hidden, position_ids)

    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)


def test_output_actually_depends_on_the_indexer_selection():
    """Guard the parity test from passing with the indexer bypassed.

    Also marks the boundary of what parity can cover: the indexer's score scale
    is a positive constant, so it cannot move ``topk`` and therefore cannot show
    up in the output at all -- which is precisely why it went missing. Pinning
    that value needs a direct assertion (``test_v4_indexer_score_scale.py``),
    not this test.
    """
    torch.manual_seed(0)
    attn = _make_csa_attention()
    attn.eval()

    B, S = 1, 16
    hidden = torch.randn(B, S, _HIDDEN, dtype=_DTYPE)
    position_ids = torch.arange(S)

    with torch.no_grad():
        base = _reference_csa_forward(attn, hidden, position_ids)
        # Flip one indexer head's weight so the selection genuinely differs.
        attn.indexer.w_dq_w.weight[attn.indexer.dq_rank].mul_(-3.0)
        perturbed = _reference_csa_forward(attn, hidden, position_ids)

    assert not torch.allclose(base, perturbed, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("shift", [0, 7])
def test_parity_holds_at_shifted_positions(shift: int):
    """Positions enter in three places; a shift must move all of them together.

    Q / KV rotation, the compressed pool's ``s * ratio``, and the output
    de-rotation all read the position axis. A parity test only at
    ``arange(S)`` would miss an off-by-origin in any of them.
    """
    torch.manual_seed(1)
    attn = _make_csa_attention()
    attn.eval()

    B, S = 1, 16
    hidden = torch.randn(B, S, _HIDDEN, dtype=_DTYPE)
    position_ids = torch.arange(S) + shift

    with torch.no_grad():
        ours = attn(hidden, position_ids)
        ref = _reference_csa_forward(attn, hidden, position_ids)

    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Group 2: gradient topology
# ---------------------------------------------------------------------------


def _backbone_names(attn: DeepseekV4Attention) -> set:
    return {n for n, _ in attn.named_parameters() if not n.startswith("indexer.")}


def test_main_loss_reaches_the_backbone_but_not_the_frozen_indexer(monkeypatch):
    """With the distillation loss off, the indexer must be unreachable.

    ``topk`` is not differentiable and the scores are discarded, so a gradient
    arriving at the indexer here would mean something else is consuming them.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=0.0)
    attn.train()

    B, S = 1, 8
    hidden = torch.randn(B, S, _HIDDEN, dtype=_DTYPE)
    attn(hidden, torch.arange(S).unsqueeze(0).expand(B, S)).sum().backward()

    got_grad = {n for n, p in attn.named_parameters() if p.grad is not None}
    assert got_grad & _backbone_names(attn), "the main loss did not reach the backbone"
    assert not any(n.startswith("indexer.") for n in got_grad)


def test_distillation_loss_reaches_only_the_indexer(monkeypatch):
    """The KL is one-directional, so its gradient must stop at the indexer.

    Run the same weights and input with the loss off and with a large
    coefficient: every backbone gradient has to be bit-identical, while the
    indexer goes from no gradient to a non-zero one.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    torch.manual_seed(0)
    off = _make_csa_attention(coeff=0.0)
    on = _make_csa_attention(coeff=1e3)
    on.load_state_dict(off.state_dict())
    off.train()
    on.train()

    B, S = 1, 8
    torch.manual_seed(2)
    base = torch.randn(B, S, _HIDDEN, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    grads = {}
    inputs = {}
    for tag, module in (("off", off), ("on", on)):
        hidden = base.clone().requires_grad_(True)
        module(hidden, position_ids).sum().backward()
        inputs[tag] = hidden.grad.clone()
        grads[tag] = {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}

    for name in _backbone_names(off):
        if name not in grads["off"]:
            continue
        torch.testing.assert_close(
            grads["on"][name],
            grads["off"][name],
            rtol=0,
            atol=0,
            msg=lambda s, n=name: f"KL perturbed backbone gradient {n}: {s}",
        )
    torch.testing.assert_close(inputs["on"], inputs["off"], rtol=0, atol=0)

    indexer_on = {n for n in grads["on"] if n.startswith("indexer.")}
    assert indexer_on, "the indexer received no gradient at coeff > 0"
    assert not any(n.startswith("indexer.") for n in grads["off"])


def test_ape_and_sink_receive_fp32_gradients(monkeypatch):
    """With the pinning opted in, it must survive into the gradients too.

    ``PRIMUS_V4_KEEP_FP32`` ships off, so this asserts the opt-in path rather
    than the default one -- a mixed-dtype model is exactly what the distributed
    optimizer cannot take (see ``keep_in_fp32``).
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    monkeypatch.setenv("PRIMUS_V4_KEEP_FP32", "1")
    torch.manual_seed(0)
    attn = _make_csa_attention()
    attn.bfloat16()
    attn.train()

    B, S = 1, 8
    hidden = torch.randn(B, S, _HIDDEN, dtype=torch.bfloat16)
    attn(hidden, torch.arange(S).unsqueeze(0).expand(B, S)).sum().backward()

    assert attn.attn_sink.grad is not None
    assert attn.attn_sink.grad.dtype == torch.float32
    assert attn.compressor.ape.grad is not None
    assert attn.compressor.ape.grad.dtype == torch.float32
    # ...and the ordinary weights still carry BF16 gradients.
    assert attn.linear_kv.weight.grad.dtype == torch.bfloat16
