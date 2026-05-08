###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P24 — refactor safety net (G22).

P24 extracts the math that previously lived inline in
:meth:`DeepseekV4Attention._attention_forward` and
:meth:`DeepseekV4Attention._csa_forward` into pure functions in
``primus.backends.megatron.core.transformer.v4_attention_kernels``.

This file is the safety-net unit-test side of P24's release gate (G22).
For each of the three V4 layer kinds (dense ``compress_ratio == 0``,
HCA ``compress_ratio == 128``, CSA ``compress_ratio == 4``) the test:

* runs the new reference op (``eager_v4_attention`` /
  ``eager_v4_csa_attention``);
* runs an **inline** verbatim copy of the **pre-P24** math (defined
  at the top of this file — see ``_pre_p24_attention_forward`` and
  ``_pre_p24_csa_attention_forward``); the inline copy is the
  tamper-proof reference because it lives in this test file and is
  never imported elsewhere;
* asserts forward output is bit-identical (or fp32-tolerance equal,
  for paths that go through ``additive_mask`` rebuild) and every leaf
  gradient matches under :func:`compare_fwd_bwd`.

In addition, a higher-level test exercises a small CPU-toy
:class:`DeepseekV4Attention` end-to-end and compares its output
against the inline reference, so the refactor at the `_attention_forward`
/ `_csa_forward` call sites stays bit-identical to the released V4
checkpoint baseline.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

# Skip the whole module if Megatron / TE imports fall over (matches the
# convention in the existing V4 attention test suite).
pytest.importorskip(
    "megatron.core.transformer.multi_latent_attention",
    reason="Megatron MLA not importable in this environment",
)

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: E402
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE  # noqa: E402
from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_attention,
    eager_v4_csa_attention,
)
from tests.unit_tests.megatron.transformer.deepseek_v4.v4_attention_test_utils import (  # noqa: E402
    FP32_TOL,
    compare_fwd_bwd,
)

# ---------------------------------------------------------------------------
# Inline pre-P24 reference math (verbatim copy — DO NOT import from elsewhere)
# ---------------------------------------------------------------------------


def _pre_p24_softmax_with_sink(
    logits: torch.Tensor,
    sink,
    num_heads: int,
) -> torch.Tensor:
    """Verbatim pre-P24 :meth:`DeepseekV4Attention._append_sink_softmax`."""
    if sink is None:
        logits = logits - logits.amax(dim=-1, keepdim=True).detach()
        return logits.softmax(dim=-1)

    ndim = logits.dim()
    view_shape = [1] * ndim
    view_shape[1] = num_heads
    view_shape[-1] = 1
    target_shape = list(logits.shape[:-1]) + [1]
    sink_col = sink.float().view(*view_shape).expand(*target_shape)
    logits_aug = torch.cat([logits, sink_col], dim=-1)
    logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
    probs = logits_aug.softmax(dim=-1)
    return probs[..., :-1]


def _pre_p24_attention_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink,
    additive_mask: torch.Tensor,
    attn_dropout: float,
    training: bool,
    scale: float,
    num_heads: int,
) -> torch.Tensor:
    """Verbatim pre-P24 :meth:`DeepseekV4Attention._attention_forward`."""
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    logits = logits + additive_mask
    probs = _pre_p24_softmax_with_sink(logits, sink, num_heads)
    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)
    return torch.matmul(probs.to(v.dtype), v)


def _pre_p24_csa_attention_forward(
    *,
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gathered: torch.Tensor,
    sink,
    sparse_mask: torch.Tensor,
    local_mask: torch.Tensor,
    attn_dropout: float,
    training: bool,
    scale: float,
    num_heads: int,
) -> torch.Tensor:
    """Verbatim pre-P24 :meth:`DeepseekV4Attention._csa_forward` (post-gather)."""
    B, H, S, Dh = q.shape
    K = gathered.shape[2]

    local_logits = torch.matmul(q.float(), k_local.float().transpose(-2, -1)) * scale
    local_logits = local_logits + local_mask  # [S, S] broadcasts over B, H

    gathered_h = gathered.unsqueeze(1).expand(B, H, S, K, Dh).float()
    sparse_logits = torch.einsum("bhsd,bhskd->bhsk", q.float(), gathered_h) * scale
    sparse_logits = sparse_logits + sparse_mask.unsqueeze(1)

    joint_logits = torch.cat([local_logits, sparse_logits], dim=-1)
    probs = _pre_p24_softmax_with_sink(joint_logits, sink, num_heads)

    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)

    probs_local = probs[..., :S].to(v_local.dtype)
    probs_sparse = probs[..., S:].to(v_local.dtype)

    out_local = torch.matmul(probs_local, v_local)
    out_sparse = torch.einsum("bhsk,bhskd->bhsd", probs_sparse, gathered_h.to(v_local.dtype))
    return out_local + out_sparse


# ---------------------------------------------------------------------------
# Small CPU shape used by every G22 test
# ---------------------------------------------------------------------------


def _toy_inputs(
    *,
    B: int = 2,
    H: int = 4,
    S: int = 8,
    Dh: int = 8,
    K: int = 3,
    swa_window: int = 0,
    sink: bool = True,
    dtype: torch.dtype = torch.float32,
    seed: int = 1234,
):
    """Build a self-contained set of tensors used by every G22 reference-op test."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, S, Dh, generator=g, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, Dh, generator=g, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, Dh, generator=g, dtype=dtype, requires_grad=True)
    gathered = torch.randn(B, S, K, Dh, generator=g, dtype=dtype, requires_grad=True)
    sink_t = torch.randn(H, generator=g, dtype=torch.float32, requires_grad=True) if sink else None
    local_mask = sliding_window_causal_mask(
        S,
        swa_window if swa_window > 0 else S,
        device=q.device,
        dtype=q.dtype,
    )
    valid = torch.ones(B, S, K, dtype=torch.bool)
    sparse_mask = torch.where(valid, 0.0, float("-inf")).to(q.dtype)
    return dict(
        B=B,
        H=H,
        S=S,
        Dh=Dh,
        K=K,
        swa_window=swa_window,
        q=q,
        k_local=k_local,
        v_local=v_local,
        gathered=gathered,
        sink=sink_t,
        local_mask=local_mask,
        sparse_mask=sparse_mask,
        scale=1.0 / math.sqrt(Dh),
    )


# ---------------------------------------------------------------------------
# G22 — `eager_v4_attention` matches the inline pre-P24 reference
# ---------------------------------------------------------------------------


class TestG22EagerV4Attention:
    """``eager_v4_attention`` reproduces the pre-P24 inline math bit-for-bit."""

    @pytest.mark.parametrize("sink_on", [True, False], ids=["sink_on", "sink_off"])
    def test_dense_with_pre_built_mask_matches(self, sink_on: bool):
        """Dense path (cr=0): caller passes pre-built SWA-causal mask."""
        toy = _toy_inputs(swa_window=0, sink=sink_on)

        def reference(q, k_local, v_local, sink):
            return _pre_p24_attention_forward(
                q=q,
                k=k_local,
                v=v_local,
                sink=sink,
                additive_mask=toy["local_mask"],
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
                num_heads=toy["H"],
            )

        def candidate(q, k_local, v_local, sink):
            return eager_v4_attention(
                q,
                k_local,
                v_local,
                sink=sink,
                swa_window=0,
                additive_mask=toy["local_mask"],
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        leaf_inputs = {"q": toy["q"], "k_local": toy["k_local"], "v_local": toy["v_local"]}
        if toy["sink"] is not None:
            leaf_inputs["sink"] = toy["sink"]
        else:
            leaf_inputs["sink"] = None

        compare_fwd_bwd(
            reference=reference,
            candidate=candidate,
            inputs=leaf_inputs,
            fwd_tol=FP32_TOL,
            bwd_tol=FP32_TOL,
            grad_keys=[k for k, v in leaf_inputs.items() if isinstance(v, torch.Tensor) and v.requires_grad],
        )

    @pytest.mark.parametrize("sink_on", [True, False], ids=["sink_on", "sink_off"])
    def test_hca_concatenated_kv_with_full_mask_matches(self, sink_on: bool):
        """HCA path (cr=128): caller pre-concatenates pool keys + full additive mask."""
        toy = _toy_inputs(sink=sink_on)
        B, H, S, Dh = toy["B"], toy["H"], toy["S"], toy["Dh"]
        # Build a tiny "compressed pool" of size P=2 broadcast over heads
        # — the kernel is shape-agnostic to Sk, this just exercises a
        # different Sk than Sq.
        P = 2
        g = torch.Generator(device="cpu").manual_seed(99)
        pool_k = torch.randn(B, H, P, Dh, generator=g, dtype=toy["q"].dtype, requires_grad=True)
        pool_v = torch.randn(B, H, P, Dh, generator=g, dtype=toy["q"].dtype, requires_grad=True)
        # Pool causal mask: pool[s] visible at t iff (s+1)*ratio - 1 <= t.
        # With ratio=4 and S=8: pool[0] visible at t>=3; pool[1] at t>=7.
        ratio = 4
        t = torch.arange(S).unsqueeze(1)  # [S, 1]
        s_end = (torch.arange(P).unsqueeze(0) + 1) * ratio - 1  # [1, P]
        extra_mask = torch.where(s_end <= t, 0.0, float("-inf")).to(toy["q"].dtype)

        k_full = torch.cat([toy["k_local"], pool_k], dim=2)
        v_full = torch.cat([toy["v_local"], pool_v], dim=2)
        full_mask = torch.cat([toy["local_mask"], extra_mask], dim=-1)

        def reference(q, k_full, v_full, sink):
            return _pre_p24_attention_forward(
                q=q,
                k=k_full,
                v=v_full,
                sink=sink,
                additive_mask=full_mask,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
                num_heads=H,
            )

        def candidate(q, k_full, v_full, sink):
            return eager_v4_attention(
                q,
                k_full,
                v_full,
                sink=sink,
                swa_window=0,
                additive_mask=full_mask,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        leaf_inputs = {
            "q": toy["q"],
            "k_full": k_full,
            "v_full": v_full,
            "sink": toy["sink"] if toy["sink"] is not None else None,
        }
        compare_fwd_bwd(
            reference=reference,
            candidate=candidate,
            inputs=leaf_inputs,
            fwd_tol=FP32_TOL,
            bwd_tol=FP32_TOL,
            grad_keys=[k for k, v in leaf_inputs.items() if isinstance(v, torch.Tensor) and v.requires_grad],
        )

    def test_swa_window_built_mask_matches_explicit_mask(self):
        """``swa_window`` path rebuilds a mask bit-identical to the explicit one."""
        toy = _toy_inputs(swa_window=4, sink=True)
        # Explicit-mask path
        explicit_mask = sliding_window_causal_mask(
            toy["S"], toy["swa_window"], device=toy["q"].device, dtype=toy["q"].dtype
        )

        def reference(q, k_local, v_local, sink):
            return eager_v4_attention(
                q,
                k_local,
                v_local,
                sink=sink,
                swa_window=0,
                additive_mask=explicit_mask,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        def candidate(q, k_local, v_local, sink):
            return eager_v4_attention(
                q,
                k_local,
                v_local,
                sink=sink,
                swa_window=toy["swa_window"],
                additive_mask=None,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        leaf_inputs = {
            "q": toy["q"],
            "k_local": toy["k_local"],
            "v_local": toy["v_local"],
            "sink": toy["sink"],
        }
        compare_fwd_bwd(
            reference=reference,
            candidate=candidate,
            inputs=leaf_inputs,
            fwd_tol=FP32_TOL,
            bwd_tol=FP32_TOL,
        )

    def test_swa_window_requires_square_kv_when_no_mask(self):
        """``additive_mask=None`` path needs ``Sq == Sk`` (HCA must pre-build mask)."""
        toy = _toy_inputs()
        B, H, S, Dh = toy["B"], toy["H"], toy["S"], toy["Dh"]
        # Sk != Sq — the kernel can't infer a sensible mask without
        # the caller supplying one. Should raise.
        k_extra = torch.randn(B, H, S + 2, Dh, dtype=toy["q"].dtype)
        v_extra = torch.randn(B, H, S + 2, Dh, dtype=toy["q"].dtype)
        with pytest.raises(ValueError, match="additive_mask"):
            eager_v4_attention(
                toy["q"],
                k_extra,
                v_extra,
                sink=toy["sink"],
                swa_window=4,
                additive_mask=None,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )


# ---------------------------------------------------------------------------
# G22 — `eager_v4_csa_attention` matches the inline pre-P24 reference
# ---------------------------------------------------------------------------


class TestG22EagerV4CSAAttention:
    """``eager_v4_csa_attention`` reproduces the pre-P24 inline math bit-for-bit."""

    @pytest.mark.parametrize("sink_on", [True, False], ids=["sink_on", "sink_off"])
    def test_csa_matches_inline_reference(self, sink_on: bool):
        """CSA fused path: joint local SWA + sparse top-K + shared sink."""
        toy = _toy_inputs(swa_window=4, sink=sink_on)

        def reference(q, k_local, v_local, gathered, sink):
            return _pre_p24_csa_attention_forward(
                q=q,
                k_local=k_local,
                v_local=v_local,
                gathered=gathered,
                sink=sink,
                sparse_mask=toy["sparse_mask"],
                local_mask=toy["local_mask"],
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
                num_heads=toy["H"],
            )

        def candidate(q, k_local, v_local, gathered, sink):
            return eager_v4_csa_attention(
                q,
                k_local,
                v_local,
                gathered,
                sink=sink,
                swa_window=toy["swa_window"],
                sparse_mask=toy["sparse_mask"],
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        leaf_inputs = {
            "q": toy["q"],
            "k_local": toy["k_local"],
            "v_local": toy["v_local"],
            "gathered": toy["gathered"],
            "sink": toy["sink"] if toy["sink"] is not None else None,
        }
        compare_fwd_bwd(
            reference=reference,
            candidate=candidate,
            inputs=leaf_inputs,
            fwd_tol=FP32_TOL,
            bwd_tol=FP32_TOL,
            grad_keys=[k for k, v in leaf_inputs.items() if isinstance(v, torch.Tensor) and v.requires_grad],
        )

    def test_csa_sparse_mask_with_invalid_slots(self):
        """Indexer ``-1`` slots → ``-inf`` ``sparse_mask`` ⇒ joint softmax ignores them."""
        toy = _toy_inputs(swa_window=4, sink=True)
        B, S, K = toy["B"], toy["S"], toy["K"]
        # Mark the first K/2 sparse slots as invalid for every query.
        valid = torch.ones(B, S, K, dtype=torch.bool)
        valid[..., : K // 2] = False
        sparse_mask = torch.where(valid, 0.0, float("-inf")).to(toy["q"].dtype)
        gathered = toy["gathered"] * valid.unsqueeze(-1).to(toy["gathered"].dtype)
        gathered = gathered.detach().clone().requires_grad_(True)

        def reference(q, k_local, v_local, gathered, sink):
            return _pre_p24_csa_attention_forward(
                q=q,
                k_local=k_local,
                v_local=v_local,
                gathered=gathered,
                sink=sink,
                sparse_mask=sparse_mask,
                local_mask=toy["local_mask"],
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
                num_heads=toy["H"],
            )

        def candidate(q, k_local, v_local, gathered, sink):
            return eager_v4_csa_attention(
                q,
                k_local,
                v_local,
                gathered,
                sink=sink,
                swa_window=toy["swa_window"],
                sparse_mask=sparse_mask,
                attn_dropout=0.0,
                training=False,
                scale=toy["scale"],
            )

        leaf_inputs = {
            "q": toy["q"],
            "k_local": toy["k_local"],
            "v_local": toy["v_local"],
            "gathered": gathered,
            "sink": toy["sink"],
        }
        compare_fwd_bwd(
            reference=reference,
            candidate=candidate,
            inputs=leaf_inputs,
            fwd_tol=FP32_TOL,
            bwd_tol=FP32_TOL,
        )


# ---------------------------------------------------------------------------
# G22 — `DeepseekV4Attention` end-to-end matches the inline pre-P24 baseline
# ---------------------------------------------------------------------------


def _make_v4_config_for_module(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_lora_rank: int,
    o_groups: int,
    o_lora_rank: int,
    attn_sink: bool,
    norm_eps: float = 1e-6,
) -> DeepSeekV4TransformerConfig:
    """Minimal V4 config for the module-level safety-net test (CPU, 1L)."""
    return DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=1,
        kv_channels=head_dim,
        qk_pos_emb_head_dim=rotary_dim,
        qk_head_dim=head_dim - rotary_dim,
        v_head_dim=head_dim,
        kv_lora_rank=head_dim,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        q_lora_rank=q_lora_rank,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        attn_sliding_window=0,
        attn_sink=attn_sink,
        compress_ratios=None,
        compress_rope_theta=160000.0,
        layernorm_epsilon=norm_eps,
        norm_epsilon=norm_eps,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


class TestG22ModuleRefactorPreservesNumerics:
    """``DeepseekV4Attention`` end-to-end output is unchanged after P24."""

    def test_dense_module_forward_finite_and_deterministic(self):
        """The dense path produces a finite output and is deterministic."""
        config = _make_v4_config_for_module(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            rotary_dim=8,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=8,
            attn_sink=True,
        )
        rope = DualRoPE(
            rotary_dim=config.qk_pos_emb_head_dim,
            rope_theta=config.rotary_base,
            compress_rope_theta=config.compress_rope_theta,
        )
        attn = DeepseekV4Attention(config, rope=rope, compress_ratio=0, submodules=None)
        attn.eval()

        torch.manual_seed(7)
        B, S = 2, 8
        hidden = torch.randn(B, S, config.hidden_size)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

        out_a = attn(hidden, position_ids)
        out_b = attn(hidden, position_ids)

        assert out_a.shape == (B, S, config.hidden_size)
        assert torch.isfinite(out_a).all()
        # Determinism: dropout is off (config.attention_dropout=0), so
        # back-to-back forward must match bit-for-bit.
        assert torch.equal(out_a, out_b), (
            "DeepseekV4Attention dense forward is no longer deterministic "
            "after the P24 refactor — RNG state must NOT be consumed when "
            "attn_dropout=0."
        )

    def test_dense_module_forward_matches_pre_p24_inline_math(self):
        """End-to-end module output equals the inline pre-P24 attention math.

        This is the strict refactor safety net: the *full*
        ``DeepseekV4Attention.forward`` output for a dense layer must
        equal what the pre-P24 inline math produces given the same
        Q / K / V / mask / sink intermediate tensors.
        """
        config = _make_v4_config_for_module(
            hidden_size=32,
            num_heads=4,
            head_dim=8,
            rotary_dim=4,
            q_lora_rank=16,
            o_groups=2,
            o_lora_rank=8,
            attn_sink=True,
        )
        rope = DualRoPE(
            rotary_dim=config.qk_pos_emb_head_dim,
            rope_theta=config.rotary_base,
            compress_rope_theta=config.compress_rope_theta,
        )
        attn = DeepseekV4Attention(config, rope=rope, compress_ratio=0, submodules=None)
        attn.eval()

        torch.manual_seed(13)
        B, S = 2, 8
        hidden = torch.randn(B, S, config.hidden_size)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

        # Run the module under test: this routes through the new
        # ``_attention_forward`` which delegates to
        # ``eager_v4_attention``.
        out_module = attn(hidden, position_ids)

        # Independently reproduce the dense math via the inline
        # pre-P24 reference. This is the same Q / K / V projections
        # the module computes (we re-use its own internals via the
        # public methods), but the attention kernel itself is the
        # verbatim pre-P24 inline body.
        with torch.no_grad():
            q = attn._apply_q(hidden)  # [B, S, H, D]
            kv = attn._apply_kv(hidden)  # [B, S, 1, D]
            q, kv = attn._apply_rope_q_k(q, kv, position_ids)
            k_h = kv.expand(B, S, attn.num_heads, attn.head_dim)
            v_h = kv.expand(B, S, attn.num_heads, attn.head_dim)
            q_bh = q.transpose(1, 2)
            k_bh = k_h.transpose(1, 2)
            v_bh = v_h.transpose(1, 2)
            local_mask = attn._local_mask(S, device=hidden.device, dtype=hidden.dtype)

            out_bh = _pre_p24_attention_forward(
                q=q_bh,
                k=k_bh,
                v=v_bh,
                sink=attn.attn_sink,
                additive_mask=local_mask,
                attn_dropout=attn.attn_dropout,
                training=attn.training,
                scale=attn._attention_scale(),
                num_heads=attn.num_heads,
            )
            out = out_bh.transpose(1, 2).contiguous().to(hidden.dtype)
            if attn.linear_o_a is not None:
                out_inline = attn._grouped_o_projection(out)
            else:
                out_inline = attn._flat_o_projection(out)

        # bit-equality (no dropout, fp32 throughout — same ops in same
        # order — so the outputs must be identical).
        assert torch.equal(out_module, out_inline), (
            "DeepseekV4Attention dense forward diverged from the pre-P24 "
            "inline reference after the P24 refactor — the extracted "
            "eager_v4_attention math is NOT bit-identical."
        )

    def test_csa_module_forward_matches_pre_p24_inline_math(self):
        """End-to-end CSA module output equals the inline pre-P24 CSA math."""
        config = _make_v4_config_for_module(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            rotary_dim=8,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=8,
            attn_sink=True,
        )
        # CSA-specific knobs (must be set before construction).
        config.index_topk = 2
        config.index_head_dim = 16
        config.index_n_heads = 2

        rope = DualRoPE(
            rotary_dim=config.qk_pos_emb_head_dim,
            rope_theta=config.rotary_base,
            compress_rope_theta=config.compress_rope_theta,
            yarn_factor=1.0,
            original_max_position_embeddings=config.original_max_position_embeddings,
        )
        attn = DeepseekV4Attention(config, rope=rope, compress_ratio=4, submodules=None)
        attn.eval()

        torch.manual_seed(21)
        # ``S`` must be ≥ 4 * compress_ratio so the compressed pool has
        # room for the indexer to pick from; here S=8 -> P=2 -> top-K=2.
        B, S = 2, 8
        hidden = torch.randn(B, S, config.hidden_size)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

        out_module = attn(hidden, position_ids)

        # Inline pre-P24 reference path: re-derive Q / K / V / pool /
        # gather / sparse_mask exactly the way the module does today,
        # then run the verbatim joint-softmax body.
        with torch.no_grad():
            q = attn._apply_q(hidden)
            kv = attn._apply_kv(hidden)
            q, kv = attn._apply_rope_q_k(q, kv, position_ids)
            k_h = kv.expand(B, S, attn.num_heads, attn.head_dim)
            v_h = kv.expand(B, S, attn.num_heads, attn.head_dim)
            q_bh = q.transpose(1, 2)
            k_bh = k_h.transpose(1, 2)
            v_bh = v_h.transpose(1, 2)
            local_mask = attn._local_mask(S, device=hidden.device, dtype=hidden.dtype)

            pool = attn._build_compressed_pool(hidden)
            P = pool.shape[1]
            topk_idxs, _ = attn.indexer(hidden)
            K = topk_idxs.shape[-1]
            valid = topk_idxs >= 0
            safe_idx = topk_idxs.clamp(min=0)
            idx_expand = safe_idx.unsqueeze(-1).expand(B, S, K, attn.head_dim)
            pool_expand = pool.unsqueeze(1).expand(B, S, P, attn.head_dim)
            gathered = torch.gather(pool_expand, dim=2, index=idx_expand)
            gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
            sparse_mask = torch.where(valid, 0.0, float("-inf")).to(hidden.dtype)

            out_bh = _pre_p24_csa_attention_forward(
                q=q_bh,
                k_local=k_bh,
                v_local=v_bh,
                gathered=gathered,
                sink=attn.attn_sink,
                sparse_mask=sparse_mask,
                local_mask=local_mask,
                attn_dropout=attn.attn_dropout,
                training=attn.training,
                scale=attn._attention_scale(),
                num_heads=attn.num_heads,
            )
            out = out_bh.transpose(1, 2).contiguous().to(hidden.dtype)
            if attn.linear_o_a is not None:
                out_inline = attn._grouped_o_projection(out)
            else:
                out_inline = attn._flat_o_projection(out)

        assert torch.equal(out_module, out_inline), (
            "DeepseekV4Attention CSA forward diverged from the pre-P24 "
            "inline reference after the P24 refactor — the extracted "
            "eager_v4_csa_attention math is NOT bit-identical."
        )
