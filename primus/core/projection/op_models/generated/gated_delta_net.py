###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5), training.

Per value head the layer keeps a ``dk x dv`` state ``S`` and, for every token,
applies the gated delta rule (Yang et al., Gated Delta Networks, eq. 10):

    S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,    o_t = S_t q_t

with ``a_t = exp(-exp(A) * softplus(x W_a + dt_bias))`` and
``b_t = sigmoid(x W_b)``.  Training runs the chunkwise form (fla
``chunk_gated_delta_rule``): inside a chunk of ``C`` tokens the delta updates
collapse into a WY representation (``A = tril(diag(b) K K^T)``, ``T = (I + A)^-1``,
``W = T diag(b) K``, ``U = T diag(b) V``), the state is carried chunk to chunk,
and the output is the inter-chunk ``Q S`` plus the masked intra-chunk scores
applied to the corrected values.

Kernels, priced the way the built-in profilers price theirs:

* ``in_proj_qkvz`` / ``in_proj_ba`` / ``out_proj`` -- GEMMs on the GEMM backend;
* short causal depthwise conv over q, k, v, the q/k L2 norms, the a/b gates and
  the gated RMSNorm -- HBM streams at the activation bandwidth fraction;
* the chunk kernels as per-workgroup tile programs on one CU times waves
  (:meth:`OpContext.fused_tiles`): the intra-chunk WY kernel and the output
  kernel parallelise over (sequence, head, chunk); the state recurrence is
  sequential over chunks and parallel over (sequence, head, ``dv / BV``) --
  which is what makes it latency-bound at small batch.  Each kernel takes the
  larger of its compute time and its HBM time (the per-chunk states it stores).

Every 4th layer stays full attention (``linear_attention_freq``); the built-in
attention profiler prices those.
"""

from __future__ import annotations

import math

from primus.core.projection.op_models.base import (
    SLOT_ATTENTION,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
    interleave_fraction,
)
from primus.core.projection.op_models.registry import register_op_model

_CHUNK = 64  # fla chunk_gated_delta_rule BT
_BV = 64  # value-dim block of the state recurrence kernel
_ACT = 2.0  # bf16 activations


@register_op_model
class GatedDeltaNet(OpModel):
    name = "gated_delta_net"
    slot = SLOT_ATTENTION
    claims = {"experimental_attention_variant": ("gated_delta_net",)}
    consumes = (
        "linear_attention_freq",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    )
    references = (
        "Yang, Kautz, Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv:2412.06464",
        "Yang et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. arXiv:2406.06484",
        "HF transformers Qwen3NextGatedDeltaNet (in_proj_qkvz, in_proj_ba, conv1d, norm, out_proj)",
        "fla-org/flash-linear-attention ops/gated_delta_rule/chunk.py",
        "InferaSim LinearAttentionProfiler (infera/projection/core/projection/module_profilers/linear_attention.py)",
    )

    def layer_fraction(self, model_config) -> float:
        return interleave_fraction(
            model_config, int(cfg_value(model_config, "linear_attention_freq", 4) or 4)
        )

    # ---- geometry ----
    @staticmethod
    def _geom(ctx: OpContext):
        hidden = int(ctx.mc.hidden_size)
        hk = int(ctx.value("linear_num_key_heads", ctx.mc.num_attention_heads))
        hv = int(ctx.value("linear_num_value_heads", hk))
        dk = int(ctx.value("linear_key_head_dim", 128))
        dv = int(ctx.value("linear_value_head_dim", 128))
        kernel = int(ctx.value("linear_conv_kernel_dim", 4) or 4)
        return hidden, hk, hv, dk, dv, kernel

    def _shape(self, ctx: OpContext):
        hidden, hk, hv, dk, dv, kernel = self._geom(ctx)
        hk_r, hv_r = ctx.heads_per_rank(hk), ctx.heads_per_rank(hv)
        seq = ctx.seq_per_cp
        n_chunks = max(1, math.ceil(seq / _CHUNK))
        tok_r = ctx.batch_size * seq  # tokens on this rank's heads
        return hidden, hk, hv, dk, dv, kernel, hk_r, hv_r, seq, n_chunks, tok_r

    # ---- cost ----
    def cost(self, ctx: OpContext) -> OpCost:
        hidden, hk, hv, dk, dv, kernel, hk_r, hv_r, seq, n_chunks, tok_r = self._shape(ctx)
        m = ctx.tokens
        pk, pv = hk * dk, hv * dv
        c, bv = _CHUNK, min(_BV, dv)
        bpe_w = 1 if ctx.gemm_dtype in ("fp8", "mx8") else 2
        parts = {}
        flops_f = flops_b = bytes_f = bytes_b = 0.0

        # Projections. M carries the TP split, as the attention profiler's do.
        for tag, n, k in (
            ("in_proj_qkvz", 2 * pk + 2 * pv, hidden),
            ("in_proj_ba", 2 * hv, hidden),
            ("out_proj", hidden, pv),
        ):
            f, b = ctx.linear(m, n, k)
            parts[tag] = (f, b)
            fl = 2.0 * m * n * k
            by = (m * k + m * n) * _ACT + (n * k * bpe_w) / ctx.tp
            flops_f += fl
            flops_b += 2 * fl
            bytes_f += by
            bytes_b += 2 * by

        # Short causal depthwise conv + SiLU over q, k, v (per-rank channels).
        conv_ch = 2 * hk_r * dk + hv_r * dv
        cf = 2.0 * tok_r * conv_ch * kernel
        cb = 2 * tok_r * conv_ch * _ACT
        parts["short_conv"] = (ctx.stream_ms(cb), ctx.stream_ms(2 * cb))
        flops_f += cf
        flops_b += 2 * cf
        bytes_f += cb
        bytes_b += 2 * cb

        # q/k L2 norm, a = exp(-exp(A) softplus(.)), b = sigmoid(.): read + write.
        nb = 2 * tok_r * (2 * hk_r * dk + 2 * hv_r) * _ACT
        parts["norms_gates"] = (ctx.stream_ms(nb), ctx.stream_ms(2 * nb))
        flops_f += 4.0 * tok_r * 2 * hk_r * dk
        flops_b += 8.0 * tok_r * 2 * hk_r * dk
        bytes_f += nb
        bytes_b += 2 * nb

        # Chunk kernels. GQA-style: k heads are broadcast to hv value heads.
        n_seq_heads = ctx.batch_size * hv_r
        wg_chunk = n_seq_heads * n_chunks
        intra = [(c, c, dk), (c, c, c), (c, dk, c), (c, dv, c)]  # K K^T, (I+A)^-1, W, U
        # The recurrence loops over the chunks inside one workgroup: fold the
        # loop into the matmuls, as FAv3 folds its KV sweep.
        L = n_chunks * c
        state = [(L, bv, dk), (dk, bv, L)]  # W S, K^T v_new
        out = [(c, c, dk), (c, dv, dk), (c, dv, c)]  # Q K^T, Q S, scores . v_new
        qkv_bytes = tok_r * hv_r * (2 * dk + dv) * _ACT
        h_bytes = n_seq_heads * n_chunks * dk * dv * _ACT  # per-chunk states stored for the output / bwd
        wu_bytes = tok_r * hv_r * (dk + dv + c) * _ACT  # W, U, T

        def kern(tiles, n_wg, nbytes):
            return max(ctx.fused_tiles(tiles, n_wg), ctx.stream_ms(nbytes))

        f_intra = kern(intra, wg_chunk, qkv_bytes + wu_bytes)
        f_state = kern(state, n_seq_heads * (dv // bv), wu_bytes + h_bytes)
        f_out = kern(out, wg_chunk, qkv_bytes + h_bytes + tok_r * hv_r * dv * _ACT)
        # Backward (fla chunk_gated_delta_rule_bwd): recompute the states, the
        # reverse dh recurrence, dq/dk/dv/dg from the stored chunks, and the WY
        # backward through T.
        bwd_dh = [(dk, bv, L), (dk, bv, L)]  # Q^T dO, W^T dv
        bwd_local = [(c, dv, c), (c, dv, dk), (c, dk, dv), (c, dk, dv), (c, c, dv), (c, dk, c), (c, dk, c)]
        bwd_wy = [(c, c, dk), (c, dk, c), (c, dv, c), (c, c, dv), (c, c, dk), (c, dk, c), (c, c, c)]
        b_state = f_state + kern(bwd_dh, n_seq_heads * (dv // bv), wu_bytes + 2 * h_bytes)
        b_local = kern(bwd_local, wg_chunk, 2 * qkv_bytes + h_bytes + 2 * tok_r * hv_r * dv * _ACT)
        b_wy = kern(bwd_wy, wg_chunk, 2 * (qkv_bytes + wu_bytes))
        parts["chunk_intra_wy"] = (f_intra, b_wy)
        parts["chunk_state"] = (f_state, b_state)
        parts["chunk_output"] = (f_out, b_local)

        def tile_flops(tiles, n_wg):
            return n_wg * sum(2.0 * a * b_ * k for a, b_, k in tiles)

        flops_f += (
            tile_flops(intra, wg_chunk)
            + tile_flops(state, n_seq_heads * (dv // bv))
            + tile_flops(out, wg_chunk)
        )
        flops_b += (
            tile_flops(state, n_seq_heads * (dv // bv))
            + tile_flops(bwd_dh, n_seq_heads * (dv // bv))
            + tile_flops(bwd_local, wg_chunk)
            + tile_flops(bwd_wy, wg_chunk)
        )
        bytes_f += 2 * qkv_bytes + wu_bytes + 2 * h_bytes + tok_r * hv_r * dv * _ACT
        bytes_b += 5 * qkv_bytes + 3 * wu_bytes + 4 * h_bytes + 2 * tok_r * hv_r * dv * _ACT

        # Gated RMSNorm: RMSNorm(o) * SiLU(z), reads o and z, writes the result.
        gb = 3 * tok_r * hv_r * dv * _ACT
        parts["gated_norm"] = (ctx.stream_ms(gb), ctx.stream_ms(2 * gb))
        flops_f += 6.0 * tok_r * hv_r * dv
        flops_b += 12.0 * tok_r * hv_r * dv
        bytes_f += gb
        bytes_b += 2 * gb

        fwd = sum(v[0] for v in parts.values())
        bwd = sum(v[1] for v in parts.values())

        params = hidden * (2 * pk + 2 * pv + 2 * hv) + pv * hidden + (2 * pk + pv) * kernel + 2 * hv + dv
        # Saved for backward: layer input, qkvz/ba, conv output, a/b, o, norm output.
        act_width = hidden + (2 * pk + 2 * pv + 2 * hv) + (2 * pk + pv) + 2 * hv + 2 * pv
        return OpCost(
            fwd_ms=fwd,
            bwd_ms=bwd,
            fwd_flops=flops_f,
            bwd_flops=flops_b,
            fwd_bytes=bytes_f,
            bwd_bytes=bytes_b,
            activation_bytes=m * act_width * _ACT,
            num_params=int(params),
            breakdown=parts,
        )

    def reference_flops(self, ctx: OpContext):
        """Chunkwise gated delta rule, counted from the algorithm (arXiv:2406.06484 sec. 3)."""
        hidden, hk, hv, dk, dv, kernel, hk_r, hv_r, seq, n_chunks, tok_r = self._shape(ctx)
        m, c = ctx.tokens, _CHUNK
        pk, pv = hk * dk, hv * dv
        proj = 2.0 * m * hidden * (2 * pk + 2 * pv + 2 * hv + pv)
        # Per token and value head: WY (K K^T row, T row via forward substitution,
        # W and U rows), the state step (W S and K^T v rank update) and the output
        # (Q S plus the masked intra-chunk scores applied to V).
        wy = 2 * c * dk + c * c + 2 * c * dk + 2 * c * dv
        state = 2 * dk * dv + 2 * dk * dv
        outp = 2 * dk * dv + 2 * c * dk + 2 * c * dv
        core = tok_r * hv_r * (wy + state + outp)
        elem = 2.0 * tok_r * (2 * hk_r * dk + hv_r * dv) * kernel + 14.0 * tok_r * hv_r * dv
        fwd = proj + core + elem
        # Backward: two matmuls per forward one for the projections; the core
        # recomputes the states and differentiates every product.
        bwd = 2 * proj + 2.2 * core + 2 * elem
        return fwd, bwd
