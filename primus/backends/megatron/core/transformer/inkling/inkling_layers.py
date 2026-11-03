###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling building blocks: RMSNorm, the causal short convolution and the
relative-bias / log-scaled attention. Tensors are sequence-first ``[s, b, h]``.

Tensor parallelism splits the attention heads (q/k/v/r column-parallel, o
row-parallel); the k/v short convolutions are depthwise, so they split with their
channels (TP axis 0 in the distributed checkpoint). Per-head parameters that every
rank applies to its own heads (q/k norms, the relative-logit table) are replicated
and their gradients summed over TP through ``copy_to_tensor_model_parallel_region``.

Precision: under bf16 / fp16 training ``Float16Module`` stores every parameter in
the training dtype (Megatron has no keep-in-fp32 mechanism for parameters); the
optimizer's fp32 main parameters hold the master copy. The short convolutions,
the norms and the softmax compute in fp32 (HF keeps the sconvs in fp32).
"""

import hashlib

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import divide

from primus.backends.megatron.core.transformer.inkling.inkling_relpos_attention import (
    relpos_flash_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (
    fused_rms_norm,
)

# Query rows per block of the eager attention. Without autograd (eval, scoring) this bounds
# the live score / bias memory to [b, n, ATTN_QUERY_CHUNK, keys] (keys = q1 - k0: at most
# window + ATTN_QUERY_CHUNK - 1 on sliding layers, up to the sequence length on global ones). In training every block's probabilities stay saved for backward, so the global layers
# still keep O(b * n * s * s) activations unless core-attention recompute is on.
ATTN_QUERY_CHUNK = 1024


# Hex digits of the SHA-256 digest kept as the seed: 15 digits = 60 bits, which fits
# torch.Generator.manual_seed (it takes any 64-bit value) and leaves collisions negligible. The
# value is fixed because changing it changes every Inkling init.
INIT_SEED_HEX_DIGITS = 15


def init_seed(base, layer, slot, index=0):
    """A deterministic seed for (base seed, 0-based layer, parameter slot, expert index): a SHA-256
    digest truncated to INIT_SEED_HEX_DIGITS (15 hex digits, 60 bits) so it fits
    torch.Generator.manual_seed; collisions are negligible."""
    digest = hashlib.sha256(f"inkling/{base}/{layer}/{slot}/{index}".encode()).hexdigest()
    return int(digest[:INIT_SEED_HEX_DIGITS], 16)


def seeded_normal(shape, std, seed):
    """A layout-independent init: the same values on every rank and layout (CPU generator)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=g) * std


def _tp_size(tp_group):
    return 1 if tp_group is None else tp_group.size()


def _replicated_for_heads(param, tp_group):
    """A replicated parameter applied to this rank's heads only: identity forward, grad all-reduce."""
    if _tp_size(tp_group) == 1:
        return param
    return copy_to_tensor_model_parallel_region(param, group=tp_group)


class InklingRMSNorm(torch.nn.RMSNorm):
    """torch's RMSNorm parameters (Megatron's ``WrappedTorchNorm`` RMSNorm) computed by the
    shared Primus RMSNorm kernel (``fused_rms_norm``, as ``LocalRMSNorm``) with the HF
    rounding: normalize in fp32, cast back to the input dtype (``mid_cast=True``), then scale
    by ``weight``. With ``weight`` in the input dtype, the product of two bf16 / fp16 values is
    exact in fp32, so the result matches HF's ``weight * h.to(dtype)`` up to the fp32 order of
    the mean reduction. Off CUDA (or with ``PRIMUS_RMSNORM_TRITON=0``) the eager reference runs.

    ``config`` is accepted (and unused) because ``TransformerBlock`` builds its final
    norm as ``layer_norm(config=, hidden_size=, eps=)``. ``tp_group`` marks a norm
    that is replicated across TP but applied to this rank's heads only (the per-head
    q / k norms): its gradient is then summed over TP.
    """

    def __init__(self, hidden_size, eps=1e-6, config=None, tp_group=None):
        del config  # see the class docstring
        super().__init__(hidden_size, eps=eps, elementwise_affine=True)
        self.tp_group = tp_group

    def forward(self, x):
        weight = _replicated_for_heads(self.weight, self.tp_group)
        return fused_rms_norm(x, weight, eps=self.eps, mid_cast=True)


SCONV_IMPLS = ("auto", "eager", "causal_conv1d")
"""``inkling_sconv_impl`` values. ``causal_conv1d`` requires Dao's ``causal_conv1d_fn`` to import
(else the layer refuses to build) and uses it for every input the kernel supports: a GPU input,
kernel size 2..4, sequence length >= 2 and a channel count divisible by 8; other inputs run the
eager reference. ``auto`` is the same, but runs eager also when the kernel does not import.
``eager`` always runs the fp32 ``F.conv1d`` reference. Inkling's short convs (kernel 4, 8-aligned
channels) meet the shape conditions."""

_CAUSAL_CONV1D = None  # the imported causal_conv1d_fn, False when it does not import


def causal_conv1d_kernel():
    """Dao's ``causal_conv1d_fn`` (ROCm build in the Primus image), or ``None`` if it does not import."""
    global _CAUSAL_CONV1D
    if _CAUSAL_CONV1D is None:
        try:
            from causal_conv1d import causal_conv1d_fn

            _CAUSAL_CONV1D = causal_conv1d_fn
        except Exception:  # noqa: BLE001  (any import failure means: no fused kernel)
            _CAUSAL_CONV1D = False
    return _CAUSAL_CONV1D or None


def eager_causal_conv1d(x, weight, seq_idx=None):
    """Reference depthwise causal conv: ``x [b, c, s]`` (fp32), ``weight [c, k]`` -> ``[b, c, s]``.

    ``seq_idx [b, s]`` (int, optional) is the document id of each position (packed sequences): a
    position sees only earlier positions of its own document, as ``causal_conv1d_fn`` with
    ``seq_idx``. Without it, this is the plain ``F.conv1d`` with left padding."""
    k, s = weight.shape[-1], x.shape[-1]
    if seq_idx is None:
        return F.conv1d(x, weight.unsqueeze(1), padding=k - 1, groups=x.shape[1])[..., :s]
    y = x * weight[:, k - 1 : k]
    for d in range(1, min(k, s)):  # tap k-1-d reads the input d steps back
        same = (seq_idx[:, d:] == seq_idx[:, :-d]).unsqueeze(1)  # [b, 1, s - d]
        shifted = F.pad(x[..., :-d] * same, (d, 0))
        y = y + shifted * weight[:, k - 1 - d : k - d]
    return y


class InklingShortConv(torch.nn.Module):
    """Causal depthwise conv over the sequence plus a residual, computed in fp32.

    ``weight`` is ``[channels, kernel]`` (the checkpoint's ``[c, 1, k]`` squeezed) and
    starts at zero (the identity map), as in HF. ``tp_sharded`` marks a convolution
    whose channels are split over TP (the k / v convs): its distributed-checkpoint
    tensor is then TP-sharded along axis 0.

    ``impl`` (``inkling_sconv_impl``, see ``SCONV_IMPLS``) picks Dao's fused
    ``causal_conv1d_fn`` (forward and backward, fp32, channel-last view of the
    ``[s, b, c]`` input, no copy) or the eager ``F.conv1d`` reference; the two are
    numerically close (within test tolerances), not bit-identical. The model refuses packed
    sequences, so the layer takes no document ids; ``eager_causal_conv1d`` and the fused kernel
    both accept ``seq_idx`` for when packing is enabled.
    """

    def __init__(self, channels, kernel_size, tp_sharded=False, tp_group=None, impl="eager"):
        super().__init__()
        if impl not in SCONV_IMPLS:
            raise ValueError(f"inkling_sconv_impl={impl!r} must be one of {SCONV_IMPLS}")
        if impl == "causal_conv1d" and causal_conv1d_kernel() is None:
            raise ImportError(
                "inkling_sconv_impl=causal_conv1d needs the causal_conv1d package (causal_conv1d_fn), "
                "which does not import; set inkling_sconv_impl: auto (fused when available) or eager"
            )
        self.kernel_size = kernel_size
        self.tp_sharded = tp_sharded
        self.tp_group = tp_group
        self.impl = impl
        self.weight = torch.nn.Parameter(torch.zeros(channels, kernel_size))
        if tp_sharded:
            tensor_parallel.set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def _use_kernel(self, h):
        if self.impl == "eager" or not h.is_cuda:
            return False
        # causal_conv1d_fn supports widths 2..4; its channel-last backward needs s >= 2 and
        # 8-aligned strides (channels and batch * channels).
        # (impl causal_conv1d: __init__ refused a missing kernel, so the import check holds.)
        ok = 2 <= self.kernel_size <= 4 and h.shape[0] >= 2 and h.shape[-1] % 8 == 0
        return ok and causal_conv1d_kernel() is not None

    def forward(self, x):  # x: [s, b, c]
        dtype = x.dtype
        h = x.float().contiguous()
        w = self.weight.float()
        xt = h.permute(1, 2, 0)  # [b, c, s]: a channel-last view (stride 1 on c)
        if self._use_kernel(h):
            y = causal_conv1d_kernel()(xt, w, None, activation=None)
        else:
            y = eager_causal_conv1d(xt, w)
        return (y.permute(2, 0, 1) + h).to(dtype)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0} if self.tp_sharded else {},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


def relative_bias(rel_logits, q_start, k_start):
    """Bias of queries ``q_start + i`` over keys ``k_start ..`` (up to the last query) from
    ``rel_logits[..., i, d]`` (the logit of distance ``d = q - k`` for ``0 <= d < E``).

    Distances outside ``[0, E)`` get 0. A pad-and-reshape ("skew") of the flipped
    table: no ``[.., s, s]`` index tensor is built.
    """
    *lead, c, E = rel_logits.shape
    z = torch.cat([rel_logits.flip(-1), rel_logits.new_zeros(*lead, c, c)], dim=-1)  # [.., c, E + c]
    y = z.reshape(*lead, c * (E + c))[..., : c * (E + c - 1)].reshape(*lead, c, E + c - 1)
    lo = q_start - E + 1  # key index of column 0 of y
    if lo > k_start:
        return torch.cat([y.new_zeros(*lead, c, lo - k_start), y], dim=-1)
    return y[..., k_start - lo :]


def add_relative_bias_(scores, rel_logits, q_start, k_start):
    """``scores += relative_bias(rel_logits, q_start, k_start)``, in place, without building the
    zero block for keys more than ``E - 1`` before the first query (a global layer's long prefix):
    only the last key columns, the ones the table can reach, are added to."""
    E = rel_logits.shape[-1]
    first = max(k_start, q_start - E + 1)  # first key the table reaches
    scores[..., first - k_start :] += relative_bias(rel_logits, q_start, first)
    return scores


class InklingAttention(Attention):
    """Inkling self-attention: per-head RMS-normed q/k (scale 1/d), short convs on
    k and v, a learned relative-position bias, sliding window on local layers and
    a position-dependent log scaling of q and the bias on global layers.

    Built by Megatron's :class:`Attention` constructor (process groups, recompute
    and offload flags, the per-partition head bookkeeping), with identity
    placeholders for its fused qkv / core-attention / output slots: Inkling has
    separate q/k/v/r projections, a per-layer head layout (sliding vs global) and a
    relative-logit bias that Megatron's core-attention modules do not take, so those
    modules and the per-partition head counts are set here after the base.

    New math: the relative-logit bias and the log scaling are not in Megatron's core
    attention. The eager core attention is computed in query blocks; a sliding layer
    only scores the keys inside its window.
    """

    def __init__(
        self,
        config,
        layer_number,
        inkling_layer_idx,
        is_sliding,
        pg_collection=None,
        sliding_window=None,
        swa_num_attention_heads=None,
    ):
        submodules = SelfAttentionSubmodules(
            linear_qkv=IdentityOp, core_attention=IdentityOp, linear_proj=IdentityOp
        )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=pg_collection,
        )
        self.is_sliding = is_sliding
        tp_group = self.tp_group
        tp = _tp_size(tp_group)
        if is_sliding:
            # sliding_window / swa_num_attention_heads: the MTP sliding depths' own values
            # (HF mtp_local_extent / mtp_swa_num_attention_heads); None = the trunk's
            nh, nkv, hd = (
                swa_num_attention_heads or config.inkling_swa_num_attention_heads,
                config.inkling_swa_num_query_groups,
                config.inkling_swa_kv_channels,
            )
            self.window = sliding_window or config.inkling_sliding_window
            self.rel_extent = self.window
        else:
            nh, nkv, hd = config.num_attention_heads, config.num_query_groups, config.kv_channels
            self.window = None
            self.rel_extent = config.inkling_rel_extent
        self.query_projection_size = nh * hd
        self.kv_projection_size = nkv * hd
        self.key_hidden_size = self.val_hidden_size = hd
        self.world_size = tp
        self.num_attention_heads_per_partition = divide(nh, tp)
        self.num_query_groups_per_partition = divide(nkv, tp)
        self.hidden_size_per_attention_head = hd
        self.d_rel = config.inkling_d_rel
        self.scaling = 1.0 / hd
        H = config.hidden_size

        def col(out):
            return tensor_parallel.ColumnParallelLinear(
                H,
                out,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
                tp_group=tp_group,
            )

        self.q_proj = col(nh * hd)
        self.k_proj = col(nkv * hd)
        self.v_proj = col(nkv * hd)
        self.r_proj = col(nh * self.d_rel)
        self.o_proj = tensor_parallel.RowParallelLinear(
            nh * hd,
            H,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        k = config.inkling_sconv_kernel_size
        kv_width = self.num_query_groups_per_partition * hd
        impl = config.inkling_sconv_impl
        self.k_sconv = InklingShortConv(kv_width, k, tp_sharded=True, tp_group=tp_group, impl=impl)
        self.v_sconv = InklingShortConv(kv_width, k, tp_sharded=True, tp_group=tp_group, impl=impl)
        self.q_norm = InklingRMSNorm(hd, config.layernorm_epsilon, tp_group=tp_group)
        self.k_norm = InklingRMSNorm(hd, config.layernorm_epsilon, tp_group=tp_group)
        self.rel_proj = torch.nn.Parameter(torch.empty(self.d_rel, self.rel_extent))
        if config.perform_initialization:
            seed = init_seed(config.inkling_init_seed, inkling_layer_idx, "rel_proj")
            with torch.no_grad():
                self.rel_proj.copy_(seeded_normal(self.rel_proj.shape, config.init_method_std, seed))

    def set_for_recompute_input_layernorm(self):
        """fp8 + layernorm recompute: not used (InklingTransformerConfig refuses fp8)."""

    def get_query_key_value_tensors(self, x, key_value_states=None, *args, **kwargs):
        """q as ``[b, nh, s, d]``, k / v as ``[b, nkv, s, d]`` and the relative-logit input
        ``r`` as ``[b, nh, s, d_rel]`` (q / k RMS-normed, k / v short-convolved).

        nh = attention heads and nkv = query groups, both per TP partition. k / v are never
        repeated to nh heads: ``_core_attention`` views q as ``[b, nkv, nh // nkv, s, d]`` and
        multiplies each group of query heads with its shared k / v head."""
        s, b, _ = x.shape
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)
        r, _ = self.r_proj(x)
        k = self.k_sconv(k)
        v = self.v_sconv(v)
        hd = self.hidden_size_per_attention_head
        nh, nkv = self.num_attention_heads_per_partition, self.num_query_groups_per_partition
        q = self.q_norm(q.view(s, b, nh, hd))
        k = self.k_norm(k.view(s, b, nkv, hd))
        v = v.view(s, b, nkv, hd)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        rel = r.view(s, b, nh, self.d_rel).permute(1, 2, 0, 3)
        return q, k, v, rel

    def _core_attention(self, q, k, v, rel, rel_proj, attention_mask=None):
        """Relative bias, log scaling, masked softmax and the value product: ``[s, b, n * d]``.

        ``attention_mask`` (optional, Megatron's convention: True = masked out) is
        broadcast to ``[b, n, s, s]`` and applied on top of the causal / window mask.
        """
        b, _, s, hd = q.shape
        rel_proj = rel_proj.to(rel.dtype)
        tau = None
        if not self.is_sliding and self.config.inkling_log_scaling_n_floor is not None:
            # Position = index in the sequence (the model refuses reset_position_ids and packed
            # sequences, so every sample starts at position 0 of its row).
            pos = torch.arange(s, device=q.device)
            tau = 1.0 + self.config.inkling_log_scaling_alpha * torch.log(
                ((pos + 1).float() / self.config.inkling_log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau.view(-1, 1)
            q = (q.float() * tau).to(q.dtype)

        nh, nkv = self.num_attention_heads_per_partition, self.num_query_groups_per_partition
        if self.config.inkling_attention_impl == "fused":
            # the Triton kernel adds the banded bias rel_logits[q, q - k] itself: no [s, s] buffer

            rel_logits = rel @ rel_proj  # [b, n, s, E]
            if tau is not None:
                rel_logits = (rel_logits.float() * tau).to(rel_logits.dtype)
            mask = None if attention_mask is None else attention_mask.bool()
            ctx = relpos_flash_attention(q, k, v, rel_logits, self.scaling, self.window, mask)
            return ctx.permute(2, 0, 1, 3).reshape(s, b, nh * hd)
        rep = nh // nkv
        # GQA without repeating k / v: query head h uses kv head h // rep, so the query heads are
        # grouped as [b, nkv, rep, s, ...] and each group's rep * chunk queries are folded into one
        # matmul against that group's k / v (a plain [b, nkv, ...] batched matmul, no copy of k / v).
        q = q.view(b, nkv, rep, s, hd)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 4 and attention_mask.shape[1] == nh and nh > 1:
                mb = attention_mask.shape[0]
                attention_mask = attention_mask.view(mb, nkv, rep, *attention_mask.shape[-2:])
            else:
                attention_mask = attention_mask.unsqueeze(-3)  # broadcast over the rep axis
        dropout = self.config.attention_dropout if self.training else 0.0
        out = []
        for q0 in range(0, s, ATTN_QUERY_CHUNK):
            q1 = min(q0 + ATTN_QUERY_CHUNK, s)
            c = q1 - q0
            k0 = 0 if self.window is None else max(0, q0 - self.window + 1)
            qc = q[:, :, :, q0:q1].reshape(b, nkv, rep * c, hd)
            scores = torch.matmul(qc, k[:, :, k0:q1].transpose(2, 3)) * self.scaling
            scores = scores.view(b, nkv, rep, c, q1 - k0)
            # relative logits of this query chunk only ([b, n, c, E]: logit of distance d = q - k)
            rel_logits = rel[:, :, q0:q1] @ rel_proj
            if tau is not None:
                rel_logits = (rel_logits.float() * tau[q0:q1]).to(rel_logits.dtype)
            rel_logits = rel_logits.view(b, nkv, rep, c, -1)
            scores = add_relative_bias_(scores, rel_logits, q0, k0)
            dist = torch.arange(q0, q1, device=q.device)[:, None] - torch.arange(k0, q1, device=q.device)
            masked = dist < 0
            if self.window is not None:
                masked = masked | (dist >= self.window)
            if attention_mask is not None:
                # a size-1 (broadcast) query or key axis is kept whole, not sliced to nothing
                qs = slice(None) if attention_mask.shape[-2] == 1 else slice(q0, q1)
                ks = slice(None) if attention_mask.shape[-1] == 1 else slice(k0, q1)
                masked = masked | attention_mask[..., qs, ks]
            scores = scores.masked_fill(masked, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if dropout > 0:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    probs = F.dropout(probs, p=dropout)
            ctx = torch.matmul(probs.view(b, nkv, rep * c, q1 - k0), v[:, :, k0:q1])
            out.append(ctx.view(b, nkv, rep, c, hd))
        ctx = torch.cat(out, dim=3) if len(out) > 1 else out[0]  # [b, nkv, rep, s, d]
        return ctx.view(b, nh, s, hd).permute(2, 0, 1, 3).reshape(s, b, nh * hd)

    def forward(self, hidden_states, attention_mask=None, **_kwargs):
        """``[s, b, H] -> ([s, b, H], None)`` (Megatron's ``(output, bias)`` convention)."""
        q, k, v, rel = self.get_query_key_value_tensors(hidden_states)
        rel_proj = _replicated_for_heads(self.rel_proj, self.tp_group)
        if self.checkpoint_core_attention and self.training:
            ctx = tensor_parallel.checkpoint(
                self._core_attention, False, q, k, v, rel, rel_proj, attention_mask
            )
        else:
            ctx = self._core_attention(q, k, v, rel, rel_proj, attention_mask)
        out, _ = self.o_proj(ctx)
        return out, None
