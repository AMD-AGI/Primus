###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 self-attention.

Differences from upstream ``SelfAttention``:

* ``q_norm`` / ``k_norm``: RMSNorm over the full projection width
  (``num_heads * head_dim``). Under tensor parallelism every rank holds a
  contiguous slice of the heads, so the weight is TP-sharded along dim 0 and the
  sum of squares is all-reduced across the TP group. The names deliberately do
  not contain ``q_layernorm`` / ``k_layernorm``: upstream sums those grads across
  TP, which is right for replicated per-head norms and wrong for sharded ones.
  ``FullWidthRMSNorm.sharded_state_dict`` marks the weight TP-sharded on dim 0,
  so distributed (``torch_dist``) checkpoints save and reshard it across TP sizes.
  The native HF converter writes TP=PP=EP=1 legacy ``torch`` checkpoints only; to load
  converted weights at TP > 1, convert them to ``torch_dist`` first.
* per-layer RoPE: the model hands every layer a stacked ``[2, s, 1, 1, d]``
  embedding (index 0 = plain RoPE, index 1 = YaRN). Full-attention layers take
  YaRN; sliding layers take plain RoPE (``olmo3_rope_layout: per_type``) or
  YaRN (``yarn_all``). With YaRN off (``olmo3_yarn_rotary_scaling_factor: 1``) every layer,
  full-attention ones included, uses plain RoPE (index 0).
* per-layer YaRN attention factor: HF multiplies cos and sin by ``m`` on the YaRN
  layers, which scales q and k by ``m`` and the logits by ``m**2``; it is applied
  here as ``softmax_scale = m**2 * base_scale`` (``base_scale``: ``config.softmax_scale``
  if set, else ``1 / sqrt(head_dim)``).
* per-layer window: HF ``sliding_window`` W (``config.olmo3_sliding_window``) lets a
  query see keys ``q - W + 1 .. q``, i.e. Megatron / TE ``window_size=(W - 1, 0)``, which
  is what ``config.window_size`` holds; full-attention layers get no window.
* ``post_attention_layernorm`` applied to the attention output (post-norm).
"""

import dataclasses
import math

import torch
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    is_layer_window_attention,
)
from megatron.core.utils import get_pg_size, make_tp_sharded_tensor_for_checkpoint
from torch import Tensor

from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
    OLMO3_ROPE_LAYOUTS,
)

__all__ = [
    "Olmo3SelfAttention",
    "FullWidthRMSNorm",
    "olmo3_layer_is_sliding",
    "olmo3_layer_uses_yarn",
]


def olmo3_layer_is_sliding(config, layer_number: int) -> bool:
    """True if the (1-indexed, global) layer uses sliding-window attention."""
    return bool(is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number))


def olmo3_layer_uses_yarn(config, sliding: bool) -> bool:
    """True if a layer gets the YaRN RoPE table and attention factor: YaRN is on (factor > 1) and
    the layer is full attention, or ``olmo3_rope_layout`` is ``yarn_all``. ``sliding`` is the
    layer's :func:`olmo3_layer_is_sliding` result (computed once per layer by the caller)."""
    return bool(config.olmo3_yarn_enabled) and (not sliding or config.olmo3_rope_layout == "yarn_all")


class _FullWidthRMSNormFn(torch.autograd.Function):
    """``(weight * (x * rstd)).to(x.dtype)`` with ``rstd = rsqrt(sum_all_ranks(x**2) / full + eps)``.

    One op for autograd: it saves the input (in its own dtype), the weight and ``rstd`` (fp32, or the input
    dtype if wider; one value per token) and recomputes the rest in backward. An eager chain of fp32 ops would
    keep about three fp32 copies of the projection (about 6x a bf16 input) per call.
    Backward: ``dx = rstd * dy - x * rstd**3 * sum_all_ranks(dy * x) / full`` with
    ``dy = grad * weight``; the row sum spans every TP rank's heads, so it is all-reduced, as the
    sum of squares is in forward.
    """

    # x: [..., heads_local, head_dim], used as given (q / k are split views of the fused qkv
    # output, so a reshape to [..., width] would copy them); weight: [heads_local, head_dim] view.
    _ROW = (-2, -1)

    @staticmethod
    def forward(ctx, x, weight, eps, full_width, group):
        ctype = torch.promote_types(x.dtype, torch.float32)  # fp32 statistics (fp64 stays fp64)
        h = x.to(ctype)
        ss = h.pow(2).sum(_FullWidthRMSNormFn._ROW, keepdim=True)
        if group is not None:
            torch.distributed.all_reduce(ss, group=group)
        rstd = torch.rsqrt(ss / full_width + eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.full_width, ctx.group = full_width, group
        return (weight * (h * rstd)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, weight, rstd = ctx.saved_tensors
        h = x.to(rstd.dtype)
        g = grad.to(rstd.dtype)
        y = h * rstd
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = (g * y).reshape(-1, *y.shape[-2:]).sum(0).to(weight.dtype)
        grad_x = None
        if ctx.needs_input_grad[0]:
            dy = g * weight.to(rstd.dtype)
            dot = (dy * h).sum(_FullWidthRMSNormFn._ROW, keepdim=True)
            if ctx.group is not None:
                torch.distributed.all_reduce(dot, group=ctx.group)
            grad_x = (rstd * dy - h * rstd.pow(3) * (dot / ctx.full_width)).to(x.dtype)
        return grad_x, grad_weight, None, None, None


class FullWidthRMSNorm(torch.nn.Module):
    """RMSNorm over ``[..., heads, head_dim]`` with the heads sharded over TP.

    Matches HF ``Olmo3RMSNorm`` applied to the un-split projection: statistics
    in fp32, ``(weight * normed).to(input_dtype)``.

    New math: one RMS statistic spans all heads, which are sharded over TP,
    so the sum of squares is all-reduced across TP. TENorm / RMSNorm / LayerNorm
    normalise a rank-local last dim only.
    """

    def __init__(
        self,
        local_width: int,
        full_width: int,
        eps: float,
        config=None,
        tp_group=None,
    ):
        super().__init__()
        self.local_width = local_width
        self.full_width = full_width
        self.eps = eps
        self.tp_group = tp_group
        # The group the sum of squares is all-reduced over (None: TP = 1), fixed at build time.
        self._reduce_group = tp_group if tp_group is not None and get_pg_size(tp_group) > 1 else None
        dtype = getattr(config, "params_dtype", torch.float32)
        use_cpu = config is None or getattr(config, "use_cpu_initialization", False)
        device = "cpu" if use_cpu or not torch.cuda.is_available() else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.ones(local_width, dtype=dtype, device=device))
        if self._reduce_group is not None:
            set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def _weight_view(self, x: Tensor) -> Tensor:
        # x: [s, b, heads_local, head_dim]; normalised in place of its layout (no reshape copy)
        heads, head_dim = x.shape[-2:]
        if heads * head_dim != self.local_width:
            raise ValueError(f"FullWidthRMSNorm: got width {heads * head_dim}, expected {self.local_width}")
        return self.weight.view(heads, head_dim)

    def forward(self, x: Tensor) -> Tensor:
        weight = self._weight_view(x)
        return _FullWidthRMSNormFn.apply(x, weight, self.eps, self.full_width, self._reduce_group)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        """TP axis map ``{'weight': 0}``: each rank saves its own heads' slice of the weight."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        key = f"{prefix}weight"
        return {
            key: make_tp_sharded_tensor_for_checkpoint(
                self.weight,
                key,
                tp_axis=0,
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        }

    def extra_repr(self):
        return f"local={self.local_width}, full={self.full_width}, eps={self.eps}"


_SCALED_CORE_ATTENTION = {}


def _with_softmax_scale(core_attention, softmax_scale):
    """``core_attention`` (class or ModuleSpec) with ``softmax_scale`` fixed, as a class or a
    ModuleSpec again: a generated subclass whose ``__init__`` replaces the ``softmax_scale``
    argument. Not a function: ``build_module`` returns a function unchanged (it would not be
    called), so the result must stay a class / spec whether upstream calls
    ``submodules.core_attention(...)`` directly (some Megatron versions instantiate it that way) or
    through ``build_module``. One class
    per (class, scale), so layers with the same scale share it."""
    if isinstance(core_attention, ModuleSpec):
        return dataclasses.replace(
            core_attention, module=_with_softmax_scale(core_attention.module, softmax_scale)
        )
    if not isinstance(core_attention, type):
        raise TypeError(
            f"OLMo-3 YaRN softmax scale: core_attention must be a class or a ModuleSpec of a class, "
            f"got {core_attention!r}"
        )
    key = (core_attention, float(softmax_scale))
    cls = _SCALED_CORE_ATTENTION.get(key)
    if cls is None:

        def __init__(self, *args, **kwargs):
            kwargs["softmax_scale"] = softmax_scale
            core_attention.__init__(self, *args, **kwargs)

        # a distinct name, so a scaled layer is visible as such in repr / logs and is not looked up
        # as the unscaled parent by qualified name; the scale is also a class attribute.
        # Cached per (class, scale); YaRN layers share one scale, so a model adds few classes.
        name = f"{core_attention.__name__}Olmo3YarnScaled"
        cls = type(
            name,
            (core_attention,),
            {
                "__init__": __init__,
                "__module__": __name__,
                "__qualname__": name,
                "olmo3_softmax_scale": float(softmax_scale),
            },
        )
        _SCALED_CORE_ATTENTION[key] = cls
    return cls


class Olmo3SelfAttention(SelfAttention):
    """Self-attention with OLMo-3's full-width QK-norm, per-layer RoPE/window and post-norm."""

    def __init__(self, config, submodules, layer_number: int, *args, post_norm=None, **kwargs):
        if post_norm is None:
            raise ValueError("OLMo-3 attention needs post_norm (its post_attention_layernorm)")
        self.olmo3_sliding = olmo3_layer_is_sliding(config, layer_number)
        # "yarn_all": one YaRN table (and attention factor) on every layer;
        # "per_type": plain RoPE on sliding layers, YaRN on full layers.
        layout = config.olmo3_rope_layout
        if layout not in OLMO3_ROPE_LAYOUTS:
            # Olmo3TransformerConfig refuses it already; kept so a hand-built config cannot get
            # an implicit layout.
            raise ValueError(f"OLMo-3 olmo3_rope_layout must be one of {OLMO3_ROPE_LAYOUTS}, got {layout!r}")
        # YaRN off (factor 1): every layer uses plain RoPE, as olmo3_layer_plan logs it.
        self.olmo3_use_yarn = olmo3_layer_uses_yarn(config, self.olmo3_sliding)
        # No private config copy: every layer shares ``config``. Upstream core attention (TE and
        # local) already applies ``config.window_size`` = (W - 1, 0) per layer through
        # is_layer_window_attention(window_size, window_attn_skip_freq, layer_number), the same
        # rule as olmo3_layer_is_sliding, so full-attention layers get no window.
        # window_size == (olmo3_sliding_window - 1, 0) is set by Olmo3TransformerConfig and
        # re-checked by olmo3_builder for a supplied config, the only way this layer is built.

        # Upstream default softmax scale: 1/sqrt(head_dim), with head_dim = kv_channels (as Attention
        # sets hidden_size_per_attention_head).
        base_scale = (
            config.softmax_scale if config.softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )
        self.olmo3_softmax_scale = None
        if self.olmo3_use_yarn:
            attention_factor = config.olmo3_yarn_effective_attention_factor
            if attention_factor != 1.0:
                self.olmo3_softmax_scale = base_scale * attention_factor * attention_factor
                # The per-layer scale goes to this layer's core attention only (upstream passes
                # config.softmax_scale); the shared config is not changed.
                submodules = dataclasses.replace(
                    submodules,
                    core_attention=_with_softmax_scale(submodules.core_attention, self.olmo3_softmax_scale),
                )
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        # The scale this layer's core attention uses. Upstream's dynamic-batching kernel
        # (Attention.flash_decode_and_prefill) reads self.softmax_scale and else falls back to
        # head_dim ** -0.5, which would drop the YaRN m**2; forward() refuses that path anyway
        # (it also applies no sliding window), and this keeps the attribute right for any reader.
        self.softmax_scale = self.olmo3_softmax_scale if self.olmo3_softmax_scale is not None else base_scale
        eps = config.layernorm_epsilon
        hd = self.hidden_size_per_attention_head
        if config.num_query_groups < self.world_size:
            # KV heads would be replicated across TP ranks: the cross-rank k statistic and the
            # TP-sharded k_norm weight assume disjoint heads (olmo3_validation refuses it earlier)
            raise ValueError(
                f"OLMo-3 full-width k_norm needs num_query_groups ({config.num_query_groups}) >= "
                f"tensor parallel size ({self.world_size})"
            )
        self.q_norm = FullWidthRMSNorm(
            self.num_attention_heads_per_partition * hd,
            config.num_attention_heads * hd,
            eps,
            config=config,
            tp_group=self.tp_group,
        )
        self.k_norm = FullWidthRMSNorm(
            self.num_query_groups_per_partition * hd,
            config.num_query_groups * hd,
            eps,
            config=config,
            tp_group=self.tp_group,
        )
        self.post_attention_layernorm = post_norm(config=config, hidden_size=config.hidden_size, eps=eps)

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        if not split_qkv or output_gate:
            # Olmo3TransformerConfig refuses fused_single_qkv_rope / attention_output_gate, the
            # settings that make upstream request these; kept as a guard that survives -O.
            raise ValueError("OLMo-3 attention needs split q/k/v and no output gate")
        query, key, value = super().get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=False, split_qkv=True
        )
        # Call the modules (not their autograd function) so module hooks fire, e.g. Megatron DDP's
        # overlap_param_gather wait for the q/k norm weights.
        query, key = self.q_norm(query), self.k_norm(key)
        return query, key, value

    def _select_rotary(self, rotary_pos_emb):
        """Pick this layer's table from the stacked ``[2, s, 1, 1, d]`` (plain, YaRN) tensor
        of ``Olmo3DualRotaryEmbedding``. Any other form would put one table on every layer
        and lose the sliding / full distinction, so it raises."""
        if rotary_pos_emb is None:
            return None
        if not torch.is_tensor(rotary_pos_emb) or rotary_pos_emb.dim() != 5 or rotary_pos_emb.shape[0] != 2:
            got = (
                tuple(rotary_pos_emb.shape)
                if torch.is_tensor(rotary_pos_emb)
                else type(rotary_pos_emb).__name__
            )
            raise ValueError(
                f"OLMo-3 attention needs the stacked [2, ...] RoPE table of Olmo3DualRotaryEmbedding, got {got}"
            )
        return rotary_pos_emb[1 if self.olmo3_use_yarn else 0]

    def forward(self, hidden_states, attention_mask, *args, rotary_pos_emb=None, **kwargs):
        # positional order upstream: key_value_states, inference_context, ...
        ctx = kwargs.get("inference_context", kwargs.get("inference_params"))
        if ctx is None and len(args) > 1:
            ctx = args[1]
        if ctx is not None and not ctx.is_static_batching():
            raise NotImplementedError(
                "OLMo-3 attention does not support dynamic-batching inference: upstream's "
                "flash_decode_and_prefill applies no sliding window and no per-layer YaRN scale"
            )
        out, bias = super().forward(
            hidden_states,
            attention_mask,
            *args,
            rotary_pos_emb=self._select_rotary(rotary_pos_emb),
            **kwargs,
        )
        if bias is not None:
            out = out + bias
            bias = None
        return self.post_attention_layernorm(out), bias
