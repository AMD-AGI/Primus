###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-exact fused replacements for chains of small kernels.

* ``PRIMUS_EXACT_CLAMPED_GEGLU`` (default on): Megatron ``MLP``'s eager clamped quick-GEGLU
  (MiniMax-M3 ``swigluoai``) runs as one forward and one backward Triton kernel with the same
  BF16 roundings (``primus/backends/megatron/core/fusions/exact_clamped_quick_geglu.py``).
  ``PrimusGroupedMLP`` (routed experts) checks the same switch in its ``bias_act_func``.
* ``PRIMUS_EXACT_TE_CROSS_ENTROPY`` (default on): Megatron's ``te_parallel_cross_entropy`` on one
  tensor-parallel rank keeps TE's online-softmax kernel but drops its FP32 gradient buffer
  (``primus/backends/megatron/core/fusions/exact_te_cross_entropy.py``). TE's in-place backward
  ``element_mul_kernel`` was seen to drop stores non-deterministically on gfx1250 in a standalone test;
  the fused backward does not, and training losses are bitwise those of TE.

Both produce bitwise the same outputs and gradients as the paths they replace.
"""

from __future__ import annotations

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def exact_clamped_geglu_enabled() -> bool:
    return os.environ.get("PRIMUS_EXACT_CLAMPED_GEGLU", "1") != "0"


def _mlp_eligible(mlp) -> bool:
    cached = getattr(mlp, "_primus_exact_geglu", None)
    if cached is not None:
        return cached
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    cfg = mlp.config
    ok = (
        cfg.gated_linear_unit
        and cfg.activation_func is quick_gelu
        and mlp.activation_func is quick_gelu
        and getattr(cfg, "activation_func_clamp_value", None) is not None
        and not getattr(cfg, "use_te_activation_func", False)
        and not cfg.bias_activation_fusion
        and not getattr(cfg, "fp8", None)
        and not getattr(cfg, "fp4", None)
    )
    mlp._primus_exact_geglu = bool(ok)
    return mlp._primus_exact_geglu


def _install_mlp() -> None:
    from megatron.core.transformer.mlp import MLP

    from primus.backends.megatron.core.fusions.exact_clamped_quick_geglu import (
        eligible_input,
        exact_clamped_quick_geglu,
    )

    if getattr(MLP.forward, "_primus_exact_geglu", False):
        return
    original_forward = MLP.forward

    def forward(self, hidden_states, per_token_scale=None, **kwargs):
        # Megatron's MLP.forward ignores **kwargs (the transformer layer passes padding_mask).
        if per_token_scale is not None or not _mlp_eligible(self):
            return original_forward(self, hidden_states, per_token_scale, **kwargs)
        intermediate, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            intermediate = intermediate + bias
        if eligible_input(intermediate):
            intermediate = exact_clamped_quick_geglu(
                intermediate, self.config.activation_func_clamp_value, self.config.glu_linear_offset
            )
        else:  # Megatron's eager branch
            x_glu, x_linear = intermediate.chunk(2, dim=-1)
            val = self.config.activation_func_clamp_value
            x_glu = x_glu.clamp(min=None, max=val)
            x_linear = x_linear.clamp(min=-val, max=val)
            intermediate = self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)
        return self.linear_fc2(intermediate)

    forward._primus_exact_geglu = True
    MLP.forward = forward


@register_patch(
    "megatron.mlp.exact_clamped_quick_geglu",
    backend="megatron",
    phase="before_train",
    description="MLP clamped quick-GEGLU (swigluoai) as one bit-exact Triton fwd/bwd kernel",
    condition=lambda ctx: exact_clamped_geglu_enabled(),
)
def patch_exact_clamped_quick_geglu(ctx: PatchContext) -> None:
    del ctx
    _install_mlp()
    log_rank_0("[Patch:megatron.mlp.exact_clamped_quick_geglu] MLP swigluoai: one bit-exact fwd and bwd kernel")


@register_patch(
    "megatron.loss.exact_te_cross_entropy",
    backend="megatron",
    phase="before_train",
    description="TE cross entropy without its FP32 gradient buffer (bit-exact, one TP rank)",
    condition=lambda ctx: os.environ.get("PRIMUS_EXACT_TE_CROSS_ENTROPY", "1") != "0",
)
def patch_exact_te_cross_entropy(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.models.common.language_module import language_module

    original = getattr(language_module, "te_parallel_cross_entropy", None)
    if original is None or getattr(original, "_primus_exact", False):
        return
    from primus.backends.megatron.core.fusions.exact_te_cross_entropy import (
        exact_te_parallel_cross_entropy,
    )

    def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
        return exact_te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable, fallback=original)

    te_parallel_cross_entropy._primus_exact = True
    language_module.te_parallel_cross_entropy = te_parallel_cross_entropy
    log_rank_0("[Patch:megatron.loss.exact_te_cross_entropy] TE cross entropy: no FP32 gradient buffer (bit-exact)")


@register_patch(
    "megatron.embedding.exact_weight_grad",
    backend="megatron",
    phase="before_train",
    description="Embedding weight gradient bitwise ATen's in one launch (persistent with direct-grad Adam)",
    condition=lambda ctx: os.environ.get("PRIMUS_EXACT_EMBEDDING_GRAD", "1") != "0",
)
def patch_exact_embedding_grad(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

    if getattr(VocabParallelEmbedding.forward, "_primus_exact", False):
        return
    from primus.backends.megatron.core.fusions import exact_embedding_grad as eg

    original_forward = VocabParallelEmbedding.forward

    def forward(self, input_):
        if (
            self.tp_group.size() > 1
            or getattr(self, "gtp_remat_size", 1) > 1
            or self.deterministic_mode
            or self.reduce_scatter_embeddings
            or not eg.eligible(input_, self.weight)
        ):
            return original_forward(self, input_)
        return eg.exact_embedding(input_, self.weight, owner=self)

    forward._primus_exact = True
    VocabParallelEmbedding.forward = forward
    log_rank_0("[Patch:megatron.embedding.exact_weight_grad] embedding weight grad: one bit-exact launch")


@register_patch(
    "megatron.moe.fused_grouped_gemm_meta",
    backend="megatron",
    phase="before_train",
    description="FlyDSL grouped-GEMM m-tile map / expert table built in one launch each (integer, same tensors)",
    condition=lambda ctx: os.environ.get("PRIMUS_FUSED_GG_META", "1") != "0",
)
def patch_fused_grouped_gemm_meta(ctx: PatchContext) -> None:
    del ctx
    from primus.backends.megatron.core.fusions import grouped_gemm_meta

    if grouped_gemm_meta.install():
        log_rank_0("[Patch:megatron.moe.fused_grouped_gemm_meta] grouped GEMM metadata: one launch per builder")


@register_patch(
    "megatron.moe.exact_router_gemm",
    backend="megatron",
    phase="before_train",
    description="FP32 router-gating backward GEMMs as bit-exact WMMA kernels (same K chain as hipBLASLt), casts fused",
    condition=lambda ctx: os.environ.get("PRIMUS_EXACT_ROUTER_GEMM", "1") != "0",
)
def patch_exact_router_gemm(ctx: PatchContext) -> None:
    del ctx
    from primus.backends.megatron.core.fusions import exact_router_gemm

    if exact_router_gemm.install():
        log_rank_0("[Patch:megatron.moe.exact_router_gemm] router gating backward: bit-exact FP32 WMMA dgrad/wgrad")


@register_patch(
    "megatron.moe.fused_routing_counts",
    backend="megatron",
    phase="before_train",
    description="routing_map.sum(dim=0) once per routing map (integer counts; cached / produced by even routing)",
    condition=lambda ctx: os.environ.get("PRIMUS_FUSED_ROUTING_COUNTS", "1") != "0",
)
def patch_fused_routing_counts(ctx: PatchContext) -> None:
    del ctx
    from primus.backends.megatron.core.fusions import routing_counts

    if routing_counts.install():
        log_rank_0("[Patch:megatron.moe.fused_routing_counts] tokens per expert: one count per routing map")


@register_patch(
    "megatron.moe.router_no_map_grad",
    backend="megatron",
    phase="before_train",
    description="Turbo fused router: no materialised zero gradient for its boolean routing-map output",
    condition=lambda ctx: os.environ.get("PRIMUS_ROUTER_NO_MAP_GRAD", "1") != "0",
)
def patch_router_no_map_grad(ctx: PatchContext) -> None:
    """Autograd fills a [T, E] zero gradient for the unused bool routing map (~45 us) before
    FusedGroupTopkRoutingWithAuxScoreFunction.backward, which ignores it. Score / probs
    gradients are still materialised (zeros) when absent, exactly as before."""
    del ctx
    import torch

    try:
        from primus_turbo.pytorch.ops.moe import fused_moe_router as fmr
    except Exception:
        return
    F = getattr(fmr, "FusedGroupTopkRoutingWithAuxScoreFunction", None)
    if F is None or getattr(F, "_primus_no_map_grad", False):
        return
    orig_fwd, orig_bwd = F.forward, F.backward

    def forward(ctx, *args, **kwargs):
        out = orig_fwd(ctx, *args, **kwargs)
        ctx.set_materialize_grads(False)
        ctx._primus_meta = [(o.shape, o.dtype, o.device) for o in out[:2]]
        return out

    def backward(ctx, g_score, g_probs, g_routing_map):
        (s0, d0, v0), (s1, d1, v1) = ctx._primus_meta
        if g_score is None:
            g_score = torch.zeros(s0, dtype=d0, device=v0)
        if g_probs is None:
            g_probs = torch.zeros(s1, dtype=d1, device=v1)
        return orig_bwd(ctx, g_score, g_probs, None)

    F.forward = staticmethod(forward)
    F.backward = staticmethod(backward)
    F._primus_no_map_grad = True
    log_rank_0("[Patch:megatron.moe.router_no_map_grad] fused router: no zero grad for the routing map")


@register_patch(
    "megatron.moe.metrics_sync_1rank",
    backend="megatron",
    phase="before_train",
    description="MoE metrics tracker: no all-reduces when the world has one rank (SUM / AVG over 1 rank are identities)",
    condition=lambda ctx: os.environ.get("PRIMUS_MOE_METRICS_SKIP_1RANK_SYNC", "1") != "0",
)
def patch_moe_metrics_sync_1rank(ctx: PatchContext) -> None:
    del ctx
    import torch

    try:
        from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
    except Exception:
        return
    orig = MoEMetricsTracker._sync_metrics
    if getattr(orig, "_primus_1rank", False):
        return

    def _sync_metrics(self, *args, **kwargs):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1:
            return None
        return orig(self, *args, **kwargs)

    _sync_metrics._primus_1rank = True
    MoEMetricsTracker._sync_metrics = _sync_metrics
    log_rank_0("[Patch:megatron.moe.metrics_sync_1rank] MoE metrics: skip 1-rank all-reduces")
