###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""No zero gradients for the routing maps of Transformer Engine's fused router.

``FusedTopkScoreFunction`` (``fused_topk_with_score_function``) and ``FusedComputeScoresForMoEAuxLoss``
(``fused_compute_score_for_moe_aux_loss``), which Megatron's router runs with ``moe_router_fusion``,
return a bool ``[tokens, experts]`` routing map next to the probabilities / scores, and their backward
ignores its gradient. Autograd still materializes one, as ``set_materialize_grads`` defaults to True:
a zero bool tensor per call, two per MoE layer and micro-batch, and on gfx1250 a large bool fill is slow
(much slower than a uint8 one).

Materialization off for the two Functions: the routing map's gradient then arrives as None, as the
probabilities' / scores' would if nothing consumed them, in which case the logits get no gradient from
the Function either (what the materialized zeros would have produced). Gradients are unchanged.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_FUNCTIONS = ("FusedTopkScoreFunction", "FusedComputeScoresForMoEAuxLoss")


def _wrap(fn) -> None:
    orig_forward, orig_backward = fn.forward, fn.backward

    def forward(ctx, *args, **kwargs):
        ctx.set_materialize_grads(False)
        return orig_forward(ctx, *args, **kwargs)

    def backward(ctx, *grads):
        # The routing map's gradient is always None now, so all None means no gradient reached the
        # probabilities / scores either.
        if all(grad is None for grad in grads):
            return (None,) * len(ctx.needs_input_grad)
        return orig_backward(ctx, *grads)

    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    fn._primus_no_materialized_grads = True


def _enabled(ctx: PatchContext) -> bool:
    # Only Megatron's moe_router_fusion runs these Functions. The bool fills this saves are slow on
    # gfx1250, where it was measured; elsewhere TE is left alone.
    return is_gfx1250() and bool(getattr(get_args(ctx), "moe_router_fusion", False))


@register_patch(
    "megatron.moe.te_router_no_materialized_grads",
    backend="megatron",
    phase="before_train",
    description="TE fused router: no zero gradients materialized for its bool routing maps (gfx1250)",
    condition=_enabled,
)
def patch_te_router_no_materialized_grads(ctx: PatchContext) -> None:
    try:
        from transformer_engine.pytorch import router as te_router
    except ImportError:
        return
    patched = []
    for name in _FUNCTIONS:
        fn = getattr(te_router, name, None)
        if fn is None or getattr(fn, "_primus_no_materialized_grads", False):
            continue
        _wrap(fn)
        patched.append(name)
    if patched:
        log_rank_0(
            f"[Patch:megatron.moe.te_router_no_materialized_grads] {', '.join(patched)}: "
            "no zero gradients for the routing maps"
        )
