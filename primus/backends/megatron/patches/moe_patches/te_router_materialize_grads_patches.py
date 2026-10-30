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
a zero bool tensor per call, two per MoE layer and micro-batch. On gfx1250 a bool fill grows with its
size up to ~32 us at 1M elements and more (uint8: ~5 us at any size), and the DeepSeek-V3 proxy's
4-GPU profile shows these two at ~95 us each in the backward: 64 per step, 6.7 ms of a 962 ms step.

Materialization off for the two Functions: the routing map's gradient then arrives as None, as the
probabilities' / scores' would if nothing consumed them, in which case the logits get no gradient from
the Function either (what the materialized zeros would have produced). Gradients are unchanged.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

# Function name -> index of its differentiable output's gradient among backward's arguments
_FUNCTIONS = {"FusedTopkScoreFunction": 0, "FusedComputeScoresForMoEAuxLoss": 1}


def _wrap(fn, grad_index: int) -> None:
    orig_forward, orig_backward = fn.forward, fn.backward

    def forward(ctx, *args, **kwargs):
        ctx.set_materialize_grads(False)
        return orig_forward(ctx, *args, **kwargs)

    def backward(ctx, *grads):
        if grads[grad_index] is None:
            return (None,) * len(ctx.needs_input_grad)
        return orig_backward(ctx, *grads)

    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    fn._primus_no_materialize_grads = True


@register_patch(
    "megatron.moe.te_router_no_materialized_grads",
    backend="megatron",
    phase="before_train",
    description="TE fused router: no zero gradients materialized for its bool routing maps",
)
def patch_te_router_no_materialized_grads(ctx: PatchContext) -> None:
    try:
        from transformer_engine.pytorch import router as te_router
    except ImportError:
        return
    patched = []
    for name, grad_index in _FUNCTIONS.items():
        fn = getattr(te_router, name, None)
        if fn is None or getattr(fn, "_primus_no_materialize_grads", False):
            continue
        _wrap(fn, grad_index)
        patched.append(name)
    if patched:
        log_rank_0(
            f"[Patch:megatron.moe.te_router_no_materialized_grads] {', '.join(patched)}: "
            "no zero gradients for the routing maps"
        )
