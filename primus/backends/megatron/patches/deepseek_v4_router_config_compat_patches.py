###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 router-config compatibility patch.

DeepSeek-V4 uses aux-loss-free ("noaux_tc") expert-bias routing with a
``sqrtsoftplus`` score function (see ``deepseek_v4_base.yaml``:
``moe_router_enable_expert_bias: true`` + ``moe_router_score_function:
sqrtsoftplus``).  This is implemented by Primus' own ``PrimusTopKRouter``
(installed by ``megatron.moe.primus_topk_router``), which natively supports
``sqrtsoftplus``.

Stock Megatron's ``TransformerConfig.__post_init__`` is stricter and rejects
expert bias for any score function other than ``sigmoid``:

    if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
        raise ValueError("Expert bias for aux-loss-free routing only supports sigmoid ...")

That hard-fails V4 model build (for both ``train`` and the projection
benchmark).  Since V4 supplies its own router, this patch relaxes *only* that
specific check by wrapping ``__post_init__`` and temporarily disabling the
``moe_router_enable_expert_bias`` flag across the stock validation, then
restoring the real value so runtime routing is unaffected.  The patch is a
no-op for configs that already satisfy the stock check (e.g. sigmoid routing).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# Score functions that Primus' V4 router implements but stock Megatron's
# expert-bias validation rejects. Keep this conservative so the relaxation only
# engages for known-supported V4 routing modes.
_V4_SUPPORTED_NONSIGMOID_SCORE_FNS = {"sqrtsoftplus", "softmax"}


def _v4_router_compat_needed(args) -> bool:
    if not getattr(args, "moe_router_enable_expert_bias", False):
        return False
    score_fn = str(getattr(args, "moe_router_score_function", "") or "").lower()
    return score_fn in _V4_SUPPORTED_NONSIGMOID_SCORE_FNS


@register_patch(
    "megatron.moe.v4_expert_bias_score_function_compat",
    backend="megatron",
    phase="build_args",  # before model build / TransformerConfig construction
    description=(
        "Relax stock Megatron's expert-bias-requires-sigmoid check so DeepSeek-V4 "
        "noaux_tc routing (sqrtsoftplus + expert bias, handled by PrimusTopKRouter) "
        "can build."
    ),
    condition=lambda ctx: _v4_router_compat_needed(get_args(ctx)),
)
def patch_v4_expert_bias_score_function_compat(ctx: PatchContext):
    """Wrap ``TransformerConfig.__post_init__`` to skip the sigmoid-only check.

    Idempotent: re-applying is a no-op.
    """
    import megatron.core.transformer.transformer_config as config_mod

    TransformerConfig = config_mod.TransformerConfig

    if getattr(TransformerConfig.__post_init__, "_primus_v4_router_compat", False):
        return

    orig_post_init = TransformerConfig.__post_init__

    def new_post_init(self):
        # Stock check raises when (enable_expert_bias and score_function != "sigmoid").
        # V4 supplies its own router for these score functions, so temporarily
        # clear the flag across the stock validation, then restore it so the
        # config still reports the real (True) value to the runtime router.
        relax = bool(getattr(self, "moe_router_enable_expert_bias", False)) and str(
            getattr(self, "moe_router_score_function", "") or ""
        ).lower() in _V4_SUPPORTED_NONSIGMOID_SCORE_FNS
        if not relax:
            orig_post_init(self)
            return
        self.moe_router_enable_expert_bias = False
        try:
            orig_post_init(self)
        finally:
            self.moe_router_enable_expert_bias = True

    new_post_init._primus_v4_router_compat = True
    new_post_init._primus_original = orig_post_init
    TransformerConfig.__post_init__ = new_post_init

    log_rank_0(
        "[Patch:megatron.moe.v4_expert_bias_score_function_compat]   wrapped "
        "TransformerConfig.__post_init__ to allow expert-bias routing with "
        "non-sigmoid score function (DeepSeek-V4 noaux_tc)."
    )
