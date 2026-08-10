###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install Kimi K3's Quantile Balancing at the expert-bias update site.

``finalize_model_grads`` is the one place in Megatron where the expert bias is
updated for a whole global batch::

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)     # finalize_model_grads.py:480-481
    reset_model_temporary_tensors(config, model)      # :483

``_update_router_expert_bias`` (``:293-319``) gathers ``local_tokens_per_expert``
and ``expert_bias`` from every router and hands them to
``get_updated_expert_bias`` (``moe_utils.py:1119-1142``), which applies
DeepSeek-V3's ``b + rate * sign(offset)``. Kimi K3 replaces that whole rule,
so this patch replaces that whole function — token counts are not the
statistic Quantile Balancing needs.

Both symbols are module-level globals looked up at call time, so rebinding
``finalize_model_grads._update_router_expert_bias`` is sufficient; there is no
second import site inside ``megatron.core.distributed`` that captures the
function object earlier.

The patch is a no-op unless ``moe_router_bias_update_rule: quantile``, so the
sign-based baseline (and every non-K3 model) is untouched, and the A/B is a
one-key config change.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _wants_quantile_balancing(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    # Kimi K3 only. This rebinds finalize_model_grads._update_router_expert_bias --
    # a function SHARED by every model that uses the aux-loss-free expert bias
    # (DeepSeek-V3/V4 included) -- so gate on model_type, not just on the generic
    # moe_router_enable_expert_bias / moe_router_bias_update_rule args. Mirrors
    # kimi_k3_flops_patches.py so all K3 patches key off the same predicate.
    if getattr(args, "model_type", None) != "kimi_k3":
        return False
    if not getattr(args, "moe_router_enable_expert_bias", False):
        return False
    return str(getattr(args, "moe_router_bias_update_rule", "sign")) == "quantile"


@register_patch(
    "megatron.kimi_k3.quantile_balancing",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace DeepSeek-V3's sign-based expert-bias update with Kimi K3's "
        "Quantile Balancing (tech report §2.3.3, Eq. 14)."
    ),
    priority=50,
    condition=_wants_quantile_balancing,
)
def patch_quantile_balancing(ctx: PatchContext):
    """Rebind ``_update_router_expert_bias`` onto the quantile rule."""
    import importlib

    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_quantile_balancing import (
        collect_quantile_balancing_routers,
        update_router_expert_bias_quantile,
    )

    # NOT ``from megatron.core.distributed import finalize_model_grads``:
    # ``megatron/core/distributed/__init__.py:10`` re-exports the *function* of
    # that name, so the attribute shadows the module and the import silently
    # hands back a function with no ``_update_router_expert_bias`` on it.
    finalize_model_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    original = finalize_model_grads._update_router_expert_bias
    state = {"step": 0, "warned": False}

    def _quantile_update_router_expert_bias(model, config):
        routers = collect_quantile_balancing_routers(model)
        if not routers:
            # The rule was selected but no router carries a margin histogram,
            # which means the MoE spec did not pick up the QB router class.
            # Falling back to the sign rule silently would look like QB working
            # badly rather than QB not running, so say so, loudly, once, and
            # then behave exactly as before the patch.
            if not state["warned"]:
                state["warned"] = True
                log_rank_0(
                    "[Patch:megatron.kimi_k3.quantile_balancing]   ERROR: "
                    "moe_router_bias_update_rule=quantile but no router exposes a "
                    "local_margin_histogram buffer. The MoE spec is not using the "
                    "Quantile Balancing router (build_stable_latent_moe_spec fills "
                    "submodules.router only when quantile_balancing_enabled(config)). "
                    "Falling back to the sign-based rule."
                )
            return original(model, config)

        state["step"] += 1
        update_router_expert_bias_quantile(model, config, step=state["step"])

    finalize_model_grads._update_router_expert_bias = _quantile_update_router_expert_bias
    log_rank_0(
        "[Patch:megatron.kimi_k3.quantile_balancing]   Patched "
        "finalize_model_grads._update_router_expert_bias -> Quantile Balancing "
        "(report §2.3.3, Eq. 14)"
    )
