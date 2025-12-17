###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo Backend Patches

Patches for integrating PrimusTurbo backend with Transformer Engine in Megatron.
"""

import importlib.util

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.te.primus_turbo_backend",
    backend="megatron",
    phase="before_train",
    description=(
        "Configure Transformer Engine to use PrimusTurbo backend when "
        "enable_primus_turbo is set and tensor_model_parallel_size == 1."
    ),
)
def patch_primus_turbo_backend(ctx: PatchContext):
    """
    Patch Transformer Engine integration to use PrimusTurbo backend.

    Behavior (moved from MegatronTrainer.patch_pt_replace_te):
        - If primus_turbo is installed AND tensor_model_parallel_size == 1 AND
          args.enable_primus_turbo is True:
            * Patch TESpecProvider to PrimusTurboSpecProvider.
            * Optionally replace GPT output layer, MoE dispatcher, and RMSNorm
              with PrimusTurbo implementations based on args flags.
        - Otherwise, fallback to TE backend and log the decision.
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        log_rank_0("[Patch:megatron.te.primus_turbo_backend] primus_turbo not found, use TE backend...")
        return

    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))

    if tp_size != 1:
        if enable_primus_turbo:
            log_rank_0(
                "[Patch:megatron.te.primus_turbo_backend] "
                "Primus Turbo does not support TP; using TE backend instead..."
            )
        else:
            log_rank_0("[Patch:megatron.te.primus_turbo_backend] TP > 1; using TE backend...")
        return

    if not enable_primus_turbo:
        log_rank_0("[Patch:megatron.te.primus_turbo_backend] enable_primus_turbo=False; using TE backend...")
        return

    # At this point we know:
    #   - primus_turbo is installed
    #   - tensor_model_parallel_size == 1
    #   - enable_primus_turbo == True
    try:
        import megatron.core.extensions as meg_ext
        from megatron.core.extensions import (
            transformer_engine as transformer_engine_spec_provider,
        )
        from megatron.core.models.gpt import (
            gpt_layer_specs,
            gpt_model,
            moe_module_specs,
        )
        from megatron.core.transformer import multi_token_prediction
        from megatron.core.transformer.moe import moe_layer, token_dispatcher

        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboColumnParallelLinearTorch,
            PrimusTurboDeepEPTokenDispatcher,
            PrimusTurboRMSNorm,
        )
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            PrimusTurboSpecProvider,
        )

        log_rank_0(
            "[Patch:megatron.te.primus_turbo_backend] "
            "Patch TESpecProvider to PrimusTurboSpecProvider; PrimusTurbo backend enabled"
        )

        assert (
            meg_ext.transformer_engine.HAVE_TE
        ), "PrimusTurboSpecProvider patch failed, can't find transformer_engine"

        # Replace TESpecProvider in all relevant locations
        transformer_engine_spec_provider.TESpecProvider = PrimusTurboSpecProvider
        gpt_layer_specs.TESpecProvider = PrimusTurboSpecProvider
        moe_module_specs.TESpecProvider = PrimusTurboSpecProvider
        multi_token_prediction.TESpecProvider = PrimusTurboSpecProvider

        # Optional: patch GPT output layer
        if getattr(args, "use_turbo_parallel_linear", False):
            gpt_model.tensor_parallel.ColumnParallelLinear = PrimusTurboColumnParallelLinearTorch

        # Optional: patch MoE dispatcher
        if getattr(args, "use_turbo_deepep", False):
            # use PrimusTurboDeepEPTokenDispatcher will auto-enable:
            #   moe_enable_deepep=True, moe_token_dispatcher_type='flex'
            args.moe_enable_deepep = True
            args.moe_token_dispatcher_type = "flex"
            token_dispatcher.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
            moe_layer.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher

        # Optional: patch RMSNorm
        if getattr(args, "use_turbo_rms_norm", False):
            import transformer_engine as te

            te.pytorch.RMSNorm = PrimusTurboRMSNorm

        log_rank_0("[Patch:megatron.te.primus_turbo_backend] Using PrimusTurbo backend (PT)")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][SKIP] Import failed: {e}")
    except AssertionError as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][WARN] Assertion failed: {e}")
    except Exception as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][ERROR] Unexpected error: {e}")
