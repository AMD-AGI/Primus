###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Muon Optimizer patches.

This module patches megatron.core.optimizer.get_megatron_optimizer to
automatically dispatch to get_megatron_muon_optimizer when args.optimizer
contains "muon", enabling muon optimizer support in the backends workflow
without maintaining a separate branch in MegatronTrainer.
"""

import dataclasses

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.optimizer.muon",
    backend="megatron",
    phase="before_train",
    description="Patch get_megatron_optimizer to dispatch to muon optimizer when optimizer name contains 'muon'.",
)
def patch_get_megatron_optimizer_muon(ctx: PatchContext) -> None:
    """
    Patch megatron.core.optimizer.get_megatron_optimizer to delegate to
    get_megatron_muon_optimizer when config.optimizer contains "muon".
    """
    import megatron.core.optimizer as optimizer_module

    original_get_megatron_optimizer = optimizer_module.get_megatron_optimizer

    if getattr(original_get_megatron_optimizer, "_primus_muon_wrapper", False):
        return

    def _patched_get_megatron_optimizer(
        config,
        model_chunks,
        no_weight_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        use_gloo_process_groups=True,
        default_skip_embedding_weight_decay=False,
        pg_collection=None,
    ):
        if not config.optimizer or "muon" not in config.optimizer:
            return original_get_megatron_optimizer(
                config,
                model_chunks,
                no_weight_decay_cond=no_weight_decay_cond,
                scale_lr_cond=scale_lr_cond,
                lr_mult=lr_mult,
                use_gloo_process_groups=use_gloo_process_groups,
                default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
                pg_collection=pg_collection,
            )

        from primus.backends.megatron.core.optimizer.moun import (
            get_megatron_muon_optimizer,
        )
        from primus.backends.megatron.core.optimizer.moun_optimizer_config import (
            MounOptimizerConfig,
        )

        args = ctx.extra.get("backend_args", {})
        kwargs = {}
        for f in dataclasses.fields(MounOptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)

        moun_config = MounOptimizerConfig(**kwargs)
        moun_config.timers = config.timers

        return get_megatron_muon_optimizer(
            moun_config,
            model_chunks,
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in config.optimizer,
            pg_collection=pg_collection,
        )

    setattr(_patched_get_megatron_optimizer, "_primus_muon_wrapper", True)
    optimizer_module.get_megatron_optimizer = _patched_get_megatron_optimizer
    log_rank_0(
        "[Patch:megatron.optimizer.muon] Patched get_megatron_optimizer to dispatch to muon when optimizer contains 'muon'"
    )
