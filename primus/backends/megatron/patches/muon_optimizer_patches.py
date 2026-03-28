###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Muon Optimizer patches.

This module patches megatron.training.training.get_megatron_optimizer to
automatically dispatch to get_megatron_muon_optimizer when args.optimizer
contains "muon". Since training.py uses `from megatron.core.optimizer import
get_megatron_optimizer`, we must patch the training module's namespace where
the function is actually used, not megatron.core.optimizer.
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
    Patch megatron.training.training.get_megatron_optimizer to delegate to
    get_megatron_muon_optimizer when config.optimizer contains "muon".

    We patch the training module (not megatron.core.optimizer) because
    training.py imports get_megatron_optimizer into its namespace at import
    time; patching the optimizer module would not affect the training module's
    local reference.
    """
    try:
        import megatron.training.training as training_module
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.optimizer.muon] Skip patch (Megatron not available): {e}")
        return

    original_get_megatron_optimizer = training_module.get_megatron_optimizer

    if getattr(original_get_megatron_optimizer, "_primus_muon_wrapper", False):
        return

    def _patched_get_megatron_optimizer(*func_args, **func_kwargs):
        if len(func_args) >= 2:
            config = func_args[0]
            model_chunks = func_args[1]
        else:
            config = func_kwargs.get("config")
            model_chunks = func_kwargs.get("model_chunks")

        config_overrides = func_kwargs.get("config_overrides")
        if config_overrides is None and len(func_args) > 2:
            config_overrides = func_args[2]

        use_gloo_process_groups = func_kwargs.get("use_gloo_process_groups")
        if use_gloo_process_groups is None and len(func_args) > 3:
            use_gloo_process_groups = func_args[3]
        if use_gloo_process_groups is None:
            use_gloo_process_groups = True

        pg_collection = func_kwargs.get("pg_collection")
        if pg_collection is None and len(func_args) > 4:
            pg_collection = func_args[4]

        dump_param_to_param_group_map = func_kwargs.get("dump_param_to_param_group_map")
        if dump_param_to_param_group_map is None and len(func_args) > 5:
            dump_param_to_param_group_map = func_args[5]

        if not config.optimizer or "muon" not in config.optimizer:
            return original_get_megatron_optimizer(*func_args, **func_kwargs)

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
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in config.optimizer,
            pg_collection=pg_collection,
            dump_param_to_param_group_map=dump_param_to_param_group_map,
        )

    setattr(_patched_get_megatron_optimizer, "_primus_muon_wrapper", True)
    training_module.get_megatron_optimizer = _patched_get_megatron_optimizer
    log_rank_0(
        "[Patch:megatron.optimizer.muon] Patched get_megatron_optimizer in megatron.training.training "
        "to dispatch to muon when optimizer contains 'muon'."
    )
