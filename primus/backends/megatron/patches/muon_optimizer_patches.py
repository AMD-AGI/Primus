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
import inspect

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
    original_signature = inspect.signature(original_get_megatron_optimizer)

    if getattr(original_get_megatron_optimizer, "_primus_muon_wrapper", False):
        return

    def _get_bound_arg(bound_arguments, name, fallback=None):
        if name in bound_arguments:
            return bound_arguments[name]

        parameter = original_signature.parameters.get(name)
        if parameter and parameter.default is not inspect.Parameter.empty:
            return parameter.default

        return fallback

    def _patched_get_megatron_optimizer(*func_args, **func_kwargs):
        config = func_kwargs.get("config")
        if config is None and func_args:
            config = func_args[0]

        optimizer_name = getattr(config, "optimizer", None)
        if not optimizer_name or "muon" not in optimizer_name:
            return original_get_megatron_optimizer(*func_args, **func_kwargs)

        bound_arguments = original_signature.bind_partial(*func_args, **func_kwargs).arguments
        model_chunks = _get_bound_arg(bound_arguments, "model_chunks")
        config_overrides = _get_bound_arg(bound_arguments, "config_overrides")
        use_gloo_process_groups = _get_bound_arg(bound_arguments, "use_gloo_process_groups", True)
        pg_collection = _get_bound_arg(bound_arguments, "pg_collection")
        dump_param_to_param_group_map = _get_bound_arg(bound_arguments, "dump_param_to_param_group_map")

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
            layer_wise_distributed_optimizer="dist" in optimizer_name,
            pg_collection=pg_collection,
            dump_param_to_param_group_map=dump_param_to_param_group_map,
        )

    setattr(_patched_get_megatron_optimizer, "_primus_muon_wrapper", True)
    training_module.get_megatron_optimizer = _patched_get_megatron_optimizer
    log_rank_0(
        "[Patch:megatron.optimizer.muon] Patched get_megatron_optimizer in megatron.training.training "
        "to dispatch to muon when optimizer contains 'muon'."
    )
