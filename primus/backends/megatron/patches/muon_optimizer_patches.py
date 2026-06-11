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


@register_patch(
    "megatron.optimizer.muon_builder",
    backend="megatron",
    phase="before_train",
    description=(
        "Redirect megatron.training.training.get_megatron_muon_optimizer to the "
        "Primus moun.get_megatron_muon_optimizer (new emerging_optimizers API + "
        "DeepSeek-V4 'deepseekv4' Newton-Schulz coefficients)."
    ),
)
def patch_get_megatron_muon_optimizer_builder(ctx: PatchContext) -> None:
    """Point Megatron's ``get_megatron_muon_optimizer`` at the Primus version.

    Megatron's ``setup_model_and_optimizer`` (training.py) branches on
    ``'muon' in config.optimizer`` and calls ``get_megatron_muon_optimizer``
    **directly** (training.py:1538) -- so the ``get_megatron_optimizer``
    dispatch patch above never sees the Muon path. The upstream
    ``megatron.core.optimizer.muon`` builder targets an OLDER
    ``emerging_optimizers`` API (``use_nesterov`` / ``mode``) that the pinned
    package (>=0.4.0a0, the first version carrying the ``deepseekv4``
    coefficient set) renamed to ``nesterov`` / ``tp_mode``, so the upstream
    builder raises ``TypeError`` at ``OrthogonalizedOptimizer.__init__``.

    The Primus ``moun.get_megatron_muon_optimizer`` is already adapted to the
    new API and auto-selects the report's ``deepseekv4`` hybrid Newton-Schulz
    schedule (8 aggressive + 2 stable) for V4 configs. We rebind the name in
    the ``training`` module namespace (where it was imported at line 130) so
    the Muon branch builds through the Primus path. R6.2-compliant: no
    third_party edit.
    """
    try:
        import megatron.training.training as training_module
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.optimizer.muon_builder] Skip patch (Megatron not available): {e}")
        return

    original_builder = getattr(training_module, "get_megatron_muon_optimizer", None)
    if original_builder is None:
        log_rank_0(
            "[Patch:megatron.optimizer.muon_builder] training module has no "
            "get_megatron_muon_optimizer symbol; nothing to patch."
        )
        return
    if getattr(original_builder, "_primus_muon_builder_wrapper", False):
        return

    def _patched_get_megatron_muon_optimizer(*func_args, **func_kwargs):
        config = func_kwargs.get("config")
        if config is None and func_args:
            config = func_args[0]
        model_chunks = func_kwargs.get("model_chunks")
        if model_chunks is None and len(func_args) >= 2:
            model_chunks = func_args[1]
        config_overrides = func_kwargs.get("config_overrides")
        # The Primus moun builder constructs/forwards a ProcessGroupCollection
        # to the inner AdamW get_megatron_optimizer, and Megatron asserts that
        # Gloo process groups are incompatible with a provided pg_collection.
        # Force gloo off for the Muon path (it does not need it).
        use_gloo_process_groups = False
        optimizer_name = getattr(config, "optimizer", "muon") or "muon"
        layer_wise_distributed_optimizer = func_kwargs.get(
            "layer_wise_distributed_optimizer", "dist" in optimizer_name
        )

        from primus.backends.megatron.core.optimizer.moun import (
            get_megatron_muon_optimizer,
        )
        from primus.backends.megatron.core.optimizer.moun_optimizer_config import (
            MounOptimizerConfig,
        )

        # Build a MounOptimizerConfig from the backend args so all muon_*
        # fields (incl. the Primus-only muon_coefficient_type) are populated;
        # fall back to copying from the passed OptimizerConfig when an arg is
        # absent.
        args = ctx.extra.get("backend_args", {})
        kwargs = {}
        for f in dataclasses.fields(MounOptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
            elif hasattr(config, f.name):
                kwargs[f.name] = getattr(config, f.name)
        moun_config = MounOptimizerConfig(**kwargs)
        moun_config.timers = getattr(config, "timers", None)

        return get_megatron_muon_optimizer(
            moun_config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer=layer_wise_distributed_optimizer,
        )

    setattr(_patched_get_megatron_muon_optimizer, "_primus_muon_builder_wrapper", True)
    training_module.get_megatron_muon_optimizer = _patched_get_megatron_muon_optimizer
    log_rank_0(
        "[Patch:megatron.optimizer.muon_builder] Redirected get_megatron_muon_optimizer "
        "in megatron.training.training to the Primus moun builder (new emerging_optimizers "
        "API + deepseekv4 coefficients)."
    )
