###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Projection config guards for combinations that are invalid or misleading in models.

Split-wgrad pipeline schedules defer weight gradients and pin cloned linear
inputs until W nodes run. Activation recomputation only frees autograd-graph
activations, so the two together (especially on MoE) produce misleading
memory projections and poor runtime memory behavior.
"""

from __future__ import annotations

from typing import Any, Optional

from primus.core.projection.training_config import ModelParallelConfig, TrainingConfig

# Schedules that split backward into B (dgrad) and W (wgrad) in Primus/Megatron.
# Includes projection CLI names and runtime ``pp_algorithm`` names.
SPLIT_WGRAD_PIPELINE_SCHEDULES = frozenset(
    {
        "zerobubble",
        "zerobubble-heuristic",
        "zero-bubble",
        "zero-bubble-heuristic",
        "zbv-formatted",
        "zbv-greedy-half",
        "zbv-greedy-min",
        "v-half",
        "v-min",
        "seaailab-ilp",
    }
)

# Map projection / agent schedule names to runtime ``pp_algorithm`` when writing YAML.
PP_SCHEDULE_TO_RUNTIME_ALGORITHM = {
    "zerobubble": "zero-bubble",
    "zerobubble-heuristic": "zero-bubble-heuristic",
    "zbv-formatted": "zbv-formatted",
    "zbv-greedy-half": "v-half",
    "zbv-greedy-min": "v-min",
    "seaailab-ilp": "seaailab-ilp",
}


def recompute_is_enabled(
    mp_config: ModelParallelConfig,
    *,
    module_cfg: Any = None,
) -> bool:
    """True when any activation-recompute lever is active."""
    granularity = getattr(mp_config, "recompute_granularity", None)
    if granularity in ("full", "selective"):
        return True
    num_layers = getattr(mp_config, "recompute_num_layers", 0) or 0
    if num_layers > 0:
        return True
    layer_ids = getattr(mp_config, "recompute_layer_ids", None)
    if layer_ids:
        return True
    if module_cfg is not None:
        if getattr(module_cfg, "recompute_granularity", None) in ("full", "selective"):
            return True
        if getattr(module_cfg, "recompute_num_layers", 0):
            return True
        if getattr(module_cfg, "recompute_layer_ids", None):
            return True
    return False


def resolve_pipeline_schedule(
    *,
    pipeline_schedule_algorithm: Optional[str] = None,
    pp_algorithm: Optional[str] = None,
    enable_zero_bubble: bool = False,
) -> str:
    """Effective schedule name for validation (CLI overrides YAML)."""
    if pipeline_schedule_algorithm and pipeline_schedule_algorithm != "auto":
        return pipeline_schedule_algorithm
    if pp_algorithm:
        return str(pp_algorithm)
    if enable_zero_bubble:
        # Matches performance projection ``auto`` + enable_zero_bubble.
        return "zerobubble"
    return "auto"


def uses_split_wgrad_schedule(schedule: str, *, enable_zero_bubble: bool = False) -> bool:
    if schedule in SPLIT_WGRAD_PIPELINE_SCHEDULES:
        return True
    if schedule == "auto" and enable_zero_bubble:
        return True
    return False


def check_recompute_pipeline_compat(
    *,
    recompute_granularity: Optional[str] = None,
    recompute_num_layers: int = 0,
    recompute_layer_ids: Optional[list] = None,
    pipeline_schedule: str = "auto",
    pp_algorithm: Optional[str] = None,
    enable_zero_bubble: Optional[bool] = None,
) -> tuple[bool, str]:
    """Return (ok, reason). Used by the tuning agent before launching projection."""
    granularity = recompute_granularity
    recompute_on = granularity in ("full", "selective") or (recompute_num_layers or 0) > 0
    if recompute_layer_ids:
        recompute_on = True
    if not recompute_on:
        return True, ""

    sched = resolve_pipeline_schedule(
        pipeline_schedule_algorithm=pipeline_schedule,
        pp_algorithm=pp_algorithm,
        enable_zero_bubble=bool(enable_zero_bubble),
    )
    if not uses_split_wgrad_schedule(sched, enable_zero_bubble=bool(enable_zero_bubble)):
        return True, ""

    return False, (
        "activation recomputation is incompatible with split-wgrad pipeline schedules "
        f"(schedule={sched!r}, recompute_granularity={granularity!r}, "
        f"recompute_num_layers={recompute_num_layers}). "
        "Use pp_schedule='auto' or '1f1b-interleaved' (inline BW), or set "
        "recompute_granularity='none'."
    )


def assert_recompute_pipeline_compat(
    training_config: TrainingConfig,
    *,
    primus_config: Any = None,
    pipeline_schedule_algorithm: Optional[str] = None,
) -> None:
    """Raise AssertionError when recompute + split-wgrad schedule are combined."""
    mp = training_config.model_parallel_config
    module_cfg = None
    if primus_config is not None:
        module_cfg = primus_config.get_module_config("pre_trainer")

    if not recompute_is_enabled(mp, module_cfg=module_cfg):
        return

    pp_algorithm = getattr(module_cfg, "pp_algorithm", None) if module_cfg is not None else None
    enable_zb = bool(getattr(module_cfg, "enable_zero_bubble", False)) if module_cfg else False

    sched = resolve_pipeline_schedule(
        pipeline_schedule_algorithm=pipeline_schedule_algorithm,
        pp_algorithm=pp_algorithm,
        enable_zero_bubble=enable_zb,
    )
    if not uses_split_wgrad_schedule(sched, enable_zero_bubble=enable_zb):
        return

    raise AssertionError(
        "[Primus:Projection] Invalid config: activation recomputation cannot be combined "
        "with split-wgrad pipeline schedules (wgrad closures pin recomputed linear inputs, "
        "so analytical memory savings and runtime peak memory are both wrong). "
        f"schedule={sched!r}, recompute_granularity={getattr(mp, 'recompute_granularity', None)!r}, "
        f"recompute_num_layers={getattr(mp, 'recompute_num_layers', 0)!r}. "
        "Use 1f1b / 1f1b-interleaved (or auto without enable_zero_bubble), or disable recompute."
    )


__all__ = [
    "SPLIT_WGRAD_PIPELINE_SCHEDULES",
    "PP_SCHEDULE_TO_RUNTIME_ALGORITHM",
    "assert_recompute_pipeline_compat",
    "check_recompute_pipeline_compat",
    "recompute_is_enabled",
    "resolve_pipeline_schedule",
    "uses_split_wgrad_schedule",
]
