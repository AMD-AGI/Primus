###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron argparse compatibility patches.

Some Megatron-LM forks/versions do not expose `add_megatron_arguments(parser)`,
but Primus's `MegatronArgBuilder` relies on it to obtain the full argparse
definition set (defaults/types).

This patch injects a best-effort `add_megatron_arguments` into
`megatron.training.arguments` if it is missing, by mirroring the internal
`parse_args()` construction flow.
"""

from __future__ import annotations

import inspect
import re
from typing import List

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0

_CANONICAL_ADD_FN_NAMES: List[str] = [
    "_add_network_size_args",
    "_add_regularization_args",
    "_add_training_args",
    "_add_rl_args",
    "_add_initialization_args",
    "_add_learning_rate_args",
    "_add_checkpointing_args",
    "_add_mixed_precision_args",
    "_add_distributed_args",
    "_add_validation_args",
    "_add_data_args",
    "_add_tokenizer_args",
    "_add_autoresume_args",
    "_add_biencoder_args",
    "_add_vision_args",
    "_add_moe_args",
    "_add_mla_args",
    "_add_heterogeneous_args",
    "_add_logging_args",
    "_add_straggler_detector_args",
    "_add_workload_inspector_server_args",
    "_add_inference_args",
    "_add_transformer_engine_args",
    "_add_retro_args",
    "_add_experimental_args",
    "_add_one_logger_args",
    "_add_inprocess_restart_args",
    "_add_ft_package_args",
    "_add_config_logger_args",
    "_add_rerun_machine_args",
    "_add_msc_args",
    "_add_kitchen_quantization_arguments",
    "_add_sft_args",
]


def _scan_add_fn_names_from_parse_args(margs) -> List[str]:
    """
    Best-effort: scan `megatron.training.arguments.parse_args` source code to
    discover which internal `_add_*` functions are used to build the parser,
    and in what order.

    If scanning fails (e.g., source not available), return [] and let caller
    fall back to `_CANONICAL_ADD_FN_NAMES`.
    """
    parse_args = getattr(margs, "parse_args", None)
    if not callable(parse_args):
        return []

    try:
        src = inspect.getsource(parse_args)
    except Exception:
        return []

    # Match patterns like:
    #   parser = _add_training_args(parser)
    #   parser=_add_training_args(parser)
    names = re.findall(r"(_add_[A-Za-z0-9_]+)\s*\(\s*parser\s*\)", src)
    if not names:
        return []

    # Preserve order, remove duplicates.
    seen = set()
    ordered: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    return ordered


@register_patch(
    "megatron.args.add_megatron_arguments_compat",
    backend="megatron",
    phase="setup",
    description="Inject add_megatron_arguments(parser) for Megatron versions that only build parsers inside parse_args().",
)
def add_megatron_arguments_compat(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs  # type: ignore
    except Exception as e:
        # Megatron not importable in this environment; nothing to do.
        log_rank_0(f"[Patch:megatron.args.add_megatron_arguments_compat] skip (import failed): {e}")
        return

    if hasattr(margs, "add_megatron_arguments") and callable(getattr(margs, "add_megatron_arguments")):
        return

    fn_names = _scan_add_fn_names_from_parse_args(margs) or list(_CANONICAL_ADD_FN_NAMES)

    def add_megatron_arguments(parser):
        """
        Compatibility shim injected by Primus.
        """
        applied = 0
        for name in fn_names:
            fn = getattr(margs, name, None)
            if callable(fn):
                parser = fn(parser)
                applied += 1
        if applied == 0:
            raise RuntimeError(
                "Megatron does not expose add_megatron_arguments and no internal _add_* "
                "functions were found to construct the parser."
            )
        return parser

    setattr(margs, "add_megatron_arguments", add_megatron_arguments)
    log_rank_0("[Patch:megatron.args.add_megatron_arguments_compat] injected add_megatron_arguments()")
