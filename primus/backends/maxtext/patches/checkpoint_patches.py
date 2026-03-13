###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Checkpoint Patches

- Wraps ``CheckpointManagerOptions`` globally to inject ``max_to_keep=5``
  (Primus default; upstream MaxText does not pass ``max_to_keep`` at all).
- Wraps ``create_training_tools`` to temporarily override that default with
  ``config.max_num_checkpoints_to_keep`` from the Primus config.
- Replaces ``load_state_if_possible`` with a Primus implementation that
  supports local-filesystem checkpoints (Dec version only).
"""

import functools

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

_PRIMUS_DEFAULT_MAX_TO_KEEP = 5


def _make_opts_wrapper(orig_opts, default_max_to_keep):
    """Return a ``CheckpointManagerOptions`` wrapper that injects ``max_to_keep``."""

    @functools.wraps(orig_opts)
    def _wrapped(*args, **kwargs):
        kwargs.setdefault("max_to_keep", default_max_to_keep)
        return orig_opts(*args, **kwargs)

    return _wrapped


def _wrap_create_training_tools(orig_func, ckpt_mod, orig_opts):
    """Wrap ``create_training_tools`` so that during its execution,
    ``CheckpointManagerOptions`` uses ``config.max_num_checkpoints_to_keep``
    instead of the global Primus default."""

    @functools.wraps(orig_func)
    def _wrapped(config, model, mesh):
        max_to_keep = getattr(config, "max_num_checkpoints_to_keep", 5)

        _saved = ckpt_mod.CheckpointManagerOptions
        ckpt_mod.CheckpointManagerOptions = _make_opts_wrapper(orig_opts, max_to_keep)
        try:
            return orig_func(config, model, mesh)
        finally:
            ckpt_mod.CheckpointManagerOptions = _saved

    return _wrapped


@register_patch(
    patch_id="maxtext.checkpoint",
    backend="maxtext",
    phase="setup",
    description="Wrap CheckpointManagerOptions, create_training_tools, and replace load_state_if_possible (Dec version)",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
)
def patch_checkpointing(ctx: PatchContext) -> None:
    log_rank_0("[Patch:maxtext.checkpoint] Patching MaxText checkpointing...")

    import MaxText.checkpointing as ckpt_mod
    import MaxText.train_utils as orig_train_utils

    from primus.backends.maxtext.checkpointing import load_state_if_possible

    orig_opts = ckpt_mod.CheckpointManagerOptions
    ckpt_mod.CheckpointManagerOptions = _make_opts_wrapper(orig_opts, _PRIMUS_DEFAULT_MAX_TO_KEEP)

    orig_train_utils.create_training_tools = _wrap_create_training_tools(
        orig_train_utils.create_training_tools, ckpt_mod, orig_opts
    )

    ckpt_mod.load_state_if_possible = load_state_if_possible

    warning_rank_0("[Patch:maxtext.checkpoint] checkpointing patched successfully.")


@register_patch(
    patch_id="maxtext.checkpoint.legacy",
    backend="maxtext",
    phase="setup",
    description="Wrap CheckpointManagerOptions and create_training_tools (Aug version)",
    condition=lambda ctx: True,
    backend_versions=["2025.*"],
)
def patch_checkpointing_legacy(ctx: PatchContext) -> None:
    log_rank_0("[Patch:maxtext.checkpoint.legacy] Patching MaxText checkpointing (legacy)...")

    import MaxText.checkpointing as ckpt_mod
    import MaxText.train_utils as orig_train_utils

    orig_opts = ckpt_mod.CheckpointManagerOptions
    ckpt_mod.CheckpointManagerOptions = _make_opts_wrapper(orig_opts, _PRIMUS_DEFAULT_MAX_TO_KEEP)

    orig_train_utils.create_training_tools = _wrap_create_training_tools(
        orig_train_utils.create_training_tools, ckpt_mod, orig_opts
    )

    warning_rank_0("[Patch:maxtext.checkpoint.legacy] checkpointing patched successfully.")
