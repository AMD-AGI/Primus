###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Granite: check the model args against the HF config of the converted checkpoint the run
loads.

Training builds the Granite model from the args, not from the checkpoint's config.json, so a
preset with a wrong multiplier would train a different model on the loaded weights. The
``build_args`` patch below runs ``native_loader_granite.check_args_against_checkpoint`` once
per run, before the trainer and the model are built, and stops the run on a mismatch. It keeps
no state and wraps no Megatron function. A checkpoint without the converter's
``granite_hf_config.json`` is checked against the args Megatron saved in it when they are a
Granite run's; only a checkpoint with neither is loaded unchecked, with a warning.
"""

from types import SimpleNamespace

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _granite_args(ctx: PatchContext):
    """The run's args if it builds Granite, else None.

    In the ``build_args`` phase the runtime has not yet merged the module params into the
    backend args (``train_runtime`` merges them after this phase), so ``model_type`` and the
    Primus-only args are still only in ``module_config.params``. This is the merged view: the
    params, overridden by the backend args, as in the runtime's merge."""
    params = getattr(ctx.extra.get("module_config", None), "params", None)
    backend = ctx.extra.get("backend_args", None)
    merged = {
        **(vars(params) if params is not None else {}),
        **(vars(backend) if backend is not None else {}),
    }
    return SimpleNamespace(**merged) if merged.get("model_type") == "granite" else None


def is_granite_run(ctx: PatchContext) -> bool:
    """Whether this run builds Granite: the merged args (``_granite_args``, the runtime's merge
    rule) have ``model_type == "granite"``."""
    return _granite_args(ctx) is not None


@register_patch(
    "megatron.granite.checkpoint_args",
    backend="megatron",
    phase="build_args",
    description="Granite: refuse model args that differ from the loaded checkpoint's HF config",
    condition=is_granite_run,
)
def patch_granite_checkpoint_args(ctx: PatchContext):
    # ``run_patches`` swallows exceptions (stop_on_error=False) and the run would go on
    # training a model that differs from the checkpoint. Escalate to a hard abort, as the
    # other must-not-continue patches do: ``SystemExit`` is not caught by ``run_patches``.
    # No collective has run yet, so every rank stops the same way. The import is inside the
    # try: a failed import would also leave the run unchecked.
    try:
        from primus.backends.megatron.checkpoint.native_loader_granite import (
            check_args_against_checkpoint,
        )

        check_args_against_checkpoint(_granite_args(ctx), log=log_rank_0)
    except Exception as e:  # noqa: BLE001 - any failure leaves the run unchecked: stop it
        log_rank_0(f"[Patch:megatron.granite.checkpoint_args] FATAL: {type(e).__name__}: {e}")
        raise SystemExit(f"[megatron.granite.checkpoint_args] {type(e).__name__}: {e}") from e
