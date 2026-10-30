###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off", ""})


def _flag(value) -> bool:
    """A YAML / env-substituted boolean: a bool, 0 / 1, or one of the usual strings
    (``"false"`` and ``"0"`` are false). Anything else is refused, not guessed."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
        return value.strip().lower() in _TRUE
    raise ValueError(
        f"keep_explicit_save_path must be true or false (true/false, 1/0, yes/no, on/off), got {value!r}"
    )


def _routes_save(ctx: PatchContext) -> bool:
    """The patch condition (no side effects): False when the experiment sets
    ``keep_explicit_save_path: true``; then ``save`` is used as given (e.g. ``--save /abs/dir``
    and a later ``--load /abs/dir``)."""
    return not _flag(get_param(ctx, "keep_explicit_save_path", False))


def _keeps_save(ctx: PatchContext) -> bool:
    """Condition of the opt-out's log patch: the inverse of ``_routes_save``."""
    return _flag(get_param(ctx, "keep_explicit_save_path", False))


@register_patch(
    "megatron.args.checkpoint_path",
    backend="megatron",
    phase="build_args",
    description="Set checkpoint save path based on experiment root",
    condition=_routes_save,
)
def patch_checkpoint_path(ctx: PatchContext):
    """
    Configure checkpoint save path.

    Behavior:
      - args.save is None  (YAML explicitly set `save: null`, or unset):
            user has opted out of saving — leave args.save = None so that
            Megatron's `checkpoint_and_decide_exit` (training.py:2340) skips
            save_checkpoint_and_time(...). This is what allows runs configured
            with `save: null` + `save_interval: <any>` to avoid hitting the
            checkpoint save path entirely.
      - args.save is non-None:
            user has opted in to saving — route the destination to
            <exp_root>/checkpoints, warning if it differs from what the user
            specified.
      - keep_explicit_save_path: true (any model; default false):
            the patch does not run, so a non-null save is used as given.
    """
    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})

    if not args or not primus_config.exp_root_path:
        return

    # Respect explicit opt-out via `save: null` in YAML.
    if not hasattr(args, "save") or args.save is None:
        log_rank_0(
            "[Patch:megatron.args.checkpoint_path] "
            "args.save is None (opt-out); skipping save-path override."
        )
        return

    ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))

    if args.save != ckpt_path:
        log_rank_0(
            f"[Patch:megatron.args.checkpoint_path][WARN] "
            f"args.save is deprecated; overriding to: {ckpt_path}"
        )

    args.save = ckpt_path
    log_rank_0(f"[Patch:megatron.args.checkpoint_path] save → {ckpt_path}")


@register_patch(
    "megatron.args.checkpoint_path_kept",
    backend="megatron",
    phase="build_args",
    description="keep_explicit_save_path: true: use save as given (made absolute) and say so",
    condition=_keeps_save,
)
def patch_keep_explicit_save_path(ctx: PatchContext):
    """The opt-out's own log line (the registry only logs "condition not met" for the patch
    above). A relative save is made absolute here, once, so the log shows the directory
    every rank writes to."""
    args = ctx.extra.get("backend_args", {})
    if not args or getattr(args, "save", None) is None:
        log_rank_0(
            "[Patch:megatron.args.checkpoint_path] keep_explicit_save_path: true, "
            "args.save is None (opt-out); no checkpoint is saved."
        )
        return
    args.save = os.path.abspath(args.save)
    log_rank_0(
        f"[Patch:megatron.args.checkpoint_path] keep_explicit_save_path: true; "
        f"save kept as given → {args.save} (not <exp_root>/checkpoints)"
    )
