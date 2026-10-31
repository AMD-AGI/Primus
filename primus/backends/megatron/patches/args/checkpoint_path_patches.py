###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.arg_utils import parse_optional_switch
from primus.core.utils.module_utils import log_rank_0

KEEP_ABSOLUTE_SAVE = "keep_absolute_save_path"


def keep_absolute_save(ctx: PatchContext) -> bool:
    """The module param ``keep_absolute_save_path`` (``parse_optional_switch``: unset is off, a
    value that is not a switch raises)."""
    return parse_optional_switch(get_param(ctx, KEEP_ABSOLUTE_SAVE, None), KEEP_ABSOLUTE_SAVE)


def _absolute_save(ctx: PatchContext) -> Optional[str]:
    """``backend_args.save`` (a str or path-like) when it is an absolute path, else None (None:
    unset, the opt-out)."""
    save = getattr(ctx.extra.get("backend_args"), "save", None)
    if not isinstance(save, (str, os.PathLike)):
        return None
    save = os.fspath(save)
    return save if os.path.isabs(save) else None


def routes_save(ctx: PatchContext) -> bool:
    """Patch condition, no side effects, never raises: False only when the run keeps an absolute
    ``save`` (``keep_absolute_save``, the module param read with ``get_param``, on and
    ``backend_args.save`` absolute). Megatron then saves at that path as given (its argument dump
    shows it), so a later run with ``load`` there finds the checkpoint. Otherwise the patch runs:
    it refuses a switch value it cannot read (whatever ``save`` is, ``save: null`` included),
    then keeps a null save, refuses a kept save that is not absolute, or routes the save."""
    try:
        keep = keep_absolute_save(ctx)
    except ValueError:
        return True  # the patch body raises the error, outside patch selection
    return not (keep and _absolute_save(ctx) is not None)


@register_patch(
    "megatron.args.checkpoint_path",
    backend="megatron",
    phase="build_args",
    description="Set checkpoint save path based on experiment root (or keep an absolute one, opt-in)",
    condition=routes_save,
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
      - ``keep_absolute_save_path: true`` (opt-in, any model) and an absolute
            args.save: the patch does not run (condition ``routes_save``) and
            the path is used as given. With the switch on and a save that is
            not absolute (relative, or ``~/...``, which is not expanded) the
            run is refused: the opt-in cannot be honoured, and routing the save
            elsewhere would make a later ``load`` at that path find nothing.
    """
    keep = keep_absolute_save(ctx)  # first: a value that is not a switch raises for every save
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

    if keep:  # on, so the save is not absolute (else the patch is skipped)
        raise ValueError(
            f"{KEEP_ABSOLUTE_SAVE} is on, but save {args.save!r} is not an absolute path (~ is not "
            f"expanded). Give an absolute save, or set {KEEP_ABSOLUTE_SAVE}: false to route the save "
            "to <exp_root>/checkpoints."
        )

    ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))

    if args.save != ckpt_path:
        # the hint only for an absolute save: with a relative one the switch would refuse the run
        hint = f" (set {KEEP_ABSOLUTE_SAVE}: true to keep it)" if os.path.isabs(str(args.save)) else ""
        log_rank_0(
            f"[Patch:megatron.args.checkpoint_path][WARN] "
            f"args.save is deprecated; overriding to: {ckpt_path}{hint}"
        )

    args.save = ckpt_path
    log_rank_0(f"[Patch:megatron.args.checkpoint_path] save → {ckpt_path}")
