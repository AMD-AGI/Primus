###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: keep the requested checkpoint save dir loadable.

``megatron.args.checkpoint_path`` (priority 50) routes ``args.save`` to
``<exp_root>/checkpoints``. A later ``--load <requested save dir>`` (save, then fine-tune
from it) then finds nothing. This patch runs first (priority 40) and, when the requested
dir does not exist yet, makes it a symlink to ``<exp_root>/checkpoints``; when it exists
and already resolves there (e.g. the link of an earlier run of the same experiment)
nothing is done; when it exists and does not resolve there, it warns that ``--load
<requested dir>`` will not see this run's checkpoints.

Scope: the rewrite of ``args.save`` affects every Megatron model, but this port must not
change the behaviour of the others (the shared patch rule), so the link is applied for
Muse Spark only; a framework-wide fix belongs in ``megatron.args.checkpoint_path``.
"""

import os

from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_TAG = "[Patch:megatron.muse_spark.link_save_dir]"


def link_requested_save_dir(requested: str, ckpt_path: str) -> bool:
    """Make the requested save dir a symlink to ``ckpt_path`` so a later ``--load <requested>``
    finds the checkpoint. Returns True when ``requested`` resolves to ``ckpt_path`` after the
    call (made now or before), else False with a warning that says why: an existing dir or
    file is left alone, and a symlink to another target (e.g. left by an earlier experiment,
    which a later ``--load <requested>`` would restore) is not replaced.

    Every rank may call it: the link is made with one ``os.symlink`` and a rank that loses
    the race (``FileExistsError``) only checks the link the winner made.
    """
    requested = os.path.abspath(requested)
    if not os.path.lexists(requested):
        try:
            os.makedirs(ckpt_path, exist_ok=True)
            os.makedirs(os.path.dirname(requested), exist_ok=True)
        except OSError as exc:  # e.g. a file where a dir is needed
            log_rank_0(f"{_TAG}[WARN] cannot link {requested} -> {ckpt_path}: {exc}")
            return False
        try:
            os.symlink(ckpt_path, requested, target_is_directory=True)
            log_rank_0(f"{_TAG} {requested} -> {ckpt_path} (symlink, for --load)")
            return True
        except FileExistsError:
            pass  # another rank made it first: check it below
        except OSError as exc:
            log_rank_0(f"{_TAG}[WARN] cannot link {requested} -> {ckpt_path}: {exc}")
            return False
    if os.path.realpath(requested) == os.path.realpath(ckpt_path):
        return True
    if os.path.islink(requested):
        why = f"is a symlink to {os.path.realpath(requested)} (an earlier experiment?), not to this run's"
    else:
        why = "already exists (not a symlink), so it is not this run's"
    log_rank_0(
        f"{_TAG}[WARN] the requested save dir {requested} {why} checkpoint dir {ckpt_path}; "
        f"this run saves to {ckpt_path}, so load from there (a later --load {requested} "
        "does not see this run's checkpoints)"
    )
    return False


@register_patch(
    "megatron.muse_spark.link_save_dir",
    backend="megatron",
    phase="build_args",
    description="Muse Spark: symlink the requested save dir to <exp_root>/checkpoints (for a later --load).",
    condition=is_muse_spark_model_type,
    priority=40,  # before megatron.args.checkpoint_path (50), which rewrites args.save (a test checks it)
)
def patch_muse_spark_link_save_dir(ctx: PatchContext):
    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})
    if not args or not getattr(primus_config, "exp_root_path", None) or getattr(args, "save", None) is None:
        return
    # the value megatron.args.checkpoint_path sets (that framework patch is not gated by
    # model, so it is not edited here to share a helper)
    ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))
    if os.path.abspath(args.save) != ckpt_path:
        link_requested_save_dir(args.save, ckpt_path)
