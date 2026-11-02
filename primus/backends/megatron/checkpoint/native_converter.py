###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Template for the Primus native (bridge-free) HF -> Megatron converters.

:meth:`NativeConverter.convert` fixes the steps every converter takes; a family
subclass fills in the model-specific ones:

1. read ``config.json`` (:func:`native_convert_common.read_hf_config`);
2. derive the Megatron fields from it (:meth:`derive_fields`, abstract) and build the
   Megatron args (:meth:`build_margs`, abstract);
3. initialise a single-rank Megatron and build the mcore model with the Primus model
   provider (:meth:`build_model`; default: the ``gpt`` provider);
4. map the HF tensors onto the model's state dict (:meth:`build_state_dict`, abstract);
5. check the mapping is exact: no unexpected or missing model keys, no unconsumed HF
   tensors (:func:`native_convert_common.check_exact_mapping`);
6. save a ``torch_dist`` checkpoint from this single process (:func:`save_torch_dist`).
   A torch_dist checkpoint carries global tensor shapes and offsets, so Megatron
   reshards it on load: the converted TP=PP=EP=1 checkpoint loads into a run with any
   TP / PP / EP.

The llama / qwen3 / deepseek converters keep their own ``convert`` functions (legacy
``torch`` checkpoints); new families subclass this.
"""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

from primus.backends.megatron.checkpoint import native_convert_common as common


def save_torch_dist(model, margs, save_dir, log=print):
    """Save ``model`` as a torch_dist checkpoint (iteration ``margs.iteration``), no optimizer.

    Runs in the single conversion process; the files hold global shapes and offsets, so
    Megatron reshards them for the training layout on load. ``latest_checkpointed_iteration.txt``
    is written as for a training save (the hook's skip-if-exists check reads it).
    """
    from megatron.training.checkpointing import save_checkpoint

    margs.ckpt_format = "torch_dist"
    margs.use_dist_ckpt = True
    margs.async_save = False
    margs.save = save_dir
    margs.no_save_optim = True
    margs.no_save_rng = True
    margs.ckpt_fully_parallel_save = False
    os.makedirs(save_dir, exist_ok=True)
    log(f"saving torch_dist checkpoint to {save_dir} ...")
    save_checkpoint(margs.iteration, [model], None, None, num_floating_point_operations_so_far=0)
    log(f"DONE. Converted checkpoint at: {save_dir}")


def existing_entries(save_dir):
    """The names in ``save_dir`` before a save (None when it does not exist): the
    ``existed`` argument of :func:`remove_partial_checkpoint`."""
    return set(os.listdir(save_dir)) if os.path.isdir(save_dir) else None


def remove_partial_checkpoint(save_dir, existed, log=print) -> None:
    """After a failed save: drop what the save wrote, so no partial checkpoint (with a
    ``latest_checkpointed_iteration.txt`` that the hook's skip-if-exists check reads) stays.

    A directory this run created (``existed`` None / False) is removed whole. A directory
    that was already there -- even an empty one (``existed == set()``: a ``mkdir``-ed output
    dir or a mount point) -- is kept: the tracker is removed, and so are the entries the
    save added (``existed`` is the set of names before it, :func:`existing_entries`), e.g. a
    new ``iter_*`` dir of partial shards; the earlier entries may be the user's and stay
    (``existed=True``: only the tracker).
    """
    if not os.path.exists(save_dir):
        return
    # the tracker first, on its own and checked: it is what makes a checkpoint look complete
    tracker = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    try:
        if os.path.lexists(tracker):
            os.remove(tracker)
    except OSError as e:
        log(
            f"conversion failed and the tracker {tracker} could not be removed ({e}): delete it "
            f"(or {save_dir}) before a relaunch, or the incomplete checkpoint is taken as complete"
        )
        return
    if existed is not None and existed is not False:  # an empty set is a pre-existing dir
        added = sorted(set(os.listdir(save_dir)) - existed) if isinstance(existed, (set, frozenset)) else []
        errors = []
        for name in added:
            path = os.path.join(save_dir, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as e:
                errors.append(f"{path}: {e}")
        kept = f"; could not remove {len(errors)}: {errors[0]}" if errors else ""
        n = len(added) - len(errors)
        log(
            f"conversion failed: removed {tracker} and the {n} new "
            f"entr{'y' if n == 1 else 'ies'} the save wrote in {save_dir}{kept} "
            "(entries that were there before are kept; the checkpoint is incomplete)"
        )
        return
    errors = []
    shutil.rmtree(save_dir, onerror=lambda _f, path, exc: errors.append(f"{path}: {exc[1]}"))
    if errors or os.path.exists(save_dir):
        log(
            f"conversion failed: removed the tracker {tracker}, but not all of the partial "
            f"checkpoint {save_dir} ({len(errors)} error(s), first: {errors[0] if errors else '?'}); "
            "it has no tracker, so it is not taken as complete"
        )
    else:
        log(f"conversion failed: removed the partial checkpoint {save_dir}")


class NativeConverter:
    """Base class of a native converter; subclasses implement the abstract steps.

    ``family`` names the converter in log lines; ``self.log`` is the log function.
    :meth:`convert` passes the per-run options (``args``: hf_dir, save_dir, dtype, parallel
    sizes, ...) to the steps as an argument; ``build_margs`` and ``check_config`` take
    ``(fields, args, cfg)`` in that order (see each step's signature for the others).
    """

    family = "native"
    # The rendezvous port of the single-process build (one place: ``convert(master_port=None)``
    # falls back to it). Distinct from the llama / qwen3 (29563) and deepseek (29561) loaders.
    master_port = 29571

    def __init__(self, log=None):
        if log is None:
            prefix = f"[{self.family}-convert]"

            def log(*a):
                print(prefix, *a, flush=True)

        self.log = log

    # ---- abstract steps -------------------------------------------------------------
    def derive_fields(self, cfg: dict) -> dict:
        """Megatron model fields from the HF ``config.json`` dict."""
        raise NotImplementedError

    def build_margs(self, fields: dict, args: SimpleNamespace, cfg: dict):
        """The Megatron args namespace for the single-process build."""
        raise NotImplementedError

    def build_state_dict(self, model, store, margs, fields: dict):
        """``(state_dict, consumed_hf_keys)``: the HF tensors mapped onto ``model``'s keys."""
        raise NotImplementedError

    # ---- overridable steps ----------------------------------------------------------
    def check_config(self, fields: dict, args: SimpleNamespace, cfg: dict) -> None:
        """Refuse configs the converter cannot handle (default: accept)."""

    def open_store(self, hf_dir: str):
        """The HF tensor store whose ``all_keys`` the exact-mapping check covers."""
        return common.SafetensorsStore(hf_dir)

    def build_model(self, margs):
        """The mcore model from the Primus model provider (default: ``gpt``)."""
        return common.build_mcore_gpt_model(margs)

    def check_model(self, model, margs) -> None:
        """Checks on the built model before mapping (default: none)."""

    def save(self, model, margs, save_dir) -> None:
        save_torch_dist(model, margs, save_dir, self.log)

    # ---- template -------------------------------------------------------------------
    def convert(
        self,
        hf_dir,
        save_dir,
        *,
        dtype="bf16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=4096,
        master_port=None,
        **extra,
    ):
        """Convert ``hf_dir`` (an HF snapshot) into a torch_dist checkpoint at ``save_dir``."""
        common.ensure_megatron_on_path()
        args = SimpleNamespace(
            hf_dir=hf_dir,
            save_dir=save_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            seq_length=seq_length,
            **extra,
        )
        cfg = common.read_hf_config(hf_dir)  # 1
        fields = self.derive_fields(cfg)  # 2
        self.check_config(fields, args, cfg)
        margs = self.build_margs(fields, args, cfg)
        margs = common.init_megatron_single_process(margs, master_port or self.master_port, log=self.log)  # 3
        model = self.build_model(margs)
        self.check_model(model, margs)
        store = self.open_store(hf_dir)
        sd, consumed = self.build_state_dict(model, store, margs, fields)  # 4
        common.check_exact_mapping(model, sd, consumed, store, self.log)  # 5
        del sd
        # the bridge-free guard runs before anything is written: a failed check leaves no
        # checkpoint that the hook's skip-if-exists test would accept later
        common.assert_bridge_free("native conversion")
        existed = existing_entries(save_dir)
        try:
            self.save(model, margs, save_dir)  # 6
            common.assert_bridge_free("native conversion (save)")
        except BaseException:
            remove_partial_checkpoint(save_dir, existed, self.log)
            raise
        return save_dir
