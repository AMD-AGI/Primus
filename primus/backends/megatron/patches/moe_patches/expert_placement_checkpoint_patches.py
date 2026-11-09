###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Checkpoint expert-axis keys under a custom expert placement (PR #166 port).

Megatron-core keys distributed-checkpoint expert tensors by global expert id,
computed as ``ep_rank * num_local_experts + local_slot``. With
``moe_expert_placement`` the global ids a rank owns are no longer that
contiguous block, so the three places that derive the id are rewritten to read
the ``local_expert_indices`` attribute the ``megatron.moe.token_dedup`` patch
records on the experts module and its grouped-linear children:

* ``TEGroupedMLP.sharded_state_dict`` (the swiglu fc1 factory offsets),
* ``SequentialMLP.sharded_state_dict``,
* ``TEGroupedLinear._sharded_state_dict_grouped`` (also used by
  ``PrimusTurboGroupedLinear``).

Checkpoints therefore stay keyed by global expert id: a run may load a
checkpoint saved with a different placement (weights are reshuffled to the new
owners by the sharded load). Positional optimizer formats are NOT remapped, so
``load_checkpoint`` warns when a distributed-optimizer state is loaded under a
placement, and a legacy ``torch`` checkpoint (rank-local layout) is refused
unless it was saved with the same placement.
"""

from __future__ import annotations

import os

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_method_source_multi
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PREFIX = "[Patch:megatron.moe.expert_placement_checkpoint]"
_KEY = "megatron.moe.expert_placement_checkpoint"


def _placement_enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_expert_placement", None))


_MLP_ANCHOR_OFFSET = "local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts"
_MLP_REPLACE_OFFSET = (
    "global_expert_ids = resolve_global_expert_ids("
    "getattr(self, 'local_expert_indices', None), self.num_local_experts, self.ep_group.rank())"
)
_TE_ANCHOR_OFFSET = "local_expert_indices_offset = get_pg_rank(self._pg_collection.ep) * self.num_gemms"
_TE_REPLACE_OFFSET = (
    "global_expert_ids = resolve_global_expert_ids("
    "getattr(self, 'local_expert_indices', None), self.num_gemms, get_pg_rank(self._pg_collection.ep))"
)


def _rewrite_expert_keys() -> None:
    from megatron.core.extensions import transformer_engine as te_mod
    from megatron.core.transformer.moe import experts as experts_mod

    from primus.backends.megatron.core.transformer.moe.expert_placement import resolve_global_expert_ids

    # A method can only be source-rewritten once (inspect cannot read an exec'd function).
    if is_patched(experts_mod.TEGroupedMLP, _KEY):
        return
    mark_patched(experts_mod.TEGroupedMLP, _KEY)

    # The rewritten methods are exec'd with their module's globals.
    experts_mod.resolve_global_expert_ids = resolve_global_expert_ids
    te_mod.resolve_global_expert_ids = resolve_global_expert_ids

    patch_method_source_multi(
        experts_mod.TEGroupedMLP,
        "sharded_state_dict",
        [
            (_MLP_ANCHOR_OFFSET, _MLP_REPLACE_OFFSET),
            (
                "(ep_axis, local_expert_indices_offset + i, num_global_experts)",
                "(ep_axis, global_expert_ids[i], num_global_experts)",
            ),
        ],
    )
    log_rank_0(f"{_PREFIX} Rewrote TEGroupedMLP.sharded_state_dict expert-axis offsets")

    patch_method_source_multi(
        experts_mod.SequentialMLP,
        "sharded_state_dict",
        [
            (_MLP_ANCHOR_OFFSET, _MLP_REPLACE_OFFSET),
            (
                "expert_global_idx = local_expert_indices_offset + expert_local_idx",
                "expert_global_idx = global_expert_ids[expert_local_idx]",
            ),
        ],
    )
    log_rank_0(f"{_PREFIX} Rewrote SequentialMLP.sharded_state_dict expert-axis offsets")

    te_grouped_linear = getattr(te_mod, "TEGroupedLinear", None)
    if te_grouped_linear is None:
        warning_rank_0(f"{_PREFIX} TEGroupedLinear not available; grouped-GEMM checkpoint keys not patched")
        return
    patch_method_source_multi(
        te_grouped_linear,
        "_sharded_state_dict_grouped",
        [
            (_TE_ANCHOR_OFFSET, _TE_REPLACE_OFFSET),
            (
                "global_expert_idx = local_expert_indices_offset + gemm_idx",
                "global_expert_idx = global_expert_ids[gemm_idx]",
            ),
        ],
    )
    log_rank_0(f"{_PREFIX} Rewrote TEGroupedLinear._sharded_state_dict_grouped expert-axis offsets")


def _check_legacy_checkpoint_placement(args) -> None:
    """Refuse a legacy ``torch`` checkpoint saved under another placement.

    Legacy checkpoints store expert weights by local EP rank/slot, so loading them
    under a different placement would assign the tensors to different global
    experts. Mirrors ``_check_legacy_checkpoint_expert_placement`` in PR #166.
    """
    from megatron.training import checkpointing as ckpt_mod

    if getattr(args, "ckpt_format", "torch_dist") != "torch" or getattr(
        args, "auto_detect_ckpt_format", False
    ):
        return
    load_dir = getattr(args, "load", None)
    if not load_dir or not ckpt_mod.checkpoint_exists(load_dir):
        return
    requested_path = getattr(args, "moe_expert_placement", None)
    state_dict, _, _, ckpt_type = ckpt_mod._load_base_checkpoint(load_dir, args, rank0=True)
    if state_dict is None or ckpt_type != ckpt_mod.CheckpointType.LEGACY:
        return
    checkpoint_args = state_dict.get("args", None)
    saved_path = getattr(checkpoint_args, "moe_expert_placement", None) or None
    if saved_path is not None and requested_path:
        if os.path.abspath(saved_path) == os.path.abspath(requested_path):
            return
        # Different paths may still hold the same placement.
        try:
            from primus.backends.megatron.core.transformer.moe.expert_placement import (
                load_moe_expert_placement,
            )

            num_experts = getattr(args, "num_experts", None)
            ep_size = int(getattr(args, "expert_model_parallel_size", 1) or 1)
            if load_moe_expert_placement(saved_path, num_experts, ep_size) == load_moe_expert_placement(
                requested_path, num_experts, ep_size
            ):
                return
        except (OSError, ValueError):
            pass
    raise RuntimeError(
        "Cannot safely load a legacy torch checkpoint with a different moe_expert_placement. "
        "Legacy torch checkpoints store expert weights by local EP rank/slot, so changing "
        "placement at load time assigns those tensors to different global experts and can "
        "diverge loss. Load this checkpoint with its original placement, then save a new "
        "checkpoint with ckpt_format: torch_dist before changing expert placement. "
        f"Checkpoint placement: {saved_path if saved_path is not None else '<none>'}; "
        f"requested placement: {requested_path}."
    )


def _wrap_load_checkpoint() -> None:
    from megatron.training import checkpointing as ckpt_mod
    from megatron.training.global_vars import get_args as get_megatron_args

    from primus.backends.megatron.patches._rebind import rebind_everywhere

    if is_patched(ckpt_mod, _KEY):
        return
    orig_load_checkpoint = ckpt_mod.load_checkpoint

    def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, *a, **kw):
        args = get_megatron_args()
        if getattr(args, "moe_expert_placement", None):
            _check_legacy_checkpoint_placement(args)
            if not getattr(args, "no_load_optim", False) and getattr(
                args, "use_distributed_optimizer", False
            ):
                warning_rank_0(
                    f"{_PREFIX} loading distributed-optimizer state with a custom moe_expert_placement. "
                    "Positional optimizer formats restore fp32 main params and Adam moments by buffer "
                    "position, NOT by global expert id -- correct ONLY if this checkpoint was saved with "
                    "the SAME placement. Resuming a differently-placed checkpoint corrupts the model on "
                    "the first step; use no_load_optim: true in that case."
                )
        return orig_load_checkpoint(ddp_model, optimizer, opt_param_scheduler, *a, **kw)

    rebound = rebind_everywhere(ckpt_mod, "load_checkpoint", load_checkpoint)
    mark_patched(ckpt_mod, _KEY)
    log_rank_0(
        f"{_PREFIX} Wrapped load_checkpoint (legacy-format placement check, dist-opt warning) in {rebound}"
    )


@register_patch(
    _KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Key distributed-checkpoint expert tensors by the placed global expert id "
        "(TEGroupedMLP / SequentialMLP / TEGroupedLinear) and guard legacy-format loads "
        "under moe_expert_placement"
    ),
    condition=_placement_enabled,
)
def patch_expert_placement_checkpoint(ctx: PatchContext):
    del ctx
    _rewrite_expert_keys()
    _wrap_load_checkpoint()
