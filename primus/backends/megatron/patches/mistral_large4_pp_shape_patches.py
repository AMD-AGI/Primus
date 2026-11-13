###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``model_type: mistral_large4``) PP tensor-shape patch.

With cross-layer KV sharing (``kv_source_layers``) the consumer layers of a source layer's
latent may sit on later pipeline stages, so the decoder block sends ``[s + n, b, h]`` between
stages: the hidden states, then the latent packed densely into ``n = kv_carrier_rows(s,
kv_lora_rank, h)`` rows (171 rows for s 2048 at the release's 512 / 6144, +8% of the wire). This
patch adds those rows to the P2P shape of the 1F1B schedule (``get_tensor_shapes``, whose
sequence dim is already the stage-local one) and of the interleaved (VPP) schedule (its
``seq_length``, which it divides by CP and, with sequence parallelism, TP). Only for this
model, only with PP > 1 and KV sharing.
"""

import functools

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    MODEL_TYPE,
    is_mistral_large4_args,
    kv_carrier_needed,
    kv_carrier_rows,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# set on the wrappers this patch installs (get_tensor_shapes and the interleaved schedule) to tell
# them from another model's
_OWNER_MARK = "_primus_mistral_large4_pp_carrier"
# a model-neutral mark on these wrappers (its value names the model): a patch that also changes
# the PP wire shape can set it on its wrappers so each refuses the other
WIRE_OWNER_ATTR = "_primus_pp_wire_owner"
# attributes other models' PP shape patches set on their wrappers: Kimi K3's get_tensor_shapes
# (kimi_k3_pp_shape_patches.py) and DeepSeek-V4's get_tensor_shapes / interleaved schedule
# (deepseek_v4_pp_shape_patches.py). A wrapper with none of them and no WIRE_OWNER_ATTR (profiling,
# NVTX, a functools.wraps decorator) does not change the shape
_FOREIGN_WIRE_ATTRS = ("_k3_pp_seq_mult", "_v4_pp_shape_patched", "_v4_pp_interleaved_patched")


# how far the ``__wrapped__`` chain is followed: a guard against a cyclic chain; real stacks of
# wrappers (profiling, NVTX, patches) are a few deep
_MAX_WRAPPER_DEPTH = 64


def _wrapper_chain(fn):
    """``fn`` and the functions it wraps (``__wrapped__``), outermost first."""
    chain = []
    while fn is not None and len(chain) < _MAX_WRAPPER_DEPTH:
        chain.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return chain


def _is_foreign_wire(f) -> bool:
    """``f`` carries another model's wire mark: a known foreign attribute, or the neutral owner
    attribute naming another model. (``functools.wraps`` copies a wrapped function's marks onto its
    wrapper, so a benign wrapper over ours carries our marks, and a wrapper over ours from another
    wire patch carries both: the foreign mark decides.)"""
    if any(hasattr(f, a) for a in _FOREIGN_WIRE_ATTRS):
        return True
    owner = getattr(f, WIRE_OWNER_ATTR, None)
    return owner is not None and owner != MODEL_TYPE


def _wire_state(fn) -> str:
    """``"foreign"`` if another wire-changing patch wrapped ``fn`` anywhere in its wrapper chain
    (checked first: also when it wrapped ours), else ``"ours"`` if this patch did, else ``"free"``."""
    chain = _wrapper_chain(fn)
    if any(_is_foreign_wire(f) for f in chain):
        return "foreign"
    if any(getattr(f, _OWNER_MARK, False) for f in chain):
        return "ours"
    return "free"


def _wants_mistral_large4_pp_shape_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    # this model only: args.model_type (is_mistral_large4_args)
    return is_mistral_large4_args(args) and kv_carrier_needed(
        getattr(args, "kv_source_layers", None),
        int(getattr(args, "num_layers", 0) or 0),
        getattr(args, "pipeline_model_parallel_size", 1),
        getattr(args, "no_rope_layers", None),
    )


def make_get_tensor_shapes(original, kv_lora_rank: int):
    """``get_tensor_shapes`` whose ``(s, b, h)`` shapes get the latent's rows: ``s + kv_carrier_rows``."""

    @functools.wraps(original)
    def get_tensor_shapes(*args, **kwargs):
        shapes = original(*args, **kwargs)
        # [()] with variable_seq_lengths: the sender transmits the real shape, nothing to change
        return [
            (
                shape
                if len(shape) == 0
                else (shape[0] + kv_carrier_rows(shape[0], kv_lora_rank, shape[2]),) + tuple(shape[1:])
            )
            for shape in shapes
        ]

    setattr(get_tensor_shapes, _OWNER_MARK, True)
    setattr(get_tensor_shapes, WIRE_OWNER_ATTR, MODEL_TYPE)
    return get_tensor_shapes


def make_interleaved_schedule(original, kv_lora_rank: int, hidden_size: int, sequence_parallel: bool):
    """The interleaved schedule builds its P2P shape as ``seq_length // cp (// tp with SP)``: it is
    given the ``seq_length`` whose division is the stage-local ``s + kv_carrier_rows(s)``.
    ``_cp_tp(kwargs)`` gives ``(cp, tp)`` at call time."""

    @functools.wraps(original)
    def schedule(*args, **kwargs):
        if kwargs.get("seq_length") is not None:
            cp, tp = _cp_tp(kwargs)
            div = cp * (tp if sequence_parallel else 1)
            s = int(kwargs["seq_length"]) // div
            kwargs["seq_length"] = (s + kv_carrier_rows(s, kv_lora_rank, hidden_size)) * div
        return original(*args, **kwargs)

    setattr(schedule, _OWNER_MARK, True)
    setattr(schedule, WIRE_OWNER_ATTR, MODEL_TYPE)
    return schedule


def _cp_tp(kwargs):
    """The (cp, tp) sizes the interleaved schedule divides by: its pg_collection's, else Megatron's."""
    pgs = kwargs.get("pg_collection")
    if pgs is not None:
        return pgs.cp.size(), pgs.tp.size()
    from megatron.core import parallel_state

    return (
        parallel_state.get_context_parallel_world_size(),
        parallel_state.get_tensor_model_parallel_world_size(),
    )


@register_patch(
    "megatron.mistral_large4.pp_tensor_shape",
    backend="megatron",
    phase="before_train",
    description="Mistral Large 4: carry the shared KV latent with the hidden states between PP stages.",
    condition=_wants_mistral_large4_pp_shape_patch,
)
def patch_mistral_large4_pp_tensor_shape(ctx: PatchContext):
    import megatron.core.pipeline_parallel.schedules as schedules_module

    args = get_args(ctx)
    for flag in ("patch_zero_bubble", "patch_primus_pipeline"):
        if getattr(args, flag, False):
            raise NotImplementedError(
                f"Mistral Large 4 KV sharing with pipeline parallelism does not support {flag}"
            )
    names = ("get_tensor_shapes", "forward_backward_pipelining_with_interleaving")
    states = {n: _wire_state(getattr(schedules_module, n)) for n in names}
    foreign = [n for n, st in states.items() if st == "foreign"]
    if foreign:
        # another model's wire (e.g. Kimi K3's): this model's rows would be added to the wrong shape
        raise RuntimeError(
            f"Mistral Large 4 PP wire: {', '.join(foreign)} already wrapped by another patch that changes "
            "the PP wire shape in this process"
        )
    if all(st == "ours" for st in states.values()):
        log_rank_0("[Patch:megatron.mistral_large4.pp_tensor_shape] PP wire already sized; skipping")
        return
    kv_lora_rank = int(args.kv_lora_rank)
    if states["get_tensor_shapes"] == "free":
        schedules_module.get_tensor_shapes = make_get_tensor_shapes(
            schedules_module.get_tensor_shapes, kv_lora_rank
        )
    if states["forward_backward_pipelining_with_interleaving"] == "free":
        schedules_module.forward_backward_pipelining_with_interleaving = make_interleaved_schedule(
            schedules_module.forward_backward_pipelining_with_interleaving,
            kv_lora_rank,
            int(args.hidden_size),
            bool(getattr(args, "sequence_parallel", False)),
        )
    log_rank_0(
        "[Patch:megatron.mistral_large4.pp_tensor_shape] PP wire seq_len + ceil(seq_len * "
        f"{kv_lora_rank} / {int(args.hidden_size)}) rows (hidden + shared KV latent)"
    )
