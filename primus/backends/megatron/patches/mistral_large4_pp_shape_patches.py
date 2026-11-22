###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``model_type: mistral_large4``) PP tensor-shape patch.

With cross-layer KV sharing (``kv_source_layers``) the consumer layers of a source layer's
latent may sit on later pipeline stages, so the decoder block sends ``[s + n, b, h]`` between
stages: the hidden states, then the latent packed densely into ``n = kv_carrier_rows(s,
kv_lora_rank, h)`` rows (``ceil(s * kv_lora_rank / h)``). This
patch adds those rows to the P2P shape of the 1F1B schedule (``get_tensor_shapes``, whose
sequence dim is already the stage-local one) and of the interleaved (VPP) schedule (its
``seq_length``, which it divides by CP and, with sequence parallelism, TP). Only for this
model, only with PP > 1 and KV sharing.
"""

import functools
import sys

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    MODEL_TYPE,
    is_mistral_large4_args,
    kv_carrier_needed,
    kv_carrier_rows,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# a model-neutral mark on the wrappers this patch installs (get_tensor_shapes and the interleaved
# schedule); its value names the model (MODEL_TYPE): it tells ours from another model's, and a patch
# that also changes the PP wire shape can set it on its wrappers so each refuses the other
WIRE_OWNER_ATTR = "_primus_pp_wire_owner"
# the values a wrapper of this patch was built with (and ``__wrapped__``, set by functools.wraps,
# is the function it wraps): a second application with other values rebuilds the wrapper from
# the unwrapped function instead of keeping the stale values
WIRE_PARAMS_ATTR = "_primus_mistral_large4_wire_params"
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
    if any(getattr(f, WIRE_OWNER_ATTR, None) == MODEL_TYPE for f in chain):
        return "ours"
    return "free"


def _mark_params(wrapper, params: dict) -> None:
    """Record on ``wrapper`` the values it was built with, with the wrapper itself: functools.wraps
    copies the attributes onto an outer wrapper, whose copy then names another function."""
    setattr(wrapper, WIRE_PARAMS_ATTR, (wrapper, dict(params)))


def _is_our_wrapper(f) -> bool:
    """``f`` itself is a wrapper this patch built (not an outer wrapper carrying a copy of its marks)."""
    mark = getattr(f, WIRE_PARAMS_ATTR, None)
    return isinstance(mark, tuple) and mark[0] is f


def _wire_params(fn):
    """The values this patch's wrapper in ``fn``'s chain was built with, None without one."""
    for f in _wrapper_chain(fn):
        if _is_our_wrapper(f):
            return getattr(f, WIRE_PARAMS_ATTR)[1]
    return None


def _wants_mistral_large4_pp_shape_patch(ctx: PatchContext) -> bool:
    # this model only: args.model_type (is_mistral_large4_args, in _run_needs_carrier)
    return _run_needs_carrier(get_args(ctx))


def _carrier_needed_by_lists(args) -> bool:
    """``args``'s pipeline and per-layer lists make the P2P tensor carry the shared KV latent."""
    return kv_carrier_needed(
        getattr(args, "kv_source_layers", None),
        int(getattr(args, "num_layers", 0) or 0),
        getattr(args, "pipeline_model_parallel_size", 1),
        getattr(args, "no_rope_layers", None),
    )


def _run_needs_carrier(args) -> bool:
    """The run of ``args`` is a Mistral Large 4 run whose P2P tensor carries the shared KV latent."""
    return is_mistral_large4_args(args) and _carrier_needed_by_lists(args)


def _current_run_needs_carrier() -> bool:
    """Whether the run calling a wrapper now needs the carrier rows, decided at call time from
    Megatron's global args (the same namespace as the Primus params: Megatron's parse_args returns
    them). The wrappers are module-level and stay installed after their run, so a later run in the
    same process (a test session, a sweep) must decide from its own args.

    ``model_type`` is not checked here: Megatron's ``get_model`` replaces it with a ``ModelType`` enum
    before the first P2P call. The per-layer lists decide instead: only this model sets
    ``kv_source_layers``, so another model's run (no lists) or a run with PP 1 gets Megatron's own
    shapes. Without global args (none set up in this process) nothing is added. The module is read
    from ``sys.modules``: a run that set global args has imported it."""
    args = _global_args()
    return args is not None and _carrier_needed_by_lists(args)


def _global_args():
    """Megatron's global args, None when this process has set none (Megatron not imported, or its
    args not initialised)."""
    global_vars = sys.modules.get("megatron.training.global_vars")
    if global_vars is None:
        return None
    try:
        return global_vars.get_args()
    except AssertionError:  # Megatron's "args is not initialized"
        return None


def make_get_tensor_shapes(original, kv_lora_rank: int):
    """``get_tensor_shapes`` whose ``(s, b, h)`` shapes get the latent's rows: ``s + kv_carrier_rows``
    (Megatron's shapes when the current run does not carry the latent: ``_current_run_needs_carrier``)."""

    @functools.wraps(original)
    def get_tensor_shapes(*args, **kwargs):
        shapes = original(*args, **kwargs)
        if not _current_run_needs_carrier():
            return shapes
        # [()] with variable_seq_lengths: the sender transmits the real shape, nothing to change
        return [
            (
                shape
                if len(shape) == 0
                else (shape[0] + kv_carrier_rows(shape[0], kv_lora_rank, shape[2]),) + tuple(shape[1:])
            )
            for shape in shapes
        ]

    setattr(get_tensor_shapes, WIRE_OWNER_ATTR, MODEL_TYPE)
    _mark_params(get_tensor_shapes, {"kv_lora_rank": kv_lora_rank})
    return get_tensor_shapes


def make_interleaved_schedule(original, kv_lora_rank: int, hidden_size: int, sequence_parallel: bool):
    """The interleaved schedule builds its P2P shape as ``seq_length // cp (// tp with SP)``: it is
    given the ``seq_length`` whose division is the stage-local ``s + kv_carrier_rows(s)``.
    ``_cp_tp(kwargs)`` gives ``(cp, tp)`` at call time, and sequence parallelism and the hidden
    size are read at call time from the model's config, as the schedule reads them (and as the
    1F1B wrapper gets ``h`` from the schedule's shape); ``sequence_parallel`` and ``hidden_size``
    are the fallbacks when the model has no config. Megatron's schedule takes its arguments by keyword only
    (``seq_length`` included); a call without ``seq_length`` is refused rather than left at the
    plain size the decoder block does not send."""

    @functools.wraps(original)
    def schedule(*args, **kwargs):
        if not _current_run_needs_carrier():
            return original(*args, **kwargs)
        if kwargs.get("seq_length") is None:
            raise TypeError(
                "Mistral Large 4 PP wire: the interleaved schedule was called without seq_length as a "
                "keyword; its P2P shape cannot be sized for the shared KV latent"
            )
        cp, tp = _cp_tp(kwargs)
        model = kwargs.get("model")
        sp = bool(_model_config_value(model, "sequence_parallel", sequence_parallel))
        h = int(_model_config_value(model, "hidden_size", hidden_size))
        div = cp * (tp if sp else 1)
        s = int(kwargs["seq_length"]) // div
        kwargs["seq_length"] = (s + kv_carrier_rows(s, kv_lora_rank, h)) * div
        return original(*args, **kwargs)

    setattr(schedule, WIRE_OWNER_ATTR, MODEL_TYPE)
    _mark_params(
        schedule,
        {"kv_lora_rank": kv_lora_rank, "hidden_size": hidden_size, "sequence_parallel": sequence_parallel},
    )
    return schedule


def _model_config_value(model, name: str, default):
    """``config.<name>`` of the (first) model chunk, as the interleaved schedule reads its config
    (``get_model_config``); ``default`` when the model has no config or the config lacks it."""
    chunk = model[0] if isinstance(model, (list, tuple)) and model else model
    if chunk is None:
        return default
    try:
        from megatron.core.utils import get_model_config

        value = getattr(get_model_config(chunk), name)
    except (AttributeError, ImportError, RuntimeError):
        return default
    return default if value is None else value


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
    kv_lora_rank = int(args.kv_lora_rank)
    hidden_size = int(args.hidden_size)
    wanted = {
        "get_tensor_shapes": {"kv_lora_rank": kv_lora_rank},
        "forward_backward_pipelining_with_interleaving": {
            "kv_lora_rank": kv_lora_rank,
            "hidden_size": hidden_size,
            "sequence_parallel": bool(getattr(args, "sequence_parallel", False)),
        },
    }
    makers = {
        "get_tensor_shapes": lambda fn, p: make_get_tensor_shapes(fn, p["kv_lora_rank"]),
        "forward_backward_pipelining_with_interleaving": lambda fn, p: make_interleaved_schedule(
            fn, p["kv_lora_rank"], p["hidden_size"], p["sequence_parallel"]
        ),
    }
    changed = False
    for n in names:
        fn = getattr(schedules_module, n)
        if states[n] == "ours":
            params = _wire_params(fn)
            if params == wanted[n]:
                continue
            if not _is_our_wrapper(fn):
                # ours is under another wrapper: it cannot be rebuilt without dropping that wrapper
                raise RuntimeError(
                    f"Mistral Large 4 PP wire: {n} was sized with {params} under another wrapper; "
                    f"this run needs {wanted[n]}"
                )
            fn = fn.__wrapped__  # rebuild from the unwrapped function with this run's values
        setattr(schedules_module, n, makers[n](fn, wanted[n]))
        changed = True
    if not changed:
        log_rank_0("[Patch:megatron.mistral_large4.pp_tensor_shape] PP wire already sized; skipping")
        return
    log_rank_0(
        "[Patch:megatron.mistral_large4.pp_tensor_shape] PP wire seq_len + ceil(seq_len * "
        f"{kv_lora_rank} / {hidden_size}) rows (hidden + shared KV latent)"
    )
