###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 PP tensor-shape patch.

V4's HyperConnections (mHC) residual carries ``K = hc_mult`` parallel
streams per position. Inside a single PP stage the layer hidden has
shape ``[B, S, K, D]``; at the PP boundary the V4 transformer block
folds K into the sequence axis via
:func:`primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block._lower_streams_out`
so the wire tensor is ``[S * K, B, D]`` (plan-2 P15 design — see
``deepseek-v4/develop/plan-2/03-phase-details.md`` P15 and the C1
finding in ``00-review-findings.md``).

Megatron has **two** code paths that compute the PP wire tensor shape,
and they need *both* to know about V4's K packing:

1. The non-interleaved 1F1B schedule (``forward_backward_pipelining_
   without_interleaving``) calls
   :func:`megatron.core.pipeline_parallel.schedules.get_tensor_shapes`
   (``schedules.py:2096-2103``).  Wrapping that function suffices for
   smokes A/B/C.

2. The interleaved 1F1B / VPP schedule (``forward_backward_pipelining_
   with_interleaving``) instead computes ``tensor_shape`` *inline* from
   the ``seq_length`` argument (``schedules.py:1001-1004``):

       tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
       tensor_shape[0] = tensor_shape[0] // cp_group.size()
       if config.sequence_parallel:
           tensor_shape[0] = tensor_shape[0] // tp_group.size()

   It does **not** call ``get_tensor_shapes``. With VPP=2 and ``hc_mult=
   4`` this leaves the recv buffer at ``[S, B, hidden]`` while the
   sender emits ``[S*K, B, hidden]``. PyTorch P2P does not validate
   shape (only ``numel * dtype_size``), so the receiver silently copies
   only the first ``S * hidden`` elements, ``_lift_streams_in``
   reshapes them as ``[B, S/K, K, D]``, and the resulting hidden
   flattens to ``S/K = 32`` instead of ``S = 128`` — :class:`Deepseek
   V4HashRouter` then trips its precondition with ``hidden=32 vs
   token_ids=128``. (P19 smoke D run reproduces this cleanly.)

We therefore install **two** complementary wrappers, both gated on the
V4 ``model_type`` + ``hc_mult > 1`` + ``PP > 1`` condition:

* :func:`_make_v4_get_tensor_shapes` multiplies the first (seq) dim of
  every tuple returned by ``get_tensor_shapes`` by ``hc_mult``, fixing
  path (1).
* :func:`_make_v4_interleaved_schedule` wraps
  ``forward_backward_pipelining_with_interleaving`` to scale its
  ``seq_length`` keyword argument by ``hc_mult`` before the schedule
  computes its inline ``tensor_shape``, fixing path (2).  Inside the
  interleaved schedule, ``seq_length`` is *only* read for that inline
  shape (see grep on ``schedules.py``), so scaling it is a no-op for
  every other concern (cudagraph, attention masking, etc.).

The companion ``adjust_tensor_shapes_fn`` parameter on the
non-interleaved 1F1B schedule is intentionally not used because
upstream Megatron explicitly asserts it is unsupported by the
interleaved / VPP schedules (``schedules.py:900-901``); going through
``get_tensor_shapes`` and ``seq_length`` keeps the behavior uniform
across all schedules.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _make_v4_get_tensor_shapes(original_fn, hc_mult: int):
    """Return a wrapper that scales the first (seq) dim by ``hc_mult``.

    ``original_fn`` returns a list of ``(seq_length, micro_batch, hidden)``
    triples; the V4 PP wire packs ``K`` streams into the seq axis so we
    multiply the seq dim only — micro_batch and hidden are unchanged.
    """

    def patched_get_tensor_shapes(*args, **kwargs):
        shapes = original_fn(*args, **kwargs)
        return [(s * hc_mult, b, h) for (s, b, h) in shapes]

    patched_get_tensor_shapes.__wrapped__ = original_fn
    patched_get_tensor_shapes._v4_pp_shape_patched = True
    return patched_get_tensor_shapes


def _make_v4_interleaved_schedule(original_fn, hc_mult: int):
    """Wrap the interleaved schedule to scale ``seq_length`` by ``hc_mult``.

    The interleaved schedule (``forward_backward_pipelining_with_
    interleaving``) computes its PP wire ``tensor_shape`` inline from
    the ``seq_length`` kwarg (``schedules.py:1001``). Inside that
    function, ``seq_length`` is consumed only by that inline
    computation, so we can safely scale it on the way in to give the
    schedule a V4-aware shape without touching any other behaviour.
    """

    def patched_schedule(*args, **kwargs):
        if "seq_length" in kwargs and kwargs["seq_length"] is not None:
            kwargs["seq_length"] = int(kwargs["seq_length"]) * hc_mult
        return original_fn(*args, **kwargs)

    patched_schedule.__wrapped__ = original_fn
    patched_schedule._v4_pp_interleaved_patched = True
    return patched_schedule


@register_patch(
    "megatron.deepseek_v4.pp_tensor_shape",
    backend="megatron",
    phase="before_train",
    description=(
        "DeepSeek-V4: pack hc_mult=K hyper-streams into the PP wire seq "
        "axis so [S*K, B, D] passes between PP stages (covers both the "
        "1F1B get_tensor_shapes path and the interleaved-1F1B / VPP "
        "inline tensor_shape path)."
    ),
    condition=lambda ctx: (
        getattr(get_args(ctx), "model_type", None) == "deepseek_v4"
        and int(getattr(get_args(ctx), "hc_mult", 1) or 1) > 1
        and int(getattr(get_args(ctx), "pipeline_model_parallel_size", 1) or 1) > 1
    ),
)
def patch_v4_pp_tensor_shape(ctx: PatchContext):
    """Multiply the PP P2P seq dim by ``hc_mult`` for V4 models."""
    import megatron.core.pipeline_parallel.schedules as schedules_module

    hc_mult = int(getattr(get_args(ctx), "hc_mult", 1))

    # Wrapper 1: get_tensor_shapes (used by the non-interleaved schedule).
    original_get_tensor_shapes = schedules_module.get_tensor_shapes
    if getattr(original_get_tensor_shapes, "_v4_pp_shape_patched", False):
        log_rank_0(
            "[Patch:megatron.deepseek_v4.pp_tensor_shape] get_tensor_shapes "
            "already patched, skip"
        )
    else:
        schedules_module.get_tensor_shapes = _make_v4_get_tensor_shapes(
            original_get_tensor_shapes, hc_mult
        )
        log_rank_0(
            f"[Patch:megatron.deepseek_v4.pp_tensor_shape] wrapped "
            f"get_tensor_shapes; PP wire seq_len * hc_mult={hc_mult} "
            "(packs K hyper-streams into the sequence axis)."
        )

    # Wrapper 2: forward_backward_pipelining_with_interleaving (VPP).
    # The interleaved schedule reads ``seq_length`` directly to build its
    # inline ``tensor_shape``; scaling the kwarg gives it the V4 wire shape
    # without rewriting the function.
    original_interleaved = schedules_module.forward_backward_pipelining_with_interleaving
    if getattr(original_interleaved, "_v4_pp_interleaved_patched", False):
        log_rank_0(
            "[Patch:megatron.deepseek_v4.pp_tensor_shape] interleaved "
            "schedule already patched, skip"
        )
    else:
        schedules_module.forward_backward_pipelining_with_interleaving = (
            _make_v4_interleaved_schedule(original_interleaved, hc_mult)
        )
        log_rank_0(
            f"[Patch:megatron.deepseek_v4.pp_tensor_shape] wrapped "
            f"forward_backward_pipelining_with_interleaving; "
            f"seq_length * hc_mult={hc_mult} on the way into the "
            "interleaved-1F1B / VPP schedule."
        )
