###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 PP tensor-shape patch.

Kimi K3 replaces the ordinary ``x = x + sublayer(x)`` residual with a
softmax mixture over a *growing* set of cross-layer checkpoints, so two
tensors flow between layers -- ``(hidden_states, block_residual)`` -- while
pipeline P2P carries exactly one ``[s, b, h]`` tensor.  Inside a stage the
second tensor is an ordinary second return value threaded by the block's
python loop (``kimi_k3_block.py``); at the stage
boundary it is folded into the sequence axis by
:func:`~primus.backends.megatron.core.models.kimi_k3.kimi_k3_block._lower_res_out`
and unfolded by ``_lift_res_in`` (``kimi_k3_block.py``).  The wire
format is therefore::

    [(1 + attn_res_num_blocks_max) * s, b, h]

a 3-D tensor of **constant** shape, which is all standard PP P2P kernels
need.  ``attn_res_num_blocks_max = ceil(num_layers / attn_res_block_size)``
(``kimi_k3_transformer_config.py``); the unused slots are
zero-padded by the sender and sliced off by the receiver using
``attn_res_num_blocks_before(layer_offset, block_size)``, so no fill count
crosses the boundary.

This is the same trick DeepSeek-V4 uses for its ``hc_mult = K`` parallel
HyperConnection streams, and this module is modelled on
``deepseek_v4_pp_shape_patches.py``.  Two differences:

* V4's ``hc_mult`` is a flat build-time constant read straight off ``args``.
  K3's multiplier has to be *derived* from ``num_layers`` and
  ``attn_res_block_size``, because ``attn_res_num_blocks_max`` is a
  ``@property`` on the transformer config and no such field exists on
  ``args``.
* The condition gates on ``attn_res_block_size`` rather than on
  ``model_type`` alone.  A K3 config with ``attn_res_block_size`` unset gets
  plain residuals and an unfolded ``[s, b, h]`` wire, i.e. the stock shape --
  scaling it would break that configuration rather than fix it.

Megatron has **two** code paths that compute the PP wire shape and both need
telling:

1. The non-interleaved 1F1B schedule calls
   :func:`megatron.core.pipeline_parallel.schedules.get_tensor_shapes`
   (``schedules.py``, called for the recv and send shapes).  Both calls are
   the same, and the first/last stage suppress their unused direction inside
   ``P2PCommunicator``, so scaling every returned triple uniformly is
   correct: every P2P transfer that actually happens is stage-to-stage and
   carries the folded tensor.
2. The interleaved 1F1B / VPP schedule computes ``tensor_shape`` *inline*
   from its ``seq_length`` argument (``schedules.py``) and never calls
   ``get_tensor_shapes``.  PyTorch P2P does not validate shape, only
   ``numel * dtype_size``, so an unpatched VPP receiver would silently copy
   the first ``s * h`` elements and ``_lift_res_in`` would then reject the
   tensor with the "not divisible by 1 + attn_res_num_blocks_max" error it
   raises at ``kimi_k3_block.py``.  Scaling the kwarg on the way in
   is safe because ``seq_length`` is read for nothing else inside that
   function.

Two schedules this patch deliberately **cannot** reach, so it refuses them
instead of producing a silently wrong wire shape:
``primus.backends.megatron.core.pipeline_parallel.zerobubble.runtime`` and
``...primuspipe.pipeline_launcher`` both bind ``get_tensor_shapes`` with a
module-level ``from ... import`` (``runtime.py``, ``pipeline_launcher.py``),
which captures the *original* function object, and ``runtime.py``
additionally recomputes the shape inline.
Rebinding the ``schedules`` module attribute therefore does not affect
either.  They are selected by ``patch_zero_bubble`` / ``patch_primus_pipeline``.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

__all__ = [
    "kimi_k3_pp_seq_multiplier",
    "patch_kimi_k3_pp_tensor_shape",
]


def kimi_k3_pp_seq_multiplier(args) -> int:
    """``1 + attn_res_num_blocks_max``, the PP wire's sequence-axis factor.

    Returns 1 when the attention-residual mechanism is off, which makes the
    whole patch a no-op by construction rather than by a separate branch.

    Deliberately recomputed from ``num_layers`` / ``attn_res_block_size``
    rather than read off ``args``: ``attn_res_num_blocks_max`` is a
    ``@property`` on :class:`KimiK3TransformerConfig`
    (``kimi_k3_transformer_config.py``) and never becomes an args
    field, so a ``getattr(args, "attn_res_num_blocks_max", 1)`` would read 1
    and quietly leave the wire shape unscaled.
    """
    block_size = int(getattr(args, "attn_res_block_size", 0) or 0)
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    if block_size <= 0 or num_layers <= 0:
        return 1
    num_blocks_max = -(-num_layers // block_size)  # ceil
    return 1 + num_blocks_max


def _make_k3_get_tensor_shapes(original_fn, seq_mult: int):
    """Return a wrapper that scales the first (seq) dim by ``seq_mult``.

    ``original_fn`` returns a list of ``(seq_length, micro_batch, hidden)``
    triples.  Only the sequence dim changes: the fold stacks
    ``1 + num_blocks_max`` copies of ``[s, b, h]`` along dim 0
    (``_lower_res_out``'s closing ``reshape``), so micro-batch and hidden are
    untouched.

    Note the scaling composes correctly with the CP and sequence-parallel
    divisions ``get_tensor_shapes`` has already applied (``schedules.py``):
    the block folds whatever *local* sequence length the stage actually
    holds, so the factor multiplies the already-divided value.
    """

    def patched_get_tensor_shapes(*args, **kwargs):
        shapes = original_fn(*args, **kwargs)
        return [(s * seq_mult, b, h) for (s, b, h) in shapes]

    patched_get_tensor_shapes.__wrapped__ = original_fn
    patched_get_tensor_shapes._k3_pp_shape_patched = True
    patched_get_tensor_shapes._k3_pp_seq_mult = seq_mult
    return patched_get_tensor_shapes


def _make_k3_interleaved_schedule(original_fn, seq_mult: int):
    """Wrap the interleaved schedule to scale its ``seq_length`` kwarg.

    ``forward_backward_pipelining_with_interleaving`` builds its PP wire
    ``tensor_shape`` inline from ``seq_length`` (``schedules.py``) and
    reads the argument for nothing else, so scaling it on the way in gives
    the schedule a K3-aware shape without rewriting the function.
    ``decoder_seq_length`` is scaled too, for symmetry with the
    ``get_tensor_shapes`` path -- which *does* prefer it when it is not None
    (``schedules.py``).  In this Megatron HEAD the interleaved schedule
    declares the kwarg and never reads it, so the second line is currently
    inert; it is here so the two paths cannot diverge if a later HEAD starts
    honouring it.
    """

    def patched_schedule(*args, **kwargs):
        if kwargs.get("seq_length") is not None:
            kwargs["seq_length"] = int(kwargs["seq_length"]) * seq_mult
        if kwargs.get("decoder_seq_length") is not None:
            kwargs["decoder_seq_length"] = int(kwargs["decoder_seq_length"]) * seq_mult
        return original_fn(*args, **kwargs)

    patched_schedule.__wrapped__ = original_fn
    patched_schedule._k3_pp_interleaved_patched = True
    return patched_schedule


def _wants_k3_pp_shape_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if getattr(args, "model_type", None) != "kimi_k3":
        return False
    if int(getattr(args, "pipeline_model_parallel_size", 1) or 1) <= 1:
        return False
    return kimi_k3_pp_seq_multiplier(args) > 1


@register_patch(
    "megatron.kimi_k3.pp_tensor_shape",
    backend="megatron",
    phase="before_train",
    description=(
        "Kimi K3: fold the attention-residual block checkpoints into the PP "
        "wire sequence axis so [(1 + attn_res_num_blocks_max) * s, b, h] "
        "passes between pipeline stages (covers both the 1F1B "
        "get_tensor_shapes path and the interleaved-1F1B / VPP inline "
        "tensor_shape path)."
    ),
    condition=_wants_k3_pp_shape_patch,
)
def patch_kimi_k3_pp_tensor_shape(ctx: PatchContext):
    """Multiply the PP P2P seq dim by ``1 + attn_res_num_blocks_max``."""
    import megatron.core.pipeline_parallel.schedules as schedules_module

    args = get_args(ctx)
    seq_mult = kimi_k3_pp_seq_multiplier(args)

    # Refuse the two schedules this patch provably cannot reach, rather than
    # let them run with an unscaled wire shape. Both capture
    # ``get_tensor_shapes`` in a module-level ``from ... import``, so the
    # rebinding below is invisible to them; the zero-bubble runtime also
    # recomputes the shape inline (runtime.py).
    for flag in ("patch_zero_bubble", "patch_primus_pipeline"):
        if getattr(args, flag, False):
            raise NotImplementedError(
                f"Kimi K3 attention residuals at pipeline_model_parallel_size > 1 are not "
                f"supported with {flag}=True. That schedule binds get_tensor_shapes at "
                "module import time (zerobubble/runtime.py, "
                "primuspipe/pipeline_launcher.py) and the zero-bubble runtime also "
                "recomputes the wire shape inline (runtime.py), so this patch "
                "cannot reach it and the PP wire would silently carry "
                f"[s, b, h] where the block emits [{seq_mult} * s, b, h]. Use the stock "
                "1F1B or interleaved-1F1B schedule."
            )

    # Wrapper 1: get_tensor_shapes (the non-interleaved 1F1B schedule, and
    # Primus's --pp-warmup helper, which imports the symbol inside its own
    # function body (pp_warmup_patches.py) and so sees the wrapper).
    original_get_tensor_shapes = schedules_module.get_tensor_shapes
    if getattr(original_get_tensor_shapes, "_k3_pp_shape_patched", False):
        log_rank_0("[Patch:megatron.kimi_k3.pp_tensor_shape] get_tensor_shapes already patched, skip")
    else:
        schedules_module.get_tensor_shapes = _make_k3_get_tensor_shapes(original_get_tensor_shapes, seq_mult)
        log_rank_0(
            f"[Patch:megatron.kimi_k3.pp_tensor_shape] wrapped get_tensor_shapes; "
            f"PP wire seq_len * (1 + attn_res_num_blocks_max) = {seq_mult} "
            f"(num_layers={getattr(args, 'num_layers', None)}, "
            f"attn_res_block_size={getattr(args, 'attn_res_block_size', None)})."
        )

    # Wrapper 2: forward_backward_pipelining_with_interleaving (VPP).
    original_interleaved = schedules_module.forward_backward_pipelining_with_interleaving
    if getattr(original_interleaved, "_k3_pp_interleaved_patched", False):
        log_rank_0("[Patch:megatron.kimi_k3.pp_tensor_shape] interleaved schedule already patched, skip")
    else:
        schedules_module.forward_backward_pipelining_with_interleaving = _make_k3_interleaved_schedule(
            original_interleaved, seq_mult
        )
        log_rank_0(
            f"[Patch:megatron.kimi_k3.pp_tensor_shape] wrapped "
            f"forward_backward_pipelining_with_interleaving; seq_length * {seq_mult} on the "
            "way into the interleaved-1F1B / VPP schedule."
        )
