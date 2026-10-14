"""Chunk the language-model head projection to dodge a large-N GEMM hang.

WHY THIS EXISTS
---------------
On gfx1250 with amdprimus:20260730, a single bf16 GEMM of

    (mbs*seq, 2816) @ (2816, 262144)

wedges the GPU hard enough to need a host reboot (PROGRESS.md 4.18). Gemma 4's
262,144-entry vocabulary means every training step issues exactly that matmul in
`GPTModel.output_layer`, so no Gemma 4 configuration can complete a step here.

Things that do NOT fix it, all measured rather than assumed:
  * TORCH_BLAS_PREFER_HIPBLASLT=0 -- rocBLAS wedges on the same shape too, so
    this is not hipBLASLt kernel selection (4.20).
  * cross_entropy_fusion_impl='te' -- chunks the *loss*, not the projection.
    The output layer still issues one full-width matmul.
  * Tensor parallelism would shard the vocab and shrink N, but we have 1 GPU.

What is left is to chunk the projection itself. Each output column depends only
on its own weight column, so splitting the weight along the vocab axis and
concatenating the results is mathematically equivalent. Autograd handles
cat/slice backward.

Numerics, measured on CPU at the real (2816 x 262144) shape with width 32768:
  * forward      -- bit-identical
  * weight grad  -- bit-identical
  * input grad   -- differs at roundoff only (2.9e-3 absolute on outputs of
                    magnitude 300, i.e. ~1e-5 relative in fp32). dgrad sums a
                    contribution per chunk instead of accumulating inside one
                    kernel, so the accumulation order changes. This is ordinary
                    floating-point reassociation, not a correctness defect, but
                    it does mean runs are not bit-comparable against an
                    unchunked baseline.

HOW
---
`ColumnParallelLinear.forward` validates that any caller-supplied weight has
shape `(output_size_per_partition, input_size)`, so a narrowed weight cannot
simply be passed in. Instead this narrows `output_size_per_partition` for the
duration of each chunk and calls megatron's real `forward`. That keeps every
piece of megatron's logic -- comm regions, `_forward_impl` dispatch, bias
handling -- rather than reimplementing it and risking a subtle divergence.

Slicing rows of a contiguous `(out_features, in_features)` weight yields
contiguous views, so no copy is made.

COST
----
The concatenation materializes the same full logit tensor as the unchunked path,
so peak memory is unchanged. This is a hang workaround, not a memory
optimization. Overhead is vocab/width extra GEMM launches on a projection that
is a few percent of step time.

CHUNK WIDTH MATTERS -- 32768 IS TOO WIDE
----------------------------------------
Measured (PROGRESS.md 4.28): a chunk width of 32768 **still wedges the GPU**. So the
trigger is not simply "the full 262144" -- the threshold is below 32768. Known-good
forward GEMM widths are 2048 and 4096, which puts the live window at (4096, 32768].

Until the threshold is measured, use a width at or below the last known-good
value rather than the 32768 this file originally suggested:

    PRIMUS_GEMMA4_LOGIT_CHUNK=4096

That is 64 chunks for a 262144 vocabulary, so launch overhead will be more
noticeable than the original estimate, though still small against a full step.

CONTROL
-------
Disabled by default so it cannot perturb runs on a healthy stack:

    PRIMUS_GEMMA4_LOGIT_CHUNK=4096
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_GEMMA4_LOGIT_CHUNK"
_PATCHED_ATTR = "_primus_gemma4_chunked_logits_patched"


def _chunk_width():
    raw = os.environ.get(_ENV, "").strip()
    if not raw:
        return 0
    try:
        width = int(raw)
    except ValueError:
        raise ValueError(f"{_ENV} must be an integer, got {raw!r}")
    if width <= 0:
        raise ValueError(f"{_ENV} must be positive, got {width}")
    return width


@contextmanager
def _narrowed(layer, size):
    saved = layer.output_size_per_partition
    layer.output_size_per_partition = size
    try:
        yield
    finally:
        layer.output_size_per_partition = saved


def _unsupported(layer, width):
    """Reasons the chunked path would be wrong rather than merely slow.

    The bypass is only safe where the tensor-parallel machinery is inert and
    where nothing downstream expects the weight to be a leaf parameter. Rather
    than silently produce wrong gradients, refuse and say why.
    """
    reasons = []
    if getattr(layer, "bias", None) is not None:
        # megatron passes the full-width bias into every call, which would not
        # broadcast against a narrowed output. Gemma 4's head is bias=False.
        reasons.append("layer has a bias")
    if getattr(layer, "gradient_accumulation_fusion", False):
        # The fused wgrad kernel reads weight.main_grad, which a slice lacks.
        reasons.append("gradient_accumulation_fusion=True")
    if getattr(layer, "sequence_parallel", False):
        reasons.append("sequence_parallel=True")
    if getattr(layer, "explicit_expert_comm", False):
        reasons.append("explicit_expert_comm=True")
    if getattr(layer.config, "defer_embedding_wgrad_compute", False):
        # Would append one activation per chunk to the wgrad buffer.
        reasons.append("defer_embedding_wgrad_compute=True")
    if layer.output_size_per_partition != layer.output_size:
        # Vocab is already sharded across TP ranks; N may be small enough
        # anyway, and chunking a shard needs care this patch does not take.
        reasons.append(
            f"vocab is tensor-parallel sharded "
            f"({layer.output_size_per_partition} of {layer.output_size})"
        )
    return reasons


@register_patch(
    "gemma4.logits.chunked",
    backend="megatron_bridge",
    phase="setup",
    description="Opt-in: split the vocab projection into chunks, to avoid a large-N GEMM that hangs gfx1250",
)
def patch_gemma4_chunked_logits(ctx: PatchContext) -> None:
    width = _chunk_width()
    if not width:
        return

    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    except Exception:
        log_rank_0("[Patch:gemma4.logits.chunked] megatron-core ColumnParallelLinear unavailable")
        return

    original_forward = ColumnParallelLinear.forward
    if getattr(original_forward, _PATCHED_ATTR, False):
        return
    state = {"logged": False}

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else getattr(self, "weight", None)

        # Only the vocab projection is wide enough to trip the bug. Every other
        # ColumnParallelLinear -- qkv, mlp, router -- stays on the fast path.
        if w is None or w.shape[0] <= width:
            return original_forward(
                self, input_, weight=weight, runtime_gather_output=runtime_gather_output
            )

        blockers = _unsupported(self, width)
        if blockers:
            raise RuntimeError(
                "[gemma4.logits.chunked] cannot chunk a "
                f"{tuple(w.shape)} projection because: {', '.join(blockers)}. "
                "Either clear those settings or unset PRIMUS_GEMMA4_LOGIT_CHUNK."
            )

        n = w.shape[0]
        if not state["logged"]:
            log_rank_0(
                f"[Patch:gemma4.logits.chunked] splitting {tuple(w.shape)} projection "
                f"into {(n + width - 1) // width} chunks of <= {width}"
            )
            state["logged"] = True

        outputs, out_bias = [], None
        for i in range(0, n, width):
            piece = w[i : i + width]
            with _narrowed(self, piece.shape[0]):
                out, out_bias = original_forward(
                    self, input_, weight=piece, runtime_gather_output=runtime_gather_output
                )
            outputs.append(out)

        # ColumnParallelLinear's output is partitioned along its last dimension,
        # which is the axis we split, so concatenating restores the full logits.
        return torch.cat(outputs, dim=-1), out_bias

    setattr(forward, _PATCHED_ATTR, True)
    ColumnParallelLinear.forward = forward
    log_rank_0(
        f"[Patch:gemma4.logits.chunked] enabled, chunk width {width} "
        f"(applies where out_features > {width})"
    )
