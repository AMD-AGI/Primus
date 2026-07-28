###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DPA Consolidated Prologue Patch

Separates the eager setup (FP8 metadata, backend selection, cu_seqlens, etc.)
from the FusedAttnFunc kernel call in DotProductAttention, so that the kernel
call can live inside a torch.compile compiled graph rather than running eagerly.

Graph-break analysis (per DPA call):

    Before:
        [compiled A] --break--> [DPA runs fully eager, kernel included] --break--> [compiled B]

    After (attention_dropout == 0, the Flux default - fast path):
        [compiled A] --break--> [eager prologue] --break--> [compiled post-ops (kernel inside)]

    After (attention_dropout > 0 - safe path):
        [compiled A] --break--> [eager prologue] --break--> [eager fork + kernel] --break--> [compiled post-ops]

Break counts: 2 for the fast path, 3 for the safe path.  The safe path runs
the FusedAttn kernel inside the model-parallel-rng fork context so the
kernel's philox state targets the correct generator.  In the fast path the
kernel doesn't consume RNG (dropout is 0), so the fork wrapper is omitted
and the kernel stays inside the compiled graph alongside the post-attention
ops (output reshape, projection, skip connections).

Mechanism:
    FusedAttnFunc.forward is monkey-patched with a thread-local "capture mode".
    When capture mode is active the patched forward stores its arguments in
    thread-local storage and raises ``_CaptureComplete``, aborting the DPA
    forward after all setup is done.  The captured args are then passed to the
    *real* ``FusedAttnFunc.apply`` inside the compiled graph.
"""

import threading
from typing import Optional

import torch
from torch import Tensor

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# Positional index of dropout_p inside FusedAttnFunc.forward's *args (i.e.
# inside our captured _capture_tls.captured_args tuple, which excludes ctx).
# Verified against transformer_engine.pytorch.attention.dot_product_attention
# .backends.FusedAttnFunc.forward as of TE 2.12. If a future TE upgrade
# reshuffles the positional layout, _new_te_dpa_forward falls back to the
# always-fork (safe) path rather than silently routing to the dropout=0 fast
# path with the wrong field.
_DROPOUT_P_IDX = 14

# ---------------------------------------------------------------------------
# Thread-local capture state
# ---------------------------------------------------------------------------
_capture_tls = threading.local()


class _CaptureComplete(Exception):
    """Raised inside the patched FusedAttnFunc.forward to abort after capture."""


# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------
@register_patch(
    "megatron.te.dpa_consolidated_prologue",
    backend="megatron",
    phase="before_train",
    priority=55,
    description=(
        "Consolidated DPA prologue: separates eager attention setup from "
        "FusedAttnFunc kernel call to reduce torch.compile graph breaks"
    ),
    condition=lambda ctx: (
        (
            (
                getattr(get_args(ctx), "torch_compile", None) is not None
                and getattr(get_args(ctx).torch_compile, "enable", False)
            )
            # Align with torch_compile_patches.py, which also honors the flat
            # enable_torch_compile flag; otherwise this prologue optimization is
            # silently skipped for configs that compile via enable_torch_compile.
            or getattr(get_args(ctx), "enable_torch_compile", False)
        )
        and not getattr(get_args(ctx), "disable_dpa_prologue_patch", False)
    ),
)
def patch_dpa_consolidated_prologue(ctx: PatchContext):
    """Replace TEDotProductAttention.forward with a two-phase version.

    Phase 1 – eager prologue (``@torch._dynamo.disable``):
        Runs the full DPA + FusedAttention setup pipeline to compute all
        arguments for ``FusedAttnFunc.apply``, without running the kernel.

    Phase 2 – compiled kernel call:
        ``FusedAttnFunc.apply`` (marked ``allow_in_graph``) runs inside the
        torch.compile compiled graph, enabling fusion with pre/post-attention
        operations.
    """
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        FusedAttnFunc,
    )

    # ---- Idempotency guard -----------------------------------------------
    # Re-running this patch would capture the already-patched forward as
    # "_orig_*", double-wrapping the capture logic. Guard against it.
    if getattr(TEDotProductAttention, "_primus_dpa_prologue_patched", False):
        log_rank_0("[Patch:megatron.te.dpa_consolidated_prologue] already applied; skipping re-patch")
        return

    # ---- TE version check ------------------------------------------------
    # _DROPOUT_P_IDX (and the captured-args layout) is pinned to TE 2.12. On a
    # different TE version the layout may drift; the fast-path detection
    # already falls back to the always-fork safe path on mismatch, but warn so
    # the perf regression is visible rather than silent.
    try:
        import transformer_engine

        _te_version = getattr(transformer_engine, "__version__", None)
    except Exception:
        _te_version = None
    if _te_version is not None and not str(_te_version).startswith("2.12"):
        log_rank_0(
            "[Patch:megatron.te.dpa_consolidated_prologue] WARNING: "
            f"_DROPOUT_P_IDX={_DROPOUT_P_IDX} was verified against TE 2.12 but "
            f"detected transformer_engine {_te_version}. The dropout=0 fast path "
            "will conservatively fall back to the safe rng-fork path if the "
            "captured-args layout differs."
        )

    # ---- Mark FusedAttnFunc as graph-safe --------------------------------
    torch._dynamo.allow_in_graph(FusedAttnFunc)

    # ---- Patch FusedAttnFunc.forward for capture mode --------------------
    _orig_fused_attn_func_fwd = FusedAttnFunc.forward

    @staticmethod
    def _capturing_fused_attn_func_fwd(ctx, *args):  # noqa: N805 – autograd ctx
        if getattr(_capture_tls, "capture_mode", False):
            _capture_tls.captured_args = args
            raise _CaptureComplete()
        return _orig_fused_attn_func_fwd(ctx, *args)

    FusedAttnFunc.forward = _capturing_fused_attn_func_fwd

    # ---- Save original TEDotProductAttention.forward ---------------------
    _orig_te_dpa_forward = TEDotProductAttention.forward

    # ---- Eager prologue --------------------------------------------------
    @torch._dynamo.disable
    def _dpa_eager_prologue(
        te_dpa,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias,
        packed_seq_params,
        num_splits,
    ):
        """Run the full DPA + FusedAttention setup eagerly.

        Returns
        -------
        captured_args : tuple | None
            Arguments for ``FusedAttnFunc.apply`` if fused backend was chosen.
        fallback_result : Tensor | None
            Final output tensor if a non-fused backend ran to completion.
        """
        _capture_tls.capture_mode = True
        _capture_tls.captured_args = None
        fallback_result = None

        try:
            fallback_result = _orig_te_dpa_forward(
                te_dpa,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                num_splits=num_splits,
            )
        except _CaptureComplete:
            pass  # expected – fused attention args captured
        finally:
            _capture_tls.capture_mode = False

        return _capture_tls.captured_args, fallback_result

    # ---- RNG-correct kernel execution ------------------------------------
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    @torch._dynamo.disable
    def _fused_attn_with_rng_fork(captured_args):
        """Run FusedAttnFunc.apply inside the model-parallel-rng fork.

        The eager prologue's _CaptureComplete exception unwinds TE's
        internal fork context before the kernel runs, so rng_gen=None
        causes the default CUDA generator to be advanced instead of
        model-parallel-rng.  This wrapper re-enters the fork so the
        kernel's philox_cuda_state call targets the correct generator.
        """
        tracker = get_cuda_rng_tracker()
        if tracker.is_initialized():
            states = tracker.get_states()
            if "model-parallel-rng" in states:
                with tracker.fork("model-parallel-rng"):
                    return FusedAttnFunc.apply(*captured_args)
        return FusedAttnFunc.apply(*captured_args)

    # ---- Max-logit bookkeeping (qk_clip) ---------------------------------
    @torch._dynamo.disable
    def _update_max_logit_stats(te_dpa, batch_max_logit):
        if hasattr(te_dpa, "current_max_attn_logits"):
            if te_dpa.current_max_attn_logits is None:
                te_dpa.current_max_attn_logits = batch_max_logit
            else:
                te_dpa.current_max_attn_logits = torch.max(te_dpa.current_max_attn_logits, batch_max_logit)

    # ---- Replacement forward ---------------------------------------------
    def _new_te_dpa_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        num_splits: Optional[int] = None,
    ) -> Tensor:
        """TEDotProductAttention.forward with consolidated prologue.

        The eager prologue handles all DPA setup (FP8 metadata, backend
        selection, cu_seqlens computation, etc.).  Then ``FusedAttnFunc.apply``
        runs inside the compiled graph.

        For non-fused backends (flash / unfused), the original forward runs
        fully inside the prologue and the result is returned directly.
        """
        captured_args, fallback_result = _dpa_eager_prologue(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias,
            packed_seq_params,
            num_splits,
        )

        if captured_args is not None:
            # Fast path: when dropout_p == 0 the FusedAttn kernel doesn't
            # consume RNG, so we can skip the eager rng-fork wrapper and
            # call FusedAttnFunc.apply directly inside the compiled graph
            # (it's already marked allow_in_graph above). This drops a
            # graph break in the common Flux case (attention_dropout=0).
            #
            # Safe path: anything that isn't a clean numeric zero (including
            # IndexError if a TE upgrade changes captured_args' layout, or
            # an unexpected tensor / object at the dropout slot) falls back
            # to the always-fork path so correctness is never compromised
            # by a layout drift.
            try:
                _dropout_p = captured_args[_DROPOUT_P_IDX]
                _use_fast_path = isinstance(_dropout_p, (int, float)) and _dropout_p == 0
            except (IndexError, TypeError):
                _use_fast_path = False

            if _use_fast_path:
                result = FusedAttnFunc.apply(*captured_args)
            else:
                result = _fused_attn_with_rng_fork(captured_args)
            return_max_logit = captured_args[-1]

            if return_max_logit:
                core_attn_out = result[0].view(*result[0].shape[:-2], -1)
                _update_max_logit_stats(self, result[1])
                return core_attn_out

            return result.view(*result.shape[:-2], -1)

        return fallback_result

    # ---- Apply the monkey-patch ------------------------------------------
    TEDotProductAttention.forward = _new_te_dpa_forward
    TEDotProductAttention._primus_dpa_prologue_patched = True

    log_rank_0("[Patch:megatron.te.dpa_consolidated_prologue] " "Applied allow_in_graph to FusedAttnFunc")
    log_rank_0(
        "[Patch:megatron.te.dpa_consolidated_prologue] "
        "Replaced TEDotProductAttention.forward with consolidated prologue"
    )
