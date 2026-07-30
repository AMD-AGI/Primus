###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Mamba/Hybrid FLA Fused Cross-Entropy Patch
===========================================

Wires flash-linear-attention's (FLA) fused cross-entropy kernels into
``MambaModel`` so GDN/KDA/Mamba2 (pure or hybrid) training never materializes
the full ``(batch*seq, vocab)`` logits tensor -- avoiding OOM on large
vocab/long sequence runs and matching FLA's own training loss numerically.

Two modes, selected by ``args.fused_ce_mode`` (resolved by
``fla_runtime_patches.py`` from ``PRIMUS_FUSED_CE`` / YAML ``fused_ce_mode``):

    0: disabled -- stock Megatron path (materializes full logits).
    1 (default): ``FusedLinearCrossEntropyLoss`` -- computes logits + CE in
       chunks (``args.fused_ce_chunks``, default 32); the full logits tensor
       is never resident. Biggest memory win.
    2: ``FusedCrossEntropyLoss`` -- materializes logits in bf16, then a fused
       Triton CE kernel; matches FLA's exact computation.

This is a "source-string rewrite" style patch: the two injection points
(``MambaModel.__init__`` / ``MambaModel.forward``) sit in the middle of
upstream method bodies, where a plain function-wrapping monkey-patch cannot
reach without duplicating the entire (large, version-sensitive) method.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_method_source
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mamba.fla_fused_ce"

# Inserted at the end of MambaModel.__init__, right after the embeddings/
# output-layer setup and before the `finish_init` loop -- mirrors exactly
# where the original megatron_patches/01-mamba_model-fused-ce.patch spliced
# in its setup code.
_INIT_ORI = "self.setup_embeddings_and_output_layer()"
_INIT_NEW = (
    _INIT_ORI + "\n\n" + "        from megatron.training import get_args as _get_args\n"
    "        _args = _get_args()\n"
    "        self._fused_ce_mode = 0\n"
    "        _ce_mode = getattr(_args, 'fused_ce_mode', 1)\n"
    "        if _ce_mode == 2:\n"
    "            try:\n"
    "                from fla.modules import FusedCrossEntropyLoss\n"
    "                self._fused_ce = FusedCrossEntropyLoss(inplace_backward=True)\n"
    "                self._fused_ce_mode = 2\n"
    "            except ImportError:\n"
    "                pass\n"
    "        elif _ce_mode == 1:\n"
    "            try:\n"
    "                from fla.modules import FusedLinearCrossEntropyLoss\n"
    "                _nc = getattr(_args, 'fused_ce_chunks', 32)\n"
    "                self._fused_lce = FusedLinearCrossEntropyLoss(reduction='mean', num_chunks=_nc)\n"
    "                self._fused_ce_mode = 1\n"
    "            except ImportError:\n"
    "                pass\n"
    "        self._use_fused_cross_entropy = self._fused_ce_mode > 0"
)

# Inserted right before the standard output-layer + loss computation, so the
# fused path can skip building the full [s, b, vocab] logits tensor entirely.
_FORWARD_ORI = "logits, _ = self.output_layer("
_FORWARD_NEW = (
    "if labels is not None and self._use_fused_cross_entropy:\n"
    "            return self._fused_cross_entropy_loss(hidden_states, labels, output_weight)\n"
    "\n"
    "        " + _FORWARD_ORI
)


def _fused_cross_entropy_loss(self, hidden_states, labels, output_weight, runtime_gather_output=None):
    """Compute loss using FLA's fused cross-entropy kernels.

    Mode 1 (FusedLinearCrossEntropyLoss): computes logits + CE in chunks,
           never materializing the full (batch*seq, vocab) logits tensor.
    Mode 2 (FusedCrossEntropyLoss): materializes logits in bf16, then
           uses a fused Triton CE kernel -- matches FLA's exact computation.
    """
    s, b, h = hidden_states.shape

    if self._fused_ce_mode == 2:
        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        logits_2d = logits.permute(1, 0, 2).reshape(b * s, -1)
        labels_1d = labels.reshape(b * s)
        loss = self._fused_ce(logits_2d, labels_1d)
        return loss.expand(b, s)
    else:
        hs_2d = hidden_states.permute(1, 0, 2).reshape(b * s, h)
        labels_1d = labels.reshape(b * s)
        weight = output_weight if output_weight is not None else self.output_layer.weight
        loss = self._fused_lce(hs_2d, labels_1d, weight)
        return loss.expand(b, s)


def _install_mamba_fused_ce_patch() -> None:
    from megatron.core.models.mamba.mamba_model import MambaModel

    if is_patched(MambaModel, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] MambaModel already patched; skipping.")
        return

    patch_method_source(MambaModel, "__init__", _INIT_ORI, _INIT_NEW)
    patch_method_source(MambaModel, "forward", _FORWARD_ORI, _FORWARD_NEW)
    MambaModel._fused_cross_entropy_loss = _fused_cross_entropy_loss

    mark_patched(MambaModel, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Patched MambaModel.__init__/forward to route through "
        "FLA's fused cross-entropy kernels when fused_ce_mode != 0."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Wire FLA's FusedLinearCrossEntropyLoss / FusedCrossEntropyLoss into MambaModel "
        "so the full (batch*seq, vocab) logits tensor is never materialized."
    ),
    # Runs after fla_runtime_knobs (priority=-100) has resolved args.fused_ce_mode.
    priority=50,
    condition=lambda ctx: getattr(get_args(ctx), "fused_ce_mode", 0) != 0,
)
def patch_mamba_fused_ce(ctx: PatchContext) -> None:
    _install_mamba_fused_ce_patch()
