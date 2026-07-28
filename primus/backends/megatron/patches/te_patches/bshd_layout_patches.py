###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine attention BSHD-layout patch.

Migrated from the source patch ``megatron_te_bshd_layout.patch`` (mpo branch).

Some TE FMHA kernels are faster with a ``bshd`` memory layout than with the
default ``sbhd`` layout Megatron feeds them. When ``NVTE_FMHA_USE_BSHD=1`` is
set, this patch transposes the ``sbhd`` ``query``/``key``/``value`` tensors to
``bshd`` before the TE ``DotProductAttention`` call and transposes the result
back to ``sbhd`` afterwards, so the change is transparent to the rest of
Megatron.

Instead of editing ``third_party/Megatron-LM`` in place, we wrap
``TEDotProductAttention.forward`` and temporarily flip the effective
``qkv_format`` for the duration of the wrapped call so TE interprets the
already-transposed tensors correctly.

Gate:
    ``NVTE_FMHA_USE_BSHD=1`` (off by default; the patch is not installed at
    all when unset, so there is zero overhead).
"""

import functools
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _bshd_enabled(_ctx: PatchContext) -> bool:
    return os.environ.get("NVTE_FMHA_USE_BSHD", "0") == "1"


def _effective_qkv_format(self, packed_seq_params):
    """Mirror TEDotProductAttention.forward's qkv_format resolution."""
    if packed_seq_params is not None:
        fmt = getattr(packed_seq_params, "qkv_format", None)
        if fmt is not None:
            return fmt
    return self.qkv_format


def _make_wrapped_forward(orig_forward):
    @functools.wraps(orig_forward)
    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        # Only convert when the effective layout is sbhd; otherwise behave
        # exactly like the original.
        if _effective_qkv_format(self, packed_seq_params) != "sbhd":
            return orig_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )

        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()

        # Temporarily flip the qkv_format TE will read so the transposed
        # tensors are interpreted as bshd. Restore it afterwards even on error.
        if packed_seq_params is not None and hasattr(packed_seq_params, "qkv_format"):
            restore_target, restore_value = packed_seq_params, packed_seq_params.qkv_format
            packed_seq_params.qkv_format = "bshd"
        else:
            restore_target, restore_value = self, self.qkv_format
            self.qkv_format = "bshd"

        try:
            core_attn_out = orig_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        finally:
            restore_target.qkv_format = restore_value

        return core_attn_out.transpose(0, 1).contiguous()

    return forward


@register_patch(
    "megatron.te.fmha_bshd_layout",
    backend="megatron",
    phase="before_train",
    description=(
        "Run TE DotProductAttention in bshd layout (transpose sbhd<->bshd "
        "around the kernel) when NVTE_FMHA_USE_BSHD=1."
    ),
    condition=_bshd_enabled,
)
def patch_te_fmha_bshd_layout(ctx: PatchContext):
    del ctx

    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError as exc:
        warning_rank_0(
            f"[Patch:megatron.te.fmha_bshd_layout] transformer_engine extension "
            f"not importable; skipping: {exc}"
        )
        return

    cls = getattr(te_ext, "TEDotProductAttention", None)
    if cls is None:
        warning_rank_0("[Patch:megatron.te.fmha_bshd_layout] TEDotProductAttention not found; skipping.")
        return

    if getattr(cls, "_primus_bshd_patched", False):
        return

    cls.forward = _make_wrapped_forward(cls.forward)
    cls._primus_bshd_patched = True
    log_rank_0(
        "[Patch:megatron.te.fmha_bshd_layout] Patched TEDotProductAttention.forward "
        "to use bshd layout (NVTE_FMHA_USE_BSHD=1)."
    )
