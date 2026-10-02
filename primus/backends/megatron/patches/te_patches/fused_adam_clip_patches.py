###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fuse global gradient clipping into the ROCm Transformer Engine Adam kernel.

Set ``PRIMUS_FUSED_ADAM_CLIP=1`` to retain Megatron's global-norm calculation
but defer application of its clip coefficient to Adam while the gradient is
already resident in registers.

GPT-OSS gives TE FP32 gradients, FP32 optimizer parameters, and FP32 moments.
TE's ROCm non-capturable path is important here: it uses one custom
device-metadata Adam launch instead of the generic path's many launches. This
patch preserves that layout and compiles a small Primus-owned HIP extension
with one additional scalar argument. It does not modify the TE installation.

This first phase intentionally leaves Megatron's norm ``.item()`` in place.
Removing that host synchronization requires a device-scalar coefficient and is
best validated separately after the HBM round-trip is removed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.te_fused_adam_clip"
_EXTENSION_NAME = "primus_te_fused_adam_clip_v1"
_extension = None


def _enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1"


def _clip_coeff(total_norm: float, max_norm: float) -> float:
    """Return Megatron's gradient clipping coefficient."""

    return min(1.0, float(max_norm) / (float(total_norm) + 1.0e-6))


def _load_extension():
    global _extension
    if _extension is not None:
        return _extension

    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("fused_adam_clip_kernel.cu")
    _extension = load(
        name=_EXTENSION_NAME,
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=False,
    )
    return _extension


def _metadata_key(chunk_size: int, tensor_lists) -> tuple[Any, ...]:
    return (
        int(chunk_size),
        tuple(
            (tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.device)
            for tensor_list in tensor_lists
            for tensor in tensor_list
        ),
    )


def _build_metadata(chunk_size: int, tensor_lists):
    if len(tensor_lists) != 4:
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP currently requires TE's four-list "
            f"FP32 Adam path (grad, param, exp_avg, exp_avg_sq), got {len(tensor_lists)} lists"
        )
    num_tensors = len(tensor_lists[0])
    if num_tensors == 0 or any(len(tensors) != num_tensors for tensors in tensor_lists):
        raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP received inconsistent empty tensor lists")

    device = tensor_lists[0][0].device
    addresses = []
    sizes = []
    block_to_tensor = []
    chunk_offsets = []
    total_chunks = 0

    for tensor_idx in range(num_tensors):
        tensors = [tensor_lists[list_idx][tensor_idx] for list_idx in range(4)]
        if any(tensor.device != device for tensor in tensors):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires one device per Adam call")
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            dtypes = tuple(tensor.dtype for tensor in tensors)
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP specializes the GPT-OSS FP32 optimizer path; "
                f"tensor {tensor_idx} has dtypes {dtypes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires contiguous optimizer tensors")
        if any(tensor.numel() != tensors[0].numel() for tensor in tensors[1:]):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP received mismatched tensor sizes")

        addresses.extend(tensor.data_ptr() for tensor in tensors)
        numel = tensors[0].numel()
        sizes.append(numel)
        chunk_offsets.append(total_chunks)
        chunks = (numel + chunk_size - 1) // chunk_size
        block_to_tensor.extend([tensor_idx] * chunks)
        total_chunks += chunks

    return (
        torch.tensor(addresses, dtype=torch.int64, device=device),
        torch.tensor(sizes, dtype=torch.int64, device=device),
        torch.tensor(block_to_tensor, dtype=torch.int32, device=device),
        torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
        total_chunks,
    )


def _install_patch() -> None:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    if is_patched(FusedAdam, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already installed; skipping.")
        return

    extension = _load_extension()
    original_adam_init = FusedAdam.__init__
    original_adam_step = FusedAdam.step

    def patched_adam_init(self, *args, **kwargs):
        if kwargs.get("capturable", False):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP preserves TE's optimized non-capturable ROCm path"
            )
        original_adam_init(self, *args, **kwargs)
        self._primus_clip_coeff = 1.0
        self._primus_adam_metadata_cache = {}

        def fused_multi_tensor_adam(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            epsilon,
            step,
            mode,
            bias_correction,
            weight_decay,
        ):
            del noop_flag
            key = _metadata_key(chunk_size, tensor_lists)
            metadata = self._primus_adam_metadata_cache.get(key)
            if metadata is None:
                metadata = _build_metadata(chunk_size, tensor_lists)
                self._primus_adam_metadata_cache[key] = metadata
            addresses, sizes, block_to_tensor, chunk_offsets, total_chunks = metadata
            extension.fused_adam_clip(
                addresses,
                sizes,
                block_to_tensor,
                chunk_offsets,
                total_chunks,
                int(chunk_size),
                float(lr),
                float(beta1),
                float(beta2),
                float(epsilon),
                int(step),
                int(mode),
                int(bias_correction),
                float(weight_decay),
                float(self._primus_clip_coeff),
            )

        # FusedAdam.step calls this attribute through TE's existing
        # multi_tensor_applier, so all Python-side grouping remains unchanged.
        self.multi_tensor_adam = fused_multi_tensor_adam

    def patched_adam_step(self, closure=None, grad_scaler=None):
        if grad_scaler is not None:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP does not currently compose with an AMP GradScaler"
            )
        try:
            return original_adam_step(self, closure=closure, grad_scaler=None)
        finally:
            self._primus_clip_coeff = 1.0

    def patched_clip_grad_norm(self, clip_grad: float) -> float:
        # Preserve Megatron's exact norm calculation and collective behavior,
        # but do not launch clip_grad_by_total_norm_fp32's scale pass.
        grad_norm = self.get_grad_norm()
        inner_optimizer = self.optimizer
        if not isinstance(inner_optimizer, FusedAdam):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires Transformer Engine FusedAdam, got "
                f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}"
            )
        inner_optimizer._primus_clip_coeff = _clip_coeff(grad_norm, clip_grad)
        return grad_norm

    FusedAdam.__init__ = patched_adam_init
    FusedAdam.step = patched_adam_step
    MegatronOptimizer.clip_grad_norm = patched_clip_grad_norm

    mark_patched(FusedAdam, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] enabled: clip coefficient is applied inside the "
        "Primus HIP Adam kernel; standalone multi_tensor_scale is disabled."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Fuse global gradient clipping into Transformer Engine FusedAdam",
    priority=39,
    condition=lambda ctx: _enabled(),
)
def patch_te_fused_adam_clip(ctx: PatchContext) -> None:
    del ctx
    _install_patch()
