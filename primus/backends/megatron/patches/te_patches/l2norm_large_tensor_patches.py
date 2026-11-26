###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Split tensors of more than 2^30 elements before Transformer Engine's l2norm.

On ROCm, TE's ``multi_tensor_l2norm`` reads out of bounds when it is given more
than one tensor and one of them has 2^31 or more elements: a memory fault in
``custom_multi_tensor_l2norm_kernel``, or a wrong norm when the address happens
to be mapped. A lone tensor of that size is fine. Seen with TE 2.19.0.dev0.

Megatron passes every gradient to one l2norm call for the grad norm, and
Primus-Turbo's grouped GEMM keeps all experts of a layer in one parameter, so
for example MiniMax-M3's 128 experts of 6144 x 6144 (fc1) are 4.8e9 elements.
The wrapper gives the kernel views of at most 2^30 elements instead; for the
total norm (``per_tensor=False``) that is the same norm.
"""

from functools import wraps

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.te_l2norm_large_tensor"
_MAX_NUMEL = 1 << 30


def _pieces(tensor: torch.Tensor):
    if tensor.numel() <= _MAX_NUMEL:
        return [tensor]
    return list(tensor.reshape(-1).split(_MAX_NUMEL))


def split_large_tensors(l2norm):
    """``l2norm`` with every tensor of more than ``_MAX_NUMEL`` elements split into views."""

    @wraps(l2norm)
    def l2norm_split(chunk_size, noop_flag, tensor_lists, per_tensor=False):
        if not per_tensor and any(t.numel() > _MAX_NUMEL for tensors in tensor_lists for t in tensors):
            tensor_lists = [[p for t in tensors for p in _pieces(t)] for tensors in tensor_lists]
        return l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    return l2norm_split


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Split tensors of more than 2^30 elements before TE multi_tensor_l2norm (grad norm)",
)
def patch_te_l2norm_large_tensor(ctx: PatchContext):
    del ctx
    from megatron.core.optimizer import clip_grads

    if is_patched(clip_grads, _PATCH_KEY):
        return
    l2norm = split_large_tensors(clip_grads.l2_norm_impl)
    rebound = rebind_everywhere(clip_grads, "l2_norm_impl", l2norm)
    if hasattr(clip_grads, "multi_tensor_l2norm"):
        rebound += rebind_everywhere(clip_grads, "multi_tensor_l2norm", l2norm)
    mark_patched(clip_grads, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] wrapped the l2norm in {', '.join(sorted(set(rebound)))}")
