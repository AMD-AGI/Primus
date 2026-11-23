###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run the gradient-clip scale of BF16 gradients as one Triton launch per tensor.

``clip_grad_by_total_norm_fp32`` scales every gradient by the on-device clip coefficient on
every step with TE's ``multi_tensor_scale_tensor``, which on ROCm runs far below the memory
bandwidth (MiniMax-M3.1-Flash proxy on one MI455X, 8.2e9 BF16 gradient elements: 12.9 ms of
a 372 ms step in 520 launches). With ``bitwise_clip_scale`` the in-place BF16 case goes to
``core/fusions/clip_scale.py`` instead: the same widen, multiply, round-to-BF16 per element,
compared with TE on the first call (``bitwise_check``); every other call, and a run whose check
fails, keeps TE. The scale is per rank over that rank's gradients, with no collective.
"""

from functools import wraps

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.bitwise_clip_scale"


def make_bitwise_scale(original, applier):
    """``original`` (TE's ``multi_tensor_scale_tensor``, called through ``applier``) with the
    in-place BF16 case taken by the Triton kernel once it has matched ``original`` bit for bit."""
    logged = []

    def log_once(msg):
        if not logged:
            logged.append(msg)
            log_rank_0(f"[Patch:{_PATCH_KEY}] {msg}")

    @wraps(original)
    def scale_tensor(chunk_size, noop_flag, tensor_lists, coeff):
        # Imported here: only a run with the switch on needs Triton.
        from primus.backends.megatron.core.fusions import clip_scale

        # eligibility before any work; everything else is TE's call, unchanged
        if not clip_scale.eligible(tensor_lists, coeff):
            log_once("first call is not an in-place scale of contiguous BF16 CUDA tensors; using TE")
            return original(chunk_size, noop_flag, tensor_lists, coeff)

        def te_scale(tensors, c):
            # the real original path: TE's applier (its chunk size) around TE's kernel
            noop = noop_flag.new_zeros(noop_flag.shape)
            applier(original, noop, [tensors, tensors], c)

        if not clip_scale.checked(te_scale, coeff):
            return original(chunk_size, noop_flag, tensor_lists, coeff)
        log_once("in-place BF16 gradient scale runs on the Triton kernel")
        clip_scale.scale_tensors(tensor_lists[1], coeff)
        return None

    scale_tensor._primus_bitwise_clip_scale = True
    return scale_tensor


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Bitwise Triton kernel for the gradient-clip scale of BF16 gradients",
    condition=lambda ctx: bool(getattr(get_args(ctx), "bitwise_clip_scale", False)),
)
def patch_bitwise_clip_scale(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.optimizer import clip_grads

    if is_patched(clip_grads, _PATCH_KEY):
        return
    original = getattr(clip_grads, "multi_tensor_scale_tensor_impl", None)
    if original is None:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] no multi_tensor_scale_tensor (TE not installed); nothing to replace"
        )
        return
    wrapped = make_bitwise_scale(original, clip_grads.multi_tensor_applier)
    rebound = rebind_everywhere(clip_grads, "multi_tensor_scale_tensor_impl", wrapped)
    mark_patched(clip_grads, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] wrapped multi_tensor_scale_tensor_impl in {', '.join(sorted(set(rebound)))}"
    )
