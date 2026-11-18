###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fast path for ``_ParamAndGradBucketGroup._post_param_sync`` (Megatron-LM main).

After every param all-gather upstream slices and ``view_as``-es every parameter's bucket
view to find the few it must copy back, then rescans all parameters for quantized
storage: ~4-5 ms per step on the DeepSeek-V4 proxy. The replacement caches each
parameter's byte offset and the quantized-parameter list and copies exactly when
upstream would. The MXFP8 grad-buffer-reuse path keeps the original implementation.
"""

import inspect

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _has_post_param_sync(ctx: PatchContext) -> bool:
    """Only Megatron-LM main's ``_post_param_sync`` has the per-parameter copy-back.

    Megatron-core 0.19's version only post-processes quantized parameters, so the
    replacement would add behaviour there.
    """
    try:
        from megatron.core.distributed.param_and_grad_buffer import (
            _ParamAndGradBucketGroup,
        )
    except ImportError:
        return False
    method = getattr(_ParamAndGradBucketGroup, "_post_param_sync", None)
    if method is None:
        return False
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    return "view_as" in source and ".copy_(" in source


@register_patch(
    "megatron.ddp.post_param_sync_fast_path",
    backend="megatron",
    phase="before_train",
    description="Replace the per-parameter slice/view_as walk in _post_param_sync with cached offsets",
    condition=_has_post_param_sync,
)
def patch_post_param_sync(ctx: PatchContext):
    import megatron.core.distributed.param_and_grad_buffer as pgb

    group_cls = pgb._ParamAndGradBucketGroup
    original = group_cls._post_param_sync
    if getattr(original, "_primus_fast_path", False):
        return

    def _build_plan(group):
        buckets = []
        for bucket in group.buckets:
            if bucket.param_data is None:
                continue
            flat = bucket.param_data.dim() == 1 and bucket.param_data.is_contiguous()
            element_size = bucket.param_data.element_size()
            plain = [
                (param, bucket.param_to_index[param][0] * element_size)
                for param in bucket.params
                if not (pgb._param_uses_quantized_storage(param) or pgb.is_grouped_tensor(param))
            ]
            buckets.append((bucket, flat, plain))
        quantized = [
            param
            for bucket in group.buckets
            for param in bucket.params
            if pgb._param_uses_quantized_storage(param)
        ]
        return buckets, quantized

    def _post_param_sync(self):
        if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            original(self)
            return
        plan = self.__dict__.get("_primus_post_param_sync_plan")
        if plan is None:
            plan = self._primus_post_param_sync_plan = _build_plan(self)
        buckets, quantized = plan
        for bucket, flat, plain in buckets:
            base = bucket.param_data.data_ptr()
            for param, offset in plain:
                if flat and param.data_ptr() == base + offset:
                    continue
                start, end = bucket.param_to_index[param]
                gathered_param = bucket.param_data[start:end].view_as(param)
                if param.data_ptr() != gathered_param.data_ptr():
                    param.data.copy_(gathered_param)
        if quantized:
            pgb.post_all_gather_processing(quantized)

    _post_param_sync._primus_fast_path = True
    group_cls._post_param_sync = _post_param_sync
    log_rank_0("[Patch:megatron.ddp.post_param_sync_fast_path] _post_param_sync uses cached param offsets")
