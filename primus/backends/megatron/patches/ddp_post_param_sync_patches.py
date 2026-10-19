###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fast path for ``_ParamAndGradBucketGroup._post_param_sync`` (Megatron-LM main).

After every param all-gather, upstream walks every parameter: it slices
``bucket.param_data``, ``view_as``-es the slice, compares ``data_ptr`` and copies only
when the parameter was redirected away from the bucket buffer (inference packing).
It then rescans every parameter for quantized storage. In training nothing is ever
copied, but the walk runs on the critical path of the next forward (with
``overlap_param_gather``) and costs ~4-5 ms per step on the DeepSeek-V4 proxy.

The replacement is behaviour-identical: it precomputes, per bucket group, the plain
parameters with their byte offset into the 1-D bucket buffer and the static list of
quantized parameters. Each call then compares ``param.data_ptr()`` against
``bucket.param_data.data_ptr() + offset`` and copies exactly when upstream would.
The MXFP8 grad-buffer-reuse path is left to the original implementation.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _has_post_param_sync(ctx: PatchContext) -> bool:
    try:
        from megatron.core.distributed.param_and_grad_buffer import (
            _ParamAndGradBucketGroup,
        )
    except ImportError:
        return False
    return hasattr(_ParamAndGradBucketGroup, "_post_param_sync")


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
            return original(self)
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
