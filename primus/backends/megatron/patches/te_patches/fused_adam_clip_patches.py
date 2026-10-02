###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fuse global gradient clipping into the ROCm Transformer Engine Adam kernel.

Set ``PRIMUS_FUSED_ADAM_CLIP=1`` to retain Megatron's global-norm calculation
but defer application of its clip coefficient to Adam while the gradient is
already resident in registers.

GPT-OSS gives TE FP32 gradients and FP32 optimizer parameters. Moments may be
FP32 or BF16; arithmetic remains FP32 in either case. TE's ROCm non-capturable
path is important here: it uses one custom
device-metadata Adam launch instead of the generic path's many launches. This
patch preserves that layout and compiles a small Primus-owned HIP extension
with a device-resident norm argument. It does not modify the TE installation.

The global norm and clip coefficient remain device-resident. The Adam kernel
loads the one-element norm tensor and derives the coefficient in-kernel, so the
optimizer no longer drains the GPU pipeline through ``Tensor.item()``.
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.te_fused_adam_clip"
_EXTENSION_NAME = "primus_te_fused_adam_clip_v5"
_extension = None


def _enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1"


def _bf16_writeback_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_BF16_WRITEBACK", "0") == "1"


def _mcore_master_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_MCORE_MASTER", "0") == "1"


def _clip_coeff(total_norm: float, max_norm: float) -> float:
    """Return Megatron's gradient clipping coefficient."""

    return min(1.0, float(max_norm) / (float(total_norm) + 1.0e-6))


def _get_grad_norm_tensor(optimizer) -> torch.Tensor:
    """Megatron's FP32 L2 norm path without its final host synchronization."""

    from megatron.core.optimizer import clip_grads

    grads_for_norm = optimizer.get_main_grads_for_grad_norm()
    if grads_for_norm:
        dummy_overflow_buf = getattr(optimizer, "_primus_norm_overflow_buf", None)
        grad_device = grads_for_norm[0].device
        if dummy_overflow_buf is None or dummy_overflow_buf.device != grad_device:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=grad_device)
            optimizer._primus_norm_overflow_buf = dummy_overflow_buf
        local_norm, _ = clip_grads.multi_tensor_applier(
            clip_grads.l2_norm_impl,
            dummy_overflow_buf,
            [grads_for_norm],
            False,
        )
        total_norm_squared = local_norm.square()
    else:
        total_norm_squared = torch.zeros(1, dtype=torch.float32, device="cuda")

    torch.distributed.all_reduce(
        total_norm_squared,
        op=torch.distributed.ReduceOp.SUM,
        group=optimizer.get_grad_stats_parallel_group(),
    )
    return total_norm_squared.sqrt()


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


def _metadata_key(
    chunk_size: int, tensor_lists, writeback_by_param_ptr
) -> tuple[Any, ...]:
    return (
        int(chunk_size),
        tuple(
            (tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.device)
            for tensor_list in tensor_lists
            for tensor in tensor_list
        ),
        tuple(
            (
                tensor.data_ptr(),
                writeback_by_param_ptr.get(tensor.data_ptr()).data_ptr()
                if tensor.data_ptr() in writeback_by_param_ptr
                else 0,
            )
            for tensor in tensor_lists[1]
        ),
    )


def _build_metadata(chunk_size: int, tensor_lists, writeback_by_param_ptr=None):
    if len(tensor_lists) != 4:
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP currently requires TE's four-list "
            f"FP32 Adam path (grad, param, exp_avg, exp_avg_sq), got {len(tensor_lists)} lists"
        )
    num_tensors = len(tensor_lists[0])
    if num_tensors == 0 or any(len(tensors) != num_tensors for tensors in tensor_lists):
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP received inconsistent empty tensor lists"
        )

    device = tensor_lists[0][0].device
    moment_dtype = tensor_lists[2][0].dtype
    if moment_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP requires FP32 or BF16 Adam moments, got "
            f"{moment_dtype}"
        )
    addresses = []
    sizes = []
    block_to_tensor = []
    chunk_offsets = []
    total_chunks = 0
    writeback_by_param_ptr = writeback_by_param_ptr or {}

    for tensor_idx in range(num_tensors):
        tensors = [tensor_lists[list_idx][tensor_idx] for list_idx in range(4)]
        if any(tensor.device != device for tensor in tensors):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires one device per Adam call"
            )
        expected_dtypes = (torch.float32, torch.float32, moment_dtype, moment_dtype)
        if any(
            tensor.dtype != expected
            for tensor, expected in zip(tensors, expected_dtypes)
        ):
            dtypes = tuple(tensor.dtype for tensor in tensors)
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires FP32 gradients/parameters and matching "
                f"FP32 or BF16 moments; tensor {tensor_idx} has dtypes {dtypes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires contiguous optimizer tensors"
            )
        if any(tensor.numel() != tensors[0].numel() for tensor in tensors[1:]):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP received mismatched tensor sizes"
            )

        writeback = writeback_by_param_ptr.get(tensors[1].data_ptr())
        if writeback is not None:
            if writeback.device != device or writeback.dtype != torch.bfloat16:
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires a BF16 destination "
                    "on the optimizer device"
                )
            if not writeback.is_contiguous() or writeback.numel() != tensors[1].numel():
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_BF16_WRITEBACK received an incompatible destination"
                )
        addresses.extend(tensor.data_ptr() for tensor in tensors)
        addresses.append(0 if writeback is None else writeback.data_ptr())
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
        0 if moment_dtype == torch.float32 else 1,
    )


def _prepare_bf16_writeback(distributed_optimizer, fused_adam_type) -> None:
    """Map each FP32 master shard to its BF16 parameter-buffer destination."""

    if not _bf16_writeback_enabled():
        return
    if getattr(distributed_optimizer, "is_stub_optimizer", False):
        return
    if distributed_optimizer.ddp_config.use_megatron_fsdp:
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_BF16_WRITEBACK does not support Megatron FSDP"
        )
    if distributed_optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires separate FP32 master parameters"
        )

    inner_optimizer = distributed_optimizer.optimizer
    if not isinstance(inner_optimizer, fused_adam_type):
        raise TypeError(
            "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires Transformer Engine FusedAdam"
        )

    cached = getattr(distributed_optimizer, "_primus_bf16_writeback_cache", None)
    if cached is not None:
        writeback_by_param_ptr, writeback_tensors = cached
        inner_optimizer._primus_bf16_writeback_by_param_ptr = writeback_by_param_ptr
        inner_optimizer._primus_bf16_writeback_tensors = writeback_tensors
        distributed_optimizer._primus_bf16_writeback_active = True
        return

    writeback_by_param_ptr = {}
    writeback_tensors = []
    for shard_main_group, model_group in zip(
        distributed_optimizer.shard_fp32_from_float16_groups,
        distributed_optimizer.model_float16_groups,
    ):
        for shard_main_param, model_param in zip(shard_main_group, model_group):
            # GPT-OSS uses BF16 model buffers. Fail closed instead of bypassing
            # Megatron's separate FP8 quantization path for another workload.
            if model_param.dtype != torch.bfloat16:
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires BF16 model parameters, "
                    f"got {model_param.dtype}"
                )
            param_range_map = distributed_optimizer._get_model_param_range_map(
                model_param
            )
            world_range = param_range_map["gbuf_world_in_bucket"]
            if world_range.size != shard_main_param.nelement():
                raise RuntimeError(
                    "BF16 writeback shard range does not match its FP32 master"
                )
            gbuf_index, _, bucket_id = distributed_optimizer.model_param_gbuf_map[
                model_param
            ]
            param_buffer = (
                distributed_optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
            )
            shard_model_param = param_buffer.view(-1)[
                world_range.start : world_range.end
            ]
            if shard_model_param.dtype != torch.bfloat16:
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires a BF16 parameter buffer"
                )
            writeback_by_param_ptr[shard_main_param.data_ptr()] = shard_model_param
            writeback_tensors.append(shard_model_param)

    expected_masters = sum(
        len(group) for group in distributed_optimizer.shard_fp32_from_float16_groups
    )
    if len(writeback_by_param_ptr) != expected_masters:
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_BF16_WRITEBACK did not map every FP32 master"
        )

    # Retain the views until the asynchronous Adam launch has consumed them.
    distributed_optimizer._primus_bf16_writeback_cache = (
        writeback_by_param_ptr,
        writeback_tensors,
    )
    inner_optimizer._primus_bf16_writeback_by_param_ptr = writeback_by_param_ptr
    inner_optimizer._primus_bf16_writeback_tensors = writeback_tensors
    distributed_optimizer._primus_bf16_writeback_active = True


def _copy_only_fp32_model_groups(distributed_optimizer) -> None:
    """Finish the copy pass for native FP32 model parameters only."""

    for shard_main_group, model_group in zip(
        distributed_optimizer.shard_fp32_groups,
        distributed_optimizer.model_fp32_groups,
    ):
        for shard_main_param, model_param in zip(shard_main_group, model_group):
            param_range_map = distributed_optimizer._get_model_param_range_map(
                model_param
            )
            world_range = param_range_map["gbuf_world_in_bucket"]
            if world_range.size != shard_main_param.nelement():
                raise RuntimeError(
                    "FP32 model shard range does not match its main parameter"
                )
            gbuf_index, _, bucket_id = distributed_optimizer.model_param_gbuf_map[
                model_param
            ]
            param_buffer = (
                distributed_optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
            )
            shard_model_param = param_buffer.view(-1)[
                world_range.start : world_range.end
            ]
            shard_model_param.copy_(shard_main_param)


def _chained_step_with_device_clip(optimizer, fused_adam_type):
    """Run ChainedOptimizer.step while deferring clipping to each fused Adam."""

    found_inf_flag = optimizer.prepare_grads()
    if found_inf_flag:
        return False, None, None
    if not optimizer.grads_states_parallel_group_is_shared():
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP requires chained optimizers to share "
            "their gradient-statistics process group"
        )

    grad_norm = _get_grad_norm_tensor(optimizer)
    for child_optimizer in optimizer.chained_optimizers:
        if getattr(child_optimizer, "is_stub_optimizer", False):
            continue
        if not child_optimizer.get_parameters():
            continue
        if child_optimizer.config.clip_grad <= 0.0:
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires a positive clip_grad")
        inner_optimizer = child_optimizer.optimizer
        if not isinstance(inner_optimizer, fused_adam_type):
            raise TypeError(
                "PRIMUS_FUSED_ADAM_CLIP requires Transformer Engine FusedAdam, got "
                f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}"
            )
        inner_optimizer._primus_clip_norm = grad_norm
        inner_optimizer._primus_clip_max_norm = float(child_optimizer.config.clip_grad)
        _prepare_bf16_writeback(child_optimizer, fused_adam_type)

    num_zeros_in_grad = (
        optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else None
    )
    update_successful = optimizer.step_with_ready_grads()
    # Returning the device tensor would make Megatron's per-step logger format
    # it as a Python float, reinstating the synchronization this patch removes.
    # The norm has already served its training purpose inside Adam, so suppress
    # the optional logging statistic.
    return update_successful, None, num_zeros_in_grad


def _install_patch() -> None:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    if is_patched(FusedAdam, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already installed; skipping.")
        return

    extension = _load_extension()
    original_adam_init = FusedAdam.__init__
    original_adam_step = FusedAdam.step
    original_copy_main_params = DistributedOptimizer._copy_main_params_to_model_params
    original_optimizer_config_post_init = OptimizerConfig.__post_init__

    @wraps(original_optimizer_config_post_init)
    def patched_optimizer_config_post_init(self):
        original_optimizer_config_post_init(self)
        if not _mcore_master_enabled():
            return
        if not self.use_precision_aware_optimizer:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_MCORE_MASTER requires use_precision_aware_optimizer"
            )
        if self.main_params_dtype != torch.float32:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_MCORE_MASTER requires FP32 main_params_dtype"
            )
        # MXFP4 is provided by Primus-Turbo rather than Megatron's fp8_recipe,
        # so upstream classifies it as the generic no-FP8 path and moves master
        # ownership into TE. Keep MCore's FP32 shards and use precision-aware
        # mode only to select BF16 moments.
        self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False

    @wraps(original_adam_init)
    def patched_adam_init(self, *args, **kwargs):
        if kwargs.get("capturable", False):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP preserves TE's optimized non-capturable ROCm path"
            )
        original_adam_init(self, *args, **kwargs)
        self._primus_clip_norm = None
        self._primus_clip_max_norm = 1.0
        self._primus_adam_metadata_cache = {}
        self._primus_bf16_writeback_by_param_ptr = {}
        self._primus_bf16_writeback_tensors = []

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
            writeback_by_param_ptr = self._primus_bf16_writeback_by_param_ptr
            key = _metadata_key(chunk_size, tensor_lists, writeback_by_param_ptr)
            metadata = self._primus_adam_metadata_cache.get(key)
            if metadata is None:
                metadata = _build_metadata(
                    chunk_size, tensor_lists, writeback_by_param_ptr
                )
                self._primus_adam_metadata_cache[key] = metadata
            (
                addresses,
                sizes,
                block_to_tensor,
                chunk_offsets,
                total_chunks,
                moment_dtype,
            ) = metadata
            if self._primus_clip_norm is None:
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_CLIP did not receive a device gradient norm"
                )
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
                int(moment_dtype),
                self._primus_clip_norm,
                float(self._primus_clip_max_norm),
            )

        # FusedAdam.step calls this attribute through TE's existing
        # multi_tensor_applier, so all Python-side grouping remains unchanged.
        self.multi_tensor_adam = fused_multi_tensor_adam

    @wraps(original_adam_step)
    def patched_adam_step(self, closure=None, grad_scaler=None):
        if grad_scaler is not None:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP does not currently compose with an AMP GradScaler"
            )
        try:
            return original_adam_step(self, closure=closure, grad_scaler=None)
        finally:
            self._primus_clip_norm = None
            self._primus_clip_max_norm = 1.0

    def patched_copy_main_params_to_model_params(self):
        if not getattr(self, "_primus_bf16_writeback_active", False):
            return original_copy_main_params(self)
        try:
            # The Adam kernel already wrote all BF16 destinations. Native FP32
            # model groups, if present, still use Megatron's ordinary copy.
            _copy_only_fp32_model_groups(self)
        finally:
            self._primus_bf16_writeback_active = False

    def patched_clip_grad_norm(self, clip_grad: float) -> None:
        # Preserve Megatron's norm kernel and global SUM reduction, but retain
        # the result on device and omit clip_grad_by_total_norm_fp32's scale pass.
        grad_norm = _get_grad_norm_tensor(self)
        inner_optimizer = self.optimizer
        if not isinstance(inner_optimizer, FusedAdam):
            raise TypeError(
                "PRIMUS_FUSED_ADAM_CLIP requires Transformer Engine FusedAdam, got "
                f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}"
            )
        inner_optimizer._primus_clip_norm = grad_norm
        inner_optimizer._primus_clip_max_norm = float(clip_grad)
        # Megatron only uses this return value for optional logging after the
        # update. Falling through with None avoids a device-to-host synchronization.

    @torch.no_grad()
    def patched_chained_step(self):
        """ChainedOptimizer.step with device clipping delegated to each Adam."""
        return _chained_step_with_device_clip(self, FusedAdam)

    FusedAdam.__init__ = patched_adam_init
    FusedAdam.step = patched_adam_step
    OptimizerConfig.__post_init__ = patched_optimizer_config_post_init
    MegatronOptimizer.clip_grad_norm = patched_clip_grad_norm
    ChainedOptimizer.step = patched_chained_step
    DistributedOptimizer._copy_main_params_to_model_params = (
        patched_copy_main_params_to_model_params
    )

    # The existing TP1/PP1 synchronization-skip patch has already established
    # that the model-parallel reduction is a no-op. Keep a tensor statistic on
    # device instead of converting it back to a host float after Adam.
    import megatron.training.training as megatron_training

    reduce_max_stat = megatron_training.reduce_max_stat_across_model_parallel_group
    if getattr(reduce_max_stat, "__name__", "") == "_passthrough_reduce_max":

        def _deferred_passthrough_reduce_max(value):
            return value

        megatron_training.reduce_max_stat_across_model_parallel_group = (
            _deferred_passthrough_reduce_max
        )

    mark_patched(FusedAdam, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] enabled: device gradient norm and clip coefficient "
        "are consumed inside the Primus HIP Adam kernel; host item() and standalone "
        "multi_tensor_scale are disabled"
        + (
            "; FP32-master to BF16 model-buffer writeback is fused into Adam."
            if _bf16_writeback_enabled()
            else "."
        )
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
