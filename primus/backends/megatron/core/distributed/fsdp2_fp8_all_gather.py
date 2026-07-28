###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FP8 all-gather tensor subclass for FSDP2.

Wraps BF16 or FP32 parameters so that FSDP2 communicates FP8 data
(1 byte/element) instead of BF16/FP32, reducing all-gather volume.

Approach A: Keep Weight in FP8 After All-Gather.
Uses a precomputed global scale (all-reduced amax across all shards) so that
the unsharded FP8 weight has a single consistent scale, compatible with
tensorwise GEMM on HIPBLASLT. After all-gather, the unsharded weight stays
in FP8 via FP8UnshardedWeightTensor, saving ~50% memory vs BF16 baseline.

Reference: torchao WeightWithDynamicFloat8CastTensor
(ao/torchao/float8/fsdp_utils.py)
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import triton
import triton.language as tl
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
)
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate
from torch.library import triton_op, wrap_triton

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


def _get_fp8_dtype(fmt: Format) -> torch.dtype:
    if fmt == Format.E4M3:
        return float8_e4m3
    elif fmt == Format.HYBRID:
        return float8_e4m3
    else:
        raise ValueError(f"Unsupported FP8 format for all-gather: {fmt}")


@triton.jit
def _quantize_fp8_prescaled_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)
    v = x * scale
    v = tl.clamp(v, min=-FP8_MAX, max=FP8_MAX)
    tl.store(output_ptr + offsets, v.to(output_ptr.dtype.element_ty), mask=mask)


@triton_op("primus::quantize_fp8_prescaled", mutates_args=())
def quantize_fp8_prescaled(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    output = torch.empty_like(x, dtype=fp8_dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_quantize_fp8_prescaled_kernel)[grid](
        x,
        output,
        scale,
        n_elements,
        fp8_max,
        BLOCK_SIZE=1024,
    )
    return output, scale_inv.clone()


@quantize_fp8_prescaled.register_fake
def _quantize_fp8_prescaled_fake(x, fp8_dtype, scale, scale_inv):
    return torch.empty_like(x, dtype=fp8_dtype), scale_inv.clone()


# ---------------------------------------------------------------------------
# Stochastic rounding variant: adds uniform [-0.5, 0.5) noise before
# truncation to make quantization error unbiased in expectation.
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_fp8_prescaled_stochastic_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    seed,
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)
    v = x * scale
    v = tl.clamp(v, min=-FP8_MAX, max=FP8_MAX)
    noise = tl.rand(seed, offsets) - 0.5
    v_sr = v + noise
    tl.store(output_ptr + offsets, v_sr.to(output_ptr.dtype.element_ty), mask=mask)


@triton_op("primus::quantize_fp8_prescaled_stochastic", mutates_args=())
def quantize_fp8_prescaled_stochastic(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(fp8_dtype).max
    output = torch.empty_like(x, dtype=fp8_dtype)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_quantize_fp8_prescaled_stochastic_kernel)[grid](
        x,
        output,
        scale,
        seed,
        n_elements,
        fp8_max,
        BLOCK_SIZE=1024,
    )
    return output, scale_inv.clone()


@quantize_fp8_prescaled_stochastic.register_fake
def _quantize_fp8_prescaled_stochastic_fake(x, fp8_dtype, scale, scale_inv, seed):
    return torch.empty_like(x, dtype=fp8_dtype), scale_inv.clone()


@torch.compile(mode="max-autotune-no-cudagraphs")
def _foreach_fp8_quantize(
    inner_tensors: list[torch.Tensor],
    fp8_outputs: list[torch.Tensor],
    scales: torch.Tensor,
    fp8_max: float,
):
    """Batch-quantize all FP8 parameters in a single fused operation.

    Uses _foreach_* ops fused by torch.compile to replace 456 individual
    kernel launches with ~1-2 fused kernels. Float32 intermediate ensures
    bitwise equivalence with the per-tensor quantize_fp8_prescaled path.
    """
    scales_list = list(scales.unbind())
    f32 = [t.float() for t in inner_tensors]
    scaled = torch._foreach_mul(f32, scales_list)
    clamped = torch._foreach_clamp_min(torch._foreach_clamp_max(scaled, fp8_max), -fp8_max)
    torch._foreach_copy_(fp8_outputs, clamped)


@torch.compile(mode="max-autotune-no-cudagraphs")
def _foreach_fp8_quantize_stochastic(
    inner_tensors: list[torch.Tensor],
    fp8_outputs: list[torch.Tensor],
    scales: torch.Tensor,
    fp8_max: float,
):
    """Batch-quantize with stochastic rounding for unbiased FP8 quantization.

    Same as _foreach_fp8_quantize but adds uniform [-0.5, 0.5) noise before
    the truncating cast to FP8, making quantization error zero in expectation.
    """
    scales_list = list(scales.unbind())
    f32 = [t.float() for t in inner_tensors]
    scaled = torch._foreach_mul(f32, scales_list)
    noise = [torch.rand_like(s) - 0.5 for s in scaled]
    noisy = torch._foreach_add(scaled, noise)
    clamped = torch._foreach_clamp_min(torch._foreach_clamp_max(noisy, fp8_max), -fp8_max)
    torch._foreach_copy_(fp8_outputs, clamped)


class WeightWithFP8AllGatherTensor(torch.Tensor):
    """Tensor subclass that quantizes to FP8 before FSDP2 all-gather.

    Wraps a BF16 or FP32 parameter tensor. FSDP2 calls fsdp_pre_all_gather
    to get FP8-quantized data for the collective, then fsdp_post_all_gather
    to return an FP8UnshardedWeightTensor (keeping weight in FP8).

    Uses a precomputed global scale (set by precompute_fp8_scales_for_fsdp)
    so all shards quantize with the same scale, producing a single consistent
    scale for the unsharded weight.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, fp8_config: Float8QuantConfig):
        if fp8_config.granularity != ScalingGranularity.TENSORWISE:
            raise ValueError(
                f"FP8 all-gather only supports TENSORWISE granularity, " f"got {fp8_config.granularity}"
            )
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, fp8_config: Float8QuantConfig):
        self._fp8_config = fp8_config
        self._tensor = tensor
        # Transient: set by precompute_fp8_scales_for_fsdp(), NOT in
        # __tensor_flatten__ (must be hashable/static for torch.compile guards).
        self._precomputed_scale = None
        self._precomputed_scale_inv = None
        self._cached_fp8_data = None
        self._use_cpp_quantize = False
        self._stochastic_rounding = False
        self._sr_counter = 0
        self._deq_after_ag = False

    def inner_data(self) -> torch.Tensor:
        """Return the underlying data tensor (bypassing subclass dispatch)."""
        return self._tensor

    def fsdp_pre_all_gather(self, mesh):
        if self._cached_fp8_data is not None:
            return (self._cached_fp8_data,), (self._precomputed_scale_inv, self._tensor.numel())
        fp8_dtype = _get_fp8_dtype(self._fp8_config.format)
        if self._precomputed_scale is None:
            raise RuntimeError(
                "precompute_fp8_scales_for_fsdp() must be called before the first forward pass"
            )
        with torch.no_grad():
            if self._use_cpp_quantize:
                from primus_turbo.pytorch.ops.quantization import quantize_fp8

                fp8_data, scale_inv = quantize_fp8(
                    self._tensor,
                    fp8_dtype,
                    self._fp8_config.granularity,
                    scale=self._precomputed_scale,
                )
            elif self._stochastic_rounding:
                if self._precomputed_scale_inv is None:
                    raise RuntimeError(
                        "precompute_fp8_scales_for_fsdp() must set scale_inv before stochastic-rounding quantize"
                    )
                if not self._tensor.is_contiguous():
                    raise RuntimeError("FP8 stochastic-rounding quantize requires contiguous input")
                self._sr_counter += 1
                fp8_data, scale_inv = quantize_fp8_prescaled_stochastic(
                    self._tensor,
                    fp8_dtype,
                    self._precomputed_scale,
                    self._precomputed_scale_inv,
                    seed=self._sr_counter,
                )
            else:
                if self._precomputed_scale_inv is None:
                    raise RuntimeError("precompute_fp8_scales_for_fsdp() must set scale_inv for Triton path")
                if not self._tensor.is_contiguous():
                    raise RuntimeError("FP8 prescaled quantize requires contiguous input")
                fp8_data, scale_inv = quantize_fp8_prescaled(
                    self._tensor,
                    fp8_dtype,
                    self._precomputed_scale,
                    self._precomputed_scale_inv,
                )
        return (fp8_data,), (scale_inv, self._tensor.numel())

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        scale_inv, shard_numel = metadata
        (fp8_gathered,) = all_gather_outputs

        if self._deq_after_ag:
            bf16_weight = fp8_gathered.to(torch.bfloat16) * scale_inv
            if out is not None:
                out.data.copy_(bf16_weight)
                return
            return bf16_weight, (fp8_gathered,)

        if out is not None:
            # Reshard path: FSDP already filled the FP8 buffer via
            # all-gather into tracked storage. Just update the scale.
            target = out.data if isinstance(out, nn.Parameter) else out
            if isinstance(target, FP8UnshardedWeightTensor):
                target._scale_inv = scale_inv
            return
        return FP8UnshardedWeightTensor(fp8_gathered, scale_inv, torch.bfloat16, self._fp8_config), (
            fp8_gathered,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithFP8AllGatherTensor(args[0]._tensor.detach(), args[0]._fp8_config)

        if func == torch.ops.aten.copy_.default:
            src = args[1]
            if isinstance(src, WeightWithFP8AllGatherTensor):
                src = src._tensor
            args[0]._tensor.copy_(src)
            return args[0]

        fp8_config = None
        inner_dtype = None

        def unwrap(t):
            nonlocal fp8_config, inner_dtype
            if fp8_config is None:
                fp8_config = t._fp8_config
                inner_dtype = t._tensor.dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(WeightWithFP8AllGatherTensor, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out

        if func == torch.ops.aten._to_copy.default:
            target_dtype = (kwargs or {}).get("dtype", None)
            if target_dtype is not None and target_dtype != inner_dtype:
                return out

        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithFP8AllGatherTensor(x, fp8_config),
            out,
        )

    def __tensor_flatten__(self):
        # Guard for torch.compile tracing: dynamo may inspect this tensor
        # via __tensor_flatten__ during __init__ before attributes are set.
        config = getattr(self, "_fp8_config", None)
        if config is None or not hasattr(self, "_tensor"):
            return [], {}
        # Float8QuantConfig is a mutable dataclass (not hashable),
        # so store fields individually as hashable metadata.
        return ["_tensor"], {
            "format": config.format,
            "granularity": config.granularity,
            "strategy": config.strategy,
            "scale_dtype": config.scale_dtype,
            "block_size": config.block_size,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        config = Float8QuantConfig(
            format=metadata["format"],
            granularity=metadata["granularity"],
            strategy=metadata["strategy"],
            scale_dtype=metadata["scale_dtype"],
            block_size=metadata["block_size"],
        )
        return WeightWithFP8AllGatherTensor(inner_tensors["_tensor"], config)

    def __repr__(self):
        return (
            f"WeightWithFP8AllGatherTensor("
            f"shape={list(self._tensor.shape)}, "
            f"dtype={self._tensor.dtype}, "
            f"device={self._tensor.device}, "
            f"granularity={self._fp8_config.granularity})"
        )


_unsharded_ops_to_preserve = {
    torch.ops.aten.as_strided.default,
    torch.ops.aten.view.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.clone.default,
}


class FP8UnshardedWeightTensor(torch.Tensor):
    """Lightweight wrapper for the unsharded FP8 weight after all-gather.

    Holds raw FP8 data + scalar inverse scale. Declares dtype=orig_dtype
    (bfloat16) so PyTorch shape/dtype inference sees BF16, but stores 1
    byte/element. FP8 linear layers detect this subclass and extract the
    pre-quantized data directly, skipping redundant quantization.
    """

    @staticmethod
    def __new__(
        cls,
        fp8_data: torch.Tensor,
        scale_inv: torch.Tensor,
        orig_dtype: torch.dtype,
        fp8_config: Float8QuantConfig,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            fp8_data.size(),
            strides=fp8_data.stride(),
            storage_offset=fp8_data.storage_offset(),
            dtype=orig_dtype,
            layout=fp8_data.layout,
            device=fp8_data.device,
            requires_grad=False,
        )

    def __init__(
        self,
        fp8_data: torch.Tensor,
        scale_inv: torch.Tensor,
        orig_dtype: torch.dtype,
        fp8_config: Float8QuantConfig,
    ):
        self._fp8_data = fp8_data
        self._scale_inv = scale_inv
        self._orig_dtype = orig_dtype
        self._fp8_config = fp8_config

    def get_fp8_data_and_scale_inv(self):
        return self._fp8_data, self._scale_inv

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            # Must preserve subclass: nn.Parameter(data) calls detach()
            # and asserts type(detach_result) == type(data).
            self = args[0]
            return FP8UnshardedWeightTensor(
                self._fp8_data.detach(),
                self._scale_inv.detach(),
                self._orig_dtype,
                self._fp8_config,
            )

        if func in _unsharded_ops_to_preserve:
            self = args[0]
            new_data = func(self._fp8_data, *args[1:], **(kwargs or {}))
            return FP8UnshardedWeightTensor(new_data, self._scale_inv, self._orig_dtype, self._fp8_config)

        raise NotImplementedError(
            f"FP8UnshardedWeightTensor does not support {func}. "
            f"Only FP8-aware module weights should be wrapped with FP8 all-gather."
        )

    def __tensor_flatten__(self):
        config = getattr(self, "_fp8_config", None)
        if config is None or not hasattr(self, "_fp8_data"):
            return [], {}
        return ["_fp8_data", "_scale_inv"], {
            "orig_dtype": self._orig_dtype,
            "format": config.format,
            "granularity": config.granularity,
            "strategy": config.strategy,
            "scale_dtype": config.scale_dtype,
            "block_size": config.block_size,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        config = Float8QuantConfig(
            format=metadata["format"],
            granularity=metadata["granularity"],
            strategy=metadata["strategy"],
            scale_dtype=metadata["scale_dtype"],
            block_size=metadata["block_size"],
        )
        return FP8UnshardedWeightTensor(
            inner_tensors["_fp8_data"],
            inner_tensors["_scale_inv"],
            metadata["orig_dtype"],
            config,
        )

    def __repr__(self):
        return (
            f"FP8UnshardedWeightTensor("
            f"shape={list(self._fp8_data.shape)}, "
            f"fp8_dtype={self._fp8_data.dtype}, "
            f"orig_dtype={self._orig_dtype}, "
            f"device={self._fp8_data.device})"
        )


def _wrap_fp8_weights_for_all_gather(
    module: nn.Module,
    fp8_config: Float8QuantConfig,
    stochastic_rounding: bool = False,
    deq_after_ag: bool = False,
) -> int:
    """Wrap FP8-eligible weight parameters with WeightWithFP8AllGatherTensor.

    Must be called BEFORE fully_shard(). FSDP2 natively handles tensor
    subclasses through fsdp_pre_all_gather/fsdp_post_all_gather.

    Args:
        stochastic_rounding: If True, use stochastic rounding during FP8
            quantization to make quantization noise unbiased.
        deq_after_ag: If True, dequantize FP8 back to BF16 after all-gather
            and return a plain tensor. Downstream dynamic quantization will
            produce a fresh scale from the actual assembled weight, avoiding
            the numerical issues of keeping weight in FP8 with a global scale.

    Returns the number of wrapped parameters.
    """
    wrapped_count = 0
    for child in module.modules():
        if not (hasattr(child, "_fp8_config") and hasattr(child, "weight")):
            continue
        w = child.weight
        if w is not None and w.dtype in (torch.bfloat16, torch.float32) and w.requires_grad:
            wrapper = WeightWithFP8AllGatherTensor(w.data, fp8_config)
            wrapper._stochastic_rounding = stochastic_rounding
            wrapper._deq_after_ag = deq_after_ag
            child.weight = nn.Parameter(wrapper, requires_grad=True)
            wrapped_count += 1
    return wrapped_count


@torch.no_grad()
def precompute_fp8_scales_for_fsdp(
    module: nn.Module,
    cache_data: bool = True,
    use_cpp_quantize: bool = False,
    stochastic_rounding: bool = False,
):
    """Precompute global FP8 scales for all FP8-all-gather parameters.

    Call after optimizer.step(), before next forward. Uses _foreach_norm
    for fused amax computation and DTensor Partial("max") -> Replicate()
    redistribution for a single batched all-reduce across all parameters.

    Args:
        module: The model module containing FP8-wrapped parameters.
        cache_data: If True, batch-quantize all weights and cache the FP8 data
            so fsdp_pre_all_gather skips per-layer quantization. If False, only
            precompute scales; quantization happens on-demand per layer.
        use_cpp_quantize: If True, use C++ quantize_fp8 from primus_turbo for
            on-demand quantization (matching run_35). Skips fp8_outputs buffer
            allocation and scale_inv computation when cache_data is False.
        stochastic_rounding: If True, use stochastic rounding during batch
            quantization to make quantization noise unbiased.
    """
    need_buffers = cache_data or not use_cpp_quantize
    need_scale_inv = cache_data or not use_cpp_quantize

    cache = getattr(module, "_fp8_scale_precompute_cache", None)
    if cache is None:
        fp8_dtensor_params = []
        fp8_plain_params = []
        inner_tensors = []
        for param in module.parameters():
            if isinstance(param, DTensor) and isinstance(param._local_tensor, WeightWithFP8AllGatherTensor):
                fp8_dtensor_params.append(param)
                inner_tensors.append(param._local_tensor._tensor)
            elif isinstance(param, WeightWithFP8AllGatherTensor):
                fp8_plain_params.append(param)
                inner_tensors.append(param._tensor)

        fp8_config = (
            fp8_dtensor_params[0]._local_tensor._fp8_config
            if fp8_dtensor_params
            else fp8_plain_params[0]._fp8_config if fp8_plain_params else None
        )
        fp8_dtype = _get_fp8_dtype(fp8_config.format) if fp8_config else None

        if need_buffers:
            fp8_outputs = [torch.empty_like(t, dtype=fp8_dtype) for t in inner_tensors]
        else:
            fp8_outputs = []
        all_local = [
            p._local_tensor if isinstance(p, DTensor) else p for p in fp8_dtensor_params + fp8_plain_params
        ]
        cache = (
            fp8_dtensor_params,
            fp8_plain_params,
            inner_tensors,
            fp8_outputs,
            all_local,
            fp8_config,
        )
        module._fp8_scale_precompute_cache = cache

    (
        fp8_dtensor_params,
        fp8_plain_params,
        inner_tensors,
        fp8_outputs,
        all_local,
        fp8_config,
    ) = cache
    if not inner_tensors:
        return

    local_amaxes = torch.stack(torch._foreach_norm(inner_tensors, ord=math.inf))

    if fp8_dtensor_params:
        mesh = fp8_dtensor_params[0].device_mesh
        partial_amaxes = DTensor.from_local(local_amaxes, device_mesh=mesh, placements=[Partial("max")])
        global_amaxes = partial_amaxes.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
    else:
        global_amaxes = local_amaxes

    fp8_max = torch.finfo(_get_fp8_dtype(fp8_config.format)).max
    global_amaxes = global_amaxes.to(torch.float64).clamp(min=1e-12)
    scales = (fp8_max / global_amaxes).to(torch.float32)
    scale_invs = (1.0 / scales) if need_scale_inv else None

    if cache_data:
        if stochastic_rounding:
            _foreach_fp8_quantize_stochastic(inner_tensors, fp8_outputs, scales, fp8_max)
        else:
            _foreach_fp8_quantize(inner_tensors, fp8_outputs, scales, fp8_max)

    for i, local_t in enumerate(all_local):
        local_t._precomputed_scale = scales[i]
        local_t._precomputed_scale_inv = scale_invs[i] if scale_invs is not None else None
        local_t._cached_fp8_data = fp8_outputs[i] if cache_data else None
        local_t._use_cpp_quantize = use_cpp_quantize


torch.serialization.add_safe_globals([WeightWithFP8AllGatherTensor])
