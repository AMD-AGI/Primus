# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FP8 all-gather tensor subclasses.

Tests the FP8 all-gather subclasses in isolation (single GPU, no
multi-GPU / NCCL required). Verifies:
- WeightWithFP8AllGatherTensor: creation, pre/post all-gather with
  precomputed scale, dispatch, flatten/unflatten
- FP8UnshardedWeightTensor: creation, dispatch propagation, detach
  dequantizes to BF16, unsupported ops raise NotImplementedError,
  flatten/unflatten roundtrip, get_fp8_data_and_scale_inv()
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
    FP8UnshardedWeightTensor,
    WeightWithFP8AllGatherTensor,
    _foreach_fp8_quantize,
    _get_fp8_dtype,
    quantize_fp8_prescaled,
)

_has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA required")


def _make_config():
    return Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)


def _make_wrapped(shape=(256, 512), device="cpu"):
    tensor = torch.randn(shape, dtype=torch.bfloat16, device=device)
    wrapped = WeightWithFP8AllGatherTensor(tensor, _make_config())
    return wrapped, tensor


def _set_precomputed_scale(wrapped):
    """Compute and set precomputed scale, scale_inv, and cached FP8 data for a single-rank scenario."""
    fp8_dtype = _get_fp8_dtype(wrapped._fp8_config.format)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = wrapped._tensor.abs().amax().float()
    scale = fp8_max / amax.clamp(min=1e-12)
    wrapped._precomputed_scale = scale
    wrapped._precomputed_scale_inv = 1.0 / scale
    fp8_data, _ = quantize_fp8_prescaled(
        wrapped._tensor,
        fp8_dtype,
        scale,
        wrapped._precomputed_scale_inv,
    )
    wrapped._cached_fp8_data = fp8_data


def _make_fp8_unsharded(shape=(256, 512), device="cpu"):
    """Create an FP8UnshardedWeightTensor from a BF16 source tensor."""
    config = _make_config()
    bf16_tensor = torch.randn(shape, dtype=torch.bfloat16, device=device)
    fp8_dtype = _get_fp8_dtype(config.format)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = bf16_tensor.abs().amax().float()
    scale = fp8_max / amax.clamp(min=1e-12)
    scale_inv = 1.0 / scale
    fp8_data = (bf16_tensor.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return (
        FP8UnshardedWeightTensor(fp8_data, scale_inv, torch.bfloat16, config),
        bf16_tensor,
        fp8_data,
        scale_inv,
    )


class TestSubclassCreation:
    def test_shape_dtype_device_preserved(self):
        wrapped, orig = _make_wrapped()
        assert wrapped.shape == orig.shape
        assert wrapped.dtype == torch.bfloat16
        assert wrapped.device == orig.device

    def test_isinstance_tensor(self):
        wrapped, _ = _make_wrapped()
        assert isinstance(wrapped, torch.Tensor)
        assert isinstance(wrapped, WeightWithFP8AllGatherTensor)

    def test_inner_tensor_accessible(self):
        wrapped, orig = _make_wrapped()
        assert wrapped._tensor is orig

    def test_tensorwise_only_validation(self):
        config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.ROWWISE)
        with pytest.raises(ValueError, match="TENSORWISE"):
            WeightWithFP8AllGatherTensor(torch.randn(4, 4, dtype=torch.bfloat16), config)

    def test_precomputed_scale_initially_none(self):
        wrapped, _ = _make_wrapped()
        assert wrapped._precomputed_scale is None


@requires_cuda
class TestPreAllGather:
    def test_returns_one_input(self):
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        all_gather_inputs, metadata = wrapped.fsdp_pre_all_gather(None)
        assert len(all_gather_inputs) == 1

    def test_fp8_data_dtype(self):
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        (fp8_data,), _ = wrapped.fsdp_pre_all_gather(None)
        assert fp8_data.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )

    def test_metadata_contains_scale_inv_and_numel(self):
        wrapped, orig = _make_wrapped((128, 64), device="cuda")
        _set_precomputed_scale(wrapped)
        _, metadata = wrapped.fsdp_pre_all_gather(None)
        scale_inv, shard_numel = metadata
        assert scale_inv.dtype == torch.float32
        assert shard_numel == 128 * 64

    def test_fp8_data_shape_matches_input(self):
        wrapped, orig = _make_wrapped((32, 64), device="cuda")
        _set_precomputed_scale(wrapped)
        (fp8_data,), _ = wrapped.fsdp_pre_all_gather(None)
        assert fp8_data.numel() == orig.numel()

    def test_asserts_without_precomputed_scale(self):
        wrapped, _ = _make_wrapped(device="cuda")
        with pytest.raises(RuntimeError, match="precompute_fp8_scales_for_fsdp"):
            wrapped.fsdp_pre_all_gather(None)


@requires_cuda
class TestPostAllGather:
    def _simulate_single_rank(self, wrapped):
        """Simulate all-gather with a single rank (data passes through)."""
        _set_precomputed_scale(wrapped)
        (fp8_data,), metadata = wrapped.fsdp_pre_all_gather(None)
        return (fp8_data,), metadata

    def test_first_call_returns_fp8_unsharded_tensor(self):
        wrapped, _ = _make_wrapped(device="cuda")
        outputs, metadata = self._simulate_single_rank(wrapped)
        result = wrapped.fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
        assert result is not None
        tensor, inner_tensors = result
        assert isinstance(tensor, FP8UnshardedWeightTensor)
        assert tensor.dtype == torch.bfloat16
        assert isinstance(inner_tensors, tuple)
        assert len(inner_tensors) == 1

    def test_steady_state_updates_scale(self):
        wrapped, _ = _make_wrapped(device="cuda")
        outputs, metadata = self._simulate_single_rank(wrapped)
        # First call to get an FP8UnshardedWeightTensor
        result = wrapped.fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
        fp8_unsharded, _ = result
        # Simulate FSDP's reshard path: pass the FP8UnshardedWeightTensor
        # directly as `out` (FSDP uses the unsharded_param, not nn.Parameter)
        new_scale_inv = metadata[0] * 2.0
        new_metadata = (new_scale_inv, metadata[1])
        result = wrapped.fsdp_post_all_gather(outputs, new_metadata, torch.bfloat16, out=fp8_unsharded)
        assert result is None
        assert torch.equal(fp8_unsharded._scale_inv, new_scale_inv)


@requires_cuda
class TestRoundtripAccuracy:
    def test_quantize_keeps_values(self):
        torch.manual_seed(42)
        wrapped, orig = _make_wrapped((128, 256), device="cuda")
        _set_precomputed_scale(wrapped)
        outputs, metadata = wrapped.fsdp_pre_all_gather(None)
        result = wrapped.fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
        fp8_unsharded, _ = result
        fp8_data, scale_inv = fp8_unsharded.get_fp8_data_and_scale_inv()
        dequantized = fp8_data.to(torch.bfloat16) * scale_inv
        dequantized = dequantized.view(orig.shape)
        torch.testing.assert_close(dequantized, orig, rtol=0.05, atol=0.1)


class TestTorchDispatch:
    def test_preserves_subclass_for_standard_ops(self):
        wrapped, _ = _make_wrapped((16, 16))
        for op in [
            lambda t: torch.empty_like(t),
            lambda t: t.view(-1),
            lambda t: t.clone(),
            lambda t: t[0:8],
        ]:
            result = op(wrapped)
            assert isinstance(
                result, WeightWithFP8AllGatherTensor
            ), f"Op should preserve subclass but got {type(result)}"

    def test_unwraps_for_compute_ops(self):
        wrapped, _ = _make_wrapped((16, 16))
        plain = torch.randn(16, 16, dtype=torch.bfloat16)
        result = wrapped + plain
        assert type(result) is torch.Tensor
        assert not isinstance(result, WeightWithFP8AllGatherTensor)

    def test_copy_inplace_modifies_inner(self):
        """copy_ modifies inner tensor in-place; target retains subclass type."""
        wrapped, _ = _make_wrapped((8, 8))
        target = torch.empty_like(wrapped)
        target.copy_(wrapped)
        assert isinstance(target, WeightWithFP8AllGatherTensor)
        torch.testing.assert_close(target._tensor, wrapped._tensor)

    def test_to_copy_dtype_change_unwraps(self):
        """`.float()` should return a plain tensor (for FP32 master copies)."""
        wrapped, _ = _make_wrapped((8, 8))
        fp32 = wrapped.float()
        assert fp32.dtype == torch.float32
        assert not isinstance(fp32, WeightWithFP8AllGatherTensor)

    def test_to_copy_same_dtype_preserves(self):
        wrapped, _ = _make_wrapped((8, 8))
        result = wrapped.to(torch.bfloat16)
        assert isinstance(result, WeightWithFP8AllGatherTensor)

    def test_detach_preserves_subclass(self):
        """detach preserves subclass so nn.Parameter() and FSDP2 work correctly."""
        wrapped, orig = _make_wrapped((8, 8))
        detached = wrapped.detach()
        assert isinstance(detached, WeightWithFP8AllGatherTensor)
        assert detached._tensor.data_ptr() == wrapped._tensor.data_ptr()
        assert detached._fp8_config.format == wrapped._fp8_config.format


class TestTensorFlattenUnflatten:
    def test_roundtrip(self):
        wrapped, orig = _make_wrapped((32, 64))
        inner_tensors_names, metadata = wrapped.__tensor_flatten__()
        assert inner_tensors_names == ["_tensor"]
        assert "format" in metadata
        assert "granularity" in metadata

        inner_tensors = {"_tensor": wrapped._tensor}
        restored = WeightWithFP8AllGatherTensor.__tensor_unflatten__(
            inner_tensors, metadata, wrapped.shape, wrapped.stride()
        )
        assert isinstance(restored, WeightWithFP8AllGatherTensor)
        assert restored._tensor is orig
        assert restored._fp8_config.format == Format.E4M3
        assert restored._fp8_config.granularity == ScalingGranularity.TENSORWISE

    def test_metadata_is_hashable(self):
        wrapped, _ = _make_wrapped()
        _, metadata = wrapped.__tensor_flatten__()
        for key, value in metadata.items():
            hash(value)

    def test_precomputed_scale_not_in_metadata(self):
        """_precomputed_scale is transient and must not appear in flatten metadata."""
        wrapped, _ = _make_wrapped()
        wrapped._precomputed_scale = torch.tensor(1.0)
        _, metadata = wrapped.__tensor_flatten__()
        assert "precomputed_scale" not in metadata
        assert "_precomputed_scale" not in metadata


# ============================================================================
# FP8UnshardedWeightTensor tests
# ============================================================================


class TestFP8UnshardedWeightTensorCreation:
    def test_shape_and_declared_dtype(self):
        fp8_unsharded, _, fp8_data, _ = _make_fp8_unsharded((64, 32))
        assert fp8_unsharded.shape == (64, 32)
        assert fp8_unsharded.dtype == torch.bfloat16

    def test_inner_fp8_data_accessible(self):
        fp8_unsharded, _, fp8_data, scale_inv = _make_fp8_unsharded()
        assert fp8_unsharded._fp8_data is fp8_data
        assert torch.equal(fp8_unsharded._scale_inv, scale_inv)

    def test_get_fp8_data_and_scale_inv(self):
        fp8_unsharded, _, fp8_data, scale_inv = _make_fp8_unsharded()
        data, sinv = fp8_unsharded.get_fp8_data_and_scale_inv()
        assert data is fp8_data
        assert torch.equal(sinv, scale_inv)


class TestFP8UnshardedDispatch:
    def test_as_strided_preserves_subclass(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((8, 8))
        result = torch.as_strided(fp8_unsharded, (64,), (1,), 0)
        assert isinstance(result, FP8UnshardedWeightTensor)
        assert result.shape == (64,)

    def test_view_preserves_subclass(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((8, 8))
        result = fp8_unsharded.view(-1)
        assert isinstance(result, FP8UnshardedWeightTensor)
        assert result.shape == (64,)

    def test_slice_preserves_subclass(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((16, 8))
        result = fp8_unsharded[0:8]
        assert isinstance(result, FP8UnshardedWeightTensor)
        assert result.shape == (8, 8)

    def test_clone_preserves_subclass(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((8, 8))
        result = fp8_unsharded.clone()
        assert isinstance(result, FP8UnshardedWeightTensor)

    def test_detach_preserves_subclass(self):
        """detach must preserve subclass (required by nn.Parameter wrapping)."""
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((8, 8))
        detached = fp8_unsharded.detach()
        assert isinstance(detached, FP8UnshardedWeightTensor)
        assert detached.dtype == torch.bfloat16

    def test_unsupported_op_raises(self):
        """Unsupported ops raise NotImplementedError (safety guard)."""
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((8, 8))
        with pytest.raises(NotImplementedError, match="does not support"):
            fp8_unsharded + torch.zeros(8, 8, dtype=torch.bfloat16)


class TestFP8UnshardedFlattenUnflatten:
    def test_roundtrip(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded((32, 64))
        inner_names, metadata = fp8_unsharded.__tensor_flatten__()
        assert set(inner_names) == {"_fp8_data", "_scale_inv"}
        assert "orig_dtype" in metadata
        assert "format" in metadata

        inner_tensors = {
            "_fp8_data": fp8_unsharded._fp8_data,
            "_scale_inv": fp8_unsharded._scale_inv,
        }
        restored = FP8UnshardedWeightTensor.__tensor_unflatten__(
            inner_tensors, metadata, fp8_unsharded.shape, fp8_unsharded.stride()
        )
        assert isinstance(restored, FP8UnshardedWeightTensor)
        assert restored._orig_dtype == torch.bfloat16
        # FP8 dtypes don't support torch.equal on CPU; compare via uint8 view
        assert torch.equal(
            restored._fp8_data.view(torch.uint8),
            fp8_unsharded._fp8_data.view(torch.uint8),
        )
        assert torch.equal(restored._scale_inv, fp8_unsharded._scale_inv)

    def test_metadata_is_hashable(self):
        fp8_unsharded, _, _, _ = _make_fp8_unsharded()
        _, metadata = fp8_unsharded.__tensor_flatten__()
        for key, value in metadata.items():
            hash(value)


class TestFP8DataCache:
    def test_cache_initially_none(self):
        wrapped, _ = _make_wrapped()
        assert wrapped._cached_fp8_data is None

    @requires_cuda
    def test_pre_all_gather_returns_cached_data(self):
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        assert wrapped._cached_fp8_data is not None
        (fp8_data,), _ = wrapped.fsdp_pre_all_gather(mesh=None)
        assert fp8_data.data_ptr() == wrapped._cached_fp8_data.data_ptr()

    @requires_cuda
    def test_pre_all_gather_fallback_without_cache(self):
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        wrapped._cached_fp8_data = None
        (fp8_data,), (scale_inv, _) = wrapped.fsdp_pre_all_gather(mesh=None)
        fp8_dtype = _get_fp8_dtype(wrapped._fp8_config.format)
        assert fp8_data.dtype == fp8_dtype
        assert fp8_data.shape == wrapped._tensor.shape

    @requires_cuda
    def test_cache_matches_direct_quantize(self):
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        cached = wrapped._cached_fp8_data
        fp8_dtype = _get_fp8_dtype(wrapped._fp8_config.format)
        fresh, _ = quantize_fp8_prescaled(
            wrapped._tensor,
            fp8_dtype,
            wrapped._precomputed_scale,
            wrapped._precomputed_scale_inv,
        )
        assert torch.equal(cached.view(torch.uint8), fresh.view(torch.uint8))

    @requires_cuda
    def test_cache_not_in_flatten_metadata(self):
        # _set_precomputed_scale runs the Triton FP8 quantize kernel which
        # requires a CUDA-resident tensor (no CPU dispatch). Match the other
        # cache tests in this class which already use device="cuda".
        wrapped, _ = _make_wrapped(device="cuda")
        _set_precomputed_scale(wrapped)
        inner_names, _ = wrapped.__tensor_flatten__()
        assert "_cached_fp8_data" not in inner_names


class TestForeachOptimizerOps:
    """Tests for _foreach_copy_ batched operations used by the optimizer."""

    @requires_cuda
    def test_foreach_copy_bf16_to_fp32_matches_float(self):
        """_foreach_copy_ BF16->FP32 matches per-tensor .float() (prepare_grads)."""
        sizes = [(128, 256), (64,), (512, 512), (32, 16)]
        bf16_tensors = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in sizes]
        ref_fp32 = [t.float() for t in bf16_tensors]

        fp32_targets = [torch.empty(s, dtype=torch.float32, device="cuda") for s in sizes]
        torch._foreach_copy_(fp32_targets, bf16_tensors)

        for i in range(len(sizes)):
            torch.testing.assert_close(
                fp32_targets[i], ref_fp32[i], rtol=0, atol=0, msg=f"Mismatch at tensor {i} shape {sizes[i]}"
            )

    @requires_cuda
    def test_foreach_copy_fp32_to_bf16_matches_copy(self):
        """_foreach_copy_ FP32->BF16 matches per-tensor .copy_() (copy-back)."""
        sizes = [(256, 128), (64,), (1024,)]
        fp32_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in sizes]
        ref_bf16 = [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in sizes]
        for r, f in zip(ref_bf16, fp32_tensors):
            r.copy_(f)

        bf16_targets = [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in sizes]
        torch._foreach_copy_(bf16_targets, fp32_tensors)

        for i in range(len(sizes)):
            torch.testing.assert_close(
                bf16_targets[i], ref_bf16[i], rtol=0, atol=0, msg=f"Mismatch at tensor {i} shape {sizes[i]}"
            )

    @requires_cuda
    def test_cache_extraction_fp8_wrapped(self):
        """to_local() + inner_data() produces the correct inner tensor for FP8-wrapped params."""
        wrapped, orig = _make_wrapped((128, 64), device="cuda")
        inner = wrapped.inner_data()
        assert inner is orig
        assert inner.dtype == torch.bfloat16
        assert inner.data_ptr() == orig.data_ptr()

    @requires_cuda
    def test_mixed_sizes_foreach_copy(self):
        """_foreach_copy_ works with mixed tensor sizes in the same list."""
        sizes = [(1,), (7,), (1024, 1024), (3, 5, 7), (131072,)]
        bf16_list = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in sizes]
        fp32_list = [torch.empty(s, dtype=torch.float32, device="cuda") for s in sizes]
        torch._foreach_copy_(fp32_list, bf16_list)
        for i, (b, f) in enumerate(zip(bf16_list, fp32_list)):
            torch.testing.assert_close(
                f, b.float(), rtol=0, atol=0, msg=f"Mixed-size mismatch at tensor {i} shape {sizes[i]}"
            )


class TestForeachFP8Quantize:
    """Tests for the compiled _foreach_fp8_quantize batch quantization helper."""

    @requires_cuda
    def test_matches_per_tensor_quantize(self):
        """Verify _foreach_fp8_quantize is bitwise identical to quantize_fp8_prescaled."""
        config = _make_config()
        fp8_dtype = _get_fp8_dtype(config.format)
        fp8_max = torch.finfo(fp8_dtype).max

        sizes = [1024, 4096, 131072, 524288]
        tensors = [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in sizes]
        amaxes = torch.stack([t.abs().amax().float() for t in tensors])
        scales = (fp8_max / amaxes.clamp(min=1e-12)).to(torch.float32)
        scale_invs = 1.0 / scales

        ref_results = []
        for i, t in enumerate(tensors):
            fp8_data, _ = quantize_fp8_prescaled(
                t,
                fp8_dtype,
                scales[i],
                scale_invs[i],
            )
            ref_results.append(fp8_data)

        fp8_outputs = [torch.empty_like(t, dtype=fp8_dtype) for t in tensors]
        _foreach_fp8_quantize.__wrapped__(tensors, fp8_outputs, scales, fp8_max)

        for i in range(len(tensors)):
            assert torch.equal(
                fp8_outputs[i].view(torch.uint8),
                ref_results[i].view(torch.uint8),
            ), f"Mismatch at tensor {i} (size {sizes[i]})"

    @requires_cuda
    def test_clamps_correctly(self):
        """Verify out-of-range values are clamped to fp8_max, not NaN."""
        config = _make_config()
        fp8_dtype = _get_fp8_dtype(config.format)
        fp8_max = torch.finfo(fp8_dtype).max

        tensor = torch.tensor([1.0, 1000.0, -1000.0, 0.001], device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(fp8_max, device="cuda", dtype=torch.float32)
        scales = scale.unsqueeze(0)

        fp8_out = [torch.empty_like(tensor, dtype=fp8_dtype)]
        _foreach_fp8_quantize.__wrapped__([tensor], fp8_out, scales, fp8_max)

        result = fp8_out[0]
        assert not result.float().isnan().any(), "NaN found in FP8 output"
        assert result.float().abs().max() <= fp8_max
