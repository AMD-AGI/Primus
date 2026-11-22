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

import math
import os
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.te_fused_adam_clip"
_EXTENSION_NAME = "primus_te_fused_adam_clip_v10"
_EXTENSION_NAME_V9 = "primus_te_fused_adam_clip_v9"
_extension = None


def _enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1"


def _wavedense_enabled() -> bool:
    """PRIMUS_FP8BLK_ADAM_WAVEDENSE=1: build the v10 kernel source (fused_adam_clip_kernel.cu): the FP8-block tile
    Adam reads and writes each tile row with whole-wave contiguous accesses and a compile-time BF16 store, -10% on
    the MegaMoE expert tiles, bit-identical outputs. 0 builds the previous v9 source (fused_adam_clip_kernel_v9.cu)."""
    return os.environ.get("PRIMUS_FP8BLK_ADAM_WAVEDENSE", "0") == "1"


def _bf16_writeback_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_BF16_WRITEBACK", "0") == "1"


def _bf16_grad_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_BF16_GRAD", "0") == "1"


def _fp8blk_enabled() -> bool:
    """PRIMUS_FP8BLK_ADAM_QUANT=1 (needs PRIMUS_FUSED_ADAM_BF16_WRITEBACK=1 and the Turbo 128-block
    UE8M0 recipe): every weight a Turbo block-FP8 GEMM quantizes is updated by a 128x128-tile Adam
    kernel that also writes its FP8 W, W^T and UE8M0 scales into Turbo's per-step weight cache
    (primus_turbo.pytorch.core.fp8blk_weight_cache), so the GEMMs skip their weight quant."""
    return os.environ.get("PRIMUS_FP8BLK_ADAM_QUANT", "0") == "1"


def _fp8blk_megamoe_enabled() -> bool:
    """PRIMUS_FP8BLK_ADAM_MEGAMOE=1 (with PRIMUS_FP8BLK_ADAM_QUANT=1): the MegaMoE fp8_block expert
    weights go through the same tile Adam kernel, which writes Turbo's FP8BlockWeightCache (q / s and
    qT / sT MX planes, bytes of quant_w_128x128) so refresh_fp8blk_weights() skips them."""
    return _fp8blk_enabled() and os.environ.get("PRIMUS_FP8BLK_ADAM_MEGAMOE", "0") == "1"


def _fp8blk_skip_bf16_enabled() -> bool:
    """PRIMUS_FP8BLK_ADAM_SKIP_BF16=1 (with PRIMUS_FP8BLK_ADAM_MEGAMOE=1): the tile Adam does not store the BF16
    MegaMoE expert weights (2 of its 22 B/param). Their GEMMs read only the Adam-written FP8 cache, so the BF16
    copy is rebuilt from the FP32 master (same RNE cast, same bytes) only before a BF16 reader: a cache refresh
    (Turbo materialize hook), save_checkpoint, the params-norm log and the DP param-hash check. Applies only where
    the expert weight's param buffer has a data-parallel group of size 1 (else the param all-gather ships it)."""
    return _fp8blk_megamoe_enabled() and os.environ.get("PRIMUS_FP8BLK_ADAM_SKIP_BF16", "0") == "1"


class _BF16Materializer:
    """Rebuilds a skipped BF16 weight (the param-buffer view) from its FP32 master shard, on the current stream,
    once per optimizer update (``stale`` is set on the host each time the Adam kernel is queued for it)."""

    _ALL = []

    def __init__(self, model_buf, master):
        self.model_buf, self.master, self.stale = model_buf, master, False
        _BF16Materializer._ALL.append(self)

    @torch.no_grad()
    def __call__(self):
        if self.stale:
            self.model_buf.copy_(self.master)  # c10 RNE, the kernel's pack_bf16x8 bytes
            self.stale = False


def materialize_skipped_bf16_weights() -> int:
    """Bring every BF16 weight the tile Adam skipped (PRIMUS_FP8BLK_ADAM_SKIP_BF16) up to date. Returns the count
    rebuilt. Call before anything outside the FP8 GEMMs reads those weights."""
    n = 0
    for m in _BF16Materializer._ALL:
        n += int(m.stale)
        m()
    return n


class _MegaMoEAdamTarget:
    """A Turbo MegaMoE ``FP8BlockWeightCache`` seen as a tile-Adam output: W = q, W^T = qT, MX scale
    planes s / sT in the e8m0 / e8m0_t slots and no scale_inv (address 0 = MX-plane mode)."""

    scale_inv = None

    def __init__(self, cache):
        self.cache = cache
        self.w, self.wt, self.e8m0, self.e8m0_t = cache.q, cache.qT, cache.s, cache.sT
        self.ready = False  # not read by MegaMoE (its cache stays valid; the forward reads it unconditionally)
        self.materializer = None  # set when the BF16 weight store is skipped (PRIMUS_FP8BLK_ADAM_SKIP_BF16)

    @staticmethod
    def find(model_param):
        """The live MegaMoE cache whose weight is ``model_param`` (None if there is none)."""
        try:
            from primus_turbo.pytorch.ops.moe import fused_mega_moe_fp8blk as mm
        except ImportError:
            return None
        if getattr(mm, "_GG_DSV4", False):  # gg_dsv4 also needs the packed scales from refresh()
            return None
        for c in list(mm._CACHES):
            if c.weight is model_param:
                return c
        return None


def _static_bf16_loss_scale_cpu_enabled() -> bool:
    return os.environ.get("PRIMUS_STATIC_BF16_LOSS_SCALE_CPU", "0") == "1"


def _uses_static_bf16_unity_scale(optimizer) -> bool:
    """Return whether every concrete optimizer uses BF16 without a grad scaler."""

    children = getattr(optimizer, "chained_optimizers", None)
    optimizers = children if children is not None else [optimizer]
    return bool(optimizers) and all(
        getattr(child.config, "bf16", False) and getattr(child, "grad_scaler", None) is None
        for child in optimizers
    )


def _return_grad_norm_enabled() -> bool:
    """Return the device grad norm (0-dim) for logging; it is read after Adam has been queued."""
    return os.environ.get("PRIMUS_FUSED_ADAM_RETURN_GRAD_NORM", "1") == "1"


def _logged_grad_norm(grad_norm: torch.Tensor):
    return grad_norm.reshape(()) if _return_grad_norm_enabled() else None


def _mcore_master_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_MCORE_MASTER", "0") == "1"


def _clip_coeff(total_norm: float, max_norm: float) -> float:
    """Return Megatron's gradient clipping coefficient."""

    return min(1.0, float(max_norm) / (float(total_norm) + 1.0e-6))


def _grads_for_grad_norm(optimizer):
    """Gradients of the main norm. Megatron-core 0.19 renamed ``get_main_grads_for_grad_norm``."""

    getter = getattr(optimizer, "get_grads_for_grad_norm", None)
    if getter is None:
        getter = optimizer.get_main_grads_for_grad_norm
    return getter()


def _separate_grad_norm_groups() -> tuple:
    """Megatron-core 0.19's separately clipped grad-norm groups (MTP); none before it."""

    from megatron.core.optimizer import optimizer as mcore_optimizer

    return tuple(getattr(mcore_optimizer, "SEPARATE_GRAD_NORM_GROUPS", ()))


def _norm_from_nan_check(optimizer):
    """g90mem: global grad norm from the DDP NaN-check sum of squares (PRIMUS_GRAD_NORM_FROM_NAN_CHECK=1).

    The deferred NaN check already reduced every grad bucket to a sum of squares in the backward; at
    gradient-statistics group size 1 its sqrt is the clip norm, so the 17 GB re-read by multi_tensor_l2norm
    (1.43 ms/step on the 1-GPU proxy) is skipped. The first use is cross-checked against the full norm
    (one host sync); on a mismatch the reuse stays off for the run. Returns None to fall back."""
    if os.environ.get("PRIMUS_GRAD_NORM_FROM_NAN_CHECK", "0") != "1":
        return None
    try:
        from primus.backends.megatron.patches.ddp_deferred_grad_check_patches import pop_last_grad_sumsq
    except Exception:
        return None
    acc = pop_last_grad_sumsq()
    if acc is None:
        return None
    grp = optimizer.get_grad_stats_parallel_group()
    if grp is not None and torch.distributed.is_initialized() and torch.distributed.get_world_size(grp) != 1:
        return None
    st = optimizer.__dict__.setdefault("_primus_norm_reuse", {"checked": False, "ok": False})
    if not st["checked"]:
        ref = _get_grad_norm_tensor_full(optimizer)
        got = acc.reshape(1).float().sqrt()
        rel = ((got - ref).abs() / ref.clamp_min(1e-30)).item()
        st["checked"], st["ok"] = True, rel < 1e-3
        log_rank_0(f"[g90mem] grad norm from NaN-check sumsq: rel diff {rel:.2e} vs l2norm -> reuse={st['ok']}")
        return ref
    return acc.reshape(1).float().sqrt() if st["ok"] else None


def _get_grad_norm_tensor(optimizer) -> torch.Tensor:
    reused = _norm_from_nan_check(optimizer)
    if reused is not None:
        return reused
    return _get_grad_norm_tensor_full(optimizer)


def _get_grad_norm_tensor_full(optimizer) -> torch.Tensor:
    """Megatron's FP32 L2 norm path without its final host synchronization."""

    from megatron.core.optimizer import clip_grads

    grads_for_norm = _grads_for_grad_norm(optimizer)
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

    wavedense = _wavedense_enabled()
    source = Path(__file__).with_name("fused_adam_clip_kernel.cu" if wavedense else "fused_adam_clip_kernel_v9.cu")
    # No -std flag: torch adds the one its headers need, and torch 2.14's need C++20.
    _extension = load(
        name=_EXTENSION_NAME if wavedense else _EXTENSION_NAME_V9,
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    return _extension


def _metadata_key(chunk_size: int, tensor_lists, writeback_by_param_ptr) -> tuple[Any, ...]:
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
                (
                    writeback_by_param_ptr.get(tensor.data_ptr()).data_ptr()
                    if tensor.data_ptr() in writeback_by_param_ptr
                    else 0
                ),
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
        raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP received inconsistent empty tensor lists")

    device = tensor_lists[0][0].device
    moment_dtype = tensor_lists[2][0].dtype
    grad_dtype = tensor_lists[0][0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"PRIMUS_FUSED_ADAM_CLIP requires FP32 or BF16 gradients, got {grad_dtype}")
    if moment_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP requires FP32 or BF16 Adam moments, got " f"{moment_dtype}"
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
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires one device per Adam call")
        expected_dtypes = (grad_dtype, torch.float32, moment_dtype, moment_dtype)
        if any(tensor.dtype != expected for tensor, expected in zip(tensors, expected_dtypes)):
            dtypes = tuple(tensor.dtype for tensor in tensors)
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires uniform FP32/BF16 gradients, FP32 "
                "parameters, and matching FP32 or BF16 moments; tensor "
                f"{tensor_idx} has dtypes {dtypes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires contiguous optimizer tensors")
        if any(tensor.numel() != tensors[0].numel() for tensor in tensors[1:]):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP received mismatched tensor sizes")

        writeback = writeback_by_param_ptr.get(tensors[1].data_ptr())
        if writeback is not None:
            if writeback.device != device or writeback.dtype != torch.bfloat16:
                raise RuntimeError(
                    "PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires a BF16 destination " "on the optimizer device"
                )
            if not writeback.is_contiguous() or writeback.numel() != tensors[1].numel():
                raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK received an incompatible destination")
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
        0 if grad_dtype == torch.float32 else 1,
    )


_FP8BLK_TILE = 128


def _fp8blk_metadata_key(tensor_lists, fp8blk_by_param_ptr) -> tuple[Any, ...]:
    return tuple(
        (tensor.data_ptr(), tensor.numel(), tensor.dtype)
        for tensor_list in tensor_lists
        for tensor in tensor_list
    ) + tuple(id(fp8blk_by_param_ptr[p.data_ptr()][1]) for p in tensor_lists[1])


def _build_fp8blk_metadata(tensor_lists, fp8blk_by_param_ptr, writeback_by_param_ptr):
    """Per tensor: 10 addresses (grad, fp32 master, exp_avg, exp_avg_sq, bf16 param, W, W^T,
    scale_inv, e8m0, e8m0_t) and dims (N, K, tiles_n, tiles_k); one 128x128 tile per block.
    A cache whose ``wt`` / ``e8m0_t`` are None (a weight whose GEMMs never read W^T) gets address
    0 there and the kernel skips its transpose."""
    device = tensor_lists[0][0].device
    grad_dtype = tensor_lists[0][0].dtype
    moment_dtype = tensor_lists[2][0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16) or moment_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"PRIMUS_FP8BLK_ADAM_QUANT: unsupported grad/moment dtypes {grad_dtype}/{moment_dtype}")
    addresses, dims, tile_to_tensor, tile_offsets = [], [], [], []
    total_tiles = 0
    for idx in range(len(tensor_lists[0])):
        grad, master, exp_avg, exp_avg_sq = (tensor_lists[j][idx] for j in range(4))
        model_param, cache = fp8blk_by_param_ptr[master.data_ptr()]
        model_buf = writeback_by_param_ptr.get(master.data_ptr())
        if model_buf is None:
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT requires the BF16 writeback destination of every weight")
        if (grad.dtype, master.dtype, exp_avg.dtype, exp_avg_sq.dtype) != (
            grad_dtype, torch.float32, moment_dtype, moment_dtype
        ):
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT received mixed optimizer dtypes in one call")
        wt, e8m0_t = getattr(cache, "wt", None), getattr(cache, "e8m0_t", None)
        if (wt is None) != (e8m0_t is None):
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT: W^T and its E8M0 bytes must both be cached or both skipped")
        out = (cache.w, wt, cache.scale_inv, cache.e8m0, e8m0_t)
        present = tuple(t for t in out if t is not None)
        dense = (grad, master, exp_avg, exp_avg_sq, model_buf)
        # BF16 address 0: the kernel skips the BF16 weight store (only MegaMoE targets with a materializer).
        skip_bf16 = getattr(cache, "materializer", None) is not None
        if any(t.numel() != model_param.numel() for t in dense) or any(not t.is_contiguous() for t in dense + present):
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT: optimizer shard does not cover the whole contiguous weight")
        if any(t.data_ptr() % 16 for t in dense + tuple(t for t in out[:2] if t is not None)):
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT requires 16-byte aligned optimizer and cache tensors")
        n, k = model_param.shape[-2:]
        batch = model_param.numel() // (n * k)
        tiles_n = (n + _FP8BLK_TILE - 1) // _FP8BLK_TILE
        tiles_k = (k + _FP8BLK_TILE - 1) // _FP8BLK_TILE
        addresses.extend(0 if t is None or (skip_bf16 and t is model_buf) else t.data_ptr() for t in dense + out)
        dims.extend((n, k, tiles_n, tiles_k))
        tiles = batch * tiles_n * tiles_k
        tile_offsets.append(total_tiles)
        tile_to_tensor.extend([idx] * tiles)
        total_tiles += tiles
    return (
        torch.tensor(addresses, dtype=torch.int64, device=device),
        torch.tensor(dims, dtype=torch.int64, device=device),
        torch.tensor(tile_to_tensor, dtype=torch.int32, device=device),
        torch.tensor(tile_offsets, dtype=torch.int64, device=device),
        total_tiles,
        0 if moment_dtype == torch.float32 else 1,
        0 if grad_dtype == torch.float32 else 1,
        [cache for _, cache in (fp8blk_by_param_ptr[p.data_ptr()] for p in tensor_lists[1])],
    )


def _copy_model_grads_to_decoupled(distributed_optimizer) -> None:
    """Attach DDP gradient-buffer shards directly to FP32 master parameters."""

    if distributed_optimizer.is_stub_optimizer:
        return
    if distributed_optimizer.ddp_config.use_megatron_fsdp:
        raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_GRAD does not support Megatron FSDP")

    def attach_group_grads(model_groups, shard_main_groups):
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                param_range = distributed_optimizer._get_model_param_range_map(model_param)["param"]
                if param_range.size != shard_main_param.nelement():
                    raise RuntimeError("Gradient shard range does not match its FP32 master")
                model_grad = model_param.main_grad
                shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                if shard_model_grad.dtype not in (torch.bfloat16, torch.float32):
                    raise RuntimeError(
                        "PRIMUS_FUSED_ADAM_BF16_GRAD requires BF16 or FP32 DDP gradients, "
                        f"got {shard_model_grad.dtype}"
                    )
                shard_main_param.decoupled_grad = shard_model_grad

    attach_group_grads(
        distributed_optimizer.model_float16_groups,
        distributed_optimizer.shard_fp32_from_float16_groups,
    )
    attach_group_grads(
        distributed_optimizer.model_fp32_groups,
        distributed_optimizer.shard_fp32_groups,
    )


def _get_decoupled_grads_for_grad_norm(optimizer):
    """Megatron's grad-norm filtering applied to decoupled gradient views."""

    from megatron.core import tensor_parallel
    from megatron.core.transformer.module import param_is_not_shared

    grads_for_norm = []
    for param in optimizer.get_parameters():
        grad = getattr(param, "decoupled_grad", None)
        if (
            grad is not None
            and param_is_not_shared(param)
            and tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param, getattr(optimizer, "tp_group", None)
            )
        ):
            grads_for_norm.append(grad)
    return grads_for_norm


def _prepare_bf16_writeback(distributed_optimizer, fused_adam_type) -> None:
    """Map each FP32 master shard to its BF16 parameter-buffer destination."""

    if not _bf16_writeback_enabled():
        return
    if getattr(distributed_optimizer, "is_stub_optimizer", False):
        return
    if distributed_optimizer.ddp_config.use_megatron_fsdp:
        raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK does not support Megatron FSDP")
    if distributed_optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
        raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires separate FP32 master parameters")

    inner_optimizer = distributed_optimizer.optimizer
    if not isinstance(inner_optimizer, fused_adam_type):
        raise TypeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires Transformer Engine FusedAdam")

    cached = getattr(distributed_optimizer, "_primus_bf16_writeback_cache", None)
    if cached is not None:
        writeback_by_param_ptr, writeback_tensors, fp8blk_by_param_ptr = cached
        inner_optimizer._primus_bf16_writeback_by_param_ptr = writeback_by_param_ptr
        inner_optimizer._primus_bf16_writeback_tensors = writeback_tensors
        inner_optimizer._primus_fp8blk_by_param_ptr = fp8blk_by_param_ptr
        distributed_optimizer._primus_bf16_writeback_active = True
        return

    writeback_by_param_ptr = {}
    writeback_tensors = []
    fp8blk_by_param_ptr = {}
    megamoe_count = 0
    skip_bf16_count = 0
    if _fp8blk_enabled():
        from primus_turbo.pytorch.core import fp8blk_weight_cache
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
            param_range_map = distributed_optimizer._get_model_param_range_map(model_param)
            world_range = param_range_map["gbuf_world_in_bucket"]
            if world_range.size != shard_main_param.nelement():
                raise RuntimeError("BF16 writeback shard range does not match its FP32 master")
            gbuf_index, _, bucket_id = distributed_optimizer.model_param_gbuf_map[model_param]
            param_buffer = distributed_optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
            shard_model_param = param_buffer.view(-1)[world_range.start : world_range.end]
            if shard_model_param.dtype != torch.bfloat16:
                raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK requires a BF16 parameter buffer")
            writeback_by_param_ptr[shard_main_param.data_ptr()] = shard_model_param
            writeback_tensors.append(shard_model_param)
            if (
                _fp8blk_enabled()
                and fp8blk_weight_cache.wants_cache(model_param)
                and model_param.dim() in (2, 3)
                and world_range.size == model_param.numel()
                and shard_model_param.data_ptr() == model_param.data_ptr()
            ):
                # The whole weight lives in this rank's master shard (DP=1 or an unsplit
                # parameter), so 128x128 tiles can be formed inside the optimizer.
                cache = fp8blk_weight_cache.attach(model_param)
                cache.ready = False
                fp8blk_by_param_ptr[shard_main_param.data_ptr()] = (model_param, cache)
            elif (
                _fp8blk_megamoe_enabled()
                and model_param.dim() == 3
                and model_param.shape[-2] % _FP8BLK_TILE == 0
                and model_param.shape[-1] % _FP8BLK_TILE == 0
                and world_range.size == model_param.numel()
                and shard_model_param.data_ptr() == model_param.data_ptr()
            ):
                mm_cache = _MegaMoEAdamTarget.find(model_param)
                if mm_cache is not None:
                    # The tile kernel now owns this cache; refresh_fp8blk_weights() skips it.
                    mm_cache.adam_managed = True
                    target = _MegaMoEAdamTarget(mm_cache)
                    dp_group = getattr(distributed_optimizer.buffers[gbuf_index], "data_parallel_group", None)
                    if (
                        _fp8blk_skip_bf16_enabled()
                        and dp_group is not None
                        and dp_group.size() == 1
                        and getattr(type(mm_cache), "MATERIALIZE_HOOK", None)  # Turbo calls it before reading
                    ):
                        target.materializer = _BF16Materializer(shard_model_param, shard_main_param)
                        setattr(model_param, type(mm_cache).MATERIALIZE_HOOK, target.materializer)
                        skip_bf16_count += 1
                    fp8blk_by_param_ptr[shard_main_param.data_ptr()] = (model_param, target)
                    megamoe_count += 1

    expected_masters = sum(len(group) for group in distributed_optimizer.shard_fp32_from_float16_groups)
    if len(writeback_by_param_ptr) != expected_masters:
        raise RuntimeError("PRIMUS_FUSED_ADAM_BF16_WRITEBACK did not map every FP32 master")

    # Retain the views until the asynchronous Adam launch has consumed them.
    distributed_optimizer._primus_bf16_writeback_cache = (
        writeback_by_param_ptr,
        writeback_tensors,
        fp8blk_by_param_ptr,
    )
    inner_optimizer._primus_bf16_writeback_by_param_ptr = writeback_by_param_ptr
    inner_optimizer._primus_bf16_writeback_tensors = writeback_tensors
    inner_optimizer._primus_fp8blk_by_param_ptr = fp8blk_by_param_ptr
    distributed_optimizer._primus_bf16_writeback_active = True
    if _fp8blk_enabled():
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] PRIMUS_FP8BLK_ADAM_QUANT: {len(fp8blk_by_param_ptr)} weights "
            f"({sum(p.numel() for p, _ in fp8blk_by_param_ptr.values()) / 1e6:.1f} M params) get FP8 "
            "W / W^T / UE8M0 scales from the Adam writeback"
            + (f", {megamoe_count} of them MegaMoE expert caches (PRIMUS_FP8BLK_ADAM_MEGAMOE)" if megamoe_count else "")
            + (
                f"; {skip_bf16_count} skip the BF16 weight store (PRIMUS_FP8BLK_ADAM_SKIP_BF16, rebuilt from the "
                "FP32 master before any BF16 reader)"
                if _fp8blk_skip_bf16_enabled()
                else ""
            )
        )


def _copy_only_fp32_model_groups(distributed_optimizer) -> None:
    """Finish the copy pass for native FP32 model parameters only."""

    for shard_main_group, model_group in zip(
        distributed_optimizer.shard_fp32_groups,
        distributed_optimizer.model_fp32_groups,
    ):
        for shard_main_param, model_param in zip(shard_main_group, model_group):
            param_range_map = distributed_optimizer._get_model_param_range_map(model_param)
            world_range = param_range_map["gbuf_world_in_bucket"]
            if world_range.size != shard_main_param.nelement():
                raise RuntimeError("FP32 model shard range does not match its main parameter")
            gbuf_index, _, bucket_id = distributed_optimizer.model_param_gbuf_map[model_param]
            param_buffer = distributed_optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
            shard_model_param = param_buffer.view(-1)[world_range.start : world_range.end]
            shard_model_param.copy_(shard_main_param)


def _chained_step_with_device_clip(optimizer, fused_adam_type):
    """Run ChainedOptimizer.step while deferring clipping to each fused Adam."""

    optimizer.grad_norms_by_group = {}
    found_inf_flag = optimizer.prepare_grads()
    if found_inf_flag:
        return False, None, None
    if not optimizer.grads_states_parallel_group_is_shared():
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP requires chained optimizers to share "
            "their gradient-statistics process group"
        )
    has_grad_norm_group = getattr(optimizer, "has_grad_norm_group", None)
    if has_grad_norm_group is not None and any(
        has_grad_norm_group(group) for group in _separate_grad_norm_groups()
    ):
        raise RuntimeError(
            "PRIMUS_FUSED_ADAM_CLIP clips with one global norm, but Megatron clips "
            "a separate grad-norm group (MTP) with its own"
        )

    grad_norm = _get_grad_norm_tensor(optimizer)
    for child_optimizer in optimizer.chained_optimizers:
        if getattr(child_optimizer, "is_stub_optimizer", False):
            continue
        if not child_optimizer.get_parameters():
            continue
        if child_optimizer.config.clip_grad <= 0.0:
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP requires a positive clip_grad")
        if math.isfinite(getattr(child_optimizer.config, "grad_norm_skip_threshold", math.inf)):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP keeps the grad norm on the device and cannot "
                "apply grad_norm_skip_threshold"
            )
        inner_optimizer = child_optimizer.optimizer
        if not isinstance(inner_optimizer, fused_adam_type):
            raise TypeError(
                "PRIMUS_FUSED_ADAM_CLIP requires Transformer Engine FusedAdam, got "
                f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}"
            )
        inner_optimizer._primus_clip_norm = grad_norm
        inner_optimizer._primus_clip_max_norm = float(child_optimizer.config.clip_grad)
        _prepare_bf16_writeback(child_optimizer, fused_adam_type)

    num_zeros_in_grad = optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else None
    update_successful = optimizer.step_with_ready_grads()
    # The device norm is returned only for logging (PRIMUS_FUSED_ADAM_RETURN_GRAD_NORM,
    # default on): Megatron formats it after the step, when the Adam launch is
    # already queued, so the host read does not stall the optimizer. Set it to 0
    # to return None and skip that read entirely.
    return update_successful, _logged_grad_norm(grad_norm), num_zeros_in_grad


def _deferred_passthrough_reduce_max(value, *args, **kwargs):
    """TP1/PP1 model-parallel max, kept on the device. Megatron-core 0.19 passes ``group=``."""

    return value


_BF16_READERS = (
    ("megatron.training.training", "save_checkpoint"),
    ("megatron.training.checkpointing", "save_checkpoint"),
    ("megatron.training.training", "calc_params_l2_norm"),
    ("megatron.training.utils", "calc_params_l2_norm"),
    ("megatron.training.training", "check_param_hashes_across_dp_replicas"),
    ("megatron.training.utils", "check_param_hashes_across_dp_replicas"),
)


def _wrap_bf16_readers() -> None:
    """PRIMUS_FP8BLK_ADAM_SKIP_BF16: rebuild the skipped BF16 weights before the Megatron functions that read model
    parameters outside the FP8 GEMMs (checkpoint save, params-norm log, DP param-hash check)."""
    import importlib

    for mod_name, attr in _BF16_READERS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is None or getattr(fn, "_primus_materializes_bf16", False):
            continue

        def make(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                materialize_skipped_bf16_weights()
                return fn(*args, **kwargs)

            wrapped._primus_materializes_bf16 = True
            return wrapped

        setattr(mod, attr, make(fn))


def _install_patch() -> None:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from transformer_engine.pytorch.optimizers import FusedAdam

    if is_patched(FusedAdam, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already installed; skipping.")
        return

    if _fp8blk_enabled():
        if not _bf16_writeback_enabled():
            raise RuntimeError("PRIMUS_FP8BLK_ADAM_QUANT requires PRIMUS_FUSED_ADAM_BF16_WRITEBACK=1")
        if os.environ.get("PRIMUS_TURBO_FP8_BLOCK", "0").strip().lower() in ("1", "true"):
            # Same default the Turbo precision setup applies later (at the first forward).
            os.environ.setdefault("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
        if os.environ.get("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "0") != "1":
            raise RuntimeError(
                "PRIMUS_FP8BLK_ADAM_QUANT emits UE8M0 scales: needs PRIMUS_TURBO_FP8_BLOCK=1 / PRIMUS_TURBO_FP8_BLOCK_UE8M0=1"
            )
        # Turbo block-FP8 GEMMs tag the weights they quantize and use the optimizer-written cache.
        os.environ["PRIMUS_TURBO_FP8_WCACHE"] = "1"
    if _fp8blk_skip_bf16_enabled():
        _wrap_bf16_readers()

    extension = _load_extension()
    original_adam_init = FusedAdam.__init__
    original_adam_step = FusedAdam.step
    original_scale_loss = MegatronOptimizer.scale_loss
    original_chained_get_loss_scale = ChainedOptimizer.get_loss_scale
    grads_for_norm_name = (
        "get_grads_for_grad_norm"
        if hasattr(MegatronOptimizer, "get_grads_for_grad_norm")
        else "get_main_grads_for_grad_norm"
    )
    original_get_main_grads = getattr(MegatronOptimizer, grads_for_norm_name)
    original_copy_model_grads = DistributedOptimizer._copy_model_grads_to_main_grads
    original_copy_main_params = DistributedOptimizer._copy_main_params_to_model_params
    original_collect_grads = DistributedOptimizer._collect_main_grad_data_for_unscaling
    original_zero_grad = DistributedOptimizer.zero_grad
    original_optimizer_config_post_init = OptimizerConfig.__post_init__

    @wraps(original_optimizer_config_post_init)
    def patched_optimizer_config_post_init(self):
        original_optimizer_config_post_init(self)
        if not _mcore_master_enabled():
            return
        if not self.use_precision_aware_optimizer:
            raise RuntimeError("PRIMUS_FUSED_ADAM_MCORE_MASTER requires use_precision_aware_optimizer")
        if self.main_params_dtype != torch.float32:
            raise RuntimeError("PRIMUS_FUSED_ADAM_MCORE_MASTER requires FP32 main_params_dtype")
        # MXFP4 is provided by Primus-Turbo rather than Megatron's fp8_recipe,
        # so upstream classifies it as the generic no-FP8 path and moves master
        # ownership into TE. Keep MCore's FP32 shards and use precision-aware
        # mode only to select BF16 moments.
        self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False

    @wraps(original_adam_init)
    def patched_adam_init(self, *args, **kwargs):
        if kwargs.get("capturable", False):
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP preserves TE's optimized non-capturable ROCm path")
        if _bf16_grad_enabled():
            kwargs["use_decoupled_grad"] = True
        original_adam_init(self, *args, **kwargs)
        self._primus_clip_norm = None
        self._primus_clip_max_norm = 1.0
        self._primus_adam_metadata_cache = {}
        self._primus_bf16_writeback_by_param_ptr = {}
        self._primus_bf16_writeback_tensors = []
        self._primus_fp8blk_by_param_ptr = {}
        self._primus_fp8blk_metadata_cache = {}

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
            if self._primus_clip_norm is None:
                raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP did not receive a device gradient norm")
            fp8blk_by_param_ptr = self._primus_fp8blk_by_param_ptr
            if fp8blk_by_param_ptr:
                sel = [i for i, p in enumerate(tensor_lists[1]) if p.data_ptr() in fp8blk_by_param_ptr]
                if sel:
                    sel_lists = [[lst[i] for i in sel] for lst in tensor_lists]
                    fkey = _fp8blk_metadata_key(sel_lists, fp8blk_by_param_ptr)
                    fmeta = self._primus_fp8blk_metadata_cache.get(fkey)
                    if fmeta is None:
                        fmeta = _build_fp8blk_metadata(sel_lists, fp8blk_by_param_ptr, writeback_by_param_ptr)
                        self._primus_fp8blk_metadata_cache[fkey] = fmeta
                    f_addr, f_dims, f_t2t, f_toff, f_tiles, f_mdt, f_gdt, f_caches = fmeta
                    extension.fused_adam_clip_fp8blk(
                        f_addr,
                        f_dims,
                        f_t2t,
                        f_toff,
                        f_tiles,
                        float(lr),
                        float(beta1),
                        float(beta2),
                        float(epsilon),
                        int(step),
                        int(mode),
                        int(bias_correction),
                        float(weight_decay),
                        int(f_mdt),
                        int(f_gdt),
                        self._primus_clip_norm,
                        float(self._primus_clip_max_norm),
                    )
                    # Queued on the current stream: later GEMMs on it read the new FP8 weights.
                    for cache in f_caches:
                        cache.ready = True
                        materializer = getattr(cache, "materializer", None)
                        if materializer is not None:
                            materializer.stale = True  # the BF16 weight now lags the FP32 master
                    sel_set = set(sel)
                    tensor_lists = [
                        [t for i, t in enumerate(lst) if i not in sel_set] for lst in tensor_lists
                    ]
                    if not tensor_lists[0]:
                        return
            key = _metadata_key(chunk_size, tensor_lists, writeback_by_param_ptr)
            metadata = self._primus_adam_metadata_cache.get(key)
            if metadata is None:
                metadata = _build_metadata(chunk_size, tensor_lists, writeback_by_param_ptr)
                self._primus_adam_metadata_cache[key] = metadata
            (
                addresses,
                sizes,
                block_to_tensor,
                chunk_offsets,
                total_chunks,
                moment_dtype,
                grad_dtype,
            ) = metadata
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
                int(grad_dtype),
                self._primus_clip_norm,
                float(self._primus_clip_max_norm),
            )

        # FusedAdam.step calls this attribute through TE's existing
        # multi_tensor_applier, so all Python-side grouping remains unchanged.
        self.multi_tensor_adam = fused_multi_tensor_adam

    @wraps(original_adam_step)
    def patched_adam_step(self, closure=None, grad_scaler=None):
        if grad_scaler is not None:
            raise RuntimeError("PRIMUS_FUSED_ADAM_CLIP does not currently compose with an AMP GradScaler")
        try:
            return original_adam_step(self, closure=closure, grad_scaler=None)
        finally:
            self._primus_clip_norm = None
            self._primus_clip_max_norm = 1.0

    @wraps(original_scale_loss)
    def patched_scale_loss(self, loss):
        if _static_bf16_loss_scale_cpu_enabled() and _uses_static_bf16_unity_scale(self):
            return loss
        return original_scale_loss(self, loss)

    @wraps(original_chained_get_loss_scale)
    def patched_chained_get_loss_scale(self):
        if _static_bf16_loss_scale_cpu_enabled() and _uses_static_bf16_unity_scale(self):
            scale = getattr(self, "_primus_cpu_loss_scale_one", None)
            if scale is None:
                scale = torch.ones(1, dtype=torch.float32, device="cpu")
                self._primus_cpu_loss_scale_one = scale
            return scale
        return original_chained_get_loss_scale(self)

    def patched_copy_main_params_to_model_params(self):
        if not getattr(self, "_primus_bf16_writeback_active", False):
            return original_copy_main_params(self)
        try:
            # The Adam kernel already wrote all BF16 destinations. Native FP32
            # model groups, if present, still use Megatron's ordinary copy.
            _copy_only_fp32_model_groups(self)
        finally:
            self._primus_bf16_writeback_active = False

    @wraps(original_get_main_grads)
    def patched_get_main_grads_for_grad_norm(self, grad_norm_group=None):
        if grad_norm_group is not None:
            return original_get_main_grads(self, grad_norm_group)
        if not _bf16_grad_enabled():
            return original_get_main_grads(self)
        return _get_decoupled_grads_for_grad_norm(self)

    @wraps(original_copy_model_grads)
    def patched_copy_model_grads_to_main_grads(self):
        if not _bf16_grad_enabled():
            return original_copy_model_grads(self)
        return _copy_model_grads_to_decoupled(self)

    @wraps(original_collect_grads)
    def patched_collect_main_grad_data_for_unscaling(self):
        if not _bf16_grad_enabled():
            return original_collect_grads(self)
        return [
            param.decoupled_grad.data
            for group in self.optimizer.param_groups
            for param in group["params"]
            if getattr(param, "decoupled_grad", None) is not None
        ]

    @wraps(original_zero_grad)
    def patched_zero_grad(self, set_to_none: bool = True):
        result = original_zero_grad(self, set_to_none=set_to_none)
        if _bf16_grad_enabled():
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    param.decoupled_grad = None
        return result

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
        # Megatron only uses this return value for logging after the update.
        return _logged_grad_norm(grad_norm)

    @torch.no_grad()
    def patched_chained_step(self):
        """ChainedOptimizer.step with device clipping delegated to each Adam."""
        return _chained_step_with_device_clip(self, FusedAdam)

    FusedAdam.__init__ = patched_adam_init
    FusedAdam.step = patched_adam_step
    MegatronOptimizer.scale_loss = patched_scale_loss
    ChainedOptimizer.get_loss_scale = patched_chained_get_loss_scale
    OptimizerConfig.__post_init__ = patched_optimizer_config_post_init
    setattr(MegatronOptimizer, grads_for_norm_name, patched_get_main_grads_for_grad_norm)
    MegatronOptimizer.clip_grad_norm = patched_clip_grad_norm
    ChainedOptimizer.step = patched_chained_step
    DistributedOptimizer._copy_main_params_to_model_params = patched_copy_main_params_to_model_params
    DistributedOptimizer._copy_model_grads_to_main_grads = patched_copy_model_grads_to_main_grads
    DistributedOptimizer._collect_main_grad_data_for_unscaling = patched_collect_main_grad_data_for_unscaling
    DistributedOptimizer.zero_grad = patched_zero_grad

    # The existing TP1/PP1 synchronization-skip patch has already established
    # that the model-parallel reduction is a no-op. Keep a tensor statistic on
    # device instead of converting it back to a host float after Adam.
    import megatron.training.training as megatron_training

    reduce_max_stat = megatron_training.reduce_max_stat_across_model_parallel_group
    if getattr(reduce_max_stat, "__name__", "") == "_passthrough_reduce_max":
        megatron_training.reduce_max_stat_across_model_parallel_group = _deferred_passthrough_reduce_max

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
        + ("; BF16 DDP gradient shards are consumed directly by Adam." if _bf16_grad_enabled() else "")
        + (
            "; static BF16 unity loss scale is logged from CPU without a device sync."
            if _static_bf16_loss_scale_cpu_enabled()
            else ""
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
