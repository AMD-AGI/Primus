###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Backend-neutral Triton Adam/AdamW kernels with a device-sized launch grid.

AdamW is a 4R3W streaming kernel, so throughput is set by the number of
in-flight memory requests. The generic ``multi_tensor_apply`` launcher used by
ATen's ``torch.optim.AdamW(fused=True)`` caps every launch at 320 workgroups
(``depth_to_max_blocks``), which cannot hide DRAM latency on MI455X (256 CUs,
~20 TB/s HBM) and is not optimal on MI355X either.

Large tensors get one grid-stride launch each with direct addressing and an
architecture-dependent grid (``workgroups_per_cu``). Small
tensors are batched into one launch per dtype combination through a
chunk -> tensor table, so models with many small parameters are not bound by
per-launch overhead. All math runs in FP32 and matches TE ``multi_tensor_adam``.
"""

import functools
import struct

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024
# Grid-stride workgroups per CU at which the per-tensor kernel saturates HBM on
# Llama 3.1 8B shapes. MI355X (gfx950) is best at 1-3 and ~6% slower at 16;
# MI455X (gfx1250) needs >= 8. gfx942 is assumed to behave like gfx950.
_WORKGROUPS_PER_CU = {"gfx942": 2, "gfx950": 2, "gfx1250": 16}
_DEFAULT_WORKGROUPS_PER_CU = {64: 2, 32: 16}  # unlisted arch: by wavefront size
# Below this, a dedicated launch costs more than the tensor's HBM traffic.
_MULTI_TENSOR_MAX_NUMEL = 1 << 22
# The batched kernel reinterprets raw addresses, so it cannot see alignment.
_MULTI_TENSOR_ALIGN_BYTES = 16
# Each table entry covers CHUNK_BLOCKS x BLOCK_SIZE elements to amortize the
# dependent metadata loads; larger chunks spill registers on gfx1250.
_MULTI_TENSOR_BLOCK_SIZE = 2048
_MULTI_TENSOR_CHUNK_BLOCKS = 8
_MULTI_TENSOR_NUM_WARPS = 8
_MULTI_TENSOR_CHUNK = _MULTI_TENSOR_BLOCK_SIZE * _MULTI_TENSOR_CHUNK_BLOCKS
_TL_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


@triton.jit
def _adam_update(
    p,
    g,
    m,
    v,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    ADAM_W_MODE: tl.constexpr,
):
    if not ADAM_W_MODE:
        g = g + weight_decay * p
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    update = (m / bias_correction1) / (tl.sqrt(v / bias_correction2) + eps)
    if ADAM_W_MODE:
        update = update + weight_decay * p
    return p - lr * update, m, v


@triton.jit
def _adam_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    v_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    ADAM_W_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64) * BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    while block_start < n_elements:
        offs = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offs < n_elements

        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
        p, m, v = _adam_update(
            p, g, m, v, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, ADAM_W_MODE
        )
        tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)
        tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
        tl.store(v_ptr + offs, v.to(v_ptr.dtype.element_ty), mask=mask)

        block_start += stride


@triton.jit
def _adam_multi_tensor_kernel(
    meta_ptr,
    chunk_tensor_ptr,
    chunk_local_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    P_DTYPE: tl.constexpr,
    G_DTYPE: tl.constexpr,
    M_DTYPE: tl.constexpr,
    V_DTYPE: tl.constexpr,
    ALIGN: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_BLOCKS: tl.constexpr,
):
    # One program per chunk; meta rows are [p_addr, g_addr, m_addr, v_addr, numel].
    chunk = tl.program_id(0)
    row = meta_ptr + tl.load(chunk_tensor_ptr + chunk).to(tl.int64) * 5
    p_ptr = tl.multiple_of(tl.load(row + 0).to(tl.pointer_type(P_DTYPE)), ALIGN)
    g_ptr = tl.multiple_of(tl.load(row + 1).to(tl.pointer_type(G_DTYPE)), ALIGN)
    m_ptr = tl.multiple_of(tl.load(row + 2).to(tl.pointer_type(M_DTYPE)), ALIGN)
    v_ptr = tl.multiple_of(tl.load(row + 3).to(tl.pointer_type(V_DTYPE)), ALIGN)
    n_elements = tl.load(row + 4)
    chunk_start = tl.load(chunk_local_ptr + chunk).to(tl.int64) * (CHUNK_BLOCKS * BLOCK_SIZE)

    for i in tl.static_range(CHUNK_BLOCKS):
        offs = chunk_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
        p, m, v = _adam_update(
            p, g, m, v, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, ADAM_W_MODE
        )
        tl.store(p_ptr + offs, p.to(P_DTYPE), mask=mask)
        tl.store(m_ptr + offs, m.to(M_DTYPE), mask=mask)
        tl.store(v_ptr + offs, v.to(V_DTYPE), mask=mask)


@triton.jit
def _adam_param_remainder_update(
    w,
    g,
    m,
    v,
    beta1,
    beta2,
    step_size,
    beta2_corr_inv,
    eps,
    lr,
    weight_decay,
    ADAM_W_MODE: tl.constexpr,
    FMA_SECOND: tl.constexpr,
):
    # TE's ROCm multi_tensor_adam_param_remainder, operation for operation, with sqrtf and the
    # division correctly rounded. hipcc contracts the moment updates `a * b + c * d` into
    # fma(a, b, c * d), except in its fp32-moment AdamW instantiation, where it fuses the second
    # product: fma(c, d, a * b) (FMA_SECOND).
    if not ADAM_W_MODE:
        g = tl.fma(weight_decay, w, g)
    if FMA_SECOND:
        m = tl.fma(1.0 - beta1, g, beta1 * m)
        v = tl.fma((1.0 - beta2) * g, g, beta2 * v)
    else:
        m = tl.fma(beta1, m, (1.0 - beta1) * g)
        v = tl.fma(beta2, v, ((1.0 - beta2) * g) * g)
    q = tl.div_rn(m, tl.sqrt_rn(v * beta2_corr_inv) + eps)
    if ADAM_W_MODE:
        w = tl.fma(-(lr * weight_decay), w, tl.fma(-step_size, q, w))
    else:
        w = tl.fma(-step_size, q, w)
    return w, m, v


@triton.jit
def _adam_param_remainder_block(
    g_ptr,
    p_ptr,
    m_ptr,
    v_ptr,
    r_ptr,
    offs,
    n_elements,
    grad_scale,
    beta1,
    beta2,
    step_size,
    beta2_corr_inv,
    eps,
    lr,
    weight_decay,
    G_DTYPE: tl.constexpr,
    M_DTYPE: tl.constexpr,
    V_DTYPE: tl.constexpr,
    HAS_GRAD_SCALE: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    FMA_SECOND: tl.constexpr,
    MASKED: tl.constexpr,
):
    # The FP32 master weight is the bf16 param's bits over the int16 remainder's, the param rounded
    # to nearest (TE's store_param_remainders layout). Unmasked blocks keep the 16-byte accesses:
    # masking every element is much slower on gfx1250. Masked-off lanes are never stored.
    if MASKED:
        mask = offs < n_elements
    else:
        mask = None
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    p16 = tl.load(p_ptr + offs, mask=mask)
    r16 = tl.load(r_ptr + offs, mask=mask)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
    if HAS_GRAD_SCALE:
        # the gradient a separate in-place `grad *= scale` (fp32 product, one rounding) leaves
        g = (g * grad_scale).to(G_DTYPE).to(tl.float32)
    p16 = tl.where(r16 < 0, p16 - 1, p16)
    w = ((p16.to(tl.int32) << 16) | (r16.to(tl.int32) & 0xFFFF)).to(tl.float32, bitcast=True)
    w, m, v = _adam_param_remainder_update(
        w, g, m, v, beta1, beta2, step_size, beta2_corr_inv, eps, lr, weight_decay, ADAM_W_MODE, FMA_SECOND
    )
    bits = w.to(tl.int32, bitcast=True)
    r16 = bits.to(tl.int16)
    p16 = (bits >> 16).to(tl.int16)
    p16 = tl.where(r16 < 0, p16 + 1, p16)
    tl.store(p_ptr + offs, p16, mask=mask)
    tl.store(r_ptr + offs, r16, mask=mask)
    tl.store(m_ptr + offs, m.to(M_DTYPE), mask=mask)
    tl.store(v_ptr + offs, v.to(V_DTYPE), mask=mask)


@triton.jit
def _adam_param_remainder_multi_tensor_kernel(
    meta_ptr,
    chunk_tensor_ptr,
    chunk_local_ptr,
    grad_scale_ptr,
    beta1,
    beta2,
    step_size,
    beta2_corr_inv,
    eps,
    lr,
    weight_decay,
    G_DTYPE: tl.constexpr,
    M_DTYPE: tl.constexpr,
    V_DTYPE: tl.constexpr,
    HAS_GRAD_SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    FMA_SECOND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_BLOCKS: tl.constexpr,
):
    # One program per chunk; meta rows are [g_addr, p_addr, m_addr, v_addr, remainder_addr, numel].
    chunk = tl.program_id(0)
    row = meta_ptr + tl.load(chunk_tensor_ptr + chunk).to(tl.int64) * 6
    g_ptr = tl.multiple_of(tl.load(row + 0).to(tl.pointer_type(G_DTYPE)), ALIGN)
    p_ptr = tl.multiple_of(tl.load(row + 1).to(tl.pointer_type(tl.int16)), ALIGN)
    m_ptr = tl.multiple_of(tl.load(row + 2).to(tl.pointer_type(M_DTYPE)), ALIGN)
    v_ptr = tl.multiple_of(tl.load(row + 3).to(tl.pointer_type(V_DTYPE)), ALIGN)
    r_ptr = tl.multiple_of(tl.load(row + 4).to(tl.pointer_type(tl.int16)), ALIGN)
    n_elements = tl.load(row + 5)
    chunk_start = tl.load(chunk_local_ptr + chunk).to(tl.int64) * (CHUNK_BLOCKS * BLOCK_SIZE)
    grad_scale = 1.0
    if HAS_GRAD_SCALE:
        grad_scale = tl.load(grad_scale_ptr)
    full = chunk_start + CHUNK_BLOCKS * BLOCK_SIZE <= n_elements
    for i in tl.static_range(CHUNK_BLOCKS):
        offs = chunk_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if full:
            _adam_param_remainder_block(
                g_ptr,
                p_ptr,
                m_ptr,
                v_ptr,
                r_ptr,
                offs,
                n_elements,
                grad_scale,
                beta1,
                beta2,
                step_size,
                beta2_corr_inv,
                eps,
                lr,
                weight_decay,
                G_DTYPE,
                M_DTYPE,
                V_DTYPE,
                HAS_GRAD_SCALE,
                ADAM_W_MODE,
                FMA_SECOND,
                False,
            )
        else:
            _adam_param_remainder_block(
                g_ptr,
                p_ptr,
                m_ptr,
                v_ptr,
                r_ptr,
                offs,
                n_elements,
                grad_scale,
                beta1,
                beta2,
                step_size,
                beta2_corr_inv,
                eps,
                lr,
                weight_decay,
                G_DTYPE,
                M_DTYPE,
                V_DTYPE,
                HAS_GRAD_SCALE,
                ADAM_W_MODE,
                FMA_SECOND,
                True,
            )


def workgroups_per_cu(arch_name: str, warp_size: int) -> int:
    arch = arch_name.split(":")[0]
    if arch in _WORKGROUPS_PER_CU:
        return _WORKGROUPS_PER_CU[arch]
    return _DEFAULT_WORKGROUPS_PER_CU.get(warp_size, 16)


@functools.lru_cache(maxsize=None)
def default_grid_size(device_index: int) -> int:
    props = torch.cuda.get_device_properties(device_index)
    wg_per_cu = workgroups_per_cu(getattr(props, "gcnArchName", ""), getattr(props, "warp_size", 32))
    return props.multi_processor_count * wg_per_cu


def _bias_corrections(beta1, beta2, step, bias_correction):
    if bias_correction:
        return 1.0 - beta1**step, 1.0 - beta2**step
    return 1.0, 1.0


def triton_adam_step_(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    bias_correction: bool = True,
    adam_w_mode: bool = True,
    grid_size: int = 0,
) -> None:
    """In-place Adam/AdamW update of one tensor, matching TE ``multi_tensor_adam`` math."""
    n_elements = param.numel()
    if n_elements == 0:
        return
    bias_correction1, bias_correction2 = _bias_corrections(beta1, beta2, step, bias_correction)
    max_grid = grid_size if grid_size > 0 else default_grid_size(param.device.index)
    grid = (min(triton.cdiv(n_elements, _BLOCK_SIZE), max_grid),)
    _adam_kernel[grid](
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        ADAM_W_MODE=bool(adam_w_mode),
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def _build_chunk_tables(rows, device):
    """Device tables of a chunked launch: the rows, and per chunk its row and its index in that row.
    A row's last entry is its element count."""
    meta = torch.tensor(rows, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
    chunks = [triton.cdiv(row[-1], _MULTI_TENSOR_CHUNK) for row in rows]
    total = sum(chunks)
    counts = torch.tensor(chunks, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
    chunk_tensor = torch.repeat_interleave(
        torch.arange(len(rows), dtype=torch.int32, device=device), counts, output_size=total
    )
    first_chunk = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts, output_size=total)
    chunk_local = (torch.arange(total, device=device) - first_chunk).to(torch.int32)
    return meta, chunk_tensor, chunk_local


def triton_multi_tensor_adam_step_(
    tensors: list,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    bias_correction: bool = True,
    adam_w_mode: bool = True,
    cache: dict = None,
    rows: list = None,
) -> None:
    """In-place Adam/AdamW update of many ``(param, grad, exp_avg, exp_avg_sq)`` tuples in one launch.

    All tuples must share the same dtype per slot, be contiguous, non-empty and
    ``_MULTI_TENSOR_ALIGN_BYTES``-aligned. ``cache`` keeps the device tables
    across calls and is reused while the addresses and sizes are unchanged.
    ``rows`` may pass precomputed ``[p_addr, g_addr, m_addr, v_addr, numel]``
    lists matching ``tensors``.
    """
    if not tensors:
        return
    if rows is None:
        rows = [[t.data_ptr() for t in ts] + [ts[0].numel()] for ts in tensors]
    if cache is not None and cache.get("rows") == rows:
        tables = cache["tables"]
    else:
        tables = _build_chunk_tables(rows, tensors[0][0].device)
        if cache is not None:
            cache.update(rows=rows, tables=tables)

    bias_correction1, bias_correction2 = _bias_corrections(beta1, beta2, step, bias_correction)
    p, g, m, v = tensors[0]
    _adam_multi_tensor_kernel[(tables[1].numel(),)](
        *tables,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        P_DTYPE=_TL_DTYPES[p.dtype],
        G_DTYPE=_TL_DTYPES[g.dtype],
        M_DTYPE=_TL_DTYPES[m.dtype],
        V_DTYPE=_TL_DTYPES[v.dtype],
        ALIGN=_MULTI_TENSOR_ALIGN_BYTES,
        ADAM_W_MODE=bool(adam_w_mode),
        BLOCK_SIZE=_MULTI_TENSOR_BLOCK_SIZE,
        CHUNK_BLOCKS=_MULTI_TENSOR_CHUNK_BLOCKS,
        num_warps=_MULTI_TENSOR_NUM_WARPS,
    )


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def param_remainder_scalars(lr, beta1, beta2, step, bias_correction=True):
    """``(step_size, beta2_corr_inv)`` as TE's ``multi_tensor_adam_param_remainder`` computes them on the
    host: ``std::pow(float, int)`` runs in double, the corrections and their uses are float."""
    bias_correction1 = bias_correction2 = 1.0
    if bias_correction:
        bias_correction1 = _f32(1.0 - float(_f32(beta1)) ** step)
        bias_correction2 = _f32(1.0 - float(_f32(beta2)) ** step)
    return _f32(_f32(lr) / bias_correction1), _f32(1.0 / bias_correction2)


def triton_multi_tensor_adam_param_remainder_step_(
    tensors: list,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    bias_correction: bool = True,
    adam_w_mode: bool = True,
    grad_scale: torch.Tensor = None,
    cache: dict = None,
) -> None:
    """In-place Adam/AdamW of bf16 params whose FP32 master weights are kept as int16 remainders
    (TE ``FusedAdam(master_weights=True, store_param_remainders=True)``), in one launch, bit for bit
    TE's ROCm ``multi_tensor_adam_param_remainder``.

    ``tensors`` holds ``(grad, param, exp_avg, exp_avg_sq, remainder)`` tuples: bf16 params, int16
    remainders, grads bf16 or fp32 and both moments bf16 or both fp32, one dtype per slot, all
    contiguous, non-empty and of the param's size. ``grad_scale`` (a one-element fp32 tensor on the
    device) applies the update to the gradients a separate ``grad *= grad_scale`` would leave: the
    fp32 product rounded to the gradient's dtype, as global-norm clipping does. ``cache`` keeps the
    device tables while the addresses and sizes are unchanged.
    """
    if not tensors:
        return
    rows = [[t.data_ptr() for t in ts] + [ts[0].numel()] for ts in tensors]
    if cache is not None and cache.get("rows") == rows:
        tables = cache["tables"]
    else:
        tables = _build_chunk_tables(rows, tensors[0][1].device)
        if cache is not None:
            cache.update(rows=rows, tables=tables)
    align = _MULTI_TENSOR_ALIGN_BYTES
    while align > 2 and any(addr % align for row in rows for addr in row[:5]):
        align //= 2
    step_size, beta2_corr_inv = param_remainder_scalars(lr, beta1, beta2, step, bias_correction)
    g, p, m, v, r = tensors[0]
    assert p.dtype == torch.bfloat16 and r.dtype == torch.int16, (p.dtype, r.dtype)
    assert m.dtype == v.dtype in (torch.bfloat16, torch.float32), (m.dtype, v.dtype)
    _adam_param_remainder_multi_tensor_kernel[(tables[1].numel(),)](
        *tables,
        grad_scale if grad_scale is not None else tables[0],
        beta1,
        beta2,
        step_size,
        beta2_corr_inv,
        eps,
        lr,
        weight_decay,
        G_DTYPE=_TL_DTYPES[g.dtype],
        M_DTYPE=_TL_DTYPES[m.dtype],
        V_DTYPE=_TL_DTYPES[v.dtype],
        HAS_GRAD_SCALE=grad_scale is not None,
        ALIGN=align,
        ADAM_W_MODE=bool(adam_w_mode),
        FMA_SECOND=bool(adam_w_mode) and m.dtype == torch.float32,
        BLOCK_SIZE=_MULTI_TENSOR_BLOCK_SIZE,
        CHUNK_BLOCKS=_MULTI_TENSOR_CHUNK_BLOCKS,
        num_warps=_MULTI_TENSOR_NUM_WARPS,
    )


def triton_supports(t: torch.Tensor) -> bool:
    return (
        type(t) in (torch.Tensor, torch.nn.Parameter)
        and t.is_cuda
        and t.dtype in _TL_DTYPES
        and t.is_contiguous()
    )


def is_batchable(tensors) -> bool:
    return 0 < tensors[0].numel() < _MULTI_TENSOR_MAX_NUMEL and all(
        t.data_ptr() % _MULTI_TENSOR_ALIGN_BYTES == 0 for t in tensors
    )
