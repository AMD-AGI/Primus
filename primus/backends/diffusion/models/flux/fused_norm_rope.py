###############################################################################
# Fused QK-RMSNorm + RoPE for the Primus/torch FLUX blocks.
#
# FLUX norms Q and K and then rotates them. Left to Inductor that is one kernel
# forward and three backward, and the rope backward writes a full-size
# intermediate the norm backward reads straight back, per block, 57 times a step.
# This does each direction in one pass.
#
# Adapted from the equivalent Megatron-side kernel. Two things differ here
# and both are in the addressing rather than the math:
#
#   * Megatron hands the kernel ``mixed_qkv`` with Q, K and V interleaved per
#     head, so every row sits at one uniform stride. Primus splits K-major
#     (``rearrange(qkv, "B L (K H D) -> K B L H D")``), so Q, K and V are three
#     contiguous blocks: rows step by D inside a token and by 3*H*D between
#     tokens. A single stride cannot describe that, so a row is addressed as
#     ``(row // H) * STOK + (row % H) * SH``. Packed outputs are the same
#     expression with STOK = H*D, so there is one path, not two.
#
#   * Primus carries RoPE as a 2x2 matrix ``freqs_cis`` of shape
#     [B, N, 1, D/2, 2, 2] holding [[cos, -sin], [sin, cos]] applied to pairs
#     (x[2i], x[2i+1]). Expanding primus/backends/diffusion/models/flux/math.py
#     gives out[2i] = cos*x[2i] - sin*x[2i+1] and out[2i+1] = sin*x[2i] +
#     cos*x[2i+1], which is exactly the interleaved branch below. So this is not
#     a numerics change; it is the same rotation with the pairing kept out of the
#     addresses. ``cos_sin_tables`` below converts the matrix once per step.
#
# Registered with @triton_op rather than @custom_op for the per-tensor entry
# point so Inductor can still fuse across it; the whole-QKV op is deliberately
# opaque because its neighbours (a GEMM and FMHA) already are.
###############################################################################

import os
from typing import NamedTuple, Tuple

import torch
import triton
import triton.language as tl
from torch.library import custom_op, triton_op, wrap_triton


class RopeTables(NamedTuple):
    """cos/sin rows for the whole sequence and for each side of a double block.

    A NamedTuple of plain tensors rather than a dict so Dynamo treats it as a
    static structure and the blocks stay fullgraph-compilable.
    """

    all_cos: torch.Tensor
    all_sin: torch.Tensor
    txt_cos: torch.Tensor
    txt_sin: torch.Tensor
    img_cos: torch.Tensor
    img_sin: torch.Tensor


@triton.jit
def _materialize(x):
    """Opaque identity: stops fp contraction and folding across this value.

    Inductor runs rope backward and norm backward as two kernels with the rope result
    stored in fp32 between them, so nothing fuses across that boundary. Within one
    kernel the compiler would otherwise contract into the next multiply or drop the
    `+ 0.0`, and the result stops being bit-identical.
    """
    return tl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _base(rows, STOK, SH, H):
    """Byte-free element offset of each row: token stride plus head stride.

    Subsumes the uniform-stride case (STOK == H * SH) and the K-major qkv column
    case (STOK == 3 * H * SH), which is the reason this is not a single stride.
    """
    return (rows // H) * STOK + (rows % H) * SH


@triton.jit
def _pair(P, rows, mask, STOK, SH, H, D: tl.constexpr, BLOCK_M: tl.constexpr, INTERLEAVED: tl.constexpr):
    """Load [BLOCK_M, D] rows of P and return the two members of each rotation pair."""
    half: tl.constexpr = D // 2
    b = _base(rows, STOK, SH, H)[:, None]
    if INTERLEAVED:
        v = tl.load(P + b + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
        return tl.split(tl.reshape(v, (BLOCK_M, half, 2)))
    lo = tl.arange(0, half)[None, :]
    a = tl.load(P + b + lo, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(P + b + lo + half, mask=mask, other=0.0).to(tl.float32)
    return a, c


@triton.jit
def _unpair(
    P, rows, mask, v_lo, v_hi, STOK, SH, H, D: tl.constexpr, BLOCK_M: tl.constexpr, INTERLEAVED: tl.constexpr
):
    """Inverse of _pair: interleave or concatenate, then store at the same addresses."""
    half: tl.constexpr = D // 2
    b = _base(rows, STOK, SH, H)[:, None]
    if INTERLEAVED:
        v = tl.reshape(tl.join(v_lo, v_hi), (BLOCK_M, D))
        tl.store(P + b + tl.arange(0, D)[None, :], v.to(P.dtype.element_ty), mask=mask)
    else:
        lo = tl.arange(0, half)[None, :]
        tl.store(P + b + lo, v_lo.to(P.dtype.element_ty), mask=mask)
        tl.store(P + b + lo + half, v_hi.to(P.dtype.element_ty), mask=mask)


@triton.jit
def _pair_w(W, D: tl.constexpr, INTERLEAVED: tl.constexpr):
    """The norm weight is D-wide and shared by every row; split it the same way."""
    half: tl.constexpr = D // 2
    if INTERLEAVED:
        v = tl.load(W + tl.arange(0, D)).to(tl.float32)
        a, b = tl.split(tl.reshape(v, (half, 2)))
        return a[None, :], b[None, :]
    lo = tl.arange(0, half)
    return (tl.load(W + lo).to(tl.float32)[None, :], tl.load(W + lo + half).to(tl.float32)[None, :])


# One config, not a search: the per-row tl.sum must add in the same order as
# Inductor's RMSNorm reduction, and its tree follows the tile shape and warp count.
def _fwd_configs():
    return [triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=2)]


@triton.autotune(configs=_fwd_configs(), key=["M", "D"])
@triton.jit
def _fwd_kernel(
    X,
    W,
    COS,
    SIN,
    OUT,
    RSTD,
    COPY_X,
    COPY_OUT,
    M,
    D: tl.constexpr,
    H,
    EPS,
    SX_TOK,
    SX_H,
    SO_TOK,
    SO_H,
    SCOPY_TOK,
    SCOPY_H,
    SC,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    COPY_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    mask = mask_m[:, None]

    # Sum the squares over the row in its natural order, as Inductor's reduction does;
    # summing the two halves of each rotation pair separately rounds differently.
    b = _base(rows, SX_TOK, SX_H, H)[:, None]
    x = tl.load(X + b + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / D + EPS)
    if INTERLEAVED:
        x_lo, x_hi = tl.split(tl.reshape(x, (BLOCK_M, D // 2, 2)))
    else:
        x_lo, x_hi = _pair(X, rows, mask, SX_TOK, SX_H, H, D, BLOCK_M, INTERLEAVED)

    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    n_lo, n_hi = x_lo * rstd * w_lo, x_hi * rstd * w_hi

    # cos/sin are [tokens, D]: one row per (batch, position), shared across heads.
    tok = rows // H
    c_lo, c_hi = _pair(COS, tok, mask, SC, SC, 1, D, BLOCK_M, INTERLEAVED)
    s_lo, s_hi = _pair(SIN, tok, mask, SC, SC, 1, D, BLOCK_M, INTERLEAVED)

    # Written as apply_rope's `f0 * x0 + f1 * x1` with f1 = -sin on the first member,
    # operands in the same order, so fp contraction fuses the same product.
    _unpair(
        OUT,
        rows,
        mask,
        tl.fma(c_lo, n_lo, (-s_lo) * n_hi),
        tl.fma(s_hi, n_lo, c_hi * n_hi),
        SO_TOK,
        SO_H,
        H,
        D,
        BLOCK_M,
        INTERLEAVED,
    )
    tl.store(RSTD + rows, tl.reshape(rstd, (BLOCK_M,)), mask=mask_m)

    # Piggyback V's compaction onto this kernel rather than paying a separate
    # copy kernel for it; the addresses are already resident.
    if COPY_ROW:
        src = _base(rows, SCOPY_TOK, SCOPY_H, H)[:, None] + tl.arange(0, D)[None, :]
        dst = _base(rows, D * H, D, H)[:, None] + tl.arange(0, D)[None, :]
        tl.store(COPY_OUT + dst, tl.load(COPY_X + src, mask=mask, other=0.0), mask=mask)


def _mix_order_split(M: int, device: torch.device) -> int:
    """Inductor's rows-per-program for this reduction (codegen/simd.py
    _codegen_mix_order_reduction): the last power of two of M / (8 * SMs), clamped to
    [16, 128]. The dw partials, and so their host-side sum, only match Inductor's bits
    when each program owns the same contiguous block of rows."""
    per = M // (torch.cuda.get_device_properties(device).multi_processor_count * 8)
    split = 1 << (per.bit_length() - 1) if per > 0 else 1
    return min(max(split, 16), 128)


# One config, not a search: tl.sum's reduction tree follows the tile shape and warp
# count. Inductor runs this backward at XBLOCK=4 on one warp; BLOCK_M=8 on one warp
# reproduces its dx and dw bits too (checked on both block shapes) and is 12-16% faster.
# Two warps change dx; BLOCK_M=16 changes dw.
#
# PRICED THREE TIMES AND STILL NOT RESOLVED. Default is 8, which is what our best run used.
#
# One node says it is worth nothing. m11-17, GBS 256, MAX_STEPS 2200, two-eval block-2, zero nan:
# g0 73.16 at BLOCK_M=4 against g2 73.18 at BLOCK_M=8, i.e. +0.03% on a box whose repeat error the
# same pair of runs puts at 0.05%. That is roughly the ceiling the arithmetic implies -- 12-16% off a
# kernel that is 15.86 of 158.91 ms/step cannot exceed ~1.5% -- so at one node the kernel is simply
# not on the critical path.
#
# Four nodes LOOKED like it disagreed, and an earlier version of this comment claimed +0.74% here with
# non-overlapping bands. That claim is withdrawn: it compared per-step medians over windows that were
# never pinned to the same steps. Over one identical window (steps 300-590, and the same four nodes in
# every run) the five runs of this recipe read
#   BLOCK_M=8   78.49 (36634376696)   78.13 (32196)
#   BLOCK_M=4   77.92 (36643010884)   77.98 (nccl-h0)   77.69 (nccl-h1b)
# so the group means differ by ~0.55% while two runs of the SAME flag differ by 0.46%. The 4-node
# per-run offset is that large because every job re-autotunes Inductor from cold under
# TORCH_COMPILE_MODE=max-autotune-no-cudagraphs and does not pick the same kernels twice -- the
# mm/addmm autotune counts differ run to run (88/66, 86/64, 88/65, 87/62). Nothing below ~1% has ever
# been measurable at 4 nodes in an unpaired cell, which is why the paired within-job harness exists.
#
# So 8 rather than 4, on two grounds that do not need the above resolved: it is bit-identical to
# Inductor's dx and dw on both block shapes (checked), and 36634376696 -- 0.586 at 8,388,608 samples
# in 58.2 minutes, our best time-to-train -- ran with it. Reproducing our best run beats diverging
# from it over a flag that is free either way.
#
# Standing lesson from e4, which claimed +2.78% here: single-digit-percent A/Bs need either m11-17,
# where the repeat error is measured at 0.05%, or a paired same-job cell on the reserved nodes.
def _bwd_configs():
    bm = int(os.getenv("FLUX_FUSED_NORM_ROPE_BWD_BLOCK_M", "8"))
    return [triton.Config({"BLOCK_M": bm}, num_warps=1, num_stages=1)]


# dw reduces every token onto a D-wide vector. Doing that with atomic_add from one
# program per row puts ~200k programs on the same D addresses. Instead, as in
# Inductor's mix-order reduction, each program owns RSPLIT contiguous rows, keeps a
# private accumulator in registers, and writes one partial row for the host to sum.
@triton.autotune(configs=_bwd_configs(), key=["M", "D"])
@triton.jit
def _bwd_kernel(
    G,
    X,
    W,
    COS,
    SIN,
    RSTD,
    DX,
    DWP,
    M,
    D: tl.constexpr,
    H,
    SG_TOK,
    SG_H,
    SX_TOK,
    SX_H,
    SDX_TOK,
    SDX_H,
    SC,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    RSPLIT: tl.constexpr,
    ROUND_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    half: tl.constexpr = D // 2
    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    dw_lo = tl.zeros((half,), dtype=tl.float32)
    dw_hi = tl.zeros((half,), dtype=tl.float32)
    dw_nat = tl.zeros((D,), dtype=tl.float32)

    for start in tl.range(pid * RSPLIT, pid * RSPLIT + RSPLIT, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        mask = (rows < M)[:, None]
        rstd = tl.load(RSTD + rows, mask=rows < M, other=0.0)[:, None]

        if INTERLEAVED:
            # Everything in the natural [rows, D] layout Inductor reduces over; building
            # the row from rotation pairs changes the order tl.sum adds in.
            cols = tl.arange(0, D)[None, :]
            first = (cols % 2) == 0
            g = tl.load(G + _base(rows, SG_TOK, SG_H, H)[:, None] + cols, mask=mask, other=0.0).to(tl.float32)
            g0, g1 = tl.split(tl.reshape(g, (BLOCK_M, D // 2, 2)))
            g_a = tl.reshape(tl.join(g0, g0), (BLOCK_M, D))
            g_b = tl.reshape(tl.join(g1, g1), (BLOCK_M, D))
            x = tl.load(X + _base(rows, SX_TOK, SX_H, H)[:, None] + cols, mask=mask, other=0.0).to(tl.float32)
            cos = tl.load(COS + (rows // H)[:, None] * SC + cols, mask=mask, other=0.0)
            sin = tl.load(SIN + (rows // H)[:, None] * SC + cols, mask=mask, other=0.0)
            w = tl.load(W + cols).to(tl.float32)
            # apply_rope's autograd in Inductor's form: member j takes g0*f[j,0] + g1*f[j,1]
            # as a plain expression (the compiler's own contraction matches Inductor's), then
            # `+ 0.0` from select_backward's where(), which turns -0.0 into +0.0.
            dn = (g_a * tl.where(first, cos, -sin) + g_b * tl.where(first, sin, cos)) + 0.0
            if ROUND_GRAD:
                # DoubleStreamBlock rotates after the txt/img concat, so the concat's
                # backward separates rope and norm backward and Inductor materializes the
                # rope gradient in apply_rope's input dtype, bf16, in between.
                dn = dn.to(tl.bfloat16).to(tl.float32)
            else:
                dn = _materialize(dn)
            u = x * rstd
            du = dn * w
            # aten._fused_rms_norm_backward: (grad_x_hat - (x_hat / D) * sum(x_hat * grad_x_hat)) * rstd
            s = tl.sum(u * du, axis=1)[:, None]
            inv_d: tl.constexpr = 1.0 / D
            dx = (du - (u * inv_d) * s) * rstd
            tl.store(
                DX + _base(rows, SDX_TOK, SDX_H, H)[:, None] + cols, dx.to(DX.dtype.element_ty), mask=mask
            )
            dw_nat += tl.sum(tl.where(mask, dn * u, 0.0), axis=0)
        else:
            g_lo, g_hi = _pair(G, rows, mask, SG_TOK, SG_H, H, D, BLOCK_M, INTERLEAVED)
            x_lo, x_hi = _pair(X, rows, mask, SX_TOK, SX_H, H, D, BLOCK_M, INTERLEAVED)
            tok = rows // H
            c_lo, c_hi = _pair(COS, tok, mask, SC, SC, 1, D, BLOCK_M, INTERLEAVED)
            s_lo, s_hi = _pair(SIN, tok, mask, SC, SC, 1, D, BLOCK_M, INTERLEAVED)
            dn_lo = g_lo * c_lo + g_hi * s_hi
            dn_hi = g_hi * c_hi - g_lo * s_lo
            u_lo, u_hi = x_lo * rstd, x_hi * rstd
            du_lo, du_hi = dn_lo * w_lo, dn_hi * w_hi
            m = (tl.sum(du_lo * u_lo, axis=1) + tl.sum(du_hi * u_hi, axis=1))[:, None] / D
            _unpair(
                DX,
                rows,
                mask,
                rstd * (du_lo - u_lo * m),
                rstd * (du_hi - u_hi * m),
                SDX_TOK,
                SDX_H,
                H,
                D,
                BLOCK_M,
                INTERLEAVED,
            )
            dw_lo += tl.sum(tl.where(mask, dn_lo * u_lo, 0.0), axis=0)
            dw_hi += tl.sum(tl.where(mask, dn_hi * u_hi, 0.0), axis=0)

    # The non-interleaved accumulators are in pair order, so unpair before the host
    # sums them onto the layout the weight actually has.
    if INTERLEAVED:
        tl.store(DWP + pid * D + tl.arange(0, D), dw_nat)
    else:
        lo = tl.arange(0, half)
        tl.store(DWP + pid * D + lo, dw_lo)
        tl.store(DWP + pid * D + lo + half, dw_hi)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def cos_sin_tables(freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn FLUX's [B, N, 1, D/2, 2, 2] rotation matrix into [B*N, D] cos/sin.

    Call once per step on ``pe`` and pass the result to every block: ``pe`` is
    built once in Model.forward and shared by all 57 of them, so this is not a
    per-block cost. Each value is duplicated across its pair so the kernel can
    read a row of D contiguously instead of gathering D/2 and broadcasting.
    """
    if freqs_cis.dim() != 6 or freqs_cis.shape[-2:] != (2, 2):
        raise ValueError(f"expected [..., D/2, 2, 2] rotation matrix, got {tuple(freqs_cis.shape)}")
    # [..., 0, 0] is cos and [..., 1, 0] is sin; the other two are -sin and cos.
    cos = freqs_cis[..., 0, 0].reshape(-1, freqs_cis.shape[-3])
    sin = freqs_cis[..., 1, 0].reshape(-1, freqs_cis.shape[-3])
    # FSDP2's mixed-precision policy re-casts these at every block call: 342
    # bfloat16_copy_kernel launches a step, 2.39 ms, for 6 tables x 57 blocks, and
    # the kernel below only ever reads the bf16 copies. Rounding here keeps the fp32
    # intermediate, so a value still sees exactly one fp32->bf16 round-to-nearest-even
    # and the tables are bitwise what the kernel is handed today.
    _dt = torch.bfloat16 if os.environ.get("FLUX_ROPE_TABLES_BF16", "0") == "1" else torch.float32
    return (
        cos.repeat_interleave(2, dim=-1).contiguous().float().to(_dt),
        sin.repeat_interleave(2, dim=-1).contiguous().float().to(_dt),
    )


def rope_tables(freqs_cis: torch.Tensor, spans) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Contiguous cos/sin tables for each (offset, length) slice of the sequence.

    DoubleStreamBlock rotates the concatenated txt+img sequence, so each side
    needs the table rows for its own positions. RoPE is positional, so rotating
    each side with its slice is identical to rotating after the concat.

    Slicing positions out of a [B, N, D] table is not a view once flattened to
    [B*L, D], so each span costs one copy -- but ``pe`` is built once in
    Model.forward and shared by all 57 blocks, so this is paid once per step, not
    once per block. The alternative, teaching the kernel a second level of table
    addressing, buys nothing at this size.
    """
    cos, sin = cos_sin_tables(freqs_cis)
    D = cos.shape[-1]
    B, N = freqs_cis.shape[0], freqs_cis.shape[1]
    cos, sin = cos.view(B, N, D), sin.view(B, N, D)
    return tuple(
        (
            cos[:, off : off + length].reshape(-1, D).contiguous(),
            sin[:, off : off + length].reshape(-1, D).contiguous(),
        )
        for off, length in spans
    )


def default_eps(dtype: torch.dtype) -> float:
    """What nn.RMSNorm(dim) uses when eps is left None.

    FLUX constructs RMSNorm without an eps to match TorchTitan. aten._fused_rms_norm
    then takes the epsilon of its *computation* dtype, which is float32 for bf16 and
    fp16 inputs: 1.19e-7, not bfloat16's 0.0078125 as the nn.RMSNorm docstring's
    "finfo(x.dtype).eps" suggests. The two differ by 25% on rows with mean(x^2) ~ 0.01.
    """
    return torch.finfo(torch.float64 if dtype == torch.float64 else torch.float32).eps


def _strides(t: torch.Tensor, H: int, D: int) -> Tuple[int, int]:
    """(token stride, head stride) for a [*, H, D] view, or raise if unaddressable.

    A row must be D contiguous elements, heads must be evenly spaced inside a token,
    and tokens must be evenly spaced across every leading dim, because the kernel
    flattens batch and sequence into one token index. A sequence slice of a larger
    tensor (e.g. one side of DoubleStreamBlock's txt/img concat) breaks the last rule.
    """
    if t.shape[-2:] != (H, D):
        raise ValueError(f"expected trailing ({H}, {D}), got {tuple(t.shape)}")
    *_, s_tok, s_h, s_d = t.stride()
    if s_d != 1 or s_h < D:
        raise ValueError(f"rows must be contiguous and non-overlapping, strides {t.stride()}")
    sizes, strides = t.shape[:-2], t.stride()[:-2]
    expected = s_tok
    for i in range(len(sizes) - 2, -1, -1):
        expected *= sizes[i + 1]
        if sizes[i] > 1 and strides[i] != expected:
            raise ValueError(f"tokens are not uniformly strided across leading dims, strides {t.stride()}")
    return s_tok, s_h


def _launch_fwd(x, w, cos, sin, eps, interleaved, traceable, copy_x=None, copy_out=None):
    H, D = x.shape[-2], x.shape[-1]
    M = x.numel() // D
    # One table row per (batch, position). A mismatch here rotates by the wrong
    # position instead of failing, so it is worth asserting rather than trusting.
    if cos.shape != (M // H, D) or sin.shape != cos.shape:
        raise ValueError(f"cos/sin must be ({M // H}, {D}), got {tuple(cos.shape)}")
    sx_tok, sx_h = _strides(x, H, D)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if copy_x is not None:
        scopy_tok, scopy_h = _strides(copy_x, H, D)
    else:
        copy_x, copy_out, scopy_tok, scopy_h = x, out, sx_tok, sx_h

    launch = wrap_triton(_fwd_kernel) if traceable else _fwd_kernel
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    launch[grid](
        x,
        w,
        cos,
        sin,
        out,
        rstd,
        copy_x,
        copy_out,
        M,
        D,
        H,
        eps,
        sx_tok,
        sx_h,
        H * D,
        D,
        scopy_tok,
        scopy_h,
        cos.stride(0),
        INTERLEAVED=interleaved,
        COPY_ROW=copy_out is not out,
    )
    return out, rstd


def _launch_bwd(g, x, w, cos, sin, rstd, dx, interleaved, traceable, round_grad=False):
    H, D = x.shape[-2], x.shape[-1]
    M = x.numel() // D
    sg_tok, sg_h = _strides(g, H, D)
    sx_tok, sx_h = _strides(x, H, D)
    sdx_tok, sdx_h = _strides(dx, H, D)
    rsplit = _mix_order_split(M, x.device)
    nprog = triton.cdiv(M, rsplit)
    # Every row is written by exactly one program, so empty() is safe.
    dwp = torch.empty(nprog, D, device=x.device, dtype=torch.float32)
    launch = wrap_triton(_bwd_kernel) if traceable else _bwd_kernel
    launch[(nprog,)](
        g,
        x,
        w,
        cos,
        sin,
        rstd,
        dx,
        dwp,
        M,
        D,
        H,
        sg_tok,
        sg_h,
        sx_tok,
        sx_h,
        sdx_tok,
        sdx_h,
        cos.stride(0),
        INTERLEAVED=interleaved,
        RSPLIT=rsplit,
        ROUND_GRAD=round_grad,
    )
    return dwp


@triton_op("primus_flux::qk_norm_rope", mutates_args=())
def _fwd_op(
    x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, interleaved: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _launch_fwd(x, w, cos, sin, eps, interleaved, traceable=True)


@_fwd_op.register_fake
def _(x, w, cos, sin, eps, interleaved):
    D = x.shape[-1]
    return (
        torch.empty(x.shape, device=x.device, dtype=x.dtype),
        torch.empty(x.numel() // D, device=x.device, dtype=torch.float32),
    )


@triton_op("primus_flux::qk_norm_rope_backward", mutates_args=())
def _bwd_op(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rstd: torch.Tensor,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    return dx, _launch_bwd(g, x, w, cos, sin, rstd, dx, interleaved, traceable=True)


@_bwd_op.register_fake
def _(g, x, w, cos, sin, rstd, interleaved):
    M = x.numel() // x.shape[-1]
    nprog = triton.cdiv(M, _mix_order_split(M, x.device))
    return (
        torch.empty(x.shape, device=x.device, dtype=x.dtype),
        torch.empty(nprog, x.shape[-1], device=x.device, dtype=torch.float32),
    )


def _fwd_setup(ctx, inputs, output):
    x, w, cos, sin, eps, interleaved = inputs
    ctx.save_for_backward(x, w, cos, sin, output[1])
    ctx.interleaved = interleaved


def _fwd_backward(ctx, grad_out, _grad_rstd):
    x, w, cos, sin, rstd = ctx.saved_tensors
    dx, dwp = _bwd_op(grad_out, x, w, cos, sin, rstd, ctx.interleaved)
    return dx, dwp.sum(0).to(w.dtype), None, None, None, None


_fwd_op.register_autograd(_fwd_backward, setup_context=_fwd_setup)


def fused_qk_norm_rope(x, weight, cos, sin, eps=None, interleaved=True):
    """RMS-norm x over its last dim then apply RoPE, one kernel each direction.

    ``x`` is [..., H, D] and need not be contiguous; a K-major slice of the qkv
    GEMM output is read in place. ``cos``/``sin`` come from ``cos_sin_tables``.
    """
    if eps is None:
        eps = default_eps(x.dtype)
    return _fwd_op(x, weight, cos, sin, eps, interleaved)[0]


# ---------------------------------------------------------------------------
# Whole-QKV entry point.
#
# The per-tensor op leaves behind one kernel the unfused path never paid for: the
# backward of the K-major split. Inductor used to fold that concatenation into
# the epilogue of its own norm-backward and write d(qkv) directly; against an
# opaque op it cannot, so it emits a standalone gather of dq, dk and dv.
#
# Taking qkv as the differentiable input removes the split, and with it the thing
# that had to be undone. The backward kernels write dq and dk straight into the q
# and k columns of d(qkv), which costs nothing now that the destination stride is
# a runtime argument. Only dv is copied, since FMHA produces it.
#
# Opaque on purpose, unlike the per-tensor op: the input is a GEMM output and the
# outputs go to FMHA, so there are no neighbours to fuse with, and tracing in
# exposes two kernels writing disjoint columns of one fresh buffer -- a pattern
# functionalization handles badly.
# ---------------------------------------------------------------------------


@custom_op("primus_flux::qkv_norm_rope", mutates_args=())
def _qkv_fwd(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
    interleaved: bool,
    round_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, T = qkv.shape
    D = T // (3 * heads)
    # K-major: q, k, v are three contiguous blocks, each a [B, L, H, D] view with
    # a 3*H*D jump between tokens.
    view = qkv.view(B, L, 3, heads, D)
    q_in, k_in, v_in = view[:, :, 0], view[:, :, 1], view[:, :, 2]
    v = torch.empty((B, L, heads, D), device=qkv.device, dtype=qkv.dtype)
    q, q_rstd = _launch_fwd(q_in, wq, cos, sin, eps, interleaved, traceable=False, copy_x=v_in, copy_out=v)
    k, k_rstd = _launch_fwd(k_in, wk, cos, sin, eps, interleaved, traceable=False)
    return q, k, v, q_rstd, k_rstd


@_qkv_fwd.register_fake
def _(qkv, wq, wk, cos, sin, heads, eps, interleaved, round_grad):
    B, L, T = qkv.shape
    D = T // (3 * heads)
    head = torch.empty((B, L, heads, D), device=qkv.device, dtype=qkv.dtype)
    rstd = torch.empty(B * L * heads, device=qkv.device, dtype=torch.float32)
    return (head, torch.empty_like(head), torch.empty_like(head), rstd, torch.empty_like(rstd))


@custom_op("primus_flux::qkv_norm_rope_backward", mutates_args=())
def _qkv_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    heads: int,
    interleaved: bool,
    round_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, T = qkv.shape
    D = T // (3 * heads)
    d_qkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    src = qkv.view(B, L, 3, heads, D)
    dst = d_qkv.view(B, L, 3, heads, D)
    dwqp = _launch_bwd(
        dq,
        src[:, :, 0],
        wq,
        cos,
        sin,
        q_rstd,
        dst[:, :, 0],
        interleaved,
        traceable=False,
        round_grad=round_grad,
    )
    dwkp = _launch_bwd(
        dk,
        src[:, :, 1],
        wk,
        cos,
        sin,
        k_rstd,
        dst[:, :, 1],
        interleaved,
        traceable=False,
        round_grad=round_grad,
    )
    dst[:, :, 2].copy_(dv)
    return d_qkv, dwqp, dwkp


@_qkv_bwd.register_fake
def _(dq, dk, dv, qkv, wq, wk, cos, sin, q_rstd, k_rstd, heads, interleaved, round_grad):
    D = qkv.shape[-1] // (3 * heads)
    M = qkv.numel() // (3 * D)
    p = torch.empty(
        triton.cdiv(M, _mix_order_split(M, qkv.device)), D, device=qkv.device, dtype=torch.float32
    )
    return (torch.empty_like(qkv, memory_format=torch.contiguous_format), p, torch.empty_like(p))


def _qkv_setup(ctx, inputs, output):
    qkv, wq, wk, cos, sin, heads, eps, interleaved, round_grad = inputs
    # qkv rather than its slices: one saved tensor, and it is the qkv GEMM's
    # output, which the graph holds for that GEMM's own backward anyway.
    ctx.save_for_backward(qkv, wq, wk, cos, sin, output[3], output[4])
    ctx.heads = heads
    ctx.interleaved = interleaved
    ctx.round_grad = round_grad


def _qkv_backward(ctx, dq, dk, dv, _dqr, _dkr):
    qkv, wq, wk, cos, sin, q_rstd, k_rstd = ctx.saved_tensors
    # DoubleStreamBlock hands back slices of the concatenated sequence's gradient,
    # which the kernel cannot address as a flat token index.
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    d_qkv, dwqp, dwkp = _qkv_bwd(
        dq, dk, dv, qkv, wq, wk, cos, sin, q_rstd, k_rstd, ctx.heads, ctx.interleaved, ctx.round_grad
    )
    return (d_qkv, dwqp.sum(0).to(wq.dtype), dwkp.sum(0).to(wk.dtype), None, None, None, None, None, None)


_qkv_fwd.register_autograd(_qkv_backward, setup_context=_qkv_setup)


def fused_qkv_norm_rope(qkv, wq, wk, cos, sin, heads, eps=None, interleaved=True, round_grad=False):
    """Split [B, L, 3*H*D] into Q, K, V with QK norm and RoPE already applied.

    Replaces the rearrange + QKNorm + apply_rope sequence in SelfAttention and
    SingleStreamBlock. Returns packed (Q, K, V), each [B, L, H, D].

    ``round_grad`` rounds the rope gradient to bf16 before the norm backward, which is
    what the unfused DoubleStreamBlock does (it rotates after the txt/img concat);
    SelfAttention and SingleStreamBlock keep it in fp32.
    """
    if eps is None:
        eps = default_eps(qkv.dtype)
    q, k, v, _, _ = _qkv_fwd(qkv, wq, wk, cos, sin, heads, eps, interleaved, round_grad)
    return q, k, v
