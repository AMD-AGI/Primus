# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""Torch reference of the MXFP6 activation operand of DLRM_HSTU_FP8_FWD_MXFP6 (yambda_5b.gin item (92)).

Used by ops/hip/mx6_fwd.py (the first-call producer check on row slices, and the prebuild self-check on a
deterministic ramp) and by the tests. Plain torch, any device; no MX library, no build.

* ``ln_y_exact``: the LN producer's pre-cast y = fma((x - mean) * rstd, w, b) * inv_scale in fp32, from the
  producer's own mean / rstd (which the first-call check requires to be bitwise the shipped store_t's). Every
  operation is rounded as the kernel rounds it: (x - mean) and (* rstd) are fp32 ops, the fma is rounded ONCE
  (exact product in fp64; the fp64 sum is rounded to odd with a TwoSum error term, so the final RNE rounding to
  fp32 is the correctly rounded fma, 53 >= 24 + 2 bits), and (* inv_scale) is an fp32 op with
  float32(inv_scale), as the producer's ctypes launch and the Triton launcher pass it.
* ``x6_xs_reference``: the MX6-T codes and MX6-A-v1 scale bytes of y. Port of pr7 iteration 24 proto.py
  (sha256 9edc2647...): q_e2m3 / enc_e2m3 / e8m0_exp / pack_rows / quant_A / a_scale_native / lds_map /
  tile_A, unchanged except for the device argument and row chunking (also embedded in
  ops/tests/ln_store_mx6_test.py, where the producer is byte-validated against it).
* ``e4m3_sat`` / ``xt8_reference``: the e4m3 bytes the store_t writes for y (saturating at +-448), row-major
  (x8) and step-tiled (xT8 [Mp / 128, 512, 128], zero pad rows).
"""

from __future__ import annotations

from typing import Tuple

import torch

D: int = 512
TILE: int = 128
KSTEPS: int = 4
FMAX: float = 7.5
CHUNK_ROWS: int = 8192

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)


# ------------------------------------------------------------------------------------------------ the LN y
def fma_f32_exact(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fma(a, b, c) with ONE rounding (RNE) for fp32 inputs (b, c broadcastable).

    a * b is exact in fp64 (24 + 24 bits). s = fl64(p + c) is rounded to odd using the TwoSum error term,
    then rounded once to fp32: rounding to odd at >= p + 2 bits followed by RNE equals RNE of the exact value
    (Boldo-Melquiond)."""
    p = a.double() * b.double()
    c64 = c.double().expand_as(p)
    s = p + c64
    pp = s - c64
    cc = s - pp
    err = (p - pp) + (c64 - cc)
    bits = s.view(torch.int64)
    fix = (err != 0) & ((bits & 1) == 0)
    up = (err > 0) == (s > 0)  # the exact value lies further from zero than s
    step = torch.where(up, torch.ones_like(bits), -torch.ones_like(bits))
    bits = torch.where(fix, bits + step, bits)
    return bits.view(torch.float64).to(torch.float32)


def ln_y_exact(
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    inv_scale: float,
    chunk: int = CHUNK_ROWS,
) -> torch.Tensor:
    """fp32 [R, 512] y of the producer for rows x [R, 512] with their fp32 mean / rstd [R]."""
    r = int(x.shape[0])
    out = torch.empty((r, D), dtype=torch.float32, device=x.device)
    w = weight.float().view(1, D)
    b = bias.float().view(1, D)
    inv = torch.tensor(float(inv_scale), dtype=torch.float32, device=x.device)
    for r0 in range(0, r, chunk):
        r1 = min(r, r0 + chunk)
        t = (x[r0:r1].float() - mean[r0:r1].view(-1, 1)) * rstd[r0:r1].view(-1, 1)
        out[r0:r1] = fma_f32_exact(t, w, b) * inv
    return out


def e4m3_sat(y: torch.Tensor) -> torch.Tensor:
    """The store_t's e4m3 of y (v_cvt_scalef32_pk_fp8_f32 with MODE.FP16_OVFL: saturate to +-448)."""
    assert _FP8_E4M3 is not None
    return y.clamp(-448.0, 448.0).to(_FP8_E4M3)


def xt8_reference(y: torch.Tensor) -> torch.Tensor:
    """Step-tiled token-major e4m3 [T, 512, 128] of y [R, 512], zero for the pad rows R..Mp-1."""
    assert _FP8_E4M3 is not None
    r = int(y.shape[0])
    t = (r + TILE - 1) // TILE
    full = torch.zeros((t * TILE, D), dtype=torch.uint8, device=y.device)
    full[:r] = e4m3_sat(y).view(torch.uint8)
    return full.view(t, TILE, D).permute(0, 2, 1).contiguous().view(_FP8_E4M3)


# --------------------------------------------------------------- pr7 iter24 proto.py (sha256 9edc2647...)
def q_e2m3(v: torch.Tensor) -> torch.Tensor:
    """RNE onto the E2M3 grid (subnormals, max 7.5); |v| <= 7.5 expected (clamped)."""
    a = v.abs()
    step = torch.where(a < 2, torch.full_like(a, 0.125),
                       torch.where(a < 4, torch.full_like(a, 0.25), torch.full_like(a, 0.5)))
    q = torch.clamp(torch.round(a / step) * step, max=7.5)
    return torch.copysign(q, v)


def enc_e2m3(q: torch.Tensor) -> torch.Tensor:
    a = q.abs()
    s = (q < 0).to(torch.int64)
    e = torch.where(a < 1, torch.zeros_like(a), torch.floor(torch.log2(torch.clamp(a, min=1.0))) + 1)
    m = torch.where(a < 1, a / 0.125, (a / torch.exp2(e - 1) - 1.0) * 8.0)
    return (s << 5) | (e.to(torch.int64) << 3) | torch.round(m).to(torch.int64)


def e8m0_exp(amax: torch.Tensor, fmax: float = FMAX) -> torch.Tensor:
    """exact e = ceil(log2(amax / fmax)), clamped to [-127, 127]; amax 0 -> 0."""
    e = torch.ceil(torch.log2(torch.clamp(amax, min=1e-38) / fmax))
    for _ in range(2):  # exact correction with power-of-two arithmetic
        e = torch.where(amax / torch.exp2(e) > fmax, e + 1, e)
        e = torch.where(amax / torch.exp2(e - 1) <= fmax, e - 1, e)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    return torch.clamp(e, -127, 127)


def pack_rows(codes: torch.Tensor) -> torch.Tensor:
    """codes int64 [R, 16, 32] (logical K order) -> MX6 v1 bytes [R, 384] (head / tail per 128-K step)."""
    r = codes.shape[0]
    c = codes.view(r, 16, 4, 8)
    sh = torch.arange(8, device=codes.device, dtype=torch.int64) * 6
    v = (c << sh).sum(-1)  # [R, 16, 4] 48-bit groups (disjoint fields)
    by = ((v.unsqueeze(-1) >> (torch.arange(6, device=codes.device, dtype=torch.int64) * 8)) & 255).to(torch.uint8)
    blk = by.view(r, 4, 4, 24)  # [R, S, q, 24]
    head = blk[..., :16].reshape(r, 4, 64)
    tail = blk[..., 16:].reshape(r, 4, 32)
    return torch.cat([head, tail], -1).reshape(r, 384).contiguous()


def quant_a(y: torch.Tensor, chunk: int = CHUNK_ROWS) -> Tuple[torch.Tensor, torch.Tensor]:
    """y fp32 [R, 512] -> (packed [R, 384] uint8, e [R, 16] int32): proto.quant_A without the dequant."""
    r = int(y.shape[0])
    packed = torch.empty((r, 384), dtype=torch.uint8, device=y.device)
    ee = torch.empty((r, 16), dtype=torch.int32, device=y.device)
    for r0 in range(0, r, chunk):
        xb = y[r0:r0 + chunk].reshape(-1, 16, 32)
        e = e8m0_exp(xb.abs().amax(-1))
        q = q_e2m3(torch.clamp(xb / torch.exp2(e).unsqueeze(-1), -7.5, 7.5))
        packed[r0:r0 + chunk] = pack_rows(enc_e2m3(q))
        ee[r0:r0 + chunk] = e.to(torch.int32)
    return packed, ee


def a_scale_native(ee: torch.Tensor) -> torch.Tensor:
    """e [R, 16] -> MX6-A-v1 bytes [T, 4, 512]: byte [T, S, (q*16 + r)*8 + m] = row 128T+16m+r, block 4S+q;
    pad rows 127 (2^0)."""
    r = ee.shape[0]
    t = (r + TILE - 1) // TILE
    full = torch.zeros((t * TILE, 16), dtype=torch.int32, device=ee.device)
    full[:r] = ee
    b = (full + 127).to(torch.uint8).view(t, 8, 16, 4, 4)  # [T, m, r, S, q]
    return b.permute(0, 3, 4, 2, 1).contiguous().view(t, KSTEPS, 512)  # [T, S, q, r, m]


def lds_map(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """slot image byte d (0..12287) -> (slot row R, byte within the 96-B K-step segment)."""
    d = torch.arange(12288, device=device, dtype=torch.int64)
    b = d % 16
    lane = (d // 16) % 64
    w = (d // 1024) % 4
    i = d // 4096
    j = 4 * i + w
    head = j < 8
    rh = 16 * j + lane // 4
    qh = (lane & 3) ^ ((rh >> 2) & 3)
    rt = 32 * (j - 8) + lane // 2
    pt = (lane & 1) ^ ((rt >> 3) & 1)
    rr = torch.where(head, rh, rt)
    cb = torch.where(head, 16 * qh + b, 64 + 16 * pt + b)
    return rr, cb


def tile_a(packed: torch.Tensor) -> torch.Tensor:
    """packed [R, 384] -> MX6-T [T, 4, 12288] (pad rows: zero codes)."""
    rr, cb = lds_map(packed.device)
    m = packed.shape[0]
    t = (m + TILE - 1) // TILE
    ap = torch.zeros((t * TILE, 384), dtype=torch.uint8, device=packed.device)
    ap[:m] = packed
    flat = ap.view(t, TILE * 384)
    return torch.stack([flat[:, rr * 384 + s * 96 + cb] for s in range(KSTEPS)], 1).contiguous()


def x6_xs_reference(y: torch.Tensor, chunk: int = CHUNK_ROWS) -> Tuple[torch.Tensor, torch.Tensor]:
    """(x6 [T, 4, 12288], xs [T, 4, 512]) of y [R, 512]: the bytes the producer must write for these rows
    when row 0 of y is a 128-row tile boundary."""
    packed, ee = quant_a(y, chunk=chunk)
    return tile_a(packed), a_scale_native(ee)
