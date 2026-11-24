###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8-block tile Adam kernel (``fused_adam_clip_fp8blk``) byte tests on gfx1250.

* The tile kernel's master / moments / BF16 weight equal the chunk Adam kernel (``fused_adam_clip``) bit for bit.
* Its FP8 outputs (W, W^T, UE8M0 scales and exponent bytes) equal ``quant_fp8_blockwise_for_weight_impl`` of the BF16
  weight.
* Skipping the BF16 store (null BF16 address) changes nothing else, and the stored BF16 equals RNE(master).
* In MegaMoE (MX-plane) mode it writes the ``FP8BlockWeightCache`` bytes of ``quant_w_128x128``, including
  exponent-edge tiles.
* The hardware FP32->BF16 convert used by the kernel matches c10's round-to-nearest-even for all 2^32 inputs
  (NaN mapped to 0x7FC0).

The kernel source is selected by ``PRIMUS_FP8BLK_ADAM_WAVEDENSE=1`` / ``PRIMUS_FP8BLK_ADAM_HWCVT=1`` (the latest
kernel); the checks are against the chunk kernel and the quant reference, so they hold for any selection.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
CHUNK = 65536
# (lr, beta1, beta2, eps, step, adamw, bias_correction, weight_decay)
HYPER = (3e-4, 0.9, 0.95, 1e-8, 7, 1, 1, 0.1)
# dense FP8-block weights: aligned and odd shapes (edge tiles, batched)
SHAPES = [(256, 384), (300, 200), (2, 384, 136), (576, 512)]


@pytest.fixture(scope="module")
def adam_ext():
    mp = pytest.MonkeyPatch()
    mp.setenv("PRIMUS_FUSED_ADAM_CLIP", "1")
    mp.setenv("PRIMUS_FP8BLK_ADAM_WAVEDENSE", "1")
    mp.setenv("PRIMUS_FP8BLK_ADAM_HWCVT", "1")
    from primus.backends.megatron.patches.te_patches import fused_adam_clip_patches as P

    try:
        yield P, P._load_extension()
    finally:
        mp.undo()


class _W:
    """One dense FP8-block weight: optimizer tensors (BF16 grad / moments) and output buffers."""

    def __init__(self, shape, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        lead, (n, k) = tuple(shape[:-2]), shape[-2:]
        self.B = 1
        for d in lead:
            self.B *= d
        self.N, self.K = n, k
        self.tn, self.tk = -(-n // 128), -(-k // 128)
        self.master = torch.randn(shape, device=DEV, generator=g) * 0.02
        self.grad = (torch.randn(shape, device=DEV, generator=g) * 1e-3).bfloat16()
        self.m = (torch.randn(shape, device=DEV, generator=g) * 1e-4).bfloat16()
        self.v = (torch.rand(shape, device=DEV, generator=g) * 1e-6).bfloat16()
        self.bf16 = torch.randn(shape, device=DEV, generator=g).bfloat16()  # stale on purpose
        u8 = dict(device=DEV, dtype=torch.uint8)
        self.q = torch.full(lead + (n, k), 0x55, **u8)
        self.qT = torch.full(lead + (k, n), 0x55, **u8)
        self.scale_inv = torch.full(lead + (self.tn, self.tk), -1.0, device=DEV)
        self.s = torch.full(lead + (self.tn, self.tk), 0x33, **u8)
        self.sT = torch.full(lead + (self.tk, self.tn), 0x33, **u8)

    def clone(self):
        c = _W.__new__(_W)
        c.__dict__.update(
            {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        )
        return c


def _tile_meta(ws, skip_bf16):
    addr, dims, t2t, toff, total = [], [], [], [], 0
    for i, w in enumerate(ws):
        sl = [w.grad, w.master, w.m, w.v, None if skip_bf16 else w.bf16, w.q, w.qT, w.scale_inv, w.s, w.sT]
        addr += [0 if t is None else t.data_ptr() for t in sl]
        dims += [w.N, w.K, w.tn, w.tk]
        toff.append(total)
        t2t += [i] * (w.B * w.tn * w.tk)
        total += w.B * w.tn * w.tk
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=DEV)
    return T(addr, torch.int64), T(dims, torch.int64), T(t2t, torch.int32), T(toff, torch.int64), total


def _chunk_meta(ws):
    addr, sizes, b2t, coff, total = [], [], [], [], 0
    for i, w in enumerate(ws):
        addr += [w.grad.data_ptr(), w.master.data_ptr(), w.m.data_ptr(), w.v.data_ptr(), w.bf16.data_ptr()]
        n = w.master.numel()
        sizes.append(n)
        coff.append(total)
        c = -(-n // CHUNK)
        b2t += [i] * c
        total += c
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=DEV)
    return T(addr, torch.int64), T(sizes, torch.int64), T(b2t, torch.int32), T(coff, torch.int32), total


def _run_chunk(ext, ws, norm):
    a, s, b2t, coff, total = _chunk_meta(ws)
    ext.fused_adam_clip(a, s, b2t, coff, total, CHUNK, *HYPER, 1, 1, norm, 1.0)


def _run_tile(ext, ws, norm, skip_bf16=False):
    ext.fused_adam_clip_fp8blk(*_tile_meta(ws, skip_bf16), *HYPER, 1, 1, norm, 1.0)


def test_tile_adam_matches_chunk_and_quant_reference(adam_ext, monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
    pytest.importorskip("primus_turbo")
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        quant_fp8_blockwise_for_weight_impl,
    )

    _, ext = adam_ext
    norm = torch.tensor([2.5], device=DEV)  # clip coefficient 0.4
    A = [_W(s, 10 + i) for i, s in enumerate(SHAPES)]
    B = [w.clone() for w in A]
    _run_chunk(ext, A, norm)
    _run_tile(ext, B, norm)
    torch.cuda.synchronize()
    for sh, a, b in zip(SHAPES, A, B):
        for name in ("master", "m", "v", "bf16"):
            assert bits_equal(getattr(a, name), getattr(b, name)), f"{name} {sh}"
        q, s = quant_fp8_blockwise_for_weight_impl(a.bf16.contiguous(), torch.float8_e4m3fn, block_size=128)
        e = ((s.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        assert torch.equal(b.q, q.view(torch.uint8)), f"W {sh}"
        assert torch.equal(b.qT, q.view(torch.uint8).transpose(-1, -2).contiguous()), f"W^T {sh}"
        assert torch.equal(b.scale_inv.view(torch.int32), s.view(torch.int32)), f"scale {sh}"
        assert torch.equal(b.s, e) and torch.equal(b.sT, e.transpose(-1, -2).contiguous()), f"e8m0 {sh}"


def test_skip_bf16_store(adam_ext):
    """A null BF16 address leaves everything else byte-identical; the stored BF16 equals RNE(master)."""
    _, ext = adam_ext
    gns = [torch.tensor([0.5], device=DEV), torch.tensor([3.0], device=DEV)]
    base = [_W(s, 40 + i) for i, s in enumerate(SHAPES)]
    for norm in gns:
        a = [w.clone() for w in base]
        b = [w.clone() for w in base]
        _run_tile(ext, a, norm, skip_bf16=False)
        _run_tile(ext, b, norm, skip_bf16=True)
        torch.cuda.synchronize()
        for x, y, w0 in zip(a, b, base):
            for k in ("grad", "master", "m", "v", "q", "qT", "s", "sT", "scale_inv"):
                assert bits_equal(getattr(x, k), getattr(y, k)), k
            assert bits_equal(y.bf16, w0.bf16), "skipped BF16 must be untouched"
            assert bits_equal(x.bf16, y.master.to(torch.bfloat16)), "stored BF16 != RNE(master)"

    # chunk kernel (embedding-sized tensors): same with a null BF16 address
    sizes = [CHUNK * 3 + 5, 7, CHUNK * 2]
    ws = [_W((1, n), 70 + i) for i, n in enumerate(sizes)]
    a = [w.clone() for w in ws]
    b = [w.clone() for w in ws]
    ma, mb = list(_chunk_meta(a)), list(_chunk_meta(b))
    addr = mb[0].view(-1, 5)
    addr[:, 4] = 0
    for m in (ma, mb):
        ext.fused_adam_clip(*m[:5], CHUNK, *HYPER, 1, 1, gns[0], 1.0)
    torch.cuda.synchronize()
    for x, y in zip(a, b):
        for k in ("master", "m", "v"):
            assert bits_equal(getattr(x, k), getattr(y, k)), k
        assert bits_equal(x.bf16, y.master.to(torch.bfloat16))


@pytest.mark.parametrize(
    "moment_dtype", [torch.float32, torch.bfloat16], ids=["fp32_moments", "bf16_moments"]
)
@pytest.mark.parametrize("lr, edges", [(1e-3, False), (0.0, True)], ids=["update", "exponent_edges"])
def test_megamoe_cache_bytes(adam_ext, lr, edges, moment_dtype):
    """MX-plane mode writes the FP8BlockWeightCache that ``quant_w_128x128`` of the updated weight produces."""
    P, ext = adam_ext
    pytest.importorskip("primus_turbo")
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import FP8BlockWeightCache
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
        quant_w_128x128,
    )

    shapes = [(2, 256, 384), (3, 128, 256)]
    torch.manual_seed(0)
    a_lists, b_lists, a_wb, b_wb, caches = [[], [], [], []], [[], [], [], []], {}, {}, []
    for shape in shapes:
        E, N, K = shape
        master = torch.randn(shape, device=DEV) * 0.02
        if (
            edges
        ):  # lr = 0 keeps the weights: all-zero, tiny, 448*2^k exact, mantissa-boundary, denormal, large tiles
            m = master.view(E, N // 128, 128, K // 128, 128)
            m[0, 0, :, 0, :] = 0
            if K // 128 > 1:
                m[0, 0, :, 1, :] *= 1e-5
            if N // 128 > 1:
                m[0, 1, :, 0, :] = 0
                m[0, 1, 5, 0, 7] = 448.0 * 2**-10
                if K // 128 > 1:
                    m[0, 1, :, 1, :] *= 0.1
                    m[0, 1, 3, 1, 3] = 1.75 * 2**-6
            if E > 1:
                m[1, 0, :, 0, :] *= 1e-30
                m[1, 0, 0, 0, 0] = -1.765625 * 2**-6
                if K // 128 > 1:
                    m[1, 0, :, 1, :] *= 300.0
        grad = (torch.randn(shape, device=DEV) * 1e-3).to(torch.bfloat16)
        ea = (torch.randn(shape, device=DEV) * 1e-3).to(moment_dtype)
        eas = (torch.rand(shape, device=DEV) * 1e-6).to(moment_dtype)
        bf = master.to(torch.bfloat16)
        for lists, wb in ((a_lists, a_wb), (b_lists, b_wb)):
            t = [grad.clone(), master.clone(), ea.clone(), eas.clone()]
            for j in range(4):
                lists[j].append(t[j])
            wb[t[1].data_ptr()] = bf.clone()
    fp8 = {}
    for i in range(len(shapes)):
        mp = b_wb[b_lists[1][i].data_ptr()]
        c = FP8BlockWeightCache(mp)
        caches.append(c)
        fp8[b_lists[1][i].data_ptr()] = (mp, P._MegaMoEAdamTarget(c))
    gn = torch.tensor([0.5], device=DEV)
    hyper = (float(lr), 0.9, 0.95, 1e-8, 7, 1, 1, 0.1)
    addr, sizes, b2t, coffs, nch, md, gd = P._build_metadata(CHUNK, a_lists, a_wb)
    f_addr, f_dims, f_t2t, f_toff, f_tiles, f_mdt, f_gdt, _ = P._build_fp8blk_metadata(b_lists, fp8, b_wb)
    ext.fused_adam_clip(addr, sizes, b2t, coffs, nch, CHUNK, *hyper, md, gd, gn, 1.0)
    ext.fused_adam_clip_fp8blk(f_addr, f_dims, f_t2t, f_toff, f_tiles, *hyper, f_mdt, f_gdt, gn, 1.0)
    torch.cuda.synchronize()
    for i in range(len(shapes)):
        for j in range(4):
            assert bits_equal(a_lists[j][i], b_lists[j][i]), f"state {j} tensor {i}"
        wa, wb_ = a_wb[a_lists[1][i].data_ptr()], b_wb[b_lists[1][i].data_ptr()]
        assert bits_equal(wa, wb_), f"bf16 weight {i}"
        c = caches[i]
        ref = quant_w_128x128(wa, torch.float8_e4m3fn, transposed=True)
        for n, x, y in zip(("q", "s", "qT", "sT"), (c.q, c.s, c.qT, c.sT), ref):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), f"cache {n} tensor {i}"


_CVT_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/util/BFloat16.h>
namespace {
__global__ void k_cmp(unsigned long long* cnt, uint32_t* first) {
  const uint64_t base = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 64;
  unsigned long long raw = 0, fixed = 0;
  for (int j = 0; j < 64; ++j) {
    const uint32_t u = static_cast<uint32_t>(base + j);
    const float f = __uint_as_float(u);
    const uint16_t s = c10::detail::round_to_nearest_even(f);
    const __bf16 h = static_cast<__bf16>(f);
    const uint16_t b = __builtin_bit_cast(uint16_t, h);
    const uint16_t x = (f != f) ? UINT16_C(0x7FC0) : b;
    raw += (b != s);
    if (x != s) { ++fixed; atomicMin(first, u); }
  }
  if (raw) atomicAdd(cnt, raw);
  if (fixed) atomicAdd(cnt + 1, fixed);
}
}  // namespace
void run(torch::Tensor cnt, torch::Tensor first) {
  const int threads = 256;
  const int64_t blocks = (INT64_C(1) << 32) / 64 / threads;
  k_cmp<<<static_cast<int>(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long*>(cnt.data_ptr()), reinterpret_cast<uint32_t*>(first.data_ptr()));
}
"""


def test_hardware_bf16_convert_matches_rne_exhaustive():
    """(__bf16)x on the GPU, with NaN mapped to 0x7FC0, equals c10's RNE for every FP32 bit pattern."""
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        name="primus_test_bf16cvt_exhaustive",
        cpp_sources="void run(torch::Tensor cnt, torch::Tensor first);",
        cuda_sources=_CVT_SRC,
        functions=["run"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
    )
    cnt = torch.zeros(2, dtype=torch.int64, device=DEV)
    first = torch.full((1,), -1, dtype=torch.int32, device=DEV)
    ext.run(cnt, first)
    torch.cuda.synchronize()
    assert cnt[1].item() == 0, f"mismatches with NaN fixup, first at bits {first.item() & 0xFFFFFFFF:#x}"
