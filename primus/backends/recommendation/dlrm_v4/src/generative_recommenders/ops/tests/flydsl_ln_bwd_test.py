#!/usr/bin/env python3

# pyre-strict

"""The FlyDSL weighted LayerNorm backward (ops/flydsl/ln_bwd.py) against the
shipped Triton path it replaces (item (14)'s v2 kernel with the fast sigmoid)
and an fp64 reference, for the four variants the training step runs plus PR
#7's bf16-weight MLP cases; the DLRM_HSTU_FP8_WLN_BWD_FLYDSL dispatch, its
arming lines and its failure latch in ops/triton/triton_layer_norm.py; and the
startup prebuild in dlrm_v4/train/blas_warmup.py. gfx950 only (the kernel's
target); skipped elsewhere."""

import unittest
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import torch

# (D, is_swish, residual, sum_dx, weight dtype)
VARIANTS = (
    (512, False, True, False, torch.bfloat16),  # STU norms (weights cast to bf16)
    (512, False, False, False, torch.bfloat16),  # targets-only STU norm, item (20e)
    (512, False, False, True, torch.float32),  # MLP output norms (fp32 Parameter)
    (256, True, False, True, torch.float32),  # MLP SwishLayerNorms (fp32 Parameter)
    (512, False, False, True, torch.bfloat16),  # PR #7's bf16-weight MLP cases
    (256, True, False, True, torch.bfloat16),
)
M = 4099  # not a multiple of any block's rows: exercises the tail
# The flyc.jit cache keys tensors by type, not by shape, so the kernel built at
# the first M serves every later M (the startup prebuild builds at 1024). The
# last M takes several passes of the persistent grid.
M_SEQUENCE = (1024, 4099, 100_003)
EPS = 1e-6
LOGGER = "generative_recommenders.ops.triton.triton_layer_norm"


def _flydsl_unavailable() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return True, "no GPU"
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx950"):
        return True, f"FlyDSL LN backward targets gfx950, not {arch or 'this device'}"
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return True, "flydsl is not installed"
    return False, ""


def _inputs(
    D: int, residual: bool, w_dtype: torch.dtype, m: int = M
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(20261007)
    x = (torch.randn(m, D, generator=g, device="cuda") * 2.0 + 0.5).to(torch.bfloat16)
    xf = x.float()
    t = {
        "x": x,
        "mean": xf.mean(1),
        "rstd": torch.rsqrt(xf.var(1, unbiased=False) + EPS),
        "dy": torch.randn(m, D, generator=g, device="cuda").to(torch.bfloat16),
        "w": (1.0 + 0.1 * torch.randn(D, generator=g, device="cuda")).to(w_dtype),
        "b": (0.1 * torch.randn(D, generator=g, device="cuda")).to(w_dtype),
    }
    if residual:
        t["res"] = torch.randn(m, D, generator=g, device="cuda").to(torch.bfloat16)
    return t


def _reference(t: Dict[str, torch.Tensor], is_swish: bool) -> Dict[str, torch.Tensor]:
    """fp64: dx (residual added unrounded), dweight, dbias, dxsum."""
    x, dy = t["x"].double(), t["dy"].double()
    w, b = t["w"].double(), t["b"].double()
    rstd = t["rstd"].double()[:, None]
    xhat = (x - t["mean"].double()[:, None]) * rstd
    if is_swish:
        sg = torch.sigmoid(xhat * w + b)
        tt = dy * x * sg * (1 - sg)
        g, dw, db = w * tt, tt * xhat, tt
    else:
        g, dw, db = w * dy, dy * xhat, dy
    dx = (g - (xhat * (xhat * g).mean(1, keepdim=True) + g.mean(1, keepdim=True))) * rstd
    if is_swish:
        dx = dx + dy * sg
    if "res" in t:
        dx = dx + t["res"].double()
    return {"dx": dx, "dweight": dw.sum(0), "dbias": db.sum(0), "dxsum": dx.sum(0)}


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).norm() / ref.norm()).item()


def _bits_equal_up_to_zero_sign(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality after mapping -0.0 to +0.0 (adding a +0 residual turns
    an exact -0.0 dx into +0.0: the sign-of-zero class item (20e) documents)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    itype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    za = torch.where(a == 0, torch.zeros_like(a), a)
    zb = torch.where(b == 0, torch.zeros_like(b), b)
    return torch.equal(za.view(itype), zb.view(itype))


def _flydsl(t: Dict[str, torch.Tensor], is_swish: bool, sum_dx: bool,
            residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
    from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd

    out = ln_bwd(t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"],
                 residual if residual is not None else t.get("res"),
                 is_swish=is_swish, sum_dx=sum_dx)
    torch.cuda.synchronize()
    return out


def _triton(tln: Any, t: Dict[str, torch.Tensor], D: int, is_swish: bool,
            sum_dx: bool) -> Tuple[torch.Tensor, ...]:
    """The production entry point; with the FlyDSL switch off (setUp) this is
    the shipped Triton path."""
    if sum_dx:
        out = tln.triton_weighted_layer_norm_bwd_dxsum(
            dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
            rstd=t["rstd"], eps=EPS, BLOCK_D=D, is_swish=is_swish)
    else:
        assert not is_swish
        out = tln.triton_weighted_layer_norm_bwd(
            dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
            rstd=t["rstd"], learnable=True, eps=EPS, BLOCK_D=D,
            residual=t.get("res"))
    torch.cuda.synchronize()
    return out


_SKIP, _WHY = _flydsl_unavailable()


@unittest.skipIf(_SKIP, _WHY)
class FlydslLnBwdTest(unittest.TestCase):
    def setUp(self) -> None:
        from generative_recommenders.ops.triton import triton_layer_norm as tln

        self.tln = tln
        # The module reads DLRM_HSTU_FP8_WLN_BWD_V2, ..._V2_FAST_SIGMOID and
        # DLRM_HSTU_FP8_WLN_BWD_FLYDSL once into these caches; force the
        # shipped Triton path (v2 + fast sigmoid) with FlyDSL off, and restore.
        self._saved = {
            name: getattr(tln, name)
            for name in ("_WLN_BWD_V2_ENABLED", "_WLN_BWD_V2_FAST_SIGMOID",
                         "_WLN_BWD_FLYDSL_ENABLED", "_WLN_BWD_FLYDSL_BROKEN",
                         "_WLN_BWD_FLYDSL_MSCALE_ENABLED")
        }
        self._saved_logged = set(tln._WLN_BWD_FLYDSL_LOGGED)
        tln._WLN_BWD_V2_ENABLED = True
        tln._WLN_BWD_V2_FAST_SIGMOID = True
        tln._WLN_BWD_FLYDSL_ENABLED = False
        tln._WLN_BWD_FLYDSL_BROKEN = None
        tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = False
        tln._WLN_BWD_FLYDSL_LOGGED.clear()

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(self.tln, name, value)
        self.tln._WLN_BWD_FLYDSL_LOGGED.clear()
        self.tln._WLN_BWD_FLYDSL_LOGGED.update(self._saved_logged)

    def _check_against_triton_and_fp64(self, D: int, is_swish: bool, residual: bool,
                                       sum_dx: bool, w_dtype: torch.dtype,
                                       m: int) -> None:
        t = _inputs(D, residual, w_dtype, m)
        ref = _reference(t, is_swish)
        tri = _triton(self.tln, t, D, is_swish, sum_dx)
        fly = _flydsl(t, is_swish, sum_dx)
        self.assertEqual(len(fly), len(tri))
        for f, r in zip(fly, tri):  # the output contract of the Triton path
            self.assertEqual((f.dtype, f.shape), (r.dtype, r.shape))
        torch.testing.assert_close(fly[0], tri[0], atol=1e-2, rtol=1e-2)
        names = ("dx", "dweight", "dbias", "dxsum")[: len(fly)]
        for name, f, r in zip(names, fly, tri):
            # as accurate as the shipped Triton path against fp64, to within
            # 10% of its error
            e_f, e_t = _rel(f, ref[name]), _rel(r, ref[name])
            self.assertLessEqual(e_f, 1.1 * e_t + 1e-6, f"{name}: {e_f} vs {e_t}")

    def test_kernel_against_triton(self) -> None:
        for D, is_swish, residual, sum_dx, w_dtype in VARIANTS:
            with self.subTest(D=D, is_swish=is_swish, residual=residual,
                              sum_dx=sum_dx, w_dtype=w_dtype):
                self._check_against_triton_and_fp64(
                    D, is_swish, residual, sum_dx, w_dtype, M)

    def test_jit_cache_serves_any_m(self) -> None:
        # One process, several M per specialization (shape-agnostic jit cache).
        for D, is_swish, residual, sum_dx, w_dtype in VARIANTS[:4]:
            for m in M_SEQUENCE:
                with self.subTest(D=D, is_swish=is_swish, residual=residual,
                                  sum_dx=sum_dx, w_dtype=w_dtype, m=m):
                    self._check_against_triton_and_fp64(
                        D, is_swish, residual, sum_dx, w_dtype, m)
                torch.cuda.empty_cache()

    def test_plain_equals_residual_variant(self) -> None:
        # The plain D=512 variant uses the residual variant's operation order,
        # grid and rows per block: with a zero residual the two agree bitwise
        # (dx up to the sign of an exact zero), and item (20e)'s plain call +
        # index_add of the candidate rows equals the dense residual call.
        for m in (M, 100_003):
            with self.subTest(m=m):
                t = _inputs(512, False, torch.bfloat16, m)
                plain = _flydsl(t, False, False)
                zero = _flydsl(t, False, False, residual=torch.zeros_like(t["x"]))
                self.assertTrue(_bits_equal_up_to_zero_sign(plain[0], zero[0]), "dx")
                self.assertTrue(torch.equal(plain[1].view(torch.int16),
                                            zero[1].view(torch.int16)), "dweight")
                self.assertTrue(torch.equal(plain[2].view(torch.int16),
                                            zero[2].view(torch.int16)), "dbias")

                g = torch.Generator(device="cuda").manual_seed(7)
                rows = torch.arange(37, m, 53, device="cuda")
                res_rows = torch.randn(rows.numel(), 512, generator=g,
                                       device="cuda").to(torch.bfloat16)
                dense = torch.zeros_like(t["x"])
                dense[rows] = res_rows
                fused = _flydsl(t, False, False, residual=dense)
                sparse = plain[0].clone()
                sparse.index_add_(0, rows, res_rows)
                torch.cuda.synchronize()
                self.assertTrue(_bits_equal_up_to_zero_sign(sparse, fused[0]),
                                "dx: plain + index_add vs dense residual")
                self.assertTrue(torch.equal(plain[1].view(torch.int16),
                                            fused[1].view(torch.int16)), "dweight")
                self.assertTrue(torch.equal(plain[2].view(torch.int16),
                                            fused[2].view(torch.int16)), "dbias")

    def test_switch_routes_to_flydsl(self) -> None:
        from torch.profiler import profile, ProfilerActivity

        tln = self.tln
        for residual, fly_name in ((True, "ln_bwd_flydsl_d512_res"),
                                   (False, "ln_bwd_flydsl_d512")):
            t = _inputs(512, residual, torch.bfloat16)
            for enabled in (True, False):
                with self.subTest(residual=residual, enabled=enabled):
                    tln._WLN_BWD_FLYDSL_ENABLED = enabled
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        _triton(tln, t, 512, False, False)
                    names = {e.name for e in prof.events()
                             if e.device_type == torch.autograd.DeviceType.CUDA}
                    fly = {n for n in names if "ln_bwd_flydsl_d512" in n
                           and "_fin_" not in n}
                    tri = {n for n in names if "_weighted_layer_norm_bwd_dx" in n}
                    if enabled:
                        self.assertTrue(fly and all(
                            ("_res" in n) == residual for n in fly), sorted(names))
                        self.assertFalse(tri, sorted(names))
                    else:
                        self.assertFalse(fly, sorted(names))
                        self.assertTrue(tri, sorted(names))
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)

    def test_arming_lines(self) -> None:
        # The exact text the DLRM_HSTU_FP8_WLN_BWD_FLYDSL gin item's ARMED list names, once per key.
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        tln = self.tln
        tln._WLN_BWD_FLYDSL_ENABLED = True
        expected = []
        # the production kernel options (XCD run length, load/store cache
        # policy), whatever the env kill switches of this process say
        opts = {(512, False, True, False): "remap=32 cache=3/3",
                (512, False, False, False): "remap=16 cache=3/3",
                (512, False, False, True): "remap=16 cache=3/3",
                (256, True, False, True): "remap=16 cache=3/3"}
        with mock.patch.object(lnmod, "XCD_REMAP", True), \
                mock.patch.object(lnmod, "STREAM_CACHE_ON", True), \
                self.assertLogs(LOGGER, level="WARNING") as logs:
            for D, is_swish, residual, sum_dx, w_dtype in VARIANTS[:4]:
                t = _inputs(D, residual, w_dtype)
                for _ in range(2):  # the second call logs nothing
                    _triton(tln, t, D, is_swish, sum_dx)
                expected.append(
                    f"[ln-bwd] flydsl: D={D} swish={int(is_swish)} "
                    f"residual={int(residual)} dxsum={int(sum_dx)} weight={w_dtype} "
                    f"{opts[(D, is_swish, residual, sum_dx)]}")
            # the fp32-input item-MLP LayerNorm: the one expected Triton line
            t = _inputs(512, False, torch.float32)
            t = {k: v.float() for k, v in t.items()}
            for _ in range(2):
                _triton(tln, t, 512, False, False)
            expected.append(
                "[ln-bwd] triton: D=512 swish=0 residual=0 dxsum=0 "
                "(dtype torch.float32/torch.float32) weight=torch.float32")
        got = [r.getMessage() for r in logs.records if "[ln-bwd]" in r.getMessage()]
        self.assertEqual(got, expected)
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)

    def test_v21_options_bitwise_vs_v2(self) -> None:
        # v2.1's XCD block order and cache policies (and the split dx-store
        # policy) change no output bit: dx, dW, dB, the dx sum AND every fp32
        # partial row equal v2's kernel (cache=0, remap=0). Outputs and PART
        # start as different NaN-payload sentinels per arm, so an unwritten
        # element cannot compare equal.
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        arms = (dict(), dict(cache=3, remap=16), dict(cache=3, remap=32, store_cache=0),
                dict(cache=2, remap=8, store_cache=3), dict(cache=0, remap=64))
        for D, is_swish, residual, sum_dx, w_dtype in VARIANTS[:4]:
            for m in (4099, 100_003):
                t = _inputs(D, residual, w_dtype, m)
                waves, _, tile_mult, _ = lnmod.DEFAULT_CFG[(D, is_swish, residual, sum_dx)]
                G = lnmod.grid_for(m, t["x"].device, waves, tile_mult)

                def filled(shape, dtype, sentinel):
                    # a NaN payload per arm (bf16: the low 16 bits' quiet-NaN form)
                    if dtype == torch.bfloat16:
                        t_ = torch.empty(shape, dtype=torch.int16, device="cuda")
                        t_.fill_(0x7FC0 | (sentinel & 0x3F))
                        return t_.view(torch.bfloat16)
                    t_ = torch.empty(shape, dtype=torch.int32, device="cuda")
                    t_.fill_(sentinel)
                    return t_.view(torch.float32)

                def run(kw, sentinel):
                    part = filled((3 * G * D,), torch.float32, sentinel)
                    out = {"part": part,
                           "dx": filled(tuple(t["x"].shape), torch.bfloat16, sentinel),
                           "dw": filled((D,), w_dtype, sentinel),
                           "db": filled((D,), w_dtype, sentinel)}
                    if sum_dx:
                        out["outs"] = filled((D,), torch.float32, sentinel)
                    res = lnmod.ln_bwd(t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"],
                                       t.get("res"), is_swish=is_swish, sum_dx=sum_dx,
                                       out=out, **kw)
                    torch.cuda.synchronize()
                    nacc = 3 if sum_dx else 2
                    return res, part[: nacc * G * D].clone()

                ref, ref_part = run(dict(cache=0, remap=0, store_cache=0), 0x7FC00001)
                for i, kw in enumerate(arms):
                    with self.subTest(D=D, swish=is_swish, residual=residual, dxsum=sum_dx,
                                      m=m, kw=kw):
                        got, got_part = run(kw, 0x7FC00002 + i)
                        for name, a, b in zip(("dx", "dw", "db", "dxsum"), got, ref):
                            itype = {torch.bfloat16: torch.int16,
                                     torch.float32: torch.int32}[a.dtype]
                            self.assertTrue(torch.equal(a.view(itype), b.view(itype)), name)
                        self.assertTrue(torch.equal(got_part.view(torch.int32),
                                                    ref_part.view(torch.int32)), "PART")

    def test_v21_kill_switches_and_options(self) -> None:
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        # the production options, whatever the env kill switches of this process say
        with mock.patch.object(lnmod, "XCD_REMAP", True), \
                mock.patch.object(lnmod, "STREAM_CACHE_ON", True):
            self.assertEqual(lnmod.kernel_options(512, False, True, False), (32, 3, 3))
            self.assertEqual(lnmod.kernel_options(256, True, False, True), (16, 3, 3))
            self.assertEqual(lnmod.kernel_options(512, False, True, False, store_cache=0),
                             (32, 3, 0))
        with self.assertRaises(TypeError):   # a bool is not a run length
            lnmod.kernel_options(512, False, True, False, remap=True)
        import numpy as np

        remap = lnmod.kernel_options(512, False, True, False, remap=np.int64(16))[0]
        self.assertIs(type(remap), int)   # a numpy int cannot reach the jit cache key
        with mock.patch.object(lnmod, "XCD_REMAP", False), \
                mock.patch.object(lnmod, "STREAM_CACHE_ON", False):
            self.assertEqual(lnmod.kernel_options(512, False, True, False), (0, 0, 0))
            self.assertEqual(lnmod.build_ln_bwd(512, False, True, False, False, 4, 4, 0,
                                                True, 0, 0).kernel_name,
                             "ln_bwd_flydsl_d512_res")
        self.assertEqual(lnmod.build_ln_bwd(512, False, True, False, False, 4, 4, 3,
                                            True, 32, 3).kernel_name,
                         "ln_bwd_flydsl_d512_res_x32_c3")
        self.assertEqual(lnmod.build_ln_bwd(512, False, True, False, False, 4, 4, 3,
                                            True, 32, 0).kernel_name,
                         "ln_bwd_flydsl_d512_res_x32_c3_s0")

    def test_failure_latches_to_triton(self) -> None:
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        tln = self.tln
        t = _inputs(512, True, torch.bfloat16)
        want = _triton(tln, t, 512, False, False)  # FlyDSL off: shipped Triton
        tln._WLN_BWD_FLYDSL_ENABLED = True
        boom = RuntimeError("synthetic FlyDSL failure\nsecond line")
        with mock.patch.object(lnmod, "ln_bwd", side_effect=boom) as fake:
            with self.assertLogs(LOGGER, level="WARNING") as logs:
                first = _triton(tln, t, 512, False, False)
                second = _triton(tln, t, 512, False, False)
        self.assertEqual(fake.call_count, 1)  # latched: tried once
        disabled = [r for r in logs.records if "flydsl DISABLED" in r.getMessage()]
        self.assertEqual(len(disabled), 1)
        self.assertEqual(
            tln._WLN_BWD_FLYDSL_BROKEN,
            "dispatch: RuntimeError: synthetic FlyDSL failure")
        for out in (first, second):
            for a, b in zip(out, want):
                self.assertTrue(torch.equal(a, b))

    def test_startup_prebuild(self) -> None:
        from generative_recommenders.dlrm_v4.train import blas_warmup
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        tln = self.tln
        tln._WLN_BWD_FLYDSL_ENABLED = True
        self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(), "ln_bwd=4")
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)
        with mock.patch.object(lnmod, "ln_bwd", side_effect=RuntimeError("boom")):
            self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(), "ln_bwd=error")
        self.assertEqual(tln._WLN_BWD_FLYDSL_BROKEN,
                         "startup prebuild: RuntimeError: boom")
        tln._WLN_BWD_FLYDSL_ENABLED = False
        self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(), "ln_bwd=off")

    def _mask(self, m: int, keep: float, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(seed)
        mask = torch.rand(m, 512, generator=g, device="cuda") < keep
        mask[1] = False  # an all-dropped and an all-kept row
        mask[2] = True
        return mask

    def test_mscale_kernel(self) -> None:
        # The layer-0 input-LayerNorm backward of item (20)(d) (dropout
        # mask*scale epilogue): dx is native_dropout_backward of the residual
        # call's dx and dW/dB are the residual call's, bit for bit; vs the
        # Triton v2 mscale kernel as close as the residual variant is.
        from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd

        for m, keep in ((M, 0.8), (100_003, 0.7)):
            with self.subTest(m=m, keep=keep):
                t = _inputs(512, True, torch.bfloat16, m)
                mask = self._mask(m, keep, 11)
                scale = 1.0 / keep
                res = _flydsl(t, False, False)
                fly = ln_bwd(t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"],
                             t["res"], dropout_mask=mask, dropout_scale=scale)
                ref = torch.ops.aten.native_dropout_backward(res[0], mask, scale)
                torch.cuda.synchronize()
                for name, a, b in zip(("dx", "dweight", "dbias"), fly, (ref,) + res[1:]):
                    self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)), name)
                tri = self.tln.triton_weighted_layer_norm_bwd(
                    dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
                    rstd=t["rstd"], learnable=True, eps=EPS, BLOCK_D=512,
                    residual=t["res"], dropout_mask=mask, dropout_scale=scale)
                torch.cuda.synchronize()
                for a, b in zip(fly, tri):
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
                    torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)

    def test_mscale_dispatch(self) -> None:
        # DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE routes the mscale call (needs the
        # FlyDSL switch too); a non-contiguous mask falls back to Triton; one
        # arming line per key.
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod

        tln = self.tln
        t = _inputs(512, True, torch.bfloat16)
        mask = self._mask(M, 0.8, 12)
        big = torch.zeros((M, 1024), dtype=torch.bool, device="cuda")
        big[:, :512] = mask
        strided = big[:, :512]

        def call(mk: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            out = tln.triton_weighted_layer_norm_bwd(
                dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
                rstd=t["rstd"], learnable=True, eps=EPS, BLOCK_D=512,
                residual=t["res"], dropout_mask=mk, dropout_scale=1.25)
            torch.cuda.synchronize()
            return out

        want = call(mask)  # FlyDSL off (setUp): the Triton v2 mscale kernel
        fly = lnmod.ln_bwd(t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"],
                           t["res"], dropout_mask=mask, dropout_scale=1.25)
        torch.cuda.synchronize()
        tln._WLN_BWD_FLYDSL_ENABLED = True
        with mock.patch.object(lnmod, "XCD_REMAP", True), \
                mock.patch.object(lnmod, "STREAM_CACHE_ON", True), \
                self.assertLogs(LOGGER, level="WARNING") as logs:
            for on, mk, ref in ((False, mask, want), (True, mask, fly), (True, strided, want)):
                with self.subTest(mscale=on, contiguous=mk.is_contiguous()):
                    tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = on
                    for _ in range(2):  # the second call logs nothing
                        for a, b in zip(call(mk), ref):
                            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        got = [r.getMessage() for r in logs.records if "[ln-bwd]" in r.getMessage()]
        self.assertEqual(got, [
            "[ln-bwd] flydsl: D=512 swish=0 residual=1 dxsum=0 mscale=1 "
            "weight=torch.bfloat16 remap=32 cache=3/3",
            "[ln-bwd] triton: D=512 swish=0 residual=1 dxsum=0 mscale=1 "
            "(mask layout) weight=torch.bfloat16"])
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)

    def test_startup_prebuild_mscale(self) -> None:
        from generative_recommenders.dlrm_v4.train import blas_warmup

        tln = self.tln
        tln._WLN_BWD_FLYDSL_ENABLED = True
        tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = True
        # the bool-mask forward (DLRM_HSTU_FP8_DROPOUT_MASK_BITS not armed)
        with mock.patch.object(blas_warmup, "_mscale_mask_bits_armed", return_value=False):
            self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(), "ln_bwd=4 mscale=1")
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)

    def test_mscale_bits_kernel(self) -> None:
        # DLRM_HSTU_FP8_DROPOUT_MASK_BITS (R-C7) hands the layer-0 call the
        # mask bit-packed, uint8 [N, D / 8]: the packed-mask kernel's dx, dW
        # and dB are the bool-mask kernel's bit for bit (so dx is
        # native_dropout_backward of the residual call's dx), and Triton's
        # packed fold is as close to it as its bool fold.
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod
        from generative_recommenders.ops.input_dropout_fold import pack_dropout_mask_bits

        built = []
        real_build = lnmod.build_ln_bwd

        def spy(*args: Any, **kwargs: Any) -> Any:
            fn = real_build(*args, **kwargs)
            built.append(fn.kernel_name)
            return fn

        for D, m, keep in ((512, M, 0.8), (512, 100_003, 0.7), (512, 7, 0.8), (256, M, 0.8)):
            with self.subTest(D=D, m=m, keep=keep):
                t = _inputs(D, True, torch.bfloat16, m)
                g = torch.Generator(device="cuda").manual_seed(13)
                mask = torch.rand(m, D, generator=g, device="cuda") < keep
                if m > 2:
                    mask[1] = False  # an all-dropped and an all-kept row
                    mask[2] = True
                packed = pack_dropout_mask_bits(mask)
                scale = 1.0 / keep
                args = (t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"], t["res"])
                res = _flydsl(t, False, False)
                built.clear()
                with mock.patch.object(lnmod, "build_ln_bwd", spy), \
                        mock.patch.object(lnmod, "XCD_REMAP", True), \
                        mock.patch.object(lnmod, "STREAM_CACHE_ON", True):
                    boolm = lnmod.ln_bwd(*args, dropout_mask=mask, dropout_scale=scale)
                    bits = lnmod.ln_bwd(*args, dropout_mask=packed, dropout_scale=scale)
                ref = torch.ops.aten.native_dropout_backward(res[0], mask, scale)
                torch.cuda.synchronize()
                suffix = "_x32_c3" if D == 512 else ""
                self.assertEqual(built, [f"ln_bwd_flydsl_d{D}_res_mscale{suffix}",
                                         f"ln_bwd_flydsl_d{D}_res_mscaleb{suffix}"])
                for name, a, b, c in zip(("dx", "dweight", "dbias"), bits, boolm,
                                         (ref,) + res[1:]):
                    self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)), name)
                    self.assertTrue(torch.equal(a.view(torch.int16), c.view(torch.int16)), name)
                tri = self.tln.triton_weighted_layer_norm_bwd(
                    dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
                    rstd=t["rstd"], learnable=True, eps=EPS, BLOCK_D=D,
                    residual=t["res"], dropout_mask=packed, dropout_scale=scale)
                torch.cuda.synchronize()
                for a, b in zip(bits, tri):
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
                    torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)

    def test_mscale_bits_dispatch(self) -> None:
        # Under the same two switches a packed mask takes the FlyDSL
        # packed-mask kernel and a bool mask the bool kernel; a strided packed
        # mask falls back to Triton's packed fold, bitwise; one line per key.
        from generative_recommenders.ops.flydsl import ln_bwd as lnmod
        from generative_recommenders.ops.input_dropout_fold import pack_dropout_mask_bits

        tln = self.tln
        t = _inputs(512, True, torch.bfloat16)
        mask = self._mask(M, 0.8, 14)
        packed = pack_dropout_mask_bits(mask)
        big = torch.zeros((M, 128), dtype=torch.uint8, device="cuda")
        big[:, :64] = packed
        strided = big[:, :64]
        self.assertTrue(tln.wln_bwd_mscale_supported(t["x"], strided))

        def call(mk: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            out = tln.triton_weighted_layer_norm_bwd(
                dy=t["dy"], x=t["x"], weight=t["w"], bias=t["b"], mean=t["mean"],
                rstd=t["rstd"], learnable=True, eps=EPS, BLOCK_D=512,
                residual=t["res"], dropout_mask=mk, dropout_scale=1.25)
            torch.cuda.synchronize()
            return out

        want = call(packed)  # FlyDSL off (setUp): Triton's packed fold
        for a, b in zip(want, call(mask)):  # == its bool fold (R-C7)
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        fly = lnmod.ln_bwd(t["dy"], t["x"], t["w"], t["b"], t["mean"], t["rstd"],
                           t["res"], dropout_mask=mask, dropout_scale=1.25)
        torch.cuda.synchronize()
        tln._WLN_BWD_FLYDSL_ENABLED = True
        with mock.patch.object(lnmod, "XCD_REMAP", True), \
                mock.patch.object(lnmod, "STREAM_CACHE_ON", True), \
                self.assertLogs(LOGGER, level="WARNING") as logs:
            for on, mk, ref in ((False, packed, want), (True, packed, fly),
                                (True, strided, want), (True, mask, fly)):
                with self.subTest(mscale=on, dtype=str(mk.dtype),
                                  contiguous=mk.is_contiguous()):
                    tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = on
                    for _ in range(2):  # the second call logs nothing
                        for a, b in zip(call(mk), ref):
                            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        got = [r.getMessage() for r in logs.records if "[ln-bwd]" in r.getMessage()]
        self.assertEqual(got, [
            "[ln-bwd] flydsl: D=512 swish=0 residual=1 dxsum=0 mscale=1 bits=1 "
            "weight=torch.bfloat16 remap=32 cache=3/3",
            "[ln-bwd] triton: D=512 swish=0 residual=1 dxsum=0 mscale=1 bits=1 "
            "(mask layout) weight=torch.bfloat16",
            "[ln-bwd] flydsl: D=512 swish=0 residual=1 dxsum=0 mscale=1 "
            "weight=torch.bfloat16 remap=32 cache=3/3"])
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)

    def test_startup_prebuild_mscale_bits(self) -> None:
        # The packed-mask kernel is prebuilt exactly when the fused
        # posenc+dropout forward will write the packed mask.
        from generative_recommenders.dlrm_v4.train import blas_warmup
        from generative_recommenders.ops import input_dropout_fold as idf
        from generative_recommenders.ops.triton import triton_fp8 as tf8
        from generative_recommenders.ops.triton import triton_position as tp

        tln = self.tln
        tln._WLN_BWD_FLYDSL_ENABLED = True
        tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = True
        with mock.patch.object(blas_warmup, "_mscale_mask_bits_armed", return_value=True):
            self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(),
                             "ln_bwd=4 mscale=1 bits=1")
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)
        # (MASK_BITS, POSENC_DROPOUT_FUSE, MSCALE_FOLD, WLN_BWD_V2, FP8 arm
        # FUSE_RESIDUAL + FUSE_LN, D) -> armed; the armed row also runs rccl's
        # first-use packed-fold self-test on this GPU (it must pass)
        for bits, fuse, fold, v2, arm, D, want in (
            (True, True, True, True, True, 512, True),
            (False, True, True, True, True, 512, False),
            (True, False, True, True, True, 512, False),
            (True, True, False, True, True, 512, False),
            (True, True, True, False, True, 512, False),
            (True, True, True, True, False, 512, False),
            (True, True, True, True, True, 1024, False),
        ):
            with self.subTest(bits=bits, fuse=fuse, fold=fold, v2=v2, arm=arm, D=D), \
                    mock.patch.object(tp, "_DROPOUT_MASK_BITS", bits), \
                    mock.patch.object(tp, "_POSENC_DROPOUT_FUSE", fuse), \
                    mock.patch.object(idf, "_FOLD_ENABLED", fold), \
                    mock.patch.object(tf8, "_HSTU_FP8_FUSE_RESIDUAL", arm), \
                    mock.patch.object(tf8, "_HSTU_FP8_FUSE_LN", arm):
                tln._WLN_BWD_V2_ENABLED = v2
                self.assertIs(blas_warmup._mscale_mask_bits_armed(D), want)
        # a failing check (e.g. a renamed R-C7 name) is only a missed prebuild:
        # it returns False and never latches the FlyDSL path off
        tln._WLN_BWD_V2_ENABLED = True
        with mock.patch.object(tp, "_DROPOUT_MASK_BITS", True), \
                mock.patch.object(tp, "_POSENC_DROPOUT_FUSE", True), \
                mock.patch.object(tp, "_dropout_mask_bits_why_not",
                                  side_effect=AttributeError("renamed")), \
                self.assertLogs(blas_warmup.logger, level="WARNING"):
            self.assertIs(blas_warmup._mscale_mask_bits_armed(512), False)
            self.assertEqual(blas_warmup._prebuild_flydsl_ln_bwd(), "ln_bwd=4 mscale=1")
        self.assertIsNone(tln._WLN_BWD_FLYDSL_BROKEN)


if __name__ == "__main__":
    unittest.main()
