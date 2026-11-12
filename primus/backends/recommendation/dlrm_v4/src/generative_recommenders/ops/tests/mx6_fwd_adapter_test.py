#!/usr/bin/env python3
# pyre-strict
"""GPU plumbing test of the DLRM_HSTU_FP8_FWD_MXFP6 adapter (ops/hip/mx6_fwd.py) with IDENTITY stand-ins.

The stand-in "MX6 GEMM" runs the shipped FP8 GEMM on the e4m3 operands behind the (dummy) MX6 bytes, with
exactly the sa / sb / bias the adapter hands it, so every adapter output must be BITWISE the shipped call's
output: this checks the adapter's plumbing -- the output-scale fold, the scaled bias, the U bias cast, the
weight-shadow scale (2- and 3-tuple APIs), shapes and dtypes, the xT8 allocation and layout, the first-call
guard (rel L2 0 -> armed), the ctx marker and the retained-graph recompute -- not MXFP6 numerics (those belong
to pr7's kernels and the offline full-output check).

Two producers, both with the REAL ln_store_mx6 signature (REVIEW B1):
* a "reference producer": the shipped Triton store_t for mean / rstd / xT8 / x8 plus the torch reference of
  x6 / xs (ops/hip/mx6_ref.py), with switchable defects to show the first-call producer check (REVIEW M2)
  latches each one: a 1-ulp mean, one x6 byte wrong only in the no-x8 kernel instantiation, one xs byte wrong
  in both instantiations;
* the REAL r14 producer (ops/hip/ln_store_mx6.py, when in the tree), wrapped only to stash the store_t's y8
  for the identity GEMM: the in-model producer check (store_t bytes, the other instantiation, x6 / xs of the
  row slices vs the torch reference) must pass, and so must the adapter prebuild's self-check on zeros and on
  the ramp (without touching the TM_NONEXP allocator; REVIEW M3).
Perturbed stand-ins (sb x 2, and sb x 1.2 -- rel L2 below the bound, caught by the slope; REVIEW m4.2) must be
REJECTED by the guard, return the shipped FP8 output and latch the site.

gfx950 + ROCm torch._scaled_mm only; not part of the CPU validator set.
Run (kernel lab): lab_py.sh <tree root> -m pytest -q -s generative_recommenders/ops/tests/mx6_fwd_adapter_test.py"""

import contextlib
import importlib
import inspect
import io
import sys
import types
import unittest
from typing import Any, Dict, Optional, Tuple
from unittest import mock

import torch

_GEMM = "_mx6_fwd_adapter_test_gemm"
_PROD = "_mx6_fwd_adapter_test_producer"
_REAL_PRODUCER = "generative_recommenders.ops.hip.ln_store_mx6"


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).startswith("gfx950")


def _real_producer() -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(_REAL_PRODUCER)
    except ImportError:
        return None


class _Kernel:
    impl = "reference-producer"
    source = "/test/ln_store_mx6_reference.cu"
    source_sha256 = "0" * 64


def _install_gemm(side: Dict[int, torch.Tensor], sb_factor: float, n_supported: Tuple[int, ...],
                  three_tuple: bool) -> Dict[str, Any]:
    from generative_recommenders.ops.triton import triton_fp8 as f8

    stats: Dict[str, Any] = {"weights": 0, "gemms": []}
    gemm = types.ModuleType(_GEMM)
    gemm.SUPPORTED_N = n_supported  # pyre-ignore[16]
    gemm.kernel = lambda n: f"identity-N{n}"  # pyre-ignore[16]

    def mx6_weight(weight, layout):  # pyre-ignore
        stats["weights"] += 1
        w8, sb = f8._get_weight_shadow(weight, layout)
        n = w8.shape[1]
        w6 = torch.zeros((n // 256, 2, 4, 12288), dtype=torch.uint8, device=weight.device)
        ws = torch.full((n // 32, 4, 4), 127, dtype=torch.uint8, device=weight.device)
        side[w6.data_ptr()] = w8
        if three_tuple:
            return w6, ws, sb * sb_factor
        return w6, ws

    def k512_mx6(x6, xs, w6, ws, bias, sa, sb, m, n, out=None):  # pyre-ignore
        stats["gemms"].append((m, n, out.dtype))
        if w6.data_ptr() not in side:  # the prebuild self-check: zero operands, zero bias -> +0
            assert int(torch.count_nonzero(x6)) == 0 and int(torch.count_nonzero(w6)) == 0
            out.view(torch.uint8).zero_()
            return out
        res = f8._scaled_mm(side[x6.data_ptr()], side[w6.data_ptr()], sa, sb, out_dtype=out.dtype, bias=bias)
        out.copy_(res)
        return out

    gemm.mx6_weight = mx6_weight  # pyre-ignore[16]
    gemm.k512_mx6 = k512_mx6  # pyre-ignore[16]
    sys.modules[_GEMM] = gemm
    return stats


def _install_reference_producer(side: Dict[int, torch.Tensor], bad_mean: bool = False,
                                bad_x6_no_x8: bool = False, bad_xs: bool = False) -> None:
    """Shipped store_t + mx6_ref, with the real ln_store_mx6 signature."""
    from generative_recommenders.ops.hip import mx6_ref
    from generative_recommenders.ops.triton.triton_layer_norm import (
        triton_weighted_layer_norm_fwd_fp8_store_t,
    )

    prod = types.ModuleType(_PROD)

    def ln_store_mx6(x, weight, bias, eps, inv_scale, mp=None, *, want_x8=False, return_y=False,  # pyre-ignore
                     xt_out=None, x6_out=None, xs_out=None, mean_out=None, rstd_out=None, x8_out=None,
                     y_out=None):
        y8, mean, rstd, yt = triton_weighted_layer_norm_fwd_fp8_store_t(
            x=x, weight=weight, bias=bias, eps=eps, inv_scale=inv_scale, pad_to=128, tiled_xt=True)
        y = mx6_ref.ln_y_exact(x, mean, rstd, weight, bias, inv_scale)
        x6r, xsr = mx6_ref.x6_xs_reference(y)
        want = want_x8 or x8_out is not None
        if bad_x6_no_x8 and not want:  # a defect of the steady-state kernel instantiation only
            x6r.view(-1)[12345] ^= 1
        if bad_xs:  # a defect of both instantiations, inside the head check window
            xsr.view(-1)[77] ^= 1

        def put(dst, src):  # pyre-ignore
            if dst is None:
                return src
            dst.copy_(src)
            return dst

        x6, xs, xt = put(x6_out, x6r), put(xs_out, xsr), put(xt_out, yt)
        m_out, r_out = put(mean_out, mean), put(rstd_out, rstd)
        if bad_mean:  # statistics that are not the store_t's
            m_out[7] = torch.nextafter(m_out[7], m_out[7] + 1)
        x8 = put(x8_out, y8) if want else None
        side[x6.data_ptr()] = y8
        return x6, xs, xt, m_out, r_out, x8

    prod.ln_store_mx6 = ln_store_mx6  # pyre-ignore[16]
    prod.ineligible_reason = lambda x, weight, bias: None  # pyre-ignore[16]
    prod.failure = lambda: None  # pyre-ignore[16]
    prod.kernel = lambda: _Kernel  # pyre-ignore[16]
    sys.modules[_PROD] = prod


@contextlib.contextmanager
def _real_producer_with_side_channel(side: Dict[int, torch.Tensor]):  # pyre-ignore[3]
    """The REAL producer; each call also stashes the shipped store_t's y8 of the same x for the identity GEMM
    (outputs untouched; every call bound against the real signature)."""
    from generative_recommenders.ops.triton.triton_layer_norm import (
        triton_weighted_layer_norm_fwd_fp8_store_t,
    )

    L6 = _real_producer()
    assert L6 is not None
    real = L6.ln_store_mx6
    sig = inspect.signature(real)
    calls = []

    def wrapper(*args, **kwargs):  # pyre-ignore
        sig.bind(*args, **kwargs)
        calls.append(sorted(kwargs))
        out = real(*args, **kwargs)
        x, w, b, eps, inv = args[:5]
        side[out[0].data_ptr()] = triton_weighted_layer_norm_fwd_fp8_store_t(
            x=x, weight=w, bias=b, eps=eps, inv_scale=inv, pad_to=128, tiled_xt=False)[0]
        return out

    with mock.patch.object(L6, "ln_store_mx6", wrapper):
        yield calls


@unittest.skipIf(not _gfx950(), "needs a gfx950 GPU with ROCm torch._scaled_mm")
class Mx6FwdAdapterIdentityTest(unittest.TestCase):
    M = 3000  # not a multiple of 128: tail tile, Mp = 3072

    def setUp(self) -> None:
        from generative_recommenders.ops.hip import mx6_fwd
        from generative_recommenders.ops.triton import triton_fp8 as f8
        from generative_recommenders.ops.triton import triton_layer_norm

        self.mx6, self.f8 = mx6_fwd, f8
        saved = (mx6_fwd.ENABLED, mx6_fwd.GEMM_MODULE, mx6_fwd.PRODUCER_MODULE, mx6_fwd.STRICT,
                 mx6_fwd.SHADOW_AUDIT)
        self.addCleanup(lambda: mx6_fwd.reset_for_testing(
            enabled=saved[0], gemm_module=saved[1], producer_module=saved[2], strict=saved[3],
            shadow_audit=saved[4]))
        self.addCleanup(sys.modules.pop, _GEMM, None)
        self.addCleanup(sys.modules.pop, _PROD, None)
        self.addCleanup(f8.clear_fp8_weight_cache)
        patches = [
            mock.patch.object(f8, "_HSTU_FP8_GEMM", True),
            mock.patch.object(triton_layer_norm, "_STORE_T_V2", True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        no_grad = torch.no_grad()  # as inside the autograd Function's forward
        no_grad.__enter__()
        self.addCleanup(no_grad.__exit__, None, None, None)
        f8.clear_fp8_weight_cache()
        dev = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=dev).manual_seed(4958)
        self.dev = dev
        self.side: Dict[int, torch.Tensor] = {}
        self.x = (torch.randn((self.M, 512), device=dev, generator=g) * 2 + 0.3).to(torch.bfloat16)
        self.nw = (1 + 0.1 * torch.randn(512, device=dev, generator=g)).to(torch.bfloat16)
        self.nb = (0.05 * torch.randn(512, device=dev, generator=g)).to(torch.bfloat16)
        self.w = torch.nn.Parameter(torch.randn((512, 2048), device=dev, generator=g) / 512**0.5)
        self.b = torch.nn.Parameter(0.1 * torch.randn(2048, device=dev, generator=g))
        self.s_a = 6.0 * 1.5 / 448.0  # LN outputs ~N(0, 1): amax ~5-6
        self.s_q = 6.0 * 1.5 / 448.0
        self.a_scale = torch.full((1, 1), self.s_a, device=dev)
        self.q_scale = torch.full((1, 1), self.s_q, device=dev)

    def use(self, producer: str, gemm_n: Tuple[int, ...], three_tuple: bool = True,
            sb_factor: float = 1.0, **defects: bool) -> Dict[str, Any]:
        mx6 = self.mx6
        mx6.reset_for_testing(enabled=True, gemm_module=_GEMM,
                              producer_module=_PROD if producer == "reference" else _REAL_PRODUCER, strict=False)
        if producer == "reference":
            _install_reference_producer(self.side, **defects)
        return _install_gemm(self.side, sb_factor, gemm_n, three_tuple)

    # -- shipped chain -------------------------------------------------------------------------
    def shipped_uvqk(self) -> Tuple[torch.Tensor, ...]:
        from generative_recommenders.ops.triton.triton_layer_norm import (
            triton_weighted_layer_norm_fwd_fp8_store_t,
        )

        f8 = self.f8
        y8, mean, rstd, yt = triton_weighted_layer_norm_fwd_fp8_store_t(
            x=self.x, weight=self.nw, bias=self.nb, eps=1e-6, inv_scale=1.0 / self.s_a,
            pad_to=128, tiled_xt=True)
        u = f8.scaled_addmm_fp8_preq(
            activation_fp8=y8, scale_a=self.a_scale, weight=self.w, bias_or_residual=self.b[:512],
            out_dtype=torch.bfloat16, weight_layout=((0, 512),), data_parallel=True).contiguous()
        vqk = f8.scaled_mm_fp8_out_preq(
            activation_fp8=y8, scale_a=self.a_scale, weight=self.w, bias=self.b[512:],
            weight_layout=((512, 2048),), output_scale=self.q_scale, scale_a_value=self.s_a,
            output_scale_value=self.s_q).contiguous()
        vk_layout = ((512, 1024), (1536, 2048))
        vk = f8.scaled_mm_fp8_out_preq(
            activation_fp8=y8, scale_a=self.a_scale, weight=self.w,
            bias=torch.cat((self.b[512:1024], self.b[1536:2048])), weight_layout=vk_layout,
            output_scale=self.q_scale, scale_a_value=self.s_a, output_scale_value=self.s_q).contiguous()
        return y8, mean, rstd, yt, u, vqk, vk

    def plan(self, kind: str = "uvqk") -> Any:  # pyre-ignore[3]
        common = dict(x=self.x, norm_weight=self.nw, norm_bias=self.nb, eps=1e-6, fp8_scale_value=self.s_a,
                      uvqk_weight=self.w, uvqk_bias=self.b, qkv_frozen=True, grad=True, store_t=True,
                      pad_to=128, tiled_xt=True)
        if kind == "uvqk":
            return self.mx6.uvqk_ln_store(u_columns=512, pack_saved=True, **common)
        return self.mx6.vk_ln_store(vk_layout=((512, 1024), (1536, 2048)), pack_handed=True, **common)

    def _check_identity(self, with_u: bool) -> str:
        mx6 = self.mx6
        y8, mean, rstd, yt, u_ref, vqk_ref, vk_ref = self.shipped_uvqk()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctx = types.SimpleNamespace(hidden_dim=128, num_heads=4, norm_eps=1e-6,
                                        fp8_input_scale_value=self.s_a, fp8_qkv_scale_value=self.s_q)
            for call in range(2):  # guarded first call, then the steady call
                p = self.plan("uvqk")
                self.assertIsNotNone(p)
                self.assertEqual(p.u_mx6, with_u)
                self.assertEqual(p.x8 is not None, call == 0 or not with_u)
                xt8, m_, r_ = p.xt8, p.mean, p.rstd
                self.assertTrue(torch.equal(m_.view(torch.int32), mean.view(torch.int32)))
                self.assertTrue(torch.equal(r_.view(torch.int32), rstd.view(torch.int32)))
                self.assertEqual(tuple(xt8.shape), (24, 512, 128))
                self.assertTrue(torch.equal(xt8.view(torch.uint8), yt.view(torch.uint8)))
                if p.u_mx6:
                    u = mx6.gemm_u(p, uvqk_weight=self.w, uvqk_bias=self.b, u_columns=512,
                                   fp8_scale=self.a_scale, out_dtype=torch.bfloat16)
                else:
                    u = self.f8.scaled_addmm_fp8_preq(
                        activation_fp8=p.x8, scale_a=self.a_scale, weight=self.w,
                        bias_or_residual=self.b[:512], out_dtype=torch.bfloat16,
                        weight_layout=((0, 512),), data_parallel=True).contiguous()
                vqk = mx6.gemm_vqk(p, ctx=ctx, uvqk_weight=self.w, uvqk_bias=self.b, u_columns=512,
                                   fp8_scale=self.a_scale, fp8_scale_value=self.s_a,
                                   fp8_qkv_scale=self.q_scale, qkv_scale_value=self.s_q)
                self.assertIsNone(p.x6)  # released after V/Q/K
                self.assertTrue(torch.equal(u.view(torch.int16), u_ref.view(torch.int16)), f"U call {call}")
                self.assertTrue(torch.equal(vqk.view(torch.uint8), vqk_ref.view(torch.uint8)), f"VQK call {call}")
                self.assertTrue(ctx._mx6_vqk)
            tctx = types.SimpleNamespace()
            pv = self.plan("vk")
            self.assertIsNotNone(pv)
            vk = mx6.gemm_vk(pv, ctx=tctx, uvqk_weight=self.w,
                             b_vk=torch.cat((self.b[512:1024], self.b[1536:2048])),
                             vk_layout=((512, 1024), (1536, 2048)), fp8_scale=self.a_scale,
                             fp8_scale_value=self.s_a, fp8_qkv_scale=self.q_scale, qkv_scale_value=self.s_q)
            self.assertTrue(torch.equal(vk.view(torch.uint8), vk_ref.view(torch.uint8)))
            self.assertTrue(tctx._mx6_vk)
            again = mx6.recompute_vqk(ctx, x=self.x, norm_weight=self.nw, norm_bias=self.nb,
                                      uvqk_weight=self.w, uvqk_bias=self.b,
                                      input_scale=torch.full((1, 1), self.s_a, device=self.dev),
                                      qkv_scale=torch.full((1, 1), self.s_q, device=self.dev))
            self.assertTrue(torch.equal(again.view(torch.uint8), vqk_ref.view(torch.uint8)))
            self.assertIsNone(mx6.recompute_vqk(types.SimpleNamespace(), x=self.x, norm_weight=self.nw,
                                                norm_bias=self.nb, uvqk_weight=self.w, uvqk_bias=self.b,
                                                input_scale=self.a_scale, qkv_scale=self.q_scale))
        text = buf.getvalue()
        self.assertNotIn("DECLINED", text)
        self.assertNotIn("REJECTED", text)
        self.assertNotIn("ERROR", text)
        self.assertEqual(text.count("producer check on cuda:"), 1)  # once per device
        self.assertIn("-> ok", text)
        self.assertIn("guard VQK on cuda:", text)
        self.assertIn("guard VK on cuda:", text)
        self.assertEqual("guard U on cuda:" in text, with_u)
        self.assertEqual(text.count("ARMED on cuda:"), 1)
        self.assertLess(text.index("guard VK on cuda:"), text.index("ARMED on cuda:"))  # REVIEW m5
        counters = mx6.counters()
        self.assertEqual(counters.get("served_VQK"), 2)
        self.assertEqual(counters.get("served_VK"), 1)
        self.assertEqual(counters.get("served_U", 0), 2 if with_u else 0)
        return text

    def test_identity_three_tuple_with_u(self) -> None:
        self.use("reference", (512, 1024, 1536))
        print(self._check_identity(with_u=True))

    def test_identity_two_tuple_u_stays_fp8(self) -> None:
        self.use("reference", (1024, 1536), three_tuple=False)
        print(self._check_identity(with_u=False))

    @unittest.skipIf(_real_producer() is None, "the r14 producer is not in this tree")
    def test_real_producer_identity_with_u(self) -> None:
        self.use("real", (512, 1024, 1536))
        with _real_producer_with_side_channel(self.side) as calls:
            text = self._check_identity(with_u=True)
        print(text)
        self.assertIn(["want_x8", "xt_out"], calls)
        self.assertIn(["mean_out", "rstd_out", "want_x8", "x6_out", "xs_out", "xt_out"], calls)  # the check call
        self.assertIn(["want_x8"], calls)  # the retained-graph recompute: no xt_out

    @unittest.skipIf(_real_producer() is None, "the r14 producer is not in this tree")
    def test_real_producer_identity_u_stays_fp8(self) -> None:
        self.use("real", (1024, 1536))
        with _real_producer_with_side_channel(self.side):
            print(self._check_identity(with_u=False))

    @unittest.skipIf(_real_producer() is None, "the r14 producer is not in this tree")
    def test_prebuild_with_the_real_producer(self) -> None:
        from generative_recommenders.ops.triton import fp8_tm_alloc

        self.use("real", (512, 1024, 1536))
        buf = io.StringIO()
        gates = [mock.patch.object(self.f8, "_HSTU_FP8_FWD_MXFP6", True)]  # as with the knob set at import
        call_site = sys.modules.get("generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention")
        if call_site is not None:
            gates.append(mock.patch.object(call_site, "_MX6_FWD", True))
        with contextlib.ExitStack() as stack:
            for gate in gates:
                stack.enter_context(gate)
            stack.enter_context(mock.patch.object(fp8_tm_alloc, "empty_xt",
                                                  side_effect=AssertionError("TM pool touched")))
            stack.enter_context(mock.patch.object(fp8_tm_alloc, "empty_tm",
                                                  side_effect=AssertionError("TM pool touched")))
            stack.enter_context(contextlib.redirect_stdout(buf))
            ok = self.mx6.prebuild()
        print(buf.getvalue())
        self.assertTrue(ok, buf.getvalue())
        self.assertIn("[ln-store-mx6] prebuilt in", buf.getvalue())
        self.assertIn("[mx6-fwd] prebuilt:", buf.getvalue())
        self.assertIn("zeros + ramp", buf.getvalue())
        self.assertIsNone(self.mx6.not_ready_reason(self.dev))

    def test_perturbed_scale_is_rejected_and_latched(self) -> None:
        mx6 = self.mx6
        self.use("reference", (1024, 1536), sb_factor=2.0)
        _, _, _, _, _, vqk_ref, _ = self.shipped_uvqk()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctx = types.SimpleNamespace()
            p = self.plan("uvqk")
            vqk = mx6.gemm_vqk(p, ctx=ctx, uvqk_weight=self.w, uvqk_bias=self.b, u_columns=512,
                               fp8_scale=self.a_scale, fp8_scale_value=self.s_a,
                               fp8_qkv_scale=self.q_scale, qkv_scale_value=self.s_q)
            self.assertTrue(torch.equal(vqk.view(torch.uint8), vqk_ref.view(torch.uint8)))  # FP8 output
            self.assertFalse(ctx._mx6_vqk)
            self.assertIsNone(self.plan("uvqk"))  # VQK latched: the next call declines
        text = buf.getvalue()
        print(text)
        self.assertIn("-> REJECTED", text)
        self.assertIn("DECLINED", text)
        self.assertIn("latched", mx6.site_reason(self.dev, "VQK"))

    def test_non_power_of_two_scale_error_is_caught_by_the_slope(self) -> None:
        mx6 = self.mx6
        self.use("reference", (1024, 1536), sb_factor=1.2)
        _, _, _, _, _, vqk_ref, _ = self.shipped_uvqk()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctx = types.SimpleNamespace()
            p = self.plan("uvqk")
            vqk = mx6.gemm_vqk(p, ctx=ctx, uvqk_weight=self.w, uvqk_bias=self.b, u_columns=512,
                               fp8_scale=self.a_scale, fp8_scale_value=self.s_a,
                               fp8_qkv_scale=self.q_scale, qkv_scale_value=self.s_q)
        text = buf.getvalue()
        print(text)
        self.assertTrue(torch.equal(vqk.view(torch.uint8), vqk_ref.view(torch.uint8)))
        line = next(ln for ln in text.splitlines() if "guard VQK" in ln)
        rel = float(line.split("rel_l2=")[1].split()[0])
        slope = float(line.split("slope=")[1].split()[0])
        self.assertLess(rel, mx6.GUARD_REL_L2_MAX)  # invisible to the rel L2 bound ...
        self.assertGreater(abs(slope - 1.0), mx6.GUARD_SLOPE_MAX)  # ... the slope rejects it
        self.assertIn("-> REJECTED", line)

    def _producer_defect(self, **defect: bool) -> str:
        mx6 = self.mx6
        self.use("reference", (1024, 1536), **defect)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(self.plan("uvqk"))  # the shipped LN then runs
            self.assertIsNone(self.plan("vk"))
        text = buf.getvalue()
        print(text)
        self.assertIn("ERROR producer check on cuda:", text)
        self.assertIn("DECLINED", text)
        self.assertIn("producer check", mx6.not_ready_reason(self.dev))
        return text

    def test_producer_not_bitwise_to_store_t_is_latched(self) -> None:
        self.assertIn("mean 1,", self._producer_defect(bad_mean=True))

    def test_producer_defect_in_the_steady_kernel_is_latched(self) -> None:  # REVIEW M2
        text = self._producer_defect(bad_x6_no_x8=True)
        self.assertIn("other instantiation x6 1,", text)
        self.assertIn("/ 0 B (reference", text)  # the first (x8) call itself matched the reference

    def test_producer_xs_defect_is_caught_by_the_reference(self) -> None:  # REVIEW M2
        text = self._producer_defect(bad_xs=True)
        self.assertIn("rows 0 / 1 B", text)


if __name__ == "__main__":
    unittest.main()
