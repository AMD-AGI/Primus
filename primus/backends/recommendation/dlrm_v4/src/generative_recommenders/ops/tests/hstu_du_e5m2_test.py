#!/usr/bin/env python3

# pyre-unsafe

"""B1a (Arbor r16, DLRM_HSTU_FP8_BWD_DU_E5M2): the LN*U*dropout backward's
e5m2 du emission and the consumer's narrowed cast.

* producer: `_triton_layer_norm_mul_dropout_bwd_du_e5m2` (v3 kernel + stock
  dW/dB finalize) against the shipped v2 path + the shipped cast of its bf16
  du: dx / dweight / dbias bytes, dz8 u-block bytes incl. the zero pad rows
  (also with more pad rows than programs: grid-stride tail), u amax bits,
  u colsum vs fp64; special-value rows (sigmoid fallback, NaN, inf, -0).
* consumer: the multi-range skip cast against the full cast on the cast
  columns (payload bytes incl. pad rows, colsum bitwise, amax), skipped
  columns untouched; `DqE5m2Sink.finish(du_emit=...)` and `finish_u_only`
  merges reproduce the full cast's dz8 / amax / colsum.
* knob off: no B1a state, the v3 kernel is never launched, the compute-output
  backward returns the shipped du (bytes) with no emission record.
Skipped without a GPU.
"""

import os
import unittest
from unittest import mock

import torch

_ENV = {"DLRM_HSTU_FP8_MASK16": "1", "DLRM_HSTU_FP8_LNMD_BWD_V2": "1"}
D = 512
W = 2048
P = 0.1


def _u8(t):
    return t.contiguous().view(torch.uint8)


def _eq(a, b):
    return bool(torch.equal(_u8(a), _u8(b)))


def _rel_l2(c, ref64):
    return float((c.double() - ref64).norm() / ref64.norm().clamp_min(1e-300))


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
class DuE5m2ProducerTest(unittest.TestCase):
    def _inputs(self, M, special=False, seed=1234):
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(seed)

        def rn(shape, s):
            return (torch.randn(shape, generator=g, device=dev) * s).to(torch.bfloat16)

        dy = rn((M, 3 * D), 1e-3)
        x = rn((M, D), 0.5)
        u = rn((M, D), 1.0)
        if special:
            r = [3, M // 3, M // 2, M - 1]
            u[r[0], 5] = -100.0  # d = inf -> wave-uniform tl.sigmoid fallback
            u[r[1], :] = -87.5  # every element in the fallback range
            u[r[2], 9] = float("nan")
            u[r[2], 11] = float("inf")
            u[r[2], 12] = float("-inf")
            dy[r[3], :] = -0.0
            u[r[3], 100:104] = torch.tensor([-89.0, -120.0, 88.0, 1e4], device=dev).to(
                torch.bfloat16
            )
        w = (1.0 + 0.05 * torch.randn(D, generator=g, device=dev)).to(torch.bfloat16)
        b = (0.05 * torch.randn(D, generator=g, device=dev)).to(torch.bfloat16)
        xf = x.float()
        mean = xf.mean(dim=1).contiguous()
        rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6).contiguous()
        with mock.patch.dict(os.environ, _ENV):
            mask = thl._create_dropout_mask(
                N=M, D=D, BLOCK_D=D, concat_u=True, concat_x=True,
                dropout_ratio=P, seed=987654321, device=dev,
            )
        return dict(dy=dy, x=x, u=u, w=w, b=b, mean=mean, rstd=rstd, mask=mask)

    def _stock(self, t, silu_u):
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        with mock.patch.dict(os.environ, _ENV):
            dx, du, dw, db, _ = thl._triton_layer_norm_mul_dropout_bwd_impl(
                t["dy"], t["x"], t["u"], t["w"], t["b"], t["mean"], t["rstd"],
                D, 2, 1e-6, True, P, 0, silu_u, True, True, "silu", False,
                t["mask"], W,
            )
        return dx, du.contiguous(), dw, db

    def _emit(self, t, silu_u, inv):
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        return thl._triton_layer_norm_mul_dropout_bwd_du_e5m2(
            dy=t["dy"], x=t["x"], u=t["u"], weight=t["w"], bias=t["b"],
            mean=t["mean"], rstd=t["rstd"], BLOCK_D=D, num_warps=2,
            dropout_ratio=P, silu_u=silu_u, random_mask=t["mask"],
            du_row_width=W, site_key=12345, inv_scale=inv,
            cast_scale=(1.0 / inv).float(), generation=None,
        )

    def _check(self, M, silu_u=True, special=False):
        from generative_recommenders.ops.triton import bwd_fp8

        t = self._inputs(M, special=special)
        dx_r, du_r, dw_r, db_r = self._stock(t, silu_u)
        finite = du_r.float().isfinite()
        amax = float(du_r.float().abs()[finite].amax())
        nominal = 57344.0 / (8.0 * amax * 0.93)
        for tag, inv_v in (("nominal", nominal), ("clip16x", nominal * 16.0),
                           ("sub64x", nominal / 64.0)):
            with self.subTest(M=M, silu_u=silu_u, special=special, inv=tag):
                inv = torch.tensor([inv_v], dtype=torch.float32, device="cuda")
                Mp = (M + 127) // 128 * 128
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                dx, du, dw, db, emit = self._emit(t, silu_u, inv)
                torch.cuda.synchronize()
                # Review fix: no (M, 2048) bf16 d(uvqk) buffer, only dx + dz8
                # (+ dW/dB, u amax / colsum: small blocks). The caching
                # allocator rounds blocks > 1 MiB up to 2 MiB, others to 512 B.
                grown = torch.cuda.memory_allocated() - before

                def rnd(b):
                    return -(-b // (2 << 20)) * (2 << 20) if b > (1 << 20) else -(-b // 512) * 512

                self.assertLessEqual(grown, rnd(M * D * 2) + rnd(Mp * W) + (64 << 10))
                ref8, ref_amax, ref_col = bwd_fp8.quantize_amax_colsum(
                    du_r, inv, pad_to=128
                )
                self.assertEqual(tuple(emit.dz8.shape), (Mp, W))
                # du is the stride-0 NaN placeholder, no buffer handed over.
                self.assertEqual(tuple(du.shape), (M, D))
                self.assertEqual(du.stride(), (0, 0))
                self.assertTrue(bool(torch.isnan(du[0, 0])))
                self.assertEqual(emit.buffer_ptr, 0)
                self.assertTrue(_eq(dx, dx_r), "dx not bitwise")
                self.assertTrue(_eq(dw, dw_r), "dweight not bitwise")
                self.assertTrue(_eq(db, db_r), "dbias not bitwise")
                self.assertTrue(_eq(emit.dz8[:, :D], ref8), "dz8 u block not bitwise")
                self.assertTrue(bool((emit.dz8[M:, :D].view(torch.uint8) == 0).all()))
                self.assertEqual(
                    emit.u_amax.reshape(1).view(torch.int32).item(),
                    ref_amax.reshape(1).view(torch.int32).item(),
                )
                if not special:
                    c64 = du_r.double().sum(dim=0)
                    self.assertLess(_rel_l2(emit.u_colsum, c64), 2e-6)
                    self.assertLess(_rel_l2(ref_col, c64), 2e-6)

    def test_producer_bitwise_vs_v2_and_shipped_cast(self) -> None:
        # 401 rows: tile_num 100 < 111 pad rows (grid-stride tail);
        # 300001 rows: tile_num 16384 (sms*64 on MI350X), 31 pad rows.
        for M in (401, 1000, 300001):
            self._check(M, silu_u=True)

    def test_producer_silu_u_false(self) -> None:
        self._check(1000, silu_u=False)

    def test_producer_special_rows(self) -> None:
        self._check(70001, silu_u=True, special=True)


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
class DuE5m2ConsumerCastTest(unittest.TestCase):
    def _value(self, M, seed=7):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(seed)
        v = (torch.randn((M, W), generator=g, device=dev) * 1e-3).to(torch.bfloat16)
        inv = (57344.0 / (8.0 * v.float().abs().amax() * 0.93)).reshape(1).float()
        return v, inv

    def test_multi_range_skip_matches_full_cast(self) -> None:
        from generative_recommenders.ops.triton import bwd_fp8
        from generative_recommenders.ops.triton.hstu_dq_e5m2 import (
            quantize_amax_colsum_skip,
        )

        for M in (1000, 77777):
            v, inv = self._value(M)
            ref8, ref_amax, ref_col = bwd_fp8.quantize_amax_colsum(v, inv, pad_to=128)
            for skips in ([(0, 512), (1024, 1536)], [(1024, 1536), (0, 512)],
                          [(0, 512)], (0, 512)):
                with self.subTest(M=M, skips=skips):
                    out = torch.full_like(ref8.view(torch.uint8), 0xFF).view(ref8.dtype)
                    pa, pc = quantize_amax_colsum_skip(v, inv, out, skips)
                    rng = [skips] if isinstance(skips[0], int) else skips
                    keep = torch.ones(W, dtype=torch.bool, device=v.device)
                    for lo, hi in rng:
                        keep[lo:hi] = False
                    cols = keep.nonzero().flatten()
                    skipped = (~keep).nonzero().flatten()
                    o8, r8 = out.view(torch.uint8), ref8.view(torch.uint8)
                    self.assertTrue(bool(torch.equal(o8[:, cols], r8[:, cols])))
                    self.assertTrue(bool((o8[:, skipped] == 0xFF).all()))
                    col = pc.sum(dim=0)
                    self.assertTrue(_eq(col[cols], ref_col[cols]), "colsum not bitwise")
                    self.assertEqual(
                        pa.amax().view(torch.int32).item(),
                        v[:, cols].float().abs().amax().view(torch.int32).item(),
                    )

    def _emit_from_reference(self, v, inv, ref8, ref_col, dz8):
        from generative_recommenders.ops.triton.hstu_du_e5m2 import DuE5m2Emit

        M = v.shape[0]
        dz8[:, :D].copy_(ref8[:, :D])  # what the LNMD producer wrote
        return DuE5m2Emit(
            dz8=dz8, u_cols=(0, D), u_amax=v[:, :D].float().abs().amax(),
            u_colsum=ref_col[:D].clone(), inv_scale=inv,
            cast_scale=(1.0 / inv).float(), rows=M, width=W, site_key=1,
            generation=None, buffer_ptr=0,
        )

    def test_finish_merges_reproduce_full_cast(self) -> None:
        from generative_recommenders.ops.triton import bwd_fp8, hstu_dq_e5m2
        from generative_recommenders.ops.triton.hstu_du_e5m2 import finish_u_only

        M = 5000
        v, inv = self._value(M, seed=11)
        ref8, ref_amax, ref_col = bwd_fp8.quantize_amax_colsum(v, inv, pad_to=128)
        q = (1024, 1536)
        # The unwritten bf16 columns: NaN must never reach the payload.
        dz = v.clone()
        dz[:, :D] = float("nan")
        # u-only finish (no filled dq sink): q comes from the BF16 cast.
        dz8 = torch.full_like(ref8.view(torch.uint8), 0xFF).view(ref8.dtype)
        emit = self._emit_from_reference(v, inv, ref8, ref_col, dz8)
        got8, amax, colsum, m = finish_u_only(emit, dz)
        self.assertEqual(m, M)
        self.assertTrue(_eq(got8, ref8))
        self.assertEqual(amax.view(torch.int32).item(), ref_amax.view(torch.int32).item())
        self.assertTrue(_eq(colsum, ref_col))
        # u|q finish through a filled M2 sink sharing the producer's dz8.
        dz[:, q[0]:q[1]] = float("nan")
        dz8 = torch.full_like(ref8.view(torch.uint8), 0xFF).view(ref8.dtype)
        emit = self._emit_from_reference(v, inv, ref8, ref_col, dz8)
        sink = hstu_dq_e5m2.DqE5m2Sink(
            rows=M, width=W, q_cols=q, inv_scale=inv, cast_scale=(1.0 / inv).float(),
            pad=128, grad_dtype=ref8.dtype, limit=57344.0, device=v.device, dz8=dz8,
        )
        self.assertIs(sink.dz8, dz8)
        dz8[:, q[0]:q[1]].copy_(ref8[:, q[0]:q[1]])  # what the dq-split wrote
        sink.filled = True
        sink.partials = ("pamax", "pcol", "pfx")
        q_amax = v[:, q[0]:q[1]].float().abs().amax()
        with mock.patch.object(
            hstu_dq_e5m2, "_dq8_finalize",
            return_value=(q_amax, ref_col[q[0]:q[1]].clone()),
        ):
            got8, amax, colsum, m = sink.finish(dz, du_emit=emit)
        self.assertTrue(_eq(got8, ref8))
        self.assertEqual(amax.view(torch.int32).item(), ref_amax.view(torch.int32).item())
        self.assertTrue(_eq(colsum, ref_col))
        # LOSSY restore decodes exactly what dequantize_prequant would.
        du = torch.empty((M, W), dtype=torch.bfloat16, device=v.device)[:, :D]
        emit.restore_bf16(du)
        want = (ref8[:M, :D].float() * (1.0 / inv).float()).to(torch.bfloat16)
        self.assertTrue(_eq(du, want))


class _Probe(torch.autograd.Function):
    """Stands in for the deferred-SiLU preprocess forward: tags its output
    like the shipped forward and records the gradient object it receives."""

    seen = []

    @staticmethod
    def forward(ctx, u, site):
        from generative_recommenders.ops.triton.triton_hstu_linear import (
            HSTU_DUVQK_SITE_ATTR,
            HSTU_DUVQK_WIDTH_ATTR,
        )

        out = u.clone()
        setattr(out, HSTU_DUVQK_WIDTH_ATTR, W)
        setattr(out, HSTU_DUVQK_SITE_ATTR, site)
        return out

    @staticmethod
    def backward(ctx, g):
        _Probe.seen.append(g)
        return g.clone(), None


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
class DuE5m2KnobOffTest(unittest.TestCase):
    def test_knob_off_shipped_path(self) -> None:
        from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        if os.environ.get("DLRM_HSTU_FP8_BWD_DU_E5M2", "0") == "1":
            self.skipTest("knob set in this process")
        self.assertFalse(bwd_fp8.du_producer_requested())
        self.assertFalse(bwd_fp8.du_producer_enabled())
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(3)
        M = 1000

        def rn(shape, s):
            return (torch.randn(shape, generator=g, device=dev) * s).to(torch.bfloat16)

        attn, u0, x = rn((M, D), 1.0), rn((M, D), 1.0), rn((M, D), 1.0)
        nw, nb = rn((D,), 1.0), rn((D,), 0.1)
        ow = rn((3 * D, D), 0.02)
        dout = rn((M, D), 1e-2)
        site = hstu_du_e5m2.DuE5m2Site(ow.data_ptr())
        grads = []
        for _ in range(2):
            _Probe.seen = []
            a = attn.clone().requires_grad_(True)
            u = u0.clone().requires_grad_(True)
            with mock.patch.dict(os.environ, _ENV), mock.patch.object(
                thl, "_du_e5m2_plan",
                side_effect=AssertionError("B1a plan evaluated with the knob off"),
            ), mock.patch.object(
                thl, "_triton_layer_norm_mul_dropout_bwd_du_e5m2",
                side_effect=AssertionError("B1a producer called with the knob off"),
            ):
                out = thl.triton_hstu_compute_output(
                    attn=a, u=_Probe.apply(u, site), x=x, norm_weight=nw,
                    norm_bias=nb, output_weight=ow, eps=1e-6, dropout_ratio=P,
                    training=True, silu_u=True, concat_u=True, concat_x=True,
                    mul_u_activation_type="silu", seed=4242,
                    recompute_y_in_backward=True,
                )
                out.backward(dout)
            (gu,) = _Probe.seen
            self.assertIsNone(getattr(gu, thl.HSTU_DU_E5M2_ATTR, None))
            self.assertIsNotNone(getattr(gu, thl.HSTU_DUVQK_BUFFER_ATTR, None))
            self.assertIsNone(site.emitted)
            self.assertIsNone(hstu_du_e5m2.broken())
            grads.append((gu.contiguous(), a.grad.clone()))
        # Deterministic shipped path: the two runs agree byte for byte.
        self.assertTrue(_eq(grads[0][0], grads[1][0]))
        self.assertTrue(_eq(grads[0][1], grads[1][1]))


_KNOB_ON_ENV = dict(
    _ENV,
    DLRM_HSTU_FP8_BWD="1",
    DLRM_HSTU_FP8_BWD_UVQK="1",
    DLRM_HSTU_FP8_BWD_UVQK_WGRAD="fp8t",
    DLRM_HSTU_FP8_BWD_DU_E5M2="1",
    # keep the out-projection dgrad in BF16 for every call of the child
    DLRM_HSTU_FP8_BWD_WARMUP_CALLS="1000",
)


def _knob_on_child() -> None:
    """Runs in a fresh process with _KNOB_ON_ENV (the knob is read at import):
    compute-output forward/backward with a tagged u -> eligibility, emission,
    take, lossy decline + latch, latched shipped path, orphan detection."""
    import json

    from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    res = {"enabled": bwd_fp8.du_producer_enabled()}
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(5)
    M = 3001

    def rn(shape, s):
        return (torch.randn(shape, generator=g, device=dev) * s).to(torch.bfloat16)

    attn0, u0, x = rn((M, D), 1.0), rn((M, D), 1.0), rn((M, D), 1.0)
    nw, nb = rn((D,), 1.0), rn((D,), 0.1)
    ow = rn((3 * D, D), 0.02)
    dout = rn((M, D), 1e-2)
    wq = torch.nn.Parameter(rn((D, W), 0.02))  # stands in for uvqk_weight
    key = wq.data_ptr()
    st = bwd_fp8._site_state(bwd_fp8._site_name("uvqk_projection", key), dev)
    st.accumulate(torch.tensor(3e-3, device=dev))
    st.publish()

    def run(site):
        _Probe.seen = []
        a = attn0.clone().requires_grad_(True)
        u = u0.clone().requires_grad_(True)
        out = thl.triton_hstu_compute_output(
            attn=a, u=_Probe.apply(u, site), x=x, norm_weight=nw, norm_bias=nb,
            output_weight=ow, eps=1e-6, dropout_ratio=P, training=True,
            silu_u=True, concat_u=True, concat_x=True,
            mul_u_activation_type="silu", seed=4242, recompute_y_in_backward=False,
        )
        out.backward(dout)
        (gu,) = _Probe.seen
        return a.grad, gu

    # Reference: same forward stats / mask, shipped v2 backward on the same dy.
    _, mean, rstd, block_d, nwarps, _, mask = thl.triton_layer_norm_mul_dropout_fwd(
        x=attn0, u=u0, weight=nw, bias=nb, eps=1e-6, dropout_ratio=P,
        training=True, silu_u=True, concat_u=True, concat_x=True,
        mul_u_activation_type="silu", seed=4242,
    )
    dy = torch.mm(dout, ow.t())
    dx_r, du_r, _, _, _ = thl._triton_layer_norm_mul_dropout_bwd_impl(
        dy, attn0, u0, nw, nb, mean, rstd, block_d, nwarps, 1e-6, True, P, 0,
        True, True, True, "silu", False, mask, W,
    )
    du_r = du_r.contiguous()
    ref8, ref_amax, _ = bwd_fp8.quantize_amax_colsum(du_r, st.inv_scale, pad_to=128)

    # 1) emission: the gradient reaching the preprocess side is the stride-0
    #    placeholder (no BUFFER tag, no d(uvqk) buffer) carrying the record.
    site = hstu_du_e5m2.DuE5m2Site(key)
    dx, gu = run(site)
    rec = getattr(gu, thl.HSTU_DU_E5M2_ATTR, None)
    res["emitted"] = rec is not None and site.emitted == rec.token
    res["placeholder"] = (
        tuple(gu.shape) == (M, D) and gu.stride() == (0, 0)
        and bool(torch.isnan(gu[0, 0]))
        and getattr(gu, thl.HSTU_DUVQK_BUFFER_ATTR, None) is None
        and rec is not None and rec.buffer_ptr == 0
    )
    res["dx_bitwise"] = _eq(dx, dx_r)
    res["dz8_u_bitwise"] = rec is not None and _eq(rec.dz8[:, :D], ref8)
    res["amax_bitwise"] = rec is not None and _eq(rec.u_amax.reshape(1), ref_amax.reshape(1))
    res["generation_current"] = rec is not None and bwd_fp8.producer_generation_matches(
        rec.generation, "uvqk_projection", key)
    # 2) take consumes it
    got = hstu_du_e5m2.take(site, gu)
    res["take_ok"] = (
        got is rec and getattr(gu, thl.HSTU_DU_E5M2_ATTR, None) is None
        and site.emitted is None and site.consumed
    )
    # 3) not consumable this call -> LOSSY restore into the u columns of the
    #    preprocess backward's fresh d(uvqk) + latch; an adopted buffer is
    #    rejected for a buffer-less emission.
    buf = torch.full((M, W), float("nan"), dtype=torch.bfloat16, device=dev)
    res["adopted_rejected"] = (hstu_du_e5m2.check_consumable(
        rec, buf, True, wq, D, fp8t_wgrad=True, tm_operand_ok=True) or ("",))[0]
    why = hstu_du_e5m2.check_consumable(
        rec, buf, False, wq, D, fp8t_wgrad=False, tm_operand_ok=True)
    res["why"] = None if why is None else why[0]
    cast_scale = rec.cast_scale.clone()
    hstu_du_e5m2.decline(rec, buf[:, :D], why[0], why[1])
    res["restored_is_decode"] = _eq(
        buf[:, :D], (ref8[:M, :D].float() * cast_scale).to(torch.bfloat16))
    res["latched"] = hstu_du_e5m2.broken() is not None
    # 4) latched: shipped path, bf16 du written (== stock)
    site2 = hstu_du_e5m2.DuE5m2Site(key)
    _, gu2 = run(site2)
    res["latched_shipped"] = (
        getattr(gu2, thl.HSTU_DU_E5M2_ATTR, None) is None
        and site2.emitted is None and _eq(gu2, du_r)
    )
    # 5) emitted, but the gradient that arrives is another tensor -> raise
    hstu_du_e5m2.reset_for_testing()
    site3 = hstu_du_e5m2.DuE5m2Site(key)
    _, gu3 = run(site3)
    try:
        hstu_du_e5m2.take(site3, gu3.clone())
        res["orphan_raises"] = False
    except RuntimeError:
        res["orphan_raises"] = True
    # 6) generation moves when the site advances (a consumer call)
    gen = bwd_fp8.producer_generation("uvqk_projection", key)
    st.calls += 1
    res["generation_moves"] = not bwd_fp8.producer_generation_matches(
        gen, "uvqk_projection", key)
    # 7) consumer: consume() on a real emission, u-only and u|q (a filled M2
    #    sink sharing the producer's dz8), against the shipped FP8 projection
    #    backward on the BF16 dz holding the stock du: d_x / d_w bitwise,
    #    v|q|k bias colsum bitwise, u colsum fp32-order.
    from generative_recommenders.ops.triton import hstu_dq_e5m2

    hstu_du_e5m2.reset_for_testing()
    mp = (M + 127) // 128 * 128
    x_t = (torch.randn((D, mp), generator=g, device=dev) * 2.0).to(torch.float8_e4m3fn)
    normed_bf16 = rn((M, D), 1.0)  # read only by a BF16 fallback
    kw = dict(normed_x=None, input_scale_value=0.05, weight=wq,
              normed_x_fp8_t=x_t, recompute_normed_x=lambda: normed_bf16)
    vqk = rn((M, W - D), 1e-3)
    ref_dz = torch.cat((du_r, vqk), dim=1).contiguous()
    ref8, _, ref_col = bwd_fp8.quantize_amax_colsum(ref_dz, st.inv_scale, pad_to=128)
    ref = bwd_fp8.uvqk_projection_backward(dz=ref_dz, **kw)
    res["ref_ran"] = ref is not None
    res["tm_broken"] = bwd_fp8._uvqk_tm_broken
    q0, q1 = 2 * D, 3 * D
    for mode in ("u", "uq"):
        site7 = hstu_du_e5m2.DuE5m2Site(key)
        _, gu7 = run(site7)
        rec7 = hstu_du_e5m2.take(site7, gu7)
        # The preprocess backward's fresh d(uvqk): u columns never written.
        buf = torch.full((M, W), float("nan"), dtype=torch.bfloat16, device=dev)
        buf[:, D:].copy_(vqk)  # what the attention backward writes
        sink = None
        if mode == "uq":
            buf[:, q0:q1] = float("nan")  # M2: BF16 dq left unwritten
            sink = hstu_dq_e5m2.DqE5m2Sink(
                rows=M, width=W, q_cols=(q0, q1), inv_scale=st.inv_scale,
                cast_scale=st.scale.clone(), pad=128,
                grad_dtype=torch.float8_e5m2, limit=57344.0, device=dev,
                dz8=rec7.dz8,
            )
            rec7.dz8[:, q0:q1].copy_(ref8[:, q0:q1])  # what the dq-split writes
            sink.filled = True
            sink.partials = ("pamax", "pcol", "pfx")
        why = hstu_du_e5m2.check_consumable(
            rec7, buf, False, wq, D, fp8t_wgrad=True,
            tm_operand_ok=bwd_fp8._tm_operand_ok(x_t, wq, M))
        res[f"consumable_{mode}"] = why is None
        q_fin = (ref_dz[:, q0:q1].float().abs().amax(), ref_col[q0:q1].clone())
        with mock.patch.object(hstu_dq_e5m2, "_dq8_finalize", return_value=q_fin):
            got = hstu_du_e5m2.consume(rec7, sink, buf, **kw)
        res[f"ran_{mode}"] = got is not None and ref is not None
        if got is not None and ref is not None:
            res[f"dx_{mode}"] = _eq(got[0], ref[0])
            res[f"dw_{mode}"] = _eq(got[1], ref[1])
            res[f"colsum_vqk_{mode}"] = _eq(got[2][D:], ref[2][D:])
            res[f"colsum_u_{mode}"] = _rel_l2(got[2][:D], du_r.double().sum(dim=0)) < 2e-6
        res[f"released_{mode}"] = rec7.dz8 is None
    res["not_latched_after_consume"] = hstu_du_e5m2.broken() is None
    # 8) retained graph (review fix): the first backward emits and is taken;
    #    the second backward of the same graph keeps the shipped v2 path
    #    (bf16 du == stock, no record), not lossy, no latch.
    site8 = hstu_du_e5m2.DuE5m2Site(key)
    _Probe.seen = []
    a8 = attn0.clone().requires_grad_(True)
    u8 = u0.clone().requires_grad_(True)
    out8 = thl.triton_hstu_compute_output(
        attn=a8, u=_Probe.apply(u8, site8), x=x, norm_weight=nw, norm_bias=nb,
        output_weight=ow, eps=1e-6, dropout_ratio=P, training=True,
        silu_u=True, concat_u=True, concat_x=True,
        mul_u_activation_type="silu", seed=4242, recompute_y_in_backward=False,
    )
    out8.backward(dout, retain_graph=True)
    rec8 = hstu_du_e5m2.take(site8, _Probe.seen[-1])
    res["retained_first_emits"] = rec8 is not None and site8.consumed
    if rec8 is not None:
        rec8.release()
    rec8 = None
    out8.backward(dout)
    gu8 = _Probe.seen[-1]
    res["retained_second_shipped"] = (
        getattr(gu8, thl.HSTU_DU_E5M2_ATTR, None) is None
        and site8.emitted is None
        and hstu_du_e5m2.take(site8, gu8) is None
        and _eq(gu8, du_r)
        and hstu_du_e5m2.broken() is None
    )
    print("KNOB_ON_RESULT " + json.dumps(res), flush=True)


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
class DuE5m2KnobOnTest(unittest.TestCase):
    def test_knob_on_producer_plumbing(self) -> None:
        import json
        import subprocess
        import sys

        env = dict(os.environ, **_KNOB_ON_ENV)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--knob-on-child"],
            env=env, capture_output=True, text=True, timeout=1800,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("KNOB_ON_RESULT ")]
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertTrue(lines, proc.stdout[-4000:] + proc.stderr[-4000:])
        res = json.loads(lines[-1][len("KNOB_ON_RESULT "):])
        self.assertIn("du_e5m2 NOT used (fp8t uvqk wgrad not taken this call)", proc.stdout)
        self.assertIn("du_e5m2 NOT used (latched off after:", proc.stdout)
        self.assertIn("ARMED du_e5m2 on cuda:0: _ln_mul_dropout_bwd_dx_du_rng_v3", proc.stdout)
        self.assertIn("-> cast skips u blocks", proc.stdout)
        self.assertIn("-> cast skips u|q blocks", proc.stdout)
        self.assertIn(
            "du_e5m2 NOT used (site already consumed (retained-graph re-backward))",
            proc.stdout,
        )
        want = dict(
            enabled=True, emitted=True, placeholder=True, dx_bitwise=True,
            dz8_u_bitwise=True, amax_bitwise=True, generation_current=True,
            take_ok=True, adopted_rejected="buffer", why="fp8t",
            restored_is_decode=True, latched=True, latched_shipped=True,
            orphan_raises=True, generation_moves=True, ref_ran=True,
            tm_broken=None, not_latched_after_consume=True,
            retained_first_emits=True, retained_second_shipped=True,
        )
        for mode in ("u", "uq"):
            for k in ("consumable", "ran", "dx", "dw", "colsum_vqk", "colsum_u",
                      "released"):
                want[f"{k}_{mode}"] = True
        self.assertEqual(res, want, proc.stdout[-4000:])


if __name__ == "__main__":
    import sys

    if "--knob-on-child" in sys.argv:
        _knob_on_child()
    else:
        unittest.main()
