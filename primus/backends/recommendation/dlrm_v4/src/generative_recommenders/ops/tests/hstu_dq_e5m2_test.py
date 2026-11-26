#!/usr/bin/env python3

# pyre-unsafe

"""The M2 dq producer (hstu_dq_e5m2, DLRM_HSTU_FP8_BWD_DQ_E5M2) must stay in
sync with r15's dq-split.

`DqE5m2Sink` runs a fork of `hstu_dq_split._dq_unit`'s main loop. These tests
run the real FlyDSL C8 attention backward twice on the same inputs -- once with
r15's stock dq-split (BF16 dq) and once through the sink -- and require the
sink's e5m2 dz payload [Mp, 2048] and amax to equal the shipped cast of the
stock result byte for byte, and its u|v|k bias column sums to be bitwise (the
q columns are a declared fp32-order change). A change to the ws layout or the
dq main loop on either side fails here instead of silently in training.
Skipped without a gfx950 GPU and FlyDSL.
"""

import os
import unittest
from unittest import mock

import torch

_C8_ENV = {
    "HSTU_ATTN_BWD": "flydsl",
    "DLRM_HSTU_FP8_BWD_QK_MFMA": "fp8",
    "DLRM_HSTU_FP8_BWD_FULL": "1",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT": "1",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED": "ilr",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT": "sp,r1,wcg",
}


def _flydsl_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    try:
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

        reason = mha.unsupported_hardware_reason()
        if reason:
            return reason
        mha.require_flydsl()
    except Exception as error:  # noqa: BLE001
        return f"FlyDSL unavailable: {error}"
    return None


@unittest.skipIf(_flydsl_reason() is not None, str(_flydsl_reason()))
class DqE5m2SinkSyncTest(unittest.TestCase):
    def _run(self, lengths, sched="ilr"):
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        from generative_recommenders.ops.triton import bwd_fp8
        from generative_recommenders.ops.triton.hstu_dq_e5m2 import DqE5m2Sink

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(7)
        L = torch.tensor(lengths, device=dev)
        offs = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev), L.cumsum(0)))
        T, N, H, D = int(offs[-1]), int(max(lengths)), 4, 128
        s = 0.0173
        vqk = (torch.randn(T, 3 * H * D, device=dev, generator=g) * (0.1 / s)).to(
            torch.float8_e4m3fn
        )
        v, q, k = (x.view(T, H, D) for x in vqk.split(H * D, dim=1))
        dout = (torch.randn(T, H, D, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
        ref = torch.zeros((T, 4 * H * D), device=dev, dtype=torch.bfloat16)
        ref[:, : H * D] = (torch.randn(T, H * D, device=dev, generator=g) * 1e-5).to(
            torch.bfloat16
        )
        got = ref.clone()

        def views(buf):
            return tuple(
                buf[:, c : c + H * D].view(T, H, D) for c in (2 * H * D, 3 * H * D, H * D)
            )

        env = dict(_C8_ENV, DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED=sched)
        if not sched:
            env["DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT"] = ""
        with mock.patch.dict(os.environ, env):
            mha.flydsl_hstu_attention_bwd(
                dout, q, k, v, offs, N, 1.0 / D**0.5, out=views(ref), qkv_scale=s
            )
            amax_q = ref[:, 2 * H * D : 3 * H * D].float().abs().amax()
            inv = (1.0 / (amax_q * (8.0 / 57344.0))).reshape(1).float()
            got[:, 2 * H * D : 3 * H * D] = float("nan")
            sink = DqE5m2Sink(
                rows=T,
                width=4 * H * D,
                q_cols=(2 * H * D, 3 * H * D),
                inv_scale=inv,
                cast_scale=1.0 / inv,
                pad=128,
                grad_dtype=torch.float8_e5m2,
                limit=57344.0,
                device=dev,
            )
            mha.flydsl_hstu_attention_bwd(
                dout, q, k, v, offs, N, 1.0 / D**0.5, out=views(got), qkv_scale=s,
                dq_fp8_sink=sink,
            )
        return bwd_fp8, sink, ref, got, inv, T

    def test_sink_matches_stock_dq_split_and_cast(self) -> None:
        bwd_fp8, sink, ref, got, inv, T = self._run([256, 1000, 37, 1500, 641])
        self.assertTrue(sink.filled, sink.declined)
        # dv / dk came from the same deterministic kernel in both runs.
        self.assertTrue(torch.equal(got[:, 512:1024].view(torch.int16), ref[:, 512:1024].view(torch.int16)))
        self.assertTrue(torch.equal(got[:, 1536:].view(torch.int16), ref[:, 1536:].view(torch.int16)))
        dz8, amax, colsum, m = sink.finish(got)
        r8, ramax, rcol = bwd_fp8.quantize_amax_colsum(ref, inv, pad_to=128)
        self.assertEqual(m, T)
        self.assertEqual(tuple(dz8.shape), tuple(r8.shape))
        self.assertTrue(torch.equal(dz8.view(torch.uint8), r8.view(torch.uint8)))
        self.assertEqual(amax.view(torch.int32).item(), ramax.view(torch.int32).item())
        uvk = torch.cat((torch.arange(0, 1024), torch.arange(1536, 2048))).to(colsum.device)
        self.assertTrue(torch.equal(colsum[uvk].view(torch.int32), rcol[uvk].view(torch.int32)))
        o64 = ref[:, 1024:1536].double().sum(0)
        err = float((colsum[1024:1536].double() - o64).norm() / o64.norm())
        self.assertLess(err, 1e-5)

    def test_unscheduled_dq_split_declines_to_stock(self) -> None:
        _, sink, ref, got, _, _ = self._run([256, 1000, 37], sched="")
        self.assertFalse(sink.filled)
        self.assertIn("sched", str(sink.declined))
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))


# The dq-split launch options the trainer passes (gin: DQ_SPLIT_SCHED=ilr, DQ_SPLIT_OPT=sp,r1,wcg,kb,
# stages 2, num_warps 8); the sink adds waves_per_eu.
_PROD_LAUNCH = dict(fp8_dot=True, bmq=128, num_warps=8, stages=2, nblk=1, order="xcd", sched="ilr",
                    rm=1, cmw=".cg", sptr=1, rsk=1)


def _gfx950_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (got {arch!r})"
    return None


def _load_dq8(diet: str):
    """A private copy of hstu_dq_e5m2 imported with DLRM_HSTU_FP8_DQ8_DIET=diet (the epilogue switch is
    read at import), so one process can run both _dq8_unit_v0 and _dq8_unit_v3."""
    import importlib.util

    from generative_recommenders.ops.triton import hstu_dq_e5m2 as mod

    with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_DQ8_DIET": diet}):
        spec = importlib.util.spec_from_file_location(f"_hstu_dq_e5m2_diet{diet}", mod.__file__)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    assert m._DQ8_DIET == (diet == "1")
    return m


def _gfx950_invalid_pk_f32_opsel(asm: str):
    """Packed-FP32 VOP3P lines whose op_sel AMD LLVM 24 (ROCm 10.2) rejects for gfx950 ("invalid
    op_sel operand"): v_pk_{mul,add}_f32 with op_sel[1] set but op_sel[0] clear, v_pk_fma_f32 with
    op_sel[1] or op_sel[2] set (gfx90a / gfx942 accept all of them)."""
    import re

    bad = []
    for line in asm.splitlines():
        mm = re.search(r"\bv_pk_(mul|add|fma)_f32\b.*\bop_sel:\[([01,]+)\]", line)
        if not mm:
            continue
        sel = [int(x) for x in mm.group(2).split(",")]
        if mm.group(1) == "fma":
            if any(sel[1:]):
                bad.append(line.strip())
        elif len(sel) > 1 and sel[1] and not sel[0]:
            bad.append(line.strip())
    return bad


@unittest.skipIf(_gfx950_reason() is not None, str(_gfx950_reason()))
class Dq8IsaDeterminismTest(unittest.TestCase):
    """_dq8_sched_kernel on synthetic, bitwise-fixed inputs (random finite e4m3 workspace and K, e8 near
    127; full and partial 128-row tiles), production launch options, both epilogues: the DIET=1 output
    (q8 bytes, pcol partial rows, pamax partials) equals DIET=0 bit for bit on every one of 12 relaunches,
    and neither compiled kernel contains a packed-FP32 op_sel form that gfx950 does not support. Under
    Triton 3.9 the DIET=1 epilogue compiled with packed-FP32 ops failed both (see
    hstu_dq_e5m2._no_packed_fp32_launch_opts): a few tiles per launch got a signed zero in lanes 48-63 of
    one register. Needs only Triton and a gfx950 GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        from generative_recommenders.ops.flydsl import hstu_dq_split as dqs
        from generative_recommenders.ops.triton import bwd_fp8

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(11)
        H, D, B, N = 4, 128, 96, 1000
        L = torch.randint(100, N + 1, (B,), device=dev, generator=g)
        L[0], L[1] = N, 128
        offs = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev), L.cumsum(0)))
        T = int(offs[-1])
        nb = (L + 127) // 128
        tiles = int((H * nb * (nb + 1)).sum())
        hdr = dqs.ws_header_bytes(B)

        def e4m3(n):
            x = torch.randint(0, 256, (n,), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
            return torch.where((x & 0x7F) == 0x7F, x - 1, x)  # no NaN codes

        ws = torch.empty(hdr + tiles * 8192, dtype=torch.uint8, device=dev)
        ws[hdr:] = e4m3(tiles * 8192)
        pre = ws[: 4 * (B + 1)].view(torch.int32)
        pre[0] = 0
        pre[1:] = torch.cumsum(H * nb * (nb + 1), 0).to(torch.int32)
        kbuf = e4m3(T * 3 * H * D).view(T, 3 * H * D)  # K at row stride 1536 as in the trainer
        k8 = kbuf[:, 2 * H * D:].view(torch.float8_e4m3fn).view(T, H, D)
        e8 = torch.randint(120, 131, (T, H), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
        grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
        pad = int(bwd_fp8.uvqk_tm_pad())
        rows_out = (T + pad - 1) // pad * pad
        cls.P = dict(dev=dev, H=H, D=D, N=N, T=T, ws=ws, k8=k8, e8=e8, offs=offs, pad=pad,
                     grad_dtype=grad_dtype, limit=limit, rows_out=rows_out,
                     one=torch.ones(1, dtype=torch.float32, device=dev),
                     dq=torch.empty((16, H, D), dtype=torch.bfloat16, device=dev))
        cls.m0, cls.m1 = _load_dq8("0"), _load_dq8("1")

    def _launch(self, m):
        P = self.P
        H, D, T = P["H"], P["D"], P["T"]
        dz8 = torch.zeros((P["rows_out"], 4 * H * D), dtype=P["grad_dtype"], device=P["dev"])
        dz8[:, 2 * H * D: 3 * H * D].view(torch.uint8).fill_(0xFF)  # unwritten bytes would show
        sink = m.DqE5m2Sink(rows=T, width=4 * H * D, q_cols=(2 * H * D, 3 * H * D),
                            inv_scale=P["one"], cast_scale=P["one"], pad=P["pad"],
                            grad_dtype=P["grad_dtype"], limit=P["limit"], device=P["dev"], dz8=dz8,
                            pad_zeroed=True)
        sink(P["ws"], P["k8"], P["e8"], P["offs"], P["dq"], P["N"], 1.0e-5, 3 * H * D, 4 * H * D,
             **_PROD_LAUNCH)
        self.assertTrue(sink.filled, sink.declined)
        pamax, pcol, pfx = sink.partials
        R = int(pfx[-1])
        return (dz8[:T, 2 * H * D: 3 * H * D].contiguous().view(torch.uint8),
                pcol[:R].contiguous().view(torch.int32), pamax[: R * H].contiguous().view(torch.int32))

    def test_diet_epilogue_bitwise_and_deterministic(self) -> None:
        ref = self._launch(self.m0)
        for rep in range(12):
            got = self._launch(self.m1)
            for name, a, b in zip(("q8", "pcol", "pamax"), ref, got):
                bad = int((a != b).sum())
                self.assertEqual(bad, 0, f"DIET=1 relaunch {rep}: {bad} {name} entries differ from DIET=0")
        again = self._launch(self.m0)
        for a, b in zip(ref, again):
            self.assertTrue(torch.equal(a, b))

    def test_isa_has_no_gfx950_invalid_packed_fp32_opsel(self) -> None:
        for m in (self.m0, self.m1):
            self._launch(m)
            kernels = [k for cache in m._dq8_sched_kernel.device_caches.values()
                       for k in cache[0].values()]
            self.assertTrue(kernels)
            for k in kernels:
                bad = _gfx950_invalid_pk_f32_opsel(k.asm["amdgcn"])
                self.assertEqual(bad, [], f"DIET={int(m._DQ8_DIET)}: {bad[:4]}")


if __name__ == "__main__":
    unittest.main()
