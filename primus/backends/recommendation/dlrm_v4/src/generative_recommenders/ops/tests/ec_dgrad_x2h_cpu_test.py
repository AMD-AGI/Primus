"""X2H (ops/hip/ec_dgrad_x2h.py): the integration on CPU (no GPU, no Triton, no hipcc).

Drives the real triton_ec_dgrad_segreduce.dgrad_segreduce with recording stand-ins for its Triton
kernels and a recording X2H runtime: the knob-off path launches exactly the shipped passes and never
imports X2H; the armed path hands production's buffers to x2h_pass and keeps combine / zero; a
decline, a missing prebuild, emit, graph capture, an import failure and a raising launch keep (or
return to) the Triton passes, and a failure latches only X2H off. The REAL runtime layer
(_CudaRuntime: ctypes argument order checked by the C parameter names of x2h_pass.cu, rc handling,
grid check, per-stream counters, zero-row sync, device check) runs against a fake library with
torch.cuda mocked. The prebuild checks (exact_check / cross_check) run against a fake E-X2b module:
a skipped write must fail them on sentinel-filled buffers, and so must leftover counters or a
touched zero row. Also: the kernel source pin and its C ABI, the census parser and gate, the cache
(rebuild of a bad entry, untrusted directory, build timeout), the deterministic cross-check input
(structure, determinism, no RNG state touched), blas_warmup's X2H hooks and the gin binding. The
GPU half is ec_dgrad_x2h_test.py.
"""

import contextlib
import hashlib
import io
import os
import random
import re
import subprocess
import sys
import textwrap
import types
import unittest
from unittest import mock

import torch
from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h
from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

W, H, BM = 512, 256, 128
_HERE = os.path.dirname(os.path.abspath(__file__))
_GR = os.path.dirname(os.path.dirname(_HERE))  # .../generative_recommenders
_GIN = os.path.join(_GR, "dlrm_v4", "train", "gin", "yambda_5b.gin")

# x2h_pass<1,0>'s remarks from the ROCm 7.2.1 container build (kernel-lab job 69883, x2h_isa/main)
_REMARKS = "\n".join(
    "x2h_pass.cu:938:1: remark: " + s + " [-Rpass-analysis=kernel-resource-usage]" for s in (
        "Function Name: _ZN3x2h8x2h_passILi1ELi0EEEvNS_6ParamsE", "    TotalSGPRs: 106",
        "    VGPRs: 214", "    AGPRs: 0", "    ScratchSize [bytes/lane]: 0", "    Dynamic Stack: False",
        "    Occupancy [waves/SIMD]: 2", "    SGPRs Spill: 0", "    VGPRs Spill: 0",
        "    LDS Size [bytes/block]: 152992",
        "Function Name: _ZN3x2h8x2h_passILi0ELi0EEEvNS_6ParamsE", "    VGPRs: 218", "    AGPRs: 0",
        "    ScratchSize [bytes/lane]: 0", "    VGPRs Spill: 3", "    LDS Size [bytes/block]: 163232"))


class _Rec:
    """A Triton kernel stand-in: kernel[grid](*args, **kw) is recorded."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def __getitem__(self, grid):
        def call(*args, **kw):
            self.calls.append((grid, args, kw))
            return f"{self.name}{grid}"
        return call


class _RT:
    """A recording X2H runtime (the device side of ec_dgrad_x2h)."""

    def __init__(self, fail_on=None, cap=False):
        self.launches, self.wpacks, self.fail_on, self.cap = [], [], fail_on, cap
        self.z = torch.zeros(H, dtype=torch.bfloat16)
        self.ctr = torch.zeros(8, dtype=torch.int32)

    def capturing(self):
        return self.cap

    def stream(self, dev):
        return 1234

    def grid(self, dev):
        return 256

    def zero_row(self, dev):
        return self.z

    def counters(self, dev, stream):
        return self.ctr

    def wpack(self, lib, w1, nhalf, stream):
        self.wpacks.append((w1.data_ptr(), nhalf, stream))
        return torch.empty(nhalf * 8 * 2 * 16 * 64 * 8, dtype=torch.bfloat16)

    def launch(self, lib, a):
        if self.fail_on == ("cat" if a["is_cat"] else "item"):
            raise RuntimeError("x2h_pass launch returned hip error 98")
        self.launches.append(dict(a))


def _members(mc=300, mi=260, u=50, seed=7):
    g = torch.Generator().manual_seed(seed)
    ia = torch.randint(0, u // 2, (2 * mc,), generator=g)
    ic = torch.randint(u // 2, u, (mi,), generator=g)

    def segs(idx):
        order = torch.argsort(idx, stable=True)
        off = torch.zeros(u + 1, dtype=torch.int64)
        off[1:] = torch.cumsum(torch.bincount(idx, minlength=u), 0)
        return order.contiguous(), off

    cat = xs.Member(torch.randn(mc, H, generator=g).to(torch.bfloat16),
                    torch.randn(H, 2 * W, generator=g).to(torch.bfloat16), ia.contiguous(), *segs(ia))
    item = xs.Member(torch.randn(mi, H, generator=g).to(torch.bfloat16),
                     torch.randn(H, W, generator=g).to(torch.bfloat16), ic.contiguous(), *segs(ic))
    for m in (cat, item):
        for t in m:
            assert t.data_ptr() % 16 == 0
    return cat, item, u


class _DispatchBase(unittest.TestCase):
    """setUp / helpers shared by the dispatch and runtime tests (no tests of its own)."""

    def setUp(self):
        x2h.reset_for_testing()
        self.said = []
        self.kern = {n: _Rec(n) for n in ("_ecx_dgrad_pass", "_ecx_combine", "_ecx_zero_uncovered")}
        self.stack = contextlib.ExitStack()
        for name, rec in self.kern.items():
            self.stack.enter_context(mock.patch.object(xs, name, rec, create=True))
        self.stack.enter_context(mock.patch.object(
            xs, "triton", types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)))
        self.stack.enter_context(mock.patch.object(xs, "_KERNELS_ERROR", None))
        self.stack.enter_context(mock.patch.object(xs, "_DEVICE_TYPES", ("cuda", "cpu")))
        self.stack.enter_context(mock.patch.object(xs, "_X2H_REQUESTED", None))
        self.stack.enter_context(mock.patch.object(xs, "_X2H_MOD", None))
        self.stack.enter_context(mock.patch.object(xs, "LAST_PASS_IMPL", "_ecx_dgrad_pass"))
        self.stack.enter_context(mock.patch.object(xs, "_ONCE", set()))
        self.stack.enter_context(mock.patch.object(x2h, "_say", self.said.append))
        self.stack.enter_context(mock.patch.object(xs, "_say", self.said.append))
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
        os.environ.pop(x2h.ENV_VAR, None)
        os.environ.pop(x2h.ROLE_ENV_VAR, None)

    def tearDown(self):
        self.stack.close()
        x2h.reset_for_testing()

    def arm(self, rt=None):
        """knob on, library 'loaded', CPU device (index -1) armed, recording runtime"""
        os.environ[x2h.ENV_VAR] = "1"
        x2h._lib = object()
        x2h._armed.add(-1)
        x2h._RT = rt or _RT()
        return x2h._RT

    def triton_passes(self):
        return self.kern["_ecx_dgrad_pass"].calls

    def check_combine_zero(self, cat, item, u, acc):
        nt = -(-cat.idx.numel() // BM) - (-item.idx.numel() // BM)
        (gc, ac, kc), = self.kern["_ecx_combine"].calls
        self.assertEqual(gc, (nt,))
        self.assertIs(ac[3], acc)
        self.assertEqual(ac[4:], (W, 8))
        self.assertEqual(kc, {"num_warps": 2})
        (gz, az, kz), = self.kern["_ecx_zero_uncovered"].calls
        self.assertEqual(gz, (-(-u // 16),))
        self.assertIs(az[0], acc)
        self.assertIs(az[1], cat.offsets)
        self.assertIs(az[2], item.offsets)
        self.assertEqual(az[3:], (u, W, 16))
        return ac


class X2HDispatchTest(_DispatchBase):
    def test_knob_off_launches_the_shipped_passes(self):
        cat, item, u = _members()
        x2h._RT = rt = _RT()
        acc = xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(xs.LAST_PASS_IMPL, "_ecx_dgrad_pass")
        self.assertFalse(rt.launches or rt.wpacks)
        (g1, a1, k1), (g2, a2, k2) = self.triton_passes()
        na, nc = -(-cat.idx.numel() // BM), -(-item.idx.numel() // BM)
        self.assertEqual((g1, g2), ((na,), (nc,)))
        self.assertEqual(a1[9:11], (cat.idx.numel(), 0))
        self.assertEqual(a2[9:11], (item.idx.numel(), na))
        self.assertEqual(a1[12:], (True, 128, 64, H, W, 2 * W, 16, H, False))
        self.assertEqual(a2[12:], (False, 128, 64, H, W, W, 16, H, False))
        # + the pass's LLVM scheduler pin where this Triton supports it (Triton 3.9: llvm_fn_attrs; else nothing)
        self.assertEqual(k1, dict(num_warps=4, num_stages=1, waves_per_eu=2, **xs._llvm_sched_kw("iterative-ilp")))
        ac = self.check_combine_zero(cat, item, u, acc)
        self.assertIs(a1[5], acc)
        self.assertIs(a1[6], ac[0])
        self.assertEqual(self.said, [])
        self.assertIsNone(xs._x2h_for(acc.device, "auto", False))

    def test_armed_runs_x2h_pass_into_the_shipped_buffers(self):
        rt = self.arm()
        cat, item, u = _members()
        acc = xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(self.triton_passes(), [])
        self.assertEqual(xs.LAST_PASS_IMPL, x2h.PASS_NAME)
        ac = self.check_combine_zero(cat, item, u, acc)
        frag_u, frag_n, frag_v = ac[0], ac[1], ac[2]
        la, lc = rt.launches
        na = -(-cat.idx.numel() // BM)
        self.assertEqual((la["is_cat"], la["frag_base"], la["npos"]), (1, 0, cat.idx.numel()))
        self.assertEqual((lc["is_cat"], lc["frag_base"], lc["npos"]), (0, na, item.idx.numel()))
        for a, m in ((la, cat), (lc, item)):
            self.assertEqual((a["dh"], a["idx"], a["order"], a["off"]),
                             (m.dh.data_ptr(), m.idx.data_ptr(), m.order.data_ptr(), m.offsets.data_ptr()))
            self.assertEqual((a["acc"], a["frag_u"], a["frag_n"], a["frag_v"]),
                             (acc.data_ptr(), frag_u.data_ptr(), frag_n.data_ptr(), frag_v.data_ptr()))
            self.assertEqual((a["role"], a["sdh"], a["grid"], a["stream"], a["zrow"]),
                             (1, H, 256, 1234, rt.z.data_ptr()))
        self.assertEqual(lc["counters"] - la["counters"], 16)
        self.assertEqual([(p, n) for p, n, _ in rt.wpacks], [(cat.w1.data_ptr(), 2), (item.w1.data_ptr(), 1)])
        xs.dgrad_segreduce(cat, item, u)
        armed = [s for s in self.said if s.startswith(x2h.ARMED_PREFIX + " fused passes")]
        self.assertEqual(len(armed), 1, self.said)
        self.assertIn("kill switch DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=0", armed[0])
        self.assertIsNone(x2h.broken())

    def test_role_env(self):
        rt = self.arm()
        os.environ[x2h.ROLE_ENV_VAR] = "0"
        cat, item, u = _members()
        xs.dgrad_segreduce(cat, item, u)
        self.assertEqual({a["role"] for a in rt.launches}, {0})
        x2h._ROLE = None
        os.environ[x2h.ROLE_ENV_VAR] = "2"
        with self.assertRaises(ValueError):
            x2h.role_swap()

    def test_not_armed_runs_triton_and_builds_nothing(self):
        os.environ[x2h.ENV_VAR] = "1"
        x2h._RT = rt = _RT()
        cat, item, u = _members()
        with mock.patch.object(x2h, "build", side_effect=AssertionError("built after RUN_START")):
            xs.dgrad_segreduce(cat, item, u)
            xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(len(self.triton_passes()), 4)
        self.assertFalse(rt.launches)
        self.assertEqual(sum("NOT armed on cuda:-1" in s for s in self.said), 1, self.said)

    def test_decline_is_per_call(self):
        rt = self.arm()
        cat, item, u = _members()
        buf = torch.empty(cat.dh.numel() + 8, dtype=torch.bfloat16)
        mis = buf[4:4 + cat.dh.numel()].view_as(cat.dh)
        mis.copy_(cat.dh)
        self.assertNotEqual(mis.data_ptr() % 16, 0)
        xs.dgrad_segreduce(cat._replace(dh=mis), item, u)
        self.assertEqual(len(self.triton_passes()), 2)
        self.assertFalse(rt.launches)
        self.assertTrue(any("DECLINED (cat dh not 16-byte aligned)" in s for s in self.said), self.said)
        xs.dgrad_segreduce(cat, item, u)  # the next aligned call takes X2H again
        self.assertEqual(len(rt.launches), 2)
        self.assertIsNone(x2h.broken())
        strided = torch.empty(cat.dh.shape[0], H + 4, dtype=torch.bfloat16)[:, :H]
        self.assertIn("row stride", x2h.decline_reason(cat._replace(dh=strided), "cat"))
        fake = types.SimpleNamespace(shape=(x2h.MAX_DH_ROWS, H), stride=lambda d: H if d == 0 else 1,
                                     data_ptr=lambda: 0)
        self.assertIn("2^22", x2h.decline_reason(cat._replace(dh=fake), "cat"))

    def test_launch_failure_latches_only_x2h(self):
        rt = self.arm(_RT(fail_on="item"))
        cat, item, u = _members()
        acc = xs.dgrad_segreduce(cat, item, u)
        self.assertIsNotNone(x2h.broken())
        self.assertIsNone(xs._BROKEN)
        self.assertEqual(len(rt.launches), 1)  # the cat pass was queued ...
        (g1, a1, _), (g2, a2, _) = self.triton_passes()  # ... and both Triton passes rerun
        self.assertIs(a1[5], acc)
        self.assertIs(a2[5], acc)
        self.assertEqual(xs.LAST_PASS_IMPL, "_ecx_dgrad_pass")
        self.assertEqual(sum("FAILED CLOSED" in s for s in self.said), 1, self.said)
        xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(len(rt.launches), 1)
        self.assertEqual(len(self.triton_passes()), 4)

    def test_emit_takes_the_triton_passes(self):
        rt = self.arm()
        cat, item, u = _members()
        emit = (torch.zeros(cat.dh.shape[0], 2 * W, dtype=torch.bfloat16),
                torch.zeros(item.dh.shape[0], W, dtype=torch.bfloat16))
        xs.dgrad_segreduce(cat, item, u, emit=emit)
        self.assertFalse(rt.launches)
        (_, a1, _), (_, a2, _) = self.triton_passes()
        self.assertIs(a1[11], emit[0])
        self.assertTrue(a1[-1] and a2[-1])

    def test_forced_impls(self):
        rt = self.arm()
        cat, item, u = _members()
        xs.dgrad_segreduce(cat, item, u, _impl="triton")
        self.assertFalse(rt.launches)
        xs.dgrad_segreduce(cat, item, u, _impl="x2h", _grid=7)
        self.assertEqual({a["grid"] for a in rt.launches}, {7})
        self.assertFalse(any("fused passes" in s for s in self.said))  # prebuild calls stay quiet
        buf = torch.empty(cat.dh.numel() + 8, dtype=torch.bfloat16)
        mis = buf[4:4 + cat.dh.numel()].view_as(cat.dh)
        with self.assertRaises(ValueError):
            xs.dgrad_segreduce(cat._replace(dh=mis), item, u, _impl="x2h")
        with self.assertRaises(ValueError):
            xs.dgrad_segreduce(cat, item, u, emit=(None, None), _impl="x2h")
        with self.assertRaises(ValueError):
            xs.dgrad_segreduce(cat, item, u, _impl="bogus")
        x2h._RT = _RT(fail_on="cat")
        with self.assertRaises(RuntimeError):
            xs.dgrad_segreduce(cat, item, u, _impl="x2h")
        self.assertIsNone(x2h.broken())  # strict: the caller (prebuild) decides

    def test_prebuild_without_gpu_fails_closed(self):
        self.assertFalse(x2h.prebuild(torch.device("cpu")))
        self.assertEqual(self.said, [])  # knob off: silent
        os.environ[x2h.ENV_VAR] = "1"
        x2h._REQUESTED = None  # the gate is read once per process
        self.assertFalse(x2h.prebuild(torch.device("cpu")))
        self.assertIn("not a GPU", x2h.broken())
        self.assertEqual(sum("FAILED CLOSED" in s for s in self.said), 1)
        self.assertFalse(x2h._armed)
        self.assertFalse(x2h.active(torch.device("cpu")))

    def test_capture_declines(self):
        rt = self.arm(_RT(cap=True))
        cat, item, u = _members()
        xs.dgrad_segreduce(cat, item, u)
        self.assertFalse(rt.launches)
        self.assertEqual(len(self.triton_passes()), 2)
        self.assertTrue(any("DECLINED (CUDA graph capture" in s for s in self.said), self.said)
        self.assertIsNone(x2h.broken())

    def test_import_failure_falls_back(self):
        os.environ[x2h.ENV_VAR] = "1"
        cat, item, u = _members()
        with mock.patch.dict(sys.modules, {"generative_recommenders.ops.hip.ec_dgrad_x2h": None}):
            xs.dgrad_segreduce(cat, item, u)
            xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(len(self.triton_passes()), 4)
        self.assertEqual(sum("X2H unavailable" in s for s in self.said), 1, self.said)
        self.assertIsNone(xs._BROKEN)
        self.assertFalse(xs._X2H_REQUESTED)

    def test_prefill_sentinels(self):
        cat, item, u = _members()
        acc = xs.dgrad_segreduce(cat, item, u, _prefill=True)  # stand-in kernels write nothing
        (_, ac, _), = self.kern["_ecx_combine"].calls
        self.assertTrue(bool((acc.view(torch.int32) == xs._PREFILL_F32_BITS).all()))
        self.assertTrue(bool((ac[0] == -1).all()) and bool((ac[1] == 0).all()))
        self.assertTrue(bool((ac[2].view(torch.int32) == xs._PREFILL_F32_BITS).all()))
        self.kern["_ecx_combine"].calls.clear()
        acc2 = xs.dgrad_segreduce(cat, item, u)  # no prefill: untouched torch.empty memory
        self.assertEqual(tuple(acc2.shape), (u, W))

    def test_decline_once_per_reason_class(self):
        self.arm()
        cat, item, u = _members()
        for k in (4, 12):
            buf = torch.empty(cat.dh.numel() + 16, dtype=torch.bfloat16)
            mis = buf[k:k + cat.dh.numel()].view_as(cat.dh)
            xs.dgrad_segreduce(cat._replace(dh=mis), item, u)
        self.assertEqual(sum("DECLINED" in s for s in self.said), 1, self.said)
        n = iter(range(10, 99, 7))  # reasons that differ only in their numbers: one line
        with mock.patch.object(x2h, "decline_reason", side_effect=lambda m, k: f"{k} dh has {next(n)} rows"):
            for _ in range(3):
                xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(sum(" rows) ->" in s for s in self.said), 1, self.said)


class _Stream:
    def __init__(self, h):
        self.cuda_stream, self.syncs = h, 0

    def synchronize(self):
        self.syncs += 1


def _c_params(name):
    """Parameter names of an extern "C" function of x2h_pass.cu, in order."""
    src = x2h._SOURCE.read_text()
    m = re.search(r'extern "C" int ' + name + r"\(([^)]*)\)", src)
    return [re.search(r"(\w+)\s*$", p.strip()).group(1) for p in m.group(1).replace("\n", " ").split(",")
            if p.strip() and p.strip() != "void"]


class _FakeLib:
    """The C ABI of libx2h.so as Python callables; arguments recorded BY C PARAMETER NAME."""

    def __init__(self, pass_rc=0, grid_bias=0):
        self.calls, self.pass_rc, self.grid_bias = [], pass_rc, grid_bias
        self.pnames, self.wnames = _c_params("x2h_pass_launch"), _c_params("x2h_wpack_launch")

    def x2h_wp_elems(self, nhalf):
        return nhalf * 8 * 2 * 16 * 64 * 8

    def x2h_wpack_launch(self, *a):
        self.calls.append(("wpack", dict(zip(self.wnames, a))))
        return 0

    def x2h_pass_grid(self, npos, grid, max_tiles):
        return (min(grid, -(-npos // BM)) + self.grid_bias) if max_tiles == 0 else 0

    def x2h_pass_launch(self, *a):
        self.calls.append(("pass", dict(zip(self.pnames, a))))
        return self.pass_rc


class X2HRuntimeTest(_DispatchBase):
    """The real _CudaRuntime against _FakeLib, torch.cuda mocked (current device 0, one stream handle)."""

    def setUp(self):
        super().setUp()
        self.cur = [_Stream(777)]
        for name, kw in (("current_device", dict(return_value=0)),
                         ("current_stream", dict(side_effect=lambda dev=None: self.cur[0])),
                         ("get_device_properties", dict(return_value=types.SimpleNamespace(multi_processor_count=256))),
                         ("is_current_stream_capturing", dict(return_value=False))):
            self.stack.enter_context(mock.patch.object(torch.cuda, name, **kw))

    def armed_real(self, lib):
        os.environ[x2h.ENV_VAR] = "1"
        x2h._lib = lib
        x2h._armed.add(-1)
        x2h._RT = x2h._CudaRuntime()
        return x2h._RT

    def test_ctypes_arguments_by_c_name(self):
        for role in ("1", "0"):
            x2h.reset_for_testing()
            os.environ[x2h.ROLE_ENV_VAR] = role
            lib = _FakeLib()
            rt = self.armed_real(lib)
            cat, item, u = _members()
            acc = xs.dgrad_segreduce(cat, item, u)
            (_, ac, _), = self.kern["_ecx_combine"].calls[-1:]
            self.assertEqual(self.triton_passes(), [])
            w = [c for k, c in lib.calls if k == "wpack"]
            p_ = [c for k, c in lib.calls if k == "pass"]
            self.assertEqual([(c["w1"], c["sw"], c["nhalf"], c["stream"]) for c in w],
                             [(cat.w1.data_ptr(), 2 * W, 2, 777), (item.w1.data_ptr(), W, 1, 777)])
            ctr = rt._counters[(-1, 777)]
            na = -(-cat.idx.numel() // BM)
            for c, m, isc, fb, off, wp in ((p_[0], cat, 1, 0, 0, w[0]["wp"]), (p_[1], item, 0, na, 16, w[1]["wp"])):
                self.assertEqual(
                    (c["role_swap"], c["emit"], c["dh"], c["sdh"], c["wp"], c["idx"], c["order"], c["off"]),
                    (int(role), 0, m.dh.data_ptr(), H, wp, m.idx.data_ptr(), m.order.data_ptr(), m.offsets.data_ptr()))
                self.assertEqual((c["acc"], c["frag_u"], c["frag_n"], c["frag_v"]),
                                 (acc.data_ptr(), ac[0].data_ptr(), ac[1].data_ptr(), ac[2].data_ptr()))
                self.assertEqual((c["dx"], c["sw_dx"], c["zrow"], c["counters"]),
                                 (None, 0, rt._zrow[-1].data_ptr(), ctr.data_ptr() + off))
                self.assertEqual((c["npos"], c["frag_base"], c["is_cat"], c["grid"], c["max_tiles"], c["stream"]),
                                 (m.idx.numel(), fb, isc, 256, 0, 777))
            self.kern["_ecx_combine"].calls.clear()

    def test_counters_per_stream_and_zero_row_sync(self):
        lib = _FakeLib()
        rt = self.armed_real(lib)
        cat, item, u = _members()
        xs.dgrad_segreduce(cat, item, u)
        xs.dgrad_segreduce(cat, item, u)
        s1 = self.cur[0]
        self.cur[0] = _Stream(888)
        xs.dgrad_segreduce(cat, item, u)
        self.assertEqual(sorted(rt._counters), [(-1, 777), (-1, 888)])
        self.assertIsNot(rt._counters[(-1, 777)], rt._counters[(-1, 888)])
        self.assertTrue(all(int(c.abs().sum()) == 0 for c in rt._counters.values()))
        self.assertEqual(s1.syncs, 1)  # the zero row: created and synchronized once
        p_ = [c for k, c in lib.calls if k == "pass"]
        self.assertEqual([c["stream"] for c in p_], [777, 777, 777, 777, 888, 888])
        self.assertEqual(p_[-1]["counters"], rt._counters[(-1, 888)].data_ptr() + 16)

    def test_rc_and_grid_checks(self):
        cat, item, u = _members()
        self.armed_real(_FakeLib(pass_rc=-3))
        xs.dgrad_segreduce(cat, item, u)
        self.assertIsNone(x2h.broken())
        self.assertTrue(any("stale HIP error 3" in s for s in self.said), self.said)
        for lib, what in ((_FakeLib(pass_rc=98), "hip error 98"), (_FakeLib(grid_bias=1), "x2h_pass_grid")):
            x2h.reset_for_testing()
            self.kern["_ecx_dgrad_pass"].calls.clear()
            self.armed_real(lib)
            xs.dgrad_segreduce(cat, item, u)
            self.assertIn(what, x2h.broken() or "")
            self.assertEqual(len(self.triton_passes()), 2)
            self.assertIsNone(xs._BROKEN)
        with self.assertRaises(RuntimeError):
            x2h._check_rc(1, "x")
        x2h._check_rc(0, "x")

    def test_stream_device_check(self):
        with self.assertRaises(ValueError):
            x2h._CudaRuntime.stream(torch.device("cuda", 3))
        self.assertEqual(x2h._CudaRuntime.stream(torch.device("cuda", 0)), 777)


class _FakeXS:
    """An E-X2b module stand-in for exact_check / cross_check: the Triton result is 'right'; x2h_pass is
    right, or skips writes (only visible on sentinel-filled buffers), or leaves counters / touches the
    zero row."""
    Member = xs.Member
    check_invariants = staticmethod(xs.check_invariants)
    _synthetic = staticmethod(xs._synthetic)
    _reference = staticmethod(xs._reference)

    def __init__(self, mode, rt):
        self.mode, self.rt, self.calls = mode, rt, []

    def dgrad_segreduce(self, cat, item, u, emit=None, *, _impl="auto", _grid=None, _prefill=False):
        self.calls.append((_impl, _grid, _prefill))
        right = torch.full((u, W), 1.5) if cat.dh.shape[0] != 500 else xs._reference(cat, item, u)
        if _impl == "triton" or self.mode == "ok":
            return right
        if self.mode == "skip":  # a kernel that skips writes: stale-but-right memory unless prefilled
            return right.view(torch.int32).fill_(xs._PREFILL_F32_BITS).view(torch.float32) if _prefill else right
        if self.mode == "counters" and _grid == x2h.STRESS_GRID:
            self.rt.ctr[1] = 5
        if self.mode == "zero_row" and _grid == x2h.STRESS_GRID:
            self.rt.z[3] = 1.0
        return right


class X2HPrebuildChecksTest(unittest.TestCase):
    def setUp(self):
        x2h.reset_for_testing()
        self.rt = x2h._RT = _RT()

    def tearDown(self):
        x2h.reset_for_testing()

    def test_cross_check_ok_uses_sentinels_everywhere(self):
        fx = _FakeXS("ok", self.rt)
        out = x2h.cross_check(fx, torch.device("cpu"), 2)
        self.assertEqual(out["grids"], (2, x2h.STRESS_GRID, 2))
        self.assertEqual(fx.calls, [("triton", None, True), ("x2h", 2, True), ("x2h", x2h.STRESS_GRID, True),
                                    ("x2h", 2, True)])

    def test_cross_check_catches_skipped_writes_counters_zero_row(self):
        for mode, msg in (("skip", "acc words differ"), ("counters", "counters not reset"),
                          ("zero_row", "zero row modified")):
            self.rt.ctr.zero_()
            self.rt.z.zero_()
            with self.assertRaisesRegex(RuntimeError, msg):
                x2h.cross_check(_FakeXS(mode, self.rt), torch.device("cpu"), 2)

    def test_exact_check(self):
        self.assertEqual(x2h.exact_check(_FakeXS("ok", self.rt), torch.device("cpu")), "96x512")
        with self.assertRaisesRegex(RuntimeError, "fp64 reference"):
            x2h.exact_check(_FakeXS("skip", self.rt), torch.device("cpu"))

    def test_census_gate(self):
        self.assertFalse(x2h.census_ok(None))
        self.assertFalse(x2h.census_ok({}))
        self.assertFalse(x2h.census_ok({"VGPRs Spill": 0}))
        self.assertFalse(x2h.census_ok({"VGPRs Spill": 3, "ScratchSize [bytes/lane]": 0}))
        self.assertTrue(x2h.census_ok(x2h.census(_REMARKS)[x2h.pass_kernel(1)]))

    def test_load_rebuilds_a_bad_cache_entry_once(self):
        good = types.SimpleNamespace(census={x2h.pass_kernel(1): {"VGPRs Spill": 0}})
        for bad in ("raises", "no_census"):
            x2h.reset_for_testing()
            calls = []

            def fake_build(fresh=False):
                calls.append(fresh)
                return (x2h.Path("/x/fresh.so" if fresh else "/x/cached.so"), "", 0.0, not fresh)

            def fake_lib(so, log, s, cached):
                if cached and bad == "raises":
                    raise OSError("truncated .so")
                return types.SimpleNamespace(census={}) if cached else good
            with mock.patch.object(x2h, "build", side_effect=fake_build), mock.patch.object(x2h, "_Lib", fake_lib):
                self.assertIs(x2h.load(1), good)
            self.assertEqual(calls, [False, True], bad)
        x2h.reset_for_testing()
        with mock.patch.object(x2h, "build", return_value=(x2h.Path("/x/new.so"), "", 1.0, False)), \
                mock.patch.object(x2h, "_Lib", side_effect=OSError("bad build")):
            with self.assertRaises(OSError):
                x2h.load(1)  # a fresh build that fails is not retried


_FAKE_HIPCC = """#!/usr/bin/env python3
import os, subprocess, sys, time
a = sys.argv[1:]
if "--version" in a:
    print("fake hipcc 0.0"); sys.exit(0)
if os.environ.get("FAKE_HIPCC_SLEEP"):
    subprocess.Popen(["sleep", "30"]); time.sleep(30)
open(a[a.index("-o") + 1], "wb").write(b"not a real library")
sys.stderr.write("x.cu:1:1: remark: Function Name: _ZN3x2h8x2h_passILi1ELi0EEEvNS_6ParamsE [-R]\\n")
sys.stderr.write("x.cu:1:1: remark:     VGPRs Spill: 0 [-R]\\n")
"""


class X2HBuildTest(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="x2h_buildtest_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.hipcc = os.path.join(self.tmp, "hipcc")
        with open(self.hipcc, "w") as fh:
            fh.write(_FAKE_HIPCC)
        os.chmod(self.hipcc, 0o755)
        self.root = os.path.join(self.tmp, "cache")
        os.mkdir(self.root)
        os.chmod(self.root, 0o755)
        self.env = mock.patch.dict(os.environ, {x2h.HIPCC_ENV_VAR: self.hipcc, x2h.CACHE_ENV_VAR: self.root})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_cached_trusted_untrusted(self):
        import shutil
        so, log, _, cached = x2h.build()
        self.assertFalse(cached)
        self.assertTrue(str(so).startswith(self.root))
        self.assertIn("VGPRs Spill: 0", log)
        self.assertTrue(x2h.build()[3])  # finished entry: no compile
        os.unlink(so.parent / "build.log")
        self.assertFalse(x2h.build()[3])  # a .so without its build.log is rebuilt
        os.chmod(self.root, 0o777)  # group/world-writable shared root: build privately
        so2, _, _, cached2 = x2h.build()
        self.assertFalse(cached2)
        self.assertFalse(str(so2).startswith(self.root))
        shutil.rmtree(so2.parent, True)
        os.chmod(self.root, 0o755)
        so3 = x2h.build(fresh=True)[0]  # a rebuild goes to a private path, the shared entry is dropped
        self.assertFalse(str(so3).startswith(self.root))
        self.assertFalse((so.parent / x2h._LIB_NAME).exists())
        shutil.rmtree(so3.parent, True)

    def test_build_timeout_kills_the_group(self):
        import time
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_SLEEP": "1"}), mock.patch.object(x2h, "_BUILD_TIMEOUT_S", 1):
            t0 = time.time()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                x2h.build()
            self.assertLess(time.time() - t0, 20)

    def test_hipcc_env_is_part_of_the_key(self):
        d1 = x2h.cache_dir()
        with mock.patch.dict(os.environ, {"HIPCC_COMPILE_FLAGS_APPEND": "-O1"}):
            self.assertNotEqual(d1, x2h.cache_dir())


class X2HStaticTest(unittest.TestCase):
    def test_source_pin_and_c_abi(self):
        src = x2h._SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(src).hexdigest(), x2h.SOURCE_SHA256,
                         "x2h_pass.cu changed: rerun the GPU gates (G0-G4) before updating the pin")
        text = src.decode()
        self.assertIn(f"#define X2H_ABI_VERSION {x2h.ABI_VERSION}", text)
        cty = {"int": x2h._I, "long long": x2h._LL, "void*": x2h._VP, "const void*": x2h._VP}
        for name, (res, args) in x2h.SIGNATURES.items():
            m = re.search(r'extern "C" int ' + name + r"\(([^)]*)\)", text)
            self.assertIsNotNone(m, name)
            params = [p.strip() for p in m.group(1).replace("\n", " ").split(",")
                      if p.strip() and p.strip() != "void"]
            self.assertEqual([cty.get(re.sub(r"\s+\w+$", "", p).strip()) for p in params], list(args), name)
            self.assertIs(res, x2h._I)

    def test_census_parser(self):
        c = x2h.census(_REMARKS)
        r1 = c[x2h.pass_kernel(1)]
        self.assertEqual((r1["VGPRs"], r1["AGPRs"], r1["VGPRs Spill"], r1["ScratchSize [bytes/lane]"],
                          r1["LDS Size [bytes/block]"], r1["Occupancy [waves/SIMD]"]),
                         (214, 0, 0, 0, 152992, 2))
        self.assertEqual(c[x2h.pass_kernel(0)]["VGPRs Spill"], 3)
        self.assertEqual(x2h.census("no remarks here"), {})

    def test_import_is_lazy(self):
        script = textwrap.dedent(r"""
            import os, sys
            os.environ["DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE"] = "1"
            os.environ["DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP"] = "1"
            from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce  # noqa
            assert "generative_recommenders.ops.hip.ec_dgrad_x2h" not in sys.modules
            from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h
            assert x2h._lib is None and not x2h._armed  # importing builds and loads nothing
        """)
        r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_knob_off_call_never_imports_x2h(self):
        script = textwrap.dedent(r"""
            import os, sys, types, torch
            os.environ.pop("DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP", None)
            from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs
            class Rec:
                def __getitem__(self, g):
                    return lambda *a, **k: None
            for n in ("_ecx_dgrad_pass", "_ecx_combine", "_ecx_zero_uncovered"):
                setattr(xs, n, Rec())
            xs.triton = types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)
            xs._KERNELS_ERROR = None
            xs._DEVICE_TYPES = ("cuda", "cpu")
            cat, item, u = xs._synthetic(torch.device("cpu"))
            xs.dgrad_segreduce(cat, item, u)
            assert xs.LAST_PASS_IMPL == "_ecx_dgrad_pass"
            assert "generative_recommenders.ops.hip.ec_dgrad_x2h" not in sys.modules, "imported with the knob off"
        """)
        r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_crosscheck_input(self):
        rng, prng = torch.get_rng_state(), random.getstate()
        grid = 4
        cat, item, u, info = x2h.crosscheck_input(torch.device("cpu"), grid, xs.Member)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(prng, random.getstate())
        self.assertIsNone(xs.check_invariants(cat, item, u))
        xs._check_member(cat, "cat", cat.dh.device, u)
        xs._check_member(item, "item", item.dh.device, u)
        sc, si = x2h.structure_stats(cat, True), x2h.structure_stats(item, False)
        for s in (sc, si):
            self.assertGreater(s["tiles"], x2h.NSTATIC * grid + 256)  # every workgroup claims tiles
            self.assertNotIn(s["last"], (0, BM))  # partial last tile
            self.assertGreater(s["tiles_gt32"], 0)  # > 2 slot passes per tile
            self.assertGreaterEqual(s["span_max"], 3)  # hot rows: FRAG_N > 1 (combine)
        self.assertGreater(sc["mixed_tiles"], sc["tiles"] // 4)  # both W1_a halves in a tile
        lens = item.offsets[1:] - item.offsets[:-1]
        self.assertGreater(int(((lens == 0) & (cat.offsets[1:] == cat.offsets[:-1])).sum()), 0)  # uncovered
        self.assertGreater(int(((item.offsets[1:] % BM == 0) & (lens > 0)).sum()), 0)  # tile-end aligned
        for k in ("cat_cancel_pairs", "item_cancel_pairs", "cat_negzero_rows", "item_negzero_rows",
                  "cat_negzero_cols", "item_negzero_cols"):
            self.assertGreater(info[k], 0, k)
        e = ((cat.dh.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
        nz = e[e > 0]
        self.assertEqual((int(nz.min()), int(nz.max())), (119, 135))  # 2^-8 .. 2^8
        again = x2h.crosscheck_input(torch.device("cpu"), grid, xs.Member)
        for a, b in zip(list(cat) + list(item), list(again[0]) + list(again[1])):
            self.assertTrue(torch.equal(a.view(torch.uint8) if a.dtype == torch.bfloat16 else a,
                                        b.view(torch.uint8) if b.dtype == torch.bfloat16 else b))
        self.assertEqual(again[2], u)
        # exactly cancelling pairs: the first two positions of a row carry opposite dh rows
        st = item.offsets[:-1][lens >= 2]
        a, b = item.order[st], item.order[st + 1]
        hits = int((item.dh[a].float() == -item.dh[b].float()).all(dim=1).sum())
        self.assertGreater(hits, info["item_cancel_pairs"] // 2)


class X2HHooksTest(unittest.TestCase):
    def setUp(self):
        try:
            from generative_recommenders.dlrm_v4.train import blas_warmup  # needs gin
            from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
        except ImportError as exc:
            self.skipTest(f"needs gin: {exc}")
        self.bw, self.apply = blas_warmup, apply_env_bootstrap
        x2h.reset_for_testing()

    def tearDown(self):
        x2h.reset_for_testing()

    def test_ignored_line_and_hook(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {x2h.ENV_VAR: "1"}), contextlib.redirect_stdout(out):
            self.bw._x2h_ignored_without_ex2b("E-X2b is off")
        self.assertIn("[ec-dgrad-x2h] DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=1 IGNORED (NOT armed): E-X2b is off",
                      out.getvalue())
        out = io.StringIO()
        with mock.patch.dict(os.environ, {x2h.ENV_VAR: "0"}), contextlib.redirect_stdout(out):
            self.bw._x2h_ignored_without_ex2b("E-X2b is off")
        self.assertEqual(out.getvalue(), "")
        fake_xs = types.SimpleNamespace(enabled=lambda: True, _say=print)
        with mock.patch.dict(os.environ, {x2h.ENV_VAR: "0"}):
            self.assertFalse(self.bw._prebuild_ec_dgrad_x2h(torch.device("cpu"), fake_xs))
        out = io.StringIO()
        with mock.patch.dict(os.environ, {x2h.ENV_VAR: "1"}), contextlib.redirect_stdout(out), \
                mock.patch.object(x2h, "prebuild", return_value=False) as pb:
            self.assertFalse(self.bw._prebuild_ec_dgrad_x2h(
                torch.device("cpu"), types.SimpleNamespace(enabled=lambda: False)))
            pb.assert_not_called()
            self.assertIn("IGNORED (NOT armed): E-X2b failed its pre-run_start prebuild", out.getvalue())
            self.assertTrue(self.bw._prebuild_ec_dgrad_x2h(torch.device("cpu"), fake_xs))
            pb.assert_called_once()

    def test_env_bootstrap_binding(self):
        for value, want in ((False, "0"), (True, "1")):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.apply(DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=value)
                self.assertEqual(os.environ.get(x2h.ENV_VAR), want)
        with mock.patch.dict(os.environ, {x2h.ENV_VAR: "1"}, clear=True):
            self.apply(DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=False)  # a pre-set env var wins
            self.assertEqual(os.environ.get(x2h.ENV_VAR), "1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.apply()
            self.assertNotIn(x2h.ENV_VAR, os.environ)

    def test_gin_binding(self):
        with open(_GIN) as f:
            lines = [ln.strip() for ln in f if not ln.lstrip().startswith("#")]
        live = [ln for ln in lines if "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP" in ln]
        self.assertEqual(live, ["apply_env_bootstrap.DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP = True"])


if __name__ == "__main__":
    unittest.main()
