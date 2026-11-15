#!/usr/bin/env python3
# pyre-strict
"""FP8_WARMUP (ship-12 gin item (35)): code default off (gin binds 2), never raises, skips without
the pre-start or a GPU, and the state guards put every latch and cache back.

CPU-safe: nothing here launches a kernel. The kernel items themselves need the
FP8 environment at import time and an MI350X; they are exercised by the GPU
smoke run (warmup twice in a fresh process with the gin env bootstrap,
FP8_WARMUP_VERIFY=1), not by this unit test."""

import logging
import sys
import types
import unittest
from unittest import mock

import torch

from generative_recommenders.dlrm_v4.train import fp8_warmup as fw


def _model() -> torch.nn.Module:
    return torch.nn.Linear(2, 2)


class Fp8WarmupEntryTest(unittest.TestCase):
    def test_default_and_zero_are_noops(self) -> None:
        model = _model()  # built first: module init draws from the CPU RNG
        cpu = torch.random.get_rng_state()
        with mock.patch.object(fw, "_run") as run:
            fw.run_fp8_first_use_warmup(
                device=torch.device("cpu"), model=model, batch_size=4,
                prestarted=True)
            for value in ("0", " ", "", 0, "-1"):
                fw.run_fp8_first_use_warmup(
                    device=torch.device("cpu"), model=model, batch_size=4,
                    prestarted=True, enabled=value)
        run.assert_not_called()
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))

    def test_malformed_value_warns_and_skips(self) -> None:
        with mock.patch.object(fw, "_run") as run, self.assertLogs(
                fw.logger, logging.WARNING):
            fw.run_fp8_first_use_warmup(
                device=torch.device("cpu"), model=_model(), batch_size=4,
                prestarted=True, enabled="two")
        run.assert_not_called()

    def test_never_raises(self) -> None:
        with mock.patch.object(fw, "_run", side_effect=RuntimeError("boom")), \
                self.assertLogs(fw.logger, logging.ERROR):
            fw.run_fp8_first_use_warmup(
                device=torch.device("cpu"), model=_model(), batch_size=4,
                prestarted=True, enabled="2")

    def test_skips_without_prestart_or_gpu(self) -> None:
        with self.assertLogs(fw.logger, logging.INFO) as cm:
            fw.run_fp8_first_use_warmup(
                device=torch.device("cpu"), model=_model(), batch_size=4,
                prestarted=False, enabled="1")
            fw.run_fp8_first_use_warmup(
                device=torch.device("cpu"), model=_model(), batch_size=4,
                prestarted=True, enabled="1")
        text = "\n".join(cm.output)
        self.assertIn("needs DL_PRESTART", text)
        self.assertIn("no CUDA device", text)
        self.assertNotIn("armed OK", text)


class _FakeLayer:
    def __init__(self, dm: int, h: int, dv: int, dq: int) -> None:
        self._uvqk_weight = torch.empty((dm, 2 * h * (dv + dq)), device="meta")
        self._output_weight = torch.empty((3 * h * dv, dm), device="meta")


class Fp8WarmupShadowSpecTest(unittest.TestCase):
    def test_production_shadow_set(self) -> None:
        """Two ordinary layers + the targets-only layer of the yambda model
        (H=4, Dv=Dq=128, Dm=512): U and VQK slices of each uvqk weight, the
        output weight, and the targets-only V|K layout; duplicates once."""
        layers = [_FakeLayer(512, 4, 128, 128) for _ in range(3)]
        specs = fw._shadow_specs(layers[:2], layers[2], True, True, 4, 128, 128)
        self.assertEqual(
            [(shape, layout) for shape, _dt, layout in specs],
            [((512, 2048), ((0, 512),)),
             ((512, 2048), ((512, 2048),)),
             ((1536, 512), ((0, 512),)),
             ((512, 2048), ((512, 1024), (1536, 2048)))])
        unfused = fw._shadow_specs(layers[:2], layers[2], False, False, 4, 128, 128)
        self.assertEqual([layout for _s, _d, layout in unfused],
                         [((0, 2048),), ((0, 512),)])


class Fp8WarmupStateTest(unittest.TestCase):
    def test_digest_detects_changes_and_equal_after_restore(self) -> None:
        name = "fp8_warmup_test_fake_module"
        mod = types.ModuleType(name)
        mod._flag = None
        mod._latch = False
        mod._set = set()
        mod._cache = {}
        mod._cell = [False]
        mod.CONST = 3
        mod.fn = lambda: 0
        sys.modules[name] = mod
        try:
            before = fw._globals_digest([name])
            mod._flag = True
            mod._set.add(("site", "cuda:0", "v2"))
            mod._cache["key"] = object()
            mod._cell[0] = True
            mod.new_attr = 1
            self.assertEqual(
                fw._digest_diff(before, fw._globals_digest([name])),
                sorted([f"{name}._flag", f"{name}._set", f"{name}._cache",
                        f"{name}._cell", f"{name}.new_attr(new)"]))
            mod._flag = None
            mod._set.clear()
            mod._cache.clear()
            mod._cell[0] = False
            del mod.new_attr
            self.assertEqual(
                fw._digest_diff(before, fw._globals_digest([name])), [])
        finally:
            del sys.modules[name]

    def test_guards_put_ship11_state_back(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
            from generative_recommenders.ops.triton import bwd_fp8
            from generative_recommenders.ops.triton import fp8_helpers_fused as fhf
            from generative_recommenders.ops.triton import triton_fp8 as tf8
            from generative_recommenders.ops.triton import triton_fp8_wshadow as twm
            from generative_recommenders.ops.triton import triton_hstu_attention as tha
            from generative_recommenders.ops.triton import triton_hstu_linear as thl
            from generative_recommenders.ops.triton import triton_layer_norm as tln
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"kernel modules not importable here: {error}")
        scalars = ((thl, "_LNMD_FWD_V3_ENABLED"), (thl, "_LNMD_FWD_MASKGEN_ENABLED"),
                   (thl, "_LNMD_FWD_MASKGEN_ARMED_LOGGED"), (tln, "_WLN_FWD_V2_ENABLED"),
                   (mha, "_fp8_fwd_pipe_logged"))
        before = [getattr(mod, name) for mod, name in scalars]
        guards = fw._Guards()
        guards.take(verify=True)
        try:
            # What the items' production wrappers do on a first call (lazy env
            # flags and once-latches flip; a sentinel is never the old value).
            bwd_fp8._uvqk_armed_logged.add(("store_t_v2", "cuda:0", "v2"))
            bwd_fp8._uvqk_once_logged.add("fp8_warmup_test")
            for mod, name in scalars:
                setattr(mod, name, "fp8_warmup_test_sentinel")
            mha._logged_layouts.add(("valu", "fp8_warmup_test"))
            # X4C19 iter50: the FP8 P@V armed/declined once-latch (F1 reaches it).
            mha._fp8_fwd_pv_logged.add("fp8_warmup_test")
            tha._DELTA_FWD_BM16_LOGGED[0] = True
            twm._plans[("fp8_warmup_test",)] = object()
            fhf._dummy[("cuda:0", torch.float32)] = torch.zeros(1)
            tf8._u_dp_logged.add(("armed", "cuda:0"))
            tf8._u_dp_workspaces[1234] = torch.zeros(1)
            tf8._quiet_seen[("u_dp", 1234)] = None
        finally:
            guards.restore()
        guards.verify_globals()
        self.assertEqual(guards.globals_changed, [])
        self.assertEqual(guards.changed, [])
        # mha's latch is put back by the 595b51e mha guard first; the other four
        # by the generic scalar restore.
        self.assertGreaterEqual(guards.restored_scalars, 4)
        self.assertEqual([getattr(mod, name) for mod, name in scalars], before)
        self.assertNotIn(("store_t_v2", "cuda:0", "v2"), bwd_fp8._uvqk_armed_logged)
        self.assertNotIn("fp8_warmup_test", bwd_fp8._uvqk_once_logged)
        self.assertNotIn(("fp8_warmup_test",), twm._plans)
        self.assertNotIn(1234, tf8._u_dp_workspaces)
        self.assertNotIn(("u_dp", 1234), tf8._quiet_seen)
        self.assertNotIn(("valu", "fp8_warmup_test"), mha._logged_layouts)
        self.assertNotIn("fp8_warmup_test", mha._fp8_fwd_pv_logged)

    def test_bwd_site_registration_is_a_guard_failure(self) -> None:
        try:
            from generative_recommenders.ops.triton import bwd_fp8
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"bwd_fp8 not importable here: {error}")
        guards = fw._Guards()
        guards.take()
        try:
            bwd_fp8._sites["fp8_warmup_test.L0"] = object()  # pyre-ignore[6]
        finally:
            guards.restore()
        self.assertIn("bwd_fp8", guards.changed)
        self.assertNotIn("fp8_warmup_test.L0", bwd_fp8._sites)


class _KernelRecorder:
    """Stands in for a @triton.jit kernel: kernel[grid](*args, **kwargs) only
    records the call (nothing is compiled or launched)."""

    def __init__(self) -> None:
        self.calls: list = []  # pyre-ignore[4]

    def __getitem__(self, grid):  # pyre-ignore[2,3]
        def launch(*args, **kwargs):  # pyre-ignore[2,3,53]
            self.calls.append((grid, args, kwargs))

        return launch


class Fp8WarmupFinRppTest(unittest.TestCase):
    """Item F2r (gin (86)): the buckets cover every RPP the production
    finalize wrappers pass for M = 0 .. 4096 B, and the prebuild calls each
    finalizer exactly once per bucket with the wrappers' own arguments.
    Meta tensors and recorded kernels: nothing is allocated or launched."""

    def setUp(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_dkdv_e5m2 as e8m
            from generative_recommenders.ops.triton import hstu_dq_e5m2 as dqe
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"sink modules not importable here: {error}")
        self.e8m, self.dqe = e8m, dqe
        self.rec = {n: _KernelRecorder() for n in (
            "_e8_pfx_kernel", "_e8_fin_kernel", "_dq8_pfx_kernel", "_dq8_fin_kernel")}
        for name, rec in self.rec.items():
            mod = e8m if name.startswith("_e8") else dqe
            patcher = mock.patch.object(mod, name, rec)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _wrapper_rpp(self, B: int, m: int) -> tuple:  # pyre-ignore[24]
        meta = torch.device("meta")
        so = torch.zeros(B + 1, dtype=torch.int64, device=meta)
        one = torch.ones(1, dtype=torch.float32, device=meta)
        sink = self.e8m.DkdvE5m2Sink(torch.zeros(1, dtype=torch.uint8, device=meta), one)
        sink.launch_kwargs(so, m, 4, 128)
        sink.finish()
        pfx, pcol, pamax = self.dqe._dq8_partials(so, m, 4, 128)
        self.dqe._dq8_finalize(pamax, pcol, pfx)
        return (self.rec["_e8_fin_kernel"].calls[-1][2]["RPP"],
                self.rec["_dq8_fin_kernel"].calls[-1][2]["RPP"])

    def test_buckets_cover_every_wrapper_rpp(self) -> None:
        for B in (1024, 1000, 8):
            n_max = 4096 * B
            e8b = dict(fw._fin_rpp_buckets(B, n_max, 4 * 32))
            dqb = dict(fw._fin_rpp_buckets(B, n_max, 4))
            for rpp, m in e8b.items():  # each bucket's M gets its RPP
                self.assertLessEqual(m, n_max)
                self.assertEqual(self._wrapper_rpp(B, m)[0], rpp)
            for rpp, m in dqb.items():
                self.assertEqual(self._wrapper_rpp(B, m)[1], rpp)
            for m in sorted(set(range(0, n_max + 1, 16384)) | {n_max}):
                e8_rpp, dq8_rpp = self._wrapper_rpp(B, m)
                self.assertIn(e8_rpp, e8b)
                self.assertIn(dq8_rpp, dqb)
        self.assertEqual([r for r, _m in fw._fin_rpp_buckets(1024, 4096 * 1024, 128)],
                         list(range(8, 137, 8)))

    def test_prebuild_calls_each_bucket_once(self) -> None:
        cpu_rng = torch.random.get_rng_state()
        out = fw._fin_rpp_prebuild(torch.device("meta"), 1024, 4, 128, 4096 * 1024,
                                   True, True)
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertEqual(out, "e8_rpp=8..136/17 dq8_rpp=8..136/17")
        for name, nc, ak in (("_e8_fin_kernel", 1024, 128), ("_dq8_fin_kernel", 512, 4)):
            calls = self.rec[name].calls
            self.assertEqual(sorted(kw["RPP"] for _g, _a, kw in calls), list(range(8, 137, 8)))
            for grid, args, kw in calls:
                self.assertEqual(grid, (256,))
                self.assertEqual(args[3], 1024)  # B
                self.assertEqual((kw["NC"], kw["AK"], kw["BLK"], kw["num_warps"]),
                                 (nc, ak, 8, 4))
                self.assertEqual(int(args[1].numel()) // int(args[0].shape[0]), ak)
        for name in ("_e8_pfx_kernel", "_dq8_pfx_kernel"):
            self.assertEqual(len(self.rec[name].calls), 17)
            for _g, args, kw in self.rec[name].calls:
                self.assertEqual((args[0].dtype, int(args[0].numel()), args[2]),
                                 (torch.int64, 1025, 1024))
        off = fw._fin_rpp_prebuild(torch.device("meta"), 1024, 4, 128, 4096 * 1024,
                                   False, True)
        self.assertEqual(off, "dq8_rpp=8..136/17")
        self.assertEqual(len(self.rec["_e8_fin_kernel"].calls), 17)


if __name__ == "__main__":
    unittest.main()
