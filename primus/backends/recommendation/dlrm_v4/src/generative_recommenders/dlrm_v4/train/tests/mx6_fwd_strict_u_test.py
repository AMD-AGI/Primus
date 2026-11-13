#!/usr/bin/env python3
# pyre-strict
"""DLRM_HSTU_FP8_FWD_MXFP6 option C: STRICT covers U (r14 iteration 51; review M2 of iteration 50).

CPU only: nothing is built or launched. hipcc / the libraries (k512_mx6_gemm._load), the GEMM self-checks, the device
and the LN producer are mocked; the REAL k512_mx6_gemm.prebuild() / kernel() and the REAL adapter
(ops/hip/mx6_fwd.py) run. The GEMM module offers U (N=512 in k512_mx6_gemm.SUPPORTED_N, i.e. k512_mx6_u.cu next to
the module = option C), and then:

* a U build failure and a U self-check failure inside k512_mx6_gemm.prebuild() are latched there and prebuild()
  still returns True (the module's contract, unchanged) -- the adapter has to decide;
* DLRM_HSTU_FP8_FWD_MXFP6_STRICT=1: mx6_fwd.prebuild() raises PrebuildError ("prebuild DECLINED: OPTION-C U NOT
  READY") before the producer self-check; a uvqk call whose U is not ready (lazy build failure without a prebuild,
  or the U site latched) raises StrictError before the producer runs; the targets-only layer (no U) is unaffected;
* without STRICT: option B -- prebuild() succeeds, VQK / VK stay ready, only N=1536 / 1024 are self-checked, U
  stays on the shipped FP8 GEMM and the producer writes x8, no decline; ONE loud "[mx6-fwd] ERROR OPTION-C U NOT
  READY ... FALLBACK TO OPTION B" line per process (check_mx6_arming.py rule L8);
* option B (no N=512 in SUPPORTED_N) and option C with U ready behave as before: no such line, no raise;
* knob off: untouched -- no output, no counters, the GEMM module is never imported by the adapter or built."""

import contextlib
import importlib
import io
import pathlib
import sys
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

_REAL_GEMM = "generative_recommenders.ops.hip.k512_mx6_gemm"
_FAKE_PRODUCER = "_mx6_strict_u_test_fake_producer"
_TAG = "OPTION-C U NOT READY"
_U_BUILD_ERROR = ("hipcc rc=1 for libk512_mx6_u.so (log /fake/build_libk512_mx6_u.so.log): error: synthetic U "
                  "build failure")
_U_SELFCHECK_ERROR = "GEMM N=512 self-check on zeros wrote 7 nonzero bytes (/fake/lib_512.so)"


def _try_import(name: str) -> Optional[Any]:  # pyre-ignore[3]
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment specific
        return None


torch = _try_import("torch")


@contextlib.contextmanager
def _capture():  # pyre-ignore[3]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _fake_outputs(x: Any, kwargs: Dict[str, Any]) -> Tuple[Any, ...]:  # pyre-ignore[2]
    """What the real ln_store_mx6 returns for these arguments (shapes / dtypes / identities), on CPU."""
    assert torch is not None
    m = int(x.shape[0])
    t = (m + 127) // 128
    e4 = torch.float8_e4m3fn

    def get(key: str, make: Any) -> Any:  # pyre-ignore[2,3]
        value = kwargs.get(key)
        return make() if value is None else value

    x6 = get("x6_out", lambda: torch.zeros((t, 4, 12288), dtype=torch.uint8))
    xs = get("xs_out", lambda: torch.full((t, 4, 512), 127, dtype=torch.uint8))
    xt = get("xt_out", lambda: torch.zeros((t, 512, 128), dtype=torch.uint8).view(e4))
    mean = get("mean_out", lambda: torch.zeros(m))
    rstd = get("rstd_out", lambda: torch.ones(m))
    want_x8 = kwargs.get("want_x8", False) or kwargs.get("x8_out") is not None
    x8 = get("x8_out", lambda: torch.zeros((m, 512), dtype=torch.uint8).view(e4)) if want_x8 else None
    return (x6, xs, xt, mean, rstd, x8)


class _FakeProducerKernel:
    impl = "fake"
    source = "/fake/ln_store_mx6.cu"
    source_sha256 = "f" * 64


def _install_fake_producer() -> types.ModuleType:
    """An API-complete stand-in for the r14 producer (the REAL ln_store_mx6 signature), CPU, no math."""
    prod = types.ModuleType(_FAKE_PRODUCER)
    prod.calls = []  # pyre-ignore[16]

    def ln_store_mx6(x: Any, weight: Any, bias: Any, eps: float, inv_scale: float,  # pyre-ignore[2]
                     mp: Optional[int] = None, *, want_x8: bool = False, return_y: bool = False,
                     xt_out: Any = None, x6_out: Any = None, xs_out: Any = None, mean_out: Any = None,
                     rstd_out: Any = None, x8_out: Any = None, y_out: Any = None) -> Tuple[Any, ...]:
        kwargs = dict(want_x8=want_x8, xt_out=xt_out, x6_out=x6_out, xs_out=xs_out, mean_out=mean_out,
                      rstd_out=rstd_out, x8_out=x8_out)
        prod.calls.append({"want_x8": bool(want_x8)})
        return _fake_outputs(x, kwargs)

    prod.ln_store_mx6 = ln_store_mx6  # pyre-ignore[16]
    prod.ineligible_reason = lambda x, weight, bias: None  # pyre-ignore[16]
    prod.failure = lambda: None  # pyre-ignore[16]
    prod.kernel = lambda: _FakeProducerKernel  # pyre-ignore[16]
    prod.prebuild = lambda: True  # pyre-ignore[16]
    sys.modules[_FAKE_PRODUCER] = prod
    return prod


@unittest.skipIf(torch is None, "torch not installed")
class _UCase(unittest.TestCase):
    """The real k512_mx6_gemm (U offered) with mocked builds / self-checks, the real adapter, a fake producer."""

    supported_n: Tuple[int, ...] = (512, 1024, 1536)  # option C

    def setUp(self) -> None:
        from generative_recommenders.ops.hip import mx6_fwd
        from generative_recommenders.ops.triton import wgrad_ck

        self.mx6 = mx6_fwd
        self.G = G = importlib.import_module(_REAL_GEMM)
        m = self.mx6
        saved = (m.ENABLED, m.GEMM_MODULE, m.PRODUCER_MODULE, m.STRICT, m.SHADOW_AUDIT)
        self.addCleanup(lambda: m.reset_for_testing(enabled=saved[0], gemm_module=saved[1], producer_module=saved[2],
                                                    strict=saved[3], shadow_audit=saved[4]))
        self.prod = _install_fake_producer()
        self.addCleanup(sys.modules.pop, _FAKE_PRODUCER, None)
        self.loads: List[Any] = []
        self.selfchecks: List[int] = []
        self.launches: List[int] = []
        self.u_build_fails = False
        self.u_selfcheck_fails = False
        self.producer_selfchecks = 0
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in (
            mock.patch.object(G, "SUPPORTED_N", self.supported_n),
            mock.patch.object(G, "_load", self._fake_load),  # no hipcc
            mock.patch.object(G, "_selfcheck_shadow", lambda device: None),
            mock.patch.object(G, "_selfcheck_gemm", self._fake_selfcheck_gemm),
            mock.patch.object(G, "_process_build_dir", lambda: pathlib.Path("/fake/mx6_build_dir")),
            mock.patch.object(G, "k512_mx6", self._fake_k512_mx6),
            mock.patch.dict(G._kernels, clear=True),
            mock.patch.dict(G._failures, clear=True),
            mock.patch.object(G, "_prebuilt", False),
            mock.patch.object(m, "_gate_reason", return_value=None),
            mock.patch.object(m, "_prebuild_device", return_value=torch.device("cpu")),
            mock.patch.object(m, "_selfcheck_producer", side_effect=self._fake_selfcheck_producer),
            # CPU tensors: let a call pass the device / scale-mode / xT-layout checks; no store_t on CPU
            mock.patch.object(m, "_operand_reason", return_value=None),
            mock.patch.object(m, "_scale_mode_reason", return_value=None),
            mock.patch.object(m, "_xt_layout_reason", return_value=None),
            mock.patch.object(m, "_producer_check", return_value=None),
            mock.patch.object(wgrad_ck, "note_tiled_producer"),
        ):
            stack.enter_context(patch)
        m.reset_for_testing(enabled=True, gemm_module=_REAL_GEMM, producer_module=_FAKE_PRODUCER, strict=False,
                            shadow_audit=0)

    # ---- mocks -------------------------------------------------------------------------------------------------
    def _fake_load(self, key: Any) -> Any:  # pyre-ignore[2,3]
        self.loads.append(key)
        if key == self.G.U_N and self.u_build_fails:
            raise RuntimeError(_U_BUILD_ERROR)
        return self.G.Kernel(key=key, n=key if isinstance(key, int) else 0, path=f"/fake/lib_{key}.so",
                             fn=lambda *args: 0, impl="fake", source="/fake/src.cu", source_sha256="0" * 64,
                             build_s=0.0, resources="fake")

    def _fake_selfcheck_gemm(self, device: Any, n: int, rows: int = 257) -> Optional[str]:  # pyre-ignore[2]
        self.selfchecks.append(int(n))
        return _U_SELFCHECK_ERROR if (n == self.G.U_N and self.u_selfcheck_fails) else None

    def _fake_k512_mx6(self, x6: Any, xs: Any, w6: Any, ws: Any, bias: Any, sa: Any, sb: Any,  # pyre-ignore[2,3]
                       m: int, n: int, out: Any = None) -> Any:
        self.launches.append(int(n))
        out.view(torch.uint8).zero_()  # a correct GEMM on zero operands
        return out

    def _fake_selfcheck_producer(self, device: Any, rows: int) -> Tuple[Optional[str], Any]:  # pyre-ignore[2]
        self.producer_selfchecks += 1
        t = (rows + 127) // 128
        plan = self.mx6.Plan(kind="selfcheck", device=device, m=rows,
                             x6=torch.zeros((t, 4, 12288), dtype=torch.uint8),
                             xs=torch.full((t, 4, 512), 127, dtype=torch.uint8), xt8=None,
                             mean=torch.zeros(rows), rstd=torch.ones(rows), x8=None)
        return None, plan

    @contextlib.contextmanager
    def on_gfx950(self):  # pyre-ignore[3]
        """The prebuild entry checks of both modules see one gfx950 device (nothing touches a device)."""
        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=props), \
                mock.patch.object(torch.cuda, "synchronize", return_value=None):
            yield

    # ---- drivers -----------------------------------------------------------------------------------------------
    def prebuild(self, strict: bool) -> Tuple[Optional[bool], Optional[BaseException], str]:
        self.mx6.reset_for_testing(strict=strict)
        ok: Optional[bool] = None
        error: Optional[BaseException] = None
        with self.on_gfx950(), _capture() as out:
            try:
                ok = self.mx6.prebuild()
            except Exception as exc:  # the STRICT cases
                error = exc
        return ok, error, out.getvalue()

    def uvqk(self, **kw: Any) -> Any:  # pyre-ignore[3]
        args: Dict[str, Any] = dict(
            x=torch.zeros((300, 512), dtype=torch.bfloat16), norm_weight=torch.ones(512, dtype=torch.bfloat16),
            norm_bias=torch.zeros(512, dtype=torch.bfloat16), eps=1e-6, fp8_scale_value=0.01,
            uvqk_weight=torch.zeros((512, 2048)), uvqk_bias=torch.zeros(2048), u_columns=512, qkv_frozen=True,
            grad=True, store_t=True, pack_saved=True, pad_to=128, tiled_xt=True)
        args.update(kw)
        return self.mx6.uvqk_ln_store(**args)

    def vk(self) -> Any:  # pyre-ignore[3]
        return self.mx6.vk_ln_store(
            x=torch.zeros((300, 512), dtype=torch.bfloat16), norm_weight=torch.ones(512, dtype=torch.bfloat16),
            norm_bias=torch.zeros(512, dtype=torch.bfloat16), eps=1e-6, fp8_scale_value=0.01,
            uvqk_weight=torch.zeros((512, 2048)), uvqk_bias=torch.zeros(2048), vk_layout=((512, 1024), (1536, 2048)),
            qkv_frozen=True, grad=True, store_t=True, pack_handed=True, pad_to=128, tiled_xt=True)

    def tag_lines(self, text: str) -> List[str]:
        return [ln for ln in text.splitlines() if _TAG in ln]

    def assert_option_b_fallback_line(self, lines: List[str]) -> None:
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("[mx6-fwd] ERROR OPTION-C U NOT READY: N=512 is in "
                                            f"{_REAL_GEMM}.SUPPORTED_N"), lines[0])
        self.assertIn("-> FALLBACK TO OPTION B for this process", lines[0])
        self.assertIn("DLRM_HSTU_FP8_FWD_MXFP6_STRICT=1 makes this fatal", lines[0])


class GemmModuleLatchesUTest(_UCase):
    """The module's own contract (unchanged): U is optional there, its failure is latched, prebuild() succeeds."""

    def run_module_prebuild(self) -> str:
        with self.on_gfx950(), _capture() as out:
            self.assertTrue(self.G.prebuild())
        return out.getvalue()

    def test_u_build_failure_is_latched_and_prebuild_succeeds(self) -> None:
        self.u_build_fails = True
        text = self.run_module_prebuild()
        G = self.G
        self.assertIn("[k512-mx6] prebuilt: N=1024,1536 + weight shadow (U N=512: K=512 MXFP6 GEMM N=512 library "
                      "unavailable: hipcc rc=1", text)
        self.assertIn("hipcc rc=1", G.failure(512) or "")
        with self.assertRaisesRegex(RuntimeError, "library unavailable"):
            G.kernel(512)
        self.assertEqual(G.kernel(1536).n, 1536)
        self.assertNotIn(512, self.selfchecks)  # not built: not self-checked

    def test_u_selfcheck_failure_is_latched_and_prebuild_succeeds(self) -> None:
        self.u_selfcheck_fails = True
        text = self.run_module_prebuild()
        G = self.G
        self.assertIn("(U N=512: GEMM N=512 self-check on zeros wrote 7 nonzero bytes", text)
        self.assertIn("U self-check failed", G.failure(512) or "")
        with self.assertRaisesRegex(RuntimeError, "U self-check failed"):
            G.kernel(512)
        self.assertEqual(sorted(self.selfchecks), [512, 1024, 1536])


class StrictUPrebuildTest(_UCase):
    def check_strict(self, needle: str) -> None:
        ok, error, text = self.prebuild(strict=True)
        mx6 = self.mx6
        self.assertIsNone(ok)
        self.assertIsInstance(error, mx6.PrebuildError)
        self.assertIn(_TAG, str(error))
        self.assertIn(needle, str(error))
        self.assertIn("STRICT=1: fatal", str(error))
        declined = [ln for ln in text.splitlines() if ln.startswith("[mx6-fwd] prebuild DECLINED: " + _TAG)]
        self.assertEqual(len(declined), 1, text)
        self.assertNotIn("[mx6-fwd] prebuilt:", text)
        self.assertEqual(self.producer_selfchecks, 0)  # refused before the producer self-check
        self.assertIsNotNone(mx6._prebuild_error)
        self.assertIn("prebuild failed", mx6.site_reason(torch.device("cpu"), "VQK") or "")

    def test_u_build_failure_is_fatal(self) -> None:
        self.u_build_fails = True
        self.check_strict("hipcc rc=1")

    def test_u_selfcheck_failure_is_fatal(self) -> None:
        self.u_selfcheck_fails = True
        self.check_strict("U self-check failed")


class NonStrictUFallbackTest(_UCase):
    def check_fallback(self, needle: str) -> None:
        mx6 = self.mx6
        cpu = torch.device("cpu")
        ok, error, text = self.prebuild(strict=False)
        self.assertIsNone(error)
        self.assertTrue(ok, text)
        lines = self.tag_lines(text)
        self.assert_option_b_fallback_line(lines)
        self.assertIn(needle, lines[0])
        self.assertIn("(prebuild: ", lines[0])
        prebuilt = [ln for ln in text.splitlines() if ln.startswith("[mx6-fwd] prebuilt:")]
        self.assertEqual(len(prebuilt), 1, text)
        self.assertIn(" N=1536,1024 (U N=512: shipped FP8; ", prebuilt[0])
        self.assertLess(text.index(_TAG), text.index("[mx6-fwd] prebuilt:"))
        self.assertEqual(self.launches, [1536, 1024])  # the adapter self-checked VQK / VK only
        self.assertIsNone(mx6._prebuild_error)
        self.assertIsNone(mx6.site_reason(cpu, "VQK"))
        self.assertIsNone(mx6.site_reason(cpu, "VK"))
        self.assertIsNotNone(mx6.site_reason(cpu, "U"))
        with _capture() as out:
            plan = self.uvqk()
            plan2 = self.uvqk()
        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan2)
        self.assertFalse(plan.u_mx6)
        self.assertIsNotNone(plan.x8)  # option B: the producer writes x8 for the FP8 U
        self.assertIsNotNone(plan2.x8)
        later = out.getvalue()
        self.assertEqual(self.tag_lines(later), [])  # one loud line per process, already printed
        self.assertEqual(later.count("[mx6-fwd] U stays on the shipped FP8 GEMM"), 1, later)
        self.assertNotIn("DECLINED", later)
        counters = mx6.counters()
        self.assertEqual(counters.get("planned_uvqk"), 2)
        self.assertEqual(counters.get("declined", 0), 0)

    def test_u_build_failure_falls_back_to_option_b(self) -> None:
        self.u_build_fails = True
        self.check_fallback("hipcc rc=1")

    def test_u_selfcheck_failure_falls_back_to_option_b(self) -> None:
        self.u_selfcheck_fails = True
        self.check_fallback("U self-check failed")


class UNotReadyAtCallTimeTest(_UCase):
    """No prebuild (DLRM_HSTU_FP8_PREBUILD=0, or the latch came later): _plan is the last line of defence."""

    def test_strict_raises_on_a_lazy_u_build_failure(self) -> None:
        self.u_build_fails = True
        mx6 = self.mx6
        mx6.reset_for_testing(strict=True)
        with _capture() as out, self.assertRaises(mx6.StrictError) as ctx:
            self.uvqk()
        self.assertIn(_TAG, str(ctx.exception))
        self.assertIn("uvqk:U", str(ctx.exception))
        self.assertIn("hipcc rc=1", str(ctx.exception))
        self.assertIn("DECLINED -> shipped FP8 path (DLRM_HSTU_FP8_FWD_MXFP6=1): uvqk:U: " + _TAG, out.getvalue())
        self.assertEqual(self.prod.calls, [])  # raised before the producer ran
        self.assertEqual(mx6.counters().get("declined"), 1)
        self.assertIn(512, self.loads)
        with _capture():  # the targets-only layer has no U: unaffected even under STRICT
            self.assertIsNotNone(self.vk())

    def test_nonstrict_lazy_u_build_failure_runs_option_b(self) -> None:
        self.u_build_fails = True
        with _capture() as out:
            plan = self.uvqk()
            self.uvqk()
        self.assertIsNotNone(plan)
        self.assertFalse(plan.u_mx6)
        self.assertIsNotNone(plan.x8)
        lines = self.tag_lines(out.getvalue())
        self.assert_option_b_fallback_line(lines)
        self.assertIn("(uvqk call on cuda:-1: ", lines[0])
        self.assertEqual(self.mx6.counters().get("declined", 0), 0)
        self.assertEqual(self.mx6.counters().get("planned_uvqk"), 2)

    def test_latched_u_site_is_fatal_under_strict_and_option_b_otherwise(self) -> None:
        mx6 = self.mx6
        cpu = torch.device("cpu")
        for strict in (True, False):
            mx6.reset_for_testing(strict=strict)
            mx6._latch(cpu, "U", "guard rejected: rel_l2=0.9 tile=1.2 slope=0.5 nonfinite=0")
            with _capture() as out:
                if strict:
                    with self.assertRaises(mx6.StrictError) as ctx:
                        self.uvqk()
                    self.assertIn("site U latched to FP8 (guard rejected", str(ctx.exception))
                else:
                    plan = self.uvqk()
                    self.assertIsNotNone(plan)
                    self.assertFalse(plan.u_mx6)
                    self.assertIsNotNone(plan.x8)
                    self.assert_option_b_fallback_line(self.tag_lines(out.getvalue()))


class UnchangedBehaviourTest(_UCase):
    def test_option_c_with_u_ready(self) -> None:
        mx6 = self.mx6
        ok, error, text = self.prebuild(strict=True)
        self.assertIsNone(error)
        self.assertTrue(ok, text)
        self.assertIn(" N=1536,1024,512; self-check", text)
        self.assertEqual(self.tag_lines(text), [])
        self.assertEqual(self.launches, [1536, 1024, 512])
        self.assertTrue(mx6.u_expected())
        with _capture() as out:
            plan = self.uvqk()
        self.assertTrue(plan.u_mx6)
        self.assertIn("U", plan.guard)  # first call: guarded, x8 written for the FP8 references
        self.assertIsNotNone(plan.x8)
        self.assertEqual(out.getvalue(), "")

    def test_knob_off_is_untouched(self) -> None:
        mx6 = self.mx6
        self.u_build_fails = True
        mx6.reset_for_testing(enabled=False, strict=True)
        with self.on_gfx950(), _capture() as out:
            self.assertFalse(mx6.prebuild())
            self.assertIsNone(self.uvqk())
            self.assertIsNone(self.vk())
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(mx6.counters(), {})
        self.assertEqual(self.loads, [])  # nothing built
        self.assertEqual(dict(self.G._failures), {})
        self.assertNotIn("gemm", mx6._mods)  # the adapter never imported the GEMM module
        self.assertFalse(mx6.u_expected())
        self.assertEqual(self.prod.calls, [])


class OptionBUnchangedTest(_UCase):
    supported_n = (1024, 1536)  # option B: no U library source

    def test_option_b_keeps_u_on_fp8_without_a_new_line(self) -> None:
        mx6 = self.mx6
        for strict in (True, False):
            ok, error, text = self.prebuild(strict=strict)
            self.assertIsNone(error, text)
            self.assertTrue(ok, text)
            self.assertIn(f" N=1536,1024 (U N=512: shipped FP8; N=512 not in {_REAL_GEMM}.SUPPORTED_N=(1024, 1536))",
                          text)
            self.assertEqual(self.tag_lines(text), [])
            self.assertFalse(mx6.u_expected())
            with _capture() as out:
                plan = self.uvqk()
            self.assertIsNotNone(plan)
            self.assertFalse(plan.u_mx6)
            self.assertIsNotNone(plan.x8)
            self.assertEqual(self.tag_lines(out.getvalue()), [])
            self.assertIn("[mx6-fwd] U stays on the shipped FP8 GEMM (N=512 not in", out.getvalue())
            self.assertNotIn("DECLINED", out.getvalue())
            self.assertNotIn(512, self.loads)


if __name__ == "__main__":
    unittest.main()
