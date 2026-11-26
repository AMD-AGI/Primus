# pyre-unsafe
"""ops/flydsl/flydsl_loc.py: FlyDSL 0.2.4's per-op source-location capture, memoized.

CPU only (FlyDSL COMPILE_ONLY=1, no GPU, no disk cache). Skips when the memo is not installed (FlyDSL 0.1.1, or a
FlyDSL / CPython whose functions differ from the verified ones): there is nothing to compare then.
  * installed in the three modules that call capture_user_location, idempotently;
  * the IR a FlyDSL build emits -- printed WITH its debug locations, the CompiledArtifact's source IR -- is the
    same string with the memoized capture and with FlyDSL's own, for an LN finalize and an LN main kernel;
  * the kill switch value "0" leaves FlyDSL's function in place (fresh interpreter).
"""

import os
import subprocess
import sys
import unittest
from unittest import mock

_ENV = {"COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}


class FlydslLocTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import flydsl  # noqa: F401
        except ImportError:
            self.skipTest("flydsl is not installed")
        # FlyDSL resolves the arch once per process; the kernels here are gfx950-only (no GPU needed to compile).
        os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
        from generative_recommenders.ops.flydsl import flydsl_loc, ln_bwd

        self.fl, self.lb = flydsl_loc, ln_bwd
        if flydsl_loc.STATE != "on":
            self.skipTest(f"location memo not installed here: {flydsl_loc.STATE}")

    def _modules(self):
        from flydsl.compiler import ast_rewriter, kernel_function
        from flydsl.expr import meta

        return meta, kernel_function, ast_rewriter

    def test_installed_in_every_caller(self) -> None:
        meta, kernel_function, ast_rewriter = self._modules()
        fast = meta.capture_user_location
        self.assertTrue(getattr(fast, "_dlrm_loc_memo", False))
        self.assertIs(kernel_function.capture_user_location, fast)
        self.assertIs(ast_rewriter.capture_user_location, fast)
        self.assertEqual(self.fl.install(), "on")
        self.assertIs(meta.capture_user_location, fast)

    def _source_ir(self, capture, build):
        import torch

        mods = self._modules()
        saved = [m.capture_user_location for m in mods]
        for m in mods:
            m.capture_user_location = capture
        try:
            with mock.patch.dict(os.environ, _ENV):
                launch, args = build(torch)
                self.assertIsNone(launch(*args))       # COMPILE_ONLY: compiled, not launched
            arts = list(launch._mem_cache.values())
            self.assertEqual(len(arts), 1)
            return arts[0]._source_ir
        finally:
            for m, c in zip(mods, saved):
                m.capture_user_location = c

    def _check(self, build) -> None:
        meta = self._modules()[0]
        fast = meta.capture_user_location
        orig = fast.__wrapped__
        a = self._source_ir(fast, build)
        b = self._source_ir(orig, build)
        self.assertIn("ln_bwd.py", a)               # carries the user source locations
        self.assertGreater(a.count("loc("), 100)
        self.assertEqual(a, b)

    def test_same_ir_finalize(self) -> None:
        lb = self.lb

        def build(torch):
            # a fresh jit per build (bypass build_finalize's lru_cache), so each one compiles
            launch = lb.build_finalize.__wrapped__(512, 3, True)
            part = torch.zeros((3, 4, 512), dtype=torch.float32)
            outs = [torch.zeros((512,), dtype=torch.float32) for _ in range(3)]
            return launch, (part, *outs, 4, lb.fx.Stream(None))

        self._check(build)

    def test_same_ir_main_kernel(self) -> None:
        lb = self.lb

        def build(torch):
            key = (512, False, False, True)
            waves, unroll, tile_mult, *rest = lb.DEFAULT_CFG[key]
            remap, cache, store_cache = lb.kernel_options(*key)
            launch = lb.build_ln_bwd.__wrapped__(512, False, False, True, True, waves, unroll, cache,
                                                 bool(rest[0]) if rest else False, remap, store_cache)
            M, G = 256, 8
            x = torch.zeros((M, 512), dtype=torch.bfloat16)
            w = torch.ones((512,), dtype=torch.float32)
            st = torch.zeros((M,), dtype=torch.float32)
            part = torch.zeros((3, G, 512), dtype=torch.float32)
            return launch, (x, x, x, x, w, w, st, st, part, M, 1, G, lb.fx.Stream(None))

        self._check(build)

    def test_kill_switch(self) -> None:
        code = ("import os; os.environ.setdefault('FLYDSL_GPU_ARCH', 'gfx950')\n"
                "from generative_recommenders.ops.flydsl import flydsl_loc, ln_bwd\n"
                "from flydsl.expr import meta\n"
                "print(flydsl_loc.STATE, getattr(meta.capture_user_location, '_dlrm_loc_memo', False))\n")
        env = dict(os.environ, DLRM_FLYDSL_LOC_MEMO="0")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "off False")


if __name__ == "__main__":
    unittest.main()
