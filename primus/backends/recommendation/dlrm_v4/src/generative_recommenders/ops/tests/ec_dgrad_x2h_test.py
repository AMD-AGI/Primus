"""X2H on the GPU (gfx950 + Triton + hipcc; skipped elsewhere): the real prebuild (build, census, exact
check, bitwise cross-check), then dgrad_segreduce with X2H armed against the Triton passes, bit for bit:
other seeds and grids, a strided dh, non-finite values (NaN payloads included), an empty member, a
second stream; and the per-call decline of a misaligned dh. CPU half: ec_dgrad_x2h_cpu_test.py.
"""

import os
import unittest

import torch


def _gfx950() -> bool:
    try:
        return (torch.cuda.is_available() and torch.version.hip is not None
                and str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950")
    except Exception:
        return False


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class X2HGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import triton  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("needs Triton")
        from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h
        from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

        cls.x2h, cls.xs = x2h, xs
        cls._env = os.environ.get(x2h.ENV_VAR)
        os.environ[x2h.ENV_VAR] = "1"
        x2h.reset_for_testing()
        xs._X2H_REQUESTED = None
        cls.dev = torch.device("cuda", torch.cuda.current_device())
        cls.ok = x2h.prebuild(cls.dev)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._env is None:
            os.environ.pop(cls.x2h.ENV_VAR, None)
        else:
            os.environ[cls.x2h.ENV_VAR] = cls._env
        cls.x2h.reset_for_testing()
        cls.xs._X2H_REQUESTED = None

    def same(self, cat, item, u, expect_x2h=True):
        xs = self.xs
        # sentinel-filled outputs on both sides: a skipped write cannot pass on recycled memory
        ref = xs.dgrad_segreduce(cat, item, u, _impl="triton", _prefill=True)
        got = xs.dgrad_segreduce(cat, item, u, _prefill=True)
        self.assertEqual(xs.LAST_PASS_IMPL, "x2h_pass" if expect_x2h else "_ecx_dgrad_pass")
        torch.cuda.synchronize()
        bad = int((got.view(torch.int32) != ref.view(torch.int32)).sum())
        self.assertEqual(bad, 0, f"{bad} acc words differ")
        return got

    def test_prebuild_armed(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        info = self.x2h.PREBUILD_INFO[self.dev.index]
        res = info["census"]
        self.assertTrue(self.x2h.census_ok(res), res)
        self.assertGreater(info["cat"]["tiles"], 6 * info["grid"])
        self.assertTrue(self.x2h.active(self.dev))

    def test_inputs_bitwise(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        xs, x2h = self.xs, self.x2h
        self.same(*xs._synthetic(self.dev))
        for grid, seed in ((13, 99), (40, 7)):
            cat, item, u, _ = x2h.crosscheck_input(self.dev, grid, xs.Member, seed=seed)
            self.same(cat, item, u)

    def test_strided_dh_and_empty_member(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        xs, x2h = self.xs, self.x2h
        cat, item, u, _ = x2h.crosscheck_input(self.dev, 5, xs.Member, seed=5)
        wide = torch.zeros(item.dh.shape[0], 264, dtype=torch.bfloat16, device=self.dev)
        wide[:, :256] = item.dh
        self.same(cat, item._replace(dh=wide[:, :256]), u)  # row stride 264
        e = cat._replace(dh=cat.dh[:0], idx=cat.idx[:0], order=cat.order[:0],
                         offsets=torch.zeros_like(cat.offsets))
        self.same(e, item, u)

    def test_nonfinite_bitwise(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        xs, x2h = self.xs, self.x2h
        cat, item, u, _ = x2h.crosscheck_input(self.dev, 6, xs.Member, seed=11)
        specials = torch.tensor([0x7F80, -0x0080, 0x7FC1, -0x003F, 0x7F81], dtype=torch.int16,
                                device=self.dev)  # +inf, -inf, NaN payloads (bf16 bits)
        for m in (cat, item):
            v = m.dh.view(torch.int16).view(-1)
            pos = (torch.arange(40, device=self.dev) * 104729) % v.numel()
            v[pos] = specials[torch.arange(40, device=self.dev) % specials.numel()]
        self.same(cat, item, u)

    def test_second_stream(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        cat, item, u = self.xs._synthetic(self.dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.same(cat, item, u)
        torch.cuda.current_stream().wait_stream(s)

    def test_misaligned_dh_declines(self) -> None:
        self.assertTrue(self.ok, self.x2h.broken())
        cat, item, u = self.xs._synthetic(self.dev)
        buf = torch.empty(cat.dh.numel() + 8, dtype=torch.bfloat16, device=self.dev)
        mis = buf[4:4 + cat.dh.numel()].view_as(cat.dh)
        mis.copy_(cat.dh)
        self.same(cat._replace(dh=mis), item, u, expect_x2h=False)
        self.assertIsNone(self.x2h.broken())


if __name__ == "__main__":
    unittest.main()
