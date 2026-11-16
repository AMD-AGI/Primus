"""Raw-bit gates for gin item (49): EC_REDUCE_FUSED (ec_gpu_plan.segment_reduce_grouped)
against the per-member EC_REDUCE_LPT calls, EC_GATHER_WIDE's launch geometry against
torch.gather, and EC_JOINT_ONESWEEP (ops/onesweep_sort.ec_joint_segments) against the item
(41)(d) joint chain."""

import unittest

import torch

from generative_recommenders.dlrm_v4.train import ec_expand_gather as eg
from generative_recommenders.dlrm_v4.train import ec_gpu_plan as gpu
from generative_recommenders.dlrm_v4.train import ec_offsets as offsets
from generative_recommenders.dlrm_v4.train import ec_segment_reduce as ec
from generative_recommenders.ops import onesweep_sort as osort


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _member(lengths, width, seed, dtype=torch.bfloat16, specials=False, stride=None):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = len(lengths)
    ln = torch.tensor(lengths, dtype=torch.int64)
    ids = torch.repeat_interleave(torch.arange(rows), ln)
    ids = ids[torch.randperm(ids.numel(), generator=gen)]
    tokens = ids.numel()
    scale = torch.exp2(torch.randint(-12, 13, (tokens, width), generator=gen).float())
    vals = (torch.randn(tokens, width, generator=gen) * scale).to(dtype)
    if specials and tokens:
        bits = vals.view(torch.int16)
        sp = torch.tensor([-32768, 0x7F80, -128, 0x0001, 0x7FC1 + seed % 7, -63 - seed % 5],
                          dtype=torch.int16)
        pick = torch.randint(0, tokens, (tokens // 4 + 1,), generator=gen)
        col = torch.randint(0, width, (pick.numel(),), generator=gen)
        bits[pick, col] = sp[torch.randint(0, sp.numel(), (pick.numel(),), generator=gen)]
    stride = stride or width
    grad = torch.zeros((tokens, stride), dtype=dtype, device="cuda")[:, :width]
    grad.copy_(vals.cuda())
    order, bounds = offsets.build_segments_valid_ids(ids.cuda(), rows)
    return grad, order, bounds


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class ECReduceFusedTest(unittest.TestCase):
    def setUp(self):
        ec.configure_segment_reduce(gpu_planner=True, fixed_size_offsets=True)
        self._saved = (gpu._LPT_ENABLED, gpu._FUSED_ENABLED)
        gpu._LPT_ENABLED, gpu._FUSED_ENABLED = True, True

    def tearDown(self):
        torch.cuda.synchronize()
        gpu._LPT_ENABLED, gpu._FUSED_ENABLED = self._saved
        ec.configure_segment_reduce()

    def _per_member(self, members, rows):
        acc = None
        for g, o, f in members:
            acc = gpu.segment_reduce_rows(g, None, rows, order=o, offsets=f, accumulate_into=acc)
        return acc

    def test_cold_and_two_piece_rows_bitwise(self):
        limit = ec._SPLIT_ABOVE
        gen = torch.Generator().manual_seed(34)
        for width in (512, 128):
            for overlap in (False, True):
                for specials in (False, True):
                    with self.subTest(width=width, overlap=overlap, specials=specials):
                        rows = 96
                        la = torch.randint(0, 80, (rows,), generator=gen)
                        lb = torch.randint(0, 80, (rows,), generator=gen)
                        if not overlap:  # disjoint member rows, as in production
                            la[rows // 2:] = 0
                            lb[: rows // 2] = 0
                        if not specials:  # deterministic hot rows: two pieces, one member
                            la[1], lb[1] = limit + 7, 0
                            la[rows - 2], lb[rows - 2] = 0, 2 * limit
                        mem = [_member(la.tolist(), width, 1, specials=specials),
                               _member(lb.tolist(), width, 2, specials=specials)]
                        got = gpu.segment_reduce_grouped(mem, rows)
                        ref = self._per_member(mem, rows)
                        one = gpu.segment_reduce_grouped(mem[:1], rows)
                        if width == 512:
                            # the production width must engage (offline-validated key at
                            # BLOCK_S=32 / SPLIT_ABOVE=1024, else the one-time self-test)
                            self.assertIsNotNone(got)
                            self.assertIsNotNone(one)
                        if got is None:
                            # other widths: only a one-time self-test decline may return None
                            self.assertIn(None, gpu._FUSED_OK.values())
                            continue
                        self.assertTrue(torch.equal(_bits(got), _bits(ref)))
                        if one is not None:
                            self.assertTrue(torch.equal(_bits(one), _bits(self._per_member(mem[:1], rows))))

    def test_production_width_engages(self):
        rows = 64
        mem = [_member([3] * 32 + [0] * 32, 512, 3), _member([0] * 32 + [5] * 32, 512, 4)]
        got = gpu.segment_reduce_grouped(mem, rows)
        self.assertIsNotNone(got)
        self.assertTrue(torch.equal(_bits(got), _bits(self._per_member(mem, rows))))

    def test_production_key_nan_payload_pin_has_power(self):
        # Overlapping member rows with NaN payloads in both members (production rows are
        # disjoint; this pins the member-1 add to LPT's compiled operand order anyway).
        rows, width = 96, 512
        gen = torch.Generator().manual_seed(35)
        mem = [_member(torch.randint(1, 60, (rows,), generator=gen).tolist(), width, 6, specials=True),
               _member(torch.randint(1, 60, (rows,), generator=gen).tolist(), width, 7, specials=True)]
        # power: rows 0..9, columns 0..63 are NaN in BOTH members with different payloads
        for k, (g, o, f) in enumerate(mem):
            toks = o[f[0]:f[10]]
            g.view(torch.int16)[toks, :64] = 0x7FC1 + 16 * k
        ref = self._per_member(mem, rows)
        self.assertTrue(bool(torch.isnan(ref[:10, :64]).all()))
        got = gpu.segment_reduce_grouped(mem, rows)
        self.assertIsNotNone(got)
        self.assertTrue(torch.equal(_bits(got), _bits(ref)))
        key = (str(got.device), ec._BLOCK_S, ec._SPLIT_ABOVE, width, 512,
               (torch.bfloat16, torch.bfloat16), (False, False), (True, True), (True, True))
        src0 = gpu._FUSED_OK[key]
        flipped = gpu._ecf_launch(mem, rows, True, not src0)
        self.assertGreater(int((_bits(flipped) != _bits(ref)).sum()), 0)  # the pin has power

    def test_disabled_and_lpt_off_return_none(self):
        mem = [_member([2, 3, 0, 1], 512, 5)]
        gpu._FUSED_ENABLED = False
        self.assertIsNone(gpu.segment_reduce_grouped(mem, 4))
        gpu._FUSED_ENABLED, gpu._LPT_ENABLED = True, False
        self.assertIsNone(gpu.segment_reduce_grouped(mem, 4))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class ECGatherWideTest(unittest.TestCase):
    def test_wide_geometry_equals_torch_gather(self):
        gen = torch.Generator(device="cuda").manual_seed(34)
        for width in (128, 512, 600):
            src = torch.randn(1000, width, device="cuda", generator=gen).to(torch.bfloat16)
            for n in (0, 1, 63, 65, 4097):
                idx = torch.randint(0, 1000, (n,), device="cuda", generator=gen)
                with self.subTest(width=width, n=n):
                    ref = torch.gather(src, 0, idx.unsqueeze(1).expand(-1, width))
                    got = eg.expand_rows(src, idx, rows=64, num_warps=16)
                    self.assertTrue(torch.equal(_bits(got), _bits(ref)))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class ECJointOnesweepTest(unittest.TestCase):
    """Item (49)(c): the joint EC segments on the onesweep extension equal the item (41)(d)
    chain (ec_offsets._joint_incumbent): order and offsets raw-bit equal, int64, contiguous and
    16-byte aligned; and the caller keeps the (41)(d) chain when the extension is unavailable."""

    def _check(self, a, b, num_rows):
        got = osort.ec_joint_segments(a, b, num_rows)
        self.assertIsNotNone(got)
        ref = offsets._joint_incumbent(a, b, num_rows)
        for m in range(2):
            for x, y in zip(got[m], ref[m]):
                self.assertEqual((x.dtype, tuple(x.shape)), (torch.int64, tuple(y.shape)))
                self.assertTrue(x.is_contiguous())
                self.assertEqual(x.data_ptr() % 16, 0)
                self.assertTrue(torch.equal(x, y))  # int64: raw bits

    def test_matches_joint_chain(self):
        gen = torch.Generator(device="cuda").manual_seed(3403)
        # U at the key-bit (2^k) and 9-bit digit-pass boundaries; odd and even member-0 sizes
        for num_rows in (1, 2, 255, 256, 257, 65536, 131071, 131072, 131073, 262145):
            for ta, tb in ((4097, 4100), (30001, 70000), (70000, 30001)):
                a = torch.randint(0, num_rows, (ta,), device="cuda", generator=gen)
                b = torch.randint(0, num_rows, (tb,), device="cuda", generator=gen)
                with self.subTest(num_rows=num_rows, ta=ta, tb=tb):
                    self._check(a, b, num_rows)
        # long runs of absent ids (more than 4096 empty rows between present ids)
        num_rows = 300_001
        a = torch.randint(0, 10, (7001,), device="cuda", generator=gen)
        b = torch.randint(num_rows - 10, num_rows, (9000,), device="cuda", generator=gen)
        with self.subTest(case="gaps"):
            self._check(a, b, num_rows)
            self._check(b, a, num_rows)
        # strided member ids (views of larger id tensors)
        big = torch.randint(0, 5000, (40002,), device="cuda", generator=gen)
        with self.subTest(case="strided"):
            self._check(big[::2], big[1:9002], 5000)

    def test_declines(self):
        ids = torch.zeros(6000, dtype=torch.int64, device="cuda")
        self.assertIsNone(osort.ec_joint_segments(ids.int(), ids, 10))
        self.assertIsNone(osort.ec_joint_segments(ids, ids[:0], 10))
        self.assertIsNone(osort.ec_joint_segments(ids, ids, 2**30))  # 2U > 2^31 - 1
        self.assertEqual(osort._MAX_ITEMS, 1 << 30)  # one onesweep batch
        saved = osort._MAX_ITEMS
        try:
            osort._MAX_ITEMS = 12000
            self.assertIsNone(osort.ec_joint_segments(ids, ids, 10))
            self.assertIsNotNone(osort.ec_joint_segments(ids, ids[:5999], 10))
        finally:
            osort._MAX_ITEMS = saved

    def test_extension_unavailable_keeps_joint_chain(self):
        self.assertIsNotNone(osort.extension())
        gen = torch.Generator(device="cuda").manual_seed(3404)
        a = torch.randint(0, 1000, (5001,), device="cuda", generator=gen)
        b = torch.randint(0, 1000, (6000,), device="cuda", generator=gen)
        ref = offsets._joint_incumbent(a, b, 1000)
        saved = (osort._EXT, osort._FAILED, offsets._JOINT_ONESWEEP)
        try:
            osort._EXT, osort._FAILED = None, "forced by the test"
            offsets._JOINT_ONESWEEP = True
            got = offsets.build_segments_joint_valid_ids([a, b], 1000)
            self.assertFalse(offsets._JOINT_ONESWEEP)  # failed closed for the process
            for m in range(2):
                for x, y in zip(got[m], ref[m]):
                    self.assertTrue(torch.equal(x, y))
        finally:
            osort._EXT, osort._FAILED, offsets._JOINT_ONESWEEP = saved


if __name__ == "__main__":
    unittest.main()
