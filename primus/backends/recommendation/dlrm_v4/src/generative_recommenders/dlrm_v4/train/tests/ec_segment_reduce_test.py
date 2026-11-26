"""Exact EC metadata and FP32 reduction gates for all four policy choices."""

import bisect
import itertools
import random
import unittest

import torch
import triton
import triton.language as tl

from generative_recommenders.dlrm_v4.train import ec_gpu_plan as gpu
from generative_recommenders.dlrm_v4.train import ec_offsets as offsets
from generative_recommenders.dlrm_v4.train import ec_segment_reduce as ec


POLICIES = tuple(itertools.product((False, True), repeat=2))


@triton.jit
def _read_hot_pieces(OFF, PREFIX, ROW, LO, HI, U, LIMIT: tl.constexpr):
    piece = tl.program_id(0)
    row, lo, hi = gpu._piece_location(OFF, PREFIX, piece, U, LIMIT)
    tl.store(ROW + piece, row)
    tl.store(LO + piece, lo)
    tl.store(HI + piece, hi)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class ECSegmentReduceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(230923)
        ec.configure_segment_reduce()

    def tearDown(self):
        torch.cuda.synchronize()
        ec.configure_segment_reduce()

    def test_exact_metadata_piece_boundaries_and_reductions(self):
        rng = random.Random(230923)
        cases = (
            ([], 17), ([0] * 17, 17), ([0, 0, 3, 0, 7, 0, 0], 17),
            ([0, 0, 4097, 0], 128), ([1] * 257, 17),
            ([0, 1, 31, 32, 33, 1023, 1024, 1025, 2048, 2049, 0, 9000], 33),
            ([rng.randrange(4097) for _ in range(257)], 17),
        )
        for lengths, width in cases:
            for dtype in (torch.int32, torch.int64):
                with self.subTest(lengths=len(lengths), width=width, dtype=dtype):
                    rows = len(lengths)
                    ids = torch.repeat_interleave(
                        torch.arange(rows, device="cuda"),
                        torch.tensor(lengths, device="cuda", dtype=torch.int64),
                    )
                    ids = ids[torch.randperm(ids.numel(), device="cuda")].to(dtype)
                    expected = [0] + list(itertools.accumulate(lengths))
                    ec.configure_segment_reduce()
                    order, bounds = ec.build_segments(ids, rows)
                    self.assertEqual(bounds.cpu().tolist(), expected)
                    before_ids, before_order = ids.clone(), order.clone()
                    integer = torch.randint(
                        -4, 5, (ids.numel(), width), device="cuda"
                    ).to(torch.bfloat16)
                    truth = torch.zeros((rows, width), device="cuda", dtype=torch.float64)
                    truth.index_add_(0, ids.long(), integer.double())
                    values = torch.randn_like(integer)
                    reference = torch.zeros_like(truth)
                    reference.index_add_(0, ids.long(), values.double())
                    control = ec.segment_reduce_rows(values, ids, rows)
                    control_error = (
                        (control.double() - reference).abs().max().item()
                        if control.numel() else 0.0
                    )
                    for planner, fixed_offsets in POLICIES:
                        ec.configure_segment_reduce(
                            gpu_planner=planner, fixed_size_offsets=fixed_offsets
                        )
                        actual_order, actual_bounds = ec.build_segments(ids, rows)
                        self.assertTrue(torch.equal(actual_order, order))
                        self.assertTrue(torch.equal(actual_bounds, bounds))
                        result = ec.segment_reduce_rows(integer, ids, rows)
                        self.assertTrue(torch.equal(result.double(), truth))
                        result = ec.segment_reduce_rows(values, ids, rows)
                        self.assertTrue(bool(torch.isfinite(result).all()))
                        error = (
                            (result.double() - reference).abs().max().item()
                            if result.numel() else 0.0
                        )
                        self.assertLessEqual(error, max(4 * control_error, 2e-5))
                    self.assertTrue(torch.equal(ids, before_ids))
                    self.assertTrue(torch.equal(order, before_order))
                    prefix = gpu.make_prefix(bounds, rows, 1024)
                    pieces = [
                        (row, lo, min(lo + 1024, expected[row + 1]))
                        for row, length in enumerate(lengths) if length > 1024
                        for lo in range(expected[row], expected[row + 1], 1024)
                    ]
                    if prefix is not None:
                        counts = [(n + 1023) // 1024 if n > 1024 else 0 for n in lengths]
                        self.assertEqual(prefix.cpu().tolist(), list(itertools.accumulate(counts)))
                    self.assertLessEqual(len(pieces), gpu.hot_capacity(ids.numel(), 1024))
                    if pieces:
                        actual = [torch.empty(len(pieces), device="cuda", dtype=torch.int64) for _ in range(3)]
                        _read_hot_pieces[(len(pieces),)](bounds, prefix, *actual, rows, 1024)
                        self.assertEqual(list(zip(*(x.cpu().tolist() for x in actual))), pieces)

    def test_offsets_layouts_supplied_order_and_redzones(self):
        source = torch.arange(96, device="cuda", dtype=torch.int64).remainder_(13)
        for ids in (source.reshape(8, 12), source.reshape(8, 12).T,
                    source[::3], source.reshape(8, 12)[:, ::3]):
            ec.configure_segment_reduce()
            order, expected = ec.build_segments(ids, 13)
            values = sorted(ids.reshape(-1).cpu().tolist())
            self.assertEqual(expected.cpu().tolist(), [bisect.bisect_left(values, i) for i in range(14)])
            actual_order, actual = offsets.build_segments_valid_ids(ids, 13, order=order)
            self.assertIs(actual_order, order)
            self.assertTrue(torch.equal(actual, expected))
            guarded = torch.full((16,), -77123456789, device="cuda", dtype=torch.int64)
            offsets.offsets_from_order_valid_ids(ids, order, 13, out=guarded[1:-1])
            self.assertTrue(torch.equal(guarded[1:-1], expected))
            self.assertEqual(guarded[0].item(), -77123456789)
            self.assertEqual(guarded[-1].item(), -77123456789)

    def test_current_stream_lifetime_and_disabled_splitting(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        for planner, fixed_offsets in POLICIES:
            ec.configure_segment_reduce(
                gpu_planner=planner, fixed_size_offsets=fixed_offsets, split_above=0
            )
            with torch.cuda.stream(producer):
                ids = torch.arange(9001, device="cuda", dtype=torch.int64).remainder_(7)
                grad = torch.ones((9001, 33), device="cuda", dtype=torch.bfloat16)
                result = ec.segment_reduce_rows(grad, ids, 7)
                ready = torch.cuda.Event()
                ready.record(producer)
                del ids, grad
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready)
                result.record_stream(consumer)
                copied = result.clone()
            del result
            with torch.cuda.stream(producer):
                pressure = [torch.empty((9001, 33), device="cuda") for _ in range(8)]
            consumer.synchronize()
            expected = torch.tensor([1286] * 6 + [1285], device="cuda").view(7, 1).expand(7, 33)
            self.assertTrue(torch.equal(copied, expected))
            del pressure

    def test_invalid_metadata_and_policy_values(self):
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        for ids, rows in ((empty, -1), (torch.zeros(1, device="cuda", dtype=torch.int64), 0),
                          (empty.float(), 1)):
            with self.assertRaises(ValueError):
                offsets.build_segments_valid_ids(ids, rows)
        ec.configure_segment_reduce(gpu_planner=True, fixed_size_offsets=True)
        for name in ("gpu_planner", "fixed_size_offsets"):
            for value in (0, 1, "False", None):
                with self.assertRaises(ValueError):
                    ec.configure_segment_reduce(**{name: value})
                self.assertTrue(ec._GPU_PLANNER)
                self.assertTrue(ec._FIXED_SIZE_OFFSETS)


if __name__ == "__main__":
    unittest.main()
