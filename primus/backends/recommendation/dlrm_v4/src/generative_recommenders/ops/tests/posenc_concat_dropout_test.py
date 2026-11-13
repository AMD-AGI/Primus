# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-unsafe

"""Bitwise tests of the fused posenc concat + HSTU input dropout
(DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE): design A (_posenc_concat_dropout_kernel)
and the V2 layout (_posenc_concat_dropout_v2_kernel,
DLRM_HSTU_FP8_POSENC_DROPOUT_V2) against the unfused pair
_AddTimestampPositionEmbeddingsConcatFunction + aten::native_dropout(x, p, True).
All comparisons are on raw bits (integer views): y, the bool mask, the CUDA
generator offset afterwards, the next torch.rand draw, the saved sorted
indices, and exact-sum gradients. Also: V2's first-use self-test is
RNG-neutral and fails closed to design A. DLRM_HSTU_FP8_DROPOUT_MASK_BITS
is off by default in code (the gin binding turns it on): run with =1 to test
the packed-mask layout (as a gin-armed run does) and unset or =0 for the bool
layout. DLRM_HSTU_FP8_POSENC_DROP_HWEPI (gfx950 only) is off by default too:
run with =1 to put V2's HW_EPI compile (hardware bf16 epilogue, v_bitop3
Philox rounds) under every V2 check; test_hw_epilogue_* force it on and off.
Run
this file after any Triton / ROCm image change: the packed LN-backward fold's
tripwire (test_mask_bits_ln_bwd_reduce_layouts) is a GPU test, and the
runtime self-test (test_mask_bits_fold_self_test) does not cover the
production > 2 GiB specialization."""

import logging
import os
import subprocess
import sys
import unittest
from typing import Dict, List, Optional

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.ops.input_dropout_fold import (
    pack_dropout_mask_bits,
    unpack_dropout_mask_bits,
)

_SPECIAL_BF16 = [
    0x7FC1,  # NaN with payload
    0xFFC3,  # negative NaN with payload
    0x7F80,  # +Inf
    0xFF80,  # -Inf
    0x8000,  # -0
    0x0001,  # smallest subnormal
    0x807F,  # negative subnormal
    0x7F7F,  # bf16 max
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(
        {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()
        ]
    )


def _nbad(a: torch.Tensor, b: torch.Tensor) -> int:
    assert a.shape == b.shape, (a.shape, b.shape)
    return int((_bits(a) != _bits(b)).sum().item())


def _make_inputs(
    right_lens: List[int],
    ctx_len: int,
    D: int,
    seed: int,
    noncontig: bool = False,
    special: bool = False,
    targets: bool = True,
    n_pos: int = 300,
    n_ts: int = 129,
) -> Dict:
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    B = len(right_lens)
    rl = torch.tensor(right_lens, dtype=torch.int64, device=dev)
    seq_lengths = rl + ctx_len
    seq_offsets = torch.zeros(B + 1, dtype=torch.int64, device=dev)
    seq_offsets[1:] = torch.cumsum(seq_lengths, 0)
    N = int(seq_offsets[-1].item())
    NR = int(rl.sum().item())
    timestamps = torch.cumsum(
        torch.randint(0, 20000, (N,), device=dev, generator=g), 0
    )
    num_targets: Optional[torch.Tensor] = None
    if targets:
        nt = torch.randint(1, 4, (B,), device=dev, generator=g)
        num_targets = torch.minimum(nt, rl)
    if noncontig:
        left = (torch.randn(B * ctx_len, D + 64, device=dev, generator=g) * 0.05).to(
            torch.bfloat16
        )[:, :D]
        right = (torch.randn(NR, 2 * D, device=dev, generator=g) * 0.05).to(
            torch.bfloat16
        )[:, D:]
        pos = torch.empty(n_pos, D + 32, device=dev).uniform_(
            -0.011, 0.011, generator=g
        )[:, 16 : 16 + D]
        ts = torch.empty(n_ts, D + 16, device=dev).uniform_(
            -0.022, 0.022, generator=g
        )[:, :D]
    else:
        left = (torch.randn(B * ctx_len, D, device=dev, generator=g) * 0.05).to(
            torch.bfloat16
        )
        right = (torch.randn(NR, D, device=dev, generator=g) * 0.05).to(
            torch.bfloat16
        )
        pos = torch.empty(n_pos, D, device=dev).uniform_(-0.011, 0.011, generator=g)
        ts = torch.empty(n_ts, D, device=dev).uniform_(-0.022, 0.022, generator=g)
    if special:
        vals = torch.tensor(_SPECIAL_BF16, dtype=torch.int32, device=dev).to(
            torch.int16
        )
        for t in (left, right):
            if t.numel() == 0:
                continue
            r = torch.randint(0, t.shape[0], (max(1, t.shape[0] // 3),), device=dev, generator=g)
            c = torch.randint(0, D, r.shape, device=dev, generator=g)
            t.view(torch.int16)[r, c] = vals[
                torch.randint(0, len(vals), r.shape, device=dev, generator=g)
            ]
        pr = torch.randint(0, n_pos, (50,), device=dev, generator=g)
        pc = torch.randint(0, D, (50,), device=dev, generator=g)
        pos[pr, pc] = float("nan")
        pos[pr[:20], pc[:20]] = -0.0
        ts[pr[:30] % n_ts, pc[:30]] = float("inf")
    for t in (left, right, pos, ts):
        t.requires_grad_(True)
    return dict(
        values_left=left,
        values_right=right,
        seq_offsets=seq_offsets,
        pos_embeddings=pos,
        ts_embeddings=ts,
        timestamps=timestamps,
        max_seq_len=max(1, int(seq_lengths.max().item())),
        max_contextual_seq_len=ctx_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=False,
        time_bucket_fn="sqrt",
        alpha=float(D) ** 0.5,
    )


def _args(inp: Dict):
    return (
        inp["values_left"],
        inp["values_right"],
        inp["seq_offsets"],
        inp["pos_embeddings"],
        inp["ts_embeddings"],
        inp["timestamps"],
        inp["max_seq_len"],
        inp["max_contextual_seq_len"],
        inp["seq_lengths"],
        inp["num_targets"],
        inp["interleave_targets"],
        inp["time_bucket_fn"],
        inp["alpha"],
    )


def _set_rng(seed: int, offset: int) -> torch.Generator:
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    gen.manual_seed(seed)
    gen.set_offset(offset)
    return gen


class PosencConcatDropoutTest(unittest.TestCase):
    P = 0.2

    def setUp(self) -> None:
        from generative_recommenders.ops.triton import (
            triton_layer_norm as tln,
            triton_position as tp,
        )

        self.tp = tp
        self.tln = tln
        self._v2 = tp._POSENC_DROPOUT_V2
        self._bits = tp._DROPOUT_MASK_BITS
        self._ok = dict(tp._POSENC_DROP_V2_OK)
        # DLRM_HSTU_FP8_DROPOUT_MASK_BITS packs the mask only where the layer-0
        # LN-backward fold consumes it (Triton v2 LN backward, gin item (14)):
        # arm v2 as production does, so a knob-ON run tests the packed layout
        self._wln_v2 = tln._WLN_BWD_V2_ENABLED
        tln._WLN_BWD_V2_ENABLED = True
        # ... and the fused residual of the FP8 forward arm, the only wiring in
        # which the layer-0 STU preprocess claims the handoff (production)
        from generative_recommenders.ops.triton import triton_fp8 as tf8

        self.tf8 = tf8
        self._fuse = (tf8._HSTU_FP8_FUSE_RESIDUAL, tf8._HSTU_FP8_FUSE_LN)
        tf8._HSTU_FP8_FUSE_RESIDUAL = True
        tf8._HSTU_FP8_FUSE_LN = True
        self._fold_ok = dict(tp._MASK_BITS_FOLD_OK)
        self._hw = tp._POSENC_DROP_HWEPI
        self._hw_ok = dict(tp._POSENC_DROP_HWEPI_OK)

    def tearDown(self) -> None:
        self.tp._POSENC_DROP_HWEPI = self._hw
        self.tp._POSENC_DROP_HWEPI_OK.clear()
        self.tp._POSENC_DROP_HWEPI_OK.update(self._hw_ok)
        self.tf8._HSTU_FP8_FUSE_RESIDUAL, self.tf8._HSTU_FP8_FUSE_LN = self._fuse
        self.tp._POSENC_DROPOUT_V2 = self._v2
        self.tp._DROPOUT_MASK_BITS = self._bits
        self.tp._POSENC_DROP_V2_OK.clear()
        self.tp._POSENC_DROP_V2_OK.update(self._ok)
        self.tp._MASK_BITS_FOLD_OK.clear()
        self.tp._MASK_BITS_FOLD_OK.update(self._fold_ok)
        self.tln._WLN_BWD_V2_ENABLED = self._wln_v2

    def _gfx950(self) -> bool:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return str(getattr(props, "gcnArchName", "") or "").startswith("gfx950")

    def _expect_bits(self, attach: bool, D: int) -> bool:
        # the packed envelope, stated independently of the code under test:
        # knob on, a handoff, LN-backward v2, D % 8 == 0, BLOCK_D <= 512
        return bool(
            self.tp._DROPOUT_MASK_BITS
            and attach
            and self.tln._WLN_BWD_V2_ENABLED
            and self.tf8._HSTU_FP8_FUSE_RESIDUAL
            and self.tf8._HSTU_FP8_FUSE_LN
            and D % 8 == 0
            and 1 << (D - 1).bit_length() <= 512
        )

    def _stock(self, inp: Dict, seed: int, offset: int):
        tp = self.tp
        gen = _set_rng(seed, offset)
        with torch.enable_grad():
            x = tp._AddTimestampPositionEmbeddingsConcatFunction.apply(*_args(inp))
            saved = x.grad_fn.saved_tensors
            y, m = torch.ops.aten.native_dropout(x, self.P, True)
        off = gen.get_offset()
        nxt = torch.rand(4096, device="cuda")
        return y, m, saved, off, nxt

    def _fused(self, inp: Dict, seed: int, offset: int, v2: bool, attach: bool):
        tp = self.tp
        tp._POSENC_DROPOUT_V2 = v2
        gen = _set_rng(seed, offset)
        with torch.enable_grad():
            y = tp._AddTimestampPositionEmbeddingsConcatDropoutFunction.apply(
                *_args(inp), self.P, attach
            )
        saved = y.grad_fn.saved_tensors
        if attach:
            from generative_recommenders.ops.input_dropout_fold import (
                DROPOUT_HANDOFF_ATTR,
            )

            m = getattr(y, DROPOUT_HANDOFF_ATTR).mask
        else:
            m = saved[5]
        off = gen.get_offset()
        nxt = torch.rand(4096, device="cuda")
        return y, m, saved, off, nxt

    def _check(
        self,
        inp: Dict,
        seed: int,
        offset: int,
        tag: str,
        attach: bool = False,
        expect_v2_used: bool = True,
        expect_hw: Optional[bool] = None,
    ) -> None:
        # DLRM_HSTU_FP8_POSENC_DROP_HWEPI: V2 runs its HW_EPI compile where the
        # knob is on and the device is gfx950 (unless a test breaks it)
        if expect_hw is None:
            expect_hw = bool(self.tp._POSENC_DROP_HWEPI and self._gfx950())
        ys, ms, ss, offs, nxs = self._stock(inp, seed, offset)
        D = ms.shape[1]
        # DLRM_HSTU_FP8_DROPOUT_MASK_BITS: the saved-mask path (attach=False)
        # keeps the bool mask; the handoff path is packed (_expect_bits)
        attaches = (
            (attach, True) if self.tp._DROPOUT_MASK_BITS and not attach else (attach,)
        )
        for att in attaches:
            for v2 in (False, True):
                yf, mf, sf, offf, nxf = self._fused(inp, seed, offset, v2, att)
                where = f"{tag} v2={v2} attach={att}"
                if v2 and expect_v2_used:
                    # V2 really ran: its self-test passed and it was not disabled
                    # (a silent fall back to design A would also be bitwise)
                    verdicts = list(self.tp._POSENC_DROP_V2_OK.values())
                    self.assertTrue(verdicts and all(verdicts), where + " V2 used")
                    hv = list(self.tp._POSENC_DROP_HWEPI_OK.values())
                    if expect_hw:
                        self.assertTrue(hv and all(hv), where + " HW_EPI used")
                    else:
                        self.assertFalse(any(hv), where + " HW_EPI not used")
                self.assertEqual(_nbad(ys, yf), 0, where + " y")
                if self._expect_bits(att, D):
                    # packed layout of stock's mask, and its exact unpacking
                    self.assertEqual(mf.dtype, torch.uint8, where)
                    self.assertEqual(
                        _nbad(pack_dropout_mask_bits(ms), mf), 0, where + " packed mask"
                    )
                    mf = unpack_dropout_mask_bits(mf, D)
                self.assertEqual(mf.dtype, torch.bool, where)
                self.assertEqual(_nbad(ms, mf), 0, where + " mask")
                self.assertEqual(offs, offf, where + " generator offset")
                self.assertEqual(_nbad(nxs, nxf), 0, where + " next draw")
                for i, nm in enumerate(
                    ("pos keys", "pos vals", "ts keys", "ts vals", "src inds")
                ):
                    self.assertEqual(_nbad(ss[i], sf[i]), 0, f"{where} {nm}")

    @unittest.skipIf(*gpu_unavailable)
    def test_fused_matches_stock_bitwise(self) -> None:
        # tails: every right length 0..40 (residues mod 8 and 16), ctx 8 / 1 / 0
        self._check(_make_inputs(list(range(41)), 8, 128, seed=1), 11, 0, "ctx8")
        self._check(
            _make_inputs(list(range(33)), 1, 64, seed=2, special=True),
            12,
            4 * 3,
            "ctx1_special",
            attach=True,
        )
        self._check(
            _make_inputs([0, 0, 1, 5, 8, 9, 0, 16, 17, 3, 0, 31, 32, 33], 0, 64, seed=3),
            13,
            8,
            "ctx0_empty",
        )
        # non-contiguous inputs with special values, seed 2^64-1, counter carry
        self._check(
            _make_inputs(
                torch.randint(
                    1, 600, (40,), generator=torch.Generator().manual_seed(4)
                ).tolist(),
                8,
                128,
                seed=4,
                noncontig=True,
                special=True,
            ),
            2**64 - 1,
            (2**32 - 100) * 4,
            "noncontig_special_carry",
        )
        # column tail (D = 136: a partial 128-column panel), D = 8, no targets
        self._check(
            _make_inputs(list(range(1, 30)), 3, 136, seed=6, special=True),
            1234,
            4 * 9,
            "D136",
        )
        self._check(
            _make_inputs(list(range(0, 30)), 2, 8, seed=7, targets=False),
            4321,
            (2**32 - 1) * 4,
            "D8_notargets_carry",
        )
        # N * D > 2^19: saturated aten thread count T (power-of-two path)
        self._check(
            _make_inputs([4000, 17, 900, 1], 8, 128, seed=8), 2**63 + 977, 4 * 77, "bigT"
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_exact_gradients(self) -> None:
        inp = _make_inputs(list(range(1, 40)), 8, 128, seed=9)
        N = inp["values_left"].shape[0] + inp["values_right"].shape[0]
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        # small integers: every dropout-scaled product and every sum is exact,
        # so the gradients are order-independent and must match bit for bit
        dy = torch.randint(-8, 9, (N, 128), device="cuda", generator=g).to(
            torch.bfloat16
        )
        leaves = [
            inp["values_left"],
            inp["values_right"],
            inp["pos_embeddings"],
            inp["ts_embeddings"],
        ]
        ys, _, _, _, _ = self._stock(inp, 77, 4 * 5)
        ref = torch.autograd.grad(ys, leaves, dy)
        # attach=False: the saved mask; attach=True with no folding consumer:
        # the handoff's native_dropout_backward fallback
        for v2 in (False, True):
            for attach in (False, True):
                yf, _, _, _, _ = self._fused(inp, 77, 4 * 5, v2, attach=attach)
                got = torch.autograd.grad(yf, leaves, dy)
                for nm, a, b in zip(("d_left", "d_right", "d_pos", "d_ts"), ref, got):
                    self.assertEqual(_nbad(a, b), 0, f"v2={v2} attach={attach} {nm}")

    @unittest.skipIf(*gpu_unavailable)
    def test_mask_bits_ln_bwd_fold(self) -> None:
        # DLRM_HSTU_FP8_DROPOUT_MASK_BITS consumer (needs no knob): the layer-0
        # LN-backward mask/scale fold with the packed mask == with the bool
        # mask, raw bits of dx, dweight, dbias; pack / unpack round trip.
        import triton
        from generative_recommenders.ops.triton import triton_layer_norm as tln

        dev = torch.device("cuda")
        g = torch.Generator(device=dev)
        g.manual_seed(31)
        for D in (8, 13, 136):
            m = torch.rand(77, D, device=dev, generator=g) < 0.8
            packed = pack_dropout_mask_bits(m)
            self.assertEqual(tuple(packed.shape), (77, (D + 7) // 8))
            self.assertEqual(_nbad(unpack_dropout_mask_bits(packed, D), m), 0, D)
        special = torch.tensor(_SPECIAL_BF16, dtype=torch.int32, device=dev).to(
            torch.int16
        )
        old = tln._WLN_BWD_V2_ENABLED
        tln._WLN_BWD_V2_ENABLED = True
        try:
            # the packed form needs D % 8 == 0 (one byte per 8 columns) and the
            # validated launch config (BLOCK_D <= 512); otherwise it declines
            for D in (13, 1024):
                xd = torch.zeros(4, D, device=dev, dtype=torch.bfloat16)
                md = torch.ones(4, D, device=dev, dtype=torch.bool)
                self.assertTrue(tln.wln_bwd_mscale_supported(xd, md), D)
                self.assertFalse(
                    tln.wln_bwd_mscale_supported(xd, pack_dropout_mask_bits(md)), D
                )
            # the tensor-free predicate the producer uses to pick the layout
            for D, ok in ((8, True), (136, True), (512, True), (520, False), (1024, False)):
                self.assertEqual(tln.wln_bwd_mscale_bits_ok(D), ok, D)
            tln._WLN_BWD_V2_ENABLED = False
            self.assertFalse(tln.wln_bwd_mscale_bits_ok(512))
            tln._WLN_BWD_V2_ENABLED = True
            for N, D in ((1, 512), (333, 512), (1037, 256), (65, 136), (40, 24), (7, 8)):
                x = torch.randn(N, D, device=dev, generator=g).to(torch.bfloat16)
                dy = (torch.randn(N, D, device=dev, generator=g) * 0.01).to(
                    torch.bfloat16
                )
                res = (torch.randn(N, D, device=dev, generator=g) * 0.01).to(
                    torch.bfloat16
                )
                for t in (dy, res):
                    k = max(1, N * D // 50)
                    idx = torch.randint(0, N * D, (k,), device=dev, generator=g)
                    t.view(-1).view(torch.int16)[idx] = special[
                        torch.randint(0, len(special), (k,), device=dev, generator=g)
                    ]
                w = (1.0 + 0.1 * torch.randn(D, device=dev, generator=g)).to(
                    torch.bfloat16
                )
                b = (0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
                xf = x.float()
                mean = xf.mean(1)
                rstd = torch.rsqrt(((xf - mean[:, None]) ** 2).mean(1) + 1e-6)
                m = torch.rand(N, D, device=dev, generator=g) >= self.P
                packed = pack_dropout_mask_bits(m)
                outs = []
                for mm in (m, packed):
                    self.assertTrue(tln.wln_bwd_mscale_supported(x, mm), (N, D))
                    outs.append(
                        tln.triton_weighted_layer_norm_bwd(
                            dy=dy,
                            x=x,
                            weight=w,
                            bias=b,
                            mean=mean,
                            rstd=rstd,
                            learnable=True,
                            eps=1e-6,
                            BLOCK_D=triton.next_power_of_2(D),
                            residual=res,
                            dropout_mask=mm,
                            dropout_scale=1.0 / (1.0 - self.P),
                        )
                    )
                for nm, a, bb in zip(("dx", "dw", "db"), *outs):
                    self.assertEqual(_nbad(a, bb), 0, f"N={N} D={D} {nm}")
        finally:
            tln._WLN_BWD_V2_ENABLED = old

    @unittest.skipIf(*gpu_unavailable)
    def test_mask_bits_layout_decision(self) -> None:
        # DLRM_HSTU_FP8_DROPOUT_MASK_BITS: the forward packs the mask only where
        # the layer-0 LN-backward fold will consume it (a handoff is attached
        # and wln_bwd_mscale_bits_ok(D)); everywhere else it keeps aten's bool
        # mask, so no backward unpacks it. One log line per outcome, the V2
        # self-test runs in the effective layout, bits as stock throughout.
        import generative_recommenders.ops.input_dropout_fold as idf

        tp, tln = self.tp, self.tln
        real_unpack = idf.unpack_dropout_mask_bits
        n_unpack = [0]

        def counting_unpack(*a, **k):
            n_unpack[0] += 1
            return real_unpack(*a, **k)

        def drop_mask_bits_keys() -> None:
            tp._LOGGED.difference_update(
                {k for k in tp._LOGGED if k.startswith("mask_bits")}
            )

        g = torch.Generator(device="cuda")
        logged = set(tp._LOGGED)
        idf.unpack_dropout_mask_bits = counting_unpack
        try:
            for knob, attach, wln_v2, D, want, fuse in (
                (False, True, True, 128, False, True),
                (True, True, True, 128, True, True),
                (True, False, True, 128, False, True),  # MSCALE_FOLD=0: no handoff
                (True, True, False, 128, False, True),  # WLN_BWD_V2=0
                (True, True, True, 1024, False, True),  # BLOCK_D > 512
                # DLRM_HSTU_FP8_FWD=0 / FUSE_RESIDUAL=0: no consumer claims it
                (True, True, True, 128, False, False),
            ):
                tp._DROPOUT_MASK_BITS = knob
                tln._WLN_BWD_V2_ENABLED = wln_v2
                self.tf8._HSTU_FP8_FUSE_RESIDUAL = fuse
                drop_mask_bits_keys()
                where = f"knob={knob} attach={attach} wln_v2={wln_v2} D={D} fuse={fuse}"
                self.assertEqual(self._expect_bits(attach, D), want, where)
                inp = _make_inputs(list(range(1, 30)), 3, D, seed=50 + D // 8)
                N = inp["values_left"].shape[0] + inp["values_right"].shape[0]
                leaves = [
                    inp["values_left"],
                    inp["values_right"],
                    inp["pos_embeddings"],
                    inp["ts_embeddings"],
                ]
                g.manual_seed(D)
                # small integers: exact sums, so all four gradients are bitwise
                dy = torch.randint(-8, 9, (N, D), device="cuda", generator=g).to(
                    torch.bfloat16
                )
                ys, ms, _, offs, _ = self._stock(inp, 61, 4 * 7)
                ref = torch.autograd.grad(ys, leaves, dy)
                for v2 in (False, True):
                    tp._POSENC_DROP_V2_OK.clear()
                    n_unpack[0] = 0
                    yf, mf, _, offf, _ = self._fused(inp, 61, 4 * 7, v2, attach)
                    w2 = f"{where} v2={v2}"
                    self.assertEqual(mf.dtype, torch.uint8 if want else torch.bool, w2)
                    self.assertEqual(_nbad(ys, yf), 0, w2 + " y")
                    self.assertEqual(
                        _nbad(pack_dropout_mask_bits(ms) if want else ms, mf), 0, w2
                    )
                    self.assertEqual(offs, offf, w2 + " generator offset")
                    if v2:
                        keys = list(tp._POSENC_DROP_V2_OK)
                        self.assertEqual([k[-1] for k in keys], [want], w2)
                    got = torch.autograd.grad(yf, leaves, dy)
                    for nm, a, b in zip(("d_left", "d_right", "d_pos", "d_ts"), ref, got):
                        self.assertEqual(_nbad(a, b), 0, f"{w2} {nm}")
                    # no consumer folds here: only a packed handoff unpacks
                    self.assertEqual(n_unpack[0], int(want), w2 + " unpacks")
                keys = {k for k in tp._LOGGED if k.startswith("mask_bits")}
                if not knob:
                    self.assertEqual(keys, set(), where)
                elif want:
                    self.assertEqual(keys, {"mask_bits"}, where)
                else:
                    self.assertEqual(len(keys), 1, where)
                    self.assertTrue(next(iter(keys)).startswith("mask_bits_off:"), where)
            # the "not armed" line is a WARNING naming the reason
            tp._DROPOUT_MASK_BITS = True
            self.tf8._HSTU_FP8_FUSE_RESIDUAL = True
            tln._WLN_BWD_V2_ENABLED = False
            drop_mask_bits_keys()
            inp = _make_inputs(list(range(1, 9)), 3, 64, seed=58)
            with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                self._fused(inp, 62, 4 * 8, True, True)
            self.assertTrue(
                any(
                    "set but NOT armed" in r and "DLRM_HSTU_FP8_WLN_BWD_V2" in r
                    for r in cm.output
                ),
                cm.output,
            )
        finally:
            idf.unpack_dropout_mask_bits = real_unpack
            tp._LOGGED.clear()
            tp._LOGGED.update(logged)
        # a packed handoff needs its [N, D]: it cannot be recovered from [N, D / 8]
        pk = pack_dropout_mask_bits(torch.ones(4, 16, device="cuda", dtype=torch.bool))
        with self.assertRaises(AssertionError):
            idf.DropoutHandoff(pk, 1.25)
        h = idf.DropoutHandoff(pk, 1.25, (4, 16))
        self.assertEqual(tuple(h.bool_mask().shape), (4, 16))

    @unittest.skipIf(*gpu_unavailable)
    def test_mask_bits_ln_bwd_reduce_layouts(self) -> None:
        # Tripwire for DLRM_HSTU_FP8_DROPOUT_MASK_BITS. The packed fold equals
        # the bool fold bit for bit only while Triton gives both compiles of
        # _weighted_layer_norm_bwd_dx_v2 the same reduction layouts (operand
        # encodings of every tt.reduce in the TTGIR; validated on Triton 3.6 /
        # gfx950). Compiled through the production wrapper at D 512 / 256, both
        # Triton specializations of N (N % 16 != 0 / == 0) and POW2 on / off,
        # then at the production size (N 2,377,259; x > 2 GiB). A failure means
        # the packed fold is no longer proven bitwise: re-run
        # probes/iter67/rc7 validate_rc7.py kmat before arming the knob.
        import re

        tln = self.tln
        dev = torch.device("cuda")
        g = torch.Generator(device=dev)
        g.manual_seed(77)
        real = tln._weighted_layer_norm_bwd_dx_v2
        launches = []

        class _Spy:  # the wrapper looks the kernel up as a module global
            def __getitem__(self, grid):
                def launch(*a, **k):
                    ck = real[grid](*a, **k)
                    launches.append((ck, k.get("MASK_BITS", False)))
                    return ck

                return launch

        def reduce_layouts(ttgir: str) -> List[str]:
            alias = dict(re.findall(r"^(#\w+) = (.+)$", ttgir, flags=re.M))

            def resolve(s: str) -> str:
                for _ in range(4):
                    s = re.sub(r"#\w+", lambda m: alias.get(m.group(0), m.group(0)), s)
                return s

            out, lines = [], ttgir.splitlines()
            for i, line in enumerate(lines):
                if '"tt.reduce"' not in line:
                    continue
                axis = re.search(r"axis = (\d+)", line)
                for j in range(i, min(i + 100, len(lines))):
                    m = re.search(r"\}\) : \((.*?)\) -> ", lines[j])
                    if m:
                        ax = axis.group(1) if axis else "?"
                        out.append(f"axis={ax} {resolve(m.group(1))}")
                        break
            return out

        def check(N: int, D: int) -> None:
            x = (torch.randn(N, D, device=dev, generator=g) * 0.7).to(torch.bfloat16)
            dy = (torch.randn(N, D, device=dev, generator=g) * 0.01).to(torch.bfloat16)
            res = (torch.randn(N, D, device=dev, generator=g) * 0.01).to(torch.bfloat16)
            w = (1.0 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
            b = (0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
            mean = torch.empty(N, device=dev)
            rstd = torch.empty(N, device=dev)
            for s in range(0, N, 1 << 18):
                xf = x[s : s + (1 << 18)].float()
                mean[s : s + (1 << 18)] = xf.mean(1)
                rstd[s : s + (1 << 18)] = torch.rsqrt(xf.var(1, correction=0) + 1e-6)
            m = torch.rand(N, D, device=dev, generator=g) >= self.P
            for pow2 in (True, False):
                tln._WLN_BWD_POW2_ENABLED = pow2
                outs, irs, flags = [], [], []
                for mm in (m, pack_dropout_mask_bits(m)):
                    launches.clear()
                    outs.append(
                        tln.triton_weighted_layer_norm_bwd(
                            dy=dy,
                            x=x,
                            weight=w,
                            bias=b,
                            mean=mean,
                            rstd=rstd,
                            learnable=True,
                            eps=1e-6,
                            BLOCK_D=D,
                            residual=res,
                            dropout_mask=mm,
                            dropout_scale=1.0 / (1.0 - self.P),
                        )
                    )
                    self.assertEqual(len(launches), 1)
                    irs.append(reduce_layouts(launches[0][0].asm["ttgir"]))
                    flags.append(launches[0][1])
                where = f"N={N} D={D} pow2={pow2}"
                self.assertEqual(flags, [False, True], where)
                self.assertGreater(len(irs[0]), 0, where)
                self.assertEqual(irs[0], irs[1], where + ": tt.reduce layouts differ")
                for nm, a, bb in zip(("dx", "dw", "db"), *outs):
                    self.assertEqual(_nbad(a, bb), 0, f"{where} {nm}")

        old_pow2 = tln._WLN_BWD_POW2_ENABLED
        # this tripwire is about the Triton fold, the fallback of every FlyDSL
        # decline: keep the FlyDSL mscale route (DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE)
        # off, whatever the environment says
        old_fly_ms = tln._WLN_BWD_FLYDSL_MSCALE_ENABLED
        tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = False
        tln._weighted_layer_norm_bwd_dx_v2 = _Spy()
        try:
            for D in (512, 256):
                self.assertTrue(tln.wln_bwd_mscale_bits_ok(D), D)
                for N in (4099, 4096):
                    check(N, D)
            if torch.cuda.get_device_properties(dev).total_memory >= 48 * 2**30:
                check(2377259, 512)
        finally:
            tln._weighted_layer_norm_bwd_dx_v2 = real
            tln._WLN_BWD_POW2_ENABLED = old_pow2
            tln._WLN_BWD_FLYDSL_MSCALE_ENABLED = old_fly_ms

    @unittest.skipIf(*gpu_unavailable)
    def test_mask_bits_fold_self_test(self) -> None:
        # The producer's first-use, fail-closed check of the packed LN-backward
        # fold (_mask_bits_fold_verified, the runtime form of the tripwire
        # above): passes on this stack, is RNG-neutral and cached, and restores
        # the kernel global. A packed fold that differs in one dx bit, in its
        # tt.reduce layouts, or that raises, keeps the bool mask with the
        # "set but NOT armed" WARNING, and the forward stays bitwise.
        import types

        tp, tln = self.tp, self.tln
        dev = torch.device("cuda", torch.cuda.current_device())
        real = tln._weighted_layer_norm_bwd_dx_v2
        n_launch = [0]

        class _Wrap:
            def __init__(self, mode: str) -> None:
                self.mode = mode

            def __getitem__(self, grid):
                def launch(*a, **k):
                    n_launch[0] += 1
                    packed = k.get("MASK_BITS", False)
                    if packed and self.mode == "raise":
                        raise RuntimeError("simulated packed-fold compile failure")
                    ck = real[grid](*a, **k)
                    if packed and self.mode == "flip":
                        a[0].view(torch.int16).view(-1)[123] ^= 1  # DX
                    if packed and self.mode == "layout":
                        return types.SimpleNamespace(asm={"ttgir": ""})
                    return ck

                return launch

        logged = set(tp._LOGGED)
        try:
            for D in (512, 136):
                self.assertTrue(tln.wln_bwd_mscale_bits_ok(D), D)
                tp._MASK_BITS_FOLD_OK.clear()
                tln._weighted_layer_norm_bwd_dx_v2 = wrap = _Wrap("none")
                n_launch[0] = 0
                gen = _set_rng(2**63 + 5, (2**32 - 1) * 4)
                self.assertEqual(tp._mask_bits_fold_verified(dev, D), "", D)
                self.assertEqual(n_launch[0], 4, D)  # bool + packed, N 4099 / 4096
                self.assertEqual(gen.initial_seed(), 2**63 + 5)
                self.assertEqual(gen.get_offset(), (2**32 - 1) * 4)
                self.assertIs(tln._weighted_layer_norm_bwd_dx_v2, wrap, D)
                self.assertEqual(tp._mask_bits_fold_verified(dev, D), "", D)
                self.assertEqual(n_launch[0], 4, f"D={D}: cached, no relaunch")
            for mode, want in (
                ("flip", "dx differ"),
                ("layout", "tt.reduce layouts differ"),
                ("raise", "raised RuntimeError"),
            ):
                tp._MASK_BITS_FOLD_OK.clear()
                tln._weighted_layer_norm_bwd_dx_v2 = wrap = _Wrap(mode)
                why = tp._mask_bits_fold_verified(dev, 512)
                self.assertIn("self-test", why, mode)
                self.assertIn(want, why, mode)
                self.assertIs(tln._weighted_layer_norm_bwd_dx_v2, wrap, mode)
                # cached verdict: the producer keeps the bool mask from now on
                self.assertEqual(tp._dropout_mask_bits_why_not(True, 512, dev), why)
            # through the producer: bool mask, WARNING, bitwise as stock
            tp._MASK_BITS_FOLD_OK.clear()
            tln._weighted_layer_norm_bwd_dx_v2 = _Wrap("flip")
            tp._DROPOUT_MASK_BITS = True
            tp._LOGGED.difference_update(
                {k for k in tp._LOGGED if k.startswith("mask_bits")}
            )
            inp = _make_inputs(list(range(1, 20)), 3, 128, seed=59)
            ys, ms, _, offs, nxs = self._stock(inp, 63, 4 * 9)
            with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                yf, mf, _, offf, nxf = self._fused(inp, 63, 4 * 9, True, True)
            self.assertEqual(mf.dtype, torch.bool)
            self.assertEqual(_nbad(ys, yf), 0, "y")
            self.assertEqual(_nbad(ms, mf), 0, "mask")
            self.assertEqual(offs, offf, "generator offset")
            self.assertEqual(_nbad(nxs, nxf), 0, "next draw")
            self.assertTrue(
                any("set but NOT armed" in r and "self-test" in r for r in cm.output),
                cm.output,
            )
        finally:
            tln._weighted_layer_norm_bwd_dx_v2 = real
            tp._LOGGED.clear()
            tp._LOGGED.update(logged)

    def _verify_args(self, inp: Dict):
        return (
            inp["values_left"].detach(),
            inp["values_right"].detach(),
            inp["seq_offsets"],
            inp["seq_lengths"],
            inp["pos_embeddings"].detach(),
            inp["ts_embeddings"].detach(),
            inp["timestamps"],
            inp["num_targets"],
            inp["interleave_targets"],
            inp["time_bucket_fn"],
            inp["alpha"],
            inp["max_contextual_seq_len"],
            self.P,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_v2_self_test_passes_and_is_rng_neutral(self) -> None:
        # in both mask layouts (bool; DLRM_HSTU_FP8_DROPOUT_MASK_BITS packed)
        tp = self.tp
        for D, ctx_len in ((512, 8), (136, 3), (64, 0)):
            inp = _make_inputs(list(range(20)), ctx_len, D, seed=D)
            for bits in (False, True):
                tp._POSENC_DROP_V2_OK.clear()
                gen = _set_rng(2**63 + 11, (2**32 - 2) * 4)
                self.assertTrue(
                    tp._posenc_drop_v2_verified(
                        *self._verify_args(inp), mask_bits=bits
                    ),
                    (D, bits),
                )
                self.assertEqual(gen.initial_seed(), 2**63 + 11)
                self.assertEqual(gen.get_offset(), (2**32 - 2) * 4)
                self.assertEqual(list(tp._POSENC_DROP_V2_OK.values()), [True])
                self.assertEqual(list(tp._POSENC_DROP_V2_OK)[0][-1], bits)

    @unittest.skipIf(*gpu_unavailable)
    def test_v2_fails_closed(self) -> None:
        tp = self.tp
        real = tp._posenc_concat_dropout_v2_kernel
        inp = _make_inputs(list(range(1, 25)), 8, 128, seed=10)

        class _Raises:
            def __getitem__(self, grid):
                raise RuntimeError("simulated V2 compile failure")

        class _FlipsOneBit:
            def __getitem__(self, grid):
                def launch(**kw):
                    real[grid](**kw)
                    kw["Y"].view(-1)[5] ^= 1

                return launch

        try:
            # (1) the self-test catches a compile/launch error and a wrong bit
            for broken in (_Raises(), _FlipsOneBit()):
                tp._posenc_concat_dropout_v2_kernel = broken
                tp._POSENC_DROP_V2_OK.clear()
                with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                    self._check(
                        inp,
                        21,
                        4 * 2,
                        f"self-test {type(broken).__name__}",
                        expect_v2_used=False,
                    )
                self.assertTrue(any("V2 layout DISABLED" in r for r in cm.output))
                # one key per mask layout _check ran (two with the knob on)
                verdicts = list(tp._POSENC_DROP_V2_OK.values())
                self.assertTrue(verdicts and not any(verdicts), verdicts)
            # (2) a launch error after a passed self-test falls back to design A
            tp._posenc_concat_dropout_v2_kernel = _Raises()
            tp._POSENC_DROP_V2_OK.clear()
            verified = tp._posenc_drop_v2_verified
            tp._posenc_drop_v2_verified = lambda *a, **k: True
            try:
                with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                    self._check(inp, 22, 4 * 3, "launch fallback", expect_v2_used=False)
                self.assertTrue(any("launch raised" in r for r in cm.output))
            finally:
                tp._posenc_drop_v2_verified = verified
        finally:
            tp._posenc_concat_dropout_v2_kernel = real

    def _hw_spy(self, mode: str, seen: List):
        """V2 kernel stand-in recording (HW_EPI, Y rows, compiled kernel) per
        launch; mode raise / flip breaks only the HW_EPI compile."""
        real = self._real_v2

        class _Spy:
            def __getitem__(self, grid):
                def launch(**kw):
                    hw = kw.get("HW_EPI", False)
                    if hw and mode == "raise":
                        raise RuntimeError("simulated HW_EPI compile failure")
                    ck = real[grid](**kw)
                    if hw and mode == "flip":
                        kw["Y"].view(-1)[5] ^= 1
                    seen.append((hw, kw["Y"].shape[0], ck))
                    return ck

                return launch

        return _Spy()

    @unittest.skipIf(*gpu_unavailable)
    def test_hw_epilogue_bitwise(self) -> None:
        # DLRM_HSTU_FP8_POSENC_DROP_HWEPI (gfx950): after its first-use
        # self-test V2 launches its HW_EPI compile (v_cvt_pk_bf16_f32 epilogue,
        # v_bitop3 Philox rounds); y, mask, generator offset, next draw and the
        # saved indices are bitwise the stock pair's (tails, special values,
        # counter carries, D 136 / 8, saturated T); one arming line
        tp = self.tp
        if not self._gfx950():
            self.skipTest("HW_EPI is gfx950-only")
        self._real_v2 = tp._posenc_concat_dropout_v2_kernel
        seen: List = []
        logged = set(tp._LOGGED)
        tp._LOGGED.discard("posenc_drop_hwepi")
        tp._POSENC_DROP_HWEPI = True
        tp._POSENC_DROP_HWEPI_OK.clear()
        tp._posenc_concat_dropout_v2_kernel = self._hw_spy("none", seen)
        cases = (
            (dict(right_lens=list(range(41)), ctx_len=8, D=128, seed=1), 11, 0, False),
            (dict(right_lens=list(range(33)), ctx_len=1, D=64, seed=2, special=True), 12, 4 * 3, True),
            (
                dict(right_lens=torch.randint(1, 600, (40,), generator=torch.Generator().manual_seed(4)).tolist(),
                     ctx_len=8, D=128, seed=4, noncontig=True, special=True),
                2**64 - 1,
                (2**32 - 100) * 4,
                False,
            ),
            (dict(right_lens=list(range(1, 30)), ctx_len=3, D=136, seed=6, special=True), 1234, 4 * 9, False),
            (dict(right_lens=list(range(0, 30)), ctx_len=2, D=8, seed=7, targets=False), 4321, (2**32 - 1) * 4, False),
            (dict(right_lens=[4000, 17, 900, 1], ctx_len=8, D=128, seed=8), 2**63 + 977, 4 * 77, False),
        )
        try:
            with self.assertLogs(tp._logger, level=logging.INFO) as cm:
                for kw, seed, off, attach in cases:
                    inp = _make_inputs(**kw)
                    N = inp["values_left"].shape[0] + inp["values_right"].shape[0]
                    seen.clear()
                    self._check(inp, seed, off, f"hw {kw['D']}", attach=attach, expect_hw=True)
                    # the forward itself (Y rows = N, not the self-test's batch) ran HW_EPI
                    self.assertTrue(any(h and n == N for h, n, _ in seen), (kw["D"], N))
                    self.assertFalse(any(not h and n == N for h, n, _ in seen), (kw["D"], N))
            arm = [
                r for r in cm.output
                if "DLRMV4 armed OK: posenc dropout hw epilogue "
                "(v_cvt_pk_bf16_f32 + v_bitop3 philox) ACTIVE" in r
            ]
            self.assertEqual(len(arm), 1, cm.output)
            self.assertFalse(any("DISABLED" in r or "NOT armed" in r for r in cm.output), cm.output)
            hw_ck = [ck for h, _, ck in seen if h]
            self.assertIn("bitop3:0x96", hw_ck[0].asm["amdgcn"])
        finally:
            tp._posenc_concat_dropout_v2_kernel = self._real_v2
            tp._LOGGED.clear()
            tp._LOGGED.update(logged)

    @unittest.skipIf(*gpu_unavailable)
    def test_hw_epilogue_kill_switch_and_fail_closed(self) -> None:
        # DLRM_HSTU_FP8_POSENC_DROP_HWEPI off (=0 / unset) or another arch: no
        # HW_EPI compile, no HW self-test, V2 exactly as before. On gfx950 a
        # HW_EPI compile error or wrong bit (self-test), a launch error after a
        # passed self-test, or an epilogue cast mismatch each fall back to V2's
        # software epilogue with a WARNING; bitwise as stock throughout; the
        # HW self-test is RNG-neutral and cached.
        tp = self.tp
        self._real_v2 = tp._posenc_concat_dropout_v2_kernel
        inp = _make_inputs(list(range(1, 25)), 8, 128, seed=10, special=True)
        seen: List = []
        logged = set(tp._LOGGED)
        idx = torch.cuda.current_device()
        arch = dict(tp._GCN_ARCH)
        cvt = dict(tp._HW_CVT_SELF_TEST)
        real_probe = tp._bf16_hw_cvt_probe_kernel
        verified = tp._posenc_drop_hwepi_verified

        def n_disabled(cm) -> int:
            return sum("hw epilogue DISABLED" in r for r in cm.output)

        try:
            # (1) kill switch
            tp._POSENC_DROP_HWEPI = False
            tp._POSENC_DROP_HWEPI_OK.clear()
            tp._posenc_concat_dropout_v2_kernel = self._hw_spy("none", seen)
            self._check(inp, 31, 4 * 2, "kill switch", expect_hw=False)
            self.assertTrue(seen)
            self.assertFalse(any(h for h, _, _ in seen))
            self.assertEqual(tp._POSENC_DROP_HWEPI_OK, {})
            # (2) another arch: knob on, the V2 binary as before, one WARNING
            tp._POSENC_DROP_HWEPI = True
            tp._GCN_ARCH[idx] = "gfx942:sramecc+:xnack-"
            tp._LOGGED.difference_update({k for k in tp._LOGGED if k.startswith("posenc_drop_hwepi")})
            seen.clear()
            with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                self._check(inp, 32, 4 * 3, "gfx942", expect_hw=False)
            self.assertFalse(any(h for h, _, _ in seen))
            self.assertEqual(tp._POSENC_DROP_HWEPI_OK, {})
            self.assertEqual(
                sum("POSENC_DROP_HWEPI set but NOT armed" in r and "gfx942" in r for r in cm.output), 1, cm.output
            )
            tp._GCN_ARCH.clear()
            tp._GCN_ARCH.update(arch)
            if not self._gfx950():
                return  # the rest needs the gfx950 HW_EPI compile
            # (3) the HW self-test catches a compile error and a wrong bit
            for mode in ("raise", "flip"):
                tp._posenc_concat_dropout_v2_kernel = self._hw_spy(mode, seen)
                tp._POSENC_DROP_HWEPI_OK.clear()
                seen.clear()
                with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                    self._check(inp, 33, 4 * 4, f"hw self-test {mode}", expect_hw=False)
                hv = list(tp._POSENC_DROP_HWEPI_OK.values())
                self.assertTrue(hv and not any(hv), mode)
                self.assertEqual(n_disabled(cm), len(hv), cm.output)  # one per key
                self.assertTrue(any("self-test" in r for r in cm.output if "DISABLED" in r), mode)
                N = inp["values_left"].shape[0] + inp["values_right"].shape[0]
                self.assertFalse(any(h and n == N for h, n, _ in seen), mode)
            # (4) a launch error after a passed self-test
            tp._posenc_concat_dropout_v2_kernel = self._hw_spy("raise", seen)
            tp._POSENC_DROP_HWEPI_OK.clear()
            tp._posenc_drop_hwepi_verified = lambda *a, **k: True
            with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                self._check(inp, 34, 4 * 5, "hw launch fallback", expect_hw=False)
            self.assertTrue(any("hw epilogue DISABLED" in r and "launch raised" in r for r in cm.output))
            tp._posenc_drop_hwepi_verified = verified
            # (5) the epilogue cast check (f.to(bf16) vs torch's cast) fails closed
            tp._posenc_concat_dropout_v2_kernel = self._real_v2

            class _FlipProbe:
                def __getitem__(self, grid):
                    def launch(*a, **k):
                        real_probe[grid](*a, **k)
                        a[1].view(-1)[7] ^= 1

                    return launch

            tp._bf16_hw_cvt_probe_kernel = _FlipProbe()
            tp._HW_CVT_SELF_TEST.clear()
            tp._POSENC_DROP_HWEPI_OK.clear()
            with self.assertLogs(tp._logger, level=logging.WARNING) as cm:
                self._check(inp, 35, 4 * 6, "hw cvt mismatch", expect_hw=False)
            self.assertTrue(any("f.to(bf16) differs" in r for r in cm.output), cm.output)
            tp._bf16_hw_cvt_probe_kernel = real_probe
            tp._HW_CVT_SELF_TEST.clear()
            # (6) a passing HW self-test is RNG-neutral and cached
            for bits in (False, True):
                tp._POSENC_DROP_HWEPI_OK.clear()
                tp._posenc_concat_dropout_v2_kernel = self._hw_spy("none", seen)
                gen = _set_rng(2**63 + 11, (2**32 - 2) * 4)
                seen.clear()
                self.assertTrue(tp._posenc_drop_hwepi_verified(*self._verify_args(inp), mask_bits=bits), bits)
                self.assertEqual(gen.initial_seed(), 2**63 + 11)
                self.assertEqual(gen.get_offset(), (2**32 - 2) * 4)
                self.assertEqual(sum(h for h, _, _ in seen), 4, bits)  # 2 RNG states x 2 runs
                self.assertTrue(tp._posenc_drop_hwepi_verified(*self._verify_args(inp), mask_bits=bits), bits)
                self.assertEqual(sum(h for h, _, _ in seen), 4, f"{bits}: cached")
                self.assertEqual(list(tp._POSENC_DROP_HWEPI_OK.values()), [True])
        finally:
            tp._posenc_concat_dropout_v2_kernel = self._real_v2
            tp._bf16_hw_cvt_probe_kernel = real_probe
            tp._posenc_drop_hwepi_verified = verified
            tp._GCN_ARCH.clear()
            tp._GCN_ARCH.update(arch)
            tp._HW_CVT_SELF_TEST.clear()
            tp._HW_CVT_SELF_TEST.update(cvt)
            tp._LOGGED.clear()
            tp._LOGGED.update(logged)

    def test_env_knobs(self) -> None:
        from generative_recommenders.ops.triton.triton_position import (
            _posenc_env_int,
        )

        name = "DLRM_HSTU_FP8_POSENC_DROP_V2_TEST_KNOB"
        old = os.environ.pop(name, None)
        try:
            for raw, want in (("", 8), ("  ", 8), ("16", 16), (" 4 ", 4)):
                os.environ[name] = raw
                self.assertEqual(_posenc_env_int(name, 8, pow2=True), want, raw)
            os.environ.pop(name)
            self.assertEqual(_posenc_env_int(name, 8, pow2=True), 8)
            for raw in ("6", "0", "-8"):
                os.environ[name] = raw
                with self.assertRaises(ValueError):
                    _posenc_env_int(name, 8, pow2=True)
        finally:
            os.environ.pop(name, None)
            if old is not None:
                os.environ[name] = old

    def test_import_with_empty_knobs(self) -> None:
        env = dict(os.environ)
        for k in (
            "DLRM_HSTU_FP8_POSENC_DROPOUT_V2",
            "DLRM_HSTU_FP8_POSENC_DROP_V2_BLOCK_D",
            "DLRM_HSTU_FP8_POSENC_DROP_V2_BLOCK_N",
            "DLRM_HSTU_FP8_POSENC_DROP_V2_WARPS",
            "DLRM_HSTU_FP8_POSENC_DROP_V2_STAGES",
            "DLRM_HSTU_FP8_POSENC_DROP_BLOCK_D",
            "DLRM_HSTU_FP8_POSENC_DROP_BLOCK_N",
            "DLRM_HSTU_FP8_POSENC_DROP_WARPS",
            "DLRM_HSTU_FP8_POSENC_DROP_STAGES",
            "DLRM_HSTU_FP8_DROPOUT_MASK_BITS",
            "DLRM_HSTU_FP8_POSENC_DROP_HWEPI",
        ):
            env[k] = ""
        code = (
            "import generative_recommenders.ops.triton.triton_position as tp; "
            "print(tp._POSENC_DROPOUT_V2, tp._POSENC_DROP_V2_BLOCK_D, "
            "tp._POSENC_DROP_V2_BLOCK_N, tp._POSENC_DROP_V2_WARPS, "
            "tp._POSENC_DROP_V2_STAGES, tp._POSENC_DROP_BLOCK_D, "
            "tp._POSENC_DROP_BLOCK_N, tp._POSENC_DROP_WARPS, tp._POSENC_DROP_STAGES, "
            "tp._DROPOUT_MASK_BITS, tp._POSENC_DROP_HWEPI)"
        )
        r = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        # Empty = the code default: the V2, MASK_BITS and HWEPI switches are off
        # (only an explicit 1/true/on/yes, e.g. the gin binding, turns them
        # on), geometry defaults.
        self.assertEqual(
            r.stdout.split()[-11:], "False 128 8 2 1 128 16 4 1 False False".split()
        )
        env["DLRM_HSTU_FP8_POSENC_DROPOUT_V2"] = "1"
        env["DLRM_HSTU_FP8_DROPOUT_MASK_BITS"] = "1"
        env["DLRM_HSTU_FP8_POSENC_DROP_HWEPI"] = "1"
        r = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(
            r.stdout.split()[-11:], "True 128 8 2 1 128 16 4 1 True True".split()
        )
        # =0 is the HWEPI kill switch
        env["DLRM_HSTU_FP8_POSENC_DROP_HWEPI"] = "0"
        r = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.split()[-1], "False")


if __name__ == "__main__":
    unittest.main()
