# pyre-strict
"""Bitwise tests for ops/glue_fusion.py.

CPU-runnable through the Triton interpreter:

    TRITON_INTERPRET=1 python -m pytest -q generative_recommenders/ops/tests/glue_fusion_test.py

(on a GPU host the same tests run on the device when TRITON_INTERPRET is unset
and DEVICE=cuda is exported). The references re-implement the SHIPPED chains
with plain torch ops (bitwise_and/gt/mul/concat/.to(bf16); arange/cumsum/
repeat_interleave/scatter_), so equality is checked against the incumbent's
per-element math, as raw bit patterns.
"""

import os
import unittest

import torch

from generative_recommenders.ops import glue_fusion as gf

DEV = torch.device(os.environ.get("DEVICE", "cpu"))


def _bits16(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _bits32(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def _rne_ref_bits(x: torch.Tensor) -> torch.Tensor:
    """Expected bf16 bits (int32 0..65535) for fp32 x: RNE via
    c10::detail::round_to_nearest_even for every non-NaN input. A NaN keeps
    its sign and upper payload with the quiet bit set, as torch's fp32->bf16
    cast does on gfx950 (iter65 exhaustive sweep)."""
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rne = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    nan = ((u >> 16) | 0x40) & 0xFFFF
    return torch.where(torch.isnan(x), nan, rne).to(torch.int32)


def _complete_cumsum(x: torch.Tensor) -> torch.Tensor:
    # == fbgemm asynchronous_complete_cumsum for integer input
    return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), torch.cumsum(x, 0)])


def _shipped_concat(left, right, off_a, off_b):
    """concat_2D_jagged(values_left=left, values_right=right, offsets_*) rows."""
    B = off_a.numel() - 1
    pieces = []
    for b in range(B):
        pieces.append(left[off_a[b]:off_a[b + 1]])
        pieces.append(right[off_b[b]:off_b[b + 1]])
    return torch.cat(pieces, 0)


def _shipped_action(actions, weights, table, target_table, off_a, off_b, T):
    exploded = torch.bitwise_and(actions.unsqueeze(-1), weights.unsqueeze(0)) > 0
    emb = (exploded.unsqueeze(-1) * table.unsqueeze(0)).view(-1, table.numel())
    right = target_table.tile(T, 1)
    return _shipped_concat(emb, right, off_a, off_b)


def _case(seed: int, B: int = 9, A: int = 3, D: int = 8):
    g = torch.Generator().manual_seed(seed)
    len_a = torch.randint(0, 300, (B,), generator=g)
    len_a[0] = 0  # empty uih sample
    len_b = torch.randint(0, 3, (B,), generator=g)
    off_a = _complete_cumsum(len_a)
    off_b = _complete_cumsum(len_b)
    N = int(off_a[-1])
    T = int(off_b[-1])
    actions = torch.randint(0, 1 << A, (N,), generator=g, dtype=torch.int64)
    weights = torch.tensor([1 << i for i in range(A)], dtype=torch.int64)
    table = torch.randn((A, D), generator=g) * 0.1  # has negatives -> -0.0 products
    table[0, 0] = 1.0 + 2.0**-8  # exact bf16 tie (round-to-even)
    table[1, 1] = 1.0 + 3 * 2.0**-8  # tie the other way
    table[2, 2] = 1e-40  # subnormal
    target = torch.randn((1, A * D), generator=g) * 0.1
    max_seq_len = int((len_a + len_b).max())
    return actions, weights, table, target, off_a, off_b, T, max_seq_len


class GlueFusionTest(unittest.TestCase):
    def test_rne_bits(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def k(X, Y, n, BLOCK: tl.constexpr):
            o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = o < n
            tl.store(Y + o, gf._f32_to_bf16_bits_rne(tl.load(X + o, mask=m)), mask=m)

        @triton.jit
        def k2(X, Y, n, BLOCK: tl.constexpr):
            o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = o < n
            tl.store(Y + o, gf._bf16_bits_to_f32(tl.load(X + o, mask=m)), mask=m)

        special = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45,
                   1e-40, 3.3895313892515355e38, -3.4028234663852886e38,
                   1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8, 1.0 + 2.0**-8 + 2.0**-20]
        # NaNs with payloads and both signs (sNaN, qNaN, payload in the rounded-off half)
        nan_u = [0x7F800001, 0x7FC00001, 0x7F810000, 0x7F808000, 0x7F80FFFF, 0x7FBFFFFF, 0x7FFFFFFF]
        nan_u = nan_u + [u | 0x80000000 for u in nan_u]
        nans = torch.tensor(nan_u, dtype=torch.int64).to(torch.int32).view(torch.float32)
        x = torch.cat([torch.tensor(special), nans,
                       torch.randn(4000) * 10.0 ** torch.randint(-30, 30, (4000,))])
        ref = _rne_ref_bits(x)
        x = x.to(DEV)
        y = torch.empty(x.numel(), dtype=torch.int16, device=DEV)
        k[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
        self.assertTrue(torch.equal(y.cpu().to(torch.int32) & 0xFFFF, ref))
        if DEV.type == "cuda":
            # On the GPU: identical to torch's own cast for EVERY input, NaN included.
            self.assertTrue(torch.equal(y.cpu(), _bits16(x.to(torch.bfloat16)).cpu()))
        else:
            finite = ~torch.isnan(x)
            self.assertTrue(torch.equal(y[finite].cpu(), _bits16(x[finite].to(torch.bfloat16)).cpu()))
        # bf16 -> fp32 is the exact widening
        h = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).to(DEV)
        f = torch.empty(h.numel(), dtype=torch.float32, device=DEV)
        k2[(triton.cdiv(h.numel(), 256),)](h, f, h.numel(), BLOCK=256)
        self.assertTrue(torch.equal(_bits32(f).cpu(), _bits32(h.view(torch.bfloat16).float()).cpu()))

    def _run_action(self, seed: int, out_bf16: bool) -> None:
        actions, weights, table, target, off_a, off_b, T, msl = _case(seed)
        # Shipped graph on the SAME device as the fused path. The d_table
        # reduction (at::sum_to over [N, A, D]) is the incumbent's on that
        # device. A CPU reference sums in a different order, so fp32 grads
        # differ in the last bits. That is why comm's iter-31 GPU run failed
        # test_action_fp32 (rccl probes/iter65/glue/gpu/t0_diag_fp32.log).
        t_ref = table.clone().to(DEV).requires_grad_(True)
        tt_ref = target.clone().to(DEV).requires_grad_(True)
        ref32 = _shipped_action(
            actions.to(DEV), weights.to(DEV), t_ref, tt_ref,
            off_a.to(DEV), off_b.to(DEV), T,
        )
        ref = ref32.to(torch.bfloat16) if out_bf16 else ref32
        # fused
        t_new = table.clone().to(DEV).requires_grad_(True)
        tt_new = target.clone().to(DEV).requires_grad_(True)
        out = gf.action_encode_concat(
            actions.to(DEV), weights.to(DEV), t_new, tt_new.tile(T, 1),
            off_a.to(DEV), off_b.to(DEV), msl, out_bf16,
        )
        self.assertEqual(out.dtype, ref.dtype)
        self.assertEqual(out.shape, ref.shape)
        bits = _bits16 if out_bf16 else _bits32
        self.assertTrue(torch.equal(bits(out).cpu(), bits(ref.detach()).cpu()))
        # backward: the consumer hands back a gradient in the output's dtype;
        # the shipped graph upcast it to fp32 (ToCopyBackward / _match) first.
        g = torch.Generator().manual_seed(seed + 1)
        go = torch.randn(ref.shape, generator=g)
        go[0, :3] = torch.tensor([0.0, -0.0, 1e-40])
        go = go.to(ref.dtype).to(DEV)
        ref.backward(go.float() if out_bf16 else go)
        out.backward(go)
        self.assertTrue(torch.equal(_bits32(t_new.grad).cpu(), _bits32(t_ref.grad).cpu()))
        self.assertTrue(torch.equal(_bits32(tt_new.grad).cpu(), _bits32(tt_ref.grad).cpu()))

    def test_action_bf16(self) -> None:
        for s in range(4):
            self._run_action(s, True)

    def test_action_fp32(self) -> None:
        """fp32 output mode (never armed in production, see the next test)."""
        for s in range(2):
            self._run_action(100 + s, False)

    def test_action_fusable_restricted_to_bf16(self) -> None:
        """The production gate arms only the bf16-consumer mode."""
        actions, weights, table, target, off_a, off_b, T, msl = _case(3)
        args = (actions.to(DEV), weights.to(DEV), table.to(DEV), target.to(DEV),
                off_a.to(DEV), off_b.to(DEV))
        self.assertFalse(gf.action_encode_fusable(*args, False))
        self.assertFalse(gf.action_encode_fusable(*args))
        if DEV.type == "cuda":
            self.assertEqual(gf.action_encode_fusable(*args, True), gf.ACTION_ENC_FUSE)

    def test_env_switch_convention(self) -> None:
        """Switches are default OFF in code and on only for an explicit
        1/true/on/yes (the gin binding writes "1"); =0 is the kill switch."""
        name = "DLRM_HSTU_FP8_GLUE_TEST_SWITCH"
        old = os.environ.get(name)
        try:
            for val, expect in (("0", False), ("false", False), ("OFF", False), (" no ", False),
                                ("", False), ("2", False), ("1", True), ("on", True),
                                (" TRUE ", True), ("yes", True)):
                os.environ[name] = val
                self.assertEqual(gf._env_true(name), expect, val)
            os.environ.pop(name)
            self.assertFalse(gf._env_true(name))
        finally:
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_action_bwd_intermediate(self) -> None:
        """The masked product handed to sum_to_size equals MulBackward0's."""
        actions, weights, table, target, off_a, off_b, T, msl = _case(7)
        N, AD = actions.numel(), table.numel()
        go = torch.randn((N + T, AD)).to(torch.bfloat16)
        d_left = torch.empty((N, AD), dtype=torch.float32, device=DEV)
        d_right = torch.empty((T, AD), dtype=torch.float32, device=DEV)
        import triton

        gf._action_enc_bwd_kernel[(triton.cdiv(msl, gf._ACT_BLOCK_N), off_a.numel() - 1)](
            go.to(DEV).view(torch.int16), actions.to(DEV), weights.to(DEV), d_left, d_right,
            off_a.to(DEV), off_b.to(DEV), D=table.shape[1], AD=AD,
            BLOCK_AD=triton.next_power_of_2(AD), BLOCK_N=gf._ACT_BLOCK_N, GRAD_BF16=True,
        )
        exploded = torch.bitwise_and(actions.unsqueeze(-1), weights.unsqueeze(0)) > 0
        g32 = go.float()
        lrows = torch.cat([torch.arange(int(off_a[b]) + int(off_b[b]), int(off_a[b + 1]) + int(off_b[b]))
                           for b in range(off_a.numel() - 1)])
        rrows = torch.cat([torch.arange(int(off_a[b + 1]) + int(off_b[b]), int(off_a[b + 1]) + int(off_b[b + 1]))
                           for b in range(off_a.numel() - 1)])
        ref_left = g32[lrows].view(N, *table.shape) * exploded.unsqueeze(-1)
        self.assertTrue(torch.equal(_bits32(d_left).cpu(), _bits32(ref_left.view(N, AD))))
        self.assertTrue(torch.equal(_bits32(d_right).cpu(), _bits32(g32[rrows])))

    def _merged_ref(self, ri_u, ri_c, len_u, len_c, n_u, n_c):
        # ec_bf16_output._merged_index as shipped (fbgemm cumsum == _complete_cumsum)
        off_u = _complete_cumsum(len_u)
        off_c = _complete_cumsum(len_c)
        samples = torch.arange(len_u.numel(), dtype=torch.int64)
        seg_u = torch.repeat_interleave(samples, len_u, output_size=n_u)
        seg_c = torch.repeat_interleave(samples, len_c, output_size=n_c)
        dst_u = torch.arange(n_u, dtype=torch.int64) + off_c[seg_u]
        dst_c = torch.arange(n_c, dtype=torch.int64) + off_u[seg_c + 1]
        out = torch.empty(n_u + n_c, dtype=ri_u.dtype)
        out.scatter_(0, dst_u, ri_u)
        out.scatter_(0, dst_c, ri_c)
        return out

    def _merged_case(self, seed: int, B: int = 37):
        g = torch.Generator().manual_seed(seed)
        len_u = torch.randint(0, 2500, (B,), generator=g)
        len_u[3] = 0
        len_c = torch.randint(0, 2, (B,), generator=g)
        n_u, n_c = int(len_u.sum()), int(len_c.sum())
        ri_u = torch.randint(0, 1 << 40, (n_u,), generator=g)
        ri_c = torch.randint(0, 1 << 40, (n_c,), generator=g)
        return ri_u, ri_c, len_u, len_c, n_u, n_c

    def test_merged_index(self) -> None:
        for s in range(3):
            ri_u, ri_c, len_u, len_c, n_u, n_c = self._merged_case(s)
            ref = self._merged_ref(ri_u, ri_c, len_u, len_c, n_u, n_c)
            out = gf.merged_index(ri_u.to(DEV), ri_c.to(DEV), len_u.to(DEV), len_c.to(DEV), n_u, n_c)
            self.assertTrue(torch.equal(out.cpu(), ref))
            # int32 lengths (the shipped path casts them to int64 first)
            out32 = gf.merged_index(ri_u.to(DEV), ri_c.to(DEV), len_u.int().to(DEV), len_c.int().to(DEV), n_u, n_c)
            self.assertTrue(torch.equal(out32.cpu(), ref))

    def test_merged_index_stacked(self) -> None:
        B = 23
        g = torch.Generator().manual_seed(5)
        len_u = torch.randint(0, 900, (B,), generator=g)
        len_c = torch.randint(0, 2, (B,), generator=g)
        n = int(len_u.sum() + len_c.sum())
        parts, refs = [], []
        for q in range(2):
            ri_u = torch.randint(0, 1 << 30, (int(len_u.sum()),), generator=g)
            ri_c = torch.randint(0, 1 << 30, (int(len_c.sum()),), generator=g)
            refs.append(self._merged_ref(ri_u, ri_c, len_u, len_c, int(len_u.sum()), int(len_c.sum())))
            parts.append((ri_u.to(DEV), ri_c.to(DEV), len_u.to(DEV), len_c.to(DEV)))
        ref = torch.stack(refs, dim=1).reshape(-1)
        out = gf.merged_index_stacked(parts, n)
        self.assertTrue(torch.equal(out.cpu(), ref))

    def test_complete_cumsum_many(self) -> None:
        g = torch.Generator().manual_seed(11)
        ls = [torch.randint(0, 3, (1024,), generator=g) for _ in range(8)]
        ls[2].zero_()
        outs = gf.complete_cumsum_many([x.to(DEV) for x in ls])
        for o, x in zip(outs, ls):
            self.assertEqual(o.dtype, torch.int64)
            self.assertTrue(o.is_contiguous())
            self.assertTrue(torch.equal(o.cpu(), _complete_cumsum(x)))

    def test_contextual_pad_cat(self) -> None:
        self._run_contextual(13, trailing=False)

    def test_contextual_pad_cat_trailing_rows(self) -> None:
        """Payload rows past offsets[-1] get a zero gradient, like the
        zero-initialised j2pd backward (iter65 port; not a production layout)."""
        self._run_contextual(14, trailing=True)

    def _run_contextual(self, seed: int, trailing: bool) -> None:
        g = torch.Generator().manual_seed(seed)
        B, D = 33, 512
        max_lens = [1, 1, 2, 1, 1]
        dtypes = [torch.float32, torch.bfloat16, torch.float32, torch.float32, torch.bfloat16]
        vals, offs = [], []
        for k, (L, dt) in enumerate(zip(max_lens, dtypes)):
            lens = torch.randint(0, 3, (B,), generator=g)  # some > L: truncated rows
            if k == 3:
                lens.fill_(1)
            off = _complete_cumsum(lens)
            extra = (k + 1) if trailing else 0
            v = (torch.randn((int(off[-1]) + extra, D), generator=g) * 3.0).to(dt)
            if dt == torch.float32 and v.numel() > 2:
                v.view(-1)[0] = 1.0 + 3 * 2.0**-8
                v.view(-1)[1] = -0.0
            vals.append(v)
            offs.append(off)
        # shipped: .to(bf16) -> jagged_to_padded_dense(.., [L], 0.0) -> flatten -> cat
        refs_in = [v.clone().requires_grad_(True) for v in vals]
        padded = []
        for v, off, L in zip(refs_in, offs, max_lens):
            v16 = v.to(torch.bfloat16)
            rows = []
            for b in range(B):
                n = min(int(off[b + 1] - off[b]), L)
                r = v16[int(off[b]):int(off[b]) + n]
                if n < L:
                    r = torch.cat([r, torch.zeros((L - n, D), dtype=torch.bfloat16)])
                rows.append(r)
            padded.append(torch.stack(rows).flatten(1, 2))
        ref = torch.cat(padded, dim=1)
        new_in = [v.clone().to(DEV).requires_grad_(True) for v in vals]
        out = gf.contextual_pad_cat(new_in, [o.to(DEV) for o in offs], max_lens)
        self.assertTrue(torch.equal(_bits16(out).cpu(), _bits16(ref.detach())))
        go = (torch.randn(ref.shape, generator=g)).to(torch.bfloat16)
        ref.backward(go)
        go_dev = go.to(DEV)
        # Leave NaN garbage in the allocator's free blocks so that an
        # uninitialised gradient row would show up.
        for v in vals:
            junk = torch.full(v.shape, float("nan"), dtype=v.dtype, device=DEV)
            del junk
        out.backward(go_dev)
        for a, b, off in zip(new_in, refs_in, offs):
            self.assertEqual(a.grad.dtype, b.grad.dtype)
            bits = _bits32 if a.dtype == torch.float32 else _bits16
            self.assertTrue(torch.equal(bits(a.grad).cpu(), bits(b.grad)))
            if trailing:
                self.assertTrue(bool((bits(a.grad[int(off[-1]):]).cpu() == 0).all()))

    # ---------------------------------------------------------------------
    # 4. Timestamp concat folds (DLRM_HSTU_FP8_CONCAT_FOLD)
    # ---------------------------------------------------------------------
    def test_concat_fold_switch_parse(self) -> None:
        """Default-on, off only for an explicit 0/false/off/no (the _env_on parse)."""
        name = "DLRM_HSTU_FP8_GLUE_TEST_SWITCH"
        old = os.environ.get(name)
        try:
            for val, expect in (("0", False), ("false", False), ("OFF", False), (" no ", False),
                                ("", True), ("1", True), ("on", True)):
                os.environ[name] = val
                self.assertEqual(gf._concat_fold_env(name), expect, val)
            os.environ.pop(name)
            self.assertTrue(gf._concat_fold_env(name))
        finally:
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_ts_concat_1d(self) -> None:
        """== concat_2D_jagged(left[:, None], right[:, None], off_l, off_r).squeeze(-1)."""
        g = torch.Generator().manual_seed(21)
        cases = [
            (37, 0, 700, 1, 1),  # B, min len, max len, max candidates, max_seq_len exact?
            (64, 0, 3, 3, 1),
            (1, 0, 2049, 1, 1),
            (1, 0, 0, 0, 1),
            (19, 0, 1500, 2, 0),  # max_seq_len underestimated: every row still written
        ]
        for B, lo, hi, mc, exact in cases:
            lu = torch.randint(lo, hi + 1, (B,), generator=g)
            lc = torch.randint(0, mc + 1, (B,), generator=g)
            left = torch.randint(-(2**63), 2**63 - 1, (int(lu.sum()),), generator=g)
            right = torch.randint(-(2**63), 2**63 - 1, (int(lc.sum()),), generator=g)
            off_l, off_r = _complete_cumsum(lu), _complete_cumsum(lc)
            ref = _shipped_concat(left, right, off_l, off_r)
            msl = int((lu + lc).max()) if exact else 1
            out = gf.jagged_concat_1d(left.to(DEV), right.to(DEV), off_l.to(DEV), off_r.to(DEV), msl)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.equal(out.cpu(), ref), (B, lo, hi, mc, exact))

    def test_ctx_ts_pad(self) -> None:
        """== (concat_2D_jagged(zeros[B*L, 1], ts[:, None], max_len_left=L,
        offsets_right=off).squeeze(-1), lens + L, complete_cumsum(lens + L))."""
        g = torch.Generator().manual_seed(22)
        for B, hi, L, exact in ((41, 900, 8, 1), (7, 3, 1, 1), (1, 0, 8, 1), (23, 2000, 3, 0)):
            lens = torch.randint(0, hi + 1, (B,), generator=g)
            ts = torch.randint(-(2**63), 2**63 - 1, (int(lens.sum()),), generator=g)
            off = _complete_cumsum(lens)
            zeros = torch.zeros(B * L, dtype=torch.int64)
            ref_ts = _shipped_concat(zeros, ts, torch.arange(0, B + 1) * L, off)
            ref_lens = lens + L
            ref_off = _complete_cumsum(ref_lens)
            msl = int(lens.max()) if exact else 1
            o_ts, o_lens, o_off = gf.ctx_ts_pad(ts.to(DEV), lens.to(DEV), off.to(DEV), L, msl)
            self.assertTrue(torch.equal(o_ts.cpu(), ref_ts), (B, hi, L))
            self.assertTrue(torch.equal(o_lens.cpu(), ref_lens), (B, hi, L))
            self.assertTrue(torch.equal(o_off.cpu(), ref_off), (B, hi, L))
            self.assertEqual((o_lens.dtype, o_off.dtype), (torch.int64, torch.int64))

    def test_ts_fold_gates(self) -> None:
        B = 5
        lens = torch.ones(B, dtype=torch.int64, device=DEV)
        off = torch.arange(0, B + 1, dtype=torch.int64, device=DEV)
        ts = torch.arange(0, B, dtype=torch.int64, device=DEV)
        on = DEV.type == "cuda"
        self.assertEqual(gf.uih_ts_concat_fusable(ts, ts, off, off), on and gf.CONCAT_FOLD_UIH_TS)
        self.assertEqual(gf.ctx_ts_pad_fusable(ts, lens, off, 8), on and gf.CONCAT_FOLD_CTX_TS)
        # declined: float values, int32 offsets, missing offsets, 2-D values, L = 0
        self.assertFalse(gf.uih_ts_concat_fusable(ts.float(), ts, off, off))
        self.assertFalse(gf.uih_ts_concat_fusable(ts, ts, off.int(), off))
        self.assertFalse(gf.uih_ts_concat_fusable(ts, ts, None, off))
        self.assertFalse(gf.ctx_ts_pad_fusable(ts[:, None], lens, off, 8))
        self.assertFalse(gf.ctx_ts_pad_fusable(ts, lens, off, 0))

    # ---------------------------------------------------------------------
    # 5. GLUE_FOLD4 (DLRM_HSTU_FP8_GLUE_FOLD4): each test runs the site with
    #    its switch off and on (module globals, restored afterwards) and
    #    compares raw bits. ACTSUM and POST rely on device behaviour (ATen's
    #    GPU reduce config, ocml cos/sin) and run on CUDA only.
    # ---------------------------------------------------------------------
    def _g4(self, **kw):  # pyre-ignore[3]
        old = (gf.G4_ACTSUM, gf.G4_CTX1, gf.G4_OFFS, gf.G4_POST)
        gf.G4_ACTSUM = kw.get("actsum", False)
        gf.G4_CTX1 = kw.get("ctx1", False)
        gf.G4_OFFS = kw.get("offs", False)
        gf.G4_POST = kw.get("post", False)
        return old

    def _g4_restore(self, old) -> None:  # pyre-ignore[2]
        gf.G4_ACTSUM, gf.G4_CTX1, gf.G4_OFFS, gf.G4_POST = old

    @unittest.skipUnless(DEV.type == "cuda", "GPU reduce config")
    def test_g4_actsum_bitwise(self) -> None:
        for seed in (21, 22):
            actions, weights, table, target, off_a, off_b, T, msl = _case(seed)
            res = []
            for on in (False, True):
                old = self._g4(actsum=on)
                try:
                    t = table.to(DEV).clone().requires_grad_(True)
                    r = target.to(DEV).tile(T, 1).clone().requires_grad_(True)
                    out = gf.action_encode_concat(actions.to(DEV), weights.to(DEV), t.unsqueeze(0), r,
                                                  off_a.to(DEV), off_b.to(DEV), msl, True)
                    g = torch.Generator().manual_seed(seed)
                    go = torch.randn(out.shape, generator=g).to(torch.bfloat16)
                    go.view(-1)[:7] = torch.tensor([0.0, -0.0, 1e-40, -3.0, 2.0**-130, 65504.0, -1e-30])
                    out.backward(go.to(DEV))
                    res.append((_bits32(t.grad).cpu(), _bits32(r.grad).cpu()))
                finally:
                    self._g4_restore(old)
            self.assertTrue(torch.equal(res[0][0], res[1][0]))
            self.assertTrue(torch.equal(res[0][1], res[1][1]))

    def test_g4_ctx1_matches_per_feature(self) -> None:
        for on in (False, True):
            old = self._g4(ctx1=on)
            try:
                self._run_contextual(31, trailing=False)
                self._run_contextual(32, trailing=True)
            finally:
                self._g4_restore(old)

    def test_g4_offsets(self) -> None:
        g = torch.Generator().manual_seed(5)
        old = self._g4(offs=DEV.type == "cuda")
        try:
            for B in (1, 2, 37, 1024):
                a = torch.randint(0, 4, (B,), generator=g)
                b = torch.randint(0, 3000, (B,), generator=g) + a
                if DEV.type != "cuda":
                    self.assertIsNone(gf.g4_offsets(a, b, preproc=True))
                    continue
                t, s, u, o = gf.g4_offsets(a.to(DEV), b.to(DEV), preproc=True)
                pa, pb = gf.g4_offsets(a.to(DEV), b.to(DEV), preproc=False)
                ra, rb = _complete_cumsum(a), _complete_cumsum(b)
                for x, y in ((t, ra), (s, rb), (u, rb - ra), (o, rb), (pa, ra), (pb, rb)):
                    self.assertEqual(x.dtype, torch.int64)
                    self.assertTrue(torch.equal(x.cpu(), y), B)
            if DEV.type == "cuda":  # declines: int32, non-contiguous
                x32 = torch.ones(8, dtype=torch.int32, device=DEV)
                self.assertIsNone(gf.g4_offsets(x32, x32, preproc=True))
                nc = torch.ones(16, dtype=torch.int64, device=DEV)[::2]
                self.assertIsNone(gf.g4_offsets(nc, nc, preproc=True))
        finally:
            self._g4_restore(old)

    @unittest.skipUnless(DEV.type == "cuda", "ocml cos/sin")
    def test_g4_time_concat(self) -> None:
        g = torch.Generator().manual_seed(9)
        B, D = 67, 64
        lens = torch.randint(1, 40, (B,), generator=g)
        N = int(lens.sum())
        ts = torch.randint(0, 1 << 41, (N,), generator=g)
        ts[:5] = torch.tensor([0, 3599, 3600, 86400, 1_700_000_000])
        pu = torch.tensor([3600.0, 86400.0])
        upp = torch.tensor([24.0, 7.0])
        emb = (torch.randn((B, D), generator=g)).to(torch.bfloat16)
        rows = _complete_cumsum(lens)[1:] - 1
        ref_tf = gf.g4_time_features_reference(ts[rows].to(DEV), pu.view(1, -1).to(DEV),
                                               upp.view(1, -1).to(DEV), torch.bfloat16)
        ref = torch.cat([emb.to(DEV), ref_tf], dim=-1)
        got = gf._g4_time_concat_launch(emb.to(DEV), lens.to(DEV), ts.to(DEV), pu.to(DEV), upp.to(DEV))
        self.assertTrue(torch.equal(_bits16(got).cpu(), _bits16(ref).cpu()))


if __name__ == "__main__":
    unittest.main()
