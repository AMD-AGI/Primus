###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime of ``mxfp6_packed_param_gather``: the MXFP6 linear weights travel as their GEMM operands, not as bf16.

The DDP buffer is laid out by ``packed_param_gather_patches`` so that in every weight bucket (``mxfp6_kind`` W6 /
W4) each shard boundary falls on a 256-row edge of the weight it cuts. Every operand a weight's GEMMs read is
*range-closed* in that unit -- any 256-row range of the weight is one contiguous byte range of it:

* forward B (the weight's rows): the MXFP6 C0 / C1 planes (W6) or the FP4 rows (W4), and the role-B scale slab,
  whose 256-row tiles are outermost;
* dgrad B (the weight's columns, contracting over its rows): FP4 codes and scale slab K256-outer (aiter tilescale
  ``kouter``; a 3-D codes tensor tells the GEMM).

So a bucket keeps one byte plane per operand at a constant number of bytes per element, a rank packs the rows it
owns straight into its shard of every plane (from the rank's own updated bf16 rows), and the collectives land every
weight's operands complete and in place. The GEMMs read views of the planes; nothing is assembled after the
gather, and the bf16 weight is never gathered.

The forward planes are round to nearest, the same bytes on every rank: one all-gather. The dgrad copy is
stochastically rounded (``fp4_sr_actw``), and each rank must see its own draw, as when it packs the weight itself
(the SR seeds differ per rank, so the rounding noise of the weight averages over the data-parallel gradient
reduction): the owner's pack emits one draw of its rows' codes per destination rank from a single read, and the
dgrad code plane goes by all-to-all (the same bytes per receiver as a gather); its scales are draw-independent and
gathered with the rest. Round to nearest, the draws are equal and the dgrad planes are gathered.

With ``packed_param_gather_prob_bits`` (4 or 2) the draws are made by the receivers instead: the owner packs one copy
of the dgrad codes rounded down plus a plane of round-up probabilities (4 bits: a nibble per code in the codes' layout;
2 bits: half the bytes), both are
all-gathered with the forward planes, and after the gather each rank rounds every code up with its probability from
its own seed (``receive``). The ranks' draws stay independent; nothing goes by all-to-all.

With ``packed_param_gather_neutral`` (W6 buckets) only the forward planes travel, in an orientation-neutral form: MXFP6
rows with no Hadamard and one scale per 32x32 tile (the A6W6 forward reads them as they arrive). The dgrad copy is
made by every rank from the gathered planes (``receive``: Turbo ``mxfp6_tile_to_fp4_col``, stochastically rounded from
the rank's own seed, or round to nearest); its planes are local, not gathered. Mode ``d2`` gathers today's forward
rows instead (H32 along K, per-32 scales), and the receiver un-rotates them before packing the round-to-nearest copy.
"""

import hashlib

import torch

_A2A = ("cc",)  # the dgrad copy's codes: one SR draw per destination (its scales do not depend on the draw)
_PROB = "cp"  # prob4: the dgrad codes' round-up probabilities (4 or 2 bits per code, the codes' element order)
_DGRAD = ("cc", "cs", _PROB)  # the dgrad copy's planes (the rest are the forward GEMM's)
SCALES = {"rs": "fwd", "cs": "bwd"}  # the scale planes and the phase whose GEMMs read them
DENSITY = {  # plane -> elements per byte
    "W6": {"c0": 2, "c1": 4, "rs": 32, "cc": 2, "cs": 32},
    "W4": {"r4": 2, "rs": 32, "cc": 2, "cs": 32},
}
ROWS = 256  # the range-closed unit


def _seed(step, bucket, idx, ra):
    """The SR seed of one owner pack: unique per (step, bucket, weight, row range) -- weights must not share random
    words, or their rounding errors correlate across layers."""
    h = hashlib.blake2b(f"{step}:{bucket}:{idx}:{ra}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "little")


class PackedBucket:
    """The planes of one W6 / W4 bucket and the per-weight views into them."""

    def __init__(self, bucket, dp, rank, fmts, external=None):
        """``external``: plane name -> a caller-owned uint8 tensor of that plane's size to use as its storage (the
        ``phased_ar`` transport keeps every bucket's scale planes in two shared buffers)."""
        self.bucket, self.dp, self.rank, self.kind = bucket, dp, rank, bucket.mxfp6_kind
        self.S, self.fmts, self.step = bucket.mxfp6_shard, fmts, 0
        n = bucket.param_data.numel()
        dev = bucket.param_data.device
        assert n == dp * self.S and all(n % d == 0 for d in DENSITY[self.kind].values())
        external = external or {}
        self.planes = {}
        for k, d in DENSITY[self.kind].items():
            t = external.get(k)
            assert t is None or (t.dtype == torch.uint8 and t.numel() == n // d), (k, None if t is None else t.shape)
            self.planes[k] = t if t is not None else torch.empty(n // d, dtype=torch.uint8, device=dev)
        # prob4: one dgrad copy (floor codes + probabilities) for every rank, rounded by each receiver
        self.prob_bits = int(fmts.get("col_prob_bits", 0))
        self.prob4 = self.prob_bits != 0
        if self.prob4:
            self.planes[_PROB] = torch.empty(n // (8 // self.prob_bits), dtype=torch.uint8, device=dev)
        # per-destination draws of this rank's dgrad rows ([dp, shard]: row d goes to rank d), SR only
        # neutral (W6): the forward planes alone travel; each rank makes its dgrad copy from them (sr or rn)
        self.neutral = fmts.get("neutral", "") if self.kind == "W6" else ""
        if self.neutral:
            self.prob_bits, self.prob4 = 0, False
            self.planes.pop(_PROB, None)
            self._none = torch.empty(0, dtype=torch.uint8, device=dev)
        self.sr = bool(fmts["col_sr"]) and not self.prob4 and not self.neutral
        self.send = {k: torch.empty_like(self.planes[k]) for k in _A2A} if self.sr else None
        self.items = []
        self.fused_done = set()  # (idx, ra) packed by the optimizer step (fused) since the last owner_pack
        for idx, p in enumerate(bucket.params_list):
            s, e = bucket.param_to_index[p]
            R, K = p.shape
            assert (e - s) == R * K and R % ROWS == 0 and K % 256 == 0, (p.shape, s, e)
            pieces = []
            for r in range(dp):
                lo, hi = max(s, r * self.S), min(e, (r + 1) * self.S)
                if lo < hi:
                    assert (lo - s) % (ROWS * K) == 0 and (hi - s) % (ROWS * K) == 0
                    pieces.append((r, (lo - s) // K, (hi - s) // K))
            self.items.append((idx, p, s, R, K, pieces))
            self._attach(p, s, R, K)

    def _view(self, plane, s, nel):
        d = (8 // self.prob_bits) if plane == _PROB else DENSITY[self.kind][plane]
        return self.planes[plane][s // d : (s + nel) // d]

    def _attach(self, p, s, R, K):
        if self.kind == "W6":
            p._ppg_row = self._view("c0", s, R * K)  # forward B: C0 plane (C1 in _ppg_c1)
            p._ppg_c1 = self._view("c1", s, R * K)
        else:
            p._ppg_row = self._view("r4", s, R * K).view(R, K // 2)
            p._ppg_c1 = None
        p._ppg_row_s = self._view("rs", s, R * K)
        p._ppg_col = self._view("cc", s, R * K).view(R // 256, K, 128)  # dgrad B, K256-outer
        p._ppg_col_s = self._view("cs", s, R * K)
        p._ppg_kind = self.kind

    def owner_pieces(self):
        """(item index, rows ra..rb) of every piece this rank owns."""
        for idx, p, s, R, K, pieces in self.items:
            for r, ra, rb in pieces:
                if r == self.rank:
                    yield idx, ra, rb

    def owner_pack(self):
        """Pack the rows this rank owns into its shard of every plane (and, SR, one dgrad draw per destination).
        Pieces the optimizer step already packed this step (``pack_piece(..., adam=...)``) are skipped."""
        self.step += 1
        done, self.fused_done = self.fused_done, set()
        for idx, ra, rb in self.owner_pieces():
            if (idx, ra) not in done:
                self.pack_piece(idx, ra, rb, self.step)

    def pack_piece(self, idx, ra, rb, step, adam=None):
        """Pack rows ra..rb of item ``idx`` for ``step``. ``adam``: (grad, exp_avg, exp_avg_sq, remainder, hyper)
        applies that optimizer step to the rows first, in the same kernel (``quantize_mx_dual_out_adam``)."""
        from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX
        from primus_turbo.pytorch.ops.quantization import set_sr_seed_next_pack

        _, p, s, R, K, _ = self.items[idx]
        w6 = self.kind == "W6"
        o, nel, w, fmt = s + ra * K, (rb - ra) * K, p.data[ra:rb], self.fmts[self.kind](R, K)
        rows = (
            self._view("c0" if w6 else "r4", o, nel),
            self._view("rs", o, nel),
            dict(row_c1=self._view("c1", o, nel) if w6 else None),
        )
        if self.neutral:  # the forward rows alone (no column direction: the receivers make it)
            assert adam is None, "packed_param_gather_neutral is not wired into the fused Adam + owner pack"
            MX.quantize_mx_dual_out(w, rows[0], rows[1], self._none, self._none, fmt, **rows[2])
            return
        if self.prob4:  # the codes rounded down + their round-up probabilities, finished by each receiver
            assert adam is None, "packed_param_gather_prob4 is not wired into the fused Adam + owner pack"
            from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import MX_FMT_FP4_COL_SR
            fmt &= ~MX_FMT_FP4_COL_SR
            cols = (self._view("cc", o, nel), self._view("cs", o, nel))
            kw = dict(rows[2], col_prob=self._view(_PROB, o, nel))
        elif not self.sr:  # round to nearest: one dual pack into the planes
            cols, kw = (self._view("cc", o, nel), self._view("cs", o, nel)), rows[2]
        else:
            # one read of the rows: the forward rows and every destination's dgrad draw (draw d -> rank d's row of
            # the send buffers, a shard apart)
            set_sr_seed_next_pack(_seed(step, self.bucket.bucket_id, idx, ra))
            cols = (self._send("cc", 0, o, nel), self._view("cs", o, nel))
            kw = dict(rows[2], draws=self.dp, draw_codes=self.S // DENSITY[self.kind]["cc"], draw_scales=0)
        if adam is None:
            MX.quantize_mx_dual_out(w, rows[0], rows[1], *cols, fmt, **kw)
        else:
            grad, m, v, rem, hyper = adam
            MX.quantize_mx_dual_out_adam(w, grad.view_as(w), m.view_as(w), v.view_as(w), rem.view_as(w), rows[0],
                                         rows[1], *cols, fmt, **hyper, **kw)
            self.fused_done.add((idx, ra))

    def _send(self, plane, dst, s, nel):
        """This rank's shard positions [s, s + nel) of the draw for rank ``dst``."""
        d, sh = DENSITY[self.kind][plane], self.S // DENSITY[self.kind][plane]
        base = dst * sh - self.rank * sh
        return self.send[plane][base + s // d : base + (s + nel) // d]

    def gather_ops(self, phase=None, scales=True):
        """(output, input) of each all-gathered plane; ``phase`` "fwd" / "bwd": only the forward GEMM's planes (rows and
        their scales) / only the dgrad copy's (column codes and scales, the prob4 probabilities); ``scales=False``
        leaves the scale planes out (``phased_ar`` all-reduces them instead)."""
        local = ("cc", "cs") if self.neutral else (_A2A if self.sr else ())
        return [(pl, pl.view(self.dp, -1)[self.rank]) for k, pl in self.planes.items()
                if k not in local and (phase is None or (k in _DGRAD) == (phase == "bwd"))
                and (scales or k not in SCALES)]

    def receive(self):
        """After the gather. prob4: round this rank's copy of every dgrad code up with its probability -- one draw,
        seeded by (step, bucket, rank), so the ranks' draws are independent as with per-rank packs. neutral: make every
        weight's dgrad copy from its gathered forward planes (sr: seeded by (step, bucket, weight, rank))."""
        if self.neutral:
            from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

            for idx, p, s, R, K, _ in self.items:
                fmt = self.fmts[self.kind](R, K)
                MX.mxfp6_tile_to_fp4_col(p._ppg_row, p._ppg_c1, p._ppg_row_s, R, K, p._ppg_col.reshape(-1),
                                         p._ppg_col_s, fmt, self.neutral == "sr",
                                         _seed(self.step, self.bucket.bucket_id, idx, f"rx{self.rank}"))
        if self.prob4:
            from primus_turbo.triton.quantization.fp4_prob_round import fp4_prob_round

            seed = _seed(self.step, self.bucket.bucket_id, "rx", self.rank)
            fp4_prob_round(self.planes["cc"], self.planes[_PROB], seed)

    def a2a_ops(self):
        """(output, input) of each all-to-all plane (SR dgrad copy): row r of the output is owner r's draw for us."""
        return [(self.planes[k], self.send[k]) for k in _A2A] if self.sr else []
