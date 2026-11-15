"""Bitwise tests for the fused seq-a2a path (seq_a2a_fused.py + seq_a2a_rowperm.py).

Send side: K1/K2 (``SendFwdTest``, ``SendBwdTest``, ``RowMapTest``, ``IntegrationTest``).
Receive side (tier 1): K3'/K4 (``RecvKernelTest``) and the patched
``All2All_Seq_Req_Wait`` + unbucketize hand-off through the REAL torchrec classes
(``RecvIntegrationTest``, GPU + real torchrec only).

Every comparison is on raw-bit integer views (int16/int32/int64), never on floats.

GPU (training container, real fbgemm/torchrec):
    HIP_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/recommendation:$PYTHONPATH \
        python3 -B <tree>/recommendation/generative_recommenders/dlrm_v4/tests/seq_a2a_fused_test.py -v
CPU (Triton interpreter): OPT-IN only, because it changes process state
(TRITON_INTERPRET, and stubs for missing fbgemm/torchrec) that other test files
collected in the same pytest process would inherit:
    SEQ_A2A_FUSED_TEST_CPU=1 PYTHONPATH=<tree>/recommendation python3 -B .../seq_a2a_fused_test.py -v
Without a GPU and without the opt-in every test here is SKIPPED (e.g. the CPU-only
correctness pytest run), and importing this file changes nothing.
On CPU without fbgemm the stock ``permute_2D_sparse_data`` is replaced by a pure-torch
reference, torchrec's ``comm_ops`` by a stub, and ``torch.distributed`` by mocks.
The CPU path is a test of the index math and the control flow; the GPU run is the
numerics arbiter. CPU-only relaxation: for *signaling* NaN inputs numpy (the
interpreter's float conversions) keeps the NaN signaling while CPU torch quiets
it, so those elements are compared with the quiet bit forced on. Quiet NaN
payloads, sign, inf, zeros and subnormals are always compared exactly.
"""

import os
import sys
import types
import unittest
import warnings
from typing import Dict, List, Optional, Tuple

import torch

_HAS_GPU = torch.cuda.is_available()
_CPU_OPT_IN = (not _HAS_GPU) and os.environ.get("SEQ_A2A_FUSED_TEST_CPU", "") == "1"
_RUN = _HAS_GPU or _CPU_OPT_IN
_SKIP_WHY = "GPU only (set SEQ_A2A_FUSED_TEST_CPU=1 for the Triton-interpreter CPU run)"
if _CPU_OPT_IN:
    os.environ.setdefault("TRITON_INTERPRET", "1")
if _RUN:  # numpy overflow/invalid warnings from the bit-pattern sweeps; not set when skipping
    warnings.filterwarnings("ignore", message=".*overflow encountered.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

try:  # real fbgemm ops (GPU container)
    import fbgemm_gpu  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from generative_recommenders.dlrm_v4 import seq_a2a_fused as saf  # noqa: E402
from generative_recommenders.dlrm_v4 import seq_a2a_rowperm as sarp  # noqa: E402
from generative_recommenders.dlrm_v4 import unbucketize_gather as ubg  # noqa: E402

DEV = torch.device("cuda") if _HAS_GPU else torch.device("cpu")
HMIN, HMAX = -65504.0, 65504.0


def _have_fbgemm_permute() -> bool:
    try:
        return callable(torch.ops.fbgemm.permute_2D_sparse_data)
    except Exception:  # noqa: BLE001
        return False


_REAL_FBGEMM = _have_fbgemm_permute()


def _ref_permute_2D_sparse_data(permute, lengths, values, weights=None, permuted_lengths_sum=None):
    """Pure-torch fbgemm.permute_2D_sparse_data (2-D lengths, no weights)."""
    T = lengths.size(0)
    l2 = lengths.reshape(T, -1)
    perm = permute.long()
    permuted_lengths = torch.index_select(l2, 0, perm)
    blk = l2.sum(dim=1).cpu()
    starts = [0]
    for v in blk.tolist():
        starts.append(starts[-1] + int(v))
    pieces = [values[starts[p]: starts[p] + int(blk[p])] for p in perm.cpu().tolist()]
    out = torch.cat(pieces) if pieces else values[:0].clone()
    return permuted_lengths, out, None


if not _REAL_FBGEMM and _CPU_OPT_IN:
    setattr(torch.ops.fbgemm, "permute_2D_sparse_data", _ref_permute_2D_sparse_data)


def _have_torchrec() -> bool:
    try:
        from torchrec.distributed import comm_ops  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


_REAL_TORCHREC = _have_torchrec()
if not _REAL_TORCHREC and _CPU_OPT_IN:  # stub just enough of torchrec.distributed.comm_ops
    _tr = types.ModuleType("torchrec")
    _trd = types.ModuleType("torchrec.distributed")
    _co = types.ModuleType("torchrec.distributed.comm_ops")
    _co.GRADIENT_DIVISION = True
    _co.All2All_Seq_Req_Wait = object()
    _trd.comm_ops = _co
    _tr.distributed = _trd
    sys.modules.setdefault("torchrec", _tr)
    sys.modules.setdefault("torchrec.distributed", _trd)
    sys.modules.setdefault("torchrec.distributed.comm_ops", _co)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def ibits(t: torch.Tensor) -> torch.Tensor:
    it = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(it)


def lowmem_encode(x: torch.Tensor) -> torch.Tensor:
    """train/utils.py _lowmem_fp16 (the shipped codec encode)."""
    return x.half().clamp_(HMIN, HMAX)


def fbgemm_stock_encode(x: torch.Tensor) -> torch.Tensor:
    """fbgemm quantize_utils.fp32_to_fp16_with_clamp (unpatched)."""
    return torch.clamp(x, HMIN, HMAX).half()


def decode(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def assert_bits(tc: unittest.TestCase, got: torch.Tensor, ref: torch.Tensor, what: str,
                nan_src: Optional[torch.Tensor] = None, nan_mode: str = "sign") -> None:
    tc.assertEqual(got.dtype, ref.dtype, what)
    tc.assertEqual(tuple(got.shape), tuple(ref.shape), what)
    g, r = ibits(got).reshape(-1), ibits(ref).reshape(-1)
    if nan_src is not None and not _HAS_GPU:
        # CPU interpreter only (the GPU run is strict everywhere):
        # * nan_mode="sign" (fp32 -> fp16, nan_src = signaling-NaN sources): numpy keeps
        #   the NaN signaling (payload bit 0 set if the truncated payload is 0), CPU torch
        #   quiets it. Both must be NaN with the same sign.
        # * nan_mode="any" (fp16 -> fp32, nan_src = all NaN sources): torch 2.14's CPU
        #   .float() scalar tail returns 0x7FFFFFFF for any NaN (vector body keeps
        #   sign/payload), so only NaN-ness is comparable.
        sn = nan_src.reshape(-1).to(g.device)
        isnan_g = torch.isnan(got.reshape(-1).float())
        isnan_r = torch.isnan(ref.reshape(-1).float())
        tc.assertTrue(bool((isnan_g[sn] & isnan_r[sn]).all()), what + ": NaN source not NaN")
        if g.dtype == torch.int16:
            sign, canon = -0x8000, 0x7E00
        else:
            sign, canon = -0x80000000, 0x7FC00000
        if nan_mode == "sign":
            g = torch.where(sn, (g & sign) | canon, g)
            r = torch.where(sn, (r & sign) | canon, r)
        else:
            g = torch.where(sn, torch.full_like(g, canon), g)
            r = torch.where(sn, torch.full_like(r, canon), r)
    bad = (g != r).nonzero().flatten()
    if bad.numel():
        i = bad[:8].tolist()
        m = 0xFFFF if g.dtype == torch.int16 else 0xFFFFFFFF
        tc.fail(f"{what}: {bad.numel()} raw-bit mismatches, first at {i}: "
                f"got {[hex(v & m) for v in g[bad[:8]].tolist()]} "
                f"ref {[hex(v & m) for v in r[bad[:8]].tolist()]}")


def snan_mask_f32(x: torch.Tensor) -> torch.Tensor:
    b = ibits(x.float())
    return ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0) & ((b & 0x00400000) == 0)


def nan_mask_f16(h: torch.Tensor) -> torch.Tensor:
    b = ibits(h).to(torch.int32) & 0xFFFF
    return ((b & 0x7C00) == 0x7C00) & ((b & 0x03FF) != 0)


def quiet_snans_f32(x: torch.Tensor) -> torch.Tensor:
    b = ibits(x)
    return torch.where(snan_mask_f32(x), b | 0x00400000, b).view(torch.float32)


def snan_mask_f16(h: torch.Tensor) -> torch.Tensor:
    b = ibits(h).to(torch.int32) & 0xFFFF
    return ((b & 0x7C00) == 0x7C00) & ((b & 0x03FF) != 0) & ((b & 0x0200) == 0)


F32_SPECIALS = [
    0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFC12345, 0x7FE00000,
    0x7FD55555, 0x7F800001, 0xFF800001, 0x7FA00000, 0x7F802000, 0x7F801FFF, 0x7FBFFFFF,  # NaNs
    0x7F800000, 0xFF800000,  # +-inf
    0x00000000, 0x80000000,  # +-0
    0x477FE000, 0xC77FE000, 0x477FE001, 0x477FEFFF, 0xC77FEFFF, 0x477FF000, 0xC77FF000,
    0x477FF001, 0x477FFFFF, 0x47800000, 0xC7800000, 0x501502F9, 0x7F7FFFFF, 0xFF7FFFFF,  # 65504..
    0x33800000, 0xB3800000, 0x33000000, 0x33000001, 0xB3000001, 0x33400000, 0x32FFFFFF,
    0x33C00000, 0x387FC000, 0x387FE000, 0x387FF000, 0x38800000, 0x387FFFFF, 0xB87FFFFF,  # f16 sub
    0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00400000,  # f32 subnormals
    0x3F800000, 0xBF800000, 0x3F800FFF, 0x3F801000, 0x3F801001, 0x3F803000,  # RNE ties near 1
]


def f32_from_bits(bits: List[int]) -> torch.Tensor:
    v = torch.tensor([b - (1 << 32) if b >= (1 << 31) else b for b in bits], dtype=torch.int64)
    return v.to(torch.int32).view(torch.float32)


def make_emb(rows: int, D: int, gen: torch.Generator, special_frac: float = 0.05) -> torch.Tensor:
    x = torch.randn(rows * D, generator=gen) * 3.0
    x[::7] *= 3e4  # many values beyond the fp16 range
    x[1::11] *= 1e-6  # fp16 subnormal range
    n_sp = int(rows * D * special_frac)
    if n_sp:
        sp = f32_from_bits(F32_SPECIALS)
        pos = torch.randint(0, rows * D, (n_sp,), generator=gen)
        x[pos] = sp[torch.randint(0, sp.numel(), (n_sp,), generator=gen)]
        # random fp16-subnormal-range values with random signs, and random bit patterns
        pos2 = torch.randint(0, rows * D, (n_sp,), generator=gen)
        mag = torch.exp2(torch.empty(n_sp).uniform_(-25.5, -13.5, generator=gen))
        x[pos2] = mag * (torch.randint(0, 2, (n_sp,), generator=gen) * 2 - 1)
        pos3 = torch.randint(0, rows * D, (n_sp,), generator=gen)
        x[pos3] = torch.randint(-(1 << 31), (1 << 31) - 1, (n_sp,), generator=gen,
                                dtype=torch.int64).to(torch.int32).view(torch.float32)
    return x.view(rows, D)


def make_bwd_wire(rows: int, D: int, gen: torch.Generator) -> torch.Tensor:
    """fp16 wire with uniformly random bit patterns (NaN payloads, inf, subnormals, -0)."""
    b = torch.randint(-32768, 32768, (rows * D,), generator=gen, dtype=torch.int32)
    return b.to(torch.int16).view(torch.float16)


def torchrec_recats(local_T: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """SequenceEmbeddingsAllToAll's registered forward/backward recat buffers (int32)."""
    fwd = [j + i * W for j in range(W) for i in range(local_T)]
    bwd = [i + j * local_T for i in range(local_T) for j in range(W)]
    return torch.tensor(fwd, dtype=torch.int32), torch.tensor(bwd, dtype=torch.int32)


def get_recat(local_T: int, W: int, stagger: int) -> torch.Tensor:
    """torchrec dist_data._get_recat (sparse_features_recat), no variable batch."""
    order = [x + W // stagger * y for x in range(W // stagger) for y in range(stagger)]
    return torch.tensor([i + j * local_T for i in range(local_T) for j in order], dtype=torch.int32)


def invert(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p.long()] = torch.arange(p.numel(), dtype=p.dtype)
    return inv


def make_lengths(n_blocks: int, B: int, rows_target: int, gen: torch.Generator,
                 zero_frac: float = 0.25, dtype=torch.int64,
                 pattern: str = "random") -> torch.Tensor:
    """Token lengths [n_blocks, B] whose blocks include zero-length ones."""
    if pattern == "alt01":  # blocks of 0 and 1 rows alternating
        blk = torch.tensor([(k % 2) for k in range(n_blocks)], dtype=torch.int64)
    elif pattern == "one":  # all rows in one block
        blk = torch.zeros(n_blocks, dtype=torch.int64)
        blk[int(torch.randint(0, n_blocks, (1,), generator=gen))] = rows_target
    else:
        w = torch.rand(n_blocks, generator=gen) ** 2
        w[torch.rand(n_blocks, generator=gen) < zero_frac] = 0
        if n_blocks > 2:
            w[0] = 0  # leading and trailing empty blocks
            w[-1] = 0
        if float(w.sum()) == 0:
            w[n_blocks // 2] = 1
        blk = torch.floor(w / w.sum() * rows_target).to(torch.int64)
    tok = torch.zeros(n_blocks, B, dtype=torch.int64)
    for k in range(n_blocks):
        v = int(blk[k])
        if v == 0:
            continue
        cut = torch.sort(torch.randint(0, v + 1, (B - 1,), generator=gen)).values
        edges = torch.cat([torch.zeros(1, dtype=torch.int64), cut, torch.tensor([v])])
        tok[k] = edges[1:] - edges[:-1]
    return tok.to(dtype)


def case_tensors(local_T: int, W: int, B: int, rows_target: int, seed: int, *,
                 recat_kind: str = "torchrec", tok_dtype=torch.int64, recat_dtype=torch.int32,
                 pattern: str = "random"):
    gen = torch.Generator().manual_seed(seed)
    n_blocks = local_T * W
    tok = make_lengths(n_blocks, B, rows_target, gen, dtype=tok_dtype, pattern=pattern)
    if recat_kind == "torchrec":
        recat, bwd = torchrec_recats(local_T, W)
    elif recat_kind.startswith("get_recat"):  # sparse_features_recat path
        stagger = int(recat_kind.split(":")[1])
        bwd = get_recat(local_T, W, stagger)
        recat = invert(bwd)
    else:
        recat = torch.randperm(n_blocks, generator=gen).to(torch.int32)
        bwd = invert(recat)
    tok = tok.view(local_T, W * B)  # lengths_after_sparse_data_all2all layout
    return gen, n_blocks, tok.to(DEV), recat.to(recat_dtype).to(DEV), bwd.to(recat_dtype).to(DEV)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
CPU = not _HAS_GPU


class RowMapTest(unittest.TestCase):
    """In-kernel row index (K1/K2 helpers) == stock row_index (permute_2D over arange)."""

    def _check(self, local_T, W, B, rows_target, seed, **kw):
        _, n_blocks, tok, recat, bwd = case_tensors(local_T, W, B, rows_target, seed, **kw)
        blk = sarp.block_rows(tok, n_blocks)  # [n,1] int64 (aten::sum)
        rows = int(blk.sum())
        # Stock references: fbgemm permute_2D_sparse_data only takes an int32
        # permute (production recats are int32); the fused helpers get the
        # original dtype.
        pblk, fidx = sarp.row_index(recat.to(torch.int32), blk, rows)
        _, bidx = sarp.row_index(bwd.to(torch.int32), pblk, rows)
        fm = saf.fwd_row_map(blk.view(-1), recat, rows)
        bm = saf.bwd_row_map(blk.view(-1), recat, bwd, rows)
        assert_bits(self, fm, fidx, f"fidx n={n_blocks} rows={rows} {kw}")
        assert_bits(self, bm, bidx, f"bidx n={n_blocks} rows={rows} {kw}")
        # bwd_recat is the inverse of recat: bidx is the inverse permutation of fidx.
        if rows:
            inv = torch.empty_like(fidx)
            inv[fidx] = torch.arange(rows, device=fidx.device)
            assert_bits(self, bm, inv, "bidx == inverse(fidx)")
        # General semantics: an unrelated backward permutation also matches stock.
        other = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed + 1))
        other = other.to(recat.dtype).to(DEV)
        _, oidx = sarp.row_index(other.to(torch.int32), pblk, rows)
        assert_bits(self, saf.bwd_row_map(blk.view(-1), recat, other, rows), oidx,
                    "bidx with a non-inverse bwd_recat")

    def test_n_blocks_and_recats(self):
        cases = [(1, 1), (3, 1), (3, 8), (1, 8), (11, 32), (2, 8), (7, 32), (14, 32)]
        if not CPU:
            cases += [(1, 16), (11, 8), (28, 8), (14, 16), (56, 8)]
        for i, (local_T, W) in enumerate(cases):
            for kind in ("torchrec", "random", "get_recat:1"):
                rows_target = 3000 if CPU else 200_000
                self._check(local_T, W, 4, rows_target, 100 + i, recat_kind=kind)
        self._check(11, 32, 2, 2000, 7, recat_kind="get_recat:2")  # TWRW-style stagger

    def test_zero_length_patterns(self):
        for pattern in ("alt01", "one"):
            for (local_T, W) in ((3, 8), (11, 32), (1, 1)):
                self._check(local_T, W, 3, 1500, 11, pattern=pattern)
        # rows == 1, and all-but-one block empty with many blocks
        self._check(11, 32, 2, 1, 13, pattern="one")

    def test_dtypes(self):
        for tok_dtype in (torch.int32, torch.int64):
            for recat_dtype in (torch.int32, torch.int64):
                self._check(3, 8, 5, 700, 17, tok_dtype=tok_dtype, recat_dtype=recat_dtype)

    @unittest.skipIf(CPU, "large rows: GPU only")
    def test_large_rows(self):
        self._check(14, 32, 1024, 2_400_000, 19)  # 448 blocks, 2.4M rows
        self._check(2, 8, 1024, 335_000, 23, recat_kind="random")


class SendFwdTest(unittest.TestCase):
    """K1 (wire + permuted lengths) == stock fbgemm permute + encode, and == rowperm."""

    def _check(self, local_T, W, B, rows_target, D, seed, **kw):
        gen, n_blocks, tok, recat, bwd = case_tensors(local_T, W, B, rows_target, seed, **kw)
        rows = int(tok.sum())
        emb = make_emb(rows, D, gen).to(DEV)
        lengths_d = tok * D
        blk = sarp.block_rows(tok, n_blocks).view(-1)
        plen, wire = saf.send_fwd(emb, blk, recat, tok, n_blocks, D)
        snan = snan_mask_f32(emb.view(-1)) if CPU else None
        recat32 = recat.to(torch.int32)  # stock fbgemm permute: int32 only
        for name, enc in (("lowmem", lowmem_encode), ("fbgemm_stock", fbgemm_stock_encode)):
            ref_pl, ref_wire = sarp.stock_fwd(recat32, lengths_d, emb, n_blocks, enc)
            what = f"{name} n={n_blocks} rows={rows} D={D} {kw}"
            assert_bits(self, plen, ref_pl, "permuted lengths " + what)
            # snan positions in wire order: map through fidx
            sn = None
            if snan is not None:
                _, fidx = sarp.row_index(recat32, blk.view(-1, 1), rows)
                sn = snan.view(rows, D)[fidx.cpu()].reshape(-1)
            assert_bits(self, wire, ref_wire, "wire " + what, nan_src=sn)
        rp_pl, rp_wire, _ = sarp.fwd_rowperm(recat32, tok, lengths_d, emb, n_blocks, D, lowmem_encode)
        assert_bits(self, plen, rp_pl, "permuted lengths vs rowperm")
        assert_bits(self, wire, rp_wire, "wire vs rowperm",
                    nan_src=None if snan is None else sn)

    def test_shapes(self):
        Ds = (8, 128, 512)
        for D in Ds:
            rows_target = {8: 5000, 128: 2500, 512: 1200}[D] if CPU else 60_000
            for i, (local_T, W) in enumerate(((1, 1), (3, 8), (11, 32))):
                self._check(local_T, W, 4, rows_target, D, 31 + i, recat_kind="random")
                self._check(local_T, W, 4, rows_target // 3, D, 41 + i)

    def test_non_pow2_D_and_dtypes(self):
        for D in (24, 136):
            self._check(3, 8, 3, 900, D, 51)
        self._check(3, 8, 3, 900, 16, 53, tok_dtype=torch.int32, recat_dtype=torch.int64)
        self._check(2, 16, 1, 500, 32, 55, pattern="alt01")

    def test_specials_exhaustive_fp16(self):
        """F1 clamp step: every fp16 pattern (as an exact fp32) through K1."""
        h = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
        x = h.float().to(DEV).view(-1, 8)  # 8192 rows, D=8, one block
        rows = x.shape[0]
        tok = torch.tensor([[rows]], dtype=torch.int64, device=DEV)
        recat = torch.zeros(1, dtype=torch.int32, device=DEV)
        plen, wire = saf.send_fwd(x, tok.view(-1), recat, tok, 1, 8)
        ref = lowmem_encode(x.view(-1))
        assert_bits(self, wire, ref, "all fp16 patterns",
                    nan_src=snan_mask_f32(x.view(-1).cpu()) if CPU else None)
        self.assertEqual(int(plen.view(-1)[0]), rows * 8)

    def test_specials_list(self):
        x = f32_from_bits(F32_SPECIALS)
        x = torch.cat([x, -x]).repeat(8).to(DEV).view(-1, 8)
        rows = x.shape[0]
        tok = torch.tensor([[rows]], dtype=torch.int64, device=DEV)
        recat = torch.zeros(1, dtype=torch.int32, device=DEV)
        _, wire = saf.send_fwd(x, tok.view(-1), recat, tok, 1, 8)
        for enc in (lowmem_encode, fbgemm_stock_encode):
            assert_bits(self, wire, enc(x.view(-1)), "specials",
                        nan_src=snan_mask_f32(x.view(-1).cpu()) if CPU else None)


class SendBwdTest(unittest.TestCase):
    """K2 == stock decode + permute + div_(W), and == rowperm + div_(W)."""

    def _check(self, local_T, W, B, rows_target, D, seed, world_sizes=(8, 16, 32, None), **kw):
        gen, n_blocks, tok, recat, bwd = case_tensors(local_T, W, B, rows_target, seed, **kw)
        rows = int(tok.sum())
        wire = make_bwd_wire(rows, D, gen).to(DEV)
        lengths_d = tok * D
        blk = sarp.block_rows(tok, n_blocks)
        permuted_lengths_d = torch.index_select(lengths_d.view(n_blocks, -1), 0, recat)
        pblk, _ = sarp.row_index(recat, blk, rows)
        ref0 = sarp.stock_bwd(bwd, permuted_lengths_d, wire, decode)
        rp0 = sarp.bwd_rowperm(bwd, pblk, wire, D, decode)
        nn = None
        if CPU:
            _, bidx = sarp.row_index(bwd, pblk, rows)
            nn = nan_mask_f16(wire).view(rows, D)[bidx.cpu()].reshape(-1)
        assert_bits(self, rp0, ref0, "rowperm vs stock (sanity)", nan_src=nn, nan_mode="any")
        for ws in world_sizes:
            ref = ref0.clone()
            if ws is not None:
                ref.div_(ws)
            for op in ("mul", "div"):
                g = saf.send_bwd(wire, blk.view(-1), recat, bwd, D, ws, scale_op=op)
                assert_bits(self, g, ref, f"grad W={ws} op={op} n={n_blocks} rows={rows} D={D} {kw}",
                            nan_src=nn, nan_mode="any")

    def test_shapes(self):
        for D in (8, 128, 512):
            rows_target = {8: 4000, 128: 2000, 512: 1000}[D] if CPU else 60_000
            for i, (local_T, W) in enumerate(((1, 1), (3, 8), (11, 32))):
                self._check(local_T, W, 4, rows_target, D, 61 + i, recat_kind="random")
                self._check(local_T, W, 4, rows_target // 3, D, 71 + i)

    def test_all_fp16_patterns(self):
        """F4 element op over all 65,536 fp16 patterns, W in {1..32} and division off."""
        h = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
        wire = h.to(DEV).view(-1)
        rows, D = wire.numel() // 8, 8
        blk = torch.tensor([rows], dtype=torch.int64, device=DEV)
        recat = torch.zeros(1, dtype=torch.int32, device=DEV)
        sn = snan_mask_f16(h) if CPU else None
        for ws in (None, 1, 2, 4, 8, 16, 32):
            ref = wire.float()
            if ws is not None:
                ref.div_(ws)
            for op in ("mul", "div"):
                g = saf.send_bwd(wire, blk, recat, recat, D, ws, scale_op=op)
                assert_bits(self, g, ref, f"all fp16 W={ws} op={op}",
                            nan_src=sn if ws is None else None)

    def test_non_pow2_world_rejected(self):
        wire = torch.zeros(16, dtype=torch.float16, device=DEV)
        blk = torch.tensor([2], dtype=torch.int64, device=DEV)
        recat = torch.zeros(1, dtype=torch.int32, device=DEV)
        with self.assertRaises(ValueError):
            saf.send_bwd(wire, blk, recat, recat, 8, 6)


class Pow2DivisionTest(unittest.TestCase):
    """x * (1/W) == x / W bitwise for W = 2**k, incl. results in the fp32 subnormal range."""

    def test_fp16_decoded_inputs(self):
        h = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
        x = h.float().to(DEV)
        for k in range(0, 11):
            W = 1 << k
            a = x / W
            b = x * (1.0 / W)
            c = x.clone().div_(W)
            assert_bits(self, b, a, f"x*(1/W) vs x/W, W={W}")
            assert_bits(self, c, a, f"div_ vs x/W, W={W}")
            fin = torch.isfinite(x) & (x != 0)
            # exactness: the quotient times W gives x back (no rounding happened)
            assert_bits(self, (a[fin] * W), x[fin], f"exact W={W}")

    def test_random_fp32_incl_subnormal_results(self):
        gen = torch.Generator().manual_seed(5)
        bits = torch.randint(-(1 << 31), (1 << 31) - 1, (1 << 18,), generator=gen, dtype=torch.int64)
        x = bits.to(torch.int32).view(torch.float32)
        tiny = torch.exp2(torch.empty(1 << 16).uniform_(-149, -110, generator=gen))  # -> subnormal
        x = torch.cat([x, tiny, -tiny, f32_from_bits(F32_SPECIALS)]).to(DEV)
        for k in (1, 3, 4, 5, 10, 20):
            W = 1 << k
            a, b = x / W, x * (1.0 / W)
            nan = torch.isnan(a)
            assert_bits(self, b[~nan], a[~nan], f"random fp32 W={W}")
            self.assertTrue(bool(torch.equal(torch.isnan(b), nan)))


# --------------------------------------------------------------------------
# Integration: the patched All2All_Seq_Req forward/backward
# --------------------------------------------------------------------------
class _FakeWork:
    def wait(self):
        return True


class _FakeCodec:
    def __init__(self, enc, dec):
        self._comm_precision = "FP16"
        self._enc, self._dec = enc, dec

    def create_context(self):
        return None

    def encode(self, t, ctx=None):
        return self._enc(t)

    def decode(self, t, ctx=None):
        return self._dec(t)

    def calc_quantized_size(self, n, ctx=None):
        return n


class IntegrationTest(unittest.TestCase):
    """_forward/_backward of seq_a2a_rowperm with the fused path, against the stock math."""

    ENV = (
        "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM", "DLRM_HSTU_FP8_SEQ_A2A_FUSED",
        "DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD",
        "DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY",
        "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY", "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK",
        "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV",
    )

    def setUp(self):
        import torch.distributed as dist

        self._saved_env = {k: os.environ.get(k) for k in self.ENV}
        self._dist = dist
        self._saved_dist = {n: getattr(dist, n, None) for n in
                            ("get_world_size", "get_rank", "all_to_all_single")}
        self.world = 8
        self.sent: List[torch.Tensor] = []

        def a2a(output=None, input=None, output_split_sizes=None, input_split_sizes=None,
                group=None, async_op=False):
            self.sent.append(input)
            return _FakeWork()

        dist.get_world_size = lambda pg=None: self.world
        dist.get_rank = lambda pg=None: 0
        dist.all_to_all_single = a2a
        self._saved_codec_check = sarp._fp16_codecs_exact
        self.calls: Dict[str, int] = {"fwd": 0, "bwd": 0}
        self._saved_send = (saf.send_fwd, saf.send_bwd)

        def spy_fwd(*a, **k):
            self.calls["fwd"] += 1
            return self._saved_send[0](*a, **k)

        def spy_bwd(*a, **k):
            self.calls["bwd"] += 1
            return self._saved_send[1](*a, **k)

        saf.send_fwd, saf.send_bwd = spy_fwd, spy_bwd
        self._saved_codecs_ok = sarp._codecs_ok
        self._saved_orig = dict(sarp._orig)
        sarp._orig["forward"] = self._stock_forward_unavailable
        sarp._orig["backward"] = self._stock_forward_unavailable

    def tearDown(self):
        sarp._orig.clear()
        sarp._orig.update(self._saved_orig)
        sarp._codecs_ok = self._saved_codecs_ok
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for n, f in self._saved_dist.items():
            if f is not None:
                setattr(self._dist, n, f)
        sarp._fp16_codecs_exact = self._saved_codec_check
        saf.send_fwd, saf.send_bwd = self._saved_send
        from torchrec.distributed import comm_ops

        comm_ops.GRADIENT_DIVISION = True
        sarp._reset_for_tests()

    @staticmethod
    def _stock_forward_unavailable(*a, **k):
        raise AssertionError("stock fallback was not expected in this test")

    def _codecs(self):
        """Real fbgemm FP16 codecs if available, else fakes with the exact semantics."""
        if _REAL_TORCHREC:
            try:
                from fbgemm_gpu.quantize_comm import QuantizedCommCodec
                from fbgemm_gpu.split_embedding_configs import SparseType
                from torchrec.distributed.types import QuantizedCommCodecs

                return QuantizedCommCodecs(
                    forward=QuantizedCommCodec(SparseType.FP16),
                    backward=QuantizedCommCodec(SparseType.FP16),
                ), True
            except Exception:  # noqa: BLE001
                pass
        # No fbgemm here: the fakes carry the exact FP16 codec semantics, so accept them.
        sarp._fp16_codecs_exact = lambda codecs: None
        sarp._codecs_ok = lambda codecs: True
        return types.SimpleNamespace(forward=_FakeCodec(lowmem_encode, decode),
                                     backward=_FakeCodec(lowmem_encode, decode)), False

    def _run(self, env: Dict[str, str], *, world: int = 8, grad_div: bool = True,
             local_T: int = 3, B: int = 4, rows_target: int = 900, D: int = 64, seed: int = 3):
        from torchrec.distributed import comm_ops

        sarp._reset_for_tests()
        for k in self.ENV:
            os.environ.pop(k, None)
        os.environ.update(env)
        self.world = world
        comm_ops.GRADIENT_DIVISION = grad_div
        codecs, real = self._codecs()
        gen, n_blocks, tok, recat, bwd = case_tensors(local_T, world, B, rows_target, seed)
        rows = int(tok.sum())
        emb = make_emb(rows, D, gen).to(DEV).contiguous()
        if CPU:  # interpreter/numpy NaN semantics differ from torch CPU; keep VERIFY strict
            emb = quiet_snans_f32(emb).contiguous()

        def a2ai():
            return types.SimpleNamespace(
                embedding_dim=D, lengths_after_sparse_data_all2all=tok,
                forward_recat_tensor=recat, backward_recat_tensor=bwd,
                input_splits=[rows // world] * world, output_splits=[rows // world] * world,
                variable_batch_size=False, codecs=codecs,
                permuted_lengths_after_sparse_data_all2all=None)

        info = a2ai()
        ctx = types.SimpleNamespace()
        myreq = types.SimpleNamespace(req=None, tensor=None, dummy_tensor=torch.empty(0),
                                      a2ai=None, qcomm_ctx=None, wait_function=None)
        self.sent.clear()
        sarp._forward(ctx, None, myreq, info, emb)
        wire = self.sent[-1]
        # stock reference (same math VERIFY uses)
        enc = (lambda t: codecs.forward.encode(t, None))
        ref_pl, ref_wire = sarp.stock_fwd(recat, tok * D, emb, n_blocks, enc)
        assert_bits(self, wire, ref_wire, f"integration wire {env}")
        assert_bits(self, info.permuted_lengths_after_sparse_data_all2all, ref_pl,
                    f"integration permuted lengths {env}")
        # backward: hand the module a received fp16 gradient wire
        gwire = make_bwd_wire(rows, D, gen).to(DEV)
        if CPU:
            gwire = torch.where(nan_mask_f16(gwire), torch.zeros_like(gwire), gwire)
        myreq.req, myreq.tensor = _FakeWork(), gwire
        out = sarp._backward(ctx)
        g = out[3]
        ref_g = sarp.stock_bwd(bwd, ref_pl, gwire, lambda t: codecs.backward.decode(t, None))
        if grad_div:
            ref_g.div_(world)
        assert_bits(self, g.reshape(-1), ref_g, f"integration grad {env}")
        self.assertEqual(tuple(g.shape), (rows, D))
        return real

    def test_fused_both(self):
        self._run({"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"})
        self.assertEqual(self.calls, {"fwd": 1, "bwd": 1})

    def test_fused_grad_division_off(self):
        self._run({"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"},
                  grad_div=False)
        self.assertEqual(self.calls, {"fwd": 1, "bwd": 1})

    def test_world_sizes(self):
        for world in (1, 16, 32):
            self.calls.update(fwd=0, bwd=0)
            self._run({"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"},
                      world=world, local_T=2, rows_target=600)
            self.assertEqual(self.calls, {"fwd": 1, "bwd": 1}, f"world={world}")

    def test_sub_switches(self):
        base = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"}
        for extra, want in (
            ({"DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD": "0"}, {"fwd": 0, "bwd": 1}),
            ({"DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD": "0"}, {"fwd": 1, "bwd": 0}),
            ({"DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND": "0"}, {"fwd": 0, "bwd": 0}),
            ({"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "0"}, {"fwd": 0, "bwd": 0}),
        ):
            self.calls.update(fwd=0, bwd=0)
            self._run({**base, **extra})
            self.assertEqual(self.calls, want, str(extra))

    def test_non_pow2_world_falls_back(self):
        self._run({"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"},
                  world=6, local_T=2, rows_target=600)
        self.assertEqual(self.calls, {"fwd": 0, "bwd": 0})

    def test_verify(self):
        env = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1",
               "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY": "2"}
        with self.assertLogs(sarp.logger, level="INFO") as cm:
            self._run(env)
        text = "\n".join(cm.output)
        self.assertIn("armed OK: seq a2a fused send (fwd cast+clamp+gather, bwd gather+decode+div)", text)
        self.assertIn("seq a2a fused verify fwd: bitwise=True", text)
        self.assertIn("seq a2a fused verify bwd: bitwise=True", text)
        self.assertFalse(sarp._state["fused_disabled"])

    def test_send_fwd_exception_falls_back(self):
        """A raising K1 (e.g. a Triton compile failure): rowperm result, fused off, one warning."""
        def boom(*a, **k):
            self.calls["fwd"] += 1
            raise RuntimeError("test: K1 launch failed")

        saf.send_fwd = boom
        env = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"}
        with self.assertLogs(sarp.logger, level="WARNING") as cm:
            self._run(env)  # asserts wire, permuted lengths and grad == stock (raw bits)
        self.assertIn("send fwd (K1) raised RuntimeError", "\n".join(cm.output))
        self.assertTrue(sarp._state["fused_disabled"])
        self.assertEqual(self.calls, {"fwd": 1, "bwd": 0})  # backward took the rowperm path

    def test_send_bwd_exception_falls_back(self):
        def boom(*a, **k):
            self.calls["bwd"] += 1
            raise RuntimeError("test: K2 launch failed")

        saf.send_bwd = boom
        env = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"}
        with self.assertLogs(sarp.logger, level="WARNING") as cm:
            self._run(env)
        self.assertIn("send bwd (K2) raised RuntimeError", "\n".join(cm.output))
        self.assertTrue(sarp._state["fused_disabled"])
        self.assertEqual(self.calls, {"fwd": 1, "bwd": 1})

    def test_static_eligibility_cached(self):
        """The codec check runs once per codecs object, not every forward."""
        env = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"}
        sarp._reset_for_tests()
        for k in self.ENV:
            os.environ.pop(k, None)
        os.environ.update(env)
        codecs, _ = self._codecs()
        inner = sarp._fp16_codecs_exact
        n = {"codec": 0}

        def counted(c):
            n["codec"] += 1
            return inner(c)

        sarp._fp16_codecs_exact = counted
        gen, n_blocks, tok, recat, bwd = case_tensors(3, 8, 4, 600, 21)
        rows = int(tok.sum())
        for i in range(3):
            emb = make_emb(rows, 64, gen).to(DEV).contiguous()
            info = types.SimpleNamespace(
                embedding_dim=64, lengths_after_sparse_data_all2all=tok,
                forward_recat_tensor=recat, backward_recat_tensor=bwd,
                input_splits=[rows // 8] * 8, output_splits=[rows // 8] * 8,
                variable_batch_size=False, codecs=codecs,
                permuted_lengths_after_sparse_data_all2all=None)
            myreq = types.SimpleNamespace(req=None, tensor=None, dummy_tensor=torch.empty(0),
                                          a2ai=None, qcomm_ctx=None, wait_function=None)
            sarp._forward(types.SimpleNamespace(), None, myreq, info, emb)
        self.assertEqual(n["codec"], 1)
        self.assertEqual(self.calls["fwd"], 3)

    def test_codec_mismatch_falls_back(self):
        env = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"}
        real = _REAL_TORCHREC
        if real:
            # a loss-scaled FP16 codec is not exact -> rowperm path
            from fbgemm_gpu.quantize_comm import QuantizedCommCodec
            from fbgemm_gpu.split_embedding_configs import SparseType
            from torchrec.distributed.types import QuantizedCommCodecs

            codecs = QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.FP16, loss_scale=2.0),
                                         backward=QuantizedCommCodec(SparseType.FP16))
            self.assertIsNotNone(sarp._fp16_codecs_exact(codecs))
        orig = self._codecs

        def codecs_rejected():
            c, r = orig()
            sarp._fp16_codecs_exact = lambda codecs: "test: rejected"
            return c, r

        self._codecs = codecs_rejected
        try:
            self._run(env)
        finally:
            self._codecs = orig
        self.assertEqual(self.calls, {"fwd": 0, "bwd": 0})

    @unittest.skipUnless(_REAL_TORCHREC, "needs real fbgemm/torchrec")
    def test_real_codec_checks(self):
        from fbgemm_gpu import quantize_comm as qc
        from fbgemm_gpu.quantize_comm import QuantizedCommCodec
        from fbgemm_gpu.split_embedding_configs import SparseType
        from torchrec.distributed.types import QuantizedCommCodecs

        fp16 = QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.FP16),
                                   backward=QuantizedCommCodec(SparseType.FP16))
        self.assertIsNone(sarp._fp16_codecs_exact(fp16), "stock or lowmem fp16 encode accepted")
        bf16 = QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.BF16),
                                   backward=QuantizedCommCodec(SparseType.FP16))
        self.assertIsNotNone(sarp._fp16_codecs_exact(bf16))
        saved = qc.fp32_to_fp16_with_clamp
        try:
            qc.fp32_to_fp16_with_clamp = lambda t: t.half()  # some other patch
            self.assertIsNotNone(sarp._fp16_codecs_exact(fp16))
        finally:
            qc.fp32_to_fp16_with_clamp = saved
        try:  # the shipped lowmem patch
            from generative_recommenders.dlrm_v4.train import utils as tu
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"train.utils import failed: {type(e).__name__}: {e}")
        from fbgemm_gpu import quantize_utils as qu

        names = ("fp32_to_fp16_with_clamp", "fp32_to_bf16_with_clamp")
        saved_fns = {(m, n): getattr(m, n) for m in (qu, qc) for n in names if hasattr(m, n)}
        saved_flag = getattr(tu, "_FBGEMM_LOWMEM_PATCHED", None)
        try:
            tu._patch_fbgemm_lowmem_clamp_cast(enabled=True)
            self.assertTrue(sarp._is_lowmem_fp16_encode(qc.fp32_to_fp16_with_clamp),
                            "lowmem patch recognised")
            self.assertIsNone(sarp._fp16_codecs_exact(fp16), "lowmem fp16 encode accepted")
            x = make_emb(300, 8, torch.Generator().manual_seed(9)).to(DEV).view(-1)
            assert_bits(self, qc.fp32_to_fp16_with_clamp(x), lowmem_encode(x), "patched encode")
        finally:  # leave fbgemm as this test found it (later tests in the process)
            for (m, n), f in saved_fns.items():
                setattr(m, n, f)
            if saved_flag is not None:
                tu._FBGEMM_LOWMEM_PATCHED = saved_flag


# --------------------------------------------------------------------------
# Receive tier 1: K3' (decode + unbucketize gather) and K4 (scatter + encode)
# --------------------------------------------------------------------------
def unbucketize_perm(U: int, W: int, T: int, gen: torch.Generator) -> torch.Tensor:
    """A row-wise unbucketize permutation with the production structure (int64, CPU).

    fbgemm ``block_bucketize_sparse_features(..., sequence=True)`` keeps the original
    (feature-major) order inside each (bucket, feature) group and lays the groups out
    bucket-major; ``perm[i]`` is the bucketized (= received) position of original row
    i, so the stock unbucketize is ``out = received[perm]``. Buckets are uniform here
    (ids uniform over the table rows).
    """
    w = torch.rand(T, generator=gen) + 0.1
    cnt = torch.floor(w / w.sum() * U).to(torch.int64)
    cnt[0] += U - int(cnt.sum())
    feat = torch.repeat_interleave(torch.arange(T), cnt)
    bucket = torch.randint(0, max(1, W), (U,), generator=gen)
    order = torch.sort(bucket * T + feat, stable=True).indices
    perm = torch.empty(U, dtype=torch.int64)
    perm[order] = torch.arange(U)
    return perm


def rand_f16_bits(n: int, seed: int) -> torch.Tensor:
    """n uniformly random fp16 bit patterns, generated on DEV."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(-32768, 32768, (n,), generator=g, device=DEV,
                         dtype=torch.int32).to(torch.int16).view(torch.float16)


def rand_grad(U: int, D: int, seed: int) -> torch.Tensor:
    """fp32 [U, D] gradient with specials (NaN payloads, +-inf, -0.0, |x| > 65504,
    fp16-subnormal range, random bit patterns), generated on DEV."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = U * D
    x = torch.randn(n, generator=g, device=DEV) * 3.0
    x[::7] *= 3e4
    x[1::11] *= 1e-6
    x[2::13] = -0.0
    n_sp = max(1, n // 50)
    sp = f32_from_bits(F32_SPECIALS).to(DEV)
    x[torch.randint(0, n, (n_sp,), generator=g, device=DEV)] = \
        sp[torch.randint(0, sp.numel(), (n_sp,), generator=g, device=DEV)]
    rb = torch.randint(-(1 << 31), (1 << 31) - 1, (n_sp,), generator=g, device=DEV, dtype=torch.int64)
    x[torch.randint(0, n, (n_sp,), generator=g, device=DEV)] = rb.to(torch.int32).view(torch.float32)
    return x.view(U, D)


def all_fp16_patterns() -> torch.Tensor:
    return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)


class RecvKernelTest(unittest.TestCase):
    """K3' == decode + unbucketize index_select; K4 == _PermuteRows backward + encode."""

    def _perms(self, U: int, W: int, seed: int) -> Dict[str, Optional[torch.Tensor]]:
        gen = torch.Generator().manual_seed(seed)
        return {
            "none": None,
            "identity": torch.arange(U, dtype=torch.int64),
            "reverse": torch.arange(U - 1, -1, -1, dtype=torch.int64),
            "random": torch.randperm(U, generator=gen),
            "unbucketize": unbucketize_perm(U, W, 14, gen),
        }

    def _check_fwd(self, wire: torch.Tensor, perm: Optional[torch.Tensor], D: int, what: str):
        U = wire.numel() // D
        p = None if perm is None else perm.to(DEV)
        out = saf.recv_fwd(wire, p, D)
        dec = decode(wire).view(U, D)
        ref = dec if p is None else torch.index_select(dec, 0, p)
        nn = None
        if CPU:  # interpreter keeps signaling NaNs; CPU torch quiets them (module docstring)
            nn = nan_mask_f16(wire).view(U, D)
            nn = (nn if p is None else nn[p]).reshape(-1)
        assert_bits(self, out, ref, what, nan_src=nn, nan_mode="any")
        if p is not None:  # the shipped _PermuteRows forward
            assert_bits(self, out, ubg._PermuteRows.apply(dec, p), what + " vs _PermuteRows",
                        nan_src=nn, nan_mode="any")

    def _check_bwd(self, grad: torch.Tensor, perm: Optional[torch.Tensor], D: int, what: str):
        U = grad.numel() // D
        p = None if perm is None else perm.to(DEV)
        wire = saf.recv_bwd(grad, p, D)
        sn = None
        if CPU:
            s = snan_mask_f32(grad.reshape(-1)).view(U, D)
            if p is not None:  # wire row p[u] comes from grad row u
                t = torch.empty_like(s)
                t[p] = s
                s = t
            sn = s.reshape(-1)
        for ename, enc in (("lowmem", lowmem_encode), ("fbgemm_stock", fbgemm_stock_encode)):
            if p is None:
                refs = {"encode": enc(grad)}
            else:
                refs = {"gather": enc(ubg.backward_gather(grad, p)),
                        "scatter": enc(ubg.backward_scatter(grad, p))}
            for k, ref in refs.items():
                assert_bits(self, wire, ref, f"{what} {ename} {k}", nan_src=sn)

    def test_fwd_all_fp16_patterns(self):
        """K3' over all 65,536 fp16 patterns (NaN payloads, +-inf, -0.0, subnormals)."""
        wire = all_fp16_patterns().to(DEV)
        for D in (8, 512):
            U = wire.numel() // D
            for name, p in self._perms(U, 8, 3 + D).items():
                self._check_fwd(wire, p, D, f"all fp16 D={D} perm={name}")

    def test_fwd_shapes(self):
        cases = ([(300, 64, 8), (257, 128, 4), (64, 24, 2), (50, 136, 8), (1, 8, 1)] if CPU else
                 [(40_000, 512, 8), (75_132, 512, 8), (108_942, 512, 16), (155_280, 512, 32),
                  (340_000, 512, 32), (11_308, 128, 8), (1, 512, 1), (3, 8, 2), (777, 136, 4)])
        for U, D, W in cases:
            wire = rand_f16_bits(U * D, U + D)
            for name, p in self._perms(U, W, U + 7).items():
                if U > 50_000 and name in ("identity", "reverse"):
                    continue
                self._check_fwd(wire, p, D, f"U={U} D={D} perm={name}")

    def test_bwd_specials_and_fp16_values(self):
        """K4 element op on every fp16 value (as exact fp32), the fp32 specials and random bits."""
        rb = torch.randint(-(1 << 31), (1 << 31) - 1, (1 << 14,),
                           generator=torch.Generator().manual_seed(5),
                           dtype=torch.int64).to(torch.int32).view(torch.float32)
        sp = f32_from_bits(F32_SPECIALS)
        x = torch.cat([all_fp16_patterns().float(), sp, -sp, rb])
        x = torch.cat([x, x[: (-x.numel()) % 8]]).view(-1, 8).to(DEV)
        U = x.shape[0]
        for name, p in self._perms(U, 8, 11).items():
            self._check_bwd(x, p, 8, f"specials perm={name}")

    @unittest.skipIf(CPU, "2**32 patterns: GPU only")
    def test_bwd_exhaustive_fp32(self):
        """K4 (the production kernel, no permutation) vs both fp16 encodes over ALL 2**32 fp32
        patterns, 16 chunks of 2**28 laid out [2**19, 512]; chunk 0 also through a random
        row scatter."""
        base = torch.arange(1 << 28, dtype=torch.int32, device=DEV)
        tot = {"lowmem": 0, "fbgemm_stock": 0}
        for c in range(16):
            off = (c << 28) if c < 8 else ((c - 16) << 28)
            x = (base + off).view(torch.float32).view(-1, 512)
            wire = saf.recv_bwd(x, None, 512)
            for name, enc in (("lowmem", lowmem_encode), ("fbgemm_stock", fbgemm_stock_encode)):
                tot[name] += int((ibits(wire).view(-1) != ibits(enc(x)).view(-1)).sum())
            if c == 0:
                p = torch.randperm(x.shape[0], device=DEV)
                ws = saf.recv_bwd(x, p, 512)
                self.assertEqual(int((ibits(ws)[p] != ibits(wire)).sum()), 0, "row scatter")
            del x, wire
        self.assertEqual(tot, {"lowmem": 0, "fbgemm_stock": 0})

    def test_bwd_shapes(self):
        cases = ([(300, 64, 8), (257, 128, 4), (64, 24, 2), (50, 136, 8), (1, 8, 1)] if CPU else
                 [(40_000, 512, 8), (75_132, 512, 8), (108_942, 512, 16), (155_280, 512, 32),
                  (340_000, 512, 32), (11_308, 128, 8), (1, 512, 1), (3, 8, 2), (777, 136, 4)])
        for U, D, W in cases:
            grad = rand_grad(U, D, U + 2 * D)
            for name, p in self._perms(U, W, U + 9).items():
                if U > 50_000 and name in ("identity", "reverse"):
                    continue
                self._check_bwd(grad, p, D, f"U={U} D={D} perm={name}")

    def test_bwd_differs_from_index_select_backward(self):
        """Why K4 engages only where _unbucketize uses _PermuteRows: the stock
        index_select backward (zeros + index_add_) turns -0.0 into +0.0."""
        U, D = 512, 64
        grad = rand_grad(U, D, 77)
        p = torch.randperm(U, generator=torch.Generator().manual_seed(1)).to(DEV)
        wire = saf.recv_bwd(grad, p, D)
        ref_is = lowmem_encode(torch.zeros_like(grad).index_add_(0, p, grad))
        diff = ibits(wire) != ibits(ref_is)
        src = torch.empty_like(grad)
        src[p] = grad  # the grad value behind every wire element
        negz = ibits(src) == ibits(torch.tensor(-0.0, device=DEV))
        self.assertGreater(int(diff.sum()), 0)
        self.assertTrue(bool((diff & ~(negz | torch.isnan(src))).sum() == 0),
                        "K4 vs index_select backward differ only at -0.0 (and NaN payloads)")
        self.assertTrue(bool(diff[negz].all()), "every -0.0 differs (0x8000 vs 0x0000)")

    def test_rejects_and_empty(self):
        w = torch.zeros(64, dtype=torch.float16, device=DEV)
        with self.assertRaises(ValueError):
            saf.recv_fwd(w.float(), None, 8)  # fp32 wire
        with self.assertRaises(ValueError):
            saf.recv_fwd(w.view(8, 8).t(), None, 8)  # non-contiguous
        with self.assertRaises(ValueError):
            saf.recv_fwd(w, torch.arange(7, device=DEV), 8)  # perm numel != U
        with self.assertRaises(ValueError):
            saf.recv_fwd(w, torch.arange(8, device=DEV).view(2, 4), 8)  # 2-D perm
        g = torch.zeros(8, 8, device=DEV)
        with self.assertRaises(ValueError):
            saf.recv_bwd(g.half(), None, 8)  # fp16 grad
        with self.assertRaises(ValueError):
            saf.recv_bwd(g.t(), torch.arange(8, device=DEV), 8)  # non-contiguous grad
        self.assertEqual(tuple(saf.recv_fwd(w[:0], None, 8).shape), (0, 8))
        self.assertEqual(tuple(saf.recv_bwd(g[:0], None, 8).shape), (0, 8))

    def test_out_of_range_perm_is_masked(self):
        """Never for a checked bijection; K3' writes NaN rows, K4 skips the store."""
        U, D = 64, 16
        wire = rand_f16_bits(U * D, 3)
        p = torch.randperm(U, generator=torch.Generator().manual_seed(2)).to(DEV)
        bad = p.clone()
        bad[5] = U + 3
        bad[9] = -1
        out = saf.recv_fwd(wire, bad, D)
        ok = torch.ones(U, dtype=torch.bool, device=DEV)
        ok[5] = ok[9] = False
        ref = torch.index_select(decode(wire).view(U, D), 0, p)
        nn = None if not CPU else nan_mask_f16(wire).view(U, D)[p][ok].reshape(-1)
        assert_bits(self, out[ok], ref[ok], "in-range rows", nan_src=nn, nan_mode="any")
        self.assertTrue(bool(torch.isnan(out[~ok]).all()))
        grad = rand_grad(U, D, 4)
        wb = saf.recv_bwd(grad, bad, D)  # must not fault; rows bad[5], bad[9] not written
        good = p[ok]
        ref_w = lowmem_encode(grad)
        sn = None if not CPU else snan_mask_f32(grad.reshape(-1)).view(U, D)[ok].reshape(-1)
        assert_bits(self, wb[good], ref_w[ok], "in-range rows", nan_src=sn)

    @unittest.skipUnless(_REAL_FBGEMM and not CPU, "needs fbgemm block_bucketize (GPU)")
    def test_unbucketize_perm_structure_matches_fbgemm(self):
        """The test generator's permutation structure == fbgemm block_bucketize's."""
        T, B, W, rows_per_shard = 5, 64, 8, 1000
        gen = torch.Generator().manual_seed(17)
        lengths = torch.randint(0, 9, (T * B,), generator=gen, dtype=torch.int64)
        n = int(lengths.sum())
        idx = torch.randint(0, rows_per_shard * W, (n,), generator=gen, dtype=torch.int64)
        block_sizes = torch.full((T,), rows_per_shard, dtype=torch.int64)
        out = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.to(DEV), idx.to(DEV), bucketize_pos=False, sequence=True,
            block_sizes=block_sizes.to(DEV), my_size=W)
        perm_fb = out[4].cpu()
        feat = torch.repeat_interleave(
            torch.arange(T), lengths.view(T, B).sum(dim=1))
        bucket = torch.clamp(idx // rows_per_shard, max=W - 1)
        order = torch.sort(bucket * T + feat, stable=True).indices
        mine = torch.empty(n, dtype=torch.int64)
        mine[order] = torch.arange(n)
        self.assertEqual(perm_fb.dtype, torch.int64)
        self.assertTrue(bool(torch.equal(perm_fb, mine)), "generator structure != fbgemm")


# --------------------------------------------------------------------------
# Integration: real torchrec Request / All2All_Seq_Req(_Wait) / SequenceEmbeddingsAwaitable
# --------------------------------------------------------------------------
_SWITCHES = (
    "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM", "DLRM_HSTU_FP8_SEQ_A2A_FUSED",
    "DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD",
    "DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV",
    "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY", "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY",
    "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK", "DLRM_HSTU_FP8_UNBUCKETIZE_GATHER",
    "DLRM_HSTU_FP8_UNBUCKETIZE_VERIFY", "DLRM_HSTU_FP8_UNBUCKETIZE_BWD_MODE",
)
_SHIPPED = {"DLRM_HSTU_FP8_SEQ_A2A_ROWPERM": "1", "DLRM_HSTU_FP8_UNBUCKETIZE_GATHER": "1"}
ALL_OFF = {"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "0"}
COMBOS = {  # (send, recv) -> env on top of the shipped switches
    ("on", "on"): {"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1"},
    ("on", "off"): {"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV": "0"},
    ("off", "on"): {"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND": "0"},
    ("off", "off"): {"DLRM_HSTU_FP8_SEQ_A2A_FUSED": "1", "DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND": "0",
                     "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV": "0"},
}


def _reset_ubg() -> None:
    ubg._state.update(verify_left=None, disabled=False, announced=False, mode=None)


@unittest.skipUnless(_REAL_TORCHREC and _HAS_GPU, "needs a GPU and real fbgemm/torchrec")
class RecvIntegrationTest(unittest.TestCase):
    """Full round trip through the REAL torchrec classes with the patches installed
    (``dist`` mocked: both all-to-alls are identity copies). Compared on raw bits
    against the shipped path (rowperm + unbucketize gather; fused off; stock wait):
    forward output, forward wire, backward wire and the lookup gradient."""

    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in _SWITCHES}
        for k in _SWITCHES:
            os.environ.pop(k, None)
        os.environ.update(_SHIPPED)
        os.environ.update(COMBOS[("on", "on")])
        sarp._reset_for_tests()
        _reset_ubg()
        ubg.install()
        sarp.install()
        assert sarp._installed and sarp._recv_installed and ubg._installed
        from fbgemm_gpu.quantize_comm import QuantizedCommCodec
        from fbgemm_gpu.split_embedding_configs import SparseType
        from torchrec.distributed.types import QuantizedCommCodecs

        cls.codecs = QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.FP16),
                                         backward=QuantizedCommCodec(SparseType.FP16))

    @classmethod
    def tearDownClass(cls):
        cls._cache.clear()
        sarp._uninstall_for_tests()
        ubg._uninstall_for_tests()
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sarp._reset_for_tests()
        _reset_ubg()

    def setUp(self):
        import torch.distributed as dist

        self._dist = dist
        self._saved_dist = {n: getattr(dist, n, None) for n in
                            ("get_world_size", "get_rank", "all_to_all_single")}
        self.world = 8
        self.sent: List[torch.Tensor] = []

        def a2a(output=None, input=None, output_split_sizes=None, input_split_sizes=None,
                group=None, async_op=False):
            self.sent.append(input.detach().clone())
            assert output.numel() == input.numel(), (output.numel(), input.numel())
            output.copy_(input)
            return _FakeWork()

        dist.get_world_size = lambda pg=None: self.world
        dist.get_rank = lambda pg=None: 0
        dist.all_to_all_single = a2a
        self.calls = {"send_fwd": 0, "send_bwd": 0, "recv_fwd": 0, "recv_bwd": 0}
        self._saved_k = {n: getattr(saf, n) for n in self.calls}

        def spy(name):
            inner = self._saved_k[name]

            def f(*a, **k):
                self.calls[name] += 1
                return inner(*a, **k)
            return f

        for n in self.calls:
            setattr(saf, n, spy(n))
        self._saved_codec_check = sarp._fp16_codecs_exact

    def tearDown(self):
        for n, f in self._saved_k.items():
            setattr(saf, n, f)
        for n, f in self._saved_dist.items():
            if f is not None:
                setattr(self._dist, n, f)
        sarp._fp16_codecs_exact = self._saved_codec_check
        if not sarp._recv_installed:
            sarp._install_recv()
        from torchrec.distributed import comm_ops

        comm_ops.GRADIENT_DIVISION = True

    # ---- one forward + backward round trip ----
    def _perm(self, kind: str, U: int, seed: int) -> Optional[torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        if kind == "none":
            return None
        if kind == "unbucketize":
            p = unbucketize_perm(U, self.world, 14, gen)
        elif kind == "random":
            p = torch.randperm(U, generator=gen)
        elif kind == "random_int32":
            p = torch.randperm(U, generator=gen).to(torch.int32)
        elif kind == "dup":  # in range, not a bijection
            p = torch.randperm(U, generator=gen)
            p[3] = p[4]
        else:
            raise ValueError(kind)
        return p.to(DEV)

    _cache: Dict[tuple, tuple] = {}

    def _inputs(self, D: int, local_T: int, B: int, rows_target: int, seed: int, perm: str):
        """Inputs built ONCE per configuration and reused by every round. The generators
        assign specials through random (duplicate) indices, which is not deterministic on
        the GPU (nor in parallel CPU index_put at large sizes), so regenerating them per
        round would compare different inputs."""
        key = (D, local_T, B, rows_target, seed, perm, self.world)
        if key not in self._cache:
            gen, n_blocks, tok, recat, bwd = case_tensors(local_T, self.world, B, rows_target, seed)
            rows = int(tok.sum())
            emb0 = make_emb(rows, D, gen).to(DEV).contiguous()
            G = rand_grad(rows, D, seed + 1)
            p = self._perm(perm, rows, seed + 2)
            self._cache[key] = (tok, recat, bwd, rows, emb0, G, p)
        return self._cache[key]

    @staticmethod
    def _set_env(env: Dict[str, str]) -> None:
        """Fresh switches: the shipped ones plus ``env``; read-once state forgotten."""
        sarp._reset_for_tests()
        _reset_ubg()
        for k in _SWITCHES:
            os.environ.pop(k, None)
        os.environ.update(_SHIPPED)
        os.environ.update(env)

    def _round(self, env: Dict[str, str], *, recv_patch: bool = True, perm: str = "unbucketize",
               grad_mode: bool = True, wait_grad_mode: Optional[bool] = None,
               emb_requires_grad: bool = True, upstream: str = "contig", D: int = 64,
               local_T: int = 3, B: int = 4, rows_target: int = 900, seed: int = 5,
               reset: bool = True, before_wait=None, codecs=None) -> Dict[str, object]:
        from torchrec.distributed import comm_ops, dist_data

        if reset:
            self._set_env(env)
        if recv_patch:
            self.assertTrue(sarp._install_recv())
        else:
            sarp._uninstall_recv_for_tests()
        for n in self.calls:
            self.calls[n] = 0
        tok, recat, bwd, rows, emb0, G, p = self._inputs(D, local_T, B, rows_target, seed, perm)
        emb = emb0.clone().requires_grad_(emb_requires_grad)
        splits = [rows // self.world] * (self.world - 1)
        splits.append(rows - sum(splits))
        a2ai = comm_ops.All2AllSequenceInfo(
            embedding_dim=D, lengths_after_sparse_data_all2all=tok,
            forward_recat_tensor=recat, backward_recat_tensor=bwd,
            input_splits=list(splits), output_splits=list(splits),
            variable_batch_size=False, codecs=codecs or self.codecs)
        self.sent.clear()
        with torch.set_grad_enabled(grad_mode):
            myreq = comm_ops.Request(None, DEV)
            comm_ops.All2All_Seq_Req.apply(None, myreq, a2ai, emb)
            aw = dist_data.SequenceEmbeddingsAwaitable(myreq, p, D)
        h = getattr(myreq, "_seqa2a_recv", None)
        if before_wait is not None:
            before_wait(myreq, aw)
        with torch.set_grad_enabled(grad_mode if wait_grad_mode is None else wait_grad_mode):
            out = aw.wait()
        res: Dict[str, object] = {
            "diverted": h is not None, "consumed": None if h is None else h.consumed,
            "out": ibits(out).clone(), "out_rg": out.requires_grad,
            "fwd_wire": ibits(self.sent[0]).clone(),
        }
        if out.requires_grad:
            if upstream == "contig":
                out.backward(G)
            else:  # the wait output's gradient arrives transposed-strided (TBackward)
                (out.t() * G.t().contiguous()).sum().backward()
            res["bwd_wire"] = ibits(self.sent[1]).clone()
            res["emb_grad"] = None if emb.grad is None else ibits(emb.grad).clone()
        res["calls"] = dict(self.calls)
        return res

    def _same(self, got: Dict[str, object], ref: Dict[str, object], what: str) -> None:
        for k in ("out", "fwd_wire", "bwd_wire", "emb_grad"):
            self.assertEqual(k in got, k in ref, f"{what}: {k} presence")
            if k in ref and ref[k] is not None:
                g, r = got[k], ref[k]
                self.assertEqual(tuple(g.shape), tuple(r.shape), f"{what}: {k} shape")
                nbad = int((g != r).sum())
                self.assertEqual(nbad, 0, f"{what}: {k} {nbad} raw-bit mismatches")
        self.assertEqual(got["out_rg"], ref["out_rg"], what)

    # ---- tests ----
    def test_probe(self):
        self.assertTrue(sarp._recorded_probe_ok())

    def test_switch_matrix(self):
        """send {on,off} x recv {on,off} (and master off with the patch installed) vs all-off."""
        for perm in ("unbucketize", "random", "none"):
            ref = self._round(ALL_OFF, recv_patch=False, perm=perm)
            self.assertEqual(ref["calls"], {"send_fwd": 0, "send_bwd": 0, "recv_fwd": 0, "recv_bwd": 0})
            self.assertFalse(ref["diverted"])
            self._same(self._round(ALL_OFF, recv_patch=False, perm=perm), ref,
                       f"perm={perm} A/A of the harness (all-off twice)")
            for (send, recv), env in COMBOS.items():
                got = self._round(env, perm=perm)
                what = f"perm={perm} send={send} recv={recv}"
                self._same(got, ref, what)
                s, r = int(send == "on"), int(recv == "on")
                self.assertEqual(got["calls"], {"send_fwd": s, "send_bwd": s, "recv_fwd": r,
                                                "recv_bwd": r}, what)
                self.assertEqual(got["diverted"], recv == "on" and perm != "none", what)
                if got["diverted"]:
                    self.assertTrue(got["consumed"], what)
            got = self._round(ALL_OFF, perm=perm)  # master off, wait patch installed
            self._same(got, ref, f"perm={perm} master off, patch installed")
            self.assertEqual(sum(got["calls"].values()), 0)

    def test_shapes_and_modes(self):
        for kw in (dict(D=128), dict(D=512, rows_target=3000), dict(D=8, local_T=1),
                   dict(D=64, local_T=2, B=3, rows_target=40)):
            for perm in ("unbucketize", "none"):
                ref = self._round(ALL_OFF, recv_patch=False, perm=perm, **kw)
                got = self._round(COMBOS[("on", "on")], perm=perm, **kw)
                self._same(got, ref, f"{kw} perm={perm}")
                self.assertEqual(got["calls"]["recv_bwd"], 1)
        scatter = {"DLRM_HSTU_FP8_UNBUCKETIZE_BWD_MODE": "scatter"}
        ref = self._round({**ALL_OFF, **scatter}, recv_patch=False)
        got = self._round({**COMBOS[("on", "on")], **scatter})
        self._same(got, ref, "scatter mode")

    def test_production_shape(self):
        """1n M2c-like RW set: ~75k received rows, D=512, W=8."""
        kw = dict(D=512, local_T=2, B=1024, rows_target=75_000, seed=31)
        ref = self._round(ALL_OFF, recv_patch=False, **kw)
        got = self._round(COMBOS[("on", "on")], **kw)
        self._same(got, ref, "75k x 512")
        self.assertEqual(got["calls"], {"send_fwd": 1, "send_bwd": 1, "recv_fwd": 1, "recv_bwd": 1})

    def test_noncontiguous_upstream_grad(self):
        ref = self._round(ALL_OFF, recv_patch=False, upstream="strided")
        got = self._round(COMBOS[("on", "on")], upstream="strided")
        self._same(got, ref, "strided upstream grad")
        self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (1, 0))

    def test_r3_nowait_not_diverted(self):
        from torchrec.distributed import dist_data
        from torchrec.distributed.types import NoWait

        self._set_env(COMBOS[("on", "on")])
        x = torch.randn(300, 16, device=DEV, requires_grad=True)
        p = torch.randperm(300, device=DEV)
        nw = NoWait(x)
        out = dist_data.SequenceEmbeddingsAwaitable(nw, p, 16).wait()
        self.assertIsNone(getattr(nw, "_seqa2a_recv", None))
        assert_bits(self, out, torch.index_select(x, 0, p), "NoWait: today's callback")
        self.assertEqual(self.calls["recv_fwd"], 0)

    def test_r3_foreign_wait_function_not_diverted(self):
        from torchrec.distributed import comm_ops

        from fbgemm_gpu.quantize_comm import QuantizedCommCodec
        from fbgemm_gpu.split_embedding_configs import SparseType
        from torchrec.distributed.types import QuantizedCommCodecs

        self._set_env(COMBOS[("on", "on")])
        D = 8  # the fused receive path needs D % 8 == 0 (a refusal reason of its own)
        p = torch.randperm(10, device=DEV)
        req = comm_ops.Request(None, DEV)
        req.req = _FakeWork()
        req.tensor = torch.zeros(10 * D, dtype=torch.float16, device=DEV)
        req.a2ai = types.SimpleNamespace(embedding_dim=D, codecs=self.codecs)
        req.wait_function = comm_ops.All2Allv_Wait
        self.assertIsNone(sarp.recv_handoff(req, p, D))
        req.wait_function = comm_ops.All2All_Seq_Req_Wait
        self.assertIsNone(sarp.recv_handoff(req, p, 2 * D), "embedding_dim mismatch")
        # Not an exact fp16 codec pair (or D % 8 != 0): K3'/K4 cannot run, so the
        # permutation is not diverted (the stock _unbucketize callback keeps it).
        req.a2ai = types.SimpleNamespace(
            embedding_dim=D,
            codecs=QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.BF16),
                                       backward=QuantizedCommCodec(SparseType.BF16)))
        self.assertIsNone(sarp.recv_handoff(req, p, D), "bf16 codecs not diverted")
        req.a2ai = types.SimpleNamespace(embedding_dim=D, codecs=None)
        self.assertIsNone(sarp.recv_handoff(req, p, D), "no codecs not diverted")
        req.a2ai = types.SimpleNamespace(embedding_dim=4, codecs=self.codecs)
        self.assertIsNone(sarp.recv_handoff(req, p, 4), "D=4 not diverted")
        req.a2ai = types.SimpleNamespace(embedding_dim=D, codecs=self.codecs)
        h = sarp.recv_handoff(req, p, D)
        self.assertIsNotNone(h)
        self.assertIsNone(sarp.recv_handoff(req, p, D), "second hand-off refused")
        req._seqa2a_recv = None
        req.tensor = None
        self.assertIsNone(sarp.recv_handoff(req, p, D), "no wire not diverted")

    def test_r3_disabled_mid_run(self):
        ref = self._round(ALL_OFF, recv_patch=False)

        def disable(myreq, aw):
            self.assertIsNotNone(getattr(myreq, "_seqa2a_recv", None))
            sarp._state["recv_disabled"] = True

        got = self._round(COMBOS[("on", "on")], before_wait=disable)
        self._same(got, ref, "disabled between hand-off and wait")
        self.assertTrue(got["diverted"] and got["consumed"])
        self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (0, 0))

    def test_r3_disabled_between_fwd_and_bwd(self):
        """Fused forward (K3'), then the path is disabled: the backward must run the
        stock _PermuteRows backward + encode, not K4."""
        from torchrec.distributed import comm_ops

        ref = self._round(ALL_OFF, recv_patch=False)
        cls = comm_ops.All2All_Seq_Req_Wait

        def fwd_then_disable(ctx, pg, myreq, *d):
            out = sarp._wait_forward(ctx, pg, myreq, *d)
            sarp._state["recv_disabled"] = True
            return out

        def swap(myreq, aw):  # after the hand-off (which checks the installed patch)
            self.assertIsNotNone(getattr(myreq, "_seqa2a_recv", None))
            cls.forward = staticmethod(fwd_then_disable)

        try:
            got = self._round(COMBOS[("on", "on")], before_wait=swap)
        finally:
            cls.forward = staticmethod(sarp._wait_forward)
        self._same(got, ref, "disabled between forward and backward")
        self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (1, 0))

    def test_r3_no_grad(self):
        ref = self._round(ALL_OFF, recv_patch=False, grad_mode=False)
        got = self._round(COMBOS[("on", "on")], grad_mode=False)
        self._same(got, ref, "no grad")
        self.assertFalse(got["out_rg"])
        self.assertTrue(got["diverted"] and got["consumed"])
        self.assertEqual(got["calls"]["recv_fwd"], 0)  # index_select branch, like _unbucketize
        self.assertIsNone(ubg._state["verify_left"], "bijectivity budget untouched")

    def test_r3_wait_without_grad_after_recorded_req(self):
        ref = self._round(ALL_OFF, recv_patch=False, wait_grad_mode=False)
        got = self._round(COMBOS[("on", "on")], wait_grad_mode=False)
        self._same(got, ref, "wait under no_grad")
        self.assertEqual(got["calls"]["recv_fwd"], 0)

    def test_r3_inputs_not_requiring_grad(self):
        ref = self._round(ALL_OFF, recv_patch=False, emb_requires_grad=False)
        got = self._round(COMBOS[("on", "on")], emb_requires_grad=False)
        self._same(got, ref, "lookup output without grad")
        self.assertEqual(got["calls"]["recv_fwd"], 0)

    def test_r3_int32_perm_keeps_index_select_backward(self):
        """int32 perm: _unbucketize uses index_select, whose backward maps -0.0 to +0.0."""
        ref = self._round(ALL_OFF, recv_patch=False, perm="random_int32")
        got = self._round(COMBOS[("on", "on")], perm="random_int32")
        self._same(got, ref, "int32 perm")
        self.assertTrue(got["diverted"] and got["consumed"])
        self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (0, 0))

    def test_r3_non_bijective_perm(self):
        ref = self._round(ALL_OFF, recv_patch=False, perm="dup")
        self.assertTrue(ubg._state["disabled"])
        got = self._round(COMBOS[("on", "on")], perm="dup")
        self._same(got, ref, "non-bijective perm")
        self.assertTrue(ubg._state["disabled"])
        self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (0, 0))
        # once disabled, later awaitables keep today's callback (no hand-off)
        got2 = self._round(COMBOS[("on", "on")], perm="random", reset=False)
        self.assertFalse(got2["diverted"])

    def test_r3_unconsumed_raises(self):
        def drop_patch(myreq, aw):
            sarp._uninstall_recv_for_tests()  # the stock wait will not consume the hand-off

        with self.assertRaisesRegex(RuntimeError, "was not applied"):
            self._round(COMBOS[("on", "on")], before_wait=drop_patch)

    def test_verify_ok_and_armed(self):
        env = {**COMBOS[("on", "on")], "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY": "2"}
        ref = self._round(ALL_OFF, recv_patch=False)
        with self.assertLogs(level="INFO") as cm:
            got = self._round(env)
        text = "\n".join(cm.output)
        self._same(got, ref, "verify")
        self.assertIn("seq a2a fused recv verify fwd: bitwise=True", text)
        self.assertIn("seq a2a fused recv verify bwd: bitwise=True", text)
        self.assertIn(sarp._ARMED_RECV_MSG, text)
        self.assertIn("DLRMV4 armed OK: unbucketize gather:", text)
        self.assertEqual(text.count(sarp._ARMED_RECV_MSG), 1)
        # Per-path proof lines: logged once each, after the backward kernels ran.
        self.assertEqual(text.count(sarp._ARMED_SEND_BWD_MSG), 1)
        self.assertEqual(text.count(sarp._ARMED_RECV_BWD_MSG), 1)
        self.assertIn(sarp._ARMED_FUSED_MSG, text)
        self.assertFalse(sarp._state["recv_disabled"])

    def test_verify_mismatch_fwd(self):
        ref = self._round(ALL_OFF, recv_patch=False)
        inner = self._saved_k["recv_fwd"]

        def corrupt(*a, **k):
            out = inner(*a, **k)
            out.view(torch.int32).view(-1)[7] ^= 1
            return out

        env = {**COMBOS[("on", "on")], "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY": "1"}
        saf.recv_fwd = corrupt  # tearDown restores the original
        with self.assertLogs(sarp.logger, level="WARNING") as cm:
            got = self._round(env)
        self._same(got, ref, "verify fwd mismatch -> stock")
        self.assertIn("verify fwd MISMATCH", "\n".join(cm.output))
        self.assertTrue(sarp._state["recv_disabled"])

    def test_verify_mismatch_bwd(self):
        ref = self._round(ALL_OFF, recv_patch=False)
        inner = self._saved_k["recv_bwd"]

        def corrupt(*a, **k):
            w = inner(*a, **k)
            w.view(torch.int16).view(-1)[3] ^= 1
            return w

        env = {**COMBOS[("on", "on")], "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY": "1"}
        saf.recv_bwd = corrupt  # tearDown restores the original
        with self.assertLogs(sarp.logger, level="WARNING") as cm:
            got = self._round(env)
        self._same(got, ref, "verify bwd mismatch -> stock")
        self.assertIn("verify bwd MISMATCH", "\n".join(cm.output))
        self.assertTrue(sarp._state["recv_disabled"])

    def test_recv_kernel_exceptions_fall_back(self):
        ref = self._round(ALL_OFF, recv_patch=False)
        for name in ("recv_fwd", "recv_bwd"):
            def boom(*a, **k):
                raise RuntimeError("test: launch failed")

            with self.assertLogs(sarp.logger, level="WARNING") as cm:
                got = self._round(COMBOS[("on", "on")],
                                  before_wait=lambda r, a, name=name: setattr(saf, name, boom))
            setattr(saf, name, self._saved_k[name])
            self._same(got, ref, f"{name} raises -> stock")
            self.assertIn("raised RuntimeError", "\n".join(cm.output))
            self.assertTrue(sarp._state["recv_disabled"])
            self.assertTrue(got["consumed"])

    def test_send_kernel_exceptions_fall_back(self):
        ref = self._round(ALL_OFF, recv_patch=False)
        for name in ("send_fwd", "send_bwd"):
            def boom(*a, **k):
                raise RuntimeError("test: launch failed")

            setattr(saf, name, boom)
            with self.assertLogs(sarp.logger, level="WARNING") as cm:
                got = self._round(COMBOS[("on", "on")])
            setattr(saf, name, self._saved_k[name])
            self._same(got, ref, f"{name} raises -> rowperm")
            self.assertIn("raised RuntimeError", "\n".join(cm.output))
            self.assertTrue(sarp._state["fused_disabled"])
            self.assertEqual((got["calls"]["recv_fwd"], got["calls"]["recv_bwd"]), (1, 1))

    def test_static_eligibility_cached(self):
        inner = self._saved_codec_check
        n = {"codec": 0}

        def counted(c):
            n["codec"] += 1
            return inner(c)

        sarp._fp16_codecs_exact = counted
        first = self._round(COMBOS[("on", "on")])
        for i in range(3):
            got = self._round(COMBOS[("on", "on")], reset=False, seed=40 + i)
            self.assertEqual(got["calls"], {"send_fwd": 1, "send_bwd": 1, "recv_fwd": 1,
                                            "recv_bwd": 1})
        self.assertEqual(first["calls"]["recv_fwd"], 1)
        self.assertEqual(n["codec"], 2, "one send + one recv check for the codecs object")

    def test_lowmem_codec(self):
        """The production codec: train/utils.py lowmem cast-then-clamp_ patch applied."""
        try:
            from generative_recommenders.dlrm_v4.train import utils as tu
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"train.utils import failed: {type(e).__name__}: {e}")
        from fbgemm_gpu import quantize_comm as qc, quantize_utils as qu

        names = ("fp32_to_fp16_with_clamp", "fp32_to_bf16_with_clamp")
        saved_fns = {(m, n): getattr(m, n) for m in (qu, qc) for n in names if hasattr(m, n)}
        saved_flag = getattr(tu, "_FBGEMM_LOWMEM_PATCHED", None)
        try:
            tu._FBGEMM_LOWMEM_PATCHED = False
            tu._patch_fbgemm_lowmem_clamp_cast(enabled=True)
            self.assertTrue(sarp._is_lowmem_fp16_encode(qc.fp32_to_fp16_with_clamp))
            for perm in ("unbucketize", "none"):
                ref = self._round(ALL_OFF, recv_patch=False, perm=perm)
                got = self._round(COMBOS[("on", "on")], perm=perm)
                self._same(got, ref, f"lowmem codec perm={perm}")
                self.assertEqual(got["calls"], {"send_fwd": 1, "send_bwd": 1, "recv_fwd": 1,
                                                "recv_bwd": 1})
        finally:
            for (m, n), f in saved_fns.items():
                setattr(m, n, f)
            if saved_flag is not None:
                tu._FBGEMM_LOWMEM_PATCHED = saved_flag

    def test_codec_not_exact_keeps_stock(self):
        from fbgemm_gpu.quantize_comm import QuantizedCommCodec
        from fbgemm_gpu.split_embedding_configs import SparseType
        from torchrec.distributed.types import QuantizedCommCodecs

        bf16 = QuantizedCommCodecs(forward=QuantizedCommCodec(SparseType.BF16),
                                   backward=QuantizedCommCodec(SparseType.BF16))
        ref = self._round(ALL_OFF, recv_patch=False, codecs=bf16)
        got = self._round(COMBOS[("on", "on")], codecs=bf16)
        self._same(got, ref, "bf16 codecs")
        self.assertEqual(sum(got["calls"].values()), 0)
        # The hand-off is refused for codecs K3'/K4 cannot serve: the stock
        # _unbucketize callback applies the permutation, as with the patch off.
        self.assertFalse(got["diverted"])


# GPU only unless opted in (see the module docstring): skip every TestCase class.
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase) and _obj is not unittest.TestCase:
        globals()[_name] = unittest.skipUnless(_RUN, _SKIP_WHY)(_obj)
del _name, _obj  # a leftover module-level TestCase binding would be collected twice


if __name__ == "__main__":
    print(f"seq_a2a_fused_test: device={DEV} interpret={saf.interpret_mode()} "
          f"real_fbgemm={_REAL_FBGEMM} real_torchrec={_REAL_TORCHREC} torch={torch.__version__}",
          flush=True)
    unittest.main()
