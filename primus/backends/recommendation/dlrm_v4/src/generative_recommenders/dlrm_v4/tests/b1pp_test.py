"""Bitwise tests for dlrm_v4/b1pp.py (B1'': fused KJT recat, TBE Rkeep rebag, Triton bounds check).

Every comparison is on raw integer bits (int64 / int32 views; fp32 compared as int32 views), never on floats.

GPU (training container, real fbgemm/torchrec):
    HIP_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/recommendation:$PYTHONPATH \
        python3 -B -m pytest -q <tree>/recommendation/generative_recommenders/dlrm_v4/tests/b1pp_test.py
CPU (Triton interpreter): OPT-IN only, because it changes process state (TRITON_INTERPRET, and pure-torch
stand-ins for missing fbgemm ops) that other test files collected in the same pytest process would inherit:
    B1PP_TEST_CPU=1 PYTHONPATH=<tree>/recommendation python3 -B -m pytest -q .../b1pp_test.py
Without a GPU and without the opt-in every test here is SKIPPED (e.g. the CPU-only correctness pytest run),
and importing this file changes nothing. The CPU run checks the index math and the hook control flow against
the pure-torch references below (ported from the comm iter-43 lab: comm43_kernels.recat_ref_cpu /
bc_ref_schedule / bc_ref_valid_vectorized, rebag_lab.rebag_torch / rocm_bwd_pred); the GPU run is the
arbiter against real fbgemm (permute_2D_sparse_data + asynchronous_complete_cumsum, bounds_check_indices v2
WARNING, a real nobag SplitTableBatchedEmbeddingBagsCodegen with fused EXACT_ROWWISE_ADAGRAD) and torchrec.
"""

import logging
import math
import os
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch

_HAS_GPU = torch.cuda.is_available()
_CPU_OPT_IN = (not _HAS_GPU) and os.environ.get("B1PP_TEST_CPU", "") == "1"
_RUN = _HAS_GPU or _CPU_OPT_IN
_SKIP_WHY = "GPU only (set B1PP_TEST_CPU=1 for the Triton-interpreter CPU run)"
if _CPU_OPT_IN:
    os.environ.setdefault("TRITON_INTERPRET", "1")

try:  # real fbgemm ops (GPU container)
    import fbgemm_gpu  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from generative_recommenders.dlrm_v4 import b1pp  # noqa: E402

if _CPU_OPT_IN and type(getattr(b1pp, "_b1pp_rebag_kernel", None)).__name__ != "InterpretedFunction":
    # b1pp was imported earlier in this process without TRITON_INTERPRET: its kernels cannot run on CPU.
    _RUN = False
    _SKIP_WHY = "b1pp was imported before TRITON_INTERPRET was set; run this file on its own"

DEV = torch.device("cuda") if _HAS_GPU else torch.device("cpu")
CPU = not _HAS_GPU
_LOGGER = "generative_recommenders.dlrm_v4.b1pp"
_SWITCH_ENVS = (
    "DLRM_HSTU_FP8_KJT_RECAT_FUSED",
    "DLRM_HSTU_FP8_TBE_REBAG",
    "DLRM_HSTU_FP8_TBE_BOUNDS_TRITON",
    "DLRM_HSTU_FP8_B1PP_VERIFY",
)


def _have_op(name: str) -> bool:
    try:
        return callable(getattr(torch.ops.fbgemm, name))
    except Exception:  # noqa: BLE001
        return False


_REAL_FBGEMM = _have_op("permute_2D_sparse_data") and _have_op("bounds_check_indices")


def _have_torchrec() -> bool:
    try:
        from torchrec.distributed import dist_data  # noqa: F401
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


_REAL_TORCHREC = _have_torchrec()


# --------------------------------------------------------------------------
# Pure-torch references (CPU or GPU)
# --------------------------------------------------------------------------
def recat_ref(recat: torch.Tensor, l2: torch.Tensor, values: torch.Tensor):
    """Block permutation of rows (comm43_kernels.recat_ref_cpu) + the complete cumsum of the result."""
    rs = l2.sum(1).cpu()
    starts = [0]
    for v in rs.tolist():
        starts.append(starts[-1] + int(v))
    perm = recat.long()
    out_l = l2[perm]
    pieces = [values[starts[r]: starts[r] + int(rs[r])] for r in perm.cpu().tolist()]
    out_v = torch.cat(pieces) if pieces else values[:0].clone()
    off = torch.zeros(out_l.numel() + 1, dtype=torch.int64, device=l2.device)
    off[1:] = torch.cumsum(out_l.reshape(-1).to(torch.int64), 0)
    return out_l, out_v, off


BC_SCHEDULES = [(order, snap, last) for order in ("asc", "desc") for snap in (False, True)
                for last in ("fix_first", "stale_read", "fix_last")]


def bc_ref_schedule(rows_per_table, indices, offsets, order="asc", snapshot=False, last="fix_first"):
    """fbgemm v2 WARNING semantics under ONE schedule (comm43_kernels.bc_ref_schedule, verbatim logic)."""
    rows = rows_per_table.cpu().tolist()
    T = len(rows)
    o = offsets.cpu().tolist()
    iv0 = indices.cpu().tolist()
    total_B = len(o) - 1
    N = len(iv0)
    if total_B <= 0 or T == 0:
        return indices.cpu().clone(), offsets.cpu().clone(), None
    B = total_B // T
    iv = list(iv0)
    w = 0
    last_orig = o[total_B]
    fix_needed = last_orig != N
    if fix_needed:
        w += 1
        if last != "fix_last":
            o[total_B] = N
    view = list(o)
    if last in ("stale_read", "fix_last"):
        view[total_B] = last_orig
    bags = range(total_B) if order == "asc" else range(total_B - 1, -1, -1)
    for bt in bags:
        if snapshot:
            s, e = view[bt], view[bt + 1]
        else:
            s, e = o[bt], o[bt + 1]
            if bt == total_B - 1 and last in ("stale_read", "fix_last"):
                e = last_orig
        if s < 0 or s > e or e > N:
            w += 1
            s = max(0, min(s, N))
            e = max(s, min(e, N))
            o[bt] = s
            o[bt + 1] = e
        nr = rows[bt // B]
        src = iv0 if snapshot else iv
        for i in range(s, e):
            x = src[i]
            if x == -1:
                continue
            if x < 0 or x >= nr:
                iv[i] = 0
                w += 1
    if fix_needed and last == "fix_last":
        o[total_B] = N
    return (torch.tensor(iv, dtype=indices.dtype), torch.tensor(o, dtype=offsets.dtype), w)


def bc_ref_valid_vectorized(rows_per_table, indices, offsets):
    """Deterministic case (valid offsets): (indices', offsets', n_bad) (comm43_kernels, verbatim logic)."""
    T = rows_per_table.numel()
    total_B = offsets.numel() - 1
    B = total_B // T
    N = indices.numel()
    off = offsets.clone()
    off[total_B] = N
    ind = indices.clone()
    pos = torch.arange(N, device=indices.device)
    tb = off[torch.arange(1, T + 1, device=off.device) * B]
    t_of = torch.bucketize(pos, tb, right=True)
    inr = (pos >= off[0]) & (pos < N)
    nr = rows_per_table[t_of.clamp(max=T - 1)]
    bad = inr & (ind != -1) & ((ind < 0) | (ind >= nr))
    ind[bad] = 0
    return ind, off, int(bad.sum())


def rocm_bwd_pred_ref(L: int, total_B: int) -> bool:
    """rebag_lab.rocm_bwd_pred: int32 total_L / int64 total_B, truncating division, > 1."""
    tl32 = ((int(L) + 2 ** 31) % 2 ** 32) - 2 ** 31
    tb = int(total_B)
    q = abs(tl32) // abs(tb)
    if (tl32 < 0) != (tb < 0):
        q = -q
    return q > 1


def rebag_ref(off: torch.Tensor, T: int, B: int, Bp: int) -> torch.Tensor:
    """rebag_lab.rebag_torch, per-feature (Rkeep) layout."""
    S = off[::B]
    n = S[1:] - S[:-1]
    j = torch.arange(Bp, device=off.device, dtype=off.dtype)
    ct = (n + Bp - 1) // Bp
    pos = torch.minimum(j[None, :] * ct[:, None], n[:, None])
    out = torch.empty(T * Bp + 1, dtype=off.dtype, device=off.device)
    torch.add(S[:-1, None], pos, out=out[:-1].view(T, Bp))
    out[-1:] = S[-1:]
    return out


def feature_of_pos(off: torch.Tensor, bags_per_feature: int, L: int) -> torch.Tensor:
    pos = torch.arange(L, device=off.device, dtype=torch.int64)
    bag = torch.searchsorted(off, pos, right=True) - 1
    return torch.div(bag, bags_per_feature, rounding_mode="floor")


def _ref_permute_2D_sparse_data(permute, lengths, values, weights=None, permuted_lengths_sum=None):
    out_l, out_v, _ = recat_ref(permute, lengths, values)
    return out_l, out_v, None


def _ref_complete_cumsum(t):
    out = torch.zeros(t.numel() + 1, dtype=t.dtype, device=t.device)
    out[1:] = torch.cumsum(t.reshape(-1), 0)
    return out


def _ref_bounds_check_indices(rows, idx, off, mode, w, weights=None, B_offsets=None, max_B=-1, b_t_map=None,
                              info_B_num_bits=-1, info_B_mask=-1, bounds_check_version=1,
                              prefetch_pipeline=False):
    ri, ro, rw = bc_ref_schedule(rows, idx, off)
    if rw is None:
        return
    idx.copy_(ri.to(idx.device))
    off.copy_(ro.to(off.device))
    if mode == 1:
        w.fill_(rw)


if _CPU_OPT_IN:  # pure-torch stand-ins for the fbgemm ops b1pp's VERIFY paths call (only when missing)
    for _op, _fn in (("permute_2D_sparse_data", _ref_permute_2D_sparse_data),
                     ("asynchronous_complete_cumsum", _ref_complete_cumsum),
                     ("bounds_check_indices", _ref_bounds_check_indices)):
        if not _have_op(_op):
            setattr(torch.ops.fbgemm, _op, _fn)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ibits(t: torch.Tensor) -> torch.Tensor:
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(it)


def assert_bits(tc: unittest.TestCase, got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    tc.assertEqual(got.dtype, ref.dtype, what + ": dtype")
    tc.assertEqual(tuple(got.shape), tuple(ref.shape), what + ": shape")
    g = ibits(got).reshape(-1)
    r = ibits(ref).reshape(-1).to(g.device)
    bad = (g != r).nonzero().flatten()
    if bad.numel():
        i = bad[:8].tolist()
        tc.fail(f"{what}: {bad.numel()} raw-bit mismatches, first at {i}: got {g[bad[:8]].tolist()} "
                f"ref {r[bad[:8]].tolist()}")


def _gen(seed: int, device: torch.device = DEV) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def mk_recat(local_split: int, W: int, device: torch.device = DEV, dtype=torch.int32) -> torch.Tensor:
    """torchrec _get_recat(local_split, W, stagger=1) without variable batch: i + j*local_split."""
    return torch.tensor([i + j * local_split for i in range(local_split) for j in range(W)],
                        dtype=dtype, device=device)


def get_recat(local_split: int, W: int, stride: int) -> torch.Tensor:
    if _REAL_TORCHREC:
        from torchrec.distributed.dist_data import _get_recat

        r = _get_recat(local_split=local_split, num_splits=W, stagger=1, device=DEV,
                       batch_size_per_rank=[stride] * W)
        if r is not None:
            return r
    return mk_recat(local_split, W)


def hstu_lengths_tbe(T, B, N_target, gen, n_long=3, seq_cap=8192, per_rank=1024):
    """comm43_common.hstu_lengths_tbe: key-major [T, B] lengths of RW-bucketized HSTU-like inputs."""
    W = max(1, B // per_rank)
    n_long = min(n_long, T)
    n_short = T - n_long
    mu = max(1.0, (N_target - n_short * per_rank) / (n_long * per_rank))
    U = torch.distributions.Gamma(torch.tensor(2.0), torch.tensor(2.0 / mu)).sample((B,)).to(DEV)
    U = U.clamp(max=seq_cap)
    L = torch.zeros((T, B), dtype=torch.int64, device=DEV)
    for t in range(n_short):
        L[t] = (torch.rand(B, generator=gen, device=DEV) < (1.0 / W)).to(torch.int64)
    for t in range(n_short, T):
        L[t] = torch.poisson(U / W, generator=gen).to(torch.int64)
    return L


def uniform_lengths(T, B, N_target, gen):
    bags = torch.randint(0, T * B, (N_target,), generator=gen, device=DEV)
    return torch.bincount(bags, minlength=T * B).view(T, B).to(torch.int64)


def recat_inputs(keys: int, W: int, stride: int, N_target: int, dist: str, seed: int):
    """comm43_recat.make_inputs: source-major a2a output layout [W*keys, stride] + int64 values."""
    gen = _gen(seed)
    torch.manual_seed(seed)
    B = W * stride
    Lt = hstu_lengths_tbe(keys, B, N_target, gen, per_rank=stride) if dist == "hstu" else \
        uniform_lengths(keys, B, N_target, gen)
    l2 = Lt.view(keys, W, stride).permute(1, 0, 2).reshape(W * keys, stride).contiguous()
    N = int(l2.sum())
    v = torch.randint(-(2 ** 62), 2 ** 62, (N,), generator=gen, device=DEV, dtype=torch.int64)
    return l2, v


def make_bc_case(T: int, B: int, mean_len: int, seed: int, device: torch.device = DEV):
    """Valid offsets (cumsum of random lengths), ids valid for the bag's feature."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, 2 * mean_len + 1, (T * B,), generator=g)
    offsets = torch.zeros(T * B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    N = int(offsets[-1])
    rows = torch.randint(5, 40, (T,), generator=g, dtype=torch.int64)
    t_of = torch.repeat_interleave(torch.arange(T), lengths.view(T, B).sum(1))
    idx = torch.randint(0, 2 ** 62, (N,), generator=g, dtype=torch.int64) % rows[t_of]
    return rows.to(device), idx.to(device), offsets.to(device)


def corrupt_ids(rows, idx, off, T, B, frac, seed):
    """comm43_bc.corrupt_indices: ~frac invalid ids (negative, >= rows, INT64 extremes) + -1 pruned markers."""
    g = torch.Generator().manual_seed(seed)
    N = idx.numel()
    bi = idx.clone()
    tb = off[torch.arange(1, T + 1, device=off.device) * B]
    t_of = torch.bucketize(torch.arange(N, device=off.device), tb, right=True).clamp(max=T - 1)
    k = max(16, int(N * frac))
    k = min(k, max(1, (2 * N) // 3))
    sel = torch.randperm(N, generator=g)[: k + k // 2].to(off.device)
    s_bad, s_prune = sel[:k], sel[k:]
    r = rows[t_of[s_bad]]
    choice = torch.arange(s_bad.numel(), device=off.device) % 10
    v = torch.empty(s_bad.numel(), dtype=torch.int64, device=off.device)
    v[choice == 0] = -2
    v[choice == 1] = -3
    v[choice == 2] = -1000
    v[choice == 3] = -(2 ** 62)
    v[choice == 4] = -(2 ** 63)
    v[choice == 5] = r[choice == 5]
    v[choice == 6] = r[choice == 6] + 1
    v[choice == 7] = r[choice == 7] * 2
    v[choice == 8] = 2 ** 62
    v[choice == 9] = 2 ** 63 - 1
    bi[s_bad] = v
    bi[s_prune] = -1
    return bi


def make_small_bc(T=4, B=64, seed=7, device: torch.device = DEV):
    """comm43_bc.make_small_bc: 1..5 ids per bag, rows chosen so a repaired range crossing a table edge
    zeroes ids valid in their own table (collateral)."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, 6, (T * B,), generator=g)
    off = torch.zeros(T * B + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(lens, 0)
    rows = torch.tensor([1000, 40, 50000, 300, 7000, 90, 123456, 5][:T], dtype=torch.int64)
    t_of = torch.repeat_interleave(torch.arange(T), lens.view(T, B).sum(1))
    idx = torch.randint(0, 2 ** 62, (int(off[-1]),), generator=g, dtype=torch.int64) % rows[t_of]
    return rows.to(device), idx.to(device), off.to(device)


def _find_bag(ol, T, B, tables, need):
    for t in tables:
        lo, hi = t * B + 3, (t + 1) * B - 3
        for k in range(lo + max(0, (hi - lo) // 3), hi):
            if need(ol, k):
                return k
    return None


def malformed_cases(rows, idx, off, T, B) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, str]]:
    """comm43_bc.malformed_cases: name -> (indices, offsets, kind in det | det_tensors | racy)."""
    N = idx.numel()
    TB = T * B
    ol = off.tolist()
    rl = rows.tolist()

    def ln(o, b):
        return o[b + 1] - o[b]

    long_tables = sorted(range(T), key=lambda t: -(ol[(t + 1) * B] - ol[t * B]))
    cases: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}

    def add(name, o2, kind, i2=None):
        cases[name] = (idx.clone() if i2 is None else i2, o2, kind)

    add("MO0_valid", off.clone(), "det")
    o = off.clone()
    o[TB] = N + 7
    add("MO1_last_high", o, "det_tensors")
    if ln(ol, TB - 1) >= 1:
        o = off.clone()
        o[TB] = N - min(2, ln(ol, TB - 1))
        add("MO2_last_low", o, "det")
    k = _find_bag(ol, T, B, long_tables, lambda q, k: ln(q, k) >= 1 and ln(q, k + 1) >= 1)
    if k is not None:
        o = off.clone()
        o[k] = ol[k + 1] + 1
        add("MO3_nonmono_up", o, "det")
        o = off.clone()
        o[k] = ol[k + 1] + 1
        i2 = idx.clone()
        i2[ol[k]] = rl[k // B] + 5
        add("MO3b_nonmono_up_badidx", o, "det", i2)
    k = _find_bag(ol, T, B, long_tables, lambda q, k: ln(q, k - 2) >= 1 and ln(q, k - 1) >= 1)
    if k is not None:
        o = off.clone()
        o[k] = ol[k - 1] - 1
        add("MO4_nonmono_down", o, "det")
    o = off.clone()
    o[0] = -3
    add("MO5_neg_first", o, "det")
    kx = next((t * B for t in range(1, T) if rl[t - 1] < rl[t] and ln(ol, t * B + 1) >= 1), None)
    if kx is None:
        kx = next((t * B for t in range(1, T) if ln(ol, t * B + 1) >= 1), None)
    if kx is not None:
        o = off.clone()
        o[kx] = ol[kx + 1] + 1
        add("MO6_nonmono_cross_table", o, "det")
    k = _find_bag(ol, T, B, long_tables, lambda q, k: ln(q, k - 1) >= 1 and ln(q, k) >= 1)
    if k is not None:
        o = off.clone()
        o[k] = -5
        add("MO7_neg_interior", o, "racy")
    if ol[TB - 1] >= 1:
        o = off.clone()
        o[TB] = ol[TB - 1] - 1
        add("MO8_last_below_prev", o, "racy")
    o = off.clone()
    k1, k2, k3 = TB // 3, TB // 2, (2 * TB) // 3
    o[k1] = N + 1000
    o[k2] = -5
    o[k3] = ol[k3 + 1] + 3
    add("MO9_combo", o, "racy")
    return cases


def run_b1pp_bounds(rows, idx, off, **kw):
    i, o = idx.clone(), off.clone()
    w = torch.full((1,), 12345, dtype=torch.int64, device=idx.device)
    if kw:
        b1pp._bounds_launch(rows, i, o, w, **kw)
    else:
        b1pp.bounds_check_warning(rows, i, o, w)
    return i, o, int(w[0])


def run_fbgemm_bounds(rows, idx, off):
    i, o = idx.clone(), off.clone()
    w = torch.full((1,), 987654321, dtype=torch.int64, device=idx.device)
    torch.ops.fbgemm.bounds_check_indices(rows, i, o, 1, w, None, B_offsets=None, max_B=-1, b_t_map=None,
                                          bounds_check_version=2)
    return i, o, int(w[0])


class _EnvMixin:
    """Save/restore the b1pp switches and its read-once state around every test."""

    def setUp(self):  # noqa: N802
        self._saved_env = {k: os.environ.get(k) for k in _SWITCH_ENVS}
        self._saved_orig = b1pp._orig_forward
        b1pp._reset_for_tests()

    def tearDown(self):  # noqa: N802
        b1pp._uninstall_for_tests()
        b1pp._orig_forward = self._saved_orig
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        b1pp._reset_for_tests()

    def switches(self, recat=False, rebag=False, bounds=False, verify=0):
        for k, on in zip(_SWITCH_ENVS[:3], (recat, rebag, bounds)):
            if on:
                os.environ[k] = "1"
            else:
                os.environ.pop(k, None)
        if verify:
            os.environ[_SWITCH_ENVS[3]] = str(verify)
        else:
            os.environ.pop(_SWITCH_ENVS[3], None)
        b1pp._reset_for_tests()


# --------------------------------------------------------------------------
# (B) fused recat
# --------------------------------------------------------------------------
class RecatTest(_EnvMixin, unittest.TestCase):
    def _check(self, recat, l2, v, what, **kw):
        if kw:
            got = b1pp._recat_launch(recat, l2, v, **kw)
        else:
            got = b1pp.recat_fused(recat, l2, v)
        ref = recat_ref(recat, l2, v)
        for name, g, r in zip(("lengths", "values", "offsets"), got, ref):
            assert_bits(self, g, r, f"{what} {name}")
        if _REAL_FBGEMM and _HAS_GPU:
            fl, fv, fw = torch.ops.fbgemm.permute_2D_sparse_data(recat, l2, v, None, v.numel())
            fo = torch.ops.fbgemm.asynchronous_complete_cumsum(fl.view(-1))
            self.assertIsNone(fw)
            for name, g, r in zip(("lengths", "values", "offsets"), got, (fl, fv, fo)):
                assert_bits(self, g, r, f"{what} {name} vs fbgemm")

    def test_small_shapes(self):
        cases = [(3, 4, 8, "rand"), (11, 2, 16, "rand"), (2, 1, 5, "rand"), (3, 4, 8, "zero_rows"),
                 (3, 4, 8, "all_zero"), (4, 3, 1, "rand"), (5, 4, 40, "rand"), (14, 2, 33, "rand")]
        if not CPU:
            cases += [(5, 4, 1500, "rand"), (14, 16, 1024, "rand"), (14, 32, 1000, "rand")]
        for (K, W, B, mode) in cases:
            R = K * W
            g = torch.Generator().manual_seed(R * 1000 + B)
            if mode == "all_zero":
                l2 = torch.zeros((R, B), dtype=torch.int64)
            else:
                l2 = torch.randint(0, 5, (R, B), generator=g, dtype=torch.int64)
                if mode == "zero_rows":
                    l2[1] = 0
                    l2[R - 1] = 0
            N = int(l2.sum())
            v = torch.randint(-2 ** 62, 2 ** 62, (N,), generator=g, dtype=torch.int64)
            recat = mk_recat(K, W)
            self._check(recat, l2.to(DEV), v.to(DEV), f"K={K} W={W} B={B} {mode} N={N}")
            # multi-chunk loops at small sizes (BLOCK / BLOCK_B < row length / B, BLOCK_R < R: the R-prefix
            # chunk loop that production shapes never enter)
            self._check(recat, l2.to(DEV), v.to(DEV), f"small-blocks K={K} W={W} B={B} {mode}",
                        block=16, block_b=16, block_r=8)
            # int64 recat: not eligible (stock permute_2D_sparse_data reads int32; torchrec builds int32)
            self.assertIn("int32", b1pp.recat_reason(recat.long(), l2.to(DEV), v.to(DEV)))
            with self.assertRaises(ValueError):
                b1pp.recat_fused(recat.long(), l2.to(DEV), v.to(DEV))

    def test_large_R_block_r(self):
        # R = 600 > BLOCK_R 512: the fixed BLOCK_R plus one more 512-row chunk (no new variant; production
        # R <= 448 never enters the loop)
        K, W, B = 15, 40, 3
        g = torch.Generator().manual_seed(5)
        l2 = torch.randint(0, 4, (K * W, B), generator=g, dtype=torch.int64)
        v = torch.randint(-2 ** 62, 2 ** 62, (int(l2.sum()),), generator=g, dtype=torch.int64)
        self._check(mk_recat(K, W), l2.to(DEV), v.to(DEV), "R=600", block=16)

    def test_reasons(self):
        l2 = torch.ones((8, 4), dtype=torch.int64, device=DEV)
        v = torch.zeros(32, dtype=torch.int64, device=DEV)
        rc = mk_recat(2, 4)
        self.assertIsNone(b1pp.recat_reason(rc, l2, v))
        self.assertIn("weights", b1pp.recat_reason(rc, l2, v, torch.ones(32, device=DEV)))
        self.assertIn("int64", b1pp.recat_reason(rc, l2.to(torch.int32), v))
        self.assertIn("int64", b1pp.recat_reason(rc, l2, v.to(torch.int32)))
        self.assertIn("permutation", b1pp.recat_reason(mk_recat(2, 3), l2, v))
        self.assertIn("empty", b1pp.recat_reason(rc, torch.ones((8, 0), dtype=torch.int64, device=DEV),
                                                 v[:0]))
        self.assertIn("contiguous", b1pp.recat_reason(rc, torch.ones((4, 8), dtype=torch.int64,
                                                                     device=DEV).t(), v))
        self.assertIn("dtype", b1pp.recat_reason(rc.to(torch.int16), l2, v))
        self.assertIn("int32", b1pp.recat_reason(rc.long(), l2, v))
        with self.assertRaises(ValueError):
            b1pp.recat_fused(rc, l2.to(torch.int32), v)
        # size cap (int32 index headroom, 2^31 - 2^17): the rule reads b1pp._SIZE_MAX at call time, so a
        # lowered cap exercises it without 2^31-element tensors
        saved = b1pp._SIZE_MAX
        try:
            b1pp._SIZE_MAX = 31  # N = 32 > cap; R*B+1 = 33 > cap
            self.assertEqual(b1pp.recat_reason(rc, l2, v), b1pp._SIZE_REASON)
            b1pp._SIZE_MAX = 32  # N = 32 fits, R*B+1 = 33 does not
            self.assertEqual(b1pp.recat_reason(rc, l2, v), b1pp._SIZE_REASON)
            b1pp._SIZE_MAX = 33
            self.assertIsNone(b1pp.recat_reason(rc, l2, v))
        finally:
            b1pp._SIZE_MAX = saved
        self.assertEqual(b1pp._SIZE_MAX, 2 ** 31 - 2 ** 17)

    @unittest.skipUnless(_REAL_FBGEMM and _HAS_GPU, "needs a GPU and real fbgemm")
    def test_lab_edge_cases_vs_fbgemm(self):
        """The 12 edge cases of probes/iter43/lab/comm43_recat.py plus B < 1024 / B > 1024 and the
        census shapes (2n R=224 B=1024, 4n R=448 B=1024)."""
        KEYS, STRIDE = 11, 1024
        cases = [(f"rand_W{W}", W, KEYS, STRIDE, "uni", 200_000) for W in (1, 2, 8, 16, 32)]
        cases += [("census4n_K14_W32", 32, 14, STRIDE, "hstu", 194_427),
                  ("census2n_K14_W16", 16, 14, STRIDE, "hstu", 140_001),
                  ("census4n_K14_W32_med", 32, 14, STRIDE, "hstu", 152_638),
                  ("census2n_K14_W16_med", 16, 14, STRIDE, "hstu", 102_686),
                  ("hstu_W32_3M", 32, KEYS, STRIDE, "hstu", 3_000_000),
                  ("B5", 8, 14, 5, "uni", 300), ("B1000", 16, 14, 1000, "hstu", 100_000),
                  ("B1500", 16, 14, 1500, "hstu", 150_000), ("B2051", 4, 14, 2051, "uni", 50_000)]
        for (name, W, K, stride, dist, Nt) in cases:
            l2, v = recat_inputs(K, W, stride, Nt, dist, 99 + W + stride)
            self._check(get_recat(K, W, stride), l2, v, name)
        g = _gen(5)
        for kind in ("zero_all", "src3_empty", "one_huge_row"):
            W = 16 if kind == "src3_empty" else 8
            l2 = torch.randint(0, 3, (W * KEYS, STRIDE), generator=g, device=DEV, dtype=torch.int64)
            if kind == "zero_all":
                l2.zero_()
            elif kind == "src3_empty":
                l2[3 * KEYS:4 * KEYS] = 0
                l2[0] = 0
            else:
                l2[5, 7] = 500_000
            v = torch.randint(-(2 ** 62), 2 ** 62, (int(l2.sum()),), generator=g, device=DEV,
                              dtype=torch.int64)
            self._check(get_recat(KEYS, W, STRIDE), l2, v, kind)
        # int32 lengths: not eligible (production is int64); the hook then runs stock
        l2, v = recat_inputs(KEYS, 16, STRIDE, 400_000, "hstu", 7)
        self.assertIsNotNone(b1pp.recat_reason(get_recat(KEYS, 16, STRIDE), l2.to(torch.int32), v))

    @unittest.skipUnless(_HAS_GPU, "GPU only")
    def test_residency_large_R(self):
        # comm43 residency shapes, reduced: R=3584 rows (grid well above residency), N=3M
        l2, v = recat_inputs(14, 256, 1024, 3_000_000, "uni", 3256)
        self._check(get_recat(14, 256, 1024), l2, v, "rows3584")


# --------------------------------------------------------------------------
# (D) Triton bounds check
# --------------------------------------------------------------------------
class BoundsTest(_EnvMixin, unittest.TestCase):
    def _cmp(self, got, ref, what):
        assert_bits(self, got[0], ref[0].to(got[0].device), what + " indices")
        assert_bits(self, got[1], ref[1].to(got[1].device), what + " offsets")
        self.assertEqual(got[2], ref[2], what + " warning")

    def test_fast_path_vs_references(self):
        shapes = [(3, 5, 3, 1), (11, 7, 2, 2), (4, 64, 6, 3), (1, 9, 4, 4), (16, 8, 2, 5), (14, 32, 1, 6)]
        for (T, B, ml, seed) in shapes:
            rows, idx, off = make_bc_case(T, B, ml, seed)
            N = idx.numel()
            vref = bc_ref_valid_vectorized(rows, idx, off)
            self.assertEqual(vref[2], 0)
            self._cmp(run_b1pp_bounds(rows, idx, off), vref, f"valid T={T} B={B}")
            bi = corrupt_ids(rows, idx, off, T, B, 0.1, seed + 1)
            ref = bc_ref_schedule(rows, bi, off)
            self._cmp(bc_ref_valid_vectorized(rows, bi, off), ref, f"refs agree T={T} B={B}")
            self._cmp(run_b1pp_bounds(rows, bi, off), ref, f"bad ids T={T} B={B} N={N}")
            # several programs in the fast path
            self._cmp(run_b1pp_bounds(rows, bi, off, BLOCK=16), ref, f"bad ids BLOCK=16 T={T} B={B}")
            if _REAL_FBGEMM and _HAS_GPU:
                self._cmp(run_b1pp_bounds(rows, bi, off), run_fbgemm_bounds(rows, bi, off),
                          f"vs fbgemm T={T} B={B}")
            # offsets[0] > 0: leading ids belong to no bag and stay unchecked
            if int(off[1]) > 2:
                o2 = off.clone()
                o2[0] = 2
                b2 = bi.clone()
                b2[0] = -77
                got = run_b1pp_bounds(rows, b2, o2)
                self._cmp(got, bc_ref_schedule(rows, b2, o2), f"start0>0 T={T} B={B}")
                self.assertEqual(int(got[0][0]), -77)

    def test_empty_tables_multi_program(self):
        rows, idx, off = make_bc_case(5, 6, 3, 9)
        lens = (off[1:] - off[:-1]).view(5, 6).clone()
        lens[2] = 0
        off = torch.zeros_like(off)
        off[1:] = torch.cumsum(lens.flatten(), 0)
        t_of = torch.repeat_interleave(torch.arange(5, device=DEV), lens.sum(1))
        g = torch.Generator().manual_seed(3)
        idx = (torch.randint(0, 2 ** 62, (int(off[-1]),), generator=g, dtype=torch.int64).to(DEV)
               % rows[t_of])
        idx[::5] = 10 ** 9
        self._cmp(run_b1pp_bounds(rows, idx, off, BLOCK=8), bc_ref_schedule(rows, idx, off), "empty table")

    def test_slow_path_one_program_is_sequential(self):
        """With one program and one bag per chunk the slow path is the asc/seq/fix_first schedule exactly:
        in the interpreter (programs and chunks run in order) for every malformed case, racy ones included.
        On the GPU a store of a shared offset and its load in the next chunk can come from different waves
        of the one program with no barrier between them (fbgemm's own race), so there the det cases must be
        exact, det_tensors exact up to the counter, and racy cases only stay in their outcome class."""
        for (T, B, seed) in [(4, 64, 7), (3, 40, 11)]:
            rows, idx, off = make_small_bc(T=T, B=B, seed=seed)
            N = idx.numel()
            big = 1 << max(12, (N - 1).bit_length())
            for name, (ci, co, kind) in malformed_cases(rows, idx, off, T, B).items():
                got = run_b1pp_bounds(rows, ci, co, BLOCK=big, BB2=1)
                ref = bc_ref_schedule(rows, ci, co, "asc", False, "fix_first")
                what = f"{name} T={T} B={B} one program"
                if CPU or kind == "det":
                    self._cmp(got, ref, what)
                elif kind == "det_tensors":
                    assert_bits(self, got[0], ref[0].to(DEV), what + " indices")
                    assert_bits(self, got[1], ref[1].to(DEV), what + " offsets")
                    self.assertIn(got[2], {bc_ref_schedule(rows, ci, co, *s)[2] for s in BC_SCHEDULES}, what)
                else:
                    self.assertTrue(bool(((got[1] >= 0) & (got[1] <= N)).all()), what + " offsets range")
                    self.assertGreater(got[2], 0, what + " warning")

    def test_malformed_outcome_classes(self):
        """Production constexprs (several programs, 16-bag chunks): det cases exact under all 12 CPU
        schedules; det_tensors tensors exact and the counter in the schedule set; racy cases keep the
        offsets in [0, N] (on the GPU also compared with fbgemm v2's outcome class)."""
        for (T, B, seed) in [(4, 64, 7), (4, 512, 7), (3, 40, 11)]:
            if CPU and B > 64:
                continue
            rows, idx, off = make_small_bc(T=T, B=B, seed=seed)
            N = idx.numel()
            for name, (ci, co, kind) in malformed_cases(rows, idx, off, T, B).items():
                refs = [bc_ref_schedule(rows, ci, co, *s) for s in BC_SCHEDULES]
                full = {(tuple(r[0].tolist()), tuple(r[1].tolist()), r[2]) for r in refs}
                got = run_b1pp_bounds(rows, ci, co)
                key = (tuple(got[0].tolist()), tuple(got[1].tolist()), got[2])
                if kind == "det":
                    self.assertEqual(len(full), 1, f"{name}: det label")
                    self.assertIn(key, full, f"{name} T={T} B={B}: det exact")
                elif kind == "det_tensors":
                    self.assertIn(key[:2], {k[:2] for k in full}, f"{name}: tensors")
                    self.assertIn(key[2], {k[2] for k in full}, f"{name}: warning in schedule set")
                else:
                    self.assertTrue(bool(((got[1] >= 0) & (got[1] <= N)).all()), f"{name}: offsets range")
                    self.assertGreater(got[2], 0, f"{name}: warning")
                if _REAL_FBGEMM and _HAS_GPU:
                    fb = run_fbgemm_bounds(rows, ci, co)
                    if kind == "det":
                        self._cmp(got, fb, f"{name} vs fbgemm")
                    elif kind == "det_tensors":
                        assert_bits(self, got[0], fb[0], f"{name} indices vs fbgemm")
                        assert_bits(self, got[1], fb[1], f"{name} offsets vs fbgemm")
                        self.assertIn(fb[2], {k[2] for k in full})
                    else:
                        self.assertTrue(bool(((fb[1] >= 0) & (fb[1] <= N)).all()))
                        self.assertGreater(fb[2], 0)

    def test_noop_and_errors(self):
        rows = torch.tensor([10, 20], dtype=torch.int64, device=DEV)
        w = torch.full((1,), 77, dtype=torch.int64, device=DEV)
        idx = torch.zeros(0, dtype=torch.int64, device=DEV)
        b1pp.bounds_check_warning(rows, idx, torch.zeros(1, dtype=torch.int64, device=DEV), w)
        self.assertEqual(int(w[0]), 77)  # total_B == 0: no warning.zero_() (as fbgemm)
        with self.assertRaises(ValueError):  # offsets size != B*T + 1
            b1pp.bounds_check_warning(rows, idx, torch.zeros(4, dtype=torch.int64, device=DEV), w)
        rows17 = torch.full((17,), 5, dtype=torch.int64, device=DEV)
        with self.assertRaises(ValueError):  # T > MAX_T
            b1pp.bounds_check_warning(rows17, idx, torch.zeros(18, dtype=torch.int64, device=DEV), w)
        with self.assertRaises(ValueError):  # int32 indices
            b1pp.bounds_check_warning(rows, idx.to(torch.int32), torch.zeros(3, dtype=torch.int64,
                                                                             device=DEV), w)

    def test_misaligned_indices_view(self):
        # A KJT split child's values are a view; its 16-byte alignment changes from step to step.
        rows, idx, off = make_bc_case(4, 16, 3, 21)
        bi = corrupt_ids(rows, idx, off, 4, 16, 0.2, 22)
        buf = torch.empty(bi.numel() + 1, dtype=torch.int64, device=DEV)
        buf[1:] = bi
        view = buf[1:]
        self.assertEqual(view.storage_offset(), 1)
        o = off.clone()
        w = torch.zeros(1, dtype=torch.int64, device=DEV)
        b1pp.bounds_check_warning(rows, view, o, w)
        ref = bc_ref_schedule(rows, bi, off)
        self._cmp((view.clone(), o, int(w[0])), ref, "misaligned view")

    def test_bad_count_chunk_loop(self):
        """BB=4 bags per validate tile and BLOCK_F=2 tiles per chunk: the check kernel's bad-count needs the
        chunk loop (n_bad > BLOCK_F; production n_bad <= 448 < 512 never enters it). Valid offsets must take
        the fast path; a bad bag that only a LATER chunk sees must still switch to the slow path."""
        T, B = 4, 16
        rows, idx, off = make_small_bc(T=T, B=B, seed=5)
        N = idx.numel()
        n_bad = -(-(T * B) // 4)
        self.assertGreater(n_bad, 2 * 2)  # at least three chunks of BLOCK_F=2 tiles
        bi = corrupt_ids(rows, idx, off, T, B, 0.2, 6)
        self._cmp(run_b1pp_bounds(rows, bi, off, BB=4, BLOCK_F=2), bc_ref_valid_vectorized(rows, bi, off),
                  "valid offsets, chunk loop (fast path)")
        # MO3 pattern ("non-monotone up", deterministic with valid ids) at bag j = T*B - 5: offsets[j] =
        # offsets[j+1] + 1 -> bag j is bad (s > e), bag j-1 grows by one id. Its tile j // 4 is beyond the
        # first BLOCK_F=2 tiles, so only the chunk loop sees the bad count.
        ol = off.tolist()
        j = T * B - 5
        self.assertTrue(ol[j + 1] - ol[j] >= 1 and ol[j + 2] - ol[j + 1] >= 1)  # as comm43_bc MO3
        self.assertGreaterEqual(j // 4, 2, "bad bag in a tile only the chunk loop reads")
        o2 = off.clone()
        o2[j] = ol[j + 1] + 1
        refs = {(tuple(r[0].tolist()), tuple(r[1].tolist()), r[2])
                for r in (bc_ref_schedule(rows, idx, o2, *s) for s in BC_SCHEDULES)}
        self.assertEqual(len(refs), 1, "deterministic under all 12 schedules")
        ref = bc_ref_schedule(rows, idx, o2, "asc", False, "fix_first")
        self.assertGreater(ref[2], 0)
        big = 1 << max(12, (N - 1).bit_length())
        self._cmp(run_b1pp_bounds(rows, idx, o2, BB=4, BLOCK_F=2, BLOCK=big, BB2=1), ref,
                  "late bad bag, chunk loop, one program (slow path)")
        self._cmp(run_b1pp_bounds(rows, idx, o2, BB=4, BLOCK_F=2), ref,
                  "late bad bag, chunk loop, production BLOCK/BB2 (slow path)")
        self._cmp(run_b1pp_bounds(rows, idx, o2), ref, "same input, production constexprs")

    def test_size_cap(self):
        rows, idx, off = make_bc_case(3, 5, 3, 1)
        w = torch.zeros(1, dtype=torch.int64, device=DEV)
        saved = b1pp._SIZE_MAX
        try:
            b1pp._SIZE_MAX = idx.numel() - 1
            with self.assertRaisesRegex(ValueError, "2\\^31 - 2\\^17"):
                b1pp.bounds_check_warning(rows, idx.clone(), off.clone(), w)
        finally:
            b1pp._SIZE_MAX = saved

    @unittest.skipUnless(_REAL_FBGEMM and _HAS_GPU, "needs a GPU and real fbgemm")
    def test_production_shapes_vs_fbgemm(self):
        """Census-like 2n/4n (T=14, B=16384/32768) and 1n (T=2, B=8192) inputs: valid, and with ~0.1%
        invalid ids (S1). Indices, offsets and the warning count bitwise vs fbgemm v2 WARNING."""
        for (T, B, Nt, seed) in [(14, 16384, 102_686, 4301), (14, 32768, 194_427, 4302),
                                 (2, 8192, 182_324, 4303), (14, 16384, 1_200_000, 4304)]:
            gen = _gen(seed)
            torch.manual_seed(seed)
            L = hstu_lengths_tbe(T, B, Nt, gen)
            off = torch.zeros(T * B + 1, dtype=torch.int64, device=DEV)
            off[1:] = torch.cumsum(L.flatten(), 0)
            N = int(off[-1])
            rows = torch.randint(10 ** 6, 5 * 10 ** 7, (T,), generator=gen, device=DEV, dtype=torch.int64)
            t_of = torch.repeat_interleave(torch.arange(T, device=DEV), L.sum(1))
            idx = torch.randint(0, 2 ** 62, (N,), generator=gen, device=DEV, dtype=torch.int64) % rows[t_of]
            got = run_b1pp_bounds(rows, idx, off)
            self.assertEqual(got[2], 0)
            self._cmp(got, run_fbgemm_bounds(rows, idx, off), f"valid T={T} B={B} N={N}")
            bi = corrupt_ids(rows, idx, off, T, B, 0.001, seed + 1)
            self._cmp(run_b1pp_bounds(rows, bi, off), run_fbgemm_bounds(rows, bi, off),
                      f"S1 T={T} B={B} N={N}")


# --------------------------------------------------------------------------
# (C) Rkeep rebag
# --------------------------------------------------------------------------
def dedup_like_offsets(T: int, B: int, n_t: List[int], device: torch.device = DEV, seed: int = 0):
    """Each feature's ids sit in one contiguous run of non-empty bags (jagged_unique_indices spreads a
    source's unique ids evenly over its bags), 1-3 ids per bag; features with n_t = 0 are empty."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.zeros(T, B, dtype=torch.int64)
    for t, n in enumerate(n_t):
        left, b = n, int(torch.randint(0, max(1, B // 4), (1,), generator=g))
        while left > 0:
            k = min(left, int(torch.randint(1, 4, (1,), generator=g)))
            lens[t, b % B] += k
            left -= k
            b += 1
    off = torch.zeros(T * B + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(lens.flatten(), 0)
    return off.to(device)


class RebagTest(_EnvMixin, unittest.TestCase):
    def test_plan_rules(self):
        self.assertEqual(b1pp.rebag_plan(0, 3, 4), (None, "L <= 0"))
        self.assertEqual(b1pp.rebag_plan(5, 0, 4)[0], None)
        self.assertEqual(b1pp.rebag_plan(5, 3, 0)[0], None)
        self.assertEqual(b1pp.rebag_plan(2 ** 31, 14, 2 ** 20)[1], "L >= 2^31")
        self.assertIn("predicate", b1pp.rebag_plan(24, 3, 4)[1])  # 24 // 12 = 2 > 1
        self.assertIn("no gain", b1pp.rebag_plan(23, 3, 4)[1])  # Bp = ceil(12/3) = 4 = B
        self.assertEqual(b1pp.rebag_plan(1, 3, 4), (1, None))
        # production census shapes (rebag_lab full run: Rkeep total_B 88214 at 4n step0, 51142 at 2n)
        self.assertEqual(b1pp.rebag_plan(176419, 14, 32768), (6301, None))
        self.assertEqual(b1pp.rebag_plan(102268, 14, 16384), (3653, None))
        self.assertIsNone(b1pp.rebag_plan(297833, 2, 8192)[0])  # 1n owner: predicate true -> stock
        g = torch.Generator().manual_seed(11)
        for _ in range(3000):
            T = int(torch.randint(1, 17, (1,), generator=g))
            B = int(torch.randint(1, 40000, (1,), generator=g))
            L = int(torch.randint(0, 2 * T * B + 5, (1,), generator=g))
            Bp, why = b1pp.rebag_plan(L, T, B)
            self.assertEqual(b1pp.rocm_bwd_pred(L, T * B), rocm_bwd_pred_ref(L, T * B))
            if Bp is None:
                self.assertTrue(L <= 0 or rocm_bwd_pred_ref(L, T * B) or
                                max(1, -(-(L // 2 + 1) // T)) >= B, (L, T, B, why))
                continue
            self.assertFalse(rocm_bwd_pred_ref(L, T * B))
            self.assertFalse(rocm_bwd_pred_ref(L, T * Bp), "kept predicate")
            self.assertLess(Bp, B)
            self.assertTrue(Bp == 1 or T * (Bp - 1) < L // 2 + 1, "fewest bags")

    def _check_layout(self, off, T, B, L, what):
        new = b1pp.rebag_rkeep(off, T, B, L)
        Bp, why = b1pp.rebag_plan(L, T, B)
        if Bp is None:
            self.assertIsNone(new, what)
            return None
        ref = rebag_ref(off, T, B, Bp)
        assert_bits(self, new, ref, what)
        self.assertTrue(bool((new[1:] >= new[:-1]).all()), what + " monotone")
        assert_bits(self, new[::Bp], off[::B], what + " feature starts")
        self.assertEqual(int(new[-1]), L, what + " last")
        if L:
            assert_bits(self, feature_of_pos(new, Bp, L), feature_of_pos(off, B, L), what + " feature of ids")
        self.assertEqual(rocm_bwd_pred_ref(L, T * Bp), rocm_bwd_pred_ref(L, T * B), what + " predicate")
        return new

    def test_layouts(self):
        cases = [(3, 64, [40, 0, 30]), (3, 64, [0, 70, 0]), (14, 64, [60, 0, 40, 0, 30, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                 (2, 50, [1, 0]), (1, 8, [3]), (4, 16, [0, 0, 0, 1]), (3, 64, [1, 1, 1])]
        if not CPU:
            cases += [(14, 32768, [0, 80336, 0, 34720, 0, 53378, 1029, 974, 937, 1024, 980, 1008, 1022, 1011]),
                      (14, 16384, [0, 44940, 0, 19437, 0, 29649, 1006, 1019, 1021, 1015, 1068, 1012, 1035, 1066])]
        for i, (T, B, n_t) in enumerate(cases):
            off = dedup_like_offsets(T, B, n_t, seed=i)
            L = int(off[-1])
            self.assertEqual(L, sum(n_t))
            self._check_layout(off, T, B, L, f"T={T} B={B} n_t={n_t}")
        # L >= 2*T*B (predicate true) -> None; L == 0 -> None
        off = dedup_like_offsets(2, 4, [9, 8])
        self.assertIsNone(b1pp.rebag_rkeep(off, 2, 4, 17))
        self.assertIsNone(b1pp.rebag_rkeep(torch.zeros(9, dtype=torch.int64, device=DEV), 2, 4, 0))
        # small BLOCK: several programs
        off = dedup_like_offsets(5, 300, [100, 0, 250, 3, 7], seed=3)
        L = int(off[-1])
        Bp, _ = b1pp.rebag_plan(L, 5, 300)
        assert_bits(self, b1pp._rebag_launch(off, 5, 300, Bp, block=16), rebag_ref(off, 5, 300, Bp), "BLOCK=16")


# --------------------------------------------------------------------------
# (C)+(D) the patched forward: control flow with a stand-in TBE (CPU and GPU)
# --------------------------------------------------------------------------
class _FakeEmb:
    """Stand-in for the fbgemm TBE: records each call; runs the WARNING check (fbgemm op or the reference)
    when its bounds_check_mode_int says so; returns a copy of the ids."""

    def __init__(self, rows_per_feature: List[int], raise_on_call: bool = False):
        self.rows_per_table = torch.tensor(rows_per_feature, dtype=torch.int64, device=DEV)
        self.bounds_check_warning = torch.zeros(1, dtype=torch.int64, device=DEV)
        self.bounds_check_mode_int = 1
        self.bounds_check_version = 2
        self.feature_table_map = list(range(len(rows_per_feature)))
        self.info_B_num_bits, self.info_B_mask = 26, (1 << 26) - 1
        self.calls: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.raise_on_call = raise_on_call

    def __call__(self, indices, offsets):
        self.calls.append((indices, offsets.clone(), self.bounds_check_mode_int))
        if self.raise_on_call:
            raise RuntimeError("emb failure (test)")
        if self.bounds_check_mode_int == 1:
            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table, indices, offsets, 1, self.bounds_check_warning, None, B_offsets=None,
                max_B=-1, b_t_map=None, info_B_num_bits=self.info_B_num_bits, info_B_mask=self.info_B_mask,
                bounds_check_version=2, prefetch_pipeline=False)
        return indices.clone()


class _FakeKJT:
    def __init__(self, values, offsets, stride, weights=None, vbe=False):
        self._values, self._offsets, self._stride = values, offsets, stride
        self._weights, self._vbe = weights, vbe

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def stride(self):
        return self._stride

    def weights_or_none(self):
        return self._weights

    def variable_stride_per_key(self):
        return self._vbe


class _Duck(torch.nn.Module):
    """Duck-typed torchrec BatchedFusedEmbedding (the attributes BaseBatchedEmbedding.forward and
    b1pp's forward read)."""

    def __init__(self, emb: Any):
        super().__init__()
        self._emb_module = emb
        self._raw_id_tracker_wrapper = None
        self._embedding_table_index_type = torch.int64
        self._embedding_table_offset_type = torch.int64

    @property
    def emb_module(self):
        return self._emb_module

    def _get_hash_zch_identities_and_metadata(self, features):
        return None


def _stock_forward(self, features):
    """BaseBatchedEmbedding.forward without hash-ZCH args (torchrec batched_embedding_kernel.py)."""
    return self.emb_module(indices=features.values().to(self._embedding_table_index_type),
                           offsets=features.offsets().to(self._embedding_table_offset_type))


class TbeHookControlFlowTest(_EnvMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        b1pp._orig_forward = _stock_forward
        self._saved_static = b1pp._tbe_static
        b1pp._tbe_static = lambda mod: (None, len(mod._emb_module.feature_table_map))

    def tearDown(self):
        b1pp._tbe_static = self._saved_static
        super().tearDown()

    def _inputs(self, seed=0, T=3, B=64, n_t=(40, 0, 30), bad=True):
        off = dedup_like_offsets(T, B, list(n_t), seed=seed)
        L = int(off[-1])
        rows = [50, 60, 70][:T] + [80] * max(0, T - 3)
        g = torch.Generator().manual_seed(seed + 100)
        fo = feature_of_pos(off, B, L).cpu()
        ids = (torch.randint(0, 2 ** 40, (L,), generator=g) % torch.tensor(rows)[fo]).to(DEV)
        if bad and L > 4:
            ids[1] = rows[int(fo[1])] + 3
            ids[3] = -5
            ids[L - 1] = -1  # pruned marker: kept as is
        return rows, ids, off, L

    def _run(self, rows, ids, off, B, **kjt_kw):
        emb = _FakeEmb(rows)
        duck = _Duck(emb)
        kjt = _FakeKJT(ids.clone(), off.clone(), B, **kjt_kw)
        out = b1pp._b1pp_tbe_forward(duck, kjt)
        return emb, kjt, out

    def test_default_off_is_stock(self):
        self.switches()
        rows, ids, off, L = self._inputs()
        emb, kjt, out = self._run(rows, ids, off, 64)
        self.assertEqual(len(emb.calls), 1)
        idx, o, mode = emb.calls[0]
        self.assertIs(idx, kjt.values())
        assert_bits(self, o, off, "stock offsets")
        self.assertEqual(mode, 1)
        self.assertEqual(b1pp._state["engaged"]["tbe"], 0)
        ref = bc_ref_schedule(torch.tensor(rows), ids, off)
        assert_bits(self, kjt.values(), ref[0].to(DEV), "stock repair")
        self.assertEqual(int(emb.bounds_check_warning[0]), ref[2])

    def test_bounds_on(self):
        self.switches(bounds=True)
        rows, ids, off, L = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            emb, kjt, out = self._run(rows, ids, off, 64)
        self.assertTrue(any("armed OK: b1pp tbe bounds triton (T=3 B=64 N=%d mode=WARNING)" % L in m
                            for m in cm.output), cm.output)
        idx, o, mode = emb.calls[0]
        self.assertEqual(mode, 3, "fbgemm ran with bounds_check_mode NONE")
        self.assertEqual(emb.bounds_check_mode_int, 1, "mode restored")
        assert_bits(self, o, off, "original offsets passed")
        ref = bc_ref_schedule(torch.tensor(rows), ids, off)
        assert_bits(self, kjt.values(), ref[0].to(DEV), "Triton repair in the KJT values")
        self.assertEqual(int(emb.bounds_check_warning[0]), ref[2])
        self.assertEqual(b1pp._state["engaged"]["tbe"], 1)

    def test_rebag_on(self):
        self.switches(rebag=True)
        rows, ids, off, L = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            emb, kjt, out = self._run(rows, ids, off, 64)
        Bp, _ = b1pp.rebag_plan(L, 3, 64)
        self.assertTrue(any("armed OK: b1pp tbe rebag Rkeep (T=3 B=64 Bp=%d L=%d)" % (Bp, L) in m
                            for m in cm.output), cm.output)
        idx, o, mode = emb.calls[0]
        self.assertEqual(mode, 1, "fbgemm keeps its own bounds check")
        assert_bits(self, o, rebag_ref(off, 3, 64, Bp), "rebagged offsets passed")
        assert_bits(self, kjt.offsets(), off, "KJT offsets untouched")

    def test_rebag_and_bounds_on_with_verify(self):
        self.switches(rebag=True, bounds=True, verify=2)
        rows, ids, off, L = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            for _ in range(3):
                emb, kjt, out = self._run(rows, ids, off, 64)
        msgs = "\n".join(cm.output)
        self.assertEqual(msgs.count("VERIFY bounds OK"), 2, msgs)
        self.assertEqual(msgs.count("VERIFY rebag OK"), 2, msgs)
        idx, o, mode = emb.calls[0]
        self.assertEqual(mode, 3)
        Bp, _ = b1pp.rebag_plan(L, 3, 64)
        assert_bits(self, o, rebag_ref(off, 3, 64, Bp), "rebag")
        self.assertEqual(b1pp._state["engaged"]["tbe"], 3)

    def test_fallbacks(self):
        self.switches(rebag=True, bounds=True)
        rows, ids, off, L = self._inputs()
        for kw, why in (({"weights": torch.ones(L, device=DEV)}, "per-sample weights"),
                        ({"vbe": True}, "variable stride per key")):
            with self.assertLogs(_LOGGER, level="INFO") as cm:
                emb, kjt, out = self._run(rows, ids, off, 64, **kw)
            self.assertEqual(emb.calls[0][2], 1)
            assert_bits(self, emb.calls[0][1], off, "stock offsets")
            self.assertTrue(any("FALLBACK: " + why in m for m in cm.output), cm.output)
        # raw-id tracker (hash-ZCH)
        emb = _FakeEmb(rows)
        duck = _Duck(emb)
        duck._raw_id_tracker_wrapper = object()
        b1pp._b1pp_tbe_forward(duck, _FakeKJT(ids.clone(), off.clone(), 64))
        self.assertEqual(emb.calls[0][2], 1)
        # int32 offsets
        emb, kjt, out = self._run(rows, ids, off.to(torch.int32), 64)
        self.assertEqual(emb.calls[0][2], 1)
        # bounds mode IGNORE: D falls back, C still runs
        emb = _FakeEmb(rows)
        emb.bounds_check_mode_int = 2
        b1pp._b1pp_tbe_forward(_Duck(emb), _FakeKJT(ids.clone(), off.clone(), 64))
        self.assertEqual(emb.calls[0][2], 2)
        Bp, _ = b1pp.rebag_plan(L, 3, 64)
        self.assertEqual(emb.calls[0][1].numel(), 3 * Bp + 1)
        # predicate true: rebag stock, bounds still Triton
        T2, B2 = 2, 4
        off2 = dedup_like_offsets(T2, B2, [9, 8])
        ids2 = torch.zeros(17, dtype=torch.int64, device=DEV)
        emb = _FakeEmb([50, 60])
        b1pp._b1pp_tbe_forward(_Duck(emb), _FakeKJT(ids2, off2.clone(), B2))
        assert_bits(self, emb.calls[0][1], off2, "predicate true: original offsets")
        self.assertEqual(emb.calls[0][2], 3)

    def test_mode_restored_when_emb_raises(self):
        self.switches(bounds=True)
        rows, ids, off, L = self._inputs()
        emb = _FakeEmb(rows, raise_on_call=True)
        with self.assertRaises(RuntimeError):
            b1pp._b1pp_tbe_forward(_Duck(emb), _FakeKJT(ids.clone(), off.clone(), 64))
        self.assertEqual(emb.calls[0][2], 3)
        self.assertEqual(emb.bounds_check_mode_int, 1)

    def test_kernel_exception_disables_part(self):
        self.switches(rebag=True, bounds=True)
        rows, ids, off, L = self._inputs()
        saved = b1pp._bounds_launch

        def boom(*a, **k):
            raise RuntimeError("compile failure (test)")

        b1pp._bounds_launch = boom
        try:
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                emb, kjt, out = self._run(rows, ids, off, 64)
                emb2, kjt2, out2 = self._run(rows, ids, off, 64)
        finally:
            b1pp._bounds_launch = saved
        self.assertEqual(sum("DISABLED" in m for m in cm.output), 1, cm.output)
        self.assertEqual(emb.calls[0][2], 1, "stock bounds check for that call")
        self.assertEqual(emb2.calls[0][2], 1, "and for the rest of the run")
        self.assertIn("bounds", b1pp._state["disabled"])
        Bp, _ = b1pp.rebag_plan(L, 3, 64)
        self.assertEqual(emb2.calls[0][1].numel(), 3 * Bp + 1, "rebag still on")

    def test_verify_mismatch_raises(self):
        self.switches(rebag=True, verify=1)
        rows, ids, off, L = self._inputs()
        saved = b1pp._rebag_launch

        def bad(offsets, T, B, Bp, **kw):
            out = saved(offsets, T, B, Bp, **kw)
            out[1] = out[2] + 1  # non-monotone
            return out

        b1pp._rebag_launch = bad
        try:
            with self.assertRaises(b1pp.VerifyMismatch):
                self._run(rows, ids, off, 64)
        finally:
            b1pp._rebag_launch = saved

    def test_jit_line_after_64_tbe_calls(self):
        self.switches(rebag=True)
        rows, ids, off, L = self._inputs(bad=False)
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            for _ in range(b1pp.JIT_REPORT_AFTER + 2):
                self._run(rows, ids, off, 64)
        lines = [m for m in cm.output if "jit after run_start" in m]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn("_b1pp_rebag_kernel", lines[0])
        # the headline separates the B1'' kernels from the (not warmed) seq-a2a K1-K4
        self.assertRegex(lines[0], r"jit after run_start: b1pp \+\d+ seq_a2a \+\d+ \(")

    def _with_stock_bounds(self, fn):
        saved = b1pp._fbgemm_bounds_check
        b1pp._fbgemm_bounds_check = fn
        self.addCleanup(setattr, b1pp, "_fbgemm_bounds_check", saved)

    def test_verify_bounds_malformed_offsets_compares_outcome_class(self):
        """A high last offset (never a cumsum): fbgemm v2 on gfx950 counts the last bag too (iter-43 lab S2:
        1202 vs 1201), a legal interleaving the Triton slow path does not take. VERIFY must not call that a
        mismatch: outcome class only, plus a WARNING naming the malformed input."""
        self.switches(bounds=True, verify=1)
        rows, ids, off, L = self._inputs()
        off2 = off.clone()
        off2[-1] = L + 7

        def stale_read(emb, ic, oc, wc):  # fbgemm's observed schedule: the last bag reads the stale last offset
            ri, ro, rw = bc_ref_schedule(emb.rows_per_table, ic, oc, "asc", False, "stale_read")
            ic.copy_(ri.to(ic.device))
            oc.copy_(ro.to(oc.device))
            wc.fill_(rw)

        self._with_stock_bounds(stale_read)
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            emb, kjt, out = self._run(rows, ids, off2, 64)
        msgs = "\n".join(cm.output)
        self.assertIn("VERIFY bounds: MALFORMED offsets", msgs)
        self.assertNotIn("VERIFY bounds OK", msgs)
        self.assertIn("armed OK: b1pp tbe bounds triton", msgs)
        self.assertEqual(emb.calls[0][2], 3)
        ref_fix = bc_ref_schedule(torch.tensor(rows), ids, off2, "asc", False, "fix_first")
        ref_stale = bc_ref_schedule(torch.tensor(rows), ids, off2, "asc", False, "stale_read")
        self.assertEqual(ref_stale[2], ref_fix[2] + 1)  # the two legal counters differ by the last bag
        self.assertEqual(int(emb.bounds_check_warning[0]), ref_fix[2])
        assert_bits(self, kjt.offsets(), ref_fix[1].to(DEV), "Triton repair of the KJT offsets")

    def test_verify_bounds_well_formed_counter_mismatch_raises(self):
        self.switches(bounds=True, verify=1)
        rows, ids, off, L = self._inputs()
        real = b1pp._fbgemm_bounds_check

        def off_by_one(emb, ic, oc, wc):
            real(emb, ic, oc, wc)
            wc.add_(1)

        self._with_stock_bounds(off_by_one)
        with self.assertRaisesRegex(b1pp.VerifyMismatch, "MISMATCH in warning"):
            self._run(rows, ids, off, 64)

    def test_verify_budget_spent_only_on_engaged_calls(self):
        self.switches(bounds=True, verify=1)
        rows, ids, off, L = self._inputs()
        saved = b1pp._bounds_launch

        def boom(*a, **k):
            raise RuntimeError("compile failure (test)")

        b1pp._bounds_launch = boom
        try:
            with self.assertLogs(_LOGGER, level="WARNING"):
                self._run(rows, ids, off, 64)
        finally:
            b1pp._bounds_launch = saved
        self.assertIn("bounds", b1pp._state["disabled"])
        self.assertEqual(b1pp._state["verify_left"]["bounds"], 1, "no VERIFY call spent on a failed launch")

    def test_size_cap_falls_back(self):
        self.switches(rebag=True, bounds=True)
        rows, ids, off, L = self._inputs()
        saved = b1pp._SIZE_MAX
        b1pp._SIZE_MAX = L - 1
        try:
            with self.assertLogs(_LOGGER, level="INFO") as cm:
                emb, kjt, out = self._run(rows, ids, off, 64)
        finally:
            b1pp._SIZE_MAX = saved
        self.assertEqual(emb.calls[0][2], 1, "stock bounds check")
        assert_bits(self, emb.calls[0][1], off, "stock offsets")
        self.assertEqual(sum("FALLBACK: values not contiguous int64" in m for m in cm.output), 2, cm.output)
        self.assertEqual(b1pp._state["disabled"], set(), "a size fallback is per call, not a failure")


# --------------------------------------------------------------------------
# (B) hook body: b1pp.kjt_recat (switch, VERIFY, fallback, failure)
# --------------------------------------------------------------------------
class KjtRecatTest(_EnvMixin, unittest.TestCase):
    def _inputs(self, K=3, W=4, B=8, seed=1):
        g = torch.Generator().manual_seed(seed)
        l2 = torch.randint(0, 4, (K * W, B), generator=g, dtype=torch.int64)
        v = torch.randint(-2 ** 62, 2 ** 62, (int(l2.sum()),), generator=g, dtype=torch.int64)
        return mk_recat(K, W), l2.reshape(-1).to(DEV), v.to(DEV), B

    def test_off_returns_none(self):
        self.switches()
        recat, lengths, v, B = self._inputs()
        self.assertIsNone(b1pp.kjt_recat(recat, lengths, v, None, B))
        self.assertEqual(b1pp._state["engaged"]["recat"], 0)

    def test_on_verify_and_arming(self):
        self.switches(recat=True, verify=2)
        recat, lengths, v, B = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            for _ in range(3):
                out = b1pp.kjt_recat(recat, lengths, v, None, B)
        msgs = "\n".join(cm.output)
        self.assertEqual(msgs.count("VERIFY recat OK"), 2, msgs)
        self.assertEqual(msgs.count("armed OK: b1pp kjt recat fused (R=12 B=8 N=%d offsets=1)" % v.numel()), 1,
                         msgs)
        ref = recat_ref(recat, lengths.view(-1, B), v)
        for name, g, r in zip(("lengths", "values", "offsets"), out, ref):
            assert_bits(self, g, r, name)
        self.assertEqual(b1pp._state["engaged"]["recat"], 3)

    def test_fallback_and_failure(self):
        self.switches(recat=True)
        recat, lengths, v, B = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            self.assertIsNone(b1pp.kjt_recat(recat, lengths, v, torch.ones(v.numel(), device=DEV), B))
            self.assertIsNone(b1pp.kjt_recat(recat, lengths, v, torch.ones(v.numel(), device=DEV), B))
        self.assertEqual(sum("recat FALLBACK: weights present" in m for m in cm.output), 1, cm.output)
        saved = b1pp._recat_launch

        def boom(*a, **k):
            raise RuntimeError("launch failure (test)")

        b1pp._recat_launch = boom
        try:
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                self.assertIsNone(b1pp.kjt_recat(recat, lengths, v, None, B))
        finally:
            b1pp._recat_launch = saved
        self.assertTrue(any("DISABLED" in m for m in cm.output))
        self.assertIsNone(b1pp.kjt_recat(recat, lengths, v, None, B), "off for the rest of the run")

    def test_int64_recat_falls_back(self):
        self.switches(recat=True)
        recat, lengths, v, B = self._inputs()
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            self.assertIsNone(b1pp.kjt_recat(recat.long(), lengths, v, None, B))
            self.assertIsNone(b1pp.kjt_recat(recat.long(), lengths, v, None, B))
        self.assertEqual(sum("recat FALLBACK: recat dtype is not int32" in m for m in cm.output), 1, cm.output)
        self.assertEqual(b1pp._state["engaged"]["recat"], 0)

    def test_verify_mismatch_raises(self):
        self.switches(recat=True, verify=1)
        recat, lengths, v, B = self._inputs()
        saved = b1pp._recat_launch

        def bad(*a, **k):
            out = saved(*a, **k)
            out[2][-1] += 1
            return out

        b1pp._recat_launch = bad
        try:
            with self.assertRaises(b1pp.VerifyMismatch):
                b1pp.kjt_recat(recat, lengths, v, None, B)
        finally:
            b1pp._recat_launch = saved


# --------------------------------------------------------------------------
# Default-off: nothing patched, nothing launched
# --------------------------------------------------------------------------
class DefaultOffTest(_EnvMixin, unittest.TestCase):
    def test_install_is_noop_and_warmup_is_noop(self):
        self.switches()
        before = b1pp.jit_cache_sizes()
        b1pp.install()
        self.assertFalse(b1pp._installed)
        self.assertIsNone(b1pp._patched_cls)
        if _REAL_TORCHREC:
            from torchrec.distributed.batched_embedding_kernel import (
                BaseBatchedEmbedding,
                BatchedFusedEmbedding,
            )

            self.assertIs(BatchedFusedEmbedding.forward, BaseBatchedEmbedding.forward)
        b1pp.run_first_use_warmup(DEV)
        self.assertFalse(b1pp._state["warmup_ran"])
        self.assertEqual(b1pp.jit_cache_sizes(), before)

    @unittest.skipUnless(_REAL_TORCHREC, "needs torchrec")
    def test_install_patches_only_when_on(self):
        from torchrec.distributed.batched_embedding_kernel import (
            BaseBatchedEmbedding,
            BatchedFusedEmbedding,
        )

        # the live torchrec forward is the body _b1pp_tbe_forward re-implements (else install() refuses)
        self.assertIsNone(b1pp._forward_mismatch(BaseBatchedEmbedding.forward))
        self.switches(rebag=True)
        b1pp.install()
        self.assertIsNone(b1pp._install_refused)
        self.assertIs(BatchedFusedEmbedding.__dict__.get("forward"), b1pp._b1pp_tbe_forward)
        self.assertIs(b1pp._orig_forward, BaseBatchedEmbedding.forward)
        b1pp.install()  # idempotent
        b1pp._uninstall_for_tests()
        self.assertIs(BatchedFusedEmbedding.forward, BaseBatchedEmbedding.forward)
        self.assertNotIn("forward", BatchedFusedEmbedding.__dict__)


# --------------------------------------------------------------------------
# install() guard: the torchrec forward must be the body _b1pp_tbe_forward re-implements
# --------------------------------------------------------------------------
def _overlay_forward_replica(self, features):
    """Verbatim body of torchrec BaseBatchedEmbedding.forward in the production overlay
    (lib_overlay_c10_v212 torchrec 1.7.0a0+bf55480, batched_embedding_kernel.py:1880-1902)."""
    forward_args: Dict[str, Any] = {}
    identities_and_metadata = self._get_hash_zch_identities_and_metadata(features)
    if identities_and_metadata is not None:
        hash_zch_identities, hash_zch_runtime_meta = identities_and_metadata
        forward_args["hash_zch_identities"] = hash_zch_identities
        if hash_zch_runtime_meta is not None:
            forward_args["hash_zch_runtime_meta"] = hash_zch_runtime_meta

    if len(forward_args) == 0:
        return self.emb_module(
            indices=features.values().to(self._embedding_table_index_type),
            offsets=features.offsets().to(self._embedding_table_offset_type),
        )
    else:
        return self.emb_module(
            indices=features.values().to(self._embedding_table_index_type),
            offsets=features.offsets().to(self._embedding_table_offset_type),
            **forward_args,
        )


def _changed_forward_replica(self, features):
    """A hypothetical later torchrec: one more forward argument plumbed into the TBE call."""
    forward_args: Dict[str, Any] = {}
    identities_and_metadata = self._get_hash_zch_identities_and_metadata(features)
    if identities_and_metadata is not None:
        forward_args["hash_zch_identities"] = identities_and_metadata[0]
    if self._total_unique_indices_enabled:
        forward_args["total_unique_indices"] = features.total_unique_indices()
    return self.emb_module(
        indices=features.values().to(self._embedding_table_index_type),
        offsets=features.offsets().to(self._embedding_table_offset_type),
        **forward_args,
    )


class InstallGuardTest(_EnvMixin, unittest.TestCase):
    def test_forward_fingerprint(self):
        self.assertIsNone(b1pp._forward_mismatch(_overlay_forward_replica))
        why = b1pp._forward_mismatch(_changed_forward_replica)
        self.assertIn("total_unique_indices", why)
        self.assertIn("argcount", b1pp._forward_mismatch(lambda self, features, extra: None))
        self.assertIn("no __code__", b1pp._forward_mismatch(len))

    @unittest.skipIf(_REAL_TORCHREC, "uses a stand-in torchrec module (CPU run only)")
    def test_install_refuses_a_forward_it_does_not_reimplement(self):
        import sys
        import types

        class Base:
            forward = _overlay_forward_replica

        class Fused(Base):
            pass

        mod = types.ModuleType("torchrec.distributed.batched_embedding_kernel")
        mod.BaseBatchedEmbedding, mod.BatchedFusedEmbedding = Base, Fused
        names = ("torchrec", "torchrec.distributed", "torchrec.distributed.batched_embedding_kernel")
        saved = {n: sys.modules.get(n) for n in names}
        sys.modules.update({names[0]: types.ModuleType(names[0]), names[1]: types.ModuleType(names[1]),
                            names[2]: mod})
        try:
            # the production body: patched
            self.switches(rebag=True)
            b1pp.install()
            self.assertIs(Fused.__dict__.get("forward"), b1pp._b1pp_tbe_forward)
            b1pp._uninstall_for_tests()
            self.assertNotIn("forward", Fused.__dict__)
            # a changed base body: refused once (WARNING), nothing patched, C and D off
            Base.forward = _changed_forward_replica
            self.switches(rebag=True, bounds=True)
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                b1pp.install()
            self.assertTrue(any("NOT installed" in m and "total_unique_indices" in m for m in cm.output),
                            cm.output)
            self.assertNotIn("forward", Fused.__dict__)
            self.assertEqual(b1pp._state["disabled"], {"rebag", "bounds"})
            self.assertFalse(b1pp._installed)
            b1pp.install()  # once refused: no second attempt
            self.assertNotIn("forward", Fused.__dict__)
            b1pp._uninstall_for_tests()
            # BatchedFusedEmbedding with its own forward (would be bypassed on engaged calls): refused
            Base.forward = _overlay_forward_replica

            def own_forward(self, features):
                return Base.forward(self, features)

            Fused.forward = own_forward
            self.switches(bounds=True)
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                b1pp.install()
            self.assertTrue(any("resolves its own forward" in m for m in cm.output), cm.output)
            self.assertIs(Fused.__dict__.get("forward"), own_forward)
        finally:
            b1pp._uninstall_for_tests()
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m


# --------------------------------------------------------------------------
# GPU: real fbgemm nobag TBE, fwd + bwd + fused EXACT_ROWWISE_ADAGRAD, A vs C vs D vs C+D
# --------------------------------------------------------------------------
def make_tbe(rows: List[int], D: int, ftm: List[int], seed: int, lr=4e-6, eps=1e-8, pooled: bool = False):
    """As torchrec BatchedFusedEmbedding builds it (pooling NONE, fp32, fused EXACT_ROWWISE_ADAGRAD,
    WARNING, int64) -- rebag_lab.make_tbe. pooled=True: a SUM-pooled TBE (not eligible, for fallbacks)."""
    from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        ComputeDevice,
        EmbeddingLocation,
        PoolingMode,
        SplitTableBatchedEmbeddingBagsCodegen,
    )
    from fbgemm_gpu.tbe.config import BoundsCheckMode

    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[(r, D, EmbeddingLocation.DEVICE, ComputeDevice.CUDA) for r in rows],
        feature_table_map=ftm,
        pooling_mode=PoolingMode.SUM if pooled else PoolingMode.NONE,
        weights_precision=SparseType.FP32,
        device=DEV,
        cache_precision=SparseType.FP32,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=lr,
        eps=eps,
        weight_decay=0,
        output_dtype=SparseType.FP32,
        bounds_check_mode=BoundsCheckMode.WARNING,
        stochastic_rounding=True,
        embedding_table_index_type=torch.int64,
        embedding_table_offset_type=torch.int64,
    )
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for w, r in zip(tbe.split_embedding_weights(), rows):
            b = math.sqrt(1.0 / r)
            w.uniform_(-b, b, generator=g)
    return tbe


def tbe_inputs(T: int, B: int, ftm: List[int], rows: List[int], n_t: List[int], seed: int,
               hot: Optional[List[Tuple[int, int, int]]] = None, n_bad: int = 0):
    """dedup-like offsets; ids random per feature; `hot` = (feature, row, occurrences) duplicated rows
    (64..4096 occurrences: the CTA path whose add order depends on the predicate); n_bad invalid ids
    (rows+7 and -3, never -1)."""
    off = dedup_like_offsets(T, B, n_t, seed=seed)
    L = int(off[-1])
    g = torch.Generator().manual_seed(seed + 1)
    fo = feature_of_pos(off, B, L).cpu()
    r_of = torch.tensor([rows[ftm[t]] for t in range(T)])[fo]
    ids = torch.randint(0, 2 ** 40, (L,), generator=g) % r_of
    S = off[::B].cpu().tolist()
    for (t, row, cnt) in (hot or []):
        n = S[t + 1] - S[t]
        pos = S[t] + torch.randperm(n, generator=g)[: min(cnt, n)]
        ids[pos] = row
    if n_bad:
        pos = torch.randperm(L, generator=g)[:n_bad]
        ids[pos[0::2]] = r_of[pos[0::2]] + 7
        ids[pos[1::2]] = -3
    return ids.to(DEV), off


@unittest.skipUnless(_REAL_FBGEMM and _REAL_TORCHREC and _HAS_GPU, "needs a GPU and real fbgemm/torchrec")
class TbeBitwiseTest(_EnvMixin, unittest.TestCase):
    """Two fwd+bwd steps (random fp32 grads, so the add order matters) per arm, from the same pristine
    weights/momentum: outputs, repaired ids, warning counters, all weights and Adagrad momentum bitwise."""

    def _arms(self, tbe, T, B, steps, grads):
        from torchrec.distributed.batched_embedding_kernel import BaseBatchedEmbedding
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        W0 = tbe.weights_dev.detach().clone()
        M0 = tbe.momentum1_dev.detach().clone()
        duck = _Duck(tbe)
        res = {}
        for arm, sw in (("A", None), ("C", dict(rebag=True)), ("D", dict(bounds=True)),
                        ("CD", dict(rebag=True, bounds=True))):
            with torch.no_grad():
                tbe.weights_dev.copy_(W0)
                tbe.momentum1_dev.copy_(M0)
            if sw is not None:
                self.switches(**sw)
            outs, idx_after, warns = [], [], []
            for st, (ids, off) in enumerate(steps):
                lengths = (off[1:] - off[:-1]).clone()
                kjt = KeyedJaggedTensor(keys=[f"f{t}" for t in range(T)], values=ids.clone(),
                                        lengths=lengths, stride=B)
                if sw is None:
                    out = BaseBatchedEmbedding.forward(duck, kjt)
                else:
                    out = b1pp._b1pp_tbe_forward(duck, kjt)
                warns.append(int(tbe.bounds_check_warning[0]))
                outs.append(out.detach().clone())
                idx_after.append(kjt.values().clone())
                out.backward(grads[st])
            torch.cuda.synchronize()
            res[arm] = dict(outs=outs, idx=idx_after, warns=warns, W=tbe.weights_dev.detach().clone(),
                            M=tbe.momentum1_dev.detach().clone(), armed=set(b1pp._state["armed"]))
        return res

    def _compare(self, res, tag):
        A = res["A"]
        for arm in ("C", "D", "CD"):
            R = res[arm]
            for st in range(len(A["outs"])):
                assert_bits(self, R["outs"][st], A["outs"][st], f"{tag} {arm} step{st} output")
                assert_bits(self, R["idx"][st], A["idx"][st], f"{tag} {arm} step{st} repaired ids")
            self.assertEqual(R["warns"], A["warns"], f"{tag} {arm} warnings")
            assert_bits(self, R["W"], A["W"], f"{tag} {arm} weights")
            assert_bits(self, R["M"], A["M"], f"{tag} {arm} momentum")
        self.assertIn("rebag", res["C"]["armed"])
        self.assertIn("bounds", res["D"]["armed"])
        self.assertEqual(res["CD"]["armed"], {"rebag", "bounds"})

    def _case(self, T, B, ftm, rows, n_ts, hot, n_bad, D, seed, tag):
        tbe = make_tbe(rows, D, ftm, seed)
        steps = [tbe_inputs(T, B, ftm, rows, n_t, seed + 10 * k, hot=hot, n_bad=n_bad)
                 for k, n_t in enumerate(n_ts)]
        g = torch.Generator(device=DEV).manual_seed(seed + 99)
        grads = [torch.randn(int(off[-1]), D, generator=g, device=DEV) * 1e-2 for _, off in steps]
        for ids, off in steps:  # the layout must engage and keep the predicate
            L = int(off[-1])
            self.assertIsNotNone(b1pp.rebag_plan(L, T, B)[0], (tag, L, T, B))
        self._compare(self._arms(tbe, T, B, steps, grads), tag)
        del tbe

    def test_small_shared_tables(self):
        # T=3 features, table 0 shared by features 0 and 1; hot rows 63/64/65/300/4096 occurrences
        self._case(T=3, B=8192, ftm=[0, 0, 1], rows=[20000, 8000], n_ts=[[9000, 4000, 1100], [8800, 4100, 1000]],
                   hot=[(0, 101, 63), (0, 202, 64), (0, 303, 65), (1, 404, 300), (0, 505, 4096), (2, 77, 64)],
                   n_bad=40, D=512, seed=1, tag="T3")

    def test_production_like_14_features(self):
        ftm = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        rows = [29346, 4042, 10524, 3125, 31250, 12500, 7500, 12500, 10000, 6250, 12500]
        n1 = [0, 8034, 0, 3472, 0, 5338, 103, 97, 94, 102, 98, 101, 102, 101]
        n2 = [10, 7900, 5, 3500, 0, 5400, 100, 99, 90, 104, 97, 103, 99, 100]
        self._case(T=14, B=2048, ftm=ftm, rows=rows, n_ts=[n1, n2], hot=[(1, 7, 70), (5, 9, 130)],
                   n_bad=30, D=512, seed=3, tag="T14")

    def test_static_eligibility_and_real_fallbacks(self):
        """_tbe_static_reason on real fbgemm TBEs, and non-eligible calls through _b1pp_tbe_forward with a
        real TBE: the stock forward runs, bitwise, and nothing arms."""
        from torchrec.distributed.batched_embedding_kernel import BaseBatchedEmbedding
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        T, B, ftm, rows = 3, 1024, [0, 0, 1], [3000, 2000]
        nobag = make_tbe(rows, 128, ftm, seed=7)
        self.assertEqual(b1pp._tbe_static_reason(_Duck(nobag)), (None, T))
        d32 = _Duck(nobag)
        d32._embedding_table_index_type = torch.int32
        self.assertIn("int64", b1pp._tbe_static_reason(d32)[0])
        pooled = make_tbe(rows, 128, ftm, seed=7, pooled=True)
        self.assertEqual(b1pp._tbe_static_reason(_Duck(pooled))[0], "pooled TBE")
        ids, off = tbe_inputs(T, B, ftm, rows, [700, 300, 200], seed=9, n_bad=6)
        lengths = (off[1:] - off[:-1]).clone()
        self.switches(rebag=True, bounds=True)
        for tbe, kw, why in ((pooled, {}, "pooled TBE"),
                             (nobag, {"weights": torch.ones(ids.numel(), device=DEV)}, "per-sample weights")):
            outs = []
            for patched in (False, True):
                kjt = KeyedJaggedTensor(keys=["a", "b", "c"], values=ids.clone(), lengths=lengths.clone(),
                                        stride=B, **kw)
                with torch.no_grad():
                    if patched:
                        with self.assertLogs(_LOGGER, level="INFO") as cm:
                            out = b1pp._b1pp_tbe_forward(_Duck(tbe), kjt)
                        self.assertTrue(any("FALLBACK: " + why in m for m in cm.output), cm.output)
                    else:
                        out = BaseBatchedEmbedding.forward(_Duck(tbe), kjt)
                outs.append((out.clone(), kjt.values().clone(), int(tbe.bounds_check_warning[0])))
            assert_bits(self, outs[1][0], outs[0][0], why + " output")
            assert_bits(self, outs[1][1], outs[0][1], why + " repaired ids")
            self.assertEqual(outs[1][2], outs[0][2], why + " warning")
        self.assertEqual(b1pp._state["armed"], set())
        self.assertEqual(b1pp._state["engaged"]["tbe"], 0)

    def test_real_batched_fused_embedding_if_constructible(self):
        """install() + a real torchrec BatchedFusedEmbedding, built as torchrec's table-wise sharding builds it
        on one rank. Its constructor runs EmbeddingFusedOptimizer, which wraps every table's weight and
        momentum in a ShardedTensor: each table needs local + global ShardMetadata (without them: a bare
        AssertionError, overlay batched_embedding_kernel.py:1510) and an initialized default process group
        (else ValueError): a 1-rank gloo group on an in-memory store is created if none exists (destroyed
        after the test). Skipped only if this torch/torchrec/fbgemm API is missing; a construction failure
        is an error, not a skip."""
        try:
            import torch.distributed as dist
            from fbgemm_gpu.split_embedding_configs import EmbOptimType
            from torch.distributed._shard.metadata import ShardMetadata
            from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
            from torchrec.distributed.batched_embedding_kernel import BatchedFusedEmbedding
            from torchrec.distributed.embedding_types import (
                EmbeddingComputeKernel,
                GroupedEmbeddingConfig,
                ShardedEmbeddingTable,
            )
            from torchrec.modules.embedding_configs import DataType, PoolingType
        except ImportError as e:
            self.skipTest(f"torch/torchrec/fbgemm API missing ({type(e).__name__}: {e})")
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
            self.addCleanup(dist.destroy_process_group)
        T, B, D = 3, 1024, 128
        placement = f"rank:{dist.get_rank()}/cuda:{torch.cuda.current_device()}"
        tables = []
        for i, (r, fn) in enumerate([(5000, ["a", "b"]), (3000, ["c"])]):
            md = ShardMetadata(shard_offsets=[0, 0], shard_sizes=[r, D], placement=placement)
            tables.append(ShardedEmbeddingTable(
                num_embeddings=r, embedding_dim=D, name=f"t{i}", feature_names=fn, data_type=DataType.FP32,
                pooling=PoolingType.NONE, local_rows=r, local_cols=D, compute_kernel=EmbeddingComputeKernel.FUSED,
                local_metadata=md, global_metadata=ShardedTensorMetadata(shards_metadata=[md],
                                                                         size=torch.Size([r, D]))))
        cfg = GroupedEmbeddingConfig(data_type=DataType.FP32, pooling=PoolingType.NONE, is_weighted=False,
                                     has_feature_processor=False, compute_kernel=EmbeddingComputeKernel.FUSED,
                                     embedding_tables=tables,
                                     fused_params={"optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                                                   "learning_rate": 4e-6, "eps": 1e-8})
        mod = BatchedFusedEmbedding(cfg, pg=dist.group.WORLD, device=DEV)
        self.assertEqual(b1pp._tbe_static_reason(mod), (None, T))  # eligible: C and D can arm on it
        ids, off = tbe_inputs(T, B, [0, 0, 1], [5000, 3000], [900, 300, 200], seed=5, n_bad=6)
        emb = mod._emb_module
        W0, M0 = emb.weights_dev.detach().clone(), emb.momentum1_dev.detach().clone()
        # random fp32 grads (as _case): the fused backward's per-row add order shows in the bits
        grad = torch.randn(int(off[-1]), D, generator=_gen(55), device=DEV) * 1e-2

        def run():
            with torch.no_grad():
                emb.weights_dev.copy_(W0)
                emb.momentum1_dev.copy_(M0)
            kjt = KeyedJaggedTensor(keys=["a", "b", "c"], values=ids.clone(), lengths=(off[1:] - off[:-1]).clone(),
                                    stride=B)
            out = mod(kjt)
            out.backward(grad)
            torch.cuda.synchronize()
            return out.detach().clone(), kjt.values().clone(), int(emb.bounds_check_warning[0]), \
                emb.weights_dev.detach().clone(), emb.momentum1_dev.detach().clone()

        ref = run()
        self.assertGreater(ref[2], 0, "the stock WARNING check repaired no id")
        self.switches(rebag=True, bounds=True)
        b1pp.install()
        self.assertIs(type(mod).forward, b1pp._b1pp_tbe_forward)
        got = run()
        self.assertEqual(b1pp._state["armed"], {"rebag", "bounds"})
        for k, (a, b) in enumerate(zip(got, ref)):
            if isinstance(a, int):
                self.assertEqual(a, b)
            else:
                assert_bits(self, a, b, f"real module field {k}")


# --------------------------------------------------------------------------
# torchrec: the KJT built with offsets, and the p_kjtt_wait hook end to end
# --------------------------------------------------------------------------
class _FakeWork:
    def wait(self):
        return True


_MISSING = object()


def _save_attrs(pairs) -> List[Tuple[Any, str, Any]]:
    """(owner, name) pairs (classes or modules) -> what each owner itself holds (not inherited)."""
    return [(owner, name, vars(owner).get(name, _MISSING)) for owner, name in pairs]


def _restore_attrs(saved: List[Tuple[Any, str, Any]]) -> None:
    for owner, name, v in saved:
        if v is _MISSING:
            if name in vars(owner):
                delattr(owner, name)
        else:
            setattr(owner, name, v)


class AttrRestoreTest(unittest.TestCase):
    def test_save_restore(self):
        import types

        class A:
            def f(self):
                return 1

        class Bc(A):
            pass

        m = types.ModuleType("m")
        saved = _save_attrs([(A, "f"), (Bc, "f"), (m, "flag")])
        A.f = lambda self: 2
        Bc.f = lambda self: 3
        m.flag = True
        _restore_attrs(saved)
        self.assertEqual(Bc().f(), 1)
        self.assertNotIn("f", vars(Bc))
        self.assertFalse(hasattr(m, "flag"))


# host_sync_free.install_kjt_lpk_patch() patches these process-wide; KjtHookTest restores them afterwards so the
# patch does not leak into later test files of the same pytest process (run_gpu_tests.sh runs two files).
_LPK_PATCH_ATTRS = (
    ("dd", None, "_dlrm_lpk_patched"),
    ("dd", "SplitsAllToAllAwaitable", "__init__"),
    ("dd", "SplitsAllToAllAwaitable", "_wait_impl"),
    ("dd", "KJTAllToAllSplitsAwaitable", "__init__"),
    ("dd", "KJTAllToAllSplitsAwaitable", "_wait_impl"),
    ("dd", "KJTAllToAllTensorsAwaitable", "__init__"),
    ("dd", "KJTAllToAllTensorsAwaitable", "_wait_impl"),
    ("es", "FusedKJTListSplitsAwaitable", "__init__"),
    ("es", "FusedKJTListSplitsAwaitable", "_wait_impl"),
)


@unittest.skipUnless(_REAL_TORCHREC and _REAL_FBGEMM, "needs real torchrec/fbgemm")
class KjtHookTest(_EnvMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        import torchrec.distributed.dist_data as dd
        import torchrec.distributed.embedding_sharding as es

        mods = {"dd": dd, "es": es}
        self._lpk_saved = _save_attrs(
            [(mods[m] if c is None else getattr(mods[m], c), n) for m, c, n in _LPK_PATCH_ATTRS]
        )

    def tearDown(self):
        _restore_attrs(self._lpk_saved)
        super().tearDown()

    def _a2a_output(self, K, W, stride, seed):
        g = torch.Generator().manual_seed(seed)
        l2 = torch.randint(0, 3, (W * K, stride), generator=g, dtype=torch.int64)  # source-major rows
        v = torch.randint(-2 ** 62, 2 ** 62, (int(l2.sum()),), generator=g, dtype=torch.int64)
        lpk = [int(l2.view(W, K, stride)[:, i, :].sum()) for i in range(K)]
        return l2.reshape(-1).to(DEV), v.to(DEV), lpk

    def _kjt_pair(self, K=3, W=4, stride=8, seed=2):
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        lengths, v, lpk = self._a2a_output(K, W, stride, seed)
        recat = get_recat(K, W, stride)
        keys = [f"k{i}" for i in range(K)]
        fl, fv, _ = torch.ops.fbgemm.permute_2D_sparse_data(recat, lengths.view(-1, stride), v, None, v.numel())
        stock = KeyedJaggedTensor(keys=keys, values=fv, lengths=fl.view(-1), stride=W * stride,
                                  length_per_key=list(lpk)).sync()
        self.switches(recat=True)
        out = b1pp.kjt_recat(recat, lengths, v, None, stride)
        self.assertIsNotNone(out)
        ours = KeyedJaggedTensor(keys=keys, values=out[1], lengths=out[0].view(-1), offsets=out[2],
                                 stride=W * stride, length_per_key=list(lpk)).sync()
        return stock, ours, keys

    def _same_kjt(self, a, b, what):
        assert_bits(self, a.values(), b.values(), what + " values")
        assert_bits(self, a.lengths(), b.lengths(), what + " lengths")
        assert_bits(self, a.offsets(), b.offsets(), what + " offsets")
        self.assertEqual(a.length_per_key(), b.length_per_key(), what + " length_per_key")
        self.assertEqual(a.offset_per_key(), b.offset_per_key(), what + " offset_per_key")
        self.assertEqual(a.keys(), b.keys(), what + " keys")
        self.assertEqual(a.stride(), b.stride(), what + " stride")

    def test_kjt_with_offsets_equals_stock(self):
        stock, ours, keys = self._kjt_pair()
        self.assertIsNotNone(ours.offsets_or_none())
        self._same_kjt(ours, stock, "kjt")
        for segs in ([3], [1, 2], [2, 1], [0, 3], [1, 0, 2], [1, 1, 1]):
            a, b = ours.split(segs), stock.split(segs)
            self.assertEqual(len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                self._same_kjt(x, y, f"split {segs}[{i}]")
            if segs == [3]:
                self.assertIs(a[0], ours, "one segment over all keys returns the KJT itself (offsets kept)")

    def test_p_kjtt_wait_hook(self):
        import torchrec.distributed.dist_data as dd
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        from generative_recommenders.dlrm_v4 import host_sync_free as hsf

        hsf.install_kjt_lpk_patch()
        K, W, stride = 3, 4, 8

        def build(seed):
            lengths, v, lpk = self._a2a_output(K, W, stride, seed)
            aw = object.__new__(dd.KJTAllToAllTensorsAwaitable)
            aw._workers = W
            aw._recat = get_recat(K, W, stride)
            aw._stride_per_rank = [stride] * W
            aw._keys = [f"k{i}" for i in range(K)]
            aw._output_tensors = [lengths, v]
            aw._input_tensors = []
            aw._awaitables = [_FakeWork()]
            aw._stagger = 1
            aw._input = KeyedJaggedTensor(keys=aw._keys, values=v[:0], lengths=torch.zeros(K * stride,
                                          dtype=torch.int64, device=DEV), stride=stride)
            setattr(aw, hsf._LPK_ATTR, lpk)
            return aw

        self.switches()
        kjt_off = dd.KJTAllToAllTensorsAwaitable._wait_impl(build(4))
        self.assertIsNone(kjt_off.offsets_or_none(), "stock: no offsets carried")
        self.switches(recat=True)
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            kjt_on = dd.KJTAllToAllTensorsAwaitable._wait_impl(build(4))
        self.assertTrue(any("armed OK: b1pp kjt recat fused" in m for m in cm.output), cm.output)
        self.assertIsNotNone(kjt_on.offsets_or_none(), "fused: offsets carried")
        self._same_kjt(kjt_on, kjt_off, "p_kjtt_wait on vs off")
        # stock torchrec dist_init on the same tensors
        aw = build(4)
        ref = KeyedJaggedTensor.dist_init(keys=aw._keys, tensors=aw._output_tensors, variable_stride_per_key=False,
                                          num_workers=W, recat=aw._recat, stride_per_rank=aw._stride_per_rank)
        self._same_kjt(kjt_on, ref, "p_kjtt_wait on vs torchrec dist_init")


# --------------------------------------------------------------------------
# Warmup: compiles the run's variants; nothing new afterwards
# --------------------------------------------------------------------------
@unittest.skipUnless(_HAS_GPU and _REAL_FBGEMM, "GPU only")
class WarmupTest(_EnvMixin, unittest.TestCase):
    def test_warmup_then_production_calls_compile_nothing(self):
        for name in b1pp.KERNEL_NAMES:
            getattr(b1pp, name).device_caches.clear()
        self.switches(recat=True, rebag=True, bounds=True)
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            b1pp.run_first_use_warmup(DEV)
        line = [m for m in cm.output if "[b1pp-warmup]" in m]
        self.assertEqual(len(line), 1, cm.output)
        self.assertIn("armed OK: [b1pp-warmup] kernels=5 triton_jit_new=5", line[0])
        c1 = b1pp.jit_cache_sizes()
        for name in b1pp.KERNEL_NAMES:
            self.assertEqual(c1[name], 1, name)
        # a second call is a no-op (train_ranker calls it twice in the streaming mode: before the startup
        # preload's gc.freeze and once more after that block)
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            b1pp.run_first_use_warmup(DEV)
            logging.getLogger(_LOGGER).info("marker (assertLogs needs one record)")
        self.assertFalse(any("[b1pp-warmup]" in m for m in cm.output), cm.output)
        self.assertEqual(b1pp.jit_cache_sizes(), c1)
        # forced re-run: compiles nothing new
        b1pp._state["warmup_ran"] = False
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            b1pp.run_first_use_warmup(DEV)
        self.assertTrue(any("triton_jit_new=0" in m for m in cm.output), cm.output)
        self.assertEqual(b1pp.jit_cache_sizes(), c1)
        # production-shaped calls: 4n/2n/1n recat, bounds (incl. a misaligned split-child view), rebag; and
        # beyond the production envelope (R = 896 > BLOCK_R, total_B = 655,360 > BLOCK_F * BB): the chunk
        # loops take them with the same variants
        for (K, W, stride, Nt) in ((14, 32, 1024, 194_427), (14, 16, 1024, 140_001), (3, 8, 1024, 15_096),
                                   (14, 16, 1000, 100_000), (14, 64, 8, 20_000)):
            l2, v = recat_inputs(K, W, stride, Nt, "hstu", 7 + W)
            b1pp.recat_fused(get_recat(K, W, stride), l2, v)
        for (T, B, Nt) in ((14, 32768, 194_427), (14, 16384, 140_001), (2, 8192, 182_324), (16, 40960, 0)):
            rows, idx, off = make_bc_case(T, B, 1, 3, device=DEV)
            w = torch.zeros(1, dtype=torch.int64, device=DEV)
            b1pp.bounds_check_warning(rows, idx, off, w)
            buf = torch.empty(idx.numel() + 1, dtype=torch.int64, device=DEV)
            buf[1:] = idx
            b1pp.bounds_check_warning(rows, buf[1:], off, w)
            L = int(off[-1])
            if b1pp.rebag_plan(L, T, B)[0] is not None:
                b1pp.rebag_rkeep(off, T, B, L)
        torch.cuda.synchronize()
        self.assertEqual(b1pp.jit_cache_sizes(), c1, "no new variant after the warmup")
        total, per = b1pp.jit_delta()
        self.assertEqual(total, 0, per)


# --------------------------------------------------------------------------
# Warmup hardening (ship-14 readiness (a)): an ordinary failure is a logged
# fallback, ANY failed synchronize (a device fault) is raised, after leaving the
# device context, as blas_warmup.warmup does. Control flow only (CUDA calls
# mocked; no kernel runs), so it runs in the CPU opt-in too.
# --------------------------------------------------------------------------
import contextlib  # noqa: E402
import types  # noqa: E402
from unittest import mock  # noqa: E402

_HIP_FAULT = "HIP error: an illegal memory access was encountered"


class WarmupFaultTest(_EnvMixin, unittest.TestCase):
    def _run(self, sync, recat=None, bounds=None, rebag=None, device_ctx=None, seq=None):
        """run_first_use_warmup with CUDA mocked; `sync` is torch.cuda.synchronize's side effect;
        `device_ctx` replaces torch.cuda.device (default: a no-op context); `seq` is the side effect
        of _warm_seq_a2a (None: the K1-K4 part is off)."""
        fakes = {"_warm_recat": recat or (lambda dev: 2), "_warm_bounds": bounds or (lambda dev: 2),
                 "_warm_rebag": rebag or (lambda dev: 1)}
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            st.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
            st.enter_context(mock.patch.object(
                torch.cuda, "device", device_ctx or (lambda d: contextlib.nullcontext())))
            st.enter_context(mock.patch.object(torch.cuda, "synchronize", side_effect=sync))
            st.enter_context(mock.patch.object(b1pp, "_seq_a2a_wanted", return_value=seq is not None))
            if seq is not None:
                st.enter_context(mock.patch.object(b1pp, "_warm_seq_a2a", side_effect=seq))
            for name, fn in fakes.items():
                st.enter_context(mock.patch.object(b1pp, name, side_effect=fn))
            b1pp.run_first_use_warmup(torch.device("cuda", 0))

    def test_ordinary_failure_is_a_logged_fallback(self):
        self.switches(recat=True, rebag=True, bounds=True)

        def bad(dev):
            raise ValueError("compile failed")

        with self.assertLogs(_LOGGER, level="INFO") as cm:
            self._run(sync=None, recat=bad)  # the device is fine: synchronize succeeds
        text = "\n".join(cm.output)
        self.assertIn("recat: [b1pp-warmup] kernels raised ValueError: compile failed", text)
        self.assertIn("recat path DISABLED", text)
        self.assertIn("armed OK: [b1pp-warmup] kernels=3", text)  # bounds 2 + rebag 1 still warmed
        self.assertIn("(parts=rebag,bounds)", text)
        self.assertIn("recat", b1pp._state["disabled"])
        self.assertTrue(b1pp._state["warmup_ran"])

    def test_any_failed_synchronize_raises(self):
        """As blas_warmup.warmup: a synchronize error raises even if it would not persist. The part
        launched fine, so it is not disabled; the run stops before RUN_START."""
        self.switches(bounds=True)
        errs = [RuntimeError("transient")]

        def sync(*a, **k):
            if errs:
                raise errs.pop()

        with self.assertRaises(b1pp.WarmupDeviceFault) as cm:
            self._run(sync=sync)
        msg = str(cm.exception)
        self.assertIn("synchronize after bounds: RuntimeError: transient", msg)
        self.assertIn("first failure: none", msg)
        self.assertNotIn("bounds", b1pp._state["disabled"])
        self.assertFalse(b1pp._state["warmup_ran"])

    def test_fault_survives_a_failing_device_context_exit(self):
        """The fault is raised after leaving torch.cuda.device(dev), so an error from that context's
        exit cannot replace it (run_first_use_warmup swallows every other error)."""
        self.switches(recat=True)

        class _ExitFails:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                raise RuntimeError("could not restore the device")

        with self.assertRaises(b1pp.WarmupDeviceFault) as cm:
            self._run(sync=RuntimeError(_HIP_FAULT), device_ctx=lambda d: _ExitFails())
        self.assertIn("synchronize after recat", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertFalse(b1pp._state["warmup_ran"])

    def test_device_context_exit_error_without_fault_is_ordinary(self):
        """No failed synchronize: an error from the device context's exit stays an ordinary one."""
        self.switches(recat=True)

        class _ExitFails:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                raise RuntimeError("could not restore the device")

        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            self._run(sync=None, device_ctx=lambda d: _ExitFails())
        self.assertIn("[b1pp-warmup] skipped: unexpected error (RuntimeError", "\n".join(cm.output))
        self.assertFalse(b1pp._state["warmup_ran"])

    def test_failed_synchronize_after_k1k4_raises(self):
        self.switches()  # no b1pp part on: only the K1-K4 part launches and synchronizes
        with self.assertRaises(b1pp.WarmupDeviceFault) as cm:
            self._run(sync=RuntimeError(_HIP_FAULT), seq=lambda dev, model, bs: (["K1 D=512"], None))
        self.assertIn("synchronize after seq_a2a K1-K4", str(cm.exception))
        self.assertFalse(b1pp._state["warmup_ran"])

    def test_sticky_device_fault_raises(self):
        self.switches(recat=True, rebag=True, bounds=True)

        def fault(dev):
            raise RuntimeError(_HIP_FAULT)

        def sync(*a, **k):
            raise RuntimeError(_HIP_FAULT)

        with self.assertRaises(b1pp.WarmupDeviceFault) as cm:
            self._run(sync=sync, recat=fault)
        msg = str(cm.exception)
        self.assertIn("device fault during the warmup", msg)
        self.assertIn("first failure: recat", msg)
        self.assertIn("DLRM_HSTU_FP8_KJT_RECAT_FUSED=0", msg)
        self.assertIn("DLRM_HSTU_FP8_SEQ_A2A_FUSED=0", msg)
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertFalse(b1pp._state["warmup_ran"])

    def test_sticky_fault_seen_only_at_the_end_raises(self):
        """Every part launched fine; only the synchronize reports the fault."""
        self.switches(recat=True)
        with self.assertRaises(b1pp.WarmupDeviceFault):
            self._run(sync=RuntimeError(_HIP_FAULT))

    def test_other_unexpected_errors_never_raise(self):
        self.switches(recat=True)
        with mock.patch.object(b1pp, "_run_warmup", side_effect=KeyError("x")), \
                self.assertLogs(_LOGGER, level="WARNING") as cm:
            b1pp.run_first_use_warmup(DEV)
        self.assertIn("[b1pp-warmup] skipped: unexpected error (KeyError", "\n".join(cm.output))

    def test_default_off_touches_no_cuda(self):
        self.switches()
        with mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no CUDA call")), \
                mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("no CUDA call")), \
                mock.patch.object(b1pp, "_seq_a2a_wanted", return_value=False):
            b1pp.run_first_use_warmup(DEV)
        self.assertFalse(b1pp._state["warmup_ran"])


# --------------------------------------------------------------------------
# K1-K4 warmup plan (ship-14 readiness (b)): which seq_a2a_fused launches the
# warmup makes for a model's sequence all-to-alls. Stand-in modules with the
# torchrec attribute layout; the four launchers are spies (no kernel runs), so
# this runs on CPU too. The GPU proof that the launched variants are exactly
# the ones production-shaped calls use is comm iter-48 lab task comm48_k1k4.
# --------------------------------------------------------------------------
class _StandInEC(torch.nn.Module):
    def __init__(self, sites):
        super().__init__()
        self._sharding_type_to_sharding = {st: None for st, _d, _od in sites}
        self._output_dists = [od for _st, _d, od in sites]
        self._dims = {st: d for st, d, _od in sites}

    def _embedding_dim_for_sharding_type(self, st):
        return self._dims[st]


def _fake_a2a(local_T: int, W: int, dtype=torch.int32):
    n = local_T * W
    fwd = [j + i * W for j in range(W) for i in range(local_T)]
    bwd = [i + j * local_T for i in range(local_T) for j in range(W)]
    return types.SimpleNamespace(
        _local_split=local_T, _num_splits=W, _codecs=object(),
        _forward_recat_tensor=torch.tensor(fwd, dtype=dtype).view(n),
        _backward_recat_tensor=torch.tensor(bwd, dtype=dtype).view(n),
    )


class SeqA2aWarmupPlanTest(_EnvMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from generative_recommenders.dlrm_v4 import seq_a2a_fused as saf
        from generative_recommenders.dlrm_v4 import seq_a2a_rowperm as sarp
        from generative_recommenders.dlrm_v4 import unbucketize_gather as ubg

        self.saf, self.sarp, self.ubg = saf, sarp, ubg
        def _copy(state):
            return {k: (set(v) if isinstance(v, set) else dict(v) if isinstance(v, dict) else v)
                    for k, v in state.items()}

        self._saved = (sarp._installed, sarp._recv_installed, _copy(sarp._state), ubg._installed,
                       ubg._RECV_HOOK, _copy(ubg._state))
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        cpu = torch.device("cpu")

        def send_fwd(emb, blk, recat, tok, n_blocks, D):
            self.calls.append(("K1", dict(D=D, n_blocks=n_blocks, tok=tuple(tok.shape), tok_dtype=tok.dtype,
                                          emb_dtype=emb.dtype, blk=tuple(blk.shape), recat=recat)))
            return torch.empty(0), torch.zeros(emb.numel(), dtype=torch.float16)

        def send_bwd(wire, blk, recat, bwd_recat, D, world_size):
            self.calls.append(("K2", dict(D=D, n_blocks=recat.numel(), world_size=world_size,
                                          wire_dtype=wire.dtype, recat=recat, bwd_recat=bwd_recat)))
            return torch.empty(0)

        def recv_fwd(wire, perm, D):
            self.calls.append(("K3", dict(D=D, perm=None if perm is None else perm.dtype, wire_dtype=wire.dtype)))
            return torch.zeros((wire.numel() // D, D), dtype=torch.float32)

        def recv_bwd(grad, perm, D):
            self.calls.append(("K4", dict(D=D, perm=None if perm is None else perm.dtype, grad=grad.dtype,
                                          contiguous=grad.is_contiguous())))
            return torch.empty(0)

        self._patches = [mock.patch.object(saf, n, f) for n, f in (
            ("send_fwd", send_fwd), ("send_bwd", send_bwd), ("recv_fwd", recv_fwd), ("recv_bwd", recv_bwd))]
        self._patches.append(mock.patch.object(sarp, "_static_reason", return_value=None))
        for p in self._patches:
            p.start()
        sarp._installed, sarp._recv_installed = True, True
        sarp._state.update(fused_cfg=(True, True), recv_cfg=True, fused_disabled=False, recv_disabled=False)
        ubg._installed, ubg._RECV_HOOK = True, (lambda *a: None)
        ubg._state["disabled"] = False
        self.dev = cpu

    def tearDown(self):
        for p in self._patches:
            p.stop()
        sarp, ubg = self.sarp, self.ubg
        sarp._installed, sarp._recv_installed = self._saved[0], self._saved[1]
        sarp._state.clear()
        sarp._state.update(self._saved[2])
        ubg._installed, ubg._RECV_HOOK = self._saved[3], self._saved[4]
        ubg._state.clear()
        ubg._state.update(self._saved[5])
        super().tearDown()

    def _model(self, sites):
        outer = torch.nn.Module()
        outer.ec = _StandInEC([(st, D, types.SimpleNamespace(_dist=a2a)) for st, D, a2a in sites])
        return outer

    def test_multinode_row_wise(self):
        """2n: 14 RW features, W=16, D=512: one K1 (blocks 224, batch 1024), K2 (scaled by W), K3'/K4 with perm."""
        a2a = _fake_a2a(14, 16)
        labels, why = b1pp._warm_seq_a2a(self.dev, self._model([("row_wise", 512, a2a)]), (1024, 1024))
        self.assertIsNone(why)
        self.assertEqual(labels, ["K1 D=512 blocks=224 B=1024", "K2 D=512 blocks=224 div=1",
                                  "K3' D=512 perm=1", "K4 D=512 perm=1"])
        k = dict((n, a) for n, a in self.calls)
        self.assertEqual(k["K1"]["tok"], (14, 16 * 1024))  # lengths_after_input_dist layout [local_T, W*B]
        self.assertEqual((k["K1"]["tok_dtype"], k["K1"]["emb_dtype"], k["K1"]["blk"]),
                         (torch.int64, torch.float32, (224,)))
        self.assertIs(k["K1"]["recat"], a2a._forward_recat_tensor)  # the module's own buffer
        self.assertIs(k["K2"]["bwd_recat"], a2a._backward_recat_tensor)
        self.assertEqual((k["K2"]["world_size"], k["K2"]["wire_dtype"]), (16, torch.float16))
        self.assertEqual((k["K3"]["perm"], k["K4"]["perm"], k["K4"]["grad"], k["K4"]["contiguous"]),
                         (torch.int64, torch.int64, torch.float32, True))

    def test_one_node_tw_cw_and_receive_only_rank(self):
        """1n forced on: TW D=512 + CW D=128 (no permutation); a rank with no local feature warms K3'/K4 only;
        a one-rank a2a (torchrec NoWait) warms nothing; equal (D, blocks) sites are deduplicated."""
        sites = [("table_wise", 512, _fake_a2a(2, 8)), ("column_wise", 128, _fake_a2a(2, 8)),
                 ("data_parallel", 512, _fake_a2a(3, 1))]
        labels, why = b1pp._warm_seq_a2a(self.dev, self._model(sites), (1024,))
        self.assertIsNone(why)
        self.assertEqual(labels, ["K1 D=512 blocks=16 B=1024", "K2 D=512 blocks=16 div=1", "K3' D=512 perm=0",
                                  "K4 D=512 perm=0", "K1 D=128 blocks=16 B=1024", "K2 D=128 blocks=16 div=1",
                                  "K3' D=128 perm=0", "K4 D=128 perm=0"])
        self.calls.clear()
        labels, why = b1pp._warm_seq_a2a(self.dev, self._model([("table_wise", 512, _fake_a2a(0, 8))]), (1024,))
        self.assertEqual(labels, ["K3' D=512 perm=0", "K4 D=512 perm=0"])
        self.assertEqual([n for n, _ in self.calls], ["K3", "K4"])

    def test_batch_sizes_and_dedup(self):
        a2a = _fake_a2a(14, 32)
        labels, _ = b1pp._warm_seq_a2a(self.dev, self._model([("row_wise", 512, a2a)]), (1024, 512, 1024, None, 0))
        self.assertEqual([lb for lb in labels if lb.startswith("K1")],
                         ["K1 D=512 blocks=448 B=512", "K1 D=512 blocks=448 B=1024"])
        self.assertEqual(sum(1 for lb in labels if lb.startswith("K2")), 1)

    def test_switches_and_handoff(self):
        sarp, ubg = self.sarp, self.ubg
        model = self._model([("row_wise", 512, _fake_a2a(14, 16))])
        ubg._RECV_HOOK = None  # no unbucketize hand-off: the RW wait decodes without the permutation
        labels, _ = b1pp._warm_seq_a2a(self.dev, model, (1024,))
        self.assertIn("K3' D=512 perm=0", labels)
        ubg._RECV_HOOK = lambda *a: None
        sarp._state.update(fused_cfg=(True, False), recv_cfg=False)  # ..._FUSED_BWD=0, ..._FUSED_RECV=0
        labels, _ = b1pp._warm_seq_a2a(self.dev, model, (1024,))
        self.assertEqual(labels, ["K1 D=512 blocks=224 B=1024"])
        sarp._state.update(fused_cfg=(False, False), recv_cfg=False)
        labels, why = b1pp._warm_seq_a2a(self.dev, model, (1024,))
        self.assertEqual((labels, why), ([], "fused seq-a2a path not installed or switched off"))
        sarp._state.update(fused_cfg=(True, True), recv_cfg=True)
        self.assertEqual(b1pp._warm_seq_a2a(self.dev, model, ())[1], "no per-rank batch size given")
        self.assertEqual(b1pp._warm_seq_a2a(self.dev, torch.nn.Linear(2, 2), (1024,))[1],
                         "no sharded sequence all-to-all in the model")
        self.assertEqual(b1pp._warm_seq_a2a(self.dev, None, (1024,))[1],
                         "no sharded sequence all-to-all in the model")

    def test_wanted_needs_the_installed_rowperm_patch(self):
        sarp = self.sarp
        self.assertTrue(b1pp._seq_a2a_wanted())
        sarp._installed = False
        self.assertFalse(b1pp._seq_a2a_wanted())
        sarp._installed = True
        sarp._state.update(fused_cfg=(False, False), recv_cfg=False)
        self.assertFalse(b1pp._seq_a2a_wanted())

    def test_send_failure_disables_the_send_path_only(self):
        sarp = self.sarp

        def boom(*a, **k):
            raise RuntimeError("compile failed")

        with mock.patch.object(self.saf, "send_fwd", boom), self.assertLogs(
                "generative_recommenders.dlrm_v4.seq_a2a_rowperm", level="WARNING") as cm:
            labels, why = b1pp._warm_seq_a2a(self.dev, self._model([("row_wise", 512, _fake_a2a(14, 16))]), (1024,))
        self.assertTrue(sarp._state["fused_disabled"])
        self.assertFalse(sarp._state["recv_disabled"])
        self.assertEqual(labels, ["K3' D=512 perm=1", "K4 D=512 perm=1"])
        self.assertIn("fused send path DISABLED", "\n".join(cm.output))


# GPU only unless opted in (see the module docstring): skip every TestCase class.
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type) and issubclass(_obj, unittest.TestCase) and _obj is not unittest.TestCase:
        globals()[_name] = unittest.skipUnless(_RUN, _SKIP_WHY)(_obj)
del _name, _obj  # a leftover module-level TestCase binding would be collected twice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"b1pp_test: device={DEV} interpret={b1pp.interpret_mode()} real_fbgemm={_REAL_FBGEMM} "
          f"real_torchrec={_REAL_TORCHREC} torch={torch.__version__}", flush=True)
    unittest.main()
