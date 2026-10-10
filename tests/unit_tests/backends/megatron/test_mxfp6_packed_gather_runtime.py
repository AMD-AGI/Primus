###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""mxfp6_packed_param_gather on one GPU: every rank's owner pack written into one set of planes (what the all-gather
produces); the views a weight then carries must equal the per-rank packs of the full weight.

* forward rows (W6: C0 / C1 planes + slab; W4: FP4 rows + slab): bitwise;
* the dgrad copy, round to nearest: bitwise against the per-rank pack's column direction re-laid K256-outer (codes
  [K/256, N, 128], slab [J, wi, 1 KiB]), the layout aiter's "kouter" kernels read;
* the dgrad copy with SR: reproducible for (step, receiver), different across receivers and steps;
* ``mxfp6_packed_param_gather_prob_bits``: floor codes + probabilities equal the full pack's, and each received code is
  its floor or one above, only where the probability is nonzero;
* ``mxfp6_packed_param_gather_neutral``: "sr" / "rn" gather a rows-only neutral pack; "d2" gathers the forward rows as
  packed, and the receiver un-rotates them before the round-to-nearest dgrad copy.
"""

import types

import pytest
import torch


def _gfx950():
    try:
        return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gfx950(), reason="the MXFP6 tilescale packs need gfx950")

DEV = "cuda"
DP = 8
BIG = ((12288, 3072), (3072, 12288), (9216, 3072))
REP = ((3072, 3072),) * 5 + ((12288, 3072),) * 3 + ((3072, 12288),) * 2  # weights cut by shard boundaries


@pytest.fixture(autouse=True)
def _reset_gates():
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates

    yield
    mxfp6_gates.reset(mxfp6_gates.Mxfp6Gates())


def _gates(**kw):
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates

    mxfp6_gates.reset(mxfp6_gates.Mxfp6Gates(gemm_layout="tilescale", bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, **kw))


def _bucket(shapes, kind):
    from primus.backends.megatron.patches.packed_param_gather_patches import plan_weight_bucket

    S, starts, _ = plan_weight_bucket(shapes, DP)
    data = torch.zeros(DP * S, dtype=torch.bfloat16, device=DEV)
    params, p2i = [], {}
    g = torch.Generator(device=DEV).manual_seed(0)
    for (R, K), s in zip(shapes, starts):
        v = data[s : s + R * K].view(R, K)
        v.copy_(torch.randn(R, K, device=DEV, generator=g) * 0.02)
        p = torch.nn.Parameter(v, requires_grad=False)
        params.append(p)
        p2i[p] = (s, s + R * K)
    return types.SimpleNamespace(params_list=params, param_to_index=p2i, param_data=data, mxfp6_kind=kind, bucket_id=0,
                                 mxfp6_shard=S)


def _gathered(b, receiver=0, step=0):
    """Every rank's owner pack; the planes as rank ``receiver`` holds them after the collectives (the SR dgrad copy
    is all-to-all: row r of its planes = owner r's draw for the receiver)."""
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import PackedBucket

    st = PackedBucket(b, DP, 0, L.ppg_formats())
    for r in range(DP):
        st.rank, st.step = r, step
        st.owner_pack()
        for k in (("cc",) if st.sr else ()):
            st.planes[k].view(DP, -1)[r].copy_(st.send[k].view(DP, -1)[receiver])
    return st


def _kouter(cc, cs, R, K, width=128):
    """The per-rank column pack ([K, R/2] codes, standard slab of K rows) re-laid K256-outer (``width`` bytes per 256
    rows: 128 for codes and 4-bit probabilities, 64 for 2-bit probabilities)."""
    codes = cc.view(torch.uint8).reshape(K, R // 256, width).transpose(0, 1).contiguous()
    slab = cs.view(torch.uint8).reshape(K // 128, R // 256, 1024).transpose(0, 1).contiguous().reshape(-1)
    return codes, slab


def _nb(n):
    return torch.empty(n, dtype=torch.uint8, device=DEV)


@pytest.mark.parametrize("kind,sr", [("W6", False), ("W4", False), ("W6", True)])
@pytest.mark.parametrize("shapes", [BIG, REP], ids=["big", "shard_cuts"])
def test_gathered_views_equal_full_weight_packs(kind, sr, shapes):
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

    _gates(fp4_sr_actw=sr, fwd_fp4_single_fc1=(kind == "W4"))
    st = _gathered(_bucket(list(shapes), kind))
    for _idx, p, _s, R, K, _pieces in st.items:
        w = p.data.contiguous()
        if kind == "W6":
            rc, rs, cc, cs = MX.quantize_mx_dual(w, MX.with_ts6_row(L._weight_fmt(w.shape, 16384), True))
            flat = rc.view(torch.uint8).reshape(-1)
            assert torch.equal(p._ppg_row, flat[: R * K // 2]) and torch.equal(p._ppg_c1, flat[R * K // 2 :])
        else:
            rc, rs, cc, cs = MX.quantize_mx_dual(w, L._fwd_fp4_fmt("weight", 16384, R, K))
            assert torch.equal(p._ppg_row.reshape(-1), rc.view(torch.uint8).reshape(-1))
        assert torch.equal(p._ppg_row_s, rs.view(torch.uint8).reshape(-1))
        if not sr:  # SR: test_sr_dgrad_copy_draws
            kc, ks = _kouter(cc, cs, R, K)
            assert p._ppg_col.dim() == 3 and torch.equal(p._ppg_col, kc)
            assert torch.equal(p._ppg_col_s, ks)


def test_sr_dgrad_copy_draws():
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import PackedBucket

    _gates(fp4_sr_actw=True)
    b = _bucket([(3072, 3072), (3072, 3072)], "W6")
    p = b.params_list[1]

    def draw(step, receiver):
        st = PackedBucket(b, DP, 0, L.ppg_formats())
        for r in range(DP):
            st.rank, st.step = r, step - 1
            st.owner_pack()
            for k in (("cc",) if st.sr else ()):
                st.planes[k].view(DP, -1)[r].copy_(st.send[k].view(DP, -1)[receiver])
        return p._ppg_col.clone()

    a, a2, b1, c = draw(5, 0), draw(5, 0), draw(5, 1), draw(6, 0)
    assert torch.equal(a, a2) and not torch.equal(a, b1) and not torch.equal(a, c)


@pytest.mark.parametrize("kind,shapes,bits", [("W6", BIG, 4), ("W4", BIG, 4), ("W6", REP, 4), ("W6", BIG, 2),
                                              ("W4", BIG, 2), ("W6", REP, 2)])
def test_prob_bits(kind, shapes, bits):
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

    _gates(fp4_sr_actw=True, packed_param_gather_prob_bits=bits, fwd_fp4_single_fc1=(kind == "W4"))
    b = _bucket(list(shapes), kind)
    st = _gathered(b)
    assert st.prob4 and not st.sr and st.a2a_ops() == [] and any(o is st.planes["cp"] for o, _ in st.gather_ops())
    width = 128 if bits == 4 else 64
    for _idx, p, s, R, K, _pieces in st.items:
        w = p.data.contiguous()
        f = (MX.with_ts6_row(L._weight_fmt(w.shape, 16384), True) if kind == "W6"
             else L._fwd_fp4_fmt("weight", 16384, R, K)) & ~MX.MX_FMT_FP4_COL_SR
        rp, rs = MX.mx_dir_sizes(R, K, f, False)
        cp, cs = MX.mx_dir_sizes(R, K, f, True)
        o = [_nb(rp), _nb(rs), _nb(cp), _nb(cs)]
        prob = _nb(cp if bits == 4 else cp // 2)
        MX.quantize_mx_dual_out(w, *o, f, col_prob=prob)
        kc, ks = _kouter(o[2], o[3], R, K)
        kp, _ = _kouter(prob, o[3], R, K, width)
        pv = st._view("cp", s, R * K).view(R // 256, K, width)
        assert torch.equal(p._ppg_col, kc) and torch.equal(p._ppg_col_s, ks) and torch.equal(pv, kp)
    floor = {id(p): p._ppg_col.clone() for p in b.params_list}

    def nib(t):
        return torch.stack([t & 15, t >> 4], -1).reshape(-1).int()

    def probs(t):  # per code, in code order
        return nib(t) if bits == 4 else torch.stack([(t >> (2 * i)) & 3 for i in range(4)], -1).reshape(-1).int()

    def rx(step, rank):
        for p in b.params_list:
            p._ppg_col.copy_(floor[id(p)])
        st.step, st.rank = step, rank
        st.receive()
        return torch.cat([p._ppg_col.reshape(-1).clone() for p in b.params_list])

    fl = torch.cat([floor[id(p)].reshape(-1) for p in b.params_list])
    pr = torch.cat([st._view("cp", s, R * K).reshape(-1) for _i, _p, s, R, K, _ in st.items])
    a, a2, r1, s1 = rx(5, 0), rx(5, 0), rx(5, 1), rx(6, 0)
    d = (nib(a) & 7) - (nib(fl) & 7)
    assert bool(((d == 0) | ((d == 1) & (probs(pr) > 0))).all()) and bool(((nib(a) & 8) == (nib(fl) & 8)).all())
    assert torch.equal(a, a2) and not torch.equal(a, r1) and not torch.equal(a, s1)


@pytest.mark.parametrize("mode,shapes", [("sr", BIG), ("rn", BIG), ("sr", REP), ("d2", BIG), ("d2", REP)])
def test_neutral(mode, shapes):
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import _seed
    from primus_turbo.pytorch.kernels.quantization import mx_a4w4_pack as MX

    if mode == "d2":
        _gates()
        plain = L.ppg_formats()["W6"]
    _gates(fp4_sr_actw=(mode == "sr"), fp4_hadamard_dgrad="h32" if mode == "d2" else "none",
           packed_param_gather_neutral=mode)
    if mode == "d2":  # d2 gathers the forward rows in the format the plain packed gather uses
        assert all(L.ppg_formats()["W6"](R, K) == plain(R, K) for R, K in shapes)
    b = _bucket(list(shapes), "W6")
    st = _gathered(b)
    assert st.neutral == mode and not st.sr and not st.prob4 and st.a2a_ops() == []
    gathered_keys = {id(o) for o, _ in st.gather_ops()}
    assert id(st.planes["cc"]) not in gathered_keys and id(st.planes["cs"]) not in gathered_keys
    for _idx, p, _s, R, K, _pieces in st.items:
        f = L.ppg_formats()["W6"](R, K)
        rp, rs = MX.mx_dir_sizes(R, K, f, False)
        c0, c1, sc = _nb(rp * 2 // 3), _nb(rp // 3), _nb(rs)
        MX.quantize_mx_dual_out(p.data.contiguous(), c0, sc, _nb(0), _nb(0), f, row_c1=c1)
        assert torch.equal(p._ppg_row, c0) and torch.equal(p._ppg_c1, c1) and torch.equal(p._ppg_row_s, sc)

    def rx(step, rank):
        st.step, st.rank = step, rank
        st.receive()
        return [(p._ppg_col.clone(), p._ppg_col_s.clone()) for p in b.params_list]

    a, a2, r1, s1 = rx(5, 0), rx(5, 0), rx(5, 1), rx(6, 0)
    same = lambda u, v: all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(u, v))  # noqa: E731
    # the copy equals the receiver run on the gathered plane, with the bucket's seed (weight 0)
    idx, p, _s, R, K, _ = st.items[0]
    f = L.ppg_formats()["W6"](R, K)
    cp, cs = MX.mx_dir_sizes(R, K, f, True)
    gc, gs = _nb(cp), _nb(cs)
    MX.mxfp6_tile_to_fp4_col(p._ppg_row, p._ppg_c1, p._ppg_row_s, R, K, gc, gs, f, mode == "sr",
                             _seed(5, b.bucket_id, idx, "rx0"))
    assert torch.equal(a[0][0].reshape(-1), gc) and torch.equal(a[0][1], gs)
    if mode == "sr":
        assert same(a, a2) and not same(a, r1) and not same(a, s1)
    else:
        assert same(a, a2) and same(a, r1) and same(a, s1)


@pytest.mark.parametrize("mode", ["rn", "sr", "prob4", "neutral_rn"])
@pytest.mark.parametrize("kind", ["W6", "W4"])
def test_phased_transport_partitions_the_planes(kind, mode):
    """``packed_param_gather_transport: phased`` sends exactly the planes of the one-call transport, split into the
    forward GEMM's (rows and their scales) and the dgrad copy's (column codes and scales, prob4 probabilities)."""
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import _DGRAD

    if mode == "neutral_rn" and kind == "W4":
        pytest.skip("neutral applies to W6 buckets only")
    kw = {"rn": {}, "sr": dict(fp4_sr_actw=True), "prob4": dict(fp4_sr_actw=True, packed_param_gather_prob_bits=4),
          "neutral_rn": dict(fp4_hadamard_dgrad="none", packed_param_gather_neutral="rn")}[mode]
    _gates(fwd_fp4_single_fc1=(kind == "W4"), **kw)
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import PackedBucket

    st = PackedBucket(_bucket(list(BIG), kind), DP, 0, L.ppg_formats())
    every = {id(o) for o, _ in st.gather_ops()}
    fwd, bwd = ({id(o) for o, _ in st.gather_ops(ph)} for ph in ("fwd", "bwd"))
    assert fwd | bwd == every and not fwd & bwd and fwd
    names = {id(t): k for k, t in st.planes.items()}
    assert {names[i] for i in bwd} <= set(_DGRAD) and not {names[i] for i in fwd} & set(_DGRAD)


@pytest.mark.parametrize("kind,sr", [("W6", False), ("W4", False), ("W6", True)])
@pytest.mark.parametrize("shapes", [BIG, REP], ids=["big", "shard_cuts"])
def test_phased_ar_scale_sum_is_exact(kind, sr, shapes):
    """``packed_param_gather_transport: phased_ar``: each rank zeroes the shared scale buffers and packs only its own
    rows; at most one rank writes any scale byte, and the byte-wise sum over ranks (the all-reduce) equals the scale
    planes the per-plane transport gathers."""
    from primus.backends.megatron.core.extensions import primus_turbo_mxfp6_local as L
    from primus.backends.megatron.core.extensions.mxfp6_packed_gather import PackedBucket

    _gates(fp4_sr_actw=sr, fwd_fp4_single_fc1=(kind == "W4"))
    b = _bucket(list(shapes), kind)
    ref = _gathered(b)  # the per-plane transport: every owner's pack in one set of planes
    n = b.param_data.numel() // 32
    total = {k: torch.zeros(n, dtype=torch.int32, device=DEV) for k in ("rs", "cs")}
    writers = {k: torch.zeros(n, dtype=torch.int32, device=DEV) for k in ("rs", "cs")}
    for r in range(DP):
        ext = {"rs": torch.zeros(n, dtype=torch.uint8, device=DEV), "cs": torch.zeros(n, dtype=torch.uint8, device=DEV)}
        st = PackedBucket(b, DP, r, L.ppg_formats(), external=ext)
        st.step = 0
        st.owner_pack()
        for k in ("rs", "cs"):
            total[k] += ext[k].int()
            writers[k] += (ext[k] != 0).int()
    covered = torch.zeros(n, dtype=torch.bool, device=DEV)  # the weights' bytes (padding is never read)
    for _i, _p, s, R, K, _pieces in ref.items:
        covered[s // 32:(s + R * K) // 32] = True
    for k in ("rs", "cs"):
        assert int(writers[k].max()) <= 1, f"{k}: a scale byte written by two ranks"
        assert torch.equal(total[k].to(torch.uint8)[covered], ref.planes[k][covered]), k
