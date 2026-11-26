"""M4-OTF: per-anchor HSTU attention length N, computed on the fly (x4c19 iter67, for comm's M4-OTF).

Tree path: generative_recommenders/dlrm_v4/train/m4_nfunc.py. numpy only (tested with numpy 1.26.4,
the ship tree's requirements.txt pin, and 2.5.3), no repo import. comm's
builder (m4_otf.py, NFUNC_MODULE / NFUNC_NAME) imports it lazily inside the per-node builder
process after run_start. There is no N table: N is computed from the pool flags of the events
before each anchor, in working memory only. Nothing is read before the first call, and nothing is
cached on disk or kept between runs.

CONTRACT (pinned by comm)
  compute_window_n_incremental(positions, user_start, flags, state, *, user_lo, user_hi)
      -> (N, state)
  compute_window_n(positions, user_start, flags, *, scan_window=20000, per_pool=1362,
                   uih_budget=4087, n_extra=9) -> N        (stateless reference, for tests only)

  positions   int64 [n]: event positions of ONE window's anchors, any order. The trained order is
              positions_L4086[ts_indices[off[ts]:off[ts+1]]].
  user_start  int64 [n_users + 1]: user u owns events [user_start[u], user_start[u+1]). The array
              must be ascending (user-major). The offline test asserts this once; this module does
              not re-check it.
  flags       (flat_is_listen_plus, flat_is_like, flat_is_skip): 1-D 1-byte 0/1 arrays or mmaps
              (bool/uint8/int8), equal length >= user_start[-1]. Only fancy-index gathers read them.
  state       None on the first call, then the object this function returned (opaque NState).
              Windows may arrive in any order, with gaps, repeats or a fresh state at any window;
              the result is the same N in every case. Ascending windows are the cheap case.
  user_lo, user_hi  users [user_lo, user_hi) covered by the state (v1: 0, n_users). Every anchor
              must belong to one of them; otherwise ValueError.
  N           int16 [n] in INPUT order, 9 <= N <= 4095 with the default parameters.
  Keyword extras: scan_window/per_pool/uih_budget/n_extra (defaults = the rule below; they are
  stored in the state) and stats (a dict the caller may pass to receive counters).

SEMANTICS (iter61 ntable PROVENANCE.md section 2; production yambda.py _history_keep_abs)
  uid_i   = searchsorted(user_start, p_i, 'right') - 1
  s_i     = max(user_start[uid_i], p_i - scan_window)
  c_k,i   = number of events e in [s_i, p_i) with flags[k][e] != 0      k = listen_plus, like, skip
  N_i     = min(uih_budget, sum_k min(c_k,i, per_pool)) + n_extra       (1362, 4087, 9)

INCREMENTAL ALGORITHM (comm DESIGN.md "STATE"; vectorized across users, no per-user Python loop)
  State per user u in [user_lo, user_hi): last_p, last_s (int64; last_p = -1 means unseen) and the 3
  UNCAPPED exact counts of [last_s, last_p) (int32; a count is at most scan_window). That is 28 B
  per user, 28 MB for 1M users. The caps are applied only when N is emitted.
  Per window: sort the anchors by p (skipped when already ascending, as every PR907 window is).
  Then process pieces of <= PIECE_ANCHORS sorted anchors in order. A user's anchors are contiguous
  after sorting, so they form one group per piece. For each group (first anchor f, last anchor l):
    incremental  if the user was seen, p_f >= last_p, s_f >= last_s, and reading the entering
                 range [last_p, p_l) plus the expiring range [last_s, s_l) is cheaper than reading
                 [s_f, p_l). Then c_i = cnt + #[last_p, p_i) - #[last_s, s_i).
    rebuild      otherwise: first sighting, p moved backward, disjoint history (s_f >= last_p),
                 or a rebuild is simply cheaper. Then c_i = #[s_i, p_i), read from [s_f, p_l).
  Both branches are exact prefix-count algebra, so the choice only changes the cost, never N.
  The group's ranges are concatenated across all groups by index arithmetic and gathered once per
  pool. The scratch is bounded by chunks of CHUNK_EVENTS events, with the running prefix carried
  across chunks. One exclusive cumsum (np.cumsum) gives every #[a, x) as a difference of two
  lookups. The state takes each group's last anchor, then N is scattered back to input order.
  Steady state (windows ascending): consecutive anchors of a user are 1-2 events apart, so each
  anchor reads about 2 flag bytes per pool. A user seen for the first time costs up to
  scan_window + its window span per pool; the first window of a fresh state costs the union of
  its users' scan ranges.

CALLER NOTES
  * Pure: no file open, no env, no global state, no threads, no madvise. Integer arithmetic only,
    deterministic, and independent of seed and rank. Scratch is bounded: per piece about 100-150 B
    per anchor (2^18 anchors), plus about 13 B per stream event in chunks of 2^20 events. On top of
    that come the int16 output, and 16 B per anchor only for unsorted input.
  * The function gathers from the caller's mmaps and never drops their pages. The builder's RSS
    therefore grows with the file-backed pages it touches. If that matters, the runtime may
    madvise(MADV_DONTNEED) its own mappings between windows, as the x4c19 prototype and tests do.
  * Exceptions are ValueError/TypeError on bad input. A state that was inside a call when an
    exception escaped is refused afterwards (start again with state=None).
"""

import operator

import numpy as np

__all__ = ["compute_window_n_incremental", "compute_window_n", "NState",
           "SCAN_WINDOW", "PER_POOL", "UIH_BUDGET", "N_EXTRA"]

SCAN_WINDOW = 20000          # dlrm_v4/utils.py scan_window
PER_POOL = 1362              # HISTORY_LENGTH 4086 // 3 (yambda.py _history_keep_abs, interleaved)
UIH_BUDGET = 4087            # MAX_SEQ_LEN 4096 - 1 candidate - 8 contextual
N_EXTRA = 9                  # 8 contextual + 1 candidate
N_POOLS = 3
# Scratch bounds. Arrays of <= 8 MB are reused by glibc's heap (its mmap threshold adapts up to
# 32 MB), whereas window-sized temporaries are fresh mmaps whose page faults dominated the cost on
# a loaded node (measured: window 40, 0.60 s at 2^20/2^22 vs 0.10 s at 2^18/2^20, sys 0.46 -> 0.01 s).
PIECE_ANCHORS = 1 << 18      # sorted anchors per piece (per-anchor scratch ~2 MB per array)
CHUNK_EVENTS = 1 << 20       # concatenated stream events per gather chunk (8 MB index + 5 MB)
_STATE_VERSION = 1


# ----------------------------------------------------------------------------------------------
# input checks (cheap: O(1) or one pass; the user-major invariant is asserted by the test)
# ----------------------------------------------------------------------------------------------
def _as_positions(positions):
    p = np.asarray(positions)
    if p.ndim != 1:
        raise ValueError(f"positions must be 1-D, got shape {p.shape}")
    if p.dtype != np.int64:
        if p.dtype.kind not in "iu":
            raise TypeError(f"positions must be integers, got {p.dtype}")
        p = p.astype(np.int64)
    return p


def _as_user_start(user_start):
    u = np.asarray(user_start)
    if u.ndim != 1 or u.shape[0] < 2:
        raise ValueError(f"user_start must be 1-D with n_users + 1 >= 2 entries, got shape {u.shape}")
    if u.dtype != np.int64:
        if u.dtype.kind not in "iu":
            raise TypeError(f"user_start must be integers, got {u.dtype}")
        u = u.astype(np.int64)
    return u


def _as_flags(flags):
    if not isinstance(flags, (tuple, list)) or len(flags) != N_POOLS:
        raise ValueError("flags must be a tuple (flat_is_listen_plus, flat_is_like, flat_is_skip)")
    out = []
    for f in flags:
        a = np.asarray(f)
        if a.ndim != 1 or a.dtype.itemsize != 1 or a.dtype.kind not in "biu":
            raise ValueError(f"pool flags must be 1-D 1-byte arrays, got {a.dtype} ndim={a.ndim}")
        out.append(a.view(np.uint8))
    if any(a.shape[0] != out[0].shape[0] for a in out):
        raise ValueError("pool flag columns differ in length")
    return tuple(out)


def _params(scan_window, per_pool, uih_budget, n_extra):
    w, pp, ub, ne = (operator.index(x) for x in (scan_window, per_pool, uih_budget, n_extra))
    if min(w, pp, ub, ne) < 0:
        raise ValueError("scan_window, per_pool, uih_budget and n_extra must be >= 0")
    if w >= 1 << 29:
        raise ValueError("scan_window must be < 2**29 (int32 counts)")
    if ub + ne > np.iinfo(np.int16).max:
        raise ValueError("uih_budget + n_extra must fit int16")
    return (w, pp, ub, ne)


def _is_ascending(p):
    return p.shape[0] < 2 or bool(np.all(p[1:] >= p[:-1]))


# ----------------------------------------------------------------------------------------------
# shared kernel: prefix counts over a concatenation of event ranges
# ----------------------------------------------------------------------------------------------
def _stream_prefix(flags, lo, ln, queries, chunk_events, stats=None):
    """Concatenate the event ranges [lo[k], lo[k] + ln[k]) (k in order) into one stream S.
    For each pool f and each query array q (int64, ascending, values in [0, len(S)]), return
    #flagged events of S[0:q] as int64 (the exclusive prefix of S at q). The stream is gathered
    and cumsummed in chunks of <= chunk_events with the prefix carried across chunks, so scratch
    is O(chunk_events) whatever len(S) is."""
    keep = np.flatnonzero(ln > 0)
    if keep.shape[0] == 0:                     # empty stream: every prefix is 0
        return [[np.zeros(q.shape[0], np.int64) for q in queries] for _ in flags], 0
    # every query lies in [0, total] and is written by exactly one chunk below
    res = [[np.empty(q.shape[0], np.int64) for q in queries] for _ in flags]
    plo = lo[keep]
    pln = ln[keep]
    pend = np.cumsum(pln)                      # stream end of each nonempty range
    total = int(pend[-1])
    poff = pend - pln                          # stream start of each nonempty range
    delta = plo - poff                         # event = stream position + delta
    cap = min(int(chunk_events), total)
    idx = np.empty(cap, np.int64)
    buf = np.empty(cap, np.uint8)
    cs = np.empty(cap + 1, np.int32)           # chunk-local exclusive prefix (<= chunk_events)
    carry = [0] * len(flags)
    nchunks = 0
    for c0 in range(0, total, cap):
        c1 = min(total, c0 + cap)
        k = c1 - c0
        last = c1 == total
        k0 = int(np.searchsorted(pend, c0, side="right"))   # first range ending after c0
        k1 = int(np.searchsorted(poff, c1, side="left"))    # first range starting at/after c1
        # stream -> event index for [c0, c1): steps of 1 inside a range, a jump at each range start
        ix = idx[:k]
        ix.fill(1)
        starts = poff[k0 + 1:k1] - c0
        ix[starts] = delta[k0 + 1:k1] - delta[k0:k1 - 1] + 1
        ix[0] = c0 + delta[k0]
        np.cumsum(ix, out=ix)
        sel = []
        for q in queries:
            i0 = int(np.searchsorted(q, c0, side="left"))
            i1 = q.shape[0] if last else int(np.searchsorted(q, c1, side="left"))
            sel.append((i0, i1, q[i0:i1] - c0 if i1 > i0 else None))
        b = buf[:k]
        c = cs[:k + 1]
        c[0] = 0
        for f, fl in enumerate(flags):
            np.take(fl, ix, out=b, mode="clip")    # ix is in range by construction
            np.cumsum(b, dtype=np.int32, out=c[1:])
            for (i0, i1, rel), out in zip(sel, res[f]):
                if rel is not None:
                    v = out[i0:i1]
                    v[...] = c[rel]                 # int32 gather widened on assignment (numpy
                    v += carry[f]                   # 1.26 refuses np.take(int32, out=int64))
            carry[f] += int(c[k])
        nchunks += 1
    if stats is not None:
        stats["chunks"] = stats.get("chunks", 0) + nchunks
    return res, total


def _n_from_counts(counts_iter, m, params):
    _, per_pool, budget, extra = params
    uih = np.zeros(m, np.int32)
    for c in counts_iter:
        np.minimum(c, per_pool, out=c)
        uih += c
    np.minimum(uih, budget, out=uih)
    uih += extra
    return uih.astype(np.int16)


# ----------------------------------------------------------------------------------------------
# incremental entry point
# ----------------------------------------------------------------------------------------------
class NState:
    """Opaque per-user running state of compute_window_n_incremental (one run, one builder)."""

    __slots__ = ("version", "user_lo", "user_hi", "n_users", "n_events", "guard", "params",
                 "last_p", "last_s", "cnt", "calls", "busy")

    def __init__(self, user_lo, user_hi, user_start, n_events, params):
        n = user_hi - user_lo
        self.version = _STATE_VERSION
        self.user_lo = user_lo
        self.user_hi = user_hi
        self.n_users = int(user_start.shape[0]) - 1
        self.n_events = int(n_events)
        self.guard = (int(user_start[0]), int(user_start[user_lo]), int(user_start[user_hi]),
                      int(user_start[-1]))
        self.params = params
        self.last_p = np.full(n, -1, np.int64)         # -1: user not seen yet
        self.last_s = np.zeros(n, np.int64)
        self.cnt = np.zeros((N_POOLS, n), np.int32)    # exact uncapped counts of [last_s, last_p)
        self.calls = 0
        self.busy = False

    @property
    def nbytes(self):
        return int(self.last_p.nbytes + self.last_s.nbytes + self.cnt.nbytes)

    def _check(self, user_lo, user_hi, user_start, n_events, params):
        if self.version != _STATE_VERSION:
            raise ValueError("state from another m4_nfunc version")
        if self.busy:
            raise ValueError("state was left inside a failed call; restart with state=None")
        got = (user_lo, user_hi, int(user_start.shape[0]) - 1, int(n_events), params,
               (int(user_start[0]), int(user_start[user_lo]), int(user_start[user_hi]), int(user_start[-1])))
        want = (self.user_lo, self.user_hi, self.n_users, self.n_events, self.params, self.guard)
        if got != want:
            raise ValueError(f"state does not match this call (user range / user_start / flags / parameters): "
                             f"state {want} vs call {got}")


def _uid_sorted(ps, ust):
    """uid = searchsorted(ust, ps, 'right') - 1 for ascending ps (int64). Uses the cheaper of a
    direct search (few anchors over many users) and a user-boundary merge (many anchors)."""
    m = ps.shape[0]
    u0 = int(np.searchsorted(ust, ps[0], side="right")) - 1
    u1 = int(np.searchsorted(ust, ps[m - 1], side="right")) - 1
    if u1 == u0:
        return np.full(m, u0, np.int64)
    if u1 - u0 >= m:
        uid = np.searchsorted(ust[u0:u1 + 1], ps, side="right")
        uid += u0 - 1
        return uid.astype(np.int64, copy=False)
    bnd = np.searchsorted(ps, ust[u0 + 1:u1 + 1], side="left")      # first anchor of users u0+1..u1
    uid = np.cumsum(np.bincount(bnd, minlength=m)[:m])
    uid += u0
    return uid.astype(np.int64, copy=False)


def _piece_incremental(ps, ust, fl, st, params, chunk_events, stats):
    w = params[0]
    m = ps.shape[0]
    uid = _uid_sorted(ps, ust)
    s = ust[uid]
    np.maximum(s, ps - w, out=s)                       # s_i = max(user start, p_i - scan_window)
    chg = np.flatnonzero(uid[1:] != uid[:-1])
    chg += 1
    gf = np.concatenate((np.zeros(1, np.int64), chg))  # first anchor of each user group
    gl = np.concatenate((chg - 1, np.full(1, m - 1, np.int64)))  # last anchor
    gsz = gl - gf + 1
    gu = uid[gf] - st.user_lo                          # state slot of each group
    del uid, chg
    ng = gf.shape[0]
    pf = ps[gf]
    sf = s[gf]
    pl = ps[gl]
    sl = s[gl]
    sp = st.last_p[gu]
    ss = st.last_s[gu]
    inc = sp >= 0
    inc &= pf >= sp
    inc &= sf >= ss
    inc &= (pl - sp) + (sl - ss) < (pl - sf)           # cheaper than a rebuild (exact either way)
    # two stream slots per group: slot 1 = expiring [last_s, s_l) or rebuild [s_f, p_l);
    # slot 2 = entering [last_p, p_l) or empty
    lo1 = np.where(inc, ss, sf)
    hi1 = np.where(inc, sl, pl)
    lo2 = np.where(inc, sp, pl)
    lo = np.empty(2 * ng, np.int64)
    ln = np.empty(2 * ng, np.int64)
    lo[0::2] = lo1
    lo[1::2] = lo2
    ln[0::2] = hi1 - lo1
    ln[1::2] = pl - lo2
    off = np.empty(2 * ng + 1, np.int64)
    off[0] = 0
    np.cumsum(ln, out=off[1:])
    off1 = off[0:2 * ng:2]
    off2 = off[1:2 * ng:2]
    # per-anchor stream offsets: c = base + G[plus] - G[minus]
    plus = np.repeat(np.where(inc, off2 - lo2, off1 - lo1), gsz)
    plus += ps
    minus = np.repeat(off1 - lo1, gsz)
    minus += s
    inc_idx = np.flatnonzero(inc)
    queries = (plus, minus, off1[inc_idx], off2[inc_idx])
    pref, total = _stream_prefix(fl, lo, ln, queries, chunk_events, stats)
    del plus, minus
    span = ps - s
    new_cnt = np.empty((N_POOLS, ng), np.int32)
    counts = []
    gu_inc = gu[inc_idx]
    for k in range(N_POOLS):
        gp, gm, g1, g2 = pref[k]
        base = np.zeros(ng, np.int64)
        base[inc_idx] = st.cnt[k, gu_inc].astype(np.int64) + g1 - g2
        c = np.repeat(base, gsz)
        c += gp
        c -= gm
        if m and (int(c.min()) < 0 or bool(np.any(c > span))):
            raise ValueError("inconsistent counts: the state does not match user_start/flags")
        new_cnt[k] = c[gl]
        counts.append(c)
        pref[k] = None
    out = _n_from_counts(counts, m, params)
    st.last_p[gu] = pl
    st.last_s[gu] = sl
    st.cnt[:, gu] = new_cnt
    if stats is not None:
        n_inc = int(inc_idx.shape[0])
        ev_inc = int(ln[0::2][inc_idx].sum() + ln[1::2][inc_idx].sum()) if n_inc else 0
        stats["pieces"] = stats.get("pieces", 0) + 1
        stats["groups"] = stats.get("groups", 0) + ng
        stats["groups_incremental"] = stats.get("groups_incremental", 0) + n_inc
        stats["groups_rebuild"] = stats.get("groups_rebuild", 0) + ng - n_inc
        stats["stream_events"] = stats.get("stream_events", 0) + total
        stats["stream_events_incremental"] = stats.get("stream_events_incremental", 0) + ev_inc
        stats["stream_events_rebuild"] = stats.get("stream_events_rebuild", 0) + total - ev_inc
    return out


def compute_window_n_incremental(positions, user_start, flags, state, *, user_lo, user_hi,
                                 scan_window=SCAN_WINDOW, per_pool=PER_POOL, uih_budget=UIH_BUDGET,
                                 n_extra=N_EXTRA, stats=None, piece_anchors=PIECE_ANCHORS,
                                 chunk_events=CHUNK_EVENTS):
    """N (int16, input order) for one window's anchor positions, plus the updated state.
    See the module docstring for the contract and the algorithm."""
    p = _as_positions(positions)
    ust = _as_user_start(user_start)
    fl = _as_flags(flags)
    params = _params(scan_window, per_pool, uih_budget, n_extra)
    piece_anchors = operator.index(piece_anchors)
    chunk_events = operator.index(chunk_events)
    if piece_anchors < 1 or chunk_events < 1:
        raise ValueError("piece_anchors and chunk_events must be >= 1")
    n_users = ust.shape[0] - 1
    user_lo = operator.index(user_lo)
    user_hi = operator.index(user_hi)
    if not 0 <= user_lo < user_hi <= n_users:
        raise ValueError(f"need 0 <= user_lo < user_hi <= n_users={n_users}, got {user_lo}, {user_hi}")
    n_events = fl[0].shape[0]
    if int(ust[0]) < 0 or int(ust[-1]) > n_events:
        raise ValueError("user_start does not lie inside the flag columns")
    if state is None:
        state = NState(user_lo, user_hi, ust, n_events, params)
    elif not isinstance(state, NState):
        raise TypeError(f"state must be None or the NState this function returned, got {type(state).__name__}")
    else:
        state._check(user_lo, user_hi, ust, n_events, params)
    n = p.shape[0]
    out = np.empty(n, np.int16)
    if stats is not None:
        stats["anchors"] = stats.get("anchors", 0) + n
    if n == 0:
        state.calls += 1
        return out, state
    if _is_ascending(p):
        order = None
        ps = p
    else:
        order = np.argsort(p, kind="stable")
        ps = p[order]
        if stats is not None:
            stats["unsorted_input"] = stats.get("unsorted_input", 0) + 1
    if int(ps[0]) < int(ust[user_lo]) or int(ps[n - 1]) >= int(ust[user_hi]):
        raise ValueError(f"anchor position outside the events of users [{user_lo}, {user_hi}): "
                         f"[{int(ps[0])}, {int(ps[n - 1])}] vs [{int(ust[user_lo])}, {int(ust[user_hi])})")
    state.busy = True
    for a in range(0, n, piece_anchors):
        b = min(n, a + piece_anchors)
        r = _piece_incremental(ps[a:b], ust, fl, state, params, chunk_events, stats)
        if order is None:
            out[a:b] = r
        else:
            out[order[a:b]] = r
    state.busy = False
    state.calls += 1
    return out, state


# ----------------------------------------------------------------------------------------------
# stateless reference (tests only): direct counting over the merged union of scan ranges
# ----------------------------------------------------------------------------------------------
def compute_window_n(positions, user_start, flags, *, scan_window=SCAN_WINDOW, per_pool=PER_POOL,
                     uih_budget=UIH_BUDGET, n_extra=N_EXTRA, piece_anchors=PIECE_ANCHORS,
                     chunk_events=CHUNK_EVENTS):
    """N (int16, input order) for any anchor positions, with no state. Reference for tests: its
    cost is the union of the anchors' scan ranges (like the prototype's strategy A)."""
    p = _as_positions(positions)
    ust = _as_user_start(user_start)
    fl = _as_flags(flags)
    params = _params(scan_window, per_pool, uih_budget, n_extra)
    w = params[0]
    n = p.shape[0]
    out = np.empty(n, np.int16)
    if n == 0:
        return out
    if int(ust[0]) < 0 or int(ust[-1]) > fl[0].shape[0]:
        raise ValueError("user_start does not lie inside the flag columns")
    order = np.argsort(p, kind="stable")
    ps = p[order]
    uid = np.searchsorted(ust, ps, side="right") - 1
    if int(uid[0]) < 0 or int(uid[-1]) >= ust.shape[0] - 1:
        raise ValueError("anchor position outside [user_start[0], user_start[-1])")
    s = np.maximum(ust[uid], ps - w)
    del uid
    for a in range(0, n, piece_anchors):
        b = min(n, a + piece_anchors)
        pp = ps[a:b]
        sp = s[a:b]
        m = b - a
        # sorted anchors: s is nondecreasing, so the union of [s_i, p_i) is a set of segments;
        # a new segment starts where s_i > p_{i-1}
        newseg = np.empty(m, bool)
        newseg[0] = True
        np.greater(sp[1:], pp[:-1], out=newseg[1:])
        first = np.flatnonzero(newseg)
        last = np.empty_like(first)
        last[:-1] = first[1:] - 1
        last[-1] = m - 1
        seg_lo = sp[first]
        seg_ln = pp[last] - seg_lo
        seg_off = np.cumsum(seg_ln) - seg_ln
        shift = np.repeat(seg_off - seg_lo, last - first + 1)
        plus = pp + shift
        minus = sp + shift
        del shift
        pref, _ = _stream_prefix(fl, seg_lo, seg_ln, (plus, minus), chunk_events)
        counts = []
        for k in range(N_POOLS):
            gp, gm = pref[k]
            gp -= gm
            counts.append(gp)
            pref[k] = None
        out[order[a:b]] = _n_from_counts(counts, m, params)
    return out
