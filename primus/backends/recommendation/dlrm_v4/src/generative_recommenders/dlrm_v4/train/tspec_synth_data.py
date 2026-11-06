#!/usr/bin/env python3
"""Synthetic yambda-5b dataset for the TSPEC record run (no training data).

User rule 2026-10-12: no prior knowledge about the training data may be
generated before run_start or offline. The TSPEC replay table
(ops/triton/triton_spec_table.jsonl) was recorded over the real training data;
this script builds a SYNTHETIC stand-in so the record run that regenerates the
table never reads a training-data file.

Every shape parameter comes from the code / config, never from the dataset:
  * vocab sizes: generative_recommenders/dlrm_v4/configs.py (yambda-5b tables:
    item_id, artist_id, album_id, uid num_embeddings), parsed from --configs;
  * history_length = min_history = 4086 (gin get_dataset.history_length /
    $MIN_HISTORY defaults), per-pool cap 4086 // 3 = 1362, scan window 20000
    (utils.get_dataset), window 86400 s (gin streaming_window_seconds);
  * per-rank batch 1024 (gin batch_size), world size from the allocation;
  * event-type codes 0..4 and LISTEN_PLUS_THRESHOLD = 50 (datasets/yambda.py,
    preprocess_public_data.EVENT_TYPE_MAP).
Per-batch token load (2026-10-13; replaces the fill-fraction sweep, which reached
3.99M tokens and a 32 GiB dq-split workspace need per rank-batch at ws 8 and ran
the 1n record out of memory). The trainer batches each window USER-MAJOR (gin
streaming_sort_within_window False, shuffle fraction 0; the offline index is
ascending inside a window), so a global batch is a contiguous run of users and its
load is set by the per-user UIH lengths along the uid order. Each user gets a base
length L0, realised exactly by its prefix pool counts:
  * calibration ramp: the first CAL_STEPS global batches of every train window are
    the trainer's FP8 calibration steps (DLRM_HSTU_FP8_CALIBRATION_STEPS = 4 in
    train/_env_bootstrap.py) at the start of each record arm. They run the BF16
    kernels, which run in no other step. Their per-rank token targets CAL_T put a
    batch on each side of every 2 GiB storage line from 2^19 to 2^21 tokens
    (tt.pointer_range classes of the [T, 2048/1536/1024/768/512]-element
    activations), up to 0.97 x CAL_T_MAX;
  * sizing level: train batch CAL_STEPS, the first FP8 step of an arm, needs
    >= SIZE_NEED, so the persistent ds workspace is sized there once to
    ceil_GiB(1.25 need) >= NEED_LIMIT, and no later batch can regrow it;
  * after these levels, the block maximum M sweeps light -> full -> light SWEEPS times
    along the user order (triangle wave from M_LO to history, flat at full length
    near the top);
  * L0 = M * S with a low-discrepancy (golden-ratio) quantile per user, so every
    run of users has the designed mean. M <= P (= history // 3): S = U^SPREAD_K
    and all of L0 sits in the anchor pool, so later anchors lengthen it by one
    each but never past P. M > P: L0 in [P, M] with the anchor pool saturated
    (P prefix events), so every anchor of that user has exactly L0 in every
    window. The spread exponent rises where needed so the block mean m keeps
    dq_need(batch * (m + 9), Mc + 9) <= NEED_DESIGN, with Mc the largest block
    maximum within two train-batch spans of the user;
  * TOP_SHARE of the users sit exactly at M, so batches reach N = M + 9.
Bounds (memory, not data; check() fails the dataset if any is broken):
  * every train rank-batch keeps dq_need <= NEED_LIMIT = 16 GiB. The persistent ds
    workspace regrows to ceil_GiB(max(1.25 need, 1.5 old)) only when a need exceeds
    it, so it stays <= 23 GiB in any batch order. With the sizing level it is sized
    once, at <= ceil_GiB(1.25 x 16 GiB) = 20 GiB, and never regrows (gated per arm);
  * every train and eval rank-batch keeps T <= T_LIMIT, and the calibration batches
    keep T <= CAL_T_MAX. Since N >= T / batch, the need gate alone already admits T
    up to only 2.79M.
  * Basis: record 1n_70691 (this kit, synthetic data) ran BF16 calibration steps up
    to 2.48M tokens per rank (step 4), and full FP8 steps with a 25 GiB workspace at
    2.89M and 3.21M (steps 5-6). At step 7 (T = 3.47M) rank 4 ran out on duvqk
    [T, 2048] bf16 while holding that workspace, and six ranks failed the
    25 -> 38 GiB regrow.
  * A regrow briefly holds old + new (the dropped buffer was not reusable in place
    there), which is why the sizing level removes every regrow after the first.
Anchors of a user all go to one listen pool (listen+ for half of the users), so
labels are about half positive. Other choices are generic:
item ids from a bounded power law (exponent 1.1, so hot rows exist) plus a
per-user favourites set, uniform artist/album mappings.

Output layout (what get_dataset("yambda-5b") reads under $DLRM_DATA_PATH, same
file set as the MLCommons reference cache manifest; see dl_prestart._REF_*):
  <root>/processed_5b/item_popularity.npy
  <root>/processed_5b/hstu_cache_L4086/{flat_*.npy,user_start.npy,user_end.npy,
      unique_uids.npy,positions_L4086.npy,anchor_ts_L4086.npy,store_meta.json,_READY}
  <root>/shared_metadata/{artist,album}_item_mapping.parquet
  <root>/_SYNTH_SPEC.json   (parameters + generator sha256; "synthetic": true)
Column dtypes and derivations follow DLRMv4YambdaDataset._ensure_cache_built /
_ensure_positions_built exactly; --check re-derives them with the same formulas.

Windows: --layout "T:8,E:1,T:2,E:1" = windows 0-7 train, 8 eval, 9-10 train,
11 eval. A train window holds train_batches * batch * world_size anchors plus
half a global batch (the per-window cap NUM_TRAIN_BATCHES binds); an eval
window holds eval_batches * batch * world_size plus a partial (dropped) batch.
Per-window anchor totals are exact; window 0 starts exactly at t_min, so the
trainer's window grid ((t - t_min) // 86400) matches the plan.

This file ships twice with identical bytes (so the generator_sha256 recorded in the
table header matches both): as the kit's gen_synth_yambda.py and in the tree as
generative_recommenders/dlrm_v4/train/tspec_synth_data.py. --configs defaults to
../configs.py next to this file (the tree layout); the kit passes it explicitly.
A configs.py that cannot be parsed is fatal (no built-in vocab fallback).

Usage (container: numpy + polars from /scratch/pip_local_yambda.tgz):
  python3 <this file> --root DIR --world-size 8 [--configs PATH] [--check-only]
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

LISTEN, LIKE = 0, 1           # datasets/yambda.py LISTEN_TYPE / LIKE_TYPE
OTHER_TYPES = (2, 3, 4)       # dislike / unlike / undislike (no pool bit)
LP_THRESHOLD = 50             # datasets/yambda.py LISTEN_PLUS_THRESHOLD
W_SECONDS = 86400             # gin get_dataset.streaming_window_seconds
T0 = 1_700_000_064            # arbitrary epoch origin (multiple of 64)
MMAP_COLS = ("flat_uid", "flat_item_ids", "flat_timestamps", "flat_event_types",
             "flat_played_ratio", "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
             "flat_is_organic", "user_start", "user_end", "unique_uids")
# yambda-5b embedding tables whose num_embeddings are read from configs.py
VOCAB_TABLES = ("item_id", "artist_id", "album_id", "uid")
DEFAULT_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs.py")
# Per-batch token-load design; see the module docstring. Values come from the code,
# the config or the record's own memory profile, none from the dataset.
CTX_TOKENS = 9                # 8 contextual (uid + 7 YAMBDA_5B_CROSS_SPECS) + 1 candidate
DQ_HEADS = 4                  # HSTU attention heads (4 x 128)
DQ_TILE = 64 * 128            # ops/flydsl/hstu_dq_split.py TILE_BYTES
GIB = 1 << 30
NEED_LIMIT = 16 * GIB         # check() gate per train rank-batch: buffer <= 1.5x = 24 GiB
NEED_DESIGN = 15 * GIB        # profile target, below the gate by the quantile rounding
SWEEPS = 2                    # light -> full -> light sweeps along the user order
M_LO = 16                     # lightest block maximum (one 16-token unit)
PLATEAU = 1.1                 # the top 1 - 1/1.1 of each sweep sits at full length
SPREAD_K = 0.15               # default in-block spread S = U^k (mean ~0.87 of M)
TOP_SHARE = 1.0 / 64          # users exactly at the block maximum M
GOLDEN = 0.6180339887498949   # Weyl sequence step: low-discrepancy user quantiles
WEIGHT_JITTER = 0.5           # per (user, window) anchor weight in 1 +- 0.25
CAL_STEPS = 4                 # train/_env_bootstrap.py DLRM_HSTU_FP8_CALIBRATION_STEPS
CAL_T_MAX = 2_480_000         # largest BF16 calibration step that ran at 1n (70691 step 4)
T_LIMIT = 2_880_000           # per-rank tokens, 0.9 x the 3.21M FP8 step that ran at 1n
# 2 GiB lines: tokens at which a [T, row] activation reaches 2^31 bytes, for the row
# sizes of the HSTU activations (bf16 [T,2048] uvqk, bf16 [T,1536] qkv, f32 [T,512] /
# fp8 [T,2048], fp8 [T,1536], bf16 [T,512] x / u / attention out)
STORAGE_LINES = tuple(sorted((1 << 31) // rb for rb in (4096, 3072, 2048, 1536, 1024)))
# calibration targets: one below the lowest line, one between each pair of line groups
# (geometric means), one above the top line
CAL_T = (int(0.75 * STORAGE_LINES[0]),
         int((STORAGE_LINES[1] * STORAGE_LINES[2]) ** 0.5),
         int((STORAGE_LINES[3] * STORAGE_LINES[4]) ** 0.5),
         int(0.97 * CAL_T_MAX))
# first FP8 step of an arm (train batch CAL_STEPS): need >= SIZE_NEED makes the buffer
# ceil_GiB(1.25 need) >= NEED_LIMIT, so no later batch can regrow it (6% margin)
SIZE_NEED = int(1.06 * NEED_LIMIT / 1.25)


def log(msg):
    print(f"[synth] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def parse_vocab(configs_path):
    out = {}
    try:
        src = open(configs_path).read()
        blk = src[src.index('elif "yambda" in dataset'):]
        blk = blk[:blk.index("YAMBDA_5B_CROSS_SPECS:")]
        for name in VOCAB_TABLES:
            m = re.search(r'"%s": EmbeddingConfig\(\s*num_embeddings=([\d_]+)' % name, blk)
            if not m:
                raise ValueError(f"{name} not found")
            out[name] = int(m.group(1).replace("_", ""))
    except Exception as ex:  # noqa: BLE001
        log(f"FATAL cannot read the yambda-5b vocab sizes from {configs_path!r}: {ex}")
        raise SystemExit(2)
    return out, configs_path


def parse_layout(s):
    roles = []
    for part in s.split(","):
        r, n = part.split(":")
        if r not in ("T", "E"):
            raise ValueError(f"bad layout role {r!r}")
        roles += [r] * int(n)
    return roles


def window_targets(roles, ws, batch, train_batches, eval_batches):
    gb = ws * batch
    out = []
    for r in roles:
        if r == "T":
            out.append(train_batches * gb + gb // 2)
        else:
            out.append(eval_batches * gb + gb // 2 + 7 * ws)  # ragged tail, dropped
    return out


def dq_need(T, N, batch):
    """ds workspace bytes of hstu_dq_split.build_workspace for a batch of `batch`
    sequences, T tokens in total and max length N (works on int64 arrays)."""
    hdr = ((batch + 1) * 4 + 255) // 256 * 256
    return hdr + DQ_HEADS * ((N + 127) // 128 + 1) * ((T + 127 * batch) // 128) * DQ_TILE


def mean_cap(M, batch):
    """Largest mean UIH length m of a block whose maximum is M such that
    dq_need(batch * (m + CTX_TOKENS), M + CTX_TOKENS) <= NEED_DESIGN."""
    N = int(M) + CTX_TOKENS
    hdr = ((batch + 1) * 4 + 255) // 256 * 256
    sum_nb = (NEED_DESIGN - hdr) // (DQ_HEADS * ((N + 127) // 128 + 1) * DQ_TILE)
    return (sum_nb * 128 - 127 * batch) / batch - CTX_TOKENS


def top_mean(k):
    """E[min(1, ((1 + TOP_SHARE) U)^k)] for U ~ uniform(0, 1)."""
    d = TOP_SHARE
    return (1.0 / (1.0 + d)) / (k + 1.0) + d / (1.0 + d)


def level_mean_max(m, history):
    """Block maximum M whose default-spread mean is m (light: m = M * e; saturated:
    m = P + (M - P) * e, e = top_mean(SPREAD_K))."""
    P, e = history // 3, top_mean(SPREAD_K)
    return int(round(min(history, m / e if m <= P * e else P + (m - P) / e)))


def level_max(history, batch):
    """Block maxima of the leading levels, one train-batch span each: the CAL_STEPS
    calibration levels (per-rank tokens CAL_T), then the sizing level for the first
    FP8 step (the smallest M from the top calibration level up whose default-spread
    batch needs >= SIZE_NEED, so the ds workspace is sized there once)."""
    P, e = history // 3, top_mean(SPREAD_K)
    out = [level_mean_max(t / batch - CTX_TOKENS, history) for t in CAL_T]
    M = out[-1]
    while M < history and dq_need(batch * (P + (M - P) * e + CTX_TOKENS), M + CTX_TOKENS,
                                  batch) < SIZE_NEED:
        M += 1
    out.append(M)
    if out != sorted(out):
        raise ValueError(f"leading levels must rise: {out}")
    return out


def block_max(x, history, x_lv, lv):
    """Block maximum UIH length at user-order position x in [0, 1): the leading
    levels on [0, x_lv), then the sweep on [x_lv, 1)."""
    x = np.asarray(x, dtype=np.float64)
    ph = (np.clip((x - x_lv) / (1.0 - x_lv), 0.0, None) * SWEEPS) % 1.0
    tri = np.where(ph < 0.5, 2 * ph, 2 * (1 - ph))
    sweep = M_LO + (history - M_LO) * np.minimum(1.0, tri * PLATEAU)
    k = np.clip((x / x_lv * len(lv)).astype(np.int64), 0, len(lv) - 1)
    return np.where(x < x_lv, np.asarray(lv, dtype=np.float64)[k], sweep)


def near_max(x, history, dx, x_lv, lv):
    """Largest block maximum within [x - dx, x + dx]. The leading levels rise and the
    sweep is piecewise linear, so the maximum sits at an end of the interval, at the
    last leading-level point inside it, or at a sweep peak."""
    lo, hi = np.clip(x - dx, 0.0, 1.0 - 1e-12), np.clip(x + dx, 0.0, 1.0 - 1e-12)
    m = np.maximum(block_max(lo, history, x_lv, lv), block_max(hi, history, x_lv, lv))
    m = np.where(lo < x_lv, np.maximum(m, block_max(np.minimum(hi, x_lv * (1 - 1e-12)),
                                                    history, x_lv, lv)), m)
    for j in range(SWEEPS):
        xp = x_lv + (j + 0.5) / SWEEPS * (1.0 - x_lv)
        m = np.where((lo <= xp) & (xp <= hi), float(history), m)
    return np.rint(m).astype(np.int64)


def base_lengths(n_users, history, batch, train_batches):
    """Per-user base UIH length L0 (see the module docstring). A train batch spans
    1 / (train_batches + 1/2) of the user order (more where anchor weights are low),
    so each user's mean is capped by the largest block maximum within two spans:
    a batch that mixes users of a lower and a higher M stays inside the budget of
    its own maximum. The first CAL_STEPS + 1 spans are the leading levels."""
    P = history // 3
    lv = level_max(history, batch)
    x_lv = len(lv) / (train_batches + 0.5)
    x = (np.arange(n_users) + 0.5) / n_users
    M = np.rint(block_max(x, history, x_lv, lv)).astype(np.int64)
    Mc = near_max(x, history, 2.0 / (train_batches + 0.5), x_lv, lv)
    q = (np.arange(n_users) * GOLDEN + 0.5) % 1.0
    qq = np.minimum(1.0, (1.0 + TOP_SHARE) * q)
    L0 = np.empty(n_users, np.int64)
    lo = M <= P
    L0[lo] = np.maximum(1, np.rint(M[lo] * qq[lo] ** SPREAD_K))
    k_of = {}
    for i in np.flatnonzero(~lo):
        m, mc = int(M[i]), int(Mc[i])
        if (m, mc) not in k_of:
            sb = P / m
            e_t = (min(m, mean_cap(mc, batch)) / m - sb) / (1.0 - sb)
            if e_t <= TOP_SHARE / (1.0 + TOP_SHARE):
                raise ValueError(f"NEED_DESIGN too small for block maximum {mc}")
            k_of[(m, mc)] = SPREAD_K if top_mean(SPREAD_K) <= e_t else \
                1.0 / ((1.0 + TOP_SHARE) * e_t - TOP_SHARE) - 1.0
        sb = P / m
        L0[i] = int(np.rint(m * (sb + (1.0 - sb) * qq[i] ** k_of[(m, mc)])))
    return np.clip(L0, 1, history), M


class Plan:
    def __init__(self, a):
        self.a = a
        self.vocab, self.vocab_src = parse_vocab(a.configs)
        self.roles = parse_layout(a.layout)
        self.nwin = len(self.roles)
        self.A = window_targets(self.roles, a.world_size, a.batch, a.train_batches,
                                a.eval_batches)
        U = a.users
        if U > self.vocab["uid"] - 1:
            raise ValueError("more users than the uid table")
        g = np.random.default_rng([a.seed, 0])
        self.uids = np.sort(g.choice(self.vocab["uid"] - 1, size=U, replace=False) + 1
                            ).astype(np.int64)
        # c[u, w]: exact per-window totals, split in proportion to a per (user,
        # window) weight 1 +- WEIGHT_JITTER / 2 (block boundaries move a little
        # from window to window), largest remainder rounding.
        gw = np.random.default_rng([a.seed, 3])
        self.c = np.zeros((U, self.nwin), dtype=np.int64)
        for w, A in enumerate(self.A):
            wgt = 1.0 + WEIGHT_JITTER * (gw.random(U) - 0.5)
            raw = A * wgt / wgt.sum()
            base = np.floor(raw).astype(np.int64)
            rem = int(A - base.sum())
            base[np.argsort(-(raw - base), kind="stable")[:rem]] += 1
            self.c[:, w] = base
        if self.c[:, 0].sum() < 1:
            raise ValueError("window 0 has no anchor")
        self.pin_user = int(np.argmax(self.c[:, 0] > 0))   # holds the t_min anchor
        self.V = self.vocab["item_id"]
        self.perm = g.permutation(self.V).astype(np.int64)  # power-law rank -> item id
        self.H = a.history
        self.P = a.history // 3
        if a.history != 3 * self.P:
            raise ValueError("history must be a multiple of 3 (three equal pools)")
        # per-user base length and prefix pool counts: the anchor pool (listen+ or
        # skip) takes min(P, L0); the rest is split between like and the other
        # listen pool, each <= P
        self.L0, self.M = base_lengths(U, self.H, a.batch, a.train_batches)
        gp = np.random.default_rng([a.seed, 4])
        self.lp_anchor = gp.random(U) < 0.5
        self.nA = np.minimum(self.P, self.L0)
        rest = self.L0 - self.nA
        nB = np.rint(rest * gp.uniform(0.3, 0.7, U)).astype(np.int64)
        nB = np.clip(nB, rest - self.P, self.P)
        self.nB, self.nC = nB, rest - nB

    def user_events(self, ui):
        """Events of user ui: H prefix events + one event per anchor."""
        return self.H + int(self.c[ui].sum())

    def zipf_items(self, rng, n, s=1.1):
        u = rng.random(n)
        V = float(self.V)
        x = (1.0 - u * (1.0 - V ** (1.0 - s))) ** (1.0 / (1.0 - s))
        r = np.minimum(np.floor(x).astype(np.int64) - 1, self.V - 1)
        return self.perm[np.maximum(r, 0)]

    def gen_user(self, ui):
        rng = np.random.default_rng([self.a.seed, 1, ui])
        H = self.H
        a_u = int(self.c[ui].sum())
        n = H + a_u
        lp_anchor = bool(self.lp_anchor[ui])
        # prefix: H events (offsets 0..H-1, never anchors) with the exact pool
        # counts of this user, shuffled. labels: 0 listen+, 1 like, 2 skip, 3 other
        pa, pc = (0, 2) if lp_anchor else (2, 0)
        nA, nB, nC = int(self.nA[ui]), int(self.nB[ui]), int(self.nC[ui])
        lab = np.concatenate([np.full(nA, pa, np.int8), np.full(nB, 1, np.int8),
                              np.full(nC, pc, np.int8),
                              np.full(H - nA - nB - nC, 3, np.int8)])
        rng.shuffle(lab)
        # region: a_u anchors, all in the anchor pool
        et = np.empty(n, np.int64)
        listen_plus = np.zeros(n, bool)
        et[:H] = np.where(lab == 1, LIKE, LISTEN)
        other_pre = lab == 3
        et[:H][other_pre] = rng.choice(OTHER_TYPES, size=int(other_pre.sum()))
        listen_plus[:H] = lab == 0
        et[H:] = LISTEN
        listen_plus[H:] = lp_anchor
        is_listen = et == LISTEN
        ratio = np.zeros(n, np.float32)
        n_lp = int((is_listen & listen_plus).sum())
        n_sk = int((is_listen & ~listen_plus).sum())
        ratio[is_listen & listen_plus] = rng.integers(LP_THRESHOLD, 101, n_lp)
        ratio[is_listen & ~listen_plus] = rng.integers(0, LP_THRESHOLD, n_sk)
        # timestamps: prefix before T0; anchors split over windows by c[ui]
        ts = np.empty(n, np.int64)
        ts[:H] = np.sort(T0 - rng.integers(3600, 60 * W_SECONDS, H))
        win = np.searchsorted(np.cumsum(self.c[ui]), np.arange(a_u), side="right")
        off = rng.integers(60, W_SECONDS, a_u)
        ts[H:] = np.sort(T0 + win * W_SECONDS + off)        # win is non-decreasing
        if ui == self.pin_user:                             # pin t_min = T0
            ts[H] = T0
        # items: favourites + global power law
        fav = self.zipf_items(rng, 64)
        items = self.zipf_items(rng, n)
        pick = rng.random(n) < 0.35
        items[pick] = fav[rng.integers(0, 64, int(pick.sum()))]
        organic = (rng.random(n) < 0.6).astype(np.int8)
        return et, ratio, ts, items, organic


def spec_of(a, plan, gen_sha):
    return {"synthetic": True, "generator_sha256": gen_sha, "seed": a.seed,
            "users": a.users, "world_size": a.world_size, "batch": a.batch,
            "train_batches": a.train_batches, "eval_batches": a.eval_batches,
            "layout": a.layout, "history": a.history, "vocab": plan.vocab,
            "vocab_src": plan.vocab_src, "window_anchors": plan.A,
            "t0": T0, "window_seconds": W_SECONDS,
            "load": {"need_limit_gib": NEED_LIMIT / GIB, "need_design_gib": NEED_DESIGN / GIB,
                     "sweeps": SWEEPS, "m_lo": M_LO, "plateau": PLATEAU,
                     "spread_k": SPREAD_K, "top_share": TOP_SHARE,
                     "weight_jitter": WEIGHT_JITTER, "cal_steps": CAL_STEPS,
                     "cal_t": list(CAL_T), "cal_t_max": CAL_T_MAX, "t_limit": T_LIMIT,
                     "size_need_gib": SIZE_NEED / GIB,
                     "levels": level_max(plan.H, a.batch)}}


def save_parquet(path, cols):
    try:
        import polars as pl
        pl.DataFrame({k: pl.Series(k, v, dtype=pl.UInt32) for k, v in cols.items()}
                     ).write_parquet(path)
        return "polars"
    except ImportError:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({k: pa.array(v, pa.uint32()) for k, v in cols.items()}), path)
        return "pyarrow"


def generate(a, root):
    plan = Plan(a)
    gen_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    spec = spec_of(a, plan, gen_sha)
    spec_path = os.path.join(root, "_SYNTH_SPEC.json")
    cache = os.path.join(root, "processed_5b", f"hstu_cache_L{a.history}")
    if os.path.exists(os.path.join(cache, "_READY")) and os.path.exists(spec_path):
        try:
            if json.load(open(spec_path)) == spec:
                log(f"up to date: {root} (spec matches); skipping generation")
                return plan, spec
        except ValueError:
            pass
    tmp = root.rstrip("/") + f".tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tcache = os.path.join(tmp, "processed_5b", f"hstu_cache_L{a.history}")
    os.makedirs(tcache)
    os.makedirs(os.path.join(tmp, "shared_metadata"))
    U = a.users
    log(f"plan: users={U} windows={plan.nwin} roles={''.join(plan.roles)} "
        f"anchors/window={plan.A} vocab={plan.vocab} ({plan.vocab_src})")
    # pass 1: sizes
    lens = np.array([plan.user_events(ui) for ui in range(U)], np.int64)
    N = int(lens.sum())
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    log(f"pass 1: total_events={N:,} anchors={int(plan.c.sum()):,} "
        f"(~{N * 37 / 2**30:.1f} GiB of columns)")
    mm = {}
    spec_cols = {"flat_uid": np.int64, "flat_item_ids": np.int64, "flat_timestamps": np.int64,
                 "flat_event_types": np.int64, "flat_is_organic": np.int8,
                 "flat_played_ratio": np.float32}
    for name, dt in spec_cols.items():
        mm[name] = np.lib.format.open_memmap(os.path.join(tcache, f"{name}.npy"), "w+", dt, (N,))
    t1 = time.time()
    for ui in range(U):
        et, ratio, ts, items, organic = plan.gen_user(ui)
        s, e = int(starts[ui]), int(starts[ui] + lens[ui])
        assert e - s == et.size, (ui, e - s, et.size)
        mm["flat_uid"][s:e] = plan.uids[ui]
        mm["flat_item_ids"][s:e] = items
        mm["flat_timestamps"][s:e] = ts
        mm["flat_event_types"][s:e] = et
        mm["flat_is_organic"][s:e] = organic
        mm["flat_played_ratio"][s:e] = ratio
        if ui and ui % max(1, U // 10) == 0:
            log(f"pass 2: {ui}/{U} users ({time.time() - t1:.0f}s)")
    for m in mm.values():
        m.flush()
    del mm
    # derived columns, exactly as _ensure_cache_built
    load = lambda n: np.load(os.path.join(tcache, f"{n}.npy"), mmap_mode="r")  # noqa: E731
    et, pr = load("flat_event_types"), load("flat_played_ratio")
    outs = {k: np.lib.format.open_memmap(os.path.join(tcache, f"{k}.npy"), "w+", np.bool_, (N,))
            for k in ("flat_is_listen_plus", "flat_is_like", "flat_is_skip")}
    CH = 1 << 24
    for s in range(0, N, CH):
        e_, p_ = et[s:s + CH], pr[s:s + CH]
        il = e_ == LISTEN
        outs["flat_is_listen_plus"][s:s + CH] = il & (p_ >= LP_THRESHOLD)
        outs["flat_is_like"][s:s + CH] = e_ == LIKE
        outs["flat_is_skip"][s:s + CH] = il & (p_ < LP_THRESHOLD)
    for m in outs.values():
        m.flush()
    del outs
    max_uid = int(plan.uids.max()) + 1
    user_start = np.full(max_uid, -1, np.int64)
    user_end = np.full(max_uid, -1, np.int64)
    user_start[plan.uids] = starts
    user_end[plan.uids] = starts + lens
    np.save(os.path.join(tcache, "user_start.npy"), user_start)
    np.save(os.path.join(tcache, "user_end.npy"), user_end)
    np.save(os.path.join(tcache, "unique_uids.npy"), plan.uids)
    # positions (_ensure_positions_built formula; min_history == history_length)
    fu = load("flat_uid")
    parts = []
    for s in range(0, N, CH):
        idx = np.arange(s, min(N, s + CH), dtype=np.int64)
        keep = (idx - user_start[fu[s:s + CH]] >= a.history) & (et[s:s + CH] == LISTEN)
        parts.append(idx[keep])
    positions = np.concatenate(parts).astype(np.int64)
    np.save(os.path.join(tcache, f"positions_L{a.history}.npy"), positions)
    ts_all = load("flat_timestamps")
    np.save(os.path.join(tcache, f"anchor_ts_L{a.history}.npy"),
            np.asarray(ts_all[positions], dtype=np.int64))
    with open(os.path.join(tcache, "store_meta.json"), "w") as f:
        json.dump({"num_users": U, "total_events": N}, f)
    open(os.path.join(tcache, "_READY"), "w").close()
    it = load("flat_item_ids")
    pop = np.zeros(plan.V, np.int64)
    for s in range(0, N, CH):
        pop += np.bincount(it[s:s + CH], minlength=plan.V)
    np.save(os.path.join(tmp, "processed_5b", "item_popularity.npy"), pop)
    g = np.random.default_rng([a.seed, 2])
    item = np.arange(plan.V, dtype=np.uint32)
    eng = save_parquet(os.path.join(tmp, "shared_metadata", "artist_item_mapping.parquet"),
                       {"artist_id": g.integers(0, plan.vocab["artist_id"], plan.V
                                                ).astype(np.uint32), "item_id": item})
    save_parquet(os.path.join(tmp, "shared_metadata", "album_item_mapping.parquet"),
                 {"album_id": g.integers(0, plan.vocab["album_id"], plan.V
                                         ).astype(np.uint32), "item_id": item})
    with open(os.path.join(tmp, "_SYNTH_SPEC.json"), "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.replace(tmp, root)
    log(f"wrote {root} (parquet via {eng}) in {time.time() - t1:.0f}s")
    return plan, spec


def check(a, root, plan):
    """Re-derive the cache with the dataset code's formulas; exit 1 on mismatch."""
    cache = os.path.join(root, "processed_5b", f"hstu_cache_L{a.history}")
    bad = []
    L = lambda n: np.load(os.path.join(cache, n), mmap_mode="r")  # noqa: E731
    meta = json.load(open(os.path.join(cache, "store_meta.json")))
    N = int(meta["total_events"])
    want = {"flat_uid": "int64", "flat_item_ids": "int64", "flat_timestamps": "int64",
            "flat_event_types": "int64", "flat_played_ratio": "float32",
            "flat_is_listen_plus": "bool", "flat_is_like": "bool", "flat_is_skip": "bool",
            "flat_is_organic": "int8", "user_start": "int64", "user_end": "int64",
            "unique_uids": "int64"}
    for n in MMAP_COLS:
        x = L(f"{n}.npy")
        if str(x.dtype) != want[n] or x.ndim != 1:
            bad.append(f"{n} dtype {x.dtype} ndim {x.ndim}")
        if n.startswith("flat_") and len(x) != N:
            bad.append(f"{n} len {len(x)} != {N}")
    fu, et, pr, ts = L("flat_uid.npy"), L("flat_event_types.npy"), L("flat_played_ratio.npy"), \
        L("flat_timestamps.npy")
    us, ue, uu = L("user_start.npy"), L("user_end.npy"), L("unique_uids.npy")
    if meta["num_users"] != len(uu):
        bad.append("num_users")
    # _ensure_cache_built user index formula on a uid-sorted flat_uid
    if np.any(np.diff(fu) < 0):
        bad.append("flat_uid not sorted")
    ch = np.where(np.diff(fu) != 0)[0] + 1
    st = np.concatenate([[0], ch])
    if not (np.array_equal(fu[st], uu) and np.array_equal(us[uu], st)
            and np.array_equal(ue[uu], np.concatenate([ch, [N]]))):
        bad.append("user_start/end/unique_uids")
    for s in range(0, N, 1 << 24):
        e_, p_ = et[s:s + (1 << 24)], pr[s:s + (1 << 24)]
        il = e_ == LISTEN
        if not (np.array_equal(L("flat_is_listen_plus.npy")[s:s + (1 << 24)], il & (p_ >= 50))
                and np.array_equal(L("flat_is_like.npy")[s:s + (1 << 24)], e_ == LIKE)
                and np.array_equal(L("flat_is_skip.npy")[s:s + (1 << 24)], il & (p_ < 50))):
            bad.append(f"bool flags chunk {s}")
            break
        if np.any((e_ < 0) | (e_ > 4)):
            bad.append("event type range")
            break
    # per-user chronological order
    d = np.diff(ts)
    if np.any(d[np.diff(fu) == 0] < 0):
        bad.append("timestamps not sorted within a user")
    pos = L(f"positions_L{a.history}.npy")
    idx = np.arange(N, dtype=np.int64)
    ref = np.where((idx - us[fu] >= a.history) & (et == LISTEN))[0]
    if not np.array_equal(pos, ref):
        bad.append("positions != _ensure_positions_built formula")
    ats = L(f"anchor_ts_L{a.history}.npy")
    if not np.array_equal(ats, ts[pos]):
        bad.append("anchor_ts != flat_timestamps[positions]")
    t_min = int(ats.min())
    nwin = int((int(ats.max()) - t_min + 1 + W_SECONDS - 1) // W_SECONDS)
    per = np.bincount((ats - t_min) // W_SECONDS, minlength=nwin).tolist()
    if t_min != T0 or per != plan.A:
        bad.append(f"window counts {per} (t_min {t_min}) != plan {plan.A}")
    # metadata (DLRMv4YambdaDataset._load_metadata)
    try:
        import polars as pl
        art = pl.read_parquet(os.path.join(root, "shared_metadata", "artist_item_mapping.parquet"))
        alb = pl.read_parquet(os.path.join(root, "shared_metadata", "album_item_mapping.parquet"))
        n_items = max(int(art["item_id"].max()) + 1, int(alb["item_id"].max()) + 1)
        if n_items != plan.vocab["item_id"]:
            bad.append(f"mapping n_items {n_items}")
        if int(art["artist_id"].max()) >= plan.vocab["artist_id"] or \
                int(alb["album_id"].max()) >= plan.vocab["album_id"]:
            bad.append("artist/album id beyond table")
    except ImportError:
        log("WARN polars missing: mapping parquet check skipped")
        n_items = plan.vocab["item_id"]
    pop = np.load(os.path.join(root, "processed_5b", "item_popularity.npy"), mmap_mode="r")
    if len(pop) != n_items or int(pop.sum()) != N:
        bad.append("item_popularity length/sum")
    if int(L("flat_item_ids.npy").max()) >= plan.vocab["item_id"] or int(uu.max()) >= plan.vocab["uid"]:
        bad.append("ids beyond table")
    # sequence-length profile per window (interleaved pools, cap L//3, scan 20000);
    # sampled, diagnostics only
    g = np.random.default_rng(1)
    samp = np.sort(g.choice(len(pos), size=min(4000, len(pos)), replace=False))
    lp, lk, sk = L("flat_is_listen_plus.npy"), L("flat_is_like.npy"), L("flat_is_skip.npy")
    cap = a.history // 3
    lens = []
    for i in samp:
        p = int(pos[i])
        s0 = max(int(us[fu[p]]), p - 20000)
        lens.append(min(cap, int(lp[s0:p].sum())) + min(cap, int(lk[s0:p].sum()))
                    + min(cap, int(sk[s0:p].sum())))
    lens = np.array(lens)
    q = np.percentile(lens, [0, 5, 25, 50, 75, 95, 100]).astype(int).tolist()
    log(f"check: events={N:,} users={len(uu):,} anchors={len(pos):,} windows={nwin} "
        f"per_window={per} uih_len_pctl[0,5,25,50,75,95,100]={q} "
        f"len%16==0 share={float(np.mean(lens % 16 == 0)):.3f}")
    # exact per-rank token load of every train batch, with the trainer's batching:
    # window (ts - t_min) // W, user-major inside a window, rank r takes
    # idx[:n][r::ws], batches of `batch`, at most train_batches per window.
    # Gate: dq-split ds workspace need <= NEED_LIMIT (see the module docstring).
    ws, bs = a.world_size, a.batch
    C = [np.concatenate([[0], np.cumsum(np.asarray(f), dtype=np.int32)]) for f in (lp, lk, sk)]
    alen = np.empty(len(pos), np.int64)
    for s in range(0, len(pos), 1 << 24):
        p = np.asarray(pos[s:s + (1 << 24)])
        s0 = np.maximum(np.asarray(us)[np.asarray(fu[p])], p - 20000)
        alen[s:s + (1 << 24)] = sum(np.minimum(cap, c[p] - c[s0]) for c in C)
    del C
    seq = alen + CTX_TOKENS
    wid = (np.asarray(ats) - t_min) // W_SECONDS
    TN = {}
    for w, role in enumerate(plan.roles):        # per window: T[b, r], N[b, r]
        sw = seq[wid == w]
        m = sw.size // ws
        nb = m // bs if role == "E" else min(m // bs, a.train_batches)
        X = sw[:m * ws].reshape(m, ws)[:nb * bs].reshape(nb, bs, ws)
        TN[w] = (X.sum(1), X.max(1))
    Ts, Ns, worst = [], [], (0, None)
    for w, role in enumerate(plan.roles):
        T, Nm = TN[w]
        if T.size and int(T.max()) > T_LIMIT:
            b, r = np.unravel_index(int(T.argmax()), T.shape)
            bad.append(f"rank-batch tokens {int(T[b, r]):,} > T_LIMIT {T_LIMIT:,} at "
                       f"window/batch/rank={w}/{b}/{r} ({role})")
        if role != "T":
            continue
        nd = dq_need(T, Nm, bs)
        b, r = np.unravel_index(int(nd.argmax()), nd.shape)
        if int(nd[b, r]) > worst[0]:
            worst = (int(nd[b, r]), (w, int(b), int(r), int(T[b, r]), int(Nm[b, r])))
        Ts.append(T.ravel())
        Ns.append(Nm.ravel())
    Ts, Ns = np.concatenate(Ts), np.concatenate(Ns)
    nd_max, where = worst
    log(f"load: ws={ws} train rank-batches={Ts.size} tokens[min,p50,max]="
        f"{Ts.min() / 1e6:.3f}/{np.median(Ts) / 1e6:.3f}/{Ts.max() / 1e6:.3f}M "
        f"N[min,max]={int(Ns.min())}/{int(Ns.max())} distinct_N={np.unique(Ns).size} "
        f"T>2^21 {float((Ts > 1 << 21).mean()):.3f} T<2^19 {float((Ts < 1 << 19).mean()):.3f} "
        f"T%16==0 {float((Ts % 16 == 0).mean()):.3f} N%16==0 {float((Ns % 16 == 0).mean()):.3f} "
        f"ds_need_max={nd_max / GIB:.2f} GiB (limit {NEED_LIMIT / GIB:.0f}) at "
        f"window/batch/rank/T/N={where}")
    if nd_max > NEED_LIMIT:
        bad.append(f"ds workspace need {nd_max / GIB:.2f} GiB > {NEED_LIMIT / GIB:.0f} GiB at "
                   f"window/batch/rank/T/N={where}")
    # record arms start at window 0 and at each train window after an eval window; their
    # first CAL_STEPS batches are the BF16 calibration steps, the next one sizes the ds
    # workspace. Gates: calibration tokens <= CAL_T_MAX, above the top 2 GiB line on every
    # rank (and below the lowest one in window 0), and no ds workspace regrow after the
    # first FP8 step of the arm.
    starts = [w for w, role in enumerate(plan.roles)
              if role == "T" and (w == 0 or plan.roles[w - 1] == "E")]
    for s in starts:
        arm = [s]
        while arm[-1] + 1 < plan.nwin and plan.roles[arm[-1] + 1] == "T":
            arm.append(arm[-1] + 1)
        Tc = TN[s][0][:CAL_STEPS]
        lo_ok = bool((Tc.min(0) < STORAGE_LINES[0]).all())
        hi_ok = bool((Tc.max(0) > STORAGE_LINES[-1]).all())
        if int(Tc.max()) > CAL_T_MAX:
            bad.append(f"arm@{s}: calibration tokens {int(Tc.max()):,} > CAL_T_MAX {CAL_T_MAX:,}")
        if not hi_ok or (s == 0 and not lo_ok):
            bad.append(f"arm@{s}: calibration batches do not straddle the 2 GiB lines on every "
                       f"rank (per-rank min {int(Tc.min(0).max()):,} max {int(Tc.max(0).min()):,})")
        bufs, regrow = [], []
        for r in range(ws):
            buf = 0
            for i, w in enumerate(arm):
                T, Nm = TN[w]
                for b in range(CAL_STEPS if i == 0 else 0, T.shape[0]):
                    nd = int(dq_need(int(T[b, r]), int(Nm[b, r]), bs))
                    if nd > buf:
                        if buf:
                            regrow.append((w, b, r, buf // GIB))
                        want = max(nd + nd // 4, buf + buf // 2)
                        buf = (want + GIB - 1) // GIB * GIB
            bufs.append(buf // GIB)
        if regrow:
            bad.append(f"arm@{s}: ds workspace regrows after the first FP8 step at "
                       f"window/batch/rank/old_GiB={regrow[:3]}")
        log(f"arm@{s} windows {arm[0]}-{arm[-1]}: calibration tokens per batch (rank 0) "
            f"{[int(t) for t in Tc[:, 0]]} lines {list(STORAGE_LINES)} straddled "
            f"low={lo_ok} high={hi_ok}; first FP8 step tokens {int(TN[s][0][CAL_STEPS].min()):,}-"
            f"{int(TN[s][0][CAL_STEPS].max()):,}; ds workspace {min(bufs)}-{max(bufs)} GiB, "
            f"regrows after it {len(regrow)}")
    if bad:
        log("CHECK FAIL: " + "; ".join(bad))
        return 1
    log("CHECK OK (dtypes, user index, flags, positions, anchor_ts, window plan, metadata, "
        "load bound)")
    return 0


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--layout", default="T:8,E:1,T:2,E:1")
    p.add_argument("--users", type=int, default=24000)
    p.add_argument("--history", type=int, default=4086)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--configs", default=os.path.normpath(DEFAULT_CONFIGS),
                   help="generative_recommenders/dlrm_v4/configs.py (default: ../configs.py)")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.check_only:
        plan = Plan(a)
    else:
        plan, _ = generate(a, root)
    if a.no_check:
        return 0
    return check(a, root, plan)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
