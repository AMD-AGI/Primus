###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Synthetic yambda-5b data for DLRMv4 throughput runs (no 227 GB download).

Writes a small dataset in the layout of the staged MLPerf cache, so the trainer
reads it with its unchanged yambda-5b loader, model, kernels and gin file:

  <root>/processed_5b/item_popularity.npy
  <root>/processed_5b/hstu_cache_L4086/{flat_*.npy, user_start.npy, user_end.npy,
      unique_uids.npy, positions_L4086.npy, anchor_ts_L4086.npy,
      ts_indices_L4086_W86400.npy, ts_offsets_L4086_W86400.npy, store_meta.json, _READY}
  <root>/shared_metadata/{artist,album}_item_mapping.parquet
  <root>/synthetic_spec.json

Two daily windows: 0 is the train window (start_ts 0, num_train_ts 1) and 1 the
eval window (eval_holdout_ts -1 resolves to it). The users' earlier events, which
fill the history of every sample, lie before them. Unlike the real cache, those
events are not anchors (positions_L4086.npy lists the anchors of the two windows
only): no run trains on them, and their index would add half to the dataset's size.

Throughput depends on the per-sample sequence length (the three event pools,
each capped at history_length // 3 within the last 20000 events), on how many
consecutive anchors come from one user (the trainer batches a window
user-major) and on how often ids repeat within a batch (embedding dedup and the
sparse all-to-all). PROFILE holds those statistics of the real data in daily
window 200, a steady-state window (windows 100-298 hold 91% of the training
samples and differ little), measured on the MLCommons cache; each user is drawn
from them:

  - anchors (listen events) in the window and events before it;
  - the shares of listen+ (played >= 50%), skip, like and other events;
  - a library of artists, and items from those artists' albums, that the user
    listens to repeatedly.

The result matches the real window within a few percent on tokens per rank-batch,
users per batch and unique items, artists and albums per batch, and on MI355X
(rocm/primus:v26.9) its training throughput within 1% (8397 vs 8454 samples/s
per GPU, steps 31-200 on 8 GPUs).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

GENERATOR_VERSION = 3

LISTEN, LIKE = 0, 1  # datasets/yambda.py LISTEN_TYPE / LIKE_TYPE
OTHER_TYPES = (2, 3, 4)  # dislike / unlike / undislike: in no pool
LP_THRESHOLD = 50  # datasets/yambda.py LISTEN_PLUS_THRESHOLD
WINDOW_SECONDS = 86400  # gin get_dataset.streaming_window_seconds
SCAN_WINDOW = 20000  # utils.get_dataset scan_window
T0 = 1_700_006_400  # start of the train window (a UTC midnight)

TRAIN_WINDOW, EVAL_WINDOW = 0, 1

SRC_ROOT = Path(__file__).resolve().parent / "src"
TSPEC = SRC_ROOT / "generative_recommenders" / "dlrm_v4" / "train" / "tspec_synth_data.py"

# Real yambda-5b, daily window 200 (10.66 M anchors, 190,697 users).
# Quantiles as (percentile, value); users are drawn by interpolating between them.
PROFILE = {
    # anchors per user in the window (the 100th percentile is 9760, capped here)
    "anchors_per_user": (
        (0, 1),
        (1, 1),
        (5, 4),
        (25, 18),
        (50, 40),
        (75, 75),
        (95, 160),
        (99, 261),
        (100, 1500),
    ),
    # events of the user before the anchor
    "prior_events": (
        (0, 4086),
        (1, 4170),
        (5, 4490),
        (25, 6170),
        (50, 8740),
        (75, 12200),
        (95, 17100),
        (99, 19700),
        (100, 24700),
    ),
    # event-type shares of a user's history. Users here are drawn per user, while a
    # batch weighs them by their anchors; the listen+ tail and the like shares are
    # adjusted (like x 0.88) so the per-anchor pools match the window's: listen+
    # 1351, like 121, skip 1209 tokens on average (UIH 2680).
    "listen_plus_share": (
        (0, 0.0096),
        (1, 0.153),
        (5, 0.264),
        (25, 0.467),
        (50, 0.631),
        (75, 0.80),
        (90, 0.92),
        (95, 0.96),
        (99, 0.993),
        (100, 1.0),
    ),
    "like_share": (
        (0, 0.0),
        (1, 0.0),
        (5, 0.0000638),
        (25, 0.00202),
        (50, 0.00691),
        (75, 0.0151),
        (95, 0.0397),
        (99, 0.0821),
        (100, 0.264),
    ),
    "other_share": (
        (0, 0.0),
        (1, 0.0),
        (5, 0.0),
        (25, 0.000917),
        (50, 0.00282),
        (75, 0.00733),
        (95, 0.029),
        (99, 0.0782),
        (100, 0.3),
    ),
    # share of organic (not recommended) plays
    "organic": 0.5,
    # anchors per hour of the day (UTC)
    "hours": (
        26775,
        27210,
        33902,
        49864,
        77249,
        116455,
        136667,
        130420,
        133224,
        135734,
        138733,
        140738,
        143709,
        146087,
        146276,
        156650,
        153651,
        136112,
        115110,
        101365,
        79211,
        58001,
        42420,
        32037,
    ),
}

# Catalog and library model (calibrated against the per-batch unique ids of the
# real window; see the module docstring).
CATALOG = {
    "artist_zipf": 0.8,  # artist popularity
    "artists_per_user": (140.0, 0.6),  # lognormal median and sigma
    "albums_per_artist_pick": 4,  # draws of an artist's albums (repeats collapse)
    "item_zipf": 1.1,  # replay concentration within a user's library
}


def _load_tspec():
    """The trainer's synthetic-data script: its vocabulary parser and parquet writer."""
    spec = importlib.util.spec_from_file_location("dlrm_v4_tspec_synth_data", TSPEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draw(rng: np.random.Generator, table: Sequence[Tuple[float, float]], n: int, log: bool = False):
    q = np.array([p for p, _ in table], dtype=np.float64) / 100.0
    v = np.array([x for _, x in table], dtype=np.float64)
    u = rng.random(n)
    if log:
        return np.exp(np.interp(u, q, np.log(np.maximum(v, 1e-12))))
    return np.interp(u, q, v)


def _zipf_rank(rng: np.random.Generator, n_ranks: int, s: float, size: int) -> np.ndarray:
    """Ranks 0..n_ranks-1 with P(r) ~ (r + 1)^-s (continuous inverse CDF)."""
    u = rng.random(size)
    if abs(s - 1.0) < 1e-9:
        x = np.exp(u * np.log(n_ranks + 1.0))
    else:
        x = (1.0 + u * ((n_ranks + 1.0) ** (1.0 - s) - 1.0)) ** (1.0 / (1.0 - s))
    return np.clip(np.floor(x).astype(np.int64) - 1, 0, n_ranks - 1)


class Catalog:
    """Items grouped into albums and albums into artists, sizes as the yambda-5b tables.

    Popular artists have more albums. Item, album and artist ids are permuted, so
    rows of one artist are spread over the embedding tables.
    """

    def __init__(self, vocab: Dict[str, int], seed: int):
        g = np.random.default_rng([seed, 1])
        n_items, n_albums, n_artists = vocab["item_id"], vocab["album_id"], vocab["artist_id"]
        # albums per artist: every artist one, the rest by popularity rank
        extra = np.bincount(_zipf_rank(g, n_artists, 0.6, n_albums - n_artists), minlength=n_artists)
        self.artist_album_start = np.concatenate([[0], np.cumsum(1 + extra)])
        # items per album: one each, the rest at random
        extra = np.bincount(g.integers(0, n_albums, n_items - n_albums), minlength=n_albums)
        self.album_item_start = np.concatenate([[0], np.cumsum(1 + extra)])
        self.item_ids = g.permutation(n_items).astype(np.int64)
        album_of_item = np.repeat(np.arange(n_albums), 1 + extra)
        artist_of_album = np.repeat(np.arange(n_artists), np.diff(self.artist_album_start))
        album_ids = g.permutation(n_albums).astype(np.int64)
        artist_ids = g.permutation(n_artists).astype(np.int64)
        # mapping tables, indexed by item id
        self.item_album = np.empty(n_items, np.int64)
        self.item_album[self.item_ids] = album_ids[album_of_item]
        self.item_artist = np.empty(n_items, np.int64)
        self.item_artist[self.item_ids] = artist_ids[artist_of_album[album_of_item]]
        self.n_artists = n_artists

    def library(self, rng: np.random.Generator, n_artists: int) -> np.ndarray:
        """Item ids of a user's library: some albums of some artists."""
        artists = np.unique(_zipf_rank(rng, self.n_artists, CATALOG["artist_zipf"], n_artists))
        a0, a1 = self.artist_album_start[artists], self.artist_album_start[artists + 1]
        picks = CATALOG["albums_per_artist_pick"]
        albums = np.concatenate([a0 + rng.integers(0, a1 - a0) for _ in range(picks)])
        albums = np.unique(albums)
        i0, i1 = self.album_item_start[albums], self.album_item_start[albums + 1]
        items = np.concatenate([np.arange(s, e) for s, e in zip(i0, i1)])
        return self.item_ids[items]


class Plan:
    """Users and their windows."""

    def __init__(
        self, *, train_samples: int, eval_samples: int, seed: int, vocab: Dict[str, int], history_length: int
    ):
        self.seed = seed
        self.history_length = history_length
        # One stream per attribute, so user i is the same whatever the dataset's size:
        # a shorter run trains on the first batches of a longer one.
        streams = [np.random.default_rng([seed, 10 + j]) for j in range(9)]
        k_train = self._split(streams[0], train_samples)
        k_eval = self._split(streams[1], eval_samples)
        self.window = np.concatenate([np.full(k_train.size, TRAIN_WINDOW), np.full(k_eval.size, EVAL_WINDOW)])
        self.anchors = np.concatenate([k_train, k_eval])
        n = self.anchors.size
        if n >= vocab["uid"]:
            raise ValueError(f"{n} users do not fit the uid table ({vocab['uid']})")
        self.uids = np.sort(streams[2].choice(vocab["uid"] - 1, size=n, replace=False) + 1).astype(np.int64)
        prior = np.rint(_draw(streams[3], PROFILE["prior_events"], n)).astype(np.int64)
        self.prior = np.clip(prior, history_length, SCAN_WINDOW + history_length)
        like = _draw(streams[4], PROFILE["like_share"], n)
        other = _draw(streams[5], PROFILE["other_share"], n)
        lp = _draw(streams[6], PROFILE["listen_plus_share"], n) * (1.0 - like - other)
        skip = np.maximum(0.0, 1.0 - like - other - lp)
        self.shares = np.stack([lp, like, skip, other], axis=1)
        self.shares /= self.shares.sum(axis=1, keepdims=True)
        self.organic = np.clip(PROFILE["organic"] + 0.15 * streams[7].standard_normal(n), 0.0, 1.0)
        median, sigma = CATALOG["artists_per_user"]
        self.library_artists = np.maximum(
            1, np.rint(median * np.exp(sigma * streams[8].standard_normal(n)))
        ).astype(np.int64)
        # events per user: the history, then the window's anchors and the other
        # event types between them
        listen = self.shares[:, 0] + self.shares[:, 2]
        self.window_events = self.anchors + np.rint(
            self.anchors * (1.0 - listen) / np.maximum(listen, 1e-3)
        ).astype(np.int64)
        self.events = self.prior + self.window_events

    @staticmethod
    def _split(rng: np.random.Generator, total: int) -> np.ndarray:
        """Anchors per user, summing to `total`."""
        out = []
        left = total
        while left > 0:
            k = np.maximum(1, np.rint(_draw(rng, PROFILE["anchors_per_user"], 4096, log=True))).astype(
                np.int64
            )
            k = k[np.cumsum(k) - k < left]
            k[-1] -= max(0, int(k.sum()) - left)
            out.append(k[k > 0])
            left -= int(k.sum())
        return np.concatenate(out)


def _user_events(plan: Plan, catalog: Catalog, hour_cdf: np.ndarray, ui: int):
    """Event columns of user `ui`, in time order."""
    rng = np.random.default_rng([plan.seed, 2, ui])
    P, D, k = int(plan.prior[ui]), int(plan.window_events[ui]), int(plan.anchors[ui])
    n = P + D
    lp, like, skip, other = plan.shares[ui]
    # 0 listen+, 1 like, 2 skip, 3 other
    kind = np.empty(n, np.int8)
    kind[:P] = rng.choice(4, size=P, p=plan.shares[ui])
    listen = lp + skip
    win = np.full(D, 3, np.int8)
    win[:k] = np.where(rng.random(k) < lp / max(listen, 1e-9), 0, 2)
    nl = D - k
    if nl:
        win[k:] = np.where(rng.random(nl) < like / max(like + other, 1e-9), 1, 3)
    rng.shuffle(win)
    if ui == 0:  # the first user's first event in the window is a listen (see below)
        first = int(np.argmax((win == 0) | (win == 2)))
        win[[0, first]] = win[[first, 0]]
    kind[P:] = win
    et = np.where(kind == 1, LIKE, LISTEN).astype(np.int64)
    oth = kind == 3
    et[oth] = rng.choice(OTHER_TYPES, size=int(oth.sum()))
    ratio = np.zeros(n, np.float32)
    is_lp, is_skip = kind == 0, kind == 2
    ratio[is_lp] = rng.integers(LP_THRESHOLD, 101, int(is_lp.sum()))
    ratio[is_skip] = rng.integers(0, LP_THRESHOLD, int(is_skip.sum()))
    # time: the history in the 60 days before the train window, then the window's events
    ts = np.empty(n, np.int64)
    ts[:P] = np.sort(T0 - rng.integers(1, 60 * WINDOW_SECONDS, P))
    day = T0 + int(plan.window[ui]) * WINDOW_SECONDS
    hour = np.searchsorted(hour_cdf, rng.random(D), side="right")
    ts[P:] = np.sort(day + hour * 3600 + rng.integers(0, 3600, D))
    if ui == 0:
        ts[P] = T0  # t_min, the earliest anchor: the window grid starts at T0
    # items: replays from the user's library, with a favourites skew
    lib = catalog.library(rng, int(plan.library_artists[ui]))
    rng.shuffle(lib)
    items = lib[_zipf_rank(rng, lib.size, CATALOG["item_zipf"], n)]
    organic = (rng.random(n) < plan.organic[ui]).astype(np.int8)
    return et, ratio, ts, items, organic


def _spec(vocab, **kw) -> dict:
    return {
        "synthetic": True,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vocab": vocab,
        "profile": "yambda-5b window 200",
        **kw,
    }


def generate(
    root: str,
    *,
    train_samples: int,
    eval_samples: int,
    seed: int = 1234,
    history_length: int = 4086,
    log=print,
) -> dict:
    """Write the dataset to `root` (replaced if present). Returns its spec."""
    tspec = _load_tspec()
    vocab, _ = tspec.parse_vocab(str(SRC_ROOT / "generative_recommenders" / "dlrm_v4" / "configs.py"))
    t0 = time.time()
    plan = Plan(
        train_samples=train_samples,
        eval_samples=eval_samples,
        seed=seed,
        vocab=vocab,
        history_length=history_length,
    )
    catalog = Catalog(vocab, seed)
    hours = np.asarray(PROFILE["hours"], np.float64)
    hour_cdf = np.cumsum(hours) / hours.sum()
    hour_cdf[-1] = 1.0
    N = int(plan.events.sum())
    U = plan.uids.size
    log(
        f"[synthetic] {U:,} users, {N:,} events, {train_samples:,} train and {eval_samples:,} eval samples "
        f"(~{N * 37 / 2**30:.1f} GiB)"
    )

    root_path = Path(root)
    tmp = Path(f"{root_path}.tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    cache = tmp / "processed_5b" / f"hstu_cache_L{history_length}"
    cache.mkdir(parents=True)
    (tmp / "shared_metadata").mkdir()

    def memmap(name, dtype, n=N):
        return np.lib.format.open_memmap(cache / f"{name}.npy", "w+", dtype, (n,))

    cols = {
        "flat_uid": memmap("flat_uid", np.int64),
        "flat_item_ids": memmap("flat_item_ids", np.int64),
        "flat_timestamps": memmap("flat_timestamps", np.int64),
        "flat_event_types": memmap("flat_event_types", np.int64),
        "flat_played_ratio": memmap("flat_played_ratio", np.float32),
        "flat_is_listen_plus": memmap("flat_is_listen_plus", np.bool_),
        "flat_is_like": memmap("flat_is_like", np.bool_),
        "flat_is_skip": memmap("flat_is_skip", np.bool_),
        "flat_is_organic": memmap("flat_is_organic", np.int8),
    }
    starts = np.concatenate([[0], np.cumsum(plan.events)[:-1]]).astype(np.int64)
    positions = []
    for ui in range(U):
        et, ratio, ts, items, organic = _user_events(plan, catalog, hour_cdf, ui)
        s, P = int(starts[ui]), int(plan.prior[ui])
        e = s + et.size
        listen = et == LISTEN
        cols["flat_uid"][s:e] = plan.uids[ui]
        cols["flat_item_ids"][s:e] = items
        cols["flat_timestamps"][s:e] = ts
        cols["flat_event_types"][s:e] = et
        cols["flat_played_ratio"][s:e] = ratio
        cols["flat_is_listen_plus"][s:e] = listen & (ratio >= LP_THRESHOLD)
        cols["flat_is_like"][s:e] = et == LIKE
        cols["flat_is_skip"][s:e] = listen & (ratio < LP_THRESHOLD)
        cols["flat_is_organic"][s:e] = organic
        # anchors: the window's listens (each has >= min_history earlier events)
        positions.append(s + P + np.flatnonzero(listen[P:]))
        if ui and ui % max(1, U // 5) == 0:
            log(f"[synthetic] {ui:,}/{U:,} users ({time.time() - t0:.0f}s)")
    for m in cols.values():
        m.flush()
    popularity = np.zeros(vocab["item_id"], np.int64)
    for s in range(0, N, 1 << 24):
        popularity += np.bincount(cols["flat_item_ids"][s : s + (1 << 24)], minlength=popularity.size)
    max_uid = int(plan.uids.max()) + 1
    user_start = np.full(max_uid, -1, np.int64)
    user_end = np.full(max_uid, -1, np.int64)
    user_start[plan.uids] = starts
    user_end[plan.uids] = starts + plan.events
    np.save(cache / "user_start.npy", user_start)
    np.save(cache / "user_end.npy", user_end)
    np.save(cache / "unique_uids.npy", plan.uids)
    positions = np.concatenate(positions).astype(np.int64)
    tag = f"L{history_length}"
    np.save(cache / f"positions_{tag}.npy", positions)
    anchor_ts = np.asarray(cols["flat_timestamps"][positions], dtype=np.int64)
    np.save(cache / f"anchor_ts_{tag}.npy", anchor_ts)
    # offline window index (datasets/offline_ts_indices.py): anchors bucketed by
    # window, in position order within a window
    window = (anchor_ts - int(anchor_ts.min())) // WINDOW_SECONDS
    counts = np.bincount(window)
    np.save(
        cache / f"ts_indices_{tag}_W{WINDOW_SECONDS}.npy", np.argsort(window, kind="stable").astype(np.int64)
    )
    np.save(
        cache / f"ts_offsets_{tag}_W{WINDOW_SECONDS}.npy",
        np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
    )
    del cols
    with open(cache / "store_meta.json", "w") as f:
        json.dump({"num_users": U, "total_events": N}, f)
    np.save(tmp / "processed_5b" / "item_popularity.npy", popularity)
    item = np.arange(vocab["item_id"], dtype=np.uint32)
    tspec.save_parquet(
        str(tmp / "shared_metadata" / "artist_item_mapping.parquet"),
        {"artist_id": catalog.item_artist.astype(np.uint32), "item_id": item},
    )
    tspec.save_parquet(
        str(tmp / "shared_metadata" / "album_item_mapping.parquet"),
        {"album_id": catalog.item_album.astype(np.uint32), "item_id": item},
    )
    (cache / "_READY").touch()
    spec = _spec(
        vocab,
        train_samples=train_samples,
        eval_samples=eval_samples,
        seed=seed,
        history_length=history_length,
        windows=counts.tolist(),
        users=U,
        events=N,
    )
    (tmp / "synthetic_spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    if counts.tolist() != [train_samples, eval_samples]:
        raise RuntimeError(
            f"[synthetic] window anchors {counts.tolist()}, expected [{train_samples}, {eval_samples}]"
        )
    shutil.rmtree(root_path, ignore_errors=True)
    os.replace(tmp, root_path)
    log(f"[synthetic] wrote {root_path} in {time.time() - t0:.0f}s (windows {counts.tolist()})")
    return spec


def ensure(
    parent: str,
    *,
    train_samples: int,
    eval_samples: int,
    seed: int = 1234,
    history_length: int = 4086,
    log=print,
) -> str:
    """The dataset for these sizes under `parent`, generated on first use.

    Datasets are kept per spec (`parent/<hash>`), so later runs of the same shape
    reuse them; a file lock serializes processes that share `parent`.
    """
    key = json.dumps([GENERATOR_VERSION, train_samples, eval_samples, seed, history_length])
    root = Path(parent) / f"yambda5b-synthetic-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    Path(parent).mkdir(parents=True, exist_ok=True)
    with open(Path(parent) / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "synthetic_spec.json").exists():
            log(f"[synthetic] reusing {root}")
        else:
            generate(
                str(root),
                train_samples=train_samples,
                eval_samples=eval_samples,
                seed=seed,
                history_length=history_length,
                log=log,
            )
    return str(root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="directory to create (replaced if present)")
    p.add_argument("--train-samples", type=int, required=True)
    p.add_argument("--eval-samples", type=int, default=16384)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)
    generate(a.root, train_samples=a.train_samples, eval_samples=a.eval_samples, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
