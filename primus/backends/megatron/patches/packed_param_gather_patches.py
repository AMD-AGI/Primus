###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP6 packed parameter all-gather: DDP bucket layout (``mxfp6_packed_param_gather``).

With the distributed optimizer each rank owns one equal, contiguous shard of every DDP bucket. The MXFP6 linear
weights are gathered as their tilescale packs instead of bf16 when this gate is on, and a pack covers whole rows: so
the weights must sit in buckets whose shard boundaries fall on row-group edges inside every weight they cut. This
module lays the buffer out that way:

* Every bucket Megatron (and the head-bucket ramp) would form becomes up to three consecutive buckets in one bucket
  group: ``W6`` (linear weights whose forward pack is MXFP6 rows), ``W4`` (linear weights whose forward is MXFP4,
  e.g. the single blocks' fc1 under ``mxfp6_fwd_fp4_single_fc1``) and ``R`` (every other parameter, gathered as bf16).
* Inside ``W6`` / ``W4`` the shard size and the minimal padding before a cut weight make every shard boundary land
  on a multiple of ``ALIGN_ROWS`` rows of that weight.

Bucket membership is decided exactly as Megatron decides it (same walk, same ``fill >= bucket_size`` test, consulted
once per parameter, so the head-bucket ramp's stand-in still drives it); only the order inside a bucket and the
padding change. The replacement ``_ParamAndGradBuffer.__init__`` is a copy of Megatron's with that layout pass; it
refuses to run against a Megatron whose ``__init__`` source differs from the one it was copied from.
"""

import hashlib
import inspect
import math
import re

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

ALIGN_ROWS = 32  # C1 (the coarsest code plane) blocks 32 rows; scales travel in row-major form
# sha256 of inspect.getsource(_ParamAndGradBuffer.__init__) this module's copy was made from.
_MEGATRON_INIT_SHA = None  # filled by the patch at first install (see _check_source)
_LIN = re.compile(
    r"layers\.(\d+)\.(self_attention\.(linear_qkv|linear_proj|added_linear_qkv|added_linear_proj)|"
    r"mlp\.linear_fc[12]|context_mlp\.linear_fc[12])\.weight$"
)


def _pad(x: int, d: int) -> int:
    return -(-x // d) * d


def pack_kind(name: str, n_joint: int, fwd_fp4_fc1: bool, fwd_fp4_l2: bool, fwd_fp4_joint_mlp: bool):
    """``"W6"``, ``"W4"`` or None (gathered as bf16) for a parameter, from its name and the forward-FP4 gates."""
    m = _LIN.search(name)
    if not m:
        return None
    layer, what = int(m[1]), m[2]
    single = layer >= n_joint
    if single and what == "mlp.linear_fc1" and fwd_fp4_fc1:
        return "W4"
    if single and what in ("mlp.linear_fc2", "self_attention.linear_proj") and fwd_fp4_l2:
        return "W4"
    if not single and what.startswith(("mlp.", "context_mlp.")) and fwd_fp4_joint_mlp:
        return "W4"
    return "W6"


def plan_weight_bucket(shapes, dp, align_rows=ALIGN_ROWS):
    """Shard size S and each weight's start offset (relative) for a weight bucket of 2-D ``shapes`` in order: every
    boundary j*S that falls inside a weight is a multiple of ``align_rows`` rows of it. S is a multiple of every
    weight's row-group unit (align_rows * K) and of 128 (so the bucket keeps Megatron's 256-byte alignment): then once
    a weight's first inside boundary is on an edge, all of them are, and that one is reached by shifting the weight's
    start forward. Returns (S, starts, padding)."""
    total = sum(r * k for r, k in shapes)
    L = math.lcm(128, *(align_rows * k for _r, k in shapes))
    S = _pad(-(-total // dp), L)
    while True:
        starts, off = [], 0
        for r, k in shapes:
            off = _pad(off, 64)
            b = (off // S + 1) * S  # first boundary after the start
            if b < off + r * k:
                off += (b - off) % (align_rows * k)
            starts.append(off)
            off += r * k
        if off <= dp * S:
            return S, starts, dp * S - total
        S += L


def plan_layout(params, numels, kinds, shapes, bucket_size, dp):
    """Megatron's bucket membership (same walk / test / padding as its own layout pass), then each bucket split into
    R / W4 / W6 sub-buckets. Returns (param -> (start, end, bucket_id), [(start, end, kind, group, S, unpadded)])."""
    lcm = math.lcm(dp, 128)
    walk = list(range(len(params)))[::-1]
    groups, cur, pstart, bstart = [], [], 0, 0
    for i in walk:
        pstart = _pad(pstart, 64)
        pend = pstart + numels[i]
        cur.append(i)
        if bucket_size is not None and (pend - bstart) >= bucket_size:  # consulted once per param, as Megatron
            groups.append(cur)
            cur, bstart = [], _pad(pend, lcm)
            pstart = bstart
        else:
            pstart = pend
    if cur:
        groups.append(cur)
    index_map, buckets, index = {}, [], 0
    for gi, g in enumerate(groups):
        for kind in (None, "W4", "W6"):
            ps = [i for i in g if kinds[i] == kind]
            if not ps:
                continue
            start, bid = index, len(buckets)
            if kind is None:
                idx = start
                for i in ps:
                    idx = _pad(idx, 64)
                    index_map[i] = (idx, idx + numels[i], bid)
                    idx += numels[i]
                end, S = _pad(idx, lcm), None
            else:
                S, rel, _ = plan_weight_bucket([shapes[i] for i in ps], dp)
                for i, r in zip(ps, rel):
                    index_map[i] = (start + r, start + r + numels[i], bid)
                end = start + dp * S
            buckets.append((start, end, kind or "R", gi, S, sum(numels[i] for i in ps)))
            index = end
    return index_map, buckets


def check_layout(index_map, buckets, shapes, kinds, dp, align_rows=ALIGN_ROWS):
    """Every shard boundary inside a W4 / W6 weight lies on an ``align_rows``-row edge; buckets tile the buffer."""
    prev = 0
    for bid, (start, end, kind, _g, S, _u) in enumerate(buckets):
        assert start == prev and (end - start) % dp == 0 and start % 128 == 0, (bid, start, end)
        prev = end
        if kind == "R":
            continue
        assert end - start == dp * S
        for i, (s, e, b) in index_map.items():
            if b != bid:
                continue
            assert kinds[i] == kind
            for j in range(1, dp):
                cut = start + j * S
                if s < cut < e:
                    assert (cut - s) % (align_rows * shapes[i][1]) == 0, (bid, i, cut - s)
    return True
