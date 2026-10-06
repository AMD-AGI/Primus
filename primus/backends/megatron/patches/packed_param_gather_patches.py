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


# ── the buffer layout (a copy of Megatron's _ParamAndGradBuffer.__init__ with plan_layout as its layout pass) ──


def _check_source(orig_init):
    """The copied constructor below matches Megatron's at the time of copying; refuse to replace a changed one."""
    src = inspect.getsource(orig_init)
    sha = hashlib.sha256(src.encode()).hexdigest()
    assert sha == _MEGATRON_INIT_SHA_VALUE, (
        "packed_param_gather: Megatron's _ParamAndGradBuffer.__init__ changed (sha " + sha[:16] + "); "
        "re-copy its constructor into packed_param_gather_patches before using mxfp6_packed_param_gather"
    )


# sha256 of inspect.getsource(_ParamAndGradBuffer.__init__) of the Megatron the copy above was made from
# (third_party/Megatron-LM d3528a213).
_MEGATRON_INIT_SHA_VALUE = "4615e5d3371631907edb53d3435bdcd875bad273517634808b1a9b709c604acb"


def make_init(orig_init, kinds_of):
    """A ``_ParamAndGradBuffer.__init__`` that lays out the packed-gather buckets when ``kinds_of(params, names)``
    returns a kind for some parameter, and is Megatron's own otherwise."""
    import torch
    from megatron.core import parallel_state
    from megatron.core.distributed import param_and_grad_buffer as pgb

    def __init__(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        names = [param_to_name[p] for p in params]
        kinds = kinds_of(params, names)
        if not ddp_config.use_distributed_optimizer or not any(kinds):
            return orig_init(
                self, ddp_config, param_dtype, grad_dtype, params, data_parallel_group, bucket_size, param_to_name,
                gradient_scaling_factor, param_indices, nccl_ub, pg_collection,
            )
        assert not nccl_ub, "packed_param_gather: nccl_ub buffers are not supported"
        assert not any(pgb.is_float8tensor(p) or pgb.is_mxfp8tensor(p) for p in params)
        assert not any(getattr(p, "shared_embedding", False) for p in params)
        assert param_dtype == torch.bfloat16, param_dtype

        # ---- Megatron's header ----
        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(with_context_parallel=True)
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp
        self.ddp_config = ddp_config
        self.params = params
        self.param_indices = param_indices
        assert len(set(params)) == len(params)
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub
        self.buckets = []
        self.param_to_bucket = {}
        self.param_index_map = {}

        # ---- layout: plan_layout instead of Megatron's pass ----
        dp = self.data_parallel_world_size
        numels = [p.data.nelement() for p in params]
        shapes = [tuple(p.shape) if p.dim() == 2 else (p.nelement(), 1) for p in params]
        index_map, plan = plan_layout(list(range(len(params))), numels, kinds, shapes, bucket_size, dp)
        check_layout(index_map, plan, shapes, kinds, dp)
        self.bucket_indices = [(s, e) for s, e, *_ in plan]
        for i, (s, e, b) in index_map.items():
            self.param_index_map[params[i]] = (s, e, b)
        self.numel = plan[-1][1]
        self.numel_unpadded = sum(numels)
        assert self.numel % dp == 0

        # ---- Megatron's storage (the non-MXFP8, non-nccl_ub branch) ----
        self.param_data = torch.zeros(self.numel, dtype=self.param_dtype, device=torch.cuda.current_device(),
                                      requires_grad=False)
        self.grad_data = torch.zeros(self.numel, dtype=self.grad_dtype, device=torch.cuda.current_device(),
                                     requires_grad=False)
        self.grad_data_size = 0
        self.param_data_size = 0
        self.param_data_cpu = None

        # ---- map param.data / main_grad, then the buckets in plan order ----
        for p in params:
            s, _e, _b = self.param_index_map[p]
            new = self._get(p.data.shape, s, buffer_type=pgb.BufferType.PARAM)
            old = p.data
            p.data = new
            assert old._base is None
            p.data.detach().copy_(old)
            del old
            p.main_grad = self._get(p.data.shape, s, buffer_type=pgb.BufferType.GRAD)
        by_bucket = [[] for _ in plan]
        for p in params[::-1]:  # Megatron's walk order inside a bucket
            by_bucket[self.param_index_map[p][2]].append(p)
        for bid, (s, e, kind, group, S, unpadded) in enumerate(plan):
            bucket = self._new_bucket(bucket_params=by_bucket[bid], start_index=s, end_index=e,
                                      numel_unpadded=unpadded, bucket_id=bid)
            bucket.mxfp6_kind = kind  # "R" (bf16 gather) / "W6" / "W4" (packed gather)
            bucket.mxfp6_group = group  # the Megatron bucket it was split from
            bucket.mxfp6_shard = S
            self.buckets.append(bucket)
        log_rank_0(
            f"[packed_param_gather] {len(plan)} buckets from {len({p[3] for p in plan})}: padding "
            f"{(self.numel - self.numel_unpadded) / 1e6:.1f} M elements; "
            + ", ".join(f"{k} {sum(p[5] for p in plan if p[2] == k) / 1e9:.2f} B" for k in ("R", "W6", "W4"))
        )

    __init__._primus_packed_param_gather = True
    return __init__


def partition_buckets_grouped(orig_partition):
    """``partition_buckets`` that keeps a packed layout's sub-buckets (same ``mxfp6_group``) in one bucket group, so
    their gathers / reduce-scatters are coalesced exactly when the original bucket's would be."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    def partition_buckets(buffers, force_single_bucket_group=False):
        if force_single_bucket_group or not any(hasattr(b, "mxfp6_group") for buf in buffers for b in buf.buckets):
            return orig_partition(buffers, force_single_bucket_group)
        groups = []
        for buf in buffers:
            cur, gid = [], None
            for b in buf.buckets:
                g = getattr(b, "mxfp6_group", None)
                if cur and (g is None or g != gid):
                    groups.append(_ParamAndGradBucketGroup(cur, buf.ddp_config, buf.data_parallel_group,
                                                           buf.data_parallel_world_size))
                    cur = []
                cur.append(b)
                gid = g
            if cur:
                groups.append(_ParamAndGradBucketGroup(cur, buf.ddp_config, buf.data_parallel_group,
                                                       buf.data_parallel_world_size))
        return groups

    partition_buckets._primus_packed_param_gather = True
    return partition_buckets


@register_patch(
    "megatron.ddp.packed_param_gather",
    backend="megatron",
    phase="before_train",
    description="MXFP6 packed parameter all-gather: DDP bucket layout with row-aligned weight shards.",
    condition=lambda ctx: bool(getattr(get_args(ctx), "mxfp6_packed_param_gather", False)),
)
def patch_packed_param_gather_layout(ctx: PatchContext) -> None:
    import megatron.core.distributed.distributed_data_parallel as ddp_mod
    import megatron.core.distributed.param_and_grad_buffer as pgb

    args = get_args(ctx)
    n_joint = int(getattr(args, "num_joint_layers", 19))
    fc1 = bool(getattr(args, "mxfp6_fwd_fp4_single_fc1", False))
    l2 = bool(getattr(args, "mxfp6_fwd_fp4_single_linear2", False))
    jm = bool(getattr(args, "mxfp6_fwd_fp4_joint_mlp", False))

    def kinds_of(params, names):
        return [pack_kind(n, n_joint, fc1, l2, jm) if p.dim() == 2 else None for p, n in zip(params, names)]

    cur = pgb._ParamAndGradBuffer.__init__
    if getattr(cur, "_primus_packed_param_gather", False) or getattr(
        getattr(cur, "_inner", None), "_primus_packed_param_gather", False
    ):
        return
    head = getattr(cur, "_primus_ddp_head_bucket", False)
    megatron_init = cur._inner if head else cur
    _check_source(megatron_init)
    ours = make_init(megatron_init, kinds_of)
    if head:  # the head ramp's stand-in bucket_size then reaches our layout pass
        cur._inner = ours
    else:
        pgb._ParamAndGradBuffer.__init__ = ours
    grouped = partition_buckets_grouped(pgb.partition_buckets)
    pgb.partition_buckets = grouped
    ddp_mod.partition_buckets = grouped
    log_rank_0(
        f"[Patch:megatron.ddp.packed_param_gather] row-aligned weight buckets ({ALIGN_ROWS}-row edges; "
        f"n_joint {n_joint}, fwd FP4 fc1/l2/joint_mlp {fc1}/{l2}/{jm})"
    )
