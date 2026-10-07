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

ALIGN_ROWS = 256  # the GEMM operands are range-closed per 256 rows (core/extensions/mxfp6_packed_gather)
_LIN = re.compile(
    r"layers\.(\d+)\.(self_attention\.(linear_qkv|linear_proj|added_linear_qkv|added_linear_proj)|"
    r"mlp\.linear_fc[12]|context_mlp\.linear_fc[12])\.weight$"
)


def _pad(x: int, d: int) -> int:
    return -(-x // d) * d


def pack_kind(
    name: str, n_joint: int, fwd_fp4_fc1: bool, fwd_fp4_l2: bool, fwd_fp4_joint_mlp: bool, joint_mlp_parts="all"
):
    """``"W6"``, ``"W4"`` or None (gathered as bf16) for a parameter, from its name and the forward-FP4 gates
    (``joint_mlp_parts``: mxfp6_fwd_fp4_joint_mlp_parts)."""
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
        from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import joint_mlp_parts as _parts

        part = f"{'img' if what.startswith('mlp.') else 'txt'}_{what.rsplit('_', 1)[1]}"
        if part in _parts(joint_mlp_parts):
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
    jp = str(getattr(args, "mxfp6_fwd_fp4_joint_mlp_parts", "all"))

    def kinds_of(params, names):
        return [pack_kind(n, n_joint, fc1, l2, jm, jp) if p.dim() == 2 else None for p, n in zip(params, names)]

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
    patch_bucket_group_sync()
    if bool(getattr(args, "mxfp6_packed_param_gather_fused_adam", False)):
        patch_fused_adam_owner_pack()
    grouped = partition_buckets_grouped(pgb.partition_buckets)
    pgb.partition_buckets = grouped
    ddp_mod.partition_buckets = grouped
    log_rank_0(
        f"[Patch:megatron.ddp.packed_param_gather] row-aligned weight buckets ({ALIGN_ROWS}-row edges; "
        f"n_joint {n_joint}, fwd FP4 fc1/l2/joint_mlp {fc1}/{l2}/{jm})"
    )


# ── the gather: W6 / W4 buckets gather their packed planes (runtime: core/extensions/mxfp6_packed_gather) ──


def refresh_packed_params(ddp):
    """Rebuild the gathered packs of a DDP model chunk from its bf16 rows (after the bf16 parameters were rewritten,
    e.g. restored after warmup steps). Bound by ``patch_bucket_group_sync``; a no-op while the gate is off."""


def _packed_states(group):
    """The PackedBucket of each W6 / W4 bucket of a bucket group (built on first use)."""
    st = getattr(group, "_ppg_states", None)
    if st is None:
        from primus.backends.megatron.core.extensions.mxfp6_packed_gather import PackedBucket
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import ppg_formats

        dp = group.intra_distributed_optimizer_instance_size
        rank = group.intra_distributed_optimizer_instance_rank
        fmts = ppg_formats()
        st = {id(b): PackedBucket(b, dp, rank, fmts) for b in group.buckets if getattr(b, "mxfp6_kind", "R") != "R"}
        group._ppg_states = st
        for b in st.values():  # the optimizer-step side (patch_fused_adam_owner_pack): this rank's pieces
            _OWNED.update({(b.items[i][1].data[ra:rb].data_ptr(), (rb - ra) * b.items[i][4]): (b, i, ra, rb)
                           for i, ra, rb in b.owner_pieces()})
    return st


# (data_ptr, numel) of this rank's owned piece of a packed weight -> (PackedBucket, item, ra, rb). The distributed
# optimizer's parameter for that piece is a view of exactly these elements (a 256-row-aligned shard of one weight).
_OWNED = {}


def patch_fused_adam_owner_pack():
    """``mxfp6_packed_param_gather_fused_adam``: Transformer Engine's FusedAdam.step leaves out the parameters that
    are owned pieces of packed weights, and each of those takes the same Adam step inside its owner pack
    (``PackedBucket.pack_piece(adam=...)``, Turbo ``quantize_mx_dual_out_adam``); the owner pack then skips them.
    Only for the configuration that kernel reproduces (bf16 parameters with store_param_remainders, fp32 moments,
    bf16 gradients, not capturable) and once TE has created the parameter's state; anything else stays in TE."""
    import torch
    from transformer_engine.pytorch.optimizers import FusedAdam

    if getattr(FusedAdam.step, "_primus_fused_owner_pack", False):
        return
    orig_step = FusedAdam.step

    def _fusable(opt, p):
        st = opt.state.get(p)
        g = getattr(p, "decoupled_grad", None) if opt.use_decoupled_grad else p.grad
        return (
            st is not None
            and (p.data_ptr(), p.numel()) in _OWNED
            and p.dtype == torch.bfloat16
            and g is not None
            and g.dtype == torch.bfloat16
            and g.is_contiguous()
            and st.get("master_param") is not None
            and st["master_param"].dtype == torch.int16
            and st["exp_avg"].dtype == torch.float32
            and st["exp_avg_sq"].dtype == torch.float32
        )

    def _why_not(opt, p):
        st = opt.state.get(p)
        g = getattr(p, "decoupled_grad", None) if opt.use_decoupled_grad else p.grad
        if st is None:
            return "no state"
        if (p.data_ptr(), p.numel()) not in _OWNED:
            return "not an owned piece"
        if g is None:
            return "no grad"
        return f"dtypes p {p.dtype} g {g.dtype} rem {getattr(st.get('master_param'), 'dtype', None)}"

    def step(self, closure=None, grad_scaler=None):
        if not _OWNED or grad_scaler is not None or self.capturable or not self.store_param_remainders:
            if not getattr(self, "_ppg_fa_logged", False) and _OWNED:
                self._ppg_fa_logged = True
                log_rank_0(f"[mxfp6 fused adam+pack] not engaged: grad_scaler {grad_scaler is not None}, "
                           f"capturable {self.capturable}, store_param_remainders {self.store_param_remainders}")
            return orig_step(self, closure, grad_scaler)
        if getattr(self, "_ppg_fa_logged", 0) < 2:  # the first two steps (step 1 has no state yet)
            self._ppg_fa_logged = getattr(self, "_ppg_fa_logged", 0) + 1
            from collections import Counter

            ps = [p for grp in self.param_groups for p in grp["params"]]
            n = sum(_fusable(self, p) for p in ps)
            why = Counter(_why_not(self, p) for p in ps if not _fusable(self, p))
            log_rank_0(f"[mxfp6 fused adam+pack] {n} of {len(ps)} optimizer params fused "
                       f"({len(_OWNED)} owned pieces); the rest: {dict(why)}")
        held = []
        for group in self.param_groups:
            mine = [p for p in group["params"] if _fusable(self, p)]
            if mine:
                ids = {id(p) for p in mine}
                held.append((group, group["params"], mine))
                group["params"] = [p for p in group["params"] if id(p) not in ids]
        try:
            loss = orig_step(self, closure, grad_scaler)
        finally:
            for group, params, _ in held:
                group["params"] = params
        for group, params, mine in held:
            if len(params) == len(mine):  # TE skipped the emptied group and so did not count its step
                group["step"] = group.get("step", 0) + 1
            b1, b2 = group["betas"]
            hyper = dict(lr=float(group["lr"]), beta1=b1, beta2=b2, eps=group["eps"],
                         weight_decay=group["weight_decay"], step=int(group["step"]), adamw=bool(self.adam_w_mode),
                         bias_correction=bool(group["bias_correction"]))
            for p in mine:
                b, idx, ra, rb = _OWNED[(p.data_ptr(), p.numel())]
                st = self.state[p]
                g = p.decoupled_grad if self.use_decoupled_grad else p.grad
                # the step number the next owner_pack runs under (its SR seed)
                b.pack_piece(idx, ra, rb, b.step + 1,
                             adam=(g, st["exp_avg"], st["exp_avg_sq"], st["master_param"], hyper))
        return loss

    step._primus_fused_owner_pack = True
    FusedAdam.step = step


class _Handles:
    """One param-gather handle over several async collectives."""

    def __init__(self, works):
        self.works = [w for w in works if w is not None]

    def wait(self):
        for w in self.works:
            w.wait()


def patch_bucket_group_sync():
    import torch
    from torch.distributed import _coalescing_manager

    import megatron.core.distributed.param_and_grad_buffer as pgb

    BG = pgb._ParamAndGradBucketGroup
    if getattr(BG.start_param_sync, "_primus_packed_param_gather", False):
        return
    orig_start = BG.start_param_sync

    def _packed(group):
        return any(getattr(b, "mxfp6_kind", "R") != "R" for b in group.buckets)

    def start_param_sync(self, force_sync: bool = False):
        if not self.ddp_config.use_distributed_optimizer or not _packed(self):
            return orig_start(self, force_sync)
        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None
        # Without the forward pre-hooks nothing waits on an overlapped gather before the next forward (Megatron's
        # first iteration, warmup steps), so the gather is synchronous then.
        async_op = self.ddp_config.overlap_param_gather and not force_sync and getattr(self, "_ppg_hooked", False)
        _gather(self, async_op, packed_only=False)
        self.param_gather_dispatched = True

    def _gather(self, async_op, packed_only):
        """Owner packs into this rank's plane shards, then one coalesced all-gather: planes for the W6 / W4
        buckets, bf16 for the rest (as Megatron)."""
        states = _packed_states(self)
        for st in states.values():
            st.owner_pack()
        grp = self.intra_distributed_optimizer_instance_group
        rank = self.intra_distributed_optimizer_instance_rank
        works = []
        for st in states.values():  # the SR dgrad copies: one draw per destination (see mxfp6_packed_gather)
            for out, inp in st.a2a_ops():
                works.append(torch.distributed.all_to_all_single(out, inp, group=grp, async_op=async_op))
        with _coalescing_manager(grp, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.buckets):
                st = states.get(id(bucket))
                if st is not None:
                    for out, inp in st.gather_ops():
                        pgb.dist_all_gather_func(out, inp, group=grp, async_op=async_op)
                    continue
                if packed_only:
                    continue
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = pgb.shard_buffer(
                        bucket.param_data, self.intra_distributed_optimizer_instance_size
                    )
                pgb.dist_all_gather_func(
                    bucket.param_data, self.cached_param_buffer_shard_list[idx][rank], group=grp, async_op=async_op
                )
        self.param_gather_handle = _Handles([cm] + works) if async_op else None

    start_param_sync._primus_packed_param_gather = True
    BG.start_param_sync = start_param_sync

    # The forward pre-hooks' state, on every bucket group of the DDP. Turning them off with param_sync=False declares
    # the bf16 parameters valid as they are (Megatron's first iteration): the packs are rebuilt from them (the owners'
    # rows; no bf16 gather), since a forward without the hooks reads the packs as they stand.
    import megatron.core.distributed.distributed_data_parallel as ddp_mod

    DDP = ddp_mod.DistributedDataParallel
    orig_enable, orig_disable = DDP.enable_forward_pre_hook, DDP.disable_forward_pre_hook

    def _groups(ddp):
        return list(getattr(ddp, "bucket_groups", [])) + list(getattr(ddp, "expert_parallel_bucket_groups", []))

    def enable_forward_pre_hook(self, *a, **k):
        orig_enable(self, *a, **k)
        for g in _groups(self):
            g._ppg_hooked = True

    def disable_forward_pre_hook(self, param_sync: bool = True):
        for g in _groups(self):
            g._ppg_hooked = False
        orig_disable(self, param_sync=param_sync)
        if not param_sync:
            _refresh(self)

    def _refresh(ddp):
        if not ddp.ddp_config.use_distributed_optimizer:
            return
        for g in _groups(ddp):
            if not _packed(g):
                continue
            if g.param_gather_handle is not None:  # superseded: the packs are rebuilt from the current rows below
                g.param_gather_handle.wait()
                g.param_gather_handle = None
            _gather(g, async_op=False, packed_only=True)

    global refresh_packed_params
    refresh_packed_params = _refresh

    BG._ppg_hooked = False
    DDP.enable_forward_pre_hook = enable_forward_pre_hook
    DDP.disable_forward_pre_hook = disable_forward_pre_hook
