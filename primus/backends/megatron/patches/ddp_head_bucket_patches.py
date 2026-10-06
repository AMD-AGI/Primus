###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ramp the DDP buckets that hold the first-forward parameters.

Megatron forms buckets walking the parameters in reverse (backprop) order and closes one every
``bucket_size`` elements, so the bucket holding the first layers is whatever remains at the end of
that walk. With overlapped parameter gathers that bucket's all-gather is waited on before the
first layer runs, and the next bucket's gather has only the head bucket's forward to hide behind;
after backward the same two buckets are the last reduce-scatters, exposed at the end of the step.

``--ddp-head-bucket-size H`` (elements, 0 = off) cuts the first-forward parameters into a ramp of
buckets H, H*r, H*r^2, ... (``--ddp-head-bucket-ramp r``) until a bucket would reach
``bucket_size``; the rest are bucketed as before. Each gather is then small enough to hide behind
the previous bucket's forward. Same parameters, same collectives, same arithmetic: only where the
bucket boundaries fall changes.

Megatron decides a boundary with one inline comparison per parameter,
``(param_end_index - bucket_start_index) >= bucket_size``, inside ``_ParamAndGradBuffer.__init__``.
The buffer is given a ``bucket_size`` that answers that comparison from a decision precomputed per
parameter; it asserts it was consulted exactly once per parameter, so a change to Megatron's loop
fails loudly instead of bucketing silently wrong.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.args.ddp_head_bucket",
    backend="megatron",
    phase="setup",
    description="Expose --ddp-head-bucket-size / --ddp-head-bucket-ramp on Megatron's distributed argparse group.",
)
def patch_ddp_head_bucket_args(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return

    orig = getattr(margs, "_add_distributed_args", None)
    if orig is None or getattr(orig, "_primus_ddp_head_bucket", False):
        return

    def _add_distributed_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="distributed")
        group.add_argument(
            "--ddp-head-bucket-size",
            type=int,
            default=0,
            help="Elements in the first-forward DDP bucket; later head buckets grow by --ddp-head-bucket-ramp "
            "up to --ddp-bucket-size. 0 keeps Megatron's bucketing.",
        )
        group.add_argument(
            "--ddp-head-bucket-ramp",
            type=float,
            default=1.4,
            help="Growth factor between consecutive head buckets (see --ddp-head-bucket-size).",
        )
        group.add_argument(
            "--ddp-param-gather-after-optimizer",
            type=int,
            default=0,
            help="With overlap_param_gather: dispatch the first N bucket groups' param all-gathers (forward order; -1 = "
            "all) right after the distributed optimizer's step instead of from the next forward's pre-hooks. 0 = off.",
        )
        return parser

    _add_distributed_args._primus_ddp_head_bucket = True
    margs._add_distributed_args = _add_distributed_args
    log_rank_0(
        "[Patch:megatron.args.ddp_head_bucket] added --ddp-head-bucket-size / --ddp-head-bucket-ramp / --ddp-param-gather-after-optimizer"
    )


def head_cuts(total: int, head: int, ramp: float, bucket_size: int):
    """Forward-order cut points of the head ramp: cumulative sizes head, head*ramp, ... while each bucket stays
    below ``bucket_size`` (and inside ``total``). Returns an ascending list; its last entry ends the ramp."""
    cuts, size, at = [], float(head), 0
    while size < bucket_size and at + int(size) < total:
        at += int(size)
        cuts.append(at)
        size *= ramp
    return cuts


def close_after(numels, head: int, ramp: float, bucket_size: int):
    """Per parameter in Megatron's walk order (reverse of ``numels``' forward order): does its bucket close after it.
    Every boundary is a forward-order cut point: the head ramp, then every ``bucket_size`` after it, so the partial
    remainder bucket lands at the TAIL (gathered last, reduce-scattered first, both hidden). Megatron's own walk would
    leave the remainder next to the ramp, where the following full-size gather cannot hide behind the small
    remainder's forward. Returned as a function of (walk index, fill); the fill is not used."""
    total = sum(numels)
    cuts = head_cuts(total, head, ramp, bucket_size)
    ramp_end = cuts[-1] if cuts else 0
    at = ramp_end + bucket_size
    while at < total:
        cuts.append(at)
        at += bucket_size
    # Elements still unassigned toward the head before / after each parameter of the walk.
    before, after, rem = [], [], total
    for n in reversed(numels):
        before.append(rem)
        rem -= n
        after.append(rem)

    def decide(i: int, fill: int) -> bool:
        b, a = before[i], after[i]
        return any(a <= c < b for c in cuts)

    return decide, cuts


class _HeadRampBucketSize:
    """Stands in for ``bucket_size`` inside one ``_ParamAndGradBuffer.__init__``: Megatron evaluates
    ``fill >= bucket_size`` once per parameter in walk order, which Python answers with ``__le__`` here."""

    def __init__(self, decide, n_params: int):
        self._decide, self._n, self.calls = decide, n_params, 0

    def __le__(self, fill):
        i = self.calls
        assert (
            i < self._n
        ), "ddp_head_bucket: Megatron consulted bucket_size more often than once per parameter"
        self.calls += 1
        return self._decide(i, int(fill))


@register_patch(
    "megatron.ddp.head_bucket",
    backend="megatron",
    phase="before_train",
    description="Cut the first-forward parameters into a ramp of small DDP buckets (exposed head gather / tail RS).",
    condition=lambda ctx: int(getattr(get_args(ctx), "ddp_head_bucket_size", 0) or 0) > 0,
)
def patch_ddp_head_bucket(ctx: PatchContext) -> None:
    import megatron.core.distributed.param_and_grad_buffer as pgb

    args = get_args(ctx)
    head, ramp = int(args.ddp_head_bucket_size), float(args.ddp_head_bucket_ramp)
    assert ramp >= 1.0, f"--ddp-head-bucket-ramp must be >= 1, got {ramp}"
    orig_init = pgb._ParamAndGradBuffer.__init__
    if getattr(orig_init, "_primus_ddp_head_bucket", False):
        return

    def __init__(
        self, ddp_config, param_dtype, grad_dtype, params, data_parallel_group, bucket_size, *a, **kw
    ):
        inner = __init__._inner  # Megatron's, or a layout patch installed later (packed_param_gather)
        if bucket_size is None or head <= 0:
            return inner(self, ddp_config, param_dtype, grad_dtype, params, data_parallel_group, bucket_size, *a, **kw)
        numels = [p.data.nelement() for p in params]
        decide, cuts = close_after(numels, head, ramp, int(bucket_size))
        stand_in = _HeadRampBucketSize(decide, len(params))
        inner(self, ddp_config, param_dtype, grad_dtype, params, data_parallel_group, stand_in, *a, **kw)
        assert stand_in.calls == len(params), (
            f"ddp_head_bucket: bucket_size consulted {stand_in.calls} times for {len(params)} params; "
            "Megatron's bucketing loop changed, update this patch"
        )
        sizes = [e - s for s, e in self.bucket_indices]
        log_rank_0(
            f"[Patch:megatron.ddp.head_bucket] {param_dtype}/{grad_dtype}: {len(sizes)} buckets (walk order, last = "
            f"first forward): {[round(x / 1e6, 1) for x in sizes]} M elements; head ramp cuts at "
            f"{[round(c / 1e6, 1) for c in cuts]} M"
        )

    __init__._primus_ddp_head_bucket = True
    __init__._inner = orig_init
    pgb._ParamAndGradBuffer.__init__ = __init__
    log_rank_0(f"[Patch:megatron.ddp.head_bucket] head bucket {head} elements, ramp {ramp}")


@register_patch(
    "megatron.optimizer.param_gather_after_step",
    backend="megatron",
    phase="before_train",
    description="Dispatch the head bucket groups' param all-gathers at the end of the optimizer step.",
    condition=lambda ctx: int(getattr(get_args(ctx), "ddp_param_gather_after_optimizer", 0) or 0) != 0,
)
def patch_param_gather_after_step(ctx: PatchContext) -> None:
    """With ``overlap_param_gather`` the first bucket group's all-gather is dispatched by the next forward's first
    pre-hook (this Megatron's ``zero_grad`` dispatches nothing), and each later group by the previous group's wait. That
    leaves the GPU's comm stream idle from the optimizer step's copy-back until the host reaches the forward. Dispatching
    the first N groups (forward order: ``reversed(bucket_groups)``) at the end of ``step_with_ready_grads`` starts them as
    soon as their params are copied back. The bookkeeping is Megatron's own: ``finish_grad_sync`` cleared
    ``param_gather_dispatched`` before the step, ``start_param_sync`` sets it, and the forward's ``finish_param_sync``
    waits on the outstanding handle and dispatches the next group only if it is not already dispatched."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    n = int(get_args(ctx).ddp_param_gather_after_optimizer)
    orig = DistributedOptimizer.step_with_ready_grads
    if getattr(orig, "_primus_param_gather_after_step", False):
        return

    stale = [0]

    def step_with_ready_grads(self):
        ok = orig(self)
        if ok and self.ddp_config.overlap_param_gather and not self.ddp_config.use_megatron_fsdp:
            for chunk in self.model_chunks:
                groups = list(reversed(chunk.bucket_groups))
                for g in groups if n < 0 else groups[:n]:
                    if g.param_gather_dispatched:
                        continue
                    if g.param_gather_handle is not None:
                        # A gather of the previous params that nothing waited on: finish it before gathering the
                        # updated ones (Megatron's own force_sync path does the same). Logged: in a normal step the
                        # forward consumes every dispatch, so this should only occur around warmup / eval paths.
                        g.param_gather_handle.wait()
                        g.param_gather_handle = None
                        stale[0] += 1
                        if stale[0] <= 3:
                            log_rank_0(
                                f"[Patch:megatron.optimizer.param_gather_after_step] waited a stale param "
                                f"gather before re-dispatch ({stale[0]})"
                            )
                    g.start_param_sync()
        return ok

    step_with_ready_grads._primus_param_gather_after_step = True
    DistributedOptimizer.step_with_ready_grads = step_with_ready_grads
    log_rank_0(
        f"[Patch:megatron.optimizer.param_gather_after_step] dispatching {n} head bucket group(s) after the step"
    )
