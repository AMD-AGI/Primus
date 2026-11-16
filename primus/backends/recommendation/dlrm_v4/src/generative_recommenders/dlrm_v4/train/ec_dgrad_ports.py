"""E-X2b dh ports: the EC side of the fused preprocessor-L1-dgrad + EC reduce.

DEFAULT OFF (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE; kernels, gate and the pair-0
protocol in ops/triton/triton_ec_dgrad_segreduce.py). Imported only from
ec_bf16_output._ex2b_plan, and only with the gate on, so the knob-off import
graph of ec_bf16_output is unchanged. Design: proposals/e_x2b/
INTEGRATION_PLAN.md sections 2.1-2.6 (actckpt-authored, rccl-reviewed).

FORWARD (plan 2.2). When the grouped EC call has exactly the two hoisted
members E-X2b covers (the CAT_FUSE artist|album member and the merged item
member, W = 512, bf16 out, grad mode on, no tracing), _merged_construct calls
ECExpandGroupPorts instead of _ECExpandGroup: the same forward plus one
stride-0 [M, 256] bf16 "dh port" output per member, tagged on the tensor pair 0
must receive (the item member output; big = the [M, 1024] cat view).
Otherwise plan() returns None and the shipped call runs unchanged.

BACKWARD (plan 2.5). Pair 0 (ops/fused_linear_norm.py, ops/preproc_fp8_l2.py,
ops/preproc_grouped_tail.py) returns dh for its port instead of dX for x when
it can (route). ECExpandGroupPorts.backward then either
  * fused: both members routed dh, neither member output has another consumer:
    one (order, offsets) build (the shipped joint sort, else per-member
    build_segments), the invariant check on the first 2 calls, and
    dgrad_segreduce -> the fp32 accumulator the shipped body (hook 2 in
    ec_bf16_output) seeds; only the kept keys' scatter_add_ runs there; or
  * late fallback (anything else, or a raising fused call, which also latches
    E-X2b off): dX = torch.mm(dh, W1) (the shipped dgrad GEMM, same shape and
    pin) is rebuilt per routed member and added to that member's incoming
    gradient (autograd's own sum for two consumers); the shipped body then
    runs the joint sort and segment reduce exactly as today.

NUMERICS: declared fp32 reduction-order class (fused path; see the kernel
module). The late fallback is the shipped arithmetic.
"""

import logging
import os
import weakref
from typing import Any, List, Optional, Sequence, Tuple

import torch
from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

logger: logging.Logger = logging.getLogger(__name__)

# First-use invariant checks (plan 2.6 guard 1) and debug self-checks.
_CHECKS_LEFT: int = 2
_SELF_CHECKS_LEFT: int = 2
_FUSED_LOGGED: bool = False
# Devices plan() engages on (the kernels are HIP-only; the CPU test widens it).
_DEVICES: Tuple[str, ...] = ("cuda",)


class ECExpandGroupPorts(ebo._ECExpandGroup):
    """_ECExpandGroup plus one stride-0 dh port output per E-X2b record.

    forward(embeddings, out_dtype, casts, records, *idxs) -> (*member outputs,
    *ports): the parent forward unchanged (ctx.set_materialize_grads(False)
    is inherited, so an unused port costs no [M, 256] zero gradient), then the
    ports. backward: ec_backward below, then the parent backward with its
    accumulator seeded through ctx.ex2b_acc (ec_bf16_output hook 2).
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # type: ignore[override]
        ctx,
        embeddings: torch.Tensor,
        out_dtype: torch.dtype,
        casts: Tuple[bool, ...],
        records: Tuple[xs.PortRecord, ...],
        *idxs: torch.Tensor,
    ):
        outs = ebo._ECExpandGroup.forward(ctx, embeddings, out_dtype, casts, *idxs)
        ctx.ex2b_records = records
        # The E-M3a proxy pattern (preproc_fp8_l2._LinearSwishLNEmitFunction):
        # one storage element; its gradient arrives full-size from pair 0.
        ports = tuple(
            torch.empty_strided((r.rows, xs.HID), (0, 0), dtype=out_dtype,
                                device=embeddings.device)
            for r in records
        )
        return tuple(outs) + ports

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grads):  # type: ignore[override]
        n = len(ctx.casts)
        acc0, member_grads = ec_backward(ctx, list(grads[:n]), grads[n:])
        ctx.ex2b_acc = acc0  # seeds the shipped body's accumulator (hook 2)
        try:
            res = ebo._ECExpandGroup.backward(ctx, *member_grads)
        finally:
            ctx.ex2b_acc = None  # never keep [U, 512] across a retain_graph backward
        return res[:3] + (None,) + res[3:]  # + None for `records`


class _Plan:
    """The two records of one EC invocation; holds the ports only until they
    are tagged (lifetime of _merged_construct)."""

    def __init__(self, cat_slot: int, cat_rows: int, item_slot: int, item_rows: int) -> None:
        self.cat = xs.PortRecord("cat", cat_slot, cat_rows, 2 * xs.WIDTH)
        self.item = xs.PortRecord("item", item_slot, item_rows, xs.WIDTH)
        self.records = tuple(sorted((self.cat, self.item), key=lambda r: r.member))
        self._ports: dict = {}

    def apply(
        self, embeddings: torch.Tensor, out_dtype: torch.dtype,
        casts: Tuple[bool, ...], idxs: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        outs = ECExpandGroupPorts.apply(embeddings, out_dtype, casts, self.records, *idxs)
        n = len(idxs)
        members = list(outs[:n])
        self._ports = {r.kind: p for r, p in zip(self.records, outs[n:])}
        values = members[self.item.member]
        _tag(values, self.item, self._ports.pop("item"), (values,))
        return members

    def tag_cat(self, flat: torch.Tensor, big: torch.Tensor) -> None:
        _tag(big, self.cat, self._ports.pop("cat"), (flat, big))


def _tag(x: torch.Tensor, rec: xs.PortRecord, port: torch.Tensor, watch) -> None:
    setattr(x, xs.PORT_ATTR, (rec, port))
    rec.out_ref = weakref.ref(x)
    rec.watch = tuple(weakref.ref(t) for t in watch)


def plan(
    embeddings: torch.Tensor, out_dtype: torch.dtype, casts: Sequence[bool],
    idxs: Sequence[torch.Tensor], cat_slot: int, w: int,
) -> Optional[_Plan]:
    """A _Plan when this grouped EC call is exactly E-X2b's (plan 2.2), else
    None (the shipped _ECExpandGroup call; silent, since other EC calls of the
    step legitimately decline). Ports are planned here, outside the Function,
    because grad mode reads False inside Function.forward."""
    if not xs.enabled():
        return None
    if not torch.is_grad_enabled() or not embeddings.requires_grad:
        return None  # eval / no_grad: no port, no attribute, byte-identical
    if xs.tracing(embeddings) or xs._graph_task_id() != -1:
        return None
    if cat_slot < 0 or len(ebo._CAT_GROUP) != 2:
        return None
    if w != xs.WIDTH or out_dtype != torch.bfloat16 or embeddings.device.type not in _DEVICES:
        return None
    hoisted = [i for i, c in enumerate(casts) if c]
    if len(hoisted) != 2 or cat_slot not in hoisted:
        return None
    item_slot = hoisted[0] if hoisted[1] == cat_slot else hoisted[1]
    n_cat = idxs[cat_slot].numel()
    if n_cat % 2:
        return None
    return _Plan(cat_slot, n_cat // 2, item_slot, idxs[item_slot].numel())


def _segments(idxs: Sequence[torch.Tensor], U: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """(order, offsets) per member, in the given (member) order: the shipped
    joint sort (ec_bf16_output._ec_joint_segments ignores g; its fail-closed
    latch applies unchanged), else the per-member shipped build."""
    segs = ebo._ec_joint_segments([(i, None) for i in idxs], U)
    if segs is None:
        from generative_recommenders.dlrm_v4.train import ec_segment_reduce as esr

        segs = [esr.build_segments(i, U) for i in idxs]
    return list(segs)


def _fused(ctx, routed: List[Tuple[xs.PortRecord, torch.Tensor]]) -> torch.Tensor:
    global _CHECKS_LEFT, _SELF_CHECKS_LEFT, _FUSED_LOGGED
    idxs = ctx.saved_tensors
    U = ctx.num_rows
    routed = sorted(routed, key=lambda rg: rg[0].member)  # member order, as the shipped sort
    segs = _segments([idxs[r.member] for r, _ in routed], U)
    mem = {}
    for (r, dh), (order, offsets) in zip(routed, segs):
        mem[r.kind] = xs.Member(dh, r.w1, idxs[r.member], order, offsets)
    cat, item = mem["cat"], mem["item"]
    if _CHECKS_LEFT > 0:
        why = xs.check_invariants(cat, item, U)
        if why is not None:
            raise RuntimeError(why)
        if _CHECKS_LEFT == 2:
            xs._say(f"{xs.TAG} invariants OK (overlap rows 0, mixed-half segments 0)")
        _CHECKS_LEFT -= 1
    acc = xs.dgrad_segreduce(cat, item, U)
    if not _FUSED_LOGGED:
        _FUSED_LOGGED = True
        xs._say(
            f"{xs.ARMED_PREFIX} fused -- preprocessor L1 dgrads fused into the grouped "
            f"EC backward (cat T={cat.idx.numel()}, item T={item.idx.numel()}, U={U}; "
            f"{xs.LAST_PASS_IMPL} x2 + _ecx_combine + _ecx_zero_uncovered; kill switch "
            f"{xs.ENV_VAR}=0)"
        )
    if _SELF_CHECKS_LEFT > 0 and os.environ.get(xs.CHECK_ENV_VAR, "0") == "1":
        _SELF_CHECKS_LEFT -= 1
        _self_check(acc, cat, item, U)
    return acc


def _self_check(acc: torch.Tensor, cat: xs.Member, item: xs.Member, U: int) -> None:
    """Debug (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_CHECK=1, first 2 calls): the late
    fallback's arithmetic member by member (peak below today's: one dX live),
    then the max row-relative difference over covered rows. Logs only."""
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import segment_reduce_rows

    ref = None
    for m, kind in ((cat, "cat"), (item, "item")):
        dx = torch.mm(m.dh, m.w1)
        if kind == "cat":
            dx = dx.view(-1, xs.WIDTH)
        ref = segment_reduce_rows(dx, m.idx, U, order=m.order, offsets=m.offsets,
                                  accumulate_into=ref)
        del dx
    assert ref is not None
    covered = (cat.offsets[1:] > cat.offsets[:-1]) | (item.offsets[1:] > item.offsets[:-1])
    diff = (acc - ref).abs().amax(dim=1)
    scale = ref.abs().amax(dim=1).clamp_min(1e-30)
    rel = torch.where(covered, diff / scale, torch.zeros_like(diff))
    unc = (~covered) & (acc.view(torch.int32) != 0).any(dim=1)
    vals = torch.stack([rel.max(), diff.max(), unc.sum().to(rel.dtype),
                        (~torch.isfinite(acc)).sum().to(rel.dtype)]).tolist()
    xs._say(
        f"{xs.TAG} self-check vs shipped reduce of rebuilt dX: max row rel diff "
        f"{vals[0]:.3e}, max abs diff {vals[1]:.3e}, uncovered rows not +0.0 "
        f"{int(vals[2])}, non-finite {int(vals[3])} (U={U})"
    )


def _late_fallback(routed, member_grads: List[Optional[torch.Tensor]], why: str) -> None:
    """dX = dh @ W1 with the shipped dgrad GEMM (aten::mm, same shape, same
    pin), added to the member's incoming gradient (plan 2.5)."""
    xs._once("fallback", why, f"{xs.TAG} FALLBACK ({why}) -> dX = dh @ W1 rebuilt "
             "with the shipped GEMM + the shipped EC reduce")
    for rec, dh in routed:
        dx = torch.mm(dh, rec.w1)
        if rec.kind == "cat":
            dx = dx.view(2 * rec.rows, xs.WIDTH)  # row 2t = artist, 2t+1 = album
        g = member_grads[rec.member]
        member_grads[rec.member] = dx if g is None else g + dx


def ec_backward(
    ctx: Any, member_grads: List[Optional[torch.Tensor]], port_grads: Sequence[Any]
) -> Tuple[Optional[torch.Tensor], List[Optional[torch.Tensor]]]:
    """(seed accumulator or None, member grads for the shipped body)."""
    routed = [(r, g) for r, g in zip(ctx.ex2b_records, port_grads) if g is not None]
    if not routed:
        return None, member_grads  # nothing routed: exactly the shipped path
    why = None
    if len(routed) != 2 or {r.kind for r, _ in routed} != {"cat", "item"}:
        why = "only one member routed dh"
    elif any(member_grads[r.member] is not None for r, _ in routed):
        why = "a member output has another consumer"
    elif not xs.enabled():
        why = "latched off"
    elif torch.is_grad_enabled():
        why = "create_graph backward"
    if why is None:
        try:
            return _fused(ctx, routed), member_grads
        except Exception as exc:  # fail closed: this step and every later one
            xs.latch_off("ec backward", exc)
            why = f"fused call raised {type(exc).__name__}"
    _late_fallback(routed, member_grads, why)
    return None, member_grads
