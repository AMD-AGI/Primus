# pyre-strict
"""DLRM_HSTU_FP8_ACTION_PREFILL: run the embedding-independent action branch of
ContextualPreprocessor inside the forward embedding all-to-all window.

WHERE THE MAIN STREAM WAITS. Under the pipeline, DlrmHSTU.preprocess's call
of the sharded EmbeddingCollection (PipelinedForward) launches the lookup, the
fwd permute and the sequence all-to-all and returns a LazyAwaitable. The first
touch of that awaitable -- ``seq_embeddings_dict[k]`` in the comprehension at
the end of DlrmHSTU.preprocess -- runs EmbeddingCollectionAwaitable._wait_impl
-> SequenceEmbeddingsAwaitable -> Request._wait_impl -> All2All_Seq_Req_Wait
.forward, whose ``myreq.req.wait()`` (torchrec/distributed/comm_ops.py) makes
the main stream wait for the collective; the unbucketize/EC expand gather
follows. Kernels issued before that touch run while the all-to-all is in
flight; everything after it queues behind it. ship10 traces: the main stream
idles there 2.70-2.93 ms on the 4-node critical rank, 1.22-1.40 at 2 nodes,
0.46-0.57 at 1 node.

WHAT MOVES. The action branch of ContextualPreprocessor.forward reads batch
payloads (the UIH action bitmask), lengths and host ints only -- ActionEncoder
uses seq_embeddings for .size(0), nothing else:
    target/seq/uih offsets -> ActionEncoder -> action MLP pair 1
    (Linear 24->256 + SwishLN)                                   "part A"
    and, with DLRM_HSTU_FP8_ACTION_PREFILL_L2 (absent = on; F1 container
    only), pair 2's two bf16 casts + addmm (256->512)             "L2"
DlrmHSTU.preprocess issues them right before the awaitable touch; the
preprocessor picks them up where the branch used to run, and only pair 2's
norm (+ the F2 residual epilogue) still runs there -- or, with the grouped
LayerNorm tail (item (30)), the prefilled head joins the group and only the
grouped LayerNorm launch runs at the end of the MLP section.

BITWISE. The same ops on the same operands under the same autocast state, on
the same stream, issued earlier: no RNG op in the branch, no host sync
(sum(num_candidates) comes from fx_length_stats' memo, or the prefill is not
issued), the GEMM pins for this token count are installed first exactly as the
preprocessor does (ops/blas_solution_pins.py), and the Triton configs are keyed
by shape/dtype, not by call order. The consumer re-checks every host-visible
input (row count, max lengths, identity of num_targets and of the action
payload, autocast/grad state, parameter versions) and falls back to the
original order on any mismatch.

BACKWARD ORDER. Autograd runs ready nodes in decreasing sequence number, and
sequence numbers follow creation order. Built early with plain autograd, the
branch's backward would move from right after pair 2's backward to after the
embedding all-to-all backward launch, and the DDP bucket of its parameters with
it. So the prefill output is detached and re-attached at the original position
by _Bridge, whose backward runs the prefill graph synchronously (reentrant
torch.autograd.backward, as checkpoint(use_reentrant=True) does) exactly when
autograd would have run it: same kernels, same order, same AccumulateGrad /
DDP-hook order. Pair 2 keeps its own autograd node and backward -- its
_LinearNormFunction, or, with ship11 item (30) armed, the grouped LayerNorm
tail node (ops/preproc_grouped_tail.py, GroupedTail.head_prefilled) -- and
only receives its Linear's results precomputed (PrefilledLinear).

Single-use and per-forward: DlrmHSTU.preprocess clears the slot, issues, and
ContextualPreprocessor.forward takes it. Training with grad only; eval,
inference, fx tracing and scripting run the original order.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.fused_linear_norm import (
    FusedLinearNormMLP,
    linear_fwd,
)
from generative_recommenders.ops.triton import preproc_fp8_snr
from torch.autograd.profiler import record_function

logger: logging.Logger = logging.getLogger(__name__)

ENV: str = "DLRM_HSTU_FP8_ACTION_PREFILL"
L2_ENV: str = "DLRM_HSTU_FP8_ACTION_PREFILL_L2"

_pending: "Optional[ActionPrefill]" = None
_logged: Set[str] = set()
# Per-process counts (tests / debugging): issued, used, fallback.
stats: Dict[str, int] = {"issued": 0, "used": 0, "fallback": 0}


def enabled() -> bool:
    """Master switch, read per call; absent = off (yambda_5b.gin turns it on)."""
    return os.environ.get(ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def l2_enabled() -> bool:
    """Also prefill pair 2's Linear (F1 container only); absent = on."""
    return os.environ.get(L2_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _rank0() -> bool:
    d = torch.distributed
    return not (d.is_available() and d.is_initialized()) or d.get_rank() == 0


def _note(key: str, warn: bool, msg: str, *args: object) -> None:
    """Log once per process per key (warnings on every rank, info on rank 0)."""
    if key in _logged:
        return
    _logged.add(key)
    if warn:
        logger.warning(msg, *args)
    elif _rank0():
        logger.info(msg, *args)


def decline(reason: str) -> None:
    _note(
        "decline:" + reason, True,
        "[action-prefill] not issued (%s); this forward runs the original "
        "order", reason,
    )


def _amp_state(device_type: str) -> Tuple[bool, torch.dtype, bool]:
    return (
        torch.is_autocast_enabled(device_type),
        torch.get_autocast_dtype(device_type),
        torch.is_grad_enabled(),
    )


def _same_storage_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.dtype == b.dtype
        and a.device == b.device
        and a.shape == b.shape
        and a.stride() == b.stride()
    )


class PrefilledLinear:
    """linear_fwd() of one (Linear, Norm) pair, issued early; taken once by
    _LinearNormFunction.forward (ops/fused_linear_norm.py) or by the grouped
    tail node _GroupedLinearLNTailFunction.forward (ops/preproc_grouped_tail.py)."""

    def __init__(
        self, x: torch.Tensor, lin_w: torch.Tensor, lin_b: torch.Tensor
    ) -> None:
        x = x.contiguous()
        self._x: torch.Tensor = x
        self._x_version: int = x._version
        self._lin_w: torch.Tensor = lin_w
        self._lin_b: torch.Tensor = lin_b
        self._versions: Tuple[int, int] = (lin_w._version, lin_b._version)
        self._state: Tuple[bool, torch.dtype, bool] = _amp_state(x.device.type)
        # _LinearNormFunction.forward runs its Linear with grad disabled.
        with torch.no_grad():
            self._out: Optional[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = linear_fwd(x, lin_w, lin_b)
        self.used: bool = False

    def take(
        self, x: torch.Tensor, lin_w: torch.Tensor, lin_b: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        out, self._out = self._out, None
        # Inside Function.forward grad is off; compare autocast only.
        ok = (
            out is not None
            and lin_w is self._lin_w
            and lin_b is self._lin_b
            and (lin_w._version, lin_b._version) == self._versions
            and _amp_state(x.device.type)[:2] == self._state[:2]
            and _same_storage_view(x, self._x)
            and x._version == self._x_version
        )
        if not ok:
            _note(
                "l2-mismatch", True,
                "[action-prefill] pair-2 Linear prefill did not match its "
                "consumer; recomputed in place (values unaffected)",
            )
            return None
        self.used = True
        return out


class _Holder:
    __slots__ = ("y",)

    def __init__(self, y: torch.Tensor) -> None:
        self.y: Optional[torch.Tensor] = y


class _Bridge(torch.autograd.Function):
    """Re-attach the prefill graph at the branch's original graph position.

    forward: an alias of the detached prefill output (no kernel). Its node is
    created where the ActionEncoder used to be called, so autograd schedules
    its backward exactly where the branch's backward used to run.
    backward: run the prefill graph now, reentrantly; its parameters'
    AccumulateGrad nodes (and any hooks on them) fire inside this call, in the
    graph's own sequence order -- the order they had as part of the outer
    graph. The anchor is a 0-element leaf that only makes this node
    differentiable; its gradient is None (AccumulateGrad no-op).
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,
        y_detached: torch.Tensor,
        anchor: torch.Tensor,
        holder: _Holder,
    ) -> torch.Tensor:
        ctx.holder = holder
        return y_detached

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, grad: torch.Tensor):  # pyre-ignore[3]
        holder = ctx.holder
        y = holder.y
        if y is None:
            raise RuntimeError(
                "[action-prefill] bridge backward ran twice (retain_graph / "
                "double backward through the action prefill is not supported); "
                f"set {ENV}=0"
            )
        holder.y = None
        torch.autograd.backward(y, grad)
        return None, None, None


class ActionPrefill:
    """One forward's issued action branch, waiting for its consumer."""

    def __init__(
        self,
        owner: torch.nn.Module,
        num_rows: int,
        max_uih_len: int,
        max_targets: int,
        num_targets: torch.Tensor,
        payloads: Dict[str, torch.Tensor],
        y: torch.Tensor,
        fused: bool,
        l2: Optional[PrefilledLinear],
    ) -> None:
        self.owner = owner
        self.num_rows = num_rows
        self.max_uih_len = max_uih_len
        self.max_targets = max_targets
        self.num_targets = num_targets
        self.payloads = payloads
        self.y = y
        self.y_version: int = y._version
        self.fused = fused
        self.l2 = l2
        self.state: Tuple[bool, torch.dtype, bool] = _amp_state(y.device.type)
        # (119) ACTION_DEDUP (ops/action_dedup.py): the codes [num_rows] when
        # this prefill carries the dedup plan (y = the action MLP on the code
        # rows), else None (the per-token branch above).
        self.dedup: Optional[torch.Tensor] = None

    def mismatch(
        self,
        max_uih_len: int,
        max_targets: int,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Optional[str]:
        """None when the consumer's inputs are the ones the prefill used."""
        if seq_embeddings.size(0) != self.num_rows:
            return f"rows {seq_embeddings.size(0)} != prefill {self.num_rows}"
        if (max_uih_len, max_targets) != (self.max_uih_len, self.max_targets):
            return "max lengths differ"
        if num_targets is not self.num_targets:
            return "num_targets is a different tensor"
        for name, t in self.payloads.items():
            if seq_payloads.get(name) is not t:
                return f"payload {name} is a different tensor"
        if _amp_state(seq_embeddings.device.type) != self.state:
            return "autocast/grad state differs"
        if self.y._version != self.y_version:
            return "prefill output modified in place"
        return None

    def attach(self) -> torch.Tensor:
        y = self.y
        if not y.requires_grad:
            return y
        anchor = torch.empty(0, device=y.device, requires_grad=True)
        return _Bridge.apply(y.detach(), anchor, _Holder(y))


def clear() -> None:
    global _pending
    _pending = None


def take(owner: torch.nn.Module) -> Optional[ActionPrefill]:
    """The pending prefill if it was issued for `owner`; the slot is emptied."""
    global _pending
    pre, _pending = _pending, None
    if pre is None or pre.owner is not owner:
        return None
    return pre


def fallback(reason: str) -> None:
    stats["fallback"] += 1
    _note(
        "fallback:" + reason.split(" ")[0], True,
        "[action-prefill] FALLBACK (%s): the issued prefill is discarded and "
        "the action branch runs in the original order", reason,
    )


def armed(pre: ActionPrefill, grouped: Optional[bool] = None) -> None:
    """The arming line, from the consumer that used it: logged once per
    process on EVERY rank (with rank=), not only on rank 0.

    `grouped`: None when the preprocessor ran without the grouped LayerNorm
    tail (ops/preproc_grouped_tail.py), else whether the tail stayed grouped
    (True) or declined to the shipped per-MLP tails (False)."""
    stats["used"] += 1
    if "active" in _logged:
        return
    _logged.add("active")
    if pre.dedup is not None:
        d = torch.distributed
        rank = d.get_rank() if d.is_available() and d.is_initialized() else 0
        logger.info(
            "[action-prefill] ACTIVE -- (119) ACTION_DEDUP plan (codes + action "
            "MLP on the 2**T+1 code rows x2) issued before the forward embedding "
            "all-to-all wait; rows=%d table rows=%d; backward re-attached in "
            "place. Kill switch %s=0 (or DLRM_HSTU_FP8_ACTION_DEDUP=0) rank=%d",
            pre.num_rows, pre.y.shape[0], ENV, rank,
        )
        return
    l2 = "off" if pre.l2 is None else ("on" if pre.l2.used else "RECOMPUTED")
    gt = "n/a" if grouped is None else ("yes" if grouped else "DECLINED")
    d = torch.distributed
    rank = d.get_rank() if d.is_available() and d.is_initialized() else 0
    # Once per process on EVERY rank (like item (30)'s armed line), so a
    # multi-node log proves the prefill ran on each rank.
    logger.info(
        "[action-prefill] ACTIVE -- action branch (offsets + ActionEncoder + "
        "action-MLP pair 1, pair-2 Linear prefill=%s, F1 container fused=%s, "
        "grouped tail=%s) issued before the forward embedding all-to-all wait; "
        "rows=%d; backward re-attached in place (kernel order unchanged). Kill "
        "switch %s=0, part A only %s=0 rank=%d",
        l2, pre.fused, gt, pre.num_rows, ENV, L2_ENV, rank,
    )


def _mlp_reason(mlp: torch.nn.Module) -> Optional[str]:
    if not isinstance(mlp, torch.nn.Sequential):
        return "action MLP is not an nn.Sequential"
    if len(list(mlp)) < 2:
        return "action MLP has fewer than 2 children"
    return None


def issue(
    pre_mod: torch.nn.Module,
    max_uih_len: int,
    max_targets: int,
    total_targets: int,
    uih_seq_lengths: torch.Tensor,
    num_targets: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
) -> None:
    """Issue the action branch of `pre_mod` (a ContextualPreprocessor) now.

    Called by DlrmHSTU.preprocess under _user_forward's autocast, right before
    its first touch of the EC awaitable. The preprocessor later receives
    seq_lengths = uih_seq_lengths + num_targets (DlrmHSTU.main_forward's merged
    lengths) and seq_embeddings with sum(uih) + total_targets rows; the
    offsets below are its own three ops on those lengths.
    """
    global _pending
    _pending = None
    # pyre-ignore[16]
    enc = pre_mod._action_encoder
    # pyre-ignore[16]
    mlp = pre_mod._action_embedding_mlp
    reason = _mlp_reason(mlp)
    if reason is not None:
        return decline(reason)
    names: List[str] = [enc._action_feature_name]
    if len(enc._watchtime_to_action_thresholds_and_weights) > 0:
        names.append(enc._watchtime_feature_name)
    payloads: Dict[str, torch.Tensor] = {}
    for name in names:
        t = seq_payloads.get(name)
        if t is None:
            return decline(f"payload {name} missing")
        payloads[name] = t
    actions = payloads[enc._action_feature_name]
    num_rows = actions.size(0) + total_targets
    if preproc_fp8_snr.ENABLED:
        # Default-off FP8 numerics probe (Arbor E-M3): its action-MLP pair-1
        # site (L1x) now runs here, before ContextualPreprocessor.forward
        # notes this forward's grad mode, so note it first, as that forward
        # does (else L1x reads the previous forward's mode, e.g. eval's).
        preproc_fp8_snr.note_grad_mode(torch.is_grad_enabled())
    if os.environ.get("DLRM_HSTU_FP8_ACTION_DEDUP", "0").strip().lower() in (
        "1", "true", "on", "yes",
    ):
        # (119) ACTION_DEDUP (ops/action_dedup.py): issue the dedup plan -- the
        # codes and the action MLP on the 2**T + 1 code rows (x2) -- in this
        # window instead of the per-token branch below. Declined: one line,
        # then the per-token prefill below.
        from generative_recommenders.ops import action_dedup as _ad

        why = _ad.decline_reason(pre_mod, actions, seq_payloads)
        got = None
        if why is None:
            with record_function("## action_prefill ##"):
                if _ad.lean():
                    # (119) LEAN: the codes take the UIH lengths themselves
                    # (no add here and sub in codes(): (uih + n) - n == uih).
                    got = _ad.try_plan(
                        pre_mod, seq_payloads, None, num_targets, num_rows,
                        uih_lengths=uih_seq_lengths,
                    )
                else:
                    got = _ad.try_plan(
                        pre_mod, seq_payloads, uih_seq_lengths + num_targets,
                        num_targets, num_rows,
                    )
        if got is not None:
            code, atab2 = got
            stats["issued"] += 1
            _pending = ActionPrefill(
                owner=pre_mod,
                num_rows=num_rows,
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                num_targets=num_targets,
                payloads=payloads,
                y=atab2,
                fused=False,
                l2=None,
            )
            _pending.dedup = code
            return
        if why is not None:
            _ad.note_decline(why)
    with record_function("## action_prefill ##"):
        # The preprocessor registers the GEMM pins for its row count before
        # its first GEMM; the prefill's GEMMs now come first.
        install_for_tokens(num_rows)
        seq_lengths = uih_seq_lengths + num_targets
        target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        uih_offsets = seq_offsets - target_offsets
        action_embeddings = enc(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            uih_offsets=uih_offsets,
            target_offsets=target_offsets,
            # Read for .size(0) only: total_targets = rows - len(actions).
            seq_embeddings=actions.new_empty((num_rows, 0)),
            seq_payloads=seq_payloads,
            # rccl ACTION_ENC_FUSE: as at the original call site, the only
            # consumer is the action MLP's first Linear under the same bf16
            # autocast, so the fused encoder may emit the bf16 rows.
            bf16_consumer=True,
        )
        if isinstance(mlp, FusedLinearNormMLP):
            y, fused = mlp.forward_head(action_embeddings)
        else:
            y, fused = action_embeddings, False
            for m in list(mlp)[:-2]:
                y = m(y)
        l2: Optional[PrefilledLinear] = None
        if fused and l2_enabled():
            # pyre-ignore[16]
            lin, _ = mlp._pairs()[-1]
            l2 = PrefilledLinear(y.detach(), lin.weight, lin.bias)
    stats["issued"] += 1
    _pending = ActionPrefill(
        owner=pre_mod,
        num_rows=num_rows,
        max_uih_len=max_uih_len,
        max_targets=max_targets,
        num_targets=num_targets,
        payloads=payloads,
        y=y,
        fused=fused,
        l2=l2,
    )
