"""Row-wise sequence-embedding unbucketize: permutation backward without index_add.

TorchRec's ``SequenceEmbeddingsAwaitable`` (row-wise sequence sharding, i.e.
only at NODES>=2 in this tree) restores the original token order of the
received embeddings with ``torch.index_select(ret, 0, unbucketize_permute)``.
Autograd's ``IndexSelectBackward0`` then runs ``zeros`` + ``index_add_``
(``indexFuncLargeIndex<float>``) over [rows, dim] fp32 on the critical rank's
serial tail before the backward embedding all_to_all (ship4 gap item N8).

``unbucketize_permute`` is a bijection on the received rows, so the backward
``dx = zeros.index_add_(0, perm, g)`` equals a plain row scatter
``dx[perm] = g``: every grad row is copied exactly once, nothing is summed and
no row is left unwritten, so neither the zero fill nor an inverse permutation
is needed. The values are bit-identical to ``0 + g`` except that ``-0.0`` is no
longer normalised to ``+0.0`` (equal under every downstream add/compare).

v2 vs iter-4 v1: v1 built the inverse permutation per backward call
(``arange`` + ``index_copy_``) and then gathered ``grad.contiguous()``. v2's
mode ``scatter`` is ONE kernel: ``empty(...).index_copy_(0, perm, g)``
(reads ``g`` through its strides, so no ``.contiguous()`` copy either). Default mode
``gather`` keeps v1's gather
but builds the inverse from a grow-only cached ``arange`` constant (no per-step
``arange`` launch) and drops ``.contiguous()``. Microbench (probes/iter5/n8v2, MI350X, D=512,
L2 flushed): stock 1.14 ms, gather 0.17-0.18 ms, scatter 0.31-0.33 ms per
backward call at 167,837 rows, so ``gather`` is the default.

Guards: the forward shape must match (``perm.numel() == rows``); the first
``DLRM_HSTU_FP8_UNBUCKETIZE_VERIFY`` calls (default 2) check bijectivity on the
device (one host sync each, warm-up only) and permanently fall back to the
stock path if it fails. Switch: ``DLRM_HSTU_FP8_UNBUCKETIZE_GATHER`` (absent =
off in code; the yambda_5b.gin ``apply_env_bootstrap`` binding turns it on;
``=0`` opts out; read once, at ``install()`` time). The ``DLRM_HSTU_FP8_``
prefix is forwarded by ``launch_slurm.sh``.

Fused receive hand-off (``seq_a2a_rowperm.py``, switch
``DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV`` under ``DLRM_HSTU_FP8_SEQ_A2A_FUSED``):
when that path is installed and on, and the awaitable is a ``comm_ops.Request``
whose wait is the patched ``All2All_Seq_Req_Wait`` (never a ``NoWait``), the
permutation is parked on the Request instead of being applied by the callback
below. The patched wait then applies it on every branch, reproducing this
module's decision (``_unbucketize``) exactly, and the awaitable only checks that
it was consumed (and raises if not, instead of returning misordered rows).
Otherwise the callback below runs, as before.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_UNBUCKETIZE_GATHER"
_VERIFY_ENV = "DLRM_HSTU_FP8_UNBUCKETIZE_VERIFY"
_MODE_ENV = "DLRM_HSTU_FP8_UNBUCKETIZE_BWD_MODE"
_MODES = ("scatter", "gather")

_installed = False
_orig_init: Any = None
_state = {"verify_left": None, "disabled": False, "announced": False, "mode": None}
# Set by seq_a2a_rowperm when its fused receive path is installed:
# (tensor_awaitable, perm, embedding_dim) -> hand-off object or None.
_RECV_HOOK: Any = None

# Grow-only constant index buffer (0..n-1), one per (device, dtype). This is a
# constant, not a cached model result: slicing it is equivalent to arange(n).
_arange_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _mode() -> str:
    if _state["mode"] is None:
        m = os.environ.get(_MODE_ENV, "gather").strip().lower() or "gather"
        if m not in _MODES:
            logger.warning("%s=%r unknown; using 'gather'", _MODE_ENV, m)
            m = "gather"
        _state["mode"] = m
    return _state["mode"]


def _arange(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    buf = _arange_cache.get(key)
    if buf is None or buf.numel() < n:
        cap = max(n, 2 * buf.numel() if buf is not None else 0)
        buf = torch.arange(cap, device=device, dtype=dtype)
        _arange_cache[key] = buf
    return buf[:n]


def backward_scatter(grad: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """dx[perm] = grad (perm is a bijection, so every row of dx is written)."""
    out = torch.empty(
        (perm.numel(),) + tuple(grad.shape[1:]), dtype=grad.dtype, device=grad.device
    )
    out.index_copy_(0, perm, grad)
    return out


def backward_gather(grad: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """dx = grad[inv] with inv[perm] = arange (cached constant arange)."""
    inv = torch.empty_like(perm)
    inv.index_copy_(0, perm, _arange(perm.numel(), perm.device, perm.dtype))
    return torch.index_select(grad, 0, inv)


_BACKWARDS = {"scatter": backward_scatter, "gather": backward_gather}


class _PermuteRows(torch.autograd.Function):
    """out = x[perm]; dx[perm] = dout (perm is a bijection)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(perm)
        return torch.index_select(x, 0, perm)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # type: ignore[override]
        (perm,) = ctx.saved_tensors
        return _BACKWARDS[_mode()](grad, perm), None


def _bijective(perm: torch.Tensor, rows: int) -> bool:
    if perm.dim() != 1 or perm.numel() != rows:
        return False
    if _state["verify_left"] is None:
        _state["verify_left"] = int(os.environ.get(_VERIFY_ENV, "2") or 0)
    if _state["verify_left"] > 0:
        _state["verify_left"] -= 1
        ok = bool(
            torch.equal(
                torch.sort(perm).values,
                torch.arange(rows, device=perm.device, dtype=perm.dtype),
            )
        )
        if not ok:
            _state["disabled"] = True
            logger.warning(
                "DLRMV4 unbucketize gather: permutation check FAILED; "
                "falling back to index_select/index_add for the rest of the run"
            )
        return ok
    return True


def _announce(rows: int, embedding_dim: int) -> None:
    if not _state["announced"]:
        _state["announced"] = True
        logger.info(
            "DLRMV4 armed OK: unbucketize gather: row-wise unbucketize backward = "
            "%s (rows=%d, dim=%d)",
            _mode(),
            rows,
            embedding_dim,
        )


def _unbucketize(
    ret: torch.Tensor, perm: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    # seq_a2a_rowperm._ubg_decision mirrors this decision for the fused receive
    # path; keep the two in step.
    x = ret.view(-1, embedding_dim)
    if (
        _state["disabled"]
        or perm.dtype != torch.int64  # index_copy_ needs a long index
        or not (torch.is_grad_enabled() and x.requires_grad)
        or not _bijective(perm, x.size(0))
    ):
        return torch.index_select(x, 0, perm)
    _announce(x.size(0), embedding_dim)
    return _PermuteRows.apply(x, perm)


def _check_consumed(ret: Any, handoff: Any) -> Any:
    if not handoff.consumed:
        raise RuntimeError(
            "DLRMV4 unbucketize gather: the row-wise unbucketize permutation handed to "
            "the fused seq a2a wait was not applied (the rows would be misordered); "
            "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV=0 runs the stock path"
        )
    return ret


def install() -> None:
    """Replace the stock unbucketize callback (idempotent)."""
    global _installed, _orig_init
    if _installed or not enabled():
        return
    try:
        from torchrec.distributed import dist_data
    except Exception:  # noqa: BLE001
        return
    cls = dist_data.SequenceEmbeddingsAwaitable
    orig_init = cls.__init__
    _orig_init = orig_init

    def __init__(
        self: Any,
        tensor_awaitable: Any,
        unbucketize_permute_tensor: Optional[torch.Tensor],
        embedding_dim: int,
    ) -> None:
        orig_init(self, tensor_awaitable, None, embedding_dim)
        if unbucketize_permute_tensor is not None:
            perm = unbucketize_permute_tensor
            hook = _RECV_HOOK
            handoff = (
                hook(tensor_awaitable, perm, embedding_dim)
                if hook is not None and not _state["disabled"]
                else None
            )
            if handoff is not None:  # the patched wait applies perm
                self.callbacks.append(lambda ret: _check_consumed(ret, handoff))
            else:
                self.callbacks.append(
                    lambda ret: _unbucketize(ret, perm, embedding_dim)
                )

    cls.__init__ = __init__
    _installed = True
    logger.info(
        "DLRMV4 unbucketize gather: SequenceEmbeddingsAwaitable patched (bwd=%s)",
        _mode(),
    )


def _uninstall_for_tests() -> None:
    """Restore the stock SequenceEmbeddingsAwaitable.__init__ (unit tests only)."""
    global _installed
    if not _installed:
        return
    from torchrec.distributed import dist_data

    dist_data.SequenceEmbeddingsAwaitable.__init__ = _orig_init
    _installed = False
