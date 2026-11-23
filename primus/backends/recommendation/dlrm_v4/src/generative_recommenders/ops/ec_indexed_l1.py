# pyre-strict
"""Indexed-X L1: the protocol between the grouped EC expand and the preprocessor L1 (pair 0 of the content and
additional MLPs). rccl R-C3 / actckpt E-X2; iteration-83 integration (probes/iter83/ixint/DESIGN_IX.md v2).

DEFAULT OFF: DLRM_HSTU_FP8_EC_INDEXED_L1=1 (1/true/on/yes) arms it; unset or "0" = off (kill switch); ROCm gfx950
only. Kernels, gate (enabled / latch_off) and the pre-RUN_START prebuild: ops/triton/triton_ec_indexed_l1.py (K).
This module is imported only with the knob on: by dlrm_v4/train/ec_bf16_output._ix_plan and by the pair-0 sites
when the tensor they get carries IX_ATTR. It imports nothing from dlrm_v4 (the EC passes its _expand in).

EC: plan() admits exactly one grouped call: the CAT_FUSE artist|album member (idx [2M] interleaved) plus the merged
(PREMERGED) uih_post_id member (idx [M]). Their casts entries become IXRecords (truthy). _ECExpandGroup.forward
(inherited by E-X2b's ECExpandGroupPorts) calls rec.issue(): a stride-0 bf16 placeholder of the member output's
shape whose single storage element is NaN; the record keeps emb_c [U, 512] bf16 and idx. The item placeholder and
the cat view `big` are tagged.
FP8 head (preproc_fp8_l2._LinearSwishLNEmitFunction): fwd() -> h = _ecix_fwd(U, idx, W1_c, b1_c); the node saves
U + idx instead of X; backward: dX = dh @ W1_c unchanged (or E-X2b's dh port); dW1 = wgrad() over ALL M tokens.
Every other consumer: rebuild() -> _IXMaterialize(x) (X = the shipped member output's bytes; identity backward to
x itself, so the graph keeps the shipped shape EC -> flat -> View(big) -> consumer).
Ledger: the EC backward requires each member's gradient to come from an Indexed-X-aware consumer.

MLPerf closed division: no computation reduction. Every token's L1 output row is computed from that token's own
gathered row; dW1 sums dh[m]^T X[m] over all M tokens (split-K over token ranges, never over unique rows); nothing is
pre-summed over tokens that share a row; nothing is reused across tokens, samples or steps. The index only says where
a row is loaded from. NUMERICS: declared fp32 reduction-order class (see K). Knob off: nothing here runs.

IX-FP8F (ship14 P4; ops/triton/triton_ec_indexed_l1_fp8.py = F8, imported lazily; DLRM_HSTU_FP8_EC_INDEXED_L1_FP8=1
arms it, default OFF, "0" = kill switch; only on the Indexed-X path above). A TRAINING forward of the FP8 head
(fwd(train=True)) takes F8.fwd8 -> (h, table, w_bwd): h from e4m3 operands (U8 per-row pow2, W1_8 per-tensor
amax/448). Consistent backward (DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD=consistent, the default): table = Ud and w_bwd =
Wd, the dequantized operands; the node saves Ud for the unchanged wgrad (and its decline path) and uses Wd for the
saved W1, dX = dh @ Wd and E-X2b's bound W1, i.e. the gradient of the quantized forward. ste: the bf16 U / W1_c.
w_bwd is None on every bf16 path. Eval (train=False) and no_grad keep the bf16 path: the FP8 forward is train-only.
F8.Unsupported -> the bf16 _ecix_fwd for that call (one loud line per member); any other F8 failure -> F8.latch_off
(IX-FP8F off for the process; Indexed-X stays on). DECLARED precision change (fp8). Knob off: F8 is never imported
and nothing here prints or computes anything new.
"""

import collections
import importlib
import logging
import os
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from generative_recommenders.ops.triton import preproc_fp8_snr
from generative_recommenders.ops.triton import triton_ec_indexed_l1 as K
from torch.utils._python_dispatch import TorchDispatchMode

logger: logging.Logger = logging.getLogger(__name__)

IX_ATTR: str = "_dlrmv4_ec_indexed_l1"  # literal copies: fused_linear_norm, preproc_fp8_l2, preproc_grouped_tail
CHECK_ENV_VAR: str = "DLRM_HSTU_FP8_EC_INDEXED_L1_CHECK"
TAG: str = K.TAG
HID: int = K.HID
WIDTH: int = K.WIDTH
FWD_TILE: str = "t128x256x64w8s2k32"
WG_TILE: str = "a256x256x64w8s2k16p256x8i"
_DEVICES: Tuple[str, ...] = ("cuda",)  # plan() engages on these device types (the CPU test widens it)
_RANGE_CALLS: int = 2
_DECLINE_SITES: Tuple[str, ...] = ("FusedLinearNormMLP",)  # rebuild sites that are declines (census-visible)
# IX-FP8F: literal copies of F8.ENV_VAR / F8.TAG, so the knob is read (once per process) without importing F8.
F8_ENV_VAR: str = "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8"
F8_TAG: str = "[ec-indexed-l1-fp8]"
F8_MODULE: str = "generative_recommenders.ops.triton.triton_ec_indexed_l1_fp8"

_ONCE: Set[Tuple[str, str]] = set()
_LOGGED: Set[str] = set()
_FWD_SEEN: Dict[str, Tuple[int, int, str]] = {}
_WG_SEEN: Dict[str, Tuple[int, int, int]] = {}
_RANGE_LEFT: Dict[str, int] = {"item": _RANGE_CALLS, "cat": _RANGE_CALLS}
_CHECK_LEFT: Optional[Dict[Tuple[str, str], int]] = None
_AUDITS: List["StorageAudit"] = []  # active debug audits (tests / lab smoke only; empty in training)
STATS: "collections.Counter[str]" = collections.Counter()  # diagnostics only
_F8_KNOB: Optional[bool] = None  # F8_ENV_VAR on (1/true/on/yes), read once per process
_F8_MOD: Any = None  # F8 after its first import; False: the import failed (FAILED CLOSED once)
_F8_LOGGED: Set[str] = set()  # IX-FP8F once-per-process lines: "item" / "cat" (armed), "eval", "ignored"


def reset_for_testing() -> None:
    global _CHECK_LEFT, _F8_KNOB, _F8_MOD
    _ONCE.clear()
    _LOGGED.clear()
    _FWD_SEEN.clear()
    _WG_SEEN.clear()
    STATS.clear()
    _RANGE_LEFT.update(item=_RANGE_CALLS, cat=_RANGE_CALLS)
    _CHECK_LEFT = None
    _F8_KNOB = None
    _F8_MOD = None
    _F8_LOGGED.clear()


def _say(msg: str) -> None:
    K._say(msg)  # arm_print_all_ranks: every rank, " [rank N]"


def _once(site: str, reason: str, msg: str) -> None:
    """Loud (every rank; census-visible) once per process per (site, reason class)."""
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        _say(msg)


def _info_once(site: str, reason: str, msg: str) -> None:
    """logger INFO once per process per (site, reason class). The text must carry no census token (eval_armed_census
    ARMED_RE / DECL_RE, case-insensitively): no 'armed', 'declin*', 'fall back', 'disabled', 'inert', 'not used'."""
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        logger.info("%s", msg)


# ------------------------------------------------------------------------------------------------ IX-FP8F gate
def _f8_knob() -> bool:
    """DLRM_HSTU_FP8_EC_INDEXED_L1_FP8 (1/true/on/yes), read once per process. Never imports F8."""
    global _F8_KNOB
    if _F8_KNOB is None:
        _F8_KNOB = os.environ.get(F8_ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
    return _F8_KNOB


def _f8_once(key: str, msg: str) -> None:
    """Loud (every rank, K._say) once per process per key."""
    if key not in _F8_LOGGED:
        _F8_LOGGED.add(key)
        _say(msg)


def _f8_eval_once() -> None:
    _f8_once("eval", f"{F8_TAG} eval pass: bf16 Indexed-X (FP8 forward is train-only)")


def _f8_ignored_once() -> None:
    """The knob is on but Indexed-X was never armed (not a ROCm build, kernels unavailable). A mid-run Indexed-X latch
    printed its own FAILED CLOSED line: nothing more here (F8.fp8f_enabled's rule)."""
    if K._BROKEN is None:
        _f8_once("ignored", f"{F8_TAG} {F8_ENV_VAR}=1 IGNORED (NOT armed): Indexed-X ({K.ENV_VAR}) is not armed on "
                 "this process; the FP8 forward runs only on the Indexed-X path")


def _f8_failed(site: str, exc: BaseException) -> None:
    """F8's FAILED CLOSED line, printed here when F8 itself cannot (import failure, a raising latch_off)."""
    global _F8_MOD
    _F8_MOD = False
    _say(f"{F8_TAG} FAILED CLOSED at {site}: {type(exc).__name__}: {str(exc)[:300]} -> bf16 Indexed-X for the rest "
         "of the run")


def _f8_module() -> Any:
    """F8, imported at its first use (knob on only), or None after a failed import (one FAILED CLOSED line)."""
    global _F8_MOD
    if _F8_MOD is None:
        try:
            _F8_MOD = importlib.import_module(F8_MODULE)
        except Exception as exc:
            _f8_failed("import", exc)
    return _F8_MOD or None


def _f8_latch(f8: Any, site: str, exc: BaseException) -> None:
    """F8.latch_off: IX-FP8F off for the rest of the process (its one FAILED CLOSED line; Indexed-X stays on)."""
    try:
        f8.latch_off(site, exc)
    except Exception:
        _f8_failed(site, exc)


def _f8_for(train: bool) -> Any:
    """F8 when this forward takes the IX-FP8F path (training forward, knob on, F8 armed and not latched), else None.
    Knob off: False before anything is imported or printed."""
    if not _f8_knob():
        return None
    if not train:  # eval keeps the bf16 Indexed-X forward
        _f8_eval_once()
        return None
    f8 = _f8_module()
    if f8 is None:
        return None
    try:
        return f8 if f8.fp8f_enabled() else None
    except Exception as exc:  # a raising gate fails closed like a raising kernel
        _f8_latch(f8, "gate", exc)
        return None


def _graph_task_id() -> int:
    f = getattr(torch._C, "_current_graph_task_id", None)
    return -1 if f is None else int(f())


def _tracing(x: Any) -> bool:
    from torch.fx._symbolic_trace import is_fx_tracing

    return is_fx_tracing() or torch.compiler.is_compiling() or isinstance(x, torch.fx.Proxy)


def _saved_tensor_hooks_active() -> bool:
    """True under torch.autograd.graph.saved_tensors_hooks (non-reentrant activation checkpointing, save_on_cpu).
    There the original forward would issue placeholders while its recompute (inside the backward, G3) takes the
    shipped expand, and checkpoint() raises on the different saved-tensor counts. Missing API: False."""
    f = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    return f is not None and f(False) is not None


# ------------------------------------------------------------------------------------------------ records
class Ledger:
    """Consumer accounting of one issued member of one EC call. No tensors (the EC ctx keeps it)."""

    __slots__ = ("kind", "fwd", "bwd", "__weakref__")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.fwd = 0  # Indexed-X-aware consumers in the forward (indexed FP8 head or rebuild)
        self.bwd: Dict[int, Tuple[int, bool]] = {}  # graph task -> (consumer backwards, dh left via E-X2b's port)

    def note_bwd(self, via_port: bool) -> None:
        t = _graph_task_id()
        n, p = self.bwd.get(t, (0, False))
        self.bwd[t] = (n + 1, p or bool(via_port))

    def pop_bwd(self, t: int) -> Tuple[int, bool]:
        return self.bwd.pop(t, (0, False))


def _placeholder(n: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """[n, w] with strides (0, 0): one storage element, set to NaN by a ONE-element fill (Tensor.fill_ on the
    stride-0 tensor itself would iterate all n * w logical elements; DESIGN_IX 0.3)."""
    ph = torch.empty_strided((n, w), (0, 0), dtype=dtype, device=device)
    ph.as_strided((1,), (1,)).fill_(float("nan"))
    return ph


class IXRecord:
    """One hoisted member of one grouped EC call. Strong refs TO it: the IXPlan (for _merged_construct's lifetime)
    and the IX_ATTR tag on its placeholder. It holds emb_c / idx (no grad_fn), the ledger and the EC's _expand, and
    only a weakref to the tagged tensor, so nothing forms a reference cycle."""

    ix_record = True  # the marker _ECExpandGroup.forward reads (bools have no such attribute)
    __slots__ = ("kind", "member", "rows", "nh", "din", "ledger", "expand", "emb_c", "idx", "U", "issued",
                 "out_ref", "__weakref__")

    def __init__(self, kind: str, member: int, rows: int, nh: int, expand: Callable[..., torch.Tensor]) -> None:
        self.kind, self.member, self.rows, self.nh = kind, member, rows, nh
        self.din = nh * WIDTH
        self.ledger = Ledger(kind)
        self.expand = expand
        self.emb_c: Optional[torch.Tensor] = None
        self.idx: Optional[torch.Tensor] = None
        self.U = 0
        self.issued = False
        self.out_ref: Optional[weakref.ref] = None

    def issue(self, emb_c: torch.Tensor, idx: torch.Tensor, w: int, out_dtype: torch.dtype) -> torch.Tensor:
        """Inside _ECExpandGroup.forward: the placeholder, or the shipped member output when Indexed-X is latched
        off (e.g. the other member of this call failed) or the placeholder cannot be made (fail closed)."""
        if not K.enabled():
            return self.expand(emb_c, idx, w)
        try:
            ph = _placeholder(idx.numel(), w, out_dtype, emb_c.device)
        except Exception as exc:
            K.latch_off("ec forward", exc)
            return self.expand(emb_c, idx, w)
        self.emb_c, self.idx, self.U, self.issued = emb_c, idx, int(emb_c.shape[0]), True
        STATS["issued_" + self.kind] += 1
        for audit in _AUDITS:
            audit.watch(ph)
        return ph

    def materialize(self) -> torch.Tensor:
        """The shipped member output's bytes as [rows, din] (inside a Function forward: no autograd)."""
        return self.expand(self.emb_c, self.idx, WIDTH).view(self.rows, self.din)


class IXPlan:
    __slots__ = ("cat", "item")

    def __init__(self, cat: IXRecord, item: IXRecord) -> None:
        self.cat, self.item = cat, item

    def tag_item(self, values: torch.Tensor) -> None:
        _tag(self.item, values)

    def tag_cat(self, big: torch.Tensor) -> None:
        _tag(self.cat, big)


def _tag(rec: IXRecord, x: torch.Tensor) -> None:
    if not rec.issued:
        return
    setattr(x, IX_ATTR, rec)
    rec.out_ref = weakref.ref(x)


# ------------------------------------------------------------------------------------------------ EC side
def _device_reason(dev: torch.device) -> Optional[str]:
    try:
        K._check_device(dev)  # Triton importable, HIP, current device, gfx950 (cached per device)
    except K.IndexedL1Unsupported as exc:
        return str(exc)
    return None


def _plan_decline(embeddings: torch.Tensor, out_dtype: torch.dtype, casts: Sequence[Any],
                  idxs: Sequence[torch.Tensor], cat_slot: int, w: int, *, cat_group: int, chain: Optional[str],
                  names: Sequence[str], premerged: Sequence[bool],
                  uih_post_id: Optional[str]) -> Optional[Tuple[bool, str, str]]:
    """None (engage) or (loud, reason class, why). loud=False: this grouped call is not THE Indexed-X call (no cat
    member, or its other hoisted member is not the merged uih_post_id member). Other grouped calls of a step may
    legitimately be like that, so it is an INFO line without a census token (E-X2b's plan() is silent there,
    ec_dgrad_ports.py:134-137). loud=True: the run is configured so that no call can engage, or this IS the
    Indexed-X call and it cannot engage."""
    from generative_recommenders.ops import preproc_fp8_l2 as pfl

    if chain is None:  # G9a: before the structural checks, so an unregistered run says so once
        return (True, "chain", "consumer chain not registered (make_model note_ec_indexed_l1_chain)")
    # G4 / G6 / G6b: is this the Indexed-X call?
    if cat_slot < 0 or cat_group != 2:
        return (False, "cat", "no CAT_FUSE artist|album member")
    hoisted = [i for i, c in enumerate(casts) if c]
    if len(hoisted) != 2 or cat_slot not in hoisted:
        return (False, "members", f"{len(hoisted)} hoisted members, need the cat member + one")
    item_slot = hoisted[0] if hoisted[1] == cat_slot else hoisted[1]
    j = item_slot - (1 if item_slot > cat_slot else 0)  # names / premerged have no cat entry (ebo:356-377)
    if not 0 <= j < len(names) or not premerged[j] or names[j] != uih_post_id:
        got = (names[j], bool(premerged[j])) if 0 <= j < len(names) else ("?", False)
        return (False, "item", f"other hoisted member {got[0]!r} premerged={got[1]}, need the merged {uih_post_id!r}")
    # G9 - G11: the run's configuration
    if chain:
        return (True, "chain", f"consumer chain is not Indexed-X-aware ({chain})")
    if not pfl.enabled() or not {0, 1} <= set(pfl.FP8_MLPS):
        return (True, "em3a", "E-M3a (DLRM_HSTU_FP8_PREPROC_L2) is off: the indexed L1 runs in its FP8 head")
    if preproc_fp8_snr.ENABLED:
        return (True, "snr", "the FP8 numerics probe (DLRM_HSTU_FP8_PREPROC_SNR) reads X")
    # G5 / G7: this call's layout
    if w != WIDTH or out_dtype != torch.bfloat16 or embeddings.device.type not in _DEVICES:
        return (True, "layout", f"W={w} out={out_dtype} device={embeddings.device}")
    if embeddings.dtype != torch.float32 or embeddings.dim() != 2 or not embeddings.is_contiguous():
        return (True, "layout", f"embeddings {embeddings.dtype} {tuple(embeddings.shape)} not fp32 2-D contiguous")
    if embeddings.device.type == "cuda":
        why = _device_reason(embeddings.device)
        if why is not None:
            return (True, "device", why)
    for i in (cat_slot, item_slot):
        t = idxs[i]
        if t.dtype != torch.int64 or t.dim() != 1 or (t.numel() > 1 and t.stride(0) != 1):
            return (True, "idx", f"member idx {t.dtype} {tuple(t.shape)} not int64 1-D contiguous")
    n_cat, n_item = idxs[cat_slot].numel(), idxs[item_slot].numel()
    U = int(embeddings.shape[0])
    if n_cat % 2:
        return (True, "idx", "cat idx length odd")
    if n_cat // 2 != n_item:  # the grouped tail would decline on the row mismatch (pgt:910-917)
        return (True, "idx", f"cat rows {n_cat // 2} != item rows {n_item}")
    if U > K.MAX_TABLE_ROWS:
        return (True, "size", f"U={U} > {K.MAX_TABLE_ROWS}")
    if n_item < pfl._MIN_ROWS or n_item > K.MAX_ROWS:
        return (True, "size", f"member rows {n_item} outside [{pfl._MIN_ROWS}, {K.MAX_ROWS}]")
    return None


def plan(embeddings: torch.Tensor, out_dtype: torch.dtype, casts: Sequence[Any], idxs: Sequence[torch.Tensor],
         cat_slot: int, w: int, *, cat_group: int, chain: Optional[str], names: Sequence[str],
         premerged: Sequence[bool], uih_post_id: Optional[str],
         expand: Callable[..., torch.Tensor]) -> Optional[IXPlan]:
    """The two records for one grouped EC call, or None (the shipped expand). Called OUTSIDE the Function (grad
    mode reads False inside Function.forward)."""
    if not K.enabled():
        if _f8_knob():
            _f8_ignored_once()
        return None
    if not torch.is_grad_enabled() or not embeddings.requires_grad:
        if _f8_knob() and not torch.is_grad_enabled():
            _f8_eval_once()  # eval runs under no_grad: the FP8 forward is train-only
        return None  # G2: eval / no_grad: no placeholder, no attribute, byte-identical
    if _tracing(embeddings) or _graph_task_id() != -1:
        return None  # G3: fx / compile / Proxy; inside a backward (reentrant or non-reentrant recompute)
    if _saved_tensor_hooks_active():  # G3b
        _once("ec", "hooks", f"{TAG} declined at ec: saved-tensor hooks are active (activation checkpointing or "
              "offloading), whose recompute takes the shipped expand -> shipped expand + L1")
        return None
    why = _plan_decline(embeddings, out_dtype, casts, idxs, cat_slot, w, cat_group=cat_group, chain=chain,
                        names=names, premerged=premerged, uih_post_id=uih_post_id)
    if why is not None:
        loud, cls, msg = why
        if loud:
            _once("ec", cls, f"{TAG} declined at ec: {msg} -> shipped expand + L1")
        else:
            _info_once("ec", cls, f"{TAG} grouped EC call not eligible ({msg}); placeholders are issued only on the "
                       "call with the cat member and the merged uih_post_id member")
        return None
    hoisted = [i for i, c in enumerate(casts) if c]
    item_slot = hoisted[0] if hoisted[1] == cat_slot else hoisted[1]
    p = IXPlan(IXRecord("cat", cat_slot, idxs[cat_slot].numel() // 2, 2, expand),
               IXRecord("item", item_slot, idxs[item_slot].numel(), 1, expand))
    if "ec" not in _LOGGED:
        _LOGGED.add("ec")
        logger.info("%s ec placeholders issued (cat slot %d, item slot %d, M=%d/%d, U=%d)", TAG, cat_slot,
                    item_slot, p.cat.rows, p.item.rows, int(embeddings.shape[0]))
    return p


# ------------------------------------------------------------------------------------------------ FP8 head
class IXCall:
    """ctx.ix of an Indexed-X-aware pair 0 (no tensors)."""

    __slots__ = ("kind", "nh", "rows", "din", "indexed", "ledger", "expand", "site")

    def __init__(self, rec: IXRecord, site: str, indexed: bool) -> None:
        self.kind, self.nh, self.rows, self.din = rec.kind, rec.nh, rec.rows, rec.din
        self.ledger, self.expand, self.site, self.indexed = rec.ledger, rec.expand, site, indexed


# w_bwd: IX-FP8F's backward W1 (Wd, or W1_c under ste) for the saved weight, dX and E-X2b; None on every bf16 path.
IXUse = collections.namedtuple("IXUse", ["h", "table", "idx", "x", "call", "w_bwd"], defaults=(None,))


def _fwd_decline(rec: IXRecord, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                 w_dtype: torch.dtype) -> Optional[Tuple[str, str]]:
    if not K.enabled():
        return ("latched", "")
    if rec.out_ref is None or rec.out_ref() is not x:
        return ("object", "x is not the tagged member output")
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (rec.rows, rec.din):
        return ("shape", f"x {x.dtype} {tuple(x.shape)} != bf16 {(rec.rows, rec.din)}")
    if w.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return ("autocast", f"W1/b1 {w.dtype}/{b.dtype}: not under autocast(bf16)")
    if tuple(w.shape) != (HID, rec.din) or tuple(b.shape) != (HID,):
        return ("w1", f"W1 {tuple(w.shape)} b1 {tuple(b.shape)}")
    if w_dtype != torch.float32:  # the prebuilt wgrad class writes fp32 dW1: forward and wgrad engage together
        return ("wgrad", f"W1 parameter dtype {w_dtype}: the prebuilt wgrad class is fp32 dW1")
    if preproc_fp8_snr.ENABLED:
        return ("snr", "the FP8 numerics probe reads X")
    return None


def _range_check(rec: IXRecord) -> None:
    if _RANGE_LEFT.get(rec.kind, 0) <= 0:
        return
    _RANGE_LEFT[rec.kind] -= 1
    why = K.check_index_range(rec.idx, rec.U)  # one host sync
    if why is not None:
        raise RuntimeError(f"EC index invariant ({rec.kind}): {why}")
    if "range" not in _LOGGED and all(v < _RANGE_CALLS for v in _RANGE_LEFT.values()):
        _LOGGED.add("range")
        _say(f"{TAG} invariants OK (item and cat idx inside [0, U); checked on the first {_RANGE_CALLS} calls)")


def fwd(x: torch.Tensor, w_c: torch.Tensor, b_c: torch.Tensor, w_dtype: torch.dtype, site: str,
        train: bool = True) -> "IXUse":
    """_LinearSwishLNEmitFunction.forward, before x.contiguous(): h from (U, idx), or the exact rebuilt X.
    w_c / b_c: the autocast casts of W1 / b1; w_dtype: W1's parameter dtype (= the wgrad's output dtype).
    train: the node's train flag. IX-FP8F (knob on, train only): h from F8.fwd8, and the IXUse carries F8's
    backward operands (table = Ud, w_bwd = Wd; ste: U, W1_c). Every other path: w_bwd None, as before."""
    rec: IXRecord = getattr(x, IX_ATTR)
    why = _fwd_decline(rec, x, w_c, b_c, w_dtype)
    h: Optional[torch.Tensor] = None
    f8: Any = None
    f8_out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    if why is None:
        try:
            _range_check(rec)
            f8 = _f8_for(train)
            if f8 is not None:
                f8_out = _fwd8(f8, rec, w_c, b_c)  # None: this call takes the bf16 _ecix_fwd
            h = K.ecix_fwd(rec.emb_c, rec.idx, rec.nh, w_c, b_c) if f8_out is None else f8_out[0]
        except K.IndexedL1Unsupported as exc:
            why = ("class", f"outside the prebuilt kernel class ({exc})")
        except Exception as exc:  # compile / launch / OOM / idx invariant: fail closed for the process
            K.latch_off(f"fwd {rec.kind}", exc)
            why = ("latched", "")
    rec.ledger.fwd += 1
    call = IXCall(rec, site, h is not None)
    if h is None:
        assert why is not None
        if why[0] != "latched":
            _once(f"fwd {rec.kind}", why[0],
                  f"{TAG} declined at fwd ({site}, {rec.kind}): {why[1]} -> X rebuilt (exact) + shipped L1")
        STATS["rebuilt_fwd_" + rec.kind] += 1
        return IXUse(None, None, None, rec.materialize(), call)
    STATS["indexed_fwd_" + rec.kind] += 1
    _armed_fwd(rec, site)
    if f8_out is not None:  # IX-FP8F
        STATS["fp8_fwd_" + rec.kind] += 1
        _armed_fwd8(f8, rec, site)
        if _check_on("fwd", rec.kind):
            _check_fwd8(f8, rec, h, w_c, b_c)  # never against the bf16 addmm (a declared precision change)
        return IXUse(h, f8_out[1], rec.idx, None, call, f8_out[2])
    if _check_on("fwd", rec.kind):
        _check_fwd(rec, h, w_c, b_c)
    return IXUse(h, rec.emb_c, rec.idx, None, call)


def _fwd8(f8: Any, rec: IXRecord, w_c: torch.Tensor,
          b_c: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """IX-FP8F forward of one member: (h, table, w_bwd), or None -> the bf16 _ecix_fwd for this call.
    F8.Unsupported (raised before any launch): one loud line per member. Anything else, including an output outside
    the contract (_check_fwd8_out): F8.latch_off (IX-FP8F off for the process; Indexed-X stays on)."""
    unsupported = getattr(f8, "Unsupported", ())
    if not (isinstance(unsupported, type) and issubclass(unsupported, BaseException)):
        unsupported = ()  # never let a malformed module turn the except clause itself into an error
    try:
        out = f8.fwd8(rec.emb_c, rec.idx, rec.nh, w_c, b_c)
        _check_fwd8_out(rec, out)
    except unsupported as exc:
        STATS["fp8_unsupported_" + rec.kind] += 1
        _once(f"fwd8 {rec.kind}", "class", f"{F8_TAG} declined at fwd ({rec.kind}): outside the prebuilt FP8 class "
              f"({exc}) -> bf16 Indexed-X forward for this call")
        return None
    except Exception as exc:
        _f8_latch(f8, f"fwd {rec.kind}", exc)
        return None
    return out


def _check_fwd8_out(rec: IXRecord, out: Any) -> None:
    """F8.fwd8's output contract (host-side; raises -> F8.latch_off): h bf16 [M, 256]; table bf16 [U, 512] (Ud or
    U); w_bwd bf16 [256, din] (Wd or W1_c); all contiguous on U's device. table and w_bwd also in the one pointer
    class the unchanged _ecix_wgrad and E-X2b's dgrad pass were prebuilt for (16-byte aligned, within 2 GiB)."""
    emb = rec.emb_c
    assert emb is not None
    if not isinstance(out, tuple) or len(out) != 3:
        raise RuntimeError(f"fwd8 returned {type(out).__name__}, not (h, table, w_bwd)")
    for name, t, shape in (("h", out[0], (rec.rows, HID)), ("table", out[1], tuple(emb.shape)),
                           ("w_bwd", out[2], (HID, rec.din))):
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or tuple(t.shape) != shape
                or t.device != emb.device or not t.is_contiguous()):
            got = (f"{t.dtype} {tuple(t.shape)} {t.device} contiguous={t.is_contiguous()}"
                   if isinstance(t, torch.Tensor) else type(t).__name__)
            raise RuntimeError(f"fwd8 {name}: {got}, need contiguous bf16 {shape} on {emb.device}")
        if name != "h":
            K._ptr_class_ok(f"fwd8 {name}", t)


def wgrad(call: IXCall, dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """_LinearSwishLNEmitFunction.backward: dW1 over ALL M tokens from (U, idx) in out_dtype, or the shipped
    (dh.t() @ X).to(out_dtype) on the exact rebuilt X. table: what the node saved (IXUse.table: U, or IX-FP8F's
    Ud, so both this kernel and the rebuilt-X decline differentiate the quantized forward)."""
    why: Optional[Tuple[str, str]] = None
    if torch.is_grad_enabled():  # create_graph=True: autograd runs the backward with grad mode on
        why = ("create_graph", "create_graph backward (second order is unsupported by the shipped pair 0 too)")
    elif not K.enabled():
        why = ("latched", "")
    else:
        try:
            dw = K.ecix_wgrad(dh, table, idx, call.nh, out_dtype=out_dtype)
        except K.IndexedL1Unsupported as exc:
            why = ("class", f"outside the prebuilt kernel class ({exc})")
        except Exception as exc:
            K.latch_off(f"wgrad {call.kind}", exc)
            why = ("latched", "")
        else:
            STATS["indexed_wgrad_" + call.kind] += 1
            _armed_wgrad(call, int(dh.shape[0]))
            if _check_on("wgrad", call.kind):
                _check_wgrad(call, dh, table, idx, out_dtype, dw)
            return dw
    if why[0] != "latched":
        _once(f"wgrad {call.kind}", why[0],
              f"{TAG} declined at wgrad ({call.kind}): {why[1]} -> X rebuilt (exact) + shipped wgrad")
    STATS["rebuilt_wgrad_" + call.kind] += 1
    x = call.expand(table, idx, WIDTH).view(call.rows, call.din)
    return (dh.t() @ x).to(out_dtype)


# ------------------------------------------------------------------------------------------------ rebuild
class _IXMaterialize(torch.autograd.Function):
    """X = the shipped member output (EC _expand on emb_c, idx) as [rows, din]; backward: the identity to the tagged
    input x (item `values`, or the cat view `big`), so the graph keeps the shipped shape EC -> flat -> View(big) ->
    consumer and hooks / retain_grad on `big` see the shipped gradient. The cat output is a view of the fresh
    expand (autograd marks it IN_CUSTOM_FUNCTION: an in-place write to X would raise; no consumer writes X)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, x: torch.Tensor, rec: IXRecord) -> torch.Tensor:  # type: ignore[override]
        ctx.set_materialize_grads(False)
        ctx.ledger = rec.ledger
        rec.ledger.fwd += 1
        out = rec.expand(rec.emb_c, rec.idx, WIDTH)  # fresh [rows * nh, 512]
        return out if rec.nh == 1 else out.view(rec.rows, rec.din)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, g: Optional[torch.Tensor]):  # type: ignore[override]
        if g is not None:
            ctx.ledger.note_bwd(False)
        return g, None


def rebuild(x: torch.Tensor, site: str) -> torch.Tensor:
    """A consumer that is not the indexed FP8 head: the exact X ([rows, din]) with x itself as the autograd input."""
    rec: IXRecord = getattr(x, IX_ATTR)
    if rec.out_ref is None or rec.out_ref() is not x:
        raise RuntimeError(f"{TAG} INVARIANT: {site} got a tagged tensor that is not the {rec.kind} member output")
    STATS[f"rebuilt_{site}_{rec.kind}"] += 1
    if site in _DECLINE_SITES:
        _once(site, rec.kind, f"{TAG} declined at {site} ({rec.kind}): the grouped tail did not run this head "
              "-> X rebuilt (exact) + shipped pair 0")
    else:
        _info_once(site, rec.kind, f"{TAG} X rebuilt at {site} ({rec.kind}) for the shipped pair 0 (E-M3a warm-up, "
                   "non-FP8 head or grouped-tail re-run); exact shipped bytes")
    return _IXMaterialize.apply(x, rec)


# ------------------------------------------------------------------------------------------------ EC backward
def ec_backward_check(ix_ledgers: Sequence[Tuple[int, Ledger]], grads: Sequence[Optional[torch.Tensor]]) -> None:
    """Top of _ECExpandGroup.backward. Per issued member, in THIS backward pass: a gradient that did not come from
    an Indexed-X-aware consumer means an IX-unaware reader read the NaN placeholder (raise); a consumer whose
    gradient got lost raises; two or more Indexed-X-aware consumers are exact (autograd summed them) but outside the
    validated envelope (one loud census line)."""
    t = _graph_task_id()
    for member, led in ix_ledgers:
        n, via_port = led.pop_bwd(t)
        arrived = grads[member] is not None or via_port
        if arrived and n == 1:
            continue
        if not arrived and n == 0:
            if led.fwd == 0:
                _once("ec backward", "unconsumed " + led.kind,
                      f"{TAG} {led.kind} placeholder issued but not consumed in its step")
            continue
        if arrived and n >= 2:
            _once("ec backward", "consumers " + led.kind,
                  f"{TAG} {led.kind} member declined the single-consumer check: {n} Indexed-X consumers returned a "
                  "gradient in one backward pass (each exact, autograd summed them; outside the validated envelope)")
            continue
        if n == 0:
            why = ("its gradient came from a consumer that is not Indexed-X-aware: that consumer read the "
                   "stride-0 NaN placeholder")
        else:
            why = f"{n} Indexed-X consumer(s) ran their backward but no gradient reached the EC backward"
        raise RuntimeError(f"{TAG} INVARIANT ({led.kind} member): {why}. Rerun with {K.ENV_VAR}=0.")


# ------------------------------------------------------------------------------------------------ logs / debug
def _armed_fwd(rec: IXRecord, site: str) -> None:
    _FWD_SEEN.setdefault(rec.kind, (rec.rows, rec.U, site))
    if "fwd" in _LOGGED or len(_FWD_SEEN) < 2:
        return
    _LOGGED.add("fwd")
    mi, u, _ = _FWD_SEEN["item"]
    mc = _FWD_SEEN["cat"][0]
    rebuilt = sum(v for k, v in STATS.items() if k.startswith("rebuilt_"))
    _say(f"{K.ARMED_PREFIX} fwd -- content/additional L1 forward reads X through the EC index (members item "
         f"[M={mi}, K={WIDTH}] and cat [M={mc}, K={2 * WIDTH}] artist|album; U={u} rows bf16; _ecix_fwd {FWD_TILE} "
         f"in {site}; X and the EC expand are not materialized; {rebuilt} earlier pair-0 calls rebuilt X; kill "
         f"switch {K.ENV_VAR}=0)")


def _armed_wgrad(call: IXCall, m: int) -> None:
    if call.kind not in _WG_SEEN:
        _, s, chunk = K.nsplit(m, call.nh)
        _WG_SEEN[call.kind] = (m, s, chunk)
    if "wgrad" in _LOGGED or len(_WG_SEEN) < 2:
        return
    _LOGGED.add("wgrad")
    (mi, si, ci), (mc, sc, cc) = _WG_SEEN["item"], _WG_SEEN["cat"]
    _say(f"{K.ARMED_PREFIX} wgrad -- L1 dW1 summed over all M tokens from (U, idx): split-K (item M={mi} S={si} "
         f"CHUNK={ci}; cat M={mc} S={sc} CHUNK={cc}) + fixed-order reduce, fp32 out (_ecix_wgrad {WG_TILE} + "
         f"_ecix_wgrad_reduce; X never materialized; kill switch {K.ENV_VAR}=0)")


def _armed_fwd8(f8: Any, rec: IXRecord, site: str) -> None:
    """IX-FP8F: once per member, at its first FP8 forward (every rank). F8.fwd8 prints its own armed line (it records
    the member in F8._ARMED); this one prints only when F8 did not, so each member gets exactly one line."""
    if rec.kind in _F8_LOGGED:
        return
    if rec.kind in getattr(f8, "_ARMED", ()):
        _F8_LOGGED.add(rec.kind)
        return
    try:
        mode = str(f8.bwd_mode())
    except Exception as exc:  # validated before fwd8 could run; never let a log line raise
        mode = f"? ({type(exc).__name__})"
    how = ("wgrad on Ud = bf16(U8 * 2^E), dgrad on Wd = bf16(W1_8 * SW): the gradient of the quantized forward"
           if mode == "consistent" else "wgrad on the bf16 U, dgrad on the bf16 W1_c: straight-through")
    _f8_once(rec.kind, f"armed OK: {F8_TAG} fwd ({rec.kind}): L1 forward on e4m3 U8 per-row pow2 (d) x W1_8 "
             f"per-tensor amax/448 read through the EC index (M={rec.rows}, K={rec.din}, U={rec.U}; _ecix_fwd8, fp32 "
             f"accumulate, bf16 h; {site}; train only, eval keeps the bf16 Indexed-X forward); bwd={mode} ({how}); "
             f"kill switch {F8_ENV_VAR}=0")


def _check_on(stage: str, kind: str) -> bool:
    global _CHECK_LEFT
    if _CHECK_LEFT is None:
        on = os.environ.get(CHECK_ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
        _CHECK_LEFT = {(s, k): (2 if on else 0) for s in ("fwd", "wgrad") for k in ("item", "cat")}
    n = _CHECK_LEFT.get((stage, kind), 0)
    if n <= 0:
        return False
    _CHECK_LEFT[(stage, kind)] = n - 1
    return True


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(t: torch.Tensor) -> torch.Tensor:
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)

    return (ordered(a) - ordered(b)).abs().max()


def _check_fwd(rec: IXRecord, h: torch.Tensor, w_c: torch.Tensor, b_c: torch.Tensor) -> None:
    """Debug (DLRM_HSTU_FP8_EC_INDEXED_L1_CHECK=1, first 2 calls per member): log only."""
    try:
        ref = torch.addmm(b_c, rec.materialize(), w_c.t())  # the shipped call on the exact X
        v = torch.stack([(h.view(torch.int16) != ref.view(torch.int16)).sum().float(),
                         (h.float() - ref.float()).abs().max(), _bf16_ulps(h, ref).float(),
                         (~torch.isfinite(h)).sum().float(), (~torch.isfinite(ref)).sum().float()]).tolist()
        _say(f"{TAG} check fwd ({rec.kind}, M={rec.rows}): {int(v[0])} of {h.numel()} elements differ from the "
             f"shipped addmm on the rebuilt X (max |d| {v[1]:.3e}, max {int(v[2])} bf16 ulp; non-finite "
             f"{int(v[3])} / shipped {int(v[4])})")
    except Exception as exc:
        _say(f"{TAG} check fwd ({rec.kind}) could not run: {type(exc).__name__}: {str(exc)[:200]}")


def _check_fwd8(f8: Any, rec: IXRecord, h: torch.Tensor, w_c: torch.Tensor, b_c: torch.Tensor) -> None:
    """Debug (DLRM_HSTU_FP8_EC_INDEXED_L1_CHECK=1, first 2 calls per member) of an IX-FP8F h: against F8's own torch
    reference (F8.reference_fwd8 on the same inputs), never against the bf16 addmm. Log only."""
    try:
        ref = f8.reference_fwd8(rec.emb_c, rec.idx, rec.nh, w_c, b_c)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        v = torch.stack([(h.view(torch.int16) != ref.view(torch.int16)).sum().float(),
                         (h.float() - ref.float()).abs().max(), _bf16_ulps(h, ref).float(),
                         (~torch.isfinite(h)).sum().float(), (~torch.isfinite(ref)).sum().float()]).tolist()
        _say(f"{F8_TAG} check fwd8 ({rec.kind}, M={rec.rows}): {int(v[0])} of {h.numel()} elements differ from "
             f"F8.reference_fwd8 (max |d| {v[1]:.3e}, max {int(v[2])} bf16 ulp; non-finite {int(v[3])} / "
             f"reference {int(v[4])})")
    except Exception as exc:
        _say(f"{F8_TAG} check fwd8 ({rec.kind}) could not run: {type(exc).__name__}: {str(exc)[:200]}")


def _check_wgrad(call: IXCall, dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor,
                 out_dtype: torch.dtype, dw: torch.Tensor) -> None:
    try:
        ref = (dh.t() @ call.expand(table, idx, WIDTH).view(call.rows, call.din)).to(out_dtype)
        v = torch.stack([(dw != ref).sum().float(),
                         (dw.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)]).tolist()
        _say(f"{TAG} check wgrad ({call.kind}, M={call.rows}): {int(v[0])} of {dw.numel()} elements differ from "
             f"the shipped (dh.t() @ X).to({out_dtype}) (max rel diff {v[1]:.3e})")
    except Exception as exc:
        _say(f"{TAG} check wgrad ({call.kind}) could not run: {type(exc).__name__}: {str(exc)[:200]}")


# ------------------------------------------------------------------------------------------------ storage audit
_META_OPS: Set[str] = {"sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "is_same_size", "dim", "size",
                       "stride", "numel", "storage_offset", "is_contiguous"}


def _is_view_op(func: Any) -> bool:
    rets = func._schema.returns
    return bool(rets) and all(r.alias_info is not None and not r.alias_info.is_write for r in rets)


class StorageAudit(TorchDispatchMode):
    """Debug / test only; the trainer never enters it. While active, IXRecord.issue() registers each placeholder,
    and every aten op that has an argument on a live placeholder's storage is recorded unless it is a view (schema:
    every return aliases an input, no write) or a metadata query. A view of a placeholder shares its storage, so a
    reader of `big`'s column views is caught too. Matching is by StorageImpl identity while the placeholder lives
    (its 2-byte block cannot be reused before that). The dispatch-mode stack is part of at::ThreadLocalState, which
    the autograd GraphTask copies into its device threads, so a backward() called inside the context is covered.
    No GPU memory."""

    def __init__(self) -> None:
        super().__init__()
        self.watched: List[weakref.ref] = []
        self.violations: List[str] = []

    def watch(self, ph: torch.Tensor) -> None:
        self.watched = [r for r in self.watched if r() is not None] + [weakref.ref(ph)]

    def __enter__(self) -> "StorageAudit":
        _AUDITS.append(self)
        return super().__enter__()

    def __exit__(self, *exc: Any) -> Any:
        if self in _AUDITS:
            _AUDITS.remove(self)
        return super().__exit__(*exc)

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        if self.watched and not _is_view_op(func) and func.overloadpacket.__name__ not in _META_OPS:
            live = {t.untyped_storage()._cdata for t in (r() for r in self.watched) if t is not None}
            for a in torch.utils._pytree.tree_leaves((args, kwargs)):
                if isinstance(a, torch.Tensor) and live:
                    try:
                        hit = a.untyped_storage()._cdata in live
                    except Exception:  # no storage (e.g. sparse)
                        hit = False
                    if hit:
                        self.violations.append(f"{func} reads/writes a placeholder ({tuple(a.shape)}, stride "
                                               f"{tuple(a.stride())})")
                        break
        return func(*args, **kwargs)
