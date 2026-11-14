# pyre-strict
"""Item (118) PREPROC_DEDUP: ContextualPreprocessor's content (c) and additional (a) MLPs run once per unique
(item, artist, album) EC row triple of the batch instead of once per token; their F2 sum is gathered back to the
tokens right before the action (x) MLP's residual epilogue.

KNOB. DLRM_HSTU_FP8_PREPROC_DEDUP: only the value "1" arms it; unset, "0" (the kill switch) or any other value is
off. Code default OFF; the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP = True turns it on and a
pre-set environment value wins. With the knob on, make_model registers the consumer chain
(dlrm_v4/train/ec_bf16_output.note_preproc_dedup_chain; arms nothing). Knob off: this module is never imported (the
EC and make_model read the knob once via ec_bf16_output.ppdd_enabled, the pre-RUN_START hook reads a literal copy,
the preprocessor looks for the tag and, untagged, only checks sys.modules for this module), nothing prints and every
computation is today's.

WHERE.
  EC (ec_bf16_output._ppdd_tag, _merged_construct's grouped branch): on the call that carries the 2-key CAT_FUSE
    member and the merged uih_post_id member, tag() sets PPDD_ATTR -> one PPDDRecord on the item member output (the
    Indexed-X placeholder in training, the written expand in eval) and on the cat view `big`.
  Preprocessor (ContextualPreprocessor._preproc_dedup): run() returns acc_tok [T, 512] bf16, or None -> today's
    statements. With acc_tok no grouped tail is begun, so the action branch runs its `tail is None` statements with
    residual = acc_tok (F2: O = bf16(bf16(LN_x(h_x)) + acc_tok)). KS-K1: an armed, eager call with T > 0 that carries
    no tag (the EC tag did not apply: EC_GROUP_BWD off, the uih key under the hoist gate, a sharding override, ...)
    runs today's statements and is counted (note_untagged: untagged / eval_untagged on the exit line; the kit
    requires untagged == 0).
  Before RUN_START (dlrm_v4/train/blas_warmup._prebuild_preproc_dedup): prebuild() = exact self-check + warm-up of
    every launch class the path reaches, then ONE ARMED line per rank. Without it nothing engages (NOT armed).

MECHANISM (per call, on the device; no host read).
  key = (item * U + artist) * U + album (int64; U < 2^21); one stable sort; segment ids = cumsum of key changes;
  M_pad = max(256, round_up(min(U, T), 256)) from host ints (U = rows of the EC unique-row table, T = tokens);
  inv / order / offsets by scatter_ / searchsorted; the unique rows by index_select of each segment's first token.
  n_u <= M_pad holds by construction when T <= U, and by the dataset schema otherwise (artist and album are functions
  of the item, so n_u <= unique items < U); it is checked by a persistent device flag, copied non-blocking and read
  only after its event completed (OVERFLOW -> RuntimeError: the run aborts; the clamp keeps every index in bounds).
  Training: the unique rows are stacked twice ([u; u], R = 2 M_pad); the gather's backward is the fp32 segment sum
  of dO over each triple's tokens (the EC segment reduce), handed to the chain as bf16 hi (first copy) + lo (second
  copy), so the chain's fp32 column sums (LN weight / bias, Linear bias) keep ~2^-16 instead of ~2^-9 relative.
  The unique chain is the shipped BF16 F1 / F2 FusedLinearNormMLP code on untagged tensors and never enters the
  grouped tail: E-M3a, Indexed-X(-FP8F), E-X2b / X2H, X2L-B, P2D and E-M1 never see it (one fixed precision path per
  run, whatever U is). Indexed-X's issued records are counted as consumed (ledger.fwd) so its EC check stays silent.

NUMERICS (declared). PRECISION: c / a run BF16 where the candidate ran FP8 (IX-FP8F e4m3 L1, E-M3a e4m3 / e5m2 L2)
+ fp32 REDUCTION-ORDER: duplicate-token gradients are summed in fp32 before the MLP backward. No dropout, RNG or
batch-level dynamic scale lies inside the deduped region. vs a BF16 per-token reference: forward ulp-level (GEMM row
count M changes), backward reduction order only.
MLPerf: a computation reduction, admitted for this item only by the owner's explicit ruling. Every token's output is
the same function of its own inputs, evaluated once per distinct input within the batch; nothing persists across
steps or batches; no data is read before RUN_START (the warm-up uses synthetic integer data, no RNG).
"""

import atexit
import collections
import importlib
import logging
import os
import time
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_DEDUP"
# Sub-knob PADSPREAD (2026-10-24, gin-bound on; only "1" arms it; "0" is the kill switch): the padding triples
# u in [n_u, M_pad) name table row u % U instead of repeating the last token's rows. Their gradients are exact
# zeros, but with one shared row the EC cast-gather backward got a single 2 (M_pad - n_u)-entry (4x for a / b)
# hot segment: SPLIT_ABOVE-row serial pieces of ~1 ms per reduce at every batch size (traced A/B + microbench).
# Spread, every row gains at most a few zero entries and stays cold.
PADSPREAD_ENV: str = "DLRM_HSTU_FP8_PREPROC_DEDUP_PADSPREAD"
# Sub-knob HILO2 (2026-10-24, gin-bound on; only "1" arms it; "0" is the kill switch): the backward's hi / lo split
# as two launches into one [2 M_pad, W] buffer (hi = copy_(s), lo = sub(s, hi) with an fp32 compute and one RNE on
# store) instead of cast / cast / sub / cast / cat over [M_pad, W] fp32 -- the same values, ~1/3 of the HBM passes.
HILO2_ENV: str = "DLRM_HSTU_FP8_PREPROC_DEDUP_HILO2"
# Sub-knob LEAN (2026-10-24; only "1" arms it; "0" is the kill switch): fewer host-issued launches per call, the
# same integer index tensors bit for bit (step at 256 rows / GPU is CPU-bound). Construction: key by two
# add(alpha=U) (4 -> 2 launches); segment ids = cumsum of an int64 change flag written by ne(out=) with flag[0] = 0
# (ones / copy / bool->int64 cast / -1 dropped: 6 -> 3); arange(M_pad + 1) / arange(M_pad) as prefix views of one
# cached arange per (device, dtype) (M_pad follows the per-step U, so an exact-size cache would miss); rows by two
# index_selects (item, and the interleaved (a, b) pair straight from the [T, 2] cat view) with PADSPREAD's where and
# the hi / lo [u; u] copies folded into the same launches (training, PADSPREAD on: 12 -> 6). Overflow note: one
# sub instead of gt / cast / cat (buf[0] holds max(n_u - M_pad, 0), read as > 0). R7: the x-independent _mlp_reason
# part cached per (module, k); hot-path modules imported once (attributes still read per call).
LEAN_ENV: str = "DLRM_HSTU_FP8_PREPROC_DEDUP_LEAN"
TAG: str = "[preproc-dedup]"
PPDD_ATTR: str = "_dlrmv4_preproc_dedup"  # literal copy: modules/preprocessors.ContextualPreprocessor._preproc_dedup
CATFUSED_ATTR: str = "_dlrmv4_catfused"  # dlrm_v4/train/ec_bf16_output.CATFUSED_ATTR (literal)
WIDTH: int = 512
PAD: int = 256
KEY_U_MAX: int = 1 << 21  # U < 2^21 keeps (i * U + a) * U + b < 2^63
FP8_MIN_ROWS: int = 1 << 16  # E-M3a's row gate (preproc_fp8_l2._MIN_ROWS); the unique chain never reaches it
_DEVICES: Tuple[str, ...] = ("cuda",)  # engages on these device types (the CPU test widens it)

# Markers (markers_118.json is generated from these; ops/tests/preproc_dedup_cpu_test.py pins them).
ARMED_MARKER: str = "armed OK: [preproc-dedup] ARMED preproc_dedup (118)"
ARMED_HEAD: str = ARMED_MARKER  # mark_armed prints "ARMED_MARKER (<detail>; kill switch ...) [rank N]"
ARMED_RX: str = (r"armed OK: \[preproc-dedup\] ARMED preproc_dedup \(118\) \(.*kill switch "
                 r"DLRM_HSTU_FP8_PREPROC_DEDUP=0\)")
DISABLED_MARKER: str = f"{TAG} DISABLED"
DECLINED_MARKER: str = f"{TAG} declined"
NOT_ARMED_MARKER: str = f"{TAG} NOT armed"
IGNORED_MARKER: str = f"{TAG} {ENV_VAR}=1 IGNORED"
OVERFLOW_MARKER: str = f"{TAG} OVERFLOW"
COUNTERS_HEAD: str = f"{TAG} counters at exit:"
COUNTERS_RX: str = (r"\[preproc-dedup\] counters at exit: train=(\d+) dedup=(\d+) eval=(\d+) eval_dedup=(\d+) "
                    r"declined=(\d+) disabled=(\d+) overflow=(\d+) max_U=(\d+) max_T=(\d+) max_n_u=(\d+) "
                    r"untagged=(\d+) eval_untagged=(\d+)")  # KS-K1 fields appended: the older prefix rx still matches
# Trace census (kernel-name forms of this stack's traces; the kit's census reads them).
TRACE_SUPERSEDED_RX: str = (r"x2h_pass|x2lb_pass|_ecix_fwd8|_ecix_wgrad|_p2_swishln256_fwd_emit|ln_bwd_grp3f8|"
                            r"_grouped_ln_residual_fwd_kernel|x2lf_pass|_w2_quant_dual_kernel")
TRACE_DEDUP_RX: str = r"^_weighted_layer_norm_fwd(\.kd)?$"

# Warm-up classes (T tokens, U table rows), chosen by kernel-class boundaries only (no data-derived size): the train
# R = 2 M_pad and the eval R = M_pad each on both sides of FP8_MIN_ROWS (the SwishLN256 v3 / old-kernel switch), T on
# both sides of 2 GiB of [T, 512] bf16 storage, each with T % 16 == 0 and != 0 (Triton's integer / storage
# specializations). M_pad is a multiple of 256, so every unique-chain row count has N % 16 == 0.
WARM_CLASSES: Tuple[Tuple[int, int], ...] = (
    (2_097_152 + 4_096, 98_304),  # > 2 GiB, T % 16 == 0: train R 196,608 / eval R 98,304 (both >= 2^16)
    (2_097_152 + 4_095, 24_576),  # > 2 GiB, T % 16 != 0: train R 49,152 / eval R 24,576 (both < 2^16)
    (1_048_576, 49_152),  # <= 2 GiB, T % 16 == 0: train R 98,304 (>= 2^16) / eval R 49,152 (< 2^16)
    (1_048_575, 40_960),  # <= 2 GiB, T % 16 != 0: train R 81,920 (>= 2^16) / eval R 40,960 (< 2^16)
)
CHECK_T: int = 20_011  # self-check tokens (odd), with a heavy triple of HEAVY tokens (> the planner's 8,192 split)
CHECK_HEAVY: int = 9_000

_STATE: str = "unset"  # unset (no prebuild yet) | armed | not_armed | ignored | disabled
_ARMED_DEVS: Set[Tuple[str, int]] = set()
_ONCE: Set[Tuple[str, str]] = set()
COUNTS: "collections.Counter[str]" = collections.Counter()
_MAX: Dict[str, int] = {"U": 0, "T": 0, "n_u": 0}
_EXIT_REGISTERED: bool = False
_KNOB: Optional[bool] = None
_PADSPREAD: Optional[bool] = None
_HILO2: Optional[bool] = None
_LEAN: Optional[bool] = None
# LEAN caches: constants only (no data-derived value), created without a host read.
_AR: Dict[Tuple[str, int, torch.dtype], torch.Tensor] = {}  # (device, dtype) -> arange(cap), cap a power of two
_MLP_STATIC: Dict[Tuple[int, int], Tuple[Any, Optional[str]]] = {}  # (id(mlp), k) -> (weakref(mlp), reason)
_MLP_STATIC_MAX: int = 16
_MODS: Dict[str, Any] = {}  # hot-path modules, imported once


def enabled() -> bool:
    """DLRM_HSTU_FP8_PREPROC_DEDUP == "1", read once per process (unset / "0" / anything else: off)."""
    global _KNOB
    if _KNOB is None:
        _KNOB = os.environ.get(ENV_VAR, "").strip() == "1"
    return _KNOB


def padspread() -> bool:
    """DLRM_HSTU_FP8_PREPROC_DEDUP_PADSPREAD == "1", read once per process (unset / "0" / other: off)."""
    global _PADSPREAD
    if _PADSPREAD is None:
        _PADSPREAD = os.environ.get(PADSPREAD_ENV, "").strip() == "1"
    return _PADSPREAD


def hilo2() -> bool:
    """DLRM_HSTU_FP8_PREPROC_DEDUP_HILO2 == "1", read once per process (unset / "0" / other: off)."""
    global _HILO2
    if _HILO2 is None:
        _HILO2 = os.environ.get(HILO2_ENV, "").strip() == "1"
    return _HILO2


def lean() -> bool:
    """DLRM_HSTU_FP8_PREPROC_DEDUP_LEAN == "1", read once per process (unset / "0" / other: off)."""
    global _LEAN
    if _LEAN is None:
        _LEAN = os.environ.get(LEAN_ENV, "").strip() == "1"
    return _LEAN


def _mod(name: str) -> Any:
    """LEAN (R7): a hot-path module imported once; callers read its attributes per call (patches still apply)."""
    m = _MODS.get(name)
    if m is None:
        m = _MODS[name] = importlib.import_module(name)
    return m


def _arange(n: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """LEAN (b): arange(n) as a read-only prefix view of one cached arange per (device, dtype). Its capacity is the
    power of two >= the largest n seen (the warm-up classes size it before RUN_START); growing is one arange launch,
    as today; never a host read. Consumers must not write into it."""
    key = _dev_key(dev) + (dtype,)
    a = _AR.get(key)
    if a is None or a.numel() < n:
        with torch.inference_mode(False):  # a normal tensor even if first built under inference_mode
            a = _AR[key] = torch.arange(1 << max(n - 1, 1).bit_length(), device=dev, dtype=dtype)
    return a[:n]


def reset_for_testing() -> None:
    global _STATE, _KNOB, _PADSPREAD, _HILO2, _LEAN
    _STATE = "unset"
    _KNOB = None
    _PADSPREAD = None
    _HILO2 = None
    _LEAN = None
    _AR.clear()
    _MLP_STATIC.clear()
    _ARMED_DEVS.clear()
    _ONCE.clear()
    COUNTS.clear()
    _MAX.update(U=0, T=0, n_u=0)
    _OVF.clear()


def _say(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)  # every rank, " [rank N]"
    except Exception:  # never abort a run over a log line
        logger.warning("%s", msg)


def _once(site: str, cls: str, msg: str) -> None:
    if (site, cls) not in _ONCE:
        _ONCE.add((site, cls))
        _say(msg)


def _dev_key(dev: torch.device) -> Tuple[str, int]:
    if dev.type == "cuda":
        return ("cuda", dev.index if dev.index is not None else torch.cuda.current_device())
    return (dev.type, -1 if dev.index is None else dev.index)


def _tracing(x: Any) -> bool:
    if lean():  # R7: no per-call import (the module attribute is still read per call)
        return (_mod("torch.fx._symbolic_trace").is_fx_tracing() or torch.compiler.is_compiling()
                or isinstance(x, torch.fx.Proxy))
    from torch.fx._symbolic_trace import is_fx_tracing

    return is_fx_tracing() or torch.compiler.is_compiling() or isinstance(x, torch.fx.Proxy)


def _graph_task_id() -> int:
    f = getattr(torch._C, "_current_graph_task_id", None)
    return -1 if f is None else int(f())


def _saved_tensor_hooks_active() -> bool:
    """Activation checkpointing / offloading (saved_tensors_hooks): its recompute would not see the same tags."""
    f = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    return f is not None and f(False) is not None


def disable(site: str, exc: BaseException) -> None:
    """Fail closed for the process: this call and every later one run today's statements (the EC tags and the
    Indexed-X placeholders are untouched, so the grouped tail / Indexed-X / E-M3a still consume them)."""
    global _STATE
    _STATE = "disabled"
    COUNTS["disabled"] += 1
    _say(f"{DISABLED_MARKER} (FAILED CLOSED) at {site}: {type(exc).__name__}: {str(exc)[:300]} -> today's "
         "statements for this call and the rest of the run")


def _decline(site: str, cls: str, why: str) -> None:
    _once(site, cls, f"{DECLINED_MARKER} at {site} ({cls}): {why} -> today's statements for this call")


def _chain() -> Tuple[Optional[str], Optional[str]]:
    """make_model's registration: (None = not registered | "" = complete | what is missing, uih_post_id)."""
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo

    return ebo._PPDD_CHAIN, ebo._PPDD_UIH_POST_ID


def mark_ignored(why: str) -> None:
    """The knob is "1" but the configuration cannot engage (registration / device): one line, nothing engages."""
    global _STATE
    _STATE = "ignored"
    _once("arm", "ignored", f"{IGNORED_MARKER} (NOT armed): {why}; today's statements")


def mark_not_armed(why: str) -> None:
    """The pre-RUN_START self-check / warm-up did not pass (or did not run): one line, nothing engages."""
    global _STATE
    _STATE = "not_armed"
    _once("arm", "not_armed", f"{NOT_ARMED_MARKER}: {why} -> today's statements")


def mark_armed(device: torch.device, detail: str) -> None:
    """The pre-RUN_START component passed on this rank's device: arm it (run() engages only on armed devices),
    register the exit counters line and print ONE ARMED line on this rank (arm_print_all_ranks, " [rank N]")."""
    global _STATE
    _STATE = "armed"
    _ARMED_DEVS.add(_dev_key(device))
    _register_exit()
    _say(f"{ARMED_MARKER} ({detail}; device={device}; c / a BF16 on the unique triples, PRECISION + "
         f"REDUCTION-ORDER class; padspread={int(padspread())} (kill switch {PADSPREAD_ENV}=0); "
         f"hilo2={int(hilo2())} (kill switch {HILO2_ENV}=0); "
         f"lean={int(lean())} (kill switch {LEAN_ENV}=0); "
         f"kill switch {ENV_VAR}=0)")


def _chain_reason(chain: Optional[str]) -> Optional[str]:
    if chain is None:
        return "make_model did not register the consumer chain (ec_bf16_output.note_preproc_dedup_chain)"
    if chain:
        return f"the consumer chain is incomplete ({chain})"
    return None


def chain_reason() -> Optional[str]:
    """None when make_model registered a complete consumer chain (knob on), else why not."""
    return _chain_reason(_chain()[0])


def device_reason(dev: torch.device) -> Optional[str]:
    """None when this device can run the path (ROCm gfx950; the CPU test widens _DEVICES), else why not."""
    return _device_reason(dev)


def _device_reason(dev: torch.device) -> Optional[str]:
    if dev.type not in _DEVICES:
        return f"device {dev} (needs {'/'.join(_DEVICES)})"
    if dev.type == "cuda":
        if getattr(torch.version, "hip", None) is None:
            return "not a ROCm build"
        try:
            arch = str(getattr(torch.cuda.get_device_properties(dev), "gcnArchName", ""))
        except Exception as exc:
            return f"device properties unavailable ({type(exc).__name__})"
        if not arch.startswith("gfx950"):
            return f"arch {arch or '?'} is not gfx950"
    return None


# ------------------------------------------------------------------------------------------------ EC side
class PPDDRecord:
    """One grouped EC call's dedup inputs: the autograd-connected unique-row table (fp32 [U, 512]), the item
    member's merged index [T] and the cat member's interleaved index [2T] (row 2t = first additional feature,
    2t + 1 = second), the issued Indexed-X ledgers, and the EC's own expand / cast-expand. No tagged tensor is held
    (no reference cycle)."""

    __slots__ = ("embeddings", "idx_item", "idx_cat", "U", "T", "ledgers", "expand", "cast_expand", "out_dtype",
                 "__weakref__")

    def __init__(self, embeddings: torch.Tensor, idx_item: torch.Tensor, idx_cat: torch.Tensor,
                 ledgers: Sequence[Any], expand: Callable[..., torch.Tensor],
                 cast_expand: Callable[..., torch.Tensor], out_dtype: torch.dtype) -> None:
        self.embeddings, self.idx_item, self.idx_cat = embeddings, idx_item, idx_cat
        self.U, self.T = int(embeddings.shape[0]), int(idx_item.numel())
        self.ledgers = tuple(ledgers)
        self.expand, self.cast_expand, self.out_dtype = expand, cast_expand, out_dtype


def _ix_ledgers(ix: Any) -> List[Any]:
    out = []
    if ix is not None:
        for rec in (getattr(ix, "item", None), getattr(ix, "cat", None)):
            if rec is not None and getattr(rec, "issued", False):
                out.append(rec.ledger)
    return out


def _tag_decline(embeddings: torch.Tensor, out_dtype: torch.dtype, idxs: Sequence[torch.Tensor], cat_slot: int,
                 item_slot: int, w: int) -> Optional[Tuple[str, str]]:
    if w != WIDTH or out_dtype != torch.bfloat16:
        return ("layout", f"W={w} out={out_dtype}")
    if embeddings.dtype != torch.float32 or embeddings.dim() != 2 or not embeddings.is_contiguous():
        return ("layout", f"embeddings {embeddings.dtype} {tuple(embeddings.shape)} not fp32 2-D contiguous")
    if _dev_key(embeddings.device) not in _ARMED_DEVS:
        return ("device", f"{embeddings.device} was not armed before RUN_START")
    for i in (cat_slot, item_slot):
        t = idxs[i]
        if t.dtype != torch.int64 or t.dim() != 1 or (t.numel() > 1 and t.stride(0) != 1):
            return ("idx", f"member idx {t.dtype} {tuple(t.shape)} not int64 1-D contiguous")
    n_cat, n_item = idxs[cat_slot].numel(), idxs[item_slot].numel()
    if n_cat != 2 * n_item:
        return ("idx", f"cat idx {n_cat} != 2 x item idx {n_item}")
    if n_item == 0:
        return ("empty", "no tokens")
    if int(embeddings.shape[0]) >= KEY_U_MAX:
        return ("size", f"U={int(embeddings.shape[0])} >= {KEY_U_MAX} (int64 key)")
    return None


def tag(embeddings: torch.Tensor, out_dtype: torch.dtype, casts: Sequence[Any], idxs: Sequence[torch.Tensor],
        cat_slot: int, w: int, *, names: Sequence[str], premerged: Sequence[bool],
        values_by: Dict[str, torch.Tensor], fused_vals: Dict[str, torch.Tensor], cat_group: Sequence[str],
        ix: Any, chain: Optional[str], uih_post_id: Optional[str], expand: Callable[..., torch.Tensor],
        cast_expand: Callable[..., torch.Tensor]) -> None:
    """ec_bf16_output._merged_construct (grouped branch, after the members are built): tag the item member output
    and `big` of THE call (2-key CAT_FUSE member + the merged uih_post_id member). Never raises (fail closed)."""
    try:
        _tag(embeddings, out_dtype, casts, idxs, cat_slot, w, names, premerged, values_by, fused_vals, cat_group,
             ix, chain, uih_post_id, expand, cast_expand)
    except Exception as exc:
        disable("ec tag", exc)


def _tag(embeddings: torch.Tensor, out_dtype: torch.dtype, casts: Sequence[Any], idxs: Sequence[torch.Tensor],
         cat_slot: int, w: int, names: Sequence[str], premerged: Sequence[bool],
         values_by: Dict[str, torch.Tensor], fused_vals: Dict[str, torch.Tensor], cat_group: Sequence[str],
         ix: Any, chain: Optional[str], uih_post_id: Optional[str], expand: Callable[..., torch.Tensor],
         cast_expand: Callable[..., torch.Tensor]) -> None:
    if _STATE == "unset":  # the pre-RUN_START hook never ran (BLAS_WARMUP off, or no prebuild call)
        why = _chain_reason(chain)
        if why is not None:
            mark_ignored(why)
        else:
            mark_not_armed("the pre-run_start warm-up + self-check did not run on this process (BLAS warm-up off?)")
    if _STATE != "armed":
        return
    if _tracing(embeddings) or _graph_task_id() != -1:
        return  # fx / compile / Proxy; inside a backward (recompute): silent
    # Is this THE call? Other grouped EC calls of a step legitimately are not: silent.
    if cat_slot < 0 or len(cat_group) != 2:
        return
    hoisted = [i for i, c in enumerate(casts) if c]
    if len(hoisted) != 2 or cat_slot not in hoisted:
        return
    item_slot = hoisted[0] if hoisted[1] == cat_slot else hoisted[1]
    j = item_slot - (1 if item_slot > cat_slot else 0)  # names / premerged have no cat entry
    if not 0 <= j < len(names) or not premerged[j] or names[j] != uih_post_id:
        return
    why = _tag_decline(embeddings, out_dtype, idxs, cat_slot, item_slot, w)
    if why is not None:
        if why[0] != "empty":  # B = 0 on this rank: nothing to dedup; silent (note_untagged skips T = 0 too)
            _decline("ec", why[0], why[1])
        return
    values = values_by[names[j]]
    big = getattr(fused_vals[cat_group[0]], CATFUSED_ATTR)[0]
    rec = PPDDRecord(embeddings, idxs[item_slot], idxs[cat_slot], _ix_ledgers(ix), expand, cast_expand, out_dtype)
    setattr(values, PPDD_ATTR, rec)
    setattr(big, PPDD_ATTR, rec)


# ------------------------------------------------------------------------------------------------ construction
class Triples(NamedTuple):
    inv: torch.Tensor  # [T] token -> triple id (clamped to M_pad - 1)
    order: torch.Tensor  # [T] tokens sorted by triple id (stable)
    offsets: torch.Tensor  # [M_pad + 1]: segment u = order[offsets[u]:offsets[u + 1]]; empty for u >= n_u
    rows: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # [M_pad] item / first / second additional row
    n_u: torch.Tensor  # [1] int64, on the device (never read in the step)


def pad_rows(U: int, T: int) -> int:
    """M_pad = max(PAD, round_up(min(U, T), PAD)): host ints only (U from the table shape, T from the index)."""
    n = min(int(U), int(T))
    return max(PAD, (n + PAD - 1) // PAD * PAD)


def m_pad(U: int, T: Optional[int] = None) -> int:
    """M_pad for a U-row table (and T tokens when given): max(256, round_up(min(U, T), 256)). No data constant."""
    return pad_rows(U, U if T is None else T)


# zero-fills: the first sort of every key width (bit_length(max_key): the pass count and digit split) also runs
# torch.sort and compares raw bits, so the production width (~48 bits) is checked at its first step even though the
# pre-RUN_START self-check (U = 4000) already checked 36 bits.
_SORT64_CHECKED: Set[int] = set()
_SORT64_OFF: bool = False  # latched: a raising call or a mismatch -> torch.sort for the rest of the process
_SORT64_LOGGED: Set[int] = set()


def _stable_sort(keys: torch.Tensor, max_key: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch.sort(keys, stable=True). zero-fills (ops/zero_fold.py): with the triple-key bound `max_key` (host int,
    keys in [0, max_key]) the onesweep extension's 64-bit-key sort (ops/onesweep_sort.sort64_pairs: the rocPRIM
    onesweep kernels torch.sort runs, over bit_length(max_key) bits, ONE fill per sort) replaces torch.sort's 8 passes
    and 17 hipMemsetAsync fills. The stable sorting permutation is unique, so the (keys, order) bits are torch.sort's;
    the first call of each key width compares raw bits against torch.sort and fails closed (torch.sort from then on)."""
    global _SORT64_OFF
    if max_key is None or _SORT64_OFF or not keys.is_cuda:
        return torch.sort(keys, stable=True)
    from generative_recommenders.ops.zero_fold import enabled as _zero_fold_on

    if not _zero_fold_on():
        return torch.sort(keys, stable=True)
    try:
        from generative_recommenders.ops import onesweep_sort as _os

        got = _os.sort64_pairs(keys, int(max_key))
    except Exception as exc:  # fail closed: torch.sort for the rest of the process
        _SORT64_OFF = True
        _say(f"{TAG} zero-fills sort64 FAILED CLOSED -> torch.sort ({type(exc).__name__}: {str(exc)[:200]})")
        return torch.sort(keys, stable=True)
    if got is None:
        return torch.sort(keys, stable=True)
    bits = int(max_key).bit_length()
    if bits not in _SORT64_CHECKED:
        _SORT64_CHECKED.add(bits)
        ref = torch.sort(keys, stable=True)
        if not (torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])):
            _SORT64_OFF = True
            _say(f"{TAG} zero-fills sort64 raw-bit MISMATCH vs torch.sort (T={keys.numel()}, max_key={max_key}) "
                 "-> torch.sort for the rest of the process")
            return ref
        if len(_SORT64_LOGGED) < 4:
            _SORT64_LOGGED.add(bits)
            _say(f"armed OK: {TAG} zero-fills sort64: triple-key stable sort on the onesweep extension "
                 f"({bits}-bit keys, one fill per sort; T={keys.numel()} raw bits == torch.sort; "
                 "kill switch DLRM_HSTU_FP8_ZERO_FOLD=0)")
    return got


def build_unique(keys: torch.Tensor, U: int, M_pad: Optional[int] = None, max_key: Optional[int] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Sync-free unique of int64 keys [T] (no nonzero / unique / item / bool index): (inv, order, offsets, M_pad,
    n_u, first). One stable sort; ids = cumsum of key changes; M_pad = m_pad(U, T) unless given (host ints);
    inv [T] (clamped to M_pad - 1), order [T] (stable), offsets [M_pad + 1] (segment u = order[offsets[u]:
    offsets[u + 1]], empty for u >= n_u), n_u [1] on the device (never read in the step), first [M_pad] (a token
    of each segment; the padding segments repeat the last token). max_key: a host bound on the keys (every key in
    [0, max_key]), or None (unknown: torch.sort)."""
    T = keys.numel()
    dev = keys.device
    mp = m_pad(U, T) if M_pad is None else int(M_pad)
    skey, order = _stable_sort(keys, max_key)
    if lean():
        seg = _segment_ids(skey)
    else:
        new = torch.ones(T, dtype=torch.bool, device=dev)
        if T > 1:
            new[1:] = skey[1:] != skey[:-1]
        seg = torch.cumsum(new, 0, dtype=torch.int64) - 1  # nondecreasing triple id in sorted order
    n_u = seg[-1:] + 1
    seg_c = seg.clamp(max=mp - 1)  # overflow (flagged) never indexes out of bounds
    inv = torch.empty_like(seg_c).scatter_(0, order, seg_c)
    ar = _arange(mp + 1, dev, torch.int64) if lean() else torch.arange(mp + 1, device=dev, dtype=torch.int64)
    offsets = torch.searchsorted(seg, ar)
    first = order.index_select(0, offsets[:-1].clamp(max=T - 1))
    return inv, order, offsets, mp, n_u, first


def _segment_ids(skey: torch.Tensor) -> torch.Tensor:
    """LEAN (a): cumsum of an int64 change flag with flag[0] = 0 == today's cumsum(ones-then-changes, int64) - 1,
    exactly, for every T (T = 0: empty; T = 1: [0]). The cumsum reads int64, so no cast launch; no -1 launch."""
    T = skey.numel()
    flag = torch.empty(T, dtype=torch.int64, device=skey.device)
    flag[:1].zero_()
    if T > 1:
        torch.ne(skey[1:], skey[:-1], out=flag[1:])  # 0 / 1 stored as int64 by the comparison itself
    return torch.cumsum(flag, 0)


def _lean_triples(idx_item: torch.Tensor, idx_cat: torch.Tensor, U: int, m_pad: int, copies: int
                  ) -> Tuple[Triples, torch.Tensor, torch.Tensor]:
    """LEAN: (Triples, r_i [copies, M_pad], r_ab [copies, M_pad, 2]) with r_i / r_ab flattened == today's
    cat([r_i] * copies) / stack([cat([r_a] * copies), cat([r_b] * copies)], 1).reshape(-1) after _spread_pad, bit
    for bit. key = b + U * (a + U * item) by add(alpha) (exact int64, == (item * U + a) * U + b). r_ab comes
    straight from the [T, 2] cat view (already the CAT_FUSE interleave); PADSPREAD's where writes the copies
    directly (broadcast over the copy dim), otherwise one repeat each. Triples.rows are views of copy 0."""
    T = idx_item.numel()
    cat2 = idx_cat.view(T, 2)
    key = torch.add(cat2[:, 1], torch.add(cat2[:, 0], idx_item, alpha=U), alpha=U)
    inv, order, offsets, mp, n_u, first = build_unique(key, U, m_pad, max_key=U * U * U - 1)  # ids in [0, U)
    r_i = idx_item.index_select(0, first)  # [M_pad]
    r_ab = cat2.index_select(0, first)  # [M_pad, 2]
    if padspread():  # _spread_pad's rule: padding triple u >= n_u names row u % U in all three members
        u = _arange(mp, r_i.device, r_i.dtype)
        pad, spread = u >= n_u, u % U
        r_i = torch.where(pad.view(1, mp), spread.view(1, mp), r_i.view(1, mp).expand(copies, mp))
        r_ab = torch.where(pad.view(1, mp, 1), spread.view(1, mp, 1), r_ab.view(1, mp, 2).expand(copies, mp, 2))
    elif copies > 1:
        r_i, r_ab = r_i.repeat(copies).view(copies, mp), r_ab.repeat(copies, 1).view(copies, mp, 2)
    else:
        r_i, r_ab = r_i.view(1, mp), r_ab.view(1, mp, 2)
    return Triples(inv, order, offsets, (r_i[0], r_ab[0, :, 0], r_ab[0, :, 1]), n_u), r_i, r_ab


def build_triples(idx_item: torch.Tensor, idx_cat: torch.Tensor, U: int, m_pad: int) -> Triples:
    """The unique (item, a, b) triples of the tokens: key = (item * U + a) * U + b, then build_unique."""
    if lean():
        return _lean_triples(idx_item, idx_cat, U, m_pad, 1)[0]
    T = idx_item.numel()
    cat2 = idx_cat.view(T, 2)
    key = (idx_item * U + cat2[:, 0]) * U + cat2[:, 1]
    inv, order, offsets, _mp, n_u, first = build_unique(key, U, m_pad, max_key=U * U * U - 1)  # ids in [0, U)
    rows = (idx_item.index_select(0, first), cat2[:, 0].index_select(0, first), cat2[:, 1].index_select(0, first))
    return Triples(inv, order, offsets, _spread_pad(rows, n_u, U), n_u)


def _spread_pad(rows: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], n_u: torch.Tensor, U: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PADSPREAD on: padding triple u >= n_u names table row u % U in all three members (n_u stays on the device;
    no host read). inv never points at a padding triple, so no token's output changes; the padding rows' backward
    is exact zeros, now spread one per row instead of one hot segment. Off: rows unchanged."""
    if not padspread():
        return rows
    mp = rows[0].numel()
    u = torch.arange(mp, device=rows[0].device, dtype=rows[0].dtype)
    out = torch.where((u >= n_u).view(1, mp), (u % U).view(1, mp), torch.stack(rows))
    return out[0], out[1], out[2]


def _segment_sum(g: torch.Tensor, inv: torch.Tensor, order: torch.Tensor, offsets: torch.Tensor,
                 m_pad: int) -> torch.Tensor:
    """fp32 [m_pad, W] per-triple sums of g's rows: the EC segment reduce (GPU planner: cold segments
    deterministic, hot pieces atomic, as shipped); index_add_ on a CPU tensor (tests)."""
    if g.stride(-1) != 1:
        g = g.contiguous()
    if g.is_cuda:
        from generative_recommenders.dlrm_v4.train.ec_segment_reduce import segment_reduce_rows

        return segment_reduce_rows(g, inv, m_pad, order=order, offsets=offsets)
    acc = torch.float64 if g.dtype == torch.float64 else torch.float32  # fp64 only in the CPU wiring test
    return torch.zeros((m_pad, g.shape[1]), dtype=acc, device=g.device).index_add_(0, inv, g.to(acc))


class GatherSegReduceBwd(torch.autograd.Function):
    """out = src[:m_pad][inv] ([T, W]); backward = the fp32 segment sum of dO per triple.

    hilo=False: src is [m_pad, W]; grad = sum cast to src's dtype.
    hilo=True: src is [2 m_pad, W], the SAME rows computed twice; the forward reads the first copy and the backward
    returns [hi; lo], hi = bf16(sum), lo = bf16(sum - hi). The chain backward is linear in dy, so hi + lo carries the
    fp32 sum to ~2^-16 relative into every fp32 column sum the norm / Linear backwards form."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, src: torch.Tensor, inv: torch.Tensor, order: torch.Tensor,  # type: ignore[override]
                offsets: torch.Tensor, m_pad: int, hilo: bool, expand: Callable[..., torch.Tensor]) -> torch.Tensor:
        ctx.save_for_backward(inv, order, offsets)
        ctx.m_pad, ctx.hilo, ctx.in_dtype = m_pad, hilo, src.dtype
        return expand(src[:m_pad], inv, src.shape[1])

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        inv, order, offsets = ctx.saved_tensors
        s = _segment_sum(g, inv, order, offsets, ctx.m_pad)
        if ctx.hilo and hilo2() and s.dtype == torch.float32:
            d = torch.empty((2 * s.shape[0], s.shape[1]), dtype=ctx.in_dtype, device=s.device)
            d[:s.shape[0]].copy_(s)  # hi = RNE(s)
            torch.sub(s, d[:s.shape[0]], out=d[s.shape[0]:])  # lo = RNE(s - f32(hi)): fp32 opmath, one rounding
            return d, None, None, None, None, None, None
        hi = s.to(ctx.in_dtype)
        if ctx.hilo:
            lo = (s - hi.to(s.dtype)).to(ctx.in_dtype)
            hi = torch.cat([hi, lo], dim=0)
        return hi, None, None, None, None, None, None


# ------------------------------------------------------------------------------------------------ overflow flag
class _Ovf:
    __slots__ = ("buf", "host", "event", "k10")

    def __init__(self, dev: torch.device) -> None:
        self.buf = torch.zeros(2, dtype=torch.int64, device=dev)  # [overflow seen, max n_u]
        cuda = dev.type == "cuda"
        self.host = torch.zeros(2, dtype=torch.int64, pin_memory=cuda)
        self.event = torch.cuda.Event() if cuda else None
        self.k10: Optional[torch.Tensor] = None  # LEAN: [1, 0] on the device (built at the first note)


_OVF: Dict[Tuple[str, int], _Ovf] = {}


def _overflow_note(n_u: torch.Tensor, m_pad: int) -> None:
    """Device-side: buf = max(buf, [n_u > m_pad, n_u]); then a non-blocking copy to pinned host + event."""
    key = _dev_key(n_u.device)
    st = _OVF.get(key)
    if st is None:
        st = _OVF[key] = _Ovf(n_u.device)
    if lean():  # one launch: [n_u - M_pad, n_u]; buf[0] = max(0, n_u - M_pad) over the run, > 0 iff overflow
        if st.k10 is None:
            st.k10 = torch.arange(1, -1, -1, dtype=torch.int64, device=n_u.device)
        upd = torch.sub(n_u.expand(2), st.k10, alpha=m_pad)
    else:
        upd = torch.cat([(n_u > m_pad).to(torch.int64), n_u])
    torch.maximum(st.buf, upd, out=st.buf)
    st.host.copy_(st.buf, non_blocking=st.event is not None)
    if st.event is not None:
        st.event.record()


def _overflow_read(block: bool = False) -> Tuple[int, int, bool]:
    """(overflow, max n_u, complete): host memory only, read when the last copy completed (event.query(), never
    blocking in the step; block=True only at teardown)."""
    ovf, n_u, complete = 0, 0, True
    for st in _OVF.values():
        if st.event is not None:
            if block:
                try:
                    st.event.synchronize()
                except Exception:
                    pass
            if not st.event.query():
                complete = False
                continue
        h = st.host.tolist()
        ovf, n_u = max(ovf, int(h[0] > 0) if lean() else int(h[0])), max(n_u, int(h[1]))
    _MAX["n_u"] = max(_MAX["n_u"], n_u)
    return ovf, n_u, complete


def _overflow_poll() -> None:
    ovf, n_u, _ = _overflow_read()
    if ovf:
        COUNTS["overflow"] += 1
        msg = (f"{OVERFLOW_MARKER}: a batch had more unique (item, a, b) triples than M_pad rows (max n_u {n_u}); "
               "the dataset invariant n_u <= unique items < U failed. Aborting; rerun with "
               f"{ENV_VAR}=0 to isolate.")
        _say(msg)
        raise RuntimeError(msg)


# ------------------------------------------------------------------------------------------------ preprocessor side
def _mlp_static_reason(mlp: Any, k: int) -> Optional[str]:
    """_mlp_reason's x-independent part: the container type, its children and the Linear shapes."""
    from generative_recommenders.ops.fused_linear_norm import FusedLinearNormMLP

    if not isinstance(mlp, FusedLinearNormMLP):
        return f"{type(mlp).__name__} is not FusedLinearNormMLP"
    kids = list(mlp)
    if len(kids) != 4 or not isinstance(kids[0], torch.nn.Linear) or not isinstance(kids[2], torch.nn.Linear):
        return "not Linear -> SwishLayerNorm -> Linear -> LayerNorm"
    if kids[0].in_features != k or kids[2].out_features != WIDTH:
        return f"Linear shapes {kids[0].in_features} -> {kids[2].out_features}, need {k} -> {WIDTH}"
    return None


def _mlp_reason(mlp: Any, k: int, x: torch.Tensor) -> Optional[str]:
    if lean():  # R7: the static part once per (module, k); _can_fuse (x, the module's kernel / enable state) per call
        hit = _MLP_STATIC.get((id(mlp), k))
        if hit is None or hit[0]() is not mlp:  # a recycled id never reuses a stale result
            why = _mlp_static_reason(mlp, k)
            try:
                ref = weakref.ref(mlp)
            except TypeError:  # not weak-referenceable: uncached
                ref = None
            if ref is not None:
                if len(_MLP_STATIC) >= _MLP_STATIC_MAX:
                    _MLP_STATIC.clear()
                _MLP_STATIC[(id(mlp), k)] = (ref, why)
        else:
            why = hit[1]
    else:
        why = _mlp_static_reason(mlp, k)
    if why is not None:
        return why
    if x.is_cuda and not mlp._can_fuse(x):
        return "FusedLinearNormMLP._can_fuse declined (non-Triton norm)"
    return None


def _run_decline(pre: Any, seq_embeddings: torch.Tensor, big: Optional[torch.Tensor],
                 rec: PPDDRecord) -> Optional[Tuple[str, str]]:
    if lean():  # R7: imported once; ENABLED is still read per call
        preproc_fp8_snr = _mod("generative_recommenders.ops.triton.preproc_fp8_snr")
    else:
        from generative_recommenders.ops.triton import preproc_fp8_snr

    if big is None or getattr(big, PPDD_ATTR, None) is not rec:
        return ("tag", "the additional features' cat view does not carry the item member's record")
    if seq_embeddings.dim() != 2 or int(seq_embeddings.shape[0]) != rec.T or tuple(big.shape) != (rec.T, 2 * WIDTH):
        return ("tag", f"rows {tuple(seq_embeddings.shape)} / {tuple(big.shape)} != the record's T={rec.T}")
    dev = rec.embeddings.device.type
    if not torch.is_autocast_enabled(dev) or torch.get_autocast_dtype(dev) != torch.bfloat16:
        return ("autocast", "not under autocast(bf16)")
    for name, mlp, k in (("content", pre._content_embedding_mlp, WIDTH),
                         ("additional", pre._additional_embedding_mlp, 2 * WIDTH)):
        why = _mlp_reason(mlp, k, seq_embeddings)
        if why is not None:
            return ("mlp", f"{name}: {why}")
    if _saved_tensor_hooks_active():
        return ("hooks", "saved-tensor hooks are active (activation checkpointing or offloading)")
    if preproc_fp8_snr.ENABLED:
        return ("snr", "the FP8 numerics probe (DLRM_HSTU_FP8_PREPROC_SNR) is armed")
    return None


def _run_core(pre: Any, rec: PPDDRecord, hilo: bool) -> torch.Tensor:
    """acc_tok = (additional(a_u, residual=content(x_u)))[:M_pad][inv]: the c + a F2 sum per token."""
    T, U = rec.T, rec.U
    m_pad = pad_rows(U, T)
    r_ab: Optional[torch.Tensor] = None
    if lean():  # the [u; u] copies and the (a, b) interleave come out of the construction's own launches
        tri, r_i2, r_ab2 = _lean_triples(rec.idx_item, rec.idx_cat, U, m_pad, 2 if hilo else 1)
        r_i, r_ab = r_i2.view(-1), r_ab2.view(-1)
    else:
        tri = build_triples(rec.idx_item, rec.idx_cat, U, m_pad)
        r_i, r_a, r_b = tri.rows
        if hilo:  # [u; u]: hi / lo copies (GatherSegReduceBwd)
            r_i, r_a, r_b = torch.cat([r_i, r_i]), torch.cat([r_a, r_a]), torch.cat([r_b, r_b])
    R = r_i.numel()
    x_u = rec.cast_expand(rec.embeddings, r_i, rec.out_dtype)  # [R, 512]
    if r_ab is None:
        r_ab = torch.stack([r_a, r_b], dim=1).reshape(-1)
    a_u = rec.cast_expand(rec.embeddings, r_ab, rec.out_dtype)
    a_u = a_u.view(R, 2 * WIDTH)  # the CAT_FUSE layout: one interleaved expand, [R, 1024]
    c_u = pre._content_embedding_mlp(x_u)
    acc_u = pre._mlp_add(pre._additional_embedding_mlp, a_u, c_u)
    out = GatherSegReduceBwd.apply(acc_u, tri.inv, tri.order, tri.offsets, m_pad, hilo, rec.expand)
    _overflow_note(tri.n_u, m_pad)
    _MAX["U"], _MAX["T"] = max(_MAX["U"], U), max(_MAX["T"], T)
    return out


def run(pre: Any, seq_embeddings: torch.Tensor, big: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """ContextualPreprocessor._preproc_dedup: acc_tok [T, 512] (the c + a F2 sum gathered to the tokens), or None
    -> today's statements. `big`: the preprocessor's _catfused_base of the additional features."""
    rec = getattr(seq_embeddings, PPDD_ATTR, None)
    if not isinstance(rec, PPDDRecord) or _STATE != "armed":
        return None
    if torch.jit.is_scripting() or _tracing(seq_embeddings) or _graph_task_id() != -1:
        return None
    train = torch.is_grad_enabled()
    COUNTS["train" if train else "eval"] += 1
    _overflow_poll()  # raises OVERFLOW (outside the fail-closed region: the run must stop)
    why = _run_decline(pre, seq_embeddings, big, rec)
    if why is not None:
        COUNTS["declined"] += 1
        _decline("run", why[0], why[1])
        return None
    try:
        out = _run_core(pre, rec, hilo=train)
    except Exception as exc:
        disable("run", exc)
        return None
    for led in rec.ledgers:  # the issued Indexed-X members are consumed here (their placeholders are never read)
        led.fwd += 1
    COUNTS["dedup" if train else "eval_dedup"] += 1
    return out


def note_untagged(seq_embeddings: torch.Tensor) -> None:
    """KS-K1 (WBND): ContextualPreprocessor._preproc_dedup's call without the EC tag (it calls this only when this
    module is loaded, i.e. knob on). Armed, eager and T > 0: the EC tag did not apply (EC_GROUP_BWD off, the uih key
    under the hoist gate, a sharding override, ...), so the call runs today's per-token statements. Counted for the
    exit line: untagged (grad mode on) / eval_untagged (off). The kit requires untagged == 0."""
    if _STATE != "armed":
        return
    if torch.jit.is_scripting() or _tracing(seq_embeddings) or _graph_task_id() != -1:
        return
    if seq_embeddings.dim() == 0 or int(seq_embeddings.shape[0]) == 0:
        return  # B = 0 on this rank: nothing to dedup
    COUNTS["untagged" if torch.is_grad_enabled() else "eval_untagged"] += 1


# ------------------------------------------------------------------------------------------------ before RUN_START
def _ids(T: int, n_items: int, U: int, heavy: int, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic integer ids (no RNG): items by a multiplicative hash, `heavy` tokens on item 7; the two additional
    rows are functions of the item (as in the dataset), in [n_items, U)."""
    t = torch.arange(T, dtype=torch.int64, device=dev)
    item = (t * 7919 + 13) % n_items
    if heavy:
        item = torch.where((t % 2 == 0) & (t < 2 * heavy), torch.full_like(item, 7), item)
    span = max(1, (U - n_items) // 2)
    a = n_items + (item * 31) % span
    b = n_items + span + (item * 17) % span
    return item, torch.stack([a, b], dim=1).reshape(-1)


def _host_reference(idx_item: torch.Tensor, idx_cat: torch.Tensor, U: int, m_pad: int) -> Triples:
    """The same construction from torch.unique on the host (CPU int64), for the self-check."""
    item, cat2 = idx_item.cpu(), idx_cat.cpu().view(-1, 2)
    key = (item * U + cat2[:, 0]) * U + cat2[:, 1]
    uniq, inv, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
    n = uniq.numel()
    order = torch.argsort(inv, stable=True)
    offsets = torch.full((m_pad + 1,), key.numel(), dtype=torch.int64)  # empty segments for u >= n
    offsets[0] = 0
    offsets[1:n + 1] = torch.cumsum(counts, 0)
    first = order[offsets[:-1].clamp(max=key.numel() - 1)]
    rows = (item[first], cat2[first, 0], cat2[first, 1])
    if padspread():  # the same rule written out on the host
        rows = tuple(torch.cat([r[:n], torch.arange(n, m_pad, dtype=r.dtype) % U]) for r in rows)
    return Triples(inv, order, offsets, rows, torch.tensor([n]))


def self_check(device: torch.device, T: int = CHECK_T, heavy: int = CHECK_HEAVY) -> str:
    """Exact checks on this device (raise on any mismatch): (a) the construction == the host torch.unique reference
    (inv, order, offsets, rows, n_u) on a batch with a heavy triple; (b) the gather and the backward's segment sums
    on exactly representable values; (d) hi + lo == the fp32 sum exactly. Synthetic integer data, no RNG."""
    U, n_items = 4_000, 3_000
    item, cat = _ids(T, n_items, U, heavy, device)
    m_pad = pad_rows(U, T)
    tri = build_triples(item, cat, U, m_pad)
    ref = _host_reference(item, cat, U, m_pad)
    n_u = int(tri.n_u.cpu()[0])
    if n_u != int(ref.n_u[0]) or n_u > m_pad:
        raise RuntimeError(f"self-check (a): n_u {n_u} != reference {int(ref.n_u[0])} (M_pad {m_pad})")
    for name, got, want in (("inv", tri.inv, ref.inv), ("order", tri.order, ref.order),
                            ("offsets", tri.offsets, ref.offsets), ("rows item", tri.rows[0], ref.rows[0]),
                            ("rows a", tri.rows[1], ref.rows[1]), ("rows b", tri.rows[2], ref.rows[2])):
        if not torch.equal(got.cpu(), want):
            raise RuntimeError(f"self-check (a): {name} differs from the host reference")
    if lean():  # (a) LEAN: the training [u; u] index rows == today's cat / stack layout of the reference rows
        _, d_i, d_ab = _lean_triples(item, cat, U, m_pad, 2)
        want_ab = torch.stack([torch.cat([ref.rows[1]] * 2), torch.cat([ref.rows[2]] * 2)], 1).reshape(-1)
        if not (torch.equal(d_i.view(-1).cpu(), torch.cat([ref.rows[0]] * 2))
                and torch.equal(d_ab.view(-1).cpu(), want_ab)):
            raise RuntimeError("self-check (a): LEAN hi / lo index rows differ from the host reference")
    # (b) / (d): src rows and dO entries are small integers (exact in bf16); per-triple sums < 2^16 (exact in fp32
    # and exactly hi + lo).
    w = WIDTH
    u = torch.arange(2 * m_pad, device=device).view(-1, 1)
    c = torch.arange(w, device=device).view(1, -1)
    src = (((u % m_pad) * 3 + c) % 61 - 30).to(torch.bfloat16).requires_grad_(True)
    from generative_recommenders.dlrm_v4.train.ec_bf16_output import _expand

    out = GatherSegReduceBwd.apply(src, tri.inv, tri.order, tri.offsets, m_pad, True, _expand)
    if not torch.equal(out.detach().cpu(), src.detach().cpu()[:m_pad][ref.inv]):
        raise RuntimeError("self-check (b): gather differs from src[inv]")
    t = torch.arange(T, device=device).view(-1, 1)
    dO = ((t * 5 + c) % 13 - 6).to(torch.bfloat16)
    out.backward(dO)
    g = src.grad.detach().cpu()
    want = torch.zeros((m_pad, w), dtype=torch.float64).index_add_(0, ref.inv, dO.cpu().double())
    hi_ref = want.float().to(torch.bfloat16)
    lo_ref = (want.float() - hi_ref.float()).to(torch.bfloat16)
    if not (torch.equal(g[:m_pad], hi_ref) and torch.equal(g[m_pad:], lo_ref)):
        raise RuntimeError("self-check (b): segment sums (hi / lo) differ from the fp64 reference")
    if not torch.equal(g[:m_pad].double() + g[m_pad:].double(), want):
        raise RuntimeError("self-check (d): hi + lo does not reconstruct the exact sum")
    return (f"construction == host reference at T={T} (n_u={n_u}, M_pad={m_pad}, heavy triple {heavy} tokens); "
            "gather + fp32 segment sums exact; hi + lo exact")


class _SynthMLP:
    """FusedLinearNormMLP's fused path (two _LinearNormFunction pairs) on synthetic weights: the warm-up launches
    exactly what the unique chain launches (pair 0 SwishLN256, pair 1 LN512, F2 residual on the last pair)."""

    def __init__(self, k: int, dev: torch.device, salt: int) -> None:
        def p(*shape: int) -> torch.Tensor:
            n = 1
            for s in shape:
                n *= s
            v = ((torch.arange(n, device=dev, dtype=torch.int64) * 37 + salt) % 29 - 14).to(torch.float32)
            return (v / 512.0).view(*shape).requires_grad_(True)

        self.p = [p(256, k), p(256), p(256), p(256), p(WIDTH, 256), p(WIDTH), p(WIDTH), p(WIDTH)]

    def __call__(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        from generative_recommenders.ops.fused_linear_norm import _LinearNormFunction

        w0, b0, n0w, n0b, w1, b1, n1w, n1b = self.p
        h = _LinearNormFunction.apply(x, w0, b0, n0w, n0b, 1e-6, True, None)
        return _LinearNormFunction.apply(h, w1, b1, n1w, n1b, 1e-6, False, residual)


class _SynthPre:
    def __init__(self, dev: torch.device) -> None:
        self._content_embedding_mlp = _SynthMLP(WIDTH, dev, 3)
        self._additional_embedding_mlp = _SynthMLP(2 * WIDTH, dev, 5)

    def _mlp_add(self, mlp: Any, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return mlp(x, residual=residual)


def _pattern(rows: int, cols: int, mod: int, scale: float, dev: torch.device,
             dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """[rows, cols] of small exact values from a repeated 1024-row block (no RNG; no [rows, cols] int64 temp)."""
    blk = min(rows, 1024)
    base = torch.arange(blk * cols, device=dev, dtype=torch.int64).view(blk, cols)
    base = ((base * 13) % mod - mod // 2).to(dtype) * scale
    reps = (rows + blk - 1) // blk
    return base.repeat(reps, 1)[:rows].contiguous() if reps > 1 else base


def _warm_class(dev: torch.device, T: int, U: int, train: bool) -> None:
    """One (T, U) class, FORWARD ONLY plus the backward's segment reduces called directly. No backward runs
    through _LinearNormFunction: its LN backward is PR #7's FlyDSL kernel, which must not be imported before the
    DataLoader pools fork (FlyDSL's jit key holds no shape; its D512 / D256-swish dx-sum variants are prebuilt after
    the fork by blas_warmup._prebuild_flydsl_ln_bwd). The bf16 GEMMs are hipBLASLt (no JIT)."""
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import segment_reduce_rows
    from generative_recommenders.ops.fused_linear_norm import _LinearNormFunction

    item, cat = _ids(T, U // 2, U, 0, dev)
    emb = _pattern(U, WIDTH, 17, 1.0 / 64, dev, torch.float32)
    rec = PPDDRecord(emb, item, cat, (), ebo._expand, ebo._CastExpandFp32Bwd.apply, torch.bfloat16)
    pre = _SynthPre(dev)
    xin = _pattern(T, 256, 23, 1.0 / 32, dev)
    xw = _SynthMLP(256, dev, 7).p  # only the last pair: the x combine (F2 epilogue at T rows)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
        acc = _run_core(pre, rec, hilo=train)  # train: R = 2 M_pad rows (hi / lo copies); eval: R = M_pad
        _LinearNormFunction.apply(xin, xw[4], xw[5], xw[6], xw[7], 1e-6, False, acc)
    if train:  # the gather backward and the two cast-gather backwards, as GatherSegReduceBwd / the EC issue them
        m = pad_rows(U, T)
        tri = build_triples(item, cat, U, m)
        _segment_sum(_pattern(T, WIDTH, 7, 1.0 / 8, dev), tri.inv, tri.order, tri.offsets, m)
        r_i = torch.cat([tri.rows[0], tri.rows[0]])
        r_ab = torch.stack([torch.cat([tri.rows[1], tri.rows[1]]), torch.cat([tri.rows[2], tri.rows[2]])], 1)
        segment_reduce_rows(_pattern(2 * m, WIDTH, 5, 1.0 / 8, dev), r_i, U)
        segment_reduce_rows(_pattern(4 * m, WIDTH, 5, 1.0 / 8, dev), r_ab.reshape(-1), U)


def _x_combine_check(dev: torch.device, n: int = 65_537) -> str:
    """Report only: the F2 epilogue O = bf16(bf16(LN(h)) + r) vs the two-step LN then aten add (bitwise under the
    LN512 autotune pins, the grouped-tail kernel's own contract)."""
    from generative_recommenders.ops.fused_linear_norm import _norm_fwd

    r_ = torch.arange(n * WIDTH, device=dev, dtype=torch.int64).view(n, WIDTH)
    h = ((r_ * 7 % 31 - 15).to(torch.bfloat16) / 8)
    res = ((r_ * 11 % 19 - 9).to(torch.bfloat16) / 16)
    w = torch.ones(WIDTH, device=dev) + (torch.arange(WIDTH, device=dev) % 5).float() / 8
    b = (torch.arange(WIDTH, device=dev) % 3).float() / 4
    y1 = _norm_fwd(h, w, b, 1e-6, False, res)[0]
    y2 = _norm_fwd(h, w, b, 1e-6, False, None)[0] + res
    bad = int((y1 != y2).sum().cpu())
    return "x combine F2 epilogue == LN + add: bitwise" if bad == 0 else f"x combine: {bad} elements differ (ulp)"


def warmup(device: torch.device, classes: Sequence[Tuple[int, int]] = WARM_CLASSES) -> str:
    """Every Triton launch class of the dedup path before RUN_START: per (T, U) class the training forward (R = 2
    M_pad) and the eval forward (R = M_pad) of the unique chain and the x combine at T rows, plus the backward's
    segment reduces (gather and cast-gathers) called directly; synthetic integer data, no RNG, no FlyDSL import
    (no LN backward runs here). Device type cuda only (Triton)."""
    if device.type != "cuda":
        return "warm-up skipped (no Triton on " + device.type + ")"
    done = []
    saved = dict(_MAX)
    try:
        for T, U in classes:
            _warm_class(device, T, U, train=True)
            _warm_class(device, T, U, train=False)
            done.append(f"T={T}/R={2 * pad_rows(U, T)}+{pad_rows(U, T)}")
        torch.cuda.synchronize(device)
    finally:  # the synthetic classes never enter the run's counters / overflow flag
        _MAX.update(saved)
        _OVF.clear()
    return "warm-up (forward + segment reduces) " + ", ".join(done) + "; " + _x_combine_check(device)


def prebuild(device: torch.device, classes: Sequence[Tuple[int, int]] = WARM_CLASSES) -> bool:
    """dlrm_v4/train/blas_warmup._prebuild_preproc_dedup, before RUN_START: IGNORED (registration / device), or the
    self-check + warm-up, then ONE ARMED line on this rank. A failure leaves the path NOT armed (today's
    statements) and startup goes on. Never raises."""
    if _STATE != "unset":
        return _STATE == "armed"
    try:
        why = chain_reason() or _device_reason(device)
        if why is not None:
            mark_ignored(why)
            return False
        t0 = time.perf_counter()
        checks = self_check(device)
        warm = warmup(device, classes)
    except Exception as exc:
        mark_not_armed(f"pre-run_start warm-up / self-check failed: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    mark_armed(device, f"pre-run_start warm-up + self-check OK: {checks}; {warm}; {time.perf_counter() - t0:.1f}s")
    return True


def _register_exit() -> None:
    global _EXIT_REGISTERED
    if not _EXIT_REGISTERED:
        _EXIT_REGISTERED = True
        # clean_exit.EXIT_HOOKS lists ("generative_recommenders.ops.preproc_dedup", "_at_exit"): the hard exit
        # (os._exit) skips plain atexit callbacks.
        atexit.register(_at_exit)


def counters_line(block: bool = False) -> str:
    ovf, n_u, complete = _overflow_read(block)
    c = COUNTS
    return (f"{COUNTERS_HEAD} train={c['train']} dedup={c['dedup']} eval={c['eval']} eval_dedup={c['eval_dedup']} "
            f"declined={c['declined']} disabled={c['disabled']} overflow={max(ovf, c['overflow'])} "
            f"max_U={_MAX['U']} max_T={_MAX['T']} max_n_u={_MAX['n_u']} untagged={c['untagged']} "
            f"eval_untagged={c['eval_untagged']}" + ("" if complete else " (flag copy pending)"))


def _at_exit() -> None:
    """Print only, no device work (the last flag copy has long completed at teardown; if not, the line says so)."""
    if _STATE in ("armed", "disabled"):
        try:
            _say(counters_line())
        except Exception:
            pass
