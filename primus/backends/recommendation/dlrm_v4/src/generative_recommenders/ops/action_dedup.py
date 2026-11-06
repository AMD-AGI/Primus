# pyre-strict
"""Action-branch dedup (gin item (119), DLRM_HSTU_FP8_ACTION_DEDUP; code default OFF).

WHAT. ContextualPreprocessor's action branch feeds every merged row through the
action MLP (Linear 24->256, SwishLayerNorm(256), Linear 256->512,
LayerNorm(512)). The MLP's input row is a pure function of a small code: for a
UIH row, concat_j(bit_j * action_table[j]) with bit_j = (action & w_j) > 0
(ActionEncoder); for a target row, the single learned target row. So there are
at most 2**T + 1 distinct input rows (T = 3 action types: 9 rows), and the MLP
is row-wise. With the knob on, each forward evaluates the module's OWN action
MLP (same modules, parameters, container and autocast) on those code rows and
adds table[code] to the residual chain instead of running the MLP on every
merged row:

    code[t]  = sum_j bit_j(action) << j  (UIH rows),  2**T  (target rows)
    E[c]     = concat_j(bit_j(c) * action_table[j]),   E[2**T] = target_table[0]
    atab2    = action_mlp(cat([E, E]))                 (2 * (2**T + 1) rows)
    out[t]   = residual[t] + atab2[code[t]]            (one rounding, as F2)

MLPerf. User-approved compute reduction for this item only (2026-10-15
17:58Z). The table is RECOMPUTED EVERY FORWARD from the current weights;
nothing is cached across forwards, steps or eval batches. It is the EXACT MLP,
not an approximation: every token's MLP input row is bitwise one of the code
rows (same fp32 products as ActionEncoder, same RNE cast in the first Linear),
and the MLP has no cross-row operation (no dropout, no RNG, no position or
time feature, no FP8 operand / amax / scale inside it).

BACKWARD (class REDUCTION-ORDER: deterministic structure, not bitwise).
d(residual) is the incoming gradient itself (no kernel). d(table) is the
per-code sum of the incoming gradient in fp32 (no atomics: a one-hot GEMM over
fixed row chunks with fp32 output, then a fixed-order fp32 sum of the chunk
partials). The two stacked copies of the table receive hi = bf16(S) and
lo = bf16(S - hi), so the module's bf16 MLP backward (run by autograd on the
2 * (2**T + 1) rows) keeps ~16 bits of S. Gradients to the action / target
embedding tables flow through E by autograd.

TORCH OPS ONLY. No new GPU kernel: index_select + add_ (forward), bmm with
out_dtype=float32 or a chunked fp32 mm (backward), small integer ops (codes).
The MLP itself runs the shipped kernels at 2 * (2**T + 1) rows.

ARMING. GPU: only after prebuild() (dlrm_v4/train/blas_warmup.run_blas_warmup,
before RUN_START, every rank) passed on the device; it prints the ARMED line.
Without it (BLAS_WARMUP off) the first forward prints one "NOT armed" line and
runs the shipped path. CPU arms without a prebuild (tests). Kill switch:
DLRM_HSTU_FP8_ACTION_DEDUP=0 (a pre-set env value wins over the gin binding).
"""

import logging
import os
from typing import Dict, Optional, Set, Tuple

import torch
from torch.autograd.function import once_differentiable
from torch.fx._symbolic_trace import is_fx_tracing

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_ACTION_DEDUP"
# Sub-knob LEAN (2026-10-24; only "1" arms it; "0" is the kill switch): host launch cuts, same values. codes()
# takes the UIH lengths from the prefill (no seq = uih + n then uih = seq - n), gets the row -> sample map from
# repeat_interleave(lens) (== repeat_interleave(arange(B), lens)), and reads 1 << arange(T), the code-bit table
# and segsum's arange(kp) from a per-(device, dtype, size) cache of constants; segsum allocates its zero
# accumulator only where it is read; the backward's hi / lo split is two launches into one buffer (as (118)
# HILO2). Forward 13 and backward 5 launches fewer per training step at T = 3 (the table is still recomputed
# every forward; only index constants are cached).
LEAN_ENV: str = "DLRM_HSTU_FP8_ACTION_DEDUP_LEAN"
TAG: str = "[action-dedup]"
ITEM: str = "(119)"
# 2**T + 1 code rows; T is 3 in production (action_weights [1, 2, 4]).
MAX_T: int = 6
# Rows per one-hot GEMM chunk of the backward per-code sum (bmm path), and per
# fp32 cast chunk of the fallback (bounded fp32 transient: 64 MiB at D = 512).
CHUNK: int = 8192
F32_CHUNK: int = 32768
# The production table: T = 3 -> 2 * 9 rows (the prebuild warms this size).
PROD_T: int = 3

_logged: Set[str] = set()
# Devices whose prebuild passed (cuda index), the backward per-code sum path per
# device ("bmm" or "f32"), and a process-wide latch (reason) after an error.
_prebuilt: Dict[int, str] = {}
_latched: Optional[str] = None
stats: Dict[str, int] = {"forwards": 0, "declined": 0}
_LEAN: Optional[bool] = None
# LEAN constants: (kind, device, dtype, n) -> tensor. Pure functions of the key
# (aranges / powers of two / the code-bit table), never of batch data.
_CONST: Dict[Tuple[str, torch.device, torch.dtype, int], torch.Tensor] = {}


def enabled() -> bool:
    """The knob, read per call: "1" / "true" / "on" / "yes" (any case) arm it; absent, "0" or any other value is
    off (yambda_5b.gin binds True)."""
    return os.environ.get(ENV_VAR, "0").strip().lower() in ("1", "true", "on", "yes")


def lean() -> bool:
    """DLRM_HSTU_FP8_ACTION_DEDUP_LEAN == "1", read once per process (unset / "0" / other: off)."""
    global _LEAN
    if _LEAN is None:
        _LEAN = os.environ.get(LEAN_ENV, "").strip() == "1"
    return _LEAN


def reset_for_testing() -> None:
    global _LEAN
    _LEAN = None
    _CONST.clear()


def _pow2(dev: torch.device, n: int) -> torch.Tensor:
    # The shipped expression: 1 << arange(n), int64.
    return torch.bitwise_left_shift(
        torch.ones((n,), dtype=torch.int64, device=dev),
        torch.arange(n, dtype=torch.int64, device=dev),
    )


def _const(kind: str, dev: torch.device, dtype: torch.dtype, n: int) -> torch.Tensor:
    """LEAN: a constant built once per (kind, device, dtype, n) by the shipped
    expression (no host read; outside inference mode so autograd may save it).
    "pow2": 1 << arange(n) int64; "bits": [2**n, n] bool, bit j of code c;
    "arange": arange(n) of dtype."""
    key = (kind, dev, dtype, n)
    t = _CONST.get(key)
    if t is None:
        with torch.inference_mode(False):
            if kind == "pow2":
                t = _pow2(dev, n)
            elif kind == "bits":
                c = torch.arange(1 << n, dtype=torch.int64, device=dev)
                t = torch.bitwise_and(c.unsqueeze(-1), _pow2(dev, n).unsqueeze(0)) > 0
            else:
                t = torch.arange(n, dtype=dtype, device=dev)
        _CONST[key] = t
    return t


def _say(key: str, msg: str) -> None:
    """One line per process per key on EVERY rank (" [rank N]")."""
    if key in _logged:
        return
    _logged.add(key)
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # never fail a step over a log line
        logger.warning("%s", msg)


def latch(where: str, error: BaseException) -> None:
    """Latch the item off process-wide (the shipped path runs from then on)."""
    global _latched
    if _latched is None:
        _latched = f"{where}: {type(error).__name__}: {error}"
        _say(
            "disabled",
            f"{TAG} DISABLED ({_latched}); the action branch runs per token "
            f"from now on (kill switch {ENV_VAR}=0)",
        )


def _num_types(enc: torch.nn.Module) -> int:
    # pyre-ignore[16]
    return int(enc._num_action_types)


def decline_reason(
    pre_mod: torch.nn.Module,
    ref: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
) -> Optional[str]:
    """None when the dedup may run for this forward of `pre_mod` (a
    ContextualPreprocessor with an action branch); `ref` is any tensor on the
    forward's device. Otherwise the reason the shipped path runs."""
    if torch.jit.is_scripting() or is_fx_tracing() or torch.compiler.is_compiling():
        return "scripting / fx tracing / torch.compile"
    if isinstance(ref, torch.fx.Proxy):
        return "fx Proxy input"
    if _latched is not None:
        return "DISABLED"
    enc = getattr(pre_mod, "_action_encoder", None)
    mlp = getattr(pre_mod, "_action_embedding_mlp", None)
    if enc is None or mlp is None:
        return "no action encoder / MLP"
    if _num_types(enc) > MAX_T:
        return f"T={_num_types(enc)} > {MAX_T} action types"
    # pyre-ignore[16]
    names = [enc._action_feature_name]
    # pyre-ignore[16]
    if len(enc._watchtime_to_action_thresholds_and_weights) > 0:
        names.append(enc._watchtime_feature_name)
    for name in names:
        if name not in seq_payloads:
            return f"payload {name} missing"
    dev = ref.device.type
    if torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) == torch.float16:
        # fp16: the per-code sum of the gradient could exceed 65504.
        return "fp16 autocast"
    if ref.is_cuda:
        if not (
            torch.is_autocast_enabled(dev)
            and torch.get_autocast_dtype(dev) == torch.bfloat16
        ):
            return "not under torch.autocast(cuda, bfloat16)"
        idx = ref.device.index
        if (torch.cuda.current_device() if idx is None else idx) not in _prebuilt:
            return "NOT armed: the pre-run_start prebuild did not run on this device"
    return None


def note_decline(reason: str) -> None:
    stats["declined"] += 1
    key = "notarmed" if reason.startswith("NOT armed") else "decline:" + reason.split(" ")[0]
    word = "NOT armed" if key == "notarmed" else "DECLINED"
    text = reason[len("NOT armed: "):] if key == "notarmed" else reason
    _say(
        key,
        f"{TAG} {word} ({text}); this forward runs the action branch per token "
        f"(kill switch {ENV_VAR}=0)",
    )


def codes(
    enc: torch.nn.Module,
    seq_payloads: Dict[str, torch.Tensor],
    seq_lengths: Optional[torch.Tensor],
    num_targets: torch.Tensor,
    num_rows: int,
    uih_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """int64 [num_rows]: each merged row's code (UIH rows of a sample first,
    then its targets: ActionEncoder's concat_2D_jagged layout). Integer torch
    ops only, no host sync. `uih_lengths` (LEAN, from the prefill) replaces
    seq_lengths - num_targets; then seq_lengths may be None."""
    # pyre-ignore[16]
    actions = seq_payloads[enc._action_feature_name]
    # Watchtime thresholds OR'ed in exactly as ActionEncoder.forward does.
    # pyre-ignore[16]
    for threshold, weight in enc._watchtime_to_action_thresholds_and_weights:
        # pyre-ignore[16]
        watchtimes = seq_payloads[enc._watchtime_feature_name]
        actions = torch.bitwise_or(
            actions, (watchtimes >= threshold).to(torch.int64) * weight
        )
    T = _num_types(enc)
    dev = actions.device
    lean_ = lean()
    # pyre-ignore[16]
    bits = torch.bitwise_and(actions.unsqueeze(-1), enc._combined_action_weights.unsqueeze(0)) > 0
    pow2 = _const("pow2", dev, torch.int64, T) if lean_ else _pow2(dev, T)
    code_uih = (bits * pow2).sum(-1)
    n_uih = actions.size(0)
    code = torch.full((num_rows,), 1 << T, dtype=torch.int64, device=dev)
    if n_uih == 0:
        return code
    num_targets = num_targets.to(torch.int64)
    if uih_lengths is not None:
        # (uih + n) - n == uih (integer add / sub, exact).
        uih_len = uih_lengths.to(torch.int64)
    else:
        # pyre-ignore[16]: seq_lengths is given when uih_lengths is not
        uih_len = seq_lengths.to(torch.int64) - num_targets
    if lean_:
        # repeat_interleave(arange(B), lens) is index_select(arange(B), r) with
        # r = repeat_interleave(lens): r itself (int64, the same values).
        batch = torch.repeat_interleave(uih_len, output_size=n_uih)
    else:
        batch = torch.repeat_interleave(
            torch.arange(num_targets.size(0), dtype=torch.int64, device=dev),
            uih_len,
            output_size=n_uih,
        )
    target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
    # UIH row i of sample b lands at merged row i + target_offsets[b].
    dst = torch.arange(n_uih, dtype=torch.int64, device=dev) + target_offsets.index_select(
        0, batch
    )
    code.index_copy_(0, dst, code_uih)
    return code


def table_inputs(enc: torch.nn.Module) -> torch.Tensor:
    """[2**T + 1, T * a]: the encoder rows of every code, built with
    ActionEncoder's own expression (bitwise its per-token rows)."""
    T = _num_types(enc)
    # pyre-ignore[16]
    table = enc._action_embedding_table
    dev = table.device
    if lean():
        # The [2**T, T] code-bit table depends on T only (built once, below's ops).
        bits = _const("bits", dev, torch.bool, T)
    else:
        c = torch.arange(1 << T, dtype=torch.int64, device=dev)
        w = _pow2(dev, T)
        bits = torch.bitwise_and(c.unsqueeze(-1), w.unsqueeze(0)) > 0
    uih_rows = (bits.unsqueeze(-1) * table.unsqueeze(0)).view(1 << T, -1)
    # pyre-ignore[16]
    return torch.cat([uih_rows, enc._target_action_embedding_table], dim=0)


def plan(
    pre_mod: torch.nn.Module,
    seq_payloads: Dict[str, torch.Tensor],
    seq_lengths: Optional[torch.Tensor],
    num_targets: torch.Tensor,
    num_rows: int,
    uih_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(code [N] int64, atab2 [2 * (2**T + 1), D]): the module's own action MLP
    on the stacked code rows, from the current weights, under the caller's
    autocast. Call after decline_reason(...) returned None. uih_lengths: see
    codes()."""
    # pyre-ignore[16]
    enc = pre_mod._action_encoder
    code = codes(enc, seq_payloads, seq_lengths, num_targets, num_rows, uih_lengths)
    e = table_inputs(enc)
    # pyre-ignore[16]
    atab2 = pre_mod._action_embedding_mlp(torch.cat([e, e], dim=0))
    return code, atab2


def try_plan(
    pre_mod: torch.nn.Module,
    seq_payloads: Dict[str, torch.Tensor],
    seq_lengths: Optional[torch.Tensor],
    num_targets: torch.Tensor,
    num_rows: int,
    uih_lengths: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """plan(...), or None after latching the item off on a (host-visible)
    error: the caller then runs the shipped per-token branch."""
    try:
        return plan(pre_mod, seq_payloads, seq_lengths, num_targets, num_rows, uih_lengths)
    except Exception as error:  # fail closed to the shipped branch
        latch("plan", error)
        return None


def _segsum_path(device: torch.device) -> str:
    if device.type != "cuda":
        return "f32"
    idx = torch.cuda.current_device() if device.index is None else device.index
    return _prebuilt.get(idx, "f32")


def segsum(code: torch.Tensor, g: torch.Tensor, k: int, path: str) -> torch.Tensor:
    """[k, D] per-code sums of g's rows: fp32 (fp64 for fp64 g), deterministic
    (fixed row chunks, fp32 partials, fixed-order sum), no atomics.

    path "bmm": bf16/fp16 g -> one-hot bmm over CHUNK-row batches with
    out_dtype=float32 (one launch), torch.sum over the batch partials, then the
    remainder rows by an fp32 mm. path "f32": each F32_CHUNK rows of g are cast
    to fp32 (bounded transient) and reduced by an fp32 one-hot mm."""
    g = g.contiguous()
    N, D = g.shape
    acc = torch.float64 if g.dtype == torch.float64 else torch.float32
    kp = ((k + 15) // 16) * 16
    lean_ = lean()
    # LEAN: the zero accumulator only where it is read (N == 0, the f32 path,
    # the bmm path without a full chunk); the fp32 / fp64 g path and the bmm
    # path with chunks overwrite it unread.
    s: Optional[torch.Tensor] = None if lean_ else torch.zeros((kp, D), dtype=acc, device=g.device)
    if N == 0:
        return (torch.zeros((kp, D), dtype=acc, device=g.device) if s is None else s)[:k]
    if lean_:
        ar = _const("arange", g.device, code.dtype, kp)
    else:
        ar = torch.arange(kp, dtype=code.dtype, device=g.device)
    if g.dtype in (torch.float32, torch.float64):
        oh = (code.unsqueeze(1) == ar).to(g.dtype)
        return torch.mm(oh.t(), g)[:k]
    start = 0
    if path == "bmm":
        oh = (code.unsqueeze(1) == ar).to(g.dtype)
        p = N // CHUNK
        start = p * CHUNK
        if p > 0:
            part = torch.bmm(
                oh[:start].view(p, CHUNK, kp).transpose(1, 2),
                g[:start].view(p, CHUNK, D),
                out_dtype=torch.float32,
            )
            s = part.sum(0)
            del part
        if start < N:
            if s is None:
                s = torch.zeros((kp, D), dtype=acc, device=g.device)
            s = s + torch.mm(oh[start:].t().float(), g[start:].float())
        # pyre-ignore[16]: N > 0, so a chunk or the remainder set s
        return s[:k]
    if s is None:
        s = torch.zeros((kp, D), dtype=acc, device=g.device)
    oh32 = (code.unsqueeze(1) == ar).to(torch.float32)
    for lo in range(0, N, F32_CHUNK):
        hi = min(N, lo + F32_CHUNK)
        s.addmm_(oh32[lo:hi].t(), g[lo:hi].float())
    return s[:k]


class CodeGatherAdd(torch.autograd.Function):
    """out = residual + atab2[:k][code] (one rounding: bf16(f32(r) + f32(t))
    for bf16, as F2's round-before-add). Backward: d(residual) = g; d(atab2) =
    [bf16(S); bf16(S - bf16(S))] with S the fp32 per-code sum of g (rows
    k..2k-1 receive the low part)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        residual: torch.Tensor,
        atab2: torch.Tensor,
        code: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        out = atab2[:k].index_select(0, code)
        if out.dtype == residual.dtype:
            out.add_(residual)  # IEEE addition commutes: bitwise residual + t
        else:  # the shipped `+` type-promotes (F1/F2 off)
            out = residual + out
        ctx.save_for_backward(code)
        ctx.k = k
        ctx.tab = (atab2.dtype, atab2.shape[0])
        ctx.res_dtype = residual.dtype
        ctx.path = _segsum_path(residual.device)
        return out

    @staticmethod
    @once_differentiable
    # pyre-ignore[14]
    def backward(ctx, g: torch.Tensor):  # pyre-ignore[2, 3]
        (code,) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        d_res = None
        if needs[0]:
            d_res = g if g.dtype == ctx.res_dtype else g.to(ctx.res_dtype)
        d_tab = None
        if needs[1]:
            dtype, rows = ctx.tab
            s = segsum(code, g, ctx.k, ctx.path)
            if rows == 2 * ctx.k and dtype != s.dtype and lean():
                # LEAN, as (118) HILO2: hi = copy_(s) (one rounding), lo = sub(s, hi)
                # computed in s's dtype with one rounding on store -- the values of
                # cast / cast / sub / cast / cat below, two launches into one buffer.
                k = ctx.k
                d_tab = torch.empty((2 * k, s.shape[1]), dtype=dtype, device=s.device)
                d_tab[:k].copy_(s)
                torch.sub(s, d_tab[:k], out=d_tab[k:])
                return d_res, d_tab, None, None
            hi = s.to(dtype)
            if rows == 2 * ctx.k:
                lo = (s - hi.to(s.dtype)).to(dtype)
                d_tab = torch.cat([hi, lo], dim=0)
            else:
                d_tab = hi
        return d_res, d_tab, None, None


def gather_add(residual: torch.Tensor, atab2: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """out = residual + table[code]; atab2 holds the table twice (hi/lo rows)."""
    k = atab2.shape[0] // 2
    stats["forwards"] += 1
    return CodeGatherAdd.apply(residual, atab2, code, k)


def take_or_plan(
    pre_mod: torch.nn.Module,
    pre,  # pyre-ignore[2]: modules/action_prefill.py ActionPrefill, or None
    seq_embeddings: torch.Tensor,
    seq_lengths: torch.Tensor,
    num_targets: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, str]]:
    """ContextualPreprocessor.forward's (119) entry, called only with the knob
    on: (code, table, prefill) for this forward, or None (the shipped action
    branch runs). A taken (65) prefill carries the plan issued before the
    embedding all-to-all wait (its table re-attached by the prefill bridge); a
    per-token prefill (the plan declined at issue, already logged) is consumed
    as shipped; without a prefill (eval, (65) off, FALLBACK) the plan runs
    here."""
    if pre is not None:
        code = getattr(pre, "dedup", None)
        if code is None:
            return None
        return code, pre.attach(), "yes"
    why = decline_reason(pre_mod, seq_embeddings, seq_payloads)
    if why is not None:
        note_decline(why)
        return None
    got = try_plan(
        pre_mod, seq_payloads, seq_lengths, num_targets, int(seq_embeddings.shape[0])
    )
    return None if got is None else (got[0], got[1], "no")


def finish(
    residual: torch.Tensor,
    dd: Tuple[torch.Tensor, torch.Tensor, str],
    tail,  # pyre-ignore[2]: ops/preproc_grouped_tail.py GroupedTail, or None
) -> torch.Tensor:
    """After the content/additional MLP section: residual + table[code]."""
    code, atab2, prefill = dd
    out = gather_add(residual, atab2, code)
    if "first" not in _logged:
        t = "off" if tail is None else ("DECLINED" if tail.declined else "grouped inputs=2")
        _say(
            "first",
            f"{TAG} first dedup forward: rows={code.shape[0]} codes={atab2.shape[0] // 2} "
            f"tail={t} prefill={prefill} segsum={_segsum_path(residual.device)} "
            f"grad={torch.is_grad_enabled()}",
        )
    return out


# --------------------------------------------------------------------------- prebuild
def _armed_line(path: str, checks: str) -> str:
    return (
        f"armed OK: {TAG} ARMED ({ENV_VAR}=1): action MLP on 2**T+1=9 code rows x2 "
        "(hi/lo), recomputed every forward, exact MLP; out = residual + table[code] "
        f"(index_select + add); fp32 per-code grad sum (segsum={path}, fixed chunks, "
        f"deterministic structure); pre-run_start self-check OK ({checks}); "
        f"REDUCTION-ORDER item {ITEM}; lean={int(lean())} (kill switch {LEAN_ENV}=0); "
        f"kill switch {ENV_VAR}=0"
    )


def _check_segsum(device: torch.device, path: str, rows: int, k: int) -> None:
    """Bitwise vs an fp64 reference on dyadic data (every partial sum exact)."""
    i = torch.arange(rows, dtype=torch.int64, device=device)
    code = (i * 5 + 3) % k
    j = torch.arange(512, dtype=torch.int64, device=device)
    q = ((i.unsqueeze(1) * 7 + j.unsqueeze(0) * 3) % 17 - 8).to(torch.float32)
    g = (q * (2.0 ** -6)).to(torch.bfloat16)
    ref = torch.zeros((k, 512), dtype=torch.float64, device=device)
    ref.index_add_(0, code, g.double())
    got = segsum(code, g, k, path)
    if got.dtype != torch.float32 or not torch.equal(got.double(), ref):
        raise RuntimeError(f"segsum path {path} rows={rows} differs from the fp64 reference")
    again = segsum(code, g, k, path)
    if not torch.equal(got, again):
        raise RuntimeError(f"segsum path {path} rows={rows} is not deterministic")


def _check_gather_add(device: torch.device, k: int) -> None:
    rows = 4099
    i = torch.arange(rows, dtype=torch.int64, device=device)
    code = (i * 3 + 1) % k
    j = torch.arange(512, dtype=torch.int64, device=device)
    atab2 = ((j.unsqueeze(0) * 0.37 + torch.arange(2 * k, device=device).unsqueeze(1) * 1.3).sin()).to(
        torch.bfloat16
    )
    res = ((i.unsqueeze(1) * 0.11 + j.unsqueeze(0) * 0.07).cos()).to(torch.bfloat16)
    with torch.no_grad():
        out = CodeGatherAdd.apply(res, atab2, code, k)
    ref = (res.float() + atab2[:k].float()[code]).to(torch.bfloat16)
    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
        raise RuntimeError("gather-add is not bf16(f32(r) + f32(t))")


def _norm_warm_ok() -> bool:
    """True when the Triton autotune pins are on (production: gin
    DLRM_HSTU_FP8_AUTOTUNE_PINS = True, seeded by train_ranker before the BLAS
    warm-up; no TRITON_FULL_AUTOTUNE=1). Then the 18-row norm launches below only
    compile. With the pins off, the first launch of an autotuned norm forward
    picks the config, and its key holds no row count, so an 18-row warm-up
    would choose the config the ~2.3M-row launches then reuse (WBND r119 K1):
    skip the norm launches then."""
    return (
        os.environ.get("DLRM_HSTU_FP8_AUTOTUNE_PINS", "0") == "1"
        and os.environ.get("TRITON_FULL_AUTOTUNE", "0") != "1"
    )


def _warm_mlp(device: torch.device, rows: int) -> bool:
    """The table MLP's forward kernels at `rows` (Triton norm forwards at this
    N, hipBLASLt problem types) and its backward GEMMs, on private tensors.
    No backward norm launch here (PR #7's FlyDSL kernels are prebuilt after
    the DataLoader forks; their jit key holds no shape). Autotune pins off:
    the backward GEMMs only (_norm_warm_ok). Returns whether the norms ran."""
    from generative_recommenders.ops.fused_linear_norm import _LinearNormFunction

    norms = _norm_warm_ok()

    f32 = torch.float32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.full((rows, 24), 0.0625, dtype=f32, device=device)
        h = x
        for k_in, n_out, swish in ((24, 256, True), (256, 512, False)):
            w = torch.full((n_out, k_in), 0.0625, dtype=f32, device=device)
            b = torch.zeros((n_out,), dtype=f32, device=device)
            nw = torch.ones((n_out,), dtype=f32, device=device)
            nb = torch.zeros((n_out,), dtype=f32, device=device)
            h_in = h
            if norms:
                h = _LinearNormFunction.apply(h_in, w, b, nw, nb, 1e-6, swish, None)
            else:
                h = torch.zeros((rows, n_out), dtype=torch.bfloat16, device=device)
            dh = torch.ones((rows, n_out), dtype=torch.bfloat16, device=device)
            wc = w.to(torch.bfloat16)
            dh @ wc
            dh.t() @ h_in.to(torch.bfloat16)
    return norms


def _warm_grouped_tail_2(device: torch.device) -> int:
    """E-M1 (gin item (30)) grouped LayerNorm(512) forward with NUM_IN = 2 (the
    tail without the action entry): not in the TSPEC table. One launch per
    Triton class a real run reaches: N % 16 == 0 / != 0, x [N, 512] bf16
    storage <= 2 GiB / > 2 GiB (output allocated alike)."""
    if os.environ.get("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", "0") != "1":
        return 0
    from generative_recommenders.ops.triton.triton_preproc_tail import (
        grouped_ln_residual_fwd,
    )

    w = torch.ones((512,), dtype=torch.float32, device=device)
    b = torch.zeros((512,), dtype=torch.float32, device=device)
    n = 0
    for rows in ((1 << 17), (1 << 17) + 1, (1 << 21) + 16, (1 << 21) + 17):
        h = torch.full((rows, 512), 0.5, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            out, _, _ = grouped_ln_residual_fwd([h, h], [w, w], [b, b], 1e-6)
        del h, out
        n += 1
    return n


def prebuild(device: torch.device) -> None:
    """Pre-RUN_START prebuild + self-check on this rank's device (called by
    dlrm_v4/train/blas_warmup.run_blas_warmup). Knob not "1": returns before
    doing anything (prints one OFF line when it is "0"). No RNG draw, no
    module / parameter / optimizer state: synthetic tensors from arange/full.
    Leaves Triton compiled kernels (table-MLP norm forwards at the table rows,
    E-M1 NUM_IN = 2) and hipBLASLt code objects behind. A failure latches the
    item off (FAILED CLOSED line, the shipped path runs); a device fault
    raises at the final synchronize, before RUN_START."""
    raw = os.environ.get(ENV_VAR)
    if not enabled():
        if raw is not None and raw.strip().lower() in ("0", "false", "off", "no"):
            _say("off", f"{TAG} OFF ({ENV_VAR}={raw.strip()}): the action branch runs per token")
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    k = (1 << PROD_T) + 1
    idx = torch.cuda.current_device() if device.index is None else device.index
    failure: Optional[BaseException] = None
    path = "f32"
    checks = ""
    try:
        with torch.cuda.device(device):
            try:
                for rows in (3 * CHUNK + 5, CHUNK, 7):
                    _check_segsum(device, "bmm", rows, k)
                path = "bmm"
            except Exception as error:  # bmm out_dtype unsupported or wrong: fp32 chunks
                logger.warning("%s segsum bmm path unavailable (%s); using fp32 chunks", TAG, error)
                path = "f32"
                for rows in (F32_CHUNK + 5, 7):
                    _check_segsum(device, "f32", rows, k)
            _check_gather_add(device, k)
            norms = _warm_mlp(device, 2 * k)
            tails = _warm_grouped_tail_2(device)
            mlp = f"table MLP rows={2 * k}" if norms else f"table MLP rows={2 * k} GEMMs only (autotune pins off)"
            checks = f"segsum={path} bitwise vs fp64, gather-add bitwise, {mlp}, E-M1 NUM_IN=2 x{tails}"
    except Exception as error:
        failure = error
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            f"{TAG} device fault during the pre-run_start prebuild; the HIP context is "
            f"unusable. Rerun with {ENV_VAR}=0 to isolate."
        ) from error
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if failure is not None:
        _say(
            "failed",
            f"{TAG} FAILED CLOSED (prebuild: {type(failure).__name__}: {failure}); "
            "the action branch runs per token",
        )
        return
    _prebuilt[idx] = path
    _say("armed", _armed_line(path, checks))
