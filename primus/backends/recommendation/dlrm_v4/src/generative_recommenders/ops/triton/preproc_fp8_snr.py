# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""In-model numerics probe for FP8 preprocessor-MLP GEMMs (Arbor E-M3).

Diagnostic only. DEFAULT OFF, and inert when off: the call sites
(ops/fused_linear_norm.py, ops/preproc_grouped_tail.py,
modules/preprocessors.py) test ENABLED and run nothing else. Arm it with

    DLRM_HSTU_FP8_PREPROC_SNR="<count>[:<stride>]"   every stride-th training
                                call of each site, <count> calls per site
    DLRM_HSTU_FP8_PREPROC_SNR_RANKS="0,5"   ranks that measure (default "0")
    DLRM_HSTU_FP8_PREPROC_SNR_ROWS=32768    rows sampled for fwd/dgrad
    DLRM_HSTU_FP8_PREPROC_SNR_EMB=1         embedding-row update metric
    DLRM_HSTU_FP8_PREPROC_SNR_DIR=<dir>     write records to
                                <dir>/preproc_fp8_snr_rank<r>.jsonl (use this:
                                ranks share one stdout pipe and long lines
                                interleave). Default: stdout, as chunked lines.

The DLRM_HSTU_FP8_ prefix is forwarded by scripts/launch_slurm.sh.

Even when armed it never changes what the step computes. It reads the bf16
operands the shipped path already holds, quantizes copies on the side,
computes fp64 references, and writes one JSON record per sampled
(site, phase). Until the last sampled call, every training call of an armed
rank re-reads each operand once for its amax (~22 GB/step, ~4-5 ms at 1n); a
sampled step costs ~1.5-15 s on the armed rank (fp64 GEMMs over all tokens,
~2k host syncs) while the other ranks wait in their next collective. Never
arm it in a timed run. It draws nothing from the global RNG. Forwards under
torch.no_grad (eval) are not counted. Knob values must contain no
whitespace (launch_slurm.sh forwards them unquoted), and _DIR must be a path
the container can see (e.g. under /scratch). If the E-M1 grouped tail
declines, the three last pairs share the label L2_K256_N512.

SITES. Every (Linear, Norm) pair of the ContextualPreprocessor MLPs:
  L1c / L1a / L1x  first Linear of the content / additional / action MLP
                   (input X: [M,512] item embeddings, [M,1024] artist|album
                   embeddings, [M,24] action features). L1x has K=24, not a
                   multiple of 16, so it stays BF16 in the E-M3 design; it is
                   reported without FP8 arms.
  L2c / L2a / L2x  last Linear of each MLP (input a1 = SwishLN(h1) [M,256]),
                   from the E-M1 grouped tail; L2_K<K> if that tail declined.

FORMATS EMULATED (the E-M3 design, proposals/redesign_E_M3_fp8.md, reusing
the shipped item-22/23 machinery in ops/triton/bwd_fp8.py; its constants are
read from that module and echoed in every record as "ctl"):
  weights   e4m3, scale amax(W)/448 (bwd_fp8.quantize_weight), from the bf16
            autocast copy (the fp32-Parameter cast differs in ~2% of codes;
            immaterial for SNR).
  X         e4m3 at amax/448 of the current tensor ("e4m3"). The EC expand
            would quantize its unique-row table at its exact current amax;
            that table holds item, artist and album rows together, so its
            scale is the max over them (here: per GEMM input).
  a1        e4m3 at the current amax ("e4m3", "e4m3_m8" = margin 8), and
            "_dly": the delayed scale of a replayed _SiteScale-style
            controller (warmup, refresh, margin as for gradients; limit 448).
  dh        e5m2 at 8*amax/57344 of the current tensor ("e5m2") and, "_dly",
            at the replayed shipped controller's published amax: warmup
            calls run BF16, first publish at the last warmup call, then
            after every refresh-th call, each publish taking the max since
            the previous one. The e4m3 token-major wgrad copy is cast at that
            scale * 2^shift, the shift stepping as DFMT_SHIFT=auto does
            (floor, +1 on a clipped window, -1 after `relax` clean windows).
            "e4m3": ideal e4m3 at amax/448 (shown to price the format).

PHASES AND ARMS (SNR dB, gain <ref,x>/|ref|^2, cosine, mean error, and the
per-row SNR tail: pooled SNR hides rows whose small values a format flushes,
while row-wise Adagrad and element-wise Adam normalize per row / element):
  fwd    sampled rows; ref = fp64(x) @ fp64(W)^T + b.
         Arms: shipped (the bf16 h the step used), e4m3 (fp64 emulation,
         output rounded to bf16), e4m3_m8 / e4m3_dly (L2), e4m3_hw
         (torch._scaled_mm on the quantized operands: the real hipBLASLt
         FP8 GEMM).
  dgrad  sampled rows; ref = fp64(dh) @ fp64(W).
         Arms: shipped, e5m2, e5m2_dly, e4m3, e5m2_hw, and for L1c/L1a
         e5m2_chain: the L1 dgrad of the dh1 that an FP8 L2 dgrad would
         deliver (e5m2 dh2 x e4m3 W2 -> bf16 da1 -> the shipped SwishLN
         backward -> e5m2 dh1 x e4m3 W1).
         Also cast_mismatch_e5m2: fraction of e5m2 bytes that differ from
         bwd_fp8.quantize_row_major (the Triton cast E-M3 would use).
  wgrad  all tokens, full [N, K] output.
         Refs: exact = fp64(dh)^T fp64(x); fwd = fp64(dh)^T deq(x_e4m3), the
         gradient of the FP8 forward that would run (fwd_dly for a delayed a1).
         Arms: shipped, nnb_e5m2 (y = dh e5m2 row-major), nna_e4m3s5 (dout_t =
         dh e4m3 copy at 2^shift headroom), e4m3 (ideal), and _dly versions.
  emb    L1c/L1a only, all tokens. dX summed per unique embedding row, as the
         EC backward's segment reduce does before the row-wise Adagrad
         update. SCOPE: rank-local unique rows found by a randomized integer
         hash of each 512-wide bf16 row block (collision odds ~1e-9 per run;
         the additional MLP's artist and album halves are separate tables),
         and only the preprocessor MLP's share of each row's gradient.
         Reported globally and per local row multiplicity (1, 2-7, 8-63, 64+):
         pooled SNR, gain, mean error, per-row SNR quantiles.
         Arms: shipped (bf16 dX), e5m2, e4m3 (L1 dgrad cast only), e5m2_chain
         (L2 dgrad + L1 dgrad both FP8).
Each record carries the operand amax, flush/subnormal/saturation fractions,
and the replayed controller state (published amax, growth over it, clipped
calls) for the site's activation and gradient histories.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

_SPEC: str = os.environ.get("DLRM_HSTU_FP8_PREPROC_SNR", "")
_CHUNK: int = 131072
_TAG: str = "[preproc-fp8-snr] "
# Ranks share one stdout pipe and a long write can be interleaved with another
# rank's, so without _DIR long records go out as numbered parts.
_PART_TAG: str = "[preproc-fp8-snr-part] "
_PART_LEN: int = 1800


def _parse(spec: str) -> Tuple[int, int]:
    if not spec:
        return 0, 1
    try:
        parts = spec.split(":")
        count = int(parts[0])
        stride = int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        raise ValueError(
            f"DLRM_HSTU_FP8_PREPROC_SNR={spec!r}: expected '<count>[:<stride>]'"
        ) from None
    return count, max(stride, 1)


_COUNT, _STRIDE = _parse(_SPEC)
ENABLED: bool = _COUNT > 0
# the last call index that is sampled; nothing is tracked after it
_LAST: int = (_COUNT - 1) * _STRIDE
# The other knobs are read only when armed, so a stray value cannot break an
# unarmed run.
_RANKS: Tuple[int, ...] = (0,)
_ROWS: int = 32768
_EMB: bool = True
_DIR: str = ""
if ENABLED:
    _RANKS = tuple(
        int(r)
        for r in os.environ.get("DLRM_HSTU_FP8_PREPROC_SNR_RANKS", "0").split(",")
        if r.strip()
    )
    _ROWS = min(
        262144, max(16, int(os.environ.get("DLRM_HSTU_FP8_PREPROC_SNR_ROWS", "32768")))
    )
    _EMB = os.environ.get("DLRM_HSTU_FP8_PREPROC_SNR_EMB", "1") == "1"
    _DIR = os.environ.get("DLRM_HSTU_FP8_PREPROC_SNR_DIR", "")

_E4M3: torch.dtype = torch.float8_e4m3fn
_E5M2: torch.dtype = torch.float8_e5m2
# (max finite, min normal) per format
_LIMITS: Dict[torch.dtype, Tuple[float, float]] = {
    _E4M3: (448.0, 2.0**-6),
    _E5M2: (57344.0, 2.0**-14),
}
_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-7", 2, 7),
    ("8-63", 8, 63),
    ("64+", 64, 1 << 62),
)
_HIST_CAP: int = 200000
# a1 controller grid (hypothetical: E-M3 has no a1 controller yet)
_ACT_GRID: Tuple[Tuple[int, float], ...] = ((1, 2.0), (1, 8.0), (200, 2.0), (200, 8.0))

_calls: Dict[str, int] = {}
_hist: Dict[str, List[torch.Tensor]] = {}
_chain: Dict[Tuple[str, int], torch.Tensor] = {}
_rank_cache: Optional[int] = None
_grad_mode: bool = True
_ctl_cache: Optional[Dict[str, Any]] = None


def note_grad_mode(enabled: bool) -> None:
    """Called by ContextualPreprocessor.forward (when ENABLED) with the caller's
    torch.is_grad_enabled(): inside autograd.Function.forward grad mode always
    reads False, so eval forwards can only be told apart from here."""
    global _grad_mode
    _grad_mode = enabled


def _ctl() -> Dict[str, Any]:
    """The shipped FP8 backward controller's constants (bwd_fp8 env knobs)."""
    global _ctl_cache
    if _ctl_cache is None:
        from generative_recommenders.ops.triton import bwd_fp8

        _ctl_cache = {
            "margin": float(bwd_fp8._BWD_MARGIN),
            "grad_max": float(bwd_fp8._GRAD_MAX),
            "grad_format": str(bwd_fp8._GRAD_FORMAT),
            "warmup": int(bwd_fp8._BWD_WARMUP_CALLS),
            "refresh": int(bwd_fp8._BWD_REFRESH_CALLS),
            "shift_floor": int(bwd_fp8._OUT_WGRAD_DFMT_SHIFT_FLOOR),
            "shift_relax": int(bwd_fp8._OUT_WGRAD_DFMT_SHIFT_RELAX),
        }
    return _ctl_cache


def _grad_dtype() -> torch.dtype:
    return _E4M3 if _ctl()["grad_format"] == "e4m3" else _E5M2


def _rank() -> int:
    global _rank_cache
    if _rank_cache is not None:
        return _rank_cache
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _rank_cache = torch.distributed.get_rank()
        return _rank_cache
    return 0


def _armed() -> bool:
    return ENABLED and _grad_mode and _rank() in _RANKS


def pair_site(is_swish: bool, in_features: int, out_features: int) -> str:
    """Site name of a _LinearNormFunction call (first or, if E-M1 declined,
    last pair of a preprocessor MLP)."""
    if is_swish:
        return {512: "L1c", 1024: "L1a", 24: "L1x"}.get(
            in_features, f"L1_K{in_features}"
        )
    return f"L2_K{in_features}_N{out_features}"


def tail_site(k: int) -> str:
    """Site name of the E-M1 grouped tail's k-th Linear (content, additional,
    action; with no additional MLP the second one is labelled L2a)."""
    return ("L2c", "L2a", "L2x")[k] if k < 3 else f"L2_{k}"


# --------------------------------------------------------------------------
# helpers


def _emit(rec: Dict[str, Any]) -> None:
    """Never raises: a diagnostic must not break training."""
    try:
        line = json.dumps(rec, sort_keys=True, default=str)
    except Exception as exc:  # pragma: no cover
        line = json.dumps({"site": rec.get("site"), "emit_error": repr(exc)})
    if _DIR:
        try:
            os.makedirs(_DIR, exist_ok=True)
            path = os.path.join(_DIR, f"preproc_fp8_snr_rank{_rank()}.jsonl")
            with open(path, "a") as f:
                f.write(line + "\n")
            return
        except OSError:
            pass  # fall back to stdout
    try:
        if len(line) <= _PART_LEN:
            print(_TAG + line, flush=True)
            return
        rid = "{}-{}-{}-{}-{}".format(
            rec.get("rank"), rec.get("site"), rec.get("phase"), rec.get("call"),
            rec.get("block", "")
        )
        parts = [line[i : i + _PART_LEN] for i in range(0, len(line), _PART_LEN)]
        for i, part in enumerate(parts):
            print(
                _PART_TAG + json.dumps({"id": rid, "i": i, "n": len(parts), "s": part}),
                flush=True,
            )
    except Exception:
        pass


def _amax_t(t: torch.Tensor) -> torch.Tensor:
    # aminmax: no [M, D] |t| temporary (X_a alone would need 4.8 GB of it)
    mn, mx = torch.aminmax(t.detach())
    return torch.maximum(mx.float(), -mn.float())


def _amax(t: torch.Tensor) -> float:
    return float(_amax_t(t))


def _track(key: str, t: torch.Tensor) -> None:
    h = _hist.setdefault(key, [])
    if len(h) < _HIST_CAP:
        h.append(_amax_t(t))


def _history(key: str) -> List[float]:
    h = _hist.get(key, [])
    return torch.stack(h).tolist() if h else []


def _replay(
    vals: List[float],
    margin: float,
    warmup: int,
    refresh: int,
    shift_floor: Optional[int] = None,
    shift_relax: int = 4,
) -> Dict[str, Any]:
    """Replay a _SiteScale-style delayed scale over one site's amax history
    (one value per training call; calls numbered from 1, the last is the
    sampled call). Mirrors bwd_fp8.quantize_gradient: BF16 while warming up,
    first publish at call `warmup`, then publish after every refresh-th call;
    each publish takes the max since the previous one. A value clips when
    amax > published * margin; the e4m3 token-major copy (shift_floor given:
    DFMT_SHIFT=auto) clips when amax > published * margin * 2^(shift-7)."""
    from generative_recommenders.ops.triton import bwd_fp8

    pending = 0.0
    pub: Optional[float] = None
    shift = shift_floor if shift_floor is not None else 7
    clean = 0
    win_t_clips = 0
    g_max = -math.inf
    clips = 0
    t_clips = 0
    fp8_calls = 0
    pub_used: Optional[float] = None
    shift_used: Optional[int] = None
    nonfinite = sum(1 for a in vals if not math.isfinite(a))
    for c, a in enumerate(vals, 1):
        if not math.isfinite(a):
            continue
        if pub is None:
            pending = max(pending, a)
            if c >= warmup:
                pub, pending = pending, 0.0
            pub_used, shift_used = None, None
            continue
        fp8_calls += 1
        pub_used, shift_used = pub, shift
        if pub > 0 and a > 0:
            g_max = max(g_max, math.log2(a / pub))
        if a > pub * margin:
            clips += 1
        if shift_floor is not None and a > pub * margin * 2.0 ** (shift - 7):
            t_clips += 1
            win_t_clips += 1
        pending = max(pending, a)
        if c % refresh == 0:
            pub, pending = pending, 0.0
            if shift_floor is not None:
                shift, clean = bwd_fp8.next_t_shift(
                    shift, win_t_clips, clean, shift_floor, shift_relax
                )
                win_t_clips = 0
    out: Dict[str, Any] = {
        "calls": len(vals),
        "fp8_calls": fp8_calls,
        "pub_used": pub_used,
        "clipped_calls": clips,
        "max_growth_log2": g_max if g_max > -math.inf else None,
    }
    if nonfinite:
        out["nonfinite"] = nonfinite
    if vals and pub_used and math.isfinite(vals[-1]) and vals[-1] > 0:
        out["now_growth_log2"] = math.log2(vals[-1] / pub_used)
    if shift_floor is not None:
        out["shift_used"] = shift_used
        out["t_clipped_calls"] = t_clips
    return out


def _sample_rows(m: int, device: torch.device) -> torch.Tensor:
    r = min(_ROWS, m)
    r = max(16, (r // 16) * 16) if m >= 16 else m
    return (torch.arange(r, device=device, dtype=torch.int64) * m) // max(r, 1)


def _metrics(ref: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
    x = x.to(torch.float64)
    err = x - ref
    rr = float((ref * ref).sum())
    ee = float((err * err).sum())
    xx = float((x * x).sum())
    rx = float((ref * x).sum())
    return {
        "snr_db": 10.0 * math.log10(rr / ee) if ee > 0 and rr > 0 else float("inf"),
        "gain": rx / rr if rr > 0 else float("nan"),
        "cos": rx / math.sqrt(rr * xx) if rr > 0 and xx > 0 else float("nan"),
        "mean_err_rel": float(err.mean()) / math.sqrt(rr / ref.numel())
        if rr > 0
        else float("nan"),
    }


def _row_stats(ref: torch.Tensor, x: torch.Tensor, dim: int = 1) -> Dict[str, float]:
    """Distribution over rows (dim=1 reduces columns) of the per-row SNR."""
    x = x.to(torch.float64)
    rr = (ref * ref).sum(dim)
    ee = ((x - ref) ** 2).sum(dim)
    sel = rr > 0
    if int(sel.sum()) == 0:
        return {}
    snr = (10.0 * torch.log10(rr[sel] / ee[sel].clamp_min(1.0e-300))).float()
    q = torch.quantile(
        snr, torch.tensor([0.01, 0.1, 0.5], device=snr.device, dtype=torch.float32)
    ).tolist()
    return {
        "p01": q[0],
        "p10": q[1],
        "p50": q[2],
        "min": float(snr.min()),
        "n": int(sel.sum()),
        "below_10db": int((snr < 10.0).sum()),
    }


def _m(ref: torch.Tensor, x: torch.Tensor, cols: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = _metrics(ref, x)
    out["rows"] = _row_stats(ref, x, 1)
    if cols:
        out["cols"] = _row_stats(ref, x, 0)
    return out


class _Frac:
    """Running flush / subnormal / saturation counts of a cast."""

    def __init__(self) -> None:
        self.nz = 0
        self.flush = 0
        self.sub = 0
        self.sat = 0

    def as_dict(self) -> Dict[str, float]:
        n = max(self.nz, 1)
        return {
            "flush": self.flush / n,
            "subnormal": self.sub / n,
            "sat": self.sat / n,
        }


def _quant(
    x: torch.Tensor, scale: float, dtype: torch.dtype, frac: Optional[_Frac] = None
) -> torch.Tensor:
    """fp8(x / scale) with saturation (torch's cast alone would give NaN past
    the e4m3fn max). Round to nearest even, as the cvt kernels do."""
    limit, min_normal = _LIMITS[dtype]
    q32 = x.to(torch.float32) / scale
    q = q32.clamp(-limit, limit).to(dtype)
    if frac is not None:
        nz = q32 != 0
        qa = q.to(torch.float32).abs()
        frac.nz += int(nz.sum())
        frac.flush += int(((qa == 0) & nz).sum())
        frac.sub += int(((qa > 0) & (qa < min_normal)).sum())
        frac.sat += int(((q32.abs() > limit) & nz).sum())
    return q


def _deq(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float64) * scale


def _act_scale(amax: float, mult: float = 1.0) -> float:
    return max(amax, 1.0e-30) * mult / 448.0


def _grad_scale(amax: float) -> float:
    c = _ctl()
    return max(amax, 1.0e-30) * c["margin"] / c["grad_max"]


def _fp8_ok(k: int, n: int) -> bool:
    return k % 16 == 0 and n % 16 == 0


def _hw_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: float,
    sb: float,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """The real hipBLASLt FP8 GEMM, through the tree's own _scaled_mm wrapper."""
    from generative_recommenders.ops.triton import triton_fp8

    dev = a.device
    return triton_fp8._scaled_mm(
        a,
        b,
        torch.tensor([sa], dtype=torch.float32, device=dev),
        torch.tensor([sb], dtype=torch.float32, device=dev),
        out_dtype=torch.bfloat16,
        bias=bias,
    )


def _base(site: str, phase: str, idx: int, m: int, k: int, n: int) -> Dict[str, Any]:
    return {
        "site": site,
        "phase": phase,
        "call": idx,
        "rank": _rank(),
        "M": m,
        "K": k,
        "N": n,
        "train": True,
        "ctl": _ctl(),
    }


def _grad_ctl(site: str) -> Dict[str, Any]:
    c = _ctl()
    return _replay(
        _history(site + ":grad"),
        c["margin"],
        c["warmup"],
        c["refresh"],
        c["shift_floor"],
        c["shift_relax"],
    )


def _act_ctl(site: str) -> Dict[str, Any]:
    c = _ctl()
    vals = _history(site + ":act")
    out: Dict[str, Any] = {"amax": vals[-1] if vals else None}
    for refresh, margin in _ACT_GRID:
        out[f"r{refresh}_m{int(margin)}"] = _replay(vals, margin, c["warmup"], refresh)
    return out


def _act_dly_scale(act: Dict[str, Any]) -> Optional[float]:
    """a1's delayed e4m3 scale: refresh 200, margin 8 (the gradient defaults)."""
    pub = act.get("r200_m8", {}).get("pub_used")
    return _act_scale(pub, 8.0) if pub else None


# --------------------------------------------------------------------------
# records


def _fwd_record(
    site: str,
    idx: int,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
) -> Dict[str, Any]:
    m, k = x.shape
    n = w.shape[0]
    rec = _base(site, "fwd", idx, m, k, n)
    act = _act_ctl(site)
    rec["act"] = act
    rows = _sample_rows(m, x.device)
    xs = x.index_select(0, rows)
    wd = w.to(torch.float64)
    bd = b.to(torch.float64)
    ref = xs.to(torch.float64) @ wd.t() + bd
    arms: Dict[str, torch.Tensor] = {"shipped": h.index_select(0, rows)}
    if _fp8_ok(k, n):
        amax_x = _amax(x)
        sw = _act_scale(_amax(w))
        wq = _quant(w, sw, _E4M3)
        wdq = _deq(wq, sw)
        rec["w_scale"] = sw
        scales: List[Tuple[str, float]] = [("e4m3", _act_scale(amax_x))]
        if site.startswith("L2"):
            scales.append(("e4m3_m8", _act_scale(amax_x, 8.0)))
            sd = _act_dly_scale(act)
            if sd:
                scales.append(("e4m3_dly", sd))
        for tag, sx in scales:
            fr = _Frac()
            xq = _quant(xs, sx, _E4M3, fr)
            arms[tag] = (_deq(xq, sx) @ wdq.t() + bd).float().to(torch.bfloat16)
            rec["frac_" + tag] = fr.as_dict()
            rec["scale_" + tag] = sx
            if tag == "e4m3":
                try:
                    arms["e4m3_hw"] = _hw_mm(xq, wq.t(), sx, sw, b.to(torch.bfloat16))
                except Exception as exc:  # report, never break training
                    rec["hw_error"] = repr(exc)
    else:
        rec["fp8"] = "ineligible (K or N not a multiple of 16): stays BF16"
    rec["metrics"] = {a + "|exact": _m(ref, v) for a, v in arms.items()}
    if "e4m3_hw" in arms:
        rec["metrics"]["e4m3_hw|e4m3"] = _metrics(
            arms["e4m3"].to(torch.float64), arms["e4m3_hw"]
        )
    return rec


def _chain_da1(dh: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The da1 an FP8 L2 dgrad would deliver, for all rows: bf16(deq(e5m2 dh)
    @ deq(e4m3 W2^T)^T), current-amax scales."""
    gd = _grad_dtype()
    s5 = _grad_scale(_amax(dh))
    wt = w.t().contiguous()
    sw = _act_scale(_amax(w))
    wtd = _deq(_quant(wt, sw, _E4M3), sw)
    out = torch.empty((dh.shape[0], w.shape[1]), dtype=torch.bfloat16, device=dh.device)
    for s in range(0, dh.shape[0], _CHUNK):
        e = min(s + _CHUNK, dh.shape[0])
        out[s:e] = (_deq(_quant(dh[s:e], s5, gd), s5) @ wtd.t()).float().to(
            torch.bfloat16
        )
    return out


def _dgrad_record(
    site: str,
    idx: int,
    dh: torch.Tensor,
    w: torch.Tensor,
    dx: torch.Tensor,
    gctl: Dict[str, Any],
    dh_chain: Optional[torch.Tensor],
) -> Dict[str, Any]:
    m, n = dh.shape
    k = w.shape[1]
    rec = _base(site, "dgrad", idx, m, k, n)
    rec["grad"] = gctl
    rows = _sample_rows(m, dh.device)
    dhs = dh.index_select(0, rows)
    wd = w.to(torch.float64)
    ref = dhs.to(torch.float64) @ wd
    arms: Dict[str, torch.Tensor] = {"shipped": dx.index_select(0, rows)}
    if _fp8_ok(k, n):
        gd = _grad_dtype()
        amax_d = _amax(dh)
        wt = w.t().contiguous()
        sw = _act_scale(_amax(w))
        wtq = _quant(wt, sw, _E4M3)
        wtd = _deq(wtq, sw)
        s5 = _grad_scale(amax_d)
        s4 = _act_scale(amax_d)
        f5, f4 = _Frac(), _Frac()
        d5 = _quant(dhs, s5, gd, f5)
        d4 = _quant(dhs, s4, _E4M3, f4)
        arms["e5m2"] = (_deq(d5, s5) @ wtd.t()).float().to(torch.bfloat16)
        arms["e4m3"] = (_deq(d4, s4) @ wtd.t()).float().to(torch.bfloat16)
        rec.update(
            {
                "scale_e5m2": s5,
                "scale_e4m3": s4,
                "frac_e5m2": f5.as_dict(),
                "frac_e4m3": f4.as_dict(),
            }
        )
        pub = gctl.get("pub_used")
        if pub:
            sdl = _grad_scale(pub)
            fd = _Frac()
            dd = _quant(dhs, sdl, gd, fd)
            arms["e5m2_dly"] = (_deq(dd, sdl) @ wtd.t()).float().to(torch.bfloat16)
            rec["scale_e5m2_dly"] = sdl
            rec["frac_e5m2_dly"] = fd.as_dict()
        if dh_chain is not None:
            dcs = dh_chain.index_select(0, rows)
            sc = _grad_scale(_amax(dh_chain))
            fc = _Frac()
            dc = _quant(dcs, sc, gd, fc)
            arms["e5m2_chain"] = (_deq(dc, sc) @ wtd.t()).float().to(torch.bfloat16)
            rec["frac_e5m2_chain"] = fc.as_dict()
        try:
            arms["e5m2_hw"] = _hw_mm(d5, wtq.t(), s5, sw, None)
        except Exception as exc:
            rec["hw_error"] = repr(exc)
        try:
            from generative_recommenders.ops.triton import bwd_fp8

            inv = torch.tensor([1.0 / s5], dtype=torch.float32, device=dh.device)
            qk, _ = bwd_fp8.quantize_row_major(
                dhs.contiguous(), inv, dtype=gd, limit=_LIMITS[gd][0]
            )
            rec["cast_mismatch_e5m2"] = float(
                (qk.view(torch.uint8) != d5.view(torch.uint8)).float().mean()
            )
        except Exception as exc:
            rec["cast_check_error"] = repr(exc)
    else:
        rec["fp8"] = "ineligible (K or N not a multiple of 16): stays BF16"
    rec["metrics"] = {a + "|exact": _m(ref, v) for a, v in arms.items()}
    if "e5m2_hw" in arms:
        rec["metrics"]["e5m2_hw|e5m2"] = _metrics(
            arms["e5m2"].to(torch.float64), arms["e5m2_hw"]
        )
    return rec


def _wgrad_record(
    site: str,
    idx: int,
    dh: torch.Tensor,
    x: torch.Tensor,
    dw: torch.Tensor,
    gctl: Dict[str, Any],
    act: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    m, n = dh.shape
    k = x.shape[1]
    rec = _base(site, "wgrad", idx, m, k, n)
    fp8 = _fp8_ok(k, n)
    dev = dh.device
    names = ["exact"]
    qspec: Dict[str, Tuple[float, torch.dtype]] = {}
    sx = sxd = 0.0
    if fp8:
        gd = _grad_dtype()
        amax_x = _amax(x)
        amax_d = _amax(dh)
        sx = _act_scale(amax_x)
        s5 = _grad_scale(amax_d)
        shift0 = _ctl()["shift_floor"]
        qspec = {
            "nnb_e5m2": (s5, gd),
            "nna_e4m3s5": (s5 * 2.0**shift0, _E4M3),
            "e4m3": (_act_scale(amax_d), _E4M3),
        }
        names += ["fwd"] + list(qspec)
        pub = gctl.get("pub_used")
        # L1's X is quantized at its exact current amax (EC table); a1 has a
        # delayed scale.
        sxd = sx
        if site.startswith("L2") and act is not None:
            sxd = _act_dly_scale(act) or 0.0
        if pub and sxd:
            sdl = _grad_scale(pub)
            shift = gctl.get("shift_used") or shift0
            qspec["nnb_e5m2_dly"] = (sdl, gd)
            qspec["nna_e4m3s5_dly"] = (sdl * 2.0**shift, _E4M3)
            names += ["fwd_dly", "nnb_e5m2_dly", "nna_e4m3s5_dly"]
        rec.update({"scale_x": sx, "scale_x_dly": sxd})
        rec.update({"scale_" + a: s for a, (s, _) in qspec.items()})
    acc = {a: torch.zeros((n, k), dtype=torch.float64, device=dev) for a in names}
    fracs = {a: _Frac() for a in qspec}
    fx, fxd = _Frac(), _Frac()
    for s in range(0, m, _CHUNK):
        e = min(s + _CHUNK, m)
        dc = dh[s:e]
        xc = x[s:e]
        dcd = dc.to(torch.float64)
        acc["exact"] += dcd.t() @ xc.to(torch.float64)
        if fp8:
            xq = _deq(_quant(xc, sx, _E4M3, fx), sx)
            acc["fwd"] += dcd.t() @ xq
            xqd = xq
            if "fwd_dly" in acc:
                xqd = xq if sxd == sx else _deq(_quant(xc, sxd, _E4M3, fxd), sxd)
                acc["fwd_dly"] += dcd.t() @ xqd
            del dcd
            for a, (sc, dt) in qspec.items():
                xa = xqd if a.endswith("_dly") else xq
                acc[a] += _deq(_quant(dc, sc, dt, fracs[a]), sc).t() @ xa
            del xq, xqd
    if fp8:
        rec["frac_x"] = fx.as_dict()
        if "fwd_dly" in acc and sxd != sx:
            rec["frac_x_dly"] = fxd.as_dict()
        rec.update({"frac_" + a: f.as_dict() for a, f in fracs.items()})
    out: Dict[str, Dict[str, Any]] = {}
    arms: Dict[str, torch.Tensor] = {"shipped": dw}
    arms.update({a: acc[a] for a in qspec})
    for a, v in arms.items():
        refs = ["exact"]
        if fp8:
            refs.append("fwd_dly" if a.endswith("_dly") else "fwd")
        if a == "shipped" and "fwd_dly" in acc:
            refs.append("fwd_dly")
        for r in refs:
            out[f"{a}|{r}"] = _m(acc[r], v, cols=True)
    if fp8:
        out["x8_dbf|exact"] = _metrics(acc["exact"], acc["fwd"])
    rec["metrics"] = out
    return rec


def _row_keys(xb: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """[m, 2] int64 randomized hash of each bf16 row (512 x 16-bit values, so
    the sums stay below 2^56: no overflow)."""
    xi = xb.contiguous().view(torch.int16).to(torch.int64) + 32768
    return torch.stack(((xi * c1).sum(1), (xi * c2).sum(1)), dim=1)


def _emb_record(
    site: str,
    idx: int,
    dh: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    dx: torch.Tensor,
    dh_chain: Optional[torch.Tensor],
) -> List[Dict[str, Any]]:
    """dX summed per unique embedding row, per 512-wide table block."""
    m, n = dh.shape
    k = w.shape[1]
    dev = dh.device
    recs: List[Dict[str, Any]] = []
    if k % 512 != 0 or x.dtype != torch.bfloat16:
        return recs
    gd = _grad_dtype()
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261008)
    c1 = torch.randint(1, 2**31 - 1, (512,), generator=gen, device=dev, dtype=torch.int64)
    c2 = torch.randint(1, 2**31 - 1, (512,), generator=gen, device=dev, dtype=torch.int64)
    amax_d = _amax(dh)
    s5 = _grad_scale(amax_d)
    s4 = _act_scale(amax_d)
    sc = _grad_scale(_amax(dh_chain)) if dh_chain is not None else 0.0
    wt = w.t().contiguous()
    sw = _act_scale(_amax(w))
    wtd = _deq(_quant(wt, sw, _E4M3), sw)  # [k, n]
    arm_names = ["ref", "shipped", "e5m2", "e4m3"] + (
        ["e5m2_chain"] if dh_chain is not None else []
    )
    for blk in range(k // 512):
        lo, hi = blk * 512, (blk + 1) * 512
        keys = torch.cat(
            [
                _row_keys(x[s : min(s + _CHUNK, m), lo:hi], c1, c2)
                for s in range(0, m, _CHUNK)
            ]
        )
        _, inv, cnt = torch.unique(
            keys, dim=0, return_inverse=True, return_counts=True
        )
        del keys
        u = cnt.numel()
        acc = {
            a: torch.zeros((u, 512), dtype=torch.float64, device=dev) for a in arm_names
        }
        wb = w[:, lo:hi].to(torch.float64)  # [n, 512]
        wtb = wtd[lo:hi]  # [512, n]
        for s in range(0, m, _CHUNK):
            e = min(s + _CHUNK, m)
            ii = inv[s:e]
            dc = dh[s:e]
            acc["ref"].index_add_(0, ii, dc.to(torch.float64) @ wb)
            acc["shipped"].index_add_(0, ii, dx[s:e, lo:hi].to(torch.float64))
            for a, (src, sca, dt) in (
                ("e5m2", (dc, s5, gd)),
                ("e4m3", (dc, s4, _E4M3)),
                ("e5m2_chain", (dh_chain[s:e] if dh_chain is not None else None, sc, gd)),
            ):
                if a not in acc or src is None:
                    continue
                acc[a].index_add_(
                    0,
                    ii,
                    (_deq(_quant(src, sca, dt), sca) @ wtb.t())
                    .float()
                    .to(torch.bfloat16)
                    .to(torch.float64),
                )
        ref = acc["ref"]
        rr_row = (ref * ref).sum(1)
        rec = _base(site, "emb", idx, m, k, n)
        rec.update(
            {
                "block": blk,
                "unique_rows": u,
                "count_max": int(cnt.max()),
                "count_mean": float(m / max(u, 1)),
                "scope": "rank-local hashed rows; preprocessor-MLP share; "
                "e5m2/e4m3 = L1 dgrad cast only, e5m2_chain = L2+L1 dgrads FP8",
            }
        )
        metrics: Dict[str, Any] = {}
        for a in arm_names[1:]:
            err = acc[a] - ref
            ee_row = (err * err).sum(1)
            arm: Dict[str, Any] = {"all": _metrics(ref, acc[a])}
            for name, lo_c, hi_c in _BUCKETS:
                sel = (cnt >= lo_c) & (cnt <= hi_c) & (rr_row > 0)
                nsel = int(sel.sum())
                if nsel == 0:
                    continue
                rr_b = float(rr_row[sel].sum())
                ee_b = float(ee_row[sel].sum())
                per_row = 10.0 * torch.log10(
                    rr_row[sel] / ee_row[sel].clamp_min(1.0e-300)
                )
                q = torch.quantile(
                    per_row.float(),
                    torch.tensor([0.1, 0.5], device=dev, dtype=torch.float32),
                ).tolist()
                bm = _metrics(ref[sel], acc[a][sel])
                arm[name] = {
                    "rows": nsel,
                    "pooled_snr_db": 10.0 * math.log10(rr_b / ee_b)
                    if ee_b > 0
                    else float("inf"),
                    "row_snr_p10": q[0],
                    "row_snr_p50": q[1],
                    "gain": bm["gain"],
                    "mean_err_rel": bm["mean_err_rel"],
                }
            metrics[a] = arm
        rec["metrics"] = metrics
        recs.append(rec)
        del acc, ref, inv, cnt
    return recs


# --------------------------------------------------------------------------
# entry points (called only when ENABLED)


@torch.no_grad()
def on_forward(
    ctx: Any,
    slot: int,
    site: str,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
) -> None:
    """After a preprocessor Linear's forward: x @ w^T + b == h (bf16)."""
    if not _armed():
        return
    idx = _calls.get(site, 0)
    _calls[site] = idx + 1
    if idx > _LAST:
        return  # past the last sample: no tracking, no ctx info, no backward work
    sampled = idx % _STRIDE == 0 and idx // _STRIDE < _COUNT
    info = getattr(ctx, "_pfp8_snr", None)
    if info is None:
        info = {}
        ctx._pfp8_snr = info
    info[slot] = (site, idx, sampled)
    with torch.autocast(device_type=x.device.type, enabled=False):
        try:
            _track(site + ":act", x)
            if sampled:
                _emit(_fwd_record(site, idx, x, w, b, h))
        except Exception as exc:  # diagnostic must never break training
            try:
                rec = _base(site, "fwd", idx, x.shape[0], x.shape[1], w.shape[0])
                rec["error"] = repr(exc)
                _emit(rec)
            except Exception:
                pass


@torch.no_grad()
def on_backward(
    ctx: Any,
    slot: int,
    dh: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    dx: Optional[torch.Tensor],
    dw: Optional[torch.Tensor],
    ln: Optional[Dict[str, Any]] = None,
) -> None:
    """After a preprocessor Linear's dgrad (dx = dh @ w) and wgrad
    (dw = dh^T @ x). `ln` (first pairs only): the SwishLN backward's saved
    inputs, for the chained arm."""
    info = getattr(ctx, "_pfp8_snr", None)
    if not info or slot not in info:
        return
    site, idx, sampled = info[slot]
    with torch.autocast(device_type=dh.device.type, enabled=False):
        try:
            _track(site + ":grad", dh)
            if not sampled:
                return
            gctl = _grad_ctl(site)
            act = _act_ctl(site) if site.startswith("L2") else None
            dh_chain: Optional[torch.Tensor] = None
            da1 = _chain.pop((site[-1], idx), None) if site in ("L1c", "L1a") else None
            if da1 is not None and ln is not None:
                from generative_recommenders.ops.triton.triton_layer_norm import (
                    triton_weighted_layer_norm_bwd_dxsum,
                )

                dh_chain = triton_weighted_layer_norm_bwd_dxsum(
                    dy=da1,
                    x=ln["h"],
                    weight=ln["norm_w"],
                    bias=ln["norm_b"],
                    mean=ln["mean"],
                    rstd=ln["rstd"],
                    eps=ln["eps"],
                    BLOCK_D=ln["block_d"],
                    is_swish=True,
                )[0]
            del da1
            if dx is not None:
                _emit(_dgrad_record(site, idx, dh, w, dx, gctl, dh_chain))
            if dw is not None:
                _emit(_wgrad_record(site, idx, dh, x, dw, gctl, act))
            if _EMB and dx is not None and site in ("L1c", "L1a"):
                for rec in _emb_record(site, idx, dh, w, x, dx, dh_chain):
                    _emit(rec)
            if _EMB and dx is not None and site in ("L2c", "L2a"):
                # autograd runs the tail before the first pairs; hand the FP8
                # da1 to the same MLP's L1 hook of this step
                for key in [key for key in _chain if key[1] < idx]:
                    del _chain[key]  # a stash whose L1 hook never came
                _chain[(site[-1], idx)] = _chain_da1(dh, w)
        except Exception as exc:
            try:
                rec = _base(site, "bwd", idx, dh.shape[0], w.shape[1], dh.shape[1])
                rec["error"] = repr(exc)
                _emit(rec)
            except Exception:
                pass
