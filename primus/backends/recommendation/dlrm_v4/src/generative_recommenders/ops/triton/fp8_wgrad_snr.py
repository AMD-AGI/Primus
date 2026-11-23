# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""In-model numerics probe for the FP8 output-projection wgrad.

Diagnostic only, default OFF. It runs only when DLRM_HSTU_FP8_BWD_OUT_WGRAD=1
(the FP8 wgrad itself) and DLRM_HSTU_FP8_BWD_OUT_WGRAD_SNR="<count>[:<stride>]"
are both set. It never changes what the step computes: it reads the operands
the armed path already holds, recomputes references on the side, and prints
one `[fp8-out-wgrad-snr]` JSON line per sampled call. It costs seconds per
sampled call (fp64 GEMMs), so never arm it in a timed run.

Two oracles, both fp64 GEMMs over the same tokens:
  ref_exact = y_bf16^T @ dout      y as the bf16 recompute of the shipped path
  ref_fwd   = deq(y_e4m3)^T @ dout y as the e4m3 bytes the forward GEMM consumed

ref_fwd is the exact gradient of the forward that actually ran (out =
deq(y_e4m3) @ W), so a path that differs from it only by the dout cast has
no y error at all. The shipped bf16 path differs from ref_fwd by the
e4m3-vs-bf16 y difference. Both are reported so the user can rule on either.

Arms (all vs both oracles; SNR dB, gain <ref,x>/|ref|^2, cosine):
  shipped     torch.mm(y_bf16^T, dout) in bf16 (the default path)
  fp8         the armed result (e4m3 y x e5m2 dout, delayed per-tensor scale)
  y8_dbf      deq(y_e4m3) x bf16 dout (fp64): y cast only
  ybf_d5      y_bf16 x deq(dout_e5m2 as armed): dout cast only
  d5_now      deq(y_e4m3) x e5m2(dout) with the current-amax pow2 scale
  d4_now      deq(y_e4m3) x e4m3(dout) with the current-amax pow2 scale
Also: dout amax, the armed dout scale, flush/saturation fractions of dout
under e5m2 (armed scale) and e4m3 (current-amax pow2 scale).
"""

import json
import math
import os
from typing import Dict, Optional, Tuple

import torch

_SPEC: str = os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_SNR", "")
_RANKS: Tuple[int, ...] = tuple(
    int(r)
    for r in os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_SNR_RANKS", "0").split(",")
    if r.strip()
)
_CHUNK: int = 262144
_calls: int = 0


def _parse(spec: str) -> Tuple[int, int]:
    if not spec:
        return 0, 1
    parts = spec.split(":")
    count = int(parts[0])
    stride = int(parts[1]) if len(parts) > 1 else 1
    return count, max(stride, 1)


_COUNT, _STRIDE = _parse(_SPEC)
SNR_ENABLED: bool = _COUNT > 0


def _rank(device: torch.device) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return device.index if device.index is not None else 0


def snr_requested(device: torch.device) -> bool:
    """True on the calls this probe samples. Advances the call counter."""
    global _calls
    if _COUNT <= 0 or _rank(device) not in _RANKS:
        return False
    idx = _calls
    _calls += 1
    return idx % _STRIDE == 0 and idx // _STRIDE < _COUNT


def _pow2_scale(amax: float, limit: float) -> float:
    # dequant scale s with q = x / s, s = 2^ceil(log2(amax / limit)) (no saturation)
    if amax <= 0.0:
        return 1.0
    return 2.0 ** math.ceil(math.log2(amax / limit))


def _fp64_wgrad(
    y: torch.Tensor, d: torch.Tensor, y_scale: float = 1.0, d_scale: float = 1.0
) -> torch.Tensor:
    """sum_m y[m]^T d[m] in fp64, chunked over tokens. y [M, C], d [M, N]."""
    m = y.shape[0]
    out = torch.zeros((y.shape[1], d.shape[1]), dtype=torch.float64, device=y.device)
    for s in range(0, m, _CHUNK):
        e = min(s + _CHUNK, m)
        out += y[s:e].to(torch.float64).t() @ d[s:e].to(torch.float64)
    return out * (y_scale * d_scale)


def _metrics(ref: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
    x = x.to(torch.float64)
    err = x - ref
    rr = float((ref * ref).sum())
    ee = float((err * err).sum())
    xx = float((x * x).sum())
    rx = float((ref * x).sum())
    return {
        "snr_db": 10.0 * math.log10(rr / ee) if ee > 0 else float("inf"),
        "gain": rx / rr if rr > 0 else float("nan"),
        "cos": rx / math.sqrt(rr * xx) if rr > 0 and xx > 0 else float("nan"),
        "mean_err_rel": float(err.mean()) / math.sqrt(rr / ref.numel())
        if rr > 0
        else float("nan"),
    }


def _cast(
    d: torch.Tensor, scale: float, dtype: torch.dtype, limit: float
) -> Tuple[torch.Tensor, float, float]:
    """Round d/scale to dtype with saturation; return (fp8, flush_frac, sat_frac).
    Chunked over tokens so the fp32 temporaries stay ~0.5 GB."""
    q = torch.empty(d.shape, dtype=dtype, device=d.device)
    nz_n = flushed_n = sat_n = 0
    for s in range(0, d.shape[0], _CHUNK):
        e = min(s + _CHUNK, d.shape[0])
        q32 = d[s:e].to(torch.float32) / scale
        nz = q32 != 0
        sat_n += int(((q32.abs() > limit) & nz).sum())
        q[s:e] = q32.clamp(-limit, limit).to(dtype)
        flushed_n += int(((q[s:e].to(torch.float32) == 0) & nz).sum())
        nz_n += int(nz.sum())
    n = max(nz_n, 1)
    return q, flushed_n / n, sat_n / n


@torch.no_grad()
def log_out_wgrad_snr(
    y_bf16: Optional[torch.Tensor],
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    dout: torch.Tensor,
    dout_fp8: torch.Tensor,
    dout_scale: torch.Tensor,
    dout_token_major: bool,
    d_output_weight: torch.Tensor,
) -> None:
    """Print one JSON line comparing the armed FP8 wgrad with the oracles."""
    m = dout.shape[0]
    if y_fp8.shape[0] != dout.shape[0]:
        # Producer mode hands over token-major y [C, M_pad]; not supported here.
        print("[fp8-out-wgrad-snr] {\"skipped\": \"token-major y\"}", flush=True)
        return
    rec: Dict[str, object] = {
        "rank": _rank(dout.device),
        "call": _calls - 1,
        "M": m,
        "C": int(y_fp8.shape[1]),
        "N": int(dout.shape[1]),
    }
    try:
        ys = float(y_scale.reshape(-1)[0])
        ds = float(dout_scale.reshape(-1)[0])
        d_rm = dout_fp8[:, :m].t() if dout_token_major else dout_fp8[:m]
        amax = float(dout.abs().amax().float())
        rec.update({"y_scale": ys, "dout_scale": ds, "dout_amax": amax})

        ref_fwd = _fp64_wgrad(y_fp8, dout, y_scale=ys)
        arms: Dict[str, torch.Tensor] = {"fp8": d_output_weight}
        arms["d_armed"] = _fp64_wgrad(y_fp8, d_rm, y_scale=ys, d_scale=ds)
        e5 = torch.float8_e5m2
        e4 = torch.float8_e4m3fn
        s5 = _pow2_scale(amax, 57344.0)
        s4 = _pow2_scale(amax, 448.0)
        q5, fl5, sat5 = _cast(dout, s5, e5, 57344.0)
        q4, fl4, sat4 = _cast(dout, s4, e4, 448.0)
        from generative_recommenders.ops.triton import bwd_fp8

        armed_dtype = dout_fp8.dtype
        armed_limit = 448.0 if armed_dtype == e4 else 57344.0
        _, fla, sata = _cast(dout, ds, armed_dtype, armed_limit)
        rec.update(
            {
                "armed_flush": fla,
                "armed_sat": sata,
                "armed_dtype": str(armed_dtype),
                "e5_now_flush": fl5,
                "e4_now_flush": fl4,
                "e4_now_sat": sat4,
                "e5_now_scale": s5,
                "e4_now_scale": s4,
            }
        )
        arms["d5_now"] = _fp64_wgrad(y_fp8, q5, y_scale=ys, d_scale=s5)
        arms["d4_now"] = _fp64_wgrad(y_fp8, q4, y_scale=ys, d_scale=s4)
        del q5, q4
        out: Dict[str, Dict[str, float]] = {}
        for k, v in arms.items():
            out[k + "|fwd"] = _metrics(ref_fwd, v)
        if y_bf16 is not None:
            ref_exact = _fp64_wgrad(y_bf16, dout)
            shipped = torch.mm(y_bf16.t(), dout)
            out["shipped|exact"] = _metrics(ref_exact, shipped)
            out["shipped|fwd"] = _metrics(ref_fwd, shipped)
            out["y8_dbf|exact"] = _metrics(ref_exact, ref_fwd)
            out["ybf_darmed|exact"] = _metrics(
                ref_exact, _fp64_wgrad(y_bf16, d_rm, d_scale=ds)
            )
            for k, v in arms.items():
                out[k + "|exact"] = _metrics(ref_exact, v)
        rec["metrics"] = out
    except Exception as exc:  # diagnostic must never break training
        rec["error"] = repr(exc)
    print("[fp8-out-wgrad-snr] " + json.dumps(rec, sort_keys=True), flush=True)
