#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP8 pre-flight check for MI455X (gfx1250) before a Primus/Megatron MXFP8 run.

Runs on one GPU in well under 2 minutes and answers one question: does the
Primus-Turbo this Python imports compute MXFP8 GEMMs correctly on this GPU?

It runs the same Turbo ops the MI455X MXFP8 yamls select (``turbo_precision: mxfp8``,
``use_turbo_gemm`` / ``use_turbo_grouped_gemm``):

  * dense:   ``primus_turbo.pytorch.ops.gemm_fp8(x, w, trans_b=True, config=MX)``
             (PrimusTurboLinear / ColumnParallel / RowParallel forward), fwd + bwd;
  * grouped: ``primus_turbo.pytorch.ops.grouped_gemm_fp8(x, w, group_lens, trans_b=True,
             config=MX)`` (PrimusTurboGroupedLinear forward), fwd + bwd.

The quant config is the one the Primus-Turbo linears use under ``turbo_precision: mxfp8``
(``primus_turbo.py`` ``_override_quant_config``; the same as ``fp8_utils.get_fp8_quant_config`` for
``fp8_recipe: mxfp8``): E4M3, MX_BLOCKWISE, block 32, E8M0 scales, dynamic scaling.
Every output (fwd, dgrad, wgrad) is compared against BF16 on identical inputs.

Thresholds (from the gfx1250 MX study): a correct MXFP8 GEMM on Gaussian inputs gives
SNR ~26-29 dB against BF16; the silent hipBLASLt MXFP8 fallback of Primus-Turbo
0.6.1.dev31 gives ~5 dB. PASS needs SNR >= 20 dB and finite values for every output.

Usage (inside the training container, with the same PYTHONPATH the training uses):
    python examples/megatron/configs/MI455X/mxfp8_preflight.py           # dense + grouped
    python examples/megatron/configs/MI455X/mxfp8_preflight.py --gpt-oss # + gpt-oss shapes
Exit code 0 = PASS, 1 = FAIL, 2 = could not run.
"""

import argparse
import importlib
import inspect
import math
import os
import statistics
import sys
import time

SNR_PASS_DB = 20.0  # correct MXFP8: ~26-29 dB on these inputs
SNR_FALLBACK_DB = 10.0  # hipBLASLt MXFP8 fallback on gfx1250: ~5 dB


def _p(msg=""):
    print(msg, flush=True)


def snr_db(out, ref):
    out = out.float()
    ref = ref.float()
    err = (out - ref).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    if not math.isfinite(err) or err == 0.0:
        return float("inf") if err == 0.0 else float("nan")
    return 10.0 * math.log10(sig / err)


def rel_l2(out, ref):
    out = out.float()
    ref = ref.float()
    return ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def time_ms(fn, warmup, reps):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


class BackendTrace:
    """Records which Turbo kernel backend each dispatcher actually executes."""

    def __init__(self):
        self.calls = []
        self.phase = None
        self._installed = False

    def install(self):
        try:
            from primus_turbo.pytorch.core.backend import AutoKernelDispatcher
        except Exception as exc:  # older / newer Turbo layout
            _p(f"  (backend trace unavailable: {exc})")
            return
        # make sure the FP8 dispatchers are imported (they register as subclasses)
        for mod in (
            "primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl",
            "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl",
        ):
            try:
                importlib.import_module(mod)
            except Exception:
                pass
        for disp in list(AutoKernelDispatcher._subclasses):
            for btype, entry in getattr(disp, "_backends", {}).items():
                impl = getattr(entry, "impl", entry)
                if not hasattr(impl, "execute") or getattr(impl, "_mxpf_wrapped", False):
                    continue
                orig = impl.execute

                def wrapped(*a, _orig=orig, _d=disp.__name__, _b=impl.__name__, **k):
                    gran = k.get("granularity")
                    self.calls.append((self.phase, _d, _b, getattr(gran, "name", str(gran))))
                    return _orig(*a, **k)

                impl.execute = staticmethod(wrapped)
                impl._mxpf_wrapped = True
        self._installed = True

    def summary(self, phase):
        seen = []
        for ph, d, b, g in self.calls:
            if ph == phase and (d, b, g) not in seen:
                seen.append((d, b, g))
        return seen


def env_report():
    import torch

    _p("== environment")
    if not torch.cuda.is_available():
        _p("  no GPU visible")
        sys.exit(2)
    prop = torch.cuda.get_device_properties(0)
    arch = getattr(prop, "gcnArchName", "?")
    _p(f"  GPU            : {prop.name}  arch={arch}  CUs={prop.multi_processor_count}")
    _p(f"  torch          : {torch.__version__}  hip={getattr(torch.version, 'hip', None)}")
    try:
        import triton

        _p(f"  triton         : {triton.__version__}")
    except Exception:
        pass
    import primus_turbo

    ver = getattr(primus_turbo, "__version__", "?")
    commit = "?"
    try:
        from primus_turbo import _version as _v

        commit = getattr(_v, "commit_id", None) or getattr(_v, "__commit_id__", None) or "?"
    except Exception:
        pass
    _p(f"  primus_turbo   : {primus_turbo.__file__}")
    _p(f"  turbo version  : {ver}  commit={commit}")
    if (
        "site-packages" in primus_turbo.__file__
        and os.environ.get("PYTHONPATH", "").find("Primus-Turbo") >= 0
    ):
        _p("  WARNING        : PYTHONPATH names a Primus-Turbo checkout but the image copy was imported")

    # gfx1250 MX backend detection
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_impl as gimpl

    has_dense_kernel = hasattr(gimpl, "_gemm_mxfp8_gfx1250")
    try:
        importlib.import_module("primus_turbo.flydsl.gemm.gemm_gfx1250_kernel")
        has_kernel_mod = True
    except Exception:
        has_kernel_mod = False
    hip_gated = False
    try:
        src = inspect.getsource(gimpl.GEMMFP8HipBLASLtBackend.can_handle)
        hip_gated = "is_gfx1250" in src
    except Exception:
        pass
    is_1250 = "gfx1250" in arch
    _p(
        f"  gfx1250 dense MX backend (FlyDSL WMMA-scale) : {'present' if has_dense_kernel and has_kernel_mod else 'ABSENT'}"
    )
    _p(
        f"  hipBLASLt MXFP8 gated off on gfx1250         : {'yes' if hip_gated else 'NO (bad ~5 dB fallback reachable)'}"
    )
    if not is_1250:
        _p("  NOTE: not a gfx1250 GPU; thresholds still apply but the gfx1250 checks are informational")
    return arch, has_dense_kernel and has_kernel_mod, hip_gated


def mx_config():
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
        ScalingStrategy,
    )

    # == primus_turbo.py _override_quant_config("mxfp8") (turbo_precision: mxfp8)
    return Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        strategy=ScalingStrategy.DYNAMIC,
        scale_dtype=ScaleDtype.E8M0,
        block_size=32,
    )


def check(name, pairs, results):
    ok_all = True
    for what, out, ref in pairs:
        s = snr_db(out, ref)
        r = rel_l2(out, ref)
        finite = bool(out.float().isfinite().all().item())
        ok = finite and (s >= SNR_PASS_DB)
        tag = "ok" if ok else "FAIL"
        if finite and s < SNR_FALLBACK_DB:
            tag += " (bad-fallback signature)"
        _p(f"    {what:6s} SNR {s:6.1f} dB  relL2 {r:.3e}  finite={finite}  {tag}")
        results.append((f"{name}/{what}", ok, s))
        ok_all &= ok
    return ok_all


def run_dense(name, m, n, k, cfg, trace, reps, results):
    import primus_turbo.pytorch as pt
    import torch

    _p(f"== dense MXFP8 linear [{name}]  x[{m},{k}] @ w[{n},{k}]^T  (gemm_fp8, trans_b=True)")
    g = torch.Generator(device="cuda").manual_seed(1234)
    x0 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w0 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    go = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    trace.phase = name
    try:
        out = pt.ops.gemm_fp8(x, w, trans_a=False, trans_b=True, out_dtype=None, config=cfg)
        out.backward(go)
        torch.cuda.synchronize()
    except Exception as exc:
        _p(f"    ERROR: {type(exc).__name__}: {str(exc)[:300]}")
        results.append((f"{name}/run", False, float("nan")))
        return
    trace.phase = None
    for d, b, gname in trace.summary(name):
        _p(f"    backend: {d} -> {b} ({gname})")

    xr = x0.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    ref = torch.nn.functional.linear(xr, wr)
    ref.backward(go)
    check(name, [("fwd", out, ref), ("dgrad", x.grad, xr.grad), ("wgrad", w.grad, wr.grad)], results)

    # timing (quantization included, as in training)
    xt = x0.clone().requires_grad_(True)
    wt = w0.clone().requires_grad_(True)

    def mx_fwd():
        with torch.no_grad():
            pt.ops.gemm_fp8(x0, w0, trans_a=False, trans_b=True, out_dtype=None, config=cfg)

    def mx_fb():
        o = pt.ops.gemm_fp8(xt, wt, trans_a=False, trans_b=True, out_dtype=None, config=cfg)
        o.backward(go)
        xt.grad = None
        wt.grad = None

    def bf_fwd():
        with torch.no_grad():
            torch.nn.functional.linear(x0, w0)

    def bf_fb():
        o = torch.nn.functional.linear(xt, wt)
        o.backward(go)
        xt.grad = None
        wt.grad = None

    fl = 2.0 * m * n * k
    t = {f.__name__: time_ms(f, 2, reps) for f in (mx_fwd, mx_fb, bf_fwd, bf_fb)}
    _p(
        f"    perf   MXFP8 fwd {t['mx_fwd']:.3f} ms ({fl / t['mx_fwd'] / 1e9:.0f} TFLOP/s)  "
        f"fwd+bwd {t['mx_fb']:.3f} ms ({3 * fl / t['mx_fb'] / 1e9:.0f} TFLOP/s)"
    )
    _p(
        f"    perf   BF16  fwd {t['bf_fwd']:.3f} ms ({fl / t['bf_fwd'] / 1e9:.0f} TFLOP/s)  "
        f"fwd+bwd {t['bf_fb']:.3f} ms ({3 * fl / t['bf_fb'] / 1e9:.0f} TFLOP/s)   "
        f"MX/BF16 fwd+bwd speedup {t['bf_fb'] / t['mx_fb']:.2f}x"
    )


def run_grouped(name, lens, n, k, cfg, trace, reps, results, timed=True):
    import primus_turbo.pytorch as pt
    import torch

    G = len(lens)
    m = int(sum(lens))
    _p(
        f"== grouped MXFP8 GEMM [{name}]  G={G} rows={m} (per group {min(lens)}..{max(lens)})  "
        f"x[{m},{k}] @ w[{G},{n},{k}]^T  (grouped_gemm_fp8, trans_b=True)"
    )
    g = torch.Generator(device="cuda").manual_seed(4321)
    x0 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w0 = torch.randn(G, n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    go = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)
    group_lens = torch.tensor(lens, device="cuda", dtype=torch.int64)

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    trace.phase = name
    try:
        out = pt.ops.grouped_gemm_fp8(x, w, group_lens, trans_b=True, config=cfg)
        out.backward(go)
        torch.cuda.synchronize()
    except Exception as exc:
        _p(f"    ERROR: {type(exc).__name__}: {str(exc)[:300]}")
        results.append((f"{name}/run", False, float("nan")))
        return
    trace.phase = None
    for d, b, gname in trace.summary(name):
        _p(f"    backend: {d} -> {b} ({gname})")

    # BF16 reference: per-expert torch matmuls on the same inputs
    xr = x0.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    outs, s = [], 0
    for i, L in enumerate(lens):
        outs.append(torch.nn.functional.linear(xr[s : s + L], wr[i]))
        s += L
    ref = torch.cat(outs, 0)
    ref.backward(go)
    check(name, [("fwd", out, ref), ("dgrad", x.grad, xr.grad), ("wgrad", w.grad, wr.grad)], results)
    if not timed:
        return

    xt = x0.clone().requires_grad_(True)
    wt = w0.clone().requires_grad_(True)

    def mx_fwd():
        with torch.no_grad():
            pt.ops.grouped_gemm_fp8(x0, w0, group_lens, trans_b=True, config=cfg)

    def mx_fb():
        o = pt.ops.grouped_gemm_fp8(xt, wt, group_lens, trans_b=True, config=cfg)
        o.backward(go)
        xt.grad = None
        wt.grad = None

    def bf_fwd():
        with torch.no_grad():
            pt.ops.grouped_gemm(x0, w0, group_lens, trans_b=True)

    def bf_fb():
        o = pt.ops.grouped_gemm(xt, wt, group_lens, trans_b=True)
        o.backward(go)
        xt.grad = None
        wt.grad = None

    fl = 2.0 * m * n * k
    t = {f.__name__: time_ms(f, 2, reps) for f in (mx_fwd, mx_fb, bf_fwd, bf_fb)}
    _p(
        f"    perf   MXFP8 fwd {t['mx_fwd']:.3f} ms ({fl / t['mx_fwd'] / 1e9:.0f} TFLOP/s)  "
        f"fwd+bwd {t['mx_fb']:.3f} ms ({3 * fl / t['mx_fb'] / 1e9:.0f} TFLOP/s)"
    )
    _p(
        f"    perf   BF16  fwd {t['bf_fwd']:.3f} ms ({fl / t['bf_fwd'] / 1e9:.0f} TFLOP/s)  "
        f"fwd+bwd {t['bf_fb']:.3f} ms ({3 * fl / t['bf_fb'] / 1e9:.0f} TFLOP/s)   "
        f"(BF16 = Turbo grouped_gemm)  MX/BF16 fwd+bwd speedup {t['bf_fb'] / t['mx_fb']:.2f}x"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpt-oss", action="store_true", help="also check the gpt-oss-120B shapes (K=2880)")
    ap.add_argument("--reps", type=int, default=10, help="timed repetitions per case (median)")
    args = ap.parse_args()

    t0 = time.time()
    import torch

    arch, has_mx, hip_gated = env_report()
    trace = BackendTrace()
    trace.install()
    cfg = mx_config()
    _p(f"  quant config   : {cfg}")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    results = []
    # Llama-3.1-8B MLP fc2-like: tokens 8192, hidden 4096, ffn 14336
    run_dense("llama3_8b", m=8192, n=14336, k=4096, cfg=cfg, trace=trace, reps=args.reps, results=results)
    # Qwen3-30B-A3B expert fc1 on one EP8 rank: 16 local experts, 2048 tokens each (balanced),
    # hidden 2048 -> 2 x moe_ffn 768
    run_grouped(
        "qwen3_30b_a3b", [2048] * 16, n=1536, k=2048, cfg=cfg, trace=trace, reps=args.reps, results=results
    )
    # same expert shape with unaligned per-expert token counts (no Primus padding), correctness only
    ragged = [2048 + d for d in (-517, 311, -64, 95, 1, -333, 200, 17, -1, 129, -250, 63, 480, -96, 5, -40)]
    run_grouped(
        "qwen3_ragged", ragged, n=1536, k=2048, cfg=cfg, trace=trace, reps=0, results=results, timed=False
    )
    if args.gpt_oss:
        # gpt-oss-120B: hidden 2880 (K not a multiple of 128); QKV 2880 -> 5120
        run_dense(
            "gpt_oss_qkv", m=4096, n=5120, k=2880, cfg=cfg, trace=trace, reps=args.reps, results=results
        )
        # expert fc1 on one EP4 rank: 32 local experts x 512 tokens, 2880 -> 2 x 2880
        run_grouped(
            "gpt_oss_moe", [512] * 32, n=5760, k=2880, cfg=cfg, trace=trace, reps=args.reps, results=results
        )

    _p("== verdict")
    _p(
        f"  threshold: SNR >= {SNR_PASS_DB:.0f} dB vs BF16 and finite for fwd/dgrad/wgrad "
        f"(correct MXFP8 ~26-29 dB; hipBLASLt fallback ~5 dB)"
    )
    failed = [r for r in results if not r[1]]
    hip_used = any("HipBLASLt" in b or "Hipblaslt" in b for _, _, b, g in trace.calls if g == "MX_BLOCKWISE")
    if "gfx1250" in arch and hip_used:
        _p("  hipBLASLt executed an MX_BLOCKWISE GEMM on gfx1250: this is the known-bad fallback")
    for nm, ok, s in results:
        if not ok:
            _p(f"  FAIL {nm}: SNR {s:.1f} dB")
    worst = min((s for _, _, s in results if math.isfinite(s)), default=float("nan"))
    if failed:
        _p(
            f"MXFP8 PREFLIGHT: FAIL ({len(failed)}/{len(results)} checks; worst SNR {worst:.1f} dB) "
            f"in {time.time() - t0:.0f} s -- do NOT train MXFP8 with this Primus-Turbo"
        )
        return 1
    _p(f"MXFP8 PREFLIGHT: PASS ({len(results)} checks; worst SNR {worst:.1f} dB) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        import traceback

        traceback.print_exc()
        _p(f"MXFP8 PREFLIGHT: ERROR ({type(exc).__name__}: {exc})")
        sys.exit(2)
