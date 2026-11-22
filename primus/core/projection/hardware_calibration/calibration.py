###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark-mode hardware calibration: effective compute clock + HBM bandwidth.

Runs on each bench GPU before the model is built, while the device is idle
and has memory to spare, and frees everything it allocated before the memory
recorder takes its first snapshot.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

from primus.core.projection.hardware_calibration.compute_clock import (
    DEFAULT_MIN_SETTLE_S,
    back_calculate_compute_clock,
    gemm_dtype_candidates,
    make_backend_predictor,
    make_origami_predictor,
    time_gemm_ms,
)
from primus.core.projection.hardware_calibration.device_info import (
    SETTLE_TIMEOUT_S,
    ClockSampler,
    _peak_hbm_gbps,
    describe_device,
    make_clock_reader,
)
from primus.core.projection.hardware_calibration.gemm_shapes import (
    collect_workload_gemms,
    select_calibration_gemms,
)
from primus.core.projection.hardware_calibration.matrix_core_loop import (
    matrix_loop_unavailable_reason,
    run_matrix_core_loop,
)
from primus.core.projection.hardware_calibration.stream_bandwidth import (
    DEFAULT_ARRAY_MIB,
    analyze_memory_clock,
    derive_model_hbm_bandwidth,
    run_stream,
    stream_workload_derate,
)

SKIP_ENV = "PRIMUS_BENCH_SKIP_HW_CALIB"
PAYLOAD_KEY = "_hardware_calibration"

TAG = "[Primus:HW Calibration]"

# Later GEMMs start with the clock already in its loaded state; the settle
# check still has to pass before they are timed.
_NEXT_GEMM_MIN_SETTLE_S = 0.3


def calibration_skip_reason(args=None) -> Optional[str]:
    """Why calibration should not run here, or *None* if it should."""
    if os.environ.get(SKIP_ENV, "0") == "1":
        return f"{SKIP_ENV}=1"
    if args is not None and getattr(args, "skip_hw_calibration", False):
        return "--skip-hw-calibration"
    if args is not None and getattr(args, "profile_only", False):
        # The bg=1 baseline subprocess; the main bench calibrates.
        return "--profile-only"
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return "no GPU visible"
    return None


def _local_device_index() -> int:
    import torch

    return int(os.getenv("LOCAL_RANK", "0")) % torch.cuda.device_count()


def run_hardware_calibration(
    training_config,
    device_index: Optional[int] = None,
    max_gemms: int = 4,
    stream_array_mib: int = DEFAULT_ARRAY_MIB,
    gemm_backend: Optional[str] = None,
    gpu_arch: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure this rank's GPU; returns a JSON-serialisable payload.

    *gemm_backend* (the projection's ``--gemm-backend``) models the GEMMs the
    clock is solved from; Origami when unset or when that backend cannot model
    the device.  *gpu_arch* (``--gpu-arch``) names the backend's hardware
    profile only on a device Primus does not recognise.

    Frees what it allocated and resets peak-memory stats; after a GPU fault
    the cleanup is best effort and the original error is the one raised.
    """
    import torch

    stream_workload_derate()  # reject a bad PRIMUS_HBM_STREAM_DERATE before measuring
    if device_index is None:
        device_index = _local_device_index()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    try:
        return _measure(
            training_config, device, device_index, max_gemms, stream_array_mib, gemm_backend, gpu_arch
        )
    finally:
        try:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            pass


def _measure(
    training_config, device, device_index, max_gemms, stream_array_mib, gemm_backend=None, gpu_arch=None
) -> Dict[str, Any]:
    info = describe_device(device_index)
    reader = make_clock_reader(info)
    try:
        return _measure_with_reader(
            training_config, device, info, reader, max_gemms, stream_array_mib, gemm_backend, gpu_arch
        )
    finally:
        if reader is not None:
            reader.close()


def _clock_model(info, gemm_backend=None, gpu_arch=None) -> Dict[str, Any]:
    """The GEMM model the clock is solved with, and what it knows about the device.

    The hardware profile is the device's own when Primus recognises it, else
    *gpu_arch*.  A registered *gemm_backend* other than Origami is tried first;
    it also supplies the spec HBM peak and matrix-core rates when Primus has
    none for the device.
    """
    profile_key, profile_source = info.profile_key, "device"
    if profile_key is None and gpu_arch:
        profile_key, profile_source = gpu_arch.strip().lower(), "--gpu-arch"
    model = {
        "profile_key": profile_key,
        "profile_source": profile_source if profile_key else None,
        "name": None,
        "predictor": None,
        "peak_hbm_gbps": info.peak_hbm_gbps,
        "matrix_rates": None,
    }
    if profile_key is None:
        return model
    name = (gemm_backend or "origami").strip().lower()
    if name != "origami":
        predictor, backend = make_backend_predictor(name, profile_key, info.num_cus)
        if predictor is not None:
            model.update(name=name, predictor=predictor, matrix_rates=backend.matrix_flops_per_cu_per_clk())
            if model["peak_hbm_gbps"] is None:
                model["peak_hbm_gbps"] = backend.hbm_bandwidth_gbps
            return model
    predictor = make_origami_predictor(profile_key, info.num_cus)
    if predictor is not None:
        model.update(name="origami", predictor=predictor)
        if model["peak_hbm_gbps"] is None:
            model["peak_hbm_gbps"] = _peak_hbm_gbps(profile_key)
    return model


def _measure_with_reader(
    training_config, device, info, reader, max_gemms, stream_array_mib, gemm_backend=None, gpu_arch=None
) -> Dict[str, Any]:
    try:
        shape_counts, dtypes = collect_workload_gemms(training_config)
        shape_source = "workload"
    except Exception as exc:
        shape_counts, dtypes = {}, {}
        shape_source = f"reference (workload shapes unavailable: {type(exc).__name__}: {exc})"
    selected = select_calibration_gemms(shape_counts, max_gemms=max_gemms)
    if shape_source == "workload" and selected[0][0] not in shape_counts:
        shape_source = "reference (no workload GEMM is large enough to be clock-bound)"

    measurements, settles, dtype_notes = [], [], []
    with ClockSampler(reader) as gemm_clocks:
        for i, (shape, count) in enumerate(selected):
            measured_shape, measured_ms, settle, note = _time_workload_gemm(
                shape,
                device,
                info.gcn_arch,
                reader=reader,
                sampler=gemm_clocks,
                min_settle_s=DEFAULT_MIN_SETTLE_S if i == 0 else _NEXT_GEMM_MIN_SETTLE_S,
            )
            measurements.append((measured_shape, count, measured_ms))
            settles.append(settle)
            if note and note not in dtype_notes:
                dtype_notes.append(note)
    gemm_clock_summary = gemm_clocks.summary()
    live_sclk = (gemm_clock_summary["sclk"] or {}).get("median_mhz")
    model = _clock_model(info, gemm_backend, gpu_arch)
    compute = back_calculate_compute_clock(
        measurements,
        info.gcn_arch,
        info.num_cus,
        model["predictor"],
        live_sclk,
        matrix_rates=model["matrix_rates"],
        model=model["name"] or "origami",
        matrix_loop=_run_matrix_loop(device, info, reader),
    )
    for row, (shape, _), settle in zip(compute["gemms"], selected, settles):
        row["settle"] = settle
        row["workload_dtype"] = shape.dtype
    compute.update(
        shape_source=shape_source,
        measured_dtype="/".join(sorted({s.dtype for s, _, _ in measurements})),
        dtype_notes=dtype_notes,
        workload_dtypes=sorted(dtypes),
        workload_unique_gemms=len(shape_counts),
        model_profile_key=model["profile_key"],
        model_profile_source=model["profile_source"],
        max_sclk_mhz=info.max_sclk_mhz,
        clock_source=gemm_clock_summary["source"],
        observed_sclk=gemm_clock_summary["sclk"],
        all_settled=all(s["settled"] for s in settles) if reader is not None else None,
    )

    with ClockSampler(reader) as stream_clocks:
        stream = run_stream(device, array_mib=stream_array_mib, reader=reader, sampler=stream_clocks)
    sustained = max(k["best_gbps"] for k in stream["kernels"].values())
    stream_clock_summary = stream_clocks.summary()
    observed_mclk = stream_clock_summary["mclk"]
    memory = {
        "stream": stream,
        "sustained_bandwidth_gbps": sustained,
        "max_mclk_mhz": info.max_mclk_mhz,
        "clock_source": stream_clock_summary["source"],
        "observed_mclk": observed_mclk,
        "memory_clock_back_calculated": False,
        **derive_model_hbm_bandwidth(sustained),
        **analyze_memory_clock(
            sustained,
            model["peak_hbm_gbps"],
            info.max_mclk_mhz,
            observed_mclk["median_mhz"] if observed_mclk else None,
        ),
    }
    return {"device": info.to_dict(), "compute": compute, "memory": memory}


def _run_matrix_loop(device, info, reader) -> Optional[Dict[str, Any]]:
    """The matrix-core loop's result with the SCLK sampled during it; *None* where it cannot run.

    A failure is recorded under ``"error"`` rather than raised: the loop is a
    cross-check, and the GEMM-based clock does not depend on it.
    """
    import torch

    if matrix_loop_unavailable_reason(info.gcn_arch) is not None:
        return None
    try:
        with ClockSampler(reader) as loop_clocks:
            # Full settle: the clock moves when the load switches from GEMMs to matrix cores only.
            result = run_matrix_core_loop(
                device, info.num_cus, sampler=loop_clocks, min_settle_s=DEFAULT_MIN_SETTLE_S
            )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    result["observed_sclk"] = loop_clocks.summary()["sclk"]
    return result


def _time_workload_gemm(shape, device, gcn_arch=None, **timing_kwargs):
    """Time *shape* in its workload precision: ``(measured_shape, ms, settle, note)``.

    Tries :func:`gemm_dtype_candidates` in order (e.g. MXFP4 natively, then
    fp8, then bf16), moving on when torch or the part cannot run one.
    *note* records any substitution and why.
    """
    import torch

    failures = []
    for dtype in gemm_dtype_candidates(shape.dtype, gcn_arch):
        measured = replace(shape, dtype=dtype)
        try:
            ms, settle = time_gemm_ms(measured, device, **timing_kwargs)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, NotImplementedError, ValueError, AttributeError, TypeError) as exc:
            if dtype == "bf16":
                raise
            failures.append(f"{dtype} unavailable: {type(exc).__name__}")
            torch.cuda.empty_cache()
            continue
        if dtype == shape.dtype or (shape.dtype == "fp4" and dtype == "mx4"):
            return measured, ms, settle, None
        reason = f" ({'; '.join(failures)})" if failures else ""
        return measured, ms, settle, f"{shape.dtype} GEMMs timed as {dtype}{reason}"
    raise AssertionError("bf16 is always a candidate")


def _rank_summary(payload: Dict[str, Any], rank: int) -> Dict[str, Any]:
    if "error" in payload:
        return {"rank": rank, "error": payload["error"]}
    observed = payload["compute"].get("observed_sclk") or {}
    return {
        "rank": rank,
        "device_index": payload["device"]["index"],
        "effective_clock_mhz": payload["compute"].get("effective_clock_mhz"),
        "observed_sclk_median_mhz": observed.get("median_mhz"),
        "observed_mclk_median_mhz": (payload["memory"].get("observed_mclk") or {}).get("median_mhz"),
        "sustained_bandwidth_gbps": payload["memory"]["sustained_bandwidth_gbps"],
        "model_hbm_bandwidth_gbps": payload["memory"].get("model_hbm_bandwidth_gbps"),
    }


def calibrated_model_hbm_bandwidth_gbps(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    """Model HBM bandwidth to project with: the slowest GPU's, since every rank waits on it.

    Re-derived from the measured STREAM bandwidth with the current derate, so a
    ``PRIMUS_HBM_STREAM_DERATE`` set when loading a saved bench still applies.
    """
    if not payload:
        return None
    values = [r.get("sustained_bandwidth_gbps") for r in payload.get("per_rank") or [] if "error" not in r]
    if "error" not in payload:
        values.append(payload.get("memory", {}).get("sustained_bandwidth_gbps"))
    values = [v for v in values if v]
    return derive_model_hbm_bandwidth(min(values))["model_hbm_bandwidth_gbps"] if values else None


def gather_rank_summaries(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Attach every rank's headline numbers to rank 0's payload (in place).

    Must be called on all ranks once ``torch.distributed`` is initialised.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return payload
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        return payload
    rank = dist.get_rank()
    local = _rank_summary(payload, rank) if payload else None
    gathered: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if payload is not None:
        payload["per_rank"] = [s for s in gathered if s is not None]
    return payload


def _fmt(value, spec: str = ".0f", unit: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{unit}"


def _settle_label(settle: Optional[Dict[str, Any]]) -> str:
    if not settle:
        return "n/a"
    if settle.get("settled") is False:
        return f"no@{settle['settle_s']:.1f}s"
    return f"{settle['settle_s']:.1f}s"


def _source_label(source: Optional[str], per_xcd: bool = False) -> str:
    if per_xcd:
        return {"amdsmi": "amdsmi, mean of all XCDs", "sysfs": "sysfs, XCD 0 only"}.get(
            source, source or "n/a"
        )
    return source or "n/a"


def _per_rank_lines(per_rank: List[Dict[str, Any]]) -> List[str]:
    lines = [
        f"  Per GPU ({len(per_rank)} ranks):",
        f"    {'rank':>4} {'origami_MHz':>11} {'live_SCLK':>9} {'live_MCLK':>9} {'STREAM_GB/s':>11} "
        f"{'model_GB/s':>10}",
    ]
    for r in per_rank:
        if "error" in r:
            lines.append(f"    {r['rank']:>4} failed: {r['error']}")
            continue
        lines.append(
            f"    {r['rank']:>4} {_fmt(r['effective_clock_mhz']):>11} "
            f"{_fmt(r['observed_sclk_median_mhz']):>9} {_fmt(r.get('observed_mclk_median_mhz')):>9} "
            f"{r['sustained_bandwidth_gbps']:>11.0f} {_fmt(r.get('model_hbm_bandwidth_gbps')):>10}"
        )
    clocks = [r["effective_clock_mhz"] for r in per_rank if r.get("effective_clock_mhz")]
    bws = [r["sustained_bandwidth_gbps"] for r in per_rank if "error" not in r]
    live = [r["observed_sclk_median_mhz"] for r in per_rank if r.get("observed_sclk_median_mhz")]
    if clocks:
        lines.append(f"    Origami-inverted clock spread {min(clocks):.0f}-{max(clocks):.0f} MHz")
    if live:
        lines.append(f"    live SCLK spread {min(live):.0f}-{max(live):.0f} MHz")
    if bws:
        lines.append(f"    STREAM bandwidth spread {min(bws):.0f}-{max(bws):.0f} GB/s")
    return lines


_INVERSION_METHODS = ("origami-inversion", "model-inversion")


def _model_label(comp: Dict[str, Any]) -> str:
    """The clock model's name for the report, with the hardware profile it was given."""
    name = comp.get("model") or "origami"
    label = "Origami" if name == "origami" else name
    if comp.get("model_profile_source") == "--gpu-arch":
        label += f" ({comp['model_profile_key']} from --gpu-arch)"
    return label


def _matrix_loop_lines(loop: Optional[Dict[str, Any]]) -> List[str]:
    if not loop:
        return []
    if "error" in loop:
        return [f"  Matrix-core loop failed: {loop['error']}"]
    lines = [
        f"  Matrix-core loop ({loop['dtype']} WMMA, {loop['waves_per_cu']} waves/CU, "
        f"settled after {_settle_label(loop.get('settle'))}): {loop['achieved_tflops']:.1f} TFLOPS",
        f"    clock from shader-cycle counter: {loop['clock_mhz']:.0f} MHz "
        f"(waves {loop['min_wave_mhz']:.0f}-{loop['max_wave_mhz']:.0f}; "
        f"real-time counter {loop['realtime_counter_mhz']:.1f} MHz)",
    ]
    rate = loop.get("rate_clock_mhz")
    if rate:
        lines.append(
            f"    clock from matrix-core rate:     {rate:.0f} MHz "
            f"({rate / loop['clock_mhz']:.0%} matrix-core utilisation at the cycle-counter clock)"
        )
    else:
        lines.append("    clock from matrix-core rate:     n/a (no FLOPs/CU/clock rate for this part)")
    sclk = loop.get("observed_sclk")
    if sclk:
        lines.append(
            f"    observed SCLK during the loop: median {sclk['median_mhz']:.0f} MHz "
            f"({loop['clock_mhz'] / sclk['median_mhz'] - 1:+.1%} cycle counter vs observed)"
        )
    return lines


def format_calibration_report(payload: Dict[str, Any]) -> str:
    per_rank = payload.get("per_rank") or []
    if "error" in payload:
        lines = ["=" * 100, f"{TAG} Calibration failed on this rank: {payload['error']}"]
        if len(per_rank) > 1:
            lines += _per_rank_lines(per_rank)
        return "\n".join(lines + ["=" * 100])

    dev = payload["device"]
    comp = payload["compute"]
    mem = payload["memory"]
    lines = [
        "=" * 100,
        f"{TAG} Device {dev['index']}: {dev['profile_key'] or dev['name']} "
        f"({dev['gcn_arch']}, {dev['num_cus']} CUs)",
        "=" * 100,
        f"  Compute clock from {len(comp['gemms'])} GEMM(s), {comp['measured_dtype']}, "
        f"shapes: {comp['shape_source']}",
        f"    {'M':>7} {'N':>7} {'K':>7} {'batch':>5} {'dtype':>5} {'count':>5} {'time_ms':>9} "
        f"{'TFLOPS':>8} {'origami_MHz':>11} {'mfma_bound_MHz':>14} {'settled':>10}",
    ]
    for g in comp["gemms"]:
        clamp = "*" if g.get("origami_clamped") else " "
        lines.append(
            f"    {g['m']:>7} {g['n']:>7} {g['k']:>7} {g['batch']:>5} {g.get('dtype', 'bf16'):>5} "
            f"{g['occurrences']:>5} {g['measured_ms']:>9.4f} {g['achieved_tflops']:>8.1f} "
            f"{_fmt(g['origami_clock_mhz']):>10}{clamp} {_fmt(g['mfma_bound_clock_mhz']):>14} "
            f"{_settle_label(g.get('settle')):>10}"
        )
    for note in comp.get("dtype_notes") or []:
        lines.append(f"    note: {note}")
    eff = comp.get("effective_clock_mhz")
    max_sclk = comp.get("max_sclk_mhz")
    pct = f", {eff / max_sclk:.0%} of {max_sclk} MHz max" if eff and max_sclk else ""
    if comp.get("method") in _INVERSION_METHODS:
        clamp_note = " (clamped to search bound)" if comp.get("clamped") else ""
        lines.append(
            f"  Effective compute clock: {eff:.0f} MHz ({_model_label(comp)} inversion{pct}){clamp_note}"
        )
        lines.append(f"  MFMA-peak lower bound:   {_fmt(comp.get('mfma_bound_clock_mhz'))} MHz")
    elif comp.get("method") == "live-sclk":
        lines.append(
            f"  Effective compute clock: {eff:.0f} MHz (live SCLK during the timed GEMMs{pct}; "
            "Origami cannot invert these GEMMs)"
        )
        lines.append(f"  MFMA-peak lower bound:   {_fmt(comp.get('mfma_bound_clock_mhz'))} MHz")
    elif comp.get("method") == "matrix-loop-cycle-counter":
        lines.append(
            f"  Effective compute clock: {eff:.0f} MHz (shader-cycle counter in the matrix-core loop{pct}; "
            "no GEMM model or live SCLK)"
        )
        lines.append(f"  MFMA-peak lower bound:   {_fmt(comp.get('mfma_bound_clock_mhz'))} MHz")
    elif comp.get("method") == "mfma-peak-lower-bound":
        lines.append(
            f"  Effective compute clock: >= {eff:.0f} MHz (MFMA-peak lower bound{pct}; "
            "install Origami for a point estimate)"
        )
    else:
        lines.append("  Effective compute clock: n/a (unknown architecture and no Origami)")
    sclk = comp.get("observed_sclk")
    if sclk:
        lines.append(
            f"  Observed SCLK ({_source_label(comp.get('clock_source'), per_xcd=True)}, during timed GEMMs): "
            f"median {sclk['median_mhz']:.0f} MHz, range {sclk['min_mhz']:.0f}-{sclk['max_mhz']:.0f} MHz "
            f"({sclk['num_samples']} samples)"
        )
        per_xcd = sclk.get("per_xcd_median_mhz") or []
        if len(per_xcd) > 1:
            lines.append(f"    per-XCD medians: {', '.join(f'{v:.0f}' for v in per_xcd)} MHz")
        if eff and comp.get("method") in _INVERSION_METHODS:
            model = "Origami" if (comp.get("model") or "origami") == "origami" else comp["model"]
            lines.append(
                f"    {model}-inverted clock is {eff / sclk['median_mhz'] - 1:+.1%} vs observed "
                f"({model} model bias on these shapes)"
            )
    if comp.get("all_settled") is False:
        lines.append(
            f"  WARNING: SCLK did not settle within {SETTLE_TIMEOUT_S:g} s for every GEMM; clocks may be unstable"
        )
    lines += _matrix_loop_lines(comp.get("matrix_loop"))
    if eff and comp.get("method") in _INVERSION_METHODS + ("live-sclk", "matrix-loop-cycle-counter"):
        lines.append(f"  To simulate at this clock: --gpu-clock-mhz {eff:.0f}")

    stream = mem["stream"]
    lines.append(
        f"  Memory bandwidth (STREAM, {stream['array_mib']} MiB {stream['dtype']} arrays, best of {stream['trials']}):"
    )
    lines.append(
        "    " + "  ".join(f"{name} {k['best_gbps']:.0f} GB/s" for name, k in stream["kernels"].items())
    )
    peak = mem.get("peak_bandwidth_gbps")
    eff_peak = f" ({mem['efficiency_vs_peak']:.0%} of {peak:.0f} GB/s peak)" if peak else ""
    lines.append(f"  Sustained HBM bandwidth: {mem['sustained_bandwidth_gbps']:.0f} GB/s{eff_peak}")
    if mem.get("workload_bandwidth_gbps"):
        lines.append(
            f"  Workload HBM bandwidth:  {mem['workload_bandwidth_gbps']:.0f} GB/s "
            f"(STREAM x {mem['stream_derate']:.2f} derate for memory-bound layer kernels)"
        )
        lines.append(
            f"  Model HBM bandwidth:     {mem['model_hbm_bandwidth_gbps']:.0f} GB/s peak-equivalent "
            f"(workload / {mem['model_reference_fraction']} sustained fraction the profilers assume); "
            f"reuse with --hbm-bandwidth-gbps {mem['model_hbm_bandwidth_gbps']:.0f}"
        )
    min_mclk = mem.get("min_mclk_mhz_at_full_efficiency")
    if min_mclk is not None:
        lines.append(
            f"  Memory clock: >= {min_mclk:.0f} MHz needed at 100% efficiency "
            f"(max MCLK {mem['max_mclk_mhz']} MHz); bandwidth alone cannot separate "
            "MCLK from efficiency"
        )
    stream_settle = stream.get("settle") or {}
    if stream_settle.get("settled") is False:
        lines.append(f"  WARNING: MCLK did not settle within {SETTLE_TIMEOUT_S:g} s before STREAM was timed")
    mclk = mem.get("observed_mclk")
    if mclk:
        at_obs = mem.get("efficiency_at_observed_mclk")
        lines.append(
            f"  Observed MCLK ({_source_label(mem.get('clock_source'))}, during timed STREAM, "
            f"settled after {_settle_label(stream_settle)}): median {mclk['median_mhz']:.0f} MHz"
            + (f" -> {at_obs:.0%} bandwidth efficiency at that clock" if at_obs else "")
        )

    if len(per_rank) > 1:
        lines += _per_rank_lines(per_rank)
    lines.append("=" * 100)
    return "\n".join(lines)
