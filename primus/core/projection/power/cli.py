###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus-cli projection power-fit`` and ``power-sample``."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import time


def add_power_projection_args(parser) -> None:
    """Power flags on the performance projection suites."""
    parser.add_argument(
        "--power-report",
        action="store_true",
        default=False,
        help=(
            "Add a power projection to the performance projection: per-GPU power under the\n"
            "power cap, the clock the cap forces, power-capped tokens/s, energy per token and\n"
            "node / cluster / facility power. Implied by --power-config or --power-cap-w.\n"
        ),
    )
    parser.add_argument(
        "--power-config",
        type=str,
        default=None,
        help=(
            "YAML with power settings: power.{gpu_arch, cap_w, cap_mode, model, profile,\n"
            "profile_files, footprint}, infrastructure.{...} and capacity.{facility_power_kw |\n"
            "it_power_kw, caps_w, max_nodes}. A fitted profile from 'power-fit' is also accepted.\n"
        ),
    )
    parser.add_argument(
        "--power-cap-w",
        type=float,
        default=None,
        help=(
            "Per-GPU board power cap in watts (the customer's power limit). Defaults to the\n"
            "GPU's published total board power. Also PRIMUS_POWER_CAP_W.\n"
        ),
    )
    parser.add_argument(
        "--power-cap-mode",
        type=str,
        default=None,
        choices=["instantaneous", "average"],
        help=(
            "How the cap is enforced: 'instantaneous' (default) caps every phase of the\n"
            "iteration; 'average' caps only the iteration-average power.\n"
        ),
    )


def add_power_fit_args(parser) -> None:
    parser.add_argument(
        "--telemetry",
        nargs="+",
        required=True,
        help="Telemetry logs: amd-smi / rocm-smi CSV, or JSON lines from 'power-sample'.",
    )
    parser.add_argument(
        "--gpu-arch", type=str, default=None, help="Architecture the profile is for (e.g. mi355x)."
    )
    parser.add_argument("--out", type=str, required=True, help="Where to write the fitted profile YAML.")
    parser.add_argument(
        "--matrix-utilization",
        type=float,
        default=None,
        help="Matrix utilization (e.g. reported MFU) to assume when the log has no matrix column.",
    )
    parser.add_argument(
        "--tbp-w", type=float, default=None, help="Board power limit, if not the published TBP."
    )
    parser.add_argument(
        "--peak-clock-mhz", type=float, default=None, help="Peak engine clock, if not published."
    )
    parser.add_argument(
        "--fit-alpha",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Fit the frequency exponent (auto: only if the clock varies in the log).",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START_UNIX", "END_UNIX"),
        default=None,
        help="Fit only samples inside the load window (Unix seconds).",
    )
    window.add_argument(
        "--window-file",
        type=str,
        default=None,
        help=(
            "JSON with the load window: start_unix/end_unix (written by 'power-sample'),\n"
            "benchmark_start_time_unix/benchmark_end_time_unix, or start/end.\n"
        ),
    )
    window.add_argument(
        "--auto-window",
        action="store_true",
        default=False,
        help="Use the span from the first to the last sample with GFX activity >= 50%%.",
    )
    parser.add_argument(
        "--trim-s",
        nargs=2,
        type=float,
        metavar=("HEAD_S", "TAIL_S"),
        default=None,
        help="Shrink the window by HEAD_S after the start (warmup) and TAIL_S before the end.",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=None,
        help="Largest gap between samples that still counts as continuous coverage (default 3 s).",
    )
    parser.add_argument(
        "--allow-invalid-window",
        action="store_true",
        default=False,
        help="Fit even when the window does not validate (gaps, partial coverage).",
    )
    parser.add_argument("--json", action="store_true", default=False, help="Print the fit report as JSON.")


def _resolve_window(args, samples):
    from primus.core.projection.power.window import (
        MeasurementWindow,
        auto_window,
        load_window,
    )

    if getattr(args, "window", None):
        window = MeasurementWindow(args.window[0], args.window[1], source="--window")
    elif getattr(args, "window_file", None):
        window = load_window(args.window_file)
    elif getattr(args, "auto_window", False):
        window = auto_window(samples)
        if window is None:
            raise SystemExit("--auto-window needs timestamps and a GFX activity column in the telemetry.")
    else:
        return None
    if getattr(args, "trim_s", None):
        window = window.trimmed(*args.trim_s)
    return window


def run_power_fit(args) -> int:
    from primus.core.projection.power.specs import resolve_power_profile
    from primus.core.projection.power.telemetry import (
        fit_power_profile,
        load_telemetry_files,
        write_profile_yaml,
    )
    from primus.core.projection.power.window import (
        MAX_SAMPLE_GAP_S,
        measure_window,
        samples_in_window,
    )

    overrides = {}
    if args.tbp_w:
        overrides["tbp_w"] = args.tbp_w
    if args.peak_clock_mhz:
        overrides["peak_clock_mhz"] = args.peak_clock_mhz
    base = resolve_power_profile(args.gpu_arch, overrides=overrides or None)
    if base is None:
        raise SystemExit(
            f"No published spec for '{args.gpu_arch}'; pass --tbp-w and --peak-clock-mhz to fit a new profile."
        )
    samples = load_telemetry_files(args.telemetry)
    window = _resolve_window(args, samples)
    measurement = None
    if window is not None:
        max_gap = args.max_gap_s if getattr(args, "max_gap_s", None) else MAX_SAMPLE_GAP_S
        measurement = measure_window(samples, window, max_gap_s=max_gap)
        if not measurement.valid and not args.allow_invalid_window:
            raise SystemExit(
                f"[Primus:Power] Load window did not validate ({', '.join(measurement.reasons)}); "
                f"fix the telemetry or pass --allow-invalid-window."
            )
        total = len(samples)
        samples = samples_in_window(samples, window)
        print(
            f"[Primus:Power] Load window {window.duration_s:.1f} s ({window.source}): {len(samples)}/{total} samples"
        )
        print(f"  {measurement.describe()}")
    elif any(s.timestamp is not None for s in samples):
        print(
            "[Primus:Power] No load window given; fitting every sample, including warmup and idle tails. "
            "Use --window-file / --window / --auto-window to restrict the fit to the load."
        )
    fit_alpha = {"auto": None, "yes": True, "no": False}[args.fit_alpha]
    report = fit_power_profile(samples, base, matrix_utilization=args.matrix_utilization, fit_alpha=fit_alpha)
    write_profile_yaml(report.profile, args.out, arch=args.gpu_arch or base.name)
    if args.json:
        doc = report.as_dict()
        if measurement is not None:
            doc["measurement"] = measurement.as_dict()
        print(json.dumps(doc, indent=2))
    else:
        p = report.profile
        print(f"[Primus:Power] Fitted {report.num_samples} samples -> {args.out}")
        print(f"  idle {p.idle_w:.1f} W, alpha {p.alpha:.2f}, coefficients {p.coefficients.as_dict()}")
        print(f"  RMSE {report.rmse_w:.1f} W, MAPE {report.mape * 100:.1f}%")
        print(f"  fitted: {', '.join(report.fitted)}; kept from base: {', '.join(report.kept) or '-'}")
        for w in report.warnings:
            print(f"  WARNING: {w}")
    return 0


def add_power_sample_args(parser) -> None:
    parser.add_argument("--device", type=int, nargs="+", default=[0], help="GPU indices to sample.")
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Workload to run while sampling; its start and end become the load window.",
    )
    parser.add_argument(
        "--duration-s", type=float, default=60.0, help="How long to sample when no --command is given."
    )
    parser.add_argument("--interval-ms", type=float, default=20.0, help="Sampling interval.")
    parser.add_argument(
        "--pre-roll-s",
        type=float,
        default=1.0,
        help="Sampling before and after the load so the window edges are bracketed by samples.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="JSON-lines output for 'power-fit'; the window is written next to it as <out>.window.json.",
    )


def window_path_for(out: str) -> str:
    stem, _ = os.path.splitext(out)
    return f"{stem}.window.json"


def run_power_sample(args) -> int:
    from primus.core.projection.power.telemetry import make_telemetry_sampler
    from primus.core.projection.power.window import MeasurementWindow, measure_window

    interval_s = args.interval_ms / 1000.0
    samplers = [make_telemetry_sampler(d, interval_s=interval_s) for d in args.device]
    if any(s is None for s in samplers):
        raise SystemExit("amdsmi is not available on this host; record telemetry with amd-smi instead.")
    roll = max(args.pre_roll_s, 2.0 * interval_s)
    what = f"'{args.command}'" if args.command else f"{args.duration_s:.0f} s"
    print(f"[Primus:Power] Sampling GPU(s) {', '.join(map(str, args.device))} during {what} ...")
    returncode = 0
    with contextlib.ExitStack() as stack:
        for s in samplers:
            stack.enter_context(s)
        time.sleep(roll)
        start = time.time()
        if args.command:
            argv = args.command if os.name == "nt" else shlex.split(args.command)
            returncode = subprocess.run(argv).returncode
        else:
            time.sleep(args.duration_s)
        end = time.time()
        time.sleep(roll)

    samples = [x for s in samplers for x in s.samples]
    with open(args.out, "w", encoding="utf-8") as f:
        for x in sorted(samples, key=lambda x: (x.timestamp or 0.0, x.device or "")):
            f.write(json.dumps(x.as_dict()) + "\n")
    window = MeasurementWindow(start, end, source="power-sample")
    measurement = measure_window(samples, window, expected_devices=len(args.device))
    doc = {
        **window.as_dict(),
        "command": args.command,
        "returncode": returncode,
        "measurement": measurement.as_dict(),
    }
    window_path = window_path_for(args.out)
    with open(window_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[Primus:Power] {len(samples)} samples -> {args.out}; window -> {window_path}")
    print(f"  {measurement.describe()}")
    if returncode:
        print(f"  WARNING: command exited with {returncode}")
    return returncode
