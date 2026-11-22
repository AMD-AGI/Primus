###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus-cli projection power-fit`` and ``power-sample``."""

from __future__ import annotations

import json
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
    parser.add_argument("--json", action="store_true", default=False, help="Print the fit report as JSON.")


def run_power_fit(args) -> int:
    from primus.core.projection.power.specs import resolve_power_profile
    from primus.core.projection.power.telemetry import (
        fit_power_profile,
        load_telemetry,
        write_profile_yaml,
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
    samples = []
    for path in args.telemetry:
        samples.extend(load_telemetry(path))
    fit_alpha = {"auto": None, "yes": True, "no": False}[args.fit_alpha]
    report = fit_power_profile(samples, base, matrix_utilization=args.matrix_utilization, fit_alpha=fit_alpha)
    write_profile_yaml(report.profile, args.out, arch=args.gpu_arch or base.name)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
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
    parser.add_argument("--device", type=int, default=0, help="GPU index to sample.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="How long to sample.")
    parser.add_argument("--interval-ms", type=float, default=20.0, help="Sampling interval.")
    parser.add_argument("--out", type=str, required=True, help="JSON-lines output for 'power-fit'.")


def run_power_sample(args) -> int:
    from primus.core.projection.power.telemetry import make_telemetry_sampler

    sampler = make_telemetry_sampler(args.device, interval_s=args.interval_ms / 1000.0)
    if sampler is None:
        raise SystemExit("amdsmi is not available on this host; record telemetry with amd-smi instead.")
    print(f"[Primus:Power] Sampling GPU {args.device} for {args.duration_s:.0f} s ...")
    with sampler:
        time.sleep(args.duration_s)
    sampler.write_jsonl(args.out)
    summary = sampler.summary()
    print(f"[Primus:Power] {summary['num_samples']} samples -> {args.out}")
    if summary["avg_power_w"] is not None:
        print(f"  avg {summary['avg_power_w']:.1f} W, max {summary['max_power_w']:.1f} W")
    return 0
