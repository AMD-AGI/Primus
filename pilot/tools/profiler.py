"""profiler tool: single-node profiling for Projection.

Wraps rocprof / RCCL profiler / torch.profiler — does not invent its own profiler.

Status: skeleton.
"""

from __future__ import annotations
import argparse
import sys


def run(model_spec: dict, configs: list[dict]) -> dict:
    """Run single-node profiling across (layers, mbs, recompute) variants.

    Returns:
        dict with per-config T_comp / Mem_peak / kernel breakdown.
    """
    raise NotImplementedError("pilot.tools.profiler.run")


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.profiler")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--model-spec", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"CLI dispatch for {args.cmd!r} not implemented")


if __name__ == "__main__":
    sys.exit(_cli())
