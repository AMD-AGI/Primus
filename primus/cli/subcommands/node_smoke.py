###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
node_smoke CLI subcommand.

Surfaces ``primus.tools.preflight.node_smoke`` as a first-class primus-cli
subcommand so users can run it via the same dispatch chain as ``preflight``,
``train``, ``benchmark``, etc.:

    primus-cli direct -- node_smoke --tier2-perf
    primus-cli slurm srun -N 4 -- direct -- node_smoke --tier2-perf
    primus-cli slurm srun -N 4 -- container -- node_smoke --tier2-perf

Design choices (consolidate-preflight-direct-wrappers plan, section 4):

1. **No inner ``run`` keyword.** The standalone CLI has
   ``run / aggregate / _per_gpu`` subparsers, but only ``run`` is user-facing.
   Hoisting its flags onto the top-level ``node_smoke`` parser means the user
   types ``primus-cli direct -- node_smoke --tier2-perf`` instead of
   ``... -- node_smoke run --tier2-perf``.

2. **Always aggregate on rank 0.** Every wrapper invocation runs ``_cmd_run``
   on every rank, then ``_cmd_aggregate`` on rank 0 only. Aggregation takes
   a few seconds (reads per-node JSONs, writes one report) and is what users
   want ~100% of the time. The rare exceptions ("run only, don't aggregate"
   / "aggregate only") stay reachable through the unchanged standalone CLI:

       python -m primus.tools.preflight.node_smoke run ...
       python -m primus.tools.preflight.node_smoke aggregate ...

3. **``allow_abbrev=False``.** Mirrors the standalone ``run`` subparser at
   ``primus/tools/preflight/node_smoke/cli.py:542``. Without this, an old
   script that still passes ``--tier2`` (legacy flag name) would silently
   match ``--tier2-perf`` as a prefix and run the wrong test set.

4. **No ``--silent`` flag.** Silencing is handled exclusively by the bash
   launcher (``primus-cli-direct.sh`` ``--silent`` before ``--``). Passing
   ``--silent`` here will be rejected by argparse, which is the desired
   behavior -- one knob in one place.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List


def run(args: Any, extra_args: List[str]) -> None:
    """Two-phase dispatch: per-node ``_cmd_run`` on every rank, then rank-0
    ``_cmd_aggregate`` with SLURM-resolved aggregator args.

    Exit-code rule (matches the deleted wrapper):
      - Non-rank-0: propagate ``_cmd_run`` exit code.
      - Rank 0: aggregator exit code wins (it knows about MISSING nodes that
        ``_cmd_run`` can't see). If aggregator returns 0 but run failed, we
        still surface the run failure so a sick rank-0 box can't paint itself
        green via a successful aggregate.
    """
    from primus.tools.preflight.node_smoke.cli import (
        _cmd_aggregate,
        _cmd_run,
        _resolve_aggregate_args_from_slurm,
    )

    if extra_args:
        # node_smoke uses allow_abbrev=False at the argparse level; reaching
        # this path means the user passed something the parser didn't claim.
        # Surface it loudly so a typo'd flag doesn't get silently dropped.
        print(
            f"[Primus:NodeSmoke] Unknown arguments: {extra_args}. "
            f"Run `primus-cli node_smoke --help` for valid options.",
        )
        raise SystemExit(2)

    rc_run = int(_cmd_run(args))
    rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0")))
    if rank != 0:
        raise SystemExit(rc_run)

    agg_ns = _resolve_aggregate_args_from_slurm(args)
    rc_agg = int(_cmd_aggregate(agg_ns))

    # Aggregator's exit code wins on rank 0 (it can detect MISSING nodes),
    # but a failed per-node run on rank 0 itself must not be hidden by a
    # successful aggregate. Surface the higher of the two.
    raise SystemExit(max(rc_agg, rc_run))


def register_subcommand(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``node_smoke`` with the primus-cli main parser.

    Hoists the ``run`` subparser's flags directly onto the top-level
    ``node_smoke`` parser (no inner ``run`` keyword) plus the aggregator
    tuning knobs the rank-0 aggregate step needs.
    """
    from primus.tools.preflight.node_smoke.cli import (
        _add_aggregate_flags,
        _add_run_flags,
    )

    parser = subparsers.add_parser(
        "node_smoke",
        help="Run per-node preflight smoke test on every node and aggregate on rank 0.",
        description=(
            "Node-local preflight smoke test. Each node runs independently "
            "(no global rendezvous, no torch.distributed). On rank 0 the per-node "
            "verdicts are aggregated into smoke_report.md + passing_nodes.txt + "
            "failing_nodes.txt -- the latter two are directly consumable by "
            "`srun --nodelist=` / `srun --exclude=`."
        ),
        # See module docstring point 3.
        allow_abbrev=False,
    )

    # Attach the canonical run-side flag surface. Anything new added to
    # `_add_run_flags` automatically flows here.
    _add_run_flags(parser)

    # Attach the aggregator-only flags. `--dump-path` / `--hbm-busy-threshold-gib`
    # / `--gpu-activity-warn-pct` are already on the parser from `_add_run_flags`
    # with identical defaults, so we skip them here to avoid argparse
    # conflicting-option errors.
    _add_aggregate_flags(parser, include_dump_path=False)

    parser.set_defaults(func=run)
    return parser
