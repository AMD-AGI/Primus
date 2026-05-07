"""Entry point: ``python -m pilot.agent``."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import DEFAULT_ORCHESTRATOR_MODEL, DEFAULT_WORKER_MODEL, OrchestratorConfig
from .orchestrator import Orchestrator
from .schemas import Budget


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("pilot-agent")
    ap.add_argument("--session", required=True, help="session id, e.g. demo_001")
    ap.add_argument("--state-dir", default="pilot/state", type=Path)
    ap.add_argument("--skills-dir", default="pilot/skills", type=Path)
    ap.add_argument("--gpu-h", type=float, default=10.0)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--wallclock-h", type=float, default=24.0)
    ap.add_argument("--token-budget", type=int, default=400_000,
                    help="orchestrator's total token budget across session")
    ap.add_argument("--orchestrator-model", default=DEFAULT_ORCHESTRATOR_MODEL)
    ap.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="no Anthropic API calls; use mocked subagent results")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    cfg = OrchestratorConfig(
        session_id=args.session,
        session_dir=args.state_dir / args.session,
        skills_dir=args.skills_dir,
        budget_total=Budget(
            gpu_h=args.gpu_h,
            rounds=args.max_rounds,
            wallclock_h=args.wallclock_h,
            total_tokens=args.token_budget,
        ),
        orchestrator_model=args.orchestrator_model,
        worker_model=args.worker_model,
        dry_run=args.dry_run,
    )

    client = None
    if not cfg.dry_run:
        try:
            from anthropic import Anthropic
        except ImportError:
            print("anthropic package missing. Install: pip install anthropic", file=sys.stderr)
            return 2
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ANTHROPIC_API_KEY not set. Use --dry-run for a no-key walkthrough.", file=sys.stderr)
            return 2
        client = Anthropic()

    orch = Orchestrator(cfg, client)
    state = orch.run()
    print(f"\n=== done ===\nfinal stage: {state.current_stage.value}\n"
          f"rounds: {state.round_id}\n"
          f"tokens used: {state.budget_used.total_tokens}\n"
          f"summary: {state.last_decision_summary}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
