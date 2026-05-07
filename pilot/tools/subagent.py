"""subagent tool: protocol abstraction for spawning Stage Workers.

This module defines the contract; concrete implementations are injected by
`integrations/<framework>/` (Claude Code Task / Cursor Task / harness fork).

Pilot core never imports an agent SDK — `subagent.spawn` here only validates
inputs and outputs against schemas, then delegates to a registered backend.

Status: skeleton.
"""

from __future__ import annotations
import argparse
import sys
from typing import Callable


# Backend registry. integrations/<framework>/ calls `register_backend(name, fn)`
# at startup. Pilot core does not import the impl.
_BACKENDS: dict[str, Callable] = {}


def register_backend(name: str, fn: Callable) -> None:
    """Integration adapters call this to register their spawn implementation."""
    _BACKENDS[name] = fn


def spawn(*,
          stage: str,
          input_refs: dict,
          skill_scope: list[str],
          max_tokens: int = 30_000,
          backend: str | None = None) -> dict:
    """Spawn a Stage Worker, return a SubagentResult (§8.11).

    Args:
        stage: stage name (DIAGNOSE / RE_PLAN / ENV_SWEEP / ...).
        input_refs: pointer-class refs into the State Layer (NOT full payloads).
        skill_scope: allowlist of Skill paths the Worker may read.
        max_tokens: Worker single-peak budget cap (§13.4).
        backend: which registered backend to use. If None, uses env var
                 `PILOT_SUBAGENT_BACKEND` or the first registered.

    Returns:
        dict matching schemas/subagent_result.schema.json
        (`summary` field validated to be < 200 tokens).
    """
    raise NotImplementedError("pilot.tools.subagent.spawn")


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.subagent")
    sub = p.add_subparsers(dest="cmd", required=True)
    spawn_p = sub.add_parser("spawn")
    spawn_p.add_argument("--stage", required=True)
    spawn_p.add_argument("--max-tokens", type=int, default=30_000)
    args = p.parse_args()
    raise NotImplementedError(f"CLI dispatch for {args.cmd!r} not implemented")


if __name__ == "__main__":
    sys.exit(_cli())
