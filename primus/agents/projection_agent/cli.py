###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Command line of the projection agent, and the ``--op-*`` flags it shares.

The same flag group is registered on ``primus projection``, on the tuning
agent and here, so an op is brought up the same way whoever asks::

    # bring the workload's missing ops up, then project
    primus projection performance --config <yaml> --profiling-mode simulate \\
        --op-bringup auto --op-coder-cmd 'cursor-agent -p "$(cat {prompt_file})" --output-format text'

    # bring them up only
    python -m primus.agents.projection_agent --workload <yaml> --op-llm-model anthropic/claude-opus-4-5

    # ... and open a draft pull request per op it brings up
    python -m primus.agents.projection_agent --workload <yaml> --op-coder-cmd '...' --op-pr pr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .agent import POLICIES, ProjectionAgent
from .bringup import llm_model_from_env, make_coder
from .publish import MODES as PR_MODES
from .publish import PublishConfig


def add_op_bringup_args(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    g = parser.add_argument_group("op bring-up (ops the projection does not model yet)")
    g.add_argument(
        "--op-bringup",
        choices=POLICIES,
        default=default,
        help=(
            "auto: write, validate and register an op model for every unmodeled op "
            "(falls back to report without a coder); report: write the context packs "
            f"only; off: price unmodeled ops with the built-in stand-in. Default: {default}."
        ),
    )
    g.add_argument(
        "--op-coder",
        choices=("auto", "dspy", "command", "none"),
        default="auto",
        help="Who writes the op model. auto: command if --op-coder-cmd, else an LLM if one is configured.",
    )
    g.add_argument(
        "--op-coder-cmd",
        default=None,
        help="Coding-agent shell template; {prompt_file}, {target_file} and {op_models_dir} are filled in.",
    )
    g.add_argument(
        "--op-llm-model",
        default=None,
        help="LiteLLM model string for --op-coder dspy (default: $PRIMUS_OP_LLM_MODEL or $LLM_MODEL).",
    )
    g.add_argument(
        "--op-paper", action="append", default=[], help="Paper text for the new op (file/URL/inline)."
    )
    g.add_argument("--hf-config", default=None, help="HF config.json path, URL or repo id.")
    g.add_argument(
        "--op-out-dir",
        type=Path,
        default=None,
        help="Where context packs and reports go (op_bringup/<op_id>/).",
    )
    g.add_argument(
        "--op-pr",
        choices=PR_MODES,
        default="off",
        help=(
            "Publish every op brought up on its own branch off HEAD. commit: local branch; "
            "push: also push it to --op-pr-remote; pr: also open a draft pull request "
            "(gh, else $GITHUB_TOKEN/$GH_TOKEN, else the compare URL is printed). Default: off."
        ),
    )
    g.add_argument(
        "--op-pr-remote", default="origin", help="Remote the branch is pushed to. Default: origin."
    )
    g.add_argument(
        "--op-pr-base",
        default=None,
        help="Branch the pull request targets. Default: the remote's default branch.",
    )
    g.add_argument(
        "--op-pr-repo",
        default=None,
        help="owner/name the pull request is opened on, when pushing to a fork. Default: the remote's repo.",
    )
    g.add_argument(
        "--op-pr-author",
        default=os.getenv("PRIMUS_OP_PR_AUTHOR"),
        help="'Name <email>' for the commit when git has no identity configured (default: $PRIMUS_OP_PR_AUTHOR).",
    )
    g.add_argument("--op-pr-signoff", action="store_true", help="Add a Signed-off-by trailer to the commit.")
    g.add_argument(
        "--op-pr-ready", action="store_true", help="Open the pull request ready for review, not draft."
    )


def publish_from_args(args) -> Optional[PublishConfig]:
    mode = getattr(args, "op_pr", "off")
    if mode == "off":
        return None
    return PublishConfig(
        mode=mode,
        remote=args.op_pr_remote,
        base=args.op_pr_base,
        repo=args.op_pr_repo,
        signoff=args.op_pr_signoff,
        author=args.op_pr_author,
        draft=not args.op_pr_ready,
    )


def _llm_available(args) -> bool:
    try:
        import dspy
    except ImportError:
        return False
    return getattr(dspy.settings, "lm", None) is not None or bool(args.op_llm_model or llm_model_from_env())


def coder_from_args(args, *, log=print):
    kind = args.op_coder
    if kind == "auto":
        kind = "command" if args.op_coder_cmd else ("dspy" if _llm_available(args) else "none")
    try:
        return make_coder(kind, cmd=args.op_coder_cmd, model=args.op_llm_model)
    except Exception as e:
        log(f"[projection-agent] op coder '{kind}' unavailable ({e}); writing context packs only")
        return None


def agent_from_args(
    args, *, out_dir: Optional[Path] = None, gpu_arch: Optional[str] = None, **kwargs
) -> Optional[ProjectionAgent]:
    """The :class:`ProjectionAgent` the ``--op-*`` flags describe, or ``None`` for ``off``."""
    if args.op_bringup == "off":
        return None
    return ProjectionAgent(
        out_dir=args.op_out_dir or out_dir or Path("op_bringup_runs"),
        gpu_arch=gpu_arch or os.getenv("PRIMUS_GPU_ARCH"),
        policy=args.op_bringup,
        coder=coder_from_args(args) if args.op_bringup == "auto" else None,
        papers=args.op_paper,
        hf_config=args.hf_config,
        publish=publish_from_args(args),
        **kwargs,
    )


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="primus.agents.projection_agent",
        description="Bring up the ops a workload needs that the Primus projection does not model.",
    )
    p.add_argument("--workload", required=True, type=Path)
    p.add_argument("--gpu-arch", default=os.getenv("PRIMUS_GPU_ARCH", "mi355x"))
    p.add_argument("--op", action="append", default=[], help="Only these op ids")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--no-register", action="store_true", help="Validate but do not copy into generated/")
    add_op_bringup_args(p)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    agent = agent_from_args(
        args, gpu_arch=args.gpu_arch, max_attempts=args.max_attempts, register=not args.no_register
    )
    if agent is None:
        return 0
    results = agent.ensure_ops(args.workload, only=args.op)
    print(json.dumps([r.as_dict() for r in results], indent=2, default=str))
    if agent.published:
        print(json.dumps({k: v.as_dict() for k, v in agent.published.items()}, indent=2))
    return 0 if all(r.status in ("registered", "acknowledged", "context_only") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
