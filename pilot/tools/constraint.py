"""constraint tools: safety checks consumed by Re-Plan / EnvSweep / Execute.

Status: skeleton.
"""

from __future__ import annotations
import argparse
import sys


def check(plan: dict, cluster: dict) -> dict:
    """Static config validity (§8.2 Plan + skills/constraints/config.md).

    Returns:
        {valid: bool, violations: list[str]}
    """
    raise NotImplementedError("pilot.tools.constraint.check")


def check_env(env_diff: dict, baseline: dict) -> dict:
    """Validate env combination against incompatibility matrix.

    Returns:
        {valid: bool, violations: list[str]}
    """
    raise NotImplementedError("pilot.tools.constraint.check_env")


def estimate_mem(plan: dict) -> dict:
    """Memory estimate (§S1 calibrated formula).

    Returns:
        {mem_gb: float, components: {param, grad, optim, act, buffer}, confidence: float}
    """
    raise NotImplementedError("pilot.tools.constraint.estimate_mem")


def diagnose_failure(snapshot_or_error: dict) -> dict:
    """Failure attribution (§8.8 FailureReport).

    Returns:
        FailureReport dict matching schemas/failure_report.schema.json
    """
    raise NotImplementedError("pilot.tools.constraint.diagnose_failure")


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.constraint")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "check_env", "estimate_mem", "diagnose_failure"):
        sub.add_parser(name)
    args = p.parse_args()
    raise NotImplementedError(f"CLI dispatch for {args.cmd!r} not implemented")


if __name__ == "__main__":
    sys.exit(_cli())
