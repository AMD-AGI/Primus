#!/usr/bin/env python3
"""Launch the in-tree diffusion backend directly with torchrun."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.runtime import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Native PyTorch diffusion training")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    text = os.path.expandvars(args.config.read_text())
    unresolved = sorted(set(re.findall(r"\$(?:\w+|\{[^}]+\})", text)))
    if unresolved:
        raise ValueError(f"Unset environment variables in {args.config}: {', '.join(unresolved)}")
    params = yaml.safe_load(text)
    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {args.config}")

    builder = DiffusionArgBuilder()
    builder.update(params)
    run_training(builder.finalize())


if __name__ == "__main__":
    main()
