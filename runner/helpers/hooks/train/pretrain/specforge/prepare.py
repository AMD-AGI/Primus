###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForge pre-train preparation hook for primus-cli direct.

Invoked by ``runner/helpers/hooks/train/pretrain/prepare_experiment.sh``.

The important job here is the run mode. ``runner/primus-cli-direct.sh`` defaults
to ``torchrun --nproc_per_node ${GPUS_PER_NODE:-8}``, but SpecForge spawns its
own distributed workers. Nesting the two launchers oversubscribes every GPU, so
this hook emits ``env.RUN_MODE=single`` (and ``env.GPUS_PER_NODE=1``) to make
Primus launch a single plain-Python process that then execs SpecForge.

It also runs a GPU-free stack preflight (hidden states, SpecForge cwd, AITER /
radix-cache, ROCm pins) so a bad image or env fails in seconds instead of after
SpecForge starts.
"""

import argparse
import os
from pathlib import Path

from primus.backends.specforge.argument_builder import specforge_mode
from primus.backends.specforge.stack_preflight import (
    apply_rocm_stack_env,
    collect_issues,
    enforce_rocm_stack,
)
from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import log_error_and_exit, log_info


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus SpecForge environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (kept for compatibility; not used by this hook)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional override for the SpecForge checkout; takes precedence over SPECFORGE_ROOT.",
    )
    return parser.parse_known_args()


def resolve_specforge_root(cli_path, pre_trainer_cfg):
    """Pick the SpecForge checkout: CLI > config > SPECFORGE_ROOT."""

    for source, value in (
        ("--backend_path", cli_path),
        ("pre_trainer.specforge_root", getattr(pre_trainer_cfg, "specforge_root", None)),
        ("SPECFORGE_ROOT", os.getenv("SPECFORGE_ROOT")),
    ):
        if value and Path(value).is_dir():
            log_info(f"SpecForge root from {source}: {value}")
            return Path(value).resolve()
        if value:
            log_info(f"SpecForge root from {source} does not exist, skipping: {value}")
    return None


def preflight_stack(pre_trainer_cfg):
    """Fail fast on missing hidden states, a CUDA clobber, or AITER/radix misconfig."""

    issues = collect_issues(pre_trainer_cfg)
    if issues:
        log_error_and_exit("stack preflight failed:\n" + "\n".join(f"  - {item}" for item in issues))
    log_info("SpecForge stack preflight OK")


def main():
    args, unknown = parse_args()

    exp_path = Path(args.config).resolve()
    log_info(f"PRIMUS_PATH: {Path(args.primus_path).resolve()}")
    log_info(f"DATA_PATH: {Path(args.data_path).resolve()}")
    log_info(f"EXP: {exp_path}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    primus_config, _ = load_primus_config(args, unknown)

    try:
        pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if specforge_mode(pre_trainer_cfg) != "capture" and not getattr(
        pre_trainer_cfg, "specforge_config", None
    ):
        log_error_and_exit("Missing required field: pre_trainer.specforge_config")

    specforge_root = resolve_specforge_root(args.backend_path, pre_trainer_cfg)
    if specforge_root is not None:
        os.environ["SPECFORGE_ROOT"] = str(specforge_root)
        print(f"env.SPECFORGE_ROOT={specforge_root}")
        print(f"extra.backend_path={specforge_root}")

    preflight_stack(pre_trainer_cfg)

    # SpecForge self-launches its workers; keep Primus to one plain-Python
    # process so the two launchers do not nest.
    log_info("Exposing run mode via env.RUN_MODE=single (SpecForge owns its launcher)")
    print("env.RUN_MODE=single")
    print("env.GPUS_PER_NODE=1")

    if enforce_rocm_stack(os.environ):
        for key, value in apply_rocm_stack_env():
            log_info(f"ROCm stack default {key}={value}")
        # Re-print even when already set, so execute_hooks.sh exports them.
        for key, default in (
            ("SGLANG_USE_AITER", "1"),
            ("SGLANG_USE_AITER_UNIFIED_ATTN", "1"),
            ("AITER_FLYDSL_FORCE", "1"),
            ("SGLANG_DISABLE_RADIX_CACHE", "1"),
        ):
            print(f"env.{key}={os.environ.get(key, default)}")


if __name__ == "__main__":
    log_info("========== Prepare SpecForge Env (pre-train hook) ==========")
    main()
