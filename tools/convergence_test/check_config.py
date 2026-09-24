###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lint a Primus config for settings that silently ruin a convergence run.

Most Primus example configs are throughput benchmarks. They are tuned to make
the step timer look good, and several of those choices make the loss curve
meaningless rather than merely slow. The worst offenders do not fail loudly:

  * ``mock_data: true``                      trains on random tokens
  * ``moe_router_force_load_balancing: true`` feeds the router *random* logits
  * ``lr: 1e-5`` with a 2-iteration warmup    barely moves the loss
  * ``lr_decay_iters < train_iters``          pins the tail of the run at min_lr

This resolves the config the same way the trainer does (module preset + model
preset + overrides) so inherited defaults are checked too, then reports errors
and warnings. Exit status is non-zero if any error is found.

Usage:
    python3 tools/convergence_test/check_config.py --config <exp.yaml> [--strict]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PRIMUS_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PRIMUS_PATH))

ERRORS = []
WARNINGS = []
NOTES = []


def error(msg, fix=None):
    ERRORS.append((msg, fix))


def warn(msg, fix=None):
    WARNINGS.append((msg, fix))


def note(msg):
    NOTES.append(msg)


def get(cfg, key, default=None):
    return getattr(cfg, key, default)


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def check_data(cfg):
    """Data plumbing: the run is worthless if this is wrong."""
    if get(cfg, "mock_data"):
        error(
            "mock_data is true -- the model trains on random tokens",
            "set mock_data: false and point train_data_path at a real dataset",
        )

    train_paths = as_list(get(cfg, "train_data_path"))
    if not train_paths:
        error(
            "train_data_path is empty",
            "run tools/convergence_test/prepare_dataset.py and set train_data_path",
        )
        return None

    if get(cfg, "split") is not None:
        error(
            f"split is {get(cfg, 'split')!r} but explicit per-split data paths are set; "
            "Megatron asserts split is None when blend_per_split is used",
            "set split: null",
        )

    missing = [p for p in train_paths if not Path(f"{p}.idx").exists()]
    for path in as_list(get(cfg, "valid_data_path")):
        if not Path(f"{path}.idx").exists():
            missing.append(path)
    if missing:
        error(
            "dataset files not found: " + ", ".join(f"{m}.idx" for m in missing),
            "build them with tools/convergence_test/prepare_dataset.py",
        )
        return None

    manifest_path = Path(train_paths[0]).parent / "dataset_info.json"
    if not manifest_path.exists():
        warn(
            f"no dataset_info.json next to {train_paths[0]}; cannot verify the "
            "dataset and the config agree on a vocabulary"
        )
        return None
    return json.loads(manifest_path.read_text())


def check_tokenizer(cfg, manifest):
    """A vocab mismatch between corpus and model is the classic silent killer."""
    if manifest is None:
        return
    tokenizer_model = get(cfg, "tokenizer_model")
    manifest_dir = manifest.get("tokenizer_dir")
    # The config may hold a relative path; compare what they actually point at.
    same = (
        tokenizer_model and manifest_dir and Path(tokenizer_model).resolve() == Path(manifest_dir).resolve()
    )
    if tokenizer_model and manifest_dir and not same:
        warn(
            f"tokenizer_model ({tokenizer_model}) is not the tokenizer the dataset "
            f"was built with ({manifest_dir})",
            f"set tokenizer_model: {manifest_dir}",
        )

    try:
        from transformers import AutoTokenizer

        vocab = len(AutoTokenizer.from_pretrained(tokenizer_model))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        warn(f"could not load tokenizer_model {tokenizer_model}: {type(exc).__name__}: {exc}")
        return

    if vocab != manifest.get("vocab_size"):
        error(
            f"vocab mismatch: config tokenizer has {vocab} tokens but the dataset was "
            f"tokenised with a {manifest.get('vocab_size')}-token vocabulary",
            "rebuild the dataset with --model pointing at this model preset",
        )
    else:
        note(f"tokenizer/vocab agree ({vocab} tokens); expect iteration-1 loss ~{math.log(vocab):.2f}")


def check_schedule(cfg):
    """Learning-rate schedule: perf configs are tuned to not learn."""
    train_iters = get(cfg, "train_iters")
    lr = get(cfg, "lr")
    decay_iters = get(cfg, "lr_decay_iters")
    warmup = get(cfg, "lr_warmup_iters") or 0

    if lr is not None and lr <= 2.0e-5:
        warn(
            f"lr is {lr:g}, which is a throughput-benchmark value; the loss will barely move",
            "use something in the 1e-4 to 3e-4 range for a convergence run",
        )
    if decay_iters is not None and train_iters is not None and decay_iters != train_iters:
        severity = error if decay_iters < train_iters else warn
        severity(
            f"lr_decay_iters ({decay_iters}) != train_iters ({train_iters}); past "
            "lr_decay_iters Megatron pins the lr at min_lr for the rest of the run",
            f"set lr_decay_iters: {train_iters}",
        )
    if train_iters and warmup < max(5, 0.01 * train_iters):
        warn(
            f"lr_warmup_iters is {warmup} for a {train_iters}-iteration run",
            "use roughly 2-10% of train_iters",
        )
    if get(cfg, "finetune"):
        warn("finetune is true, which resets the scheduler and skips optimizer state", "set finetune: false")


def check_moe(cfg):
    """MoE fast paths that are only valid under benchmark routing."""
    if not get(cfg, "num_experts"):
        return
    if get(cfg, "moe_router_force_load_balancing"):
        error(
            "moe_router_force_load_balancing is true -- Megatron replaces the router "
            "logits with random values, so the router never learns",
            "set moe_router_force_load_balancing: false",
        )
    if get(cfg, "use_turbo_deepep"):
        warn(
            "use_turbo_deepep is true; with real (unbalanced) routing this produced Inf "
            "gradients at iteration 3 on MI325X. It is only exercised with "
            "force_load_balancing in the perf configs",
            "set use_turbo_deepep: false, or watch the first ~10 iterations closely",
        )
    if get(cfg, "turbo_sync_free_moe_stage"):
        warn(
            f"turbo_sync_free_moe_stage is {get(cfg, 'turbo_sync_free_moe_stage')}; the fused "
            "router path it enables is not exercised by the benchmark configs",
            "set turbo_sync_free_moe_stage: 0 if you see Inf/NaN gradients",
        )
    if not get(cfg, "moe_aux_loss_coeff"):
        warn(
            "moe_aux_loss_coeff is unset or zero; with real routing the experts can collapse",
            "use 1e-2 for Mixtral-style models",
        )


def check_determinism(cfg):
    """Megatron's deterministic_mode has hard prerequisites."""
    if not get(cfg, "deterministic_mode"):
        if os.environ.get("PRIMUS_DETERMINISTIC") != "1":
            note(
                "run-to-run bit-exactness is off; set PRIMUS_DETERMINISTIC=1 to pin "
                "NCCL_ALGO/rocBLAS atomics/TE algorithms"
            )
        return
    if get(cfg, "use_flash_attn"):
        error(
            "deterministic_mode requires use_flash_attn: false (Megatron asserts this)",
            "set use_flash_attn: false, but expect much higher activation memory",
        )
    if get(cfg, "cross_entropy_loss_fusion"):
        error(
            "deterministic_mode requires cross_entropy_loss_fusion: false",
            "set cross_entropy_loss_fusion: false",
        )
    if os.environ.get("NCCL_ALGO") not in ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"):
        error(
            "deterministic_mode requires NCCL_ALGO to be set to a deterministic algorithm",
            "export PRIMUS_DETERMINISTIC=1 (sets NCCL_ALGO=Ring) before launching",
        )


def check_observability(cfg):
    """You cannot read a loss curve you never logged."""
    if str(get(cfg, "stderr_sink_level", "")).upper() != "DEBUG":
        warn(
            f"stderr_sink_level is {get(cfg, 'stderr_sink_level')}; Primus emits Megatron's "
            "per-iteration loss line at DEBUG, so it will not reach the console",
            "set stderr_sink_level: DEBUG",
        )
    if get(cfg, "disable_tensorboard", True):
        warn(
            "disable_tensorboard is true; no TensorBoard scalars will be written",
            "set disable_tensorboard: false",
        )
    if not get(cfg, "eval_iters") or not get(cfg, "eval_interval"):
        warn(
            "eval_iters/eval_interval are unset or zero; you will get no validation curve",
            "set eval_interval: 25 and eval_iters: 2 or similar",
        )
    if not get(cfg, "save"):
        note("save is null: a crash loses the whole run, but checkpoints for a large MoE are huge")


def check_budget(cfg, manifest):
    """Token budget versus dataset size."""
    train_iters = get(cfg, "train_iters")
    gbs = get(cfg, "global_batch_size")
    seq = get(cfg, "seq_length")
    mbs = get(cfg, "micro_batch_size")
    if not (train_iters and gbs and seq):
        return

    if mbs and gbs % mbs:
        error(f"global_batch_size ({gbs}) is not divisible by micro_batch_size ({mbs})")
    if seq > (get(cfg, "max_position_embeddings") or seq):
        error(
            f"seq_length ({seq}) exceeds max_position_embeddings " f"({get(cfg, 'max_position_embeddings')})"
        )

    total = train_iters * gbs * seq
    note(f"token budget: {train_iters} iters x {gbs} x {seq} = {total/1e6:.0f}M tokens")

    if manifest:
        available = (manifest.get("splits", {}).get("train", {}) or {}).get("tokens")
        if available:
            epochs = total / available
            if epochs > 1.0:
                warn(
                    f"the run consumes {total/1e6:.0f}M tokens but the dataset holds only "
                    f"{available/1e6:.0f}M ({epochs:.1f} epochs, so data repeats)",
                    f"rebuild with --target-tokens {total/1e6*1.1:.0f}e6",
                )
            else:
                note(f"dataset holds {available/1e6:.0f}M tokens ({epochs:.2f} epochs, no repetition)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    from primus.core.launcher.parser import PrimusParser

    # Configs may hold dataset paths relative to the Primus root, which is what
    # the trainer uses as its cwd. Match that so existence checks agree.
    config_path = Path(args.config).resolve()
    os.chdir(PRIMUS_PATH)
    primus_config = PrimusParser().parse(SimpleNamespace(config=str(config_path)))
    cfg = primus_config.get_module_config("pre_trainer")

    manifest = check_data(cfg)
    check_tokenizer(cfg, manifest)
    check_schedule(cfg)
    check_moe(cfg)
    check_determinism(cfg)
    check_observability(cfg)
    check_budget(cfg, manifest)

    print(f"\nconvergence lint: {args.config}\n" + "=" * 72)
    for msg in NOTES:
        print(f"  note  {msg}")
    for msg, fix in WARNINGS:
        print(f"  WARN  {msg}")
        if fix:
            print(f"        -> {fix}")
    for msg, fix in ERRORS:
        print(f"  ERROR {msg}")
        if fix:
            print(f"        -> {fix}")
    print("=" * 72)
    print(f"{len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")

    if ERRORS or (args.strict and WARNINGS):
        sys.exit(1)


if __name__ == "__main__":
    main()
