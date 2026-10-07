###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generate a convergence config from a Primus example config.

The examples are throughput benchmarks: mock or synthetic data, a learning rate
that barely moves the loss, and for MoE models routing that is forced to be
balanced. They do carry what ships for a model -- architecture, parallelism,
precision, kernels -- so the generated file ``extends:`` the example and
overrides only what a convergence run needs: real data, a schedule that learns,
evaluation, logging, a fixed global batch, and the MoE settings that are only
valid under benchmark routing. Each group of overrides says why in the file.

    python3 tools/convergence_test/make_config.py \
        --example examples/megatron/configs/MI325X/qwen3_8B-BF16-pretrain.yaml

Writes output/convergence/configs/<backend>/<model>-<precision>-convergence.yaml
and prints its path. A generated config is a starting point: probe it before a
long run (run_convergence_test.sh --config <file> --probe 20).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolve_config as rc  # noqa: E402
from resolve_config import PRIMUS_PATH, as_number  # noqa: E402

GLOBAL_BATCH = 128
SEQ_LENGTH = 4096
DEFAULT_ITERS = 1000


def model_size_b(name):
    """Parameter count in billions from a model name: llama2_7B -> 7, qwen3_30B_A3B -> 30."""
    sizes = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)[Bb](?![a-zA-Z])", name)]
    return max(sizes) if sizes else None


def peak_lr(size_b):
    """Peak learning rate for a ~0.5M-token batch; smaller models take a higher one."""
    if size_b is None:
        return 1.5e-4
    if size_b <= 3:
        return 3.0e-4
    if size_b <= 20:
        return 1.5e-4
    return 1.0e-4


def _relative_to_primus(path):
    path = Path(path).resolve()
    try:
        return "./" + str(path.relative_to(PRIMUS_PATH))
    except ValueError:
        return str(path)


def _schedule_comment(iters, lr, size_b):
    size = f"~{size_b:g}B" if size_b else "unknown-size"
    return (
        f"run length and schedule: {iters} iterations x {GLOBAL_BATCH} x {SEQ_LENGTH} tokens; "
        f"cosine {lr:g} -> {lr / 10:g} after 10% warmup (peak lr for a {size} model)"
    )


def megatron_overrides(cfg, iters, data_dir, size_b):
    get = cfg.get
    notes = []
    lr = peak_lr(size_b)
    world = cfg.num_devices
    model_parallel = 1
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        model_parallel *= int(as_number(get(key), 1) or 1)
    dp = max(1, world // model_parallel)
    moe = int(as_number(get("num_experts"), 0) or 0) > 0

    mbs = int(as_number(get("micro_batch_size"), 1) or 1)
    if moe and mbs > 1:
        notes.append(
            f"MoE: micro_batch_size {mbs} -> 1. The example is tuned for forced-balanced routing; "
            "real routing sends uneven token counts to experts. Raise it if a probe shows headroom."
        )
        mbs = 1
    while mbs > 1 and GLOBAL_BATCH % (mbs * dp):
        mbs -= 1

    seq = SEQ_LENGTH
    max_positions = int(as_number(get("max_position_embeddings"), 0) or 0)
    if max_positions and max_positions < SEQ_LENGTH:
        seq = max_positions
        notes.append(f"seq_length {seq}: the model's max_position_embeddings is below {SEQ_LENGTH}")

    groups = [
        (
            "logging: Primus emits the per-iteration line at DEBUG",
            {
                "disable_tensorboard": False,
                "stderr_sink_level": "DEBUG",
                "log_interval": 10,
                "log_throughput": True,
                "profile": False,
                "check_for_nan_in_loss_and_grad": True,
            },
        ),
        (
            _schedule_comment(iters, lr, size_b),
            {
                "train_iters": iters,
                "global_batch_size": GLOBAL_BATCH,
                "micro_batch_size": mbs,
                "seq_length": seq,
                "lr": lr,
                "min_lr": lr / 10,
                "lr_warmup_iters": max(1, iters // 10),
                "lr_decay_iters": iters,
                "lr_decay_style": "cosine",
                "weight_decay": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "clip_grad": 1.0,
                "eod_mask_loss": True,
            },
        ),
    ]

    init_std = as_number(get("init_method_std"), 0.02)
    hidden = int(as_number(get("hidden_size"), 0) or 0)
    if init_std > 0.01 and hidden >= 4096:
        groups.append(
            (
                "init: random logits of std init_method_std x sqrt(hidden) would start the loss well above ln(vocab)",
                {"init_method_std": 0.008},
            )
        )

    data = data_dir
    groups.append(
        (
            "real data, built by prepare_dataset.py; the driver points PRIMUS_CONVERGENCE_DATA at it",
            {
                "mock_data": False,
                "tokenizer_type": get("tokenizer_type") or "HuggingFaceTokenizer",
                "tokenizer_model": f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/tokenizer",
                "train_data_path": [f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/train_text_document"],
                "valid_data_path": [f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/valid_text_document"],
                "test_data_path": [f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/valid_text_document"],
                "data_cache_path": f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/cache",
                "split": None,
                "dataloader_type": "single",
                "num_workers": 8,
            },
        )
    )
    groups.append(
        (
            f"validation every {max(1, iters // 10)} iterations over 10 batches",
            {"eval_interval": max(1, iters // 10), "eval_iters": 10},
        )
    )
    groups.append(
        (
            "no checkpoints",
            {"finetune": False, "load": None, "save": None, "disable_last_saving": True},
        )
    )

    if moe:
        moe_fix = {}
        if get("moe_router_force_load_balancing"):
            moe_fix["moe_router_force_load_balancing"] = False
        moe_fix["moe_router_dtype"] = "fp32"
        if get("use_turbo_deepep"):
            moe_fix["use_turbo_deepep"] = False
        if as_number(get("turbo_sync_free_moe_stage"), 0):
            moe_fix["turbo_sync_free_moe_stage"] = 0
        if get("moe_token_dispatcher_type") == "flex":
            moe_fix["moe_token_dispatcher_type"] = "alltoall"
        if not as_number(get("moe_aux_loss_coeff"), 0) and not get("moe_router_enable_expert_bias"):
            moe_fix["moe_aux_loss_coeff"] = 0.01
        if get("moe_router_load_balancing_type") == "none" and not get("moe_router_enable_expert_bias"):
            moe_fix["moe_router_load_balancing_type"] = "aux_loss"
        if not get("recompute_granularity"):
            moe_fix["recompute_activations"] = True
            moe_fix["recompute_granularity"] = "selective"
        groups.append(
            (
                "MoE: the example forces balanced routing (random router logits) and pairs it with fast "
                "paths that produced Inf gradients under real routing on Mixtral; learn the router, "
                "balance it with an auxiliary loss",
                moe_fix,
            )
        )
    return groups, {}, notes


def maxtext_overrides(cfg, iters, data_dir, size_b, run_name):
    get = cfg.get
    notes = []
    lr = peak_lr(size_b)
    per_device = GLOBAL_BATCH // max(1, cfg.num_devices)
    example_pdbs = as_number(get("per_device_batch_size"), per_device) or per_device
    pdbs = per_device
    while pdbs > 1 and (pdbs > example_pdbs or per_device % pdbs):
        pdbs -= 1
    accumulation = per_device // pdbs
    if accumulation > 1:
        notes.append(
            f"per_device_batch_size {pdbs} x gradient_accumulation_steps {accumulation}: the example "
            f"fits only {example_pdbs:g} sequences per device"
        )

    data = data_dir
    groups = [
        (
            "logging and output",
            {
                "run_name": run_name,
                "base_output_directory": "./output",
                "enable_tensorboard": True,
                "profiler": "",
            },
        ),
        (
            _schedule_comment(iters, lr, size_b),
            {
                "steps": iters,
                "per_device_batch_size": pdbs,
                "gradient_accumulation_steps": accumulation,
                "max_target_length": SEQ_LENGTH,
                "learning_rate": lr,
                "lr_schedule_type": "cosine",
                "learning_rate_final_fraction": 0.1,
                "warmup_steps_fraction": 0.1,
                "learning_rate_schedule_steps": -1,
                "adam_b1": 0.9,
                "adam_b2": 0.95,
                "adam_eps": 1.0e-8,
                "adam_weight_decay": 0.1,
                "gradient_clipping_threshold": 1.0,
            },
        ),
        (
            "fp32 master weights and Adam moments (bf16 rounds late-run updates away); compute stays bf16",
            {"weight_dtype": "float32", "dtype": "bfloat16"},
        ),
        (
            "real data: pre-tokenized parquet from prepare_dataset.py --format maxtext",
            {
                "dataset_type": "grain",
                "grain_file_type": "parquet",
                "grain_train_files": f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/train-*.parquet",
                "grain_eval_files": f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/valid-*.parquet",
                "train_data_columns": ["tokens"],
                "eval_data_columns": ["tokens"],
                "tokenize_train_data": False,
                "tokenize_eval_data": False,
                "tokenizer_type": "huggingface",
                "tokenizer_path": f"${{PRIMUS_CONVERGENCE_DATA:{data}}}/tokenizer",
                "packing": True,
                "grain_worker_count": 4,
                "enable_data_shuffling": True,
                "data_shuffle_seed": 0,
                "init_weights_seed": 0,
            },
        ),
        (
            f"validation every {max(1, iters // 10)} steps over 10 batches",
            {"eval_interval": max(1, iters // 10), "eval_steps": 10},
        ),
        ("no checkpoints", {"enable_checkpointing": False, "async_checkpointing": False}),
    ]

    if int(as_number(get("num_experts"), 1) or 1) > 1:
        moe_fix = {}
        if not as_number(get("load_balance_loss_weight"), 0):
            moe_fix["load_balance_loss_weight"] = 0.01
        capacity = as_number(get("capacity_factor"), -1)
        if 0 < capacity < 1.25:
            moe_fix["capacity_factor"] = 1.25
        if moe_fix:
            groups.append(
                (
                    "MoE: MaxText's default balancing loss is 0, so the router can collapse; "
                    "drop fewer tokens than the benchmark's capacity factor",
                    moe_fix,
                )
            )

    # MaxText applies its model file after the overlay; route any key the model
    # file sets to something else through override_model, which beats it.
    model_cfg = yaml.safe_load(cfg.model_file.read_text()) if cfg.model_file else {}
    override_model = {}
    for _, values in groups:
        for key in list(values):
            if key in (model_cfg or {}) and model_cfg[key] != values[key]:
                override_model[key] = values.pop(key)
    return groups, override_model, notes


def _yaml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        # YAML 1.1 reads an exponent without a dot (1e-08) as a string.
        if "e" in text and "." not in text.split("e")[0]:
            mantissa, exponent = text.split("e")
            text = f"{mantissa}.0e{exponent}"
        return text
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def render(example, out, exp_name, groups, override_model, notes):
    extends = _extends_path(example, out)
    lines = [
        f"# Generated by tools/convergence_test/make_config.py from",
        f"#   {_relative_to_primus(example)}",
        "# The example supplies the model, parallelism, precision and kernels; the",
        "# overrides below are what a convergence run changes, each with its reason.",
        "# Probe before a long run:",
        f"#   tools/convergence_test/run_convergence_test.sh --config {_relative_to_primus(out)} --probe 20",
    ]
    for note in notes:
        lines.append(f"# NOTE: {note}")
    lines += [
        f"extends: {_yaml_value(extends)}",
        "",
        f'exp_name: "${{PRIMUS_EXP_NAME:{exp_name}}}"',
        'workspace: "${PRIMUS_WORKSPACE:./output}"',
        "",
        "modules:",
        "  pre_trainer:",
        "    overrides:",
    ]
    for comment, values in groups:
        if not values:
            continue
        lines.append(f"      # {comment}")
        lines += [f"      {key}: {_yaml_value(value)}" for key, value in values.items()]
        lines.append("")
    if override_model:
        lines.append("      # keys MaxText's model file also sets: only override_model beats it")
        lines.append("      override_model_config: true")
        lines.append("      override_model:")
        lines += [f"        {key}: {_yaml_value(value)}" for key, value in override_model.items()]
    return "\n".join(lines).rstrip() + "\n"


def _extends_path(example, out):
    """Relative inside the checkout, which the container mounts at the same path."""
    example, out = Path(example).resolve(), Path(out).resolve()
    try:
        example.relative_to(PRIMUS_PATH)
        out.relative_to(PRIMUS_PATH)
    except ValueError:
        return str(example)
    return str(
        Path(*[".."] * len(out.parent.relative_to(PRIMUS_PATH).parts), example.relative_to(PRIMUS_PATH))
    )


def config_stem(example):
    """examples/.../qwen3_8B-BF16-pretrain.yaml -> qwen3_8B-BF16"""
    return re.sub(r"-pretrain$", "", Path(example).stem)


def generate(example, out=None, iters=DEFAULT_ITERS, source="fineweb-edu", tokenizer=None):
    """Write a convergence config for ``example`` and return what was decided."""
    # Both before rc.load(), which moves to the Primus root.
    example = Path(example).resolve()
    out = Path(out).resolve() if out else None
    cfg = rc.load(example)
    if cfg.framework not in rc.FRAMEWORKS:
        raise SystemExit(f"{example}: framework {cfg.framework!r} is not supported")
    stem = config_stem(example)
    suffix = "-maxtext-convergence" if cfg.framework == "maxtext" else "-convergence"
    exp_name = f"{stem}{suffix}"
    if out is None:
        out = PRIMUS_PATH / "output" / "convergence" / "configs" / cfg.framework / f"{stem}-convergence.yaml"

    tokenizer_name = tokenizer or rc.tokenizer_from_preset(cfg.model_preset, cfg.framework)[1]
    seq = SEQ_LENGTH
    data_dir = _relative_to_primus(rc.default_dataset_dir(tokenizer_name, cfg.framework, seq, source))
    size_b = model_size_b(Path(cfg.model_preset).stem or stem)

    if cfg.framework == "megatron":
        groups, override_model, notes = megatron_overrides(cfg, iters, data_dir, size_b)
    else:
        groups, override_model, notes = maxtext_overrides(cfg, iters, data_dir, size_b, exp_name)
    if size_b is None:
        notes.append("the model size is not in its name; using the 7B-class peak learning rate 1.5e-4")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(example, out, exp_name, groups, override_model, notes))

    check = rc.load(out)
    problems = []
    if check.global_batch_size != GLOBAL_BATCH:
        problems.append(f"global batch is {check.global_batch_size}, expected {GLOBAL_BATCH}")
    if check.train_iters != iters:
        problems.append(f"iterations resolve to {check.train_iters}, expected {iters}")
    problems += [f"{k} is discarded by the MaxText model file" for k in check.shadowed]
    if problems:
        raise SystemExit(f"generated {out}, but it does not resolve as intended: " + "; ".join(problems))
    return {
        "path": str(out),
        "example": str(example),
        "backend": cfg.framework,
        "model_preset": cfg.model_preset,
        "exp_name": exp_name,
        "iterations": iters,
        "global_batch_size": check.global_batch_size,
        "seq_length": check.seq_length,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--example", required=True, help="A Primus example config to start from")
    parser.add_argument("--out", help="Where to write (default: output/convergence/configs/<backend>/)")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Run length in iterations")
    parser.add_argument("--source", default="fineweb-edu", help="Corpus, for the default dataset path")
    parser.add_argument("--tokenizer", help="Tokenizer substituted for the model preset's")
    parser.add_argument("--json", action="store_true", help="Print the decisions as JSON")
    args = parser.parse_args()

    info = generate(args.example, args.out, args.iters, args.source, args.tokenizer)
    if args.json:
        print(json.dumps(info, indent=2))
        return
    print(info["path"])
    for note in info["notes"]:
        print(f"  note: {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
