###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Lint a Primus config for settings that silently ruin a convergence run.

Most Primus example configs are throughput benchmarks. They are tuned to make
the step timer look good, and several of those choices make the loss curve
meaningless rather than merely slow. The worst offenders do not fail loudly:

  Megatron
  * ``mock_data: true``                      trains on random tokens
  * ``moe_router_force_load_balancing: true`` feeds the router *random* logits
  * ``lr: 1e-5`` with a 2-iteration warmup    barely moves the loss
  * ``lr_decay_iters < train_iters``          pins the tail of the run at min_lr

  MaxText
  * ``dataset_type: synthetic``               trains on random tokens
  * an override of a key the MaxText model file also sets is discarded
  * ``weight_dtype: bfloat16``                rounds late-run updates away
  * ``load_balance_loss_weight: 0``           lets the MoE router collapse
  * token ids >= ``vocab_size``               do not raise in JAX

This resolves the config the same way the trainer does (module preset + model
preset + overrides, and for MaxText its own base.yml and model file) so
inherited defaults are checked too, then reports errors and warnings. Exit
status is non-zero if any error is found.

Usage:
    python3 tools/convergence_test/check_config.py --config <exp.yaml> [--strict] [--probe] [--key value ...]

Trailing ``--key value`` pairs are applied like primus-cli overrides, so the lint
sees the run the driver will actually launch.
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolve_config  # noqa: E402
from resolve_config import as_number  # noqa: E402

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
    return cfg.get(key, default)


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def load_tokenizer(path):
    """Load a tokenizer that is on disk; never reach for the Hub from a lint."""
    if not path or not Path(path).is_dir():
        return None, f"{path} is not a local tokenizer directory"
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path), None
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Megatron
# ---------------------------------------------------------------------------


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


def check_schedule(cfg, probe=False):
    """Learning-rate schedule: perf configs are tuned to not learn."""
    train_iters = int(as_number(get(cfg, "train_iters"), 0))
    lr = as_number(get(cfg, "lr"))
    decay_iters = as_number(get(cfg, "lr_decay_iters"))
    warmup = as_number(get(cfg, "lr_warmup_iters"), 0)

    if lr is not None and lr <= 2.0e-5:
        warn(
            f"lr is {lr:g}, which is a throughput-benchmark value; the loss will barely move",
            "use something in the 1e-4 to 3e-4 range for a convergence run",
        )
    if decay_iters is not None and train_iters and decay_iters != train_iters:
        if probe:
            note(f"probe: the first {train_iters} iterations of a {decay_iters:g}-iteration schedule")
        else:
            severity = error if decay_iters < train_iters else warn
            severity(
                f"lr_decay_iters ({decay_iters:g}) != train_iters ({train_iters}); past "
                "lr_decay_iters Megatron pins the lr at min_lr for the rest of the run",
                f"set lr_decay_iters: {train_iters}",
            )
    horizon = decay_iters if decay_iters is not None else train_iters
    if horizon and warmup >= horizon:
        error(
            f"lr_warmup_iters ({warmup:g}) is not below the {horizon:g}-iteration decay horizon; "
            "Megatron asserts warmup < decay at start-up",
            f"use a warmup of roughly 2-10% of {horizon:g} iterations",
        )
    elif train_iters and warmup < max(5, 0.01 * train_iters):
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


def check_observability(cfg, probe=False):
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
    if not probe and (
        not as_number(get(cfg, "eval_iters"), 0) or not as_number(get(cfg, "eval_interval"), 0)
    ):
        warn(
            "eval_iters/eval_interval are unset or zero; you will get no validation curve",
            "set eval_interval: 25 and eval_iters: 2 or similar",
        )
    if not get(cfg, "save"):
        note("save is null: nothing is checkpointed, so a crash loses the whole run")


def check_budget(cfg, manifest):
    """Token budget versus dataset size."""
    train_iters = int(as_number(get(cfg, "train_iters"), 0))
    gbs = int(as_number(get(cfg, "global_batch_size"), 0))
    seq = int(as_number(get(cfg, "seq_length"), 0))
    mbs = int(as_number(get(cfg, "micro_batch_size"), 0))
    if not (train_iters and gbs and seq):
        return

    model_parallel = 1
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        model_parallel *= int(as_number(get(cfg, key), 1) or 1)
    dp = cfg.num_devices // model_parallel if cfg.num_devices % model_parallel == 0 else 0
    if mbs and dp and gbs % (mbs * dp):
        error(
            f"global_batch_size ({gbs}) is not a multiple of micro_batch_size x data-parallel size "
            f"({mbs} x {dp}); Megatron refuses to start",
            f"use a global_batch_size divisible by {mbs * dp}",
        )
    max_positions = int(as_number(get(cfg, "max_position_embeddings"), seq))
    if seq > max_positions:
        error(f"seq_length ({seq}) exceeds max_position_embeddings ({max_positions})")

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


def run_megatron_checks(cfg, probe=False):
    manifest = check_data(cfg)
    check_tokenizer(cfg, manifest)
    check_schedule(cfg, probe)
    check_moe(cfg)
    check_determinism(cfg)
    check_observability(cfg, probe)
    check_budget(cfg, manifest)


# ---------------------------------------------------------------------------
# MaxText
# ---------------------------------------------------------------------------


def check_maxtext_resolution(cfg):
    """Overrides that MaxText will quietly throw away."""
    if cfg.maxtext_configs is None:
        warn(
            "could not find MaxText's configs/base.yml (third_party/maxtext or $MAXTEXT_PATH); "
            "only the Primus overlay was checked, not MaxText's defaults"
        )
        return
    model_file = cfg.model_file.name if cfg.model_file else "the model file"
    for key, (ours, theirs) in sorted(cfg.shadowed.items()):
        if key == "logical_axis_rules":  # merged rule by rule, not replaced
            continue
        error(
            f"{key}: {ours!r} is discarded -- MaxText applies {model_file} ({key}: {theirs!r}) "
            "after the Primus overrides",
            f"move it under override_model: (or pass --override_model.{key} {ours})",
        )


def maxtext_files(pattern):
    if not pattern:
        return []
    return sorted(glob.glob(str(Path(pattern).expanduser())))


def check_maxtext_data(cfg):
    """Data plumbing. Returns the dataset manifest when there is one to read."""
    dataset_type = get(cfg, "dataset_type")
    if dataset_type == "synthetic":
        error(
            "dataset_type is synthetic -- the model trains on random tokens",
            "use dataset_type: grain with a dataset from prepare_dataset.py --format maxtext",
        )
        return None
    if not get(cfg, "packing", True):
        warn(
            "packing is false: every row is padded to max_target_length, so much of each "
            "batch is padding and the loss is averaged over far fewer tokens than a packed run",
            "set packing: true unless padding is what you are testing",
        )
    if dataset_type == "hf":
        warn(
            "dataset_type is hf: the corpus streams from the Hub for the whole run and is "
            "tokenised inside the container, so a network hiccup kills the run and a "
            "transformers upgrade in the image changes the data. Its tokenizer also "
            "truncates every document at max_target_length and discards the rest",
            "build a pre-tokenized dataset with prepare_dataset.py --format maxtext",
        )
        return None
    if dataset_type != "grain":
        warn(f"dataset_type is {dataset_type}; only grain datasets can be verified by this lint")
        return None

    file_type = get(cfg, "grain_file_type")
    train_pattern = get(cfg, "grain_train_files", "")
    eval_pattern = get(cfg, "grain_eval_files", "")
    if any(sep in str(train_pattern) for sep in (";", ",")) or get(cfg, "grain_train_mixture_config_path"):
        note("grain_train_files is a mixture; the lint cannot verify its contents")
        return None
    train_files = maxtext_files(train_pattern)
    if not train_files:
        error(
            f"grain_train_files matches no files: {train_pattern!r}",
            "build them with tools/convergence_test/prepare_dataset.py --format maxtext",
        )
        return None
    eval_interval = as_number(get(cfg, "eval_interval"), -1)
    if eval_interval > 0 and not maxtext_files(eval_pattern):
        error(f"eval_interval is {eval_interval:g} but grain_eval_files matches no files: {eval_pattern!r}")

    nnodes = int(os.environ.get("NNODES", "1"))
    workers = int(as_number(get(cfg, "grain_worker_count"), 1))
    if file_type == "parquet" and workers > len(train_files) // max(1, nnodes):
        error(
            f"grain_worker_count ({workers}) exceeds the {len(train_files)} parquet files split "
            f"over {nnodes} node(s); MaxText refuses to start",
            "lower grain_worker_count or rebuild with more --num-shards",
        )

    columns = as_list(get(cfg, "train_data_columns"))
    manifest_path = Path(train_files[0]).parent / "dataset_info.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if get(cfg, "tokenize_train_data", True):
        if get(cfg, "use_truncation", True):
            warn(
                "tokenize_train_data with use_truncation: true cuts every document at "
                "max_target_length and discards the rest",
                "use a pre-tokenized dataset, or set use_truncation: false",
            )
        if manifest and manifest.get("format") == "maxtext":
            error(
                "the dataset is pre-tokenized but tokenize_train_data is true",
                "set tokenize_train_data: false and tokenize_eval_data: false",
            )
        return manifest

    if manifest is None:
        warn(
            f"no dataset_info.json next to {train_files[0]}; cannot verify the "
            "dataset and the config agree on a vocabulary"
        )
        return None
    if manifest.get("format") != "maxtext":
        error(f"{manifest_path.parent} is a {manifest.get('format', 'megatron')} dataset, not a MaxText one")
        return None
    if eval_interval > 0 and get(cfg, "tokenize_eval_data", True):
        error("the dataset is pre-tokenized but tokenize_eval_data is true", "set tokenize_eval_data: false")
    if columns != [manifest.get("column", "tokens")]:
        error(
            f"train_data_columns is {columns} but the dataset column is {manifest.get('column')!r}",
            f'set train_data_columns: ["{manifest.get("column")}"] (and eval_data_columns)',
        )
    seq = int(as_number(get(cfg, "max_target_length"), 0))
    rows = manifest.get("seq_length")
    if rows and seq < rows:
        error(
            f"the dataset holds rows of up to {rows} tokens but max_target_length is {seq}; "
            "grain refuses rows longer than max_target_length",
            f"rebuild with prepare_dataset.py --format maxtext --seq-length {seq}",
        )
    elif rows and seq > rows:
        note(f"dataset rows are cut at {rows} tokens, shorter than max_target_length {seq}")
    return manifest


def check_maxtext_tokenizer(cfg, manifest):
    """Vocabulary: out-of-range ids do not fail in JAX, they corrupt quietly."""
    vocab_size = int(as_number(get(cfg, "vocab_size"), 0))
    path = get(cfg, "tokenizer_path")
    if manifest and manifest.get("tokenizer_dir") and path:
        if Path(path).resolve() != Path(manifest["tokenizer_dir"]).resolve():
            warn(
                f"tokenizer_path ({path}) is not the tokenizer the dataset was built with "
                f"({manifest['tokenizer_dir']})",
                f"set tokenizer_path: {manifest['tokenizer_dir']}",
            )

    tokenizer, problem = load_tokenizer(path)
    if tokenizer is None:
        warn(f"could not load tokenizer_path: {problem}")
        return
    vocab = len(tokenizer)
    if manifest and manifest.get("vocab_size") not in (None, vocab):
        error(
            f"vocab mismatch: tokenizer_path has {vocab} tokens but the dataset was "
            f"tokenised with a {manifest['vocab_size']}-token vocabulary",
            "rebuild the dataset with --model pointing at this model preset",
        )
    if vocab_size and vocab > vocab_size:
        error(
            f"the tokenizer has {vocab} tokens but the model's vocab_size is {vocab_size}; "
            "JAX does not bounds-check embedding lookups, so the extra ids train on garbage",
            "use the tokenizer that matches model_name, or raise vocab_size via override_model",
        )
    elif vocab_size:
        note(
            f"tokenizer ({vocab}) fits vocab_size ({vocab_size}); expect step-1 loss ~{math.log(vocab_size):.2f}"
        )

    pad = next((i for i in (tokenizer.pad_token_id, tokenizer.unk_token_id) if i is not None), None)
    if pad is None:
        warn(
            f"the tokenizer has no pad or unk token, so MaxText pads with id 0 "
            f"({tokenizer.convert_ids_to_tokens(0)!r}) and drops every target equal to it from the loss",
            "build the dataset with prepare_dataset.py --format maxtext, which assigns a reserved pad token",
        )
    elif manifest and manifest.get("pad_id") not in (None, pad):
        warn(
            f"MaxText will pad with id {pad}, but the dataset was verified against pad id {manifest['pad_id']}"
        )


def check_maxtext_schedule(cfg, probe=False):
    """Learning-rate schedule and optimizer precision."""
    steps = int(as_number(get(cfg, "steps"), 0))
    lr = as_number(get(cfg, "learning_rate"))
    schedule_steps = int(as_number(get(cfg, "learning_rate_schedule_steps"), -1))
    warmup = as_number(get(cfg, "warmup_steps_fraction"), 0.0)

    if lr is not None and lr <= 3.0e-5:
        warn(
            f"learning_rate is {lr:g} (MaxText's base.yml default is 3e-5); the loss will barely move",
            "use something in the 1e-4 to 3e-4 range for a convergence run",
        )
    horizon = schedule_steps if schedule_steps > 0 else steps
    if schedule_steps > 0 and steps:
        if schedule_steps < steps:
            error(
                f"learning_rate_schedule_steps ({schedule_steps}) < steps ({steps}); MaxText "
                "sets the learning rate to 0 after the schedule ends",
                "set learning_rate_schedule_steps: -1 to follow steps",
            )
        elif probe:
            note(f"probe: the first {steps} steps of a {schedule_steps}-step schedule")
        elif schedule_steps > steps:
            warn(
                f"learning_rate_schedule_steps ({schedule_steps}) > steps ({steps}); the run "
                "stops before the learning rate has decayed",
                "set learning_rate_schedule_steps: -1 unless this is a prefix of a longer run",
            )
    if horizon and warmup * horizon < max(5, 0.01 * horizon):
        warn(
            f"warmup_steps_fraction {warmup:g} gives {int(warmup * horizon)} warmup steps",
            "use roughly 0.02-0.1",
        )
    if str(get(cfg, "weight_dtype", "float32")) != "float32":
        warn(
            f"weight_dtype is {get(cfg, 'weight_dtype')}: master weights and Adam moments are "
            "kept in that precision, so late-run updates are rounded away",
            "set weight_dtype: float32 (compute stays in dtype)",
        )
    if get(cfg, "quantization"):
        note(f"quantization is {get(cfg, 'quantization')!r}; compare against a bf16 run of the same recipe")


def check_maxtext_moe(cfg):
    if int(as_number(get(cfg, "num_experts"), 1)) <= 1:
        return
    if not as_number(get(cfg, "load_balance_loss_weight"), 0):
        warn(
            "load_balance_loss_weight is 0 (MaxText's default); with real routing the experts can collapse",
            "use 0.01 for Mixtral-style models",
        )
    capacity = as_number(get(cfg, "capacity_factor"), -1)
    if capacity > 0:
        severity = warn if capacity < 1 else note
        severity(
            f"capacity_factor is {capacity:g}: tokens beyond that multiple of an expert's fair share are dropped"
        )


def check_maxtext_observability(cfg):
    eval_interval = as_number(get(cfg, "eval_interval"), -1)
    if eval_interval <= 0:
        warn(
            "eval_interval is not positive; you will get no validation curve",
            "set eval_interval: 100 or similar",
        )
    elif as_number(get(cfg, "eval_steps"), -1) <= 0:
        note("eval_steps is unset, so every evaluation reads the whole validation split")
    if not get(cfg, "enable_tensorboard", True):
        warn(
            "enable_tensorboard is false; learning rate and grad norm are only written to TensorBoard",
            "set enable_tensorboard: true",
        )
    if get(cfg, "metrics_file") and eval_interval > 0:
        warn(
            "metrics_file is truncated at every evaluation (MaxText writes the running eval "
            "metrics with step 0), so it will not hold the full run",
            "read the log or TensorBoard instead; plot_loss.py does",
        )
    if get(cfg, "profiler"):
        warn(
            f"profiler is {get(cfg, 'profiler')!r}; profiled steps are slower and their "
            "metrics can be hidden",
            'set profiler: ""',
        )
    if not get(cfg, "enable_checkpointing"):
        note("enable_checkpointing is false: a crash loses the whole run")
    deterministic = os.environ.get("PRIMUS_DETERMINISTIC") == "1"
    if not deterministic and os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") != "0":
        note(
            "TE may pick non-deterministic attention kernels, so runs are not bit-exact; "
            "PRIMUS_DETERMINISTIC=1 pins them and NCCL_ALGO"
        )


def check_maxtext_budget(cfg, manifest):
    steps = cfg.train_iters
    gbs = cfg.global_batch_size
    seq = cfg.seq_length
    if not (steps and gbs and seq):
        return
    pdbs = as_number(get(cfg, "per_device_batch_size"), 0)
    note(
        f"token budget: {steps} steps x {gbs} ({pdbs:g}/device x {cfg.num_devices} GPUs) x {seq} "
        f"= {steps*gbs*seq/1e6:.0f}M token slots (padding included)"
    )
    if not manifest:
        return
    splits = manifest.get("splits", {})
    available = (splits.get("train") or {}).get("tokens")
    need = steps * gbs * seq
    if available:
        epochs = int(as_number(get(cfg, "num_epoch"), 1))
        if available * epochs < need:
            severity = error if available * epochs < 0.9 * need else warn
            severity(
                f"the run has {need/1e6:.0f}M token slots but the dataset holds {available/1e6:.0f}M "
                f"x {epochs} epoch(s); MaxText stops when the data runs out",
                f"rebuild with --target-tokens {need/1e6*1.1:.0f}e6",
            )
        else:
            note(f"dataset holds {available/1e6:.0f}M tokens ({need/available:.2f} epochs at most)")
    valid = (splits.get("valid") or {}).get("tokens")
    eval_steps = int(as_number(get(cfg, "eval_steps"), -1))
    if valid and eval_steps > 0 and eval_steps * gbs * seq > valid:
        note(
            f"eval_steps {eval_steps} asks for {eval_steps*gbs*seq/1e6:.1f}M tokens but the validation "
            f"split holds {valid/1e6:.1f}M, so each evaluation stops early"
        )


def run_maxtext_checks(cfg, probe=False):
    check_maxtext_resolution(cfg)
    manifest = check_maxtext_data(cfg)
    check_maxtext_tokenizer(cfg, manifest)
    check_maxtext_schedule(cfg, probe)
    check_maxtext_moe(cfg)
    check_maxtext_observability(cfg)
    check_maxtext_budget(cfg, manifest)


def main():
    # Trailing arguments are training overrides; never let one abbreviate ours.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="The run is a throughput probe: the first iterations of a longer schedule, without evaluation",
    )
    args, overrides = parser.parse_known_args()

    cfg = resolve_config.load(args.config, overrides)
    if cfg.framework == "megatron":
        run_megatron_checks(cfg, args.probe)
    elif cfg.framework == "maxtext":
        run_maxtext_checks(cfg, args.probe)
    else:
        error(f"framework {cfg.framework!r} is not supported by the convergence lint")

    print(f"\nconvergence lint ({cfg.framework}): {args.config}\n" + "=" * 72)
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
