###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Argument defaults for the AMDiffusionBenchmark workloads."""

from __future__ import annotations

import copy
from typing import Any

BENCHMARK_TRAINER: dict[str, Any] = {
    "name": "amdb_benchmark",
    "args": {
        "output_dir": "./output/amdb",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "gradient_checkpointing_ratio": 1.0,
        "attention_backend": "flash_attn_aiter",
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "constant",
        "warmup_steps": 10,
        "weight_decay": 0.0,
        "num_train_epochs": 1,
        "max_steps": 100,
        "logging_steps": 1,
        "save_steps": 0,
        "save_total_limit": 3,
        "dataloader_num_workers": 0,
        # AMDiffusionBenchmark uses Accelerator(even_batches=True): a short tail
        # is padded back to train_batch_size, and TFLOPS divides the first
        # step's flop count by each step's GPU time.
        "dataloader_even_batches": True,
        "report_to": "none",
        "run_name": "amdb",
        "bf16": True,
        "seed": 42,
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 0.0,
        "fsdp2_wrap_target": "dit",
        "fsdp_transformer_layer_cls_to_wrap": "",
        "fsdp_module_paths_to_wrap": "",
        "fsdp2_reshard_after_forward": True,
        "fsdp2_pure_low_precision": True,
        "compile_transformer_blocks": False,
        "save_strategy": "dit_only",
        "sp_size": 1,
        "dp_replicate": 1,
    },
}

_SPECS: dict[str, dict[str, Any]] = {
    "sdxl": {
        "modality": "image",
        "fsdp_layers": "BasicTransformerBlock",
        "learning_rate": 1.0e-4,
        "lr_scheduler_type": "polynomial",
        "warmup_steps": 100,
        "gradient_checkpointing": False,
    },
    "mochi-1": {
        "modality": "video",
        "fsdp_layers": "AsymmetricJointBlock",
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "constant",
        "warmup_steps": 10,
        "gradient_checkpointing": False,
    },
    "hunyuan-video": {
        "modality": "video",
        "fsdp_layers": "MMDoubleStreamBlock,MMSingleStreamBlock",
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "constant",
        "warmup_steps": 10,
        "gradient_checkpointing": False,
    },
    "wan2.1-i2v": {
        "modality": "video",
        "fsdp_layers": "DiTBlock",
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "constant",
        "warmup_steps": 10,
        "gradient_checkpointing": True,
    },
}


def _spec(model_name: str) -> dict[str, Any] | None:
    if model_name == "stable-diffusion-xl":
        model_name = "sdxl"
    return _SPECS.get(model_name)


def benchmark_defaults(
    model_name: str,
    *,
    flux_dataset: dict[str, Any],
    flux_trainer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if model_name == "flux-dev":
        dataset = copy.deepcopy(flux_dataset)
        dataset["config"]["dataset_type"] = "raw"
        dataset["config"]["dataset_format"] = "hf_repo"
        dataset["config"]["dataset_path"] = ""
        processor = dataset["config"]["processor_config"]
        processor["img_size"] = 512
        processor["processor_name"] = "flux_raw"
        processor["processor_type"] = "flux_raw"
        processor["skip_low_resolution"] = False
        dataset["config"]["benchmark_loader"] = True
        trainer = copy.deepcopy(BENCHMARK_TRAINER)
        trainer["args"]["run_name"] = "flux-dev"
        trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] = "DoubleStreamBlock,SingleStreamBlock"
        trainer["args"]["fsdp_module_paths_to_wrap"] = flux_trainer["args"]["fsdp_module_paths_to_wrap"]
        trainer["args"]["fsdp_module_paths_no_reshard"] = flux_trainer["args"]["fsdp_module_paths_no_reshard"]
        return dataset, trainer

    spec = _spec(model_name)
    if spec is None:
        return None
    dataset = {
        "name": "amdb",
        "config": {
            "dataset_path": "",
            "modality": spec["modality"],
            "max_train_samples": -1,
            "split": "train",
        },
    }
    trainer = copy.deepcopy(BENCHMARK_TRAINER)
    trainer["args"]["learning_rate"] = spec["learning_rate"]
    trainer["args"]["lr_scheduler_type"] = spec["lr_scheduler_type"]
    trainer["args"]["warmup_steps"] = spec["warmup_steps"]
    trainer["args"]["gradient_checkpointing"] = spec["gradient_checkpointing"]
    trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] = spec["fsdp_layers"]
    trainer["args"]["run_name"] = model_name
    return dataset, trainer


def apply_benchmark_sections(
    model_name: str,
    *,
    data: dict[str, Any],
    trainer_args: dict[str, Any],
    dataset_cfg: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    if model_name == "flux-dev":
        if str(data.get("dataset_type", "")).lower() == "precomputed":
            normalized["dataset"]["name"] = "amdb"
            dataset_cfg["modality"] = "image"
        resolution = data.get("resolution")
        model_cfg = normalized["model"].get("config")
        if resolution is None and isinstance(model_cfg, dict):
            resolution = model_cfg.get("resolution")
        if resolution is not None:
            size = int(str(resolution).split(",")[0])
            processor = dataset_cfg.setdefault("processor_config", {})
            processor["img_size"] = size
        if int(trainer_args.get("sp_size", 1)) != 1:
            raise ValueError("flux-dev diffusion training requires `parallelism.sp_size: 1`.")
        return

    spec = _spec(model_name)
    if spec is None:
        return
    if int(trainer_args.get("sp_size", 1)) != 1:
        raise ValueError(f"{model_name} diffusion training requires `parallelism.sp_size: 1`.")
    dataset_cfg["modality"] = data.get("modality") or spec["modality"]
    for key in ("max_train_samples", "revision", "split"):
        if data.get(key) is not None:
            dataset_cfg[key] = data[key]
    model_cfg = normalized["model"].setdefault("config", {})
    if isinstance(model_cfg, dict):
        for key in ("resolution", "num_frames", "center_crop", "shift"):
            if data.get(key) is not None:
                model_cfg[key] = data[key]
