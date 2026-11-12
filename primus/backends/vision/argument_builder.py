###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Translate Primus vision YAML sections into model, dataset, and trainer args."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from primus.core.utils.yaml_utils import nested_namespace_to_dict

_TRAINER_ARGS: dict[str, Any] = {
    "output_dir": "./output/vision",
    "max_steps": 100,
    "num_train_epochs": 1000,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1.0e-4,
    "weight_decay": 0.05,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 0,
    "eta_min_ratio": 0.1,
    "attention_backend": "aiter",
    "seed": 42,
    "bf16": True,
    "gradient_checkpointing": False,
    "compile_transformer_blocks": False,
    "logging_steps": 1,
    "save_steps": 0,
    "save_total_limit": 3,
    "parallelism_mode": "ddp",
    "fsdp2_reshard_after_forward": True,
    "profile": False,
    "resume_from_checkpoint": None,
    "run_name": "vision",
    "dataloader_num_workers": 4,
}

_DATASETS: dict[str, dict[str, Any]] = {
    "vggt": {
        "name": "vggt",
        "config": {
            "dataset_path": "/shared_nfs/3dwm/hywm-3dwm-sub20/co3d",
            "annotation_dir": "",
            "max_images_per_gpu": 24,
            "num_workers": 4,
            "img_size": 518,
            "patch_size": 14,
            "image_num_range": [2, 24],
            "aspect_ratios": [0.33, 1.0],
        },
    },
    "worldmirror": {
        "name": "worldmirror",
        "config": {
            "hypersim_dir": "/shared_nfs/3dwm/hywm-3dwm-sub20/hypersim",
            "max_images_per_gpu": 24,
            "min_num_of_views": 2,
            "num_workers": 4,
            "aspect_ratios": [0.5, 2.0],
            "num_pixels_range": [100000, 250000],
            "train_datasets": None,
        },
    },
    "synthetic": {
        "name": "synthetic",
        "config": {
            "length": 8,
            "views": 2,
            "image_size": 32,
            "batch_size": 1,
            "num_workers": 0,
        },
    },
}

_BOOL_TRAINER_KEYS = {
    "bf16",
    "gradient_checkpointing",
    "compile_transformer_blocks",
    "profile",
    "fsdp2_reshard_after_forward",
}


def _trainer(learning_rate: float, run_name: str) -> dict[str, Any]:
    args = copy.deepcopy(_TRAINER_ARGS)
    args["learning_rate"] = learning_rate
    args["run_name"] = run_name
    args["output_dir"] = f"./output/{run_name}"
    return {"name": "fsdp2", "args": args}


class VisionArgBuilder:
    """Build the compact config object consumed by the vision trainer."""

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def update(self, params: Any) -> None:
        if isinstance(params, SimpleNamespace):
            self._params = nested_namespace_to_dict(params)
        elif isinstance(params, dict):
            self._params = copy.deepcopy(params)
        else:
            raise TypeError(f"VisionArgBuilder expects dict or SimpleNamespace, got {type(params).__name__}")

    def finalize(self) -> SimpleNamespace:
        params = copy.deepcopy(self._params)
        if "model" not in params:
            raise ValueError("Vision backend config requires a model preset.")
        for legacy_section in ("dataset", "trainer"):
            if legacy_section in params:
                raise ValueError(
                    f"Vision backend does not accept a public `{legacy_section}` section. "
                    "Use `data`, `training`, `parallelism`, `optimizer`, `lr_scheduler`, "
                    "`runtime`, and `metrics`."
                )
        normalized = self._normalize(params)
        return SimpleNamespace(
            model=normalized["model"],
            dataset=normalized["dataset"],
            trainer=normalized["trainer"],
            stage=normalized.get("stage", "pretrain"),
        )

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model") or {}
        if not isinstance(model, dict):
            raise TypeError(f"Vision model preset must be a dict, got {type(model).__name__}")
        model_name = str(model.get("name", "")).strip().lower()
        if model_name not in _DATASETS:
            known = ", ".join(sorted(_DATASETS))
            raise ValueError(f"Unsupported vision model name: {model_name!r}. Known models: {known}.")

        learning_rate = {"vggt": 5.0e-5, "worldmirror": 1.0e-4, "synthetic": 1.0e-3}[model_name]
        normalized = {
            "model": model,
            "dataset": copy.deepcopy(_DATASETS[model_name]),
            "trainer": _trainer(learning_rate, model_name),
            "stage": params.get("stage", "pretrain"),
        }
        dataset_cfg = normalized["dataset"]["config"]
        trainer_args = normalized["trainer"]["args"]
        model_cfg = model.get("config") or {}
        self._require_stage2_checkpoint(model_name, model_cfg)
        self._apply_sections(params, dataset_cfg, trainer_args, model_name)
        return normalized

    @staticmethod
    def _require_stage2_checkpoint(model_name: str, model_cfg: dict[str, Any]) -> None:
        if model_name != "worldmirror":
            return
        enabled = model_cfg.get("enable_gs", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
        pretrained = str(model_cfg.get("pretrained") or "").strip()
        if enabled and not pretrained:
            raise ValueError(
                "WorldMirror stage 2 (model.config.enable_gs) requires model.config.pretrained, "
                "the stage-1 checkpoint. Load weights only."
            )

    def _apply_sections(
        self,
        params: dict[str, Any],
        dataset_cfg: dict[str, Any],
        trainer_args: dict[str, Any],
        model_name: str,
    ) -> None:
        training = params.get("training") or {}
        data = params.get("data") or {}
        parallelism = params.get("parallelism") or {}
        optimizer = params.get("optimizer") or {}
        lr_scheduler = params.get("lr_scheduler") or {}
        runtime = params.get("runtime") or {}
        metrics = params.get("metrics") or {}
        checkpoint = params.get("checkpoint") or {}

        training_map = {
            "steps": "max_steps",
            "max_steps": "max_steps",
            "num_train_epochs": "num_train_epochs",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "output_dir": "output_dir",
            "save_steps": "save_steps",
            "save_total_limit": "save_total_limit",
            "run_name": "run_name",
            "resume_from_checkpoint": "resume_from_checkpoint",
        }
        for source, target in training_map.items():
            if training.get(source) is not None:
                trainer_args[target] = training[source]
        if model_name == "synthetic" and training.get("local_batch_size") is not None:
            dataset_cfg["batch_size"] = training["local_batch_size"]

        for key in (
            "dataset_path",
            "annotation_dir",
            "hypersim_dir",
            "max_images_per_gpu",
            "min_num_of_views",
            "num_workers",
            "img_size",
            "patch_size",
            "image_num_range",
            "aspect_ratios",
            "num_pixels_range",
            "train_datasets",
            "length",
            "views",
            "image_size",
            "batch_size",
        ):
            if data.get(key) is not None:
                dataset_cfg[key] = data[key]
        if data.get("num_workers") is not None:
            trainer_args["dataloader_num_workers"] = data["num_workers"]

        if parallelism.get("mode") is not None:
            trainer_args["parallelism_mode"] = str(parallelism["mode"]).strip().lower()
        if parallelism.get("fsdp2_reshard_after_forward") is not None:
            trainer_args["fsdp2_reshard_after_forward"] = self._coerce_bool(
                parallelism["fsdp2_reshard_after_forward"]
            )

        optimizer_map = {
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "adam_beta1": "adam_beta1",
            "adam_beta2": "adam_beta2",
            "adam_epsilon": "adam_epsilon",
            "max_grad_norm": "max_grad_norm",
        }
        for source, target in optimizer_map.items():
            if optimizer.get(source) is not None:
                trainer_args[target] = optimizer[source]

        for source, target in (
            ("lr_scheduler_type", "lr_scheduler_type"),
            ("warmup_steps", "warmup_steps"),
            ("eta_min_ratio", "eta_min_ratio"),
        ):
            if lr_scheduler.get(source) is not None:
                trainer_args[target] = lr_scheduler[source]

        runtime_map = {
            "attention_backend": "attention_backend",
            "seed": "seed",
            "bf16": "bf16",
            "gradient_checkpointing": "gradient_checkpointing",
            "compile_transformer_blocks": "compile_transformer_blocks",
            "profile": "profile",
        }
        for source, target in runtime_map.items():
            if runtime.get(source) is None:
                continue
            value = runtime[source]
            if target in _BOOL_TRAINER_KEYS:
                value = self._coerce_bool(value)
            trainer_args[target] = value

        if metrics.get("log_freq") is not None:
            trainer_args["logging_steps"] = metrics["log_freq"]
        if checkpoint.get("resume_from_checkpoint") is not None:
            trainer_args["resume_from_checkpoint"] = checkpoint["resume_from_checkpoint"]

        mode = str(trainer_args.get("parallelism_mode", "ddp")).strip().lower()
        if mode not in {"ddp", "fsdp2"}:
            raise ValueError(f"parallelism.mode must be 'ddp' or 'fsdp2', got {mode!r}.")
        trainer_args["parallelism_mode"] = mode
