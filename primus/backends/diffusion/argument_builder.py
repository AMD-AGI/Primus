###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from primus.core.utils.yaml_utils import nested_namespace_to_dict


class WanArgBuilder:
    """Build the compact config object consumed by the Wan trainer."""

    DEFAULT_DATASET: dict[str, Any] = {
        "name": "wan",
        "config": {
            "dataset_type": "vision",
            "dataset_format": "jsonl",
            "dataset_path": "/path/to/meta.jsonl",
            "data_folder": "/path/to/videos",
            "video_sampling_strategy": "frame_num",
            "frame_num": 81,
            "shuffle": True,
            "video_backend": "imageio",
            "processor_config": {
                "processor_name": "wanvideo",
                "processor_type": "wanvideo",
                "max_text_length": 512,
                "text_tokenizer": "/path/to/umt5-xxl",
                "extra_kwargs": {
                    "do_resize": True,
                    "size": {
                        "height": 480,
                        "width": 832,
                    },
                    "do_normalize": True,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                },
            },
        },
    }
    DEFAULT_TRAINER: dict[str, Any] = {
        "name": "fsdp2",
        "args": {
            "output_dir": "./output/wan",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "attention_backend": "sdpa",
            "learning_rate": 1.0e-5,
            "lr_scheduler_type": "constant",
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "num_train_epochs": 1,
            "max_steps": 100,
            "logging_steps": 1,
            "save_steps": 0,
            "dataloader_num_workers": 4,
            "report_to": "none",
            "run_name": "wan-fsdp2",
            "bf16": True,
            "seed": 10007,
            "optim": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "fsdp2_wrap_target": "dit",
            "fsdp_transformer_layer_cls_to_wrap": "DiTBlock",
            "fsdp2_reshard_after_forward": True,
            "save_strategy": "dit_only",
            "sp_size": 1,
            "dp_replicate": 1,
            "flow_match_scheduler": {
                "shift": 5,
                "sigma_min": 0.0,
                "extra_one_step": True,
                "num_train_timesteps": 1000,
            },
        },
    }

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def update(self, params: Any) -> None:
        if isinstance(params, SimpleNamespace):
            self._params = nested_namespace_to_dict(params)
        elif isinstance(params, dict):
            self._params = copy.deepcopy(params)
        else:
            raise TypeError(f"WanArgBuilder expects dict or SimpleNamespace, got {type(params).__name__}")

    def finalize(self) -> SimpleNamespace:
        params = copy.deepcopy(self._params)
        if "model" not in params:
            raise ValueError("Wan backend config requires a model preset.")
        for legacy_section in ("dataset", "trainer"):
            if legacy_section in params:
                raise ValueError(
                    f"Wan backend no longer accepts public `{legacy_section}` overrides. "
                    "Use Primus-style `data`, `training`, `parallelism`, `optimizer`, "
                    "`runtime`, and `metrics` sections instead."
                )

        params = self._normalize_primus_style_sections(params)

        return SimpleNamespace(
            model=params["model"],
            dataset=params["dataset"],
            trainer=params["trainer"],
            stage=params.get("stage", "pretrain"),
            primus=params.get("primus", {}),
        )

    @staticmethod
    def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        cursor = target
        for key in path[:-1]:
            next_value = cursor.get(key)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[key] = next_value
            cursor = next_value
        cursor[path[-1]] = value

    @staticmethod
    def _get_any(source: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in source:
                return source[key]
        return None

    def _normalize_primus_style_sections(self, params: dict[str, Any]) -> dict[str, Any]:
        """Translate concise Primus-style Wan sections into trainer arguments.

        Public configs use high-level sections such as `training`, `data`,
        `parallelism`, and `runtime`; internal defaults supply the compact
        `dataset` and `trainer` objects consumed by the Wan runtime.
        """

        normalized = {
            "model": params["model"],
            "dataset": copy.deepcopy(self.DEFAULT_DATASET),
            "trainer": copy.deepcopy(self.DEFAULT_TRAINER),
            "stage": params.get("stage", "pretrain"),
            "primus": params.get("primus", {}),
        }

        dataset_cfg = normalized["dataset"]["config"]
        trainer_args = normalized["trainer"]["args"]

        training = params.get("training") or {}
        data = params.get("data") or {}
        parallelism = params.get("parallelism") or {}
        optimizer = params.get("optimizer") or {}
        runtime = params.get("runtime") or {}
        metrics = params.get("metrics") or {}

        training_map = {
            ("steps",): ("max_steps",),
            ("local_batch_size",): ("per_device_train_batch_size",),
            ("global_batch_size",): ("global_batch_size",),
            ("gradient_accumulation_steps",): ("gradient_accumulation_steps",),
            ("output_dir",): ("output_dir",),
            ("save_steps",): ("save_steps",),
            ("run_name",): ("run_name",),
            ("num_train_epochs",): ("num_train_epochs",),
            ("resume_from_checkpoint",): ("resume_from_checkpoint",),
        }
        for source_path, target_path in training_map.items():
            value = self._get_any(training, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)

        data_map = {
            ("dataset_path",): ("dataset_path",),
            ("data_folder",): ("data_folder",),
            ("frame_num",): ("frame_num",),
            ("video_backend",): ("video_backend",),
            ("text_tokenizer",): ("processor_config", "text_tokenizer"),
            ("processor_name",): ("processor_config", "processor_name"),
            ("processor_type",): ("processor_config", "processor_type"),
        }
        for source_path, target_path in data_map.items():
            value = self._get_any(data, *source_path)
            if value is not None:
                self._set_nested(dataset_cfg, target_path, value)

        height = data.get("height")
        width = data.get("width")
        if height is not None:
            self._set_nested(dataset_cfg, ("processor_config", "extra_kwargs", "size", "height"), height)
        if width is not None:
            self._set_nested(dataset_cfg, ("processor_config", "extra_kwargs", "size", "width"), width)

        parallelism_map = {
            ("sp_size",): ("sp_size",),
            ("dp_replicate",): ("dp_replicate",),
        }
        for source_path, target_path in parallelism_map.items():
            value = self._get_any(parallelism, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)

        optimizer_map = {
            ("lr",): ("learning_rate",),
            ("learning_rate",): ("learning_rate",),
            ("weight_decay",): ("weight_decay",),
            ("adam_beta1",): ("adam_beta1",),
            ("adam_beta2",): ("adam_beta2",),
            ("adam_epsilon",): ("adam_epsilon",),
            ("max_grad_norm",): ("max_grad_norm",),
        }
        for source_path, target_path in optimizer_map.items():
            value = self._get_any(optimizer, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)

        runtime_map = {
            ("attention_backend",): ("attention_backend",),
            ("report_to",): ("report_to",),
            ("seed",): ("seed",),
        }
        for source_path, target_path in runtime_map.items():
            value = self._get_any(runtime, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)

        log_freq = metrics.get("log_freq")
        if log_freq is not None:
            self._set_nested(trainer_args, ("logging_steps",), log_freq)

        enable_wandb = metrics.get("enable_wandb")
        if enable_wandb is False and runtime.get("report_to") is None:
            self._set_nested(trainer_args, ("report_to",), "none")
        elif enable_wandb is True and runtime.get("report_to") is None:
            self._set_nested(trainer_args, ("report_to",), "wandb")

        checkpoint = params.get("checkpoint") or {}
        resume_from_checkpoint = checkpoint.get("resume_from_checkpoint")
        if resume_from_checkpoint is not None:
            self._set_nested(trainer_args, ("resume_from_checkpoint",), resume_from_checkpoint)

        return normalized
