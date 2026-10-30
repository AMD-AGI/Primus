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


class DiffusionArgBuilder:
    """Build the compact config object consumed by diffusion trainers."""

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
    DEFAULT_WAN_TRAINER: dict[str, Any] = {
        "name": "fsdp2",
        "args": {
            "output_dir": "./output/wan",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "attention_backend": "flash_attn_aiter",
            "learning_rate": 1.0e-5,
            "lr_scheduler_type": "constant",
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "num_train_epochs": 1,
            "max_steps": 100,
            "logging_steps": 1,
            "save_steps": 0,
            "save_total_limit": 3,
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
    DEFAULT_FLUX_DATASET: dict[str, Any] = {
        "name": "flux",
        "config": {
            "dataset_type": "precomputed",
            "dataset_format": "hf_dataset",
            "dataset_path": "/path/to/flux_precomputed_dataset",
            "eval_dataset_path": None,
            "dataset": None,
            "shuffle": True,
            "processor_config": {
                "processor_name": "flux_precomputed",
                "processor_type": "flux_precomputed",
                "prompt_dropout_prob": 0.0,
                "empty_encodings_path": None,
                "img_size": 256,
                "skip_low_resolution": True,
            },
        },
    }
    DEFAULT_FLUX_TRAINER: dict[str, Any] = {
        "name": "fsdp2",
        "args": {
            "output_dir": "./output/flux",
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "gradient_checkpointing_ratio": 1.0,
            "attention_backend": "flash_attn_aiter",
            "learning_rate": 2.0e-4,
            "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": 1600,
            "weight_decay": 0.1,
            "num_train_epochs": 1,
            "max_steps": 100,
            "logging_steps": 1,
            "save_steps": 0,
            "save_total_limit": 3,
            "dataloader_num_workers": 4,
            "report_to": "none",
            "run_name": "flux-schnell-fsdp2",
            "bf16": True,
            "seed": 10007,
            "optim": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "fsdp2_wrap_target": "dit",
            "fsdp_transformer_layer_cls_to_wrap": "DoubleStreamBlock,SingleStreamBlock",
            "fsdp_module_paths_to_wrap": "img_in,time_in,vector_in,txt_in,final_layer",
            "fsdp_module_paths_no_reshard": "final_layer",
            "fsdp2_reshard_after_forward": True,
            "compile_transformer_blocks": True,
            "save_strategy": "dit_only",
            "sp_size": 1,
            "dp_replicate": 1,
            "flow_match_scheduler": {
                "shift": 1,
                "sigma_min": 0.0,
                "extra_one_step": False,
                "num_train_timesteps": 1000,
            },
            "mlperf_enable": False,
            "mlperf_target_eval_loss": 0.586,
            "mlperf_eval_samples": 262144,
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
            raise TypeError(
                f"DiffusionArgBuilder expects dict or SimpleNamespace, got {type(params).__name__}"
            )

    def finalize(self) -> SimpleNamespace:
        params = copy.deepcopy(self._params)
        if "model" not in params:
            raise ValueError("Diffusion backend config requires a model preset.")
        for legacy_section in ("dataset", "trainer"):
            if legacy_section in params:
                raise ValueError(
                    f"Diffusion backend no longer accepts public `{legacy_section}` overrides. "
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

    def _model_family(self, params: dict[str, Any]) -> str:
        model = params.get("model") or {}
        if not isinstance(model, dict):
            raise TypeError(f"Diffusion model preset must be a dict, got {type(model).__name__}")
        name = str(model.get("name", "")).strip().lower()
        if not name:
            raise ValueError("Diffusion model preset requires `model.name`.")
        return name

    def _defaults_for_model(self, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        benchmark = benchmark_defaults(
            model_name,
            flux_dataset=self.DEFAULT_FLUX_DATASET,
            flux_trainer=self.DEFAULT_FLUX_TRAINER,
        )
        if benchmark is not None:
            return benchmark
        if model_name == "wan":
            return self.DEFAULT_DATASET, self.DEFAULT_WAN_TRAINER
        if model_name == "flux" or model_name.startswith("flux."):
            return self.DEFAULT_FLUX_DATASET, self.DEFAULT_FLUX_TRAINER
        raise ValueError(f"Unsupported diffusion model name: {model_name!r}")

    def _normalize_primus_style_sections(self, params: dict[str, Any]) -> dict[str, Any]:
        """Translate concise Primus-style sections into trainer arguments.

        Public configs use high-level sections such as `training`, `data`,
        `parallelism`, and `runtime`; internal defaults supply compact
        `dataset` and `trainer` objects consumed by the diffusion runtime.
        """
        model_name = self._model_family(params)
        default_dataset, default_trainer = self._defaults_for_model(model_name)

        normalized = {
            "model": params["model"],
            "dataset": copy.deepcopy(default_dataset),
            "trainer": copy.deepcopy(default_trainer),
            "stage": params.get("stage", "pretrain"),
            "primus": params.get("primus", {}),
        }

        dataset_cfg = normalized["dataset"]["config"]
        trainer_args = normalized["trainer"]["args"]

        training = params.get("training") or {}
        data = params.get("data") or {}
        parallelism = params.get("parallelism") or {}
        optimizer = params.get("optimizer") or {}
        lr_scheduler = params.get("lr_scheduler") or {}
        runtime = params.get("runtime") or {}
        metrics = params.get("metrics") or {}

        training_map = {
            ("steps",): ("max_steps",),
            ("local_batch_size",): ("per_device_train_batch_size",),
            ("eval_batch_size",): ("per_device_eval_batch_size",),
            ("global_batch_size",): ("global_batch_size",),
            ("gradient_accumulation_steps",): ("gradient_accumulation_steps",),
            ("output_dir",): ("output_dir",),
            ("save_steps",): ("save_steps",),
            ("save_strategy",): ("save_strategy",),
            ("save_total_limit",): ("save_total_limit",),
            ("run_name",): ("run_name",),
            ("num_train_epochs",): ("num_train_epochs",),
            ("dataloader_num_workers",): ("dataloader_num_workers",),
            ("resume_from_checkpoint",): ("resume_from_checkpoint",),
        }
        for source_path, target_path in training_map.items():
            value = self._get_any(training, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)
        if training.get("local_batch_size") is not None and training.get("eval_batch_size") is None:
            self._set_nested(trainer_args, ("per_device_eval_batch_size",), training["local_batch_size"])

        data_map = {
            ("dataset_path",): ("dataset_path",),
            ("eval_dataset_path",): ("eval_dataset_path",),
            ("data_folder",): ("data_folder",),
            ("frame_num",): ("frame_num",),
            ("video_backend",): ("video_backend",),
            ("text_tokenizer",): ("processor_config", "text_tokenizer"),
            ("processor_name",): ("processor_config", "processor_name"),
            ("processor_type",): ("processor_config", "processor_type"),
            ("dataset_format",): ("dataset_format",),
            ("dataset_type",): ("dataset_type",),
            ("dataset",): ("dataset",),
            ("shuffle",): ("shuffle",),
            ("empty_encodings_path",): ("processor_config", "empty_encodings_path"),
            ("prompt_dropout_prob",): ("processor_config", "prompt_dropout_prob"),
            ("img_size",): ("processor_config", "img_size"),
            ("skip_low_resolution",): ("processor_config", "skip_low_resolution"),
        }
        for source_path, target_path in data_map.items():
            value = self._get_any(data, *source_path)
            if value is not None:
                self._set_nested(dataset_cfg, target_path, value)

        height = data.get("height")
        width = data.get("width")
        if height is not None:
            self._set_nested(
                dataset_cfg,
                ("processor_config", "extra_kwargs", "size", "height"),
                height,
            )
        if width is not None:
            self._set_nested(
                dataset_cfg,
                ("processor_config", "extra_kwargs", "size", "width"),
                width,
            )

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

        lr_scheduler_map = {
            ("lr_scheduler_type",): ("lr_scheduler_type",),
            ("warmup_steps",): ("warmup_steps",),
        }
        for source_path, target_path in lr_scheduler_map.items():
            value = self._get_any(lr_scheduler, *source_path)
            if value is not None:
                self._set_nested(trainer_args, target_path, value)

        runtime_map = {
            ("attention_backend",): ("attention_backend",),
            ("report_to",): ("report_to",),
            ("seed",): ("seed",),
            ("bf16",): ("bf16",),
            ("fp16",): ("fp16",),
            ("gradient_checkpointing",): ("gradient_checkpointing",),
            ("gradient_checkpointing_ratio",): ("gradient_checkpointing_ratio",),
            ("compile_transformer_blocks",): ("compile_transformer_blocks",),
            ("fsdp2_reshard_after_forward",): ("fsdp2_reshard_after_forward",),
        }
        for source_path, target_path in runtime_map.items():
            value = self._get_any(runtime, *source_path)
            if value is not None:
                if target_path in {
                    ("bf16",),
                    ("fp16",),
                    ("gradient_checkpointing",),
                    ("compile_transformer_blocks",),
                    ("fsdp2_reshard_after_forward",),
                }:
                    value = self._coerce_bool(value)
                self._set_nested(trainer_args, target_path, value)

        log_freq = metrics.get("log_freq")
        if log_freq is not None:
            self._set_nested(trainer_args, ("logging_steps",), log_freq)

        enable_wandb = metrics.get("enable_wandb")
        if enable_wandb is False and runtime.get("report_to") is None:
            self._set_nested(trainer_args, ("report_to",), "none")
        elif enable_wandb is True and runtime.get("report_to") is None:
            self._set_nested(trainer_args, ("report_to",), "wandb")

        mlperf = params.get("mlperf") or {}
        mlperf_map = {
            ("enable",): ("mlperf_enable",),
            ("target_eval_loss",): ("mlperf_target_eval_loss",),
            ("eval_samples",): ("mlperf_eval_samples",),
            ("eval_steps",): ("mlperf_eval_steps",),
            ("train_samples",): ("mlperf_train_samples",),
            ("eval_total_samples",): ("mlperf_eval_total_samples",),
            ("output_file",): ("mlperf_output_file",),
        }
        for source_path, target_path in mlperf_map.items():
            value = self._get_any(mlperf, *source_path)
            if value is not None:
                if target_path == ("mlperf_enable",):
                    value = self._coerce_bool(value)
                self._set_nested(trainer_args, target_path, value)

        checkpoint = params.get("checkpoint") or {}
        resume_from_checkpoint = checkpoint.get("resume_from_checkpoint")
        if resume_from_checkpoint is not None:
            self._set_nested(trainer_args, ("resume_from_checkpoint",), resume_from_checkpoint)

        if (model_name == "flux" or model_name.startswith("flux.")) and int(
            trainer_args.get("sp_size", 1)
        ) != 1:
            raise ValueError("FLUX diffusion training currently requires `parallelism.sp_size: 1`.")

        apply_benchmark_sections(
            model_name,
            data=data,
            trainer_args=trainer_args,
            dataset_cfg=dataset_cfg,
            normalized=normalized,
        )
        return normalized


# Backwards-compatible import path for existing Wan-only callers.
WanArgBuilder = DiffusionArgBuilder


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
        # ``constant`` ignores warmup. A non-zero value never changes the LR.
        "warmup_steps": 0,
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

# Registered on later benchmark PRs. Reject them here so this release does not
# build trainer defaults, or download checkpoints, for a model it cannot run.
_UNBUILT_BENCHMARKS = {
    "sdxl",
    "stable-diffusion-xl",
    "mochi-1",
    "hunyuan-video",
    "wan2.1-i2v",
}


def _positive_image_size(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"flux-dev image size must be a positive integer, got {value!r}") from exc
    if number <= 0 or number != int(number):
        raise ValueError(f"flux-dev image size must be a positive integer, got {value!r}")
    return int(number)


def _square_image_size(resolution: Any) -> int:
    """Parse a square benchmark resolution.

    Accepts ``512``, ``512.0``, ``"512"``, ``"512x512"``, ``"512,512"`` and
    ``[512, 512]``. A non-square pair is rejected instead of keeping one side.
    """
    if isinstance(resolution, bool) or resolution is None:
        raise ValueError(f"flux-dev resolution must be a positive size, got {resolution!r}")
    if isinstance(resolution, (int, float)):
        return _positive_image_size(resolution)
    if isinstance(resolution, (list, tuple)):
        parts = list(resolution)
    else:
        text = str(resolution).strip().lower().replace(" ", "")
        if not text:
            raise ValueError("flux-dev resolution is empty")
        parts = [text]
        for separator in ("x", "*", ","):
            if separator in text:
                parts = [part for part in text.split(separator) if part]
                break
    if len(parts) == 1:
        return _positive_image_size(parts[0])
    if len(parts) == 2:
        height = _positive_image_size(parts[0])
        width = _positive_image_size(parts[1])
        if height != width:
            raise ValueError(
                f"flux-dev resolution must be square, got {resolution!r}. Set data.img_size to one side length."
            )
        return height
    raise ValueError(f"flux-dev resolution must be a size or HxW pair, got {resolution!r}")


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
    if model_name in _UNBUILT_BENCHMARKS:
        raise ValueError(
            f"{model_name} has no diffusion model builder. "
            "This release registers the flux-dev benchmark only, so the config is rejected "
            "before any checkpoint or dataset download."
        )
    return None


def apply_benchmark_sections(
    model_name: str,
    *,
    data: dict[str, Any],
    trainer_args: dict[str, Any],
    dataset_cfg: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    if model_name == "flux-dev":
        # Precomputed FLUX samples are mean/logvar encodings, not AMDB images.
        # Leaving the dataset on the flux loader avoids the latent-cache path
        # looking up a missing ``image`` key.
        resolution = data.get("resolution")
        model_cfg = normalized["model"].get("config")
        if resolution is None and isinstance(model_cfg, dict):
            resolution = model_cfg.get("resolution")
        # An explicit data.img_size is already copied onto the processor.
        # The model YAML always sets resolution, so it must not replace that size.
        if data.get("img_size") is None and resolution is not None:
            processor = dataset_cfg.setdefault("processor_config", {})
            processor["img_size"] = _square_image_size(resolution)
        if int(trainer_args.get("sp_size", 1)) != 1:
            raise ValueError("flux-dev diffusion training requires `parallelism.sp_size: 1`.")
