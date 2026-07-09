###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import importlib.util
from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.yaml_utils import nested_namespace_to_dict


class DiffusionPretrainTrainer(BaseTrainer):
    """Primus lifecycle wrapper for Wan diffusion training."""

    def __init__(self, backend_args: Any):
        super().__init__(backend_args=backend_args)
        self.wan_trainer = None

    @staticmethod
    def _as_dict(value: Any) -> dict:
        if isinstance(value, dict):
            return value
        return nested_namespace_to_dict(value)

    def setup(self):
        trainer_cfg = self._as_dict(self.backend_args.trainer)
        dataset_cfg = self._as_dict(self.backend_args.dataset)
        trainer_args = trainer_cfg.get("args", {})
        attention_backend = trainer_args.get("attention_backend")

        missing = [
            package
            for package in (
                "torch",
                "loguru",
                "safetensors",
                "transformers",
                "PIL",
                "torchvision",
                "requests",
                "packaging",
            )
            if importlib.util.find_spec(package) is None
        ]
        video_backend = (dataset_cfg.get("config", {}) or {}).get("video_backend")
        if video_backend == "imageio" and importlib.util.find_spec("imageio") is None:
            missing.append("imageio")
        if video_backend == "decord" and importlib.util.find_spec("decord") is None:
            missing.append("decord")
        if missing:
            raise RuntimeError(
                "Diffusion backend missing required Python packages: "
                f"{', '.join(missing)}. Install the Wan diffusion training extras first."
            )

        if attention_backend:
            from primus.backends.diffusion.attention import set_attention_backend

            set_attention_backend(attention_backend)
            log_rank_0(f"[Primus:Diffusion] attention_backend={attention_backend}")

        if attention_backend == "flash_attn_aiter":
            from primus.backends.diffusion.attention.aiter import (
                AITER_FLASH_ATTN_AVAILABLE,
            )

            if not AITER_FLASH_ATTN_AVAILABLE:
                raise RuntimeError(
                    "attention_backend=flash_attn_aiter was requested, but AITER flash attention "
                    "is unavailable in this environment."
                )

    def init(self):
        from primus.backends.diffusion.registry import (
            get_dataset_builder,
            get_model_builder,
            get_trainer_builder,
        )

        model_cfg = self._as_dict(self.backend_args.model)
        dataset_cfg = self._as_dict(self.backend_args.dataset)
        trainer_cfg = self._as_dict(self.backend_args.trainer)

        model_name = model_cfg["name"]
        dataset_name = dataset_cfg["name"]
        trainer_name = trainer_cfg["name"]

        model_config = model_cfg["config"]
        dataset_config = dataset_cfg["config"]
        trainer_args = trainer_cfg["args"]

        log_rank_0(
            f"[Primus:Diffusion] Building model={model_name}, dataset={dataset_name}, trainer={trainer_name}"
        )
        model = get_model_builder(model_name)(model_config)
        dataset, processor = get_dataset_builder(dataset_name)(dataset_config)
        self.wan_trainer = get_trainer_builder(trainer_name)(
            model=model,
            dataset=dataset,
            processor=processor,
            trainer_args=trainer_args,
        )

    def train(self):
        if self.wan_trainer is None:
            raise RuntimeError("DiffusionPretrainTrainer.init() must be called before train().")

        self.wan_trainer.train()
        self.wan_trainer.save_model()

    def cleanup(self, on_error: bool = False):
        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.finish(exit_code=1 if on_error else 0)
        except Exception as exc:
            log_rank_0(f"[Primus:Diffusion] wandb cleanup failed: {exc}")

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            log_rank_0(f"[Primus:Diffusion] distributed cleanup failed: {exc}")
