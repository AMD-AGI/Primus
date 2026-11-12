###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus lifecycle wrapper for vision backend training."""

from __future__ import annotations

import os
from typing import Any

import torch.distributed as dist

from primus.backends.vision.registry import get_dataset_builder, get_model_builder, get_trainer_builder
from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.yaml_utils import nested_namespace_to_dict


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return nested_namespace_to_dict(value)


def ensure_distributed(local_rank: int) -> None:
    """Join the torchrun process group before data samplers are built."""
    import torch

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized() and (world > 1 or "RANK" in os.environ):
            dist.init_process_group(backend="nccl")
        return
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")


class VisionPretrainTrainer(BaseTrainer):
    """Build a registered vision model and run the Primus training loop."""

    def __init__(self, backend_args: Any, *args, **kwargs):
        super().__init__(backend_args=backend_args, *args, **kwargs)
        self.vision_trainer = None

    def setup(self):
        trainer_args = _as_dict(self.backend_args.trainer).get("args", {})
        backend = str(trainer_args.get("attention_backend") or "sdpa").strip().lower()
        if backend in {"flash_attn_aiter", "aiter"}:
            backend = "aiter"
        os.environ["VISION_ATTN_BACKEND"] = backend
        os.environ["WORLD_MIRROR_ATTN_BACKEND"] = backend
        log_rank_0(f"[Primus:Vision] attention_backend={backend}")

    def init(self):
        import torch

        model_cfg = _as_dict(self.backend_args.model)
        dataset_cfg = _as_dict(self.backend_args.dataset)
        trainer_cfg = _as_dict(self.backend_args.trainer)
        trainer_args = trainer_cfg["args"]
        seed = trainer_args.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        ensure_distributed(self.local_rank)
        model_name = model_cfg["name"]
        dataset_name = dataset_cfg["name"]
        trainer_name = trainer_cfg["name"]
        log_rank_0(
            f"[Primus:Vision] Building model={model_name}, dataset={dataset_name}, trainer={trainer_name}"
        )
        model = get_model_builder(model_name)(model_cfg["config"])
        dataset, processor = get_dataset_builder(dataset_name)(dataset_cfg["config"])
        self.vision_trainer = get_trainer_builder(trainer_name)(
            model=model,
            dataset=dataset,
            processor=processor,
            trainer_args=trainer_args,
        )

    def train(self):
        if self.vision_trainer is None:
            raise RuntimeError("VisionPretrainTrainer.init() must be called before train().")
        self.vision_trainer.train()

    def cleanup(self, on_error: bool = False):
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            log_rank_0(f"[Primus:Vision] distributed cleanup failed: {exc}")
