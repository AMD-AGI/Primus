###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Launcher-neutral construction and execution for diffusion training."""

from __future__ import annotations

import gc
import importlib.util
from typing import Any

from primus.backends.diffusion.utils.log import logger


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {key: _as_value(item) for key, item in vars(value).items()}


def _as_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_as_value(item) for item in value)
    if hasattr(value, "__dict__"):
        return _as_dict(value)
    return value


def prepare_runtime(config: Any) -> None:
    """Validate dependencies and select the configured attention backend."""
    trainer_cfg = _as_dict(config.trainer)
    dataset_cfg = _as_dict(config.dataset)
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
    dataset_config = dataset_cfg.get("config", {}) or {}
    video_backend = dataset_config.get("video_backend")
    if dataset_cfg.get("name") == "flux":
        for package in ("datasets", "huggingface_hub", "sentencepiece"):
            if importlib.util.find_spec(package) is None:
                missing.append(package)
        if (
            dataset_config.get("dataset_type") == "raw"
            and dataset_config.get("dataset_format") == "webdataset"
            and importlib.util.find_spec("webdataset") is None
        ):
            missing.append("webdataset")
    if video_backend == "imageio" and importlib.util.find_spec("imageio") is None:
        missing.append("imageio")
    if video_backend == "decord" and importlib.util.find_spec("decord") is None:
        missing.append("decord")
    if missing:
        raise RuntimeError(
            "Diffusion backend missing required Python packages: "
            f"{', '.join(missing)}. Install the diffusion training extras first."
        )

    if attention_backend:
        from primus.backends.diffusion.attention import set_attention_backend

        set_attention_backend(attention_backend)
        logger.info(f"Diffusion attention_backend={attention_backend}")

    if attention_backend == "flash_attn_aiter":
        from primus.backends.diffusion.attention.aiter import AITER_FLASH_ATTN_AVAILABLE

        if not AITER_FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "attention_backend=flash_attn_aiter was requested, but AITER flash attention "
                "is unavailable in this environment."
            )


def build_trainer(config: Any):
    """Build the model, datasets, and backend trainer from canonical config."""
    from primus.backends.diffusion.registry import (
        get_dataset_builder,
        get_model_builder,
        get_trainer_builder,
    )

    model_cfg = _as_dict(config.model)
    dataset_cfg = _as_dict(config.dataset)
    trainer_cfg = _as_dict(config.trainer)
    trainer_args = trainer_cfg["args"]

    seed = trainer_args.get("seed")
    if seed is not None:
        from primus.backends.diffusion.utils.train_utils import set_seed

        set_seed(int(seed))

    logger.info(
        f"Building model={model_cfg['name']}, dataset={dataset_cfg['name']}, "
        f"trainer={trainer_cfg['name']}"
    )
    model = get_model_builder(model_cfg["name"])(model_cfg["config"])
    dataset_parts = get_dataset_builder(dataset_cfg["name"])(dataset_cfg["config"])
    if len(dataset_parts) == 2:
        dataset, processor = dataset_parts
        eval_dataset = eval_processor = None
    elif len(dataset_parts) == 4:
        dataset, processor, eval_dataset, eval_processor = dataset_parts
    else:
        raise ValueError(f"Dataset builder returned {len(dataset_parts)} values; expected 2 or 4.")

    return get_trainer_builder(trainer_cfg["name"])(
        model=model,
        dataset=dataset,
        processor=processor,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
        trainer_args=trainer_args,
    )


def cleanup(on_error: bool = False) -> None:
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=1 if on_error else 0)
    except Exception as exc:
        logger.warning(f"Diffusion wandb cleanup failed: {exc}")

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        logger.warning(f"Diffusion distributed cleanup failed: {exc}")


def run_training(config: Any) -> None:
    """Run diffusion directly, without the Primus runtime lifecycle."""
    try:
        prepare_runtime(config)
        trainer = build_trainer(config)
        gc.disable()
        try:
            trainer.train()
            trainer.save_model()
        finally:
            gc.enable()
    except BaseException:
        cleanup(on_error=True)
        raise
    cleanup()
