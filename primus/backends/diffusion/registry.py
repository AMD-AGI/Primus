###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Explicit factory registry for diffusion model, dataset, and trainer builders."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def _build_wan_model(model_config: dict):
    from primus.backends.diffusion.models.registrations.wan import build_wan_model

    return build_wan_model(model_config)


def _build_wan_dataset(dataset_config: dict):
    from primus.backends.diffusion.data.registrations.wan import build_wan_dataset

    return build_wan_dataset(dataset_config)


def _build_flux_model(model_config: dict):
    from primus.backends.diffusion.models.registrations.flux import build_flux_model

    return build_flux_model(model_config)


def _build_flux_preset_model(preset: str):
    def _builder(model_config: dict):
        from primus.backends.diffusion.models.registrations.flux import build_flux_model

        config = dict(model_config)
        config["model_preset"] = preset
        return build_flux_model(config)

    return _builder


def _build_flux_dataset(dataset_config: dict):
    from primus.backends.diffusion.data.registrations.flux import build_flux_dataset

    return build_flux_dataset(dataset_config)


def _build_fsdp2_trainer(
    *, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None
):
    from primus.backends.diffusion.trainers.fsdp2 import build_fsdp2_trainer

    return build_fsdp2_trainer(
        model=model,
        dataset=dataset,
        processor=processor,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
        trainer_args=trainer_args,
    )


def _build_sdxl(model_config: dict):
    from primus.backends.diffusion.models.registrations.sdxl import build_sdxl_model

    return build_sdxl_model(model_config)


def _build_amdb_dataset(dataset_config: dict):
    from primus.backends.diffusion.data.registrations.benchmark import (
        build_benchmark_dataset,
    )

    return build_benchmark_dataset(dataset_config)


def _build_amdb_benchmark_trainer(
    *, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None
):
    from primus.backends.diffusion.trainers.benchmark import (
        build_amdb_benchmark_trainer,
    )

    return build_amdb_benchmark_trainer(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args=trainer_args,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
    )


MODEL_BUILDERS: Dict[str, Callable[[dict], Any]] = {
    "flux": _build_flux_model,
    "flux.1-dev": _build_flux_preset_model("flux.1-dev"),
    "flux.1-schnell": _build_flux_preset_model("flux.1-schnell"),
    "wan": _build_wan_model,
}
DATASET_BUILDERS: Dict[str, Callable[[dict], Tuple[Any, Any]]] = {
    "flux": _build_flux_dataset,
    "wan": _build_wan_dataset,
}
TRAINER_BUILDERS: Dict[str, Callable[..., Any]] = {
    "fsdp2": _build_fsdp2_trainer,
}

MODEL_BUILDERS["sdxl"] = _build_sdxl
MODEL_BUILDERS["stable-diffusion-xl"] = _build_sdxl
DATASET_BUILDERS["amdb"] = _build_amdb_dataset
TRAINER_BUILDERS["amdb_benchmark"] = _build_amdb_benchmark_trainer


def get_model_builder(name: str) -> Callable[[dict], Any]:
    try:
        return MODEL_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown model name: {name}")


def get_dataset_builder(name: str) -> Callable[[dict], Tuple[Any, Any]]:
    try:
        return DATASET_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset name: {name}")


def get_trainer_builder(name: str) -> Callable[..., Any]:
    try:
        return TRAINER_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown trainer name: {name}")
