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


def _build_fsdp2_trainer(*, model, dataset, processor, trainer_args: dict):
    from primus.backends.diffusion.trainers.fsdp2 import build_fsdp2_trainer

    return build_fsdp2_trainer(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args=trainer_args,
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
