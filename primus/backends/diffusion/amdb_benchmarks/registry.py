###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the AMDiffusionBenchmark model, dataset, and trainer builders."""

from __future__ import annotations

from typing import Any, Callable


def _build_flux_dev(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.flux.registration import (
        build_flux_model,
    )

    config = dict(model_config or {})
    config["model_preset"] = config.get("model_preset") or "flux.1-dev"
    return build_flux_model(config)


def _build_sdxl(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.sdxl_registration import (
        build_sdxl_model,
    )

    return build_sdxl_model(model_config)


def _build_mochi(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.mochi_registration import (
        build_mochi_model,
    )

    return build_mochi_model(model_config)


def _build_hunyuan(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.hunyuan_registration import (
        build_hunyuan_model,
    )

    return build_hunyuan_model(model_config)


def _build_wan21_i2v(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.wan.registration import (
        build_wan_model,
    )

    config = dict(model_config or {})
    nested = dict(config.get("config") or {})
    nested.setdefault("model_type", "i2v")
    nested.setdefault("dit_hidden_size", 5120)
    nested.setdefault("dit_num_layers", 40)
    nested.setdefault("dit_num_heads", 40)
    nested.setdefault("dit_intermediate_size", 13824)
    nested.setdefault("dit_in_channels", 36)
    nested.setdefault("dit_out_channels", 16)
    nested.setdefault("dit_patch_size", (1, 2, 2))
    nested.setdefault("dit_freq_dim", 256)
    nested.setdefault("dit_text_dim", 4096)
    nested.setdefault("text_len", 512)
    nested.setdefault("vae_type", "wan_video_vae")
    nested.setdefault("separated_timestep", False)
    nested.setdefault("fuse_vae_embedding_in_latents", False)
    nested.setdefault("trainable_modules", "dit")
    nested.setdefault("height", 544)
    nested.setdefault("width", 960)
    nested.setdefault("num_frames", 9)
    nested.setdefault("amdb_sampling", True)
    config["config"] = nested
    return build_wan_model(config)


def _build_dataset(dataset_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.data import build_amdb_dataset

    return build_amdb_dataset(dataset_config)


def _build_trainer(*, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None):
    from primus.backends.diffusion.amdb_benchmarks.trainer import (
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


def register_amdb_benchmarks(
    model_builders: dict[str, Callable[[dict], Any]],
    dataset_builders: dict[str, Callable[[dict], Any]],
    trainer_builders: dict[str, Callable[..., Any]],
) -> None:
    model_builders.update(
        {
            "flux-dev": _build_flux_dev,
            "sdxl": _build_sdxl,
            "stable-diffusion-xl": _build_sdxl,
            "mochi-1": _build_mochi,
            "hunyuan-video": _build_hunyuan,
            "wan2.1-i2v": _build_wan21_i2v,
        }
    )
    dataset_builders["amdb"] = _build_dataset
    trainer_builders["amdb_benchmark"] = _build_trainer
