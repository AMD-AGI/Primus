###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the AMDiffusionBenchmark model, dataset, and trainer builders."""

from __future__ import annotations

from typing import Any, Callable


def _build_mochi(model_config: dict):
    from primus.backends.diffusion.amdb_benchmarks.mochi_registration import (
        build_mochi_model,
    )

    return build_mochi_model(model_config)


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
    model_builders.update({"mochi-1": _build_mochi})
    dataset_builders["amdb"] = _build_dataset
    trainer_builders["amdb_benchmark"] = _build_trainer
