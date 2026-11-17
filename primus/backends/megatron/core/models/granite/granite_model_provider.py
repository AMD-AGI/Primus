###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Granite ``model_provider``: upstream Megatron-LM ``model_provider(model_builder, ...)``,
refusing ModelOpt. ``get_model_provider("granite")`` returns
``partial(model_provider, granite_builders.granite_builder)``.

Megatron is imported inside the functions (``megatron.training`` probes the device at
import), so this module imports without a GPU.
"""

import inspect
from typing import Optional

from primus.core.utils.import_utils import lazy_import

__all__ = ["model_provider"]


def _upstream_model_provider():
    """Upstream Megatron-LM ``model_provider`` (top-level ``model_provider.py``).

    Raises ``ImportError`` when Megatron-LM has no ``model_provider.py`` (the Old layout),
    and ``RuntimeError`` when it has one without the ``model_builder`` parameter (the Mid
    layout): the module imports, its API is the wrong version."""
    try:
        upstream = lazy_import(["model_provider"], "model_provider", log_prefix="[Primus][MegatronCompat]")
    except ImportError as e:
        raise ImportError(
            "[Primus][MegatronCompat] Granite needs Megatron-LM's model_provider.py "
            f"(model_provider(model_builder, ...), the gpt_builder layout): {e}"
        ) from e
    params = list(inspect.signature(upstream).parameters)
    if not params or params[0] != "model_builder":
        raise RuntimeError(
            "[Primus][MegatronCompat] Granite needs Megatron-LM's model_provider(model_builder, ...); "
            f"this Megatron-LM has model_provider{tuple(params)}"
        )
    return upstream


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``: upstream
    ``model_provider(model_builder, ...)``, refusing ModelOpt.

    ``get_model_provider`` binds :func:`granite_builder` as ``model_builder``. With
    ``modelopt_enabled`` (set by Megatron's ``get_model`` for a ModelOpt checkpoint) upstream
    ``model_provider`` replaces the builder with ``modelopt_gpt_mamba_builder``, which builds a
    plain GPT layer spec without Granite's scaled residual add, so it is refused.
    """
    from megatron.training import get_args

    if getattr(get_args(), "modelopt_enabled", False):
        raise ValueError(
            "Granite does not support ModelOpt (modelopt_enabled): the ModelOpt builder would "
            "replace the Granite builder, so the model would be a plain GPT model without "
            "GraniteTransformerConfig (residual_multiplier) and its scaled residual add"
        )
    if model_builder is None:
        from primus.backends.megatron.core.models.granite.granite_builders import (
            granite_builder as model_builder,
        )
    return _upstream_model_provider()(
        model_builder,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
