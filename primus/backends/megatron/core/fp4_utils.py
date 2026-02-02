###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


"""Utility functions related to FP4 that are used throughout Megatron core"""

from contextlib import nullcontext

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

from primus.backends.megatron.core.enums import Fp4Recipe
from primus.backends.megatron.core.extensions.primus_turbo import (
    primus_turbo_fp4_autocast,
)
from primus.modules.module_utils import warning_rank_0

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

# Check if Primus-Turbo is installed
HAVE_TURBO = False
try:
    import primus_turbo  # pylint: disable=W0611

    HAVE_TURBO = True
except (ImportError, ModuleNotFoundError):
    # Primus-Turbo not found
    pass


WARN_ONCE = True


if HAVE_TE and HAVE_TURBO:
    from primus_turbo.pytorch.core.low_precision import (
        Format,
        ScaleDtype,
        ScalingGranularity,
    )

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboQuantConfig,
    )

    def get_fp4_recipe(config: TransformerConfig):
        """Return fp4 recipe."""
        fp4_recipe = None
        fp4_recipe_none_reason = ""
        if is_te_min_version("2.7.0.dev0"):
            if config.fp4_recipe == Fp4Recipe.nvfp4:
                try:
                    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
                except AttributeError:
                    fp4_recipe_none_reason = "NVFP4BlockScaling recipe is not available in this version of Transformer Engine. Please make sure you are using TE version >= 2.7.0.dev0."
            else:
                fp4_recipe_none_reason = "NVFP4BlockScaling is the only supported FP4 recipe. Please make sure you are using a compatible TE version >= 2.7.0.dev0."
        else:
            fp4_recipe_none_reason = (
                "FP4 support requires TransformerEngine version >= 2.7.0.dev0 for NVFP4BlockScaling."
            )

        return fp4_recipe, fp4_recipe_none_reason

    def get_fp4_quant_config(config: TransformerConfig):
        """Return fp4 quant config."""
        fp4_quant_config = None
        fp4_quant_config_none_reason = ""
        if config.fp4_recipe == Fp4Recipe.mxfp4:
            fp4_quant_config = PrimusTurboQuantConfig(
                granularity=ScalingGranularity.MX_BLOCKWISE,
                format=Format.E2M1_X2,
                block_size=32,
                scale_dtype=ScaleDtype.E8M0,
            )
        else:
            fp4_quant_config_none_reason = "Only MXFP4 is supported in Primus-Turbo."

        return fp4_quant_config, fp4_quant_config_none_reason

    def get_fp4_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp4 context manager."""
        num_bf16_layers_at_start = config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        num_bf16_layers_at_end = config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp4_context = config.fp4 if not is_init else config.fp4_param

        if not need_fp4_context:
            fp4_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            fp4_context = nullcontext()
        else:
            fp4_recipe, fp4_recipe_none_reason = get_fp4_recipe(config)
            fp4_quant_config, fp4_quant_config_none_reason = get_fp4_quant_config(config)

            global WARN_ONCE
            if WARN_ONCE:
                if fp4_recipe is None:
                    warning_rank_0(
                        f"TransformerEngine FP4 {config.fp4_recipe} not work since {fp4_recipe_none_reason}"
                    )
                if fp4_quant_config is None:
                    warning_rank_0(
                        f"Primus-Turbo FP4 {config.fp4_recipe} not work since {fp4_quant_config_none_reason}"
                    )
                if is_init:
                    warning_rank_0(
                        f"Primus-Turbo FP4 {config.fp4_recipe} not work since Primus-Turbo not support fp4 model init."
                    )
                WARN_ONCE = False

            fp4_group = None
            if parallel_state.model_parallel_is_initialized():
                fp4_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                fp4_context = primus_turbo_fp4_autocast(
                    enabled=True if fp4_recipe is not None else False,
                    fp4_recipe=fp4_recipe,
                    fp4_group=fp4_group,
                    enabled_turbo=True if fp4_quant_config is not None else False,
                    turbo_quant_config=fp4_quant_config,
                )
            else:
                import inspect

                context_args = {"enabled": True}
                if "recipe" in inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters:
                    context_args["recipe"] = fp4_recipe
                fp4_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        return fp4_context

elif HAVE_TE:

    def get_fp4_recipe(config: TransformerConfig):
        """Return fp4 recipe."""
        if is_te_min_version("2.7.0.dev0"):
            if config.fp4_recipe == Fp4Recipe.nvfp4:
                try:
                    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
                except AttributeError:
                    raise ValueError(
                        """NVFP4BlockScaling recipe is not available in this version of
                        Transformer Engine. Please make sure you are using TE version
                        >= 2.7.0.dev0."""
                    )
            else:
                raise ValueError(
                    "NVFP4BlockScaling is the only supported FP4 recipe. "
                    "Please make sure you are using a compatible TE version >= 2.7.0.dev0."
                )
        else:
            raise ValueError(
                """FP4 support requires TransformerEngine version >= 2.7.0.dev0
                for NVFP4BlockScaling."""
            )
        return fp4_recipe

    def get_fp4_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp4 context manager."""
        num_bf16_layers_at_start = config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        num_bf16_layers_at_end = config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp4_context = config.fp4 if not is_init else config.fp4_param

        if not need_fp4_context:
            fp4_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            fp4_context = nullcontext()
        else:
            fp4_recipe = get_fp4_recipe(config)
            fp4_group = None
            if parallel_state.model_parallel_is_initialized():
                fp4_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                # TE currently uses fp8_autocast for fp8 and fp4 quantization.
                fp4_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp4_recipe, fp8_group=fp4_group
                )
            else:
                import inspect

                context_args = {"enabled": True}
                if "recipe" in inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters:
                    context_args["recipe"] = fp4_recipe
                fp4_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        return fp4_context

else:

    def get_fp4_recipe(config: TransformerConfig):
        """Return None when Transformer Engine is not available."""
        return None

    def get_fp4_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return nullcontext when Transformer Engine is not available."""
        return nullcontext()
