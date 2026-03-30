###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Utility functions related to FP8 that are used throughout Megatron core"""
from contextlib import nullcontext

import torch
from megatron.core.fp8_utils import is_first_last_bf16_layer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

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


SCALING_BLOCK_SIZE = 128
MX_SCALING_BLOCK_SIZE = 32

WARN_ONCE = True


if HAVE_TE and HAVE_TURBO:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import TEDelayedScaling
    from primus_turbo.pytorch.core.low_precision import ScaleDtype, ScalingGranularity

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboQuantConfig,
    )

    def te_fp8_format_mapping(te_format):
        from primus_turbo.pytorch.core.low_precision import Format as TurboFormat
        from transformer_engine.common.recipe import Format as TEFormat

        format_mapping = {
            TEFormat.E4M3: TurboFormat.E4M3,
            TEFormat.HYBRID: TurboFormat.HYBRID,
            TEFormat.E5M2: TurboFormat.E5M2,
        }

        return format_mapping[te_format]

    def get_fp8_recipe(config: TransformerConfig):
        """Return fp8 recipe.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 recipe: Transformer Engine FP8 recipe.
            FP8 None reason: reason why the fp8 recipe is None.
        """
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        # Select fp8 recipe (TE version >= 2.1.0).
        fp8_recipe = None
        fp8_recipe_none_reason = ""
        if is_te_min_version("2.1.0"):
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
                )
            elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
                fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                fp8_recipe_none_reason = "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and DelayedScaling are the only supported FP8 recipes. Please also make sure you are using a compatible TE version."
        else:
            # Assert that the user is using delayed scaling.
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            else:
                fp8_recipe_none_reason = "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for Float8BlockScaling."

        return fp8_recipe, fp8_recipe_none_reason

    def get_fp8_quant_config(config: TransformerConfig):
        """Return fp8 quant config.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 quant config: Primus-Turbo FP8 quant config.
            FP8 quant config none reason: reason why the fp8 quant config is None.
        """
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        fp8_quant_config = None
        fp8_quant_config_none_reason = ""
        if config.fp8_recipe == Fp8Recipe.delayed:
            # NOTE: Primus-Turbo not support delayed scaling.
            fp8_quant_config_none_reason = "Primus-Turbo not support delayed scaling."
        elif config.fp8_recipe == Fp8Recipe.tensorwise:
            fp8_quant_config = PrimusTurboQuantConfig(
                granularity=ScalingGranularity.TENSORWISE, format=te_fp8_format_mapping(fp8_format)
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise:
            fp8_quant_config = PrimusTurboQuantConfig(
                granularity=ScalingGranularity.BLOCKWISE,
                format=te_fp8_format_mapping(fp8_format),
                block_size=SCALING_BLOCK_SIZE,
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            fp8_quant_config = PrimusTurboQuantConfig(
                granularity=ScalingGranularity.MX_BLOCKWISE,
                format=te_fp8_format_mapping(fp8_format),
                block_size=MX_SCALING_BLOCK_SIZE,
                scale_dtype=ScaleDtype.E8M0,
            )

        return fp8_quant_config, fp8_quant_config_none_reason

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).
            is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

        Returns:
            FP8 context.
            If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
            We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
            that needs to be trained in bf16.
        """

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
            # bf16 training or bf16 layer in fp8 training
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            fp8_recipe, fp8_recipe_none_reason = get_fp8_recipe(config)
            fp8_quant_config, fp8_quant_config_none_reason = get_fp8_quant_config(config)

            global WARN_ONCE
            if WARN_ONCE:
                if fp8_recipe is None:
                    warning_rank_0(
                        f"TransformerEngine FP8 {config.fp8_recipe} not work since {fp8_recipe_none_reason}"
                    )
                if fp8_quant_config is None:
                    warning_rank_0(
                        f"Primus-Turbo FP8 {config.fp8_recipe} not work since {fp8_quant_config_none_reason}"
                    )
                if is_init:
                    warning_rank_0(
                        f"Primus-Turbo FP8 {config.fp8_recipe} not work since Primus-Turbo not support fp8 model init."
                    )
                WARN_ONCE = False

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                from primus.backends.megatron.core.extensions.primus_turbo import (
                    primus_turbo_fp8_autocast,
                )

                fp8_context = primus_turbo_fp8_autocast(
                    enabled=True if fp8_recipe is not None else False,
                    fp8_recipe=fp8_recipe,
                    fp8_group=fp8_group,
                    enabled_turbo=True if fp8_quant_config is not None else False,
                    turbo_quant_config=fp8_quant_config,
                )
            else:
                import inspect

                context_args = {"enabled": True}
                # Check if fp8_model_init supports setting recipe
                if "recipe" in (inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters):
                    context_args["recipe"] = fp8_recipe
                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

elif HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    def get_fp8_recipe(config: TransformerConfig):
        """Return fp8 recipe.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 recipe.
        """
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        # Select fp8 recipe (TE version >= 2.1.0).
        fp8_recipe = None
        if is_te_min_version("2.1.0"):
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
                )
            elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
                fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(
                    "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                    "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                    "you are using a compatible TE version."
                )
        else:
            # Assert that the user is using delayed scaling.
            assert config.fp8_recipe == Fp8Recipe.delayed, (
                "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
                "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
                "Float8BlockScaling."
            )
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        return fp8_recipe

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).
            is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

        Returns:
            FP8 context.
            If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
            We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
            that needs to be trained in bf16.
        """

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
            # bf16 training or bf16 layer in fp8 training
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            fp8_recipe = get_fp8_recipe(config)

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                fp8_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
                )
            else:
                import inspect

                context_args = {"enabled": True}
                # Check if fp8_model_init supports setting recipe
                if "recipe" in (inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters):
                    context_args["recipe"] = fp8_recipe
                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

else:

    def get_fp8_recipe(config: TransformerConfig):
        """Returns None since TE is not available."""
        return None

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()
