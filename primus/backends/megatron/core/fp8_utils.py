###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Utility functions related to FP8 that are used throughout Megatron core"""
from contextlib import nullcontext

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

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


if HAVE_TE and HAVE_TURBO:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import TEDelayedScaling
    from primus_turbo.pytorch.core.float8 import Float8QuantConfig, ScalingGranularity

    def te_fp8_format_mapping(te_format):
        from primus_turbo.pytorch.core.float8 import Format as TurboFormat
        from transformer_engine.common.recipe import Format as TEFormat

        format_mapping = {
            TEFormat.E4M3: TurboFormat.E4M3,
            TEFormat.HYBRID: TurboFormat.HYBRID,
            TEFormat.E5M2: TurboFormat.E5M2,
        }

        return format_mapping[te_format]

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
        num_bf16_layers_at_start = config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        num_bf16_layers_at_end = config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        # Since layer_no is a global layer index, additional checks on whether
        # we are in the first or last pipeline-parallel rank are not needed.
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context:
            # bf16 training
            fp8_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            # fp8 training but this layer_no should be bf16
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            # Select TE fp8 recipe and turbo fp8 quant config
            fp8_recipe = None
            fp8_quant_config = None
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            elif config.fp8_recipe == Fp8Recipe.tensorwise:
                if is_te_min_version("2.2.0.dev0"):
                    fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(fp8_format=fp8_format)
                fp8_quant_config = Float8QuantConfig(
                    granularity=ScalingGranularity.TENSORWISE, format=te_fp8_format_mapping(fp8_format)
                )
            elif config.fp8_recipe == Fp8Recipe.blockwise:
                if is_te_min_version("2.3.0.dev0"):
                    fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
                fp8_quant_config = Float8QuantConfig(
                    granularity=ScalingGranularity.BLOCKWISE,
                    format=te_fp8_format_mapping(fp8_format),
                    block_size=128,
                )
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
                if is_te_min_version("2.1.0.dev0"):
                    fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
                fp8_quant_config = Float8QuantConfig(
                    granularity=ScalingGranularity.MX_BLOCKWISE,
                    format=te_fp8_format_mapping(fp8_format),
                    block_size=32,
                )

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
                    turbo_fp8_quant_config=fp8_quant_config,
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
                    context_args["preserve_high_precision_init_val"] = True
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

elif HAVE_TURBO:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe

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
        num_bf16_layers_at_start = config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        num_bf16_layers_at_end = config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        # Since layer_no is a global layer index, additional checks on whether
        # we are in the first or last pipeline-parallel rank are not needed.
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context:
            # bf16 training
            fp8_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            # fp8 training but this layer_no should be bf16
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            import primus_turbo

            if config.fp8 == "e4m3":
                fp8_format = primus_turbo.pytorch.core.float8.Format.E4M3
            elif config.fp8 == "hybrid":
                fp8_format = primus_turbo.pytorch.core.float8.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            # Select fp8
            fp8_quant_config = None
            if config.fp8_quant_config == Fp8Recipe.tensorwise:
                fp8_quant_config = primus_turbo.pytorch.core.float8.TensorwiseQuantConfig(
                    fp8_format=fp8_format
                )
            elif config.fp8_quant_config == Fp8Recipe.blockwise:
                fp8_quant_config = primus_turbo.pytorch.core.float8.BlockwiseQuantConfig(
                    fp8_format=fp8_format
                )
            else:
                raise ValueError("Primus-Turbo only supports TensorwiseQuantConfig and BlockwiseQuantConfig.")

            if not is_init:
                from primus.backends.megatron.core.extensions.primus_turbo import (
                    primus_turbo_fp8_autocast,
                )

                # NOTE: Disable Transformer Engine FP8
                fp8_context = primus_turbo_fp8_autocast(
                    enabled=False,
                    fp8_recipe=None,
                    fp8_group=None,
                    enabled_turbo=True,
                    turbo_fp8_quant_config=fp8_quant_config,
                )
            else:
                # NOTE: Primus-Turbo does not support fp8_model_init yet.
                fp8_context = nullcontext()

        return fp8_context

elif HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

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
        num_bf16_layers_at_start = config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        num_bf16_layers_at_end = config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        # Since layer_no is a global layer index, additional checks on whether
        # we are in the first or last pipeline-parallel rank are not needed.
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context:
            # bf16 training
            fp8_context = nullcontext()
        elif layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
            # fp8 training but this layer_no should be bf16
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            import transformer_engine  # To keep out TE dependency when not training in fp8

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
                    fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(fp8_format=fp8_format)
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
                    context_args["preserve_high_precision_init_val"] = True
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

else:

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()
