###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


"""Utility functions related to FP4 that are used throughout Megatron core"""

from contextlib import nullcontext

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

from primus.backends.megatron.core.enums import Fp4Recipe
from primus.core.utils.module_utils import warning_rank_0

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


def _primus_turbo_enabled() -> bool:
    if not HAVE_TURBO:
        return False
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
        return enable_primus_turbo
    except Exception:
        return False


MXFP4_SCALING_BLOCK_SIZE = 32

WARN_ONCE = True


if HAVE_TE and HAVE_TURBO:

    def get_fp4_recipe(config: TransformerConfig):
        """Return fp4 recipe."""
        fp4_recipe = None
        fp4_recipe_none_reason = ""
        if is_te_min_version("2.7.0.dev0"):
            if config.fp4_recipe == Fp4Recipe.nvfp4:
                try:
                    fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
                except AttributeError:
                    fp4_recipe_none_reason = (
                        "NVFP4BlockScaling recipe is not available in this version of "
                        "Transformer Engine. Please make sure you are using TE version >= 2.7.0.dev0."
                    )
            elif config.fp4_recipe == Fp4Recipe.mxfp4:
                try:
                    import os

                    fp4_recipe = transformer_engine.common.recipe.MXFP4BlockScaling()
                    fp4_recipe.use_hadamard = os.environ.get("NVTE_MXFP4_USE_HADAMARD", "0") == "1"
                except AttributeError:
                    fp4_recipe_none_reason = (
                        "MXFP4BlockScaling recipe is not available in this version of "
                        "Transformer Engine. MXFP4 requires ROCm TE with AITER support."
                    )
            else:
                fp4_recipe_none_reason = (
                    f"Unsupported fp4_recipe '{config.fp4_recipe}'. "
                    "Supported recipes: 'nvfp4' (NVIDIA), 'mxfp4' (AMD ROCm with AITER)."
                )
        else:
            fp4_recipe_none_reason = "FP4 support requires TransformerEngine version >= 2.7.0.dev0."

        return fp4_recipe, fp4_recipe_none_reason

    def get_fp4_quant_config(config: TransformerConfig):
        """Return Primus-Turbo fp4 quant config.

        The Primus-Turbo extension is imported lazily here so that a missing or
        incompatible ``primus_turbo`` API only affects the Turbo FP4 path and
        cannot break module import (which would silently disable the FP4 patch
        and fall back to upstream Megatron's nvfp4-only recipe handling).
        """
        if config.fp4_recipe != Fp4Recipe.mxfp4:
            return None, "Only MXFP4 is supported in Primus-Turbo."

        try:
            from primus_turbo.pytorch.core.low_precision import (
                Format,
                ScaleDtype,
                ScalingGranularity,
            )

            from primus.backends.megatron.core.extensions.primus_turbo import (
                PrimusTurboQuantConfig,
            )
        except (ImportError, ModuleNotFoundError) as e:
            return None, f"Primus-Turbo FP4 quant config unavailable: {e}"

        fp4_quant_config = PrimusTurboQuantConfig(
            granularity=ScalingGranularity.MX_BLOCKWISE,
            format=Format.E2M1_X2,
            block_size=MXFP4_SCALING_BLOCK_SIZE,
            scale_dtype=ScaleDtype.E8M0,
            use_gradient_sr=getattr(config, "mxfp4_gradient_stochastic_rounding", False),
        )
        return fp4_quant_config, ""

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
            # Local spec bypasses TE autocast -- quantization is handled
            # internally by PrimusTurboMXFP4LocalSpecProvider's custom ops.
            if getattr(config, "transformer_impl", "transformer_engine") == "local":
                fp4_context = nullcontext()
            else:
                fp4_recipe, fp4_recipe_none_reason = get_fp4_recipe(config)
                turbo_enabled = _primus_turbo_enabled()

                global WARN_ONCE
                if WARN_ONCE:
                    if fp4_recipe is None:
                        warning_rank_0(
                            f"TransformerEngine FP4 {config.fp4_recipe} not work since {fp4_recipe_none_reason}"
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
                    # Only touch the Primus-Turbo extension when the Turbo FP4
                    # autocast path is explicitly enabled; otherwise use TE directly.
                    if turbo_enabled:
                        fp4_quant_config, fp4_quant_config_none_reason = get_fp4_quant_config(config)
                        if WARN_ONCE and fp4_quant_config is None:
                            warning_rank_0(
                                f"Primus-Turbo FP4 {config.fp4_recipe} not work since {fp4_quant_config_none_reason}"
                            )

                        from primus.backends.megatron.core.extensions.primus_turbo import (
                            primus_turbo_fp4_autocast,
                        )

                        # FP4 is done entirely by Primus-Turbo (``enabled_turbo``);
                        # keep the TE fp8 autocast DISABLED so a non-turbo TE module
                        # still inside this context (e.g. the TEDotProductAttention
                        # used when sliding-window attention forces
                        # use_turbo_attention=False) does not try to build a TE
                        # ``MXFP4BlockScaling`` RecipeState, which the shipped TE
                        # rejects ("MXFP4BlockScaling is not supported"); those
                        # modules are meant to stay BF16 under MXFP4 anyway.
                        fp4_context = primus_turbo_fp4_autocast(
                            enabled=False,
                            fp4_recipe=fp4_recipe,
                            fp4_group=fp4_group,
                            enabled_turbo=True if fp4_quant_config is not None else False,
                            turbo_quant_config=fp4_quant_config,
                        )
                    else:
                        # TE currently uses fp8_autocast for fp8 and fp4 quantization.
                        fp4_context = transformer_engine.pytorch.fp8_autocast(
                            enabled=True if fp4_recipe is not None else False,
                            fp8_recipe=fp4_recipe,
                            fp8_group=fp4_group,
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
                        "NVFP4BlockScaling recipe is not available in this version of "
                        "Transformer Engine. Please make sure you are using TE version "
                        ">= 2.7.0.dev0."
                    )
            elif config.fp4_recipe == Fp4Recipe.mxfp4:
                try:
                    import os

                    fp4_recipe = transformer_engine.common.recipe.MXFP4BlockScaling()
                    fp4_recipe.use_hadamard = os.environ.get("NVTE_MXFP4_USE_HADAMARD", "0") == "1"
                except AttributeError:
                    raise ValueError(
                        "MXFP4BlockScaling recipe is not available in this version of "
                        "Transformer Engine. MXFP4 requires ROCm TE with AITER support."
                    )
            else:
                raise ValueError(
                    f"Unsupported fp4_recipe '{config.fp4_recipe}'. "
                    "Supported recipes: 'nvfp4' (NVIDIA), 'mxfp4' (AMD ROCm with AITER)."
                )
        else:
            raise ValueError("FP4 support requires TransformerEngine version >= 2.7.0.dev0.")
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
            # Local spec bypasses TE autocast -- quantization is handled
            # internally by PrimusTurboMXFP4LocalSpecProvider's custom ops.
            if getattr(config, "transformer_impl", "transformer_engine") == "local":
                fp4_context = nullcontext()
            else:
                fp4_recipe = get_fp4_recipe(config)
                fp4_group = None
                if parallel_state.model_parallel_is_initialized():
                    fp4_group = parallel_state.get_amax_reduction_group(
                        with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                    )

                if not is_init:
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
