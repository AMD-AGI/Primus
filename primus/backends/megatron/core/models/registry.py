###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model families the Megatron backend builds, keyed by the YAML ``model_type``.

Every reference is a ``"module:attribute"`` string resolved on first use, so
importing this table imports no model code. ``primus/core/config/schema_check.py``
parses it without importing it, which is why entries must stay plain literals.
"""

import importlib
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_MODEL_TYPE = "gpt"


@dataclass(frozen=True)
class ModelFamily:
    """How one ``model_type`` is built and trained.

    A family either names an upstream ``builder`` (``gpt``, ``mamba``), or
    declares the pieces the shared builder in ``builder.py`` assembles:

    * ``config_class``: the dataclass handed to ``core_transformer_config_from_args``.
    * ``model_class``: constructed with GPTModel-style arguments.
    * ``decoder_spec``: ``fn(config, vp_stage=...) -> spec``, passed as
      ``transformer_layer_spec``. Empty when the model builds its own decoder.
    * ``prepare_args``: ``fn(args)``, run before the config is built.
    * ``model_kwargs``: ``fn(args) -> dict`` of extra constructor arguments.
    * ``check_model``: ``fn(args, config, model)``, run on the built model.

    Module paths are relative to ``package`` when it is set. ``training`` is
    the upstream entry point providing ``forward_step`` and
    ``train_valid_test_datasets_provider``.
    """

    package: str = ""
    builder: str = ""
    config_class: str = ""
    model_class: str = ""
    decoder_spec: str = ""
    prepare_args: str = ""
    model_kwargs: str = ""
    check_model: str = ""
    training: str = "pretrain_gpt"

    def resolve(self, ref: str) -> Any:
        module, _, attr = ref.partition(":")
        if self.package:
            module = f"{self.package}.{module}"
        return getattr(importlib.import_module(module), attr)


MODEL_FAMILIES = {
    "gpt": ModelFamily(builder="gpt_builders:gpt_builder"),
    "mamba": ModelFamily(builder="mamba_builders:mamba_builder", training="pretrain_mamba"),
    "deepseek_v4": ModelFamily(
        package="primus.backends.megatron.core.models.deepseek_v4",
        config_class="deepseek_v4_transformer_config:DeepSeekV4TransformerConfig",
        model_class="deepseek_v4_model:DeepseekV4Model",
        decoder_spec="deepseek_v4_layer_specs:get_deepseek_v4_runtime_decoder_spec",
        prepare_args="deepseek_v4_builders:prepare_deepseek_v4_args",
        model_kwargs="deepseek_v4_builders:deepseek_v4_model_kwargs",
    ),
    "glm5_next": ModelFamily(
        package="primus.backends.megatron.core.models.glm5_next",
        config_class="glm5_next_transformer_config:Glm5NextTransformerConfig",
        model_class="glm5_next_model:Glm5NextModel",
        decoder_spec="glm5_next_layer_specs:get_glm5_next_runtime_decoder_spec",
    ),
    "kimi_k3": ModelFamily(
        package="primus.backends.megatron.core.models.kimi_k3",
        config_class="kimi_k3_transformer_config:KimiK3TransformerConfig",
        model_class="kimi_k3_model:KimiK3Model",
        decoder_spec="kimi_k3_layer_specs:get_kimi_k3_runtime_decoder_spec",
        model_kwargs="kimi_k3_builders:kimi_k3_model_kwargs",
        check_model="kimi_k3_builders:check_kimi_k3_model",
    ),
    "zaya1": ModelFamily(
        package="primus.backends.megatron.core.models.zaya1",
        config_class="zaya1_transformer_config:Zaya1TransformerConfig",
        model_class="zaya1_model:Zaya1Model",
        prepare_args="zaya1_builders:prepare_zaya1_args",
        model_kwargs="zaya1_builders:zaya1_model_kwargs",
    ),
}


def get_model_family(model_type: Optional[str]) -> ModelFamily:
    """The family ``model_type`` names. Unregistered values (e.g. the diffusion
    trainers' ``diffusion_model``, which supply their own provider) train as ``gpt``."""
    return MODEL_FAMILIES.get(model_type or DEFAULT_MODEL_TYPE, MODEL_FAMILIES[DEFAULT_MODEL_TYPE])
