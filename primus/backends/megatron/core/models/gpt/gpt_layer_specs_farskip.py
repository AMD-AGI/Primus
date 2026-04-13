###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.core.models.gpt import gpt_layer_specs
from megatron.training.global_vars import get_args
from primus.backends.megatron.core.transformer.farskip_transformer_layer import (
    SimpleFarSkipTransformerLayer,
    OverlappedFarSkipTransformerLayer,
)


def get_gpt_layer_with_transformer_engine_spec(num_experts=None, **kwargs):
    use_simple = get_args().use_simple_farskip_layer
    use_overlapped = get_args().use_overlapped_farskip_layer

    result = gpt_layer_specs._original_get_te_spec(num_experts=num_experts, **kwargs)

    if use_overlapped and num_experts is not None:
        result.module = OverlappedFarSkipTransformerLayer
    elif use_simple or (use_overlapped and num_experts is None):
        result.module = SimpleFarSkipTransformerLayer

    return result


def get_gpt_mtp_block_spec_for_backend(config, spec, backend, **kwargs):
    """Wrap MTP block spec to strip farskip from MTP layers (farskip is not supported for MTP)."""
    from copy import deepcopy
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    if isinstance(spec, TransformerBlockSubmodules):
        layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec):
        layer_spec = spec
    else:
        layer_spec = None

    if layer_spec is not None and layer_spec.module in (SimpleFarSkipTransformerLayer, OverlappedFarSkipTransformerLayer):
        spec = deepcopy(spec)
        if isinstance(spec, TransformerBlockSubmodules):
            spec.layer_specs[-1].module = TransformerLayer
        elif isinstance(spec, ModuleSpec):
            spec.module = TransformerLayer

    return gpt_layer_specs._original_get_mtp_block_spec_for_backend(config, spec, backend, **kwargs)


def get_gpt_layer_local_spec(num_experts=None, **kwargs):
    use_simple = get_args().use_simple_farskip_layer
    use_overlapped = get_args().use_overlapped_farskip_layer

    result = gpt_layer_specs._original_get_local_spec(num_experts=num_experts, **kwargs)

    if use_overlapped and num_experts is not None:
        result.module = OverlappedFarSkipTransformerLayer
    elif use_simple or (use_overlapped and num_experts is None):
        result.module = SimpleFarSkipTransformerLayer

    return result
