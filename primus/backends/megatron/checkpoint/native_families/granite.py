###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native converter registration of dense IBM Granite (GraniteForCausalLM, e.g. granite-4.1-3b).

The model_type ``granite`` is shared by task heads (GraniteForSequenceClassification, ...)
that the converter does not build, so it selects the family only when config.json lists no
architectures. The model is built with Primus's ``granite`` model provider (scaled residual
branches)."""

from primus.backends.megatron.checkpoint.native_registry import (
    NativeFamily,
    register_native_family,
)

register_native_family(
    NativeFamily(
        name="granite",
        label="Granite dense (GraniteForCausalLM)",
        converter="primus.backends.megatron.checkpoint.native_loader_granite",
        architectures=("GraniteForCausalLM",),
        model_types=("granite",),
        model_type_needs_no_architectures=True,
        model_type="granite",
    )
)
