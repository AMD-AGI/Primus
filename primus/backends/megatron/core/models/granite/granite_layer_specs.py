###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Layer spec for IBM Granite: the upstream GPT dense layer spec with
``GraniteTransformerLayer`` as the layer module."""

import dataclasses

from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

from primus.backends.megatron.core.transformer.granite import GraniteTransformerLayer


def _granite_layer(spec):
    module = getattr(spec, "module", None)
    if not (isinstance(module, type) and issubclass(module, TransformerLayer)):
        raise TypeError(f"Granite expects a TransformerLayer spec, got module {module!r}")
    if issubclass(module, GraniteTransformerLayer):
        return spec
    if module is not TransformerLayer:
        # Replacing a custom subclass would silently drop its behaviour.
        raise TypeError(
            f"Granite cannot use the custom layer class {module.__name__}: the Granite layer "
            "replaces the layer module, so a TransformerLayer subclass would lose its own code. "
            "Subclass GraniteTransformerLayer instead."
        )
    return dataclasses.replace(spec, module=GraniteTransformerLayer)


def granite_layer_spec_from(base_spec):
    """Return ``base_spec`` with each layer module replaced by ``GraniteTransformerLayer``.

    ``base_spec`` is what upstream ``gpt_builder`` accepts for ``--spec``: a ``ModuleSpec``
    for a ``TransformerLayer`` or a ``TransformerBlockSubmodules`` with ``layer_specs``
    (each layer spec is converted; the block's ``layer_norm`` is kept). Submodules
    (attention, MLP, norms) are kept. A spec whose module is already a
    ``GraniteTransformerLayer`` (or a subclass) is kept as it is. Any other
    ``TransformerLayer`` subclass is refused with ``TypeError``, since replacing it would
    silently drop its behaviour.
    """
    if isinstance(base_spec, TransformerBlockSubmodules):
        if not base_spec.layer_specs:
            raise TypeError("Granite expects a block spec with layer_specs, got none")
        return dataclasses.replace(base_spec, layer_specs=[_granite_layer(s) for s in base_spec.layer_specs])
    return _granite_layer(base_spec)


def get_granite_layer_spec(args, config):
    """Granite decoder layer spec, selected the way upstream ``gpt_builder`` selects the
    dense GPT layer spec.

    It calls ``gpt_builders._get_transformer_layer_spec`` (private upstream) on purpose:
    that is the function Primus patches and upstream ``gpt_builder`` uses, so Granite gets
    the same TE / local / Primus Turbo spec as a GPT model with the same args. A rename
    upstream fails here with ``AttributeError`` at build time, not silently.
    """
    import gpt_builders  # third_party/Megatron-LM, on sys.path at runtime

    use_te = args.transformer_impl == "transformer_engine"
    base_spec = gpt_builders._get_transformer_layer_spec(use_te, config)
    return granite_layer_spec_from(base_spec)
