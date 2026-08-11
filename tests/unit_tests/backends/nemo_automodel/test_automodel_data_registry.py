###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the AutoModel offline cache-builder registry.

WHY THIS TEST EXISTS:
  The registry maps a model name to a ``"<module>:<callable>"`` string, resolved by
  import only when ``primus data automodel-cache --model <name>`` actually runs. That
  laziness is deliberate — it keeps the CLI from importing torch, diffusers and every
  model's optional dependencies just to print ``--help``.

  The cost of laziness is that a typo in a registry string, or a builder that gets
  renamed or moved, is invisible until someone runs a cache build — which is a
  GPU-bound job needing gated encoder weights, i.e. the most expensive possible place
  to discover it. These tests resolve every registered entry so that breakage surfaces
  here instead.

  ``build_cache`` itself is not called: it needs a GPU and the encoder weights.
"""

import inspect

import pytest

from primus.backends.nemo_automodel.data.registry import (
    CACHE_BUILDERS,
    available_models,
    get_cache_builder,
)


def test_registry_is_not_empty():
    assert CACHE_BUILDERS, "no cache builders registered"


def test_available_models_is_sorted_and_matches_registry():
    assert available_models() == sorted(CACHE_BUILDERS)


def test_ideogram4_is_registered():
    assert "ideogram4" in available_models()


@pytest.mark.parametrize("model", sorted(CACHE_BUILDERS))
def test_every_registered_builder_resolves_and_is_callable(model):
    """The whole point: a stale registry string must fail here, not on a GPU node."""
    builder = get_cache_builder(model)
    assert callable(builder), f"{model} builder is not callable"


@pytest.mark.parametrize("model", sorted(CACHE_BUILDERS))
def test_every_builder_accepts_the_arguments_the_cli_passes(model):
    """The CLI calls builders purely by keyword, so the names are the contract."""
    builder = get_cache_builder(model)
    parameters = inspect.signature(builder).parameters
    accepts_extra_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    required = {
        "image_dir",
        "caption_dir",
        "output_dir",
        "num_samples",
        "resolution",
        "max_text_tokens",
        "vae_source",
        "text_encoder_source",
        "tokenizer_source",
        "device",
        "dtype",
        "seed",
        "shuffle",
    }
    if not accepts_extra_kwargs:
        missing = required - set(parameters)
        assert not missing, f"{model} builder cannot accept CLI arguments: {sorted(missing)}"


def test_unknown_model_names_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        get_cache_builder("no-such-model")
    message = str(excinfo.value)
    assert "no-such-model" in message
    # An unknown --model should tell the user what they could have typed.
    for model in available_models():
        assert model in message
