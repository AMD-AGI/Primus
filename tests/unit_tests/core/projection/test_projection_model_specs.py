###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Drift checks for the projection's transcribed model specs.

``BUILTIN_MODEL_SPECS`` exists so a cluster can be sized without a training
checkout, but a transcription that drifts from the backend it was copied from
mis-sizes that cluster silently.  These tests re-derive the same architectures
from every authority that happens to be reachable and compare:

* **Primus' own Megatron presets**, which ship in this repo, so this check always
  runs -- a Llama 3 8B is the same 32 layers whichever backend trains it.
* **The installed TorchTitan flavor table**, when TorchTitan is importable.

The backend-sourced check skips rather than fails when the backend is absent,
which is the normal case in CI and on a projection-only machine.
"""

from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional

import pytest
import yaml

from primus.core.projection.frameworks.model_specs import (
    BUILTIN_MODEL_SPECS,
    TORCHTITAN_FLAVOR_ALIASES,
    ModelSpec,
    get_builtin_spec,
)
from primus.core.projection.frameworks.torchtitan import (
    _spec_from_model_args,
    _torchtitan_model_args,
)

_MEGATRON_PRESET_DIR = Path(__file__).resolve().parents[4] / "primus" / "configs" / "models" / "megatron"

# Canonical spec name -> the Megatron preset describing the same model.
_MEGATRON_PRESETS: Dict[str, str] = {
    "llama3.2-1b": "llama3.2_1B.yaml",
    "llama3-8b": "llama3_8B.yaml",
    "llama3-70b": "llama3_70B.yaml",
    "llama3.1-405b": "llama3.1_405B.yaml",
    "llama4-17bx16e": "llama4_17B16E.yaml",
    "llama4-17bx128e": "llama4_17B128E.yaml",
    "deepseek-v2-16b": "deepseek_v2_lite.yaml",
    "deepseek-v2-236b": "deepseek_v2.yaml",
    "deepseek-v3-671b": "deepseek_v3.yaml",
    "qwen3-4b": "qwen3_4B.yaml",
    "qwen3-8b": "qwen3_8B.yaml",
    "qwen3-14b": "qwen3_14B.yaml",
    "qwen3-32b": "qwen3_32B.yaml",
    "qwen3-30b-a3b": "qwen3_30B_A3B.yaml",
    "qwen3-235b-a22b": "qwen3_235B_A22B.yaml",
    "gpt-oss-20b": "gpt_oss_20B.yaml",
    "gpt-oss-120b": "gpt_oss_120B.yaml",
}

# Megatron preset key -> ModelSpec attribute holding the same number.  The
# vocabulary is deliberately absent: Megatron pads it to the tokenizer Primus
# trains with, while TorchTitan carries the model's own.
_MEGATRON_FIELD_MAP = {
    "num_layers": "num_layers",
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_query_groups": "num_query_groups",
    "kv_channels": "kv_channels",
    "num_experts": "num_experts",
    "moe_router_topk": "moe_router_topk",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
}

# Fields Megatron only reads under multi-head latent attention.  The presets
# carry values for them either way, so they are only a claim when MLA is on.
_MEGATRON_MLA_FIELD_MAP = {
    "kv_lora_rank": "kv_lora_rank",
    "q_lora_rank": "q_lora_rank",
    "qk_head_dim": "qk_head_dim",
    "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
    "v_head_dim": "v_head_dim",
}


def _load_megatron_preset(filename: str) -> Optional[Dict[str, object]]:
    """Load one Megatron preset file, without following its ``extends`` chain.

    Only the leaf file is read on purpose.  The bases it extends carry Megatron's
    own defaults -- ``moe_router_topk: 2`` and ``kv_lora_rank: 32`` sit in
    ``language_model.yaml`` and apply to dense models too -- so an inherited value
    is not a claim about this model's architecture, and comparing against one
    would fail every dense preset in the table.
    """
    path = _MEGATRON_PRESET_DIR / filename
    if not path.is_file():
        return None
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle) or {}
    return {key: value for key, value in loaded.items() if key != "extends"}


@pytest.mark.parametrize("canonical,preset", sorted(_MEGATRON_PRESETS.items()))
def test_spec_matches_the_megatron_preset_for_the_same_model(canonical, preset):
    spec = get_builtin_spec(canonical)
    assert spec is not None, f"{canonical} is mapped to a preset but has no spec"

    config = _load_megatron_preset(preset)
    assert config is not None, f"Megatron preset {preset} is missing"

    fields = dict(_MEGATRON_FIELD_MAP)
    if spec.multi_latent_attention and config.get("multi_latent_attention"):
        fields.update(_MEGATRON_MLA_FIELD_MAP)

    compared = 0
    for preset_key, spec_attr in fields.items():
        expected = config.get(preset_key)
        if expected is None:
            continue
        assert getattr(spec, spec_attr) == expected, (
            f"{canonical}: {spec_attr} is {getattr(spec, spec_attr)} but Megatron "
            f"preset {preset} says {preset_key}={expected}"
        )
        compared += 1

    # The FFN width is spelled one way for a dense model and another for one that
    # routes every layer through experts, where Megatron falls back to
    # ``ffn_hidden_size`` for the expert width when no explicit one is given.
    dense_width = config.get("ffn_hidden_size")
    if dense_width is not None:
        if not spec.num_experts or not all(spec.moe_layer_pattern()):
            assert spec.ffn_hidden_size == dense_width, f"{canonical}: dense FFN width"
            compared += 1
        elif config.get("moe_ffn_hidden_size") is None:
            assert spec.moe_ffn_hidden_size == dense_width, f"{canonical}: expert FFN width"
            compared += 1

    assert compared >= 4, f"{preset} pinned too little of the architecture to be a check"


@pytest.mark.parametrize("canonical,preset", sorted(_MEGATRON_PRESETS.items()))
def test_moe_layer_pattern_matches_the_megatron_preset(canonical, preset):
    config = _load_megatron_preset(preset)
    assert config is not None
    spec = get_builtin_spec(canonical)

    freq = config.get("moe_layer_freq")
    if freq is None or not spec.num_experts:
        pytest.skip(f"{preset} does not pin a MoE layer pattern")

    pattern = spec.moe_layer_pattern()
    if isinstance(freq, str):
        # Megatron accepts a Python expression; the presets use it to spell out
        # DeepSeek's leading dense layers.
        expected = eval(freq, {"__builtins__": {}}, {})  # noqa: S307  (repo-owned data)
        if isinstance(expected, int):
            expected = [1 if i % expected == 0 else 0 for i in range(spec.num_layers)]
    elif isinstance(freq, int):
        expected = [1 if i % freq == 0 else 0 for i in range(spec.num_layers)]
    else:
        expected = list(freq)

    assert pattern == list(expected), f"{canonical}: MoE layer pattern drifted from {preset}"


# --------------------------------------------------------------------------- #
# Live backend cross-checks
# --------------------------------------------------------------------------- #


def _torchtitan_installed() -> bool:
    try:
        import torchtitan  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _torchtitan_installed(), reason="torchtitan is not installed")
@pytest.mark.parametrize("flavor_key", sorted(TORCHTITAN_FLAVOR_ALIASES))
def test_spec_matches_the_installed_torchtitan_flavor(flavor_key):
    name, flavor = flavor_key
    # The alias table is keyed lowercase; TorchTitan's own registry is not.
    model_args = _torchtitan_model_args(name, flavor) or _torchtitan_model_args(name, flavor.upper())
    if model_args is None:
        pytest.skip(f"installed torchtitan has no flavor {name}/{flavor}")
    live = _spec_from_model_args(name, model_args)
    if live is None:
        pytest.skip(f"torchtitan {name}/{flavor} does not pin an FFN width")

    transcribed = get_builtin_spec(TORCHTITAN_FLAVOR_ALIASES[flavor_key])
    # Field by field rather than whole-spec equality, so a failure names what
    # moved instead of printing two dataclass reprs to diff by eye.
    for field in fields(ModelSpec):
        assert getattr(transcribed, field.name) == getattr(live, field.name), (
            f"transcribed spec for {name}/{flavor}: {field.name} is "
            f"{getattr(transcribed, field.name)!r} but the installed torchtitan says "
            f"{getattr(live, field.name)!r}; update BUILTIN_MODEL_SPECS"
        )


# --------------------------------------------------------------------------- #
# Coverage: every model the shipped examples name has to resolve
# --------------------------------------------------------------------------- #


def test_every_builtin_spec_is_reachable_from_a_backend_alias():
    reachable = set(TORCHTITAN_FLAVOR_ALIASES.values())
    orphans = sorted(set(BUILTIN_MODEL_SPECS) - reachable)
    assert not orphans, f"specs no backend alias points at: {orphans}"


def test_specs_are_comparable_by_value():
    # The drift checks above rely on ModelSpec equality being structural.
    assert isinstance(BUILTIN_MODEL_SPECS["llama3-8b"], ModelSpec)
    assert BUILTIN_MODEL_SPECS["llama3-8b"] == BUILTIN_MODEL_SPECS["llama3-8b"]
    assert BUILTIN_MODEL_SPECS["llama3-8b"] != BUILTIN_MODEL_SPECS["llama3-70b"]
