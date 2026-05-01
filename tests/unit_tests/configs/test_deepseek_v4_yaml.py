###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-2 P18 — DeepSeek-V4 YAML schema gate (G1).

Each V4 yaml under ``primus/configs/models/megatron/deepseek_v4_*.yaml``
must:

* parse via the standard primus :func:`parse_yaml` loader (extends +
  env resolution included);
* construct a :class:`DeepSeekV4TransformerConfig` after the standard
  schema massage (no CRIT-level KeyError on plan-2 fields);
* surface ``compress_ratios`` as a tuple of ints after
  ``__post_init__`` runs (P18 D4 — the YAML may carry the legacy string
  form ``"[0, 0, 4, ...]"`` or a real list, and the dataclass must
  normalize both);
* not carry retired fields (``v4_use_custom_mtp_block`` /
  ``mtp_compress_ratios`` were dropped in plan-2 P17);
* not silently lose the V4-specific MoE / HC / sliding-window / sink
  fields the schema relies on.

Plus a single provider-singleton check (P18 D1):
:func:`resolve_v4_provider` returns the same instance on repeated
calls within the same config.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_YAML_DIR = _REPO_ROOT / "primus" / "configs" / "models" / "megatron"


# ---------------------------------------------------------------------------
# Helpers — schema-friendly subset of the V4 config kwargs
# ---------------------------------------------------------------------------


def _config_kwargs_from_yaml(yaml_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter the merged YAML dict down to keys consumed by
    :class:`DeepSeekV4TransformerConfig` (and its parents).

    The full primus config tree mixes train / data / launcher fields
    with the model-config fields we care about here. The dataclass
    ignores unknown keys via Python's normal dataclass rules — but
    only when we hand it kwargs that exist on the dataclass. We do
    that by intersecting with ``DeepSeekV4TransformerConfig``'s
    ``__dataclass_fields__``.
    """
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    valid = set(DeepSeekV4TransformerConfig.__dataclass_fields__.keys())
    return {k: v for k, v in yaml_dict.items() if k in valid}


def _build_v4_config(yaml_dict: Dict[str, Any]):
    """Construct a V4 config from a parsed YAML dict.

    Some upstream parents require ``num_attention_heads`` and a few
    other fields. We rely on the YAML to provide them; missing fields
    fall back to dataclass defaults.
    """
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    kwargs = _config_kwargs_from_yaml(yaml_dict)
    return DeepSeekV4TransformerConfig(**kwargs)


# ---------------------------------------------------------------------------
# YAML parsing — base / flash / pro all parse and merge cleanly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parse_yaml_fn():
    """Bind the shared loader once for all tests in this module."""
    from primus.core.config.yaml_loader import parse_yaml

    return parse_yaml


@pytest.mark.parametrize(
    "yaml_name",
    ["deepseek_v4_base.yaml", "deepseek_v4_flash.yaml", "deepseek_v4_pro.yaml"],
)
def test_v4_yaml_parses(parse_yaml_fn, yaml_name: str) -> None:
    parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
    assert isinstance(parsed, dict)
    # Every V4 yaml must declare its core shape.
    for required in ("num_layers", "hidden_size", "num_attention_heads"):
        # base.yaml inherits these from elsewhere, so this is best-effort.
        if yaml_name == "deepseek_v4_base.yaml":
            continue
        assert required in parsed, f"{yaml_name} missing required key {required!r}"


# ---------------------------------------------------------------------------
# compress_ratios is normalized to tuple[int, ...] (P18 D4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    ["deepseek_v4_flash.yaml", "deepseek_v4_pro.yaml"],
)
def test_compress_ratios_normalized_to_tuple(parse_yaml_fn, yaml_name: str) -> None:
    """The dataclass ``__post_init__`` must convert
    ``compress_ratios`` (which may arrive as a YAML string or a list)
    into ``tuple[int, ...]``.

    The base yaml does not pin a schedule (it is provided per-variant);
    flash + pro do. Both must round-trip to a tuple of ints with no
    string survivors and no value drift vs the raw schedule.
    """
    parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
    raw = parsed["compress_ratios"]

    # Sanity: the YAML form is either a string or a list.
    assert isinstance(raw, (str, list, tuple))

    cfg = _build_v4_config(parsed)
    normalized = cfg.compress_ratios

    assert normalized is not None
    assert isinstance(
        normalized, tuple
    ), f"{yaml_name}: compress_ratios should normalize to tuple, got {type(normalized).__name__}"
    for i, r in enumerate(normalized):
        assert isinstance(r, int), f"{yaml_name}: compress_ratios[{i}]={r!r} is not int"

    # Length / values: strip the YAML form and compare to the parsed-int form.
    if isinstance(raw, str):
        import ast as _ast

        raw_list = _ast.literal_eval(raw)
    else:
        raw_list = list(raw)
    assert (
        tuple(int(x) for x in raw_list) == normalized
    ), f"{yaml_name}: compress_ratios value drift — yaml={raw_list}, normalized={normalized}"


def test_compress_ratios_canonical_dispatch_values_only(parse_yaml_fn) -> None:
    """Plan-2 P17 fixed the comment inversion in V4 yamls; here we
    enforce the *value* contract: every entry must be one of
    ``{0, 4, 128}`` (V4 attention only knows these branches —
    anything else is a schema bug)."""
    for yaml_name in ("deepseek_v4_flash.yaml", "deepseek_v4_pro.yaml"):
        parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
        cfg = _build_v4_config(parsed)
        bad = [(i, r) for i, r in enumerate(cfg.compress_ratios) if r not in (0, 4, 128)]
        assert not bad, (
            f"{yaml_name}: compress_ratios contains non-canonical values "
            f"(allowed: 0=dense+SWA, 4=CSA, 128=HCA); offenders: {bad}"
        )


# ---------------------------------------------------------------------------
# Retired schema fields (P17 hygiene)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retired_field",
    ["v4_use_custom_mtp_block", "mtp_compress_ratios"],
)
def test_retired_fields_not_in_v4_config(retired_field: str) -> None:
    """Plan-2 P17 deleted the legacy MTP block; both gating fields
    must be removed from ``DeepSeekV4TransformerConfig``."""
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    names = {f.name for f in fields(DeepSeekV4TransformerConfig)}
    assert (
        retired_field not in names
    ), f"{retired_field} must be removed from DeepSeekV4TransformerConfig (plan-2 P17)."


@pytest.mark.parametrize(
    "yaml_name",
    ["deepseek_v4_base.yaml", "deepseek_v4_flash.yaml", "deepseek_v4_pro.yaml"],
)
def test_yaml_does_not_set_retired_fields(parse_yaml_fn, yaml_name: str) -> None:
    """If a V4 YAML still references a retired field (e.g. someone
    forgot to update a downstream yaml) we want a loud schema error,
    not silent acceptance.

    The dataclass would now reject these as ``TypeError`` (unexpected
    keyword), but we filter unknown keys for forward-compat in
    ``_config_kwargs_from_yaml``; here we explicitly enforce the
    contract on the parsed dict instead.
    """
    parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
    for retired in ("v4_use_custom_mtp_block", "mtp_compress_ratios"):
        assert retired not in parsed, (
            f"{yaml_name} still references retired field {retired!r} "
            "(plan-2 P17). Remove the line or update training scripts."
        )


# ---------------------------------------------------------------------------
# V4-specific schema fields the runtime depends on (D5 / D6 hygiene)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    [
        # Hyper-Connection
        "hc_mult",
        "hc_eps",
        # Hybrid attention extras
        "compress_ratios",
        "compress_rope_theta",
        "attn_sliding_window",
        "attn_sink",
        # Grouped low-rank O projection
        "o_groups",
        "o_lora_rank",
        # MoE-specific extras
        "num_hash_layers",
        "swiglu_limit",
    ],
)
def test_v4_config_carries_runtime_field(field_name: str) -> None:
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    names = {f.name for f in fields(DeepSeekV4TransformerConfig)}
    assert field_name in names, (
        f"DeepSeekV4TransformerConfig must declare {field_name!r}; "
        "removing it silently breaks the V4 runtime."
    )


# ---------------------------------------------------------------------------
# Provider singleton (P18 D1) — same instance per config
# ---------------------------------------------------------------------------


def test_resolve_v4_provider_caches_per_config():
    from primus.backends.megatron.core.models.deepseek_v4.build_context import (
        resolve_v4_provider,
    )
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    # Build a minimal V4 config; we only need the dataclass instance —
    # the provider does not touch most fields at construction.
    cfg_a = DeepSeekV4TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
    )
    cfg_b = DeepSeekV4TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
    )

    p_a1 = resolve_v4_provider(cfg_a)
    p_a2 = resolve_v4_provider(cfg_a)
    p_b1 = resolve_v4_provider(cfg_b)

    assert p_a1 is p_a2, "resolve_v4_provider must reuse the cached provider for the same config (P18 D1)."
    assert p_a1 is not p_b1, (
        "Different config instances should each get their own provider so "
        "test isolation and runtime overrides are not silently shared."
    )


# ---------------------------------------------------------------------------
# Provider activation_func helper (P18 D2)
# ---------------------------------------------------------------------------


def test_v4_mlp_activation_func_respects_use_te_activation_func() -> None:
    """``provider.v4_mlp_activation_func()`` returns ``None`` when the
    config keeps Megatron's eager activation path (the V4 default,
    needed for clamped-SwiGLU); only when the user opts into TE does
    the spec slot carry the TE class.
    """
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        DeepSeekV4SpecProvider,
        TEActivationOp,
    )

    class _FakeCfg:
        use_te_activation_func = False

    p = DeepSeekV4SpecProvider(config=_FakeCfg())
    assert p.v4_mlp_activation_func() is None, (
        "Default V4 path must leave activation_func slot empty so Megatron "
        "MLP uses config.activation_func (clamped-SwiGLU)."
    )

    class _TeCfg:
        use_te_activation_func = True

    p2 = DeepSeekV4SpecProvider(config=_TeCfg())
    assert p2.v4_mlp_activation_func() is TEActivationOp, (
        "When the user opts into TE activation, the spec slot must carry "
        "TEActivationOp so Megatron MLP instantiates the TE-fused path."
    )
