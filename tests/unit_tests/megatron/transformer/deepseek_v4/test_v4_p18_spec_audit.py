###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-2 P18 — V4 spec system audit (light-weight CPU tests).

This file complements ``test_deepseek_v4_yaml.py`` (the schema gate
G1) with structural / static checks on the V4 spec system itself:

* No V4 module re-instantiates ``DeepSeekV4SpecProvider`` directly —
  the only allowed call site is ``build_context.resolve_v4_provider``.
  This enforces the P18 D1 audit ("provider built **once** per
  builder call and threaded down").
* The package surface ``__init__`` exposes only the live classes
  (``DeepseekV4MTPBlock`` was retired in P17; cross-checked here).
* The V4 layer-spec / MTP-spec helpers route their activation_func
  via ``provider.v4_mlp_activation_func()`` — not the unconditional
  ``provider.activation_func()``. (P18 D2.)

We deliberately keep these checks AST-only so they run without torch.
The unit tests in ``test_deepseek_v4_yaml.py`` cover the runtime
behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
_V4_MODELS = _REPO_ROOT / "primus" / "backends" / "megatron" / "core" / "models" / "deepseek_v4"
_V4_TRANSFORMER = _REPO_ROOT / "primus" / "backends" / "megatron" / "core" / "transformer"
_PROVIDER_PATH = (
    _REPO_ROOT
    / "primus"
    / "backends"
    / "megatron"
    / "core"
    / "extensions"
    / "transformer_engine_spec_provider.py"
)
_BUILD_CONTEXT_PATH = _V4_MODELS / "build_context.py"


def _v4_python_sources() -> list[Path]:
    """Sources subject to the P18 audit (V4 model + V4 transformer
    helpers). The provider class itself is excluded — it is the
    *target* of the audit, not a consumer."""
    return sorted(
        list(_V4_MODELS.glob("*.py"))
        + [
            _V4_TRANSFORMER / "deepseek_v4_attention.py",
            _V4_TRANSFORMER / "compressor.py",
            _V4_TRANSFORMER / "indexer.py",
            _V4_TRANSFORMER / "dual_rope.py",
            _V4_TRANSFORMER / "hyper_connection.py",
            _V4_TRANSFORMER / "attn_sink.py",
            _V4_TRANSFORMER / "sliding_window_kv.py",
            _V4_TRANSFORMER / "clamped_swiglu.py",
            _V4_TRANSFORMER / "local_rmsnorm.py",
        ]
    )


# ---------------------------------------------------------------------------
# D1 — provider singleton
# ---------------------------------------------------------------------------


def test_no_direct_DeepSeekV4SpecProvider_construction_outside_build_context() -> None:
    """The only place that may construct ``DeepSeekV4SpecProvider``
    is ``build_context.resolve_v4_provider`` (which caches the
    instance on the config). Every other consumer must call the
    helper instead.

    The audit walks the V4 source AST and rejects any direct
    ``DeepSeekV4SpecProvider(...)`` ``Call`` node.
    """
    offenders = []
    for path in _v4_python_sources():
        if not path.exists():
            continue
        if path.resolve() == _BUILD_CONTEXT_PATH.resolve():
            # build_context.py is the only allowed instantiator.
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "DeepSeekV4SpecProvider":
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', '?')}")
            elif isinstance(func, ast.Attribute) and func.attr == "DeepSeekV4SpecProvider":
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', '?')}")
    assert not offenders, (
        "Plan-2 P18 D1: V4 modules must call resolve_v4_provider(config) "
        "instead of constructing DeepSeekV4SpecProvider() directly.\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )


def test_build_context_module_exposes_resolve_helper() -> None:
    """``resolve_v4_provider`` is the entry point — make sure the
    helper survives renames."""
    src = _BUILD_CONTEXT_PATH.read_text()
    assert "def resolve_v4_provider" in src, "build_context.py must expose `resolve_v4_provider(config)`."
    # And exports it.
    assert '"resolve_v4_provider"' in src, (
        "build_context.__all__ must list `resolve_v4_provider` so consumers " "import the public name."
    )


# ---------------------------------------------------------------------------
# D2 — activation_func consistency
# ---------------------------------------------------------------------------


def test_provider_exposes_v4_mlp_activation_func_helper() -> None:
    """The V4-aware helper exists in the provider."""
    src = _PROVIDER_PATH.read_text()
    assert "def v4_mlp_activation_func" in src, (
        "DeepSeekV4SpecProvider must expose v4_mlp_activation_func() so "
        "spec builders return None when use_te_activation_func is False."
    )


def test_v4_specs_use_v4_mlp_activation_func_helper() -> None:
    """Layer-spec / block.py SharedExpertMLP wiring must call the
    V4-aware helper (P18 D2). The unconditional
    ``provider.activation_func()`` form is forbidden in spec builders."""
    layer_specs = (_V4_MODELS / "deepseek_v4_layer_specs.py").read_text()
    block_src = (_V4_MODELS / "deepseek_v4_block.py").read_text()
    for src, name in ((layer_specs, "deepseek_v4_layer_specs.py"), (block_src, "deepseek_v4_block.py")):
        assert "v4_mlp_activation_func" in src, (
            f"{name}: spec builder must call provider.v4_mlp_activation_func(), "
            "not the unconditional provider.activation_func() (which silently "
            "passes a class into MLPSubmodules.activation_func)."
        )


# ---------------------------------------------------------------------------
# Package surface — D1 / P17 cross-check
# ---------------------------------------------------------------------------


def test_v4_package_init_exports_only_live_symbols() -> None:
    init_src = (_V4_MODELS / "__init__.py").read_text()
    assert "DeepseekV4MTPBlock" not in init_src.split("__all__")[1] if "__all__" in init_src else True
    # Stronger AST check below.
    tree = ast.parse(init_src)
    all_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_names.append(elt.value)
    assert (
        "DeepseekV4MTPBlock" not in all_names
    ), "Package __all__ must not re-export the retired DeepseekV4MTPBlock."


# ---------------------------------------------------------------------------
# Eager-construction audit — only spec-built submodules in spec-driven init
# ---------------------------------------------------------------------------


_SPEC_REPLACEABLE_NAMES = (
    # Norms / linears / activations are the things a downstream user
    # might plausibly want to swap via spec — they should always be
    # built via build_module, never instantiated directly inside
    # __init__ for spec-driven paths.
    "TENorm",
    "TEColumnParallelLinear",
    "TERowParallelLinear",
    "TELinear",
    "TEActivationOp",
)


@pytest.mark.parametrize(
    "rel_path",
    [
        "primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py",
        "primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_mtp_specs.py",
    ],
)
def test_spec_builders_do_not_eagerly_construct_te_modules(rel_path: str) -> None:
    """Spec builders should only emit ``ModuleSpec(module=...)``
    references — they must not eagerly construct TE modules in spec
    code. The runtime caller resolves the spec via ``build_module``.

    This is the static counterpart to D1: "no spec-replaceable
    module instantiated outside ``build_module``" inside the spec
    helpers themselves.
    """
    src = (_REPO_ROOT / rel_path).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = None
        if isinstance(func, ast.Name):
            target = func.id
        elif isinstance(func, ast.Attribute):
            target = func.attr
        if target in _SPEC_REPLACEABLE_NAMES:
            offenders.append(f"{rel_path}:{getattr(node, 'lineno', '?')} {target}(...)")
    assert not offenders, (
        "Spec builders must use ModuleSpec(module=...) for TE modules "
        "instead of constructing them eagerly.\nOffenders:\n  " + "\n  ".join(offenders)
    )
