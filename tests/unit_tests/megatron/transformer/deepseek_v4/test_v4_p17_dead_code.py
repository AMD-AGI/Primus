###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-2 P17 dead-code audit (gate G14).

These tests guarantee the cleanup landed in plan-2 P17 stays clean —
they are intentionally light-weight (no torch.distributed, no GPU) so
they run on every PR. They cover:

* The legacy primus-owned MTP block module is gone (no
  ``deepseek_v4_mtp.py`` file, no import path).
* The ``v4_use_custom_mtp_block`` and ``mtp_compress_ratios`` config
  fields are gone from :class:`DeepSeekV4TransformerConfig`.
* The retired CSA / HCA standalone modules
  (``csa_attention.py`` / ``hca_attention.py``) are gone.
* The runtime ``decoder._v4_token_ids`` stash has zero AST-level
  references in the V4 source tree (docstrings are exempt).
* The block-level ``_RMSNorm`` duplicates have been replaced by the
  shared :class:`LocalRMSNorm`; no V4 module re-defines a class called
  ``_RMSNorm``.
* The ``DeepseekV4MTPBlock`` symbol is **not** re-exported from the
  V4 model package (already covered by ``test_v4_mtp.py``; mirrored
  here so the dead-code audit is self-contained).
* The ``compress_ratios`` yaml comments document the canonical
  ``4 = CSA`` / ``128 = HCA`` mapping.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
_V4_MODELS = _REPO_ROOT / "primus" / "backends" / "megatron" / "core" / "models" / "deepseek_v4"
_V4_TRANSFORMER = _REPO_ROOT / "primus" / "backends" / "megatron" / "core" / "transformer"
_YAML_DIR = _REPO_ROOT / "primus" / "configs" / "models" / "megatron"


# ---------------------------------------------------------------------------
# Retired files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relpath",
    [
        # Legacy primus-owned MTP head — replaced by spec-based path in P16
        # and deleted in P17.
        "primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_mtp.py",
        # Standalone CSA / HCA attention classes — folded into
        # DeepseekV4Attention in P13 and deleted then; re-asserted here.
        "primus/backends/megatron/core/transformer/csa_attention.py",
        "primus/backends/megatron/core/transformer/hca_attention.py",
    ],
)
def test_retired_files_are_gone(relpath: str) -> None:
    """The dead modules must not be present on disk."""
    p = _REPO_ROOT / relpath
    assert not p.exists(), f"plan-2 P17 expects {relpath} to be removed; still present at {p}"


def test_legacy_mtp_block_import_path_does_not_resolve() -> None:
    """Importing the retired module must raise ``ImportError``."""
    with pytest.raises(ImportError):
        importlib.import_module("primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp")


# ---------------------------------------------------------------------------
# Retired config fields / package surface
# ---------------------------------------------------------------------------


def test_v4_config_no_legacy_mtp_fields() -> None:
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    fields = set(DeepSeekV4TransformerConfig.__dataclass_fields__.keys())
    assert "v4_use_custom_mtp_block" not in fields, (
        "v4_use_custom_mtp_block must be removed from DeepSeekV4TransformerConfig " "(plan-2 P17)."
    )
    assert "mtp_compress_ratios" not in fields, (
        "mtp_compress_ratios was only consumed by the legacy MTP block; "
        "it must be removed alongside the block (plan-2 P17)."
    )


def test_package_surface_drops_legacy_mtp_block() -> None:
    pkg = importlib.import_module("primus.backends.megatron.core.models.deepseek_v4")
    assert "DeepseekV4MTPBlock" not in getattr(
        pkg, "__all__", []
    ), "Package __all__ must not re-export the retired DeepseekV4MTPBlock."
    assert not hasattr(
        pkg, "DeepseekV4MTPBlock"
    ), "Package must not expose the retired DeepseekV4MTPBlock attribute."


# ---------------------------------------------------------------------------
# AST audits — V4 sources
# ---------------------------------------------------------------------------


def _v4_python_sources() -> list[Path]:
    """All .py files under V4 model and shared transformer modules."""
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


def test_no_runtime_v4_token_ids_attribute_access() -> None:
    """``_v4_token_ids`` must not appear as an AST attribute or a name
    binding anywhere in the V4 source tree.

    Docstring mentions are **exempt** — they document the legacy stash
    has been removed; the audit walks the syntax tree, not the source
    text, so they are naturally ignored.
    """
    offenders = []
    for path in _v4_python_sources():
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_v4_token_ids":
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', '?')} attribute access"
                )
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "_v4_token_ids":
                        offenders.append(
                            f"{path.relative_to(_REPO_ROOT)}:{getattr(target, 'lineno', '?')} setattr"
                        )
            if isinstance(node, ast.Name) and node.id == "_v4_token_ids":
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', '?')} name ref")
    assert not offenders, (
        "Plan-2 P15 retired the decoder._v4_token_ids stash; the runtime "
        "tree must not regress.\nOffenders:\n  " + "\n  ".join(offenders)
    )


def test_no_v4_module_redefines_local_rmsnorm() -> None:
    """Plan-2 P17 dedups ``_RMSNorm`` into ``LocalRMSNorm``; no other
    V4 module may declare a ``class _RMSNorm`` shadow definition.

    The shared helper lives in
    ``primus.backends.megatron.core.transformer.local_rmsnorm`` and
    exposes the canonical class as ``LocalRMSNorm`` — so the only
    ``_RMSNorm`` references that survive in the V4 tree are doc /
    comment strings, which the AST walk ignores.
    """
    offenders = []
    for path in _v4_python_sources():
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "_RMSNorm":
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', '?')}")
    assert not offenders, (
        "Plan-2 P17 dedups _RMSNorm; no V4 module should re-declare it.\n"
        "Use primus.backends.megatron.core.transformer.local_rmsnorm.LocalRMSNorm "
        "instead.\nOffenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# YAML comments — canonical compress_ratios mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    ["deepseek_v4_flash.yaml", "deepseek_v4_pro.yaml", "deepseek_v4_base.yaml"],
)
def test_yaml_documents_canonical_compress_ratio_mapping(yaml_name: str) -> None:
    """Plan-2 P17 fixes the inverted comment in the V4 yamls so the
    schedule list says ``4 = CSA`` / ``128 = HCA`` (matching the
    DeepseekV4Attention dispatch in ``deepseek_v4_attention.py``).

    Each yaml that ships a ``compress_ratios`` schedule (or hands the
    schedule down to per-variant yamls) must spell the mapping
    correctly, so accidental sed-style edits later don't silently
    invert the labels.
    """
    text = (_YAML_DIR / yaml_name).read_text()
    assert "compress_ratios" in text
    assert "4 = CSA" in text or "4   = CSA" in text, (
        f"{yaml_name}: must document `4 = CSA` (overlap, per-query top-K from "
        "compressed pool); current comment is missing or inverted."
    )
    assert "128 = HCA" in text or "128 = HCA" in text, (
        f"{yaml_name}: must document `128 = HCA` (non-overlap, full visibility " "over compressed pool)."
    )
