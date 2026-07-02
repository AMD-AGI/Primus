###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-3 P23 — Turbo DeepEP dispatcher in V4 specs.

Today the V4 spec build captures
:class:`megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher`
at module-import time, while the Primus turbo patch
(``primus.backends.megatron.patches.turbo.moe_dispatcher_patches``) only
rebinds that module attribute at ``before_train``.  The patch fires
*after* V4 spec build, so V4 silently runs the upstream
:class:`MoEFlexTokenDispatcher` even when the user opted into
``use_turbo_deepep=True``.

P23 fixes this by resolving the dispatcher class **at V4 spec-build time**
through :func:`primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs._pick_v4_dispatcher_cls`,
which checks ``args.enable_primus_turbo``, ``args.use_turbo_deepep``,
``args.tensor_model_parallel_size`` and the import status of the
``primus_turbo`` package.  Two side-effects ensure the rest of the
stack agrees:

* :func:`_maybe_plumb_v4_turbo_deepep_args` (in
  ``deepseek_v4_builders.py``) mutates ``args.moe_enable_deepep`` and
  ``args.moe_token_dispatcher_type`` BEFORE
  ``core_transformer_config_from_args``, so the V4 ``config`` carries
  the right ``moe_token_dispatcher_type``.
* :meth:`DeepseekV4MoE._resolve_dispatcher_type_from_spec` recognises
  ``PrimusTurboDeepEPTokenDispatcher`` (by class name, no
  ``primus_turbo`` import required) and returns ``"flex"``, so the
  per-layer log line ``"dispatcher active via …"`` reports the
  correct type.

Test gates exercised here:

* **G20a — gating predicate**
  :func:`is_v4_turbo_deepep_active` matches the four conditions used
  by the upstream patch.
* **G20b — args plumbing**
  :func:`_maybe_plumb_v4_turbo_deepep_args` mutates only when all
  gates pass; respects an explicit ``"allgather"`` opt-in.
* **G20c — class resolution**
  :func:`_pick_v4_dispatcher_cls` returns the right
  ``(cls, type_name)`` tuple for each
  ``(config.moe_token_dispatcher_type, args)`` combination.
* **G20d — V4 MoE resolver recognises Turbo class**
  :meth:`DeepseekV4MoE._resolve_dispatcher_type_from_spec` returns
  ``"flex"`` for a class named
  ``PrimusTurboDeepEPTokenDispatcher`` (mocked when ``primus_turbo``
  is not installed).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.models.deepseek_v4 import (
    deepseek_v4_builders,
    deepseek_v4_layer_specs,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
    _pick_v4_dispatcher_cls,
    is_v4_turbo_deepep_active,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import DeepseekV4MoE

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_args(
    *,
    enable_primus_turbo: bool = False,
    use_turbo_deepep: bool = False,
    tensor_model_parallel_size: int = 1,
    moe_token_dispatcher_type: str = "alltoall",
    moe_enable_deepep: bool = False,
):
    """Minimal ``args`` namespace mirroring Megatron's runtime args."""
    return SimpleNamespace(
        enable_primus_turbo=enable_primus_turbo,
        use_turbo_deepep=use_turbo_deepep,
        tensor_model_parallel_size=tensor_model_parallel_size,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_enable_deepep=moe_enable_deepep,
    )


def _make_cfg(*, moe_token_dispatcher_type: str = "alltoall"):
    """Minimal ``config`` namespace with the dispatcher-type field only."""
    return SimpleNamespace(moe_token_dispatcher_type=moe_token_dispatcher_type)


@pytest.fixture
def primus_turbo_available(monkeypatch):
    """Pretend the ``primus_turbo`` package is importable."""
    real_find_spec = deepseek_v4_layer_specs.importlib.util.find_spec

    def _fake(name):
        if name == "primus_turbo":
            return MagicMock()
        return real_find_spec(name)

    monkeypatch.setattr(deepseek_v4_layer_specs.importlib.util, "find_spec", _fake)


@pytest.fixture
def primus_turbo_unavailable(monkeypatch):
    """Pretend the ``primus_turbo`` package is NOT importable."""
    real_find_spec = deepseek_v4_layer_specs.importlib.util.find_spec

    def _fake(name):
        if name == "primus_turbo":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(deepseek_v4_layer_specs.importlib.util, "find_spec", _fake)


# ``type(..., (), {})`` produces a class whose ``__name__`` is the
# string we pass — using a plain ``class`` block would set ``__name__``
# to the local symbol (e.g. ``_FakeTurboDispatcher``) regardless of
# any class-level ``__name__`` assignment.  Only the class name
# matters for ``_resolve_dispatcher_type_from_spec``; construction is
# never exercised in these unit tests.
_FakeTurboDispatcher = type("PrimusTurboDeepEPTokenDispatcher", (), {})


@pytest.fixture
def fake_turbo_class(monkeypatch):
    """Inject a fake ``PrimusTurboDeepEPTokenDispatcher`` import target."""
    monkeypatch.setattr(
        deepseek_v4_layer_specs,
        "_import_primus_turbo_deepep_dispatcher_cls",
        lambda: _FakeTurboDispatcher,
    )
    return _FakeTurboDispatcher


# ---------------------------------------------------------------------------
# G20a — gating predicate (is_v4_turbo_deepep_active)
# ---------------------------------------------------------------------------


class TestGatingPredicate:
    """Mirror of the conditions enforced by ``moe_dispatcher_patches``."""

    def test_all_gates_open(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=1,
        )
        assert is_v4_turbo_deepep_active(args) is True

    def test_primus_turbo_missing(self, primus_turbo_unavailable):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=1,
        )
        assert is_v4_turbo_deepep_active(args) is False

    def test_enable_primus_turbo_off(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=False,
            use_turbo_deepep=True,
            tensor_model_parallel_size=1,
        )
        assert is_v4_turbo_deepep_active(args) is False

    def test_use_turbo_deepep_off(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=False,
            tensor_model_parallel_size=1,
        )
        assert is_v4_turbo_deepep_active(args) is False

    def test_tp_gt_1_blocks(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=2,
        )
        assert is_v4_turbo_deepep_active(args) is False


# ---------------------------------------------------------------------------
# G20b — args plumbing (_maybe_plumb_v4_turbo_deepep_args)
# ---------------------------------------------------------------------------


class TestArgsPlumbing:
    """The plumbing helper sets ``moe_enable_deepep`` + dispatcher type."""

    def test_plumb_when_active(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            moe_token_dispatcher_type="alltoall",
            moe_enable_deepep=False,
        )
        deepseek_v4_builders._maybe_plumb_v4_turbo_deepep_args(args)
        assert args.moe_enable_deepep is True
        assert args.moe_token_dispatcher_type == "flex"

    def test_no_plumb_when_inactive(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=False,
            use_turbo_deepep=True,
            moe_token_dispatcher_type="alltoall",
            moe_enable_deepep=False,
        )
        deepseek_v4_builders._maybe_plumb_v4_turbo_deepep_args(args)
        assert args.moe_enable_deepep is False
        assert args.moe_token_dispatcher_type == "alltoall"

    def test_no_plumb_when_package_missing(self, primus_turbo_unavailable):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            moe_token_dispatcher_type="alltoall",
        )
        deepseek_v4_builders._maybe_plumb_v4_turbo_deepep_args(args)
        assert args.moe_token_dispatcher_type == "alltoall"

    def test_explicit_allgather_preserved(self, primus_turbo_available):
        """User opted into ``allgather`` — never silently overridden."""
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            moe_token_dispatcher_type="allgather",
        )
        deepseek_v4_builders._maybe_plumb_v4_turbo_deepep_args(args)
        # ``moe_enable_deepep`` may flip (Turbo wants it for any deepep
        # path), but the user's dispatcher choice is preserved.
        assert args.moe_token_dispatcher_type == "allgather"

    def test_idempotent_when_already_flex(self, primus_turbo_available):
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            moe_token_dispatcher_type="flex",
            moe_enable_deepep=True,
        )
        deepseek_v4_builders._maybe_plumb_v4_turbo_deepep_args(args)
        assert args.moe_token_dispatcher_type == "flex"
        assert args.moe_enable_deepep is True


# ---------------------------------------------------------------------------
# G20c — class resolution (_pick_v4_dispatcher_cls)
# ---------------------------------------------------------------------------


class TestPickDispatcherCls:
    """Exhaustive class-resolution table."""

    def test_alltoall_default(self):
        cfg = _make_cfg(moe_token_dispatcher_type="alltoall")
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=_make_args())
        assert cls is MoEAlltoAllTokenDispatcher
        assert type_name == "alltoall"

    def test_allgather_explicit(self):
        cfg = _make_cfg(moe_token_dispatcher_type="allgather")
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=_make_args())
        assert cls is MoEAllGatherTokenDispatcher
        assert type_name == "allgather"

    def test_flex_without_turbo(self):
        cfg = _make_cfg(moe_token_dispatcher_type="flex")
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=_make_args())
        assert cls is MoEFlexTokenDispatcher
        assert type_name == "flex"

    def test_flex_with_turbo_active(self, primus_turbo_available, fake_turbo_class):
        cfg = _make_cfg(moe_token_dispatcher_type="flex")
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=1,
        )
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=args)
        assert cls is fake_turbo_class
        assert type_name == "flex"

    def test_flex_with_turbo_active_but_class_missing(self, primus_turbo_available, monkeypatch):
        """Gracefully fall back to the upstream class with a warning."""
        monkeypatch.setattr(
            deepseek_v4_layer_specs,
            "_import_primus_turbo_deepep_dispatcher_cls",
            lambda: None,
        )
        cfg = _make_cfg(moe_token_dispatcher_type="flex")
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=1,
        )
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=args)
        assert cls is MoEFlexTokenDispatcher
        assert type_name == "flex"

    def test_flex_with_turbo_inactive_tp_gt_1(self, primus_turbo_available, fake_turbo_class):
        """TP > 1 keeps the upstream Flex dispatcher."""
        cfg = _make_cfg(moe_token_dispatcher_type="flex")
        args = _make_args(
            enable_primus_turbo=True,
            use_turbo_deepep=True,
            tensor_model_parallel_size=2,
        )
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=args)
        assert cls is MoEFlexTokenDispatcher
        assert type_name == "flex"

    def test_unknown_type_falls_back_to_alltoall(self, caplog):
        cfg = _make_cfg(moe_token_dispatcher_type="random_string")
        with caplog.at_level("WARNING", logger=deepseek_v4_layer_specs.__name__):
            cls, type_name = _pick_v4_dispatcher_cls(cfg, args=_make_args())
        assert cls is MoEAlltoAllTokenDispatcher
        assert type_name == "alltoall"
        assert any("unsupported moe_token_dispatcher_type" in rec.message for rec in caplog.records)

    def test_args_none_falls_back_when_megatron_not_initialised(
        self, primus_turbo_available, fake_turbo_class, monkeypatch
    ):
        """``args=None`` + no Megatron args available → non-turbo branch."""

        def _raises():
            raise RuntimeError("Megatron not initialised in unit test")

        # Patch the lazy import target inside the helper.
        import megatron.training as _megatron_training  # noqa: WPS433

        monkeypatch.setattr(_megatron_training, "get_args", _raises)
        cfg = _make_cfg(moe_token_dispatcher_type="flex")
        cls, type_name = _pick_v4_dispatcher_cls(cfg, args=None)
        assert cls is MoEFlexTokenDispatcher
        assert type_name == "flex"


# ---------------------------------------------------------------------------
# G20d — V4 MoE resolver recognises Turbo class
# ---------------------------------------------------------------------------


class TestV4MoEResolver:
    """``DeepseekV4MoE._resolve_dispatcher_type_from_spec`` must label
    :class:`PrimusTurboDeepEPTokenDispatcher` as ``"flex"``."""

    def test_turbo_class_resolves_to_flex(self):
        spec = ModuleSpec(module=_FakeTurboDispatcher)
        assert DeepseekV4MoE._resolve_dispatcher_type_from_spec(spec) == "flex"

    def test_turbo_class_bare_resolves_to_flex(self):
        # The resolver also accepts a bare class (some call sites pass
        # the type directly, not a ModuleSpec).
        assert DeepseekV4MoE._resolve_dispatcher_type_from_spec(_FakeTurboDispatcher) == "flex"

    def test_alltoall_unchanged(self):
        spec = ModuleSpec(module=MoEAlltoAllTokenDispatcher)
        assert DeepseekV4MoE._resolve_dispatcher_type_from_spec(spec) == "alltoall"

    def test_flex_unchanged(self):
        spec = ModuleSpec(module=MoEFlexTokenDispatcher)
        assert DeepseekV4MoE._resolve_dispatcher_type_from_spec(spec) == "flex"

    def test_allgather_unchanged(self):
        spec = ModuleSpec(module=MoEAllGatherTokenDispatcher)
        assert DeepseekV4MoE._resolve_dispatcher_type_from_spec(spec) == "allgather"

    def test_unknown_class_falls_back_to_alltoall(self, caplog):
        class _Unknown:
            __name__ = "_Unknown"

        spec = ModuleSpec(module=_Unknown)
        with caplog.at_level("WARNING"):
            result = DeepseekV4MoE._resolve_dispatcher_type_from_spec(spec)
        assert result == "alltoall"
        assert any("unsupported dispatcher module" in rec.message for rec in caplog.records)
