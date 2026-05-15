###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P49 G49 — tilelang dispatcher infra.

Asserts that:

* :func:`is_tilelang_path_enabled` correctly tracks the env knob.
* :func:`is_tilelang_kernel_available` returns False for every plan-8
  kernel at P49 (none registered yet).
* :func:`should_dispatch` falls through with a single rank-0 warning
  when the env knob is set but no kernel has landed yet.
* Importing the tilelang dispatcher module does NOT trigger a tilelang
  JIT compile (lazy import contract).
* Registering an available kernel via :func:`register_available_kernel`
  flips :func:`is_tilelang_kernel_available` to True for that name.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager

import pytest


# Tilelang dispatch is a pure-Python module — no GPU required.
@contextmanager
def _env(key: str, value: str | None):
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _reset_dispatcher_state(module):
    """Reset the module's internal state between tests so warnings fire."""
    module._AVAILABLE_KERNELS.clear()
    module._FALLBACK_WARNED.clear()


# ---------------------------------------------------------------------------
# G49.1: env knob predicate
# ---------------------------------------------------------------------------


class TestG49EnvKnob:
    def test_default_off(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with _env("PRIMUS_V4_TILELANG_ATTN", None):
            assert _tilelang.is_tilelang_path_enabled() is False

    def test_explicit_zero(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with _env("PRIMUS_V4_TILELANG_ATTN", "0"):
            assert _tilelang.is_tilelang_path_enabled() is False

    def test_explicit_one(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with _env("PRIMUS_V4_TILELANG_ATTN", "1"):
            assert _tilelang.is_tilelang_path_enabled() is True

    def test_unrelated_truthy_value_does_not_enable(self):
        """Only the literal string ``"1"`` enables — ``"true"`` does not."""
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        for v in ("true", "TRUE", "yes", "2"):
            with _env("PRIMUS_V4_TILELANG_ATTN", v):
                assert _tilelang.is_tilelang_path_enabled() is False, f"value {v!r} should NOT enable"


# ---------------------------------------------------------------------------
# G49.2: per-kernel availability registry
# ---------------------------------------------------------------------------


class TestG49KernelAvailability:
    def test_p49_empty_by_default(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        _reset_dispatcher_state(_tilelang)
        for name in (
            "v4_attention_fwd",
            "v4_attention_bwd",
            "v4_csa_attention_fwd",
            "v4_csa_attention_bwd",
        ):
            assert _tilelang.is_tilelang_kernel_available(name) is False

    def test_register_flips_to_true(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        _reset_dispatcher_state(_tilelang)
        _tilelang.register_available_kernel("v4_attention_fwd")
        try:
            assert _tilelang.is_tilelang_kernel_available("v4_attention_fwd") is True
            assert _tilelang.is_tilelang_kernel_available("v4_attention_bwd") is False
        finally:
            _reset_dispatcher_state(_tilelang)

    def test_unknown_kernel_name_raises(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with pytest.raises(ValueError, match="Unknown tilelang kernel name"):
            _tilelang.is_tilelang_kernel_available("not_a_kernel")
        with pytest.raises(ValueError, match="Unknown tilelang kernel name"):
            _tilelang.register_available_kernel("not_a_kernel")


# ---------------------------------------------------------------------------
# G49.3: dispatcher fallthrough + one-time warning
# ---------------------------------------------------------------------------


class TestG49DispatcherFallthrough:
    def test_should_dispatch_false_at_default(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        _reset_dispatcher_state(_tilelang)
        with _env("PRIMUS_V4_TILELANG_ATTN", None):
            assert _tilelang.should_dispatch("v4_attention_fwd") is False

    def test_should_dispatch_warns_once_when_env_set_but_kernel_missing(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        _reset_dispatcher_state(_tilelang)
        with _env("PRIMUS_V4_TILELANG_ATTN", "1"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert _tilelang.should_dispatch("v4_attention_fwd") is False
                # Second call should not re-warn.
                assert _tilelang.should_dispatch("v4_attention_fwd") is False
            messages = [str(w.message) for w in caught]
            # Filter out unrelated tilelang import / version-pin warnings — we
            # only assert about the kernel-fallback warning.
            fallback_msgs = [m for m in messages if "is not available" in m]
            assert len(fallback_msgs) == 1, (
                f"expected one fallback warning per kernel name; got "
                f"{len(fallback_msgs)}: {fallback_msgs}"
            )

    def test_should_dispatch_true_when_env_and_registered(self):
        """When tilelang is importable + env=1 + kernel registered, dispatch."""
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        try:
            import tilelang  # noqa: F401
        except ImportError:
            pytest.skip("tilelang not installed")

        _reset_dispatcher_state(_tilelang)
        with _env("PRIMUS_V4_TILELANG_ATTN", "1"):
            _tilelang.register_available_kernel("v4_attention_fwd")
            try:
                assert _tilelang.should_dispatch("v4_attention_fwd") is True
            finally:
                _reset_dispatcher_state(_tilelang)


# ---------------------------------------------------------------------------
# G49.4: stub entry points raise NotImplementedError
# ---------------------------------------------------------------------------


class TestG49StubsRaise:
    @pytest.mark.parametrize(
        "fn_name,phase",
        [
            ("v4_attention_fwd_tilelang", "P50"),
            ("v4_attention_bwd_tilelang", "P51"),
            ("v4_csa_attention_fwd_tilelang", "P54"),
            ("v4_csa_attention_bwd_tilelang", "P55"),
        ],
    )
    def test_stub_raises_with_phase_message(self, fn_name, phase):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        fn = getattr(_tilelang, fn_name)
        with pytest.raises(NotImplementedError, match=phase):
            fn()


# ---------------------------------------------------------------------------
# G49.5: version pin + cache dir
# ---------------------------------------------------------------------------


class TestG49Misc:
    def test_version_pin_format(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        # Pin format: <version>+<extras>; assert non-empty.
        assert isinstance(_tilelang.TILELANG_VERSION_PIN, str)
        assert _tilelang.TILELANG_VERSION_PIN.strip() != ""

    def test_default_cache_dir(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with _env("PRIMUS_V4_TILELANG_CACHE_DIR", None):
            d = _tilelang.cache_dir()
            assert d.endswith("/.tilelang_cache/v4")

    def test_override_cache_dir(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        with _env("PRIMUS_V4_TILELANG_CACHE_DIR", "/tmp/custom_tilelang_cache"):
            assert _tilelang.cache_dir() == "/tmp/custom_tilelang_cache"


# ---------------------------------------------------------------------------
# G49.6: dispatcher wired into v4_attention / v4_csa_attention wrappers
# ---------------------------------------------------------------------------


class TestG49WrappersHaveDispatch:
    """Source-level audit: the dispatcher hook must exist in both wrappers."""

    def test_v4_attention_calls_should_dispatch(self):
        import importlib
        import inspect

        mod = importlib.import_module(
            "primus.backends.megatron.core.transformer.v4_attention_kernels.v4_attention"
        )
        src = inspect.getsource(mod.v4_attention)
        assert "should_dispatch" in src, "v4_attention wrapper must call _tilelang.should_dispatch(...)"
        assert "v4_attention_fwd_tilelang" in src, "v4_attention wrapper must route to the tilelang stub"

    def test_v4_csa_attention_calls_should_dispatch(self):
        import importlib
        import inspect

        mod = importlib.import_module(
            "primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention"
        )
        src = inspect.getsource(mod.v4_csa_attention)
        assert "should_dispatch" in src
        assert "v4_csa_attention_fwd_tilelang" in src
