###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from unittest.mock import Mock

from primus.core.patches.context import PatchContext
from primus.core.patches.patch import FunctionPatch


class TestFunctionPatch:
    def test_applies_to_basic(self):
        patch = FunctionPatch(id="test", description="", handler=Mock())
        ctx = PatchContext(backend="megatron", phase="setup")

        # Should apply with no constraints
        assert patch.applies_to(ctx)

    def test_applies_to_backend_phase(self):
        patch = FunctionPatch(
            id="test",
            description="",
            handler=Mock(),
            backend="megatron",
            phase="setup",
        )

        assert patch.applies_to(PatchContext(backend="megatron", phase="setup"))
        assert not patch.applies_to(PatchContext(backend="torchtitan", phase="setup"))
        assert not patch.applies_to(PatchContext(backend="megatron", phase="run"))

    def test_applies_to_versions_wildcard(self):
        patch = FunctionPatch(
            id="test",
            description="",
            handler=Mock(),
            backend_version_patterns=["0.8.*"],
            primus_version_patterns=["1.0.0"],
        )

        # Match
        ctx = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.8.5",
            primus_version="1.0.0",
        )
        assert patch.applies_to(ctx)

        # Mismatch backend_version
        ctx_bad_backend = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.9.0",
            primus_version="1.0.0",
        )
        assert not patch.applies_to(ctx_bad_backend)

        # Mismatch primus_version
        ctx_bad_primus = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.8.5",
            primus_version="1.0.1",
        )
        assert not patch.applies_to(ctx_bad_primus)

    def test_applies_to_versions_with_range_and_comparators(self):
        # Use the extended range syntax supported by version_in_range.
        # Note: patterns are combined with OR semantics.
        patch = FunctionPatch(
            id="range_patch",
            description="",
            handler=Mock(),
            backend_version_patterns=["0.8.0~0.8.5"],
            primus_version_patterns=["1.0.0~1.0.5"],
        )

        # In range for both backend and primus versions
        ctx_ok = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.8.3",
            primus_version="1.0.2",
        )
        assert patch.applies_to(ctx_ok)

        # Backend version too low
        ctx_low_backend = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.7.9",
            primus_version="1.0.2",
        )
        assert not patch.applies_to(ctx_low_backend)

        # Backend version too high
        ctx_high_backend = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.9.0",
            primus_version="1.0.2",
        )
        assert not patch.applies_to(ctx_high_backend)

        # Primus version out of range (too low)
        ctx_low_primus = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.8.3",
            primus_version="0.9.9",
        )
        assert not patch.applies_to(ctx_low_primus)

        # Primus version out of range (too high)
        ctx_high_primus = PatchContext(
            backend="x",
            phase="y",
            backend_version="0.8.3",
            primus_version="1.0.6",
        )
        assert not patch.applies_to(ctx_high_primus)

    def test_applies_to_condition(self):
        patch = FunctionPatch(
            id="test",
            description="",
            handler=Mock(),
            condition=lambda ctx: ctx.model_name == "llama",
        )

        assert patch.applies_to(PatchContext(backend="x", phase="y", model_name="llama"))
        assert not patch.applies_to(PatchContext(backend="x", phase="y", model_name="bert"))

    def test_apply_invokes_handler(self):
        mock_handler = Mock()
        patch = FunctionPatch(
            id="apply-test",
            description="",
            handler=mock_handler,
        )
        ctx = PatchContext(backend="megatron", phase="setup")

        patch.apply(ctx)

        mock_handler.assert_called_once_with(ctx)
