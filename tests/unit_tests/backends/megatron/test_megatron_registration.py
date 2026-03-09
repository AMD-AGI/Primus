###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for Megatron backend registration in __init__.py.

This test ensures that the registration logic in primus/backends/megatron/__init__.py
correctly registers the backend with BackendRegistry. Without proper registration,
the backend will be unavailable to the runtime.

Test coverage:
    1. Adapter registration (MegatronAdapter)
    2. Trainer class registration (MegatronPretrainTrainer)
    3. Integration: get_adapter returns correct instance
"""

import pytest

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

_SUPPORTS_TRAINER_CLASS_REGISTRY = all(
    hasattr(BackendRegistry, attr)
    for attr in ("_trainer_classes", "register_trainer_class", "get_trainer_class", "has_trainer_class")
)

if not _SUPPORTS_TRAINER_CLASS_REGISTRY:
    pytest.skip(
        "BackendRegistry trainer-class registration API is not available; "
        "skip megatron registration tests that depend on it.",
        allow_module_level=True,
    )


class TestMegatronBackendRegistration:
    """Test that megatron backend is properly registered via __init__.py."""

    @pytest.fixture(autouse=True)
    def ensure_backend_loaded(self):
        """Ensure megatron backend module is loaded before each test."""
        if not _SUPPORTS_TRAINER_CLASS_REGISTRY:
            pytest.skip("BackendRegistry trainer-class API not available in this version.")
        # Import the __init__ module to trigger registration
        import primus.backends.megatron  # noqa: F401

    def test_adapter_is_registered(self):
        """Verify that MegatronAdapter is registered for 'megatron' backend."""
        # Check registration
        assert BackendRegistry.has_adapter("megatron"), (
            "MegatronAdapter not registered. " "Check BackendRegistry.register_adapter() call in __init__.py"
        )

        # Verify correct adapter class is registered (not instance)
        adapter_cls = BackendRegistry._adapters.get("megatron")
        assert adapter_cls is MegatronAdapter, (
            f"Expected MegatronAdapter class, got {adapter_cls}. "
            "Check BackendRegistry.register_adapter('megatron', MegatronAdapter) in __init__.py"
        )

    def test_trainer_class_is_registered(self):
        """Verify that MegatronPretrainTrainer is registered for 'megatron' backend."""
        # Check registration
        assert BackendRegistry.has_trainer_class("megatron"), (
            "MegatronPretrainTrainer not registered. "
            "Check BackendRegistry.register_trainer_class() call in __init__.py"
        )

        # Verify correct trainer class
        trainer_cls = BackendRegistry.get_trainer_class("megatron")
        assert trainer_cls is MegatronPretrainTrainer, (
            f"Expected MegatronPretrainTrainer, got {trainer_cls}. "
            "Check BackendRegistry.register_trainer_class(MegatronPretrainTrainer, 'megatron')"
        )

        # Explicit stage should also work
        assert BackendRegistry.has_trainer_class("megatron", stage="pretrain")

    def test_adapter_can_be_instantiated_via_registry(self):
        """Verify that get_adapter returns a working MegatronAdapter instance."""
        # This tests the full integration: registry lookup + instantiation
        # Note: We skip path setup since Megatron may not be installed in test env
        from unittest.mock import patch

        # get_adapter() has two paths:
        # - lazy-load path (backend not registered yet): defines resolved_path
        # - already-registered path: may not define resolved_path
        #
        # Force the lazy-load path here, and mock the backend "import" to re-register
        # megatron without relying on Python module reload side-effects.
        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()
        try:
            BackendRegistry._adapters.pop("megatron", None)
            BackendRegistry._trainer_classes.pop(("megatron", "pretrain"), None)

            def _fake_load_backend(_backend: str) -> None:
                BackendRegistry.register_adapter("megatron", MegatronAdapter)
                BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")

            with patch.object(BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
                adapter = BackendRegistry.get_adapter("megatron")
        finally:
            BackendRegistry._adapters = original_adapters
            BackendRegistry._trainer_classes = original_trainers

        # Verify instance type
        assert isinstance(
            adapter, MegatronAdapter
        ), f"Expected MegatronAdapter instance, got {type(adapter).__name__}"

        # Verify adapter framework attribute
        assert adapter.framework == "megatron", f"Expected framework='megatron', got '{adapter.framework}'"

    def test_trainer_class_can_be_retrieved(self):
        """Verify trainer class retrieval through adapter."""
        from unittest.mock import patch

        # Force lazy-load path; see rationale in test_adapter_can_be_instantiated_via_registry.
        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()
        try:
            BackendRegistry._adapters.pop("megatron", None)
            BackendRegistry._trainer_classes.pop(("megatron", "pretrain"), None)

            def _fake_load_backend(_backend: str) -> None:
                BackendRegistry.register_adapter("megatron", MegatronAdapter)
                BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")

            with patch.object(BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
                adapter = BackendRegistry.get_adapter("megatron")
        finally:
            BackendRegistry._adapters = original_adapters
            BackendRegistry._trainer_classes = original_trainers

        # Adapter should be able to load the registered trainer class
        trainer_cls = adapter.load_trainer_class()
        assert (
            trainer_cls is MegatronPretrainTrainer
        ), f"Expected MegatronPretrainTrainer from adapter, got {trainer_cls}"

    def test_megatron_in_available_backends_list(self):
        """Verify megatron appears in list of available backends."""
        available = BackendRegistry.list_available_backends()
        assert "megatron" in available, (
            f"'megatron' not in available backends: {available}. "
            "Registration may have failed in __init__.py"
        )


class TestMegatronRegistrationOrder:
    """Test that registration happens in correct order and is idempotent."""

    def test_registration_is_idempotent(self):
        """Verify that re-importing __init__ doesn't cause errors."""
        import importlib

        import primus.backends.megatron

        # Re-import should not raise errors
        importlib.reload(primus.backends.megatron)

        # Verify registration still works after reload
        assert BackendRegistry.has_adapter("megatron")
        assert BackendRegistry.has_trainer_class("megatron")

    def test_registration_happens_at_import_time(self):
        """Verify registration occurs when module is imported."""
        # This is tested implicitly by all other tests, but we make it explicit
        # Registration should happen automatically without any function calls

        # Clear registrations (simulate fresh import)
        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()
        try:
            # Remove megatron registrations
            BackendRegistry._adapters.pop("megatron", None)
            BackendRegistry._trainer_classes.pop(("megatron", "pretrain"), None)

            # Verify it's gone
            assert not BackendRegistry.has_adapter("megatron")

            # Re-import should trigger registration
            import importlib

            import primus.backends.megatron

            importlib.reload(primus.backends.megatron)

            # Now it should be registered again
            assert BackendRegistry.has_adapter("megatron")
            assert BackendRegistry.has_trainer_class("megatron")

        finally:
            # Restore original state
            BackendRegistry._adapters = original_adapters
            BackendRegistry._trainer_classes = original_trainers


class TestMegatronRegistrationFailures:
    """Test error handling when registration is missing or incorrect."""

    def test_missing_adapter_registration_would_fail(self):
        """Demonstrate what happens if adapter registration is missing."""
        # Simulate missing registration
        original = BackendRegistry._adapters.pop("megatron", None)

        try:
            # Without registration, get_adapter should fail gracefully
            # (after attempting lazy load)
            # In the new core runtime, this is treated as an unrecoverable error
            # and results in an AssertionError from BackendRegistry.
            with pytest.raises(AssertionError, match="Backend 'megatron' not found"):
                # Mock setup_backend_path to prevent actual path operations
                from unittest.mock import patch

                with patch.object(BackendRegistry, "_try_load_backend", return_value=False):
                    BackendRegistry.get_adapter("megatron")
        finally:
            # Restore
            if original:
                BackendRegistry._adapters["megatron"] = original

    def test_missing_trainer_registration_would_fail(self):
        """Demonstrate what happens if trainer class registration is missing."""
        # Simulate missing trainer registration
        original = BackendRegistry._trainer_classes.pop(("megatron", "pretrain"), None)
        try:
            # Without trainer registration, get_trainer_class should fail
            with pytest.raises(ValueError, match="No trainer class registered"):
                BackendRegistry.get_trainer_class("megatron")
        finally:
            # Restore
            if original:
                BackendRegistry._trainer_classes[("megatron", "pretrain")] = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
