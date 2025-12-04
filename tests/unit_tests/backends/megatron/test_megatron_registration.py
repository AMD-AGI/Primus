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
    1. Path name registration (megatron -> Megatron-LM)
    2. Adapter registration (MegatronAdapter)
    3. Trainer class registration (MegatronPretrainTrainer)
    4. Integration: get_adapter returns correct instance
"""

import pytest

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry


class TestMegatronBackendRegistration:
    """Test that megatron backend is properly registered via __init__.py."""

    @pytest.fixture(autouse=True)
    def ensure_backend_loaded(self):
        """Ensure megatron backend module is loaded before each test."""
        # Import the __init__ module to trigger registration
        import primus.backends.megatron  # noqa: F401

    def test_path_name_is_registered(self):
        """Verify that megatron -> Megatron-LM path mapping is registered."""
        # BackendRegistry should have path name registered
        assert BackendRegistry.has_adapter("megatron"), (
            "Megatron backend not loaded. " "Check primus/backends/megatron/__init__.py imports are working."
        )

        path_name = BackendRegistry.get_path_name("megatron")
        assert path_name == "Megatron-LM", (
            f"Expected path name 'Megatron-LM' for megatron backend, got '{path_name}'. "
            "Check BackendRegistry.register_path_name() call in __init__.py"
        )

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
            "Check BackendRegistry.register_trainer_class('megatron', MegatronPretrainTrainer)"
        )

    def test_adapter_can_be_instantiated_via_registry(self):
        """Verify that get_adapter returns a working MegatronAdapter instance."""
        # This tests the full integration: registry lookup + instantiation
        # Note: We skip path setup since Megatron may not be installed in test env
        from unittest.mock import patch

        with patch.object(BackendRegistry, "setup_backend_path"):
            adapter = BackendRegistry.get_adapter("megatron")

        # Verify instance type
        assert isinstance(
            adapter, MegatronAdapter
        ), f"Expected MegatronAdapter instance, got {type(adapter).__name__}"

        # Verify adapter framework attribute
        assert adapter.framework == "megatron", f"Expected framework='megatron', got '{adapter.framework}'"

    def test_trainer_class_can_be_retrieved(self):
        """Verify trainer class retrieval through adapter."""
        from unittest.mock import patch

        with patch.object(BackendRegistry, "setup_backend_path"):
            adapter = BackendRegistry.get_adapter("megatron")

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
        assert BackendRegistry.get_path_name("megatron") == "Megatron-LM"

    def test_registration_happens_at_import_time(self):
        """Verify registration occurs when module is imported."""
        # This is tested implicitly by all other tests, but we make it explicit
        # Registration should happen automatically without any function calls

        # Clear registrations (simulate fresh import)
        original_path_names = BackendRegistry._path_names.copy()
        original_adapters = BackendRegistry._adapters.copy()
        original_trainers = BackendRegistry._trainer_classes.copy()

        try:
            # Remove megatron registrations
            BackendRegistry._path_names.pop("megatron", None)
            BackendRegistry._adapters.pop("megatron", None)
            BackendRegistry._trainer_classes.pop("megatron", None)

            # Verify it's gone
            assert not BackendRegistry.has_adapter("megatron")

            # Re-import should trigger registration
            import importlib

            import primus.backends.megatron

            importlib.reload(primus.backends.megatron)

            # Now it should be registered again
            assert BackendRegistry.has_adapter("megatron")
            assert BackendRegistry.has_trainer_class("megatron")
            assert BackendRegistry.get_path_name("megatron") == "Megatron-LM"

        finally:
            # Restore original state
            BackendRegistry._path_names = original_path_names
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
            with pytest.raises(ValueError, match="Backend 'megatron' not found"):
                # Mock setup_backend_path to prevent actual path operations
                from unittest.mock import patch

                with patch.object(BackendRegistry, "setup_backend_path"):
                    with patch.object(BackendRegistry, "_try_load_backend", return_value=False):
                        BackendRegistry.get_adapter("megatron")
        finally:
            # Restore
            if original:
                BackendRegistry._adapters["megatron"] = original

    def test_missing_trainer_registration_would_fail(self):
        """Demonstrate what happens if trainer class registration is missing."""
        # Simulate missing trainer registration
        original = BackendRegistry._trainer_classes.pop("megatron", None)

        try:
            # Without trainer registration, get_trainer_class should fail
            with pytest.raises(KeyError, match="No trainer class registered"):
                BackendRegistry.get_trainer_class("megatron")
        finally:
            # Restore
            if original:
                BackendRegistry._trainer_classes["megatron"] = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
