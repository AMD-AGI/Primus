###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for BackendRegistry improvements.
"""

import sys

import pytest

import primus.core.backend.backend_registry as registry_module
from primus.core.backend.backend_adapter import BackendAdapter


class MockAdapter(BackendAdapter):
    """Mock adapter for testing."""

    def prepare_backend(self, config):
        pass

    def convert_config(self, params):
        return {}

    def load_trainer_class(self, stage: str = "pretrain"):
        return object

    def detect_backend_version(self) -> str:
        return "test-version"


class TestBackendRegistryErrorHandling:
    """Test improved error handling in BackendRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save original state
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        registry_module.BackendRegistry._adapters.clear()
        # Silence logging dependencies (logger may not be initialized in tests)
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._adapters = self._original_adapters
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_get_adapter_not_found_helpful_error(self):
        """Test that get_adapter provides helpful error when backend not found."""
        # Register one backend
        registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        # Try to get non-existent backend
        # get_adapter first calls _load_backend, which fails with ModuleNotFoundError
        # when the backend module doesn't exist.
        with pytest.raises(ModuleNotFoundError, match="No module named 'primus.backends.non_existent'"):
            registry_module.BackendRegistry.get_adapter("non_existent")

    def test_get_adapter_creation_failure(self):
        """Test error handling when adapter creation fails."""

        class FailingAdapter(BackendAdapter):
            def __init__(self, _framework):
                super().__init__(_framework)
                raise RuntimeError("Adapter initialization failed")

            def prepare_backend(self, config):
                pass

            def convert_config(self, config):
                return {}

            def load_trainer_class(self, stage: str = "pretrain"):
                return object

            def detect_backend_version(self) -> str:
                return "test-version"

        registry_module.BackendRegistry.register_adapter("failing", FailingAdapter)

        with pytest.raises(RuntimeError) as exc_info:
            registry_module.BackendRegistry.get_adapter("failing")

        error_msg = str(exc_info.value)
        assert "Adapter initialization failed" in error_msg


class TestBackendRegistryLazyLoading:
    """Test lazy loading functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        # Reset adapter registry state
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        registry_module.BackendRegistry._adapters.clear()

        # Ensure backend module can be re-imported so that lazy loading
        # re-runs registration even if other tests imported it earlier.
        self._orig_megatron_module = sys.modules.get("primus.backends.megatron")
        if "primus.backends.megatron" in sys.modules:
            del sys.modules["primus.backends.megatron"]

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._adapters = self._original_adapters
        registry_module.log_rank_0 = self._orig_log_rank_0

        # Restore original backend module to avoid impacting other tests
        if self._orig_megatron_module is not None:
            sys.modules["primus.backends.megatron"] = self._orig_megatron_module
        else:
            sys.modules.pop("primus.backends.megatron", None)

    def test_try_load_backend_non_existent(self):
        """Test that _try_load_backend handles non-existent backends gracefully."""
        with pytest.raises(ImportError):
            registry_module.BackendRegistry._try_load_backend("definitely_not_a_backend")

    def test_get_adapter_with_lazy_loading(self):
        """Test that get_adapter triggers lazy loading."""
        # Don't pre-register, let it lazy load
        # Avoid importing real `primus.backends.megatron` here because its __init__.py may
        # depend on optional registry features (e.g., trainer-class registration) that
        # are not present in all versions under test.
        from unittest.mock import patch

        def _fake_load_backend(_backend: str) -> None:
            registry_module.BackendRegistry.register_adapter("megatron", MockAdapter)

        with patch.object(registry_module.BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
            adapter = registry_module.BackendRegistry.get_adapter("megatron", backend_path=None)
        # If megatron is installed and registered correctly, this should not raise
        # and must return a non-None adapter instance.
        assert adapter is not None

    def test_list_available_backends(self):
        """Test listing available backends."""
        registry_module.BackendRegistry.register_adapter("backend1", MockAdapter)
        registry_module.BackendRegistry.register_adapter("backend2", MockAdapter)

        available = registry_module.BackendRegistry.list_available_backends()
        assert "backend1" in available
        assert "backend2" in available
        assert len(available) == 2


class TestBackendRegistryGetAdapterIntegration:
    """Test get_adapter with automatic path setup."""

    def setup_method(self):
        """Save original state."""
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        self._original_sys_path = sys.path.copy()
        registry_module.BackendRegistry._adapters.clear()

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore original state."""
        registry_module.BackendRegistry._adapters.clear()
        registry_module.BackendRegistry._adapters.update(self._original_adapters)
        sys.path[:] = self._original_sys_path
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_get_adapter_with_backend_path(self, tmp_path):
        """Test adapter can set up backend path via setup_backend_path()."""
        # Create backend directory
        backend_dir = tmp_path / "test_backend_dir"
        backend_dir.mkdir()

        # Force the lazy-load path (backend not registered yet) and simulate backend
        # registering its adapter class. BackendRegistry.get_adapter no longer mutates
        # sys.path; path setup is owned by adapter.setup_backend_path().
        # We patch _load_backend to avoid importing a real primus.backends.test_backend
        # module; instead we simulate the backend registering its adapter class.
        from unittest.mock import patch

        def _fake_load_backend(_backend: str) -> None:
            registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        with patch.object(registry_module.BackendRegistry, "_load_backend", side_effect=_fake_load_backend):
            adapter = registry_module.BackendRegistry.get_adapter(
                "test_backend", backend_path=str(backend_dir)
            )

        assert adapter is not None
        adapter.setup_backend_path(backend_path=str(backend_dir))
        assert str(backend_dir) in sys.path

    def test_get_adapter_path_not_found_error(self):
        """Test get_adapter provides helpful error when path not found."""
        # Force the lazy-load path (backend not registered yet) so that
        # _load_backend() tries to import the module first and fails with ModuleNotFoundError.
        with pytest.raises(ModuleNotFoundError, match="No module named 'primus.backends.test_backend'"):
            registry_module.BackendRegistry.get_adapter("test_backend", backend_path="/non/existent/path")


class TestBackendRegistrySetupHooks:
    """Test setup hook registration and execution."""

    def setup_method(self):
        """Clear setup hooks before each test."""
        self._original_setup_hooks = registry_module.BackendRegistry._setup_hooks.copy()
        registry_module.BackendRegistry._setup_hooks.clear()

    def teardown_method(self):
        """Restore setup hooks after each test."""
        registry_module.BackendRegistry._setup_hooks = self._original_setup_hooks

    def test_register_and_run_setup_hooks_in_order(self, capsys):
        """Test that setup hooks are run in registration order."""
        calls = []

        def hook1():
            calls.append("hook1")

        def hook2():
            calls.append("hook2")

        registry_module.BackendRegistry.register_setup_hook("test_backend", hook1)
        registry_module.BackendRegistry.register_setup_hook("test_backend", hook2)

        registry_module.BackendRegistry.run_setup("test_backend")
        captured = capsys.readouterr()

        assert "[Primus:BackendSetup] Running 2 setup hooks for backend 'test_backend'." in captured.out
        assert calls == ["hook1", "hook2"]

    def test_run_setup_no_hooks_is_noop(self, capsys):
        """Test that run_setup is a no-op when no hooks are registered."""
        registry_module.BackendRegistry.run_setup("unregistered_backend")
        captured = capsys.readouterr()
        # No output expected when there are no hooks
        assert captured.out == ""

    def test_run_setup_handles_exceptions(self, capsys):
        """Test that run_setup continues when a hook raises an exception."""

        def failing_hook():
            raise RuntimeError("hook failed")

        registry_module.BackendRegistry.register_setup_hook("test_backend", failing_hook)

        registry_module.BackendRegistry.run_setup("test_backend")
        captured = capsys.readouterr()

        assert "[Primus:BackendSetup] Running 1 setup hooks for backend 'test_backend'." in captured.out
        assert "Error in setup hook" in captured.out


class TestBackendRegistryTrainerClass:
    """Test the trainer-class registry API.

    ``register_trainer_class`` / ``get_trainer_class`` / ``has_trainer_class``
    back the stage-based fallback in ``MegatronAdapter.load_trainer_class`` (and
    the other backend adapters), so they must keep (backend, stage) keying and
    raise on lookups for unregistered combinations.
    """

    def setup_method(self):
        self._original_trainer_classes = registry_module.BackendRegistry._trainer_classes.copy()
        registry_module.BackendRegistry._trainer_classes.clear()

    def teardown_method(self):
        registry_module.BackendRegistry._trainer_classes = self._original_trainer_classes

    def test_register_and_get_trainer_class(self):
        class DummyTrainer:
            pass

        registry_module.BackendRegistry.register_trainer_class(DummyTrainer, "megatron", stage="pretrain")

        assert registry_module.BackendRegistry.get_trainer_class("megatron", stage="pretrain") is DummyTrainer
        assert registry_module.BackendRegistry.has_trainer_class("megatron", stage="pretrain") is True

    def test_register_defaults_to_pretrain_stage(self):
        class DummyTrainer:
            pass

        # Default stage is "pretrain" for both register and lookup.
        registry_module.BackendRegistry.register_trainer_class(DummyTrainer, "megatron")
        assert registry_module.BackendRegistry.get_trainer_class("megatron") is DummyTrainer

    def test_stage_is_part_of_the_key(self):
        class PretrainTrainer:
            pass

        registry_module.BackendRegistry.register_trainer_class(PretrainTrainer, "megatron", stage="pretrain")

        # A different stage for the same backend must NOT resolve to the
        # pretrain class; it is a distinct registry key.
        assert registry_module.BackendRegistry.has_trainer_class("megatron", stage="sft") is False
        with pytest.raises(ValueError):
            registry_module.BackendRegistry.get_trainer_class("megatron", stage="sft")

    def test_get_unregistered_trainer_class_raises(self):
        assert registry_module.BackendRegistry.has_trainer_class("nonexistent") is False
        with pytest.raises(ValueError, match="No trainer class registered"):
            registry_module.BackendRegistry.get_trainer_class("nonexistent")

    def test_register_overwrites_same_key(self):
        class TrainerA:
            pass

        class TrainerB:
            pass

        registry_module.BackendRegistry.register_trainer_class(TrainerA, "megatron", stage="pretrain")
        registry_module.BackendRegistry.register_trainer_class(TrainerB, "megatron", stage="pretrain")
        assert registry_module.BackendRegistry.get_trainer_class("megatron", stage="pretrain") is TrainerB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
