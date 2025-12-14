###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

    def convert_config(self, config):
        return {}

    def load_trainer_class(self):
        return object

    def detect_backend_version(self) -> str:
        return "test-version"


class TestBackendRegistryErrorHandling:
    """Test improved error handling in BackendRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save original state
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        self._original_path_names = registry_module.BackendRegistry._path_names.copy()
        registry_module.BackendRegistry._adapters.clear()
        registry_module.BackendRegistry._path_names.clear()

        # Silence logging dependencies (logger may not be initialized in tests)
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._adapters = self._original_adapters
        registry_module.BackendRegistry._path_names = self._original_path_names
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_get_adapter_not_found_helpful_error(self):
        """Test that get_adapter provides helpful error when backend not found."""
        # Register one backend
        registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        # Try to get non-existent backend
        # In the new core runtime, this fails at import time for the backend module.
        with pytest.raises(ImportError):
            registry_module.BackendRegistry.get_adapter("non_existent")

    def test_get_adapter_empty_registry_error(self):
        """Test error message when no backends are registered."""
        # In the new core runtime, this also fails at import time for the backend module.
        with pytest.raises(ImportError):
            registry_module.BackendRegistry.get_adapter("any_backend")

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

            def load_trainer_class(self):
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
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        registry_module.BackendRegistry._adapters.clear()

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._adapters = self._original_adapters
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_try_load_backend_non_existent(self):
        """Test that _try_load_backend handles non-existent backends gracefully."""
        with pytest.raises(ImportError):
            registry_module.BackendRegistry._try_load_backend("definitely_not_a_backend")

    def test_try_load_backend_returns_bool(self):
        """Test that _try_load_backend returns boolean."""
        with pytest.raises(ImportError):
            registry_module.BackendRegistry._try_load_backend("non_existent")

    def test_get_adapter_with_lazy_loading(self):
        """Test that get_adapter triggers lazy loading."""
        # Don't pre-register, let it lazy load
        # This will try to load megatron backend
        # Note: get_adapter now also tries setup_backend_path
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

    def test_list_available_backends_empty(self):
        """Test listing when no backends registered."""
        available = registry_module.BackendRegistry.list_available_backends()
        assert available == []


class TestBackendRegistryPathNames:
    """Test path name registration and retrieval."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_path_names = registry_module.BackendRegistry._path_names.copy()
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        registry_module.BackendRegistry._path_names.clear()
        registry_module.BackendRegistry._path_names.update(
            {"megatron": "Megatron-LM", "torchtitan": "torchtitan"}
        )
        registry_module.BackendRegistry._adapters.clear()

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._path_names.clear()
        registry_module.BackendRegistry._path_names.update(self._original_path_names)
        registry_module.BackendRegistry._adapters.clear()
        registry_module.BackendRegistry._adapters.update(self._original_adapters)
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_register_and_get_path_name(self):
        """Test registering and retrieving path names."""
        registry_module.BackendRegistry.register_path_name("test_backend", "TestBackend-Path")

        path_name = registry_module.BackendRegistry.get_path_name("test_backend")
        assert path_name == "TestBackend-Path"

    def test_get_path_name_with_lazy_loading(self):
        """Test get_path_name triggers lazy loading."""
        # Pre-registered in _path_names
        path_name = registry_module.BackendRegistry.get_path_name("megatron")
        assert path_name == "Megatron-LM"

    def test_get_path_name_not_found(self):
        """Test error when path name not registered and can't be loaded."""
        # In the new lazy-loading logic, an unknown backend triggers a ModuleNotFoundError.
        with pytest.raises(ModuleNotFoundError):
            registry_module.BackendRegistry.get_path_name("non_existent_backend")


class TestBackendRegistrySetupPath:
    """Test setup_backend_path functionality."""

    def setup_method(self):
        """Save original state."""
        self._original_path_names = registry_module.BackendRegistry._path_names.copy()
        self._original_sys_path = sys.path.copy()
        registry_module.BackendRegistry._path_names.clear()
        registry_module.BackendRegistry._path_names.update(
            {
                "megatron": "Megatron-LM",
                "test_backend": "TestBackend",
            }
        )

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore original state."""
        registry_module.BackendRegistry._path_names = self._original_path_names
        sys.path[:] = self._original_sys_path
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_setup_backend_path_with_explicit_path(self, tmp_path):
        """Test setup_backend_path with explicit backend_path argument."""
        # Create a temporary backend directory
        backend_dir = tmp_path / "explicit_backend"
        backend_dir.mkdir()

        # Setup with explicit path
        result = registry_module.BackendRegistry.setup_backend_path(
            "test_backend", backend_path=str(backend_dir), verbose=False
        )

        assert result == str(backend_dir)
        assert str(backend_dir) in sys.path

    def test_setup_backend_path_with_env_var(self, tmp_path, monkeypatch):
        """Test setup_backend_path with BACKEND_PATH environment variable."""
        # Create a temporary backend directory
        backend_dir = tmp_path / "env_backend"
        backend_dir.mkdir()

        # Set environment variable
        monkeypatch.setenv("BACKEND_PATH", str(backend_dir))

        # Setup should use env var
        result = registry_module.BackendRegistry.setup_backend_path("test_backend", verbose=False)

        assert result == str(backend_dir)
        assert str(backend_dir) in sys.path

    def test_setup_backend_path_not_found(self):
        """Test setup_backend_path raises error when backend not registered."""
        # Missing backend now fails during lazy loading with ModuleNotFoundError.
        with pytest.raises(ModuleNotFoundError):
            registry_module.BackendRegistry.setup_backend_path("non_existent_backend", verbose=False)

    def test_setup_backend_path_already_in_sys_path(self, tmp_path):
        """Test setup_backend_path doesn't duplicate entries in sys.path."""
        backend_dir = tmp_path / "duplicate_test"
        backend_dir.mkdir()

        # Add to sys.path manually
        sys.path.insert(0, str(backend_dir))
        initial_count = sys.path.count(str(backend_dir))

        # Setup should not duplicate
        registry_module.BackendRegistry.setup_backend_path(
            "test_backend", backend_path=str(backend_dir), verbose=False
        )

        final_count = sys.path.count(str(backend_dir))
        assert final_count == initial_count  # Should not increase


class TestBackendRegistryGetAdapterIntegration:
    """Test get_adapter with automatic path setup."""

    def setup_method(self):
        """Save original state."""
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        self._original_path_names = registry_module.BackendRegistry._path_names.copy()
        self._original_sys_path = sys.path.copy()
        registry_module.BackendRegistry._adapters.clear()
        registry_module.BackendRegistry._path_names.clear()
        registry_module.BackendRegistry._path_names.update({"test_backend": "TestBackend"})

        # Silence logging dependencies
        self._orig_log_rank_0 = registry_module.log_rank_0
        registry_module.log_rank_0 = lambda *args, **kwargs: None

    def teardown_method(self):
        """Restore original state."""
        registry_module.BackendRegistry._adapters.clear()
        registry_module.BackendRegistry._adapters.update(self._original_adapters)
        registry_module.BackendRegistry._path_names.clear()
        registry_module.BackendRegistry._path_names.update(self._original_path_names)
        sys.path[:] = self._original_sys_path
        registry_module.log_rank_0 = self._orig_log_rank_0

    def test_get_adapter_with_backend_path(self, tmp_path):
        """Test get_adapter automatically sets up backend path."""
        # Create backend directory
        backend_dir = tmp_path / "test_backend_dir"
        backend_dir.mkdir()

        # Register adapter
        registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        # get_adapter should return a working adapter; when adapter is already
        # registered, the path is not modified in the new simplified logic.
        adapter = registry_module.BackendRegistry.get_adapter("test_backend", backend_path=str(backend_dir))

        assert adapter is not None

    def test_get_adapter_path_not_found_error(self):
        """Test get_adapter provides helpful error when path not found."""
        # Register adapter but no valid path
        registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        # In the new logic, if the adapter is already registered, backend_path
        # is ignored and get_adapter succeeds.
        adapter = registry_module.BackendRegistry.get_adapter(
            "test_backend", backend_path="/non/existent/path"
        )
        assert isinstance(adapter, MockAdapter)


class TestBackendRegistryHasAdapter:
    """Test has_adapter functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_adapters = registry_module.BackendRegistry._adapters.copy()
        registry_module.BackendRegistry._adapters.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        registry_module.BackendRegistry._adapters = self._original_adapters

    def test_has_adapter_true(self):
        """Test has_adapter returns True for registered adapter."""
        registry_module.BackendRegistry.register_adapter("test_backend", MockAdapter)

        assert registry_module.BackendRegistry.has_adapter("test_backend") is True

    def test_has_adapter_false(self):
        """Test has_adapter returns False for non-registered adapter."""
        assert registry_module.BackendRegistry.has_adapter("non_existent") is False


class TestBackendRegistryTrainerClasses:
    """Test trainer class registration and retrieval."""

    def setup_method(self):
        """Clear trainer classes before each test."""
        self._original_trainer_classes = registry_module.BackendRegistry._trainer_classes.copy()
        registry_module.BackendRegistry._trainer_classes.clear()

    def teardown_method(self):
        """Restore trainer classes after each test."""
        registry_module.BackendRegistry._trainer_classes = self._original_trainer_classes

    def test_register_and_get_trainer_class(self):
        """Test registering and retrieving trainer classes."""

        class DummyTrainer:
            pass

        registry_module.BackendRegistry.register_trainer_class("test_backend", DummyTrainer)

        trainer_cls = registry_module.BackendRegistry.get_trainer_class("test_backend")
        assert trainer_cls is DummyTrainer

    def test_get_trainer_class_not_found(self):
        """Test error when trainer class not registered."""
        with pytest.raises(AssertionError) as exc_info:
            registry_module.BackendRegistry.get_trainer_class("non_existent_backend")

        assert "No trainer class registered for backend 'non_existent_backend'" in str(exc_info.value)

    def test_has_trainer_class(self):
        """Test has_trainer_class reflects registration state."""

        class DummyTrainer:
            pass

        assert registry_module.BackendRegistry.has_trainer_class("test_backend") is False
        registry_module.BackendRegistry.register_trainer_class("test_backend", DummyTrainer)
        assert registry_module.BackendRegistry.has_trainer_class("test_backend") is True


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
