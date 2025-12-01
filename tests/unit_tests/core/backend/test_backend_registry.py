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

from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry


class MockAdapter(BackendAdapter):
    """Mock adapter for testing."""

    def prepare_backend(self, config):
        pass

    def convert_config(self, config):
        return {}

    def load_trainer_class(self):
        return object


class TestBackendRegistryErrorHandling:
    """Test improved error handling in BackendRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save original state
        self._original_adapters = BackendRegistry._adapters.copy()
        self._original_path_names = BackendRegistry._path_names.copy()
        BackendRegistry._adapters.clear()
        BackendRegistry._path_names.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        BackendRegistry._adapters = self._original_adapters
        BackendRegistry._path_names = self._original_path_names

    def test_get_adapter_not_found_helpful_error(self):
        """Test that get_adapter provides helpful error when backend not found."""
        # Register one backend
        BackendRegistry.register_adapter("test_backend", MockAdapter)

        # Try to get non-existent backend
        with pytest.raises(ValueError) as exc_info:
            BackendRegistry.get_adapter("non_existent")

        error_msg = str(exc_info.value)
        assert "Backend 'non_existent' not found" in error_msg
        assert "Available backends: test_backend" in error_msg
        assert "Hint:" in error_msg

    def test_get_adapter_empty_registry_error(self):
        """Test error message when no backends are registered."""
        with pytest.raises(ValueError) as exc_info:
            BackendRegistry.get_adapter("any_backend")

        error_msg = str(exc_info.value)
        assert "Backend 'any_backend' not found" in error_msg
        assert "Available backends: none" in error_msg

    def test_get_adapter_creation_failure(self):
        """Test error handling when adapter creation fails."""

        class FailingAdapter(BackendAdapter):
            def __init__(self, framework):
                raise RuntimeError("Adapter initialization failed")

            def prepare_backend(self, config):
                pass

            def convert_config(self, config):
                return {}

            def load_trainer_class(self):
                return object

        BackendRegistry.register_adapter("failing", FailingAdapter)

        with pytest.raises(RuntimeError) as exc_info:
            BackendRegistry.get_adapter("failing")

        error_msg = str(exc_info.value)
        assert "Failed to create adapter for 'failing'" in error_msg
        assert "Adapter initialization failed" in error_msg


class TestBackendRegistryLazyLoading:
    """Test lazy loading functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_adapters = BackendRegistry._adapters.copy()
        BackendRegistry._adapters.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        BackendRegistry._adapters = self._original_adapters

    def test_try_load_backend_non_existent(self, capsys):
        """Test that _try_load_backend handles non-existent backends gracefully."""
        result = BackendRegistry._try_load_backend("definitely_not_a_backend")

        # Should return False but not crash
        assert result is False
        capsys.readouterr()

    def test_try_load_backend_returns_bool(self):
        """Test that _try_load_backend returns boolean."""
        result = BackendRegistry._try_load_backend("non_existent")
        assert isinstance(result, bool)
        assert result is False

    def test_get_adapter_with_lazy_loading(self):
        """Test that get_adapter triggers lazy loading."""
        # Don't pre-register, let it lazy load
        # This will try to load megatron backend
        try:
            # Note: get_adapter now also tries setup_backend_path
            adapter = BackendRegistry.get_adapter("megatron", backend_path=None)
            # If megatron is installed, should succeed
            assert adapter is not None
        except (ValueError, FileNotFoundError):
            # If not installed or path not found, should get helpful error
            pass

    def test_list_available_backends(self):
        """Test listing available backends."""
        BackendRegistry.register_adapter("backend1", MockAdapter)
        BackendRegistry.register_adapter("backend2", MockAdapter)

        available = BackendRegistry.list_available_backends()
        assert "backend1" in available
        assert "backend2" in available
        assert len(available) == 2

    def test_list_available_backends_empty(self):
        """Test listing when no backends registered."""
        available = BackendRegistry.list_available_backends()
        assert available == []


class TestBackendRegistryPathNames:
    """Test path name registration and retrieval."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_path_names = BackendRegistry._path_names.copy()
        self._original_adapters = BackendRegistry._adapters.copy()
        BackendRegistry._path_names = {"megatron": "Megatron-LM", "torchtitan": "torchtitan"}
        BackendRegistry._adapters.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        BackendRegistry._path_names = self._original_path_names
        BackendRegistry._adapters = self._original_adapters

    def test_register_and_get_path_name(self):
        """Test registering and retrieving path names."""
        BackendRegistry.register_path_name("test_backend", "TestBackend-Path")

        path_name = BackendRegistry.get_path_name("test_backend")
        assert path_name == "TestBackend-Path"

    def test_get_path_name_with_lazy_loading(self):
        """Test get_path_name triggers lazy loading."""
        # Pre-registered in _path_names
        path_name = BackendRegistry.get_path_name("megatron")
        assert path_name == "Megatron-LM"

    def test_get_path_name_not_found(self):
        """Test error when path name not registered and can't be loaded."""
        with pytest.raises(KeyError) as exc_info:
            BackendRegistry.get_path_name("non_existent_backend")

        assert "No path name registered for backend 'non_existent_backend'" in str(exc_info.value)


class TestBackendRegistrySetupPath:
    """Test setup_backend_path functionality."""

    def setup_method(self):
        """Save original state."""
        self._original_path_names = BackendRegistry._path_names.copy()
        self._original_sys_path = sys.path.copy()
        BackendRegistry._path_names = {"megatron": "Megatron-LM", "test_backend": "TestBackend"}

    def teardown_method(self):
        """Restore original state."""
        BackendRegistry._path_names = self._original_path_names
        sys.path[:] = self._original_sys_path

    def test_setup_backend_path_with_explicit_path(self, tmp_path):
        """Test setup_backend_path with explicit backend_path argument."""
        import sys

        # Create a temporary backend directory
        backend_dir = tmp_path / "explicit_backend"
        backend_dir.mkdir()

        # Setup with explicit path
        result = BackendRegistry.setup_backend_path(
            "test_backend", backend_path=str(backend_dir), verbose=False
        )

        assert result == str(backend_dir)
        assert str(backend_dir) in sys.path

    def test_setup_backend_path_with_env_var(self, tmp_path, monkeypatch):
        """Test setup_backend_path with BACKEND_PATH environment variable."""
        import sys

        # Create a temporary backend directory
        backend_dir = tmp_path / "env_backend"
        backend_dir.mkdir()

        # Set environment variable
        monkeypatch.setenv("BACKEND_PATH", str(backend_dir))

        # Setup should use env var
        result = BackendRegistry.setup_backend_path("test_backend", verbose=False)

        assert result == str(backend_dir)
        assert str(backend_dir) in sys.path

    def test_setup_backend_path_not_found(self):
        """Test setup_backend_path raises error when backend not registered."""
        with pytest.raises(KeyError) as exc_info:
            BackendRegistry.setup_backend_path("non_existent_backend", verbose=False)

        error_msg = str(exc_info.value)
        assert "No path name registered" in error_msg
        assert "Available backends:" in error_msg

    def test_setup_backend_path_already_in_sys_path(self, tmp_path):
        """Test setup_backend_path doesn't duplicate entries in sys.path."""
        import sys

        backend_dir = tmp_path / "duplicate_test"
        backend_dir.mkdir()

        # Add to sys.path manually
        sys.path.insert(0, str(backend_dir))
        initial_count = sys.path.count(str(backend_dir))

        # Setup should not duplicate
        BackendRegistry.setup_backend_path("test_backend", backend_path=str(backend_dir), verbose=False)

        final_count = sys.path.count(str(backend_dir))
        assert final_count == initial_count  # Should not increase


class TestBackendRegistryGetAdapterIntegration:
    """Test get_adapter with automatic path setup."""

    def setup_method(self):
        """Save original state."""
        self._original_adapters = BackendRegistry._adapters.copy()
        self._original_path_names = BackendRegistry._path_names.copy()
        self._original_sys_path = sys.path.copy()
        BackendRegistry._adapters.clear()
        BackendRegistry._path_names = {"test_backend": "TestBackend"}

    def teardown_method(self):
        """Restore original state."""
        BackendRegistry._adapters = self._original_adapters
        BackendRegistry._path_names = self._original_path_names
        sys.path[:] = self._original_sys_path

    def test_get_adapter_with_backend_path(self, tmp_path):
        """Test get_adapter automatically sets up backend path."""
        # Create backend directory
        backend_dir = tmp_path / "test_backend_dir"
        backend_dir.mkdir()

        # Register adapter
        BackendRegistry.register_adapter("test_backend", MockAdapter)

        # get_adapter should setup path automatically
        adapter = BackendRegistry.get_adapter("test_backend", backend_path=str(backend_dir))

        assert adapter is not None
        assert str(backend_dir) in sys.path

    def test_get_adapter_path_not_found_error(self):
        """Test get_adapter provides helpful error when path not found."""
        # Register adapter but no valid path
        BackendRegistry.register_adapter("test_backend", MockAdapter)

        with pytest.raises(FileNotFoundError) as exc_info:
            BackendRegistry.get_adapter("test_backend", backend_path="/non/existent/path")

        error_msg = str(exc_info.value)
        assert "Requested backend: 'test_backend'" in error_msg


class TestBackendRegistryHasAdapter:
    """Test has_adapter functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_adapters = BackendRegistry._adapters.copy()
        BackendRegistry._adapters.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        BackendRegistry._adapters = self._original_adapters

    def test_has_adapter_true(self):
        """Test has_adapter returns True for registered adapter."""
        BackendRegistry.register_adapter("test_backend", MockAdapter)

        assert BackendRegistry.has_adapter("test_backend") is True

    def test_has_adapter_false(self):
        """Test has_adapter returns False for non-registered adapter."""
        assert BackendRegistry.has_adapter("non_existent") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
