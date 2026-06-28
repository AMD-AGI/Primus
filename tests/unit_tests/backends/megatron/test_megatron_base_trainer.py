###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronBaseTrainer.

Tests path resolution, parse_args patching, and setup orchestration.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer


def _build_trainer(monkeypatch: pytest.MonkeyPatch, backend_args=None):
    """Helper to build MegatronBaseTrainer with stubbed dependencies."""

    # Create a concrete subclass for testing (MegatronBaseTrainer is abstract)
    class ConcreteMegatronBaseTrainer(MegatronBaseTrainer):
        def train(self):
            pass

    # Stub out BaseTrainer.__init__ to avoid real distributed env setup
    def dummy_base_init(self, backend_args=None, *args, **kwargs):
        self.backend_args = backend_args
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.master_addr = "localhost"
        self.master_port = 12345

    monkeypatch.setattr(
        "primus.core.trainer.base_trainer.BaseTrainer.__init__",
        dummy_base_init,
    )

    # Silence logging
    monkeypatch.setattr(
        "primus.backends.megatron.megatron_base_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )

    if backend_args is None:
        backend_args = SimpleNamespace()

    return ConcreteMegatronBaseTrainer(backend_args=backend_args)


class TestMegatronBaseTrainer:
    """Tests for MegatronBaseTrainer setup and path resolution."""

    def test_ensure_megatron_path_from_primus_path_env(self, monkeypatch: pytest.MonkeyPatch):
        """Test path resolution from PRIMUS_PATH environment variable."""
        trainer = _build_trainer(monkeypatch)

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            primus_path = Path(tmpdir) / "Primus"
            megatron_path = primus_path / "third_party" / "Megatron-LM"
            megatron_path.mkdir(parents=True)

            monkeypatch.setenv("PRIMUS_PATH", str(primus_path))

            # Force `import megatron` to raise ImportError by caching None in
            # sys.modules.  Unlike mocking builtins.__import__, this leaves
            # Python's import machinery intact for all other modules.
            saved_megatron = sys.modules.get("megatron", _SENTINEL := object())
            sys.modules["megatron"] = None
            # Strip paths that either contain "megatron" in the name or have a
            # megatron/ subdirectory, so Method 4 doesn't short-circuit.
            original_path = sys.path[:]
            sys.path[:] = [
                p for p in sys.path if "megatron" not in p.lower() and not (Path(p) / "megatron").is_dir()
            ]
            try:
                trainer._ensure_megatron_path()
                assert str(megatron_path) in sys.path
            finally:
                sys.path[:] = original_path
                if saved_megatron is _SENTINEL:
                    sys.modules.pop("megatron", None)
                else:
                    sys.modules["megatron"] = saved_megatron

    def test_ensure_megatron_path_from_cwd(self, monkeypatch: pytest.MonkeyPatch):
        """Test path resolution from current working directory."""
        trainer = _build_trainer(monkeypatch)

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            primus_path = Path(tmpdir) / "Primus"
            megatron_path = primus_path / "third_party" / "Megatron-LM"
            megatron_path.mkdir(parents=True)

            monkeypatch.delenv("PRIMUS_PATH", raising=False)

            def mock_cwd():
                return primus_path / "some" / "subdirectory"

            monkeypatch.setattr(Path, "cwd", staticmethod(mock_cwd))

            saved_megatron = sys.modules.get("megatron", _SENTINEL := object())
            sys.modules["megatron"] = None
            original_path = sys.path[:]
            sys.path[:] = [
                p for p in sys.path if "megatron" not in p.lower() and not (Path(p) / "megatron").is_dir()
            ]
            try:
                trainer._ensure_megatron_path()
                assert str(megatron_path) in sys.path
            finally:
                sys.path[:] = original_path
                if saved_megatron is _SENTINEL:
                    sys.modules.pop("megatron", None)
                else:
                    sys.modules["megatron"] = saved_megatron

    def test_ensure_megatron_path_from_file_location(self, monkeypatch: pytest.MonkeyPatch):
        """Test path resolution from current file location."""
        trainer = _build_trainer(monkeypatch)

        # Unset PRIMUS_PATH and mock cwd to not contain Primus
        monkeypatch.delenv("PRIMUS_PATH", raising=False)
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/some/other/path")))

        # Mock __file__ to point to a Primus subdirectory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            primus_path = Path(tmpdir) / "Primus"
            megatron_path = primus_path / "third_party" / "Megatron-LM"
            megatron_path.mkdir(parents=True)

            # Mock __file__ to be in primus/backends/megatron/
            fake_file = primus_path / "primus" / "backends" / "megatron" / "megatron_base_trainer.py"
            fake_file.parent.mkdir(parents=True)
            fake_file.touch()

            def mock_resolve(self):
                return fake_file

            monkeypatch.setattr(Path, "resolve", mock_resolve)

            saved_megatron = sys.modules.get("megatron", _SENTINEL := object())
            sys.modules["megatron"] = None
            original_path = sys.path[:]
            # Isolate Method 3 (file-location) by neutralizing Method 4, which
            # returns early if any sys.path entry already contains a real
            # ``megatron`` package dir. A name-only filter misses a pip-installed
            # megatron-core under site-packages (no "megatron" in the path name),
            # so also drop entries that actually contain a ``megatron`` dir.
            sys.path[:] = [
                p for p in sys.path if "megatron" not in p.lower() and not (Path(p) / "megatron").is_dir()
            ]
            try:
                trainer._ensure_megatron_path()
                assert str(megatron_path) in sys.path
            finally:
                sys.path[:] = original_path
                if saved_megatron is _SENTINEL:
                    sys.modules.pop("megatron", None)
                else:
                    sys.modules["megatron"] = saved_megatron

    def test_ensure_megatron_path_already_importable(self, monkeypatch: pytest.MonkeyPatch):
        """Test that path setup is skipped if megatron is already importable."""
        trainer = _build_trainer(monkeypatch)

        megatron_mod = types.ModuleType("megatron")
        monkeypatch.setitem(sys.modules, "megatron", megatron_mod)

        path_before = sys.path[:]
        trainer._ensure_megatron_path()

        assert sys.path == path_before

    def test_patch_parse_args(self, monkeypatch: pytest.MonkeyPatch):
        """Test that parse_args is patched in both locations."""
        trainer = _build_trainer(monkeypatch)

        import megatron.training.arguments as megatron_args_mod
        import megatron.training.initialize as megatron_init_mod

        original_parse_args_args = megatron_args_mod.parse_args
        original_parse_args_init = megatron_init_mod.parse_args

        backend_args = SimpleNamespace(test_param="value")
        trainer.backend_args = backend_args

        try:
            trainer._patch_parse_args()

            assert megatron_args_mod.parse_args is not original_parse_args_args
            assert megatron_init_mod.parse_args is not original_parse_args_init

            result_args = megatron_args_mod.parse_args()
            result_init = megatron_init_mod.parse_args()

            assert result_args is backend_args
            assert result_init is backend_args
        finally:
            megatron_args_mod.parse_args = original_parse_args_args
            megatron_init_mod.parse_args = original_parse_args_init
