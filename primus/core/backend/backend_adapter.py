###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from typing import Any, Optional


class BackendAdapter(ABC):
    """
    The unified interface for all backend frameworks.

    BackendAdapter is responsible for four main tasks:
        1. Backend initialization (env, patching, paths)
        2. Convert Primus TypedConfig → backend native config / args
        3. Build Trainer class or call backend launcher
        4. Optional hooks (patch, override behavior)
    """

    def __init__(self, framework: str):
        self.framework = framework
        # Default folder name under <repo_root>/third_party/ for this backend.
        # Subclasses can override if the directory name differs from `framework`.
        self.third_party_dir_name: Optional[str] = None

    # ============================================================================
    # Default Methods (May be overridden by subclasses)
    # ============================================================================

    def prepare_backend(self, config: Any):
        """
        Prepare backend environment before building args / constructing trainer.

        Default behavior:
          - Run backend-specific setup hooks registered in `BackendRegistry`.

        Backends can override this method if they need extra environment setup
        beyond `BackendRegistry.run_setup(self.framework)`.
        """
        from primus.core.backend.backend_registry import BackendRegistry
        from primus.modules.module_utils import log_rank_0

        BackendRegistry.run_setup(self.framework)
        log_rank_0(f"[Primus:{self.framework}] Backend prepared")

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Resolve and insert the backend's python path, then allow backend-specific sys.path tweaks.

        Note:
        This method intentionally does NOT depend on `BackendRegistry` so adapters can
        own backend loading logic. Registry should focus on adapter discovery only.
        """
        import os
        import sys
        from pathlib import Path

        from primus.modules.module_utils import log_rank_0

        def _use_path(path: str, error_msg: str) -> str:
            norm_path = os.path.abspath(os.path.normpath(str(path)))
            if os.path.exists(norm_path):
                if norm_path not in sys.path:
                    sys.path.insert(0, norm_path)
                    try:
                        log_rank_0(f"[Primus:{self.framework}] sys.path.insert → {norm_path}")
                    except Exception:
                        # Logger may not be initialized yet; sys.path is already updated.
                        pass
                return norm_path
            assert False, error_msg

        # 1) CLI argument: if provided, it must exist. No fallback.
        if backend_path:
            cli_error = (
                f"Backend path not found for '{self.framework}'.\n"
                f"Requested path: {backend_path}\n"
                f"Hint: Fix --backend_path or remove it to use BACKEND_PATH or the default third_party location."
            )
            resolved = _use_path(backend_path, cli_error)
            return resolved

        # 2) Environment variable: if set, it must exist. No fallback.
        env_path = os.getenv("BACKEND_PATH")
        if env_path:
            env_error = (
                f"BACKEND_PATH does not exist for '{self.framework}'.\n"
                f"BACKEND_PATH={env_path}\n"
                f"Hint: Fix BACKEND_PATH or unset it to use the default third_party location."
            )
            resolved = _use_path(env_path, env_error)
            return resolved

        # 3) Default: <repo_root>/third_party/<dir_name> must exist.
        dir_name = self.third_party_dir_name or self.framework
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "third_party" / dir_name
        default_error = (
            f"No valid backend path for '{self.framework}'.\n"
            f"Tried default path: {default_path}\n"
            f"Hint: Install backend to third_party/{dir_name} or provide a valid --backend_path/BACKEND_PATH."
        )
        resolved = _use_path(str(default_path), default_error)
        return resolved

    # ============================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # ============================================================================

    @abstractmethod
    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Return backend Trainer class registered in `BackendRegistry`.

        Default behavior:
          - Lookup trainer via `BackendRegistry.get_trainer_class(self.framework, stage=stage)`

        Backends can override this method if they need special resolution rules.
        """

    @abstractmethod
    def convert_config(self, params: Any) -> Any:
        """
        Convert Primus params to backend-specific arguments.

        Args:
            params: Parameters from module_config.params (SimpleNamespace or dict)

        Returns:
            Backend-specific arguments (SimpleNamespace or dict)
        """

    @abstractmethod
    def detect_backend_version(self) -> str:
        """
        Detect backend version for version-specific patches.

        Returns:
            Version string (e.g., "0.8.0", "commit:abc123")

        Raises:
            RuntimeError: If version cannot be detected

        Subclasses must implement this method and should fail fast
        if version detection is not possible.
        """
