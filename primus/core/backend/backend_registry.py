###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

###############################################################################
# Primus BackendRegistry
#
# This module manages registration & lookup for:
#   - Backend Path Names (used by _setup_backend_path)
#   - Backend Adapters (MegatronAdapter, TitanAdapter, TurboAdapter...)
#   - Backend Trainer Classes (optional)
#   - Backend Setup Hooks (patches or environment initialization)
#
# Design Philosophy:
#   - Pure Lazy Loading: Backends are loaded on-demand when first accessed
#   - No Hard-coded Backend List: All backends discovered dynamically
#   - Zero Startup Overhead: No upfront imports or initialization
#   - Fail-Safe: Missing backends don't break other backends
#
# This is the foundation of Primus's plugin-based backend system.
###############################################################################

from typing import Callable, Dict, List, Type

from primus.modules.module_utils import error_rank_0, log_rank_0


class BackendRegistry:
    """
    Global registry for backend integration.

    Primus supports different training backends:
        - Megatron
        - Titan
        - JAX
        - Third-party plug-in backend

    This registry enables:
        - path name registration (for third_party/<path_name>)
        - adapter registration (BackendAdapter)
        - trainer class registration (optional)
        - framework-specific setup hook registration
    """

    # Backend → third_party folder name
    _path_names: Dict[str, str] = {
        # Pre-register known path names to avoid chicken-egg problem
        "megatron": "Megatron-LM",
        "torchtitan": "torchtitan",
    }

    # Backend → AdapterClass (class, not instance)
    _adapters: Dict[str, Type] = {}

    # Backend → TrainerClass (optional)
    _trainer_classes: Dict[str, Type] = {}

    # Backend → list of setup hooks
    _setup_hooks: Dict[str, List[Callable]] = {}

    # ----------------------------------------------------------------------
    #  Path Name Registration
    # ----------------------------------------------------------------------
    @classmethod
    def register_path_name(cls, backend: str, path_name: str):
        """
        Register mapping: framework_name → directory name under third_party/.
        e.g., register_path_name("megatron", "Megatron-LM")
        """
        cls._path_names[backend] = path_name

    @classmethod
    def get_path_name(cls, backend: str) -> str:
        """
        Get path name for backend, with lazy loading support.

        If backend not registered, try to load it first.
        """
        # Try lazy load if not registered
        if backend not in cls._path_names:
            cls._try_load_backend(backend)

        if backend not in cls._path_names:
            raise KeyError(
                f"[Primus] No path name registered for backend '{backend}'.\n"
                f"Available backends: {', '.join(cls._path_names.keys())}"
            )
        return cls._path_names[backend]

    # ----------------------------------------------------------------------
    #  Backend Adapter Registration
    # ----------------------------------------------------------------------
    @classmethod
    def register_adapter(cls, backend: str, adapter_cls: Type):
        """
        Register BackendAdapter subclass:
            register_adapter("megatron", MegatronAdapter)
        """
        cls._adapters[backend] = adapter_cls

    @classmethod
    def get_adapter(cls, backend: str, backend_path=None):
        """
        Get adapter for backend (with lazy loading and automatic path setup).

        This method automatically:
        1. Sets up backend path in sys.path (if not already done)
        2. Lazy loads backend module if not registered
        3. Creates and returns adapter instance

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")
            backend_path: Optional explicit path(s) to backend installation

        Returns:
            Backend adapter instance

        Raises:
            ValueError: If backend not found or unavailable
            RuntimeError: If adapter creation fails
            FileNotFoundError: If backend path cannot be found
        """
        # Step 1: Setup backend path (idempotent - won't duplicate if already in sys.path)
        try:
            cls.setup_backend_path(backend, backend_path=backend_path, verbose=True)
        except KeyError:
            # Backend path name not registered yet, will try to load backend first
            pass
        except FileNotFoundError as e:
            # Path not found - provide helpful error
            raise FileNotFoundError(
                f"{e}\n" f"Requested backend: '{backend}'\n" "This backend requires installation before use."
            ) from e

        # Step 2: Try lazy load if not registered
        if backend not in cls._adapters:
            loaded = cls._try_load_backend(backend)
            if loaded:
                # After loading, try path setup again (backend may have registered path_name)
                try:
                    cls.setup_backend_path(backend, backend_path=backend_path, verbose=True)
                except (KeyError, FileNotFoundError):
                    # Ignore if still can't setup - backend might not need external path
                    pass

        # Step 3: Check if adapter is now available
        if backend not in cls._adapters:
            available = list(cls._adapters.keys()) if cls._adapters else ["none"]
            raise ValueError(
                f"[Primus] Backend '{backend}' not found.\n"
                f"Available backends: {', '.join(available)}\n"
                f"Hint: Make sure '{backend}' is installed and properly configured."
            )

        # Step 4: Create adapter instance with error handling
        try:
            return cls._adapters[backend](backend)
        except Exception as e:
            raise RuntimeError(f"[Primus] Failed to create adapter for '{backend}': {e}") from e

    @classmethod
    def has_adapter(cls, backend: str) -> bool:
        """Check if adapter is registered for backend."""
        return backend in cls._adapters

    # ----------------------------------------------------------------------
    #  Backend Discovery & Lazy Loading
    # ----------------------------------------------------------------------
    @classmethod
    def setup_backend_path(cls, backend: str, backend_path=None, verbose=True) -> str:
        """
        Insert Python import path for backend modules.

        Priority:
            1. backend_path argument (--backend-path CLI option)
            2. BACKEND_PATH environment variable
            3. primus/third_party/<backend_dir_name>

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")
            backend_path: Optional explicit path(s) to backend
            verbose: Whether to print sys.path insertion message

        Returns:
            The path that was added to sys.path

        Raises:
            FileNotFoundError: If no valid backend path found
        """
        import os
        import sys
        from pathlib import Path

        candidate_paths = []

        # 1) CLI argument
        if backend_path:
            candidate_paths.append(backend_path)

        # 2) Environment variable
        env_path = os.getenv("BACKEND_PATH")
        if env_path:
            candidate_paths.append(env_path)

        # 3) Default fallback: third_party/<backend_dir_name>
        backend_dir_name = cls.get_path_name(backend)
        # Navigate from this file to project root: primus/core/backend/backend_registry.py -> <repo_root>/
        primus_root = Path(__file__).resolve().parents[3]
        default_path = primus_root / "third_party" / backend_dir_name
        candidate_paths.append(default_path)

        # Normalize paths and remove duplicates
        normalized = list(dict.fromkeys(os.path.abspath(os.path.normpath(p)) for p in candidate_paths))

        # Insert first valid path
        for p in normalized:
            if os.path.exists(p):
                if p not in sys.path:
                    sys.path.insert(0, p)
                    if verbose:
                        log_rank_0(f"[Primus] sys.path.insert → {p}")
                return p

        raise FileNotFoundError(
            f"[Primus] No valid backend path for '{backend}'.\n"
            f"Tried: {normalized}\n"
            f"Hint: Use --backend_path to specify the backend installation path, or\n"
            f"      set BACKEND_PATH environment variable, or\n"
            f"      install backend to primus/third_party/{backend_dir_name}"
        )

    @classmethod
    def _try_load_backend(cls, backend: str) -> bool:
        """
        Attempt to lazily load a backend module.

        This enables on-demand loading of backends without importing
        all backends at startup.

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")

        Returns:
            True if backend was loaded successfully, False otherwise
        """
        import importlib

        try:
            module_path = f"primus.backends.{backend}"
            importlib.import_module(module_path)
            return True
        except Exception as e:
            error_rank_0(f"[Primus] Warning: Failed to load backend '{backend}': {e}")
            return False

    @classmethod
    def list_available_backends(cls) -> list:
        """
        List all currently registered backends.

        Returns:
            List of backend names
        """
        return list(cls._adapters.keys())

    @classmethod
    def discover_all_backends(cls):
        """
        Auto-discover and load all backends from primus/backends/.

        This scans the backends directory and attempts to load each
        backend module found.
        """
        from pathlib import Path

        # Find backends directory relative to this file
        backends_dir = Path(__file__).parent.parent.parent / "backends"

        if not backends_dir.exists():
            print(f"[Primus] Warning: Backends directory not found: {backends_dir}")
            return

        discovered = []
        for item in backends_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
                cls._try_load_backend(item.name)
                if item.name in cls._adapters:
                    discovered.append(item.name)

        if discovered:
            print(f"[Primus] Discovered backends: {', '.join(discovered)}")
        else:
            print("[Primus] Warning: No backends discovered")

    # ----------------------------------------------------------------------
    #  TrainerClass Registration (optional)
    # ----------------------------------------------------------------------
    @classmethod
    def register_trainer_class(cls, backend: str, trainer_cls: Type):
        """
        Register trainer class for backend (optional).
        This is useful for simple backends or Primus-native trainer classes.
        """
        cls._trainer_classes[backend] = trainer_cls

    @classmethod
    def get_trainer_class(cls, backend: str):
        if backend not in cls._trainer_classes:
            raise KeyError(f"[Primus] No trainer class registered for backend '{backend}'.")
        return cls._trainer_classes[backend]

    @classmethod
    def has_trainer_class(cls, backend: str) -> bool:
        return backend in cls._trainer_classes

    # ----------------------------------------------------------------------
    # Setup Hook Registration
    # ----------------------------------------------------------------------
    @classmethod
    def register_setup_hook(cls, backend: str, hook_fn: Callable):
        """
        Register a function to run during backend setup.
        Example uses:
            - environment fixes
            - rank synchronization setup
            - patch pipeline initialization
        """
        if backend not in cls._setup_hooks:
            cls._setup_hooks[backend] = []
        cls._setup_hooks[backend].append(hook_fn)

    @classmethod
    def run_setup(cls, backend: str):
        """
        Run setup hooks registered for this backend.
        Adapter.prepare_backend() will typically call this first.

        Hooks run in registration order.
        """
        hooks = cls._setup_hooks.get(backend, [])
        if not hooks:
            return

        print(f"[Primus:BackendSetup] Running {len(hooks)} setup hooks for backend '{backend}'.")
        for hook in hooks:
            try:
                hook()
            except Exception as e:
                print(f"[Primus:BackendSetup] Error in setup hook: {e}")

    # ----------------------------------------------------------------------
    # Debug / Dump
    # ----------------------------------------------------------------------
    @classmethod
    def debug_dump(cls):
        print("\n========== Primus BackendRegistry ==========")
        print("Path Names:       ", cls._path_names)
        print("Adapters:         ", cls._adapters)
        print("Trainer Classes:  ", cls._trainer_classes)
        print("Setup Hooks:      ", {k: len(v) for k, v in cls._setup_hooks.items()})
        print("=============================================\n")
