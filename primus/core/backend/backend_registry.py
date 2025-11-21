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
# This is the foundation of Primus's plugin-based backend system.
###############################################################################

from typing import Callable, Dict, List, Type


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
    _path_names: Dict[str, str] = {}

    # Backend → AdapterClass (class, not instance)
    _adapters: Dict[str, Type] = {}

    # Backend → TrainerClass (optional)
    _trainer_classes: Dict[str, Type] = {}

    # Backend → list of setup hooks
    _setup_hooks: Dict[str, List[Callable]] = {}

    @classmethod
    def initialize(cls):
        """
        Initialize backend registry by loading all available backends.

        This should be called early in the application lifecycle,
        typically in main.py before any training commands are executed.
        """
        import importlib

        backend_modules = [
            "primus.backends.megatron",
            "primus.backends.torchtitan",
        ]

        print("[Primus] Initializing BackendRegistry...")
        loaded = []
        failed = []

        for mod in backend_modules:
            try:
                importlib.import_module(mod)
                backend_name = mod.split(".")[-1]
                loaded.append(backend_name)
            except ModuleNotFoundError:
                # Backend not installed, skip
                backend_name = mod.split(".")[-1]
                failed.append(f"{backend_name} (not installed)")
            except Exception as e:
                backend_name = mod.split(".")[-1]
                failed.append(f"{backend_name} (error: {e})")

        if loaded:
            print(f"[Primus] Loaded backends: {', '.join(loaded)}")
        if failed:
            print(f"[Primus] Skipped backends: {', '.join(failed)}")

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
        if backend not in cls._path_names:
            raise KeyError(f"[Primus] No path name registered for backend '{backend}'.")
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
    def get_adapter(cls, backend: str):
        """
        Get adapter for backend (with lazy loading support).

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")

        Returns:
            Backend adapter instance

        Raises:
            ValueError: If backend not found or unavailable
            RuntimeError: If adapter creation fails
        """
        # Try lazy load if not registered
        if backend not in cls._adapters:
            cls._try_load_backend(backend)

        # Still not found - provide helpful error
        if backend not in cls._adapters:
            available = list(cls._adapters.keys()) if cls._adapters else ["none"]
            raise ValueError(
                f"[Primus] Backend '{backend}' not found.\n"
                f"Available backends: {', '.join(available)}\n"
                f"Hint: Make sure '{backend}' is installed and properly configured."
            )

        # Create adapter instance with error handling
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
    def _try_load_backend(cls, backend: str):
        """
        Attempt to lazily load a backend module.

        This enables on-demand loading of backends without importing
        all backends at startup.

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")
        """
        import importlib

        try:
            module_path = f"primus.backends.{backend}"
            importlib.import_module(module_path)
            print(f"[Primus] Loaded backend: {backend}")
        except ModuleNotFoundError:
            # Backend not installed, ignore silently
            pass
        except Exception as e:
            print(f"[Primus] Warning: Failed to load backend '{backend}': {e}")

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
