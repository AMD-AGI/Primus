###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

###############################################################################
# Primus BackendRegistry
#
# Responsibilities:
#   - Register and lookup backend adapter classes
#   - Lazy-load backend modules (on demand)
#   - Register and lookup trainer classes (per backend/stage)
#   - Register and run backend setup hooks
#
# Notes:
#   - Backend path / sys.path setup is owned by `BackendAdapter.setup_backend_path()`
#     and is orchestrated by the runtime, not the registry.
###############################################################################

from typing import Callable, Dict, List, Type

from primus.modules.module_utils import log_rank_0


class BackendRegistry:
    """
    Global registry for backend integration.

    Primus supports different training backends:
        - Megatron
        - Titan
        - JAX
        - Third-party plug-in backend

    This registry enables:
        - adapter registration (BackendAdapter subclass)
        - trainer class registration (per backend/stage)
        - framework-specific setup hook registration
    """

    # Backend → AdapterClass (class, not instance)
    _adapters: Dict[str, Type] = {}

    # (Backend, Stage) → TrainerClass
    # _trainer_classes: Dict[tuple, Type] = {}

    # Backend → list of setup hooks
    _setup_hooks: Dict[str, List[Callable]] = {}

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
        Get adapter for backend (with lazy loading).

        This method automatically:
        1. Returns an already-registered adapter if available
        2. Otherwise lazily loads the backend module (which is expected to register the adapter)
        3. Creates and returns the adapter instance

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")
            backend_path: Deprecated; backend path setup is owned by the adapter/runtime.

        Returns:
            Backend adapter instance

        Raises:
            AssertionError: If backend cannot be found/registered
            ImportError: If backend module import fails
        """
        # Ensure adapter class is registered (lazy import)
        if backend not in cls._adapters:
            cls._load_backend(backend)

        assert backend in cls._adapters, (
            f"[Primus] Backend '{backend}' not found.\n"
            f"Available backends: {', '.join(cls._adapters.keys()) if cls._adapters else 'none'}\n"
            f"Hint: Make sure '{backend}' is installed and properly configured."
        )

        # Create adapter instance; backend path setup is owned by adapter.setup_backend_path().
        return cls._adapters[backend](backend)

    @classmethod
    def has_adapter(cls, backend: str) -> bool:
        """Check if adapter is registered for backend."""
        return backend in cls._adapters

    @classmethod
    def _load_backend(cls, backend: str) -> None:
        """
        Attempt to lazily load a backend module.

        This enables on-demand loading of backends without importing
        all backends at startup.

        Args:
            backend: Backend name (e.g., "megatron", "torchtitan")

        Returns:
            None. Import errors are treated as unrecoverable and will be raised directly.
        """
        import importlib

        module_path = f"primus.backends.{backend}"
        importlib.import_module(module_path)
        log_rank_0(f"[Primus:BackendRegistry] lazy-load backend='{backend}' module='{module_path}'")

    # Backward-compatible alias; old name kept for tests and external callers.
    _try_load_backend = _load_backend

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
                cls._load_backend(item.name)
                if item.name in cls._adapters:
                    discovered.append(item.name)

        if discovered:
            log_rank_0(f"[Primus] Discovered backends: {', '.join(discovered)}")
        else:
            log_rank_0("[Primus] Warning: No backends discovered")

    # ----------------------------------------------------------------------
    #  TrainerClass Registration (optional)
    # ----------------------------------------------------------------------
    # @classmethod
    # def register_trainer_class(cls, trainer_cls: Type, backend: str, stage: str = "pretrain"):
    #     """
    #     Register trainer class for backend with optional stage.

    #     Args:
    #         trainer_cls: Trainer class to register
    #         backend: Backend name (e.g., "megatron", "torchtitan")
    #         stage: Stage name (e.g., "pretrain", "sft"). Defaults to "pretrain".
    #     """
    #     cls._trainer_classes[(backend, stage)] = trainer_cls

    # @classmethod
    # def get_trainer_class(cls, backend: str, stage: str = "pretrain"):
    #     """
    #     Get trainer class for backend.

    #     Args:
    #         backend: Backend name (e.g., "megatron", "torchtitan")
    #         stage: Stage name. Defaults to "pretrain".

    #     Returns:
    #         Trainer class

    #     Raises:
    #         ValueError: If no trainer class found for backend/stage combination
    #     """
    #     key = (backend, stage)
    #     if key in cls._trainer_classes:
    #         return cls._trainer_classes[key]

    #     raise ValueError(f"[Primus] No trainer class registered for backend '{backend}' stage '{stage}'.")

    # @classmethod
    # def has_trainer_class(cls, backend: str, stage: str = "pretrain") -> bool:
    #     """Check if trainer class is registered for backend/stage."""
    #     return (backend, stage) in cls._trainer_classes

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
        print("Adapters:         ", cls._adapters)
        print("Setup Hooks:      ", {k: len(v) for k, v in cls._setup_hooks.items()})
        print("=============================================\n")
