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
        Create a new adapter instance for backend.
        """
        if backend not in cls._adapters:
            raise KeyError(f"[Primus] No adapter registered for backend '{backend}'.")
        return cls._adapters[backend](backend)

    @classmethod
    def has_adapter(cls, backend: str) -> bool:
        return backend in cls._adapters

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
