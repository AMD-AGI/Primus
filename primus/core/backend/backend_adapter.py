###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


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

    # -----------------------------------------------------------
    # 1) Backend Env Setup (Patch + Distributed + Path)
    # -----------------------------------------------------------
    @abstractmethod
    def prepare_backend(self, config: Any):
        """
        Called before creating Trainer.
        Should:
            - setup sys.path for backend
            - run backend-specific patches
            - initialize distributed detect if needed
        """

    # -----------------------------------------------------------
    # 2) Config Translation Layer
    # -----------------------------------------------------------
    @abstractmethod
    def convert_config(self, config: Any) -> Dict[str, Any]:
        """
        Convert TypedConfig (PrimusConfig + ModuleConfig) to backend args.

        Example:
            Primus PretrainConfig -> Megatron args OR Titan args

        Return:
            A dict or namespace containing args for the backend trainer.
        """

    # -----------------------------------------------------------
    # 2.5) Backend Version Detection (Optional)
    # -----------------------------------------------------------
    def detect_backend_version(self) -> Optional[str]:
        """
        Detect backend version for version-specific patches.

        Returns:
            Version string (e.g., "0.8.0", "commit:abc123") or None

        Subclasses can override this to provide version detection.
        """
        return None

    # -----------------------------------------------------------
    # 2.6) Apply Setup Patches (Automatic)
    # -----------------------------------------------------------
    def _apply_setup_patches(self, module_config):
        """
        Apply setup phase patches before backend preparation.

        This is called automatically by create_trainer() and should NOT
        be overridden by subclasses unless you have a very good reason.

        Args:
            module_config: ModuleConfig instance
        """
        from primus.core.patches import run_patches

        backend_version = self.detect_backend_version()
        model_name = getattr(module_config, "model", None)

        run_patches(
            backend=self.framework,
            phase="setup",
            backend_version=backend_version,
            model_name=model_name,
            extra={
                "config": module_config.params,
                "module_config": module_config,
            },
        )

    # -----------------------------------------------------------
    # 2.7) Apply Build Args Patches (Automatic)
    # -----------------------------------------------------------
    def _apply_build_args_patches(self, module_config, backend_args):
        """
        Apply build_args phase patches after config conversion.

        This is called automatically by create_trainer() and should NOT
        be overridden by subclasses unless you have a very good reason.

        Args:
            module_config: ModuleConfig instance
            backend_args: Backend-specific args (output of convert_config)
        """
        from primus.core.patches import run_patches

        backend_version = self.detect_backend_version()
        model_name = getattr(module_config, "model", None)

        run_patches(
            backend=self.framework,
            phase="build_args",
            backend_version=backend_version,
            model_name=model_name,
            extra={
                "args": backend_args,
                "config": module_config.params,
                "primus_config": None,  # Will be set in create_trainer
                "module_config": module_config,
            },
        )

    # -----------------------------------------------------------
    # 3) Trainer Loader
    # -----------------------------------------------------------
    @abstractmethod
    def load_trainer_class(self):
        """
        Return backend Trainer class.

        Megatron → MegatronTrainer
        Titan → TitanTrainer
        Turbo → TurboTrainer
        """

    # -----------------------------------------------------------
    # 4) Trainer Launcher (Final Step)
    # -----------------------------------------------------------
    def create_trainer(self, primus_config, module_config):
        """
        Default implementation:
            - apply setup patches (automatic)
            - prepare backend
            - convert Primus config to backend args
            - apply build_args patches (automatic)
            - instantiate trainer
        """

        # 1) apply setup patches (automatic for all backends)
        self._apply_setup_patches(module_config)

        # 2) backend env/patch/detect
        self.prepare_backend(module_config)

        # 3) config translation
        backend_args = self.convert_config(module_config)

        # 4) apply build_args patches (automatic for all backends)
        self._apply_build_args_patches(module_config, backend_args)

        # 5) load trainer class from backend
        TrainerClass = self.load_trainer_class()

        return TrainerClass(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )
