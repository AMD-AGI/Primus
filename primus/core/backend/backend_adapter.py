###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from typing import Any, Dict


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
            - prepare backend
            - convert Primus config to backend args
            - instantiate trainer
        """

        # Step 1: backend env/patch/detect
        self.prepare_backend(module_config)

        # Step 2: config translation
        backend_args = self.convert_config(module_config)

        # Print backend args for debugging (one per line)
        print(f"\n{'='*80}")
        print(f"Backend Arguments ({self.framework}):")
        print(f"{'='*80}")
        if hasattr(backend_args, "__dict__"):
            # SimpleNamespace or object with __dict__
            for key, value in sorted(vars(backend_args).items()):
                print(f"  {key:30s} = {value}")
        elif isinstance(backend_args, dict):
            # Dictionary
            for key, value in sorted(backend_args.items()):
                print(f"  {key:30s} = {value}")
        else:
            print(f"  {backend_args}")
        print(f"{'='*80}\n")

        # Step 3: load trainer class from backend
        TrainerClass = self.load_trainer_class()

        return TrainerClass(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )
