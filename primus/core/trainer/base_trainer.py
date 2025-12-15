###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
BaseTrainer: Universal base class for all backend trainers.

This class provides a unified training workflow with patch management
that works across all backends (Megatron, TorchTitan, JAX/Maxtext, etc.).

Design Pattern: Template Method
    - run(): Template method defining the universal training workflow
    - run_train(): Abstract method to be implemented by subclasses
    - Patch management is handled universally via run_patches()
"""

from abc import abstractmethod
from typing import Any

from primus.core.patches import run_patches
from primus.core.trainer.trainer_component import TrainerComponent
from primus.modules import module_utils


class BaseTrainer(TrainerComponent):
    """
    Universal base trainer for all backend frameworks.

    This class implements the Template Method pattern to provide a consistent
    training workflow across all backends while allowing flexibility for
    backend-specific implementations.

    Workflow (Template Method):
        1) Apply before_train patches (via run_patches)
        2) Execute training (run_train)
        3) Apply after_train patches (via run_patches)

    Subclasses (backend-specific trainers) must implement:
        - run_train(): The actual training logic
        - setup(), init(): Lifecycle methods
        - _detect_version(): Version detection (optional)

    Example hierarchy:
        BaseModule (ABC)
            ↓
        BaseTrainer (this class)
            ↓
        MegatronBaseTrainer, TorchtitanBaseTrainer, MaxtextBaseTrainer
            ↓
        MegatronPretrainTrainer, TorchtitanPretrainTrainer, etc.
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any = None):
        """
        Initialize base trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Backend-specific arguments (optional)
        """
        # Initialize BaseModule (sets self.module_config, self.module_name, etc.)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
        )

        # Store backend-specific arguments
        self.backend_args = backend_args

        # Backend metadata for patch application (required for training)
        if not hasattr(self.module_config, "framework") or not self.module_config.framework:
            raise ValueError(
                f"[{self.__class__.__name__}] 'framework' is required in module_config for training. "
                f"Please specify framework in your configuration (e.g., framework: megatron)"
            )
        if not hasattr(self.module_config, "model") or not self.module_config.model:
            raise ValueError(
                f"[{self.__class__.__name__}] 'model' is required in module_config for training. "
                f"Please specify model in your configuration (e.g., model: llama2_7B)"
            )

        self.backend_name = self.module_config.framework
        self.model_name = self.module_config.model

    def run(self):
        """
        Template method for universal training workflow.

        This method defines the standard training workflow that applies
        to all backends:
            1) Apply before_train patches (via run_patches)
            2) Execute training (via run_train())
            3) Apply after_train patches (via run_patches)

        Subclasses only need to implement:
            - run_train(): The actual training logic

        DO NOT override this method unless you have a very good reason.
        """
        module_utils.log_rank_0("=" * 80)
        module_utils.log_rank_0(f"Starting {self.backend_name.upper()} training workflow...")
        module_utils.log_rank_0("=" * 80)

        # 1) Apply before_train patches
        module_utils.log_rank_0("[1/3] Applying before_train patches...")
        patch_count = run_patches(
            backend=self.backend_name,
            phase="before_train",
            backend_version=type(self).detect_version(),
            model_name=self.model_name,
            extra={
                "args": self.backend_args,
                "primus_config": self.primus_config,
                "module_config": self.module_config,
            },
        )
        module_utils.log_rank_0(f"Applied {patch_count} patches")

        # 2) Execute training (implemented by subclass)
        module_utils.log_rank_0("[2/3] Executing training...")
        self.run_train()

        # 3) Apply after_train patches (if any)
        module_utils.log_rank_0("[3/3] Applying after_train patches...")
        patch_count = run_patches(
            backend=self.backend_name,
            phase="after_train",
            backend_version=type(self).detect_version(),
            model_name=self.model_name,
            extra={
                "args": self.backend_args,
                "primus_config": self.primus_config,
                "module_config": self.module_config,
            },
        )
        module_utils.log_rank_0(f"Applied {patch_count} patches")

        module_utils.log_rank_0("=" * 80)
        module_utils.log_rank_0("Training workflow completed successfully.")
        module_utils.log_rank_0("=" * 80)

    @abstractmethod
    def run_train(self):
        """
        Execute the actual training loop.

        This method must be implemented by subclasses to provide
        the task-specific training logic.

        Example (MegatronPretrainTrainer):
            def run_train(self):
                from megatron.training import pretrain
                pretrain(train_valid_test_datasets_provider, model_provider, ...)

        Example (TorchtitanPretrainTrainer):
            def run_train(self):
                from torchtitan.train import train
                train(config, ...)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run_train()")

    @classmethod
    @abstractmethod
    def detect_version(cls) -> str:
        """
        Detect backend version.

        This method must be implemented by backend-specific trainers
        to provide accurate version detection for their backend.

        Version detection is critical for:
            - Applying version-specific patches
            - Logging and debugging
            - Ensuring compatibility

        Returns:
            Version string (e.g., "0.15.0rc8", "commit:abc123")

        Raises:
            RuntimeError (or similar): If version detection fails and is critical.
            Implementations should fail fast rather than silently return "unknown".

        Example (MegatronBaseTrainer - fail fast):
            @classmethod
            def detect_version(cls) -> str:
                try:
                    from megatron.core import package_info
                    return package_info.__version__
                except Exception as e:
                    raise RuntimeError("Failed to detect Megatron-LM version") from e

        Example (Optional backend - graceful fallback):
            @classmethod
            def detect_version(cls) -> str:
                try:
                    import some_backend
                    return some_backend.__version__
                except Exception:
                    return "unknown"  # Only if truly optional
        """
        raise NotImplementedError(f"{cls.__name__} must implement detect_version()")
