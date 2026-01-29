###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


# Trigger registration of all Megatron patches (args_patches, env_patches, etc.)
import primus.backends.megatron.patches  # noqa: F401
from primus.backends.megatron.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class MegatronAdapter(BackendAdapter):
    """
    The complete BackendAdapter implementation for Megatron-LM.

    This adapter is designed to:
        - Handle multi-version Megatron differences
        - Convert Primus config → Megatron args using ArgBuilder
        - Apply patches automatically (PR fixes, kernel bugs, attention fixes)
        - Load the appropriate Trainer class depending on Megatron version
    """

    def __init__(self, framework="megatron"):
        super().__init__(framework)
        # Temporary storage for module_config during trainer creation
        # This allows load_trainer_class() to access task information
        self._current_module_config = None

    # Backend Setup & Patches
    def prepare_backend(self, config):
        """
        Megatron-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - Set environment variables

        Note: setup patches are applied automatically by the base class
        before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("megatron")

        log_rank_0("[Primus:MegatronAdapter] Backend prepared")

    # Override base class method for version detection
    def detect_backend_version(self) -> str:
        """
        Detect Megatron-LM version.

        Delegates to the trainer class's detect_version() classmethod to ensure
        consistency and proper separation of concerns.

        Returns:
            Version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected
        """
        # Get trainer class and call its detect_version classmethod
        TrainerClass = self.load_trainer_class()
        return TrainerClass.detect_version()

    # Config → Megatron Args
    def convert_config(self, module_config):
        """
        Convert Primus ModuleConfig → final Megatron-LM argument Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Fills missing fields using Megatron-LM defaults
            - Injects distributed environment variables (via builder)
            - Produces a Megatron-compatible argparse-like Namespace

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron args
        """
        # Store module_config for later use in load_trainer_class()
        self._current_module_config = module_config
        
        # Instantiate the builder
        builder = MegatronArgBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # Produce the final Megatron Namespace (with distributed env injected)
        megatron_args = builder.finalize()

        log_rank_0(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args")

        return megatron_args

    # Load Trainer Class (Task Adaptive - pretrain vs SFT)
    def load_trainer_class(self):
        """
        Load appropriate Megatron trainer class based on task type.
        
        The megatron backend supports multiple training tasks:
        - pretrain: Uses MegatronPretrainTrainer  
        - SFT (supervised fine-tuning): Uses MegatronSFTTrainer
        
        Task detection is based on configuration markers:
        - If is_instruction_dataset=True or similar SFT markers → SFT trainer
        - Otherwise → Pretrain trainer (default)
        
        Returns:
            Trainer class for the current task
        """
        # Detect training task based on module_config
        is_sft = self._is_sft_task()
        
        if is_sft:
            log_rank_0("[Primus:MegatronAdapter] Detected SFT task, loading MegatronSFTTrainer")
            from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
            return MegatronSFTTrainer
        else:
            log_rank_0("[Primus:MegatronAdapter] Detected pretrain task, loading MegatronPretrainTrainer")
            # Use the default registered trainer (MegatronPretrainTrainer)
            try:
                return BackendRegistry.get_trainer_class(self.framework)
            except (ValueError, AssertionError) as exc:
                raise RuntimeError(
                    "[Primus:MegatronAdapter] 'megatron' backend not registered. "
                    "Ensure primus.backends.megatron defines the trainer "
                    "and imports BackendRegistry."
                ) from exc
    
    def _is_sft_task(self):
        """
        Determine if the current task is SFT based on module_config.
        
        Detection strategy:
        1. Check for explicit SFT marker (is_instruction_dataset, is_sft)
        2. Default to False (pretrain)
        
        Note: We DO NOT use finetune_lr as an indicator, as it can be used
        for non-SFT fine-tuning tasks or continued pretraining.
        
        Returns:
            True if SFT task, False otherwise
        """
        if self._current_module_config is None:
            return False
        
        # Check for explicit SFT markers in params
        params = getattr(self._current_module_config, 'params', None)
        if params:
            # Handle both SimpleNamespace and dict params
            if isinstance(params, dict):
                # params is a dictionary
                if params.get('is_instruction_dataset', False):
                    return True
                if params.get('is_sft', False):
                    return True
            else:
                # params is a SimpleNamespace or similar object
                if getattr(params, 'is_instruction_dataset', False):
                    return True
                if getattr(params, 'is_sft', False):
                    return True
        
        # Default to pretrain
        return False
