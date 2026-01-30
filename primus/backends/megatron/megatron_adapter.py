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
        # For version detection, use the base pretrain trainer (all trainers share same version)
        TrainerClass = self.load_trainer_class(module_config=None)
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
        # Instantiate the builder
        builder = MegatronArgBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # Produce the final Megatron Namespace (with distributed env injected)
        megatron_args = builder.finalize()

        log_rank_0(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args")

        return megatron_args

    # Load Trainer Class (Version Adaptive)
    def load_trainer_class(self, module_config=None):
        """
        Load Megatron trainer class registered via BackendRegistry.
        
        Args:
            module_config: Module configuration containing the module name
                          (e.g., "sft_trainer", "pre_trainer")
                        
        Returns:
            Trainer class for the specified module type
        """
        try:
            # Determine trainer key based on module name
            module_name = module_config.name if module_config and hasattr(module_config, 'name') else None
            
            if module_name and module_name == "sft_trainer":
                trainer_key = "megatron_sft"
            else:
                # Default to pretrain trainer
                trainer_key = self.framework
            
            return BackendRegistry.get_trainer_class(trainer_key)
        except ValueError as exc:
            raise RuntimeError(
                f"[Primus:MegatronAdapter] Trainer for '{trainer_key}' not registered. "
                f"Ensure primus.backends.megatron defines the trainer "
                f"and imports BackendRegistry."
            ) from exc
