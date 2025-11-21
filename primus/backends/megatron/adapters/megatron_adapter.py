###############################################################################
# Megatron Adapter (Full Production Implementation)
#
# This is the unified integration layer between Primus Runtime and Megatron-LM.
#
# Responsibilities:
#   1. Apply Megatron version-specific patches
#   2. Convert Primus ModuleConfig → Megatron native args
#   3. Load Megatron Trainer class (multiple version fallback)
#
###############################################################################

import os

from primus.backends.megatron.builders.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry

# from primus.backend.megatron.patches import apply_megatron_patches


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

    # 1. Backend Setup & Patches
    def prepare_backend(self, config):
        """
        Megatron-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - Apply patches (multi-version compatible)
            - Lazy-import Megatron utilities
        """
        # Run setup hooks from BackendRegistry (patch, version detection)
        BackendRegistry.run_setup("megatron")

        # Apply internal Megatron patches (fix known issues)
        # apply_megatron_patches()

        # Extra env handling: distributed variables
        if "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        print(f"[Primus:MegatronAdapter] Backend prepared.")

    # 2. Config → Megatron Args
    def convert_config(self, module_config):
        """
        Convert Primus ModuleConfig → final Megatron-LM argument Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Fills missing fields using Megatron-LM defaults
            - Produces a Megatron-compatible argparse-like Namespace

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron args
        """

        # 1. Instantiate the builder
        builder = MegatronArgBuilder()

        # 2. Feed in config params (already merged with CLI overrides in train_launcher)
        #    module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # 3. Produce the final Megatron Namespace
        megatron_args = builder.finalize()

        print(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args.")

        return megatron_args

    # 3. Load Trainer Class (Version Adaptive)
    def load_trainer_class(self):
        """Load Megatron trainer class registered via BackendRegistry."""
        try:
            return BackendRegistry.get_trainer(self.framework)
        except ValueError as exc:
            raise RuntimeError(
                "[Primus:MegatronAdapter] 'megatron' backend not registered. "
                "Ensure primus.modules.trainer.megatron.pre_trainer defines the trainer "
                "and imports BackendRegistry."
            ) from exc
