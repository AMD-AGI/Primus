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

        Delegates to the Megatron base trainer's detect_version() to keep
        version detection independent of the selected stage.

        Returns:
            Version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected
        """
        from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer

        return MegatronBaseTrainer.detect_version()

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

    # Load Trainer Class (Stage-Aware)
    def load_trainer_class(self, stage: str = "pretrain"):
        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except (ValueError, AssertionError) as exc:
            raise RuntimeError(
                "[Primus:MegatronAdapter] 'megatron' backend trainer not registered. "
                "Ensure primus.backends.megatron registers the trainer class via BackendRegistry."
            ) from exc
