###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


###############################################################################
# Megatron Adapter (Full Production Implementation)
#
# This is the unified integration layer between Primus Runtime and Megatron-LM.
#
# Responsibilities:
#   1. Apply Megatron version-specific patches
#   2. Convert Primus ModuleConfig → Megatron native args
#   3. Load Megatron Trainer class (multiple version fallback)
#   4. Inject arguments into Megatron's runtime
#
###############################################################################


from primus.backends.megatron.builders.argument_builder import MegatronArgBuilder
from primus.backends.megatron.patches import apply_megatron_patches
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry


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
            - Detect Megatron version
            - Apply patches (version/model-specific)
            - Set environment variables
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("megatron")

        # Detect Megatron version
        megatron_version = self._detect_megatron_version()
        model_name = config.model if hasattr(config, "model") else None

        # Phase: Setup environment
        apply_megatron_patches(
            backend_version=megatron_version,
            model_name=model_name,
            phase="setup",
            extra={"config": config.params if hasattr(config, "params") else {}},
        )

        print(f"[Primus:MegatronAdapter] Backend prepared (version: {megatron_version})")

    # Override base class method for version detection
    def detect_backend_version(self) -> str:
        """
        Detect Megatron-LM version.

        Returns:
            Version string (e.g., "0.8.0") or "unknown"
        """
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
            elif hasattr(megatron, "version"):
                return megatron.version
            else:
                # Try to detect from git or package metadata
                try:
                    from importlib.metadata import version

                    return version("megatron-lm")
                except Exception:
                    pass
        except Exception:
            pass

        return "unknown"

    # 2. Config → Megatron Args
    def convert_config(self, module_config):
        """
        Convert Primus ModuleConfig → final Megatron-LM argument Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Fills missing fields using Megatron-LM defaults
            - Produces a Megatron-compatible argparse-like Namespace

        Note: build_args patches are applied automatically by the base class
        after this method returns.

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
            return BackendRegistry.get_trainer_class(self.framework)
        except ValueError as exc:
            raise RuntimeError(
                "[Primus:MegatronAdapter] 'megatron' backend not registered. "
                "Ensure primus.backends.megatron.trainers defines the trainer "
                "and imports BackendRegistry."
            ) from exc

    # Note: Use default create_trainer() from BackendAdapter
    # Megatron-specific logic (arg injection) is handled in MegatronBaseTrainer
