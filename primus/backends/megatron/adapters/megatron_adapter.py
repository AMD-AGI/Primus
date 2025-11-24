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

from types import SimpleNamespace

from primus.backends.megatron.builders.argument_builder import MegatronArgBuilder
from primus.backends.megatron.patches import apply_megatron_patches
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.utils.distributed_logging import log_rank_0


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

        # Phase 1: Before importing backend
        apply_megatron_patches(
            backend_version=megatron_version,
            model_name=model_name,
            phase="before_import_backend",
        )

        # Phase 2: After importing backend
        apply_megatron_patches(
            backend_version=megatron_version,
            model_name=model_name,
            phase="after_import_backend",
        )

        print(f"[Primus:MegatronAdapter] Backend prepared (version: {megatron_version})")

    def _detect_megatron_version(self) -> str:
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
            - Applies patches (before/after build_args phases)
            - Produces a Megatron-compatible argparse-like Namespace

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron args
        """
        megatron_version = self._detect_megatron_version()
        model_name = module_config.model if hasattr(module_config, "model") else None

        # Phase: Before building args
        apply_megatron_patches(
            backend_version=megatron_version,
            model_name=model_name,
            phase="before_build_args",
            extra={"config": module_config.params},
        )

        # 1. Instantiate the builder
        builder = MegatronArgBuilder()

        # 2. Feed in config params (already merged with CLI overrides in train_launcher)
        #    module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # 3. Produce the final Megatron Namespace
        megatron_args = builder.finalize()

        # Phase: After building args
        apply_megatron_patches(
            backend_version=megatron_version,
            model_name=model_name,
            phase="after_build_args",
            extra={"args": megatron_args},
        )

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

    # 4. Inject Arguments (Shared across all Megatron trainers)
    @staticmethod
    def inject_args(backend_args: SimpleNamespace) -> bool:
        """
        Inject pre-configured arguments into Megatron's runtime.

        This method provides multiple strategies for argument injection:
        1. Direct injection: Set megatron.training.global_vars._GLOBAL_ARGS
        2. Monkey patching: Replace parse_args() to return our args

        This is a static method so it can be called by any Megatron trainer
        (pretrain, sft, posttrain, etc.) without needing adapter instance.

        Args:
            backend_args: Megatron argument namespace to inject

        Returns:
            True if injection succeeded via any strategy, False otherwise
        """
        # Strategy 1: Try direct injection first (fastest, least invasive)
        direct_success = MegatronAdapter._try_direct_injection(backend_args)

        # Strategy 2: Always patch parse_args as well (most reliable)
        patch_success = MegatronAdapter._patch_parse_args(backend_args)

        if direct_success:
            log_rank_0("Args injected via both direct assignment and parse_args patching")
        elif patch_success:
            log_rank_0("Args injected via parse_args patching only")
        else:
            log_rank_0("WARNING: All injection strategies failed")
            return False

        return True

    @staticmethod
    def _try_direct_injection(backend_args: SimpleNamespace) -> bool:
        """
        Try to directly inject args into Megatron's global state.

        Args:
            backend_args: Megatron argument namespace

        Returns:
            True if successful, False otherwise
        """
        try:
            from megatron.training import global_vars  # type: ignore

            # Try to set directly (some versions have _GLOBAL_ARGS)
            if hasattr(global_vars, "_GLOBAL_ARGS"):
                global_vars._GLOBAL_ARGS = backend_args
                return True
            elif hasattr(global_vars, "_set_args"):
                global_vars._set_args(backend_args)
                return True
            else:
                return False
        except (ImportError, AttributeError) as e:
            log_rank_0(f"Cannot directly inject args: {e}")
            return False

    @staticmethod
    def _patch_parse_args(backend_args: SimpleNamespace) -> bool:
        """
        Monkey patch Megatron's parse_args to return our prepared args.

        This is the most reliable way to inject args because:
        1. Works with all Megatron versions
        2. Intercepts parse_args() wherever it's called
        3. Allows us to add custom logic (e.g., argument validation)

        Args:
            backend_args: Megatron argument namespace

        Returns:
            True if patching succeeded, False otherwise
        """
        try:
            import megatron.training.arguments as megatron_args  # type: ignore
            import megatron.training.initialize as megatron_init  # type: ignore

            # Create a function that always returns our prepared args
            def patched_parse_args(*args, **kwargs):
                log_rank_0("parse_args() called, returning pre-configured args")
                return backend_args

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(f"Patched parse_args with {len(vars(backend_args))} arguments")
            return True
        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Cannot patch parse_args: {e}")
            return False
