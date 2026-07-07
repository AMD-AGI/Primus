###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import torch.distributed as dist

from primus.backends.megatron.training.global_vars import (
    get_mlflow_writer,
    set_primus_global_variables,
)
from primus.backends.megatron.training.mlflow_setup import upload_mlflow_artifacts
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0, warning_rank_0


class MegatronBaseTrainer(BaseTrainer):
    """Base trainer for Megatron-LM, handles parse_args patching."""

    def setup(self):
        """Setup Megatron runtime: set global vars and patch parse_args."""
        self._ensure_megatron_path()
        set_primus_global_variables(self.backend_args)
        self._patch_parse_args()

    def _ensure_megatron_path(self):
        """Ensure Megatron-LM path is in sys.path before any megatron imports."""
        import os
        import sys
        from pathlib import Path

        # First, check if megatron is already importable
        try:
            import megatron  # type: ignore

            return  # Path already set correctly
        except ImportError:
            pass

        # Try multiple methods to find the Megatron-LM path
        megatron_paths = []

        # Method 1: From PRIMUS_PATH environment variable (most reliable)
        primus_path = os.getenv("PRIMUS_PATH")
        if primus_path:
            megatron_path = Path(primus_path) / "third_party" / "Megatron-LM"
            if megatron_path.exists():
                megatron_paths.append(str(megatron_path))

        # Method 2: From current working directory (works in container)
        try:
            cwd = Path.cwd()
            # Check if we're in /workspace/Primus or a subdirectory
            if "Primus" in str(cwd):
                # Find the Primus root
                primus_root = cwd
                while primus_root.name != "Primus" and primus_root != primus_root.parent:
                    primus_root = primus_root.parent
                if primus_root.name == "Primus":
                    megatron_path = primus_root / "third_party" / "Megatron-LM"
                    if megatron_path.exists():
                        megatron_paths.append(str(megatron_path))
        except Exception:
            pass

        # Method 3: From current file location
        try:
            repo_root = Path(__file__).resolve().parents[3]
            megatron_path = repo_root / "third_party" / "Megatron-LM"
            if megatron_path.exists():
                megatron_paths.append(str(megatron_path))
        except Exception:
            pass

        # Method 4: Check if already in sys.path
        for path in sys.path:
            path_obj = Path(path)
            if path_obj.exists():
                megatron_pkg = path_obj / "megatron"
                if megatron_pkg.exists() and megatron_pkg.is_dir():
                    return  # Path already set correctly

        # Add paths to sys.path if not already present
        for path in megatron_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                log_rank_0(f"[Primus:MegatronBaseTrainer] Added Megatron-LM to sys.path: {path}")

        # Verify the path was set correctly by trying to import
        try:
            import megatron  # type: ignore

            log_rank_0("[Primus:MegatronBaseTrainer] Successfully verified megatron import")
        except ImportError as e:
            log_rank_0(
                f"[Primus:MegatronBaseTrainer] WARNING: Failed to import megatron after path setup: {e}"
            )
            log_rank_0(f"[Primus:MegatronBaseTrainer] sys.path: {sys.path[:5]}")

    def init(self):
        """Initialize Megatron training components."""
        log_rank_0("Initializing Megatron training...")
        # log_dict_aligned("Backend arguments", self.backend_args)

    def cleanup(self, on_error: bool = False):
        """Megatron cleanup: optional fast exit.

        * ``PRIMUS_EXIT_FAST=1`` — after ``super().cleanup()``, skip Python
          shutdown and call ``os._exit(0)`` directly. Saves ~20 s of Python
          interpreter shutdown / torchrun reaper lag at the cost of any
          ``atexit`` handlers below us. OFF by default.

        When both are off this method just delegates to ``super().cleanup()``,
        identical to the previous behavior.
        """
        super().cleanup(on_error=on_error)

        if not on_error:
            self._finalize_mlflow_artifacts()
        else:
            self._end_mlflow_run()

        exit_fast = os.environ.get("PRIMUS_EXIT_FAST", "0") == "1"

        if exit_fast and not on_error:
            log_rank_0("[MegatronBaseTrainer] PRIMUS_EXIT_FAST=1 -> os._exit(0)")
            # Flush stdout/stderr so the final log lines are not lost.
            try:
                import sys

                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # pragma: no cover
                pass
            os._exit(0)

    def _finalize_mlflow_artifacts(self):
        """Generate/upload Megatron training artifacts after the core runtime finishes."""
        args = self.backend_args
        tensorboard_dir = getattr(args, "tensorboard_dir", None)
        save_dir = getattr(args, "save", None)
        exp_root_path = None
        if save_dir:
            save_dir_abs = os.path.abspath(save_dir)
            if os.path.basename(save_dir_abs) == "checkpoints":
                exp_root_path = os.path.dirname(save_dir_abs)
        if exp_root_path is None and tensorboard_dir:
            tensorboard_dir_abs = os.path.abspath(tensorboard_dir)
            if os.path.basename(tensorboard_dir_abs) == "tensorboard":
                exp_root_path = os.path.dirname(tensorboard_dir_abs)
                log_rank_0(f"[MLflow] Fallback exp_root_path from tensorboard_dir: {exp_root_path}")

        # Bookend rank-0 console so the finalization window is visible
        # (the actual work runs on the writer/last rank).
        will_finalize = (not getattr(args, "disable_mlflow", True)) or getattr(
            args, "generate_tracelens_report", False
        )
        if will_finalize:
            log_rank_0("[MLflow] Finalizing artifacts...")

        # Barrier so all ranks flush trace/log files before the writer rank uploads.
        if dist.is_initialized():
            dist.barrier()

        try:
            upload_mlflow_artifacts(
                tensorboard_dir=tensorboard_dir,
                exp_root_path=exp_root_path,
                upload_traces=getattr(args, "mlflow_upload_traces", False),
                upload_logs=getattr(args, "mlflow_upload_logs", False),
                generate_tracelens_report=getattr(args, "generate_tracelens_report", False),
                upload_tracelens_report=getattr(args, "mlflow_upload_tracelens_report", False),
                tracelens_ranks=getattr(args, "mlflow_tracelens_ranks", None),
                tracelens_output_format=getattr(args, "mlflow_tracelens_output_format", "xlsx"),
                tracelens_cleanup_after_upload=getattr(args, "mlflow_tracelens_cleanup_after_upload", False),
                tracelens_auto_install=getattr(args, "mlflow_tracelens_auto_install", True),
            )
        except Exception as e:
            warning_rank_0(f"[MLflow] Artifact finalization failed: {e}")
        finally:
            self._end_mlflow_run()
            if dist.is_initialized():
                dist.barrier()

        if will_finalize:
            log_rank_0("[MLflow] Artifact finalization done.")

    def _end_mlflow_run(self):
        mlflow_writer = get_mlflow_writer()
        if mlflow_writer:
            mlflow_writer.end_run()

    def _patch_parse_args(self):
        """Patch Megatron's parse_args to return pre-configured Primus arguments."""
        import megatron.training.arguments as megatron_args  # type: ignore
        import megatron.training.initialize as megatron_init  # type: ignore

        log_rank_0("Patching Megatron-LM parse_args()")

        patched_parse_args = lambda *args, **kwargs: (
            log_rank_0("parse_args() called; returning Primus arguments") or self.backend_args
        )

        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        log_rank_0(f"Patched parse_args(); Primus provided {len(vars(self.backend_args))} arguments")
