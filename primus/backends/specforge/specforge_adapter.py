###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForge BackendAdapter implementation.

Unlike the in-process backends (Megatron, TorchTitan, MaxText), SpecForge owns
its own distributed launcher. Primus therefore treats it as an out-of-process
backend: the adapter only normalizes config into a command line, and the
trainer execs it. The ``pretrain/specforge/prepare.py`` hook emits
``env.RUN_MODE=single`` so ``primus-cli`` does not wrap this in torchrun.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

from primus.backends.specforge.argument_builder import (
    DEFAULT_ENTRYPOINT,
    flatten_overrides,
    resolve_specforge_root,
    specforge_mode,
)
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


class SpecForgeAdapter(BackendAdapter):
    """BackendAdapter for the SpecForge speculative-decoding trainer."""

    def __init__(self, framework: str = "specforge"):
        super().__init__(framework)
        self.third_party_dir_name = "SpecForge"

    def setup_backend_path(self, backend_path=None) -> str:
        """Resolve the SpecForge checkout, tolerating a pip-installed SpecForge.

        The base implementation asserts when no ``third_party/<name>`` exists.
        SpecForge is normally installed as a wheel in the training image, so a
        missing checkout is expected rather than fatal.
        """

        candidate = backend_path or os.getenv("SPECFORGE_ROOT") or os.getenv("BACKEND_PATH")
        if candidate and os.path.isdir(candidate):
            resolved = os.path.abspath(candidate)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            log_rank_0(f"[Primus:specforge] SpecForge root -> {resolved}")
            return resolved

        if candidate:
            warning_rank_0(f"[Primus:specforge] SpecForge path '{candidate}' does not exist; ignoring.")
        log_rank_0("[Primus:specforge] No SpecForge checkout; relying on the installed 'specforge' package.")
        return ""

    def convert_config(self, params: Any) -> Any:
        """Normalize Primus params into the fields the SpecForge CLI needs."""

        specforge_mode_value = specforge_mode(params)
        specforge_config = getattr(params, "specforge_config", None)
        if specforge_mode_value != "capture" and not specforge_config:
            raise ValueError(
                "[Primus:specforge] modules.pre_trainer must set 'specforge_config' "
                "(path to the SpecForge YAML)."
            )

        overrides = flatten_overrides(getattr(params, "specforge_overrides", None))
        output_dir = getattr(params, "output_dir", None)
        if output_dir and "output_dir" not in overrides:
            overrides["output_dir"] = str(output_dir)

        root = resolve_specforge_root(params)
        capture = flatten_overrides(getattr(params, "specforge_capture", None))

        backend_args = SimpleNamespace(
            specforge_mode=specforge_mode_value,
            specforge_config=str(specforge_config) if specforge_config else None,
            specforge_entrypoint=str(getattr(params, "specforge_entrypoint", None) or DEFAULT_ENTRYPOINT),
            specforge_root=str(root) if root is not None else None,
            specforge_overrides=overrides,
            specforge_capture=capture,
        )

        log_rank_0(
            f"[Primus:specforge] config={backend_args.specforge_config} "
            f"root={backend_args.specforge_root} overrides={len(overrides)}"
        )
        return backend_args

    def load_trainer_class(self, stage: str = "pretrain"):
        if stage == "pretrain":
            from primus.backends.specforge.specforge_pretrain_trainer import (
                SpecForgePretrainTrainer,
            )

            return SpecForgePretrainTrainer
        raise ValueError(f"[Primus:specforge] Unsupported stage: {stage}")

    def detect_backend_version(self) -> str:
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version("specforge")
            except PackageNotFoundError:
                pass
        except Exception as exc:  # pragma: no cover - importlib.metadata always present on 3.8+
            warning_rank_0(f"[Primus:specforge] importlib.metadata unavailable: {exc}")

        try:
            import specforge

            return getattr(specforge, "__version__", "unknown")
        except Exception as exc:
            warning_rank_0(f"[Primus:specforge] Failed to detect SpecForge version: {exc}")

        return "unknown"
