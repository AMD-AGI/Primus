###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo AutoModel BackendAdapter.

The AutoModel counterpart of ``TorchTitanAdapter`` / ``MaxTextAdapter``:

    - Resolve the AutoModel backend path (the ``third_party/Automodel`` submodule,
      or an already-installed ``nemo_automodel`` if the base image ships one).
    - Convert Primus ``module_config.params`` -> AutoModel config namespace.
    - Provide the AutoModel trainer class to the Primus runtime.
    - Expose a backend version string for diagnostics.
"""

from __future__ import annotations

import os
from typing import Any

from primus.backends.nemo_automodel.argument_builder import NemoAutomodelConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.modules.module_utils import log_rank_0, warning_rank_0


class NemoAutomodelAdapter(BackendAdapter):
    """Complete BackendAdapter implementation for NeMo AutoModel."""

    def __init__(self, framework: str = "nemo_automodel"):
        super().__init__(framework)
        # Submodule lives at ``third_party/Automodel`` (the upstream repo name),
        # matching the megatron/torchtitan convention where third_party_dir_name
        # == the checkout dir. The python package it provides is ``nemo_automodel``.
        self.third_party_dir_name = "Automodel"

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Resolve the AutoModel python path.

        The canonical source is the ``third_party/Automodel`` submodule (init via
        ``git submodule update --init third_party/Automodel`` or ``primus-cli deps
        sync``), which ``prepare.py`` installs editable on first run -- identical
        to torchtitan/megatron.

        As a fast path, if no explicit ``--backend_path`` / ``BACKEND_PATH`` is
        given and ``nemo_automodel`` is already importable (e.g. a base image that
        ships it pre-installed), we use its install location and skip the
        third_party requirement. Otherwise we
        fall back to the default resolution (CLI/env/third_party/deps-sync).
        """
        if not backend_path and not os.getenv("BACKEND_PATH"):
            try:
                import nemo_automodel  # noqa: F401

                # Return the *import root* (the directory containing the package),
                # not the package dir itself: Primus puts this path on PYTHONPATH
                # for the training subprocess, so `import nemo_automodel` must
                # resolve from it. For an installed package that is site-packages,
                # matching the third_party/<name> convention where the backend root
                # is the repo dir that *contains* the package (e.g. third_party/
                # Megatron-LM for `megatron`). Returning the package dir would put
                # the wrong dir on PYTHONPATH and risk shadowing its submodules.
                import_root = os.path.dirname(os.path.dirname(os.path.abspath(nemo_automodel.__file__)))
                log_rank_0(
                    f"[Primus:NemoAutomodelAdapter] using installed nemo_automodel import root {import_root}"
                )
                return import_root
            except Exception as exc:  # not installed -> fall back to third_party
                warning_rank_0(
                    f"[Primus:NemoAutomodelAdapter] nemo_automodel not importable ({exc}); "
                    "falling back to third_party/backend_path resolution."
                )
        return super().setup_backend_path(backend_path)

    def convert_config(self, params: Any):
        """Convert Primus params -> AutoModel config namespace (pass-through)."""
        builder = NemoAutomodelConfigBuilder()
        builder.update(params)
        cfg = builder.finalize()
        log_rank_0("[Primus:NemoAutomodelAdapter] Converted Primus module params -> AutoModel config")
        return cfg

    def load_trainer_class(self, stage: str = "pretrain"):
        """Resolve the trainer via ``BackendRegistry`` (registered in __init__)."""
        from primus.core.backend.backend_registry import BackendRegistry

        # AutoModel diffusion currently exposes a single (pretrain) trainer; treat
        # the common ``train`` alias as ``pretrain``. Reject any other stage up
        # front so the error names the real cause (unsupported stage) rather than
        # implying a broken registration.
        if stage not in ("pretrain", "train"):
            raise RuntimeError(
                f"[Primus:NemoAutomodelAdapter] '{self.framework}' backend only exposes a "
                f"'pretrain' stage (got stage={stage!r}). Finetune runs also use the pretrain "
                "trainer via model.mode: finetune in the experiment config."
            )
        resolved_stage = "pretrain"
        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=resolved_stage)
        except (ValueError, AssertionError) as exc:
            raise RuntimeError(
                f"[Primus:NemoAutomodelAdapter] '{self.framework}' trainer not registered for "
                f"stage {resolved_stage!r}. Ensure primus.backends.nemo_automodel.__init__ "
                "registers the trainer via BackendRegistry.register_trainer_class."
            ) from exc

    def detect_backend_version(self) -> str:
        try:
            import importlib.metadata as importlib_metadata
        except Exception:  # pragma: no cover
            return "unknown"
        for dist in ("nemo_automodel", "nemo-automodel"):
            try:
                return importlib_metadata.version(dist)
            except Exception:
                continue
        return "unknown"
