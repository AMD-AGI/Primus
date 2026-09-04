###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion TransformerEngine preload patch

``transformer_engine_context`` in ``maxdiffusion.train_utils`` imports
TransformerEngine as its first statement. On this ROCm/JAX stack that import
segfaults unless TensorFlow has already been loaded in the process.

This was previously repaired by a ``sed`` in
``examples/maxdiffusion/setup_maxdiffusion_env.sh`` (step 4b) that inserted
``import tensorflow`` immediately above the TE import in the vendored submodule.
That edit has the same durability problem as the Shardy ``sed``: any
``git restore`` / ``checkout`` / ``submodule update`` in
``third_party/maxdiffusion`` drops it, and the setup script then skips the
patch section because MaxDiffusion is already importable.

This patch does the equivalent from tracked Primus code:

1. Import TensorFlow at ``setup``, before ``before_train`` (which imports JAX)
   and before ``train()`` (which enters ``transformer_engine_context``).
2. Wrap ``transformer_engine_context`` so a later TE import still preloads TF
   even if something else imported TransformerEngine first through a different
   path.

The intended process-wide order is tensorflow -> jax -> transformer_engine.
"""

from contextlib import contextmanager

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import error_rank_0, log_rank_0, warning_rank_0


def _preload_tensorflow() -> bool:
    """Import TensorFlow if it is not already in the process. Return success."""
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:  # noqa: BLE001 - a missing TF must not abort training
        warning_rank_0(
            f"[Patch:maxdiffusion.te_preload] tensorflow is not importable ({exc!r}); "
            "TransformerEngine may segfault on import."
        )
        return False
    return True


@register_patch(
    patch_id="maxdiffusion.te_preload",
    backend="maxdiffusion",
    phase="setup",
    description="Preload TensorFlow before TransformerEngine to avoid an import-order segfault",
    condition=lambda ctx: True,  # Always enabled
    priority=10,  # Before other setup work; logger is independent
)
def patch_maxdiffusion_te_preload(ctx: PatchContext) -> None:
    """Load TensorFlow, then wrap MaxDiffusion's TE context so it always preloads."""
    del ctx

    if not _preload_tensorflow():
        return

    try:
        from maxdiffusion import train_utils
    except ImportError as exc:  # noqa: BLE001 - TF is already loaded; TE import is still safer
        error_rank_0(f"[Patch:maxdiffusion.te_preload] Failed to import maxdiffusion.train_utils: {exc!r}")
        log_rank_0("[Patch:maxdiffusion.te_preload] tensorflow preloaded; TE context wrap skipped.")
        return

    upstream = train_utils.transformer_engine_context
    if getattr(upstream, "_primus_tf_preload", False):
        return

    @contextmanager
    def transformer_engine_context():
        _preload_tensorflow()
        with upstream() as result:
            yield result

    transformer_engine_context._primus_tf_preload = True  # type: ignore[attr-defined]
    train_utils.transformer_engine_context = transformer_engine_context

    log_rank_0("[Patch:maxdiffusion.te_preload] tensorflow preloaded; TE context wrapped.")
