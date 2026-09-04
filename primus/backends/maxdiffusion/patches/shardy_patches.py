###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion Shardy Patch

MaxDiffusion turns the Shardy partitioner off wherever it builds a
``cudnn_flash_te`` attention layer (two call sites in
``src/maxdiffusion/models/attention_flax.py``)::

    if attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention
      jax.config.update("jax_use_shardy_partitioner", False)

TransformerEngine's fused attention is registered through
``custom_partitioning`` with Shardy sharding rules only, so with Shardy off JAX
falls back to GSPMD and lowering aborts before the first step::

    NotImplementedError: Custom-partitioned function 'impl at
    transformer_engine/jax/cpp_extensions/attention.py:563' does not support
    GSPMD sharding propagation rules.

Every WAN and FLUX config Primus ships sets ``attention: cudnn_flash_te``, so
this is on the path of every MaxDiffusion run.

This was previously repaired by a ``sed`` in
``examples/maxdiffusion/setup_maxdiffusion_env.sh`` that rewrote both call sites
in the vendored submodule. That fix only survived while the submodule working
tree kept the edit -- a ``git restore``, ``git checkout`` or ``submodule update``
in ``third_party/maxdiffusion`` reverted it -- and the script could not re-apply
it, because it exits early whenever MaxDiffusion is already importable (true on
the MaxText image, which ships its own copy at ``/workspace/maxdiffusion``).
Guarding the config setter here keeps the behavior in tracked, reviewable code
and independent of the submodule's working-tree state.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import error_rank_0, log_rank_0, warning_rank_0

_SHARDY_FLAG = "jax_use_shardy_partitioner"

_suppression_logged = False


@register_patch(
    patch_id="maxdiffusion.shardy",
    backend="maxdiffusion",
    phase="before_train",
    description="Keep the Shardy partitioner enabled for TransformerEngine fused attention",
    condition=lambda ctx: True,  # Always enabled
)
def patch_maxdiffusion_shardy(ctx: PatchContext) -> None:
    """Ignore attempts to disable Shardy, which TE fused attention requires."""
    del ctx

    try:
        import jax
    except ImportError as exc:  # noqa: BLE001 - never abort a run over a config guard
        error_rank_0(f"[Patch:maxdiffusion.shardy] Failed to import jax: {exc!r}")
        return

    upstream_update = jax.config.update
    if getattr(upstream_update, "_primus_shardy_guard", False):
        return

    global _suppression_logged
    _suppression_logged = False

    def update(name, value, *args, **kwargs):
        # Only the disable is refused; every other config update passes through
        # untouched, including an explicit re-enable.
        if name == _SHARDY_FLAG and not value:
            global _suppression_logged
            if not _suppression_logged:
                _suppression_logged = True
                # One layer per transformer block reaches this, so report once.
                warning_rank_0(
                    "[Patch:maxdiffusion.shardy] Ignoring MaxDiffusion's attempt to disable "
                    "Shardy: TransformerEngine fused attention registers Shardy partitioning "
                    "rules only, and GSPMD lowering fails for it."
                )
            return None
        return upstream_update(name, value, *args, **kwargs)

    update._primus_shardy_guard = True
    jax.config.update = update

    # Assert the value once through the unwrapped setter: an attention layer built
    # before this patch landed would have left Shardy off.
    upstream_update(_SHARDY_FLAG, True)

    log_rank_0("[Patch:maxdiffusion.shardy] Shardy partitioner pinned on.")
