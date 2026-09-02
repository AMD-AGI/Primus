###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Transformer Engine native MXFP4 linear swap.

A third four-bit option, distinct from the Primus-Turbo MXFP4 path in
``mxfp4_linear.py``: instead of a local module wrapping Turbo's ``gemm_fp4``,
this swaps ``nn.Linear`` for ``transformer_engine.pytorch.Linear`` and runs it
under TE's own ``MXFP4BlockScaling`` recipe. Which is better is an empirical
question about two different quantization implementations, which is the reason
both exist.

Requires a TE build with MXFP4 recipe support. On a build without it, TE may
still expose the recipe *name* while lacking the machinery, so ``install`` builds
the recipe up front to fail at startup rather than mid-training.

TWO THINGS ARE NEEDED, NOT ONE:
  A bare ``te.pytorch.Linear`` runs in bf16 outside an autocast. So the swap
  alone changes nothing -- each swapped module's forward also has to be wrapped in
  an MXFP4 autocast for the TE Linears to actually quantize. A swap without the
  wrap is the quiet failure this path is most prone to, which is why the summary
  line reports what got wrapped and not just what got swapped.

=============================================================================
WHY THE AUTOCAST GOES ON EACH BLOCK AND NOT ON THE TRANSFORMER
=============================================================================
This is the subtle part, and it is an interaction with activation checkpointing.

AutoModel wraps each transformer block in a non-reentrant
``checkpoint_wrapper``. Under non-reentrant checkpointing the block's forward is
re-run during the backward pass to recompute activations, and that re-run happens
*standalone* -- it is not nested inside whatever context managers were active when
the original forward ran.

So if the autocast is entered once at the top-level transformer forward, the
recompute runs outside it, in bf16. The two executions of the same block then
quantize differently, which means they save a different number of tensors, and
non-reentrant checkpointing detects that mismatch and raises. Forced past the
error, the gradients would be wrong.

Wrapping each block's forward instead puts the autocast *inside* the checkpoint
boundary, so it is re-entered identically on recompute. Every TE Linear lives
inside a block -- the skip-list keeps the conditioning and projection layers in
bf16 -- so per-block coverage is complete.

A model may hold its blocks in more than one list. Wrapping only the first list
found leaves the rest running their TE Linears outside the autocast, silently in
bf16, so every list is wrapped.

Activation (env, no config schema change):
  PRIMUS_TE_MXFP4=1               enable the swap and autocast (default off)
  NVTE_MXFP4_USE_HADAMARD=0       apply TE's random Hadamard transform in the
                                  recipe. Read by TE itself as well; set here so
                                  the recipe object agrees with the environment.

Recommended config pairing (this path drives the autocast itself):
  model.transformer_engine_linear: true
  model.transformer_engine_fp8: false
"""
from __future__ import annotations

import logging

from primus.backends.nemo_automodel._env import env_flag
from primus.backends.nemo_automodel.quantization import _common, _fp4_common

logger = logging.getLogger(__name__)

BACKEND_NAME = "te_mxfp4"
_LOG_PREFIX = "[TE-MXFP4]"

# Block-list attributes to wrap, covering the naming conventions in use. This is
# the same enumeration the shared activation-checkpointing helper takes as an
# argument; it is not shared with it because the action differs -- that helper
# wraps blocks in a checkpoint, this one enters an autocast inside one -- and only
# the list of names is common.
DEFAULT_BLOCK_ATTRS = ("blocks", "transformer_blocks", "single_transformer_blocks")

_WRAPPED_FLAG = "_primus_te_mxfp4_wrapped"


def is_enabled() -> bool:
    """Whether the TE-native MXFP4 swap was requested."""
    return env_flag("PRIMUS_TE_MXFP4")


# Highest precedence of the low-precision swaps: it is the most specific request,
# naming both a precision and an implementation, so it should not lose to a
# broader one. See _common.register_backend for why registration is not
# activation.
_common.register_backend(
    BACKEND_NAME,
    precedence=30,
    is_requested=is_enabled,
    description="Transformer Engine native MXFP4 (te.pytorch.Linear + MXFP4BlockScaling)",
)


def build_recipe():
    """Construct TE's MXFP4 recipe.

    ``use_hadamard`` is set as an instance attribute from the environment because
    that is how TE's own recipe helpers do it, and leaving it unset would let the
    recipe object and the environment disagree about whether the transform is
    applied.
    """
    from transformer_engine.common.recipe import MXFP4BlockScaling

    recipe = MXFP4BlockScaling()
    recipe.use_hadamard = env_flag("NVTE_MXFP4_USE_HADAMARD")
    return recipe


def autocast_for(recipe):
    """Return a TE autocast context for ``recipe``, across TE API spellings.

    Newer TE exposes ``autocast(recipe=...)``; the older API is
    ``fp8_autocast(fp8_recipe=...)``, and there is an intermediate spelling that
    takes ``recipe=``. Probed rather than version-sniffed, because the version
    number has not been a reliable guide to which of these is present.
    """
    import transformer_engine.pytorch as te

    if hasattr(te, "autocast"):
        try:
            return te.autocast(enabled=True, recipe=recipe)
        except TypeError:
            pass
    try:
        return te.fp8_autocast(enabled=True, recipe=recipe)
    except TypeError:
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)


def wrap_forward(module, recipe) -> bool:
    """Wrap ``module.forward`` so each call runs under the MXFP4 autocast.

    Bound on the instance, before FSDP2 wrapping. FSDP2 keeps the module's
    ``forward`` and drives it through hooks, so the autocast is entered on every
    real forward. Returns whether it wrapped, so a double-wrap is visible to the
    caller rather than silently counted twice.
    """
    if getattr(module, _WRAPPED_FLAG, False):
        return False

    original_forward = module.forward

    def forward(*args, **kwargs):
        with autocast_for(recipe):
            return original_forward(*args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]
    setattr(module, _WRAPPED_FLAG, True)
    return True


def wrap_block_forwards(module, recipe, block_attrs=DEFAULT_BLOCK_ATTRS) -> str:
    """Enter the autocast per transformer block. See the module docstring for why.

    Returns a description of what was wrapped, for the summary line. Falls back to
    wrapping the whole module when no block list is found, which is correct but
    incompatible with activation checkpointing -- so it says so.
    """
    wrapped = []
    for attr in block_attrs:
        blocks = getattr(module, attr, None)
        if blocks is None or len(blocks) == 0:
            continue
        count = sum(wrap_forward(block, recipe) for block in blocks)
        wrapped.append(f"{attr}[{count}]")

    if wrapped:
        return "per-block " + "+".join(wrapped)

    wrap_forward(module, recipe)
    logger.warning(
        "%s no block list found among %s, so the autocast is on the whole module. "
        "This will NOT survive activation checkpointing: the recompute re-runs each "
        "block outside the autocast and the tensor counts will not match.",
        _LOG_PREFIX,
        ", ".join(block_attrs),
    )
    return "whole module (no block list found)"


def replace_linears(module, module_name: str, *, fp8_safe_only: bool = False) -> int:
    """Drop-in replacement for AutoModel's TE swap, using TE Linear plus autocast.

    ``fp8_safe_only`` is accepted for signature compatibility with the symbol
    being replaced and is ignored: the skip-list is always applied.

    Eligibility is the MXFP4 predicate, not the FP8 one, since this is a four-bit
    path with the same 128-alignment requirement.
    """
    import torch.nn as nn
    from transformer_engine.pytorch import Linear as TELinear

    converted = 0
    kept_bf16 = 0

    def walk(parent, prefix: str = "") -> None:
        nonlocal converted, kept_bf16
        # list() because children are reassigned during the walk.
        for child_name, child in list(parent.named_children()):
            fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TELinear):
                continue
            if not isinstance(child, nn.Linear):
                walk(child, fqn)
                continue
            if not _fp4_common.is_fp4_training_safe_linear(fqn, child):
                kept_bf16 += 1
                logger.debug(
                    "%s keeping %s.%s in bf16 (ineligible for MXFP4); weight=%s",
                    _LOG_PREFIX,
                    module_name,
                    fqn,
                    tuple(child.weight.shape),
                )
                continue
            replacement = TELinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=child.weight.dtype,
                device=child.weight.device,
            )
            _common.copy_linear_params(replacement, child)
            setattr(parent, child_name, replacement)
            converted += 1

    walk(module)

    # Only wrap if something was swapped: an autocast around bf16 Linears costs
    # nothing but claiming it in the log would be misleading.
    wrap_description = "nothing (no Linear was swapped)"
    if converted:
        wrap_description = wrap_block_forwards(module, build_recipe())

    logger.info(
        "%s %s: %d -> te.pytorch.Linear, %d kept in bf16; MXFP4 autocast on %s",
        _LOG_PREFIX,
        module_name,
        converted,
        kept_bf16,
        wrap_description,
    )
    return converted


def install() -> bool:
    """Rebind AutoModel's TE swap symbol to the TE-native MXFP4 swap."""
    # Build the recipe now, so a TE without MXFP4 support fails at startup. TE can
    # expose the recipe name without the machinery behind it, so importing the
    # symbol is not sufficient evidence that this path will work.
    build_recipe()

    # Best-effort capability check. Deliberately not a hard requirement: the
    # helper has moved between modules and is absent from some builds where MXFP4
    # nonetheless executes, so the recipe build above is the real gate and this
    # only improves the error message.
    for module_path in ("transformer_engine.pytorch", "primus_turbo.pytorch.core.low_precision"):
        try:
            module = __import__(module_path, fromlist=["check_mxfp4_support"])
        except ImportError:
            continue
        check = getattr(module, "check_mxfp4_support", None)
        if check is None:
            continue
        supported, reason = check()
        if not supported:
            raise RuntimeError(f"MXFP4 reported unsupported by {module_path}: {reason}")
        break
    else:
        logger.warning(
            "%s no check_mxfp4_support() found in TE or primus_turbo; proceeding "
            "because the recipe built. Confirm this image has MXFP4 support.",
            _LOG_PREFIX,
        )

    _common.install_linear_swap(replace_linears, _LOG_PREFIX)
    return True
