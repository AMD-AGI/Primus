###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared policy and configuration for the MXFP4 linear path.

Separated from the GEMM implementations and the module swap because it is all
*policy* -- which layers are eligible, what precision the knobs select -- and
policy is the part worth reading on its own. The kernels in ``mxfp4_gemm.py``
take these as arguments and make no decisions.

ALIGNMENT, IN TWO PLACES, FOR TWO REASONS:
  ``FP4_ALIGN`` (128) applies to ``in_features``/``out_features`` and is enforced
  at swap time, so an ineligible Linear simply stays bf16. ``AITER_K_MULTIPLE``
  (256) applies to the *contraction* dimension of each GEMM, one of which is the
  token count -- a runtime property no swap-time check can see. That one is
  handled by padding in ``mxfp4_gemm.py``; see its docstring, since getting it
  wrong is silent.

Kept free of torch at import time so the patch condition and these tests can run
without it.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from primus.backends.nemo_automodel._env import env_flag, env_int, env_str
from primus.backends.nemo_automodel.quantization import _common

logger = logging.getLogger(__name__)

# Alignment the HIPBLASLT FP4 GEMM requires of in_features/out_features. The
# forward and both backward GEMMs are NT-layout, and FP4 packs two values per
# byte, which together make 128 the smallest workable multiple.
FP4_ALIGN = 128

# AITER's FP4 GEMM requires the CONTRACTION dim be a multiple of this. See
# mxfp4_gemm.py -- violating it does not raise, it returns wrong numbers.
AITER_K_MULTIPLE = 256

# Conditioning projections kept in bf16 by default.
#
# AutoModel's FP8-safe predicate names the FLUX module tree (``time_text_embed.``,
# ``norm_out.``), which does not match every model's naming, so on some models the
# timestep embedder, text embedder and AdaLN modulation would all be swapped
# despite being the classic keep-in-high-precision set. MXFP4 is considerably more
# aggressive than FP8, so the omission matters more here. These are a handful of
# Linears out of hundreds, so the cost of excluding them is negligible; the knob
# exists to A/B full coverage rather than because the default is in doubt.
FP4_SENSITIVE_PREFIXES = ("condition_embedder.",)

BACKWARD_PRECISIONS = ("mxfp4", "fp8")
SENSITIVE_PRECISIONS = ("tw_fp8", "bf16")

# Matches a transformer-block list and index in a module FQN, covering both
# ``blocks.N.`` and prefixed forms such as ``transformer_blocks.N.``.
_BLOCK_RE = re.compile(r"(?:^|\.)((?:\w+_)*blocks)\.(\d+)(?:\.|$)")


def _one_of(name: str, default: str, allowed: Tuple[str, ...]) -> str:
    """Read an env value and reject anything not in ``allowed``.

    Raising beats falling back to the default: a typo in a precision knob that
    silently trains in a different precision than the one requested leaves no
    trace in the logs.
    """
    value = env_str(name, default).lower()
    if value not in allowed:
        raise ValueError(f"{name}={value!r} invalid; expected one of {', '.join(allowed)}")
    return value


BACKEND_NAME = "turbo_mxfp4"


def is_enabled() -> bool:
    """Whether the MXFP4 swap was requested. Not the same as it being active."""
    return env_flag("PRIMUS_TURBO_FP4")


def keep_sensitive_bf16() -> bool:
    return env_flag("PRIMUS_TURBO_FP4_KEEP_SENSITIVE", default=True)


def gradient_sr_enabled() -> bool:
    """Whether to stochastically round the backward gradient quantization.

    Stochastic rounding debiases quantization, which matters more at four bits
    than at eight, and the gradients are where it matters most.

    This is a plain flag rather than a scope because Turbo now owns the decision:
    ``Float4QuantConfig.use_gradient_sr`` applies SR to exactly the backward
    gradient quantization, and the forward recipes are Turbo's to set. Applying
    SR to the forward activation and weight quantization as well would mean
    keeping a local copy of Turbo's quantization recipes -- which is what made
    this code drift out of date in the first place -- so it is deliberately not
    offered here. If it turns out to be wanted, it belongs upstream as another
    config field, not as a fork.
    """
    return env_flag("PRIMUS_TURBO_FP4_SR")


def preshuffle_enabled() -> bool:
    """Whether to use the preshuffled scale/output layout.

    Turbo dispatches to AITER when this is on and HIPBLASLT when it is off, so
    this selects the backend as much as the layout. Default on, matching Turbo's
    own tuned path.
    """
    return env_flag("PRIMUS_TURBO_FP4_PRESHUFFLE", default=True)


def backward_precision() -> str:
    """Precision of the two backward GEMMs.

      mxfp4 (default) pure MXFP4 forward and backward
      fp8             MXFP4 forward, tensorwise FP8 backward

    The FP8 backward saves bf16 activations and weights and requantizes them at
    backward time, so it gives up the activation-memory saving in exchange for a
    more mature backward GEMM. That trade is the reason to keep it as an option
    rather than a fallback.
    """
    return _one_of("PRIMUS_TURBO_FP4_BACKWARD", "mxfp4", BACKWARD_PRECISIONS)


def sensitive_band_enabled() -> bool:
    return env_flag("PRIMUS_TURBO_FP4_SENSITIVE_LAYERS")


def sensitive_start() -> int:
    """How many leading transformer blocks are in the band."""
    return env_int("PRIMUS_TURBO_FP4_SENSITIVE_START", 2)


def sensitive_end() -> int:
    """How many trailing transformer blocks are in the band."""
    return env_int("PRIMUS_TURBO_FP4_SENSITIVE_END", 8)


def sensitive_precision() -> str:
    """What the band runs in instead of MXFP4: tensorwise FP8, or bf16."""
    return _one_of("PRIMUS_TURBO_FP4_SENSITIVE_PRECISION", "tw_fp8", SENSITIVE_PRECISIONS)


def block_index(fqn: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse a block-list name and index out of a module FQN.

    ``blocks.7.attn1.to_q`` gives ``("blocks", 7)``; ``proj_out`` gives
    ``(None, None)``.
    """
    match = _BLOCK_RE.search(fqn)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def count_blocks(module) -> Dict[str, int]:
    """Map each transformer-block list prefix to its block count.

    Walks the module's Linears rather than looking for an attribute name, so the
    band can be resolved on any model without knowing how it names its block
    lists. A model with both a dual-stream and a single-stream list gets an entry
    for each, and the band applies to each independently.
    """
    import torch.nn as nn

    counts: Dict[str, int] = {}
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        list_name, idx = block_index(name)
        if list_name is not None:
            counts[list_name] = max(counts.get(list_name, 0), idx + 1)
    return counts


def is_sensitive_block(fqn: str, block_counts: Dict[str, int]) -> bool:
    """Whether ``fqn`` is in the band of first-N and last-M blocks."""
    list_name, idx = block_index(fqn)
    if list_name is None or list_name not in block_counts:
        return False
    count = block_counts[list_name]
    return idx < sensitive_start() or idx >= (count - sensitive_end())


def is_fp4_training_safe_linear(name: str, linear) -> bool:
    """Whether a Linear is eligible for MXFP4.

    Three gates, in increasing specificity: AutoModel's own FP8-safety predicate,
    then the conditioning-prefix policy above, then the 128-alignment the FP4 GEMM
    requires. Deferring to AutoModel first means the set kept in bf16 is a
    superset of the TE path's rather than a competing opinion.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    if not _is_fp8_training_safe_linear(name, linear):
        return False
    if keep_sensitive_bf16() and name.startswith(FP4_SENSITIVE_PREFIXES):
        return False
    if linear.weight.shape[0] % FP4_ALIGN != 0 or linear.weight.shape[1] % FP4_ALIGN != 0:
        return False
    return True


def resolve_fp4_config():
    """Build the MXFP4 quantization config.

    No granularity or format knobs, unlike the FP8 path: Turbo's
    ``Float4QuantConfig.__post_init__`` asserts MX_BLOCKWISE with E2M1, block size
    32 and an E8M0 scale, and its defaults already select exactly that. Offering
    choices the library rejects would only produce confusing assertion failures.

    The two fields that *are* set are checked for existence first. This code runs
    against a Turbo it was not written against, and a config field silently not
    taking effect is precisely the failure this path cannot afford -- so a
    requested option that the installed library does not have raises here rather
    than being quietly dropped.
    """
    from primus_turbo.pytorch.core.low_precision import Float4QuantConfig

    kwargs = {}
    for field, requested in (
        ("use_gradient_sr", gradient_sr_enabled()),
        ("use_preshuffle", preshuffle_enabled()),
    ):
        if field in Float4QuantConfig.__dataclass_fields__:
            kwargs[field] = requested
        elif requested:
            raise RuntimeError(
                f"Float4QuantConfig has no {field!r} field in the installed "
                f"primus_turbo, so the corresponding request cannot be honoured. "
                f"Upgrade primus_turbo or unset the env knob."
            )
    return Float4QuantConfig(**kwargs)


def resolve_hybrid_fp8_config():
    """Tensorwise FP8 config for the hybrid backward and the sensitive band.

    HYBRID format means E4M3 for forward-stage operands and E5M2 for gradients,
    which is the convention Turbo's own tensorwise FP8 GEMM uses.
    """
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )

    return Float8QuantConfig(format=Format.HYBRID, granularity=ScalingGranularity.TENSORWISE)


# Registered here rather than alongside Float4Linear so that the selector, and
# the patch condition that consults it, stay usable without torch -- Float4Linear
# subclasses nn.Linear and so cannot be imported without it. Above FP8 (10):
# asking for four bits and eight bits at once should give four.
_common.register_backend(
    BACKEND_NAME,
    precedence=20,
    is_requested=is_enabled,
    description="Primus-Turbo MXFP4 Float4Linear (gemm_fp4)",
)


def pad_multiple(rows: int, multiple: int = AITER_K_MULTIPLE) -> int:
    """Round ``rows`` up to a multiple of ``multiple``.

    Kept here, and torch-free, so the arithmetic behind the token padding can be
    tested without a GPU. The reason it exists is in ``mxfp4_gemm.py``.
    """
    return ((rows + multiple - 1) // multiple) * multiple
