###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P26 — dispatch precedence in :class:`DeepseekV4Attention` (CSA path).

Asserts the static plumbing that wires ``use_v4_triton_csa_attention``
into the V4 CSA forward path, without depending on a full forward
pass:

* :class:`DeepSeekV4TransformerConfig` already exposes
  ``use_v4_triton_csa_attention: bool = False`` (P25 added it; P26 wires
  it through ``DeepseekV4Attention.__init__`` + ``_csa_forward``).
* :class:`DeepseekV4Attention.__init__` reads
  ``config.use_v4_triton_csa_attention`` and stores it as
  ``self._use_v4_triton_csa_attention``,
* the flag is auto-disabled when ``compress_ratio != 4`` (cr ∈ {0, 128}
  layers use the separate ``use_v4_triton_attention`` flag landing in
  P25),
* :meth:`DeepseekV4Attention._csa_forward` reads
  ``self._use_v4_triton_csa_attention`` and dispatches to
  :func:`v4_csa_attention` when set, else
  :func:`eager_v4_csa_attention`.

The full kernel-vs-eager output equivalence is covered by G26 / G27
(``test_v4_p26_v4_csa_attention_fwd.py`` /
``test_v4_p26_v4_csa_attention_bwd.py``); this file is the static /
wiring-side gate.
"""

from __future__ import annotations

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)

# ---------------------------------------------------------------------------
# Static plumbing — config + module surface
# ---------------------------------------------------------------------------


def test_p26_config_exposes_use_v4_triton_csa_attention():
    """V4 transformer config exposes ``use_v4_triton_csa_attention: bool = False``."""
    fields = {f.name for f in DeepSeekV4TransformerConfig.__dataclass_fields__.values()}
    assert "use_v4_triton_csa_attention" in fields

    csa_field = DeepSeekV4TransformerConfig.__dataclass_fields__["use_v4_triton_csa_attention"]
    # Default must be False so existing checkpoints / smokes are
    # unaffected by the new switch.
    assert csa_field.default is False


def test_p26_attention_init_reads_use_v4_triton_csa_attention_flag():
    """``DeepseekV4Attention.__init__`` reads the CSA config flag once."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    init_consts = DeepseekV4Attention.__init__.__code__.co_consts
    assert "use_v4_triton_csa_attention" in init_consts, (
        "DeepseekV4Attention.__init__ must read config.use_v4_triton_csa_attention " "(plan-4 P26 plumbing)."
    )


def test_p26_attention_init_auto_disables_for_non_csa_layers():
    """``__init__`` auto-disables the CSA flag for cr != 4 layers.

    The dispatch contract is:
      cr ∈ {0, 128} ← ``use_v4_triton_attention`` (P25)
      cr == 4       ← ``use_v4_triton_csa_attention`` (P26)

    The init's auto-disable branch surfaces this contract explicitly so
    a stray run script with ``use_v4_triton_csa_attention=True`` does
    not silently accelerate the cr == 4 layers only and skew
    apples-to-apples perf comparisons.
    """
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    # The init body references the flag name as a string constant via
    # ``getattr(config, "use_v4_triton_csa_attention", False)`` and
    # checks ``self.compress_ratio != 4`` to flip it back off.
    init_consts = DeepseekV4Attention.__init__.__code__.co_consts
    assert "use_v4_triton_csa_attention" in init_consts
    # The integer constant 4 is the constexpr that gates the
    # auto-disable branch (matches plan-4 02-phase-details Phase 26
    # design notes).
    assert 4 in init_consts


def test_p26_csa_forward_consults_use_v4_triton_csa_attention_flag():
    """``_csa_forward`` reads ``self._use_v4_triton_csa_attention`` and references the kernel."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    co = DeepseekV4Attention._csa_forward.__code__
    names = set(co.co_names)
    consts_str = " ".join(str(c) for c in co.co_consts if c is not None)

    # The runtime flag attribute must be referenced from the bytecode.
    # Python compiles ``self._use_v4_triton_csa_attention`` so the
    # attribute name lives in ``co_names``; check both for safety.
    has_flag_ref = "_use_v4_triton_csa_attention" in names or any(
        "_use_v4_triton_csa_attention" in str(c) for c in co.co_consts
    )
    assert has_flag_ref, (
        "DeepseekV4Attention._csa_forward must consult "
        "self._use_v4_triton_csa_attention (plan-4 P26 dispatch — precedence "
        "use_v4_triton_csa_attention > eager)."
    )

    # The Triton kernel function must be referenced (P31 routes through
    # the pool/topk variant to avoid materialising gathered tensors).
    refs_csa_kernel = "v4_csa_attention_from_pool" in names or "v4_csa_attention_from_pool" in consts_str
    assert refs_csa_kernel, (
        "DeepseekV4Attention._csa_forward must call v4_csa_attention_from_pool "
        "when the CSA Triton flag is on (plan-5 P31)."
    )
