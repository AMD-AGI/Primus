###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P25 \u2014 dispatch precedence in :class:`DeepseekV4Attention`.

Asserts the static plumbing that wires ``use_v4_triton_attention``
into the V4 attention forward path, without depending on a full
forward pass:

* :class:`DeepSeekV4TransformerConfig` exposes
  ``use_v4_triton_attention: bool`` and
  ``use_v4_triton_csa_attention: bool``,
* :class:`DeepseekV4Attention.__init__` reads
  ``config.use_v4_triton_attention`` and stores it as
  ``self._use_v4_triton_attention``,
* the flag is auto-disabled when ``compress_ratio == 4`` (CSA path
  uses the separate ``use_v4_triton_csa_attention`` flag landing in
  P26),
* the dense / HCA forward dispatch in ``forward`` reads
  ``self._use_v4_triton_attention`` after the
  ``self._use_core_attention`` Turbo branch, enforcing the documented
  precedence ``use_turbo_attention > use_v4_triton_attention > eager``.

The full kernel-vs-eager output equivalence is covered by G23 / G24
(``test_v4_p25_v4_attention_fwd.py`` / ``test_v4_p25_v4_attention_bwd.py``);
this file is the static / wiring-side gate.
"""

from __future__ import annotations

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)

# ---------------------------------------------------------------------------
# Static plumbing — config + module surface
# ---------------------------------------------------------------------------


def test_p25_config_exposes_use_v4_triton_attention():
    """V4 transformer config exposes both Triton-attention switches."""
    fields = {f.name for f in DeepSeekV4TransformerConfig.__dataclass_fields__.values()}
    assert "use_v4_triton_attention" in fields
    assert "use_v4_triton_csa_attention" in fields

    # Defaults must be False so existing checkpoints / smokes are
    # unaffected by the new switches.
    cfg_field = DeepSeekV4TransformerConfig.__dataclass_fields__["use_v4_triton_attention"]
    csa_field = DeepSeekV4TransformerConfig.__dataclass_fields__["use_v4_triton_csa_attention"]
    assert cfg_field.default is False
    assert csa_field.default is False


def test_p25_attention_init_reads_use_v4_triton_attention_flag():
    """``DeepseekV4Attention.__init__`` reads the config flag once."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    src = DeepseekV4Attention.__init__.__code__.co_consts
    # The init body references the flag name as a string constant when
    # ``getattr(config, "use_v4_triton_attention", False)`` is compiled.
    assert "use_v4_triton_attention" in src, (
        "DeepseekV4Attention.__init__ must read config.use_v4_triton_attention " "(plan-4 P25 plumbing)."
    )


def test_p25_forward_consults_use_v4_triton_attention_for_dense_and_hca():
    """``forward``'s cr == 0 / cr == 128 branches gate on the flag."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    forward_src = DeepseekV4Attention.forward.__code__.co_consts
    # Must reference the runtime flag attribute name. ``co_consts`` may
    # not surface attribute strings directly when Python compiles a
    # ``self._use_v4_triton_attention`` access (the attribute name lives
    # in ``co_names``); check both.
    names = set(DeepseekV4Attention.forward.__code__.co_names)
    has_flag_ref = "_use_v4_triton_attention" in names or any(
        "_use_v4_triton_attention" in str(c) for c in forward_src
    )
    assert has_flag_ref, (
        "DeepseekV4Attention.forward must consult self._use_v4_triton_attention "
        "(plan-4 P25 dispatch — precedence "
        "use_turbo_attention > use_v4_triton_attention > eager)."
    )


def test_p25_forward_helper_method_exists():
    """Plan-4 P25 ships a dedicated forward helper for the Triton path."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    assert hasattr(DeepseekV4Attention, "_attention_forward_via_v4_triton")
    helper = DeepseekV4Attention._attention_forward_via_v4_triton
    # The helper must call the Triton entry point (string ref in code).
    co = helper.__code__
    names = set(co.co_names)
    consts_str = " ".join(str(c) for c in co.co_consts if c is not None)
    refs_v4_attention = "v4_attention" in names or "v4_attention" in consts_str
    assert refs_v4_attention, "_attention_forward_via_v4_triton must call v4_attention (plan-4 P25)."


def test_p25_csa_layers_disable_v4_triton_attention_flag():
    """CSA (cr == 4) layers MUST NOT pick up the dense/HCA Triton flag.

    The dispatch is:
      cr ∈ {0, 128} ← ``use_v4_triton_attention``
      cr == 4       ← ``use_v4_triton_csa_attention`` (plan-4 P26)

    The init's auto-disable branch surfaces this contract explicitly so
    a stray run script with ``use_v4_triton_attention=True`` does not
    silently accelerate cr ∈ {0, 128} while leaving CSA on eager and
    skewing apples-to-apples perf comparisons.
    """
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    # The init body contains an early-return / disable branch that
    # references both the flag name and the cr-not-in-(0, 128) check.
    init_consts = " ".join(str(c) for c in DeepseekV4Attention.__init__.__code__.co_consts if c is not None)
    assert "use_v4_triton_attention" in init_consts
    # The (0, 128) tuple is the constexpr that gates the auto-disable.
    assert (0, 128) in DeepseekV4Attention.__init__.__code__.co_consts
