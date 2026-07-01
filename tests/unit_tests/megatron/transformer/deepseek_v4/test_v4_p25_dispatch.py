###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense/HCA attention dispatch plumbing in :class:`DeepseekV4Attention`.

Static (wiring) gate for the unified ``use_v4_attention_backend`` selector:

* :class:`DeepSeekV4TransformerConfig` exposes ``use_v4_attention_backend: str``
  (and ``use_v4_csa_attention_backend``) defaulting to ``"gluon"``;
* :class:`DeepseekV4Attention.__init__` reads them into ``self._attn_backend`` /
  ``self._csa_backend``;
* the dense/HCA dispatch goes through ``_attention_backend_forward``, which
  switches on ``self._attn_backend`` (``use_turbo_attention`` still wins via the
  earlier ``_use_core_attention`` branch), and the ``triton_v1`` path calls
  ``v4_attention_v1`` through ``_attention_forward_via_v4_triton``.

Output equivalence is covered by the G23/G24 fwd/bwd tests; this is the static
wiring gate.
"""

from __future__ import annotations

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


def test_config_exposes_backend_selectors():
    """V4 transformer config exposes the unified string selectors, default gluon."""
    fields = DeepSeekV4TransformerConfig.__dataclass_fields__
    assert "use_v4_attention_backend" in fields
    assert "use_v4_csa_attention_backend" in fields
    assert fields["use_v4_attention_backend"].default == "gluon"
    assert fields["use_v4_csa_attention_backend"].default == "gluon"


def test_attention_init_reads_backend_selector():
    """``DeepseekV4Attention.__init__`` reads config.use_v4_attention_backend."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    init_consts = " ".join(str(c) for c in DeepseekV4Attention.__init__.__code__.co_consts if c is not None)
    assert "use_v4_attention_backend" in init_consts
    assert "use_v4_csa_attention_backend" in init_consts


def test_backend_forward_helper_switches_on_attn_backend():
    """The dense/HCA dispatch helper switches on ``self._attn_backend``."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    assert hasattr(DeepseekV4Attention, "_attention_backend_forward")
    co = DeepseekV4Attention._attention_backend_forward.__code__
    names = set(co.co_names)
    consts = " ".join(str(c) for c in co.co_consts if c is not None)
    assert "_attn_backend" in names
    # references each supported dense/HCA backend value + its entry
    for be in ("gluon", "triton_v2", "triton_v1"):
        assert be in consts, f"helper must handle backend value {be!r}"


def test_forward_uses_backend_forward_helper():
    """``forward`` routes the dense/HCA paths through the helper."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    names = set(DeepseekV4Attention.forward.__code__.co_names)
    assert "_attention_backend_forward" in names


def test_triton_v1_helper_calls_v4_attention_v1():
    """The triton_v1 path still calls ``v4_attention_v1`` via the launcher helper."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    assert hasattr(DeepseekV4Attention, "_attention_forward_via_v4_triton")
    co = DeepseekV4Attention._attention_forward_via_v4_triton.__code__
    names = set(co.co_names)
    consts = " ".join(str(c) for c in co.co_consts if c is not None)
    assert "v4_attention_v1" in names or "v4_attention_v1" in consts
