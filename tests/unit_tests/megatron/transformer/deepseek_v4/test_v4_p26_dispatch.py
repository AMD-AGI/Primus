###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CSA (cr=4) attention dispatch plumbing in :class:`DeepseekV4Attention`.

Static (wiring) gate for the unified ``use_v4_csa_attention_backend`` selector:

* :class:`DeepSeekV4TransformerConfig` exposes ``use_v4_csa_attention_backend: str``
  defaulting to ``"triton_v1"``;
* :class:`DeepseekV4Attention.__init__` reads it into ``self._csa_backend``;
* :meth:`DeepseekV4Attention._csa_forward` switches on ``self._csa_backend``,
  dispatching to ``v4_csa_attention_gluon`` / ``v4_csa_attention_v2`` /
  ``v4_csa_attention_v1`` (pool) / ``v4_csa_attention_v0`` (gathered, deprecated)
  / ``eager_v4_csa_attention``.

Output equivalence is covered by the G26/G27 fwd/bwd tests; this is the static
wiring gate.
"""

from __future__ import annotations

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


def test_config_exposes_csa_backend_selector():
    """V4 config exposes ``use_v4_csa_attention_backend: str = 'triton_v1'``."""
    fields = DeepSeekV4TransformerConfig.__dataclass_fields__
    assert "use_v4_csa_attention_backend" in fields
    assert fields["use_v4_csa_attention_backend"].default == "triton_v1"


def test_attention_init_reads_csa_backend_selector():
    """``__init__`` reads config.use_v4_csa_attention_backend into ``_csa_backend``."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    init_consts = " ".join(str(c) for c in DeepseekV4Attention.__init__.__code__.co_consts if c is not None)
    assert "use_v4_csa_attention_backend" in init_consts


def test_csa_forward_switches_on_csa_backend_and_references_kernels():
    """``_csa_forward`` switches on ``self._csa_backend`` and references each entry."""
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        DeepseekV4Attention,
    )

    co = DeepseekV4Attention._csa_forward.__code__
    names = set(co.co_names)
    consts = " ".join(str(c) for c in co.co_consts if c is not None)

    assert "_csa_backend" in names
    for be in ("gluon", "triton_v2", "triton_v1", "flydsl_v0"):
        assert be in consts, f"_csa_forward must handle CSA backend value {be!r}"
    # references the pool (v1), sparse-MLA (v2), gathered (v0) and eager entries;
    # gluon is loaded lazily onto ``self._v4_csa_attention_gluon`` (attribute access).
    for fn in (
        "v4_csa_attention_v1",
        "v4_csa_attention_v2",
        "_v4_csa_attention_gluon",
        "v4_csa_attention_v0",
        "eager_v4_csa_attention",
    ):
        assert fn in names or fn in consts, f"_csa_forward must reference {fn}"
