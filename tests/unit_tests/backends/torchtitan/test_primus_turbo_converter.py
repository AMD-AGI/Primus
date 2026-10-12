###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The primus_turbo model converter must follow the turbo_attention patch gate.

Model configs list ``converters: [primus_turbo]`` unconditionally. Replacing
attention with TurboAttention while the stock TorchTitan Attention is in place
(turbo attention or Primus-Turbo disabled) fails at the first forward with
unexpected ``block_mask`` / ``enable_gqa`` keyword arguments.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchtitan")

from primus.backends.torchtitan.primus_turbo_extensions import (  # noqa: E402
    primus_turbo_converter as converter_mod,
)


def _job_config(enable_primus_turbo, use_turbo_attention):
    return SimpleNamespace(
        primus_turbo=SimpleNamespace(
            enable_primus_turbo=enable_primus_turbo,
            use_turbo_attention=use_turbo_attention,
            enable_attention_float8=False,
        )
    )


@pytest.mark.parametrize(
    "enable_primus_turbo, use_turbo_attention, replaced",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_converter_replaces_attention_only_with_turbo_attention(
    monkeypatch, enable_primus_turbo, use_turbo_attention, replaced
):
    calls = []
    monkeypatch.setattr(converter_mod, "replace_turbo_attention_modules", lambda m, cfg: calls.append(m))
    converter = converter_mod.PrimusTurboConverter(
        _job_config(enable_primus_turbo, use_turbo_attention), parallel_dims=None
    )
    model = torch.nn.Linear(2, 2)

    assert converter.convert(model) is model
    assert calls == ([model] if replaced else [])
