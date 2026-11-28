###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PrimusTurboAttention and Megatron's per-layer sliding window.

``window_size`` / ``window_attn_skip_freq`` choose, layer by layer, sliding-window or full causal
attention (TEDotProductAttention reads them the same way). PrimusTurboAttention hands the layer's
window to Primus-Turbo's ``flash_attn_func`` on every call, ``(-1, -1)`` for none.
"""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType

from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboAttention
from tests.utils import PrimusUT

HEADS, GROUPS, HEAD_DIM = 8, 2, 16


def _attention(window_size, window_attn_skip_freq, layer_number, context_parallel_size=1):
    config = SimpleNamespace(  # the attributes PrimusTurboAttention reads from a TransformerConfig
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        kv_channels=HEAD_DIM,
        attention_dropout=0.0,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        window_size=window_size,
        window_attn_skip_freq=window_attn_skip_freq,
    )
    return PrimusTurboAttention(
        config=config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=0.5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
class TestPrimusTurboAttentionWindow(PrimusUT):
    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        import megatron.training.global_vars as global_vars_module

        args = SimpleNamespace(
            offload=False,
            offload_ops=[],
            enable_turbo_attention_float8=False,
            deterministic_mode=False,
            use_sink_attention=False,
        )
        monkeypatch.setattr(global_vars_module, "_GLOBAL_ARGS", args)

    def test_layer_window(self):
        cases = {
            # a list has one entry per layer, 1 = sliding window (the Muse Spark pattern)
            "per_layer_list": ((2047, 0), [1, 0, 1, 1], [(2047, 0), None, (2047, 0), (2047, 0)]),
            # an int n: every n-th layer attends in full
            "every_nth_full": ((127, 0), 2, [(127, 0), None, (127, 0), None]),
            "all_layers": ((127, 0), None, [(127, 0)] * 2),
            "no_window": (None, None, [None] * 2),
        }
        for name, (window_size, skip_freq, expected) in cases.items():
            for layer_number, window in enumerate(expected, start=1):
                with self.subTest(case=name, layer_number=layer_number):
                    attention = _attention(window_size, skip_freq, layer_number)
                    assert attention.layer_window_size == window
                    # what the layer reports is what it attends with; TE keeps (-1, 0) for plain causal
                    assert tuple(int(w) for w in attention.window_size) == (window or (-1, 0))

    def test_forward_passes_the_layer_window(self):
        seq, batch = 12, 2
        for layer_number, window in ((1, (5, 0)), (2, (-1, -1))):
            with self.subTest(layer_number=layer_number):
                attention = _attention((5, 0), [1, 0], layer_number)
                calls = []
                attention.attn = lambda q, k, v, **kwargs: calls.append(kwargs) or torch.zeros_like(q)
                query = torch.randn(seq, batch, HEADS, HEAD_DIM)
                key = torch.randn(seq, batch, GROUPS, HEAD_DIM)
                out = attention(query, key, torch.randn_like(key), None, attn_mask_type=AttnMaskType.causal)

                assert out.shape == (seq, batch, HEADS * HEAD_DIM)
                assert calls[0]["window_size"] == window and calls[0]["causal"] is True

    def test_unsupported_windows_are_refused(self):
        # Primus-Turbo takes a causal window (left, 0); its context-parallel attention takes none
        with pytest.raises(AssertionError, match="causal sliding window"):
            _attention((127, 127), None, 1)
        with pytest.raises(AssertionError, match="sliding window needs CP 1"):
            _attention((127, 0), None, 1, context_parallel_size=2)
