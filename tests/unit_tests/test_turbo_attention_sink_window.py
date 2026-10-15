# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Check attention dispatch metadata without requiring TE or a GPU installation."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class MaskType(Enum):
    causal = 1
    no_mask = 2


@pytest.fixture(scope="module")
def forward():
    # Execute the actual forward method, with only tensor/backend operations mocked.
    path = Path(__file__).parents[2] / "primus/backends/megatron/core/extensions/primus_turbo.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrimusTurboAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    namespace = {"AttnMaskType": MaskType}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"]


@pytest.mark.parametrize("sink_enabled", [False, True])
@pytest.mark.parametrize("even_only", [False, True])
@pytest.mark.parametrize("window", [0, 128])
@pytest.mark.parametrize("layer", range(1, 25))
def test_sink_toggle_preserves_attention_window(forward, sink_enabled, even_only, window, layer):
    tensor = MagicMock()
    tensor.contiguous.return_value = tensor
    tensor.permute.return_value = tensor
    tensor.view.return_value = tensor
    tensor.shape = (8192, 4, 64, 64)
    sink = MagicMock() if sink_enabled else None
    attention = MagicMock(return_value=tensor)
    module = SimpleNamespace(
        qkv_format="sbhd",
        use_sink_attention=sink_enabled,
        sinks=sink,
        sink_sliding_window=window,
        sink_window_even_layers_only=even_only,
        layer_number=layer,
        offload=False,
        softmax_scale=0.125,
        deterministic_mode=False,
        attn_kwargs={},
        attn=attention,
    )

    forward(module, tensor, tensor, tensor, None, MaskType.causal)

    kwargs = attention.call_args.kwargs
    sliding_layers = range(1, 25, 2) if even_only else range(1, 25)
    expected = (window, 0) if window and layer in sliding_layers else (-1, -1)
    assert kwargs["window_size"] == expected
    assert kwargs["sink"] is (sink.float.return_value if sink_enabled else None)
    assert kwargs["causal"] is True
