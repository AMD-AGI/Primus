###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PrimusTurboAttention's sliding window against TEDotProductAttention's.

Muse Spark's head shape (64-dim heads, 8 query heads per KV head) with a
window shorter than the sequence. Layer 1 is a sliding-window layer, layer 2
a full-attention one, picked by ``window_attn_skip_freq`` as in Muse.
"""

import socket
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("primus_turbo")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEDotProductAttention,
)
from megatron.core.transformer.enums import AttnBackend, AttnMaskType  # noqa: E402
from megatron.core.transformer.transformer_config import (  # noqa: E402
    TransformerConfig,
)

SEQ, BATCH, HEADS, GROUPS, HEAD_DIM, WINDOW = 1024, 2, 16, 2, 64, 256


@pytest.fixture(scope="module")
def parallel_state():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
        )
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    yield
    ps.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture
def modules(parallel_state, monkeypatch):
    del parallel_state
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    from megatron.training.global_vars import set_args

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboAttention,
    )

    set_args(
        SimpleNamespace(
            use_sink_attention=False,
            sink_sliding_window=0,
            sink_window_even_layers_only=True,
            deterministic_mode=False,
            offload=False,
            offload_ops="",
            enable_turbo_attention_float8=False,
        )
    )
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        kv_channels=HEAD_DIM,
        window_size=(WINDOW - 1, 0),
        window_attn_skip_freq=[1, 0],
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.auto,
    )

    def build(cls, layer_number):
        return cls(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

    return {
        layer: (build(PrimusTurboAttention, layer), build(TEDotProductAttention, layer)) for layer in (1, 2)
    }


def _qkv():
    g = torch.Generator(device="cuda").manual_seed(0)
    shapes = [(SEQ, BATCH, HEADS, HEAD_DIM)] + [(SEQ, BATCH, GROUPS, HEAD_DIM)] * 2
    return [torch.randn(s, generator=g, device="cuda", dtype=torch.bfloat16).requires_grad_() for s in shapes]


def _run(attention, qkv, grad):
    qkv = [t.detach().clone().requires_grad_() for t in qkv]
    out = attention(*qkv, attention_mask=None, attn_mask_type=AttnMaskType.causal)
    out.backward(grad)
    return out.float(), [t.grad.float() for t in qkv]


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("layer, window", [(1, (WINDOW - 1, 0)), (2, (-1, -1))])
def test_turbo_matches_te(modules, layer, window):
    turbo, te = modules[layer]
    assert turbo.flash_window_size == window
    qkv = _qkv()
    grad = torch.randn(SEQ, BATCH, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    out, grads = _run(turbo, qkv, grad)
    ref_out, ref_grads = _run(te, qkv, grad)
    errors = [_rel(out, ref_out)] + [_rel(g, r) for g, r in zip(grads, ref_grads)]
    print(f"\nlayer {layer} window {window}: rel err out/dq/dk/dv " + " ".join(f"{e:.2e}" for e in errors))
    assert max(errors) < 2e-2


def test_window_changes_the_output(modules):
    qkv = _qkv()
    grad = torch.zeros(SEQ, BATCH, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    windowed, _ = _run(modules[1][0], qkv, grad)
    full, _ = _run(modules[2][0], qkv, grad)
    assert _rel(windowed, full) > 0.1


def test_checkpoint_keys_match_te(modules):
    """No TE ``_extra_state`` is saved, so either attention class loads the other's checkpoint."""
    from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

    metadata = ensure_metadata_has_dp_cp_group(None)
    turbo, te = modules[1]
    assert turbo.sharded_state_dict("attn.", metadata=metadata) == {}
    assert te.sharded_state_dict("attn.", metadata=metadata) == {}

    turbo.sinks = torch.nn.Parameter(torch.zeros(HEADS, device="cuda"))
    assert list(turbo.sharded_state_dict("attn.", metadata=metadata)) == ["attn.sinks"]


def test_muse_shape_runs_on_flydsl(modules):
    """Forcing FlyDSL raises if it cannot take the inputs instead of falling back to aiter."""
    backend = pytest.importorskip("primus_turbo.pytorch.core.backend")
    BackendType, GlobalBackendManager = backend.BackendType, backend.GlobalBackendManager

    if torch.cuda.get_device_capability() < (9, 5):
        pytest.skip("FlyDSL attention needs gfx950")
    if not hasattr(GlobalBackendManager, "set_attn_backend") or not hasattr(BackendType, "FLYDSL"):
        pytest.skip("this primus_turbo cannot pin FlyDSL attention in code")
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL)
    try:
        qkv = _qkv()
        grad = torch.randn(SEQ, BATCH, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        out, _ = _run(modules[1][0], qkv, grad)
    finally:
        GlobalBackendManager.set_attn_backend(None)
    ref, _ = _run(modules[1][1], qkv, grad)
    assert _rel(out, ref) < 2e-2
