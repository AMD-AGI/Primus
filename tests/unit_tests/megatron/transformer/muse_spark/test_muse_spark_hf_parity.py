###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end parity of Megatron's Muse Spark with transformers' reference.

A tiny random ``MuseSparkForConditionalGeneration`` is saved with
``save_pretrained``, converted by ``native_loader_muse_spark`` and run through
``MuseSparkGPTModel``. The mapping must be exact and the Megatron bf16 logits
must sit as close to the HF fp32 logits as HF's own bf16 run does.

Needs a transformers release that ships ``models/muse_spark`` (>= 5.20) and a GPU.
"""

import os
import socket

import pytest
import torch

pytest.importorskip("transformers.models.muse_spark")

from transformers import MuseSparkConfig, MuseSparkForConditionalGeneration  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)
from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_muse_spark as conv,
)
from primus.core.utils import logger as primus_logger  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 256
BATCH, SEQ = 2, 48
# 6 layers puts the full-attention NoPE layers at 1 and 5 (counted back from the
# last), and a 16-token window is shorter than the sequence.
TEXT = dict(
    vocab_size=VOCAB,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
    sliding_window=16,
    num_local_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    moe_hidden_size=128,
    bos_token_id=1,
    eos_token_id=2,
)
VISION = dict(
    patch_size=2,
    pos_emb_height=2,
    pos_emb_width=2,
    num_attention_heads=2,
    num_hidden_layers=1,
    hidden_size=32,
    intermediate_size=64,
    merge_size=2,
    max_position_embeddings=4,
)
AUDIO = dict(patch_sizes=[2], hidden_sizes=[16], depths=[1], num_hidden_layers=1, out_hidden_size=256)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_logger() -> None:
    if primus_logger._logger is not None:
        return
    primus_logger.setup_logger(
        primus_logger.LoggerConfig(
            exp_root_path=os.environ.get("UT_LOG_PATH", "ut_out"),
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name="UT-muse-spark-hf",
            file_sink_level="DEBUG",
            stderr_sink_level="ERROR",
            node_ip="localhost",
            rank=0,
            world_size=1,
        ),
        is_head=False,
    )


@torch.no_grad()
def _perturb(model: MuseSparkForConditionalGeneration) -> None:
    """Move every weight away from its init so a wrong mapping shows up."""
    g = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        if not name.startswith("model.language_model.") and name != "lm_head.weight":
            continue
        if name.endswith("norm.weight"):
            p.add_(0.3 * torch.randn(p.shape, generator=g))
        elif name.endswith("residual_gate.gate"):
            p.copy_(torch.randn(p.shape, generator=g))
        elif name.endswith("mlp.gate.weight") or name == "lm_head.weight":
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(0.2 * torch.randn(b.shape, generator=g))


def _hf_logits(path, ids, dtype):
    model = MuseSparkForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="eager")
    model = model.cuda().eval()
    with torch.no_grad():
        return model(input_ids=ids.cuda()).logits.float().cpu()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.fixture(scope="module")
def hf_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("muse_spark_hf")
    cfg = MuseSparkConfig(
        text_config=TEXT,
        vision_config=VISION,
        audio_config=AUDIO,
        out_hidden_size=128,
        projector_hidden_size=64,
    )
    model = MuseSparkForConditionalGeneration(cfg).float()
    _perturb(model)
    model.save_pretrained(path, safe_serialization=True)
    ids = torch.randint(3, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(2))
    return str(path), ids


def _megatron_logits(path, ids, dtype):
    _ensure_logger()
    from megatron.core import parallel_state
    from megatron.training import global_vars

    global_vars.unset_global_variables()
    parallel_state.destroy_model_parallel()
    common.ensure_megatron_on_path()
    common.apply_convert_patches(module_name="checkpoint_convert")
    try:
        model, margs = conv.build_model(path, dtype=dtype, seq_length=SEQ, master_port=_free_port())
        store = common.SafetensorsStore(path)
        sd, consumed = conv.build_mcore_state_dict(model, store, margs, log=lambda *a: None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected
        assert not [k for k in missing if "_extra_state" not in k]
        assert store.all_keys() == consumed

        model = model.cuda().eval()
        pos = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, -1)
        with torch.no_grad():
            logits = model(input_ids=ids.cuda(), position_ids=pos, attention_mask=None).float().cpu()
    finally:
        global_vars.unset_global_variables()
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return logits[..., :VOCAB]


def _agree(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


def test_megatron_matches_hf(hf_checkpoint):
    path, ids = hf_checkpoint
    ref = _hf_logits(path, ids, torch.float32)
    hf_bf16 = _hf_logits(path, ids, torch.bfloat16)

    # Top-k routing over random experts flips under bf16 noise, so only the
    # fp32 run can pin the mapping down tightly.
    fp32 = _megatron_logits(path, ids, "fp32")
    err, agree = _rel(fp32, ref), _agree(fp32, ref)
    print(f"\nfp32: rel err {err:.2e}, argmax agreement {agree:.3f}")
    assert err < 1e-3
    assert agree > 0.99

    bf16 = _megatron_logits(path, ids, "bf16")
    err, floor = _rel(bf16, ref), _rel(hf_bf16, ref)
    print(f"bf16: rel err {err:.4f} (HF bf16 {floor:.4f}), argmax agreement {_agree(bf16, ref):.3f}")
    assert err < max(2.0 * floor, 1e-2)
