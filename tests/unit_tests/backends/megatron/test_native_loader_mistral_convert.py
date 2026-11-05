###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-PR test of the real native Mistral conversion through the posttrain hook,
with nothing stubbed: ``01_convert_checkpoints.py --config <SFT experiment>`` (as
``primus-cli ... train posttrain`` runs it) detects the Mistral family and runs
native_loader_mistral.convert (config checks -> Llama's convert: Megatron init,
mcore model build, weight mapping, exact-mapping validation, save) on a tiny random
Mistral checkpoint with real GQA (4 query heads in 2 KV groups) and ``head_dim:
null``, with a tokenizer built locally (no Hub access). The saved Megatron
checkpoint is then compared tensor by tensor with the HF weights. (The weekly
trainer case test_mistral_7B_v0_3 runs the same hook and then SFT from its output.)

The hook runs in a subprocess, as in a real run (Megatron's process-wide globals).
It needs a GPU to import the model's kernels (the per-PR unit-test job has one)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tests.tiny_mistral import (  # noqa: E402
    assert_matches_hf,
    load_converted,
    write_checkpoint,
)

REPO = Path(__file__).resolve().parents[4]

# Building the Megatron model imports Transformer Engine / mamba_ssm, which probe a GPU
# at import; the weights are converted on CPU. Per-PR CI runs tests/unit_tests on a
# GPU runner (.github/workflows/ci.yaml, "Run Primus Core Tests").
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="building the Megatron model needs a GPU"
)
# Every size, the head size (64) and the RoPE base differ from the
# mistral_7B_v0_3.yaml preset the experiment loads (head_dim 128, base 1e6, ...), so
# the checkpoint's args show the converter took them from config.json, not the preset.
SIZES = dict(vocab=384, hidden=384, heads=6, kv_heads=2, ffn=768, layers=3)
ROPE_THETA = 250000.0
RMS_EPS = 1e-6


HOOK = REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"


def _sft_experiment(path, hf_dir):
    """An SFT experiment as a user writes it: the Mistral-7B-v0.3 preset, the HF
    checkpoint as tokenizer_model, native conversion required (no Bridge fallback)."""
    path.write_text(
        textwrap.dedent(
            f"""
            work_group: amd
            user_name: root
            exp_name: tiny-mistral-sft
            workspace: ./output
            modules:
              post_trainer:
                framework: megatron
                config: sft_trainer.yaml
                model: mistral_7B_v0_3.yaml
                overrides:
                  tokenizer_model: {hf_dir}
                  native_ckpt_convert: true
            """
        )
    )
    return path


def test_real_native_conversion_through_the_hook_matches_hf(tmp_path):
    hf_dir, data = tmp_path / "hf", tmp_path / "data"
    # head_dim: null, as recent transformers write it (hidden_size // heads)
    hf = write_checkpoint(hf_dir, head_dim=None, rope_theta=ROPE_THETA, rms_norm_eps=RMS_EPS, **SIZES)
    exp = _sft_experiment(tmp_path / "sft.yaml", hf_dir)

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DATA_PATH=str(data), NODE_RANK="0")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT"):
        env.pop(key, None)
    run = subprocess.run(
        [sys.executable, str(HOOK), "--config", str(exp)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    log = run.stdout + run.stderr
    assert run.returncode == 0, (
        log[-1500:] + "\n...\n" + "\n".join(l for l in log.splitlines() if "Error" in l)[-1500:]
    )
    out = data / "megatron_checkpoints" / "hf_native"
    assert "Native (mistral) conversion: dtype=fp32 TP=1 PP=1 EP=1" in log
    assert "mapping is COMPLETE and EXACT" in log
    assert (
        "head_dim=None" in log
    )  # head_dim: null went through Llama's translation (kv_channels from validate_args)
    # what the hook hands to the trainer
    assert f"extra.pretrained_checkpoint={out}" in run.stdout and "extra.finetune=true" in run.stdout

    ckpt = load_converted(out)
    assert_matches_hf(ckpt["model"], hf, **{k: SIZES[k] for k in ("heads", "kv_heads", "ffn", "layers")})
    args = ckpt["args"]
    got = (
        args.num_layers,
        args.hidden_size,
        args.ffn_hidden_size,
        args.num_attention_heads,
        args.num_query_groups,
        args.kv_channels,
        args.rotary_base,
        args.norm_epsilon,
    )
    want = (
        SIZES["layers"],
        SIZES["hidden"],
        SIZES["ffn"],
        SIZES["heads"],
        SIZES["kv_heads"],
        SIZES["hidden"] // SIZES["heads"],  # 64: head_dim null -> hidden // heads
        ROPE_THETA,
        RMS_EPS,
    )
    assert got == want
