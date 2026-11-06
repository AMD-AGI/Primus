###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for the native-converter unit tests: the Megatron import guard, a stand-in
for ``common.SafetensorsStore`` and a builder of Llama-layout HF tensors."""

import importlib.util
from pathlib import Path

import pytest

# runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py (the posttrain hook).
CONVERT_HOOK = (
    Path(__file__).resolve().parents[5]
    / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
)


def load_convert_hook(name="convert_hook_under_test"):
    """Import the posttrain checkpoint-conversion hook (a script, not a package module)."""
    spec = importlib.util.spec_from_file_location(name, CONVERT_HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_megatron_training():
    """Skip when megatron.core cannot be imported, then import megatron.training.

    On a CPU-only host the first import of megatron.training raises "No CUDA GPUs are
    available" from an optional dependency (fla -> mamba_ssm asks the GPU for its arch);
    fla then treats the module as unavailable and the second import succeeds."""
    pytest.importorskip("megatron.core")
    try:
        import megatron.training  # noqa: F401
    except RuntimeError as exc:
        if "CUDA" not in str(exc) and "GPU" not in str(exc):
            raise
        import megatron.training  # noqa: F401,F811


class DictStore:
    """Stands in for common.SafetensorsStore: HF tensors by name."""

    def __init__(self, tensors):
        self.tensors = tensors

    def has(self, name):
        return name in self.tensors

    def get(self, name, dtype):
        return self.tensors[name].to(dtype)


def llama_layout_hf_tensors(layers, hidden, ffn, heads, kv_heads, vocab, lm_head=False):
    """Random HF tensors of a Llama-layout checkpoint (seed 0)."""
    import torch

    head_dim = hidden // heads
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    t = {"model.embed_tokens.weight": r(vocab, hidden), "model.norm.weight": r(hidden)}
    if lm_head:
        t["lm_head.weight"] = r(vocab, hidden)
    for i in range(layers):
        h = f"model.layers.{i}."
        t[h + "input_layernorm.weight"] = r(hidden)
        t[h + "post_attention_layernorm.weight"] = r(hidden)
        t[h + "self_attn.q_proj.weight"] = r(heads * head_dim, hidden)
        t[h + "self_attn.k_proj.weight"] = r(kv_heads * head_dim, hidden)
        t[h + "self_attn.v_proj.weight"] = r(kv_heads * head_dim, hidden)
        t[h + "self_attn.o_proj.weight"] = r(hidden, heads * head_dim)
        t[h + "mlp.gate_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.up_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.down_proj.weight"] = r(hidden, ffn)
    return t


def stub_convert_pipeline(monkeypatch, cfg, tensors, build_model, save):
    """Fake the heavy native_convert_common steps of ``convert`` (Megatron setup, model build,
    save) so the converter's own functions run on CPU: ``build_model(margs)`` builds the model,
    ``save(model, sd, consumed, store, margs, save_dir, log)`` records what would be saved."""
    from primus.backends.megatron.checkpoint import native_convert_common as common

    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda stage: None)
    monkeypatch.setattr(common, "read_hf_config", lambda hf_dir: dict(cfg))
    monkeypatch.setattr(common, "init_megatron_single_process", lambda margs, port, log=None: margs)
    monkeypatch.setattr(common, "build_mcore_gpt_model", build_model)
    monkeypatch.setattr(common, "SafetensorsStore", lambda hf_dir: DictStore(tensors))
    monkeypatch.setattr(common, "load_validate_save", save)


def margs_args(hf_dir="/nonexistent", save_dir="/nonexistent"):
    """The ``args`` a converter's ``build_margs`` takes (as ``convert`` builds it), for a small
    fp32 single-rank model."""
    from types import SimpleNamespace

    return SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype="fp32",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=128,
        make_vocab_size_divisible_by=128,
        ckpt_format="torch",
    )
