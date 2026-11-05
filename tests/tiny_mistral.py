###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""A tiny random Mistral HF checkpoint (``MistralForCausalLM``) with a local
tokenizer (no Hub access), and the full HF <-> converted-Megatron comparison.

Used by the per-PR conversion test (tests/unit_tests/backends/megatron/
test_native_loader_mistral_convert.py) and the weekly SFT trainer case
(tests/trainer/test_megatron_trainer.py::test_mistral_7B_v0_3). torch, safetensors,
tokenizers and transformers are imported when a function runs."""

import json
import os


def write_tokenizer(hf_dir, vocab):
    """Save a word-level HF tokenizer with exactly ``vocab`` tokens and Mistral's
    special-token ids (unk 0, bos 1, eos 2) into ``hf_dir``."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = {"<unk>": 0, "<s>": 1, "</s>": 2}
    words.update({f"w{i}": i for i in range(3, vocab)})
    tok = Tokenizer(models.WordLevel(vocab=words, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    fast.save_pretrained(hf_dir)
    assert len(fast) == vocab


def write_checkpoint(
    hf_dir,
    *,
    vocab,
    hidden,
    heads,
    kv_heads,
    ffn,
    layers,
    head_dim=None,
    dtype="float32",
    max_position_embeddings=4096,
    rope_theta=1000000.0,
    rms_norm_eps=1e-05,
    seed=0,
):
    """Write config.json, a tokenizer and random ``model.safetensors`` of a Mistral-7B-v0.3
    style checkpoint with these sizes into ``hf_dir`` (created if missing). Every
    tensor, norms included, is random, so a dropped or swapped mapping shows.
    ``head_dim`` None writes ``head_dim: null`` (hidden // heads). Returns the tensors."""
    import torch
    from safetensors.torch import save_file

    os.makedirs(hf_dir, exist_ok=True)
    write_tokenizer(hf_dir, vocab)
    hd = head_dim or hidden // heads
    assert hidden == heads * hd
    config = {
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_act": "silu",
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": dtype,
        "vocab_size": vocab,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(config, f)
    g = torch.Generator().manual_seed(seed)
    tdtype = getattr(torch, dtype)

    def r(*shape):
        return (0.02 * torch.randn(*shape, generator=g)).to(tdtype)

    def norm():  # around 1, distinct per tensor
        return (1.0 + 0.1 * torch.randn(hidden, generator=g)).to(tdtype)

    t = {
        "model.embed_tokens.weight": r(vocab, hidden),
        "model.norm.weight": norm(),
        "lm_head.weight": r(vocab, hidden),
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        t.update(
            {
                p + "input_layernorm.weight": norm(),
                p + "post_attention_layernorm.weight": norm(),
                p + "self_attn.q_proj.weight": r(heads * hd, hidden),
                p + "self_attn.k_proj.weight": r(kv_heads * hd, hidden),
                p + "self_attn.v_proj.weight": r(kv_heads * hd, hidden),
                p + "self_attn.o_proj.weight": r(hidden, heads * hd),
                p + "mlp.gate_proj.weight": r(ffn, hidden),
                p + "mlp.up_proj.weight": r(ffn, hidden),
                p + "mlp.down_proj.weight": r(hidden, ffn),
            }
        )
    save_file(t, os.path.join(hf_dir, "model.safetensors"), metadata={"format": "pt"})
    return t


def load_converted(megatron_dir):
    """The ``model`` state dict of the single-rank legacy checkpoint in ``megatron_dir``."""
    import glob

    import torch

    files = glob.glob(os.path.join(megatron_dir, "iter_*", "mp_rank_00", "model_optim_rng.pt"))
    assert len(files) == 1, f"expected one TP=1 PP=1 checkpoint file under {megatron_dir}, found {files}"
    return torch.load(files[0], map_location="cpu", weights_only=False)


def assert_matches_hf(model, hf, *, heads, kv_heads, ffn, layers):
    """Every converted Megatron tensor equals its HF source: embedding, final norm and
    head; per layer the fused QKV (per KV group: its query heads, then K, then V), the
    output projection, both norms, fc1 (gate rows, then up rows) and fc2. Every HF
    tensor is compared once."""
    import torch

    hidden = hf["model.embed_tokens.weight"].shape[1]
    hd = hf["model.layers.0.self_attn.k_proj.weight"].shape[0] // kv_heads
    used = set()

    def same(meg, name, value=None):
        used.add(name)
        got = model[meg] if value is None else value
        assert torch.equal(got, hf[name]), f"{meg} != HF {name}"

    same("embedding.word_embeddings.weight", "model.embed_tokens.weight")
    same("output_layer.weight", "lm_head.weight")
    same("decoder.final_layernorm.weight", "model.norm.weight")
    for i in range(layers):
        p, m = f"model.layers.{i}.", f"decoder.layers.{i}."
        qkv = model[m + "self_attention.linear_qkv.weight"]
        assert qkv.shape == ((heads + 2 * kv_heads) * hd, hidden)
        per_group = heads // kv_heads
        q, k, v = qkv.reshape(kv_heads, per_group + 2, hd, hidden).split([per_group, 1, 1], dim=1)
        same(m + "linear_qkv[q]", p + "self_attn.q_proj.weight", q.reshape(-1, hidden))
        same(m + "linear_qkv[k]", p + "self_attn.k_proj.weight", k.reshape(-1, hidden))
        same(m + "linear_qkv[v]", p + "self_attn.v_proj.weight", v.reshape(-1, hidden))
        same(m + "self_attention.linear_qkv.layer_norm_weight", p + "input_layernorm.weight")
        same(m + "self_attention.linear_proj.weight", p + "self_attn.o_proj.weight")
        fc1 = model[m + "mlp.linear_fc1.weight"]
        assert fc1.shape == (2 * ffn, hidden)
        same(m + "linear_fc1[gate]", p + "mlp.gate_proj.weight", fc1[:ffn])
        same(m + "linear_fc1[up]", p + "mlp.up_proj.weight", fc1[ffn:])
        same(m + "mlp.linear_fc1.layer_norm_weight", p + "post_attention_layernorm.weight")
        same(m + "mlp.linear_fc2.weight", p + "mlp.down_proj.weight")
    assert used == set(hf), f"HF tensors not compared: {sorted(set(hf) - used)}"
