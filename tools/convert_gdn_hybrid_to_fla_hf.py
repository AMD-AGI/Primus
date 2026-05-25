#!/usr/bin/env python3
"""Convert Primus (Megatron) 75% Hybrid GDN+MLA checkpoint to FLA HuggingFace format.

The hybrid model has 12 FLA mixer blocks: MLA at indices [0, 4, 8] and GDN
at the other 9 indices (matches FLA `gated_deltanet_300M_hybrid.json` exactly).

Primus Megatron stores them as 24 sublayers alternating mixer/MLP:

    sublayer   pattern  FLA mixer  what
    --------   -------  ---------  ----
    0          *        0          MLA
    1          -                   MLP
    2          M        1          GDN
    3          -                   MLP
    4          M        2          GDN
    5          -                   MLP
    6          M        3          GDN
    7          -                   MLP
    8          *        4          MLA
    9          -                   MLP
    ...                            (repeats with `*-M-M-M-` pattern)

This converter walks the 12 FLA mixer indices, finds the corresponding Megatron
sublayer (mixer + MLP), and emits FLA-format weights.

The MLA mapping uses our spec's `q_layernorm=WrappedTorchNorm` / `kv_layernorm=
WrappedTorchNorm` (added in the parity fix). Without those, the FLA-side `q_proj.1`
and `kv_proj.1` RMSNorm weights would be missing and the FLA model would NaN.

Usage (inside the container):

    python tools/convert_gdn_hybrid_to_fla_hf.py \
        --checkpoint-path output/amd/root/zebra_llama_300M_gdn_hybrid-pretrain/checkpoints/iter_0004768 \
        --output-dir output/gdn_hybrid_300M_fla_hf \
        --config /home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/gated_deltanet_300M_hybrid.json
"""
import argparse
import json
import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict

_megatron_path = str(Path(__file__).resolve().parents[1] / "third_party" / "Megatron-LM")
if _megatron_path not in sys.path:
    sys.path.insert(0, _megatron_path)


def load_megatron_checkpoint(checkpoint_path):
    model_path = Path(checkpoint_path) / "mp_rank_00" / "model_optim_rng.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"Loaded. Iteration: {checkpoint.get('iteration', '?')}")
    return checkpoint


def _get_first(state, *candidates):
    for k in candidates:
        if k in state:
            return state[k], k
    raise KeyError(
        f"None of these expected keys were found:\n  "
        + "\n  ".join(candidates)
        + "\nFirst 30 keys in checkpoint:\n  "
        + "\n  ".join(sorted(state.keys())[:30])
    )


def _build_layer_map(hybrid_pattern):
    """Walk the hybrid_override_pattern and return:
       fla_to_megatron[fla_layer_idx] = (mixer_megatron_sublayer, mlp_megatron_sublayer, kind)
       kind: 'mla' | 'gdn'

    `*` = MLA, `M` = GDN (Mamba slot), `-` = MLP."""
    fla_to_megatron = []
    cur_mixer = None
    cur_kind = None
    for i, c in enumerate(hybrid_pattern):
        if c in ("*", "M"):
            cur_mixer = i
            cur_kind = "mla" if c == "*" else "gdn"
        elif c == "-":
            if cur_mixer is None:
                # MLP without preceding mixer — should never happen for our pattern
                continue
            fla_to_megatron.append((cur_mixer, i, cur_kind))
            cur_mixer = None
            cur_kind = None
    return fla_to_megatron


def _megatron_to_fla_rope_channels(w_rope, qk_rope_head_dim):
    """Permute the rope channels of a tensor whose LAST dim is `qk_rope_head_dim`
    from Megatron's storage order to FLA's RoPE-ready order.

    Megatron stores rope channels as [c0, c1, c2, ..., c_{d-1}] and on-the-fly
    rearranges them to [c0, c2, c4, ..., c_{d-2}, c1, c3, ..., c_{d-1}] right
    before applying RoPE (multi_latent_attention branch in rope_utils.py).
    FLA's mla.py applies no such permutation — it assumes the weights are
    already in the rearranged layout.

    So when going Megatron→FLA we apply the permutation ONCE, baked into the
    saved weights. The mapping is index `i` in Megatron → index
    `i//2`               if i even   (lands in first half)
    `qk_rope_head_dim//2 + i//2`  if i odd    (lands in second half)
    which is equivalent to `cat([even-indexed, odd-indexed], dim=-1)`.
    """
    d = qk_rope_head_dim
    assert w_rope.shape[-1] == d, (
        f"_megatron_to_fla_rope_channels: last dim {w_rope.shape[-1]} != "
        f"qk_rope_head_dim {d}"
    )
    even = w_rope[..., 0::2]
    odd = w_rope[..., 1::2]
    return torch.cat([even, odd], dim=-1).contiguous()


def convert_mla_block(state, sub_mixer, prefix, fla_cfg):
    """MLA mixer Primus→FLA mapping. Writes:
        {prefix}.attn.q_proj.0/1/2.weight
        {prefix}.attn.kv_proj.0/1/2.weight
        {prefix}.attn.k_rope.weight
        {prefix}.attn.o_proj.weight

    Critical: applies the [::2, 1::2] → [first-half, second-half] permutation
    to the RoPE-carrying channels of `linear_q_up_proj` and `linear_kv_down_proj
    [kv_lora_rank:]` to match FLA's RoPE convention (see rope_utils.py
    multi_latent_attention branch which Megatron applies on-the-fly but FLA
    does not).
    """
    out = OrderedDict()
    attn_cfg = fla_cfg["attn"]
    q_lora_rank = attn_cfg["q_lora_rank"]
    kv_lora_rank = attn_cfg["kv_lora_rank"]
    qk_rope_head_dim = attn_cfg["qk_rope_head_dim"]
    num_heads = attn_cfg["num_heads"]
    qk_nope_head_dim = attn_cfg["qk_nope_head_dim"]
    v_head_dim = attn_cfg["v_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_size = fla_cfg["hidden_size"]

    # ── q path: down → norm → up ──
    q_down_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_q_down_proj.weight"]
    q_up_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_q_up_proj.weight"]
    q_norm_w, _ = _get_first(
        state, f"decoder.layers.{sub_mixer}.self_attention.q_layernorm.weight",
    )
    assert q_down_w.shape == (q_lora_rank, hidden_size)
    assert q_up_w.shape == (num_heads * qk_head_dim, q_lora_rank)
    assert q_norm_w.shape == (q_lora_rank,)

    # Permute the rope half of q_up_proj's output (per head). Reshape into
    # (num_heads, qk_head_dim, q_lora_rank), split, permute rope, recombine.
    q_up_3d = q_up_w.view(num_heads, qk_head_dim, q_lora_rank)
    q_up_nope = q_up_3d[:, :qk_nope_head_dim, :]
    q_up_rope_meg = q_up_3d[:, qk_nope_head_dim:, :]            # (h, rope, r)
    q_up_rope_meg = q_up_rope_meg.transpose(-1, -2)             # (h, r, rope)
    q_up_rope_fla = _megatron_to_fla_rope_channels(q_up_rope_meg, qk_rope_head_dim)
    q_up_rope_fla = q_up_rope_fla.transpose(-1, -2)             # (h, rope, r)
    q_up_fla = torch.cat([q_up_nope, q_up_rope_fla], dim=1)     # (h, qk_head_dim, r)
    q_up_fla = q_up_fla.reshape(num_heads * qk_head_dim, q_lora_rank).contiguous()

    out[f"{prefix}.attn.q_proj.0.weight"] = q_down_w
    out[f"{prefix}.attn.q_proj.1.weight"] = q_norm_w
    out[f"{prefix}.attn.q_proj.2.weight"] = q_up_fla

    # ── kv path: fused down → split [kv_compressed | k_rope] → norm → up ──
    kv_down_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_kv_down_proj.weight"]
    expected_down_rows = kv_lora_rank + qk_rope_head_dim
    assert kv_down_w.shape == (expected_down_rows, hidden_size)
    out[f"{prefix}.attn.kv_proj.0.weight"] = kv_down_w[:kv_lora_rank].contiguous()

    # k_rope's output channels need the same permutation. Shape (rope, hidden):
    # rope dim is the output (last after transpose), so transpose → permute → back.
    k_rope_meg = kv_down_w[kv_lora_rank:].contiguous()         # (rope, hidden)
    k_rope_meg = k_rope_meg.transpose(0, 1)                    # (hidden, rope)
    k_rope_fla = _megatron_to_fla_rope_channels(k_rope_meg, qk_rope_head_dim)
    k_rope_fla = k_rope_fla.transpose(0, 1).contiguous()       # (rope, hidden)
    out[f"{prefix}.attn.k_rope.weight"] = k_rope_fla

    kv_norm_w, _ = _get_first(
        state, f"decoder.layers.{sub_mixer}.self_attention.kv_layernorm.weight",
    )
    assert kv_norm_w.shape == (kv_lora_rank,)
    out[f"{prefix}.attn.kv_proj.1.weight"] = kv_norm_w

    kv_up_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_kv_up_proj.weight"]
    expected_up_rows = num_heads * (qk_nope_head_dim + v_head_dim)
    assert kv_up_w.shape == (expected_up_rows, kv_lora_rank)
    out[f"{prefix}.attn.kv_proj.2.weight"] = kv_up_w

    out[f"{prefix}.attn.o_proj.weight"] = state[
        f"decoder.layers.{sub_mixer}.self_attention.linear_proj.weight"
    ]
    return out


def convert_gdn_block(state, sub_mixer, prefix, fla_cfg):
    """GDN mixer Primus→FLA mapping (same as convert_gdn_to_fla_hf.py)."""
    out = OrderedDict()
    hidden_size = fla_cfg["hidden_size"]
    num_heads = fla_cfg["num_heads"]
    num_v_heads = fla_cfg.get("num_v_heads", num_heads)
    head_dim = fla_cfg["head_dim"]
    expand_v = fla_cfg.get("expand_v", 1.0)
    key_dim = num_heads * head_dim
    value_dim = num_v_heads * int(head_dim * expand_v)

    in_proj_w = state[f"decoder.layers.{sub_mixer}.mixer.in_proj.weight"]
    expected_in = key_dim * 2 + value_dim * 2 + num_v_heads * 2
    assert in_proj_w.shape[0] == expected_in, (
        f"in_proj shape {tuple(in_proj_w.shape)} expected ({expected_in},{hidden_size})"
    )
    out[f"{prefix}.attn.q_proj.weight"] = in_proj_w[:key_dim]
    out[f"{prefix}.attn.k_proj.weight"] = in_proj_w[key_dim : key_dim * 2]
    out[f"{prefix}.attn.v_proj.weight"] = in_proj_w[key_dim * 2 : key_dim * 2 + value_dim]
    out[f"{prefix}.attn.g_proj.weight"] = in_proj_w[
        key_dim * 2 + value_dim : key_dim * 2 + value_dim * 2
    ]
    out[f"{prefix}.attn.b_proj.weight"] = in_proj_w[
        key_dim * 2 + value_dim * 2 : key_dim * 2 + value_dim * 2 + num_v_heads
    ]
    out[f"{prefix}.attn.a_proj.weight"] = in_proj_w[
        key_dim * 2 + value_dim * 2 + num_v_heads :
    ]

    out[f"{prefix}.attn.A_log"] = state[f"decoder.layers.{sub_mixer}.mixer.A_log"]
    out[f"{prefix}.attn.dt_bias"] = state[f"decoder.layers.{sub_mixer}.mixer.dt_bias"]

    conv_w = state[f"decoder.layers.{sub_mixer}.mixer.conv1d.weight"]
    out[f"{prefix}.attn.q_conv1d.weight"] = conv_w[:key_dim]
    out[f"{prefix}.attn.k_conv1d.weight"] = conv_w[key_dim : key_dim * 2]
    out[f"{prefix}.attn.v_conv1d.weight"] = conv_w[key_dim * 2 :]

    out[f"{prefix}.attn.o_norm.weight"] = state[
        f"decoder.layers.{sub_mixer}.mixer.out_norm.weight"
    ]
    out[f"{prefix}.attn.o_proj.weight"] = state[
        f"decoder.layers.{sub_mixer}.mixer.out_proj.weight"
    ]
    return out


def convert(checkpoint, fla_config_path, hybrid_pattern):
    state = checkpoint["model"]
    with open(fla_config_path) as f:
        fla_cfg = json.load(f)

    num_hidden_layers = fla_cfg["num_hidden_layers"]
    intermediate_size = fla_cfg.get("intermediate_size", fla_cfg["hidden_size"] * 4)

    layer_map = _build_layer_map(hybrid_pattern)
    assert len(layer_map) == num_hidden_layers, (
        f"hybrid_pattern produced {len(layer_map)} FLA mixer blocks "
        f"but FLA config wants {num_hidden_layers}"
    )

    print(f"\nFLA layer index → Megatron sublayer mapping:")
    for fla_idx, (sub_mixer, sub_mlp, kind) in enumerate(layer_map):
        print(f"  FLA layer {fla_idx} ({kind.upper()}): mixer=sub{sub_mixer}, mlp=sub{sub_mlp}")

    hf_state = OrderedDict()
    hf_state["model.embeddings.weight"] = state["embedding.word_embeddings.weight"]

    for fla_idx, (sub_mixer, sub_mlp, kind) in enumerate(layer_map):
        prefix = f"model.layers.{fla_idx}"

        attn_norm_w, _ = _get_first(
            state,
            f"decoder.layers.{sub_mixer}.mixer.in_proj.layer_norm_weight",
            f"decoder.layers.{sub_mixer}.norm.weight",
            f"decoder.layers.{sub_mixer}.input_layernorm.weight",
        )
        hf_state[f"{prefix}.attn_norm.weight"] = attn_norm_w

        if kind == "mla":
            hf_state.update(convert_mla_block(state, sub_mixer, prefix, fla_cfg))
        else:
            hf_state.update(convert_gdn_block(state, sub_mixer, prefix, fla_cfg))

        # MLP sublayer (shared between MLA and GDN paths)
        mlp_norm_w, _ = _get_first(
            state,
            f"decoder.layers.{sub_mlp}.mlp.linear_fc1.layer_norm_weight",
            f"decoder.layers.{sub_mlp}.pre_mlp_layernorm.weight",
            f"decoder.layers.{sub_mlp}.input_layernorm.weight",
        )
        hf_state[f"{prefix}.mlp_norm.weight"] = mlp_norm_w

        fc1_w = state[f"decoder.layers.{sub_mlp}.mlp.linear_fc1.weight"]
        assert fc1_w.shape[0] == intermediate_size * 2, (
            f"MLP fc1 shape {tuple(fc1_w.shape)} expected ({intermediate_size*2},...)"
        )
        hf_state[f"{prefix}.mlp.gate_proj.weight"] = fc1_w[:intermediate_size]
        hf_state[f"{prefix}.mlp.up_proj.weight"] = fc1_w[intermediate_size:]
        hf_state[f"{prefix}.mlp.down_proj.weight"] = state[
            f"decoder.layers.{sub_mlp}.mlp.linear_fc2.weight"
        ]

    final_norm_w, _ = _get_first(
        state,
        "decoder.final_norm.weight",
        "decoder.final_layernorm.weight",
        "decoder.norm.weight",
    )
    hf_state["model.norm.weight"] = final_norm_w

    if "output_layer.weight" in state:
        hf_state["lm_head.weight"] = state["output_layer.weight"]
    else:
        hf_state["lm_head.weight"] = state["embedding.word_embeddings.weight"]

    return hf_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default="/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/gated_deltanet_300M_hybrid.json",
    )
    parser.add_argument(
        "--hybrid-pattern",
        default="*-M-M-M-*-M-M-M-*-M-M-M-",
        help="Same as YAML hybrid_override_pattern (24 chars for 300M hybrid)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Primus 75% Hybrid GDN+MLA → FLA HuggingFace Conversion")
    print("=" * 72)

    checkpoint = load_megatron_checkpoint(args.checkpoint_path)
    hf_state = convert(checkpoint, args.config, args.hybrid_pattern)
    print(f"\nConverted {len(hf_state)} parameters")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file

        if (
            "lm_head.weight" in hf_state
            and "model.embeddings.weight" in hf_state
            and hf_state["lm_head.weight"].data_ptr()
            == hf_state["model.embeddings.weight"].data_ptr()
        ):
            hf_state["lm_head.weight"] = hf_state["lm_head.weight"].clone()
        save_file(hf_state, str(output_dir / "model.safetensors"))
        print(f"  Saved {output_dir / 'model.safetensors'}")
    except ImportError:
        torch.save(hf_state, output_dir / "pytorch_model.bin")
        print(f"  Saved {output_dir / 'pytorch_model.bin'}")

    with open(args.config) as f:
        config = json.load(f)
    config["architectures"] = ["GatedDeltaNetForCausalLM"]
    config["fuse_cross_entropy"] = False
    config["fuse_norm"] = False
    config["fuse_swiglu"] = False
    config["fuse_linear_cross_entropy"] = False
    config["torch_dtype"] = "bfloat16"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 2048,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\nDone. To evaluate:")
    print(f"  lm_eval --model hf \\")
    print(f"    --model_args pretrained={output_dir},trust_remote_code=True,dtype=bfloat16 \\")
    print(f"    --tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge,lambada_openai \\")
    print(f"    --batch_size 16")


if __name__ == "__main__":
    main()
