#!/usr/bin/env python3
"""
Convert Primus (Megatron) Pure KDA checkpoint → FLA HuggingFace format.

This is the inverse of `tools/convert_fla_kda_init_to_megatron.py`. It takes
the trained `iter_NNNNNNN/mp_rank_00/model_optim_rng.pt` file Primus writes
and emits a directory loadable by `transformers.AutoModelForCausalLM` via
FLA's `KDAForCausalLM` (`trust_remote_code=True`).

Primus's KDA layer fuses six FLA "hidden_states → X" projections into a
single column-parallel `in_proj`:

    Primus `decoder.layers.{kda}.mixer.in_proj.weight`
        ←→  cat([q_proj, k_proj, v_proj, f_proj.0, g_proj.0, b_proj], dim=0)

The converter splits this back apart and remaps the rest 1:1.

Layer ordering: Primus's HybridStack interleaves
    even  index → KDA mixer
    odd   index → MLP
so `fla_layer_idx` maps to `gdn_idx = 2*i` and `mlp_idx = 2*i+1`.

Usage
-----
    python3 tools/convert_kda_to_fla_hf.py \\
        --checkpoint-path output/amd/root/zebra_llama_300M_kda_pure-pretrain/checkpoints/iter_0004768 \\
        --output-dir output/kda_pure_300M_fla_hf \\
        --config /home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/kda_300M_pure.json

Then evaluate with lm-eval:
    lm_eval --model hf \\
        --model_args pretrained=output/kda_pure_300M_fla_hf,trust_remote_code=True,dtype=bfloat16 \\
        --tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge \\
        --batch_size 16
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

# Megatron must be on sys.path so torch.load can unpickle the ShardedTensor
# wrappers Primus writes into the checkpoint.
_megatron_path = str(Path(__file__).resolve().parents[1] / "third_party" / "Megatron-LM")
if _megatron_path not in sys.path:
    sys.path.insert(0, _megatron_path)


def load_megatron_checkpoint(checkpoint_path: Path) -> dict:
    model_path = Path(checkpoint_path) / "mp_rank_00" / "model_optim_rng.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Expected layout: <checkpoint-path>/mp_rank_00/model_optim_rng.pt"
        )
    print(f"[load] {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"[load] iteration={ckpt.get('iteration', '?')}")
    return ckpt


def _get_first(state, *candidates):
    for k in candidates:
        if k in state:
            return state[k], k
    raise KeyError(
        f"None of these expected keys were found in checkpoint:\n  "
        + "\n  ".join(candidates)
        + "\nFirst 30 actually-present keys:\n  "
        + "\n  ".join(sorted(state.keys())[:30])
    )


def convert(checkpoint: dict, fla_config_path: Path) -> OrderedDict:
    """Map Primus KDA Megatron state_dict → FLA HF KDA state_dict."""
    state = checkpoint["model"]

    with open(fla_config_path) as f:
        cfg = json.load(f)

    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_heads"]
    num_v_heads = cfg.get("num_v_heads") or num_heads
    head_dim = cfg["head_dim"]
    expand_v = cfg.get("expand_v", 1.0)
    intermediate_size = cfg["intermediate_size"]
    num_hidden_layers = cfg["num_hidden_layers"]

    head_v_dim = int(head_dim * expand_v)
    qk_dim = num_heads * head_dim                # 256 for 300M
    v_dim = num_v_heads * head_v_dim             # 512 for 300M
    fused_in_proj_dim = (
        qk_dim * 2          # q + k
        + v_dim             # v
        + head_v_dim        # f_a (low-rank gate bottleneck)
        + head_v_dim        # g_a (low-rank output-gate bottleneck)
        + num_v_heads       # beta
    )
    print(
        f"[cfg ] hidden={hidden_size} num_heads={num_heads} num_v_heads={num_v_heads}\n"
        f"       head_dim={head_dim} expand_v={expand_v} head_v_dim={head_v_dim}\n"
        f"       qk_dim={qk_dim} v_dim={v_dim} intermediate={intermediate_size}\n"
        f"       layers={num_hidden_layers} fused_in_proj_dim={fused_in_proj_dim}"
    )

    hf = OrderedDict()
    hf["model.embeddings.weight"] = state["embedding.word_embeddings.weight"]

    for fla_idx in range(num_hidden_layers):
        kda_i = 2 * fla_idx
        mlp_i = 2 * fla_idx + 1
        dst = f"model.layers.{fla_idx}"

        # ── attn pre-norm ─────────────────────────────────────────────
        attn_norm_w, _ = _get_first(
            state,
            f"decoder.layers.{kda_i}.mixer.in_proj.layer_norm_weight",  # TE-fused spec
            f"decoder.layers.{kda_i}.norm.weight",                       # no-TE spec
            f"decoder.layers.{kda_i}.input_layernorm.weight",
        )
        hf[f"{dst}.attn_norm.weight"] = attn_norm_w

        # ── fused in_proj split: [q | k | v | f_a | g_a | beta] ───────
        in_proj_w = state[f"decoder.layers.{kda_i}.mixer.in_proj.weight"]
        assert in_proj_w.shape == (fused_in_proj_dim, hidden_size), (
            f"in_proj shape {tuple(in_proj_w.shape)} != "
            f"expected ({fused_in_proj_dim}, {hidden_size}) for layer {kda_i}. "
            "Did you train with the post-fusion KDA code?"
        )
        o = 0
        q_w   = in_proj_w[o:o + qk_dim];      o += qk_dim
        k_w   = in_proj_w[o:o + qk_dim];      o += qk_dim
        v_w   = in_proj_w[o:o + v_dim];       o += v_dim
        f_a_w = in_proj_w[o:o + head_v_dim];  o += head_v_dim
        g_a_w = in_proj_w[o:o + head_v_dim];  o += head_v_dim
        b_w   = in_proj_w[o:o + num_v_heads]; o += num_v_heads
        assert o == fused_in_proj_dim, (o, fused_in_proj_dim)

        hf[f"{dst}.attn.q_proj.weight"]   = q_w
        hf[f"{dst}.attn.k_proj.weight"]   = k_w
        hf[f"{dst}.attn.v_proj.weight"]   = v_w
        hf[f"{dst}.attn.b_proj.weight"]   = b_w
        hf[f"{dst}.attn.f_proj.0.weight"] = f_a_w
        hf[f"{dst}.attn.g_proj.0.weight"] = g_a_w

        # ── low-rank bottleneck expansions (still separate in Primus) ─
        hf[f"{dst}.attn.f_proj.1.weight"] = state[
            f"decoder.layers.{kda_i}.mixer.f_b_proj.weight"
        ]
        hf[f"{dst}.attn.g_proj.1.weight"] = state[
            f"decoder.layers.{kda_i}.mixer.g_b_proj.weight"
        ]
        # g_proj.1 has bias=True (FLA reference: fla/layers/kda.py:189)
        hf[f"{dst}.attn.g_proj.1.bias"] = state[
            f"decoder.layers.{kda_i}.mixer.g_b_proj.bias"
        ]

        # ── fused conv1d split: [q_conv | k_conv | v_conv] ─────────────
        conv_w = state[f"decoder.layers.{kda_i}.mixer.conv1d.weight"]
        # FLA stores each conv as [channels, 1, kernel]
        q_conv = conv_w[:qk_dim]
        k_conv = conv_w[qk_dim:qk_dim * 2]
        v_conv = conv_w[qk_dim * 2:]
        hf[f"{dst}.attn.q_conv1d.weight"] = q_conv
        hf[f"{dst}.attn.k_conv1d.weight"] = k_conv
        hf[f"{dst}.attn.v_conv1d.weight"] = v_conv

        # ── A_log / dt_bias ───────────────────────────────────────────
        # Primus stores A_log as [1, 1, num_v_heads, 1]; FLA wants flat [num_v_heads].
        A_log = state[f"decoder.layers.{kda_i}.mixer.A_log"].reshape(num_v_heads)
        hf[f"{dst}.attn.A_log"]   = A_log
        hf[f"{dst}.attn.dt_bias"] = state[f"decoder.layers.{kda_i}.mixer.dt_bias"]

        # ── output norm + projection ──────────────────────────────────
        hf[f"{dst}.attn.o_norm.weight"] = state[
            f"decoder.layers.{kda_i}.mixer.out_norm.weight"
        ]
        hf[f"{dst}.attn.o_proj.weight"] = state[
            f"decoder.layers.{kda_i}.mixer.out_proj.weight"
        ]

        # ── MLP sublayer ──────────────────────────────────────────────
        mlp_norm_w, _ = _get_first(
            state,
            f"decoder.layers.{mlp_i}.mlp.linear_fc1.layer_norm_weight",  # TE spec
            f"decoder.layers.{mlp_i}.pre_mlp_layernorm.weight",          # no-TE spec
            f"decoder.layers.{mlp_i}.input_layernorm.weight",
        )
        hf[f"{dst}.mlp_norm.weight"] = mlp_norm_w

        # SwiGLU fc1 = cat([gate_proj, up_proj])
        fc1_w = state[f"decoder.layers.{mlp_i}.mlp.linear_fc1.weight"]
        assert fc1_w.shape == (intermediate_size * 2, hidden_size), (
            f"fc1 shape {tuple(fc1_w.shape)} != "
            f"expected ({intermediate_size * 2}, {hidden_size}) for layer {mlp_i}"
        )
        hf[f"{dst}.mlp.gate_proj.weight"] = fc1_w[:intermediate_size]
        hf[f"{dst}.mlp.up_proj.weight"]   = fc1_w[intermediate_size:]
        hf[f"{dst}.mlp.down_proj.weight"] = state[
            f"decoder.layers.{mlp_i}.mlp.linear_fc2.weight"
        ]

    # ── final norm + LM head ─────────────────────────────────────────
    final_norm_w, _ = _get_first(
        state,
        "decoder.final_norm.weight",
        "decoder.final_layernorm.weight",
        "decoder.norm.weight",
    )
    hf["model.norm.weight"] = final_norm_w

    if "output_layer.weight" in state:
        hf["lm_head.weight"] = state["output_layer.weight"]
    else:
        # Llama-style tied embeddings
        hf["lm_head.weight"] = state["embedding.word_embeddings.weight"]

    return hf


def save_hf_dir(hf_state: OrderedDict, output_dir: Path, fla_config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights — prefer safetensors, fall back to pytorch_model.bin.
    try:
        from safetensors.torch import save_file
        # Clone tied weights so they don't share storage (safetensors rejects that).
        if "lm_head.weight" in hf_state and "model.embeddings.weight" in hf_state:
            if hf_state["lm_head.weight"].data_ptr() == hf_state["model.embeddings.weight"].data_ptr():
                hf_state["lm_head.weight"] = hf_state["lm_head.weight"].clone()
        save_file(hf_state, str(output_dir / "model.safetensors"))
        print(f"[save] model.safetensors ({(output_dir / 'model.safetensors').stat().st_size / 1e6:.1f} MB)")
    except ImportError:
        torch.save(hf_state, output_dir / "pytorch_model.bin")
        print(f"[save] pytorch_model.bin (safetensors not installed)")

    # config.json — copy FLA's pretrain config, force HF/eval-friendly toggles.
    with open(fla_config_path) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["KDAForCausalLM"]
    cfg["model_type"] = cfg.get("model_type", "kda")
    cfg["torch_dtype"] = "bfloat16"
    # Disable fused-loss / fused-norm / fused-swiglu auto-detection at eval
    # time — FLA's HF model has these as optional fast paths that aren't
    # needed for evaluation and avoid extra triton compiles.
    cfg["fuse_cross_entropy"] = False
    cfg["fuse_norm"] = False
    cfg["fuse_swiglu"] = False
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] config.json (architectures={cfg['architectures']})")

    # generation_config.json (so lm-eval can use HF .generate cleanly)
    gen_cfg = {
        "_from_model_config": True,
        "transformers_version": None,
        "pad_token_id": cfg.get("pad_token_id"),
        "eos_token_id": cfg.get("eos_token_id"),
        "bos_token_id": cfg.get("bos_token_id"),
    }
    gen_cfg = {k: v for k, v in gen_cfg.items() if v is not None}
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_cfg, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True, type=Path,
                   help="Primus iter dir, e.g. .../checkpoints/iter_0004768")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--config", type=Path, default=None,
                   help="Path to FLA KDA config JSON. Defaults to kda_300M_pure.json "
                        "when '300m' appears in --checkpoint-path, else kda_1B_pure.json.")
    p.add_argument("--tokenizer-src", type=Path, default=None,
                   help="Optional tokenizer dir to copy into --output-dir.")
    args = p.parse_args()

    if args.config is None:
        configs_dir = Path("/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs")
        if "300m" in str(args.checkpoint_path).lower():
            args.config = configs_dir / "kda_300M_pure.json"
        else:
            args.config = configs_dir / "kda_1B_pure.json"
        print(f"[auto] --config defaulted to {args.config}")

    print("=" * 78)
    print(" Primus KDA → FLA HuggingFace converter")
    print("=" * 78)
    print(f"  checkpoint = {args.checkpoint_path}")
    print(f"  output     = {args.output_dir}")
    print(f"  fla config = {args.config}")
    print()

    ckpt = load_megatron_checkpoint(args.checkpoint_path)
    hf_state = convert(ckpt, args.config)
    print(f"[map ] converted {len(hf_state)} tensors")

    save_hf_dir(hf_state, args.output_dir, args.config)

    # Optionally copy tokenizer files
    if args.tokenizer_src is not None:
        import shutil
        copied = 0
        for name in (
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
        ):
            src = args.tokenizer_src / name
            if src.exists():
                shutil.copy(src, args.output_dir / name)
                copied += 1
        print(f"[tok ] copied {copied} tokenizer files from {args.tokenizer_src}")

    print()
    print("Done. To load:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}', trust_remote_code=True)")
    print()
    print("To evaluate with lm-eval:")
    print(f"  lm_eval --model hf \\")
    print(f"    --model_args pretrained={args.output_dir},trust_remote_code=True,dtype=bfloat16 \\")
    print(f"    --tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge \\")
    print(f"    --batch_size 16")


if __name__ == "__main__":
    sys.exit(main())
