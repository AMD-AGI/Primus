#!/usr/bin/env python3
"""
Convert Primus (Megatron) Pure GDN checkpoint to FLA HuggingFace format.

Primus uses fused projections (in_proj, conv1d, mlp.linear_fc1) and alternating
GDN/MLP sublayers. FLA uses separate projections and combined layers.

Usage:
    python tools/convert_gdn_to_fla_hf.py \
        --checkpoint-path output/amd/root/zebra_llama_1B_gdn_pure-pretrain/checkpoints/iter_0076294 \
        --output-dir output/gdn_pure_1B_fla_hf \
        --config /path/to/gated_deltanet_1B_pure.json
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Ensure Megatron is importable (needed for torch.load to unpickle checkpoint)
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
    """Return state[k] for the first k present in candidates, else raise with a helpful message."""
    for k in candidates:
        if k in state:
            return state[k], k
    raise KeyError(
        f"None of these expected keys were found in checkpoint:\n  "
        + "\n  ".join(candidates)
        + "\nFirst 30 actually-present keys:\n  "
        + "\n  ".join(sorted(state.keys())[:30])
    )


def convert(checkpoint, fla_config_path):
    """Convert Megatron state dict to FLA GatedDeltaNet HuggingFace format.

    Handles both layouts:
      * TE spec     (gdn_hybrid_stack_spec):
          norm folded into linear → `mixer.in_proj.layer_norm_weight`,
                                    `mlp.linear_fc1.layer_norm_weight`
      * no-TE spec  (gdn_hybrid_stack_spec_no_te):
          separate WrappedTorchNorm → `norm.weight`, `pre_mlp_layernorm.weight`
    """
    state = checkpoint['model']

    with open(fla_config_path) as f:
        fla_cfg = json.load(f)

    hidden_size = fla_cfg['hidden_size']
    num_heads = fla_cfg['num_heads']
    num_v_heads = fla_cfg.get('num_v_heads', num_heads)
    head_dim = fla_cfg['head_dim']
    expand_v = fla_cfg.get('expand_v', 1.0)
    intermediate_size = fla_cfg.get('intermediate_size', hidden_size * 4)
    num_hidden_layers = fla_cfg['num_hidden_layers']

    head_k_dim = head_dim
    head_v_dim = int(head_dim * expand_v)
    key_dim = num_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    print(f"\nModel config:")
    print(f"  hidden_size={hidden_size}, num_heads={num_heads}, num_v_heads={num_v_heads}")
    print(f"  head_dim={head_dim}, expand_v={expand_v}")
    print(f"  key_dim={key_dim}, value_dim={value_dim}")
    print(f"  intermediate_size={intermediate_size}, num_hidden_layers={num_hidden_layers}")
    print(f"  in_proj_dim = {key_dim*2 + value_dim*2 + num_v_heads*2}")

    hf_state = OrderedDict()

    # Embeddings (FLA uses 'model.embeddings.weight', not 'model.embed_tokens.weight')
    hf_state['model.embeddings.weight'] = state['embedding.word_embeddings.weight']

    for fla_layer_idx in range(num_hidden_layers):
        gdn_idx = fla_layer_idx * 2
        mlp_idx = fla_layer_idx * 2 + 1
        prefix = f'model.layers.{fla_layer_idx}'

        # ── GDN sublayer ──
        attn_norm_w, _ = _get_first(
            state,
            f'decoder.layers.{gdn_idx}.mixer.in_proj.layer_norm_weight',
            f'decoder.layers.{gdn_idx}.norm.weight',
            f'decoder.layers.{gdn_idx}.input_layernorm.weight',
        )
        hf_state[f'{prefix}.attn_norm.weight'] = attn_norm_w

        # Fused in_proj split: [q(key_dim), k(key_dim), v(value_dim), gate(value_dim), beta(num_v_heads), alpha(num_v_heads)]
        in_proj_w = state[f'decoder.layers.{gdn_idx}.mixer.in_proj.weight']
        assert in_proj_w.shape[0] == key_dim * 2 + value_dim * 2 + num_v_heads * 2, \
            f"in_proj shape mismatch: {in_proj_w.shape}"

        q_w = in_proj_w[:key_dim]
        k_w = in_proj_w[key_dim:key_dim*2]
        v_w = in_proj_w[key_dim*2:key_dim*2+value_dim]
        g_w = in_proj_w[key_dim*2+value_dim:key_dim*2+value_dim*2]
        b_w = in_proj_w[key_dim*2+value_dim*2:key_dim*2+value_dim*2+num_v_heads]
        a_w = in_proj_w[key_dim*2+value_dim*2+num_v_heads:]

        hf_state[f'{prefix}.attn.q_proj.weight'] = q_w
        hf_state[f'{prefix}.attn.k_proj.weight'] = k_w
        hf_state[f'{prefix}.attn.v_proj.weight'] = v_w
        hf_state[f'{prefix}.attn.g_proj.weight'] = g_w
        hf_state[f'{prefix}.attn.b_proj.weight'] = b_w
        hf_state[f'{prefix}.attn.a_proj.weight'] = a_w

        # A_log and dt_bias
        hf_state[f'{prefix}.attn.A_log'] = state[f'decoder.layers.{gdn_idx}.mixer.A_log']
        hf_state[f'{prefix}.attn.dt_bias'] = state[f'decoder.layers.{gdn_idx}.mixer.dt_bias']

        # Fused conv1d split: [q_conv(key_dim, 1, 4), k_conv(key_dim, 1, 4), v_conv(value_dim, 1, 4)]
        conv_key = f'decoder.layers.{gdn_idx}.mixer.conv1d.weight'
        if conv_key in state:
            conv_w = state[conv_key]  # (key_dim*2 + value_dim, 1, kernel_size)
            q_conv = conv_w[:key_dim]
            k_conv = conv_w[key_dim:key_dim*2]
            v_conv = conv_w[key_dim*2:]
            hf_state[f'{prefix}.attn.q_conv1d.weight'] = q_conv
            hf_state[f'{prefix}.attn.k_conv1d.weight'] = k_conv
            hf_state[f'{prefix}.attn.v_conv1d.weight'] = v_conv

        # Output norm (per-head RMSNorm)
        hf_state[f'{prefix}.attn.o_norm.weight'] = state[f'decoder.layers.{gdn_idx}.mixer.out_norm.weight']

        # Output projection
        hf_state[f'{prefix}.attn.o_proj.weight'] = state[f'decoder.layers.{gdn_idx}.mixer.out_proj.weight']

        # ── MLP sublayer ──
        mlp_norm_w, _ = _get_first(
            state,
            f'decoder.layers.{mlp_idx}.mlp.linear_fc1.layer_norm_weight',
            f'decoder.layers.{mlp_idx}.pre_mlp_layernorm.weight',
            f'decoder.layers.{mlp_idx}.input_layernorm.weight',
        )
        hf_state[f'{prefix}.mlp_norm.weight'] = mlp_norm_w

        # Fused SwiGLU fc1 split: [gate_proj(intermediate), up_proj(intermediate)]
        fc1_w = state[f'decoder.layers.{mlp_idx}.mlp.linear_fc1.weight']
        assert fc1_w.shape[0] == intermediate_size * 2, \
            f"fc1 shape mismatch: {fc1_w.shape}, expected ({intermediate_size*2}, {hidden_size})"
        gate_proj = fc1_w[:intermediate_size]
        up_proj = fc1_w[intermediate_size:]
        hf_state[f'{prefix}.mlp.gate_proj.weight'] = gate_proj
        hf_state[f'{prefix}.mlp.up_proj.weight'] = up_proj

        # Down projection
        hf_state[f'{prefix}.mlp.down_proj.weight'] = state[f'decoder.layers.{mlp_idx}.mlp.linear_fc2.weight']

    final_norm_w, _ = _get_first(
        state,
        'decoder.final_norm.weight',
        'decoder.final_layernorm.weight',
        'decoder.norm.weight',
    )
    hf_state['model.norm.weight'] = final_norm_w

    # LM head (tied with embeddings)
    if 'output_layer.weight' in state:
        hf_state['lm_head.weight'] = state['output_layer.weight']
    else:
        hf_state['lm_head.weight'] = state['embedding.word_embeddings.weight']

    return hf_state


def main():
    parser = argparse.ArgumentParser(description='Convert Primus GDN to FLA HuggingFace format')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to FLA config JSON (gated_deltanet_1B_pure.json)')
    args = parser.parse_args()

    # Default config path — auto-detect model size from checkpoint
    if args.config is None:
        fla_configs_dir = Path("/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs")
        alt_dir = Path(__file__).parent.parent / "third_party" / "flash-linear-attention" / "legacy" / "training" / "configs"
        configs_dir = fla_configs_dir if fla_configs_dir.exists() else alt_dir

        # Detect from checkpoint path name
        ckpt_str = str(args.checkpoint_path).lower()
        if "300m" in ckpt_str:
            args.config = str(configs_dir / "gated_deltanet_300M_pure.json")
        else:
            args.config = str(configs_dir / "gated_deltanet_1B_pure.json")

    print("=" * 70)
    print("Primus GDN → FLA HuggingFace Conversion")
    print("=" * 70)

    checkpoint = load_megatron_checkpoint(args.checkpoint_path)
    hf_state = convert(checkpoint, args.config)

    print(f"\nConverted {len(hf_state)} parameters")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")
    torch.save(hf_state, output_dir / "model.safetensors.bin")

    # Also save as safetensors if available
    try:
        from safetensors.torch import save_file
        # Clone tied weights to avoid shared memory error
        if 'lm_head.weight' in hf_state and 'model.embeddings.weight' in hf_state:
            if hf_state['lm_head.weight'].data_ptr() == hf_state['model.embeddings.weight'].data_ptr():
                hf_state['lm_head.weight'] = hf_state['lm_head.weight'].clone()
        save_file(hf_state, str(output_dir / "model.safetensors"))
        (output_dir / "model.safetensors.bin").unlink()
        print("  Saved as safetensors format")
    except ImportError:
        os.rename(output_dir / "model.safetensors.bin", output_dir / "pytorch_model.bin")
        print("  Saved as pytorch_model.bin (safetensors not available)")

    # Save config.json (FLA format)
    with open(args.config) as f:
        config = json.load(f)
    config['architectures'] = ['GatedDeltaNetForCausalLM']
    config['fuse_cross_entropy'] = False
    config['fuse_norm'] = False
    config['fuse_swiglu'] = False

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    # Tokenizer config (use Llama-3.2-1B tokenizer)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 2048,
    }
    with open(output_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\n{'='*70}")
    print("Conversion complete!")
    print(f"{'='*70}")
    print(f"\nTo load:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
    print(f"\nTo evaluate with lm-eval:")
    print(f"  lm_eval --model hf \\")
    print(f"    --model_args pretrained={output_dir},trust_remote_code=True \\")
    print(f"    --tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge \\")
    print(f"    --batch_size 16")


if __name__ == '__main__':
    main()
