#!/usr/bin/env python3
"""
Initialize a Megatron (Primus) checkpoint from FLA's GatedDeltaNet model weights.

Creates a fake Megatron checkpoint directory that Primus can load via --load,
ensuring both frameworks start from identical weights for loss-curve comparison.

Usage:
    python tools/convert_fla_gdn_init_to_megatron.py \
        --fla-config /path/to/gated_deltanet_300M_pure.json \
        --output-dir output/fla_init_ckpt_300M \
        --seed 42 \
        --no-te   # use no-TE key names (WrappedTorchNorm, ColumnParallelLinear)

This script was reconstructed verbatim from the agent transcript dated 2026-05-13
(see GDN_FLA_PARITY.md §"Files in the repo for this work").  It is one of the
"forensics scripts" kept untracked in tools/ per the parity-doc footnote.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict

_primus_root = Path(__file__).resolve().parents[1]
_fla_root = _primus_root.parent / "flash-linear-attention"
if str(_fla_root) not in sys.path:
    sys.path.insert(0, str(_fla_root))


def init_fla_model(config_path, seed=42):
    """Initialize an FLA GatedDeltaNet model and return its state_dict."""
    torch.manual_seed(seed)

    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    with open(config_path) as f:
        cfg_dict = json.load(f)

    config = GatedDeltaNetConfig(**cfg_dict)
    # Disable fusions for the init pass — we just want the raw weights, not the
    # fused-kernel-specific buffer initialisation.  Primus will turn fusions on
    # at training time via PRIMUS_FLA_* env vars regardless of these.
    config.fuse_cross_entropy = False
    config.fuse_norm = False
    config.fuse_swiglu = False

    model = GatedDeltaNetForCausalLM(config)
    model.eval()

    print(f"FLA model initialized: {sum(p.numel() for p in model.parameters()):,} params")
    return model.state_dict(), config


def convert_fla_to_megatron(fla_state, fla_cfg, use_te=True):
    """Convert FLA HF state dict to Megatron naming convention.

    Each FLA layer becomes TWO Megatron decoder layers: the mixer sub-layer
    (GDN) at index 2k and the MLP sub-layer at index 2k+1.  This mirrors
    HybridStack's split layout (see primus/backends/megatron/core/models/hybrid).
    """
    hidden_size = fla_cfg.hidden_size
    num_heads = fla_cfg.num_heads
    num_v_heads = getattr(fla_cfg, 'num_v_heads', num_heads)
    head_dim = fla_cfg.head_dim
    expand_v = getattr(fla_cfg, 'expand_v', 1.0)
    intermediate_size = fla_cfg.intermediate_size
    num_hidden_layers = fla_cfg.num_hidden_layers

    head_k_dim = head_dim
    head_v_dim = int(head_dim * expand_v)
    key_dim = num_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    print(f"\nConverting FLA -> Megatron (use_te={use_te}):")
    print(f"  hidden_size={hidden_size}, num_heads={num_heads}, num_v_heads={num_v_heads}")
    print(f"  head_dim={head_dim}, expand_v={expand_v}")
    print(f"  key_dim={key_dim}, value_dim={value_dim}")
    print(f"  intermediate_size={intermediate_size}, num_hidden_layers={num_hidden_layers}")

    mg_state = OrderedDict()

    mg_state['embedding.word_embeddings.weight'] = fla_state['model.embeddings.weight'].clone()

    for fla_idx in range(num_hidden_layers):
        gdn_idx = fla_idx * 2
        mlp_idx = fla_idx * 2 + 1
        fp = f'model.layers.{fla_idx}'

        # ── GDN sub-layer ──
        if use_te:
            mg_state[f'decoder.layers.{gdn_idx}.mixer.in_proj.layer_norm_weight'] = \
                fla_state[f'{fp}.attn_norm.weight'].clone()
        else:
            mg_state[f'decoder.layers.{gdn_idx}.norm.weight'] = \
                fla_state[f'{fp}.attn_norm.weight'].clone()

        # Fuse separate projections into in_proj: [q, k, v, gate, beta, alpha]
        q_w = fla_state[f'{fp}.attn.q_proj.weight']
        k_w = fla_state[f'{fp}.attn.k_proj.weight']
        v_w = fla_state[f'{fp}.attn.v_proj.weight']
        g_w = fla_state[f'{fp}.attn.g_proj.weight']
        b_w = fla_state[f'{fp}.attn.b_proj.weight']
        a_w = fla_state[f'{fp}.attn.a_proj.weight']
        in_proj_w = torch.cat([q_w, k_w, v_w, g_w, b_w, a_w], dim=0)
        mg_state[f'decoder.layers.{gdn_idx}.mixer.in_proj.weight'] = in_proj_w

        mg_state[f'decoder.layers.{gdn_idx}.mixer.A_log'] = \
            fla_state[f'{fp}.attn.A_log'].clone()
        mg_state[f'decoder.layers.{gdn_idx}.mixer.dt_bias'] = \
            fla_state[f'{fp}.attn.dt_bias'].clone()

        # Fuse conv1d: [q_conv, k_conv, v_conv]
        q_conv = fla_state[f'{fp}.attn.q_conv1d.weight']
        k_conv = fla_state[f'{fp}.attn.k_conv1d.weight']
        v_conv = fla_state[f'{fp}.attn.v_conv1d.weight']
        conv_w = torch.cat([q_conv, k_conv, v_conv], dim=0)
        mg_state[f'decoder.layers.{gdn_idx}.mixer.conv1d.weight'] = conv_w

        mg_state[f'decoder.layers.{gdn_idx}.mixer.out_norm.weight'] = \
            fla_state[f'{fp}.attn.o_norm.weight'].clone()
        mg_state[f'decoder.layers.{gdn_idx}.mixer.out_proj.weight'] = \
            fla_state[f'{fp}.attn.o_proj.weight'].clone()

        # ── MLP sub-layer ──
        if use_te:
            mg_state[f'decoder.layers.{mlp_idx}.mlp.linear_fc1.layer_norm_weight'] = \
                fla_state[f'{fp}.mlp_norm.weight'].clone()
        else:
            mg_state[f'decoder.layers.{mlp_idx}.pre_mlp_layernorm.weight'] = \
                fla_state[f'{fp}.mlp_norm.weight'].clone()

        gate_w = fla_state[f'{fp}.mlp.gate_proj.weight']
        up_w = fla_state[f'{fp}.mlp.up_proj.weight']
        fc1_w = torch.cat([gate_w, up_w], dim=0)
        mg_state[f'decoder.layers.{mlp_idx}.mlp.linear_fc1.weight'] = fc1_w

        mg_state[f'decoder.layers.{mlp_idx}.mlp.linear_fc2.weight'] = \
            fla_state[f'{fp}.mlp.down_proj.weight'].clone()

    mg_state['decoder.final_norm.weight'] = fla_state['model.norm.weight'].clone()

    print(f"  Converted {len(mg_state)} parameter tensors")
    return mg_state


def save_megatron_checkpoint(mg_state, output_dir, iteration=0):
    """Save as a Megatron checkpoint directory structure."""
    ckpt_dir = Path(output_dir) / f"iter_{iteration:07d}" / "mp_rank_00"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model': mg_state,
        'checkpoint_version': 3.0,
    }

    ckpt_path = ckpt_dir / "model_optim_rng.pt"
    torch.save(checkpoint, ckpt_path)

    iter_file = Path(output_dir) / "latest_checkpointed_iteration.txt"
    iter_file.write_text(str(iteration))

    print(f"\nSaved Megatron checkpoint to: {ckpt_dir}")
    print(f"  {ckpt_path}: {ckpt_path.stat().st_size / 1e6:.1f} MB")
    return str(Path(output_dir))


def verify_conversion(fla_state, mg_state, fla_cfg, use_te):
    """Verify total parameter count matches."""
    fla_params = sum(v.numel() for v in fla_state.values())
    mg_params = sum(v.numel() for v in mg_state.values())

    # FLA has both model.embeddings.weight and lm_head.weight (tied).
    # Megatron only has embedding.word_embeddings.weight (output_layer shares).
    fla_unique = fla_params - fla_state['model.embeddings.weight'].numel()

    print(f"\nVerification:")
    print(f"  FLA params (unique): {fla_unique:,}")
    print(f"  Megatron params:     {mg_params:,}")

    if fla_unique != mg_params:
        print(f"  WARNING: param count mismatch! Diff = {abs(fla_unique - mg_params):,}")
    else:
        print(f"  OK -- param counts match")

    emb_match = torch.equal(
        fla_state['model.embeddings.weight'],
        mg_state['embedding.word_embeddings.weight']
    )
    norm_key = 'decoder.layers.0.norm.weight' if not use_te else \
               'decoder.layers.0.mixer.in_proj.layer_norm_weight'
    norm_match = torch.equal(
        fla_state['model.layers.0.attn_norm.weight'],
        mg_state[norm_key]
    )
    print(f"  Embedding match: {emb_match}")
    print(f"  Layer 0 norm match: {norm_match}")


def main():
    parser = argparse.ArgumentParser(
        description='Initialize Primus checkpoint from FLA weights')
    parser.add_argument('--fla-config', type=str, required=True,
                        help='Path to FLA config JSON')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Megatron checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for FLA model initialization')
    parser.add_argument('--no-te', action='store_true',
                        help='Use no-TE key names (WrappedTorchNorm, ColumnParallelLinear)')
    args = parser.parse_args()

    print("=" * 70)
    print("FLA -> Primus (Megatron) Weight Initialization")
    print("=" * 70)

    fla_state, fla_cfg = init_fla_model(args.fla_config, args.seed)

    use_te = not args.no_te
    mg_state = convert_fla_to_megatron(fla_state, fla_cfg, use_te=use_te)

    verify_conversion(fla_state, mg_state, fla_cfg, use_te)

    ckpt_path = save_megatron_checkpoint(mg_state, args.output_dir)

    print(f"\n{'='*70}")
    print("Done! To use in Primus, add to your training config:")
    print(f"  load: {ckpt_path}")
    print(f"  finetune: true    # load weights only, fresh optimizer")
    print(f"  no_load_optim: true")
    print(f"  no_load_rng: true")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
