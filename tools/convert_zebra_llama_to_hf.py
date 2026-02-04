#!/usr/bin/env python3
"""
Convert Megatron Zebra-Llama checkpoint to HuggingFace format.

This script converts a trained Zebra-Llama model (Hybrid Mamba+MLA) from 
Megatron-LM format to a HuggingFace-compatible format for evaluation.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict

# Add Megatron-LM to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_megatron_checkpoint(checkpoint_path):
    """Load Megatron checkpoint from disk."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the model weights
    model_path = Path(checkpoint_path) / "mp_rank_00" / "model_optim_rng.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    print("Loading checkpoint (this may take a moment)...")
    try:
        # Try with weights_only=True first (safer)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        print("✓ Loaded with weights_only=True")
    except Exception as e:
        print(f"weights_only=True failed ({e}), trying full load...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print("✓ Loaded with weights_only=False")
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    return checkpoint


def extract_model_state(checkpoint):
    """Extract model state dict from Megatron checkpoint."""
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        # Try to find the model state in the checkpoint
        for key in checkpoint.keys():
            if 'model' in key.lower():
                model_state = checkpoint[key]
                break
        else:
            model_state = checkpoint
    
    print(f"Model state contains {len(model_state)} parameters")
    
    # Print some example keys
    print("\nExample parameter names:")
    for i, (key, v) in enumerate(list(model_state.items())):
        if v is not None and hasattr(v, 'shape'):
            print(f"  {key}: {v.shape} ({v.dtype})")
        else:
            print(f"  {key}: {type(v)} (non-tensor)")
    
    return model_state


def convert_to_hf_format(model_state, config):
    """Convert Megatron model state to HuggingFace format."""
    hf_state = OrderedDict()
    
    # This is a template - you'll need to customize based on your model architecture
    # The key mapping depends on how your Zebra-Llama model is structured
    
    print("\nConverting to HuggingFace format...")
    
    for key, value in model_state.items():
        # Remove 'module.' prefix if present
        if key.startswith('module.'):
            key = key[7:]
        
        # Convert layer names
        # Example mappings (customize for your architecture):
        # decoder.layers.0.mixer.in_proj.weight -> model.layers.0.mamba.in_proj.weight
        new_key = key
        if key.startswith('decoder.'):
            if key.startswith('decoder.final_norm.'):
                new_key = key.replace('decoder.final_norm.', 'model.norm.')
            else:
                new_key = key.replace('decoder.', 'model.')
        if key.startswith('embedding.word_embeddings.'):
            new_key = key.replace('embedding.word_embeddings.', 'model.embed_tokens.')
        
        if 'linear_kv_up_proj.layer_norm_weight' in new_key:
            new_key = new_key.replace('linear_kv_up_proj.layer_norm_weight', 'kv_layernorm.weight')
        if 'linear_q_up_proj.layer_norm_weight' in new_key:
            new_key = new_key.replace('linear_q_up_proj.layer_norm_weight', 'q_layernorm.weight')
        if 'linear_fc1.layer_norm_weight' in new_key:
            new_key = new_key.replace('linear_fc1.layer_norm_weight', 'pre_mlp_layernorm.weight')
        if 'mixer.in_proj.layer_norm_weight' in new_key:
            new_key = new_key.replace('mixer.in_proj.layer_norm_weight', 'norm.weight')
        if 'mlp.pre_mlp_layernorm' in new_key:
            new_key = new_key.replace('mlp.pre_mlp_layernorm', 'pre_mlp_layernorm')

        # if 'layer_norm_weight' in new_key:
        #     new_key = new_key.replace('layer_norm_weight', 'weight')
        # if 'layer_norm_bias' in new_key:
        #     new_key = new_key.replace('layer_norm_bias', 'bias')
        
        if '_extra_state' not in new_key:
            hf_state[new_key] = value

    # Ensure lm_head.weight exists (Megatron uses output_layer.weight)
    # Prefer explicit output layer if present; otherwise fall back to tying with embeddings.
    if "lm_head.weight" not in hf_state:
        if "output_layer.weight" in model_state:
            hf_state["lm_head.weight"] = model_state["output_layer.weight"]
        elif "model.output_layer.weight" in hf_state:
            hf_state["lm_head.weight"] = hf_state["model.output_layer.weight"]
        elif "model.embed_tokens.weight" in hf_state:
            # Fallback: tie lm_head to embeddings (common when weights are tied)
            hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"]
        elif "embedding.word_embeddings.weight" in model_state:
            hf_state["lm_head.weight"] = model_state["embedding.word_embeddings.weight"]
    
    return hf_state


def save_hf_checkpoint(hf_state, config, output_dir):
    """Save checkpoint in HuggingFace format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_dir / "pytorch_model.bin"
    print(f"\nSaving model weights to: {model_path}")
    torch.save(hf_state, model_path)
    
    # Save config
    config_path = output_dir / "config.json"
    print(f"Saving config to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create a basic model card
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Zebra-Llama {config.get('hidden_size', 'N/A')}M

This is a converted checkpoint from Megatron-LM format.

## Model Details
- Hidden Size: {config.get('hidden_size', 'N/A')}
- Num Layers: {config.get('num_layers', 'N/A')}
- Num Attention Heads: {config.get('num_attention_heads', 'N/A')}
- Vocab Size: {config.get('vocab_size', 'N/A')}

## Architecture
Hybrid Mamba + Multi-Latent Attention (MLA)
""")
    
    print(f"\n✓ Conversion complete! Saved to: {output_dir}")


def create_config_from_checkpoint(checkpoint, args):
    """Create HuggingFace config from Megatron checkpoint metadata."""
    
    # Try to extract config from checkpoint
    
    megatron_args = checkpoint['args']
    config = {
        'architectures': ['ZebraLlamaForCausalLM'],
        'model_type': 'zebra_llama',
        'hidden_size': getattr(megatron_args, 'hidden_size', args.hidden_size),
        'num_layers': getattr(megatron_args, 'num_layers', args.num_layers),
        'num_attention_heads': getattr(megatron_args, 'num_attention_heads', args.num_attention_heads),
        'intermediate_size': getattr(megatron_args, 'ffn_hidden_size', None),
        'vocab_size': getattr(megatron_args, 'padded_vocab_size', args.vocab_size),
        'original_max_position_embeddings': getattr(megatron_args, 'original_max_position_embeddings', 4096),
        'rms_norm_eps': getattr(megatron_args, 'norm_epsilon', 1e-5),
        'hybrid_attention_ratio': getattr(megatron_args, 'hybrid_attention_ratio', 0.25),
        'mamba_state_dim': getattr(megatron_args, 'mamba_state_dim', 64),
        'mamba_head_dim': getattr(megatron_args, 'mamba_head_dim', 64),
        'mamba_num_groups': getattr(megatron_args, 'mamba_num_groups', 8),
        'num_attention_heads': getattr(megatron_args, 'num_attention_heads', 32),
        'q_lora_rank': getattr(megatron_args, 'q_lora_rank', 1344),
        'kv_lora_rank': getattr(megatron_args, 'kv_lora_rank', 128),
        'qk_head_dim': getattr(megatron_args, 'qk_head_dim', 32),
        'qk_pos_emb_head_dim': getattr(megatron_args, 'qk_pos_emb_head_dim', 32),
        'v_head_dim': getattr(megatron_args, 'v_head_dim', 64),
        'rotary_scaling_factor': getattr(megatron_args, 'rotary_scaling_factor', 1.0),
        'mscale': getattr(megatron_args, 'mscale', 1.0),
        'mscale_all_dim': getattr(megatron_args, 'mscale_all_dim', 1.0),
        'rotary_base': getattr(megatron_args, 'rotary_base', 500000),
        'beta_fast': getattr(megatron_args, 'beta_fast', 32.0),
        'beta_slow': getattr(megatron_args, 'beta_slow', 1.0),
        'torch_dtype': 'bfloat16' if getattr(megatron_args, 'bf16', False) else 'float32',
    }
     
    return config


def main():
    parser = argparse.ArgumentParser(description='Convert Megatron Zebra-Llama checkpoint to HuggingFace format')
    
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to Megatron checkpoint directory (e.g., output/zebra_llama_1B-pretrain/iter_0001000)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for HuggingFace checkpoint')
    parser.add_argument('--hidden-size', type=int, default=2048,
                        help='Hidden size of the model')
    parser.add_argument('--num-layers', type=int, default=24,
                        help='Number of transformer layers')
    parser.add_argument('--num-attention-heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--vocab-size', type=int, default=128256,
                        help='Vocabulary size')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Megatron to HuggingFace Checkpoint Conversion")
    print("="*70)
    
    # Step 1: Load Megatron checkpoint
    checkpoint = load_megatron_checkpoint(args.checkpoint_path)
    
    # Step 2: Extract model state
    model_state = extract_model_state(checkpoint)
    
    # Step 3: Create config
    config = create_config_from_checkpoint(checkpoint, args)
    print(f"\nModel config: {json.dumps(config, indent=2)}")
    
    from tools.modeling_zebra_llama import ZebraLlamaConfig, ZebraLlamaForCausalLM
    zebra_config = ZebraLlamaConfig(**config)
    model = ZebraLlamaForCausalLM(zebra_config)
    
    # Step 4: Convert to HuggingFace format
    hf_state = convert_to_hf_format(model_state, zebra_config)
    sd = model.state_dict()

    # Compare keys between converted state and model's expected keys
    hf_keys = set(hf_state.keys())
    model_keys = set(sd.keys())
    missing_in_hf = sorted(model_keys - hf_keys)
    extra_in_hf = sorted(hf_keys - model_keys)

    print("\n" + "=" * 70)
    print("State dict key comparison")
    print("=" * 70)
    print(f"HF state keys:     {len(hf_keys)}")
    print(f"Model state keys:  {len(model_keys)}")
    print(f"Missing in hf_state (expected by model): {len(missing_in_hf)}")
    print(f"Extra in hf_state (not in model):        {len(extra_in_hf)}")

    if missing_in_hf:
        print("\nFirst missing keys:")
        for k in missing_in_hf[:50]:
            shape = tuple(sd[k].shape) if hasattr(sd[k], "shape") else None
            dtype = str(sd[k].dtype) if hasattr(sd[k], "dtype") else None
            print(f"  - {k}  shape={shape} dtype={dtype}")
        if len(missing_in_hf) > 50:
            print(f"  ... and {len(missing_in_hf) - 50} more")

    if extra_in_hf:
        print("\nFirst extra keys:")
        for k in extra_in_hf[:50]:
            shape = tuple(hf_state[k].shape) if hasattr(hf_state[k], "shape") else None
            dtype = str(hf_state[k].dtype) if hasattr(hf_state[k], "dtype") else None
            print(f"  - {k}  shape={shape} dtype={dtype}")
        if len(extra_in_hf) > 50:
            print(f"  ... and {len(extra_in_hf) - 50} more")

    # Shape check for intersection
    common = sorted(hf_keys & model_keys)
    shape_mismatches = []
    for k in common:
        v_hf = hf_state[k]
        v_md = sd[k]
        if hasattr(v_hf, "shape") and hasattr(v_md, "shape") and tuple(v_hf.shape) != tuple(v_md.shape):
            shape_mismatches.append((k, tuple(v_hf.shape), tuple(v_md.shape)))

    print(f"\nShape mismatches on common keys: {len(shape_mismatches)}")
    for k, sh_hf, sh_md in shape_mismatches[:50]:
        print(f"  - {k}: hf_state{sh_hf} vs model{sh_md}")
    if len(shape_mismatches) > 50:
        print(f"  ... and {len(shape_mismatches) - 50} more")
    
    model.to(torch.bfloat16)
    # Use strict=False so we can see all missing/extra keys without crashing,
    # but we expect lm_head.weight to be present now.
    missing, unexpected = model.load_state_dict(hf_state, strict=False)

    if missing:
        print("\nMissing keys when loading into model (strict=False):")
        for k in missing[:50]:
            print(f"  - {k}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    if unexpected:
        print("\nUnexpected keys when loading into model (strict=False):")
        for k in unexpected[:50]:
            print(f"  - {k}")
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")
    
    # import pdb; pdb.set_trace()
    # Step 5: Save HuggingFace checkpoint
    save_hf_checkpoint(hf_state, config, args.output_dir)
    
    print("\n" + "="*70)
    print("Next steps:")
    print("="*70)
    print("1. Review the converted checkpoint")
    print("2. Create a custom modeling file for Zebra-Llama if needed")
    print("3. Run evaluation with lm-eval-harness")
    print("="*70)


if __name__ == '__main__':
    main()
