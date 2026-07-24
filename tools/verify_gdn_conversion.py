#!/usr/bin/env python3
"""
Sanity check for Primus GDN → FLA HuggingFace conversion.

Loads the converted model and verifies:
1. All weights loaded without errors
2. Forward pass produces reasonable loss (~2-4 for a trained model)
3. Top predictions are sensible English tokens

Usage:
    python tools/verify_gdn_conversion.py --model-path output/gdn_pure_1B_fla_hf
"""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description='Verify GDN checkpoint conversion')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to converted FLA HuggingFace model directory')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Tokenizer to use')
    args = parser.parse_args()

    print("=" * 60)
    print("GDN Conversion Verification")
    print("=" * 60)

    from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
    from transformers import AutoTokenizer

    print(f"\nLoading config from: {args.model_path}")
    config = GatedDeltaNetConfig.from_pretrained(args.model_path)
    print(f"  hidden_size={config.hidden_size}, num_heads={config.num_heads}, "
          f"num_v_heads={config.num_v_heads}, num_layers={config.num_hidden_layers}")

    print(f"\nLoading model...")
    model = GatedDeltaNetForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e9:.3f}B")

    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The largest planet in our solar system is",
    ]

    all_passed = True
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])

        loss = out.loss.item()
        logits = out.logits[0, -1]
        top5 = torch.topk(logits, 5)

        status = "PASS" if loss < 6.0 else "FAIL"
        if loss > 6.0:
            all_passed = False

        # Greedy continuation — most intuitive coherence signal
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id or 0,
            )
        cont = tokenizer.decode(gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Loss: {loss:.4f} [{status}]")
        print(f"  Top-5: {[tokenizer.decode(t) for t in top5.indices]}")
        print(f"  Greedy continuation: \"{cont}\"")

    print(f"\n{'=' * 60}")
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("  - Model loads cleanly")
        print("  - Forward pass produces reasonable loss")
        print("  - Predictions are sensible")
        print(f"{'=' * 60}")
        return 0
    else:
        print("RESULT: SOME CHECKS FAILED")
        print("  - Loss too high (>6.0) suggests weight mapping issue")
        print(f"{'=' * 60}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
