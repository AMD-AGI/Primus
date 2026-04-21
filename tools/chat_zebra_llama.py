#!/usr/bin/env python3
"""
Simple interactive chat for Zebra-Llama (HF-converted checkpoint).

Loads:
  - model code from: tools/modeling_zebra_llama.py
  - weights from:    output/zebra_llama_1B_hf_iter_0150000

Notes:
  - KV cache is disabled in the model implementation, so generation is slower.
  - Chat formatting here is intentionally simple; adjust the prompt template as needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer


def build_prompt(history: List[Tuple[str, str]]) -> str:
    """
    Very simple chat prompt:
      User: ...
      Assistant: ...
    """
    parts: List[str] = []
    for role, text in history:
        if role == "user":
            parts.append(f"User: {text}")
        else:
            parts.append(f"Assistant: {text}")
    parts.append("Assistant:")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Zebra-Llama (converted HF checkpoint)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/zebra_llama_1B_hf_iter_0200000",
        help="Path to converted HF checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer name/path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    args = parser.parse_args()

    primus_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(primus_root))

    from tools.modeling_zebra_llama import ZebraLlamaConfig, ZebraLlamaForCausalLM  # noqa: E402

    ckpt_dir = (primus_root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing pytorch_model.bin: {weights_path}")

    torch.manual_seed(args.seed)

    # Device / dtype
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[Info] Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Info] Loading config: {config_path}")
    cfg_dict = json.loads(config_path.read_text())
    config = ZebraLlamaConfig(**cfg_dict)

    print(f"[Info] Building model on {device} ({dtype})")
    model = ZebraLlamaForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    print(f"[Info] Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys ({len(missing)}). Showing first 20:")
        for k in missing[:20]:
            print(f"  - {k}")
    if unexpected:
        print(f"[Warn] Unexpected keys ({len(unexpected)}). Showing first 20:")
        for k in unexpected[:20]:
            print(f"  - {k}")

    history: List[Tuple[str, str]] = []

    print("\n=== Zebra-Llama chat ===")
    print("Type your message and press enter.")
    print("Commands: /reset, /exit\n")

    # while True:
    #     try:
    #         user = input("User> ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         print("\n[Exit]")
    #         return

    #     if not user:
    #         continue
    #     if user == "/exit":
    #         return
    #     if user == "/reset":
    #         history.clear()
    #         print("[Info] History cleared.\n")
            # continue

    # history.append(("user", user))
    # prompt = build_prompt(history)
    
    prompt = "Once upon a time, "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=1.0,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # Decode only the newly generated portion
    gen_ids = out_ids[0, input_ids.shape[1] :]
    assistant = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    #history.append(("assistant", assistant))
    print(f"OUTPUT> {prompt}{assistant}\n")


if __name__ == "__main__":
    main()

