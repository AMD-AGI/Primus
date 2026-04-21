
import sys
import os
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.utils import make_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_zebra_llama import ZebraLlamaForCausalLM, ZebraLlamaConfig

AutoConfig.register("zebra_llama", ZebraLlamaConfig)
AutoModelForCausalLM.register(ZebraLlamaConfig, ZebraLlamaForCausalLM)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tasks", type=str,
                        default="arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_path", type=str, default="eval_results")
    parser.add_argument("--limit", type=float, default=None)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model_path
    dtype_str = args.dtype if args.dtype != "auto" else "auto"
    model_args = f"pretrained={args.model_path},tokenizer={tokenizer_path},trust_remote_code=True,dtype={dtype_str}"

    print("=" * 72)
    print("Zebra-Llama lm-eval")
    print("=" * 72)
    print(f"  Model path:  {args.model_path}")
    print(f"  Tokenizer:   {tokenizer_path}")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Num fewshot: {args.num_fewshot}")
    print(f"  Device:      {args.device}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Output:      {args.output_path}")
    print("=" * 72)

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot if args.num_fewshot > 0 else None,
        device=args.device,
        log_samples=True,
        limit=args.limit,
    )

    print(make_table(results))

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, "results.json")
    with open(out_file, "w") as f:
        json.dump(results["results"], f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
