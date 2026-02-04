
import torch
import json
import transformers
from transformers import AutoTokenizer

from torch import nn as nn
from pathlib import Path

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from lm_eval.models.huggingface import HFLM

import sys

from modeling_zebra_llama import ZebraLlamaForCausalLM, ZebraLlamaConfig

@register_model("zebra_llama")
class ZebraLlamaEvalWrapper(HFLM):

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        # `lm-eval` passes `--model_args` values as strings, so normalize here.
        device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
        dtype_obj = dtype
        if isinstance(dtype, str):
            key = dtype.strip().lower().replace("torch.", "")
            alias_map = {
                "bf16": "bfloat16",
                "fp16": "float16",
                "fp32": "float32",
                "half": "float16",
                "float": "float32",
            }
            key = alias_map.get(key, key)
            if key == "auto":
                dtype_obj = None
            elif hasattr(torch, key):
                dtype_obj = getattr(torch, key)
            else:
                raise ValueError(f"Unsupported dtype value: {dtype!r}")
        
        primus_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(primus_root))

        pretrained_path = Path(pretrained)
        ckpt_dir = (
            (primus_root / pretrained_path).resolve()
            if not pretrained_path.is_absolute()
            else pretrained_path.resolve()
        )
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        config_path = ckpt_dir / "config.json"
        weights_path = ckpt_dir / "pytorch_model.bin"

        cfg_dict = json.loads(config_path.read_text())
        config = ZebraLlamaConfig(**cfg_dict)
        
        model = ZebraLlamaForCausalLM(config)
        model.eval()
        
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
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        print(model)
        if dtype_obj is None:
            model = model.to(device=device_obj)
        else:
            model = model.to(device=device_obj, dtype=dtype_obj)

        # Initialize harness HF wrapper on top of our already-loaded model.
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            max_length=max_length,
            batch_size=batch_size,
            device=str(device_obj),
            dtype=dtype_obj,
        )

    def _model_call(self, inps, **kwargs):
        """
        lm-eval's `HFLM` assumes `self.model(...)` returns an object with `.logits`.
        Our Zebra model can return a raw tuple if `return_dict` is false, so force it.
        """
        kwargs.setdefault("return_dict", True)
        out = self.model(inps, **kwargs)
        if hasattr(out, "logits"):
            return out.logits
        # Fallback for unexpected tuple returns
        if isinstance(out, tuple) and len(out) > 0:
            return out[0]
        raise TypeError(f"Unexpected model output type: {type(out)}")
  

if __name__ == "__main__":
    cli_evaluate()
    