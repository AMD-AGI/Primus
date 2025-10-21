# Supported Models

Primus supports a growing list of state-of-the-art foundation models for pretraining, fine-tuning, and evaluation, with seamless backend switching and flexible configuration.

---

## 📚 Model Support Matrix

| Model Family          | Variants/Sizes            | Backend(s)        | Pretrain | Fine-tune | Eval | SFT | RLHF | FlashAttention | Notes               |
|---------------------- |--------------------------|-------------------|----------|-----------|------|-----|------|---------------|---------------------|
| **Llama-3.1**         | 8B, 70B, 405B            | Megatron, Titan   | ✔️       | ✔️        | ✔️   | ✔️  | WIP  | ✔️            | Full precision & FP8|
| **DeepSeek-V3**       | 16B, 67B, 236B           | Megatron, Titan   | ✔️       | ✔️        | ✔️   | ✔️  | WIP  | ✔️            | Long context        |
| **Mixtral**           | 8×22B, 8×7B              | Megatron          | ✔️       | ✔️        | ✔️   | ✔️  | WIP  | Partial        | MoE, dynamic routing|
| **InternLM2**         | 7B, 20B                  | Megatron          | ✔️       | ✔️        | ✔️   | ✔️  |      | Partial        |                     |
| **Baichuan**          | 7B, 13B, 53B             | Megatron          | ✔️       | ✔️        | ✔️   | ✔️  |      | Partial        |                     |
| **Yi**                | 6B, 34B                  | Megatron          | ✔️       | ✔️        | ✔️   | ✔️  |      | Partial        |                     |
| **Qwen2**             | 7B, 72B                  | Megatron          | ✔️       | ✔️        | ✔️   | ✔️  |      | Partial        |                     |
| **StableLM**          | 3B, 7B                   | Titan             | ✔️       | ✔️        | ✔️   |     |      |                |                     |
| **GPT-3**             | 125M, 355M, 1.3B, 2.7B   | Megatron          | ✔️       | ✔️        | ✔️   |     |      |                |                     |

> _Note: "WIP" = Work In Progress, "Partial" = Partial support (e.g. only inference)_

---

## 🔄 Backend Support

- **Megatron**: Large-scale model training, 3D parallelism, full model family support.
- **TorchTitan**: Next-gen backend, out-of-box FP8/FlashAttention, and PyTorch 2.0 graph mode (Eager/Inductor).
- **Future**: JAX and custom backends in roadmap.

---

## 🎯 Roadmap

- Full support for MoE models (Mixtral, DeepSeek-MoE)
- Expanded multi-lingual and instruction-tuned models
- Evaluation/Inference for very long context (>128K tokens)
- Integration with Hugging Face transformers for easy export/import

---

## 📝 Adding New Models

Primus supports flexible model registration via YAML/TOML configs. To add a new model:

1. Create a config under `primus/configs/models/`
2. Set `model.name`, `model.flavor` and tokenizer parameters
3. Specify backend (`megatron` or `torchtitan`)
4. (Optional) Tune training/eval parameters for your hardware

See [Experiment Configuration](./config/overview.md) for details and templates.

---

## 📢 Contribution

Want to add your favorite model? [Open a pull request](https://github.com/amd/primus/pulls) or [file an issue](https://github.com/amd/primus/issues)!

---

_Last updated: 2025-09-17_
