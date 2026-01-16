## JAX MaxText Patch Notes & Extended Arguments

Primus integrates JAX MaxText as a backend for running LLaMA and related transformer models on AMD GPUs.
At the moment, Primus does not apply any additional patch-layer arguments on top of MaxText – the
MaxText configuration surface (YAML + CLI) is used as-is.

Use this page together with:

- [`docs/backends/overview.md`](../overview.md#supported-models) for a high-level model overview
- [`docs/cli/PRIMUS-CLI-GUIDE.md`](../../cli/PRIMUS-CLI-GUIDE.md) for Primus CLI usage patterns
- The official MaxText documentation for the full set of MaxText-specific arguments

### 1. MaxText-Specific Notes

- Primus currently wires MaxText via `primus/configs/models/maxtext` and `primus/configs/modules/maxtext`.
- Model families currently exercised in examples include:
  - LLaMA2 7B/70B
  - LLaMA3 8B/70B
  - LLaMA3.3 70B
  - DeepSeek-V2 16B
  - Mixtral-8x7B
  - Grok1
- There are no extra Primus-only flags for MaxText yet; as we add MaxText-specific patches
  (e.g., ROCm optimizations, logging helpers), they will be documented in tables here in the
  same style as the Megatron-LM and TorchTitan patch notes.
