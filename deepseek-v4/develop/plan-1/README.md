# DeepSeek-V4 Plan-1 (Phase 8+)

This directory contains the Phase 8+ replan for DeepSeek-V4 integration in Primus.
The goal is to close architecture and performance gaps discovered after the first
bring-up sequence (Phase 1-7).

## Review Baseline

- Branch baseline commit: `0a25e20030b38cb46f840f0971c8ed97746eae4c`.
- Reviewed implementation commits on top:
  - `d3383c02` (P1 configs)
  - `8ae10000` (P2 dispatch)
  - `a5d2a561` (P3 scaffolding)
  - `3b7ad8c8` (P4 HC + hybrid attention)
  - `5e4008dc` (P5 MoE + MTP)
  - `97b9720d` (P6-P7 distributed bring-up)

## Confirmed Findings From Review

### 1) `deepseek_v4_layer_specs.py` and Megatron spec logic

The current implementation in
`primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py`
is a placeholder delegation layer to GPT helpers and does not define a V4-native
`ModuleSpec` tree for runtime decoder construction.

Runtime behavior is currently:

1. `GPTModel.__init__` builds a stock decoder using placeholder spec.
2. `DeepseekV4Model` replaces `self.decoder` with standalone
   `DeepseekV4TransformerBlock`.

This means the real V4 decoder path does not primarily rely on
`ModuleSpec -> build_module` composition for V4 internals.

### 2) TE reuse (`TESpecProvider`) in runtime V4 path

Megatron TE wrappers and providers exist as expected:

- `third_party/Megatron-LM/megatron/core/extensions/transformer_engine.py`
- `third_party/Megatron-LM/megatron/core/extensions/transformer_engine_spec_provider.py`
- `third_party/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py`

However, V4 runtime modules are still mostly pure PyTorch custom paths
(`_RMSNorm`, `nn.Linear`, custom MoE loop and all-reduce in `v4_moe.py`) and do
not systematically reuse TE-backed submodules (`TENorm`, TE parallel linears,
TE grouped GEMM experts) through spec/provider composition.

So the concern about missing TE reuse in the effective runtime path is valid.

## Replan Scope (Phase 8+)

The new plan starts from Phase 8 and focuses on:

1. Re-centering V4 runtime around `ModuleSpec`-driven composition.
2. Reusing TE-backed modules where applicable for performance.
3. Converging MoE parallel/dispatcher integration to Megatron patterns.
4. Completing regression, convergence, and release readiness gates.

## Documents

- `00-roadmap.md`: phase overview, milestones, dependencies.
- `01-phase-details.md`: task-level plan for Phase 8-11.
- `02-test-strategy.md`: test matrix, pass/fail gates, reporting templates.
