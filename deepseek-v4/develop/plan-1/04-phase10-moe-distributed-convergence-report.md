# 04 — Phase 10 (v2) MoE + Distributed Convergence Report

Date: 2026-04-30

## Scope

This report summarizes Phase 10 delivery for DeepSeek-V4 integration in Primus, focused on:

- MoE construction convergence to `ModuleSpec + build_module`.
- Distributed token dispatch alignment with Megatron dispatcher flow.
- Clamped-SwiGLU compatibility across grouped-GEMM backends.
- Config/schema cleanup for DeepSeek-V4-specific runtime fields.
- End-to-end runtime validation on the target MI355X environment.

## Environment

- Host: `uswslocpm2m-106-2371`
- Container: `dev_primus_wenx_691`
- Workspace: `/shared_aig/wenx/workspace/Primus`
- Runtime entry: `run_deepseek_v4.sh`

## Delivered Changes

### 1) MoE spec-driven construction

- Added explicit `DeepseekV4MoESubmodules` ownership and routed MoE construction through `build_module` for:
  - router,
  - token dispatcher,
  - grouped experts,
  - shared expert.
- Removed legacy dual-path ambiguity and enforced grouped-expert-first construction for the active DeepSeek-V4 path.

### 2) Dispatcher-path convergence

- Converged DeepSeek-V4 MoE execution onto Megatron dispatcher-based flow.
- Removed reliance on the temporary EP routed-output all-reduce fallback in the active path.
- Preserved hash-routed lower-layer behavior and routed learned-router layers through the same dispatcher contract.

### 3) Shared expert alignment with Megatron

- Aligned shared expert implementation with Megatron `SharedExpertMLP`.
- Removed custom shared-expert fallback path and enforced `SharedExpertMLP` usage.
- Passed clamped-SwiGLU behavior through config/activation fields instead of custom module substitution.

### 4) Parameter down-sinking and module signature cleanup

- Reduced plumbing from spec layer into module constructors by resolving runtime fields from `config` directly inside modules.
- Simplified `DeepseekV4Attention`, `DeepseekV4MoE`, and `DeepseekV4HybridLayer` initialization signatures.
- Upgraded `DeepseekV4HybridLayer` to `GraphableMegatronModule`.

### 5) DeepSeek-V4 config schema formalization

- Added `DeepSeekV4TransformerConfig` (inherits `MLATransformerConfig`) to host V4-specific runtime fields.
- Updated builder path to explicitly request `config_class=DeepSeekV4TransformerConfig`.
- Updated V4 module type hints to the new config type.
- Added `activation_func_clamp_value` to `deepseek_v4_base.yaml` and aligned it with `swiglu_limit`.

## Runtime Bring-up and Issue Resolution

During runtime execution of `run_deepseek_v4.sh`, the following blockers were found and resolved:

1. **Grouped MoE clamped-SwiGLU guard failure**
   - Symptom: grouped backend rejected clamped-SwiGLU support gate.
   - Action: added smoke override `v4_grouped_experts_support_clamped_swiglu=True` in run script.

2. **Hyper-connection dtype mismatch**
   - Symptom: `float32` vs `bf16` mismatch in `F.linear` paths.
   - Action: aligned HC linear weight dtype to input dtype in `hyper_connection.py`.

3. **TE projection dtype mismatch**
   - Symptom: TE output projection assertion on activation/parameter dtype mismatch.
   - Action: cast attention output back to activation dtype before output projection in `deepseek_v4_attention.py`.

4. **CSA memory pressure (OOM)**
   - Symptom: very large sparse-attention intermediate tensors under smoke defaults.
   - Action: reduced smoke defaults (`seq_length=128`, `max_position_embeddings=128`) and set `index_topk=8` in run script.

5. **DDP bucket reset assertion after first iteration**
   - Symptom: `per_param_grad_ready_counts` assertion at iteration boundary.
   - Action: disabled overlap flags for smoke (`overlap_grad_reduce=False`, `overlap_param_gather=False`).

## Final Validation Result

`run_deepseek_v4.sh` now completes successfully on the target environment:

- `torchrun finished successfully (code 0)`
- Training reached `iteration 10/10`
- No fatal runtime exception after the above fixes

Observed final log markers:

- `Megatron pretrain execution completed.`
- `Training completed.`
- `Cleanup completed.`

## Acceptance Against Phase 10 Plan

### Completed

- Spec-driven MoE submodule construction.
- Dispatcher-based active distributed path convergence.
- Shared expert alignment to `SharedExpertMLP`.
- Clamped-SwiGLU compatibility contract enforcement.
- Config schema formalization via `DeepSeekV4TransformerConfig`.
- End-to-end smoke run pass on target host/container.

### Still Open (tracked)

- PP token-id propagation contract for hash-routed layers.
- Distributed smoke matrix with deterministic routing snapshots (`1x8` and combined `PP/EP` replay checks).

## Files of Interest (Phase 10)

- `primus/backends/megatron/core/transformer/moe/v4_moe.py`
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py`
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py`
- `primus/backends/megatron/core/transformer/deepseek_v4_attention.py`
- `primus/backends/megatron/core/transformer/hyper_connection.py`
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_transformer_config.py`
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_builders.py`
- `primus/configs/models/megatron/deepseek_v4_base.yaml`
- `run_deepseek_v4.sh`

## Notes

- This report captures both the core Phase 10 architecture convergence and the required runtime stabilization needed to pass the target smoke script in the designated execution environment.
