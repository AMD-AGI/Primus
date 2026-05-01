# 00 — DeepSeek-V4 Integration Roadmap

> This document defines phase breakdown, milestones, and dependencies.
> Detailed tasks, file-level landing list, and test strategy live in
> `01-code-layout.md` / `02-phase-details.md` / `03-testing-strategy.md`.

## Guiding Principles

| Principle | What it means |
|---|---|
| **Follow existing conventions** | Use Primus's `model_type` + builder + LayerSpec + patches pattern (mirror the mamba extension path). |
| **Never edit `third_party/`** | All V4-specific changes live under `primus/`. |
| **MVP first** | Start with the simplest reference-faithful implementation that runs end-to-end; do perf / quantization later. |
| **Align with the reference** | `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py` is the source of truth. NeMo Automodel is a secondary reference (it fixed several HF-PR bugs). |
| **Each phase is independently reviewable** | Aim for self-contained, mergeable phases. |

## Phase Overview

| # | Phase | Type | Key Deliverables | Exit Criteria |
|---|---|---|---|---|
| **1** | Configs & yaml | scaffolding | `deepseek_v4_base.yaml`, `deepseek_v4_flash.yaml`, `MI355X/deepseek_v4_flash-BF16-pretrain.yaml` | yaml passes the Primus loader; every field maps to a valid Megatron arg |
| **2** | Register `deepseek_v4` model_type | scaffolding | `deepseek_v4_builders.py` + new dispatch branch in `MegatronPretrainTrainer` | trainer reaches the builder (even if internally still a stub GPT) |
| **3** | Layer spec + decoder block scaffolding | scaffolding | `primus/backends/megatron/core/models/deepseek_v4/{deepseek_v4_layer_specs.py, deepseek_v4_block.py, deepseek_v4_model.py}` | 1L scaffold (attention=MLA spec / FFN=MoE spec) forwards without NaN |
| **4** | HC + Hybrid Attention core | **core** | New modules under `core/transformer/`: hyper_connection / compressor / indexer / csa_attention / hca_attention; patch swap-in | Forward output matches the V4-Flash reference attention to within 1e-3 |
| **5** | MoE family + activation + RoPE + MTP | **core** | hash router, sqrtsoftplus scoring, clamped SwiGLU, dual-RoPE / partial-RoPE, MTP head | Full V4-Flash forward (43L + 1MTP) runs, logits shapes correct, token-0 logits close to reference |
| **6** | Trainer end-to-end + TP/PP/EP | integration | full dispatch, `pretrain_deepseek_v4.py` (peer of `pretrain_gpt.py`), PP layout, HC across PP | Single-node 8-GPU BF16 produces a steadily-decreasing loss curve, no hangs |
| **7** | Muon Optimizer integration | integration | reuse existing muon patches; extend the param-group split for V4 (HC scale & MTP head go to AdamW) | With muon enabled, grad-norm is stable and loss slope ≥ AdamW baseline |
| **8** | Convergence validation + FP8 + quantization | integration (deferable) | short-run convergence comparison, FP8 path, (optional) FP4 | Loss on a 0.5e9 token run agrees with reference experiments to within ±0.05 |

> **MVP scope**: phases 1–6 are what makes "V4-Flash trainable in Primus".
> Phases 7–8 follow the MVP but P4 / P5 must keep the right hooks open (e.g.
> the router class exposes a `score_fn` abstraction, attention exposes a
> `compress_ratio` config) so that they can be plugged in directly later.

## Inter-Phase Dependency Graph

```
P1 (configs)
  └── P2 (register model_type)
        └── P3 (layer specs / block / model scaffolding)
              ├── P4 (HC + Hybrid Attn)         ┐
              └── P5 (MoE / Activation / RoPE / ├── P6 (trainer + TP/PP/EP)
                       MTP)                      │      ├── P7 (Muon)
                                                 │      └── P8 (convergence + FP8/FP4)
```

P4 and P5 can run **in parallel** (different sub-teams), but **P6 only starts
once both P4 and P5 reach a working forward pass**.

## Milestones (suggested external communication cadence)

| Milestone | Rough scope | Phases |
|---|---|---|
| **M0: Scaffolding done** | `deepseek_v4` selectable, scaffold forward does not crash | P1+P2+P3 |
| **M1: HC + Hybrid Attn module-level pass** | 4 streams + CSA / HCA agree with reference on a 1L toy model | P4 |
| **M2: Full V4-Flash forward aligned** | 43L + 1MTP forward within 1e-2 of reference | P4+P5 |
| **M3: Distributed training works** | Single 8-GPU / 4-node 32-GPU BF16 runs with falling loss | P6 |
| **M4: Muon + longer-run convergence** | 50 / 200 / 1000-step convergence curve vs AdamW | P7+P8 |
| **M5: FP8 / FP4 path** | Throughput improvement with FP8 enabled | P8 |

## Top-Level Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **HC vs Megatron PP partitioning** | The 4 HC streams must flow across layers; if PP splits without carrying all 4 streams together it breaks the math | In P3, lift hidden shape to `[B, S, hc_mult, D]` in the spec; PP send/recv serializes all 4 streams together. |
| **Compressor / Indexer have no fused Megatron op** | v1 has no fused kernel, perf is poor | Implement in pure PyTorch for correctness; fused kernels deferred to P8 (turbo / triton). |
| **Hash routing vs EP** | Hash routing is a static token-id → expert mapping; EP partitioning must remain consistent | When designing `HashGate` in P5, query EP rank info directly and dispatch tokens not owned by this rank to the correct EP rank. |
| **MTP head shares weights with main lm_head** | Shared embedding sits at the last PP stage; MTP runs another forward on top | Reuse Megatron's existing mtp hooks (V3 already exercises `mtp_num_layers`). |
| **YaRN only applies to compressed layers** | Cannot just swap base; must respect per-layer compress_ratio | In P5, make RoPE layer-aware. |
| **Clamped SwiGLU vs Megatron's fused activation** | Fused path defaults to swiglu without clamp | In P5, provide an unfused fallback first; register a new activation kind. |

## Time Estimate (for scheduling only)

> Person-day estimate, single FT engineer; with parallel work this compresses.

| Phase | Estimate (person-days) | Notes |
|---|---|---|
| P1 | 0.5 | yaml only |
| P2 | 0.5 | builder + dispatch + import smoke test |
| P3 | 2 | LayerSpec / Block / Model scaffolding |
| P4 | 5 | HC + Compressor + Indexer + CSA + HCA + Attn sink |
| P5 | 5 | Hash MoE + sqrtsoftplus + clamped SwiGLU + dual-RoPE + MTP |
| P6 | 3 | Trainer dispatch + PP layout + 1-node / multi-node smoke |
| P7 | 1 | Muon configuration |
| P8 | 4 | Numerical alignment + short runs + FP8 + FP4 |
| **Total** | **~21 person-days** | with parallelism → **7–12 calendar days** |
