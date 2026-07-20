# Primus Weekly Engineering Report — 2026-W21

## 1. Time Window

- Start: Monday 2026-05-18 00:00:00 Asia/Shanghai (GMT+8)
- End: Friday 2026-05-22 09:01 Asia/Shanghai (GMT+8) (report generation time)
- Branch observed: `origin/main`

## 2. Executive Summary

- **1 PR merged to `main`** in the weekly window (Mon 2026-05-18 00:00 GMT+8 → Fri 2026-05-22 09:01 GMT+8).
- Category breakdown: **Bug Fix: 1**; Performance Optimization, Turbo/Dependency Version Update, CI/Infra, Refactor, Docs, Other: 0.
- **No backend version/pin change in the W21 window.** `third_party/Megatron-LM` is still pinned at `d3528a21`, `third_party/torchtitan` at `5fb7cc2e`, and `.github/workflows/ci.yaml` still pins `PRIMUS_TURBO_COMMIT=06b8d3fefd91be26d6adfb5cd43c7524ef87b825`, `PRIMUS_TURBO_AITER_COMMIT=b5e03ed191fca11ee423226537ef8d9435e432a6`, and `TRITON_COMMIT=88b227e23f0445f3f695bad05bbf1a363b4f50e0` on `origin/main`. The current Primus-Turbo pin landed in PR #723 on 2026-05-15 (W20) and was not changed in this window.
- **No weekly report was generated for 2026-W20.** The W19 report covered 2026-05-04 → 2026-05-08 GMT+8; the next entry is this W21 report. PRs merged between Mon 2026-05-11 00:00 GMT+8 and Sun 2026-05-17 23:59 GMT+8 (notably #694's follow-up Turbo bump in #723, #722, #594, #644, #709, #662, #715, #439, #725 and #594) are visible in `git log origin/main` but are not summarized in this report's PR table because they fall outside the W21 window. The W19 report already covered the W19 portion of those changes and the W21 dashboard "Highlights since W19" sections below note the W20-window Turbo pin advance and the unchanged submodule SHAs.
- **PR #721 (Bug Fix, Slurm launcher).** `runner/primus-cli-slurm-entry.sh` now resolves the first hostname from `SLURM_NODELIST` via `scontrol show hostnames`, uses that as `MASTER_ADDR` when unset, and **fails fast (exit 2)** with explicit error logs when a user-provided `MASTER_ADDR` does not match the first SLURM host. Prevents distributed Slurm jobs from hanging due to mismatched Slurm node-rank mapping.
- **Megatron-LM upstream drift: `plan sync` (unchanged).** Pin is `d3528a21` (2026-03-06); upstream `main` HEAD is `e27607a8` (2026-05-22) — **554 commits ahead** (+96 since the W19 snapshot `7cdf652c`). Notable new upstream activity since W19: full removal of legacy code (#4759), upstream Mamba builder migration (#4550), TE 2.15.0 dependency bump (#4682), Hybrid `recompute` support (#4496), `M-FSDP` MXFP8 `fine_grained_param_gather` toggle (#4181), MTP recompute crash fix on packed sequences (#4593), Layer-wise distributed-optimizer + DDP buffer integration (#4509), Megatron-FSDP A2A overlap (#3797), MoE GroupedExperts memory-estimation fix (#4687), GEMM+SwiGLU fused MLP roll-up (#4636), and the `Add high-priority A2A stream and HybridEP preprocessing SMs` change (#4694).
- **torchtitan upstream drift: `urgent sync` (unchanged).** Pin is `5fb7cc2e` (2025-10-15); upstream `main` HEAD is `b5852826` (2026-05-21) — **667 commits ahead** (+57 since the W19 snapshot `4ebb0895`). Notable new upstream activity since W19: DeepEP import fix for latest DeepEP (#3414), MoE split of DeepEP/HybridEP dispatchers (#3389), MoE refactor for clean DTensor boundaries between shared/routed experts (#3386), GraphTrainer rework of `full_inductor_compilation_pass` via `regional_inductor` + CPU-attr migration (#3346), `[graph_trainer] AutoParallel AOT FX Trace Integration` (#2725), `[GraphTrainer] Add Context Parallel support` (#3305), `Remove MoE expert for-loop fallback` (#3308), `Make ChunkedCELoss support torch.autograd.grad` (#3249), `Full DTensor` config-based sharding for Llama3 (#3159), and `[LoRA] Add LoRA converter and rename quantization to converters` (#3264).
- **Primus-Turbo monthly drift: pin is now `06b8d3fe`, +39 commits ahead of month-start CI pin.** Both CI and benchmark `PRIMUS_TURBO_COMMIT` are now `06b8d3fefd91be26d6adfb5cd43c7524ef87b825` (Primus-Turbo #278, *Add HYBRID FP8 format support for Triton backend in gemm and grouped_gemm*, 2026-05-13). CI/benchmark pin is **+39 commits ahead** of the month-start CI pin `333b68d7` (2026-03-27) and **+44 commits ahead** of the month-start benchmark pin `a4488f6c` (2026-03-19). Since the W19 snapshot pin `ef5b58ea`, Primus-Turbo advanced **7 commits** (W20 PR #723), adding HYBRID FP8 format for Triton gemm/grouped_gemm (#278), removing the attn workaround (#313), correcting `hipblaslt_gemm` workspace size on MI300X (#333), adding `flash_attn_varlen_func` with the aiter backend (#332), binding per-stream `hipBLASLt` handles to fix the BF16 multi-stream stall on gfx942 (#319), adding `SECURITY.md` (#331), and fixing a DeepEP combine hang (#327). `PRIMUS_TURBO_AITER_COMMIT` advanced from `857f4d15` to `b5e03ed191` in PR #723 to pick up the *Fix dsink bf16 noise in Triton MHA one-kernel backward* fix (AITER #3070). Recommendation: **monitor**.
- **No backend-gap report regenerated this week.** No submodule SHA changed (`third_party/Megatron-LM`, `third_party/torchtitan`, `third_party/Megatron-Bridge`, `third_party/Emerging-Optimizers`, `third_party/HummingbirdXT`, `third_party/maxtext` are all unchanged on `origin/main`) and no Primus-Turbo / AITER / Triton / ROCSHMEM / UCCL pin changed in the W21 window. Primus-Turbo is also not currently surfaced as a separate backend under `docs/backend-gap/`, so the existing torchtitan baseline report (`5fb7cc2e` vs upstream `main`) is unaffected.

## 3. Weekly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#721](https://github.com/AMD-AGI/Primus/pull/721) `Fix Slurm MASTER_ADDR validation` (author: `WangLingxun`) | 2026-05-18 16:41 | Bug Fix | Hardens `runner/primus-cli-slurm-entry.sh`. Resolves `SLURM_MASTER_ADDR` from `scontrol show hostnames "$SLURM_NODELIST"` (first entry) up front and exits 2 with a clear log if that fails. When the caller does not export `MASTER_ADDR`, the script now uses `SLURM_MASTER_ADDR` (replacing the prior `head -n1 \|\| echo localhost` fallback that could silently default to `localhost`). When the caller does export `MASTER_ADDR`, the script compares it against `SLURM_MASTER_ADDR` and aborts with explicit `MASTER_ADDR=… expected=…` error logs on mismatch, preventing distributed Slurm jobs from hanging due to mismatched Slurm node-rank mapping. Single-file, +13/-5 change. |

## 4. Megatron-LM Drift Overview

- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned in Primus `main` (`third_party/Megatron-LM`): `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` — *fix(moe): fix TE general_gemm API change (#3582)*, 2026-03-06
- Upstream `main` HEAD: `e27607a89b4e36ef084590eb10567031f4798a21` — *Update copy-pr-bot.yaml [skip ci]* (2026-05-22)
- Last upstream functional change in this window: `547fb1713` — *Move bert and t5 pretrain files (#4820)* (2026-05-21)
- Commit gap: **upstream is 554 commits ahead of Primus pin** (+96 since the W19 snapshot `7cdf652c`).
- Submodule SHA on Primus side: unchanged in W21 (also unchanged in W20); last submodule SHA bump on `main` was `3bec9aa9` → `d3528a21` inside PR #654 (merged 2026-04-10).
- Recommendation: **plan sync** (unchanged from W17/W18/W19). Upstream continues to land large structural cleanups (`fully remove legacy code` #4759, legacy GPT/transformer/modules already removed in W18/W19), a TE 2.15.0 dependency bump, Megatron-FSDP A2A overlap, and additional inference/MoE work. The eventual sync diff continues to grow, but the overall sync risk profile is unchanged this week.

### Notable upstream areas that have moved since the pin

- **Legacy code removal (continued)**: `fully remove legacy code` (#4759); on top of W19's legacy `transformer`/`modules` removal (#4207) and legacy GPT removal (#4322). These large structural deletes continue to dominate the eventual Primus sync diff.
- **Dependency / build**: Update `transformer-engine` dependency to **2.15.0** (#4682); upgrade `mamba-ssm` to `2.3.2.post1` and `causal-conv1d` to `1.6.2.post1` (#4712); widen `flashinfer-python` pin to `<0.7.0` (#4700); bump `nvidia-modelopt>=0.44.0` (#4803).
- **Inference / CUDA graphs**: Refactor CUDA graph API: decompose `cuda_graph_scope` into full-iteration impl, inference scope, and per-layer capture modules (#4292); inference prefill engine step optimization for Nemotron (#4764); fix recompute checkpointing + training CGs (#3919); size `DynamicInferenceContext` KV `layer_map` for non-uniform PP (#4775); on top of W19's input/position-ID view cache (#4634) and FlashInfer sampling (#2456).
- **MoE / EP**: Add high-priority A2A stream and HybridEP preprocessing SMs (#4694); thread custom process groups through MoE grad finalization (#4782); MoE GroupedExperts memory estimation respects EP (#4687); HybridEP IB Python-side guardrail (#4719); on top of W19's vLLM grouped-gemm MoE backend (#4566) and partial-cudagraphs + HybridEP DDP-hook fix (#4500).
- **FSDP / DDP / optimizer**: `[Main] Support A2A Overlap for Megatron-FSDP` (#3797); `M-FSDP: Make fine_grained_param_gather configurable for MXFP8` (#4181); `Integrate LayerWiseDistributedOptimizer with DDP buffer infrastructure` (#4509); `Allow optimizer CG to share the same pool as full-iter CG` (#4698); `Route non-Muon params through DistributedOptimizer` (#4771); `Combine GEMM + SwiGLU fused MLP PRs (3890, 4071, 4095, 4219, 4311, 4324) → main` (#4636).
- **Hybrid / Mamba / SSM**: Support recomputing in `HybridModel` (#4496); `[training migration] Migrate mamba builder` (#4550); on top of W19's checkpoint conversion between `GPT_model` and `Hybrid_model` (#4482) and the `[ssm]` optional-extra split (#4517).
- **Reliability / nightlies**: Fix MTP recompute crash with packed sequences (#4593); NCCL UB fix: reduce memory and correctly deregister NCCL mem pool (#4492); fix `no_shard` training convergence + unittest (#3754); Add periodic GPU sniff tests to detect hardware stragglers (#4662); on top of W19's `nvidia_resiliency_ext` fault-injection (#4370) and gradient-corruption fix for layerwise param AG overlap (#4609).
- **Tokenizers / data**: `Fix tokenizers bug in nightly` (#4833); `Tokenizers updates` (#4780); fix tokenizers in respect to newer transformers (#4608); on top of W19's *convert tokenizer args to config* (#4406) and *Finalize all builders in preprocess_data* (#4573).

### Megatron-LM upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| Legacy code removal | `fully remove legacy code`;<br>(carries) legacy `transformer`/`modules` removal;<br>(carries) legacy GPT removal | NVIDIA/Megatron-LM #4759, #4207, #4322 | The remaining legacy paths are now fully gone; any Primus patches under `primus/backends/megatron/**` still referencing legacy classes must be audited before the next pin bump. |
| Dependency / build | TE **2.15.0** bump;<br>`mamba-ssm` 2.3.2.post1 + `causal-conv1d` 1.6.2.post1;<br>widen `flashinfer-python` pin to `<0.7.0`;<br>bump `nvidia-modelopt>=0.44.0` | NVIDIA/Megatron-LM #4682, #4712, #4700, #4803 | Primus Megatron docker stack will need the matching TE version on the next sync; align this with the existing Turbo `TRITON_COMMIT` source build path. |
| Inference / CUDA graphs | Refactor CUDA graph API (decompose `cuda_graph_scope`);<br>Nemotron prefill engine step opt;<br>recompute-CG fix;<br>`DynamicInferenceContext` non-uniform PP fix;<br>(carries) input/position-ID view cache, FlashInfer sampling | NVIDIA/Megatron-LM #4292, #4764, #3919, #4775, #4634, #2456 | Inference/post-train paths for Primus DSV3/MoE configs benefit; expand validation when the sync lands. |
| MoE / EP | High-priority A2A stream + HybridEP preprocessing SMs;<br>thread custom PGs through MoE grad finalization;<br>MoE GroupedExperts memory estimation respects EP;<br>HybridEP IB guardrail | NVIDIA/Megatron-LM #4694, #4782, #4687, #4719 | Reinforces the already-stale MoE sync surface; once Primus syncs, the EP-aware memory estimate and high-priority A2A stream become relevant for Primus DSV3/Qwen3 MoE runs. |
| FSDP / DDP / optimizer | Megatron-FSDP A2A overlap;<br>`M-FSDP` MXFP8 `fine_grained_param_gather` toggle;<br>LayerWise-DistOpt + DDP-buffer integration;<br>optimizer CG pool sharing;<br>non-Muon params → DistributedOptimizer;<br>GEMM+SwiGLU fused MLP roll-up | NVIDIA/Megatron-LM #3797, #4181, #4509, #4698, #4771, #4636 | Significant FSDP/DDP/optimizer reshape; Primus Megatron backend trainer wrappers (`primus/backends/megatron/**`) will need re-validation when the pin moves. |
| Hybrid / Mamba / SSM | `Support recomputing in HybridModel`;<br>`Migrate mamba builder`;<br>(carries) `[ssm]` optional extra | NVIDIA/Megatron-LM #4496, #4550, #4517 | The Hybrid/SSM stack continues to evolve; Primus Mamba/Hybrid configs (LFM/LFM2 added in W18 #651) must be re-tested against the upstream builder migration at sync time. |
| Reliability / nightlies | MTP recompute crash fix (packed seq);<br>NCCL UB memory fix;<br>`no_shard` convergence fix;<br>periodic GPU sniff tests | NVIDIA/Megatron-LM #4593, #4492, #3754, #4662 | Useful for Primus pretrain-at-scale on MI300X/MI355X clusters; this week's Primus #721 Slurm MASTER_ADDR validation aligns with the same "fail fast at launch" theme. |
| Tokenizers / data | Tokenizers fixes + updates;<br>fix tokenizers for newer transformers;<br>(carries) tokenizer-args→config | NVIDIA/Megatron-LM #4833, #4780, #4608, #4406 | Builds on the Primus-side tokenizer-alignment work in W19 #675; the tokenizer-args→config refactor remains the next candidate Primus must adopt at sync time. |

## 5. torchtitan Drift Overview

- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned in Primus `main` (`third_party/torchtitan`): `5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021` — *Deepseek-V3 toml file minor fix (#1894)*, 2025-10-15
- Upstream `main` HEAD: `b5852826b98278c705954ca95ae2cfcb7e94789e` — *Fix imports for latest DeepEP (#3414)* (2026-05-21)
- Commit gap: **upstream is 667 commits ahead of Primus pin** (+57 since the W19 snapshot `4ebb0895`).
- Submodule SHA on Primus side: unchanged in W21 (also unchanged in W20). The most recent Primus-side torchtitan-adjacent change was W20 #723 (*primus-turbo attention add sbhd format support*), which extends the Primus outer adapter under `primus/backends/torchtitan/models/` for the new Primus-Turbo sbhd attention layout — it does **not** bump the upstream pin.
- Recommendation: **urgent sync** (unchanged from W17/W18/W19). The pin is now ~7 months stale; another wave of GraphTrainer / MoE / DTensor landings this week further widens the gap.

### Notable upstream areas that have moved since the pin

- **MoE refactor / DeepEP / HybridEP**: `Fix imports for latest DeepEP` (#3414); `[MoE][6/n] Extract local_reorder, split DeepEP/HybridEP dispatchers` (#3389); `[MoE][5/n] Refactor MoE to clean DTensor boundaries for shared/routed experts` (#3386); `Remove MoE expert for-loop fallback` (#3308); on top of W19's HybridEP-with-GraphTrainer integration tests (#3184), `AllToAllTokenDispatcher` sp_size padding (#3193), MXFP8GroupedExpertsConverter swap fix (#3199).
- **GraphTrainer / precompile**: `[graph_trainer] Fix fsdp_passes compat with BitsetAncestors from pytorch passes` (#3416); `[graph_trainer] Rework full_inductor_compilation_pass via regional_inductor + CPU attr migration` (#3346); `[graph_trainer] Gate overlap_fsdp_ag_rs_pass behind a config flag` (#3241); `[graph_trainer] AutoParallel AOT FX Trace Integration` (#2725); `[GraphTrainer] Add Context Parallel support` (#3305); `[graph_trainer] Refactor selective activation remat to in-place` (#3270); `[graph_trainer] Refactor passes.py into focused modules` (#3319); `[GraphTrainer] Add compile.disable_passes to selectively skip graph passes` (#3273); `[graph_trainer] Skip dense numerics tests due to upstream DTensor regression` (#3372); H100 CI failure fix from DeepEP compilation break (#3390); on top of W19's memory-policy registry + extra-passes hook (#3215), `[graph_trainer] FSDP AG RS overlap` (#3156), and CPU-offloading prefetch pass (#3166).
- **Quantization / LoRA / Float8**: `[Quantization] Use class factory for MXFP8/Float8 GroupedExperts to fix _owner` (#3251); `[LoRA] Add LoRA converter and rename quantization to converters` (#3264); `Merge lora and float8 tests` (#3310); on top of W19's float8 GroupedExperts quantization (#3233).
- **DTensor / sharding**: `[Full DTensor] Config-based Full DTensor for Llama3` (#3159); `[graph_trainer] Skip dense numerics tests due to upstream DTensor regression` (#3372); `Back out "[graph_trainer] Fix SAC remat to produce fresh FakeTensor storage for recomputed nodes (#3343)" (#3358)`.
- **ChunkedCELoss / loss / training loop**: `Make ChunkedCELoss support torch.autograd.grad` (#3249); `Remove unnecessary clone from ChunkedCELoss` (#3272); `Avoid repeat_interleave output-size sync` (#3274) and its revert (#3335); `Fix optimizer state and module state coupling` (#3356); `[graph_trainer] Pass global_valid_tokens into loss_fn instead of dividing externally` (#3244).
- **RL / observability / CI**: `[rl] use wandb=False in CI` (#3408); `[rl] fix CI timeout issue by properly tear down for vllm engine V2` (#3365); `[rl] switch to vllm v2 engine` (#3330); `RL: observability spans across trainer/generator/controller` (#3234); `[RL] - Enable experiment metrics` (#3237); `Enhance Lychee Link Checker (Resiliency & Performance)` (#3203); on top of W19's `[CI] Run integration tests in parallel` (#3144) and structured logging (#3176).

### torchtitan upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| MoE refactor / DeepEP / HybridEP | DeepEP import fix;<br>split DeepEP / HybridEP dispatchers;<br>clean DTensor boundaries for shared/routed experts;<br>remove MoE for-loop fallback | pytorch/torchtitan #3414, #3389, #3386, #3308 | Continues the MoE consolidation Primus is gated on; Primus's torchtitan-side MoE/DSV3/Qwen3 paths under `primus/backends/torchtitan/**` will need re-validation against the new shared/routed split when the pin moves. |
| GraphTrainer / precompile | `fsdp_passes` compat fix with BitsetAncestors;<br>`full_inductor_compilation_pass` rework;<br>AutoParallel AOT FX integration;<br>Context Parallel support;<br>passes.py refactor;<br>`compile.disable_passes` selective gating | pytorch/torchtitan #3416, #3346, #2725, #3305, #3319, #3273 | Major continuing GraphTrainer infrastructure remains unavailable behind the stale pin. CP support inside GraphTrainer is a new capability surface Primus will want once it syncs. |
| Quantization / LoRA / Float8 | Class-factory MXFP8/Float8 GroupedExperts owner fix;<br>LoRA converter + quantization→converters rename;<br>merged lora+float8 tests | pytorch/torchtitan #3251, #3264, #3310 | LoRA + GroupedExperts converters land together; useful for Primus MoE + LoRA experiments after sync. |
| ChunkedCELoss / loss / training loop | `ChunkedCELoss` `torch.autograd.grad` support;<br>remove unnecessary clone;<br>`Avoid repeat_interleave` output-size sync (and revert);<br>optimizer/module state coupling fix | pytorch/torchtitan #3249, #3272, #3274, #3335, #3356 | Targeted loss-path correctness/perf work; relevant when Primus syncs torchtitan to revalidate Primus-side loss instrumentation. |
| RL / observability / CI | `[rl] vllm v2 engine` switch;<br>vllm-V2 CI teardown fix;<br>RL observability spans;<br>RL experiment metrics;<br>Lychee link checker upgrade | pytorch/torchtitan #3330, #3365, #3234, #3237, #3203 | Reference patterns for Primus RL CI; the vLLM v2 engine switch is the most consequential RL change since the W19 snapshot. |

## 6. Primus-Turbo Monthly Drift Overview

- Drift type: **in-repo**, not upstream — compares Turbo version/SHA referenced on Primus `main` now vs the latest commit at or before `month_start_ts = 2026-05-01 00:00 Asia/Shanghai` (`2026-04-30 16:00 UTC`).
- Turbo is **not a submodule** in Primus. Canonical version source:
  - `.github/workflows/ci.yaml` → `PRIMUS_TURBO_COMMIT`, `PRIMUS_TURBO_AITER_COMMIT`, `TRITON_COMMIT` (also wired through `.github/workflows/docker/Dockerfile`)
  - `.github/workflows/benchmark.yaml` → `PRIMUS_TURBO_COMMIT`
- Reference Primus commit at month start (May) on `main`: `0a25e20030b38cb46f840f0971c8ed97746eae4c` (*fix: keep legacy groupedgemm on megatron backend (#693)*, 2026-04-28 15:27 GMT+8). The Turbo pins at that commit are byte-identical to the values reported in W17/W18.
- Current state on `origin/main` (W21):
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `06b8d3fefd91be26d6adfb5cd43c7524ef87b825` — *Add HYBRID FP8 format support for Triton backend in gemm and grouped_gemm (#278)*, 2026-05-13
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `b5e03ed191fca11ee423226537ef8d9435e432a6` — *Fix dsink bf16 noise in Triton MHA one-kernel backward (AITER #3070)*
  - `ci.yaml` `TRITON_COMMIT`: `88b227e23f0445f3f695bad05bbf1a363b4f50e0` (unchanged since W19)
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `06b8d3fefd91be26d6adfb5cd43c7524ef87b825`
- Month-start (2026-05-01) state on `main`:
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `333b68d7c81b722b21b4aad10cd250c45f15027c` — *fix sm_scale none bug (#263)*, 2026-03-27
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `e83f9903c07001a0ec29e85d223f6e6cdbe00859`
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `a4488f6cdb15cfff4383c61af7922bb50803f0ea` — *feat: update triton impl for mi300 & mi355 (#252)*, 2026-03-19
  - `ci.yaml` `TRITON_COMMIT`: not present (the env var was introduced 2026-05-07 by Primus PR #694).
- **Primus-Turbo pin was not changed in the W21 window.** The current pin landed in PR #723 on 2026-05-15 (W20). Pin movements this month: month-start `333b68d7` → W19 PR #694 `ef5b58ea` (+32 commits, 2026-05-07) → W20 PR #723 `06b8d3fe` (+7 more commits, 2026-05-15) → no change in W21. AITER pin: `e83f9903` → W19 `857f4d15` → W20 `b5e03ed191` → no change in W21. `TRITON_COMMIT` pin: introduced in W19 at `88b227e23f04` → no change in W20/W21.
- Recommendation: **monitor**. Pins are in sync (CI and benchmark both at `06b8d3fe`); the next Turbo upstream batch (Primus-Turbo `main` will inform the next bump) is the watch item.

### Notable areas changed since month start

- **HYBRID FP8 grouped GEMM (latest)**: Primus-Turbo #278 adds HYBRID FP8 format support for the Triton backend in `gemm` and `grouped_gemm`; combined with the W19 default switch (#320) and the in-tree Triton source build (`TRITON_COMMIT=88b227e`), this is now the primary kernel path used by Primus-Turbo's `TEGroupedMLP` integration on the Megatron backend.
- **Attention pinning / kernel correctness (W20)**: Primus-Turbo #333 corrects `hipblaslt_gemm` workspace sizing on MI300X; #319 binds per-stream `hipBLASLt` handles to fix the BF16 multi-stream stall on gfx942; #313 removes a stale attention workaround; #332 adds `flash_attn_varlen_func` with the aiter backend.
- **DeepEP combine hang fix (W20)**: Primus-Turbo #327 fixes the DeepEP combine hang; relevant when running DeepEP/MoE paths on ROCm.
- **AITER pin moved (W20)**: `PRIMUS_TURBO_AITER_COMMIT` changed from `857f4d15` to `b5e03ed191` to pick up *Fix dsink bf16 noise in Triton MHA one-kernel backward* (AITER #3070); the Turbo build pulls AITER at this SHA inside the docker image.
- **Carries from earlier this month (W19)**: Grouped-GEMM default → Triton for BF16 and FP8 (#320); `Torch.compile` custom attention wrappers (#310); MoE EPBackend Protocol + `EPBufferConfig` refactor (#297); Symmetric-Memory rewrite (#276); MXFP8/FP8 quantization fixes (#306, #307, #308); `deepep_use_comm_stream` re-introduction (#314); attn `bhsd` (#304) and `sbhd` (#275) layout coverage. (Primus-side consumer for sbhd shipped in W20 #723's outer adapter.)
- **Triton built from source (W19, still active)**: a new explicit `TRITON_COMMIT=88b227e` is wired through `ci.yaml`; `benchmark.yaml` does not pin Triton; the Dockerfile clones `triton-lang/triton@release/3.7.x`, checks out `${TRITON_COMMIT}`, and `pip install --no-build-isolation -v .` to compile Triton from source.

### Primus-Turbo monthly drift table

| Component | Current Version/SHA | Month-start Version/SHA | Delta Summary | Key Changes | Evidence |
| --- | --- | --- | --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI build) | `06b8d3fefd91be26d6adfb5cd43c7524ef87b825` (*Add HYBRID FP8 format support for Triton backend in gemm and grouped_gemm (#278)*, 2026-05-13) | `333b68d7c81b722b21b4aad10cd250c45f15027c` (*fix sm_scale none bug (#263)*, 2026-03-27) | **+39 commits** | HYBRID FP8 grouped GEMM (#278);<br>`hipblaslt_gemm` MI300X workspace fix (#333);<br>per-stream `hipBLASLt` handles for BF16 multi-stream stall on gfx942 (#319);<br>`flash_attn_varlen_func` with aiter backend (#332);<br>DeepEP combine hang fix (#327);<br>carries: grouped-GEMM default → Triton (#320), `Torch.compile` attention wrappers (#310), EPBackend Protocol + EPBufferConfig refactor (#297), SymmetricMemory rewrite (#276), MXFP8/FP8 quant fixes (#306/#307/#308), `deepep_use_comm_stream` (#314). | [`.github/workflows/ci.yaml` L17](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L17), [Primus-Turbo compare](https://github.com/AMD-AGI/Primus-Turbo/compare/333b68d7c81b722b21b4aad10cd250c45f15027c...06b8d3fefd91be26d6adfb5cd43c7524ef87b825) |
| `PRIMUS_TURBO_AITER_COMMIT` (CI build) | `b5e03ed191fca11ee423226537ef8d9435e432a6` (*Fix dsink bf16 noise in Triton MHA one-kernel backward (AITER #3070)*) | `e83f9903c07001a0ec29e85d223f6e6cdbe00859` | **Pin advanced (2-step)** | Month-start `e83f9903` → W19 `857f4d15` → W20 `b5e03ed191`; build pulls AITER at the new SHA inside the docker image. | [`.github/workflows/ci.yaml` L18](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L18) |
| `PRIMUS_TURBO_COMMIT` (benchmark) | `06b8d3fefd91be26d6adfb5cd43c7524ef87b825` (*Add HYBRID FP8 format support for Triton backend in gemm and grouped_gemm (#278)*, 2026-05-13) | `a4488f6cdb15cfff4383c61af7922bb50803f0ea` (*feat: update triton impl for mi300 & mi355 (#252)*, 2026-03-19) | **+44 commits** | Same set of upstream changes as the CI pin plus the earlier interval (`a4488f6c` → `333b68d7`); benchmark and CI pins remain in sync at `06b8d3fe`. | [`.github/workflows/benchmark.yaml` L9](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/benchmark.yaml#L9), [Primus-Turbo compare](https://github.com/AMD-AGI/Primus-Turbo/compare/a4488f6cdb15cfff4383c61af7922bb50803f0ea...06b8d3fefd91be26d6adfb5cd43c7524ef87b825) |
| `TRITON_COMMIT` (CI build) | `88b227e23f0445f3f695bad05bbf1a363b4f50e0` (`triton-lang/triton@release/3.7.x`) | not present | **Unchanged since W19** | Compiles Triton from source inside the docker image; wired through both the main and JAX docker builds. | [`.github/workflows/ci.yaml` L21](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml#L21), [`Dockerfile`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker/Dockerfile) |

## 7. Source Links

- Primus main branch: https://github.com/AMD-AGI/Primus/tree/main
- Primus weekly PR listing (window): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-05-17T16%3A00%3A00Z
- PR #721 (Fix Slurm MASTER_ADDR validation): https://github.com/AMD-AGI/Primus/pull/721
- Megatron-LM pin: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (at report time): https://github.com/NVIDIA/Megatron-LM/commit/e27607a89b4e36ef084590eb10567031f4798a21
- Megatron-LM compare: https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
- torchtitan pin: https://github.com/pytorch/torchtitan/commit/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021
- torchtitan upstream HEAD (at report time): https://github.com/pytorch/torchtitan/commit/b5852826b98278c705954ca95ae2cfcb7e94789e
- torchtitan compare: https://github.com/pytorch/torchtitan/compare/5fb7cc2e3bbb9b9dc0ab7af34ed5cc58b5f32021...main
- Primus-Turbo current pin: https://github.com/AMD-AGI/Primus-Turbo/commit/06b8d3fefd91be26d6adfb5cd43c7524ef87b825
- Primus-Turbo month-start (April→May) compare (CI): https://github.com/AMD-AGI/Primus-Turbo/compare/333b68d7c81b722b21b4aad10cd250c45f15027c...06b8d3fefd91be26d6adfb5cd43c7524ef87b825
- Primus-Turbo month-start (April→May) compare (benchmark): https://github.com/AMD-AGI/Primus-Turbo/compare/a4488f6cdb15cfff4383c61af7922bb50803f0ea...06b8d3fefd91be26d6adfb5cd43c7524ef87b825
- Primus-Turbo W19→W21 compare (since previous pin): https://github.com/AMD-AGI/Primus-Turbo/compare/ef5b58ea3de0a2956d57dba518be466b7a092442...06b8d3fefd91be26d6adfb5cd43c7524ef87b825
- Triton release branch (current pin): https://github.com/triton-lang/triton/commit/88b227e23f0445f3f695bad05bbf1a363b4f50e0
- Month-start reference commit on `main` (May): https://github.com/AMD-AGI/Primus/commit/0a25e200
- Last week's report (W19): https://github.com/AMD-AGI/Primus/blob/main/docs/weekly_reports/2026-W19-primus-weekly.md

---

*Generated automatically by the Primus weekly report automation. Factual statements are derived from `git log origin/main`, the pinned submodule SHAs in `third_party/`, and the `PRIMUS_TURBO_COMMIT`/`PRIMUS_TURBO_AITER_COMMIT`/`TRITON_COMMIT` values in `.github/workflows/{ci,benchmark}.yaml` as observed at 2026-05-22 09:01 GMT+8. Upstream-HEAD SHAs and commit counts are snapshots at report generation time.*
