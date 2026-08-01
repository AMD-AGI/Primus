# Primus Monthly Engineering Report — 2026-08

## 1. Time Window

- Start: Saturday 2026-08-01 09:00 Asia/Shanghai (GMT+8)
- End: Saturday 2026-08-01 17:04 Asia/Shanghai (GMT+8) (report generation time)
- Branch observed: `origin/main`
- Cadence: **monthly** (August 2026 kickoff snapshot)

## 2. Executive Summary

- **0 PRs merged to `main`** in the 2026-08 monthly window (Sat 2026-08-01 09:00 GMT+8 → Sat 2026-08-01 17:04 GMT+8). The window covers only the first ~8 hours of the month, so this is a short-cycle August kickoff snapshot rather than a full-month aggregate. The most recent merge to `main` prior to the window was PR [#937](https://github.com/AMD-AGI/Primus/pull/937) on 2026-07-31 14:05 GMT+8, before the window opened.
- Category breakdown: all categories **0** (no merged PRs in the window).
- **No backend dependency pin/version change inside the monthly window.** There are no commits on `origin/main` at or after `2026-08-01T01:00:00Z` (= 2026-08-01 09:00 GMT+8), so no submodule SHA and no tracked-config pin (`PRIMUS_TURBO_COMMIT`, `PRIMUS_TURBO_AITER_COMMIT`, `TRITON_COMMIT`, `UCCL_COMMIT`) changed in this window.
- **Submodule SHAs (current on `origin/main`):** `third_party/Megatron-LM` `d3528a21`, `third_party/torchtitan` `73a0e697`, `third_party/Megatron-Bridge` `9577b128`, `third_party/Emerging-Optimizers` `93d9eb3a`, `third_party/HummingbirdXT` `ed7b7bd0`, `third_party/maxtext` `a7c6c7e5`, `third_party/mamba` `4d67c534`. None changed inside the August window.
- **torchtitan pin was refreshed pre-window (Q3).** `third_party/torchtitan` advanced from `5fb7cc2e` (2025-10-15) to `73a0e697` (*Bump version to v0.2.2 (#2412)*, 2026-02-20) in PR [#871](https://github.com/AMD-AGI/Primus/pull/871) *feat(torchtitan): upgrade to v0.2.2 for torch 2.12 + GPT-OSS support*, merged 2026-07-17 — inside Q3 but **before** the August monthly window. It is surfaced in the drift section for completeness but does not trigger a backend-gap regeneration this run (the change is outside the window).
- **Megatron-LM upstream drift: `plan sync`.** Pin is `d3528a21` (2026-03-06); upstream `main` HEAD is `f2d4dfadbbdc85a3545791571d38a11ee2d47058` (2026-08-01) — **1045 commits ahead**. The pin is now ~5 months stale and the gap has crossed 1000 commits (632 at the 2026-06 snapshot). Notable upstream activity since the pin includes MFSDP v2 MCore integration with `FullyShardedOptimizer` (#5865), NCCL EP zero copy (#5735), disaggregated KV transfer backends (#5861), inference-optimized Qwen MoE (#5700), streaming replies for the dynamic engine (#5727), and a large body of MFSDP / MoE / inference / CUDA-graph work.
- **torchtitan upstream drift: `plan sync`.** Pin is `73a0e697` (v0.2.2, 2026-02-20); upstream `main` HEAD is `9228564523aa63f78c5e3e038068a886572e90de` (2026-07-31) — **739 commits ahead**. The pin was intentionally moved to the tagged `v0.2.2` release in July (#871), so the drift is measured against fast-moving `main`; the pin itself is now ~5.4 months behind `main` HEAD. Notable upstream activity since the pin includes Kimi K2 model support (#3532), Qwen 3.5 Varlen attention (#3801), MoE support in the transformers modeling backend (#2679), SFT for the HF-Transformer-backed trainer (#3243), and a large RL / GraphTrainer / MoE / FA4 body of work.
- **Primus-Turbo quarterly drift (Q3 2026, anchor 2026-07-01 00:00 GMT+8):** the canonical CI pin advanced from `3c39ef259aa6d724c77c481e926466e7a167e938` (*fix(moe): declare probs layout explicitly in moe_permute (#355)*, 2026-06-03) to `9b5d3092efcbc087657b233d8e9ae662cee6ec6b` (*fix: megakernel nondeterministic and accuracy issues in gfx950. (#429)*, 2026-07-28) — **+67 commits**; the benchmark pin advanced from the same `3c39ef25` to `a04a233cbfb468dbe21600cbf9db70953428b25c` (*feat: force use nt layout gemm in bwd (#386)*, 2026-06-29) — **+35 commits** (CI is +32 commits ahead of the benchmark pin). `PRIMUS_TURBO_AITER_COMMIT` moved from `b5e03ed1` to `0f3c58e6` (AITER v0.1.14.post1) and `TRITON_COMMIT` moved from `88b227e2` to `09500db9` since the quarter start. All of this movement landed earlier in Q3 (July), before the August window. Recommendation: **monitor**.
- **No backend-gap report regenerated this month.** Both surfaced backend report sets (`docs/backend-gap/reports/megatron/upstream-main`, `docs/backend-gap/reports/torchtitan/upstream-main`) compare submodule pins to upstream, and no submodule SHA changed inside the August window. The torchtitan pin bump (#871) and the Primus-Turbo pin bumps are Q3 (July) changes outside this window, so no existing backend-gap report is regenerated during this routine monthly run.

## 3. Monthly PR Update Table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| - | - | - | No merged PRs in this window |

## 4. Megatron-LM Drift Overview

- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned in Primus `main` (`third_party/Megatron-LM`): `d3528a21301db2d12e92912b3ec025dc8a2ed4d6` — *fix(moe): fix TE general_gemm API change (#3582)*, 2026-03-06
- Upstream `main` HEAD: `f2d4dfadbbdc85a3545791571d38a11ee2d47058` — *deps: Update black dependency to version 26.3.0 (#6180)* (2026-08-01)
- Commit gap: **upstream is 1045 commits ahead of Primus pin** (632 at the 2026-06 monthly snapshot; the gap has grown by ~413 commits over two months and crossed the 1000-commit mark).
- Submodule SHA on Primus side: unchanged in the monthly window; last submodule bump on `main` was in PR #654 (merged 2026-04-10, pre-window).
- Recommendation: **plan sync**. The gap continues to grow with large MFSDP v2 / MoE / EP / inference / CUDA-graph work; nothing in the (empty) August window changes Primus's sync risk profile, but the pin is now ~5 months stale and >1000 commits behind.

### Notable upstream areas that have moved since the pin

- **Megatron-FSDP (MFSDP) v2**: MFSDP v2 MCore integration with `FullyShardedOptimizer` (#5865); Normalize MFSDP data-parallel axes at `fully_shard` (#6067); Rebind MFSDP v2 sharded grads once per gradient reduction (#6041); Fix Megatron FSDP optimizer under-update (#5976); Add MFSDP runtime schedule design (#6031).
- **MoE / EP**: NCCL EP zero copy (#5735); inference-optimized Qwen MoE (#5700); update MoE single-weight unit test (#6035); optimize unit metadata for fused shared experts (#6053); fix gradient counting for muon + expert biases (#6099); `Bs invar moe` (#4871).
- **Inference / dynamic engine**: streaming replies for the dynamic engine (#5727); disaggregated KV transfer backends (#5861); extend dynamic-inference async scheduling (#5939); fix LRU block accounting for resumed requests (#5995); harden dynamic prefix-cache allocation for mamba (#6091); fix MLA dynamic-inference decode flag (#4902).
- **CUDA graphs / precision**: Fix GTP full-iteration CUDA-graph capture regression (#6077); align DDP initialization with capture stream (#6021); skip per-param `copy_` dispatch in the MXFP8 param copy-back (#6094); `dist_ckpt: add --stream-ckpt-dequant` for OOM on large FP8/MXFP8 loads (#4451).
- **Checkpoint / distributed correctness**: fix ckpt conversion issue (#6151); send pg group for distributed-checkpoint validation (#6092); enforce identical optimizer shard count between layout computation and training iteration (#6048); populate dp process group in auto-built `ProcessGroupCollection` in pipeline schedules (#5901).

### Megatron-LM upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| Megatron-FSDP v2 | MFSDP v2 MCore integration with `FullyShardedOptimizer`;<br>normalize DP axes at `fully_shard`;<br>rebind v2 sharded grads once per reduction;<br>fix optimizer under-update | NVIDIA/Megatron-LM #5865, #6067, #6041, #5976 | Large FSDP-path evolution that will dominate the Megatron sync diff; Primus patches under `primus/backends/megatron/**` touching FSDP/optimizer must be audited before the next pin bump. |
| MoE / EP | NCCL EP zero copy;<br>inference-optimized Qwen MoE;<br>optimize unit metadata for fused shared experts;<br>fix muon + expert-bias gradient counting | NVIDIA/Megatron-LM #5735, #5700, #6053, #6099 | Aligns with Primus DSv3/Qwen MoE training on MI300X/MI355X; concrete EP zero-copy/perf opportunities unlocked at sync, but require validation against Primus `transformer/moe/` patches. |
| Inference / dynamic engine | Streaming replies for the dynamic engine;<br>disaggregated KV transfer backends;<br>async scheduling extensions;<br>dynamic prefix-cache hardening | NVIDIA/Megatron-LM #5727, #5861, #5939, #6091 | Expands the Megatron inference/serving surface relevant to Primus post-train/eval flows; non-trivial code-path migration expected when Primus adopts the new engine APIs. |
| CUDA graphs / precision | Full-iteration CUDA-graph capture regression fix;<br>DDP init aligned with capture stream;<br>skip per-param `copy_` in MXFP8 param copy-back;<br>`--stream-ckpt-dequant` for large FP8/MXFP8 loads | NVIDIA/Megatron-LM #6077, #6021, #6094, #4451 | Useful for Primus FP8/MXFP8 pretrain configs and CUDA-graph paths; validate against Primus-Turbo FP8 integration after the next sync. |
| Checkpoint / distributed correctness | ckpt conversion fix;<br>pg group for dist-ckpt validation;<br>enforce optimizer-shard-count invariant;<br>populate dp process group in pipeline schedules | NVIDIA/Megatron-LM #6151, #6092, #6048, #5901 | Direct correctness/robustness improvements for Primus large-scale distributed checkpoint + pipeline runs; coordinate with Primus launcher/preflight at sync. |

## 5. torchtitan Drift Overview

- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned in Primus `main` (`third_party/torchtitan`): `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce` — *Bump version to v0.2.2 (#2412)*, 2026-02-20 (tagged `v0.2.2` release)
- Upstream `main` HEAD: `9228564523aa63f78c5e3e038068a886572e90de` — *Always Pre-Split Microbatches for PP (#3856)* (2026-07-31)
- Commit gap: **upstream is 739 commits ahead of Primus pin** (692 at the 2026-06 snapshot, when the pin was the older `5fb7cc2e`).
- Submodule SHA on Primus side: unchanged in the monthly window; the pin was moved from `5fb7cc2e` (2025-10-15) to `73a0e697` (v0.2.2) in PR [#871](https://github.com/AMD-AGI/Primus/pull/871), merged 2026-07-17 (Q3, pre-window).
- Recommendation: **plan sync**. The pin was intentionally upgraded to the tagged `v0.2.2` release in July for torch 2.12 + GPT-OSS support, so Primus is now on a maintained release rather than an ~7.5-month-stale `main` commit. The drift vs fast-moving upstream `main` remains large (739 commits); track the next torchtitan release for the following sync.

### Notable upstream areas that have moved since the pin

- **New models**: Kimi K2 (`k2_7`) model support (#3532); Qwen 3.5 Varlen attention (#3801) + DTensor TP vision position-cache fix (#3899); MoE support added to the transformers modeling backend (#2679); GPT-OSS flex-mask shared decoder helper (#3814).
- **Trainer / SFT / RL**: enable SFT for the HF-Transformer-backed trainer (#3243); RL window FIFO scheduling (#3927); RL trainer entropy metric (#3848); bitwise-parity/RL CI stabilization (#3959, #3981); single-node Qwen3 DAPO math example (#3951).
- **MoE / quantization / float8**: `[MoE]` sibling `token_dispatcher` + `grouped_experts` for composable override (#3859); mxfp8 MoE `ep=1` enablement (#3935); gate `deepseek_v3_671b` float8 converters on hardware capability (#3945); fix float8 `filter_fqns` to exclude `lm_head` not `output` (#4008); MinimalAsyncEP int32 top-k address-overflow fix (#3969).
- **Attention / parallelism / perf**: Always pre-split microbatches for PP (#3856); Select FA4 varlen attention on Blackwell (#4012); CP PTRR load balancer mask-from-dict support (#3972); Unify CosSin/Complex RoPE YaRN computation (#3787); Helion RoPE override for ComplexRoPE (#3767); avoid per-microbatch D2H caused by `isfinite` (#3873).
- **Checkpoint / dataloader / infra**: remote fsspec checkpoint paths (e.g. `gs://`) (#3887); clarify initial-load vs resume behavior (#3732); fix dataloader restart on second resume (#3908); support transformers 5.9.0 + hub 1.24.0 tokenizer download (#3962); Bump DeepEP HybridEP pin (#4033).

### torchtitan upstream feature delta table

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
| --- | --- | --- | --- |
| New models | Kimi K2 (`k2_7`);<br>Qwen 3.5 Varlen attention + vision position-cache fix;<br>MoE in transformers modeling backend;<br>GPT-OSS flex-mask helper | pytorch/torchtitan #3532, #3801, #2679, #3814 | Broadens the model set Primus can pull through its torchtitan backend; adopting Kimi K2 / Qwen 3.5 would need Primus config + adapter work in `examples/torchtitan/configs/` and `primus/backends/torchtitan/**`. |
| Trainer / SFT / RL | SFT for HF-Transformer trainer;<br>RL window FIFO scheduling;<br>RL trainer entropy metric;<br>Qwen3 DAPO math example | pytorch/torchtitan #3243, #3927, #3848, #3951 | Expands Primus's torchtitan-backed post-training surface (SFT/RL); reference patterns for Primus RL/SFT flows after sync. |
| MoE / quantization / float8 | Composable MoE `token_dispatcher` + `grouped_experts`;<br>mxfp8 MoE `ep=1`;<br>hardware-gated dsv3-671b float8 converters;<br>float8 `filter_fqns` `lm_head` fix | pytorch/torchtitan #3859, #3935, #3945, #4008 | Touches the same surface as Primus's torchtitan Qwen3/MoE adapter; after sync the Primus MoE/float8 patches must be re-validated against the composable-override boundaries. |
| Attention / parallelism / perf | Always pre-split microbatches for PP;<br>FA4 varlen attention on Blackwell;<br>CP PTRR mask-from-dict;<br>unify RoPE YaRN + Helion RoPE override | pytorch/torchtitan #3856, #4012, #3972, #3787, #3767 | PP/CP/RoPE changes affect the Primus torchtitan training paths; requires coordinated patches when Primus moves off v0.2.2. |
| Checkpoint / dataloader / infra | Remote fsspec (`gs://`) checkpoints;<br>initial-load vs resume clarification;<br>dataloader second-resume fix;<br>transformers 5.9.0 tokenizer support | pytorch/torchtitan #3887, #3732, #3908, #3962 | Robustness/infra improvements for Primus torchtitan checkpoint + data pipelines on MI300X/MI355X after sync. |

## 6. Primus-Turbo Quarterly Drift Overview

- Drift type: **in-repo, quarterly** — compares the Primus-Turbo pin/version on Primus `main` now vs the value on `main` at the quarter-start anchor `quarter_start_ts = 2026-07-01 00:00 Asia/Shanghai` (`2026-06-30 16:00 UTC`). Q3 2026 is in progress; the next quarterly anchor is `2026-10-01 00:00 GMT+8`.
- Turbo is **not a submodule** in Primus. Canonical version source:
  - `.github/workflows/ci.yaml` → `PRIMUS_TURBO_COMMIT`, `PRIMUS_TURBO_AITER_COMMIT`, `TRITON_COMMIT` (also wired through `.github/workflows/docker/Dockerfile`)
  - `.github/workflows/benchmark.yaml` → `PRIMUS_TURBO_COMMIT`
- Reference Primus commit at quarter start on `main`: `cfaba77815e50a952e1fb42fbaf7d4197150f38a` (*fix: fix padding bug and remove use_turbo_permute_padding flag (#828)*, 2026-06-30 14:40 GMT+8) — last `main` commit at or before `2026-06-30 16:00 UTC`.
- Current state on `origin/main` (2026-08):
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `9b5d3092efcbc087657b233d8e9ae662cee6ec6b` — *fix: megakernel nondeterministic and accuracy issues in gfx950. (#429)*, 2026-07-28
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `0f3c58e6edb6754940bcf9fd5f09ccb6f389f52e` — AITER v0.1.14.post1 (tag commit)
  - `ci.yaml` `TRITON_COMMIT`: `09500db9f0fe66fd176d1f080e2017b37e7e995d`
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `a04a233cbfb468dbe21600cbf9db70953428b25c` — *feat: force use nt layout gemm in bwd (#386)*, 2026-06-29
- Quarter-start (2026-07-01) state on `main`:
  - `ci.yaml` `PRIMUS_TURBO_COMMIT`: `3c39ef259aa6d724c77c481e926466e7a167e938` — *fix(moe): declare probs layout explicitly in moe_permute (#355)*, 2026-06-03
  - `ci.yaml` `PRIMUS_TURBO_AITER_COMMIT`: `b5e03ed191fca11ee423226537ef8d9435e432a6`
  - `ci.yaml` `TRITON_COMMIT`: `88b227e23f0445f3f695bad05bbf1a363b4f50e0`
  - `benchmark.yaml` `PRIMUS_TURBO_COMMIT`: `3c39ef259aa6d724c77c481e926466e7a167e938`
- **Primus-Turbo pin advanced in Q3.** CI pin is **+67 commits** ahead of quarter start; benchmark pin is **+35 commits** ahead of quarter start (CI is +32 commits ahead of the benchmark pin — the two YAML pins are not currently in sync); the AITER pin moved (`b5e03ed1` → `0f3c58e6`, v0.1.14.post1) and the Triton pin moved (`88b227e2` → `09500db9`). All Primus-Turbo movement landed in July; there was **no** Primus-Turbo pin change inside the August monthly window.
- Recommendation: **monitor**. Primus-Turbo is actively maintained and the pins advanced healthily this quarter. Track the next CI/benchmark pin alignment (CI is 32 commits ahead of benchmark) and validate the FlyDSL-based MoE/GEMM and gfx1250 build paths on MI300X/MI355X.

### Notable areas changed since quarter start

- **FlyDSL kernel backends**: Primus-Turbo #412 (*fused Mega Kernel MoE on FlyDSL*), #410 (native FlyDSL grouped MXFP8 GEMM for gfx950), #390/#384/#356/#398/#424 (FlyDSL fp8/mxfp8/mxfp4 dense + grouped GEMM), #420 (Triton & FlyDSL sparse attention) build out the new FlyDSL kernel path that Primus MoE/GEMM configs will exercise.
- **gfx1250 enablement**: Primus-Turbo #374 and #423 add build support for gfx1250 (Parts 1 & 2), extending Primus-Turbo hardware coverage beyond gfx942/gfx950.
- **Quantized-tensor surface expansion**: Primus-Turbo #401/#416/#418/#422 add quantized-tensor support for mxfp8 grouped GEMM, mxfp4 grouped GEMM, blockwise GEMM, and blockwise grouped GEMM; #376 optimizes the mxfp4/mxfp8 dequantize kernel; #365 enables stochastic rounding on MXFP4 gradients.
- **Grouped-GEMM work-stealing + perf**: Primus-Turbo #348/#353 add CK and Triton work-stealing grouped-GEMM variants with a schedule API; #407 improves fp8 dgrad/wgrad performance; #408 zeroes uncovered padding tails; #425 fixes grouped fp8 wgrad memory faults on empty (0-token) expert groups.
- **Attention & communication**: Primus-Turbo #377 adds Ulysses context-parallel Varlen attention; #409 migrates the ODC rocSHMEM communication backend into Primus-Turbo; #415 adds a transpose-2d kernel to replace `aten::t`.
- **Correctness / stability**: Primus-Turbo #429 (the current CI pin) fixes megakernel nondeterminism and accuracy on gfx950; #388 fixes an MXFP8 backward crash on non-contiguous `grad_out`; #414/#383 fix fp4/fp8 recipe correctness and scale-pad bugs.

### Primus-Turbo quarterly drift table

| Component | Current Version/SHA | Quarter-start (2026-07-01) Version/SHA | Delta Summary | Key Changes | Evidence |
| --- | --- | --- | --- | --- | --- |
| `PRIMUS_TURBO_COMMIT` (CI build) | `9b5d3092efcbc087657b233d8e9ae662cee6ec6b` (*fix: megakernel nondeterministic and accuracy issues in gfx950. (#429)*, 2026-07-28) | `3c39ef259aa6d724c77c481e926466e7a167e938` (*fix(moe): declare probs layout explicitly in moe_permute (#355)*, 2026-06-03) | **+67 commits** | FlyDSL MoE/GEMM backends (#412, #410, #390, #384, #398, #424);<br>gfx1250 build support (#374, #423);<br>quantized-tensor surface for grouped/blockwise GEMM (#401, #416, #418, #422);<br>grouped-GEMM work-stealing + perf (#348, #353, #407, #408);<br>Ulysses CP Varlen attention (#377);<br>ODC rocSHMEM backend migration (#409);<br>megakernel gfx950 correctness fix (#429). | [`.github/workflows/ci.yaml`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml), [Primus-Turbo compare](https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...9b5d3092efcbc087657b233d8e9ae662cee6ec6b) |
| `PRIMUS_TURBO_COMMIT` (benchmark) | `a04a233cbfb468dbe21600cbf9db70953428b25c` (*feat: force use nt layout gemm in bwd (#386)*, 2026-06-29) | `3c39ef259aa6d724c77c481e926466e7a167e938` (*fix(moe): declare probs layout explicitly in moe_permute (#355)*, 2026-06-03) | **+35 commits** | Subset of the CI change set through #386 (nt-layout bwd GEMM); the benchmark pin trails the CI pin by 32 commits. | [`.github/workflows/benchmark.yaml`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/benchmark.yaml), [Primus-Turbo compare](https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...a04a233cbfb468dbe21600cbf9db70953428b25c) |
| `PRIMUS_TURBO_AITER_COMMIT` (CI build) | `0f3c58e6edb6754940bcf9fd5f09ccb6f389f52e` (AITER v0.1.14.post1) | `b5e03ed191fca11ee423226537ef8d9435e432a6` | **Pin advanced** | AITER pin advanced to the v0.1.14.post1 tag commit in Q3; the Turbo build pulls AITER at the new SHA inside the docker image. | [`.github/workflows/ci.yaml`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml) |
| `TRITON_COMMIT` (CI build) | `09500db9f0fe66fd176d1f080e2017b37e7e995d` | `88b227e23f0445f3f695bad05bbf1a363b4f50e0` | **Pin advanced** | Triton source-build commit advanced in Q3; compiled from source inside the docker image. `benchmark.yaml` does not pin Triton. | [`.github/workflows/ci.yaml`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/ci.yaml), [`Dockerfile`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker/Dockerfile) |

## 7. Source Links

- Primus main branch: https://github.com/AMD-AGI/Primus/tree/main
- Primus August monthly PR listing (window): https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A%3E%3D2026-08-01T01%3A00%3A00Z
- PR #871 (feat(torchtitan): upgrade to v0.2.2 for torch 2.12 + GPT-OSS support): https://github.com/AMD-AGI/Primus/pull/871
- Megatron-LM pin: https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- Megatron-LM upstream HEAD (at report time): https://github.com/NVIDIA/Megatron-LM/commit/f2d4dfadbbdc85a3545791571d38a11ee2d47058
- Megatron-LM compare (pin → HEAD): https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main
- torchtitan pin: https://github.com/pytorch/torchtitan/commit/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce
- torchtitan upstream HEAD (at report time): https://github.com/pytorch/torchtitan/commit/9228564523aa63f78c5e3e038068a886572e90de
- torchtitan compare (pin → HEAD): https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main
- Primus-Turbo current CI pin: https://github.com/AMD-AGI/Primus-Turbo/commit/9b5d3092efcbc087657b233d8e9ae662cee6ec6b
- Primus-Turbo current benchmark pin: https://github.com/AMD-AGI/Primus-Turbo/commit/a04a233cbfb468dbe21600cbf9db70953428b25c
- Primus-Turbo quarter compare (CI, Q3-start → current): https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...9b5d3092efcbc087657b233d8e9ae662cee6ec6b
- Primus-Turbo quarter compare (benchmark, Q3-start → current): https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...a04a233cbfb468dbe21600cbf9db70953428b25c
- Triton commit (current pin): https://github.com/triton-lang/triton/commit/09500db9f0fe66fd176d1f080e2017b37e7e995d
- Quarter-start reference commit on `main` (2026-07-01): https://github.com/AMD-AGI/Primus/commit/cfaba778
- Previous monthly report (2026-06): https://github.com/AMD-AGI/Primus/blob/dashboard-data/docs/monthly_reports/2026-06-primus-monthly.md

---

*Generated automatically by the Primus monthly report automation. Factual statements are derived from `git log origin/main`, the pinned submodule SHAs in `third_party/`, and the `PRIMUS_TURBO_COMMIT`/`PRIMUS_TURBO_AITER_COMMIT`/`TRITON_COMMIT` values in `.github/workflows/{ci,benchmark}.yaml` as observed at 2026-08-01 17:04 GMT+8. Upstream-HEAD SHAs and commit counts are snapshots at report generation time.*
