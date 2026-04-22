# Primus Weekly Engineering Report — 2026-W17

- **Repository:** [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)
- **Base branch:** `main`
- **Report ID:** `2026-W17`
- **Generated:** 2026-04-22 (Asia/Shanghai, GMT+8)

## 1. Time window

- **Start:** 2026-04-20 00:00:00 Asia/Shanghai (Monday 00:00 GMT+8)
- **End:** 2026-04-22 18:11 Asia/Shanghai (report generation time)
- **ISO week:** 2026-W17

## 2. Executive summary

- **Total merged PRs on `main` this window:** 1.
- **Category breakdown:** Bug Fix ×1 (100%). No PRs in Performance Optimization, Turbo/Dependency Version Update, CI/Infra, Refactor, Docs, or Other categories.
- **Turbo / dependency version updates this week:** None observed.
- **Primus `main` HEAD at report time:** `4990e517` — `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT (#674)`.
- **Megatron-LM submodule drift:** Pinned at `d3528a21` (2026-03-06); upstream `main` HEAD `bbc6b4d1` (2026-04-22). **Primus is ~344 commits behind upstream `main`, ~47 calendar days of drift.**
- **Notable upstream themes accumulated since pinned SHA:** MoE router/score function changes, Mamba → HybridModel rename + new attention features, TransformerEngine 2.14 bump, FA4 inference, async checkpoint CPU-SHM support, multiple RL fixes, rampup batch size scheduler rework.
- **Risk posture:** The only in-scope change this week is a safety/robustness fix that tightens our compatibility layer against the upstream `TransformerBlock._checkpointed_forward` (adds a SHA256 fingerprint guard + UT). It does not pull new upstream code; it only hardens detection of upstream drift.
- **Recommendation:** **plan sync.** Drift is large (344 commits, multiple MoE / RL / checkpoint / TE-version changes) but no urgent regression has been reported against pinned `d3528a21`. A scheduled Megatron-LM rebase PR is recommended before another week of drift accumulates.

## 3. Weekly PR update table

Time window (GMT+8): 2026-04-20 00:00 → 2026-04-22 18:11

| PR | Merged Time (GMT+8) | Category | Key Update |
|---|---|---|---|
| [#674](https://github.com/AMD-AGI/Primus/pull/674) | 2026-04-22 16:30 | Bug Fix | `fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT` — Rewrites `primus/backends/megatron/patches/recompute_layer_patches.py` so the inner `custom` / `checkpoint_handler` closures are byte-for-byte identical to upstream Megatron's `TransformerBlock._checkpointed_forward`. Delegates fast-path to upstream when `config.recompute_layer_ids` is `None`. Adds a SHA256 fingerprint guard test, a pipeline-stage offset test, and an FP8/no-grad skip test so future upstream edits surface as actionable CI failures. Removes the stale `PrimusTransformerBlock` subclass. Author: `lhzhang333`. |

### Per-PR summary

- **PR #674 — Bug Fix / Compatibility hardening**
  - Files touched: `primus/backends/megatron/patches/recompute_layer_patches.py`, `tests/unit_tests/backends/megatron/test_recompute_layer_patches.py`, deletion of legacy `primus/backends/megatron/core/transformer/transformer_block.py`.
  - Behavior change for users: none when `recompute_layer_ids` is unset (delegated to upstream unchanged).
  - Notable invariants now covered by UT:
    1. Fingerprint lock against upstream `_checkpointed_forward` source.
    2. `recompute_layer_ids` are global indices, correctly mapped to block-local indices via `get_transformer_layer_offset`.
    3. Mirrors upstream's `(fp8 or fp4) and not hidden_states.requires_grad` skip rule.

## 4. Megatron-LM drift overview

| Field | Value |
|---|---|
| Submodule path | `third_party/Megatron-LM` |
| Upstream | `https://github.com/NVIDIA/Megatron-LM.git` (`main`) |
| Pinned SHA in Primus `main` | [`d3528a21301db2d12e92912b3ec025dc8a2ed4d6`](https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6) |
| Pinned commit subject | `fix(moe): fix TE general_gemm API change (#3582)` |
| Pinned commit date | 2026-03-06 22:21:48 +0800 |
| Upstream `main` HEAD SHA | [`bbc6b4d14bd8f787b803b3382147dac4cecb20ec`](https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec) |
| Upstream HEAD commit subject | `chore: rotate oncall schedule` |
| Upstream HEAD commit date | 2026-04-22 17:34:12 +0800 |
| Upstream commit gap (ahead count) | **344 commits** |
| Calendar gap | ~47 days |
| Drift trend this week | No submodule bump merged this week. Drift grew by the natural upstream cadence. |

**Assumption (labelled):** commit gap was computed from a fetched shallow clone of `NVIDIA/Megatron-LM` deepened to 800 commits from `main`; the full-history gap is `rev-list --count d3528a21..bbc6b4d1 = 344`, which is conclusive because the pinned SHA is already reachable.

## 5. Upstream feature delta table

High-signal upstream additions/fixes present in `bbc6b4d1` but missing in pinned `d3528a21`.

| Area | New Upstream Capability | Evidence (PR/Commit) | Potential Impact to Primus |
|---|---|---|---|
| MoE / Router | New score function for the router; improvement of shared expert overlap, plus shared-expert overlap for `FlexDispatcher` | [#3673](https://github.com/NVIDIA/Megatron-LM/pull/3673), [#2207](https://github.com/NVIDIA/Megatron-LM/pull/2207) | MoE recipes in `primus/backends/megatron/*` and `primus/configs/modules/megatron` may want to expose / default these; routing numerics could shift if Primus later rebases. |
| Mamba / Hybrid models | `MambaModel`/`MambaStack` renamed to `HybridModel`/`HybridStack`; QK-LayerNorm for DPA in MambaModel; DeepSeek Sparse Attention ported to MambaModel | [#4099](https://github.com/NVIDIA/Megatron-LM/pull/4099), [#4067](https://github.com/NVIDIA/Megatron-LM/pull/4067), [#3553](https://github.com/NVIDIA/Megatron-LM/pull/3553) | Rename is a breaking import change; Primus Mamba/Hybrid recipes (if any consumer exists downstream) will need symbol updates at rebase time. |
| Attention / Inference | FA4 Inference integration | [#4186](https://github.com/NVIDIA/Megatron-LM/pull/4186) | Opens door to a faster inference path; requires evaluating ROCm/FA4 parity before Primus adopts. |
| TransformerEngine | Bump TransformerEngine to `release_v2.14`; fix TE version check for `retain_pinned_cpu_buffers` in CPU offload | [#4331](https://github.com/NVIDIA/Megatron-LM/pull/4331), [#4267](https://github.com/NVIDIA/Megatron-LM/pull/4267) | Primus docker/env pins for TE must be re-validated during next Megatron-LM bump; FP8 / CPU-offload paths most exposed. |
| Checkpointing | `--async-ckpt-use-cpu-shm` argument; coredump fix when saving checkpoints; `save_checkpoint_and_time()` elapsed-duration accounting | [#4355](https://github.com/NVIDIA/Megatron-LM/pull/4355), [#1871](https://github.com/NVIDIA/Megatron-LM/pull/1871), [#4263](https://github.com/NVIDIA/Megatron-LM/pull/4263) | Reduces checkpoint stall on large-scale runs; Primus trainer configs can surface the CPU-SHM flag once rebased. |
| Training schedule | Replace rampup batch size scheduler with custom step batch size schedules (was reverted once and re-landed) | [#3779](https://github.com/NVIDIA/Megatron-LM/pull/3779), [#4411](https://github.com/NVIDIA/Megatron-LM/pull/4411), revert [#4404](https://github.com/NVIDIA/Megatron-LM/pull/4404) | CLI-level change to batch-size scheduling; watch for config-key rename when rebasing Primus launch scripts. |
| RL / post-training | Onload optimizer after logprobs; make `--skip-train` work again; fix reward due to stop token; fix non-partial rollouts; RL staleness tables/histogram | [#4235](https://github.com/NVIDIA/Megatron-LM/pull/4235), [#4249](https://github.com/NVIDIA/Megatron-LM/pull/4249), [#4096](https://github.com/NVIDIA/Megatron-LM/pull/4096), [#3964](https://github.com/NVIDIA/Megatron-LM/pull/3964), [#4097](https://github.com/NVIDIA/Megatron-LM/pull/4097) | Directly relevant to Primus posttraining surfaces; several are correctness fixes (reward / `--skip-train`). |
| Distributed | `param_index_map` uses unpacked (full numel) offsets; wait for async P2P send before deallocating output tensor; Megatron-FSDP 0.5.0 | [#4328](https://github.com/NVIDIA/Megatron-LM/pull/4328), [#4047](https://github.com/NVIDIA/Megatron-LM/pull/4047), commit `97f9ab6f` | Subtle correctness fixes for optimizer-state mapping and pipeline P2P; good to absorb during next rebase. |
| CP / UT | CP: fix UT timeout | [#4373](https://github.com/NVIDIA/Megatron-LM/pull/4373) | Reduces flakiness of context-parallel UTs downstream. |
| Init / CLI | Fix Megatron initialization with `extra_args_provider`; extract args init to launch scripts; remove legacy biencoder/realm/vision models | [#4327](https://github.com/NVIDIA/Megatron-LM/pull/4327), [#4225](https://github.com/NVIDIA/Megatron-LM/pull/4225), [#4202](https://github.com/NVIDIA/Megatron-LM/pull/4202), [#4205](https://github.com/NVIDIA/Megatron-LM/pull/4205) | Primus launchers that pass `extra_args_provider` benefit from the fix; legacy-model removals may require cleaning any stale imports at rebase time. |
| CI / infra (upstream) | Retry loops on `uv install` and `apt-get update`; GB200 1-node MR functional tests | [#4387](https://github.com/NVIDIA/Megatron-LM/pull/4387), [#4209](https://github.com/NVIDIA/Megatron-LM/pull/4209), [#4334](https://github.com/NVIDIA/Megatron-LM/pull/4334) | Informational — NVIDIA-side CI; no direct Primus action. |

**Assumption (labelled):** Impact statements above are best-effort inferences from PR titles/descriptions; exact Primus impact requires code-level diffing at rebase time.

## 6. Source links

- **Primus `main` at report time:** https://github.com/AMD-AGI/Primus/commit/4990e517
- **Merged PRs on Primus `main` this week:**
  - https://github.com/AMD-AGI/Primus/pull/674
- **Megatron-LM pinned SHA:** https://github.com/NVIDIA/Megatron-LM/commit/d3528a21301db2d12e92912b3ec025dc8a2ed4d6
- **Megatron-LM upstream HEAD:** https://github.com/NVIDIA/Megatron-LM/commit/bbc6b4d14bd8f787b803b3382147dac4cecb20ec
- **Megatron-LM compare (pinned → HEAD):** https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...bbc6b4d14bd8f787b803b3382147dac4cecb20ec
- **Primus submodule configuration:** https://github.com/AMD-AGI/Primus/blob/main/.gitmodules

---

*Generated by the Primus weekly engineering report automation. Source: `git log` on `origin/main` restricted to Asia/Shanghai Mon 00:00 — now; Megatron-LM drift computed from a shallow clone of `NVIDIA/Megatron-LM` `main` deepened to 800 commits (pinned SHA reachable, `rev-list --count` conclusive).*
