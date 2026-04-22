# Primus Weekly Update: Main PRs + Megatron-LM Upstream Delta (Mon-to-Now)

- **Report window:** 2026-04-20 00:00 UTC (Monday) → 2026-04-22 08:36 UTC
- **Repositories:** `AMD-AGI/Primus` (branch `main`), submodule `third_party/Megatron-LM`
- **Primus `Megatron-LM` pin:** `d3528a21` (2026-03-06) — "fix(moe): fix TE general_gemm API change (#3582)"
- **Upstream `NVIDIA/Megatron-LM` main tip:** `e7789676` (2026-04-21) — "RL: Onload optimizer after logprobs computation (#4235)"
- **Commit distance:** `0` ahead / **`343`** behind (`342` first-parent PR merges).

---

## 1. Executive Summary

- Only **1 PR** merged into `Primus` `main` this week (Mon → now): PR **#674** — a `Megatron-LM` compatibility / hardening fix with unit tests around `recompute_layer_patches`.
- The single PR is low-risk: it keeps the patch byte-for-byte aligned with upstream `TransformerBlock._checkpointed_forward`, adds a source-fingerprint guard + behavior tests, and removes a stale dead-code subclass.
- `Primus` is pinned to `Megatron-LM @ d3528a21` (2026-03-06); upstream `main` has advanced by **343 commits** since then — ~6.5 weeks of NVIDIA work to digest.
- Notable upstream net-new features since the pin: **FA4 inference**, **NVFP4 native weights in DDP**, **FP8 DPA for MXFP8 recipe**, **DeepSeek Sparse Attention on `MambaModel`**, **Gated Delta Net CP**, **absorbed-MLA** and **fused MLA DOWN GEMM**, **Lion optimizer**, **speculative decoding with MTP layers**, **async-ckpt CPU-SHM**, **hybrid prefix caching**, **shared-expert overlap for FlexDispatcher**, and a **new MoE router score function**.
- Notable breaking/renaming churn: `MambaModel/MambaStack → HybridModel/HybridStack`; removal of legacy **vision / biencoder-realm / t5 / BERT / data / mpu / encoder_and_decoder** modules; `TransformerEngine` bumped to **v2.13 → v2.14**; DeepEP bump; `megatron-fsdp` → 0.5.0; `async_allgather` renamed to `overlap_param_gather`; rampup batch-size scheduler was replaced (and the replacement landed after one revert round-trip).
- Several **checkpoint / RL / inference** fixes upstream are directly relevant to Primus code paths (recompute, RL loops, CP, async save).
- Recommended near-term adoption candidates: FP8-DPA-on-MXFP8, NVFP4 DDP weights, async-ckpt CPU-SHM, shared-expert-overlap + FlexDispatcher, absorbed-MLA / fused MLA DOWN GEMM.
- Recommended to defer / gate: `HybridModel` rename, TE v2.14 bump, rampup-batch-size replacement, and legacy-module removals — these require API audits across Primus patches.

---

## 2. Weekly `main` PR Table (2026-04-20 → now)

**Total merged PRs:** 1

### Category summary

| Category | Count |
|---|---|
| Bug Fix (Megatron compatibility) | 1 |
| Performance Optimization | 0 |
| Turbo Update | 0 |
| Feature | 0 |
| Refactor | 0 |
| Infra / CI | 0 |
| Docs | 0 |
| Other | 0 |

### PRs

| PR # | Title | Author | Merged At (UTC) | Category | Key Changes |
|---|---|---|---|---|---|
| [#674](https://github.com/AMD-AGI/Primus/pull/674) | fix(megatron): adapt recompute_layer_patches to the upstream Megatron and add UT | `lhzhang333` | 2026-04-22 08:30:54 | Bug Fix (Megatron compat) | • Rewrites `recompute_layer_patches` so that when `config.recompute_layer_ids is None` it delegates verbatim to Megatron, and the inner `custom`/`checkpoint_handler` closures are byte-for-byte identical to upstream; removes stale `PrimusTransformerBlock`. • Adds fingerprint + behavior tests (pipeline-stage offset, FP8 no-grad skip, negative-index validation) to catch future Megatron drift in CI. |

---

## 3. `Megatron-LM` Upstream Delta (d3528a21 → e7789676, 343 commits)

### 3.1 Feature-delta table (meaningful changes only)

| Area | Upstream New Feature / Change | Evidence (hash · PR) | Potential Impact on Primus |
|---|---|---|---|
| Training perf · FP8 | Enable FP8 DPA for the MXFP8 recipe | `22e0bb5fd` · PR #4066 | Unlocks MXFP8 + FP8 dot-product-attention on MI-class FP8 paths; validate with Primus MXFP8 recipes. |
| Training perf · FP4/FP8 | NVFP4 native weights for DDP | `e1db4a03d` · PR #4005 | New DDP quant path; needs interop check with Primus DDP/FSDP patches. |
| Training perf · MLA | Fuse MLA DOWN projection GEMMs; add `absorbed-mla` | `8fd390d1f` · PR #3039; `204e7d57b` · PR #3198 | Large MLA speed win for DeepSeek-style models; intersects Primus MLA patches. |
| Training perf · MoE | Shared-expert overlap support for FlexDispatcher | `ebfa13852` · PR #2207 | Hides shared-expert compute behind dispatch; good fit for Primus MoE recipes. |
| Training perf · MoE | Reduce number of shared-expert streams | `6da62672c` · PR #3752 | Reduces stream pressure / memory for MoE shared experts. |
| Training perf · MoE | Reuse grad buffer for layer-wise param allgather | `28e13c484` · PR #3751 | Memory footprint reduction for layer-wise distributed optimizer. |
| MoE features | New score function added to the router | `eb80b7491` · PR #3673 | New routing option — verify config plumbing in Primus MoE configs. |
| MoE inference | Torch grouped-gemm BF16 + MXFP8 + CUDA-graphed inference-optimized MoEs | `589cd9e12` · PR #3858 | Improves MoE inference on non-NV backends; cross-check with Primus inference. |
| Distributed / CP | `[MoE]` Gated Delta Net context parallel | `20ba03fec` · PR #2642 | Enables CP for new MoE variant; extends Primus CP coverage. |
| Distributed / CP | Reset AG_pipeline bucket status after validation step | `5f80f0ac6` · PR #3155 | Fixes stale AG bucket in eval-after-train — directly relevant for Primus pretrain loop. |
| Distributed / CP | `[MLA]` Pad V when Q/V head dims differ for THD | `5dcda195a` · PR #3003 | Correctness for THD + asymmetric MLA; check THD code in Primus. |
| Distributed / AG-RS | AG/RS overlap via explicit process-group passing | `2ebfbb21f` · PR #3249 | API change for overlap — adjust Primus TP/SP wiring. |
| Runtime / API | Rename `MambaModel`/`MambaStack` → `HybridModel`/`HybridStack` | `15e07a2dd` · PR #4099 | **Breaking rename.** Any Primus import of `MambaModel`/`MambaStack` will need aliasing. |
| Runtime / args | Fix Megatron initialization with `extra_args_provider` | `afae25b5f` · PR #4327 | Primus uses custom arg providers — review for compatibility. |
| Runtime / args | Extract args init to launch scripts | `97aca2f88` · PR #4225 | Launcher flow change; impacts Primus pretrain/launch wrappers. |
| Runtime / async | `async_allgather` renamed to `overlap_param_gather` | `1b425abdd` · PR #4217 | Rename propagation needed in Primus configs/flags. |
| Scheduler | Replace rampup batch-size scheduler with custom step batch-size schedules | `c9e03d0c3` / `532ad926b` · PR #3779/#4411 (one revert #4404 in between) | **Behavior change** for batch-size ramp; re-tune Primus pretrain configs. |
| Scheduler / loops | Support multimodule pipelining in 1F1B schedule | `0ca9b6395` · PR #3129 | May enable new pipeline topologies for Primus. |
| Checkpointing | `--async-ckpt-use-cpu-shm` flag | `997896883` · PR #4355 | New async-ckpt option worth enabling in Primus large-scale runs. |
| Checkpointing | Remove cross-rank sync during checkpoint load; deprecate `load_state_dict` | `0602523f7` · PR #2864 | Speedup + **deprecation**; verify Primus `ckpt` loaders. |
| Checkpointing | Fix potential coredump when saving checkpoint | `ded22f428` · PR #1871 | Stability fix for Primus saves. |
| Checkpointing | `save_checkpoint_and_time()` restructuring / elapsed timing | `7928a84e8` · PR #4263 | Minor API / log format change for ckpt timing. |
| Checkpointing | Optimize process mgmt + delete ops for async save | `17de0dbd9` · PR #3262 | Faster async save cleanup; transparent win. |
| Checkpointing | Zero-copy storage sharing in `CheckpointWithoutOutput` | `f54403492` · PR #3649 | Memory win for activation ckpt + output tensor. |
| Optimizer | Add **Lion** optimizer | `83498ef9c` · PR #3813 | New optimizer path — expose via Primus configs. |
| Optimizer | Make `param_index_map` always use unpacked offsets | `3315c86bc` · PR #4328 | Distributed optimizer invariant change; audit Primus DistOpt patches. |
| Optimizer | Remove unnecessary args for layerwise distributed optimizer | `1ca250c6f` · PR #4272 | Signature trim — check Primus callers. |
| Optimizer | Fix layerwise optimizer with `expt_dp_size=1` & elementwise DistOpt contention | `51bcf1470` · PR #4138 | Bug fix for MoE+DistOpt. |
| Offload / memory | Fine-grained activation offloading for Mamba/Hybrid | `92d5c1f4d` · PR #4173 | Memory win for Hybrid models. |
| Offload / memory | Retain pinned CPU buffers for CPU offloading | `fde4059a9` · PR #3151 | Throughput win for CPU offload path. |
| Offload / memory | Fix TE version check for `retain_pinned_cpu_buffers` | `c2d1a8f7e` · PR #4267 | Follow-up fix; matters when bumping TE. |
| Offload / memory | Enable CPU offloading with full-iteration CUDA graph | `d30c3ae54` · PR #3969 | Memory win; verify with Primus CUDA-graph paths. |
| Offload / memory | Reset activation offload manager after eval | `b8e23d587` · PR #3739 | Bug fix for eval-after-train; relevant to Primus pretrain. |
| TE / builds | Bump `TransformerEngine` to `release_v2.14` | `ceac26946` · PR #4331 | **Env bump.** Requires Primus image / ROCm-TE compat review. |
| TE / builds | Bump DeepEP to `34152ae` | `123645bb7` · PR #4228 | DeepEP refresh — verify expert-parallel perf. |
| TE / builds | Set `megatron-fsdp` to 0.5.0 | `97f9ab6f2` | MFSDP minor bump. |
| Model support | Port DeepSeek Sparse Attention to `MambaModel` | `a00e9443c` · PR #3553 | Enables DSA on Hybrid — model expansion for Primus. |
| Model support | Add QK layernorm for DPA in `MambaModel` | `e15ec3c04` · PR #4067 | Stability for hybrid training. |
| Model support | `conditions_embeddings` on `TransformerBlock`/`Layer` for DiT | `1daa19f89`/revert `cc4cb0119` · PR #4134/#4270 | DiT support landed then reverted — currently **not** in main. |
| Model support | Add activation logging & tokens-per-expert logging | `8be1e79d1` · PR #3842 | Better observability for MoE; easy adopt. |
| Inference | **FA4 inference** (FlashAttention-4) | `76ac7c24b` · PR #4186 | Major inference perf for supported HW. |
| Inference | Hybrid prefix caching | `c4bffde9e` · PR #3225 | Inference perf for hybrid models. |
| Inference | Speculative decoding with MTP layers | `8f539df74` · PR #3594 | New decoding path — interesting for posttraining eval. |
| Inference | MTP inference fixes + `materialize_only_last_token_logits` | `980211ae6`/`c1463050b` · PR #4191/#4166 | Stability for MTP inference. |
| Inference | Mamba EP inference unit test (eager fallback + mixed CUDA graphs) | `8cf6b355a` · PR #4085 | Coverage only. |
| Inference / fusion | `M4` leftover for TE cuda graph | `01eb7e87e` · PR #3137 | TE cudagraph path consolidation. |
| Inference / fusion | Fused dLN + add in backwards pass | `8318b8093` · PR #3384 | Backward fusion win; portable to Primus. |
| Inference / fusion | Exposing `interleave` for `fused_apply_rotary_pos_emb_thd` | `87eb3c2b7` · PR #3794 | API surface — trivial adopt. |
| RL | Onload optimizer after logprobs computation | `e7789676f` · PR #4235 | RL memory footprint fix. |
| RL | Hybrid MoE training cudagraphs + training↔inference transition fix | `e19fbe2e1` · PR #3373 | Major RL perf/stability fix. |
| RL | RL works again with `--skip-train` / inference-only | `98a51eb2c`/`29e798a48` · PR #4249/#3744 | RL eval loops. |
| RL | Fix RL reward due to stop token | `e4d3a4c4f` · PR #4096 | Correctness. |
| RL | Fix bug with non-partial rollouts | `ed5de26a4` · PR #3964 | Correctness. |
| RL | Tables + histogram for RL staleness | `23663a870` · PR #4097 | Observability. |
| RL | Reverse polarity of off-policy measurement | `43675d4b2` · PR #3580 | Metric semantics — update dashboards. |
| Dataloader / data | Defensively close GPU device FDs in dataloader workers | `1259982ed` · PR #3684 | FD leak fix — relevant to Primus long runs. |
| Dataloader / data | Find optimal number of workers | `f26190677` · PR #3699 | Auto-tune for data loader. |
| NVTX / debug | Enhance and fix NVTX for training | `df929f574` · PR #3642 | Profiler surface — trivial adopt. |
| NVTX / debug | NCCL flight recorder configuration support | `251a7545f` · PR #3806 | Debuggability for comm hangs. |
| P2P / comms | Wait for async P2P send before deallocating output tensor | `260cba713` · PR #4047 | Correctness for PP; relevant to Primus. |
| Misc cleanup / breaking | Remove legacy **vision**, **biencoder+REALM**, **T5**, **BERT**, **data**, **mpu**, **encoder_and_decoder** modules | `e69cf431a`, `d245c4438`, `81b8b5acc`, `7f8f37e0c`, `925341d28`, `b5b1994bd`, `b7437fe36` · PR #4202/#4205/#4203/#4204/#3853/#3854/#3836 | **Breaking deletions.** Any Primus code importing these paths will break on next bump. |
| Misc cleanup | Remove `packed_attention_mask` unused parameter | `b562151f4` · PR #3859 | Signature change. |
| Misc cleanup | Move inference guards out of `arguments.py` | `be09bb65a` · PR #4210 | Arg-parsing refactor; check Primus arg overrides. |
| Misc cleanup | Handle list-typed process groups in `ProcessGroupCollection.__repr__` | `dda69017d` · PR #3753 | Log hygiene. |
| MFSDP | Fix decoupled_grad + DistOpt mechanics | `ab43d43f0` · PR #4133 | MFSDP correctness. |
| MFSDP | Build `expt_device_mesh` only for MoE | `25129bf3d` · PR #3831 | Startup fix. |
| MFSDP | Auto default for MixedPrecisionPolicy | `ff70b2488` · PR #3810 | Usability. |
| Upcycling | Fix `TransformerConfig` validation for mixed dense/MoE upcycling | `e8e79a43e` · PR #3647 | Matters for Primus MoE upcycling recipes. |

### 3.2 Out of scope (noise/churn, not included above)

- Docs, `copy-pr-bot` / skills / oncall rotations, README edits, CLAUDE.md ↔ AGENTS.md rename, CI-only changes, Nemotron tokenizer updates, flaky-test markers, and similar.

---

## 4. Risks / Recommendations

### Adopt now (low risk, clear value)
- **Checkpoint stability:** cherry-pick `ded22f428` (coredump fix), `17de0dbd9` (async save perf), `7928a84e8` (elapsed timing).
- **MoE observability:** `8be1e79d1` (activation + tokens-per-expert logging).
- **Profiling:** `df929f574` (NVTX fixes), `251a7545f` (NCCL flight recorder).
- **PP correctness:** `260cba713` (wait for async P2P before dealloc), `5f80f0ac6` (AG_pipeline reset after validation).
- **Dataloader stability:** `1259982ed` (FD leak in workers).

### Pilot behind flags / perf dashboards
- **FA4 inference** (`76ac7c24b`), **FP8 DPA for MXFP8** (`22e0bb5fd`), **NVFP4 native weights for DDP** (`e1db4a03d`) — validate on target AMD HW paths.
- **MLA:** `8fd390d1f` (fused DOWN GEMM) + `204e7d57b` (absorbed-MLA) — prospective large wins on DeepSeek-style MLA training.
- **MoE:** `ebfa13852` (shared-expert overlap + FlexDispatcher), `6da62672c`, `28e13c484`, `eb80b7491` (new router score fn).
- **RL loop:** `e19fbe2e1` (MoE cudagraphs + train↔infer transition) and `e7789676f` (optimizer onload after logprobs).

### Monitor / gate before bumping submodule
- **TE v2.14 bump** (`ceac26946`) — coordinate with ROCm-TE compatibility; note `c2d1a8f7e` adds the version check.
- **DeepEP bump** (`123645bb7`), **megatron-fsdp 0.5.0** (`97f9ab6f2`).
- **Rampup batch-size scheduler** replaced (`c9e03d0c3` + `532ad926b`, one revert in between) — re-tune any Primus configs that rely on the legacy rampup.
- **Async API rename** `async_allgather → overlap_param_gather` (`1b425abdd`) — search/replace in Primus configs.
- **`MambaModel` → `HybridModel` rename** (`15e07a2dd`) — add compat aliases in Primus where imported.

### Defer until planned Megatron bump PR
- Removals of **vision / biencoder-realm / T5 / BERT / data / mpu / encoder_and_decoder** legacy modules — any Primus code depending on these must be audited first.
- DiT `conditions_embeddings` (`1daa19f89`) — landed and later reverted (`cc4cb0119`); not currently in upstream `main`, do not build on it yet.

### Delivery
- Report is committed under `docs/weekly_reports/` and delivered via a pull request opened by the release-analysis automation.
