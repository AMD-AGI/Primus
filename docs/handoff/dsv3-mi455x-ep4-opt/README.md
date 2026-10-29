# DeepSeek-V3 proxy on 4x MI455X: end-to-end optimisation handoff

Status: **paused 2026-10-29** (user decision; work moves to MiniMax-M3.1-Flash end to end). Every code
change below compiles and is pushed, but **none has been run on a GPU yet**: the queued validation
batches never got an idle GPU window. Resume by running the batches in [How to resume](#how-to-resume).

## Branches

| Repo | Branch | Base | Head |
|---|---|---|---|
| AMD-AGI/Primus | `dev/boyang/dsv3_opt` | `32d0d74a` | this doc |
| AMD-AGI/Primus-Turbo | `dev/boyang/dsv3_opt` | `73de5397` (PR #583 head, FlyDSL DeepEP V2) | `41655cf2` |

## Setup

- Config `examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml`, launcher
  `examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh`.
- Proxy: 4 MoE layers, hidden 7168, 128 experts, top-8, 1 shared expert (2048); EP4, TP1, MBS 2,
  GBS 64 (8 microbatches), seq 4096, BF16; FlyDSL attention and grouped GEMM; container
  `dev_mi455_deepep_yangbo` (image `amdprimus/amdprimus:tas-20261019`).
- Metric: tokens/s/GPU, harmonic mean over iterations >= 4 of 20; ms/iter = 65536 / (tokens/s/GPU).
- DeepEP: `--use_turbo_deepep True --turbo_deepep_num_cu 160`, `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl_v2`,
  `TURBO_DIR=<Primus-Turbo checkout>`; even routing (`--moe_router_force_load_balancing_type even`).

## Measured so far (2026-10-28, even routing)

| Case | tokens/s/GPU | ms/iter | Note |
|---|---:|---:|---|
| flydsl_v2, 160 CU (baseline) | 65231 | 1005 | 20 iterations |
| flydsl_v2 + sync-free stage 2 | 63238 | 1036 | only 3 iterations, not conclusive |
| flydsl (V1) + sync-free stage 3 | 61449 | 1066 | only 3 iterations |
| flydsl_v2 + sync-free stage 3 | NaN grads at iteration 1 | | root-caused and fixed, see below |

Profile of the baseline (rank 0, `scripts/analysis_v2_even_rank0.txt`, ~1085 ms/step under the profiler):

- GPU idle 113 ms/step: 68 ms in 100 us-1 ms gaps; MoEDispatch host syncs (~14 ms) are what sync-free removes.
- Routed-expert grouped GEMM ~260 ms (FlyDSL `kernel_grouped_nt` 162 ms, `kernel_grouped_variable_k` 98 ms).
- Dense hipBLASLt GEMMs ~270 ms; MLA attention ~133 ms (`k_dkdv64` 67, `k_dqg96` 42, `mla_fwd` 24).
- DeepEP kernels ~66 ms (combine_v2 25, notify 18 + cached_notify 12, dispatch_v2 11); notify time is mostly
  cross-rank wait / skew.
- bf16 `add_` 36 ms: main_grad accumulation (expert fc1 13 ms, fc2 6.5 ms, embedding / output 6.3 ms, dense ~7 ms),
  because `gradient_accumulation_fusion: false` in the yaml.
- HtoD memcpy 332 ms on the copy stream is batch data queued behind compute (harmless), except ~0.7 ms per
  microbatch not overlapped, mostly the 33 MB attention mask.

## Code changes

### Primus-Turbo

- `41655cf2` **flydsl_v2 combine fix.** Under a worst-case allocation (`num_worst_tokens`, sync-free stage >= 2)
  `combine_v2_kernel`'s pushers for the last source rank used `row_end = num_rows` = `x.size(0)` = the
  worst-case row count, so the tail rows were pushed over other ranks' combine slots with their flags raised.
  Timing dependent (gone with sync probes). Fix: push only `rank_prefix` rows. Test
  `tests/pytorch/deep_ep/test_intranode.py::test_combine_ignores_the_worst_case_tail` (NaN tail must not leak).
  Compile-only ISA check by another window: 16/16 builds, only combine_v2 changes, no register growth.
  Also cherry-picked onto the V2-unified branch.
- `b1abeff3` **FlyDSL gfx1250 variable-K grouped GEMM beta=1 epilogue** (`accumulate=True`: `out[g] += a.T @ b`),
  wired into `GroupedGEMMVariableKFlyDSLBackend` (`inplace_add_to_out` on gfx1250 for a contiguous same-dtype
  `out`). Lets `gradient_accumulation_fusion: true` keep the FlyDSL wgrad (grad buffer is bf16). Expected
  saving 12-15 ms/step. Compile-only (no GPU): fc1 [32,4096,7168] VGPR 774->848, fc2 [32,7168,2048] 904->908,
  no spills / scratch.
- `bc9fd6b3` count sync-free permute overflows on the GPU (no host sync).
- `554119d4` debug only: `PRIMUS_TURBO_DEBUG_NAN=1` device-side non-finite probes in the token dispatcher.

### Primus

- `008a3687` host-sync removal for the MoE step: `--turbo_sync_free_permute_factor` (permute buffer budget
  `tokens*topk*factor` instead of the worst case), `--get_batch_skip_tp1_broadcast`, NaN check once on the grad
  norm (`--check_for_nan_in_grad_norm`) instead of per bucket, sync-free overflow check.
- `fc305f29` **shared-expert overlap under the DeepEP dispatcher**: `PrimusTurboDeepEPTokenDispatcher` calls the
  shared expert's overlap hooks like Megatron's alltoall dispatcher (fc1 + act on the side stream from before
  dispatch, fc2 after combine, output added in `combine_postprocess`); the ROCm arg check rejecting
  `moe_shared_expert_overlap` with DeepEP is dropped and TransformerConfig's alltoall-only check is skipped
  for the flex dispatcher. Enable with `--moe_shared_expert_overlap True`.
- `2d80bf05` debug only: expert NaN probes + per-step report under `PRIMUS_TURBO_DEBUG_NAN=1`.

No-code option found: Turbo attention ignores `attention_mask` (only `attn_mask_type`), so
`--create_attention_mask_in_dataloader False` drops the 33 MB mask per microbatch.

## Queued validation (not run)

`scripts/` holds the scripts as used (absolute paths of the original machine: `/home/boyan/project/deepep`,
worktrees `Primus-dsv3opt`, `Primus-Turbo-dsv3opt`, logs in `ut_logs/e2e_dsv3opt_*`).

- `batch7.sh`: `unit_tests.sh` (new combine test on V1 and V2, V2 regressions, `bench_vk_accum.py` beta=1
  correctness and timing vs FlyDSL + add and Triton beta=1), then e2e `v2s3` (stage 3), `v2sf` (+ factor 1.05),
  `v2sfh` (+ host-sync flags).
- `batch8.sh`: `--gradient_accumulation_fusion True` on v2 and v2sfh, plus a profiled run (steps 6-8, all ranks);
  skipped unless the beta=1 bench prints ALL OK.
- `batch9.sh`: shared-expert overlap, `--turbo_deepep_use_comm_stream True`, attention mask off, combinations.

Each e2e case takes ~2 min. GPU rules used: start only after the 4 GPUs are idle 30 min (`can_start.sh`,
also restricted to 00:00-07:00 UTC+8), `gpu_guard.sh` kills our runs when another process appears.

## How to resume

1. Check out both branches (worktrees as above), start `scripts/gpu_guard.sh`, then run `batch7.sh`,
   `batch8.sh`, `batch9.sh` (each waits for the previous one).
2. Check: combine test passes, beta=1 bench ALL OK, v2s3 / v2sfh finite grad norm and loss parity with v2,
   ms/iter vs 1005.
3. Next candidates after that: profile the best config; V2 notify / combine skew; CU count of DeepEP with
   overlap on; embedding grad add; GEMM and MLA attention tuning; uniform routing.
