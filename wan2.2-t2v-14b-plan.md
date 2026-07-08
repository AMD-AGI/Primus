# Wan2.2-T2V-A14B Training Support Plan

## Goal

Add Wan2.2-T2V-A14B training support on top of the Primus diffusion backend introduced by PR #779.

The implementation should stay incremental and should validate that the current diffusion backend can be extended to a larger Wan variant without adding a separate training stack.

## Base Branch Plan

- Current base PR: https://github.com/AMD-AGI/Primus/pull/779 (`dev/zirui/wan`).
- Draft A14B PR: https://github.com/AMD-AGI/Primus/pull/861 (`dev/zirui/wan-A2.2-A14B`).
- Keep the A14B PR based on `dev/zirui/wan` while #779 is open so the diff remains small.
- After #779 lands in `main`, rebase or retarget the A14B branch to `main` before final review.

## Proposed Scope

### In Scope

- Add model presets for Wan2.2-T2V-A14B.
- Add high-noise and low-noise SFT stage presets.
- Add MI355X example configs for both stages.
- Align stage behavior with Nvidia/NeMo AutoModel where practical:
  - `high_noise`: train Diffusers `transformer` on sigma range `[boundary_ratio, 1.0]`.
  - `low_noise`: train Diffusers `transformer_2` on sigma range `[0.0, boundary_ratio]`.
  - Default `boundary_ratio=0.875`.
  - Default flow shift `3.0`.
- Reuse existing Primus diffusion backend components:
  - `WanArgBuilder`.
  - `WanForTraining`.
  - `WanDiT`.
  - `FlowMatchScheduler`.
  - FSDP2 trainer.
  - Existing Wan dataset and processor.
- Add unit tests around config mapping and A14B-specific checkpoint key handling.

### Out of Scope for the First Formal PR

- Full Diffusers pipeline loader support.
- Native Diffusers VAE loading.
- Native Diffusers UMT5/text encoder loading.
- Automatic two-stage orchestration in a single Primus run.
- Generation/inference support for combining high-noise and low-noise trained checkpoints.
- Performance tuning or benchmark parity claims before E2E smoke tests pass.

## Checkpoint Format Decision

Wan2.2-T2V-A14B is published as a Diffusers pipeline with:

- `model_index.json`
- `transformer/`
- `transformer_2/`
- `vae/`
- `text_encoder/`
- `tokenizer/`

The current Primus diffusion backend uses native Wan-style assets for frozen encoders:

- DiT `.safetensors` or `.bin` files.
- UMT5 encoder `.pth` file.
- Wan VAE `.pth` file.
- Tokenizer directory.

For the initial A14B PR, keep the checkpoint-format scope intentionally narrow:

- Support loading the A14B DiT from the Diffusers `transformer` or `transformer_2` subfolder.
- Do not require full Diffusers pipeline loading.
- Continue using native `.pth` UMT5/VAE assets for the frozen text encoder and VAE.
- Default the A14B presets to compatible Wan2.1-14B encoder assets:
  - `/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth`
  - `/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth`
- Use the A14B Diffusers tokenizer subfolder:
  - `/models/Wan2.2-T2V-A14B-Diffusers/tokenizer`

Leave the broader checkpoint format decision as a follow-up TODO:

- Option A: keep Primus training on native Wan-style `.pth` encoder assets and only add a lightweight Diffusers DiT key converter.
- Option B: add full Diffusers component loading for DiT, VAE, text encoder, and tokenizer.
- Option C: provide an offline conversion tool from Diffusers layout to Primus native Wan layout.

The first formal PR should explicitly document whichever temporary behavior is used.

## Current Draft Implementation Summary

Draft PR #861 currently adds:

- `primus/configs/models/diffusion/wan2.2_t2v_a14b.yaml`
- `primus/configs/models/diffusion/wan2.2_t2v_a14b_high_noise_sft.yaml`
- `primus/configs/models/diffusion/wan2.2_t2v_a14b_low_noise_sft.yaml`
- `examples/diffusion/configs/MI355X/wan2.2_t2v_a14b-posttrain-high-noise.yaml`
- `examples/diffusion/configs/MI355X/wan2.2_t2v_a14b-posttrain-low-noise.yaml`
- Diffusers Wan DiT key remapping in `primus/backends/diffusion/models/registrations/wan.py`.
- Scheduler override plumbing in `WanArgBuilder`.
- README documentation for the A14B two-stage setup.
- Focused unit tests for scheduler mapping, Diffusers subfolder selection, and key remapping.

The key-level smoke check passed locally:

- A14B Diffusers `transformer` state dict keys: 1095.
- Meta-initialized Primus A14B `WanDiT` state dict keys: 1095.
- Missing keys after remap: 0.
- Unexpected keys after remap: 0.

Prepare-hook validation passed locally for both high-noise and low-noise example configs using:

- `/mnt/shared/zirui/models/Wan2.2-T2V-A14B-Diffusers`
- `/mnt/shared/zirui/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth`
- `/mnt/shared/zirui/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth`
- `/mnt/shared/zirui/data/tiny-video-samples`

Execution update on 2026-07-07:

- Fixed trainer-side `sigma_max` plumbing in the A14B worktree.
- Added a focused unit test for trainer scheduler `sigma_min`/`sigma_max` propagation.
- Ran focused Wan scheduler/builder tests: 9 passed.
- Ran full diffusion backend unit tests: 12 passed.
- Added `/zirui/code/Primus-wan-A2.2-A14B/local_runs/run.sh` as a local high/low/both E2E smoke launcher.
- Added rank-0 trainer logging for the resolved `FlowMatchScheduler` sigma range.
- Re-ran `python -m pytest tests/unit_tests/backends/diffusion`: 12 passed.
- Ran `PREPARE_ONLY=1 bash local_runs/run.sh both`; both high-noise and low-noise prepare hooks passed with `/models` and `/data` inputs.

Execution update on 2026-07-08:

- Ran a real single-node MI355X 8-GPU high-noise A14B training smoke on `chi2761`.
- Command used the A14B worktree, `GPUS_PER_NODE=8`, `SP_SIZE=1`, `MAX_STEPS=1`, `FRAME_NUM=1`, `HEIGHT=128`, `WIDTH=128`, `ATTENTION_BACKEND=flash_attn_aiter`, `VIDEO_BACKEND=decord`, and `BLAS_BACKEND=hipblas`.
- Data path on `chi2761`: `/zirui/data/tiny-video-samples/meta.jsonl` and `/zirui/data/tiny-video-samples/data`.
- Full A14B DiT load passed on all ranks with `files=12 missing=0 unexpected=0`.
- FSDP2 initialized with `DeviceMesh created: dims={'dp_shard': 8}`, wrapped 40 DiT blocks, and moved the sharded model to GPU after sharding.
- Training completed one forward/backward/optimizer step: `step=1 loss=0.5935`, `gnorm=19.2500`, `step_time=146.22s`.
- Final DiT save succeeded: `output/wan2.2_t2v_a14b-posttrain-high-noise/dit_model.safetensors` (about 27G).
- Log file: `/zirui/code/Primus-wan-A2.2-A14B/e2e_logs/high-noise-8gpu-tiny-sp1-aiter-hipblas-chi2761.log`.
- Runtime completed cleanly with `Training completed.` and `Cleanup completed.`; GPUs were released afterward.

## Findings and Issues

### 1. The Current Approach Does Extend the Diffusion Backend

The draft implementation is based on the existing diffusion backend rather than a new backend:

- It reuses `WanArgBuilder` for Primus-style config normalization.
- It reuses the existing Wan model registration path.
- It reuses `WanDiT` rather than importing Diffusers' runtime model.
- It reuses `WanFlowMatchTrainPipeline` and `FlowMatchScheduler`.
- It reuses FSDP2 trainer and the current dataset path.

This is a good test of backend extensibility.

### 2. The Code Is Mostly Lightweight, But Key Remapping Needs Care

The main implementation is compact, but the Diffusers-to-Primus key remapping is string-replacement based.

This is acceptable for a Draft PR if it remains tightly scoped to `WanTransformer3DModel` and is covered by tests, but formal PR should:

- Name the helper clearly as Wan-specific, not generic Diffusers conversion.
- Keep full-key smoke tests or unit tests that protect the mapping.
- Fail loudly if missing/unexpected keys are non-zero for A14B loading.

### 3. `sigma_max` Trainer Plumbing Was Fixed

The A14B configs set stage-specific `sigma_min` and `sigma_max`. The trainer previously only passed these values to `FlowMatchScheduler` partially.

Original issue:

- `WanArgBuilder` maps `scheduler.sigma_max` into `trainer.args.flow_match_scheduler.sigma_max`.
- `BaseWanTrainer` currently passes `shift`, `sigma_min`, and `extra_one_step` into `FlowMatchScheduler`.
- `BaseWanTrainer` does not pass `sigma_max`.

Fix landed in the A14B worktree:

- `BaseWanTrainer` now passes `sigma_max=float(scheduler_cfg.get("sigma_max", 1.0))` into `FlowMatchScheduler`.
- Scheduler construction is covered by a focused trainer unit test.
- `python -m pytest tests/unit_tests/backends/diffusion` passes locally.

Remaining validation:

- Low-noise E2E smoke training still needs to verify that the stage loads `transformer_2` and samples from the intended sigma window.

### 4. PR #861 CI State Needs Cleanup

PR #861 had a CI `code-lint (3.12)` failure due to isort changing import order in `test_wan_model_registration.py`.

Local worktree currently has an unpushed commit:

- `25600096 fix isort issue`

Before using #861 as a formal PR:

- Push that commit, or squash it into the feature commit.
- Re-run CI and verify code-lint passes.

### 5. High-Noise E2E Training Smoke Passed

Completed local validation:

- Unit tests pass locally.
- `python -m pytest tests/unit_tests/backends/diffusion` passed locally after the scheduler fix.
- `pre-commit run --all-files` passed locally.
- A14B high-noise prepare hook passed.
- A14B low-noise prepare hook passed.
- Full A14B key-remap smoke check passed.
- Local E2E runner exists and prepare-only validation passes for both stages.
- High-noise real 8-GPU A14B smoke passed on `chi2761` for one training step.

Still not yet completed:

- Low-noise real 8-GPU smoke test.
- Multi-step high-noise or low-noise smoke test.
- Larger video shape validation beyond the tiny `1x128x128` smoke.
- Any loss comparison with AutoModel.

Until the low-noise stage also passes, claim only high-noise E2E smoke coverage, not full two-stage E2E support.

## Implementation Plan

### Phase 1: Stabilize the Minimal A14B Draft

1. Keep PR #861 as a Draft PR based on `dev/zirui/wan`.
2. Done: fix scheduler plumbing:
   - Pass `sigma_max` through `BaseWanTrainer` into `FlowMatchScheduler`.
   - Add a focused unit test.
3. Done: run full diffusion backend unit tests locally.
4. Push or squash the isort-only commit and the scheduler fix.
5. Keep checkpoint support scoped to DiT Diffusers subfolders only.
6. Update PR description to say explicitly:
   - Diffusers DiT loading is supported.
   - Full Diffusers VAE/T5 loading is not yet supported.
   - A14B presets currently use native Wan2.1-14B `.pth` encoder assets.
7. Add the 2026-07-08 `chi2761` high-noise 8-GPU smoke result to the PR notes.

### Phase 2: E2E Smoke Validation

AutoModel reference:

- `/zirui/code/Automodel/examples/diffusion/finetune/wan2_2_t2v_flow.yaml`
- `/zirui/code/Automodel/run_wan2.2-t2v-a14b-benchmark.sh`

AutoModel's A14B recipe assumes a single 8-GPU node for the benchmark path:

- `GPUS_PER_NODE=8`
- `fsdp.dp_size=8`
- `activation_checkpointing=true`
- `local_batch_size=1`
- `global_batch_size=8`
- `flow_shift=3.0`
- `boundary_ratio=0.875`
- One stage per run:
  - `high_noise`: active `transformer`, sigma range `[0.875, 1.0]`
  - `low_noise`: active `transformer_2`, sigma range `[0.0, 0.875]`

Primus should treat E2E as a single-node MI355X 8-GPU smoke first. A single MI355X GPU is not a good initial E2E target because `world_size=1` skips FSDP2 sharding in the current trainer; the 14B DiT is expected to need 8-way sharding plus activation checkpointing.

#### E2E Preconditions

Before launching training:

1. Run on the A14B branch/worktree with the `sigma_max` trainer fix applied.
2. Confirm local inputs exist:
   - `INIT_CHECKPOINT=/models/Wan2.2-T2V-A14B-Diffusers`
   - `TEXT_ENCODER=/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth`
   - `VAE_CHECKPOINT=/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth`
   - `TEXT_TOKENIZER=/models/Wan2.2-T2V-A14B-Diffusers/tokenizer`
   - `DATASET_PATH=/data/tiny-video-samples/meta.jsonl`
   - `DATA_FOLDER=/data/tiny-video-samples/data`
   - On `chi2761`, the tiny sample data path used for the passing high-noise smoke was `/zirui/data/tiny-video-samples`.
3. Use `GPUS_PER_NODE=8`.
4. Start with `MAX_STEPS=1` or `2`.
5. Start with `FRAME_NUM=81`, `HEIGHT=480`, `WIDTH=832`; test `FRAME_NUM=121` only after the shorter smoke passes.
6. Use `ATTENTION_BACKEND=flash_attn_aiter` first. If kernel/runtime issues block launch, retry with `ATTENTION_BACKEND=sdpa` only as a debugging fallback.
7. Use `SP_SIZE=8` for the first 14B smoke if memory is the concern. A14B has 40 attention heads, so `SP_SIZE=4` and `SP_SIZE=8` both divide the head count. `SP_SIZE=1` is useful as a simpler code path but is likely less memory-friendly.
8. Use `BLAS_BACKEND=hipblas` on ROCm if hipBLASLt reports `HIPBLAS_STATUS_INTERNAL_ERROR` during backward.

#### Local E2E Runner

`/zirui/code/Primus-wan-A2.2-A14B/local_runs/run.sh` now wraps the two-stage
smoke path and defaults to the mounted `/models` and `/data` prefixes:

```bash
cd /zirui/code/Primus-wan-A2.2-A14B
bash local_runs/run.sh both
```

Useful overrides:

- `STAGE=high` or `STAGE=low` to run a single stage.
- `MAX_STEPS=2` or `5` after the 1-step smoke passes.
- `RUN_PREPARE=1` to run the posttrain diffusion prepare hook before `torchrun`.
- `ATTENTION_BACKEND=sdpa` as a debugging fallback if `flash_attn_aiter` blocks launch.
- `BLAS_BACKEND=hipblas` to force PyTorch BLAS away from hipBLASLt on ROCm.

#### High-Noise Smoke Command

Passing `chi2761` tiny smoke command from 2026-07-08:

```bash
cd /zirui/code/Primus-wan-A2.2-A14B
mkdir -p e2e_logs
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 \
BLAS_BACKEND=hipblas \
STAGE=high GPUS_PER_NODE=8 SP_SIZE=1 \
VIDEO_BACKEND=decord \
DATASET_PATH=/zirui/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/zirui/data/tiny-video-samples/data \
MAX_STEPS=1 FRAME_NUM=1 HEIGHT=128 WIDTH=128 \
ATTENTION_BACKEND=flash_attn_aiter MASTER_PORT=29580 \
bash local_runs/run.sh high 2>&1 | tee e2e_logs/high-noise-8gpu-tiny-sp1-aiter-hipblas-chi2761.log
```

Expected pass markers from that run:

- `Init completed.`
- `Training started.`
- `step=1 loss=0.5935`
- `Saved DiT weights to ./output/wan2.2_t2v_a14b-posttrain-high-noise/dit_model.safetensors`
- `Training completed.`
- `Cleanup completed.`

Larger-shape high-noise smoke target:

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 GPUS_PER_NODE=8 \
INIT_CHECKPOINT=/models/Wan2.2-T2V-A14B-Diffusers \
TEXT_ENCODER=/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
VAE_CHECKPOINT=/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
TEXT_TOKENIZER=/models/Wan2.2-T2V-A14B-Diffusers/tokenizer \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=1 FRAME_NUM=81 HEIGHT=480 WIDTH=832 SP_SIZE=8 ATTENTION_BACKEND=flash_attn_aiter \
torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=8 \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_t2v_a14b-posttrain-high-noise.yaml
```

Expected high-noise behavior:

- Loads Diffusers `transformer`.
- Does not load/train `transformer_2`.
- Uses scheduler sigma range `[0.875, 1.0]`.
- Runs at least one forward/backward/optimizer step.
- Logs finite loss.

#### Low-Noise Smoke Command

Use the same environment and command shape, but switch the config:

```bash
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 GPUS_PER_NODE=8 \
INIT_CHECKPOINT=/models/Wan2.2-T2V-A14B-Diffusers \
TEXT_ENCODER=/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
VAE_CHECKPOINT=/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
TEXT_TOKENIZER=/models/Wan2.2-T2V-A14B-Diffusers/tokenizer \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=1 FRAME_NUM=81 HEIGHT=480 WIDTH=832 SP_SIZE=8 ATTENTION_BACKEND=flash_attn_aiter \
torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29501 --nproc_per_node=8 \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_t2v_a14b-posttrain-low-noise.yaml
```

Expected low-noise behavior:

- Loads Diffusers `transformer_2`.
- Does not load/train `transformer`.
- Uses scheduler sigma range `[0.0, 0.875]`.
- Runs at least one forward/backward/optimizer step.
- Logs finite loss.

#### E2E Acceptance Criteria

Minimum smoke pass:

- High-noise job launches on one 8-GPU MI355X node. Done on `chi2761` with the tiny shape.
- Low-noise job launches on one 8-GPU MI355X node.
- Both jobs load full A14B DiT weights with zero missing/unexpected keys after conversion. Done for high-noise.
- Both jobs complete at least one train step. Done for high-noise.
- Loss is finite for both jobs. Done for high-noise.
- Logs show the intended active transformer and sigma range. Done for high-noise.
- No OOM, distributed hang, or FSDP2 wrapping error. Done for high-noise.

Stronger validation after smoke:

- Increase `MAX_STEPS` to `2` or `5`.
- Retry with `FRAME_NUM=81`, then `FRAME_NUM=121`, to match AutoModel benchmark shape more closely.
- Compare Primus loss scale/trend against AutoModel on the same tiny-video samples, without claiming benchmark parity.
- Enable a small checkpoint save and verify the `dit_only` output can be reloaded.

#### Failure Triage Order

1. If launch fails before model load, check environment and dependency/kernel availability.
2. If model load fails, inspect Diffusers key remapping and missing/unexpected keys.
3. If OOM occurs, keep `SP_SIZE=8`, `FRAME_NUM=81`, `local_batch_size=1`, `MAX_STEPS=1`, and confirm gradient checkpointing is enabled.
4. If attention kernels fail, retry with `ATTENTION_BACKEND=sdpa` to separate kernel issues from model/trainer issues.
5. If ROCm GEMM fails with `HIPBLAS_STATUS_INTERNAL_ERROR`, retry with `BLAS_BACKEND=hipblas` to avoid hipBLASLt.
6. If distributed training hangs, retry with a fresh `MASTER_PORT`, verify `GPUS_PER_NODE=8`, and check whether all ranks reached FSDP2 wrapping.
7. If loss is NaN/Inf, use `FIXED_SEED` and `FIXED_TIMESTEP` for a reproducible debug run.

### Phase 3: Decide Checkpoint Format Direction

After minimal E2E passes, decide one of:

1. Keep hybrid support:
   - Diffusers DiT subfolder loading.
   - Native `.pth` frozen encoder assets.
2. Add full Diffusers loading:
   - Load Diffusers VAE.
   - Load Diffusers UMT5/text encoder.
   - Keep tokenizer from Diffusers path.
3. Add conversion tooling:
   - Convert A14B Diffusers checkpoint into native Primus/Wan layout.
   - Keep training runtime simple.

This decision should be made before marking the PR ready for final review.

## Formal PR Checklist

Before converting the A14B Draft PR to a formal PR:

- [ ] Rebase or retarget from `dev/zirui/wan` to `main` after PR #779 lands.
- [x] Fix `sigma_max` scheduler plumbing.
- [ ] Push/squash the isort fix.
- [ ] Ensure CI code-lint is green.
- [ ] Run `pre-commit run --all-files`.
- [x] Run `python -m pytest tests/unit_tests/backends/diffusion`.
- [x] Add or keep a unit test for scheduler `sigma_min/sigma_max` propagation.
- [x] Keep key-remap unit tests.
- [ ] Add a model-build smoke test if practical without loading full weights in CI.
- [x] Add a local A14B E2E smoke runner.
- [x] Run local high-noise prepare hook.
- [x] Run local low-noise prepare hook.
- [x] Run high-noise E2E smoke training.
- [ ] Run low-noise E2E smoke training.
- [ ] Document checkpoint format limitations clearly.
- [x] Document exact model/data environment variables for both stages.
- [ ] Avoid claiming benchmark/performance parity before benchmark runs exist.

## Open TODOs

- Decide whether A14B should support full Diffusers pipeline loading or remain hybrid.
- Decide whether to add a checkpoint conversion tool.
- Decide whether Primus should orchestrate high-noise and low-noise stages in one top-level config or keep them as separate explicit runs.
- Validate that using Wan2.1-14B VAE/T5 assets with Wan2.2-T2V-A14B DiT is acceptable for training quality, not just code execution.
- Compare loss behavior against AutoModel after E2E smoke succeeds.
- Add generation/inference support for combining two stage checkpoints if needed.
