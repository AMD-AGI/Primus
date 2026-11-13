## Hard Requirements (MUST be in every agent's prompt)

Optimize the end-to-end training step time of MiniMax-M3.1-Flash on this machine. Shorter steady-state ms/iter is better. The only score that counts is `bash eat/eval.sh` from the repository root. It prints `SCORE: <x>` where `x = 1000 / steady_ms`. Correctness failures, incomplete runs, NaNs, and GPU collisions score 0.

The tree already has the bit-exact step-time work from the first run: recompute off, bf16 grad buffer with direct HIP Adam reads on this 1-rank 1-micro-batch shape, Turbo RMSNorm with TE linear, and fused Triton RoPE for MSA. Keep those. Do not revert them, and do not turn on PRIMUS_FUSED_ADAM_PARAM_REMAINDERS, bf16 Adam moments, or any other optimizer representation that changes the bf16 parameter update. That remainder path was measured faster but moved iteration-6 lm loss from 12.28070 to 12.28152, so it is out of scope. The official eval scores 0 when iteration-6 lm loss differs from 12.28070 by more than 1e-4.

Work on the MSA operator and fusing the ops around it. The gfx1250 kernels are `primus/backends/megatron/core/transformer/minimax_m3/flydsl/msa_token_fwd_w32.py`, `msa_token_bwd_w32.py`, and the indexer next to them. A profile of the previous tree put about 9 ms in those MSA kernels, with the rest of the step in hipBLASLt GEMMs, grouped GEMM, and the Adam kernel. A change that is not bitwise must still land inside the loss band above, and a microbench against eager or the current kernel should be in the notes.

Read first: `examples/megatron/models/minimax-m3/README.md` (section MiniMax-M3.1-Flash and the single-GPU notes) and `examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh`. The model preset is `primus/configs/models/megatron/minimax_m3.1_flash.yaml`.

The official workload is fixed inside `eat/eval.sh` and must not be weakened: MiniMax-M3.1-Flash, 1 GPU, 4 layers, MBS 1, GBS 1, sequence 4096, mock data, `TRAIN_ITERS=6`, FlyDSL MSA, the launcher's other defaults. Steady-state time is the logged average after `log_avg_skip_iterations=2` (the number after the slash on the last iteration). Do not special-case this eval, its iteration count, shapes, or log parser. A speedup has to come from the training code path this command actually runs.

This machine has one GPU, index 0. Every GPU command, including experiments and the official eval, must be wrapped by `gpulock` so it queues with everyone else:

```bash
gpulock run -g 0 -n "<what this run is>" -- <command>
```

`HIP_VISIBLE_DEVICES` must be 0. Do not use any other GPU, do not kill or reset another user's process, and do not switch cards when the lock is busy. `gpulock` waits. A result whose output contains `INTRUDERS` or `is on a GPU this run locked` is invalid. Do not hold the GPU across more than one training run: lock per run, then release. Two runs at once need different `MASTER_PORT`s. The official eval already picks one.

Only one agent may use the GPU at a time. The supervisor must put GPU-using roles in separate phases so they do not run together. Do not start a second `gpulock run` until your previous one has exited. A local lock already enforces this for notes that start with `eat`: a second EAT run waits before it even joins the `gpulock` queue, so do not work around that lock and do not stack jobs.

You may change Primus code, the MiniMax launcher, and kernels used by this workload. Do not modify `eat/eval.sh`, `eat/task.md`, `eat/env.sh`, or `eat/bin/`. Do not commit `output/`, secrets, or `~/llm_api`. Do not push to any git remote. Do not change submodule URLs. Prefer not to move submodule pointers unless the speedup cannot be done in Primus itself.

## Shared Environment

All agents share this working tree and the one GPU. Each agent edits only its own copies (for example `work_<role>/`) until it has a result that beats the best score already recorded in `eat/log/RESULTS.md`. Only that agent copies the change into the tree, keeps a snapshot, and appends one line to `eat/log/RESULTS.md` with the score, steady ms/iter, and what changed. Do not revert someone else's better result. Launch at most one training process at a time.
