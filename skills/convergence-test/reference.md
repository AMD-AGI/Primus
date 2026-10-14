# Convergence test reference

## How the planner decides

- **Model**: any spelling is reduced to a key (`Meta-Llama-3.1-8B`, `llama3.1_8B`
  and `llama 3.1 8b` all match), then looked up in the bundled recipes
  (`tools/convergence_test/configs/<backend>/`) and in the example configs
  (`examples/<backend>/configs/<GPU>/`). No match: it offers the closest names.
- **Backend**: `--backend`, else the image name (`jax`/`maxtext` -> MaxText;
  `primus`/`torch`/`megatron` -> Megatron), else what is installed in a local
  image, else the only backend that has the model; otherwise it asks.
- **Config**: a bundled recipe for the model and precision when there is one
  (it has reference results), else one generated from the example config for the
  detected GPU (`--from-example` forces generation). A generated config is probed
  before the full run.
- **Precision**: fp8 is `FP8` on Megatron and `nanoo_fp8` on MaxText for
  MI300X/MI325X (`fp8` on MI355X).
- **Tokenizer**: the model preset's. When it is gated and no Hugging Face token
  is set, an ungated copy with the same vocabulary is used (for example
  `NousResearch/Llama-2-7b-hf`); no token is needed if the dataset was already
  built.
- **GPU**: detected with `amd-smi` (or `rocm-smi`): the model picks the example
  config directory and the fp8 flavour, the count is checked against
  `GPUS_PER_NODE` (8). Not detected: a warning, and `--gpu` sets it. On MI300X
  (192 GB) bundled recipes, sized on 256 GB GPUs, probe first.
- **Image**: the default is `container.options.image` in `runner/.primus.yaml`
  for Megatron and `MAXTEXT_DEFAULT_IMAGE` in `run_convergence_test.sh` for
  MaxText.

## What a generated config changes

`tools/convergence_test/make_config.py` writes
`output/convergence/configs/<backend>/<model>-<precision>-convergence.yaml`,
which `extends:` the example (so it keeps the example's model, parallelism,
precision and kernels) and overrides, each group commented in the file:

- 1000 iterations x 128 sequences x 4096 tokens; cosine schedule with 10% warmup,
  peak lr 3e-4 up to 3B parameters, 1.5e-4 up to 20B, 1e-4 above
- real data from `prepare_dataset.py`, validation every 10% of the run
- logging the loss can be read from; no checkpoints
- Megatron: `micro_batch_size` kept if the batch divides, `init_method_std`
  0.008 for hidden size >= 4096 when the example uses 0.02
- MaxText: `per_device_batch_size` 16, or the example's with gradient
  accumulation; `weight_dtype: float32`
- MoE: real routing (`moe_router_force_load_balancing: false`), an auxiliary
  balancing loss, Megatron `micro_batch_size` 1 with recompute and the DeepEP /
  sync-free fast paths off; MaxText `load_balance_loss_weight` 0.01 and
  `capacity_factor` >= 1.25

## When a run fails

| Symptom in `jobs.py status` | Cause | Fix |
| --- | --- | --- |
| `lint failed` with `ERROR` lines | the config would not train correctly | follow the `->` hint; for a generated config edit the file and start again |
| `out of memory`, `RESOURCE_EXHAUSTED`, HIP OOM | the batch or activations do not fit | generated Megatron config: halve `micro_batch_size`, add `recompute_granularity: full`; generated MaxText: halve `per_device_batch_size` and double `gradient_accumulation_steps`, or `remat_policy: full`; re-probe |
| Inf/NaN gradients in the first iterations of a Megatron MoE model | Turbo MoE fast paths under real routing | `use_turbo_deepep: false`, `turbo_sync_free_moe_stage: 0` (generated configs already do this) |
| MaxText exits 4 after zero steps, `Training stopped` | grain worker processes without parsed absl flags (Primus branches before the fix) | append `-- --grain_worker_count 0 --grain_worker_count_eval 0` |
| could not load tokenizer / 401 / gated | gated tokenizer, no token | `export HF_TOKEN=...` or `--tokenizer <ungated copy>` |
| `initial vs ln(V)` SUSPICIOUS | tokenizer and data disagree, or a large init | check `tokenizer` in the plan; re-build with `--force` in `prepare_dataset.py` |
| `EADDRINUSE`, container already running | a previous container | `jobs.py stop <job>`, or `docker rm -f $(docker ps -q -f name=primus-training)` |
| `the node is busy` from `jobs.py start` (exit 2) | another convergence job, or a process or container it names | ask the user; stop only a convergence job, and only if they say so; never touch anything else |
| exit 3 | the loss moved from `--baseline` by more than `--tolerance` | report both curves; a probe is too short to gate |

## Exit codes

A run (the `state` line of `jobs.py status`): 0 PASS (every iteration trained;
baseline matched if given), 3 baseline FAIL, 4 stopped early, 6 the probe had
nan/skipped iterations so the full run was not started, anything else a lint
error, a crash, or an unreadable baseline.

`jobs.py start`: 0 started, 1 failed at once (status printed), 2 node busy.

## Another Primus branch

The driver trains the checkout it is in. For a branch, tag or commit:

```bash
git worktree add --detach ../primus-<name> <ref>
git -C ../primus-<name> submodule update --init third_party/Megatron-LM   # or third_party/maxtext
# if that ref has no tools/convergence_test, copy this checkout's
cp -r tools/convergence_test ../primus-<name>/tools/
export DATA_PATH=$PWD/data        # reuse the datasets built here
cd ../primus-<name> && python3 tools/convergence_test/plan_request.py ...   # then jobs.py there
```

Jobs, plans and results then live in that worktree's `output/convergence/`.
MaxText on refs without the grain fix needs the `grain_worker_count 0` overrides above.

## Examples

"run convergence test for llama2 7b fp8 using c4, image rocm/primus:v26.7"

```bash
python3 tools/convergence_test/plan_request.py --model "llama2 7b" --precision fp8 \
    --source c4 --image rocm/primus:v26.7
python3 tools/convergence_test/jobs.py start --plan <plan_file>
```

"can you check qwen3 30b a3b still converges on the new jax image
unifiedtrainingdockers.azurecr.io/utd/ci:jax_ci_92ca3a6_20260925_maxdiffusion, 2 hours max"

```bash
python3 tools/convergence_test/plan_request.py --model "qwen3 30b a3b" --hours 2 \
    --image unifiedtrainingdockers.azurecr.io/utd/ci:jax_ci_92ca3a6_20260925_maxdiffusion
python3 tools/convergence_test/jobs.py start --plan <plan_file>
# one job: the probe, then the full run sized to 2 hours
```

"run llama2 7b fp8 on c4, then qwen3 8b" (several models, one after another)

```bash
python3 tools/convergence_test/plan_request.py --model "llama2 7b" --precision fp8 --source c4   # plan A
python3 tools/convergence_test/plan_request.py --model "qwen3 8b"                                # plan B
python3 tools/convergence_test/jobs.py start --plan <plan A> --queue   # starts now if the node is free
python3 tools/convergence_test/jobs.py start --plan <plan B> --queue   # starts when A is done
```

Each job is independent: B starts when A ends, whether A passed or not, and
neither needs the session to stay open. `jobs.py list` shows both;
`jobs.py stop <job>` removes a queued one. Plan every model before starting
any, so questions come up front.

"how is the convergence test going?" -> `jobs.py status`

"compare it with last month's run" ->
`plot_loss.py <new>.csv <old>.csv --labels new old --out output/convergence/compare`
