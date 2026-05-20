---
name: run-and-profile
description: Subagent contract for running one Primus training job and collecting its profile. Use whenever the main agent (driving `tuning-loop`) needs to actually launch training — baseline run, candidate evaluation, env sweep — on the cluster. Every "submit + observe + summarize" must go through a subagent following this skill, so the main session never absorbs raw profile / log content. Triggers include: submit training, run a candidate plan, collect profile, measure throughput, run an env sweep.
---

# Run-and-Profile — The Subagent Boundary

This is the **only** skill in Pilot that defines an interface, because it is the only place that absorbs heavy artifacts (logs, profiler traces). The convention is rigid: every training run lives inside its own subagent; the subagent returns one short markdown block to the parent; raw artifacts stay on disk under `output/pilot/runs/<run_id>/`.

The main agent (running `tuning-loop`) calls a subagent with a `plan` + `cluster` + `purpose`. The subagent does the actual `bash …/run_pretrain_cli.sh` invocation, polls until done, parses metrics + profiler trace, writes artifacts to disk, and returns the markdown summary defined below. The main agent reads only that markdown.

## Subagent Input Contract

The parent passes this YAML-shaped input as part of the subagent prompt. Required fields are marked. `env_diff` is **only** the diff against the cluster's env baseline (do not re-state the baseline).

```yaml
plan:
  parallelism:
    tp: <int>             # required
    pp: <int>             # required
    dp: <int>             # required
    ep: <int>             # required for MoE, default 1
    vpp: <int>            # default 1
    cp: <int>             # default 1
  runtime:
    mbs: <int>            # required (micro batch size)
    gbs: <int>            # required (global batch size)
    recompute: full | selective | none   # required
    seq_len: <int>        # optional override
  comm:
    bucket_size_mb: <int> # optional
    overlap: true | false # optional
  env_diff:               # only diff vs cluster baseline; empty {} is valid
    NCCL_BUFFSIZE: 16777216
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
    # ...

cluster:
  mode: single | slurm    # required
  nodes: <int>            # required
  gpus_per_node: <int>    # required
  partition: <str>        # slurm only
  nodelist: <str>         # slurm only, optional, e.g. "node[01-04]"
  image: <str>            # docker / podman image, optional, default rocm/primus:v26.2

config:
  exp_yaml: <path>        # required, e.g. examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

purpose: baseline | candidate-eval | env-sweep | preflight   # required
max_steps: <int>          # required; baseline=200, candidate-eval=50-100, env-sweep=30-50
profile: true | false     # default true
parent_run_id: <str>      # optional, the champion this candidate derived from
notes: <str>              # optional, one-line intent like "vpp 1->2 to lower bubble"
```

## What the Subagent Does (Workflow)

### Step 1: Materialize the run directory

```bash
RUN_ID="r$(date +%Y%m%d_%H%M%S)_${RANDOM}"
RUN_DIR="output/pilot/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
```

Persist the input YAML to `${RUN_DIR}/plan.yaml` so the run is reproducible.

### Step 2: Translate `plan` into Primus CLI overrides

Primus runs are launched via a wrapper script. Use `examples/run_local_pretrain_cli.sh` for `mode=single` and `examples/run_slurm_pretrain_cli.sh` for `mode=slurm`. Both accept:

- `EXP=<path>` — the `config.exp_yaml`
- Trailing `--key value` overrides forwarded to `train pretrain --config $EXP $@`

Translation table:

| Plan field | Primus override flag |
|---|---|
| `parallelism.tp` | `--tensor_model_parallel_size <tp>` |
| `parallelism.pp` | `--pipeline_model_parallel_size <pp>` |
| `parallelism.ep` | `--expert_model_parallel_size <ep>` |
| `parallelism.vpp` | `--virtual_pipeline_model_parallel_size <vpp>` |
| `parallelism.cp` | `--context_parallel_size <cp>` |
| `runtime.mbs` | `--micro_batch_size <mbs>` |
| `runtime.gbs` | `--global_batch_size <gbs>` |
| `runtime.recompute` | `--recompute_granularity {full,selective}` (+ `--recompute_method block --recompute_num_layers <N>` for full) |
| `runtime.seq_len` | `--seq_length <n>` |
| `max_steps` | `--train_iters <max_steps>` |
| `comm.bucket_size_mb` | `--ddp_bucket_size <bytes>` (×1024×1024) |
| `comm.overlap` | `--overlap_grad_reduce True --overlap_param_gather True` |
| `profile=true` | `--profile True --use_pytorch_profiler True --profile_step_start <max_steps-3> --profile_step_end <max_steps-1>` |

Backend-specific names follow the YAML template under `examples/megatron/configs/...` or `examples/torchtitan/configs/...`. When in doubt, `Read` the chosen `exp_yaml` once to confirm the override key names; never invent flag names.

### Step 3: Export `env_diff`

Export every flag in `plan.env_diff` before invoking the wrapper, e.g.:

```bash
export NCCL_BUFFSIZE=16777216
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
```

Do not export the cluster baseline env again — the wrapper / preset already sources it.

### Step 4: Launch

For `mode=single`:

```bash
EXP="${exp_yaml}" \
bash examples/run_local_pretrain_cli.sh \
  --tensor_model_parallel_size ${tp} \
  --pipeline_model_parallel_size ${pp} \
  ... \
  > "${RUN_DIR}/log.txt" 2>&1
```

For `mode=slurm`, use `examples/run_slurm_pretrain_cli.sh` with `NNODES=${nodes}` and `NODES_LIST=${nodelist}` exported. If the user has a working SLURM allocation already, prefer attaching via `--jobid` (see `.cursor/skills/slurm-xiaoming-dev-container`).

### Step 5: Poll, early-stop, timeout

| Condition | Action |
|---|---|
| `iteration <max_steps>` printed | Run completed; proceed to Step 6 |
| `OutOfMemoryError` / `HIP error: out of memory` in log | Mark `status: oom`, kill job, skip profile parse |
| `RuntimeError` / `Exited with exit code 1` and no `iteration 3` yet | Mark `status: failed` with first matched error |
| `NCCL.*timeout` / `ncclSystemError` / `IB.*HCA` | Mark `status: hang` |
| Wallclock > `max(15min, 3 × estimated_step_ms × max_steps)` | Mark `status: hang`, kill (`scancel` for slurm) |
| `loss=nan` or `loss=inf` after step 5 | Mark `status: numerical`, kill |

For SLURM jobs, capture the SLURM job_id from the log (`Submitted batch job <id>`) so the subagent can `scancel` on timeout.

### Step 6: Parse metrics

Pull these from the last 20% of training-log iteration lines (skip warmup):

| Metric | Source |
|---|---|
| `tps` (tokens/sec/GPU) | Megatron prints `tokens/s/GPU` per iteration; average post-warmup |
| `step_time_ms` | average iteration time post-warmup |
| `mem_peak_gb` | `max_mem_allocated` from log, or `rocm-smi --showmemuse` snapshot if unavailable |
| `loss` | last logged loss (sanity only) |

If `profile=true` and the profiler trace exists at `${RUN_DIR}/profile/*.json` (or `*.pt.trace.json` PyTorch profiler output), compute:

| Metric | Approximation |
|---|---|
| `comm_ratio` | sum(`nccl*` / `rccl*` / `all_reduce` / `all_to_all` / `all_gather` / `reduce_scatter` kernel time) / step time |
| `bubble_ratio` | sum of pipeline-stage idle gaps / step time (PP > 1 only; else 0) |
| `overlap_ratio` | overlapped(comm, comp) time / total comm time |
| `gpu_util_avg` | mean GPU SM utilization across ranks if available; else from `rocm-smi` polled during run |

If profiler data is unavailable, set the corresponding fields to `n/a` rather than guessing.

### Step 7: Pick a bottleneck hint

Apply the first matching rule (these are coarse — the parent will re-run `bottleneck-diagnose` for the final classification):

| Condition | Hint |
|---|---|
| `comm_ratio > 0.25` | `COMM_BOUND` |
| `bubble_ratio > 0.15` | `PIPELINE_BOUND` |
| `mem_peak_gb / hbm_capacity_gb > 0.92` or status was OOM at higher mbs | `MEMORY_BOUND` |
| `gpu_util_avg < 0.65` and the above are all low | `COMPUTE_BOUND` |
| MoE model and `alltoall` time > 15% of step | `MOE_DISPATCH_BOUND` |
| Multiple thresholds tripped | `MIXED` |
| Profiler unavailable | `UNKNOWN` |

### Step 8: Write artifacts and snapshot.yaml

Persist these under `${RUN_DIR}/`:

```
output/pilot/runs/<run_id>/
├── plan.yaml          # the input plan
├── log.txt            # full stdout/stderr
├── profile/           # raw profiler trace (kept for parent on-demand Read)
├── snapshot.yaml      # parsed metrics (the "machine-readable summary")
└── result.md          # the markdown summary returned to the parent
```

`snapshot.yaml` shape:

```yaml
run_id: <id>
plan_id: <id>
purpose: <baseline|...>
status: completed | early_stopped | oom | hang | failed | numerical
metrics:
  tps: <float|null>
  step_time_ms: <float|null>
  comm_ratio: <float|null>
  bubble_ratio: <float|null>
  overlap_ratio: <float|null>
  mem_peak_gb: <float|null>
  gpu_util_avg: <float|null>
  loss: <float|null>
warnings: ["slow node smc04", ...]
collected_at: <iso8601>
```

### Step 9: Return the markdown summary to the parent

This is the **entire** payload the parent agent sees. Keep it under ~200 words. Do NOT paste log lines, NCCL traces, or profiler JSON.

```markdown
## Run Result (run_id: r20260515_1142_8842)

- **plan**: tp=2 pp=4 ep=8 mbs=1 vpp=2 recompute=selective, env_diff={NCCL_BUFFSIZE:16M, NCCL_MIN_NCHANNELS:16}
- **purpose**: candidate-eval (parent: r20260515_1031_8801)
- **status**: completed
- **metrics**: tps=17600, step_time_ms=412, comm_ratio=0.18, bubble_ratio=0.09, overlap_ratio=0.61, mem_peak_gb=158, gpu_util_avg=0.74
- **bottleneck hint**: COMPUTE_BOUND
- **one-line summary**: vpp 1→2 closed the bubble (0.18→0.09), step ~5% faster; gpu_util headroom suggests mbs scaling next.
- **artifacts**:
  - profile: output/pilot/runs/r20260515_1142_8842/profile/
  - log: output/pilot/runs/r20260515_1142_8842/log.txt
  - snapshot.yaml: output/pilot/runs/r20260515_1142_8842/snapshot.yaml
```

The `bottleneck hint` is exactly one of: `COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND | COMPUTE_BOUND | MOE_DISPATCH_BOUND | MIXED | UNKNOWN`.

The `status` is exactly one of: `completed | early_stopped | oom | hang | failed | numerical`.

## Important Notes

- **No raw artifacts in chat**. The parent reads `result.md`. If the parent decides it needs profiler detail, it `Read`s the snapshot.yaml or the trace file by path — do not pre-emptively dump.
- **One run per subagent**. If the parent wants to test 3 candidates in parallel, it spawns 3 subagents. This isolates context and gives natural parallelism (especially useful for env sweeps).
- **`env_diff` is a diff, not a full env**. When the parent wants to roll back an env tweak, it sends `env_diff: {NCCL_BUFFSIZE: <baseline_value>}` or omits the key. Listing the cluster baseline in `env_diff` pollutes the audit trail.
- **Failure is a normal return**, not an exception. `status: oom / hang / failed / numerical` is part of the contract; the subagent still produces `result.md` so the parent's ledger has a record. Only escalate to the user when the failure is unrecoverable (cluster-level breakage).
- **Honour `purpose: env-sweep`**: typically `max_steps ≤ 50`, several runs in parallel from one parent, each with one tiny `env_diff`. Always lock structure (no parallelism / mbs / recompute changes).
- **Wrapper script names are fixed**: `examples/run_local_pretrain_cli.sh` and `examples/run_slurm_pretrain_cli.sh`. If a backend-specific wrapper exists (e.g. `examples/torchtitan/...`), use that; never reinvent the launch command.
- **Cluster prerequisites are not Pilot's job**. If `mode=slurm`, the SLURM allocation must already exist; if `mode=single`, the container must already be running. Pilot does not `salloc` / `docker run`. Surface a clear error to the parent if the prerequisite is missing.
