# diffusion_bench

End-to-end automation for the **AMDiffusionBenchmark** (FLUX.1-dev, Stable Diffusion XL, Mochi 1,
HunyuanVideo, Wan 2.1 I2V) and **DLRMBenchmark** (TF32 / FP32) workloads on AMD Instinct
**MI300** (MI300X / MI325X) and **MI355** (MI350X / MI355X) nodes.

One command starts the container, downloads the Hugging Face assets, runs every benchmark with the
batch size for the target GPU, saves all logs, and writes a parsed summary. What to run is defined
in a YAML config file.

```bash
export HF_TOKEN=hf_xxx
cd tools/perf/diffusion_bench       # from the Primus repo root
./run_benchmarks.sh                 # uses config.yaml in this folder
```

The perf batch runner can also run it as part of a batch, after the Primus models:
`BACKEND=...,diffusion_dlrm bash tools/perf/run_batch.sh` (see
[`tools/perf/README.md`](../README.md#diffusion-and-dlrm-diffusion_dlrm)). It uses this folder's
`config.yaml`, and passes the image, GPU type, output directory and a container label.

---

## Contents

1. [What it does](#what-it-does)
2. [Files](#files)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Configuration file](#configuration-file)
6. [Hugging Face cache and reusing downloaded data](#hugging-face-cache-and-reusing-downloaded-data)
7. [Command-line options](#command-line-options)
8. [Results and metrics](#results-and-metrics)
9. [Failure handling](#failure-handling)
10. [Examples](#examples)
11. [How it works](#how-it-works)
12. [Troubleshooting](#troubleshooting)

---

## What it does

The script automates the manual procedure below. It runs on the host; you don't need to enter the
container yourself.

| Step | Manual procedure | Automated by |
|---|---|---|
| 1 | `docker run ...` the benchmark image | `run_benchmarks.sh` starts a detached container |
| 2 | `cd /workspace/AMDiffusionBenchmark` | done inside each step |
| 3 | `sed` the Makefile to use `hf auth login` / `hf download` | `setup` step |
| 4 | `make download_assets` (and log in to HF) | `download` step, using `HF_TOKEN` |
| 5 | `python launcher.py train_args=<model> train_args.train_batch_size=<bs>` for each model | one `diffusion_<model>_bs<bs>` step per run |
| 6 | `cd /workspace/DLRMBenchmark` | done inside each step |
| 7 | `./launch_training_single_node.sh -p TF32` | `dlrm_TF32` step |
| 8 | `./launch_training_single_node.sh -p FP32` | `dlrm_FP32` step |
| 9 | Collect numbers from `runs_summary.csv` / `results.csv` | `parse_results.py` -> `summary.md/csv/json` |

Default batch sizes (per GPU), taken from the benchmark instructions:

| Benchmark | MI300 | MI355 |
|---|---:|---:|
| FLUX.1-dev (`flux-dev`) | 10 | 16 |
| Stable Diffusion XL (`stable-diffusion-xl`) | 20 | 12 |
| Mochi 1 (`mochi-1`) | 1 | 4 |
| HunyuanVideo (`hunyuan-video`) | 1 | 3 |
| Wan 2.1 I2V (`wan2_1-i2v`) | 1 | 1 |

DLRM uses the repository's `training_config.sh` (batch size 32768, 1000 training batches) unless you
override the batch size.

---

## Files

| File | Runs on | Purpose |
|---|---|---|
| `run_benchmarks.sh` | host | Entry point. Detects the GPU, starts/stops the container, runs every step with a timeout, records step status, calls the parser. |
| `config.yaml` | – | Default configuration: GPU, image, HF cache, models, batch sizes, train args, DLRM. |
| `config.py` | host | Loads and validates the YAML, resolves per-GPU values, applies CLI overrides, produces the run plan. |
| `in_container.sh` | container | The individual steps: `setup`, `download`, `diffusion`, `dlrm`, `cleanup`, `finalize`. |
| `parse_results.py` | host | Builds `summary.md`, `summary.csv`, `summary.json` from a results directory (Python standard library only). |

The folder is mounted read-only into the container at `/diffusion_bench`.

---

## Requirements

- A single node with 8 MI300-class or MI355-class GPUs (both benchmarks assume 8 GPUs).
- Docker, with your user in the `docker` group (`docker info` must work without sudo).
- The benchmark image available locally or pullable (default:
  `unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_0ce782a_20261002`; run `docker login` for
  private registries). The image must contain `/workspace/AMDiffusionBenchmark` and
  `/workspace/DLRMBenchmark`.
- `python3` with PyYAML on the host (`python3 -c "import yaml"`). Install with
  `pip install --user pyyaml` or `sudo apt install python3-yaml` if missing.
- A Hugging Face token with access to the gated models (FLUX.1-dev in particular: accept the license
  on its model page first), exported as `HF_TOKEN`. Not needed if the cache is already populated and
  `download: false`.
- Disk space for the HF cache: about **370 GB** for all diffusion models and datasets.

---

## Quick start

```bash
cd tools/perf/diffusion_bench       # from the Primus repo root

# 1. Check the plan without using the GPUs (starts the container, launcher.py in dry-run mode)
./run_benchmarks.sh --dry-run

# 2. Full run. It takes several hours, so run it in tmux/screen or with nohup:
export HF_TOKEN=hf_xxx
nohup ./run_benchmarks.sh > bench_$(date +%Y%m%d_%H%M).out 2>&1 &
tail -f bench_*.out

# 3. Read the results
cat results/<gpu>_<host>_<timestamp>/summary.md
```

The same command works on MI300 and MI355 nodes: with `gpu: auto` the GPU type is read from sysfs
(`gfx942` -> `mi300`, `gfx950` -> `mi355`) and the matching batch sizes are applied.

---

## Configuration file

`config.yaml` in this folder is used by default; pass another with `-c/--config FILE`. Keeping one
config per experiment (e.g. `configs/mi355_sweep.yaml`) makes runs reproducible, and every results
directory contains a copy of the config used (`config.yaml`) and the fully resolved plan
(`resolved_config.yaml`).

### Full reference

```yaml
gpu: auto                 # auto | mi300 | mi355
image: unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_0ce782a_20261002

hf_cache_dir: null        # null -> $HF_HOME or ~/.cache/huggingface (see the HF cache section)
hf_cache_readonly: false  # mount read-only; implies download: false and HF_HUB_OFFLINE=1
download: true            # run the download/verify step before the diffusion runs

output_dir: ./results     # relative paths are relative to the config file
step_timeout: 14400       # seconds per benchmark step, 0 = no timeout
download_timeout: 0       # seconds for the download step, 0 = no timeout
keep_container: false     # keep the container afterwards for debugging

diffusion:
  enabled: true
  train_args: {}          # train_args.<key>=<value> for every model
  hydra_overrides: []     # raw hydra overrides for every launcher.py call
  models:
    flux-dev:
      enabled: true
      batch_size: {mi300: 10, mi355: 16}
      train_args: {}      # per-model, wins over diffusion.train_args
      hydra_overrides: []
    stable-diffusion-xl:
      batch_size: {mi300: 20, mi355: 12}
    mochi-1:
      batch_size: {mi300: 1, mi355: 4}
    hunyuan-video:
      batch_size: {mi300: 1, mi355: 3}
    wan2_1-i2v:
      batch_size: {mi300: 1, mi355: 1}

dlrm:
  enabled: true
  image: null             # separate DLRM image, null = same as `image`
  precisions: [TF32, FP32]
  batch_size: null        # overrides BATCH_SIZE in training_config.sh, null = repo value
```

### Rules

- **Per-GPU values.** `batch_size`, any `train_args` value, and `dlrm.batch_size` can be a plain
  value (used on every GPU) or a mapping `{mi300: ..., mi355: ...}`. One config therefore serves
  both node types.
- **Batch-size sweeps.** A list runs one benchmark per value:
  `batch_size: {mi300: [8, 10, 12], mi355: [12, 16, 20]}`.
- **Selecting models.** A model listed under `models` runs unless it has `enabled: false`; a model
  missing from `models` does not run. `diffusion.enabled: false` disables all diffusion runs and
  `dlrm.enabled: false` disables DLRM.
- **train_args** become `train_args.<key>=<value>` hydra overrides for `launcher.py`, e.g.
  `num_iterations: 50` -> `train_args.num_iterations=50`. Any key accepted by the model config in
  `AMDiffusionBenchmark/config/train_args/` works (`num_iterations`, `resolution`, `num_frames`,
  `gradient_checkpointing`, ...). Keys not in those configs need the `+train_args.<key>=<value>`
  form via `hydra_overrides`.
- **hydra_overrides** are passed verbatim, e.g. `launcher.skip_larger_bs=false`.
- **Validation.** Unknown keys are rejected with a suggestion (`batch_sise` -> "did you mean
  `batch_size`?"), as are unknown models, invalid precisions, and non-positive batch sizes, before
  anything is started.
- **Paths** accept `~` and `$ENV_VARS`. Relative paths in the file are relative to the config file;
  relative paths on the command line are relative to the current directory.

---

## Hugging Face cache and reusing downloaded data

The host directory `hf_cache_dir` (or `--hf-cache DIR`) is bind-mounted into the container, so
downloads persist across runs and the second run doesn't download anything. **You can point it at
any existing cache to reuse data you already have.**

Two layouts are recognised automatically:

| You point to | Recognised by | Mounted at |
|---|---|---|
| An HF home directory, e.g. `~/.cache/huggingface` or `$HF_HOME` | contains a `hub/` subdirectory (or is empty/new) | `/root/.cache/huggingface` |
| A hub directory, e.g. `/shared/hf/hub` | named `hub`, or directly contains `models--*` / `datasets--*` | `/root/.cache/huggingface/hub` |

Default when `hf_cache_dir` is `null`: `$HF_HOME` if set, otherwise `~/.cache/huggingface`.

With `download: true` and a populated cache, the download step only checks the files against the
Hub (it takes a few minutes and re-downloads only what is missing or changed). To skip it entirely,
use `download: false` or `--skip-download`.

**Shared caches you don't own:** use `hf_cache_readonly: true` (or `--hf-cache-readonly`). The cache
is mounted read-only, the download step is skipped, and `HF_HUB_OFFLINE=1` is set so nothing tries to
write to it. All needed models and datasets must already be present.

**Ownership:** the container runs as root. When the run ends, only the root-owned files it created
in the cache are given to your user; files belonging to others are left untouched.

**Needed assets.** If all five models are selected, the download step runs
`make download_assets` exactly as in the manual procedure. If only some are selected, it downloads
only those models and their datasets, using `model_path` / `train_data_path` from each model's
config:

| Model | Hugging Face model | Dataset |
|---|---|---|
| flux-dev | `black-forest-labs/FLUX.1-dev` | `frank-chieng/chinese_architecture_siheyuan` |
| stable-diffusion-xl | `stabilityai/stable-diffusion-xl-base-1.0` | `frank-chieng/chinese_architecture_siheyuan` |
| mochi-1 | `genmo/mochi-1-preview` | `Wild-Heart/Disney-VideoGeneration-Dataset` |
| hunyuan-video | `hunyuanvideo-community/HunyuanVideo` | `Wild-Heart/Disney-VideoGeneration-Dataset` |
| wan2_1-i2v | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `Wild-Heart/Disney-VideoGeneration-Dataset` |

**Saving data from an existing container.** If you downloaded the assets inside a container started
with `--rm`, copy them out before it exits, then point `hf_cache_dir` at the copy:

```bash
docker cp <container>:/root/.cache/huggingface/hub ~/.cache/huggingface/
```

**Choosing the filesystem.** Models are read from the cache at the start of each run. A local disk is
faster than NFS, but a cache on shared NFS can be reused from several nodes (e.g. an MI300 node and an
MI355 node) without downloading twice.

---

## Command-line options

Command-line options override the config for a single run. `./run_benchmarks.sh --help` prints this
list.

| Option | Overrides |
|---|---|
| `-c, --config FILE` | config file (default: `config.yaml` next to the script) |
| `--gpu auto\|mi300\|mi355` | `gpu` |
| `--image IMAGE` | `image` |
| `--dlrm-image IMAGE` | `dlrm.image` |
| `--models LIST` | which diffusion models run (comma separated, or `none`); ignores `enabled` |
| `--bs MODEL=N[,N...]` | batch size(s) of one model; repeatable |
| `--num-iterations N` | `train_args.num_iterations` for all models |
| `--extra-train-args "..."` | extra raw hydra overrides for every run |
| `--dlrm-precisions LIST` | `dlrm.precisions` (or `none`) |
| `--dlrm-batch-size N` | `dlrm.batch_size` |
| `--hf-cache DIR` | `hf_cache_dir` |
| `--hf-cache-readonly` | `hf_cache_readonly: true` |
| `--skip-download` | `download: false` |
| `--output-dir DIR` | `output_dir` |
| `--timeout SEC` | `step_timeout` |
| `--keep-container` | `keep_container: true` |
| `--label KEY=VALUE` | adds a docker label to the containers; repeatable (the perf batch runner finds and stops its containers by label) |
| `--dry-run` | starts the container and runs `launcher.py` with `launcher.dry_run=true`; skips downloads and DLRM; uses no GPU time |
| `--parse-only RUN_DIR` | only regenerates the summary of an existing results directory |

Model aliases accepted on the command line: `flux`, `sdxl`, `mochi`, `hunyuan`, `wan`.

Environment variables: `HF_TOKEN` (required for downloads), `HF_HOME` (default cache location).

---

## Results and metrics

Each run creates `output_dir/<gpu>_<host>_<YYYYmmdd_HHMMSS>/`. The default `./results` folder is
git-ignored, so runs never show up as untracked files in the repo.

```
mi300_tus1-p3-g35_20261002_014509/
├── summary.md               # human-readable tables (also printed at the end of the run)
├── summary.csv              # one row per benchmark run
├── summary.json             # run info + step status + results
├── status.tsv               # step, status, exit code, duration, log path
├── run.log                  # complete console output of the run
├── run_info.txt             # GPU, host, image, HF cache, driver, date
├── config.yaml              # copy of the config used
├── resolved_config.yaml     # plan after per-GPU resolution and CLI overrides
├── env/                     # versions (ROCm, torch, repo commits), pip freeze, image id, repo git status
├── logs/                    # one console log per step: setup.log, download.log, diffusion_<model>_bs<bs>.log, dlrm_<prec>.log
├── diffusion/<model>_bs<bs>/  # launcher output: runs_summary.csv, training log, config, launcher.log, tensorboard
└── dlrm/<TF32|FP32>/        # train.log, results.csv, training_config.sh used
```

### Diffusion metrics

From AMDiffusionBenchmark's `runs_summary.csv` (averaged over the measured steps after warm-up):

| Column | Meaning |
|---|---|
| Step time GPU (s) | `avg_step_time_gpu`, GPU time per training step |
| Samples/s/GPU | `avg_fps_gpu` = batch size / step time, per GPU |
| Samples/s (node) | Samples/s/GPU × number of GPUs (8) |
| TFLOPS/GPU | `avg_tflops`, achieved TFLOPS per GPU |
| Wall time | whole step including model load and data preprocessing |

If `runs_summary.csv` is missing (e.g. the run was killed), the metrics are recomputed from the
`Step N - {...}` lines of the training log, skipping the first 5 warm-up steps.

### DLRM metrics

From DLRMBenchmark's `results.csv`, which the repo's `process_training_log.py` writes:
**Recommendations/s (mean)** = iterations/s × batch size, averaged over the last progress samples,
plus the coefficient of variation (std/mean). If `results.csv` is missing, the same calculation is
done on `train.log`.

Example `summary.md`:

```
## DLRM (DLRMBenchmark)

| Precision | Batch size | Status | Recommendations/s (mean) | std/mean | Wall time (min) |
|---|---:|---|---:|---:|---:|
| TF32 | 32768 | success | 690,439 | 0.010 | 2.0 |
```

---

## Failure handling

Each benchmark is a separate step, so one failure does not stop the others. Every step gets a status:

| Status | Meaning |
|---|---|
| `success` | completed and produced metrics |
| `oom` | out of GPU memory (from `runs_summary.csv` status or the log) |
| `failed` | the benchmark reported an error (e.g. Python traceback, `status=error`) |
| `timeout` | exceeded `step_timeout`; processes were killed |
| `not_run` | `--dry-run` |
| `skipped` | DLRM in `--dry-run` |
| `error` | any other non-zero exit |

Details:

- `launcher.py` **always exits 0**, even if training failed. The script reads the real outcome from
  the `status` column of `runs_summary.csv`.
- After every step, leftover training processes (`train.py`, `accelerate`, `torchrun`,
  `dlrm_main.py`) are killed so the next step starts with free GPUs.
- Model checkpoints written by the diffusion training (tens of GB) are deleted after each run; they
  are not needed for benchmarking.
- On Ctrl-C or `kill`, the container is removed and the results collected so far are still parsed.
- The script exits non-zero if any step was not successful, so CI jobs can detect failures.

**HF_TOKEN and the Makefile.** The Makefile includes `.env.example`, which sets `HF_TOKEN=` (empty)
and would hide the exported token, so `make download_assets` would stop at an interactive login.
The script logs in with `hf auth login --token` and passes `HF_TOKEN=...` on the `make` command line.

---

## Examples

**Full suite, auto-detected GPU:**
```bash
./run_benchmarks.sh
```

**Force MI355 batch sizes (e.g. config prepared on another node):**
```bash
./run_benchmarks.sh --gpu mi355
```

**One model with a custom batch size:**
```bash
./run_benchmarks.sh --models flux --bs flux=12 --dlrm-precisions none
```

**Batch-size sweep from the command line:**
```bash
./run_benchmarks.sh --models sdxl --bs sdxl=12,16,20,24 --dlrm-precisions none
```

**Short smoke test (fewer iterations):**
```bash
./run_benchmarks.sh --num-iterations 20 --dlrm-precisions none
```

**DLRM only:**
```bash
./run_benchmarks.sh --models none
```

**Reuse an existing shared cache without downloading:**
```bash
./run_benchmarks.sh --hf-cache /shared/hf/hub --hf-cache-readonly
```

**Dedicated experiment config** (`configs/mi355_sweep.yaml`):
```yaml
gpu: mi355
image: unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_3979e53_20260930
hf_cache_dir: /shared/huggingface
download: false
output_dir: ~/primus-bench/diffusion/mi355_sweep
diffusion:
  train_args:
    num_iterations: 100
  models:
    flux-dev:
      batch_size: [12, 16, 20]
    hunyuan-video:
      batch_size: [2, 3, 4]
dlrm:
  enabled: false
```
```bash
./run_benchmarks.sh -c configs/mi355_sweep.yaml
```

**Regenerate a summary:**
```bash
./run_benchmarks.sh --parse-only results/mi300_tus1-p3-g35_20261002_014509
```

**Debug a failing model interactively:**
```bash
./run_benchmarks.sh --models wan --dlrm-precisions none --keep-container
docker exec -it diffusion-bench-main-<...> bash
```

---

## How it works

```
run_benchmarks.sh (host)
 ├─ detect GPU (sysfs gfx_target_version)
 ├─ config.py: YAML + CLI overrides -> run plan
 ├─ create results dir, save config + resolved plan
 ├─ docker run -d --rm <image> sleep infinity
 │     mounts: HF cache -> /root/.cache/huggingface[/hub]
 │             results dir -> /results
 │             this folder -> /diffusion_bench (ro)
 ├─ docker exec ... in_container.sh setup        (Makefile sed, versions)
 ├─ docker exec ... in_container.sh download     (make download_assets / hf download)
 ├─ for each (model, batch size):
 │     docker exec ... timeout in_container.sh diffusion <model> <bs> <overrides>
 │        python launcher.py train_args=<model> train_args.train_batch_size=<bs>
 │                           train_args.logging_dir=/results/diffusion/<model>_bs<bs> ...
 ├─ for each precision:
 │     docker exec ... timeout in_container.sh dlrm <TF32|FP32> [bs]
 │        ./launch_training_single_node.sh -p <prec>   (log streamed, artifacts copied)
 ├─ finalize: hand results (and new cache files) to the host user
 ├─ docker rm -f
 └─ parse_results.py -> summary.md / summary.csv / summary.json
```

The container uses the same flags as the manual procedure (`--privileged --network host --ipc host
--device /dev/dri --device /dev/kfd --cap-add SYS_PTRACE --security-opt seccomp=unconfined
--group-add video`), plus `--ulimit memlock=-1:-1`. With `--rm`, a container that is stopped from
outside (for example by the perf batch runner's cleanup) is removed as well.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CONFIG ERROR: Could not auto-detect the GPU type` | No readable `/sys/class/kfd`; set `gpu: mi300\|mi355` or pass `--gpu`. |
| `CONFIG ERROR: Unknown key ...` | Typo in the YAML; the message suggests the closest valid key. |
| `HF_TOKEN is not set and no token found` | `export HF_TOKEN=hf_xxx`, or set `download: false` if the cache is already populated. |
| Download fails with 401/403 for FLUX.1-dev | Accept the model license on Hugging Face with the account that owns the token. |
| `Failed to pull <image>` | `docker login unifiedtrainingdockers.azurecr.io`, or pull the image manually. |
| `PyYAML is required on the host` | `pip install --user pyyaml` or `sudo apt install python3-yaml`. |
| `WARNING: other running containers may be using the GPUs` | Another container is running; make sure it isn't using the GPUs, or numbers will be wrong. |
| A diffusion step shows `oom` | Batch size too large for this GPU; lower it in the config or with `--bs`. |
| A step shows `timeout` | Increase `step_timeout` (first runs also preprocess data), or check its log in `logs/`. |
| Diffusion step `failed` | See `logs/diffusion_<model>_bs<bs>.log` and `diffusion/<model>_bs<bs>/*_logs_*.txt`. |
| Results or cache files owned by root | The run was killed before `finalize`; run `docker run --rm -v <dir>:/d <image> chown -R $(id -u):$(id -g) /d`. |
| Left-over container | `docker ps -a --filter name=diffusion-bench-` then `docker rm -f <name>`. |
