# Primus Auto Benchmarking Tool

<img width="1024" height="468" alt="Primus Banner" src="https://github.com/user-attachments/assets/f1b2bf61-d612-4e62-bac4-ac115928632a" />

An interactive bash menu for running Primus training benchmarks on AMD Instinct GPUs and collecting a metrics table from the logs.

Supported backends: Megatron, TorchTitan, Megatron Bridge, MLPerf, MaxText, and MaxDiffusion. Supported devices: MI300X, MI325X, and MI355X.

---

## Quick Start

Run the script with bash from anywhere. It finds the Primus root from its own location (`PRIMUS_ROOT` overrides that).

```bash
cd /path/to/Primus
export HF_TOKEN=hf_...
bash tools/auto_benchmark/run_primus_autobenchmark.sh
```

### Launch mode

The tool picks `primus-cli` mode from where it is running:

| Condition | Mode |
| --- | --- |
| `PRIMUS_MODE=direct` or `PRIMUS_MODE=container` | That mode |
| Running inside a container | `primus-cli direct` |
| Running on the host | `primus-cli container` |

Inside a container, training uses the current environment. On the host, container mode asks for an image. There is no default. An empty answer stops with `Docker image is required.`

```text
➜ Enter Docker image:
```

A pasted leading colon or extra spaces, such as `:  my-registry/image:tag`, is removed before Docker sees the name. An image that is still not a valid reference stops the suite instead of failing every model.

NCCL and Gloo use an interface that is up on the machine running the job. The benchmark script chooses it. A name you already exported is kept when that interface is up. Otherwise the default route is used, then any other up interface. Loopback and virtual devices such as `docker0` are skipped. `runner/helpers/envs/get_ip_interface.sh` is left unchanged for other Primus launches. GPU collectives still use the InfiniBand devices selected inside the container.

`PRIMUS_MODE` must be `direct` or `container`. `primus-cli` is taken from the repo root, with a fallback to `runner/primus-cli`.

### Docker release suites

Choose **Docker release suite** at the first prompt, then select Primus or JAX. The GPU is detected from `rocminfo`, and the suite runs with no model prompts.

| Release suite | What runs |
| --- | --- |
| `1` or `primus` | Primus: megatron, megatron_bridge, torchtitan |
| `2` or `jax` | JAX: maxtext, maxdiffusion |

On a login node that same choice submits one Slurm job. On a machine with a GPU, the run moves into its own session, so closing the SSH connection does not stop it. Details are in [What happens after you choose](#4-what-happens-after-you-choose).

---

## Features

- Separate Docker release and interactive benchmarking workflows
- One choice detects the GPU and runs that suite
- Manual selection by number, comma list, range, or `all` for a single backend
- Optional view and edit of a working copy; repo YAML files stay unchanged
- Direct or container launch through `primus-cli`
- MLPerf models launched from each module's own scripts
- One failed model is logged and the rest of the list still runs
- A dropped SSH session does not stop a run that has already started
- A login node submits the run to Slurm without an extra prompt
- Metrics table and CSV after the runs, with the first five steps dropped and a harmonic mean for throughput
- Each log and CSV row records the image digest, Primus commit, submodule pins, GPU, ROCm version, and framework versions
- One JSON manifest per suite lists every model, status, runtime, artifact path, and that same environment

---

## Walkthrough

### 1. Workflow

```
★ What do you want to run?
  ● 1) Docker release suite (automatic model selection)
  ● 2) Interactive benchmarking (choose backend and models)

➜ Enter number or name:
```

Choose `1` or `release` for an automated Docker release run. Choose `2` or `interactive` to select one backend and its models.

#### Docker release suite

```
★ Choose Docker Release Suite:
  ● 1) Primus (megatron, megatron_bridge, torchtitan)
  ● 2) JAX (maxtext, maxdiffusion)

➜ Enter number or name:
```

`1` or `primus` runs the Primus release suite. `2` or `jax` runs the JAX MaxText release suite. Either choice skips model, view, and edit prompts.

#### Interactive benchmarking

```
★ Choose Backend:
  ● 1) megatron
  ● 2) torchtitan
  ● 3) megatron_bridge
  ● 4) mlperf (MI355X only; not supported on MI300X/MI325X)
  ● 5) maxtext
  ● 6) maxdiffusion (MI300X/MI355X configs; MI325X uses MI300X)
```

Each backend writes logs under `results/logs_<backend>/` and gets its own metrics table.

### 2. Device

The tool looks for `rocminfo` on `PATH`, under `/opt/rocm/bin`, under `ROCM_PATH`, and under `/opt/rocm-*/bin`.

1. It reads a marketing name matching `MI300X`, `MI325X`, or `MI355X`.
2. If that name is missing, `gfx950` maps to MI355X.
3. `gfx942` is shared by MI300X and MI325X, so the tool asks you to choose.
4. On a login node with no GPU, it asks you to choose the target GPU. That choice is what a queued job will run.

```
★ Please select Device manually:
  ● 1) MI300X
  ● 2) MI325X
  ● 3) MI355X
```

MLPerf exits if the device is MI300X or MI325X. MaxDiffusion on MI325X uses the MI300X config directory.

Config roots:

- `examples/megatron/configs/<GPU>/`
- `examples/torchtitan/configs/<GPU>/`
- `examples/megatron_bridge/configs/<GPU>/`
- `examples/maxtext/configs/<GPU>/`
- `examples/maxdiffusion/configs/<GPU>/` (MI325X reads `MI300X`)
- `examples/mlperf/<model>/`

### 3. Model selection

The Docker release workflow does not ask for models. It prints every release config for the detected GPU and runs them. A missing YAML stops the tool before any run starts. The list lives in `run_primus_autobenchmark.sh` (`load_release_configs`).

| Suite | MI300X | MI325X | MI355X |
| --- | ---: | ---: | ---: |
| Primus (megatron + megatron_bridge + torchtitan) | 30 | 30 | 32 |
| JAX (maxtext + maxdiffusion) | 24 | 24 | 26 |

Per-backend counts inside those suites:

| Backend | MI300X | MI325X | MI355X |
| --- | ---: | ---: | ---: |
| megatron | 23 | 23 | 25 |
| torchtitan | 5 | 5 | 5 |
| megatron_bridge | 2 | 2 | 2 |
| maxtext | 21 | 21 | 23 |
| maxdiffusion | 3 | 3 | 3 |

Megatron on every GPU: DeepSeek-V2-Lite BF16, GPT-OSS 20B BF16 and FP8, Llama 3.1 8B and 70B BF16 and FP8, Llama 3.3 70B BF16 and FP8, Mixtral 8x7B BF16, Qwen2.5 7B BF16 and FP8, Qwen2.5 72B BF16, Qwen3 30B-A3B BF16 and FP8. MI355X also runs Llama 3.1 8B MXFP8 and MXFP4.

MI300X, MI325X, and MI355X Megatron also include Mamba 370M, Zebra-Llama-Mamba 1B/3B/8B, GDN 1B, KDA 1B, Zebra-Llama-GDN 1B, and Zebra-Llama-KDA 1B. MI325X uses its own copy when that file exists, and the MI300X copy when the MI325X file is absent.

TorchTitan: DeepSeek-V3 16B BF16 and Llama 3.1 8B and 70B, each in BF16 and FP8.

Megatron Bridge on every GPU: `qwen3_32b_lora_posttrain.yaml` and `qwen3_32b_sft_posttrain.yaml`. MI300X and MI355X use their own copies. MI325X uses the MI300X copies. Both run as `train posttrain`.

MaxText BF16 on every GPU: Llama 2 7B and 70B, Llama 3 8B and 70B, DeepSeek-V2 16B, Mixtral 8x7B, Llama 3.3 70B, Qwen3 14B, Qwen3 30B-A3B, Gemma 4 26B and 31B, and Mixtral 8x22B.

Nanoo FP8 on MI300X and MI325X, and not on MI355X: Llama 2 7B and 70B, Llama 3 8B, DeepSeek-V2 16B, Mixtral 8x7B, Qwen3 14B, Qwen3 30B-A3B, and Gemma 4 26B and 31B. Llama 3 70B, Llama 3.3 70B, and Mixtral 8x22B stay BF16 on those GPUs.

FP8 on MI355X: the Nanoo list plus Llama 3 70B and Llama 3.3 70B. Mixtral 8x22B stays BF16. Filenames use `nanoo_fp8` on MI300X and MI325X, and `fp8` on MI355X.

MaxDiffusion is part of the JAX release suite: `flux_dev`, `wan2.1_1.3b`, and `wan2.1_14b`, all BF16. MI325X uses the MI300X copies. It uses the same JAX image as MaxText.

#### Add a model

The Docker release suite runs the filenames returned by `load_release_configs` in `run_primus_autobenchmark.sh`. A YAML that only sits under `examples/` shows up in the interactive menu. The release suite includes it after the name is added in that function. Update the suite counts in the tables above when the list changes.

`resolve_release_config` looks up `examples/<backend>/configs/<GPU>/<file>`. For MI325X, a missing file falls back to the MI300X copy. A path that still does not exist stops the suite before the first model and prints `Built-in release config does not exist`.

`pretrain` versus `posttrain` comes from the YAML. `metrics.py` reads the backend's existing log lines, so the new model shows up in that backend's CSV when the log contains those lines. MLPerf stays on the interactive menu.

##### Primus

Primus runs three cases, in this order: `megatron`, `megatron_bridge`, then `torchtitan`. Each case takes the full filename.

| Backend | Config directory | Where to add the name |
| --- | --- | --- |
| megatron | `examples/megatron/configs/<GPU>/` | `names=(...)` in the `megatron` case |
| megatron_bridge | `examples/megatron_bridge/configs/<GPU>/` | the `for name in` list in the `megatron_bridge` case |
| torchtitan | `examples/torchtitan/configs/<GPU>/` | the `for name in` list in the `torchtitan` case |

A model that runs on MI300X, MI325X, and MI355X goes in the shared list. Copy the YAML into each GPU directory that should have its own file. MI325X can omit its copy and use the MI300X file.

A model that runs on one GPU goes in the `if [[ "$gpu" == "..." ]]` block for that GPU. Llama 3.1 8B MXFP8 and MXFP4 are added that way, inside `if [[ "$gpu" == "MI355X" ]]`, and only in the `megatron` case.

Megatron example. Add the file, then add the same filename to `names`:

```text
examples/megatron/configs/MI300X/my_model-BF16-pretrain.yaml
examples/megatron/configs/MI355X/my_model-BF16-pretrain.yaml
```

```bash
names=(
    ...
    my_model-BF16-pretrain.yaml
)
```

Megatron Bridge and TorchTitan use the same two steps with their own directory and `for name in` list. A Megatron Bridge post-train file is still just a filename in that list. The YAML's `post_trainer` key selects `train posttrain`.

##### JAX

JAX runs two cases, in this order: `maxtext`, then `maxdiffusion`.

MaxText builds the filename from a stem. The files live in `examples/maxtext/configs/<GPU>/`.

| List in the `maxtext` case | File it selects | GPUs |
| --- | --- | --- |
| `bf16_names` | `<stem>-bf16-pretrain.yaml` | MI300X, MI325X, MI355X |
| `nanoo_names` | `<stem>-nanoo_fp8-pretrain.yaml` | MI300X, MI325X |
| `fp8_names` | `<stem>-fp8-pretrain.yaml` | MI355X |

Add the stem to `bf16_names` when the BF16 config should run. Add it to `nanoo_names` when MI300X and MI325X should also run Nanoo FP8. Add it to `fp8_names` when MI355X should run FP8. Leave a stem out of a precision list when that precision should stay off. Llama 3 70B is in `bf16_names` and `fp8_names`, and out of `nanoo_names`, so MI300X runs it as BF16 only.

```text
examples/maxtext/configs/MI300X/my_model-bf16-pretrain.yaml
examples/maxtext/configs/MI300X/my_model-nanoo_fp8-pretrain.yaml
examples/maxtext/configs/MI355X/my_model-bf16-pretrain.yaml
examples/maxtext/configs/MI355X/my_model-fp8-pretrain.yaml
```

```bash
bf16_names=( ... my_model )
nanoo_names=( ... my_model )
fp8_names=( ... my_model )
```

MaxDiffusion takes the full filename in the `for name in` list of the `maxdiffusion` case. Files live in `examples/maxdiffusion/configs/<GPU>/`. The current entries are `flux_dev-pretrain.yaml`, `wan2.1_1.3b-pretrain.yaml`, and `wan2.1_14b-pretrain.yaml`. Add the new filename to that list. MI325X uses the MI300X copy when its own file is absent. MaxDiffusion runs in the same JAX job as MaxText, after the MaxText models.

#### Manual selection

Choosing a single backend lists every YAML in the device directory. Duplicate basenames are shown once.

```
➜ Select config number(s) (comma-separated, range, or 'all'):
(Examples: 1,3,5 or 4-8 or all)
```

- `1` one config
- `1,3,5` several configs
- `4-8` a range
- `all` every listed config

#### MLPerf

```
★ Available MLPerf Models: (mlperf / MI355X)
  ● 1) llama3.1_8b
  ● 2) llama2_70b
  ● 3) gpt_oss_20b
  ● 4) flux1
```

Same number, range, and `all` selection. MLPerf is separate from the release suite.

### 4. What happens after you choose

If `rocminfo` finds a GPU, the tool keeps the terminal attached so you can watch the log, and starts the work in a separate session. Closing SSH, or pressing Ctrl-C, drops the view and leaves the run going. The session log is `results/sessions/autobench_<GPU>_<suite>_<timestamp>.log`. The PID is printed above that path. Stop the run with `kill <pid>` when you mean to stop it.

If this machine has no GPU, the tool submits one Slurm job and returns. There is no second confirm. The job is one node, one task, 8 GPUs, and 24 hours unless you exported overrides before starting the tool:

- `SBATCH_PARTITION` — omit it to use the site default
- `SBATCH_TIME` — default `24:00:00`
- `SBATCH_GPUS_PER_NODE` — default `8`
- `SBATCH_ACCOUNT` — omitted when unset

`HF_TOKEN` is required. It is inherited with `sbatch --export=ALL` and is not written into the job script. Container mode passes `HF_TOKEN`, `WANDB_API_KEY`, `DATABRICKS_TOKEN`, and `DOCKER_LOGIN_KEY` with `docker -e KEY` so the value is not part of the `docker run` arguments or the launch log. `DOCKER_IMAGE` is required for a host run and is inherited the same way. A shell already inside a container uses direct mode and does not ask for an image.

`SBATCH_DRY_RUN=1` writes the script and does not call `sbatch`.

The job re-runs this script from a saved selection file. It runs every model in the suite, continues after a failure, and writes a metrics table per backend. The Slurm log is `results/slurm/<jobid>_<GPU>_<suite>.out`.

`sbatch` must be on `PATH`. The submit shell must be allowed to pass the environment with `--export=ALL`.

### 5. View and edit

View and edit are offered for a manual, non-MLPerf selection. The release suite and MLPerf skip both.

**View** prints each selected YAML with comments and blank lines removed.

**Edit** copies the file to `/tmp/primus_edit_<model>_<pid>.yaml` and opens an editor. The copy under `examples/` is left as it is. The run later saves that copy as `results/logs_<backend>/<model>_<backend>_<device>_runN_working.yaml` and passes that path to `primus-cli`.

Editor search order: `$EDITOR`, `$VISUAL`, `vim`, `vi`, `nano`, `emacs -nw`, `code --wait`, `cursor --wait`. If none of those exist, the tool tries to install `vim`.

### 6. Environment

```
★ Setting up environment...
✓ Primus launch mode: container
✓ Docker image: <the image you entered>
✓ Using HuggingFace token from the environment
```

If `HF_TOKEN` is unset, the interactive menu asks for it (input is hidden). A batch job exits when `HF_TOKEN` is missing. On the host, container mode asks for a Docker image and exits with `Docker image is required` when none is given. A shell already inside a container uses direct mode and does not ask for an image.

The tool also marks the Primus git directories as safe for the current user.

MLPerf dataset roots, used when a download is still required:

| Model | Data |
| --- | --- |
| llama3.1_8b | `/data/mlperf_llama31_8b` |
| llama2_70b | `PACKED_DATA_DIR=/data`, checkpoint `/data/megatron_checkpoints/Llama-2-70b-hf` |
| gpt_oss_20b | `/data/gpt_oss_20b` |
| flux1 | `/data/mlperf_flux1` |

`RUN_RUNTIME_TUNABLES` defaults to `1` for MLPerf. Set it to `0` to skip the runtime tunables those modules apply.

### 7. Execution

Before the first model, on the machine that will run it, the tool stops every other running Docker or Podman container and terminates processes that still hold the AMD GPU devices (`/dev/kfd` and `/dev/dri`). The container this job is already running inside is left alone. If another process still holds GPU memory after that cleanup, the benchmark stops instead of starting on a busy GPU.

Each selected model then runs in order. The loop uses `|| true`, so a non-zero exit is recorded and the next model starts after a short pause.

Non-MLPerf launch:

```bash
primus-cli <direct|container> -- train <pretrain|posttrain> --config <working.yaml>
```

`pretrain` versus `posttrain` comes from the YAML (`pre_trainer` / `post_trainer`, or a `posttrain` / `sft` name). Container mode tees stdout and stderr into the log. Direct mode passes `--log_file`. Both append `AUTO_BENCHMARK_EXIT_CODE=<code>` to the log.

MLPerf follows each module's own scripts on MI355X:

- `llama3.1_8b`, `llama2_70b`, and `gpt_oss_20b` source `config_MI355X_1x8x1.sh` and run `run_and_time.sh`
- `flux1` runs the training command from `run_with_docker.sh` (`primus-cli direct -- train pretrain --config examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml`) and appends the MLLOG file to the benchmark log

Those MLPerf scripts expect to already be in the MLPerf environment (`/data`, `/results`, `/model`).

Logs:

```
tools/auto_benchmark/results/logs_<backend>/<model>_<backend>_<device>_runN.log
```

`runN` increments from logs already in that directory.

Each log ends with an `AUTO BENCHMARK ENVIRONMENT` block: image digest, Primus commit, submodule pins, GPU, ROCm version, and framework versions (`torch`, `jax`, and `transformers` when they import). Those same fields are copied onto every metrics CSV row.

The suite also writes one JSON manifest, `results/manifest_<suite>_<device>_<timestamp>.json`. It is updated after every model and holds the environment plus each model's status, exit code, runtime in seconds, and artifact paths.

Each run also writes two files beside that log:

- `..._runN_snapshot.txt` records the kernel, Primus git commit, ROCm version, GPU firmware, container image digest, and the launch environment. Firmware comes from `amd-smi` when `amd-smi version` succeeds, and from `rocm-smi` otherwise. Tokens are not included. A suite-level copy is `results/system_snapshot.txt`.
- `..._runN_reproduce.sh` is one command that reruns that model. Export `HF_TOKEN` first. The script sets the image, train suite, and config that produced the result.

A non-zero exit also writes `..._runN_failure.tar.gz`. The archive holds the log, the working YAML, the snapshot, the reproduce script, and a GPU capture from the moment the model failed. That capture uses `amd-smi` when `amd-smi version` succeeds, and `rocm-smi` when that is the tool that runs.

### 8. Metrics

After the loop, the tool runs:

```bash
python metrics.py <backend>
```

`metrics.py` reads `results/logs_<backend>/`, prints a table, and writes:

- `results/metrics_<backend>.csv` (latest)
- `results/metrics_<backend>_<timestamp>.csv` (snapshot)

Throughput columns drop the first five steps, then use the harmonic mean. Memory, MFU, step time, and loss stay arithmetic means. Megatron iteration lines that are printed twice are collapsed by iteration number before the average. Every row also carries Image Digest, Primus Commit, Submodule Pins, GPU, ROCm, and Framework Versions from that run's log.

A failed log is labeled `Error found - check log (<class>)`. The class is the most specific match, not every keyword in the log. Classes: `killed_sigkill` (exit 137), `killed_sigterm` (exit 143), `huggingface_auth`, `out_of_memory`, `distributed_or_network`, `dataset_or_checkpoint`, `dependency_or_import`, `jax_or_compilation`, `configuration`, `permission`, `timeout`, otherwise `runtime_or_incomplete`. A clean exit with no parseable metrics is `incomplete_metrics`. Non-finite loss or grad norm is `Diverged (non-finite loss) - invalid`.

You can rebuild a table without re-running training:

```bash
cd tools/auto_benchmark
python metrics.py megatron
```

---

## How results are stored

Everything the tool writes stays under `tools/auto_benchmark/results/`. One model uses one prefix:

```text
results/logs_<backend>/<model>_<backend>_<device>_runN
```

`runN` is the next number after the logs already in that directory. A second run of the same model becomes `run2` and does not overwrite `run1`.

### Every model

These files are written whether the model succeeds or fails:

| File | What it holds |
| --- | --- |
| `..._runN.log` | Training output, then `AUTO_BENCHMARK_EXIT_CODE=<code>`, then the environment block and the record block |
| `..._runN_working.yaml` | The YAML passed to `primus-cli`. The file under `examples/` is unchanged |
| `..._runN_snapshot.txt` | Kernel, Primus commit, ROCm, GPU firmware, image digest, and this run's exit code. No tokens |
| `..._runN_reproduce.sh` | One command that reruns this model. Export `HF_TOKEN` first |

The log ends with:

```text
AUTO_BENCHMARK_EXIT_CODE=0
===== AUTO BENCHMARK ENVIRONMENT =====
image_digest=...
primus_commit=...
submodule_pins=...
gpu=...
rocm_version=...
framework_versions=...
runtime_seconds=...
===== AUTO BENCHMARK RECORD =====
SNAPSHOT=..._runN_snapshot.txt
REPRODUCE=..._runN_reproduce.sh
MANIFEST=results/manifest_<suite>_<device>_<timestamp>.json
```

A successful model stops there. Its metrics row has the throughput numbers, and the manifest entry has `"status": "ok"`.

### When a model fails

A non-zero exit still keeps the four files above and adds two more:

| File | What it holds |
| --- | --- |
| `..._runN_gpu_state.txt` | GPU capture from the moment the model failed (`amd-smi`, or `rocm-smi` when that is the tool that runs) |
| `..._runN_failure.tar.gz` | The log, working YAML, snapshot, reproduce script, and GPU capture in one archive |

The log also gains a `FAILURE_BUNDLE=` line. The manifest entry has `"status": "failed"`, the exit code, and `artifacts.failure_bundle` set to that archive. The CSV row for that model is `Error found - check log (<class>)` and the numeric columns are blank. The class is one of: `killed_sigkill` (exit 137), `killed_sigterm` (exit 143), `huggingface_auth`, `out_of_memory`, `distributed_or_network`, `dataset_or_checkpoint`, `dependency_or_import`, `jax_or_compilation`, `configuration`, `permission`, `timeout`, otherwise `runtime_or_incomplete`. Exit 0 with no parseable throughput is `incomplete_metrics`. Non-finite loss or grad norm is `Diverged (non-finite loss) - invalid`.

The suite records that model and starts the next one. Open the log for the traceback, or unpack the failure archive when you want the YAML and GPU state with it.

### Suite files

These cover the whole run, not one model:

| File | What it holds |
| --- | --- |
| `results/system_snapshot.txt` | The same machine snapshot, written once before the first model |
| `results/manifest_<suite>_<device>_<timestamp>.json` | Environment plus every model so far: status, exit code, runtime, and artifact paths. Updated after each model |
| `results/metrics_<backend>.csv` | Latest table for that backend. Rewritten when the backend's models finish |
| `results/metrics_<backend>_<timestamp>.csv` | A copy of that table that is not overwritten later |
| `results/sessions/autobench_<GPU>_<suite>_<timestamp>.log` | The detached session log on a GPU machine |
| `results/sessions/autobench_<GPU>_<suite>_<timestamp>.state` | The saved model list for that session |
| `results/slurm/<jobid>_<GPU>_<suite>.out` | The Slurm log when the tool queued the run |
| `results/slurm/autobench_<GPU>_<suite>_<timestamp>.sbatch` | The job script. It does not contain `HF_TOKEN` or `DOCKER_IMAGE` |

A Primus release writes three CSVs, one each for megatron, megatron_bridge, and torchtitan. A JAX release writes `metrics_maxtext.csv` and `metrics_maxdiffusion.csv`.

```text
tools/auto_benchmark/results/
├── system_snapshot.txt
├── manifest_primus_MI300X_20261013_182247.json
├── logs_megatron/
│   ├── llama3.1_8B-BF16-pretrain_megatron_MI300X_run1.log
│   ├── llama3.1_8B-BF16-pretrain_megatron_MI300X_run1_working.yaml
│   ├── llama3.1_8B-BF16-pretrain_megatron_MI300X_run1_snapshot.txt
│   └── llama3.1_8B-BF16-pretrain_megatron_MI300X_run1_reproduce.sh
├── logs_torchtitan/
├── logs_megatron_bridge/
├── logs_maxtext/
│   ├── llama2_7B-bf16-pretrain_maxtext_MI300X_run1.log
│   ├── llama2_7B-bf16-pretrain_maxtext_MI300X_run1_working.yaml
│   ├── llama2_7B-bf16-pretrain_maxtext_MI300X_run1_snapshot.txt
│   ├── llama2_7B-bf16-pretrain_maxtext_MI300X_run1_reproduce.sh
│   ├── llama2_7B-bf16-pretrain_maxtext_MI300X_run1_gpu_state.txt
│   └── llama2_7B-bf16-pretrain_maxtext_MI300X_run1_failure.tar.gz
├── logs_maxdiffusion/
├── logs_mlperf/
├── metrics_megatron.csv
├── metrics_megatron_2026-10-13_18-22-47.csv
├── sessions/
│   ├── autobench_MI300X_primus_20261013_182247.state
│   └── autobench_MI300X_primus_20261013_182247.log
└── slurm/
    ├── autobench_MI300X_primus_20261013_171215.sbatch
    └── 8168_MI300X_primus.out
```

The two failure files in that tree are an example of a failed model. A model that exits 0 does not get `_gpu_state.txt` or `_failure.tar.gz`.

---

## Metrics columns

**Megatron:** iterations, iteration time, harmonic-mean TFLOPS/GPU, harmonic-mean tokens/GPU. Batch size, sequence length, global batch size, and precision come from the YAML when it can be found.

**Megatron Bridge:** iterations, iteration time, and harmonic-mean TFLOPS/GPU. Iteration time and `MODEL_TFLOP/s/GPU` are merged from separate log lines.

**TorchTitan:** steps after warm-up, memory, harmonic-mean TPS, harmonic-mean TFLOPS, MFU.

**MaxText and MaxDiffusion:** step time, harmonic-mean TFLOPS/device, tokens/s/device, samples/s/device, and frames/s/device, plus loss. Values come from `completed step:` lines. A model logs whichever of those rates it prints.

**MLPerf:** samples, run duration, and throughput from `:::MLLOG` lines. Llama 3.1 8B, GPT-OSS 20B, and Flux use `overall_throughput`. Llama 2 70B uses `throughput`.

---

## Example: release suite on a GPU node

```bash
export DOCKER_IMAGE=registry.example/image:tag
export HF_TOKEN=hf_...
bash tools/auto_benchmark/run_primus_autobenchmark.sh
# Workflow: 1 (Docker release suite)
# Release suite: 2 (JAX)
# Device: detected MI355X
```

The tool runs every built-in MI355X MaxText release config, then Flux and both Wan MaxDiffusion configs. It keeps going if one fails and writes `results/metrics_maxtext.csv` and `results/metrics_maxdiffusion.csv`. The session log is under `results/sessions/`.

## Example: Primus release from a login node

```bash
export DOCKER_IMAGE=registry.example/image:tag
export HF_TOKEN=hf_...
export SBATCH_PARTITION=gpu
export SBATCH_TIME=24:00:00
bash tools/auto_benchmark/run_primus_autobenchmark.sh
# Workflow: 1 (Docker release suite)
# Release suite: 1 (Primus)
# Device: 3  (MI355X, because the login node has no GPU)
```

The tool submits one job. Check it with `squeue`, then read `tools/auto_benchmark/results/slurm/<jobid>_MI355X_primus.out` and the metrics CSVs for megatron, megatron_bridge, and torchtitan.

## Example: one manual config, direct mode

```bash
export HF_TOKEN=hf_...
bash tools/auto_benchmark/run_primus_autobenchmark.sh
# Workflow: 2 (interactive)
# Backend: 1 (megatron)
# Device: MI300X
# Select: 1
# View parameters? y
# Edit configuration files? n
```

---

## Troubleshooting

**`primus-cli` not found.** Expect it at the repo root or at `runner/primus-cli`.

**A release YAML is missing.** The tool stops before the first run and prints the path. Megatron Bridge expects `qwen3_32b_lora_posttrain.yaml` and `qwen3_32b_sft_posttrain.yaml` under `examples/megatron_bridge/configs/MI300X` or `MI355X`. MI325X uses the MI300X pair.

**MLPerf on MI300X or MI325X.** MLPerf in this tool is MI355X only.

**`sbatch` was not found.** A login node needs Slurm on `PATH` before the tool can queue the run.

**Batch job exits immediately with `HF_TOKEN is not set` or `Docker image is required`.** Export `HF_TOKEN` and `DOCKER_IMAGE` before you start a non-interactive run. Both are passed with `--export=ALL` and are not stored in the `.sbatch` file. An interactive host run asks for the image and stops if the answer is empty.

**Login node says it could not detect a GPU.** Choose the target GPU. After the token prompt, the tool submits the job.

**The terminal returned or the connection dropped.** The run keeps going. Read `results/sessions/autobench_<GPU>_<suite>_<timestamp>.log`, or the Slurm file under `results/slurm/`.

**Metrics row says `Error found - check log (...)`.** Open the matching file in `results/logs_<backend>/`. The same prefix has `_failure.tar.gz` with the log, working YAML, system snapshot, reproduce script, and GPU state. The class is a hint: `killed_sigkill` / `killed_sigterm` for a signal, `huggingface_auth` for a rejected or gated Hugging Face repo, plus memory, import, data, network, JAX compile, config, permission, or timeout. `incomplete_metrics` means the process ended without a parseable throughput line.

**Rerun one result.** Export `HF_TOKEN`, then run the `_reproduce.sh` file next to that log.

---

## Support

1. Logs: `tools/auto_benchmark/results/logs_<backend>/`
2. Latest table: `tools/auto_benchmark/results/metrics_<backend>.csv`
3. Slurm output: `tools/auto_benchmark/results/slurm/`
4. Confirm the GPU with `rocminfo` when you are on a compute node
5. Confirm `HF_TOKEN`, and `DOCKER_IMAGE` when you want container mode
