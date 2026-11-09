# Primus Perf Batch Runner

Batch-runs Primus training benchmarks across Megatron, Megatron-Bridge,
TorchTitan, MaxText and MaxDiffusion, plus the diffusion and DLRM benchmarks
of [`diffusion_bench/`](./diffusion_bench/), and extracts the results into
`results.csv` / `results.json`, which record not just the throughput but the
settings that produced it and, for a run that failed, why.

Two goals shape this tool:

- **The script is never edited.** Every setting comes from the environment, so
  starting a run is a handful of exports rather than a diff. Which models run
  is chosen in [`configs.yaml`](./configs.yaml): `GPU` and `BACKEND` narrow it
  to one backend's suite, and commenting entries in or out picks a subset of
  that — in the catalog itself, or in a copy. Running something outside the
  curated suite means adding its `examples/...` path to the catalog, which is
  the only case where a path gets typed.
- **Every number is attributable.** Each log carries the image digest and ID,
  submodule pins, GPU model, host and container ROCm, framework versions,
  the platform it ran on (server, BIOS, CPU, memory, GPU settings, host GPU
  driver, NICs, OS) with a one-line system description, host settings, world
  size and the full perf-relevant environment, and those land in the CSV as
  columns.
- **Every run ends with a verdict.** Runs are stopped when they hang or run
  too long, nothing starts on busy GPUs, and each log and CSV row says why a
  run ended.

## Quick start

```bash
export HF_TOKEN=hf_...                                    # required
export DOCKER_IMAGE=rocm/primus:<tag>   # required
export BACKEND=maxtext,maxdiffusion                        # required
export RESULT_DIR=~/primus-bench/mi325x-v26.7

# GPU is auto-detected from amd-smi / rocm-smi; set it to override.
# GPU=MI325X

# RCCL still needs a real NIC on a single node.
# NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME are auto-detected from get_ip_interface.sh,
# which maps the first IPv4 from hostname -I to an interface.
# On some nodes that address belongs to docker0.
# So, the following may need to be set manually.

# export NCCL_SOCKET_IFNAME=eno0                            # e.g. eno0 / ibp...
# export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

DRY_RUN=1 bash tools/perf/run_batch.sh                    # check the plan first
bash tools/perf/run_batch.sh                              # single node
python3 tools/perf/extract_results.py "$RESULT_DIR"       # optional: already run at the end of a batch
```

`DRY_RUN=1` prints every selected config with its batch size, steps, log
filename and exact launch command, then exits without touching docker or the
GPUs. It does not need `HF_TOKEN` or `DOCKER_IMAGE`, so it also works on a
login node (set `GPU` there, since there is no GPU to detect it from).

Multi-node, from inside a Slurm allocation (`salloc -N 8 -p mi325x`):

```bash
bash tools/perf/run_batch_multinode.sh
```

The runner can be invoked from any directory.

## Dependencies

[`yq`](https://github.com/mikefarah/yq) v4+ is the only external requirement.
Everything else (`bash`, `docker`, `git`, `python3`, `amd-smi` or `rocm-smi`) is already
present on a node that can run Primus. `BACKEND=...,diffusion_dlrm` also needs PyYAML
in the host's `python3` (`pip install --user pyyaml`), even for `DRY_RUN=1`. The Excel
report needs pandas and XlsxWriter, which `report.py` installs for you (see "Excel report").

### Automatic installation

If `yq` is missing, the runner installs it. Nothing needs doing beforehand:

```
[INFO] yq not found; fetching v4.53.2 (linux_amd64) into ~/.cache/primus-perf/bin
[INFO] installed yq (https://github.com/mikefarah/yq/) version v4.53.2
```

What it does, exactly:

1. Downloads the static binary for your architecture, from the pinned release:
   `https://github.com/mikefarah/yq/releases/download/v4.53.2/yq_linux_amd64`
   (`uname -m` selects `amd64`, `arm64` or `386`), using `curl` or `wget`,
   whichever is present.
2. Marks it executable and checks it actually runs and reports mikefarah v4,
   so a truncated download or an HTML error page fails immediately rather than
   surfacing later as a confusing expression error.
3. Moves it to **`~/.cache/primus-perf/bin/yq`**.
4. Prepends `~/.cache/primus-perf/bin` to `PATH` **for that run only**.

No root is needed, nothing is installed system-wide, and your shell
configuration is not modified. Later runs reuse the cached binary.

| Variable | Default | Effect |
|---|---|---|
| `PERF_CACHE_DIR` | `${XDG_CACHE_HOME:-~/.cache}/primus-perf` | Where the binary is cached; it lands in `$PERF_CACHE_DIR/bin/yq`. |
| `YQ_VERSION` | `v4.53.2` | Release tag to fetch. |

The version is pinned rather than tracking `latest` because a benchmark tool
whose own dependency silently changes between runs works against the
provenance everything else here exists to record.

### Manual installation

To install it yourself — also without root — put the binary anywhere on your
`PATH`:

```bash
mkdir -p ~/.local/bin
curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
  -o ~/.local/bin/yq
chmod +x ~/.local/bin/yq
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.bashrc to make it stick
```

Swap `yq_linux_amd64` for `yq_linux_arm64` on ARM. With root, the same binary
can go in `/usr/local/bin` instead. Package managers work too
(`snap install yq`, `brew install yq`, `apk add yq` on Alpine, or
`go install github.com/mikefarah/yq/v4@latest`).

### Verifying

```bash
$ yq --version
yq (https://github.com/mikefarah/yq/) version v4.53.2
```

That output must mention **mikefarah** and be **v4 or newer**. Two unrelated
tools are called `yq`: mikefarah's Go binary, whose v4 expressions these
scripts use, and [kislyuk's](https://github.com/kislyuk/yq) Python jq wrapper
installed by `pip install yq`, which prints just `yq 3.x.x` and takes
different expressions. Having the wrong one on `PATH` is the likeliest way for
this to go wrong, so the runner checks rather than assuming, and bootstraps a
private copy if needed:

```
[WARN] the yq on PATH is not mikefarah's v4+ (yq 3.2.3).
       These scripts use v4 expressions; bootstrapping a private copy.
```

## Choosing what to run

### Catalog mode (default)

[`configs.yaml`](./configs.yaml) is the curated release suite, keyed by GPU
then backend:

```yaml
MI325X:
  megatron:
    - examples/megatron/configs/MI325X/llama3.1_8B-BF16-pretrain.yaml
    - examples/megatron/configs/MI325X/llama3.3_70B-BF16-pretrain.yaml
    # - examples/megatron/configs/MI325X/deepseek_v3-BF16-pretrain.yaml   # OOM, re-enable after #1127
```

Entries are **enabled by default**, so selecting a GPU and backend runs that
whole suite with no editing. To run a subset, comment entries out — ideally
with a reason, as above.

To run a model that is not in the suite, add its path under the right GPU and
backend. Paths are relative to the repo root, and the backend key must match
the framework the config declares:

```yaml
MI325X:
  megatron:
    - examples/megatron/configs/MI325X/qwen3_30B_A3B-FP8-pretrain.yaml   # added
```

Both of those are checked by the catalog test, so a typo or a misfiled entry
fails a PR rather than a benchmark night.

`GPU` selects the top-level key and `BACKEND` the second-level keys, as a
comma-separated list. Both scope the run; comments exclude within that scope:

```bash
GPU=MI325X BACKEND=maxtext,maxdiffusion bash tools/perf/run_batch.sh
```

Only combine backends that the image you are benchmarking actually supports —
see [below](#which-backends-go-with-which-image).

For an ad-hoc set, copy the catalog and point at your copy, which keeps the
tracked file clean:

```bash
cp tools/perf/configs.yaml "$RESULT_DIR/configs.yaml"
$EDITOR "$RESULT_DIR/configs.yaml"
CONFIG_FILE="$RESULT_DIR/configs.yaml" bash tools/perf/run_batch.sh
```

[`perf_configs.yaml`](./perf_configs.yaml) is a second catalog in the same
format: the models published in the AI Training section of AMD's
[performance results page](https://www.amd.com/en/developer/resources/rocm-hub/ai-cloud-development/performance-results.html#ai-training),
for MI355X, MI325X and MI300X. Its header notes where a config's batch size
differs from the published number. Run it with
`CONFIG_FILE=tools/perf/perf_configs.yaml`. The catalog test does not cover
this file.

Editing `tools/perf/configs.yaml` directly is the right move when the change
really is a change to the release suite: it is reviewed like any other file,
and [`tests/unit_tests/tools/test_perf_catalog.py`](../../tests/unit_tests/tools/test_perf_catalog.py)
checks that every path still exists, declares a train module, and sits under
the backend key matching its framework.

> Do **not** comment out a backend key such as `megatron:` — that orphans its
> indented list items and breaks the YAML. Use `BACKEND=` to skip a backend.

### Directory mode

`CONFIG_DIR` runs every `*.yaml` under a directory, recursively and sorted:

```bash
CONFIG_DIR=~/my-configs bash tools/perf/run_batch.sh
```

Renaming a config to `*.yaml.done` removes it from future runs, since discovery
matches `*.yaml` on the basename. That is the resume trick: when a batch dies
partway, mark the ones that finished and re-run.

Setting both `CONFIG_DIR` and `CONFIG_FILE` is an error. Missing paths and
duplicate entries are reported before anything launches.

### Diffusion and DLRM (`diffusion_dlrm`)

`diffusion_dlrm` is a backend outside the catalog. Adding it to `BACKEND`
runs [`diffusion_bench/run_benchmarks.sh`](./diffusion_bench/) once per
repetition (`NUM_REPS`), after every other config of the batch, for the diffusion models (FLUX.1-dev, SDXL,
Mochi, HunyuanVideo, Wan 2.1) and DLRM precisions its config enables. So
everything a `rocm/primus` image can run is one command:

```bash
GPU=MI355X BACKEND=megatron,torchtitan,megatron_bridge,diffusion_dlrm \
DOCKER_IMAGE=rocm/primus:v26.8 RESULT_DIR=~/primus-bench/mi355x-v26.8 \
bash tools/perf/run_batch.sh
```

- **What runs** comes from diffusion_bench's own config,
  [`diffusion_bench/config.yaml`](./diffusion_bench/config.yaml): models,
  batch sizes per GPU type, DLRM precisions and the Hugging Face cache.
  Point `DIFFUSION_DLRM_CONFIG` at a copy to change them without touching
  the shared file. The runner passes it `DOCKER_IMAGE`, the GPU type (`GPU`
  MI300X/MI308X/MI325X → `mi300`, MI350X/MI355X → `mi355`, otherwise read
  from the GPU's gfx942 / gfx950 architecture), its container label,
  and `EXTRA_ENV`. `TRAIN_STEPS` is not applied: every model keeps the
  `num_iterations` of that config, and DLRM its own length. `HF_TOKEN`
  reaches it through the environment, never a command line. The runner's own Primus defaults
  (`HSA_NO_SCRATCH_RECLAIM` and the two `*_ATOMIC_FP32` settings) are not
  applied, so the numbers match a standalone diffusion_bench run. `GPU`
  need not be a catalog key when `diffusion_dlrm` is the only backend.
- **No runner time limit.** `MAX_RUN_SECONDS` and `MAX_IDLE_SECONDS` do not
  apply to this run. diffusion_bench stops a step that exceeds its
  `step_timeout` (4 hours in the default config), marks it `timeout` and
  moves on to the next model. Its asset download has a separate
  `download_timeout`, which is 0 (none) by default, so a stalled first
  download can hold the batch until it is stopped.
- **Hugging Face cache.** The default config caches under `/tmp/huggingface`
  on the node; a full first download is about 370 GB. Set
  `DIFFUSION_DLRM_HF_CACHE` (or `hf_cache_dir` in the config) to reuse a
  cache that is already populated.
- **Results.** diffusion_bench writes its usual folder,
  `$RESULT_DIR/diffusion_dlrm/<gpu>_<host>_<time>/` (per-step logs,
  `status.tsv`, `summary.md`), and the runner keeps its own log next to the
  others, `diffusion_dlrm-<config>-<hash>-rep<r>_<stamp>.log`. The batch
  summary lists each step's outcome, and `results.csv` gets a row per model
  (see "Extracting results"). The run counts as failed when any step did,
  the setup and download steps included, even if every model then ran.
- **Image.** `DOCKER_IMAGE` must contain `/workspace/AMDiffusionBenchmark`
  and `/workspace/DLRMBenchmark`, as `rocm/primus` images do; this is checked
  before the batch starts.
- Catalog mode, single node only. With `CONFIG_DIR`, `BACKEND` is ignored and
  so is `diffusion_dlrm` (the runner warns); `run_batch_multinode.sh` refuses
  it. It also needs `PRIMUS_MODE=container`, since diffusion_bench starts its
  own container.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Never defaulted, so it cannot be baked into a shared script. |
| `DOCKER_IMAGE` | — | **Required.** Which build to benchmark; a stale default would silently measure the wrong one. |
| `BACKEND` | — | **Required.** Comma-separated backends, e.g. `megatron,torchtitan`; `diffusion_dlrm` adds the diffusion and DLRM benchmarks. |
| `GPU` | auto-detected | Catalog top-level key, e.g. `MI325X`. Read from `amd-smi` or `rocm-smi` when unset. |
| `DRY_RUN` | `0` | `1` prints the plan and exits. |
| `MAX_RUN_SECONDS` | `3600` | Stop a run still going after this long. `0` = no limit. |
| `MAX_IDLE_SECONDS` | `1200` | Stop a run whose log has not grown for this long. `0` = no limit. |
| `POLL_INTERVAL` | `15` | Seconds between checks of the run log. |
| `GPU_BUSY_MIB` | `2048` | A GPU with more memory in use than this counts as busy. |
| `GPU_BUSY_WAIT_SECONDS` | `60` | How long to wait for busy GPUs to free up before refusing. |
| `ALLOW_BUSY_GPUS` | `0` | `1` runs even when the GPUs are busy. |
| `FAILING_RANK_OUTPUT` | `1` | `0` does not copy the failing rank's own log into the log of a failed run (see "Time limits and failure reasons"). |
| `EXTRACT` | `1` | `0` skips refreshing `results.csv` / `results.json` at the end of a batch. |
| `REPORT` | `1` | `0` skips writing `report.xlsx` at the end of a batch (see "Excel report"); `EXTRACT=0` skips it too. |
| `PULL_IMAGE` | `0` | `1` (or `true` / `yes`) pulls `DOCKER_IMAGE` on every node before the batch even if the tag is present. |
| `RESULT_DIR` | `$PWD/primus-perf-<date>` | Where logs and snapshots go; created if missing. |
| `CONFIG_FILE` | `tools/perf/configs.yaml` | Catalog to read. |
| `CONFIG_DIR` | — | Switches to directory mode. |
| `NUM_REPS` | `1` | Repetitions per config. |
| `TRAIN_STEPS` | whatever the YAML says | Caps training length when set. Not applied to `megatron_bridge`, whose configs keep their own `train_iters` (their learning-rate warmup and decay are sized to it, and a run no longer than the warmup fails Megatron's `lr_warmup_steps < lr_decay_steps` check), nor to `diffusion_dlrm`, which keeps diffusion_bench's `num_iterations`. A Megatron config for which it is not longer than the warmup is not run (see "Before each run"). |
| `CHECK_TRAIN_STEPS` | `1` | `0` runs a Megatron config even when `TRAIN_STEPS` is not longer than its warmup. |
| `EXTRA_ENV` | — | Space-separated `KEY=VALUE` pairs forwarded into the container as `--env`. See below. |
| `DATA_PATH` | `<checkout>/data` | Where Primus keeps its Hugging Face downloads (models, datasets). Set it to keep them out of the checkout; the runner creates it, mounts it and passes it into the container. See below. |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | auto (single node); `eno0` (multi-node) | RCCL/Gloo bootstrap NIC. Set even for 1-node 8-GPU; must not be `docker0`. See below. |
| `NNODES`, `GPUS_PER_NODE` | `1`, `8` | Cluster topology. |
| `PRIMUS_MODE` | `container` / `slurm srun` | Launcher mode. |
| `DIFFUSION_DLRM_CONFIG` | `tools/perf/diffusion_bench/config.yaml` | diffusion_bench config for `diffusion_dlrm`. |
| `DIFFUSION_DLRM_HF_CACHE` | `hf_cache_dir` of that config | Hugging Face cache for the diffusion models. |

### Which backends go with which image

Training runs on one of two images, and each carries only one framework stack:

| Image | Backends |
|---|---|
| `...utd/ci:primus_the_rock_ci_<sha>_<date>` (PyTorch) | `megatron`, `torchtitan` |
| `...utd/ci:jax_the_rock_ci_<sha>_<date>` (JAX) | `maxtext`, `maxdiffusion` |

So `BACKEND=megatron,torchtitan` and `BACKEND=maxtext,maxdiffusion` are the
two valid groupings. Mixing across the split — `BACKEND=megatron,maxtext` —
cannot work whichever image you pick, because half the runs have no framework
to execute. `megatron_bridge` and `diffusion_dlrm` go with the PyTorch side,
on an image that carries them, such as `rocm/primus:v26.8`.

This is why `BACKEND` is required rather than defaulting to every backend
listed for the GPU: there is no image for which "all" is correct, so a default
would only ever queue runs doomed to fail.

`GPU` is the opposite case: the node already knows what it is, so when unset
the runner reads the part number out of the GPU's marketing name from
`amd-smi` (or `rocm-smi`) (`AMD Instinct MI325X` → `MI325X`) and says so:

```
[INFO] GPU not set; detected MI325X (AMD Instinct MI325X).
```

Set it explicitly to benchmark one device's configs on another. When neither
tool is available the runner errors and lists the catalog keys rather than
guessing.

### Extra environment variables

Use `EXTRA_ENV` — a space-separated list of `KEY=VALUE` pairs:

```bash
EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0 GPU_MAX_HW_QUEUES=2" bash tools/perf/run_batch.sh
```

**Do not rely on plain `export` for this.** Exporting a variable in your shell
does not necessarily reach the training container. primus-cli forwards only
two categories:

- names listed in [runner/.primus.yaml](../../runner/.primus.yaml) under
  `container.options.env` (`HSA_NO_SCRATCH_RECLAIM`, `NVTE_CK_IS_V3_ATOMIC_FP32`,
  `GPU_MAX_HW_QUEUES`, `REBUILD_BNXT`, and about 40 more), and
- anything matching the prefixes `PRIMUS_`, `NCCL_`, `RCCL_`, `GLOO_`,
  `IONIC_`, `HIPBLASLT_`.

Anything else is dropped, silently. `DEBUG_HIP_DYNAMIC_QUEUES` is neither
listed nor prefix-matched, so `export DEBUG_HIP_DYNAMIC_QUEUES=0` before a run
has no effect inside the container even though it shows up in your shell and
in `batch_env_<stamp>.txt`.

`EXTRA_ENV` sidesteps the filter by turning each pair into an explicit
`--env KEY=VALUE` on the launcher, which primus-cli always honours. It is also
set in the runner's own shell, so it reaches direct (non-container) mode too
and is recorded in the banner:

```
# Extra env (--env): DEBUG_HIP_DYNAMIC_QUEUES=0 GPU_MAX_HW_QUEUES=2
```

That line sits apart from the general `Perf environment` block precisely
because these are guaranteed to have reached the container, while the others
were merely present on the host.

The alternative — adding a name to `container.options.env` in
`runner/.primus.yaml` — is the right move when a variable should always be
forwarded for everyone, rather than for one batch.

### Where downloads go (`DATA_PATH`)

Primus downloads Hugging Face models and datasets into
`$DATA_PATH/huggingface` (its `HF_HOME`). Unset, `DATA_PATH` is
`<checkout>/data`, which fills a home directory with a quota quickly: the
Qwen3-32B checkpoint the `megatron_bridge` configs load is 62 GB. Point it
somewhere bigger:

```bash
DATA_PATH=/tmp/primus-data bash tools/perf/run_batch.sh
```

The container sees only the checkout, and primus-cli does not forward
`DATA_PATH`, so the runner adds `--volume $DATA_PATH:$DATA_PATH` and
`--env DATA_PATH=$DATA_PATH` to every training launch (`diffusion_dlrm` runs
keep their own container and cache, `DIFFUSION_DLRM_HF_CACHE`). It leaves the
mount out when the path is inside the checkout or already a volume of the
config file primus-cli loads (`~/.primus.yaml` if it exists, otherwise
`runner/.primus.yaml`), since docker refuses the same mount twice. Before the
first run it creates the directory on every node, so it belongs to you
rather than to root, and the batch configuration records the free space:

```
# DATA_PATH    : /tmp/primus-data (4.4T free on node-1; mounted by the runner)
```

On a multi-node batch the path is the same on every node. A node-local path
(`/tmp/...`) means each node downloads its own copy; a shared filesystem
downloads once, provided root can write there (the container runs as root,
so the files it writes are owned by root). Run logs stay in the checkout
(`PRIMUS_WORKSPACE`, `output/` by default); they are small.

A `DATA_PATH` that `EXTRA_ENV` sets is passed in as before but not mounted:
mount it yourself, as a volume in `.primus.yaml`. Setting it both ways is
refused unless the two agree. A `DATA_PATH` exported in your shell for some
other reason now takes effect too: in container mode primus-cli used to drop
it.

Some small files stay in `<checkout>/data` regardless:
`runner/helpers/hooks/train/pretrain/prepare_experiment.sh` hands the
pre-train hooks `./data/` rather than `DATA_PATH`, so their pip cache,
torchtitan's tokenizer files and any dataset they build (none in the catalog:
its Megatron configs use mock data and its MaxText and MaxDiffusion configs
synthetic data) land there. Their downloads through `HF_HOME` do not.

### RCCL socket interface (single node included)

RCCL still needs a usable bootstrap NIC on a **single node**. A job with
`world_size=1` and eight local devices still creates communicators for
intra-node collectives (including MaxText expert-parallel AllToAll). If
bootstrap is pinned to `docker0`, training dies at the first collective with:

```
RCCL operation ncclGetUniqueId(&id) failed: invalid usage
Last RCCL warning: 'Bootstrap : no socket interface found'
```

The launcher does not leave this unset. `base_env.sh` fills
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` from
[`get_ip_interface.sh`](../../runner/helpers/envs/get_ip_interface.sh), which
maps the first IPv4 from `hostname -I` to an interface. On some nodes that
address belongs to **docker0**.
[`10_auto_nccl_net.sh`](../../runner/helpers/hooks/10_auto_nccl_net.sh) would
skip `docker0` / `lo` / `veth*`, but it never runs once `base_env.sh` has
already set the variables.

The runner guards against this before the first run. An explicit
`NCCL_SOCKET_IFNAME` or `GLOO_SOCKET_IFNAME` that selects `lo`, `docker*`,
`br-*`, `veth*` or another bridge is refused (comma lists are checked entry
by entry; a `^` exclusion list is accepted). When they are unset and Primus's
own pick would be such an interface, the runner exports the default-route
interface instead and logs it in the summary.

To choose the interface yourself, export it before `run_batch.sh`. `NCCL_*`
and `GLOO_*` are prefix-forwarded into the container, so a plain export is
enough:

```bash
export NCCL_SOCKET_IFNAME=eno0          # whatever `ip -br addr` shows, not docker0
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN                  # optional: confirm UniqueId / bootstrap
```

### Multi-node extras

Multi-node additionally honours `MAX_ATTEMPTS_PER_CONFIG` (default 6) and
`FAIL_PATTERN` for its retry loop: an attempt whose output matches
`FAIL_PATTERN`, or that hits one of the run limits below, is stopped and
retried. A config that has not succeeded yet is not retried after a failure
that would recur, so it does not spend all its attempts on it: a
`python_error` during setup whose exception, as the traceback that stopped
the run shows it, comes from the config or the code (`AssertionError`,
`ValueError`, `KeyError`, `TypeError`, `AttributeError`, `IndexError`,
`ZeroDivisionError`, `NotImplementedError`, `ModuleNotFoundError`, or an
omegaconf, hydra or yaml error), a `nic_error`, or a second `oom` in a row
(see "Extracting results"). A `RuntimeError`, `OSError`, `ImportError` or
distributed error during setup is still retried, as a port, rendezvous,
network or file system hiccup raises those too, and so is any error whose
message mentions a connection, a timeout or an unavailable service. The summary then
says `not retried: another attempt would fail the same way`, and the
config's `GAVE UP` line `(not retried: the failure would recur)`.
`RETRY_ALL_FAILURES=1` retries those too. `run_batch_multinode.sh` defaults `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
to `eno0` (Fremont); override them the same way on any other site. The
single-node `docker0` failure above applies here too if the wrong NIC is
pinned.

## Before each run

These checks run on every node before the first run, and the GPU part again
before every later run. A failed check at the start refuses the batch with
exit code 2 before anything launches; a failed GPU check later stops the
batch and leaves the remaining runs unstarted.

- **Image present, and the same everywhere.** A missing image is pulled once,
  up front, so the pull neither counts against the run limits nor leaves the
  image digest unknown (`PULL_IMAGE=1` pulls even when the tag is present).
  For a private image, export `DOCKER_LOGIN_USER` and `DOCKER_LOGIN_KEY` (and
  `DOCKER_LOGIN_REGISTRY` if not Docker Hub), as for `primus-cli`; the runner
  logs in before pulling.
  Multi-node batches refuse to start when the nodes hold different images
  under the same tag.
- **Leftover containers stopped.** Every container this tool starts carries
  the label `primus.perf=<user>/<batch stamp>/<host>/<pid>`. Containers that
  earlier batches of the same user left running (after a SIGKILL, a scheduler
  time limit or a lost session) are stopped. Other users' containers, and
  containers whose batch is still running on this host, are never touched.
  "Still running" can only be checked on the host that ran the batch: on the
  other nodes of a multi-node batch, any earlier batch of yours counts as
  left over.
- **GPUs idle.** No GPU may have more than `GPU_BUSY_MIB` of memory in use
  (idle MI355X GPUs show about 280 MiB). After `GPU_BUSY_WAIT_SECONDS` the
  batch refuses and lists the busy GPUs and the running containers.
  `ALLOW_BUSY_GPUS=1` overrides this. Memory is read with `amd-smi` (or
  `rocm-smi`), which only sees the GPUs this job was given. A Slurm job step
  that was not given the GPUs (how the multi-node check reaches the other
  nodes) sees none that way, so the amdgpu driver's counters in sysfs are the
  fallback; those cover every GPU on the node, so on a node shared with
  another job, its GPUs count too. When nothing is readable the check is
  recorded as "not checked" and does not block.
- **NCCL interface usable.** See the socket interface section above.
- **diffusion_dlrm ready**, when selected. Its config is validated before
  anything else (exit 2 on a bad key or model name), the image must carry
  the benchmark repos it needs, and the Hugging Face cache must be writable.

The result of the GPU check is recorded in every run's banner (`GPU check`).

Before each config, its `TRAIN_STEPS` is checked against its learning-rate
schedule. With `lr_decay_iters` unset, Megatron decays over `train_iters`
and asserts `lr_warmup_steps < lr_decay_steps`, so a `TRAIN_STEPS` no longer
than `lr_warmup_iters` fails during setup: the catalog's Megatron configs
warm up for 2 iterations, and a config that does not set it gets Primus's
default of 40. Such a config is not launched. It still gets a log for each
repetition, with exit code 2 and the failure reason `not_run: TRAIN_STEPS=10
is not more than lr_warmup_iters=40, ...`, and the summary a `[NOT RUN]`
line; `DRY_RUN=1` shows it too. Each value comes from the config's
overrides, else from its module config (`primus/configs/modules/...`); a
schedule set by `lr_warmup_fraction` or by samples is not checked.
`CHECK_TRAIN_STEPS=0` runs such a config anyway. A config whose launch
command cannot be built gets a `not_run` log the same way.

## Time limits and failure reasons

Every run is launched in the background, in its own process group, and its
log is checked every `POLL_INTERVAL` seconds. A run is stopped when it is
still going after `MAX_RUN_SECONDS` (default 1 hour), or when its log has not
grown for `MAX_IDLE_SECONDS` (default 20 minutes), which is how a hang shows
up. Both apply to single-node and multi-node runs; set either to `0` to
disable it. Stopping a run kills its whole process group and then its
containers.

Every log ends with a footer that says why the run ended:

```
# Exit code        : 143
# Elapsed (sec)    : 1206
# Failure reason   : idle: no new output for MAX_IDLE_SECONDS=1200s
```

`Failure reason` is `none` for a successful run, otherwise `<category>:
<detail>` with category `timeout`, `idle`, `fail_pattern` (multi-node),
`signal` (e.g. `killed by SIGKILL`), `exit` or `interrupted`. A
`diffusion_dlrm` run has none of the runner's limits (diffusion_bench times
out each step itself) and, when some of its steps failed, the category
`steps`, e.g. `steps: 2 of 8 diffusion_bench steps not successful:
diffusion_hunyuan-video_bs3 oom, dlrm_FP32 timeout`.

The run output only has local rank 0's console (and, multi-node, the last
local rank of the last node's), so when another rank fails first, its error
is not there. When torchrun's summary names that rank (`Root Cause (first
observed failure)`), the runner copies its ERROR records from this run out
of Primus's own per-rank log,
`<workspace>/<team>/<user>/<exp>/logs/<module>/rank-<N>/error.log`, into the
log, between the run output and the footer:

```
########################## Begin Failing Rank Output ###########################
# Failing rank     : 2 (local rank 2) on node-1.cluster.local, exit code 1
# Source           : .../output/amd/root/llama3.1_8B-pretrain/logs/pre_trainer/rank-2/error.log (its ERROR records from this run)
[20261022 18:52:27][root/amd][pre_trainer][ip-localhost][rank-2/8][ERROR]    [---traceback.py:1050] : Traceback (most recent call last):
...
########################### End Failing Rank Output ############################
```

The workspace is the config's `workspace` as the container saw it, with
`EXTRA_ENV` and the runner's `PRIMUS_*` variables (`PRIMUS_WORKSPACE`, say)
applied, and the experiment is the run's own. Primus appends every run of an
experiment to the same files, so only records stamped after the run started
are copied. When that rank logged no ERROR record, the section has the last
lines of its `debug.log` instead (a native crash, for one, leaves no Python
error), which are not read as the cause; when no log of that rank was
written during the run, it says where it looked. The records are the last
200 lines at most. Multi-node, each node looks for the file. This usually
takes under a second, and at most two minutes (three multi-node).
`FAILING_RANK_OUTPUT=0` turns it off.

The batch summary gives each failed run's cause under its `[FAIL]` line,
as `results.csv` will have it (unless it only repeats the `[FAIL]` line, as
for a timeout):

```
[20:22:04] [run 1/2] megatron/pretrain/.../fail_gbs_assert [7891462e] MBS=4 GBS=130 steps=10 ...
                    [FAIL] exit: launcher exited with code 1 exit=1 elapsed=36s
                           cause: python_error at setup: AssertionError: global batch size (130) is not divisible by ...
```

`python3 tools/perf/extract_results.py --verdict <log>` prints the same for
one log: its `status`, `failure_reason`, `failed_at`, `from_traceback`
(`yes` when the detail is the exception a traceback shows stopping the run)
and `failure_detail`, separated by `\x1f`.

Exit codes of both runners:

| Code | Meaning |
|---|---|
| `0` | Every run passed. |
| `1` | At least one run failed or was not started. |
| `2` | Bad input or a failed pre-run check; nothing was launched. |
| `130` | Interrupted. |

## What a batch writes

Into `$RESULT_DIR`:

| File | Contents |
|---|---|
| `<framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>_<stamp>.log` | Banner, full config dump, raw output, exit code, elapsed time and failure reason. |
| `diffusion_dlrm-<config>-<hash>-rep<r>_<stamp>.log`, `diffusion_dlrm/<gpu>_<host>_<time>/` | The `diffusion_dlrm` run: the same log layout, and diffusion_bench's results folder (per-step logs, `status.tsv`, `summary.md`). |
| `batch_summary_<stamp>.txt` | System info, the resolved config list with content hashes, per-run status. |
| `batch_env_<stamp>.txt` | Full environment, **secrets redacted**. |
| `batch_submodules_<stamp>.txt` | `git submodule status` — which vendored code ran. |
| `batch_gpu_<stamp>.txt` | `rocm-smi` (or `amd-smi`) clocks, power caps, topology. |
| `batch_stack_<stamp>.txt` | Image digest and ID, ROCm and HIP versions, library versions from inside the container. |
| `batch_host_<stamp>.txt` | Every node's platform as `key: value` lines (see "Platform" below), then the raw data behind them: `lscpu`, `numactl -H`, idle states, DIMMs, each GPU's sysfs values, RDMA ports, amdgpu module parameters, driver packages and repositories, Docker and network interfaces. Multi-node adds the settings that differ between nodes. |
| `batch_manifest_<stamp>/` | The image's `/workspace/.manifest` (build recipe, dpkg and pip lists), when it has one. |
| `run_batch_<stamp>.sh` | The driver as it was run, copied *before* the first run. |
| `results.csv`, `results.json` | Every log in the directory parsed by `extract_results.py`; refreshed at the end of each batch. |
| `report.xlsx` | The Excel report of the directory (see "Excel report"), when pandas and XlsxWriter are installed. |

`<stamp>` is shared by every artefact of one batch, so repeated batches in the
same directory accumulate instead of overwriting. Only `results.csv` and
`results.json` are rewritten each time, from all the logs together.

## Stopping a batch

Ctrl+C, `kill` (SIGTERM) or a closed terminal (SIGHUP) stops the run in
flight and abandons the rest of the batch instead of rolling on to the next
config. It also stops the training container, which matters because
`docker run` losing its client does not bring the container down on its own —
without this you would be left holding the GPUs.

The in-flight log still gets its footer (exit code 130, elapsed time and
`Failure reason: interrupted`) and the summary records where the batch
stopped, so an interrupted batch is still readable by `extract_results.py`.

Only containers this batch started are stopped: cleanup matches this batch's
`primus.perf` label exactly, on every allocated node for multi-node. The same
cleanup runs when a run hits a limit or a multi-node attempt fails, so the
GPUs are released before the next run starts.

diffusion_bench's container runs as root, so the files it writes are
root-owned until it hands them back at the end of its run. When the runner
stopped it first, the runner does that hand-back itself for the run's folder
and the root-owned files in the Hugging Face cache, so you can still delete
the results folder.

A SIGKILL (some schedulers send one at the time limit) cannot be handled.
The batch summary still shows which runs started and finished, the
interrupted log simply has no footer, and the container it leaves behind is
stopped by the next batch on that node (see "Before each run").

## Provenance

The per-run banner carries the fields that explain a number:

```
# Docker image     : rocm/primus:v26.8
# Image digest     : rocm/primus@sha256:79629b5e5ba4...
# Image ID         : sha256:fe8e1551d200...
# Primus commit    : 9a05d3fc9e5c-dirty
# Submodule pins   : b2ceb2bd
# GPU model        : AMD Instinct MI355X
# ROCm (host)      : 10.1.0
# Container ROCm   : 10.1.0
# Container HIP    : 7.16.26385-0000000
# Torch version    : 2.14.0+rocm10.1.0
# World size       : 8  (NNODES=1 x GPUS_PER_NODE=8)
# System           : Dual AMD EPYC 9575F 64-Core Processor-based Supermicro AS -4126GS-NMR-LCC server with 8x AMD Instinct MI355X (288GB HBM3E 1400W) GPUs, 1 NUMA node per socket, System BIOS 1.7, Ubuntu 22.04.5 LTS, Host GPU driver amdgpu 30.10.1 (ROCm 7.0.1)
# Server           : Supermicro AS -4126GS-NMR-LCC
# BIOS             : 1.7 (09/12/2025, American Megatrends International, LLC.)
# OS               : Ubuntu 22.04.5 LTS
# Kernel           : 6.8.0-65-generic
# Kernel cmdline   : ro panic=0 nowatchdog ... amd_iommu=on iommu=pt amdgpu.noretry=1 ...
# CPU              : AMD EPYC 9575F 64-Core Processor
# CPU topology     : 2 sockets x 64 cores x 2 threads (256 CPUs), 2 NUMA nodes
# CPU frequency    : acpi-cpufreq, boost on, max 3300 MHz
# CPU governor     : schedutil x256
# CPU idle states  : acpi_idle (menu): POLL C1 C2
# NUMA balancing   : 0
# Host memory      : 3023 GiB usable; installed 3072 GiB Registered-DDR5, 24 channels; DIMMs 128 GiB Samsung M321RAJA0MB2-CCPWF, 6400 MT/s, configured 6000 MT/s
# Transparent HP   : always
# IOMMU            : amd_iommu=on iommu=pt (default domain: identity)
# GPU VRAM         : 288 GiB
# GPU power cap    : 1400 W
# GPU partition    : SPX/NPS1
# GPU clocks       : perf level auto, max sclk 2400 MHz
# GPU VBIOS        : 113-M355-01-1K1-040C
# GPU PCIe         : 32.0 GT/s PCIe x16
# Host GPU driver  : amdgpu 30.10.1 (ROCm 7.0.1)
# Driver package   : amdgpu-dkms 1:6.14.14.30100100-2212064.22.04
# amdgpu driver    : 6.14.14
# RDMA NICs        : 8x ionic 400 Gb/sec (fw 1.117.5-a-147), 2x bnxt_en 200 Gb/sec (fw 233.0.207.0)
# Perf environment (host shell):
#   HSA_NO_SCRATCH_RECLAIM=1
#   NVTE_CK_IS_V3_ATOMIC_FP32=1
#   ...
```

A few notes on why these specific fields:

- The **digest** is recorded alongside the tag because a tag can be re-pushed;
  two runs with the same tag are not necessarily the same image. The **image
  ID** covers the case the digest misses: a CI image re-tagged locally as
  `rocm/primus:v26.8` still reports the CI repository's digest, which says
  nothing about the tag it ran under, but its ID identifies it. On nodes
  using the same docker image store, two runs used the same image exactly
  when their IDs match (docker's classic and containerd stores report
  different IDs for the same image). Multi-node batches refuse to start when
  the nodes report different IDs under the same tag; `PULL_IMAGE=1` (or
  `true` / `yes`) pulls the registry's copy on every node first.
- **ROCm** is reported twice, in every run banner and in the batch summary:
  the host's user-space install (`ROCm (host)`, from `ROCM_PATH` or
  `amd-smi`) and the container's, which is what the run actually used
  (`Container ROCm`, read from `/opt/rocm` or, in TheRock images, from the
  pip `rocm` package). The HIP version is a separate number, and the kernel
  driver is the `amdgpu driver` field. In the CSV they are `rocm_version`
  and `container_rocm_version`. With `PRIMUS_MODE=direct` the run does not
  use the image's ROCm, so `Container ROCm` only describes `DOCKER_IMAGE`.
- **Host settings** (CPU frequency governor, NUMA balancing, transparent huge
  pages, IOMMU mode, amdgpu driver) explain why one node is slower than
  another with the same image and code. The banner carries the node running
  the batch; `batch_host_<stamp>.txt` has every node.
- The **platform** is what a published number is quoted with. `System` is
  that sentence, written the way performance footnotes give it (CPUs and
  server, GPUs with their memory and power cap, NUMA nodes per socket, BIOS,
  OS, host GPU driver); the batch summary and the Excel Overview carry it
  too. The lines below it are the details that can move a number on their
  own: DIMM speed (the example's DIMMs are rated 6400 MT/s but run at 6000),
  boost and idle states, GPU power cap, compute and memory partition, clock
  limits, VBIOS and PCIe link, and the scale-out NICs and their firmware.
  Everything is read without root:
  - server, board and BIOS from DMI (`/sys/class/dmi/id`), placeholders
    such as "To Be Filled By O.E.M." left out;
  - CPU topology from `lscpu`, frequency driver, boost, EPP and idle states
    from `/sys/devices/system/cpu`;
  - installed memory and populated channels from EDAC, DIMM model and
    speeds from udev's copy of the DMI tables (which can miss some DIMMs,
    so they are named but not counted);
  - GPU memory, power cap, partition, clocks, VBIOS and PCIe link from the
    amdgpu driver's sysfs files, which every Slurm job step can read (amd-smi
    only sees the GPUs a step was given). The GPU's marketing name and
    memory type come from amd-smi on the node running the batch;
  - `Host GPU driver` from the `amdgpu-dkms` package (deb or rpm), whose
    version ends in the driver release (`...30100100` is amdgpu 30.10.1),
    and the ROCm release of the host's repo.radeon.com repository, which
    amdgpu-install points at the ROCm release that ships that driver. Older
    packages numbered the driver by ROCm release and show as `ROCm 6.4.1`.
    Without the package, it is the loaded module's version;
  - RDMA NICs from `/sys/class/infiniband`, active ports only.

  A value the GPUs or CPUs disagree on is a tally, e.g. `1400 W x7, 1000 W x1`.
  For multi-node, `Host check` says whether every node reported the same
  platform; when one does not, the batch summary warns with the keys that
  differ, and the end of `batch_host_<stamp>.txt` lists each of their values
  with the number of nodes that had it (the node sections above show which).
- The image's **build manifest** (`/workspace/.manifest`, including
  `docker-build-recipe.txt`) is copied once per batch, so the result directory
  says how its image was built.
- **Submodule pins** matter because Primus vendors ten submodules, and bumping
  one changes performance while leaving the Primus commit unchanged.
- **Library versions are read from inside the container**, since the image
  digest does not cover packages installed at runtime.
- The **perf environment** block is the runner's own shell, captured by
  prefix (`HSA_*`, `NCCL_*`, `XLA_*`, `TORCH_*`, `NVTE_*`, and so on) rather
  than as a hand-maintained list, which is always out of date. Anything
  matching `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*_KEY` or `AWS_*` is
  redacted. Inside the container, Primus prints its main environment
  settings, and each hook's exports, at the start of the run output; the
  GPU-specific file (`runner/helpers/envs/<GPU>.sh`) may set more than is
  echoed there.

## Extracting results

```bash
python3 tools/perf/extract_results.py "$RESULT_DIR"
```

Both runners do this automatically at the end of every batch (also after an
interrupt; `EXTRACT=0` skips it). It writes `results.csv` and `results.json`
into the directory, replacing the previous ones: they are derived from every
log there, including earlier batches, so re-running is always safe. The JSON
holds the same rows plus `parser_version`, the tools commit and the time it
was generated. Throughput uses the harmonic mean over post-warmup steps,
memory the arithmetic mean; at least the first three logged steps are always
dropped as compile/autotune. Multi-rank logs are filtered to a single rank so
the iteration count stays honest.

Alongside the metrics, each row carries `world_size`, `docker_image`,
`image_digest`, `primus_commit`, `submodule_pins`, `gpu_model`, `rocm_version`,
`torch_version`/`jax_version` and, for multi-node, `slurm_job_id`/`nodelist`.
Columns added later sit at the end of the CSV, so positional readers of older
files keep working:

| Column | Meaning |
|---|---|
| `status` | `ok`, `failed`, `interrupted`, `incomplete` (no footer: still running or killed), `unknown` (legacy log). |
| `failure_reason`, `failure_detail` | Why a non-`ok` run ended, and the log line behind it (see below). |
| `attempt`, `failed_attempt` | Multi-node attempt number; set on `-failed-attempt<n>` logs. Filter these out for performance numbers. |
| `config_path`, `batch_stamp`, `num_reps`, `gpu_check` | From the run banner. |
| `tps_per_gpu_*`, `tflops_*` (`min`, `max`, `stdev`, `cv`) | Spread over the post-warmup steps; tells a few slow steps from a uniformly slow run. |
| `version_warnings` | Distinct `... version mismatch ...` messages, e.g. aiter installed vs expected. |
| `parser_version` | Bumped when columns change. |
| `image_id`, `container_rocm_version`, `container_hip_version` | Exact image and the user-space ROCm/HIP inside it. |
| `cpu_governor`, `numa_balancing`, `transparent_hugepages`, `iommu`, `amdgpu_version` | Host settings of the node that ran the batch. |
| `rec_per_s`, `rec_per_s_cv` | DLRM recommendations/s and its std/mean, as DLRMBenchmark reports them (iterations/s × its per-GPU `BATCH_SIZE`). |
| `step_log` | For `diffusion` / `dlrm` rows, the diffusion_bench step log, relative to the result directory; empty for a step that never ran. |
| `system_description` | The one-line platform description (see "Provenance"). |
| `host_check` | Multi-node: whether every node reported the same platform. |
| `server`, `bios`, `os`, `kernel`, `kernel_cmdline` | Server vendor and model, BIOS version (date, vendor), OS, kernel and its command line. |
| `cpu_model`, `cpu_topology`, `cpu_frequency`, `cpu_idle`, `host_memory` | CPU model; sockets, cores, threads and NUMA nodes; frequency driver, boost, max clock and EPP; idle states; usable and installed memory and DIMMs. |
| `gpu_vram`, `gpu_power_cap`, `gpu_partition`, `gpu_clocks`, `gpu_vbios`, `gpu_driver`, `rdma_nics` | Per-GPU memory, power cap, compute/memory partition, perf level and max clock, VBIOS, the host GPU driver release, and the active RDMA ports. |
| `hmean_nominal_tflops`, `throughput_note` | Megatron-Bridge post-training without packed sequences: its `MODEL_TFLOP/s/GPU` estimate, kept under this name instead of `hmean_tflops`, and why (see below). |
| `failed_at` | For a training run that failed, was interrupted or is incomplete: `setup` when it logged no training step, otherwise `after step N`, the last step it logged, numbered as the backend numbers them (MaxText from 0). Empty for `no_metrics`, which finished. Empty for `diffusion_dlrm` runs and their `diffusion` / `dlrm` rows. |
| `failing_rank`, `failing_host` | The rank torchrun named as the first to fail, and its node, when the log has its section (see "Time limits and failure reasons"). |

Megatron-Bridge's `MODEL_TFLOP/s/GPU` is an estimate, not a measurement:
global batch × `seq_length` × the FLOPs of one training token, divided by the
step time, so it counts every sample as `seq_length` tokens. That holds for
pretraining and for post-training with `packed_sequence: true`. Without
packed sequences, post-training pads each micro-batch only to its longest
sample: SQuAD samples are a few hundred tokens against a `seq_length` of
8192, and the estimate exceeds the GPU's peak (about 14,000 TFLOP/s per GPU
for the Qwen3-32B LoRA recipe on MI355X). Those rows (`Train Module` is
`post_trainer` and the resolved `packed_sequence` is false, Megatron-Bridge's
default) therefore keep the estimate as *nominal* TFLOP/s
(`hmean_nominal_tflops`, "Nominal TFLOP/s per GPU" in the report: the rate
the step would reach if every sample were `seq_length` tokens) and leave the
TFLOP/s columns empty. Their throughput is samples/s per GPU, from each
iteration's global batch size and step time, and `throughput_note` says why.
Because the estimate is a fixed amount of work divided by the step time,
nominal TFLOP/s still compares runs of one config (same model, dataset,
batch size and `seq_length`). LoRA's estimate also counts the weight gradients its
frozen base weights never compute, so treat a packed LoRA run's TFLOP/s as
an upper bound.

A `diffusion_dlrm` log becomes one row per diffusion model and DLRM
precision, with backend `diffusion` or `dlrm`. Those numbers come from
diffusion_bench's own parser, so they match its `summary.md`: TFLOP/s and
samples/s per GPU are AMDiffusionBenchmark's averages, not this tool's
harmonic means, and `micro_batch_size` is the batch size per GPU (DLRM's
`BATCH_SIZE` is per GPU too). Their `exit_code`, `elapsed_sec`, `status` and
`failure_reason` are the step's:

- `timeout`: diffusion_bench stopped the step at its `step_timeout`;
  `killed`: it got SIGKILL before that (host out of memory, `docker kill`).
- `no_metrics`: the step succeeded without reporting a throughput. Such a
  row is `failed` even though the batch summary shows the run as passed.
- `config_error`: Hydra rejected `launcher.py`'s overrides; the detail is its
  message.
- `not_run`: the step never ran, e.g. because the setup step failed first
  (`interrupted` or `incomplete` instead, when the batch was stopped).
- Otherwise the step's own log is searched as described below.

Every step diffusion_bench planned gets a row, including those that left no
output folder, and a setup or download step that failed gets one too, with
backend `diffusion_dlrm`. When the config gives DLRM its own image
(`dlrm.image`), the DLRM rows name that image and leave the main image's
container versions empty.

The rules below live in [`failure.py`](./failure.py).

`failure_reason` takes the runner's verdict from the footer when it is
definitive (`timeout`, `idle`, `interrupted`, `not_run` for a config the
runner did not launch, and `steps` for a `diffusion_dlrm` run with no step
rows). Otherwise the run output is
searched, in this order, for `nic_error` (no usable socket interface), `oom`
(including RCCL's `Failed to CUDA calloc 536870912 bytes`, which PyTorch
reports as an "unhandled cuda error", unless the output shows the GPU had
faulted first), `gpu_contended` (out of memory while
this process held less than half the GPU, i.e. another process had it),
`segfault`, `nccl_error` and `python_error`.

The detail of `nccl_error` and `python_error` is the exception that stopped
the run, when the output has one (an NCCL one for `nccl_error`): the root of
the traceback chain printed just before Primus's `[Primus] Error:` line, or
of the last chain. That is back along `raise ... from` to the original
error, past Primus's `RuntimeError: Training execution failed` and torchrun's
`ChildFailedError`; a chain that ends in `KeyboardInterrupt` or `SystemExit`
does not count. A bare exception gets the location it was raised at, and a
bare `AssertionError` the failed assert too, e.g.
`AssertionError: assert self.lr_warmup_steps < self.lr_decay_steps
(optimizer_param_scheduler.py:74)`. Otherwise the detail is the first NCCL
line, or an exception line. An NCCL error's detail ends with the
`Last error:` that PyTorch quotes from NCCL, which says what failed, with
PyTorch's source location dropped to make room:
`torch.distributed.DistBackendError: NCCL error: unhandled system error,
NCCL version 2.27.7 | Last error: Call to ibv_modify_qp failed with 110
Connection timed out, on dev rdma4:1, ...`. Primus's own `[Patch]` messages are ignored,
along with the traceback a patch logs when it is skipped for a missing
dependency.

Failing all that, the reason is `killed` (SIGKILL with no error in the log,
e.g. the host OOM killer), `signal`, `fail_pattern`, `exit`, or `unknown`; a
failed run always has a reason. `incomplete` rows only get one when the
output shows a cause. A training run that exited 0 without logging a
training step is `failed` with `no_metrics`, even though the batch summary
shows it as passed.

When the log has the failing rank's own records (see "Time limits and
failure reasons"), they are searched first, and `failing_rank` and
`failing_host` say which rank that was. Without them the detail is local
rank 0's view of the failure, as the PyTorch backends' console shows no
other rank (but the last local rank of the last node, multi-node).

Logs written before these fields existed simply leave the columns empty.
Older logs still get a `status` and a `failure_reason`: from their exit code
(130 counts as `interrupted`), from the older multi-node footer
(`Failure pattern? : true`), and from their output.

## Excel report

```bash
python3 tools/perf/report.py "$RESULT_DIR"                       # -> $RESULT_DIR/report.xlsx
python3 tools/perf/report.py "$RESULT_DIR" --baseline ~/primus-bench/mi355x-v26.7
python3 tools/perf/report.py dir_a dir_b -o release.xlsx         # several batches in one report
```

`report.py` turns a result directory into a workbook that is easier to read
than `results.csv`:

| Sheet | Contents |
|---|---|
| Overview | One column per batch: outcome (with the errors of a refused batch), timing, image (tag, digest, ID), container ROCm/HIP and Torch, Primus commit and submodule pins, the platform (system description, GPU model, nodes and GPUs, server, BIOS, CPU, memory, GPU settings, host GPU driver, NICs, OS), host settings, run settings (reps, `TRAIN_STEPS`, limits, the `diffusion_dlrm` config), and run counts (planned, started, failed, not started). With `--baseline`, each baseline's image, commit and batches. |
| Performance | One row per unique backend, model, precision, config hash, sequence length, MBS, GBS, nodes, GPUs and image ID, with runs ok / total and the means over the successful runs of TFLOP/s per GPU and throughput (per GPU, and for all GPUs), with its unit: tokens/s, samples/s, frames/s or recommendations/s. Megatron-Bridge post-training without packed sequences shows nominal TFLOP/s instead of TFLOP/s, in its own column (see "Extracting results"). Also the step counts run, run-to-run CV, GPU memory and minutes per run. |
| Runs | Every run, with its status, failure reason, where it failed and the detail, throughput note, image, result directory and a link to its log: the table to build your own PivotTable on (Insert > PivotTable). |
| Chart | TFLOP/s per GPU for each config (and the baseline). |

- **Averages.** A config's numbers are the arithmetic mean over its
  successful runs (`NUM_REPS` repetitions, possibly across batches, and
  whatever step count each ran; the Steps column lists them, e.g. `10; 50`).
  Failed runs and failed multi-node attempts are removed before averaging;
  "runs ok / total" counts them and the Runs sheet says why each one
  failed. Runs with another image are a separate row, and
  the Image column appears when there are several images. Run-to-run CV
  above 3% is highlighted; "runs ok / total" is red when runs failed, amber
  when only retried multi-node attempts did.
- **Baseline.** `--baseline DIR` (repeatable), e.g. the previous release,
  adds the baseline's TFLOP/s and throughput and the change, red or green
  past `--threshold` percent (default 5). Configs are matched by backend,
  model, precision and GPU count; the Baseline config column says whether
  the config file was the same.
- **Inputs.** It reads each directory's `results.json` when that is current
  (newer than the logs, from this parser version), and otherwise parses the
  logs itself; it never writes into the input or baseline directories.
- **Automatic.** Both runners also write `report.xlsx` at the end of a
  batch, after `results.csv`, when its libraries are installed; `REPORT=0`
  skips it, and so does `EXTRACT=0`.
- **Dependencies.** pandas and XlsxWriter. A copy in
  `~/.cache/primus-perf/python<X.Y>-<platform>` is used first; the first
  `report.py` run installs whatever is missing or unusable there (about
  130 MB, no root, nothing else on the system changes). A batch never
  installs them itself, and `--no-install` turns this off. Without network
  access, install them into a Python of your own (`python3 -m venv ~/venv-perf
  && ~/venv-perf/bin/pip install pandas XlsxWriter`) and run `report.py` with it.

## Relationship to the other benchmark tools

- [`tools/auto_benchmark/`](../auto_benchmark/) is the *interactive* runner:
  single node, menu-driven, Megatron and TorchTitan only.
- [`tools/daily/daily_report.py`](../daily/daily_report.py) is the CI-side
  aggregator for the scheduled `benchmark.yaml` workflow; it derives its
  dimensions from a fixed `{date}/{gpu}/{framework}/` path layout.
- This tool is the non-interactive batch runner: five frameworks, pre- and
  post-training, container and Slurm, with provenance carried in the logs.
  [`diffusion_bench/`](./diffusion_bench/) runs the diffusion and DLRM
  benchmarks on its own too; `BACKEND=...,diffusion_dlrm` runs it as part of
  a batch.
