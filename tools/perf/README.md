# Primus Perf Batch Runner

Batch-runs Primus training benchmarks across Megatron, Megatron-Bridge,
TorchTitan, MaxText and MaxDiffusion, and extracts the results into a CSV that
records not just the throughput but the settings that produced it.

Two goals shape this tool:

- **The script is never edited.** Every setting comes from the environment, so
  starting a run is a handful of exports rather than a diff. Which models run
  is chosen in [`configs.yaml`](./configs.yaml): `GPU` and `BACKEND` narrow it
  to one backend's suite, and commenting entries in or out picks a subset of
  that — in the catalog itself, or in a copy. Running something outside the
  curated suite means adding its `examples/...` path to the catalog, which is
  the only case where a path gets typed.
- **Every number is attributable.** Each log carries the image digest,
  submodule pins, GPU model, ROCm and framework versions, world size and the
  full perf-relevant environment, and those land in the CSV as columns.

## Quick start

```bash
export HF_TOKEN=hf_...                                    # required
export DOCKER_IMAGE=unifiedtrainingdockers.azurecr.io/utd/ci:<tag>   # required
export BACKEND=maxtext,maxdiffusion                        # required
export RESULT_DIR=~/primus-bench/mi325x-v26.7

# GPU is auto-detected from rocm-smi; set it to override.
# GPU=MI325X

# RCCL still needs a real NIC on a single node. 
# NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME are auto-detected from get_ip_interface.sh,
# which maps the first IPv4 from hostname -I to an interface.
# On some nodes that address belongs to docker0.
# So, the following may need to be set manually.

# export NCCL_SOCKET_IFNAME=eno0                            # e.g. eno0 / ibp...
# export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

bash tools/perf/run_batch.sh                              # single node
python3 tools/perf/extract_results.py "$RESULT_DIR"       # logs -> CSV
```

Multi-node, from inside a Slurm allocation (`salloc -N 8 -p mi325x`):

```bash
bash tools/perf/run_batch_multinode.sh
```

The runner can be invoked from any directory.

## Dependencies

[`yq`](https://github.com/mikefarah/yq) v4+ is the only external requirement.
Everything else (`bash`, `docker`, `git`, `python3`, `rocm-smi`) is already
present on a node that can run Primus.

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

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Never defaulted, so it cannot be baked into a shared script. |
| `DOCKER_IMAGE` | — | **Required.** Which build to benchmark; a stale default would silently measure the wrong one. |
| `BACKEND` | — | **Required.** Comma-separated backends, e.g. `megatron,torchtitan`. |
| `GPU` | auto-detected | Catalog top-level key, e.g. `MI325X`. Read from `rocm-smi` when unset. |
| `RESULT_DIR` | `$PWD/primus-perf-<date>` | Where logs and snapshots go; created if missing. |
| `CONFIG_FILE` | `tools/perf/configs.yaml` | Catalog to read. |
| `CONFIG_DIR` | — | Switches to directory mode. |
| `NUM_REPS` | `1` | Repetitions per config. |
| `TRAIN_STEPS` | whatever the YAML says | Caps training length when set. |
| `EXTRA_ENV` | — | Space-separated `KEY=VALUE` pairs forwarded into the container as `--env`. See below. |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | auto (single node); `eno0` (multi-node) | RCCL/Gloo bootstrap NIC. Set even for 1-node 8-GPU; must not be `docker0`. See below. |
| `NNODES`, `GPUS_PER_NODE` | `1`, `8` | Cluster topology. |
| `PRIMUS_MODE` | `container` / `slurm srun` | Launcher mode. |

### Which backends go with which image

Training runs on one of two images, and each carries only one framework stack:

| Image | Backends |
|---|---|
| `...utd/ci:primus_the_rock_ci_<sha>_<date>` (PyTorch) | `megatron`, `torchtitan` |
| `...utd/ci:jax_the_rock_ci_<sha>_<date>` (JAX) | `maxtext`, `maxdiffusion` |

So `BACKEND=megatron,torchtitan` and `BACKEND=maxtext,maxdiffusion` are the
two valid groupings. Mixing across the split — `BACKEND=megatron,maxtext` —
cannot work whichever image you pick, because half the runs have no framework
to execute.

This is why `BACKEND` is required rather than defaulting to every backend
listed for the GPU: there is no image for which "all" is correct, so a default
would only ever queue runs doomed to fail.

`GPU` is the opposite case: the node already knows what it is, so when unset
the runner reads the part number out of the `rocm-smi` product name
(`AMD Instinct MI325X` → `MI325X`) and says so:

```
[INFO] GPU not set; detected MI325X from rocm-smi.
```

Set it explicitly to benchmark one device's configs on another. When
`rocm-smi` is unavailable the runner errors and lists the catalog keys rather
than guessing.

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

Export a real NIC before `run_batch.sh`. `NCCL_*` and `GLOO_*` are
prefix-forwarded into the container, so a plain export is enough:

```bash
export NCCL_SOCKET_IFNAME=eno0          # whatever `ip -br addr` shows, not docker0
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN                  # optional: confirm UniqueId / bootstrap
```

Check `hostname -I` and `ip -br addr` on the node you will run on. Do not
unset these hoping RCCL will pick: Primus will fill them from `hostname -I`
again, which is how `docker0` got selected in the first place.

### Multi-node extras

Multi-node additionally honours `MAX_ATTEMPTS_PER_CONFIG`, `POLL_INTERVAL`,
`MAX_RUN_SECONDS` and `FAIL_PATTERN` for its retry loop.
`run_batch_multinode.sh` defaults `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
to `eno0` (Fremont); override them the same way on any other site. The
single-node `docker0` failure above applies here too if the wrong NIC is
pinned.

## What a batch writes

Into `$RESULT_DIR`:

| File | Contents |
|---|---|
| `<framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>_<stamp>.log` | Banner, full config dump, raw output, exit code and elapsed time. |
| `batch_summary_<stamp>.txt` | System info, the resolved config list with content hashes, per-run status. |
| `batch_env_<stamp>.txt` | Full environment, **secrets redacted**. |
| `batch_submodules_<stamp>.txt` | `git submodule status` — which vendored code ran. |
| `batch_gpu_<stamp>.txt` | `rocm-smi` clocks, power caps, topology. |
| `batch_stack_<stamp>.txt` | Image digest plus library versions from inside the container. |
| `run_batch_<stamp>.sh` | The driver as it was run, copied *before* the first run. |

`<stamp>` is shared by every artefact of one batch, so repeated batches in the
same directory accumulate instead of overwriting.

## Stopping a batch

Ctrl+C stops the run in flight and abandons the rest of the batch instead of
rolling on to the next config. It also stops the training container, which
matters because `docker run` losing its client does not bring the container
down on its own — without this you would be left holding the GPUs.

The in-flight log still gets its footer (exit code 130 and elapsed time) and
the summary records where the batch stopped, so an interrupted batch is still
readable by `extract_results.py`.

Only containers this batch started are stopped. primus-cli names them
`primus-training-<SLURM_JOB_ID or pid>`, so there is no fixed name to match;
the runner snapshots which `primus-training*` containers were already up
before it launched anything and leaves those alone. On a shared node, a
colleague's run — or your own earlier one — survives your Ctrl+C. The snapshot
is echoed into the summary so you can see what was excluded.

Multi-node cleanup works the same way, and runs on every allocated node. Under
Slurm the container name is fully determined (`primus-training-$SLURM_JOB_ID`,
identical across the allocation and unique per job), so cleanup targets that
name exactly and can never touch another job's containers; the prefix-plus-
snapshot path is only the fallback for running outside an allocation. The name
being cleaned up is printed in the batch summary. The same cleanup runs on the
retry paths, so a failed or hung attempt releases the GPUs before the next one
starts.

## Provenance

The per-run banner carries the fields that explain a number:

```
# Docker image     : unifiedtrainingdockers.azurecr.io/utd/ci:primus_the_rock_ci_b154fd0_20260903
# Image digest     : unifiedtrainingdockers.azurecr.io/utd/ci@sha256:ec9992a8065d...
# Primus commit    : 137d6c256119
# Submodule pins   : 43f426fa
# GPU model        : AMD Instinct MI325X
# ROCm version     : 7.2.1
# Torch version    : 2.12.0+rocm10.0.0
# World size       : 8  (NNODES=1 x GPUS_PER_NODE=8)
# Perf environment :
#   HSA_NO_SCRATCH_RECLAIM=1
#   NVTE_CK_IS_V3_ATOMIC_FP32=1
#   ...
```

A few notes on why these specific fields:

- The **digest** is recorded alongside the tag because a tag can be re-pushed;
  two runs with the same tag are not necessarily the same image.
- **Submodule pins** matter because Primus vendors ten submodules, and bumping
  one changes performance while leaving the Primus commit unchanged.
- **Library versions are read from inside the container**, since the image
  digest does not cover packages installed at runtime.
- The **perf environment** is captured by prefix (`HSA_*`, `NCCL_*`, `XLA_*`,
  `TORCH_*`, `NVTE_*`, and so on) rather than as a hand-maintained list, which
  is always out of date. Anything matching `*TOKEN*`, `*SECRET*`, `*PASSWORD*`,
  `*_KEY` or `AWS_*` is redacted.

## Extracting results

```bash
python3 tools/perf/extract_results.py "$RESULT_DIR"
```

Writes `benchmark_results_<timestamp>.csv` into that directory. Throughput uses
the harmonic mean over post-warmup steps, memory the arithmetic mean; at least
the first three logged steps are always dropped as compile/autotune. Multi-rank
logs are filtered to a single rank so the iteration count stays honest.

Alongside the metrics, each row carries `world_size`, `docker_image`,
`image_digest`, `primus_commit`, `submodule_pins`, `gpu_model`, `rocm_version`,
`torch_version`/`jax_version` and, for multi-node, `slurm_job_id`/`nodelist`.
Logs written before these fields existed simply leave the columns empty.

## Relationship to the other benchmark tools

- [`tools/auto_benchmark/`](../auto_benchmark/) is the *interactive* runner:
  single node, menu-driven, Megatron and TorchTitan only.
- [`tools/daily/daily_report.py`](../daily/daily_report.py) is the CI-side
  aggregator for the scheduled `benchmark.yaml` workflow; it derives its
  dimensions from a fixed `{date}/{gpu}/{framework}/` path layout.
- This tool is the non-interactive batch runner: five frameworks, pre- and
  post-training, container and Slurm, with provenance carried in the logs.
