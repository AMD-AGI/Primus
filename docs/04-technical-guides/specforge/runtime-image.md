# SpecForge ROCm runtime image

Image for [Run SpecForge with Primus](running.md). That page is the `primus-cli`
entrypoint and MI355X results; YAML/env/CLI knobs are in
[configuration](configuration.md). This page is the install reference.

The published `primus` wheel does **not** include `primus/backends/specforge/`.
The [Dockerfile](https://github.com/AMD-AGI/Primus/blob/main/examples/specforge/docker/Dockerfile)
installs **this Primus checkout** editable with `--no-deps`.

For the specialized Muse SGLang base and the unmerged EAGLE4 implementation,
see the [Muse Spark EAGLE4 development image](muse-spark-eagle4-image.md).
That image has separate GPU-free checks and a documented capture limitation.

## What Primus owns

| Piece | Owner |
| --- | --- |
| ROCm SGLang + PyTorch | Official SGLang image (do not let pip replace it) |
| Draft architecture, data scripts, `specforge train` / `specforge export` | [SpecForge](https://github.com/sgl-project/SpecForge) |
| Capture / train launch, experiment YAML, ROCm stack preflight | Primus `framework: specforge` |

## Conflict policy

| Component | Install method |
| --- | --- |
| SGLang / torch | Keep the base image versions |
| SpecForge | `pip install -e . --no-deps --ignore-requires-python` at a pinned git SHA |
| Capture plugin | `pip install -e plugins/sglang-spec-capture --no-deps` from that same checkout |
| Primus | `pip install -e /opt/primus --no-deps` (this tree, not the PyPI wheel) |
| Extra imports | Allowlist in [`install-light-deps.sh`](https://github.com/AMD-AGI/Primus/blob/main/examples/specforge/docker/install-light-deps.sh), each `--no-deps` |
| tensorboard | Exception. The Dockerfile installs `tensorboard>=2.14,<2.21` with dependencies, constrained to versions already in the base image, so pip cannot upgrade numpy, protobuf, grpcio, setuptools, or pillow. `SummaryWriter` imports tensorboard's own tree, and a `--no-deps` copy leaves `--report-to tensorboard` broken. tensorboard 2.21 requires `protobuf>=6.31.1` and `grpcio>=1.74`; the upper bound keeps that release out so the constraint holds instead of upgrading the SGLang stack. |

Do **not** run `primus-cli deps sync` in this image. That path installs Megatron /
CUDA wheels and will clobber the ROCm stack.

## Build

Build from the **Primus repository root** (this repository). The trailing `.`
is the Docker build context; the Dockerfile copies `primus/`, `runner/`,
`primus_cli.py`, and `examples/specforge/` from that context into `/opt/primus`.
Do not use a SpecForge checkout as the context.

```bash
docker build -f examples/specforge/docker/Dockerfile \
  -t primus-specforge:v0.5.23-rocm720-mi35x-20261103 .
```

`BASE_IMAGE` defaults to the tested nightly tag
`lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261103` and `SPECFORGE_REF` to
`af1c3f3868e0ad1d1bc4a3bc4c577f2c9d3fa7b5`. Both are Dockerfile `ARG`
defaults; `--build-arg` overrides them. `BASE_IMAGE` includes the tested digest
below, so re-pushing the nightly tag does not change the default build.

The Dockerfile:

1. Starts from the tested SGLang ROCm nightly.
2. Clones SpecForge at `SPECFORGE_REF`.
3. Installs SpecForge and its SGLang capture plugin editable `--no-deps`.
   SpecForge also uses `--ignore-requires-python` for the tested base interpreter.
4. Copies this Primus tree and installs it editable `--no-deps`. It links
   `primus/runner` to the copied `runner/` directory so the packaged `primus-cli`
   entry point can find its Bash launcher. Runner scripts are normalized to LF.
5. Fills missing light packages (`accelerate`, `wandb`, …) `--no-deps`.
6. Installs `tensorboard>=2.14,<2.21` with dependencies, constrained to the
   base image's already-installed versions.
7. Runs GPU-free smoke checks (plugin entry-point discovery, forward-observer
   and deferred-completion APIs, capture server arguments, `import specforge`,
   a `SummaryWriter` event-file write, `primus-cli --help`, HIP torch).

On 2026-11-04, this image passed GPU-free build checks and a Primus Qwen3.5-4B
DFlash online smoke test on MI355X: one capture GPU, one trainer GPU, 55 captured
samples, and 20 training steps with finite losses. The server advertised the
plugin backend, and the producer, consumer, and sidecars shut down successfully.
Offline training also passed two steps using precomputed Qwen3.5-4B features.
The temporary offline bridge below also passed fresh capture of four shards,
resume without rewriting them, exact tensor parity between compressed and
uncompressed output, and two training steps from the freshly captured features.
This is functional validation; historical performance results predate the migration.
The tested base-image digest was
`sha256:4c9b1497e3793b2f5d216440688a53151ad67d656b8dcdc99311e79fbf734577`.
Only MI355X has been validated with this pin; MI300X/MI325X are unvalidated.
For another GPU architecture, select a compatible ROCm image with
AITER and the plugin extension APIs. Stock older images that only support the
capture patch are incompatible. Prefer an image digest when available.

Primus online mode enables `SPECFORGE_SPEC_CAPTURE=1` in each capture server and
adds `specforge_spec_capture` to an existing `SGLANG_PLUGINS` allowlist. It sends
`deployment.disaggregated.server_capture=plugin` to the producer and consumers,
which selects the plugin request format. Remove any explicit `server_capture=patch`
or `DISAGG_SERVER_CAPTURE=patch` setting before migrating. Primus owns the capture
flags; remove legacy capture flags from `sglang_extra_args`.

Preflight checks plugin capabilities and requires the SGLang version prefix
`0.5.23`. `PRIMUS_SPECFORGE_PIN_SGLANG` can override that prefix (an empty value
disables only the version check). Capability checks still apply to online mode.
Offline training does not require the server capture plugin. Default offline
capture uses it through the bridge below. The `0.5.23` version prefix alone
is insufficient: stock `v0.5.23` lacks the forward-observer/deferred-response
APIs from [SGLang PR #43450](https://github.com/sgl-project/sglang/pull/43450)
that this pinned nightly includes.

## Temporary offline capture bridge

The pinned SpecForge native offline
adapter requires `parallel_state_wrapper.ParallelState`, which this nightly
does not provide. Primus therefore defaults capture to
`primus.backends.specforge.offline_capture`, which replaces the preparation
script's target factory with its plugin client and Mooncake store. SpecForge
still owns tokenization, capture layers, loss masks, vocabulary mapping, shard
writing, compression, and resume. Only the worker's imported preparation module
is adapted; the SpecForge and SGLang source trees are unchanged.

Each local TP group owns a loopback SGLang server and Mooncake master. Capture
reads host tensors, frees the store objects after each batch, and closes sidecars
on success or failure. TP must fit within one node and divide `nproc_per_node`;
data parallel capture can span nodes. ROCr allocation masks are preserved.
An explicit `specforge_capture.script` continues to select a custom script.

Once SpecForge ships native plugin-backed offline preparation, update
`SPECFORGE_REF`, restore the upstream capture entry point in the argument builder,
and remove `offline_capture.py` and its bridge tests/help smoke check. No capture
YAML or shard-format migration should be needed.

## Run

```bash
docker run -it --name primus-specforge \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g --network host \
  -e HF_HOME=/data/hf-cache \
  -e HF_TOKEN \
  -v /path/to/data:/data \
  primus-specforge:v0.5.23-rocm720-mi35x-20261103 \
  bash
```

`--device=/dev/kfd --device=/dev/dri --group-add video` are required for ROCm.
`--ipc=host --shm-size=64g` is enough shared memory for multi-GPU capture/train.
Re-enter with `docker exec -it primus-specforge bash`.

From the host, `primus-cli container` is the same as `docker run` plus
`primus-cli direct` inside. ROCm devices are mapped by default; set
`--shm-size 64g` and this image (not the default `rocm/primus` tag):

```bash
./runner/primus-cli container --image primus-specforge:v0.5.23-rocm720-mi35x-20261103 \
  --shm-size 64g --volume /path/to/data:/data \
  -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
```

Inside the image:

- Primus: `/opt/primus` (`WORKDIR`)
- SpecForge: `/workspace/SpecForge` (`SPECFORGE_ROOT`)
- Capture plugin is installed in the SGLang environment

Set `HF_TOKEN` if the target model is gated. Point `HF_HOME` at a durable cache
so model weights are not re-downloaded.

Launch capture or train with `primus-cli` from `/opt/primus`. See
[Run SpecForge with Primus](running.md).

Capture and online SGLang on this image use AITER with the radix cache off.
Primus sets that at launch; you do not put it in YAML or export it.
