# SpecForge ROCm runtime image

Image for [Run SpecForge with Primus](running.md). That page is the `primus-cli`
entrypoint and MI355X results; YAML/env/CLI knobs are in
[configuration](configuration.md). This page is the install reference.

The published `primus` wheel does **not** include `primus/backends/specforge/`.
The [Dockerfile](https://github.com/AMD-AGI/Primus/blob/main/examples/specforge/docker/Dockerfile)
installs **this Primus checkout** editable with `--no-deps`.

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

Build from the **Primus repository root** using the shared helper:

```bash
uv run --no-project python examples/specforge/docker/build.py mi35x --build
uv run --no-project python examples/specforge/docker/build.py mi30x --build
```

[`images.json`](https://github.com/AMD-AGI/Primus/blob/main/examples/specforge/docker/images.json)
is the single source of truth for SGLang's version, ROCm tag, architecture-specific
base digests, and SpecForge's version and commit pin. Primus's version comes from
its existing `primus/__init__.py`; it is not duplicated in the manifest.

The helper supplies the Dockerfile build arguments in both CI and local builds.
It uses an allowlisted tar context from this repository and requires Docker with
BuildKit. A bare `docker build` no longer supplies the required pins automatically.
To inspect the resolved pins and default local tag without building:

```bash
uv run --no-project python examples/specforge/docker/build.py mi35x
```

The default tag combines Primus, SGLang, and SpecForge versions, the first 12
characters of the SpecForge commit, and the architecture variant:
`v<primus>-sglang<sglang>-specforge<specforge>-g<commit>-mi35x`.
The commit suffix distinguishes SpecForge snapshots that share a package version.
For the convenient local name used in this guide's run examples:

```bash
uv run --no-project python examples/specforge/docker/build.py mi35x --build \
  --tag primus-specforge:local-mi35x
```

The Dockerfile installs the SpecForge SGLang capture plugin and
checks that the SpecForge package version matches the manifest. It installs both
source trees with `--no-deps` (SpecForge also uses `--ignore-requires-python`), preserves the base ROCm stack, installs the constrained
tensorboard dependencies, and runs the existing GPU-free smoke checks.
Changing pins still requires rebuilding and validating the resulting runtime.

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

## CI image publication

The `Build SpecForge Images` workflow builds two separate images in
`docker.io/tasimage/primus-specforge`, following SGLang's architecture variants:

- `mi30x`: gfx942, using the corresponding pinned SGLang base image.
- `mi35x`: gfx950, using the corresponding pinned SGLang base image.

Both base digests are recorded in `examples/specforge/docker/images.json`.
The matrix uses the same Dockerfile and SpecForge source pin for both variants.
It streams the required files from the Primus checkout as the build context and
requires a Linux Docker runner with BuildKit enabled. No GPU is needed to build.

Relevant non-draft PRs and pushes to `main`, version-tag pushes, and manual
dispatches trigger the workflow. Draft PRs wait until marked ready for review.
The `ci:gpu` label does not gate image builds, matching the existing Primus image
workflow. This workflow only runs the Dockerfile's existing GPU-free checks;
it does not launch training or add a GPU test.

Main and version-tag builds publish the version-bearing tag described above.
PR builds publish `pr-<number>-<version-bearing-tag>` so they cannot overwrite
main/release aliases. Every build also publishes a unique tag ending in
`-run-<run-id>-<attempt>`; these are GitHub Actions identifiers, not software
versions. Manual branch runs outside `main` publish only that unique tag.
There is no architecture-neutral `latest` tag. Tags can move on rebuild;
use a registry digest to identify exact image contents.

The `specforge-image-mi30x` and `specforge-image-mi35x` workflow artifacts record
the base image, Primus revision, local image ID, and published registry digest.
Use the published `repository@sha256:...` reference for reproducible downstream
jobs; convenience tags can move on a later run. These artifacts describe image
builds, not GPU validation of either architecture.

Maintainers must provision `tasimage/primus-specforge` and grant the existing
`PRIMUS_DOCKER_HUB_TOKEN` permission to push and, for private images, pull it.
Trusted runs fail publication if that secret is missing. Fork and Dependabot
PRs build without publishing or receiving registry credentials. Registry login
uses a temporary Docker configuration removed when the publish step exits.

## Run

```bash
docker run -it --name primus-specforge \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g --network host \
  -e HF_HOME=/data/hf-cache \
  -e HF_TOKEN \
  -v /path/to/data:/data \
  primus-specforge:local-mi35x \
  bash
```

`--device=/dev/kfd --device=/dev/dri --group-add video` are required for ROCm.
`--ipc=host --shm-size=64g` is enough shared memory for multi-GPU capture/train.
Re-enter with `docker exec -it primus-specforge bash`.

From the host, `primus-cli container` is the same as `docker run` plus
`primus-cli direct` inside. ROCm devices are mapped by default; set
`--shm-size 64g` and this image (not the default `rocm/primus` tag):

```bash
./runner/primus-cli container --image primus-specforge:local-mi35x \
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
