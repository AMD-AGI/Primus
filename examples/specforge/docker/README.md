# SpecForge ROCm runtime image

Image for [SpecForge on Primus](../README.md). That page is the `primus-cli`
entrypoint sketch; this file is the install reference.

The published `primus` wheel does **not** include `primus/backends/specforge/`.
This Dockerfile installs **this Primus checkout** editable with `--no-deps`.

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
| SpecForge | `pip install -e . --no-deps` at a pinned git SHA |
| Primus | `pip install -e /opt/primus --no-deps` (this tree, not the PyPI wheel) |
| Extra imports | Allowlist in `install-light-deps.sh`, each `--no-deps` |

Do **not** run `primus-cli deps sync` in this image. That path installs Megatron /
CUDA wheels and will clobber the ROCm stack.

## Build

Build from the **Primus repository root** (this repository). The trailing `.`
is the Docker build context; the Dockerfile copies `primus/`, `runner/`,
`primus_cli.py`, and `examples/specforge/` from that context into `/opt/primus`.
Do not use a SpecForge checkout as the context.

```bash
docker build -f examples/specforge/docker/Dockerfile \
  -t primus-specforge:v0.5.14-rocm700-mi35x .
```

Pins (`BASE_IMAGE` digest, `SPECFORGE_REF`) are in
[`pins.env`](pins.env) and as Dockerfile `ARG`s.

The Dockerfile:

1. Starts from `lmsysorg/sglang:v0.5.14-rocm700-mi35x`.
2. Clones SpecForge at `SPECFORGE_REF` and applies
   `patches/sglang/v0.5.14/spec-capture.patch`.
3. Installs SpecForge editable `--no-deps`.
4. Copies this Primus tree and installs it editable `--no-deps`.
5. Fills missing light packages (`accelerate`, `tensorboard`, …) `--no-deps`.
6. Runs GPU-free smoke checks (`enable_spec_capture`, `import specforge`,
   `primus-cli --help`, HIP torch).

Validated on AMD Instinct MI355X (gfx950). Newer SGLang ROCm tags may work if
they still ship AITER and the speculative algorithm you will serve.

For MI300X (gfx942), override the base tag to an `mi30x` image that still has
those pieces:

```bash
docker build -f examples/specforge/docker/Dockerfile \
  --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.14-rocm700-mi30x \
  -t primus-specforge:v0.5.14-rocm700-mi30x .
```

## Run

```bash
docker run -it --name primus-specforge \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g --network host \
  -e HF_HOME=/data/hf-cache \
  -e HF_TOKEN \
  -v /path/to/data:/data \
  primus-specforge:v0.5.14-rocm700-mi35x \
  bash
```

`--device=/dev/kfd --device=/dev/dri --group-add video` are required for ROCm.
`--ipc=host --shm-size=64g` is enough shared memory for multi-GPU capture/train.
Re-enter with `docker exec -it primus-specforge bash`.

Inside the image:

- Primus: `/opt/primus` (`WORKDIR`)
- SpecForge: `/workspace/SpecForge` (`SPECFORGE_ROOT`)
- Capture patch is already applied to the image SGLang

Set `HF_TOKEN` if the target model is gated. Point `HF_HOME` at a durable cache
so model weights are not re-downloaded.

Launch capture or train with `primus-cli` from `/opt/primus`. See
[SpecForge on Primus](../README.md).
