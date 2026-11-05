# Muse Spark EAGLE4 development image

This image combines the Muse Spark SGLang ROCm runtime with the EAGLE4
implementation in [SpecForge PR #1024](https://github.com/sgl-project/SpecForge/pull/1024)
and this Primus checkout. It contains the software environment; model weights,
a Muse EAGLE4 draft configuration, and training recipes are separate.

Build from the Primus repository root:

```bash
docker build -f examples/specforge/docker/Dockerfile.muse-spark-eagle4 \
  -t primus-specforge:muse-spark-eagle4-1c253f04 .
```

The defaults pin:

- Base: `lmsysorg/sglang-rocm:muse-spark-open` at digest
  `sha256:72065dfc0b87818ab415f96c0fc08848233d25f22b002ac8f28c33fad6ec82a4`.
- SpecForge repository: `https://github.com/yudigege86/SpecForge.git`.
- SpecForge commit: `1c253f04c5fd0a321753a9ead28739c0a29ff7a7`
  from the PR's `feat/eagle4-draft` branch.

`BASE_IMAGE`, `SPECFORGE_REPOSITORY`, and `SPECFORGE_REF` are build arguments.
The SpecForge fork is intentional because EAGLE4 has not merged upstream.
The Primus source and task branch belong to `AMD-AGI/Primus` directly.

SpecForge, its capture plugin, and Primus are installed editable with
`--no-deps`. SpecForge also uses `--ignore-requires-python` for the base's
Python 3.10 interpreter. The image preserves the Muse SGLang/Torch stack and
uses the same protected lightweight dependency and constrained TensorBoard
installation as the [generic runtime](runtime-image.md). Do not run
`primus-cli deps sync` in this environment.

## Validation and capture limitation

Build-time checks cover CLI imports, the HIP Torch build, a TensorBoard event
write, capture plugin entry-point discovery, EAGLE4 algorithm registration,
and a tiny EAGLE4 forward/backward pass on CPU. The tiny model checks software
compatibility; it is not a Muse draft architecture or model validation.

On 2026-11-04, both build-time and post-build GPU-free checks passed for these
pins. A comparison against the base confirmed unchanged installed versions of
SGLang, Torch, Transformers, Triton, NumPy, protobuf, grpcio, Pillow, and packaging.

The pinned base contains SGLang `0.5.21+gc722c0a1` and lacks
`ModelRunner.forward_observer` and `deferred_output.ResponseCompletion`.
Installing the capture plugin does not supply those server APIs. This image
does not patch or replace the Muse-specific SGLang source. Consequently,
**Primus online and plugin-backed offline capture are unavailable in this
image**. The normal capture preflight still rejects this stack. A compatible
Muse runtime or a separately validated API backport is needed before capture.
Training from existing features is not blocked by these capture checks.

The image sets `PRIMUS_SPECFORGE_PIN_SGLANG=0.5.21` to match its specialized
base. This changes the version expectation only; it does not bypass capture
capability checks or enable a patch fallback.

Read the installed revision and capability report without GPU access:

```bash
docker run --rm --entrypoint cat primus-specforge:muse-spark-eagle4-1c253f04 \
  /opt/primus/muse-spark-eagle4-image.json
```

Repeat the GPU-free checks with:

```bash
docker run --rm --entrypoint bash primus-specforge:muse-spark-eagle4-1c253f04 \
  /opt/primus/examples/specforge/docker/smoke-check-muse-eagle4.sh
```

No GPU capture, target-model load, or training workload is covered by these
image checks. Prepare the Muse draft configuration and recipe, resolve capture
compatibility, and validate those workloads separately.
