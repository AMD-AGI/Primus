# Muse Spark EAGLE4 development image

This image combines the Muse Spark SGLang ROCm runtime with the EAGLE4
implementation in [SpecForge PR #1024](https://github.com/sgl-project/SpecForge/pull/1024)
and this Primus checkout. It contains the software environment; model weights,
a Muse EAGLE4 draft configuration, and training recipes are separate.

Build from the Primus repository root:

```bash
docker build -f examples/specforge/docker/Dockerfile.muse-spark-eagle4 \
  -t primus-specforge:muse-spark-1027-eagle4-1c253f04 .
```

The defaults pin:

- Base: `lmsysorg/sglang-rocm:muse-spark-open-1027` at digest
  `sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`.
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

## Capture API backport

The unmodified `1027` base lacks the generic SGLang APIs required by the
SpecForge capture plugin. This Dockerfile applies a version-specific
[API backport](../../../examples/specforge/docker/patches/README.md) derived
from SGLang PRs #43449 and #43450 before installing SpecForge. It adds target
auxiliary-state capture, forward observers, deferred responses, and a native
Muse EAGLE3 capture hook while preserving the Muse ROCm kernels and AITER patches.
SpecForge's plugin owns feature publication and metadata; there is no legacy
capture backend or patch fallback.

Primus selects the EAGLE3 capture convention for EAGLE3/EAGLE4 draft configs
and DFLASH for DFlash drafts. Capture layer IDs still come from the draft JSON.
The Muse hook captures zero-based decoder layer outputs and the final normalized
target state. The configured IDs must match the intended draft feature layout.
The target model and draft JSON are supplied at run time, outside the image.

## Validation

Build-time checks cover CLI imports, the HIP Torch build, a TensorBoard event
write, plugin entry-point discovery and extension APIs, EAGLE4 registration,
and a tiny EAGLE4 forward/backward pass on CPU. Missing capture APIs fail the
build. The tiny model checks software compatibility; it is not a Muse draft
architecture or model validation. The backport also passed 76 upstream CPU
checks for auxiliary capture, token-row alignment, forward observers, deferred
completion, cancellation, and failed publication.

The image sets `PRIMUS_SPECFORGE_PIN_SGLANG=0.5.21` to match its specialized
base. This changes the version expectation only; it does not bypass capture
capability checks or enable a patch fallback.

Read the installed revision and capability report without GPU access:

```bash
docker run --rm --entrypoint cat primus-specforge:muse-spark-1027-eagle4-1c253f04 \
  /opt/primus/muse-spark-eagle4-image.json
```

Repeat the GPU-free checks with:

```bash
docker run --rm --entrypoint bash primus-specforge:muse-spark-1027-eagle4-1c253f04 \
  /opt/primus/examples/specforge/docker/smoke-check-muse-eagle4.sh
```

The image was also validated on MI355X GPUs with the Muse Spark Open
816B-A42B Quark MXFP4 target and a Muse EAGLE4 draft configuration:

- Two TP4 capture replicas advertised the plugin and published finite BF16
  features through Mooncake. Layers `[1, 29, 57]` produced 24,576 auxiliary
  channels and 8,192 final-state channels; replica parity checks passed.
- Eight training workers completed 200 optimizer steps with an 8,192-token
  maximum sequence length. All recorded losses were finite. The first-position
  distillation loss changed from 11.832 at step 1 to 5.115 at step 200.
- The final checkpoint recorded step 200, strategy `eagle4`, and world size 8,
  with all eight rank-state files present. Capture and training exited cleanly.

This is a capture-to-training integration test, not an acceptance-rate or
inference-speed benchmark. Model weights, the draft JSON, and the training recipe
remain runtime inputs and are not bundled in the image.
