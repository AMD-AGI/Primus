# Muse Spark EAGLE4 online training

Build the Muse runtime, install the host `primus-cli`, and launch capture and
training together with one Slurm command. The recipe uses two nodes with eight
MI355X GPUs each: two TP4/EP4 target replicas on the capture node and eight FSDP
workers on the trainer node.

The configuration files are:

- [Draft architecture](../../../examples/specforge/configs/muse-spark-eagle4.json):
  one EAGLE4 decoder layer, hidden size 8192, and capture layers `[1, 29, 57]`.
- [SpecForge training](../../../examples/specforge/configs/muse-spark-eagle4-training.yaml):
  the objective, data format, optimizer, and Mooncake settings.
- [Primus experiment](../../../examples/specforge/configs/muse-spark-eagle4-online.yaml):
  role assignment and runtime path overrides.
- [Host launcher](../../../examples/specforge/configs/muse-spark-eagle4-launcher.yaml):
  container devices, environment forwarding, and runtime options.

Model weights and training data are supplied separately. Bash, Python 3.10+,
`uv`, and a Slurm client are required on the submission host. Execution nodes
need Docker access, ROCm devices, and network connectivity between the roles.
The checkout and host Python environment must be accessible at the same paths
on the execution nodes.

## Check out Primus and install the host CLI

Use a shared checkout so the launcher and recipes are visible on both nodes:

```bash
cd /path/to/shared/checkouts
git clone --branch codex/muse-spark-eagle4-image \
  https://github.com/AMD-AGI/Primus.git
cd Primus
ln -s ../runner primus/runner
uv venv .venv-primus-cli
uv pip install --python .venv-primus-cli/bin/python --no-deps -e . PyYAML
source .venv-primus-cli/bin/activate
primus-cli --help
```

The runner link provides the editable-install layout used by the image as well.
This installs the CLI from this checkout without installing the training stack
on the host. The container supplies SpecForge, SGLang, and HIP Torch. Do not run
`primus-cli deps sync` for this image.

## Build the image

On a Linux Docker build host, from the same Primus checkout:

```bash
export IMAGE=primus-specforge:muse-spark-1027-eagle4-1c253f04
docker build --network host \
  -f examples/specforge/docker/Dockerfile.muse-spark-eagle4 \
  -t "$IMAGE" .
```

Make this image tag available on both execution nodes before launching. Image
distribution is handled by your container registry or node provisioning process.

The Dockerfile pins:

- `lmsysorg/sglang-rocm:muse-spark-open-1027` at digest
  `sha256:7184d3ae2bfdc75a5bb8eb497b768e370ef7b213879cb4d97fbaaa3e8e5f5040`.
- The EAGLE4 implementation from [SpecForge PR #1024](https://github.com/sgl-project/SpecForge/pull/1024),
  repository `https://github.com/yudigege86/SpecForge.git`, commit
  `1c253f04c5fd0a321753a9ead28739c0a29ff7a7`.

`BASE_IMAGE`, `SPECFORGE_REPOSITORY`, and `SPECFORGE_REF` are build arguments.
SpecForge, its capture plugin, and Primus are installed editable with `--no-deps`.
SpecForge also uses `--ignore-requires-python` for the base's Python 3.10.
The dependency protections preserve the Muse SGLang/Torch stack while adding
lightweight requirements and TensorBoard.

## Prepare paths and data

Set these variables in the submission shell. Model and shared-data directories
must be available at the same absolute paths on both nodes; the local-state
root may be backed by each node's local disk.

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/shared/data
export LOCAL_STATE_ROOT=/path/to/node-local/specforge
export TARGET_MODEL=$MODEL_ROOT/Muse-Spark-Open-816B-A42B-Quark-MXFP4
export TRAIN_DATA_PATH=$DATA_ROOT/train.jsonl
export RUN_ID=muse-eagle4-$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT=$DATA_ROOT/runs/$RUN_ID
export OUTPUT_DIR=$RUN_ROOT/output
export CONSUMER_STATE_DIR=$LOCAL_STATE_ROOT/$RUN_ID/consumer-state
export MAX_STEPS=200
mkdir -p "$RUN_ROOT"
```

`TARGET_MODEL` is the local target snapshot directory, including its weights,
configuration, and tokenizer files. The input is JSONL with a `text` field in
each row. Text must already be formatted with the target's Muse chat template
and include an assistant response: the recipe sets `is_preformatted: true`.

Defaults are batch size 1 per trainer GPU, accumulation 1, a maximum sequence
length of 8192, and checkpoints every 50 steps. `MAX_STEPS`, `NUM_EPOCHS`, and
`SAVE_INTERVAL` can override their recipe defaults. Supply enough valid samples
for the requested steps, or increase `NUM_EPOCHS`; `max_steps` is an upper bound.

Use a fresh `RUN_ID` and state paths for every launch. The local-state mount
must be writable, and Primus creates the consumer-state leaf directory itself.

## Launch both roles with Primus

From the shared Primus checkout, with the host CLI environment active:

```bash
export IMAGE=primus-specforge:muse-spark-1027-eagle4-1c253f04
primus-cli --config examples/specforge/configs/muse-spark-eagle4-launcher.yaml \
  slurm srun -N 2 -n 2 --gres=gpu:8 --cpus-per-task=32 --mem=0 \
  -- container --image "$IMAGE" \
  --volume "$MODEL_ROOT:$MODEL_ROOT:ro" \
  --volume "$DATA_ROOT:$DATA_ROOT" \
  --volume "$LOCAL_STATE_ROOT:$LOCAL_STATE_ROOT" \
  -- train pretrain \
  --config examples/specforge/configs/muse-spark-eagle4-online.yaml
```

Add your scheduler's allocation options as needed. The launcher config requests
a two-hour wall time; adjust it for your workload. Each Slurm task runs one
Primus supervisor. Primus starts Mooncake, the two capture servers, and the
producer on rank 0, then starts the eight-worker consumer on rank 1. You do not
launch producer and consumer scripts separately. The checkout is bind-mounted
automatically, so configuration paths resolve against the same source tree.

Automatic interface selection is used by default. If necessary, export
`PRIMUS_SPECFORGE_BIND_IFACE` with a reachable interface name before launching;
the launcher forwards `PRIMUS_*` and common communication environment variables.
The recipe transports captured features with Mooncake TCP.

The example requests all host memory on its allocated nodes (`--mem=0`) for
target loading and Mooncake buffers. Adjust the memory request to your site's
resource policy and the available host RAM.

Coordination and role logs are under `$RUN_ROOT/control`, including
`producer.log`, `consumer.log`, and capture-server logs. Checkpoints and
TensorBoard events are under `$OUTPUT_DIR`. TensorBoard is enabled by
`tracking.report_to: tensorboard` in the training YAML.

## Capture API compatibility

The unmodified `1027` base lacks the extension APIs required by the SpecForge
capture plugin. The Dockerfile applies a version-specific
[API backport](../../../examples/specforge/docker/patches/README.md) before
installing SpecForge. It preserves the Muse ROCm kernels and AITER patches;
SpecForge's plugin owns feature publication and capture metadata.

EAGLE4 uses the EAGLE3 capture convention without loading an inference-time
draft model. The draft JSON's layer IDs `[1, 29, 57]` are zero-based decoder-layer
outputs, captured after the 2nd, 30th, and 58th layers. Primus resolves these IDs
and enables the plugin only for capture-server processes.

The image sets `PRIMUS_SPECFORGE_PIN_SGLANG=0.5.21` for this specialized base.
This changes the version expectation only; capture capability checks remain
required. Changing the base image requires rechecking the backport and model
capture semantics.

Build-time checks cover CLI imports, HIP Torch, TensorBoard event writing,
plugin discovery, capture APIs, and a tiny EAGLE4 CPU forward/backward pass.
Inspect the image metadata or repeat these GPU-free checks with:

```bash
docker run --rm --entrypoint cat "$IMAGE" /opt/primus/muse-spark-eagle4-image.json
docker run --rm --entrypoint bash "$IMAGE" \
  /opt/primus/examples/specforge/docker/smoke-check-muse-eagle4.sh
```
