# Qwen3.5-4B / DFlash single-node reference (draft)

This PR is a data, recipe, commands, and reference-results handoff; it is not an
integration-ready test framework. The 100-step GPU reference is **pending**.
The old 200-step measurements do not apply to this fixture/schedule.

Use the existing `examples/specforge/configs/qwen3.5-4b-dflash-online.yaml` from a
Primus checkout containing #1421 (main after merge). Do not use this handoff
branch as the runtime checkout: it does not duplicate #1421's feature code.
`primus-cli container` mounts and runs its host checkout, so its revision matters
in addition to the image digest. No generated training YAML or Python launcher
is needed. The Python tools below only check completed runs and collect bounds.

## Inputs and one-time preparation

- Data: `tests/trainer/fixtures/specforge/sharegpt_800.jsonl`, all 800 conversations,
  4,243,587 bytes. Source IDs, provenance, and declared license are alongside it.
- Pins: `specforge_smoke_fixture.json` records data checksums, model/tokenizer,
  image digest, SpecForge revision, seeds, and the MI355X reference target.
- Shape: one capture GPU, one trainer GPU, batch size 2, accumulation 4,
  100 optimizer steps = 800 consumed samples. The runtime shuffles with seed 42.
- Pre-download the Qwen model/tokenizer snapshot below into the Hugging Face
  cache. Ordinary runs need no dataset download. Use the pinned image's Python
  dependencies; do not install Primus's full dependency tree into it.

The commands are Bash. Set absolute paths without spaces or quotes. For container
and Slurm modes, the model cache, smoke checkout, and results must be available
on the compute host and mounted at these same paths. `PRIMUS_CHECKOUT` points to
the runtime source; `SMOKE_CHECKOUT` points to this draft's data/checker checkout.

```bash
export PRIMUS_CHECKOUT=/path/to/Primus
export SMOKE_CHECKOUT=/path/to/Primus-smoke
export HF_HOME=/path/to/hf-cache
export RESULTS=/path/to/results
export IMAGE=tasimage/primus-specforge@sha256:2fcef9af3d29d53132b2e548fbda31ca8844e6c043f236249050a3398288ed2d
export MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

One-time model download, using Python with `huggingface_hub` installed (for
example inside the pinned runtime with the cache mounted at `$HF_HOME`):

```bash
uv run --no-project --python "$(command -v python)" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B", revision=os.environ["MODEL_REVISION"])
PY
```

## Common recipe settings

Run this block once for each attempt, choosing a fresh run ID. It creates only
the parent run directory; leave the control and consumer-state directories absent.

```bash
export RUN_ID="qwen35-dflash-100-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export RUN_DIR="$RESULTS/$RUN_ID"
mkdir -p "$RESULTS"
mkdir "$RUN_DIR"  # fails if this attempt already exists
export RUN_ROOT="$RUN_DIR/control"
export CONSUMER_STATE_DIR="$RUN_DIR/consumer-state"
export OUTPUT_DIR="$RUN_DIR/output"
export PRIMUS_WORKSPACE="$RUN_DIR/primus"
export TRAIN_DATA_PATH="$SMOKE_CHECKOUT/tests/trainer/fixtures/specforge/sharegpt_800.jsonl"
export TARGET_MODEL="$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/$MODEL_REVISION"
test -f "$TARGET_MODEL/config.json"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PRIMUS_SPECFORGE_SINGLE_NODE=true
export CAPTURE_NNODES=1 TRAINER_NNODES=1 SERVER_COUNT=1 SERVER_TP=1
export SERVER_GPUS=0 TRAINER_GPUS=1 NPROC_PER_NODE=1 GPUS_PER_NODE=2
export MAX_STEPS=100
export SPECFORGE_ROOT=/workspace/SpecForge
export SPECFORGE_CONFIG="$SPECFORGE_ROOT/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-amd.yaml"

train_args=(train pretrain
  --config examples/specforge/configs/qwen3.5-4b-dflash-online.yaml
  "specforge_overrides={'training.seed':42,'training.prompt_seed':42,'training.batch_size':2,'training.accumulation_steps':4,'data.cache_dir':'$RUN_DIR/dataset-cache'}")

# Verify the fixture, and record comparable conditions without run-specific paths.
uv run --no-project python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
manifest = json.loads((Path(os.environ["SMOKE_CHECKOUT"]) / "tests/scripts/specforge_smoke_fixture.json").read_text())
assert hashlib.sha256(Path(os.environ["TRAIN_DATA_PATH"]).read_bytes()).hexdigest() == manifest["dataset"]["fixture_sha256"]
revision = subprocess.check_output(["git", "-C", os.environ["PRIMUS_CHECKOUT"], "rev-parse", "HEAD"], text=True).strip()
inputs = {"manifest": manifest, "runtime_primus_revision": revision, "runtime_image": os.environ["IMAGE"]}
(Path(os.environ["RUN_DIR"]) / "reference-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
PY
printf '%q ' "${train_args[@]}" > "$RUN_DIR/training-command.txt"
```

The YAML supplies `training.num_epochs=1`, `training.log_interval=5`, the data
path, `model.use_liger_kernel=false`, and max/save steps from `MAX_STEPS`.
The target snapshot path is passed to both capture and training. The flat-key
`specforge_overrides` dictionary above merges into those YAML overrides.
Save the resolved runtime configuration/logs as evidence too. The recorded input
identity describes the intended conditions; the checker does not attest the
runtime, image, GPU hardware, or model files on your behalf.

## Choose one native launch path

### Direct, in the configured runtime

Use exactly two allocated MI355X GPUs. Normalize a scheduler-provided two-device
mask before launching; HIP indices 0 and 1 are relative to the ROCr mask.

```bash
mask=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}
[[ -n "$mask" ]] || { echo 'Set the two allocated GPU IDs before launching'; exit 1; }
[[ "$mask" == *,* && "$mask" != *,*,* ]] || exit 1
export ROCR_VISIBLE_DEVICES="$mask" HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
cd "$PRIMUS_CHECKOUT"
set +e
timeout --signal=TERM --kill-after=60s 90m \
  bash runner/primus-cli direct --log_file "$RUN_DIR/launcher.log" -- "${train_args[@]}"
rc=$?
printf '%s\n' "$rc" > "$RUN_DIR/launcher.exit"
set -e
```

### Container, on an already allocated host

First apply the same visibility normalization as the direct example. The pinned
image supplies dependencies; the native launcher runs `PRIMUS_CHECKOUT` source.
Use your site's approved Primus launcher configuration for Docker permissions.
The default container settings may require adjustment for your environment.

```bash
container_args=(container --image "$IMAGE" --name "$RUN_ID" --shm-size 64g
  --volume "$SMOKE_CHECKOUT:$SMOKE_CHECKOUT:ro"
  --volume "$RESULTS:$RESULTS" --volume "$HF_HOME:$HF_HOME")
for key in RUN_ID RUN_ROOT CONSUMER_STATE_DIR OUTPUT_DIR PRIMUS_WORKSPACE \
  TRAIN_DATA_PATH TARGET_MODEL HF_HOME HF_HUB_OFFLINE HF_DATASETS_OFFLINE \
  PRIMUS_SPECFORGE_SINGLE_NODE CAPTURE_NNODES TRAINER_NNODES SERVER_COUNT \
  SERVER_TP SERVER_GPUS TRAINER_GPUS NPROC_PER_NODE GPUS_PER_NODE MAX_STEPS \
  SPECFORGE_ROOT SPECFORGE_CONFIG ROCR_VISIBLE_DEVICES; do
  container_args+=(--env "$key")
done
container_args+=(--env HIP_VISIBLE_DEVICES=0,1 --env CUDA_VISIBLE_DEVICES=0,1)
cd "$PRIMUS_CHECKOUT"
set +e
timeout --signal=TERM --kill-after=60s 90m \
  bash runner/primus-cli "${container_args[@]}" -- "${train_args[@]}" \
  > "$RUN_DIR/launcher.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$RUN_DIR/launcher.exit"
set -e
```

### Primus CLI Slurm → container → training

Build `container_args` as above but **do not execute the container launch**.
Submit from a shell without stale visibility masks from a different allocation.
This example requires the site to supply a two-device `ROCR_VISIBLE_DEVICES`
mask on the allocated node; the container launcher's bare `--env
ROCR_VISIBLE_DEVICES` resolves it there, not on the login host. HIP/CUDA ordinals
0 and 1 are relative to that mask. If your site only supplies physical IDs in
CUDA/HIP variables, adapt the site's GPU mapping before using this command;
do not assume physical GPUs 0 and 1 belong to the allocation.

```bash
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
cd "$PRIMUS_CHECKOUT"
set +e
bash runner/primus-cli slurm srun -N 1 --gres=gpu:2 --cpus-per-task=16 \
  --time=01:30:00 -- "${container_args[@]}" -- "${train_args[@]}" \
  > "$RUN_DIR/launcher.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$RUN_DIR/launcher.exit"
set -e
```

Add the site's scheduler account/partition flags before the first `--`.
This exercises **Primus's Slurm launcher**, rather than an external sbatch script
that invokes a Python wrapper around direct mode. Queue waiting is separate from
the scheduler's execution time limit. Container cleanup after a hard timeout
must be verified for your runtime; retain logs and check for surviving processes.
All three 100-step command paths remain GPU-unqualified in this draft.

## Check a completed run (optional Python helper)

```bash
uv run --no-project python "$SMOKE_CHECKOUT/tests/scripts/check_specforge_smoke.py" \
  --run-dir "$RUN_DIR" --launcher-exit-code "$(cat "$RUN_DIR/launcher.exit")" \
  --inputs "$RUN_DIR/reference-inputs.json"
```

The checker never launches training. It requires launcher success, both role
statuses zero, 800 consumed samples, all 20 five-step metric records, finite
loss/accuracy/gradient norm, and nonempty step-100 shared/rank-0 checkpoints.
It writes final metrics and means of the ten observations at steps 55–100 to
`summary.json`. Checkpoint reload correctness is not tested.

There is **no accepted 100-step reference yet**. GPU runs remain pending. When
requested, collect three runs with identical inputs, then freeze bounds:

```bash
uv run --no-project python "$SMOKE_CHECKOUT/tests/scripts/calibrate_specforge_smoke.py" calibrate \
  --summaries "$RESULTS"/calibration-{1,2,3}/summary.json --output "$RESULTS/baseline.json"
```

Choose those run IDs explicitly when collecting references. Run two fresh
validations and pass `--baseline "$RESULTS/baseline.json"` to the checker, then:

```bash
uv run --no-project python "$SMOKE_CHECKOUT/tests/scripts/calibrate_specforge_smoke.py" validate \
  --baseline "$RESULTS/baseline.json" --summaries "$RESULTS"/validation-{1,2}/summary.json
```

Bounds are two-sided: twice the largest calibration deviation from the median,
with floors of 1% of loss (at least 1e-6) and 0.005 absolute accuracy. These are
initial engineering tolerances, not a statistical confidence interval. Never
retune them just to pass validation. The pinned online backend has no held-out
evaluation support, so these are training metrics. Future Python test cases can
parameterize models, draft architectures, and GPUs after this reference is useful.

CPU checks: `uv run --no-project python -m unittest discover -s tests/scripts -p test_specforge_smoke_results.py`.
To regenerate data for provenance verification, use `prepare_specforge_smoke.py`
with the pinned source and a new output path; ordinary training uses checked-in data.
