# Qwen3.5-4B / DFlash single-node reference (draft)

This PR is a data, recipe, commands, and reference-results handoff; it is not an
integration-ready test framework. The 100-step MI355X reference passed three
calibration runs and two independent validation runs on 2026-11-12.
The old 200-step measurements do not apply to this fixture/schedule.

Use the existing `examples/specforge/configs/qwen3.5-4b-dflash-online.yaml` from a
separate checkout of `AMD-AGI/Primus` main. #1421 merged as
`aa454b0233346b9cecaa01685d0d217595ac2b4a`; use that commit or a later main
revision for new runs. Do not use this handoff branch as the runtime checkout:
it does not include the merged feature code.
`primus-cli container` mounts and runs its host checkout, so its revision matters
in addition to the image digest. No generated training YAML or Python launcher
is needed. The Python tools below only check completed runs and collect bounds.

## Inputs and one-time preparation

- Data: `tests/trainer/fixtures/specforge/sharegpt_800.jsonl`, all 800 conversations,
  4,243,587 bytes. Source IDs, provenance, and declared license are alongside it.
- Workload pins: `specforge_smoke_fixture.json` records data checksums,
  model/tokenizer revisions, training settings, and the MI355X reference target.
  Its `runtime` section is historical calibration provenance, not a requirement
  for nightly/weekly candidates. Test new Primus/SpecForge versions and record
  their actual commits and resolved image digest on every run.
- Shape: one capture GPU, one trainer GPU, batch size 2, accumulation 4,
  100 optimizer steps = 800 consumed samples. The runtime shuffles with seed 42.
- Pre-download the Qwen model/tokenizer snapshot below into the Hugging Face
  cache. Ordinary runs need no dataset download. Use the candidate image's Python
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
export IMAGE="tasimage/primus-specforge:<mi35x-tag-under-test>"
export MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

For container/Slurm runs, select a published candidate tag, resolve it once, and
record the SpecForge commit from that image (these metadata commands need no GPU):

```bash
docker pull "$IMAGE"
export IMAGE=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")
export RUNTIME_SPECFORGE_REVISION=$(docker run --rm --entrypoint git "$IMAGE" -C /workspace/SpecForge rev-parse HEAD)
```

For direct mode, set `SPECFORGE_ROOT` to the actual checkout and export
`RUNTIME_SPECFORGE_REVISION=$(git -C "$SPECFORGE_ROOT" rev-parse HEAD)` instead.
Keep the workload settings fixed when changing software; retain the resolved
recipe with the run evidence.

One-time model download, using Python with `huggingface_hub` installed (for
example inside the candidate runtime with the cache mounted at `$HF_HOME`):

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
manifest.pop("runtime", None)  # Historical software pins are not candidate provenance.
inputs = {"manifest": manifest, "runtime_primus_revision": revision,
          "runtime_specforge_revision": os.environ["RUNTIME_SPECFORGE_REVISION"],
          "runtime_image": os.environ["IMAGE"]}
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

First apply the same visibility normalization as the direct example. The candidate
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
The recorded batch used native `primus-cli slurm sbatch` to start one container,
then a private batch driver invoked the direct training command five times with
fresh processes, run directories, and dataset caches. All five executions used
the same two GPUs, allocation, container, and fixed seed. The `srun` variant above
and five separate container launches were not independently tested. For `sbatch`,
submission success is not training success: wait for the job to finish and verify
its final exit status before running the checker.

## Accepted MI355X reference

[specforge_smoke_baseline_mi355x.json](specforge_smoke_baseline_mi355x.json)
contains the current numerical regression policy. The
[qualification record](specforge_smoke_qualification_mi355x.json) includes the
exact reference inputs, runtime versions, five run summaries and their hashes,
and every logged loss/accuracy/gradient-norm observation.

All five runs completed 100 optimizer steps, consumed all 800 conversations,
reported successful role/launcher exits, and wrote nonempty step-100 shared and
rank-0 checkpoints. All 800 rows survived preprocessing. The logged loss,
accuracy, and gradient norm matched exactly across all five runs; this is
repeatability under these fixed conditions, not a guarantee across hardware or
software revisions.

The means of the ten logged observations at steps 55–100 are:

- Training loss reference: **7.522446346282959**; fail above **7.898568663597107** (+5%).
- Training accuracy reference: **0.05101585015654564**; fail below **0.04101585015654564** (one percentage point lower).
- Gradient norm: **0.21905557662248612**, recorded for inspection; only finiteness is asserted.

The original validations passed tighter two-sided bounds frozen after the third
calibration run, without retuning. The current one-sided policy was adopted
after those runs for evolving software versions. Replaying all five recorded
summaries passes the new policy; no new GPU runs or cross-version calibration
were performed for this policy change. The historical qualification record and
its original baseline hash are unchanged. The runtime checkout was
`55fec472fda464449ebc417c3e0e1037f0bb912d` (the pre-merge #1421 head), with the
manifest's pinned image, SpecForge, model/tokenizer, and data. PyTorch was
`2.9.1+rocm7.2.0.git7e1940d4`; HIP was `7.2.26015-fc0010cf6a`.
The manifest's image Primus revision describes the image contents; the mounted
runtime checkout revision above is the source actually executed.

The gate compares `workload_identity`: fixture checksum/row count, model/tokenizer,
training settings, and GPU type. Primus/SpecForge commits and the image digest
are recorded in `reference_inputs` and the full `reference_identity` for diagnosis,
but do not need to match the baseline. Calibration still requires three runs
with identical full inputs; candidate comparisons may use new software versions.
Changed workload settings require a separate reference. The checker relies on
recorded inputs; it does not attest that the runtime honored those settings.

Lower loss and higher accuracy pass, provided values remain finite and valid.
Loss improvements over 5% or accuracy improvements over 0.01 add non-failing
`review_notes` to the summary for investigation. The +5% loss / -0.01 accuracy
limits are provisional
engineering choices, not statistically calibrated cross-version tolerances.
Keep the baseline fixed until deliberately reviewed; never refresh it from each
passing candidate. These runs do not qualify other GPU types, seeds, model
configurations, checkpoint reload, or performance. Nightly/weekly scheduling and
CI wiring remain follow-up work.

## Check a completed run (optional Python helper)

```bash
uv run --no-project python "$SMOKE_CHECKOUT/tests/scripts/check_specforge_smoke.py" \
  --run-dir "$RUN_DIR" --launcher-exit-code "$(cat "$RUN_DIR/launcher.exit")" \
  --inputs "$RUN_DIR/reference-inputs.json" \
  --baseline "$SMOKE_CHECKOUT/tests/scripts/specforge_smoke_baseline_mi355x.json"
```

The checker never launches training. It requires launcher success, both role
statuses zero, 800 consumed samples, all 20 five-step metric records, finite
loss/accuracy/gradient norm, and nonempty step-100 shared/rank-0 checkpoints.
It writes final metrics and means of the ten observations at steps 55–100 to
`summary.json`. Checkpoint reload correctness is not tested.

This comparison also applies to newer main checkouts and candidate images with
the same workload. Omit `--baseline` only when collecting calibration runs or
when intentionally requesting structural checks without a numerical gate.

To deliberately collect a new reference, run three executions with identical
inputs, then freeze bounds to a new file:

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

Bounds are one-sided: loss at most 1.05 times the calibration median, accuracy
at least max(0, calibration median minus 0.01). These are initial engineering
tolerances, not a statistical confidence interval. Never
retune them just to pass validation. This reference collects training metrics,
not held-out evaluation. Future Python test cases can
parameterize models, draft architectures, and GPUs after this reference is useful.

CPU checks: `uv run --no-project python -m unittest discover -s tests/scripts -p test_specforge_smoke_results.py`.
To regenerate data for provenance verification, use `prepare_specforge_smoke.py`
with the pinned source and a new output path; ordinary training uses checked-in data.
