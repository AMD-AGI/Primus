# SpecForge single-node online smoke (draft)

This follow-up depends on Primus PR #1421. It does not include that feature's
implementation; until it merges, run this script against the published PR #1421
runtime image. The script explicitly requests 200 optimizer steps without changing
the regular online/offline recipes' defaults.

## Run inside the runtime

Expose two allocated GPUs, with capture on visible GPU 0 and the trainer on GPU 1.
Mount this checkout at `/smoke` and a writable data/results directory at `/data`.
The image's Primus source stays at `/opt/primus`. Use a persistent session and a
fresh directory for every attempt:

```bash
python /smoke/tests/scripts/run_specforge_single_node_smoke.py \
  --primus-root /opt/primus \
  --data /data/dataset/sharegpt_train.jsonl \
  --data-sha256 "$FIXTURE_SHA256" \
  --run-dir /data/runs/unique-attempt
```

`FIXTURE_SHA256` must be the recorded checksum of an already frozen dataset.
The script never downloads or prepares data. It checks that checksum, rejects
existing run directories, sets the single-node layout, and invokes `primus-cli
direct` with a 90-minute timeout plus a 60-second termination grace period.
Run it in a disposable container so container teardown also removes any leftover
processes after a hard timeout. Preserve the mounted results on failure.

Success requires a zero launcher exit, both role statuses zero, step-200 finite
loss/accuracy/gradient metrics, and nonempty step-200 shared/rank-0 state files.
`summary.json` records final metrics and a final-window mean for baseline collection.
This checks execution and checkpoint presence, not checkpoint reload correctness.
Pass `--baseline` to enforce the calibrated numerical gate. Logs and `inputs.json` remain in
the run directory. Record the image digest, source revision, GPU model, and complete
training configuration alongside them when collecting a reference.

## Frozen inputs

`specforge_smoke_fixture.json` pins the dataset source file/revision/SHA-256,
prepared fixture SHA-256, model and tokenizer revision, runtime image digest,
seeds, and GPU architecture. The image currently includes PR #1421; refreshing
that runtime requires new calibration. Model and tokenizer use the same exact
predownloaded Hugging Face snapshot, with offline loading during training.

The source is `Aeala/ShareGPT_Vicuna_unfiltered`, specifically
`ShareGPT_V4.3_unfiltered_cleaned_split.json`. Its dataset card describes a cleaned
reupload and lists Apache-2.0. The fixture contains the first 8,192 valid,
unique-ID conversations in source order, retaining complete alternating
human/assistant conversations with nonempty messages. The preparation script
records the exact selection rule; it does not sample randomly or rewrite text.
The resulting JSONL is 44,531,680 bytes. Conversation text is not included in Git
or the runtime image. Review source terms/content before redistributing a copy;
the structural filter is not a privacy or sensitive-content scrubber.

Download the source from the revision recorded in the manifest, then prepare it
once, outside the training job:

```bash
uv run --no-project python tests/scripts/prepare_specforge_smoke.py \
  --source /data/ShareGPT_V4.3_unfiltered_cleaned_split.json \
  --output /data/dataset/sharegpt_train.jsonl
```

The script verifies both checksums and refuses to replace existing output.
Pre-download `Qwen/Qwen3.5-4B` at the manifest's exact model revision into the
runtime's Hugging Face cache. Mount that cache for every run. The runner checks
that exactly two MI355X GPUs are visible and uses a fresh tokenization cache and
fresh run/output/control directories for each execution.

## Calibration and independent validation

Run three fresh 200-step executions with the same fixture, seeds, configuration,
image digest, and MI355X device layout. Preserve `inputs.json`, `recipe.yaml`,
logs, checkpoints, and `summary.json` from each. All 40 expected five-step metric
records must exist exactly once and be finite. The comparison uses the mean of
the ten logged observations at steps 155 through 200. Gradient norm is checked
for finiteness but is not given a numerical regression threshold.

```bash
uv run --no-project python tests/scripts/calibrate_specforge_smoke.py calibrate \
  --summaries /data/runs/calibration-{1,2,3}/summary.json \
  --output /data/baseline.json
```

The predeclared tolerance is twice the largest absolute deviation from the
three-run median, with a floor of 1% of median loss (at least 1e-6), or 0.005
absolute accuracy (0.5 percentage points). Bounds are two-sided, since unexpected
improvements can also indicate changed inputs or execution. Three runs are an
initial calibration, not a statistical guarantee about the future flake rate.
Inspect the measured spread before accepting a baseline; do not hide instability
by automatically widening tolerances after validation failure.

Freeze that baseline before starting two additional runs, each passing
`--baseline /data/baseline.json` to the runner. Then independently verify both:

```bash
uv run --no-project python tests/scripts/calibrate_specforge_smoke.py validate \
  --baseline /data/baseline.json \
  --summaries /data/runs/validation-{1,2}/summary.json
```

Validation rejects reused calibration summaries and mismatched input identities.
The pinned SpecForge online path does not support held-out evaluation: these are
training metrics, and “validation runs” means independent executions testing the
frozen thresholds. If validation fails, investigate before defining a new
calibration experiment; retain the failed result.

CPU checks:

```bash
uv run --no-project python -m unittest discover -s tests/scripts -p test_specforge_smoke_results.py
```

## Work remaining before CI qualification

- Complete GPU calibration and two independent validation executions, then check
  in the reviewed baseline and a compact result report.
- Add a trainer pytest entry point, JUnit reporting, backend selection, cleanup,
  and the existing non-draft `ci:gpu` gate. Keep the SpecForge runtime isolated.
- Checkpoint reload correctness and end-to-end hard-timeout cleanup remain
  separate validation work.
