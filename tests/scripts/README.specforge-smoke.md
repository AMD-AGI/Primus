# SpecForge single-node online smoke (draft)

This test depends on the single-node online enablement in PR #1421. It runs
**100 optimizer steps**, with batch size 2, gradient accumulation 4, and one
trainer GPU: **800 consumed conversation samples**. Capture uses a second GPU
on the same MI355X host. Regular online/offline recipe defaults are unchanged.

## Checked-in data

The complete 800-conversation fixture is
`tests/trainer/fixtures/specforge/sharegpt_800.jsonl` (4,243,587 bytes).
Its source IDs, attribution, and declared Apache-2.0 license are in the same
directory. No dataset download or conversion is needed for a normal test.

`specforge_smoke_fixture.json` pins the source revision/checksum, fixture checksum,
Qwen3.5-4B model/tokenizer revision, image digest, seed 42, and prompt seed 42.
The fixture's conversations were selected from the previously eligible sample
pool, excluding common contact/credential patterns. The new 800-row dataset is
shuffled by the pinned runtime; it does not preserve the 200-step run's ordering
or learning-rate schedule, so the old baseline cannot be reused.

The runner verifies the fixture checksum and requires a consumed-sample count of
800. The 100-step configuration has not yet been GPU-qualified; if runtime
filtering or execution loses examples, an early exit is a failure, not a pass.

For provenance verification only, regenerate from the pinned original source:

```bash
uv run --no-project python tests/scripts/prepare_specforge_smoke.py \
  --source /data/ShareGPT_V4.3_unfiltered_cleaned_split.json \
  --output /data/regenerated-sharegpt_800.jsonl
```

The script selects the checked-in source IDs in canonical order, verifies source
and output checksums, and refuses to overwrite existing files.

## Run inside the runtime

Expose exactly two allocated MI355X GPUs. Mount this checkout at `/smoke`, a
writable results directory at `/data`, and the predownloaded Hugging Face cache
at `$HF_HOME`. The image's Primus source stays at `/opt/primus`.
Pre-download `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; both capture and training use that
exact cached snapshot, with offline model loading during the run.

```bash
uv run --no-project --python "$(command -v python)" python \
  /smoke/tests/scripts/run_specforge_single_node_smoke.py \
  --run-dir /data/runs/unique-attempt
```

The default data file and manifest come from the smoke checkout. Optional `--data`
and `--data-sha256` arguments must still match the canonical manifest checksum.
The runner preserves the allocated GPU pair through Primus preflight, creates
fresh tokenization/control/output directories, and invokes `primus-cli direct`.
Each attempt must use a new run directory. The command has a 90-minute timeout
plus a 60-second termination grace period; use a disposable container to remove
any leftover processes after a hard timeout. Keep mounted evidence on failure.

Success requires a zero launcher exit, zero producer/trainer role statuses,
exactly 800 consumed samples, all 20 five-step metric records through step 100,
finite loss/accuracy/gradient norm, and nonempty step-100 shared/rank-0 checkpoint
files. `inputs.json`, `recipe.yaml`, logs, and `summary.json` record the execution.
Checkpoint presence does not establish checkpoint reload correctness.

## Numerical calibration: pending

The old 200-step baseline and qualification report have been removed from this
branch; the prior measurements remain in Git history. **No GPU run or accepted
numerical baseline is claimed for the new 100-step fixture.**

The planned budget remains three calibration runs followed by two independent
validation runs, on MI355X only. Keep inputs, seeds, image, and device layout
identical across all five executions. The comparison uses the mean of the ten
logged observations at steps 55,60,...,100. Gradient norm must be finite but has
no numerical threshold.

```bash
uv run --no-project python tests/scripts/calibrate_specforge_smoke.py calibrate \
  --summaries /data/runs/calibration-{1,2,3}/summary.json \
  --output /data/baseline.json
```

The predeclared two-sided tolerance is twice the largest deviation from the
three-run median, with a floor of 1% of median loss (at least 1e-6), or 0.005
absolute accuracy (0.5 percentage points). Inspect the measured spread before
accepting it; three runs do not establish a long-term flake rate.

Freeze the baseline before starting two new runs, each passing
`--baseline /data/baseline.json` to the runner. Do not widen bounds merely to
make failed validation pass. Then verify both summaries independently:

```bash
uv run --no-project python tests/scripts/calibrate_specforge_smoke.py validate \
  --baseline /data/baseline.json \
  --summaries /data/runs/validation-{1,2}/summary.json
```

Input-identity mismatches and reused calibration summaries are rejected.
The pinned online backend does not support held-out evaluation: these are
training metrics, and validation means separate executions testing the bounds.
Always check the runner's exit code: a summary file alone does not prove numerical
success. Without `--baseline`, the runner performs structural checks only.

CPU checks:

```bash
uv run --no-project python -m unittest discover -s tests/scripts -p test_specforge_smoke_results.py
```

## Remaining work

- Run three 100-step calibration and two independent validation executions;
  commit the accepted baseline and qualification report.
- Add the trainer pytest entry point, backend selection, JUnit reporting, cleanup,
  and the existing non-draft `ci:gpu` gate. Keep this runtime isolated.
- Validate checkpoint reload correctness and hard-timeout cleanup separately.
