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
This checks execution and checkpoint presence, not checkpoint reload correctness
or a calibrated numerical regression threshold. Logs and `inputs.json` remain in
the run directory. Record the image digest, source revision, GPU model, and complete
training configuration alongside them when collecting a reference.

## Work remaining before CI qualification

- Choose a small redistributable fixture, record provenance/order/preprocessing and
  checksum, and ensure it supplies enough usable examples for 200 steps.
- Freeze target model/tokenizer revisions and configuration; repeat baseline runs
  per GPU architecture and establish loss/accuracy tolerances from measured variation.
- Add a trainer pytest entry point, JUnit reporting, backend selection, cleanup,
  and the existing non-draft `ci:gpu` gate. Keep the SpecForge runtime isolated.
- Validate the exact image/script combination on GPUs. No GPU run is claimed by
  this draft; CPU validation only checks result parsing and failure conditions.
