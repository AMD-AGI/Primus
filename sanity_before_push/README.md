# `sanity_before_push/` — FLA parity check before pushing

Quick, FLA-schedule-matched re-run of the 5 models we ported to Primus, to
confirm **speed and loss match FLA** before pushing / merging.

| # | Model | Spec | FLA-init? | iters (warmup + 500) |
|---|---|---|:--:|---|
| 1 | `300M_gdn_pure`     | pure Gated DeltaNet  | yes | 200 + 500 = 700 |
| 2 | `300M_kda_pure`     | pure Kimi Delta Attn | yes | 200 + 500 = 700 |
| 3 | `300M_gdn_hybrid`   | GDN + MLA hybrid     | no  | 200 + 500 = 700 |
| 4 | `300M_mamba_hybrid` | Mamba2 + MLA hybrid  | no  | 200 + 500 = 700 |
| 5 | `1B_gdn_pure`       | pure GDN (FSDP/ZeRO-2) | no | 2000 + 500 = 2500 |

## Methodology

Each config keeps FLA's **full** `lr_warmup_iters` and `lr_decay_iters`, then
stops at `warmup + 500`. Because the cosine LR depends only on
`(step, warmup, decay)`, the LR at every compared step is bit-identical to
FLA's full-length run — we just compare the first `warmup + 500` steps.

- **Speed** is judged on steady-state ms/iter (Megatron's running average,
  measured after warmup). PASS if Primus is within tolerance of FLA
  (≤ +5% for 300M, ≤ +6% for 1B).
- **Loss** is judged at the final step. The two **pure** models load FLA-init
  weights so their loss should match FLA to within ±2%. The **hybrids** have no
  FLA-init weight converter, so their absolute loss is offset — only the curve
  shape and speed are meaningful, so loss is reported as INFO (not PASS/FAIL).
- All runs use `use_fla_data=true` → `FLAOrderGPTDataset` feeds the exact same
  token order as FLA's HF `DistributedSampler`.

## Prerequisites (on the training box)

- FLA-init checkpoints (pure models): `output/fla_init_ckpt_300M`,
  `output/fla_init_kda_300M`. Build with
  `tools/convert_fla_gdn_init_to_megatron.py` /
  `tools/convert_fla_kda_init_to_megatron.py` if missing.
- FLA token caches: `.../fineweb-edu/sample-10BT/train` (300M) and
  `.../fineweb-edu/sample-100BT/train` (1B).
- FLA reference logs under `.../legacy/training/` (paths in
  `summarize_sanity.py::MODELS`).

The driver checks each prerequisite and skips a model (with a clear message)
rather than failing the whole batch.

## Run

```bash
cd /home/vanbhati@amd.com/Primus

# all 5 (fastest first; ~2-3 h total on 8×MI300X)
bash sanity_before_push/run_sanity.sh

# subset
bash sanity_before_push/run_sanity.sh 300M_gdn_pure 1B_gdn_pure

# background
nohup bash sanity_before_push/run_sanity.sh \
    > sanity_before_push/logs/_run_all.log 2>&1 &
```

## Summarize

```bash
python3 sanity_before_push/summarize_sanity.py --print
```

Prints a Speed table, a Loss table, and an overall **READY TO PUSH /
NOT READY** verdict (also written to `sanity_before_push/summary.md`).

## Notes

- This folder is git-ignored — it's a local pre-push harness, not part of the
  Primus public surface. The configs are derived from the validated
  `eval/configs/*_real.yaml` with only the iteration/warmup budget changed.
- Logs land in `sanity_before_push/logs/<model>.log`; checkpoints (final only)
  in `output/sanity/<model>/`.
