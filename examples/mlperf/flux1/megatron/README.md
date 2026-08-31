# FLUX.1-Schnell MLPerf Training — Megatron backend

The sibling directory (`examples/mlperf/flux1/`) runs FLUX.1-Schnell through the
`diffusion` backend. This one runs it through Megatron, which is the path the
MXFP6 work sits on, and produces runtime logs meant to be read by
`mlperf_logging.compliance_checker` rather than by a person.

## What the launcher supplies, and why it has to

The logging patch refuses to invent any value that ends up in the submitted
log. Startup fails, loudly, if any of these is missing:

| Variable | What it decides |
| --- | --- |
| `MLLOG_OUTPUT_FILE` | Where the result file is written. Rank zero writes it directly, so the artifact the checker reads is the artifact the run produced — not a filtered copy of stdout. |
| `MLLOG_SUBMISSION_ORG` / `_DIVISION` / `_PLATFORM` | Which division the log is judged in. A wrong default here is a wrong submission. |
| `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR` / `_ATTN` / `_COMM` | The numerics disclosure. |
| `MLPERF_CLEAR_CACHES` | Whether the machine actually started cold. |
| `EXP` | The recipe, which a reviewer has to be able to find in the submission. |

`run_and_time.sh` sets all of them, drops the page cache, and reports
`cache_clear=false` if it could not — a run without the privileges to drop
caches is still a valid run, just not a cold one.

## One run

```bash
RUN_INDEX=0 RESULTS_DIR=/results \
bash examples/mlperf/flux1/megatron/run_and_time.sh
```

This writes `/results/result_0.txt` and immediately runs the compliance checker
against it. Checking at the end of each run is deliberate: the alternative is
discovering after ten runs that none of the logs parse.

## A submission campaign

```bash
RESULTS_DIR=/results bash examples/mlperf/flux1/megatron/run_campaign.sh
```

Ten runs with seeds `42..51`, then an RCP comparison over the collected
results. Ten is what `mlperf_logging/rcp_checker/rcp_checker.py` requires for
`flux1`. Runs that finish without reaching the target still produce a result
file and still count — they are part of the distribution being compared, and
discarding them would bias it.

## Where the numbers come from

The reference convergence points for `flux1` at global batch size 512
(`rcp_checker/training_6.0.0/rcps_flux1.json`, 20 NVIDIA BF16 runs) span
7,077,888 to 7,602,176 samples, which is 13,824 to 14,848 steps. The recipe's
`train_iters` is a safety cap set above that range, not a target: a run ends
when `eval_accuracy` reaches 0.586, and how many samples that took is the
result.

`closed_flux1.yaml` also pins several hyperparameters exactly — AdamW betas
0.9/0.95, epsilon 1e-8, weight decay 0.1, gradient clip 1.0, and
`evaluation_frequency` at exactly 262,144 samples (`eval_interval: 512` at
GBS 512). Changing any of them in the recipe makes the log fail the checker.

## MXFP6 and the disclosure vocabulary

`training_6.0.0/common.yaml` accepts a fixed set of values for
`lowest_numerical_precision_in_*`, and `mxfp6` is not in it. An MXFP6 run
therefore produces a structurally valid log that the checker rejects until the
format is accepted upstream. The logger emits the configured value anyway and
warns — describing an MXFP6 run as `fp8` would be a false disclosure, which is
worse than a log that has to wait for approval.
