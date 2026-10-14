---
name: convergence-test
description: Run, monitor, stop and report Primus convergence tests -- training a model on a real corpus and checking that the loss curve is healthy -- from a plain-language request such as "run convergence test for llama2 7B fp8 on c4 with image rocm/primus:v26.7". Covers Megatron and MaxText, bundled recipes or configs generated from the model's example config, Hugging Face or local datasets, docker images, time budgets, baselines and result locations. Use when the user asks to run, start, check, stop or compare a convergence test, loss-curve or accuracy regression run, or asks whether a model, image or branch still converges.
---

# Convergence test

The user names a model and, optionally, a dataset, a docker image and a few
other things; you plan the run with a script, launch it in the background, keep
an eye on it, and report the verdict. The tool is `tools/convergence_test/`;
every command below runs from the Primus root. This file and
[reference.md](reference.md) are enough; read the tool's README only for
something they do not cover.

## 1. Read the request

Only the model is required. Everything else has a default, so do not ask for it.

| The user says | Pass to the planner |
| --- | --- |
| a model in any spelling: "llama 2 7b", "Qwen3-8B", "mixtral 8x7b" | `--model "<as said>"` |
| bf16, fp8, nanoo_fp8, mxfp8, mxfp4 | `--precision <it>` |
| a docker image | `--image <image>` (also decides the backend) |
| megatron / maxtext, torch / jax | `--backend megatron\|maxtext` |
| a dataset: fineweb-edu (default), c4, wikitext103, a Hugging Face id `owner/name[:subset]`, or a local path | `--source <it>`; add `--text-field <col>` if they name the text column |
| "N iterations / steps" | `--iters N` |
| "for N hours", "within N hours" | `--hours N` |
| where results should go | `--output-dir <dir>` |
| "compare with / gate against <run or csv>" | `--baseline <path>` |
| a specific tokenizer | `--tokenizer <repo-or-path>` |
| training settings ("lr 2e-4", "no fp8 for attention") | append `-- --<key> <value>` |
| "use the example config" | `--from-example` |
| a Primus branch, tag or commit | see "Another Primus branch" in [reference.md](reference.md) |

## 2. Plan

```bash
python3 tools/convergence_test/plan_request.py --model "<model>" [flags from step 1]
```

It prints JSON. If `status` is `needs_input`, ask its `questions` (they are in
AskQuestion's format; without that tool, ask in plain text and list the
options), then plan again passing the chosen option's `id` to the flag named by
the question's `id`: `model` -> `--model`, `backend` -> `--backend`,
`precision` -> `--precision`, `source` -> `--source`. For `hf_token`, wait for
the user to export `HF_TOKEN`, or pass the tokenizer they name as `--tokenizer`.
If `status` is `ready`, the plan is saved in `plan_file`.

## 3. Tell the user, then start

Summarize the plan in a few lines, from the JSON:

- config: `config.kind` is `bundled` (a tested recipe; give `config.name`; the
  "Reference results" table in `tools/convergence_test/README.md` has its
  measured numbers) or `generated` (give `config.example` it came from,
  `config.path`, and each of `config.notes`)
- backend and why (`backend_reason`), precision, the detected `gpu` and
  `gpu_count`; the image `image.name`,
  saying "default" when `image.default` is true and "will be pulled" when
  `image.present_locally` is false
- the corpus `source`; the dataset is ready if `dataset.ready`, otherwise it is
  built first (`dataset.build_minutes`, at most `dataset.disk_gb_at_most` GB under
  `dataset.dir`, reused by later runs). `needed_tokens` is 10% more than the run
  trains on, so no document repeats.
- the tokenizer: `tokenizer.note` when an ungated copy replaces a gated one
- run length: `iterations` x `global_batch_size` x `seq_length` = `tokens`; the
  expected time `duration.minutes` (measured on this node), or that it will be
  known once training starts when it is null
- results go to `results_dir`
- every entry of `warnings`

Then start it without waiting for confirmation, unless a warning needs a
decision or the user asked to review first:

```bash
python3 tools/convergence_test/jobs.py start --plan <plan_file>
```

That runs everything the plan needs as one background job. When `probe_first`
is true (a generated config, a time budget, or an MI300X node), the job runs a
20-iteration probe and then, only if the probe trained cleanly, the full run.
With `--hours`, the full run takes the iteration count the probe measured to fit
the budget. The budget covers the full run's wall clock, start-up included; the
probe's 5-10 minutes come on top. Add `--probe` when the user wants only the
probe.

`start` prints `started <job>` (or `queued <job>`); `<job>` is the id that
`status` and `stop` take. It exits 0 when the job is running or queued, 1 if
it failed at once (it prints the status), and 2 if the node is busy.

A busy message names what holds the GPUs: another convergence job, or a process
or container that is not ours. **Never stop, kill or remove anything you did
not start.** When the user asked for several runs, or the node is busy with
another convergence job, start with `--queue`: the job waits its turn (first
in, first out) and starts by itself when the node is free. When something else
holds the GPUs, tell the user what it is and ask whether to queue behind it or
to start anyway with `--force` (sharing the GPUs).

## 4. Watch it

Check `python3 tools/convergence_test/jobs.py status <job>` every one to two
minutes for the first ten minutes or so, waiting in between: that is when the
dataset build, the config check, the container start-up, out-of-memory and the
first iterations fail. The `state` line shows the phase:

| Phase | Usually takes |
| --- | --- |
| building the dataset | 1-3 min per 0.5B tokens (C4 is slower); skipped when it exists |
| checking the config | seconds |
| starting the container and compiling | 2-5 min, MaxText up to ~8; nothing is printed meanwhile |
| training | the rest; `progress`, `speed` with an ETA, and `health` lines appear |

More than 15 minutes in "starting the container and compiling" is not normal:
look at the end of the `console` log. Once it is training healthily, tell the
user the ETA and that they can ask for the status at any time; do not block for
the hours a run takes.

Healthy means: the loss is falling, the `health` line shows `nan/skipped 0` and
peak memory under ~95%, and, once it finishes, `initial vs ln(V)` says
"as expected".

For a two-stage job, `state` names the stage: `probe (stage 1/2)`, then
`full run (stage 2/2)`; `budget` shows the iterations the probe fitted to the
user's hours. If the probe fails, the job ends with the probe's verdict and the
full run never starts: report it, and for a generated config fix the file as
in [reference.md](reference.md) "When a run fails" and start again (say what
you changed).

## 5. Report

When the user asks, or a run you are watching ends, run `jobs.py status <job>`
and report:

- the verdict line (`state`): PASS, or which FAIL and why
- loss first -> last, validation loss, step time, tokens/s/GPU, peak memory
- the loss curve: embed the `.png` from `results` as an image
- the result files (`.log`, `.csv`, `.png`, and the `.yaml` copy of the config)
- for a failure: the `errors` lines and the fix from [reference.md](reference.md)

Other requests:

- "status" with no job named: `jobs.py status` (latest) or `jobs.py list`
- "stop it": `jobs.py stop <job>`. It also removes the job's container. A stopped
  run has no verdict; its `status` prints a `partial` command that plots the
  iterations it trained.
- "compare these runs": `python3 tools/convergence_test/plot_loss.py <a.csv> <b.csv>
  --labels "<a>" "<b>" --out <prefix>`, then embed `<prefix>.png`; for a gate use
  `--baseline <reference.csv>` (exit 3 means FAIL)

## Rules

- One run on the node at a time; `jobs.py` enforces it, and `--queue` lines up
  the rest.
- Never edit a bundled config under `tools/convergence_test/configs/` to make a
  run fit. A generated config (under `output/convergence/configs/`) may be
  adjusted after a failed probe; say what you changed.
- Do not commit generated configs, plans or results.
- Results compare only within the same corpus, recipe and grain worker count;
  say so if the user compares across them.
