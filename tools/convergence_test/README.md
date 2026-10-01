# Convergence Test

Train a model on a **real corpus** and check that the loss actually goes down.

Primus ships a lot of pretraining configs, but almost all of them are
throughput benchmarks: Megatron configs run on `mock_data` (random tokens) with
a learning rate of `1e-5` and a two-iteration warmup; MaxText configs run on
`dataset_type: synthetic` with MaxText's default `3e-5`. They are excellent for
measuring TFLOP/s and useless for telling you whether the stack learns. This
tool covers the other case, for both the **Megatron** and the **MaxText**
backend.

It is a thin wrapper around the normal `primus-cli` pretrain path, so anything
you learn here transfers directly to a real training run.

---

## Quick start

```bash
tools/convergence_test/run_convergence_test.sh --list

# Megatron, rocm/primus image
tools/convergence_test/run_convergence_test.sh --model megatron/llama3.2_1B       # ~45 min
tools/convergence_test/run_convergence_test.sh --model megatron/llama2_7B         # ~85 min
tools/convergence_test/run_convergence_test.sh --model megatron/llama2_7B-FP8
tools/convergence_test/run_convergence_test.sh --model megatron/mixtral_8x7B_v0.1 # ~2h10m

# MaxText, rocm/jax-training image
tools/convergence_test/run_convergence_test.sh --model maxtext/llama3.2_1B        # ~45 min
tools/convergence_test/run_convergence_test.sh --model maxtext/llama2_7B          # ~85 min
tools/convergence_test/run_convergence_test.sh --model maxtext/llama2_7B-nanoo_fp8
tools/convergence_test/run_convergence_test.sh --model maxtext/mixtral_8x7B       # ~75 min

# Another corpus, and results kept somewhere else
tools/convergence_test/run_convergence_test.sh --model maxtext/llama2_7B --source c4 \
    --output-dir /shared/convergence/v26.8
```

(Times on 8x MI325X.) That single command builds the dataset, lints the
config, trains, plots the loss curve, and checks that every iteration ran. The
backend comes from the config's `framework:`; `--model` takes
`<backend>/<model>`, or just `<model>` when only one backend has it.
`--config <file>` runs any other experiment YAML, as long as it trains on real
data (see [Adding a model](#adding-a-model)).

### Gated tokenizers

Most Llama presets point at gated `meta-llama` repos. For those, accept the
licence on huggingface.co with an account that has access and export a token
before the first run:

```bash
export HF_TOKEN=hf_...
```

Only dataset preparation needs it. `prepare_dataset.py` saves the tokenizer
into the dataset directory and the configs point at that local copy, so
training itself never touches the Hub. If you would rather not deal with
gating, `--tokenizer <repo-or-path>` substitutes any equivalent vocabulary, for
example the ungated mirrors `--tokenizer NousResearch/Llama-3.2-1B` and
`--tokenizer NousResearch/Llama-2-7b-hf`.

### What a run produces

Every file of a run is named `<exp>[-<corpus>]_<timestamp>` and written to
`output/convergence/` in the Primus checkout, or to `--output-dir`:

| File | Contents |
| --- | --- |
| `.log` | The console output, headed by the command line, the Primus and backend commits, the config, the dataset and the overrides |
| `.yaml` | A copy of the config as it was when the run started |
| `.csv` | Per-iteration training and validation loss, learning rate, grad norm, throughput and memory; usable as a `--baseline` |
| `.png` | The loss curve, with learning-rate, grad-norm and throughput panels |

The trainer's own logs go to `<workspace>/<team>/<user>/<exp>/`, where
`workspace` is `output/` unless `--output-dir` (or `PRIMUS_WORKSPACE`) says
otherwise, and MaxText's TensorBoard files go to
`<base_output_directory>/<run_name>/`, which `--output-dir` also moves. The
driver prints all of these locations before it starts. An `--output-dir`
outside the checkout, on shared storage for example, is mounted into the
container.

`--plot-only` re-plots the latest run of a config from its console log; give
it the same `--source` and `--output-dir` as the run.

### Choosing the data

`--source` picks the corpus. The dataset is built on the host, tokenised with
the vocabulary of the model being trained, and reused by later runs.

| `--source` | Corpus |
| --- | --- |
| `fineweb-edu` (default) | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |
| `c4` | `allenai/c4`, `en`; validated on its official `validation` split |
| `wikitext103` | `Salesforce/wikitext`, `wikitext-103-raw-v1`; about 100M words, less than most recipes consume |
| `<owner>/<name>[:<subset>]` | Any Hugging Face dataset with a text column, e.g. `HuggingFaceFW/fineweb:sample-10BT` |
| a path, directory or glob | Your own `.jsonl`, `.json`, `.parquet` or `.txt` files; start it with `/`, `./`, `../` or `~` |

`--text-field` names the column that holds the text when it is not `text`.
Plain-text files are split into documents at blank lines. Every source except
C4 lacks a held-out split, so its first ~10M tokens become the validation set
and training starts after them; a local corpus needs to be larger than that.

Datasets are kept in `$DATA_PATH/convergence/<corpus>-<tokenizer>/` (`DATA_PATH`
defaults to `data/` in the checkout and can point at shared storage).
`--data-dir` chooses the directory instead; with `--skip-prepare` it can be a
dataset you built yourself with [`prepare_dataset.py`](#1-build-a-dataset).
`--tokenizer` builds with a substitute tokenizer that has the same vocabulary,
such as an ungated mirror.

A run on another corpus is a different experiment. Its output names carry the
corpus, and it should only be compared with runs on the same corpus.

### Options

| Option | Meaning |
| --- | --- |
| `--model <backend>/<name>` | A bundled config; `--list` shows them |
| `--config <file>` | Any Primus experiment YAML |
| `--train-iters N` | Change the run length, and the learning-rate schedule with it |
| `--probe N` | Train only the first N iterations of the run, to measure speed |
| `--budget-hours H` | With `--probe`, report how many iterations fit in H hours |
| `--source <corpus>`, `--text-field <name>` | The corpus; see [Choosing the data](#choosing-the-data) |
| `--tokenizer <repo-or-path>` | Build the dataset with a substitute tokenizer |
| `--data-dir <dir>` | Where the dataset is kept |
| `--skip-prepare`, `--prepare-only` | Use the dataset as it is; build it and stop |
| `--output-dir <dir>` | Where results and trainer logs go |
| `--baseline <run>`, `--tolerance <loss>` | Gate the run against a reference run; exit 3 on FAIL |
| `--image <image>` | Training image (default below) |
| `--env KEY=VALUE` | Set a container environment variable (repeatable) |
| `--deterministic` | Export `PRIMUS_DETERMINISTIC=1` for a repeatable run |
| `--strict` | Fail on lint warnings as well as errors |
| `--plot-only` | Re-plot the latest run |
| `-- <overrides>` | Everything after `--` is a training override, e.g. `-- --lr 2e-4` |

## Requirements

Training happens inside the container image, but **the helper scripts run on
the host**, so the host python needs `numpy`, `transformers`, `datasets` and
`pyyaml`, plus `torch` for Megatron datasets or `pyarrow` for MaxText ones, and
`matplotlib` for plots. The driver checks this before doing anything. `--image`
changes only the training environment; it has no effect on dataset preparation.

The default image is `container.options.image` in `runner/.primus.yaml` for
Megatron and `rocm/jax-training:maxtext-v26.7` (`MAXTEXT_DEFAULT_IMAGE` in the
driver) for MaxText.

## Fitting a time budget

To find out how many iterations fit in a time budget before committing:

```bash
tools/convergence_test/run_convergence_test.sh --model megatron/mixtral_8x7B_v0.1 \
    --probe 20 --budget-hours 3
```

A probe trains the first iterations of the full run's learning-rate schedule
and builds the dataset for the full run, so the run that follows reuses it. A
dataset that is too small for the run asked of it is rebuilt; the corpus stream
is deterministic, so the rebuild is a superset. A probe also evaluates once
(MaxText after its first step, Megatron after its last), so a broken validation
pipeline shows up in the probe rather than an hour into the real run.

A probe is too short to gate: in its first iterations the loss falls several
hundredths per step, so two healthy runs differ by more than the default
tolerance. Compare full runs.

---

## Release workflow

The bundled configs are fixed recipes on fixed, pre-tokenized data, so the same
config on two images differs only in the software stack. For each release
candidate:

```bash
# 1. Train, and gate against the previous release's curve
tools/convergence_test/run_convergence_test.sh --model maxtext/llama3.2_1B \
    --image rocm/jax-training:maxtext-v26.8 \
    --baseline baselines/maxtext-llama3.2_1B-v26.7.csv

# 2. Keep this run's CSV as the next baseline, with the header of its log,
#    which records the command, commits and config that produced it
cp output/convergence/llama3.2_1B-maxtext-convergence_<timestamp>.csv \
   baselines/maxtext-llama3.2_1B-v26.8.csv
```

The exit status is what automation should look at:

| Status | Meaning |
| --- | --- |
| 0 | trained every iteration (and matched the baseline, if one was given) |
| 3 | the loss moved from the baseline by more than `--tolerance` (default 0.05) |
| 4 | fewer iterations were logged than requested |
| other | lint error, training crashed (the trainer's own status), or the baseline could not be read |

Status 4 matters for MaxText in particular: it turns a failing data iterator,
or running out of data, into a graceful "Training stopped" and exits 0.

The comparison averages the training loss over the last 5% of the common
iterations (at least 10; single steps are noisy) and compares validation loss
at the last common evaluation. It reads a log, an experiment directory, or a
CSV this tool wrote, so baselines are small files you can check in anywhere.

The driver is safe to leave running while the checkout changes under it (a
`git pull` mid-run): bash parses the whole script before executing it.

To A/B a single knob, keep everything else fixed and pass it through:
`--env NVTE_CK_USES_FWD_V3=0` sets a container environment variable, and
anything after `--` is a training override, e.g. `-- --packing false`.

---

## What the pieces do

| File | Purpose |
| --- | --- |
| `run_convergence_test.sh` | End-to-end driver: dataset, lint, train, plot, gate |
| `resolve_config.py` | Resolves a config the way the trainer will; the driver's plan |
| `prepare_dataset.py` | Streams a real corpus into Megatron `.bin`/`.idx` or MaxText parquet |
| `check_config.py` | Flags settings that silently ruin a convergence run |
| `plot_loss.py` | Log to loss curve, CSV, health summary, baseline verdict |
| `configs/<backend>/` | Ready-to-run convergence configs |

Each script is usable on its own; the driver just chains them.

---

## 1. Build a dataset

```bash
# Megatron: indexed .bin/.idx
python3 tools/convergence_test/prepare_dataset.py \
    --model mixtral_8x7B_v0.1.yaml --target-tokens 600e6

# MaxText: pre-tokenized parquet, rows of at most --seq-length tokens
python3 tools/convergence_test/prepare_dataset.py --format maxtext \
    --model mixtral_8x7B.yaml --seq-length 4096 --target-tokens 600e6
```

The corpus is tokenised with **the vocabulary of the model you are going to
train**. This matters more than anything else here: a corpus built with the
128k Llama-3 vocab is unusable for a 32k Mixtral model, and the failure mode is
not a clean error. Passing `--model` resolves the tokenizer from the Primus
model preset of that `--format` so you cannot get it wrong.

`--source` and `--text-field` take the corpora listed in
[Choosing the data](#choosing-the-data). Sources stream, so nothing is
downloaded twice. C4 is validated on its official `validation` split; the
others have none, so their validation documents come from the head of the
training stream, ahead of the training documents. Output lands in
`$DATA_PATH/convergence/<source>-<tokenizer>/` (MaxText:
`<source>-<tokenizer>-maxtext-<seq-length>/`) with a `dataset_info.json`
manifest recording the format, corpus and splits, text column, tokenizer,
vocab size, and exact token counts. A directory is never reused for a
different corpus, format, vocabulary or MaxText row length (`--force`
rebuilds it), and one whose build was interrupted is rebuilt. Both formats
read the same document streams, so a Megatron and a MaxText dataset for the
same corpus and tokenizer hold the same documents.

A run on another corpus is a different experiment: compare it only with runs
on the same corpus. Earlier MaxText accuracy runs trained on C4 with the
Llama-2 tokenizer; `--model maxtext/llama2_7B --source c4` uses the same corpus
and vocabulary, though not their Hub-streamed, in-container data pipeline.

Roughly 6.5-7M tokens/s on the host, so 1B tokens takes about 3 minutes.
Megatron storage is 2 bytes/token for vocabularies under 65500 and 4 bytes
otherwise; MaxText parquet is zstd-compressed int32, about 1.7 bytes/token.

**Why MaxText gets pre-tokenized data.** MaxText can stream a Hub dataset and
tokenise it in the container (`dataset_type: hf`), which is what earlier
MaxText accuracy runs did. For a release test that is the wrong trade: the run
needs the network for hours, a `transformers` upgrade in the image silently
changes the data, and that path truncates every document at
`max_target_length`. `--format maxtext` instead writes every document as
`BOS + text + EOS`, exactly as MaxText's own tokenizer would, split into rows of
at most `--seq-length` tokens (grain rejects longer rows), and round-robined
over 64 parquet shards. The config reads them with `dataset_type: grain`,
`tokenize_train_data: false`, and grain packs rows into sequences with
per-document attention masking.

MaxText's grain pipeline pads with, and **drops from the loss**, the
tokenizer's pad id, falling back to unk and then to id 0. Llama-3 tokenizers
have neither, so every target equal to token 0 (`!`) would quietly vanish from
the loss. `--format maxtext` points pad at a reserved token instead
(`<|finetune_right_pad_id|>` for Llama-3.x); the vocabulary is unchanged.

A few presets name repos that cannot be loaded: `llama3.2_1B.yaml` (Megatron)
asks for `meta-llama/Meta-Llama-3.2-1B`, but the real repo drops the `Meta-`
prefix, and the MaxText `llama2_*` presets name Meta's original-format repos,
which `AutoTokenizer` cannot read. Nobody notices because mock and synthetic
data never load a tokenizer. `TOKENIZER_NAME_FIXES` corrects them; the
corrected repos are still gated and still need `HF_TOKEN`.

If you substitute a tokenizer by hand, use a **base** model, not an instruct
one. They often differ in `eos_token_id` — `Llama-3.3-70B-Instruct` reports
`128009` (`<|eot_id|>`) where the base model reports `128001`
(`<|end_of_text|>`) — and that id is appended to every document as EOD.

> The stock `examples/megatron/prepare.py` cannot do this job: it fetches
> BookCorpus through a dataset loading script, which `datasets>=3.0` refuses to
> execute. `examples/megatron/preprocess_data.py` cannot either: it imports
> `megatron.training`, which needs a GPU at import time, and its worker pool
> deadlocks because `huggingface_hub`'s fork handler logs through Primus'
> loguru bridge while the lock is held.

## 2. Lint the config

```bash
python3 tools/convergence_test/check_config.py --config <your-config.yaml> [--key value ...]
```

This resolves the config exactly as the trainer does, so inherited defaults are
checked too. Trailing `--key value` pairs are applied like primus-cli overrides;
the driver passes the same ones it trains with. Errors exit non-zero.

For MaxText, "exactly as the trainer does" includes MaxText's own merge: Primus
only carries an overlay, and MaxText layers `configs/base.yml`, the overlay,
and then `configs/models/<model_name>.yml` on top. The model file wins over
anything under `overrides:`, so **an override of a key the model file also
sets is silently discarded**; only `override_model:` beats it. The lint replays
that merge on the host (no JAX) and reports discarded keys as errors.

Running it against the stock benchmark configs is instructive (the MaxText
`vocab_size` line is what `--vocab_size 64000` on `llama2_7B` produces):

```
Megatron
ERROR mock_data is true -- the model trains on random tokens
ERROR moe_router_force_load_balancing is true -- Megatron replaces the router
      logits with random values, so the router never learns
WARN  lr is 1e-05, which is a throughput-benchmark value
WARN  use_turbo_deepep is true; with real routing this produced Inf gradients

MaxText
ERROR dataset_type is synthetic -- the model trains on random tokens
ERROR vocab_size: 64000 is discarded -- MaxText applies llama2-7b.yml
      (vocab_size: 32000) after the Primus overrides
WARN  weight_dtype is bfloat16: master weights and Adam moments are kept in
      that precision, so late-run updates are rounded away
WARN  load_balance_loss_weight is 0 (MaxText's default); with real routing the
      experts can collapse
```

Megatron: data plumbing (`mock_data`, missing `.idx` files, `split` set
alongside per-split paths), tokenizer/dataset vocabulary agreement, the LR
schedule, MoE routing flags, `deterministic_mode` prerequisites,
observability, and whether the token budget exceeds the dataset.

MaxText: discarded overrides; data plumbing (synthetic or Hub-streamed data,
missing grain files, more grain workers than files, pre-tokenized data with
`tokenize_train_data: true`, rows longer than `max_target_length`, `packing:
false`); a tokenizer larger than the model's `vocab_size` (JAX does not
bounds-check embedding lookups, so those ids train on garbage) and the pad-id
problem above; the LR schedule, including `learning_rate_schedule_steps <
steps`, after which MaxText sets the learning rate to 0; `weight_dtype`; MoE
balancing and capacity; observability; and whether the run needs more tokens
than the dataset holds (MaxText stops when the data runs out).

## 3. Train

The driver launches `primus-cli container -- train pretrain` and tees the
console to `<exp>[-<corpus>]_<timestamp>.log` (see
[What a run produces](#what-a-run-produces) and [Options](#options)).

Every MaxText run gets its own `run_name` (`<exp>-<timestamp>`), so its
TensorBoard file under `base_output_directory` is never mixed with an earlier
run's.

## 4. Plot

Run automatically by the driver, or on its own:

```bash
# One run: its console log, or the latest run in an experiment directory
python3 tools/convergence_test/plot_loss.py output/convergence/<exp>_<timestamp>.log
python3 tools/convergence_test/plot_loss.py output/amd/root/<exp>

# Compare two runs, including across backends
python3 tools/convergence_test/plot_loss.py runA.log runB.csv \
    --labels "megatron" "maxtext" --x tokens --seq-length 4096
```

The driver reads each run from its own console log, so a run that trained
nothing is never scored on an earlier one. Megatron needs a second source.
Primus prints its per-iteration line at `DEBUG` on the last rank and, on a
single node, forwards it to the console (every bundled config sets
`stderr_sink_level: DEBUG`, and the lint warns when a config does not), but it
does not forward evaluations. Those are only in the experiment directory's
`logs/pre_trainer/rank-<last>/debug.log`, which **appends** every run of the
experiment. The driver takes the validation points of the run there whose
losses match the console's iteration for iteration (`--validation-from`).
Pointed at an experiment directory directly, the script splits runs on the
iteration counter resetting and uses the last one; `--run-index` selects
another.

MaxText prints its per-step line at `INFO` on rank 0, with the loss to three
decimals and no learning rate or grad norm. `--tensorboard
<base_output_directory>/<run_name>` merges those and full-precision losses from
MaxText's TensorBoard file (read without a TensorFlow dependency); the driver
does this for you. MaxText counts steps from 0; they are reported as
iterations from 1, and an evaluation "after train step N" as iteration N+1,
so both backends line up. With `--x tokens`, MaxText runs are plotted against
the tokens actually trained (`total_weights`, padding excluded).

You get a PNG, a CSV, and a summary:

```
  iterations        : 1 -> 2000 (of 2000)
  lm loss           : 12.1470 -> 3.2851
  validation loss   : 6.0783 -> 3.2966
  initial vs ln(V)  : 12.15 vs 11.76 -- as expected
  median s/iter     : 1.3
  nan / skipped     : 0 / 0
  peak memory       : 55.8 GB (21.8%)
```

The `initial vs ln(V)` line is the cheapest correctness check available: an
untrained model over a vocabulary of size V must start at `ln(V)`. If iteration
1 is far off, the tokenizer and the data disagree and nothing downstream is
worth reading. The one benign offset is the initialisation: random logits with
standard deviation σ start about σ²/2 higher, and for an untied output layer σ
is roughly `init_method_std` × √hidden. Megatron's 0.02 at hidden 4096 adds
0.8, which is why the 4096-wide recipes use 0.008. `--detailed` adds learning
rate, grad norm, and throughput panels.

---

## Reference results

Measured on 1 node x 8 MI325X; Megatron on `rocm/primus:v26.7`, MaxText on
`rocm/jax-training:maxtext-v26.7`.

| Backend | Model | Corpus | Iters | Tokens | Wall clock | Loss | Valid | tokens/s/GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Megatron | Llama-3.2-1B BF16 | FineWeb-Edu | 2000 | 1.05B | 45 min | 12.15 -> 3.29 | 3.30 | 49200 |
| Megatron | Llama-2-7B BF16 | C4 | 1000 | 524M | 84 min | 10.50 -> 3.17 | 3.16 | 13800 |
| Megatron | Mixtral-8x7B BF16 | FineWeb-Edu | 300 | 157M | ~2h10m | 10.52 -> ~6.5 | | 2100-2800 |
| MaxText | Llama-3.2-1B BF16 | FineWeb-Edu | 2000 | 1.04B | 44 min | 12.23 -> 3.38 | 3.42 | 56800 |
| MaxText | Llama-2-7B BF16 | C4 | 1000 | 521M | 85 min | 10.86 -> 3.38 | 3.34 | 13600 |
| MaxText | Mixtral-8x7B BF16 | FineWeb-Edu | 300 | 155M | 75 min | 10.84 -> 4.62 | 4.72 | 4700 |

Use these as smoke-test targets, not as golden values. Run-to-run drift from
floating-point non-determinism is normal: two identical Mixtral runs matched
exactly at iteration 1, differed by 7e-6 at iteration 5, and by 2.8e-3 at
iteration 10. Do not attribute loss differences below ~0.01 to a config change
unless you ran with `--deterministic`.

MaxText's token count is the non-padding tokens it trained (98.8% of the slots
for Llama-3.2-1B); its tokens/s/GPU counts slots, like Megatron's.

The two backends train the same network on the same documents with the same
schedule. On the dense models MaxText ends higher: by about 0.1 on
Llama-3.2-1B (3.38 vs 3.29 train, 3.42 vs 3.30 validation) and about 0.2 on
Llama-2-7B (3.38 vs 3.17, 3.34 vs 3.16). They are not built to be bit-comparable:
MaxText packs documents with per-document attention masking and a BOS token
where Megatron concatenates them across attention; MaxText's loss includes the
EOS predictions that Megatron's `eod_mask_loss` drops; the initialisations and
the validation subsets differ. Gate each backend against its own baseline, and
treat a cross-backend gap that moves between releases as the signal.

On Mixtral the gap is far larger and goes the other way: MaxText reaches 4.62
(4.72 validation) with no loss spikes after warmup, while Megatron runs of the
same recipe on the same documents end near 6 and have shown mid-run spikes
(loss to 17.8, grad norm 106). That is worth investigating on the Megatron side.

---

## Reproducibility

Megatron: `seed: 1234` (from `trainer_base.yaml`) already fixes weight
initialisation and data order, which is what you need to compare two configs
fairly. Override with `--seed`. MaxText: `init_weights_seed` and
`data_shuffle_seed` (both 0) do the same.

Bit-exactness is a separate problem. `--deterministic` exports
`PRIMUS_DETERMINISTIC=1`, which sets `NCCL_ALGO=Ring`,
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `ROCBLAS_DEFAULT_ATOMICS_MODE=0`,
`TORCH_COMPILE_DISABLE=1`, and `PRIMUS_TURBO_AUTO_TUNE=0`. It costs throughput
but removes the largest sources of variation.

It does **not** make MaxText runs bit-exact: two identical Llama-3.2-1B runs
with `--deterministic` already differed at iteration 1 (by 3e-5) and by 6.5e-4
after 30 iterations. Something the flag does not pin, such as XLA's runtime
GEMM autotuning, still varies. The drift is two orders of magnitude below the
default `--tolerance`.

Megatron's stronger `deterministic_mode: true` additionally requires
`use_flash_attn: false` and `cross_entropy_loss_fusion: false`. Disabling flash
attention raises activation memory sharply, so it is usually impractical for a
large model at `seq_length 4096`. `tests/trainer/test_megatron_trainer.py`
exercises that path for `llama3_8B` and `deepseek_v2_lite` and asserts
bit-exact losses across two runs.

---

## MaxText notes

Things the MaxText recipes do differently from the perf configs, and why:

- **`weight_dtype: float32`.** `pre_trainer.yaml` sets `bfloat16`, which keeps
  master weights and Adam moments in bf16. Late in a run the updates fall below
  bf16 resolution and are rounded away. Compute stays in bf16 either way.
- **`load_balance_loss_weight: 0.01`.** MaxText's default is 0, so the perf
  MoE configs train the router with no balancing term at all.
- **`capacity_factor: 1.25`.** Same dense-dispatch MoE kernels as every Primus
  MaxText MoE config on AMD, with less token dropping than the perf configs' 1.
- **`profiler: ""`.** `pre_trainer.yaml` profiles step 3.
- **`eval_interval`/`eval_steps`.** MaxText evaluates nothing by default.

And things to know when reading a MaxText run:

- Do not rely on `metrics_file` with evaluation on. MaxText truncates it
  whenever it writes a step-0 record, and running evaluation metrics are
  written with the evaluation's own step counter, so every evaluation wipes
  it: an 8-step run evaluating every 3 steps kept only steps 6 and 7. The
  console log and TensorBoard are complete; `plot_loss.py` reads those.
- The first steps of a fresh container include JIT compilation of
  TransformerEngine's CK attention kernels (their cache lives in the
  container's `/root/.cache`), a few minutes of start-up on every run.
- Primus starts MaxText without `absl.app.run`; the trainer marks absl flags
  as parsed so grain's worker processes (`grain_worker_count > 0`) can start.

---

## Adding a model

Copy a bundled config from the same backend and change `model:` plus the
dataset paths. Megatron:

```yaml
model: qwen2.5_7B.yaml
overrides:
  mock_data: false
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen2.5-7b}/tokenizer
  train_data_path:
    - ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen2.5-7b}/train_text_document
  valid_data_path:
    - ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen2.5-7b}/valid_text_document
  test_data_path:
    - ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen2.5-7b}/valid_text_document
  split: null
```

MaxText (the directory carries the sequence length):

```yaml
model: qwen3_14B.yaml
overrides:
  dataset_type: grain
  grain_file_type: parquet
  grain_train_files: ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen3-14b-maxtext-4096}/train-*.parquet
  grain_eval_files: ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen3-14b-maxtext-4096}/valid-*.parquet
  train_data_columns: ["tokens"]
  eval_data_columns: ["tokens"]
  tokenize_train_data: false
  tokenize_eval_data: false
  tokenizer_type: huggingface
  tokenizer_path: ${PRIMUS_CONVERGENCE_DATA:./data/convergence/fineweb-edu-qwen3-14b-maxtext-4096}/tokenizer
  max_target_length: 4096
```

Name it `configs/<backend>/<model>-convergence.yaml` and `--model
<backend>/<model>` finds it. The default path keeps the config portable; the
driver overrides `PRIMUS_CONVERGENCE_DATA` with the absolute directory it
prepared. Then run `check_config.py` and fix whatever it reports.

Sane starting hyper-parameters: `lr` 1e-4 to 3e-4 with cosine decay to
`lr/10`, warmup at 2-10% of the run, the decay horizon equal to the run length
(Megatron `lr_decay_iters`, MaxText `learning_rate_schedule_steps: -1`), grad
clipping at 1.0, Adam beta2 0.95.

---

## Troubleshooting

**Inf or NaN gradients in the first few iterations on a Megatron MoE model.**
Turn off the Turbo MoE fast paths: `use_turbo_deepep: false` and
`turbo_sync_free_moe_stage: 0`. On MI325X, DeepEP produced `Inf` at iteration 3
on every attempt once `moe_router_force_load_balancing` was disabled; the
`alltoall` dispatcher is stable. This is the single biggest throughput cost of
running a *real* MoE convergence test (~2.1k versus ~6k tokens/s/GPU), so it is
worth re-testing as Primus-Turbo evolves.

**Out of memory on a Megatron MoE model.** The `alltoall` dispatcher allocates
per-expert buffers from actual token counts, and those grow as routing becomes
less balanced. Lower `micro_batch_size` and add `recompute_activations: true`
with `recompute_granularity: selective`. For Mixtral-8x7B, `mbs 2` peaks at 95%
of HBM and `mbs 1` at 88% for only 5% less throughput.

**A MaxText run "finished" with exit 0 but trained nothing, or stopped early.**
Look for `Training stopped:` in the console log; the driver prints it and
exits 4. `next(self.data_iterator) failed` means the data pipeline raised:
missing files, rows longer than `max_target_length`, or the data ran out.

**`grain_worker_count (N) exceeds the number of parquet files per host`.**
Each grain worker needs its own files. Lower `grain_worker_count` or rebuild
with more `--num-shards`.

**`EADDRINUSE` on port 1234.** A previous container is still holding the
torchrun master port. `docker rm -f $(docker ps -q -f name=primus-training)`.
The driver checks for this before launching.

**The loss is flat.** Check the learning rate and its decay horizon first, then
confirm iteration 1 is near `ln(vocab_size)`. `check_config.py` catches all of
these.

**Permission denied deleting `output/`.** The container runs as root, so its
logs are root-owned. Harmless: `plot_loss.py` handles appended runs, and the
dataset cache is hash-keyed so stale entries are ignored.
