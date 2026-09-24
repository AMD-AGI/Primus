# Convergence Test

Train a model on a **real corpus** and check that the loss actually goes down.

Primus ships a lot of pretraining configs, but almost all of them are
throughput benchmarks: they run on `mock_data` (random tokens) with a learning
rate of `1e-5` and a two-iteration warmup. They are excellent for measuring
TFLOP/s and completely useless for telling you whether the stack learns. This
tool covers the other case.

It is a thin wrapper around the normal `primus-cli` pretrain path, so anything
you learn here transfers directly to a real training run.

---

## Quick start

```bash
# 1. Small dense model, ~45 minutes on 8x MI325X
tools/convergence_test/run_convergence_test.sh --model llama3.2_1B

# 2. MoE model, ~2h10m on 8x MI325X
tools/convergence_test/run_convergence_test.sh --model mixtral_8x7B_v0.1
```

That single command builds the dataset, lints the config, trains, and writes a
loss curve to `output/convergence/`. No Hugging Face token is required: the
corpus and all default tokenizers are ungated.

To find out how many iterations fit in a time budget before committing:

```bash
tools/convergence_test/run_convergence_test.sh --model mixtral_8x7B_v0.1 \
    --probe 20 --budget-hours 3
```

---

## What the pieces do

| File | Purpose |
| --- | --- |
| `run_convergence_test.sh` | End-to-end driver: dataset, lint, train, plot |
| `prepare_dataset.py` | Streams a real corpus into Megatron `.bin`/`.idx` |
| `check_config.py` | Flags settings that silently ruin a convergence run |
| `plot_loss.py` | Log to loss curve, CSV, and a health summary |
| `configs/` | Ready-to-run convergence configs |

Each script is usable on its own; the driver just chains them.

---

## 1. Build a dataset

```bash
python3 tools/convergence_test/prepare_dataset.py \
    --model mixtral_8x7B_v0.1.yaml --target-tokens 600e6
```

The corpus is tokenised with **the vocabulary of the model you are going to
train**. This matters more than anything else here: a corpus built with the
128k Llama-3 vocab is unusable for a 32k Mixtral model, and the failure mode is
not a clean error. Passing `--model` resolves the tokenizer from the Primus
model preset so you cannot get it wrong.

Sources are parquet-native and stream, so nothing is downloaded twice:
`fineweb-edu` (default), `c4`, `wikitext103`. Output lands in
`$DATA_PATH/convergence/<source>-<tokenizer>/` with a `dataset_info.json`
manifest recording the tokenizer, vocab size, and exact token counts.

Roughly 6.5M tokens/s on a 48-core host, so 600M tokens takes about 3 minutes.
Storage is 2 bytes/token for vocabularies under 65500 and 4 bytes otherwise, so
600M Mixtral tokens is 1.2 GB but 600M Llama-3 tokens is 2.4 GB.

Some model presets point at gated or non-existent repos (`llama3.2_1B.yaml`
asks for `meta-llama/Meta-Llama-3.2-1B`, which is not a real repo name — nobody
notices because `mock_data` swaps in `NullTokenizer`). Known cases are mapped to
ungated mirrors with identical vocabularies; add to `TOKENIZER_MIRRORS` as
needed, or pass `--tokenizer` explicitly.

> The stock `examples/megatron/prepare.py` cannot do this job: it fetches
> BookCorpus through a dataset loading script, which `datasets>=3.0` refuses to
> execute. `examples/megatron/preprocess_data.py` cannot either: it imports
> `megatron.training`, which needs a GPU at import time, and its worker pool
> deadlocks because `huggingface_hub`'s fork handler logs through Primus'
> loguru bridge while the lock is held.

## 2. Lint the config

```bash
python3 tools/convergence_test/check_config.py --config <your-config.yaml>
```

This resolves the config exactly as the trainer does (module preset + model
preset + overrides), so inherited defaults are checked too. Errors exit
non-zero. Running it against a stock benchmark config is instructive:

```
ERROR mock_data is true -- the model trains on random tokens
ERROR train_data_path is empty
ERROR moe_router_force_load_balancing is true -- Megatron replaces the router
      logits with random values, so the router never learns
WARN  lr is 1e-05, which is a throughput-benchmark value
WARN  use_turbo_deepep is true; with real routing this produced Inf gradients
```

It checks the data plumbing (`mock_data`, missing `.idx` files, `split` set
alongside per-split paths), that the dataset and config agree on a vocabulary,
the LR schedule, MoE routing flags, `deterministic_mode` prerequisites,
observability, and whether the token budget exceeds the dataset.

## 3. Train

The driver launches `primus-cli container -- train pretrain` and tees the
console to `output/convergence/<exp>_<timestamp>.log`. Useful options:

```
--train-iters N      Override length (also fixes lr_decay_iters)
--probe N            Run N iterations to measure throughput, then stop
--budget-hours H     Report the train_iters that fit in H hours
--deterministic      PRIMUS_DETERMINISTIC=1 for a repeatable run
--image <image>      Override the docker image
--strict             Fail on lint warnings too
--                   Everything after this is forwarded to primus-cli
```

## 4. Plot

Run automatically by the driver, or on its own:

```bash
# Latest run in an experiment directory
python3 tools/convergence_test/plot_loss.py output/amd/root/<exp>

# Compare two runs
python3 tools/convergence_test/plot_loss.py runA runB \
    --labels "rocm 7.14" "rocm 7.15" --x tokens --seq-length 4096
```

Primus emits Megatron's per-iteration line at `DEBUG`, so the numbers live in
`logs/pre_trainer/rank-<last>/debug.log`, not on the console. Point the script
at the experiment directory and it finds the right rank. Re-running an
experiment **appends** to the same file, so it splits runs on the iteration
counter resetting and uses the last one; `--run-index` selects another.

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
worth reading. `--detailed` adds learning rate, grad norm, and throughput
panels.

---

## Reference results

Measured on 1 node x 8 MI325X, `rocm/primus:v26.7`, FineWeb-Edu.

| Model | Iters | Tokens | Wall clock | Loss | tokens/s/GPU |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B BF16 | 2000 | 1.05B | 45 min | 12.15 -> 3.29 | 49200 |
| Mixtral-8x7B BF16 | 300 | 157M | ~2h10m | 10.52 -> ~6.5 | 2100-2800 |

Use these as smoke-test targets, not as golden values. Run-to-run drift from
floating-point non-determinism is normal: two identical Mixtral runs matched
exactly at iteration 1, differed by 7e-6 at iteration 5, and by 2.8e-3 at
iteration 10. Do not attribute loss differences below ~0.01 to a config change
unless you ran with `--deterministic`.

---

## Reproducibility

`seed: 1234` (from `trainer_base.yaml`) already fixes weight initialisation and
data order, which is what you need to compare two configs fairly. Override with
`--seed`.

Bit-exactness is a separate problem. `--deterministic` exports
`PRIMUS_DETERMINISTIC=1`, which sets `NCCL_ALGO=Ring`,
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `ROCBLAS_DEFAULT_ATOMICS_MODE=0`,
`TORCH_COMPILE_DISABLE=1`, and `PRIMUS_TURBO_AUTO_TUNE=0`. It costs throughput
but removes the largest sources of variation.

Megatron's stronger `deterministic_mode: true` additionally requires
`use_flash_attn: false` and `cross_entropy_loss_fusion: false`. Disabling flash
attention raises activation memory sharply, so it is usually impractical for a
large model at `seq_length 4096`. `tests/trainer/test_megatron_trainer.py`
exercises that path for `llama3_8B` and `deepseek_v2_lite` and asserts
bit-exact losses across two runs.

---

## Adding a model

Copy a bundled config and change `model:` plus the dataset paths:

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

Name it `configs/<model>-convergence.yaml` and `--model <model>` finds it. The
default path keeps the config portable; the driver overrides
`PRIMUS_CONVERGENCE_DATA` with the absolute directory it prepared. Then run
`check_config.py` and fix whatever it reports.

Sane starting hyper-parameters: `lr` 1e-4 to 3e-4 with cosine decay to
`lr/10`, `lr_warmup_iters` at 2-10% of `train_iters`, `lr_decay_iters` equal to
`train_iters`, `clip_grad: 1.0`, `adam_beta2: 0.95`.

---

## Troubleshooting

**Inf or NaN gradients in the first few iterations on an MoE model.** Turn off
the Turbo MoE fast paths: `use_turbo_deepep: false` and
`turbo_sync_free_moe_stage: 0`. On MI325X, DeepEP produced `Inf` at iteration 3
on every attempt once `moe_router_force_load_balancing` was disabled; the
`alltoall` dispatcher is stable. This is the single biggest throughput cost of
running a *real* MoE convergence test (~2.1k versus ~6k tokens/s/GPU), so it is
worth re-testing as Primus-Turbo evolves.

**Out of memory on an MoE model.** The `alltoall` dispatcher allocates
per-expert buffers from actual token counts, and those grow as routing becomes
less balanced. Lower `micro_batch_size` and add `recompute_activations: true`
with `recompute_granularity: selective`. For Mixtral-8x7B, `mbs 2` peaks at 95%
of HBM and `mbs 1` at 88% for only 5% less throughput.

**`EADDRINUSE` on port 1234.** A previous container is still holding the
torchrun master port. `docker rm -f $(docker ps -q -f name=primus-training)`.
The driver checks for this before launching.

**The loss is flat.** Check `lr` and `lr_decay_iters` first, then confirm
iteration 1 is near `ln(vocab_size)`. `check_config.py` catches all three.

**Permission denied deleting `output/`.** The container runs as root, so its
logs are root-owned. Harmless: `plot_loss.py` handles appended runs, and the
dataset cache is hash-keyed so stale entries are ignored.
