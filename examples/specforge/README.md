# SpecForge on Primus

`primus-cli` is the entrypoint for SpecForge **offline capture** and **draft
training**. SpecForge still owns data prep, draft configs, export, and SGLang
serving — follow the
[SpecForge AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md)
for those steps.

## Runtime image

Build and run: [`docker/README.md`](docker/README.md). Inside the container,
Primus is `/opt/primus` and SpecForge is `/workspace/SpecForge`.

## Launch with `primus-cli`

From `/opt/primus`:

```bash
./runner/primus-cli direct -- train pretrain --config <experiment.yaml>
```

Same command for capture and train; only the YAML changes. Example pair for
Qwen3.5-4B DFlash (20-step smoke defaults):

**Capture** — `specforge_mode: capture` runs SpecForge
`scripts/prepare_hidden_states.py`:

```yaml
# examples/specforge/configs/qwen3.5-4b-dflash-offline-capture.yaml
modules:
  pre_trainer:
    framework: specforge
    overrides:
      specforge_mode: capture
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}
      specforge_capture:
        target_model_path: ${TARGET_MODEL:Qwen/Qwen3.5-4B}
        data_path: ${CAPTURE_DATA_PATH}
        output_path: ${OUTPUT_DIR}/hidden_states_raw
        nproc_per_node: ${NPROC_PER_NODE:1}
        sglang_attention_backend: aiter
        sglang_disable_radix_cache: true
```

```bash
cd /opt/primus
export CAPTURE_DATA_PATH=/data/sharegpt.jsonl
export OUTPUT_DIR=/data/runs/capture
export NPROC_PER_NODE=8
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline-capture.yaml
```

**Train** — `specforge_config` points at SpecForge’s YAML; Primus forwards
overrides onto `specforge train`:

```yaml
# examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
modules:
  pre_trainer:
    framework: specforge
    overrides:
      specforge_config: ${SPECFORGE_CONFIG:/workspace/SpecForge/examples/configs/offline/colocated/qwen3.5-4b-dflash-offline-amd.yaml}
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}
      specforge_overrides:
        training.max_steps: ${MAX_STEPS:20}
        data.hidden_states_path: ${HIDDEN_STATES_PATH}
        deployment.trainer.nproc_per_node: ${NPROC_PER_NODE:1}
```

```bash
cd /opt/primus
export HIDDEN_STATES_PATH=/data/runs/capture/hidden_states
export OUTPUT_DIR=/data/runs/train
export NPROC_PER_NODE=8
export MAX_STEPS=20
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
```

Dotted CLI keys override YAML, for example
`specforge_overrides.training.max_steps=1000`.

## Reference results on MI355X

Qwen3.5-4B DFlash on 8× MI355X. ShareGPT was prepared as in the
[SpecForge AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md).
256 unique prompts were held out for eval and removed from train (exact-row
and first-user-turn leak filter).

Capture and train used the example YAMLs above via `primus-cli`, with
`NPROC_PER_NODE=8`. Capture:

```bash
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline-capture.yaml
```

Train, one epoch (~40k shards, `MAX_STEPS=2510`):

```bash
export MAX_STEPS=2510
export NPROC_PER_NODE=8
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml \
  specforge_overrides.training.num_epochs=1 \
  specforge_overrides.training.batch_size=2 \
  specforge_overrides.training.accumulation_steps=1 \
  specforge_overrides.training.save_interval=240 \
  specforge_overrides.training.log_interval=20
```

![train/loss and train/acc](assets/qwen35-4b-dflash-train-curves.png)

Held-out SGLang eval on **1 GPU**: `--tp-size 1`, `--max-running-requests 1`,
`--mem-fraction-static 0.75`, `--attention-backend aiter`,
`--disable-radix-cache`, `--disable-cuda-graph`, `--reasoning-parser qwen3`.
256 prompts, thinking on, EOS on, 64 new tokens, concurrency 1.

| | Target-only | DFLASH |
| --- | ---: | ---: |
| tok/s | 52.14 | **151.76** |
| MAL | — | **3.66** |
