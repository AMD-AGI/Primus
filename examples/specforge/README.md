# SpecForge on Primus

`primus-cli` is the entrypoint for SpecForge **capture** and **draft training**.
SpecForge still owns data prep, draft configs, export, and SGLang serving —
follow the
[SpecForge AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md)
for those steps.

## Intro

Two knobs:

| Knob | Values | Meaning |
| --- | --- | --- |
| `specforge_mode` | `train` \| `capture` | Which program Primus runs (`train` if omitted) |
| `specforge_train_mode` | `online` \| `offline` | How train runs. **Required** when mode is `train` (no default). Ignored for capture. |

`offline` train is `specforge train` on pre-captured hidden states. `online`
train is live Mooncake + SGLang capture plus `--role producer` / `--role
consumer`. Capture is SpecForge `scripts/prepare_hidden_states.py`, not a
train mode.

Build and run the overlay: [`docker/README.md`](docker/README.md). Inside the
container, Primus is `/opt/primus` and SpecForge is `/workspace/SpecForge`.

From `/opt/primus`:

```bash
./runner/primus-cli direct -- train pretrain --config <experiment.yaml>
```

Same command for capture and train; only the YAML changes. Dotted CLI keys
override YAML, for example `specforge_overrides.training.max_steps=1000`.

## Offline

Capture, then train on disk hidden states. Example pair for Qwen3.5-4B DFlash
(20-step smoke defaults).

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

**Train** — `specforge_mode: train` with `specforge_train_mode: offline`.
`specforge_config` points at SpecForge’s YAML; Primus forwards overrides onto
`specforge train`:

```yaml
# examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
modules:
  pre_trainer:
    framework: specforge
    overrides:
      specforge_mode: train
      specforge_train_mode: offline
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

## Online

`specforge_mode: train` with `specforge_train_mode: online` is live
capture+train. SpecForge's own CLI will not start Mooncake/SGLang across
nodes, so Primus does that and then launches `--role producer` /
`--role consumer`.

One Slurm allocation, the same `primus-cli` command on every node. Shape is
**C capture nodes + T trainer nodes** (`CAPTURE_NNODES` + `TRAINER_NNODES`,
default 1+1 so `-N 2`):

| Slurm `NODE_RANK` | Role |
| --- | --- |
| `0 .. C-1` | Capture: SGLang on GPU. Rank 0 also starts Mooncake and the SpecForge **producer** (CPU HTTP client to those servers). |
| `C .. C+T-1` | Wait for `inference.ready` → `--role consumer` (`--node-rank` when `T>1`) |

Checked-in SpecForge YAML keeps `127.0.0.1`. After allocate, Primus injects a
routable `HEAD_IP` (override with `PRIMUS_SPECFORGE_HEAD_IP`, or set
`PRIMUS_SPECFORGE_BIND_IFACE` to pick a NIC). `control_dir` / `output_dir`
must be on shared storage; `consumer_state_dir` must be trainer-local. SpecForge
refuses a reused control/SQLite tree, so `RUN_ROOT` and `CONSUMER_STATE_DIR`
must be empty (a new `RUN_ID` is the usual way to get that).

`./runner/primus-cli slurm` runs **on the login node** from this checkout; it
does not need a pip-installed Primus. The overlay image (or a bind-mount of
this tree at `/opt/primus`) is what actually trains. `--gres=gpu:N` is Slurm's
**per-node** GPU request — the same `N` on every allocated node. Set it to the
GPUs that node uses (`SERVER_COUNT` on a capture node, `NPROC_PER_NODE` on a
trainer node).

```bash
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT=/path/to/shared/run/$RUN_ID
export OUTPUT_DIR=$RUN_ROOT/output
export CONSUMER_STATE_DIR=/path/to/local/nvme/specforge/$RUN_ID/consumer-state
export TRAIN_DATA_PATH=/path/to/sharegpt_train.jsonl
export MAX_STEPS=20
cd /opt/primus
./runner/primus-cli slurm srun -N 2 \
  --gres=gpu:1 \
  -- container --image primus-specforge:v0.5.14-rocm700-mi35x \
  --volume /shared:/shared \
  -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-online-2node.yaml
```

The example above is **1 capture GPU + 1 trainer GPU** on 2 nodes. Load the
overlay image on **every** node if the scheduler's container store is
node-local. `GPUS_PER_NODE=1` from the prepare hook means one Primus process
per node, not the device mask: the producer has empty `CUDA_VISIBLE_DEVICES`;
SGLang/consumer use `SERVER_GPUS` / `TRAINER_GPUS`.

Do not wrap this in `managed_local` or `--role both`. SpecForge
`deployment.trainer.nnodes` is `TRAINER_NNODES` (consumer nodes only). Raise
`SERVER_COUNT` / `TRAINER_GPUS` / `--gres` for 8 GPUs per role on 2 nodes, or
raise `-N` with `CAPTURE_NNODES` / `TRAINER_NNODES`.

## Reference results on MI355X

Qwen3.5-4B DFlash, one epoch. ShareGPT prompts were leak-filtered (exact-row
and first-user-turn) and **256 unique rows** were held out for eval. Train
labels are the target’s own **greedy thinking** continuations
(`temperature=0`, reasoning saved) — not original ShareGPT assistant text.

Both runs used the example YAMLs above via `primus-cli`, 8 trainer GPUs,
batch 2, accumulation 1, `log_interval=20`, `save_interval=240`,
`MAX_STEPS=2510` (~40k refs). Offline captured hidden states to disk
(`max_length=2048`) then trained. Online recaptured the same train prompts
live (8 SGLang `TP=1` + 8 trainer GPUs). Curves are in the same band, not
bit-identical.

Offline train:

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

Online train (8+8):

```bash
export MAX_STEPS=2510
export NPROC_PER_NODE=8
export SERVER_COUNT=8
export SERVER_TP=1
./runner/primus-cli slurm srun -N 2 --gres=gpu:8 \
  -- container --image primus-specforge:v0.5.14-rocm700-mi35x \
  --volume /shared:/shared \
  -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-online-2node.yaml \
  specforge_overrides.training.num_epochs=1 \
  specforge_overrides.training.batch_size=2 \
  specforge_overrides.training.accumulation_steps=1 \
  specforge_overrides.training.save_interval=240 \
  specforge_overrides.training.log_interval=20 \
  specforge_overrides.runtime.in_flight_high_watermark=64 \
  specforge_overrides.runtime.in_flight_low_watermark=32 \
  specforge_online.mooncake_lease_ttl_ms=8000
```

![train/loss, offline vs online](assets/train-loss-online-vs-offline.png)

![train/acc, offline vs online](assets/train-acc-online-vs-offline.png)

Held-out SGLang eval on **1 GPU** after `specforge export --to hf` and
normalizing the draft to `DFlashDraftModel` / `block_size=16`: `--tp-size 1`,
`--max-running-requests 1`, `--mem-fraction-static 0.75`,
`--attention-backend aiter`, `--disable-radix-cache`, `--disable-cuda-graph`,
`--reasoning-parser qwen3`. 256 prompts, thinking on, EOS on, 64 new tokens,
concurrency 1.

| | Offline | Online |
| --- | ---: | ---: |
| Target-only tok/s | 52.14 | 52.12 |
| DFLASH tok/s | **151.76** | **156.45** |
| MAL | **3.662** | **3.649** |
| Speedup | 2.911x | 3.002x |
