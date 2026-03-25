# Checkpoint Management

Checkpoints capture **model state**, **optimizer state**, and **training progress** (iteration or step counters, schedulers, and related metadata). They are essential for **fault tolerance** (resume after failure), **experiment management** (reproducibility and comparison), and **hand-offs** between pretraining, fine-tuning, and conversion workflows.

Primus is YAML-driven: checkpoint behavior is configured per backend. **Megatron-LM**, **TorchTitan**, and **MaxText** each expose their own checkpoint surfaces; this guide maps the knobs you set in Primus configs.

**Primary sources in this repository**

| Area | File |
|------|------|
| Megatron trainer defaults | `primus/configs/modules/megatron/trainer_base.yaml` |
| Primus Megatron extensions | `primus/configs/modules/megatron/primus_megatron_module.yaml` |
| TorchTitan defaults | `primus/configs/modules/torchtitan/pre_trainer.yaml` |
| Megatron checkpoint benchmark | `benchmark/megatron/checkpoint/README.md` |

---

## 1. Overview

- **What is saved:** Typically model parameters, optimizer state, RNG state, and iteration/step tracking—exact contents depend on flags such as `no_save_optim` / `no_save_rng` (Megatron) or `initial_load_model_only` (TorchTitan).
- **Why it matters:** Long runs on AMD GPU clusters benefit from periodic saves to durable storage; resuming or branching experiments requires consistent paths and formats.
- **Backend-specific systems:** Each training backend integrates its own checkpoint pipeline; Primus wires YAML into those backends without forcing a single universal format across Megatron, TorchTitan, and MaxText.

---

## 2. Megatron checkpoint configuration

Megatron-related options live on the trainer configuration merged from `trainer_base.yaml` and `primus_megatron_module.yaml`. Defaults below are taken from `trainer_base.yaml` unless noted.

### Core paths and cadence

| Parameter | Default (`trainer_base.yaml`) | Description |
|-----------|-------------------------------|-------------|
| `save` | `null` | Directory where new checkpoints are written. |
| `load` | `null` | Directory to load from when **resuming** training. |
| `save_interval` | `20000` | Save every *N* iterations. |
| `finetune` | `false` | When `true`, loads weights but **resets** the iteration counter (typical fine-tune entry). |

### Format and detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ckpt_format` | `torch_dist` | Checkpoint format: `torch` (legacy single-file style), `torch_dist` (distributed), or `zarr`. |
| `auto_detect_ckpt_format` | `false` | When loading, infer format automatically. |
| `pretrained_checkpoint` | `null` | Path to a **pretrained** checkpoint. |
| `ckpt_step` | `null` | Load a specific step from the pretrained checkpoint when applicable. |

### Optimizer and RNG inclusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `no_save_optim` | `null` | When set truthy, **omit** optimizer state from saves. |
| `no_save_rng` | `null` | When set truthy, **omit** RNG state from saves. |
| `no_load_optim` | `null` | When set truthy, **do not** restore optimizer from checkpoint. |
| `no_load_rng` | `null` | When set truthy, **do not** restore RNG from checkpoint. |

### Performance and distributed I/O

| Parameter | Default | Description |
|-----------|---------|-------------|
| `async_save` | `null` | Asynchronous checkpoint saving to reduce time blocking the training loop. |
| `ckpt_fully_parallel_save` | `true` | Parallel save path for distributed checkpoints. |
| `ckpt_fully_parallel_load` | `false` | Parallel load for distributed checkpoints. |
| `ckpt_assume_constant_structure` | `false` | Optimization when model structure is fixed across saves/loads. |
| `non_persistent_save_interval` | `null` | Save to **fast local** storage on a different cadence than persistent saves. |

Related keys in `trainer_base.yaml` for non-persistent checkpoints include `non_persistent_ckpt_type`, `non_persistent_global_ckpt_dir`, `non_persistent_local_ckpt_dir`, and `non_persistent_local_ckpt_algo` (default `"fully_parallel"`).

### Format conversion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ckpt_convert_format` | `null` | Target format for conversion (`torch`, `torch_dist`, or `zarr`). |
| `ckpt_convert_save` | `null` | Output directory for converted checkpoints. |
| `ckpt_convert_update_legacy_dist_opt_format` | `false` | Update legacy distributed optimizer layout when converting. |

### Primus extensions

Defined in `primus/configs/modules/megatron/primus_megatron_module.yaml` and implemented in `primus/modules/trainer/megatron/trainer.py` and `primus/backends/megatron/patches/checkpoint_patches.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_continue_train` | `false` | When `true`, **automatically resume** from the latest checkpoint under `save` (adjusts load/finetune and related flags). |
| `disable_last_saving` | `false` | When `true`, **skip** the final checkpoint at shutdown (useful for benchmarking or when only periodic saves matter). |

---

## 3. TorchTitan checkpoint configuration

TorchTitan checkpoint options are grouped under `checkpoint` in `primus/configs/modules/torchtitan/pre_trainer.yaml`.

| Parameter | Default (`pre_trainer.yaml`) | Description |
|-----------|------------------------------|-------------|
| `checkpoint.enable` | `false` | Master switch for checkpointing. |
| `checkpoint.folder` | `checkpoint` | Output directory (relative to run layout unless given as absolute). |
| `checkpoint.interval` | `500` | Save every *N* **steps**. |
| `checkpoint.initial_load_path` | `null` | Path for **initial** load (cold start or migration). |
| `checkpoint.initial_load_model_only` | `true` | Load **weights only**, not optimizer state. |
| `checkpoint.initial_load_in_hf` | `false` | Load initial weights from **Hugging Face** layout. |
| `checkpoint.last_save_model_only` | `true` | On last save, write **model only**. |
| `checkpoint.last_save_in_hf` | `false` | Write final checkpoint in **Hugging Face** format. |
| `checkpoint.export_dtype` | `float32` | Dtype for exported checkpoints. |
| `checkpoint.async_mode` | `disabled` | Asynchronous checkpoint mode. |
| `checkpoint.keep_latest_k` | `10` | Retain only the **K** most recent checkpoints. |
| `checkpoint.load_step` | `-1` | Load a specific step (`-1` typically means latest or default behavior per backend). |
| `checkpoint.exclude_from_loading` | `[]` | Glob or pattern list to **exclude** from restore. |
| `checkpoint.enable_first_step_checkpoint` | `false` | Optional checkpoint at step 0. |
| `checkpoint.create_seed_checkpoint` | `false` | Create a seed checkpoint when enabled. |

TorchTitan also defines `activation_checkpoint` (activation recomputation) separately from persistent training checkpoints—do not confuse the two sections in `pre_trainer.yaml`.

---

## 4. MaxText checkpoint configuration

MaxText (JAX) uses configuration keys surfaced in Primus documentation and MaxText configs under `third_party/maxtext`. Typical training flags:

| Parameter | Typical default | Description |
|-----------|-----------------|-------------|
| `enable_checkpointing` | `false` | Enable Orbax (or configured) checkpoint saves. |
| `async_checkpointing` | `false` | When checkpointing is enabled, use **async** checkpoint workers. |

See `production_docs/03-configuration-reference/maxtext-parameters.md` for the full MaxText parameter table and interaction with training runs.

---

## 5. Checkpoint formats (Megatron)

| Format | Behavior | Notes |
|--------|----------|--------|
| `torch` | Classic PyTorch save/load; often **one file per rank** in distributed settings. | Simple but less flexible for topology changes. |
| `torch_dist` | **Distributed** checkpoint format with **resharding** support (e.g., changing tensor/pipeline parallel degree between save and load). | **Recommended** for many production flows that may change parallelism. |
| `zarr` | Zarr-backed checkpoint storage. | Useful when the stack and storage backend support it. |

**Recommendation:** Prefer `torch_dist` for production when you need **flexibility across parallel layouts** and scalable I/O (see `ckpt_fully_parallel_save` / `ckpt_fully_parallel_load` in Megatron config).

---

## 6. Common workflows

**Resume training**

- Set `load` to the checkpoint directory produced by a previous run.
- Keep `save` pointed at the directory for **new** checkpoints (often the same tree with a new run id, depending on your layout).
- Ensure `finetune` is `false` when you want to **continue** iteration counts.

**Fine-tune from a pretrained checkpoint**

- Set `load` (and optionally `pretrained_checkpoint` / `ckpt_step` as appropriate).
- Set `finetune: true` so iteration counters reset while weights load.

**Auto-resume (Primus Megatron extension)**

- Set `auto_continue_train: true` in the Megatron module config.
- Primus searches for the latest checkpoint under `save` and aligns load/optimizer flags; see `primus/modules/trainer/megatron/trainer.py` for behavior details.

**Convert checkpoint format**

- Set `ckpt_convert_format` (for example `torch_dist`) and `ckpt_convert_save` to the output directory.

**Import Hugging Face weights (TorchTitan)**

- Set `checkpoint.initial_load_in_hf: true` and `checkpoint.initial_load_path` to the HF model directory.

---

## 7. Benchmarking checkpoints

The Megatron checkpoint benchmark lives in `benchmark/megatron/checkpoint/`.

**Entry points**

- `benchmark/megatron/checkpoint/ckpt_launch.py` — main launcher (requires a Primus YAML config).
- `benchmark/megatron/checkpoint/ckpt_report.py` — reporting utility (can be run separately).

**Example** (from `benchmark/megatron/checkpoint/README.md`):

```bash
export DATA_PATH=/PATH/TO/DATA
python3 benchmark/megatron/checkpoint/ckpt_launch.py \
    --yaml-config-path examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-pretrain.yaml \
    --nnodes 1
```

The tool reports save/load times, bandwidth, and configuration echoes (world size, `ckpt_format`, `async_save`, paths, and more). Truncate or clean leftover output directories between runs if permissions or stale outputs cause issues.

---

## 8. Best practices

- Enable **`async_save`** (Megatron) for large models when supported, to limit training stalls during checkpoint windows.
- Set **`save_interval`** from **economic** criteria: frequent enough to limit lost work, infrequent enough to avoid storage and throughput bottlenecks (Megatron default in `trainer_base.yaml` is `20000`—override per job).
- Use **`non_persistent_save_interval`** with fast **local SSD** for frequent snapshots and a slower interval to **NFS** or object storage for durability.
- **Validate** resume and fine-tune paths on short runs before multi-week jobs; confirm `finetune` and `auto_continue_train` behave as intended.
- For TorchTitan, enable **`checkpoint.enable`** explicitly and set **`checkpoint.keep_latest_k`** to bound disk usage.

---

## Related documentation

- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
- [MaxText parameters](../03-configuration-reference/maxtext-parameters.md)
- [Benchmark suite](../02-user-guide/benchmarking.md)
