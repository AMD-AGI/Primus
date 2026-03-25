# MaxText Backend Configuration Reference

Primus routes experiment YAML into the [MaxText](https://maxtext.readthedocs.io/) stack (JAX / XLA). Configuration is a **flat map of keys** (no nested `training.`* trees like TorchTitan): Primus merges module and model presets, writes a temporary YAML, and MaxText’s `pyconfig.initialize` loads it on top of upstream defaults.

The Primus overlay keeps `base_config: "base.yml"` so MaxText loads its own `[configs/base.yml](https://github.com/google/maxtext)` at runtime. This page lists **Primus-defined defaults and commonly overridden Primus fields**. For the full upstream parameter set (hundreds of keys), see the [MaxText documentation](https://maxtext.readthedocs.io/) and upstream `base.yml`.

---

## How parameters flow

1. YAML presets under `primus/configs/modules/maxtext/` and `primus/configs/models/maxtext/` are merged with CLI overrides.
2. `MaxTextAdapter.convert_config` passes the merged namespace through `MaxTextConfigBuilder` (currently a thin pass-through).
3. `export_params_to_yaml` writes a flat YAML file; MaxText ignores unknown Primus-private keys via pydantic filtering.
4. Unknown keys from upstream still resolve through environment overrides inside MaxText (`pyconfig`), not shown here.

---

## 1. Base module parameters

Shared with all Primus modules via `module_base.yaml` and trainer extensions.


| Parameter           | Default (Primus)                                                  | Description                                                                                  |
| ------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `trainable`         | `true` in `trainer_base.yaml` (overrides `module_base`’s `false`) | When `true`, the module participates in training orchestration.                              |
| `sink_level`        | `null`                                                            | Structured logging sink level for the module (if the logging stack is configured to use it). |
| `file_sink_level`   | `DEBUG`                                                           | File sink verbosity.                                                                         |
| `stderr_sink_level` | `INFO`                                                            | Stderr sink verbosity.                                                                       |


---

## 2. Training

From `pre_trainer.yaml` (extends `trainer_base.yaml`).


| Parameter     | Default      | Description                                                   |
| ------------- | ------------ | ------------------------------------------------------------- |
| `base_config` | `"base.yml"` | Upstream MaxText base file loaded by `pyconfig._load_config`. |
| `hardware`    | `"gpu"`      | Hardware target string consumed by MaxText.                   |
| `steps`       | `1000`       | Global optimizer steps for the run.                           |
| `log_period`  | `100`        | Steps between log emissions.                                  |


---

## 3. Data


| Parameter        | Default        | Description                                                          |
| ---------------- | -------------- | -------------------------------------------------------------------- |
| `dataset_type`   | `"hf"`         | Dataset backend selector (Hugging Face in the default path).         |
| `hf_path`        | `"allenai/c4"` | Hugging Face dataset repo or identifier.                             |
| `hf_data_dir`    | `"en"`         | Subdirectory / config slice within the HF dataset.                   |
| `hf_train_files` | `""`           | Optional explicit train file list (format per MaxText HF loader).    |
| `packing`        | `true`         | Sequence packing for efficiency when supported by the data pipeline. |


---

## 4. Checkpointing


| Parameter              | Default | Description                                                        |
| ---------------------- | ------- | ------------------------------------------------------------------ |
| `enable_checkpointing` | `false` | See Training section.                                              |
| `async_checkpointing`  | `false` | When `enable_checkpointing` is true, use async checkpoint workers. |


---

## 5. Profiling


| Parameter                         | Default    | Description                             |
| --------------------------------- | ---------- | --------------------------------------- |
| `profiler`                        | `"xplane"` | Profiler backend (e.g. XPlane for JAX). |
| `skip_first_n_steps_for_profiler` | `3`        | Warmup steps excluded from capture.     |
| `profiler_steps`                  | `1`        | Number of steps to profile once active. |


---

## 6. Memory and recomputation


| Parameter                       | Default  | Description                                                                        |
| ------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `remat_policy`                  | `'full'` | Activation rematerialization policy (`none`, `minimal`, `full`, etc.—see MaxText). |
| `optimizer_memory_host_offload` | `false`  | Offload optimizer state to host memory when supported.                             |
| `scan_layers`                   | `true`   | Use scanned layer implementation where applicable.                                 |
| `param_scan_axis`               | `1`      | Axis for parameter scanning / partitioning layout.                                 |


---

## 7. Precision and quantization


| Parameter                 | Default           | Description                                                           |
| ------------------------- | ----------------- | --------------------------------------------------------------------- |
| `dtype`                   | `"bfloat16"`      | Default compute dtype for many ops.                                   |
| `quantization`            | `""`              | Quantization mode string (empty = none; set per MaxText AQT recipes). |
| `quantize_kvcache`        | `false`           | Quantize KV cache tensors.                                            |
| `kv_quant_axis`           | `"heads_and_dkv"` | KV quantization axis naming for kernels.                              |
| `kv_quant_dtype`          | `"int8"`          | Storage dtype for KV cache when quantization is on.                   |
| `weight_dtype`            | `bfloat16`        | Weight storage / compute dtype for non-quantized paths.               |
| `checkpoint_is_quantized` | `false`           | Set `true` when loading an AQT-quantized checkpoint.                  |
| `logits_dot_in_fp32`      | `false`           | Compute logits matmul in `float32` for numerical stability.           |


---

## 8. Model

From `model_base.yaml` and per-model files such as `llama3_8B.yaml`.


| Parameter               | Default                                                             | Description                                                             |
| ----------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `model_name`            | `"default"` in `model_base`; e.g. `"llama3-8b"` in `llama3_8B.yaml` | Selects MaxText’s bundled model YAML when present.                      |
| `override_model_config` | `true`                                                              | When `true`, CLI / kwargs override values from the loaded model config. |
| `attention`             | `"cudnn_flash_te"`                                                  | Attention implementation (Primus default favors TE flash on AMD GPUs).  |
| `use_iota_embed`        | `true`                                                              | Use iota-based embedding for performance on accelerator backends.       |
| `tokenizer_path`        | e.g. `"meta-llama/Meta-Llama-3-8B"`                                 | Hugging Face tokenizer id or local path.                                |


---

## 9. Advanced


| Parameter | Default | Description                                                           |
| --------- | ------- | --------------------------------------------------------------------- |
| `shardy`  | `false` | Enable Shardy-related integration in MaxText when building shardings. |


---

## Related reading

- [MaxText documentation](https://maxtext.readthedocs.io/) — full parameter reference and recipes.
- Primus implementation: `primus/backends/maxtext/argument_builder.py`, `maxtext_pretrain_trainer.py`, `maxtext_adapter.py`.

