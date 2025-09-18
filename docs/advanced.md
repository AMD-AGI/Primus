# Advanced Features

Primus provides powerful training features for large-scale foundation models, enabling optimal performance and flexibility on AMD GPUs.

---

## 🎯 Mixed Precision Support

Primus supports:

- `bf16` (recommended on ROCm 6.3+)
- `fp16` (with manual loss scaling or AMP)
- `fp8` (via TorchTitan backend)

Set via config:

```yaml
train:
  precision: bf16
```

---

## 🧠 Parallelism

| Type               | Description                           | Support |
|--------------------|---------------------------------------|---------|
| TP (Tensor)        | Partition model weights across GPUs   | ✔️      |
| PP (Pipeline)      | Stage-based model partitioning        | ✔️      |
| EP (Expert)        | MoE-style sparse activation           | ✔️      |
| VPP / CP           | Virtual pipeline / Chunk parallelism  | WIP     |

Configure in `parallelism` block:

```yaml
parallelism:
  tensor: 4
  pipeline: 2
  expert: 1
```

---

## 🧩 Checkpointing & Resuming

- Save model, optimizer, RNG states
- Resume with `--load-dir` or set in config:

```yaml
checkpoint:
  save_dir: ./ckpt/
  load_dir: ./ckpt/
  save_interval: 1000
```

---

## 📈 Evaluation & Logging

- Evaluate on validation set during training
- Configure via `eval:` block with frequency & dataset
- Log metrics to stdout and files (optional: tensorboard support)

---

## 🔍 Performance Optimization

- Enable `HipBLASLt` autotuning (enabled by default)
- Use `benchmark` mode to profile GEMM / RCCL
- Profile tokens/sec, TFLOPS, memory usage

---

_Last updated: 2025-09-17_
