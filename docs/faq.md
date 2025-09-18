# FAQ: Frequently Asked Questions

---

## ❓ Primus CLI reports "config not found"?

Ensure the config file path is correct and relative to your working directory:

```bash
primus-cli direct -- train --config ./examples/configs/llama3_8B-pretrain.yaml
```

---

## ❓ GPUs not visible / PyTorch can't see HIP devices?

- Make sure ROCm is properly installed and accessible.
- Run the following to verify:

```bash
rocminfo
python -c "import torch; print(torch.cuda.is_available())"
```

If you get `False`, check if you're using the ROCm PyTorch wheel.

---

## ❓ Slurm training job hangs or has no output?

- Make sure `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and `NODE_RANK` are properly set (auto-inferred in most cases).
- Set debug logging:

```bash
export NCCL_DEBUG=INFO
export PRIMUS_DEBUG=1
```

---

## ❓ NCCL/RCCL errors about transport or P2P?

- If you're using MI300 across nodes, ensure IB or xGMI is active.
- Try disabling P2P for debug:

```bash
export RCCL_P2P_DISABLE=1
```

---

## ❓ Can I use Hugging Face tokenizers/models?

Yes! Just set `tokenizer_path` to a Hugging Face model name in your config:

```yaml
model:
  tokenizer_path: meta-llama/Llama-3-8B
```

---

## ❓ How do I resume training?

Either:

```bash
primus-cli direct -- train --config exp.yaml --load-dir ./ckpt/
```

Or add to your YAML:

```yaml
checkpoint:
  load_dir: ./ckpt/
```

---

## ❓ How to enable profiling?

Use the benchmark suite:

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 4096
```

Or use ROCm tools like `rocprof`, `rocm-smi`, or `rocminfo`.

---

_Last updated: 2025-09-17_
