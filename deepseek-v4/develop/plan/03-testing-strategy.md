# 03 — Testing Strategy

> Phase mapping in [`02-phase-details.md`](02-phase-details.md).
> This document defines **how we validate** that each phase's output is
> actually correct.

## Test Pyramid

```
                ▲ convergence tests           (phase 8)
              ▲ end-to-end distributed smoke  (phase 6)
            ▲ multi-module forward alignment  (phase 4 / 5 / 6)
          ▲ unit tests (single module)        (phase 4 / 5)
        ▲ schema / import / arg checks         (phase 1 / 2 / 3)
```

## 1. Schema / Import / Arg Checks (cheapest, mandatory)

### 1.1 yaml schema validation

| Test | Goal | Command |
|---|---|---|
| `pytest tests/configs/test_deepseek_v4_yaml.py::test_load_base` | yaml parses through Primus loader | `pytest -k deepseek_v4_yaml` |
| `pytest tests/configs/test_deepseek_v4_yaml.py::test_extra_args_registered` | once the builder registers V4 fields into Megatron argparse, they parse | same |

### 1.2 Trainer dispatch

| Test | Goal | Check |
|---|---|---|
| `model_type=deepseek_v4` boots and imports `pretrain_deepseek_v4` | dispatch path works | trainer log shows `Using DeepSeek-V4 model provider` |

## 2. Unit Tests (mandatory in phase 4 / 5)

> Files live in `tests/backends/megatron/core/transformer/test_*.py`.
> Framework: pytest (existing) plus `tests/conftest.py` fixtures (`tiny_config` etc.).

| Module | Key tests |
|---|---|
| `hyper_connection.HyperConnection` | 1) Sinkhorn output is doubly stochastic (row/col sums ≈ 1, error < 1e-4 after 20 iter)<br>2) under fixed weights, max abs error < 1e-4 vs NeMo `DeepseekV4HyperConnection`<br>3) backward grads do not explode |
| `hyper_connection.HyperHead` | 1) sigmoid weighted sum matches manual computation<br>2) hc_mult=1 reduces to identity |
| `compressor.Compressor` (overlap=True, ratio=4) | 1) output shape `[B, 1, S/4, head_dim]`<br>2) under fixed weights, max abs error < 1e-4 vs reference `Compressor` |
| `compressor.Compressor` (overlap=False, ratio=128) | as above, shape `[B, 1, S/128, head_dim]` |
| `indexer.Indexer` | 1) output indices fall in `[-1, num_compressed_kv-1]`<br>2) causality: indices for query `q` are all `< (q+1)//ratio`<br>3) topk count == `index_topk` |
| `dual_rope` | 1) compress layers use `compress_rope_theta`, non-compress layers use `rotary_base`<br>2) partial RoPE rotates only the first `qk_pos_emb_head_dim=64` dims<br>3) YaRN scaling activates only when `compress_ratio != 0` |
| `clamped_swiglu` | 1) inputs outside the clamp range are truncated<br>2) within range, equivalent to `silu(x[:d])*x[d:]` |
| `csa_attention.CSASelfAttention` | 1) 1L toy model forward passes<br>2) sparse attn mask correct (causal + SWA + indexer top-K)<br>3) attn_sink probability column reduces final probs (sum < 1 after softmax slice) |
| `hca_attention.HCASelfAttention` | 1) as above<br>2) all queries see the full pool (after causal clamp) |
| `hash_router.HashRouter` | 1) the same token id always routes to the same expert across batches<br>2) under EP > 1, dispatch / drop is correct |
| `sqrtsoftplus_router` | 1) `sqrt(softplus(x))` numerics agree<br>2) interacts correctly with the noaux_tc bias |

Every unit test uses a **fixed seed** (`torch.manual_seed(0)`) plus **small
sizes** (hidden=128, S=64, hc_mult=2~4) and runs in < 5 seconds.

## 3. Module-Level Forward Alignment (mandatory in phase 4 / 5)

> Use the reference implementation as oracle, load identical weights, compare forward.

### Procedure

1. Take the reference `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py`,
   freeze the weights via the reference's own loader.
2. Build a **bit-equivalent 1L toy config** in Primus (hc_mult=4, 1 CSA + 1 HCA
   + 1 hash MoE) and copy the weights 1:1.
3. Run forward; compare hidden states at every layer entry / exit point via
   max abs error.

### Thresholds

| Dtype | Threshold |
|---|---|
| FP32 (oracle) | < 1e-5 |
| BF16 (training reality) | < 1e-2 |

### Test placement

- `tests/integration/deepseek_v4/test_forward_alignment.py`
- Fixtures stored in `tests/integration/deepseek_v4/fixtures/`.

## 4. End-to-End Distributed Smoke (mandatory in phase 6)

| Test | Configuration | Goal |
|---|---|---|
| 1-node 8-GPU BF16 | TP=1, PP=8, EP=1, DP=1 | PP partitioning carries 4-stream HC |
| 1-node 8-GPU BF16 | TP=2, PP=4, EP=1, DP=1 | TP partitioning across attention / Compressor / Indexer |
| 1-node 8-GPU BF16 | TP=1, PP=1, EP=8, DP=1 | EP routing (must work for hash + sqrtsoftplus) |
| 4-node 32-GPU BF16 | TP=1, PP=8, EP=4, DP=1 | realistic large-cluster config |

For each smoke:

- 50 iterations without crash
- loss monotone decreasing (occasional bumps allowed)
- grad-norm stable (< 1e3)
- save 1 ckpt + load round-trip

## 5. Convergence Tests (phase 8)

### 5.1 Short runs

- mock_data + 50 iter / 200 iter / 1000 iter
- compare BF16 vs FP8 loss curves
- compare AdamW vs Muon loss curves

### 5.2 Medium run

- bookcorpus tokenized data, 100M tokens
- shrunken V4-Flash 4L (first 4 layers)
- loss curve vs the relative numbers reported in the paper

### 5.3 Long run (one representative experiment)

- 1B tokens
- full V4-Flash 43L + 1 MTP
- save curve screenshots into `notes/2026-XX-convergence-1b.md`.

## 6. Performance Baseline (phase 8)

| Metric | Method | Expected (MI355X 8-GPU BF16) |
|---|---|---|
| **TFLOPs / GPU** | Megatron `log_throughput` | ≥ 80% of V3 (V4 loses some to HC 4×, gains some via sparse attn) |
| **Iter time** | timer output | 1×8 GPU BF16 batch_size=4096 < 5s/iter |
| **HBM usage** | `nvidia-smi` | < 220 GB/GPU (MI355X has 256 GB) |
| **Throughput (token/s)** | derived | comparable to V3 at the same scale |

## 7. Regression Tests

- On every PR merge: phase 1–3 schema + dispatch tests (lightweight, all green).
- Weekly: phase 4 / 5 unit tests (medium cost).
- Pre-release: phase 6 1-node smoke + phase 8 short-run convergence.

## 8. Test Tree

```
tests/
├── configs/
│   └── test_deepseek_v4_yaml.py
├── backends/megatron/
│   └── core/transformer/
│       ├── test_hyper_connection.py
│       ├── test_compressor.py
│       ├── test_indexer.py
│       ├── test_dual_rope.py
│       ├── test_clamped_swiglu.py
│       ├── test_csa_attention.py
│       ├── test_hca_attention.py
│       └── test_hash_router.py
└── integration/deepseek_v4/
    ├── test_forward_alignment.py
    ├── test_smoke_1n8gpu_bf16.py
    └── fixtures/
        └── tiny_v4_flash_weights.pt
```
