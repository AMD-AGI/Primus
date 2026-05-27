# 01 — Code Landing List for DeepSeek-V4 in Primus

> This document only lists **file paths and their responsibilities**, not
> implementation details (those live in `02-phase-details.md`).
> Path prefix: `+ ` = new file; `~ ` = modify existing file; `!` = create a new sub-package.

## 0. Top-Level Layout

```
primus/
├── backends/megatron/
│   ├── megatron_pretrain_trainer.py             ~ add deepseek_v4 dispatch branch
│   ├── core/
│   │   ├── models/
│   │   │   └── deepseek_v4/                     ! new sub-package — V4-specific model / spec / block
│   │   │       ├── __init__.py                  +
│   │   │       ├── deepseek_v4_layer_specs.py   + LayerSpec (attn / moe / hash, three kinds)
│   │   │       ├── deepseek_v4_block.py         + 4-stream HC TransformerBlock (replaces Megatron's TransformerBlock)
│   │   │       ├── deepseek_v4_model.py         + top-level Model (embed → block → HyperHead → lm_head + MTP)
│   │   │       └── README.md                    + entry doc for this sub-package
│   │   ├── transformer/
│   │   │   ├── hyper_connection.py              + mHC (compute_weights / collapse / expand / HyperHead)
│   │   │   ├── compressor.py                    + Compressor (overlap pool ratio=4 / non-overlap ratio=128)
│   │   │   ├── indexer.py                       + Indexer (CSA top-K selector)
│   │   │   ├── csa_attention.py                 + CSA SelfAttention (Compressor + Indexer + sparse-attn-w-mask)
│   │   │   ├── hca_attention.py                 + HCA SelfAttention (heavy compress, no Indexer)
│   │   │   ├── sliding_window_kv.py             + SWA buffer manager (last K tokens of KV)
│   │   │   ├── attn_sink.py                     + per-head learnable scalar concatenated into softmax
│   │   │   ├── dual_rope.py                     + dual RoPE base (rotary_base / compress_rope_theta) + partial RoPE + selective YaRN
│   │   │   ├── attention.py                     ~ existing: add a V4 dispatcher branch (only if needed)
│   │   │   └── multi_latent_attention.py        ~ existing: keep as the V3 MLA reference; verify V4 does not reuse and goes via the new modules
│   │   ├── fusions/
│   │   │   └── clamped_swiglu.py                + clamped SwiGLU activation
│   │   └── optimizer/
│   │       └── (reuse existing muon)            ~ extend muon param-group split with V4-specific params (HC weight, MTP head)
│   ├── patches/
│   │   ├── deepseek_v4_layer_specs_patches.py   + at build_args, register V4 LayerSpec into Megatron's model_provider
│   │   ├── hyper_connection_patches.py          + swap Megatron TransformerBlock for DeepseekV4Block (only when model_type==deepseek_v4)
│   │   ├── csa_hca_attention_patches.py         + (optional) register V4 attention classes via patch
│   │   ├── moe_patches/
│   │   │   ├── hash_router_patches.py           + Hash routing router
│   │   │   └── sqrtsoftplus_router_patches.py   + sqrtsoftplus + noaux_tc scoring
│   │   └── mtp_v4_patches.py                    + MTP head: separate HC head config, shared embedding/lm_head
│   └── training/
│       └── (no change; reuse training.py)
│
├── configs/models/megatron/
│   ├── deepseek_v4_base.yaml                    + V4 shared defaults (HC, hybrid attn switches)
│   ├── deepseek_v4_flash.yaml                   + V4-Flash hyperparameters (43 layers, 64 heads, etc.)
│   └── deepseek_v4_pro.yaml                     + V4-Pro hyperparameters
│
└── pretrain_deepseek_v4.py                      + builder entry; loaded via lazy_import by the trainer when Megatron starts
                                                   (peer of pretrain_gpt.py / pretrain_mamba.py, but lives inside the
                                                    primus repo and is injected via sys.path; see Phase 2)

examples/megatron/configs/MI355X/
├── deepseek_v4_flash-BF16-pretrain.yaml         + main training yaml (BF16 first, FP8 later)
└── deepseek_v4_flash-FP8-pretrain.yaml          + added later in Phase 8

deepseek-v4/develop/                             ← design docs only, **not production code**
```

## 1. Phase × File Mapping (the most important table)

| Phase | New / modified files | Depends on | Notes |
|---|---|---|---|
| **P1** | `+ primus/configs/models/megatron/deepseek_v4_base.yaml`<br>`+ primus/configs/models/megatron/deepseek_v4_flash.yaml`<br>`+ examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` | Only the Megatron arg schema | Every yaml field must map to a Megatron CLI argument; the new fields (HC / Hybrid Attn / Hash MoE) are injected into argparse via a P2 patch. |
| **P2** | `+ primus/pretrain_deepseek_v4.py`<br>`~ primus/backends/megatron/megatron_pretrain_trainer.py`<br>`~ primus/core/utils/import_utils.py` (add `deepseek_v4` branch in `get_model_provider`)<br>`+ primus/backends/megatron/core/models/deepseek_v4/__init__.py`<br>`+ primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py` (stub) | P1 | The stub model = GPTModel + a dummy `Block` swap-in; HC not implemented yet, the goal is for end-to-end import to succeed. |
| **P3** | `+ primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py`<br>`+ primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py` (HC=1 stream) | P2 | LayerSpec only points at attention=MLA-like + ffn=MoE; HC is still 1 stream (hc_mult=1 degenerates to baseline). |
| **P4** | `+ primus/backends/megatron/core/transformer/hyper_connection.py`<br>`+ primus/backends/megatron/core/transformer/compressor.py`<br>`+ primus/backends/megatron/core/transformer/indexer.py`<br>`+ primus/backends/megatron/core/transformer/csa_attention.py`<br>`+ primus/backends/megatron/core/transformer/hca_attention.py`<br>`+ primus/backends/megatron/core/transformer/sliding_window_kv.py`<br>`+ primus/backends/megatron/core/transformer/attn_sink.py`<br>`+ primus/backends/megatron/core/transformer/dual_rope.py`<br>`~ primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py` (enable 4-stream HC)<br>`+ primus/backends/megatron/patches/hyper_connection_patches.py` | P3 | Core focus; inject V4 attention into the spec via patch. |
| **P5** | `+ primus/backends/megatron/patches/moe_patches/hash_router_patches.py`<br>`+ primus/backends/megatron/patches/moe_patches/sqrtsoftplus_router_patches.py`<br>`+ primus/backends/megatron/core/fusions/clamped_swiglu.py`<br>`+ primus/backends/megatron/patches/mtp_v4_patches.py`<br>`~ primus/backends/megatron/core/transformer/dual_rope.py` (layer-aware YaRN scaling) | P3 (parallel with P4) | MoE / activation / RoPE / MTP are modular; can be parallelized across people. |
| **P6** | `~ primus/backends/megatron/megatron_pretrain_trainer.py` (replace P2 stub builder with the real model from P3+P4+P5)<br>`~ examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` (PP layout, recompute settings) | P4 + P5 | Validate 1×8 BF16 first. |
| **P7** | `~ examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` (enable muon)<br>`~ primus/backends/megatron/patches/muon_optimizer_patches.py` (V4-specific param groups: HC weight / MTP head go to AdamW; attention/MoE weights go to Muon) | P6 | Reuse the existing muon framework, only extend param-group split. |
| **P8** | `+ examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml`<br>`+ deepseek-v4/develop/notes/<convergence-runs>.md`<br>(optional) `+ primus/backends/megatron/extensions/v4_kernels.py` (fused sparse-attn / Compressor) | P6 + P7 | FP4 is further optional. |

## 2. Interface Contracts (the two most important)

### 2.1 `model_type` selection

`primus/configs/models/megatron/deepseek_v4_base.yaml` must contain:

```yaml
model_type: deepseek_v4   # string literal; the trainer uses this to pick the builder
```

`MegatronPretrainTrainer.train()` adds a branch:

```python
elif model_type == "deepseek_v4":
    from pretrain_deepseek_v4 import (
        forward_step,
        train_valid_test_datasets_provider,
    )
```

`primus/core/utils/import_utils.py:get_model_provider` adds:

```python
elif model_type == "deepseek_v4":
    model_provider = lazy_import(
        ["model_provider", "pretrain_deepseek_v4"], "model_provider", ...)
    deepseek_v4_builder = lazy_import(
        ["deepseek_v4_builders"], "deepseek_v4_builder", ...)
    return partial(model_provider, deepseek_v4_builder)
```

> We follow the same two-layer structure as mamba: the upper-level `model_provider`
> (in `primus/pretrain_deepseek_v4.py`) is just a shell, the actual implementation
> lives in `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py`.

### 2.2 hidden-shape contract

**All V4 internals flow as `[B, S, hc_mult, D]`**; only the boundaries entering /
leaving the transformer block do collapse. Specifically:

- entering the model: after embed, `unsqueeze(2).expand(-1, -1, hc_mult, -1)`
- inside the transformer block: HC's 4 streams in parallel
- leaving the block: HyperHead computes `Σ_h sigmoid(...) · x[h]` → `[B, S, D]`
- after that, RMSNorm / lm_head behave the same as GPT

This contract is documented in `deepseek_v4_block.py`'s docstring; all upstream
and downstream modules respect it.

## 3. Mapping to the Reference Implementation

| Reference module (`deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py`) | Primus location |
|---|---|
| `Compressor` | `primus/backends/megatron/core/transformer/compressor.py` |
| `Indexer` | `primus/backends/megatron/core/transformer/indexer.py` |
| `Attention` (with CSA / HCA branches) | split into `csa_attention.py` + `hca_attention.py` (shared parts in a base class) |
| `attn_sink` (in `Attention.forward`) | `attn_sink.py` |
| `Block.attn_hc / ffn_hc` call chain | `hyper_connection.py` |
| `MoE` + `Gate` (`sqrtsoftplus`) | `patches/moe_patches/sqrtsoftplus_router_patches.py` |
| `MoE` + `Gate` (hash) | `patches/moe_patches/hash_router_patches.py` |
| `Block`'s `compress_ratio` switch | `deepseek_v4_block.py` decides which attention each layer uses |
| `Transformer.head_fn` (HC head) | `hyper_connection.py:HyperHead` |
| `MTPBlock` | `patches/mtp_v4_patches.py` |
| Dual RoPE: `precompute_freqs_cis(theta=rotary_base)` and `precompute_freqs_cis(theta=compress_rope_theta)` | `dual_rope.py` |
| `clamp_swiglu` | `clamped_swiglu.py` |

## 4. Mapping to NeMo Automodel (secondary reference)

`deepseek-v4/NVIDIA-NeMo/Automodel/nemo_automodel/components/models/deepseek_v4/`

- `layers.py` — we mirror its
  - `_apply_partial_rope_interleaved` (into `dual_rope.py`)
  - `_overlap_transform` (into `compressor.py`)
  - `_pool_windows` (into `compressor.py`)
  - `DeepseekV4HyperConnection` (into `hyper_connection.py`)
- `config.py` — used to cross-check our yaml field naming.

> NeMo also uses mcore but does it inline rather than via patches. We stick
> with the patch pattern to remain consistent with Primus conventions.
