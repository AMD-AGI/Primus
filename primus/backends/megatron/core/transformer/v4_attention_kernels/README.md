# DeepSeek-V4 attention kernels

This package holds every attention backend used by `DeepseekV4Attention`
(`primus/backends/megatron/core/transformer/deepseek_v4_attention.py`).
`__init__.py` is the single entry point: it maps each backend to its functional
entry, and the attention module imports everything from here.

## Attention variants

DeepSeek-V4 has three attention shapes, selected per layer by `compress_ratio`:

| `compress_ratio` | Variant   | What it does                                                        |
| ---------------- | --------- | ------------------------------------------------------------------- |
| `0`              | dense/SWA | Causal + sliding-window attention over the local KV.                |
| `128`            | HCA       | Hierarchical compressed attention: local KV concatenated with a compressed pool, joint softmax. |
| `4`              | CSA       | Compressed sparse attention: per-query top-K gather from the compressed pool, joint softmax. |

Two config selectors pick the kernel per group:

- **`use_v4_attention_backend`** → dense (`cr=0`) and HCA (`cr=128`) layers.
- **`use_v4_csa_attention_backend`** → CSA (`cr=4`) layers.

Both default to `gluon`.

## Backends

| Selector value | Applies to     | Folder                   | Entry point(s)                              | Notes                                                                 |
| -------------- | -------------- | ------------------------ | ------------------------------------------- | --------------------------------------------------------------------- |
| `eager`        | dense/HCA, CSA | `_eager/`                | `eager_v4_attention`, `eager_v4_csa_attention` | Pure-PyTorch reference. Bit-identical baseline shared with unit tests; slow, used for correctness. |
| `triton_v0`    | CSA only       | `_triton_v0_deprecated/` | `v4_csa_attention_v0`                        | **Deprecated.** Gathered per-query CSA with scalar GEMV (~30–260× slower). Not for production. |
| `triton_v1`    | dense/HCA, CSA | `_triton_v1/`            | `v4_attention_v1`, `v4_csa_attention_v1`     | Separate K/V, pool-based CSA + dense/HCA Triton kernels (previous production default). |
| `triton_v2`    | dense/HCA, CSA | `_triton_v2/`            | `v4_attention_v2`, `v4_csa_attention_v2`     | Fused single-latent sparse-MLA (K=V) using plain Triton `tl.dot` / MFMA. |
| `gluon`        | dense/HCA, CSA | `_gluon_dsa/`            | `v4_attention_gluon`, `v4_csa_attention_gluon` | **Default.** Hand-tuned fused single-latent sparse-MLA for gfx950 (CDNA4). |
| `flydsl_v0`    | CSA only       | `_flydsl_v0_deprecated/` | routed via `v4_csa_attention_v0` (`use_flydsl=True`) | **Deprecated**, forward-only legacy FlyDSL scalar CSA.               |

Support folders (not directly selectable):

- `_triton_common/` — shared Triton helpers (indexer, compressor, sinkhorn, RoPE, HC).
- `_flydsl_v1/` — WIP native FlyDSL MFMA sparse-MLA backend (forward kernel not yet implemented).
- `_tilelang/` — experimental TileLang path (not wired into the selectors).
- `v4_sparse_mla_adapter.py` — kernel-agnostic adapter mapping V4 tensors to the fused sparse-MLA interface (used by `triton_v2` / `gluon`).

Valid selector values (enforced in `DeepseekV4Attention.__init__`):

- dense/HCA: `eager | triton_v1 | triton_v2 | gluon`
- CSA: `eager | triton_v0 | triton_v1 | triton_v2 | gluon | flydsl_v0`

> Note: when a Turbo `core_attention` module is built, `use_turbo_attention`
> still takes precedence for the dense (`cr=0`) path.

## How to enable each backend

Set the two selectors to any valid value. All three mechanisms below set the
same config fields.

### 1. Config YAML

```yaml
# primus/configs/models/megatron/deepseek_v4_*.yaml
use_v4_attention_backend: gluon      # dense (cr=0) + HCA (cr=128)
use_v4_csa_attention_backend: gluon  # CSA (cr=4)
```

### 2. CLI flags

```bash
--use_v4_attention_backend triton_v2 \
--use_v4_csa_attention_backend gluon
```

### 3. Environment variables (root run scripts)

The `run_deepseek_v4*.sh` scripts read these and forward them as CLI flags:

```bash
export USE_V4_ATTENTION_BACKEND=triton_v1       # dense + HCA
export USE_V4_CSA_ATTENTION_BACKEND=triton_v1   # CSA
```

### Examples

```bash
# Default: gfx950 hand-tuned gluon backend everywhere
export USE_V4_ATTENTION_BACKEND=gluon
export USE_V4_CSA_ATTENTION_BACKEND=gluon

# Production Triton v1 (separate K/V) for all groups
export USE_V4_ATTENTION_BACKEND=triton_v1
export USE_V4_CSA_ATTENTION_BACKEND=triton_v1

# Fused single-latent sparse-MLA (Triton v2) for all groups
export USE_V4_ATTENTION_BACKEND=triton_v2
export USE_V4_CSA_ATTENTION_BACKEND=triton_v2

# Eager reference (correctness / debugging)
export USE_V4_ATTENTION_BACKEND=eager
export USE_V4_CSA_ATTENTION_BACKEND=eager
```

The two selectors are independent, so you can mix backends per group, e.g.
`USE_V4_ATTENTION_BACKEND=triton_v1` with `USE_V4_CSA_ATTENTION_BACKEND=gluon`.
