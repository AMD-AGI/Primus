# DeepSeek-V4 Tech Blog

Engineering reference for adding DeepSeek-V4 support to Primus.

> Active plan of record (2026-05-01 onward):
> [`develop/plan-2/`](../plan-2/README.md) — architecture-faithful rewrite.
> See [`02-plan-1-as-built-and-plan-2-pointer.md`](02-plan-1-as-built-and-plan-2-pointer.md)
> for the as-built notes that close out plan-0 / plan-1 and the rationale for plan-2.

## Documents

| File | Contents |
|---|---|
| [`01-deepseek-v4-architecture-deep-dive.md`](01-deepseek-v4-architecture-deep-dive.md) | A deep dive on the DeepSeek-V4 architecture and its core changes vs V3 / V3.2. Covers CSA / HCA / mHC / Hash Routing / sqrtsoftplus / clamped SwiGLU / dual RoPE / Muon / MTP. |
| [`02-plan-1-as-built-and-plan-2-pointer.md`](02-plan-1-as-built-and-plan-2-pointer.md) | As-built closure of plan-0 / plan-1 (what shipped, what fell short of real V4) + pointer to plan-2 (the architecture-faithful rewrite). Cross-links progress tracker, HTML timeline, and PPT roadmap. |

## Diagrams

`diagrams/` contains PNGs rendered directly via Pillow (we previously used SVG
but several markdown viewers do not display SVG, so SVG has been retired).
Source code: `render_diagrams.py`; regenerate with `python3 render_diagrams.py`.

| File | Contents |
|---|---|
| [`diagrams/architecture.png`](diagrams/architecture.png) | V4-Flash overall architecture (43 layers + 1 MTP, with the compress_ratios pattern strip) |
| [`diagrams/csa.png`](diagrams/csa.png) | CSA — Compressed Sparse Attention (compress_ratio=4, with the Indexer top-K path) |
| [`diagrams/hca.png`](diagrams/hca.png) | HCA — Heavily Compressed Attention (compress_ratio=128, no Indexer, all queries share the same pool) |
| [`diagrams/mhc.png`](diagrams/mhc.png) | mHC — Manifold-Constrained Hyper-Connections (Sinkhorn mixing across 4 hidden streams) |

## Sources

- Official weights / inference implementation: `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/`
  - `DeepSeek_V4.pdf` (technical report)
  - `inference/model.py` (reference implementation, includes Compressor / Indexer / HC / Hash Gate)
  - `inference/kernel.py` (FP8/FP4 GEMM, `sparse_attn`, `hc_split_sinkhorn`)
  - `config.json` (hyperparameters for V4-Flash & V4-Pro)
- HF Transformers PRs 45616 / 45643 ("Add DeepSeek V4")
- NVIDIA NeMo AutoModel V4 training-side port: `deepseek-v4/NVIDIA-NeMo/Automodel/nemo_automodel/components/models/deepseek_v4/` — **important reference**: `layers.py` documents three common bugs in HF PR 45616 along with their fixes.
- RedNote slides: `deepseek-v4/references/deepseek-rednote-1/` (a 12-slide visual walkthrough)
  - originals: `images/slide_*.webp`
  - **converted to PNG in this repo**: `images_png/slide_*.png` (converted with Pillow 12, inlined directly in the markdown)
  - main additions surfaced from RedNote: mHC's mathematical constraints (Birkhoff polytope), Muon's Hybrid Newton-Schulz coefficients, the Anticipatory Routing rescue mechanism, and the Muon vs AdamW parameter assignment table — see `01-...md` §9.
