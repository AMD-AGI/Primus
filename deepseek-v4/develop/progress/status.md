# DeepSeek-V4 Integration Progress Tracker

> Tick `[x]` and fill in the commit hash + date when a task is done.
> Task granularity matches [`../plan/02-phase-details.md`](../plan/02-phase-details.md).
> Any blockers / decisions go in the `> note` row right under the task.

## Phase 0 — Investigation & Preparation

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | Architecture investigation + tech blog | — | 2026-04-28 | `deepseek-v4/develop/techblog/01-deepseek-v4-architecture-deep-dive.md` |
| [x] | 4 architecture diagrams (PNG) | — | 2026-04-28 | rendered directly via Pillow; `techblog/render_diagrams.py` reproduces them |
| [x] | Development plan | — | 2026-04-28 | all documents in this directory |

## Phase 1 — Configs & yaml

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `primus/configs/models/megatron/deepseek_v4_base.yaml` | | | |
| [ ] | `primus/configs/models/megatron/deepseek_v4_flash.yaml` | | | |
| [ ] | `examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` | | | |
| [ ] | `examples/megatron/configs/MI355X/deepseek_V4_Pro-BF16-pretrain.yamls` | | | |
| [ ] | yaml schema test | | | `tests/configs/test_deepseek_v4_yaml.py` |

## Phase 2 — Register `model_type=deepseek_v4`

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `primus/pretrain_deepseek_v4.py` (shell) | | | |
| [ ] | `primus/deepseek_v4_builders.py` (shell) | | | |
| [ ] | dispatch branch in `primus/backends/megatron/megatron_pretrain_trainer.py` | | | |
| [ ] | `deepseek_v4` branch in `primus/core/utils/import_utils.py:get_model_provider` | | | |
| [ ] | `primus/backends/megatron/core/models/deepseek_v4/{__init__.py, deepseek_v4_model.py}` (stub) | | | |
| [ ] | trainer-dispatch test | | | |

## Phase 3 — Layer Spec + Block scaffolding

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `deepseek_v4_layer_specs.py` (attention / dense_attention / mtp specs) | | | |
| [ ] | `deepseek_v4_block.py` (1-stream version, hc_mult=1 degenerate) | | | |
| [ ] | `deepseek_v4_model.py` (top-level assembly with pre/post_process) | | | |
| [ ] | register all V4 fields in `pretrain_deepseek_v4.py:extra_args_provider` | | | |
| [ ] | `deepseek_v4_builder` wired to `DeepseekV4Model` | | | |
| [ ] | 4-layer tiny config forward + backward passes | | | |

## Phase 4 — HC + Hybrid Attention

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `core/transformer/hyper_connection.py` (HyperConnection + HyperHead + Sinkhorn) | | | |
| [ ] | `core/transformer/compressor.py` (overlap=True ratio=4 / overlap=False ratio=128) | | | |
| [ ] | `core/transformer/indexer.py` | | | |
| [ ] | `core/transformer/sliding_window_kv.py` | | | |
| [ ] | `core/transformer/attn_sink.py` | | | |
| [ ] | `core/transformer/dual_rope.py` (dual base + partial RoPE) | | | |
| [ ] | `core/transformer/csa_attention.py` | | | |
| [ ] | `core/transformer/hca_attention.py` | | | |
| [ ] | `patches/hyper_connection_patches.py` swaps in V4 block | | | |
| [ ] | upgrade `deepseek_v4_block.py` to 4-stream HC | | | |
| [ ] | unit test: HyperConnection / Sinkhorn doubly-stochastic | | | |
| [ ] | unit test: Compressor numerical alignment vs reference | | | |
| [ ] | unit test: Indexer causality + topk | | | |
| [ ] | integration: 4L V4-Flash toy model 50 iter loss decreases | | | |

## Phase 5 — MoE / Activation / RoPE / MTP

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `patches/moe_patches/hash_router_patches.py` | | | |
| [ ] | `patches/moe_patches/sqrtsoftplus_router_patches.py` | | | |
| [ ] | `core/fusions/clamped_swiglu.py` + register on Megatron activation | | | |
| [ ] | upgrade `dual_rope.py` to be layer-aware (YaRN only on compress layers) | | | |
| [ ] | `patches/mtp_v4_patches.py` (separate MTP HC head) | | | |
| [ ] | unit test: HashRouter token-determinism | | | |
| [ ] | unit test: sqrtsoftplus numerics | | | |
| [ ] | unit test: clamped_swiglu | | | |
| [ ] | numerical alignment: token-0 logits vs reference inference/model.py within 1e-2 | | | |

## Phase 6 — Trainer end-to-end + PP / EP

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | swap builder in to use real model with HC + V4 attention + V4 MoE | | | |
| [ ] | PP layout design (HC 4-stream sent atomically) | | | |
| [ ] | TP partitioning (QKV / Compressor / Indexer end-to-end) | | | |
| [ ] | EP routing (hash + sqrtsoftplus) | | | |
| [ ] | smoke: 1×8 BF16 50 iter | | | |
| [ ] | smoke: 1×8 PP=2 EP=8 BF16 | | | |
| [ ] | smoke: 4×8 BF16 | | | |

## Phase 7 — Muon Optimizer

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | param-group split (HC / MTP head / embed / lm_head → AdamW; rest → Muon) | | | |
| [ ] | NS coefficients / 5 iter aligned with RedNote / reference | | | |
| [ ] | 50-iter BF16 run with Muon enabled passes | | | |
| [ ] | grad-norm + loss slope vs AdamW comparison | | | |

## Phase 8 — Convergence + FP8 / FP4

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | short runs (mock_data 50 / 200 / 1000 iter) | | | |
| [ ] | medium run (bookcorpus 100M tokens) | | | |
| [ ] | long run (1B tokens, one representative experiment) | | | |
| [ ] | `examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml` | | | |
| [ ] | FP8 throughput ≥ +30% vs BF16 | | | |
| [ ] | (optional) FP4 path | | | |

## Blockers / Risks Log

| date | description | status | decision |
|---|---|---|---|
|  | (example) HC 4-stream PP send/recv interface does not directly support 4D tensor | open | plan to add a reshape buffer in PP launcher |
