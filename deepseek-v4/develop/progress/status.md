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
| [x] | `primus/configs/models/megatron/deepseek_v4_base.yaml` | `d3383c02` | 2026-04-28 | extends `llama_base`; sets `model_type=deepseek_v4` and all V4 defaults |
| [x] | `primus/configs/models/megatron/deepseek_v4_flash.yaml` | `d3383c02` | 2026-04-28 | values from `DeepSeek-V4-Flash/config.json` |
| [x] | `primus/configs/models/megatron/deepseek_v4_pro.yaml` | `d3383c02` | 2026-04-28 | values from `DeeSeek-v4-Pro/config.json` |
| [x] | `examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` | `d3383c02` | 2026-04-28 | scaffold; parallelism + perf knobs to be retuned in P6 |
| [x] | `DeepSeekV4Tokenizer` accepted by `_add_tokenizer_args` | `d3383c02` | 2026-04-28 | `primus/backends/megatron/training/tokenizer/tokenizer.py` |
| [-] | ~~Register V4 fields into Megatron argparse~~ | (n/a) | 2026-04-28 | Not needed: `merge_namespace` (`train_runtime.py:_initialize_trainer`) copies yaml-only fields onto `backend_args` after `convert_config`, and `MegatronBaseTrainer._patch_parse_args` makes Megatron return `backend_args` verbatim. The V4 builder reads V4 fields directly via `args.<field>`. |
| [ ] | yaml schema test | | | `tests/configs/test_deepseek_v4_yaml.py` (deferred to P3) |

## Phase 2 — Register `model_type=deepseek_v4`

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | dispatch branch in `primus/backends/megatron/megatron_pretrain_trainer.py` | `8ae10000` | 2026-04-28 | reuses `pretrain_gpt.forward_step` for V4 |
| [x] | `deepseek_v4` branch in `primus/core/utils/import_utils.py:get_model_provider` | `8ae10000` | 2026-04-28 | imports primus-owned V4 builder/provider |
| [x] | `primus/backends/megatron/core/models/deepseek_v4/__init__.py` (stub) | `8ae10000` | 2026-04-28 | re-exports `DeepseekV4Model` |
| [x] | `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py` (stub) | `8ae10000` | 2026-04-28 | thin subclass of `GPTModel`; replaced in P3 |
| [x] | `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_builders.py` (shell) | `8ae10000` | 2026-04-28 | bundles `model_provider` + `deepseek_v4_builder` |
| [ ] | trainer-dispatch test | | | (added in P3 along with model spec) |

## Phase 3 — Layer Spec + Block scaffolding

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | `deepseek_v4_layer_specs.py` (layer / decoder-block / MTP-block specs + P4/P5 hooks) | `a5d2a561` | 2026-04-28 | delegates to GPT helpers; per-layer hooks `_resolve_attention_module_spec` / `_resolve_mlp_module_spec` reserved for P4/P5 |
| [x] | `deepseek_v4_block.py` (1-stream version, hc_mult=1 degenerate) | `a5d2a561` | 2026-04-28 | `DeepseekV4TransformerBlock` subclass; stashes V4 config attrs for P4 patches |
| [x] | `deepseek_v4_model.py` (uses `DeepseekV4TransformerBlock` as decoder) | `a5d2a561` | 2026-04-28 | post-`super().__init__` swap-in keeps a stable target for P4/P5 patches |
| [-] | ~~register V4 fields in `pretrain_deepseek_v4.py:extra_args_provider`~~ | (n/a) | 2026-04-28 | superseded — yaml fields reach builder via `merge_namespace`; no argparse step required |
| [x] | `deepseek_v4_builder` wired to V4 layer specs + `DeepseekV4Model` | `a5d2a561` | 2026-04-28 | `_resolve_layer_spec` + `_resolve_mtp_block_spec` use the V4-prefixed helpers |
| [ ] | 4-layer tiny config forward + backward passes | | | needs GPU; deferred to env validation |

## Phase 4 — HC + Hybrid Attention

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | `core/transformer/hyper_connection.py` (HyperMixer + HyperHead + Sinkhorn) | `3b7ad8c8` | 2026-04-28 | unit-tested: row/col errs ~1e-6, hc_mult=1 degenerate exact, fp32 params + fp32 sinkhorn |
| [x] | `core/transformer/compressor.py` (overlap=True ratio=4 / overlap=False ratio=128) | `3b7ad8c8` | 2026-04-28 | unit-tested: HCA / CSA pool shapes correct, APE shape matches, RMSNorm + (RoPE applied externally) |
| [x] | `core/transformer/indexer.py` | `3b7ad8c8` | 2026-04-28 | causality verified (q=0 all -1, q=3 sees pool[0], etc.); idxs in [0,P) ∪ {-1}; backward OK |
| [x] | `core/transformer/sliding_window_kv.py` | `3b7ad8c8` | 2026-04-28 | causal SWA mask + per-query KV indices |
| [x] | `core/transformer/attn_sink.py` | `3b7ad8c8` | 2026-04-28 | sinks=0 → probs sum ≤ 1; sinks=50 → ~0; backward propagates to sinks |
| [x] | `core/transformer/dual_rope.py` (dual base + partial interleaved RoPE + YaRN) | `3b7ad8c8` | 2026-04-28 | YaRN m_scale = 0.1·log(factor)+1 verified; partial RoPE preserves norm; nope channels untouched |
| [x] | `core/transformer/deepseek_v4_attention.py` (shared Q/KV/SWA/sink/output base) | `3b7ad8c8` | 2026-04-28 | dense attention: forward + backward + causality OK |
| [x] | `core/transformer/csa_attention.py` | `3b7ad8c8` | 2026-04-28 | overlap compressor + indexer + per-query top-K gather + joint softmax (incl. sink); causality verified |
| [x] | `core/transformer/hca_attention.py` | `3b7ad8c8` | 2026-04-28 | non-overlap compressor + full compressed-pool concat; causal-mask verified |
| [x] | upgrade `deepseek_v4_block.py` to multi-stream HC + per-layer attention dispatch | `3b7ad8c8` | 2026-04-28 | standalone module (does not subclass `TransformerBlock`); 8-layer mixed dense/CSA/HCA + hc_mult=4 forward/backward/causality OK |
| [x] | finalize `deepseek_v4_layer_specs.py` (placeholder for super().__init__) | `3b7ad8c8` | 2026-04-28 | V4 block bypasses Megatron's spec mechanism; placeholder spec keeps `GPTModel.__init__` happy until P6 refactor |
| [-] | ~~`patches/hyper_connection_patches.py`~~ | (n/a) | 2026-04-28 | superseded — V4 block is standalone, no patch needed; the swap-in happens inside `DeepseekV4Model.__init__` |
| [ ] | unit test: HyperConnection / Sinkhorn doubly-stochastic | | | |
| [ ] | unit test: Compressor numerical alignment vs reference | | | |
| [ ] | unit test: Indexer causality + topk | | | |
| [ ] | integration: 4L V4-Flash toy model 50 iter loss decreases | | | |

## Phase 5 — MoE / Activation / RoPE / MTP

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | `core/transformer/clamped_swiglu.py` (activation + MLP wrapper) | `(this commit)` | 2026-04-28 | clamp tight verified, MLP forward+backward OK; placed under `core/transformer/` (not `fusions/`) since v1 is eager |
| [x] | `core/transformer/moe/v4_hash_router.py` (HashRouter; static `tid2eid` table) | `(this commit)` | 2026-04-28 | per-token top-K distinct, deterministic across instantiations w/ same seed, probs sum to 1; standalone (Megatron MoE integration → P6) |
| [x] | `core/transformer/moe/v4_topk_router.py` (V4TopKRouter; sqrtsoftplus / sigmoid / softmax) | `(this commit)` | 2026-04-28 | three score functions verified, top-K renorm OK, optional noaux_tc bias; backward OK |
| [x] | `core/transformer/moe/v4_moe.py` (DeepseekV4MoE: routed + shared experts + clamped SwiGLU) | `(this commit)` | 2026-04-28 | hash mode + learned mode forward+backward; same-token determinism OK |
| [x] | layer-aware `dual_rope.py` (YaRN only on `compress_ratio != 0`) | `3b7ad8c8` | 2026-04-28 | already covered by P4: `DualRoPE.get_rope(compress_ratio)` returns the right cache; main_rope built without YaRN, compress_rope with YaRN |
| [x] | upgrade `deepseek_v4_block.py` to use V4 MoE (replace `_SwiGLUMLP`) | `(this commit)` | 2026-04-28 | per-layer router pick (hash if `layer_idx < num_hash_layers`); token_ids threaded through forward via model-side stash on the block |
| [x] | `models/deepseek_v4/deepseek_v4_mtp.py` (DeepseekV4MTPBlock, separate per-layer HyperHead) | `(this commit)` | 2026-04-28 | shares `rope` with main decoder, separate HC head per MTP layer; loss-side wiring → P6 |
| [x] | `models/deepseek_v4/deepseek_v4_model.py` instantiates MTP block (when `mtp_num_layers > 0`) | `(this commit)` | 2026-04-28 | `forward()` overridden to stash `input_ids` on the V4 block for hash routers |
| [x] | smoke test: clamped_swiglu / HashRouter / V4TopKRouter / V4MoE / V4Block w/ MoE / V4 MTP | `(this commit)` | 2026-04-28 | 7-test suite (`/tmp/p5_smoke.py`); all green on the dev box container |
| [ ] | numerical alignment: token-0 logits vs reference `inference/model.py` within 1e-2 | | | deferred to P6+ (needs reference checkpoint loaded into V4 model) |
| [-] | ~~`patches/moe_patches/hash_router_patches.py`~~ / ~~`sqrtsoftplus_router_patches.py`~~ / ~~`mtp_v4_patches.py`~~ | (n/a) | 2026-04-28 | superseded — V4 block is standalone and uses these modules directly. Real Megatron-MoE / token-dispatcher / EP integration is P6's responsibility. |

## Phase 6 — Trainer end-to-end + PP / EP

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | swap builder in to use real model with HC + V4 attention + V4 MoE | | | |
| [ ] | PP layout design (HC 4-stream sent atomically) | | | |
| [ ] | TP partitioning (QKV / Compressor / Indexer end-to-end) | | | |
| [ ] | EP routing (hash + sqrtsoftplus) | | | |
| [ ] | smoke: 1×8 BF16 50 iter | | | |
| [ ] | smoke: 1×8 PP=2 EP=4 BF16 | | | |

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
