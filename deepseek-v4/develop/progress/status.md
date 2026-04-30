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
| [x] | swap builder in to use real model with HC + V4 attention + V4 MoE | (working tree) | 2026-04-28 | `deepseek_v4_builders.py` now matches upstream provider signature (`config` / `pg_collection`) and forwards into `DeepseekV4Model` |
| [x] | PP layout design (HC 4-stream sent atomically) | (working tree) | 2026-04-28 | `DeepseekV4TransformerBlock` now builds per-PP local layers (`get_num_layers_to_build` + `get_transformer_layer_offset`) and supports `set_input_tensor` |
| [ ] | TP partitioning (QKV / Compressor / Indexer end-to-end) | | | |
| [x] | EP routing (hash + sqrtsoftplus) | (working tree) | 2026-04-28 | `DeepseekV4MoE` now shards experts by EP rank and all-reduces routed outputs across EP group |
| [ ] | smoke: 1×8 BF16 50 iter | | | not run yet (completed 3-iter smoke for functional bring-up) |
| [x] | smoke: 1×8 PP=2 EP=4 BF16 | (working tree) | 2026-04-28 | passed on `uswslocpm2m-106-2371` / `dev_primus_wenx_691` with `TRAIN_ITERS=3`, no fatal error, `iteration        3/       3` reached |

## Phase 7 — Single-node bring-up (PP=2, EP=4)

| | Task | commit | date | note |
|---|---|---|---|---|
| [-] | ~~Muon optimizer phase~~ | (n/a) | 2026-04-28 | cancelled: Primus already includes distributed Muon optimizer; no dedicated integration phase needed here |
| [x] | create `run_deepseek_v4.sh` (reference: `run_qwen.bak.sh`) | (working tree) | 2026-04-28 | fixed knobs `MBS=1`, `GBS=16`, `PRIMUS_PP=2`, `PRIMUS_EP=4`; added lightweight smoke overrides (`num_layers=8`, `num_experts=8`, `mtp_num_layers=0`) |
| [x] | run `run_deepseek_v4.sh` on `uswslocpm2m-106-2371` in container `dev_primus_wenx_691` | (working tree) | 2026-04-28 | single-node smoke passed with PP/EP groups initialized, training reached `iteration        3` and torchrun exited code 0 |

## Phase 8 — ~~Convergence + FP8 / FP4~~ (cancelled)

| | Task | commit | date | note |
|---|---|---|---|---|
| [-] | ~~convergence / FP8 / FP4 campaign~~ | (n/a) | 2026-04-28 | cancelled for current DeepSeek-V4 bring-up scope |

## Phase 8 (v2) — ModuleSpec main-path refactor

> Replan baseline: `deepseek-v4/develop/plan-1/`.

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | define V4 runtime `ModuleSpec` topology in `deepseek_v4_layer_specs.py` | (working tree) | 2026-04-29 | landed full DeepSeek layer/submodules spec tree (attention/ffn/hc) rooted at `get_deepseek_v4_runtime_decoder_spec` |
| [x] | make `DeepseekV4Model` runtime path spec-driven (remove decoder swap as default) | (working tree) | 2026-04-29 | `DeepseekV4Model` now inherits `LanguageModule` and builds decoder directly from external runtime spec (`build_module`) |
| [x] | align builder (`deepseek_v4_builders.py`) layer/mtp spec resolution with spec-driven runtime | (working tree) | 2026-04-29 | builder now resolves/passes DeepSeek runtime decoder spec only; removed GPT placeholder/super-init spec dependence |
| [x] | PP/VP/MTP compatibility validation for the refactored spec path | (working tree) | 2026-04-29 | validated runtime instantiate/forward in container `dev_primus_wenx_691` with expected decoder + attention topology |
| [x] | document retirement plan for legacy placeholder/swap path | (working tree) | 2026-04-29 | package/model docs updated to record LanguageModule-based path and retirement of GPT placeholder/swap strategy |

## Phase 9 (v2) — TE provider reuse integration

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | revise Phase 9 plan with provider-inheritance direction | (working tree) | 2026-04-30 | switched plan to `DeepSeekV4SpecProvider(PrimusTurboSpecProvider)` as the single V4 provider entry |
| [x] | add `DeepSeekV4SpecProvider` class in TE spec provider module | (working tree) | 2026-04-30 | landed in `core/extensions/transformer_engine_spec_provider.py`; inherits `PrimusTurboSpecProvider`, adds V4 mode resolution (`local`/`te`/`turbo`) and grouped-MLP selector |
| [x] | wire V4 spec construction to `DeepSeekV4SpecProvider` | (working tree) | 2026-04-30 | `deepseek_v4_layer_specs.py` now resolves one provider instance, injects provider mode into block/layer params, and routes norm + MoE spec payload through provider |
| [x] | migrate norm + projection path to provider-driven selection | (working tree) | 2026-04-30 | norm path remains provider-driven; attention projections now use `DeepseekV4AttentionSubmodules + build_module`, and dense-MLP projections route through provider `linear()` in duplicated mode (TE/Turbo) with local fallback when provider modules are unavailable |
| [x] | providerize V4 MoE expert grouped-GEMM path | (working tree) | 2026-04-30 | `v4_moe.py` now supports provider grouped-MLP instantiation and grouped forward dispatch (expert bucketing + grouped forward + scatter-add), and falls back to local expert path when runtime dependencies are missing |
| [x] | deliver provider-mode A/B validation report | (working tree) | 2026-04-30 | report added in `deepseek-v4/develop/plan-1/03-phase9-provider-ab-report.md`; local forward passes, TE module-map build + CUDA forward both pass, and TE/Turbo mode host-input path now has explicit CUDA guard in decoder forward |

## Phase 10 (v2) — MoE + distributed path convergence

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | refresh Phase 10 landing scope and task decomposition | (working tree) | 2026-04-30 | aligned with MoE review outcomes: submodules+build_module, dispatcher reuse, and clamped-SwiGLU backend compatibility |
| [x] | define `DeepseekV4MoESubmodules` and wire spec-driven MoE construction | (working tree) | 2026-04-30 | `v4_moe.py` now defines `DeepseekV4MoESubmodules` and builds router/dispatcher/expert/shared-expert modules through `build_module`; `deepseek_v4_layer_specs.py` now passes MoE submodules in FFN spec |
| [x] | integrate Megatron dispatcher bridge into DeepSeek-V4 MoE forward path | (working tree) | 2026-04-30 | V4 MoE now supports dispatcher bridge flow (`dispatch_preprocess -> token_dispatch -> dispatch_postprocess -> expert_compute -> combine`) with runtime dispatcher selection and local fallback |
| [x] | retire routed-output `all_reduce` fallback from active EP path | (working tree) | 2026-04-30 | local fallback now disables EP all-reduce by default and only enables it with explicit debug gate `v4_enable_ep_allreduce_fallback` |
| [ ] | implement PP token-id propagation contract for hash-routed layers | | | explicit stage ownership, transport rules, and fail-fast assertions |
| [x] | add clamped-SwiGLU backend compatibility checks for grouped-gemm modes | (working tree) | 2026-04-30 | grouped backend now requires declared clamped-SwiGLU support (`supports_clamped_swiglu` or config override) or it is downgraded to local experts with warning |
| [ ] | distributed smoke: 1x8 and PP/EP combined run with deterministic routing snapshots | | | no hang, no dispatcher regressions, stable hash and learned routing |

## Phase 11 (v2) — Validation + release gates

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | execute full regression gate matrix (G1-G6) | | | see `deepseek-v4/develop/plan-1/02-test-strategy.md` |
| [ ] | numerical alignment run and tolerance report | | | fixed seed/checkpoint snapshots |
| [ ] | short-run convergence campaign and baseline comparison | | | track loss slope and stability |
| [ ] | TE on/off throughput + memory comparison report | | | performance gate for release decision |
| [ ] | publish release checklist and blocker disposition | | | go/no-go output with risk owners |

## Blockers / Risks Log

| date | description | status | decision |
|---|---|---|---|
| 2026-04-28 | PyTorch warns `c10d::allreduce_` autograd kernel is not registered for the EP routed-output allreduce path in `v4_moe.py` | open | Keep functional path for Phase 7 bring-up; replace with proper token dispatcher / autograd-safe EP path in follow-up optimization work |
|  | (example) HC 4-stream PP send/recv interface does not directly support 4D tensor | open | plan to add a reshape buffer in PP launcher |
