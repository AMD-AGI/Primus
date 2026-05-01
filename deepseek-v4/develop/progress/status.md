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

## Phase 12 (v3) — Plan-2 lockdown

> Replan baseline: `deepseek-v4/develop/plan-2/`.
> Plan-1 phases 9 / 10 / 11 are paused — their tracking rows above remain
> for history but are no longer the active program of work.

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | Architecture review of `dev/wenx/deepseek-v4` from `e194e039`..HEAD | (review) | 2026-05-01 | findings recorded in `deepseek-v4/develop/plan-2/00-review-findings.md` |
| [x] | Plan-2 documents (`README.md`, `00`-`04`) | (review) | 2026-05-01 | replaces plan-0 / plan-1 as the active plan |
| [x] | Phase 12+ tracking section opened in `status.md` | (this commit) | 2026-05-01 | append-only |
| [x] | Update tech-blog with as-built notes for plan-1 + pointer to plan-2 | (working tree) | 2026-05-01 | added `techblog/02-plan-1-as-built-and-plan-2-pointer.md`; techblog `README.md` now points plan-2 as active plan of record |
| [x] | Refresh `progress/timeline.html` (standard fonts + day-by-day Gantt) | (working tree) | 2026-05-01 | swapped Google Fonts → system stack; Gantt switched to daily columns with May 02–05 holiday band; P13–P21 packed into May 06–09 |
| [x] | Refresh `progress/deepseek_v4_roadmap.pptx` (add 开发计划 schedule slide) | (working tree) | 2026-05-01 | new slide 7/13 with 3-row date / phase / work-content layout + holiday-gap arrow; section eyebrows renumbered |
| [ ] | **Stakeholder sign-off on plan-2 scope** | | | **external blocker — required before P13 starts (May 06)**. Sign-off owner: project lead. Material to review: `plan-2/README.md`, `00-review-findings.md`, `01-roadmap.md`, `progress/timeline.html`, `progress/deepseek_v4_roadmap.pptx`. |

> **P12 status (2026-05-01 EOD):** all engineering deliverables are done.
> The only open item is the external sign-off; plan-2 P13 starts on May 06
> (after the May 02–05 holiday) once that sign-off lands.

## Phase 13 (v3) — Faithful attention (`MLASelfAttention`-rooted)

> P13 landed in two commits inside the May 06 budget:
> 1. **First commit** (`cad0fb38`) — V4-faithful dense (`compress_ratio == 0`)
>    attention rooted on `MLASelfAttention`, with single-latent KV, per-head
>    `q_rms`, attn-sink, grouped low-rank O, and an inline numerical-alignment
>    test scaffold. The legacy plan-1 `_LegacyDeepseekV4Attention` kept
>    backing CSA / HCA so the smoke matrix continued to run.
> 2. **Follow-up commit (this one)** — fold Compressor / Indexer into the
>    new `DeepseekV4Attention.forward` as spec submodules so all three V4
>    layer types (dense / HCA / CSA) share one class; switch
>    `linear_q_up_proj` / `linear_o_b` projection specs from
>    `parallel_mode="duplicated"` to ColumnParallel / RowParallel; add a
>    TP=2 sharding-parity scaffold; retire `csa_attention.py` /
>    `hca_attention.py` / `_LegacyDeepseekV4Attention`.

| | Task | commit | date | note |
|---|---|---|---|---|
| [x] | `DeepseekV4Attention(MLASelfAttention)` + V4 `DeepseekV4AttentionSubmodules` dataclass | `cad0fb38` | 2026-05-01 | dense path; CSA / HCA folded in via the follow-up |
| [x] | Single-latent KV (K = V = `wkv`) | `cad0fb38` | 2026-05-01 | new attention drops `linear_k_proj` / `linear_v_proj`; `linear_kv` projects `hidden -> head_dim` and broadcasts |
| [x] | Per-head `q_rms` after `linear_q_up_proj` | `cad0fb38` | 2026-05-01 | parameter-less RMS — matches the released checkpoint (no `q_rms.weight` key) |
| [x] | Reuse MLA `kv_layernorm` for V4 `kv_norm` | `cad0fb38` | 2026-05-01 | provider-built RMSNorm on `head_dim`; same module name as MLA |
| [x] | Learnable per-head `attn_sink` parameter | `cad0fb38` | 2026-05-01 | direct `nn.Parameter` on the attention (key: `attn_sink`); inline softmax-with-sink — matches released checkpoint key exactly. Spec-built `attn_sink_module` retained as forward-compat for a future TE-fused sink |
| [x] | Grouped low-rank O projection (`linear_o_a` + `linear_o_b`, `o_groups`/`o_lora_rank`) | `cad0fb38` | 2026-05-01 | added `o_groups` / `o_lora_rank` to `DeepSeekV4TransformerConfig`; einsum-based `wo_a` per group + `wo_b` matches `inference/model.py`. `o_lora_rank == 0` falls back to MLA-style flat `linear_proj` |
| [x] | Compressor / Indexer as spec submodules of attention | (this commit) | 2026-05-01 | added `compressor` / `indexer` to `DeepseekV4AttentionSubmodules`; `DeepseekV4Attention.forward` now dispatches on `compress_ratio in {0, 4, 128}` and shares the attn-sink softmax across local + compressed branches; `_LegacyDeepseekV4Attention`, `csa_attention.py`, `hca_attention.py` deleted |
| [x] | Switch projection specs from `parallel_mode="duplicated"` to TP | (this commit) | 2026-05-01 | `linear_q_up_proj` → `provider.column_parallel_linear()` (`gather_output=True`); `linear_o_b` / `linear_proj` → `provider.row_parallel_linear()` (`input_is_parallel=False`). At TP > 1 the projection weights are sharded across TP ranks; at TP = 1 the result is bit-identical to the previous duplicated path. Grouped-O `linear_o_a` + the rest stay duplicated; full sharded grouped-O TP plan tracked in P14 |
| [x] | Inline numerical-alignment test scaffold (CPU fp32, compress_ratio=0) | `cad0fb38` | 2026-05-01 | `tests/unit_tests/megatron/transformer/deepseek_v4/test_deepseek_v4_attention.py`: state-dict-key check, shape-and-finite, sink on/off forward equivalence ≤1e-3 vs inline reference, fallback path coverage |
| [x] | Compressed branches: HCA shape + numerical alignment, CSA shape | (this commit) | 2026-05-01 | extended the unit-test file with `test_hca_forward_shape_and_finite`, `test_hca_forward_matches_inline_reference` (compress-base RoPE + compressed-causal mask + joint softmax-with-sink, ≤1e-3 vs inline reference) and `test_csa_forward_shape_and_finite` (overlap Compressor + Indexer top-K + per-query joint softmax) |
| [x] | Spec wiring contract tests | (this commit) | 2026-05-01 | `test_attention_spec_uses_column_and_row_parallel` asserts the q-up-proj / o-b spec uses `ColumnParallel` / `RowParallel`; `test_attention_spec_includes_compressor_and_indexer` asserts compressed-branch submodules carry `Compressor` (always) + `Indexer` (CSA only) |
| [ ] | 1L attention forward agrees with HF reference within 1e-3 (CPU fp32) | | | **partial**: inline reference equivalence verified this commit; HF-reference comparison waits for P17 state-dict adapter |
| [x] | TP=2 sharding parity test (scaffold) | (this commit) | 2026-05-01 | `test_tp2_sharding_parity_scaffold` skips on CPU / single-rank; runnable only under `torchrun --nproc_per_node=2`. Implementation of the bit-equality check vs a duplicated baseline is tracked in P19 (distributed re-validation) |

## Phase 14 (v3) — Faithful MoE + activation + router

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `clamped_swiglu_pre_mul(gate, up, alpha)` activation; separate `w1`/`w3` for the eager MLP | | | matches HF reference (gate clamp `max=α`, up clamp `(-α,+α)`) |
| [ ] | `DeepseekV4LearnedRouter(TopKRouter)` with sqrtsoftplus / sigmoid / softmax + bias | | | honors `moe_router_topk_scaling_factor` |
| [ ] | `DeepseekV4HashRouter(TopKRouter)` with learnable `gate_linear` + `tid2eid` lookup | | | weights from learned score, expert ids from table |
| [ ] | `DeepseekV4MoE(MoELayer)` | | | inherits load-balance / z-loss / dispatcher lifecycle |
| [ ] | Provider `v4_grouped_mlp_spec(swiglu_limit)` + `v4_router_spec(learned)` | | | activation_func returns callable instance |
| [ ] | Token-ids threaded as forward kwarg (no `decoder._v4_token_ids`) | | | |
| [ ] | Router unit tests: identical (probs, indices) + gradient on gate (G4) | | | |
| [ ] | 1L MoE forward within 1e-3 of HF reference (G5) | | | |

## Phase 15 (v3) — Faithful layer + block + HC × PP

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `DeepseekV4HybridLayer(TransformerLayer)` with HC residual override | | | replaces `GraphableMegatronModule` plain layer |
| [ ] | `DeepseekV4TransformerBlock(TransformerBlock)` | | | reuses upstream PP / recompute / SP |
| [ ] | Lift / lower helpers carry `[S,B,K,D]` across PP via `[S*K,B,D]` packing | | | fixes C1 |
| [ ] | `HyperHead` only on the post_process stage | | | |
| [ ] | Caller-supplied `position_ids` honored | | | fixes C3 |
| [ ] | Token-ids removed from `decoder` attribute stash | | | fixes C2 |
| [ ] | PP=1 vs PP=2 vs PP=4 equivalence on 4L V4 toy (G6) | | | |

## Phase 16 (v3) — MTP integration

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `get_v4_mtp_block_spec` builder | | | mirrors `_mamba_mtp_block_spec` |
| [ ] | `DeepseekV4Model.__init__` builds `MultiTokenPredictionBlock` when `mtp_num_layers > 0` | | | |
| [ ] | Per-MTP-layer separate `HyperHead` | | | |
| [ ] | `process_mtp_loss` wired in `DeepseekV4Model.forward` | | | |
| [ ] | `DeepseekV4MTPBlock` retired or moved to `research/` | | | |
| [ ] | MTP loss appears in train log; ablation `mtp_num_layers=0` matches LM loss to 1e-6 (G7) | | | |

## Phase 17 (v3) — State-dict adapter + checkpoint load

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | `DeepSeekV4StateDictAdapter` (HF / inference key map → Primus) | | | |
| [ ] | `scripts/load_v4_flash_check.py` (CPU forward, 64-token prompt) | | | |
| [ ] | `tid2eid` loaded as non-trainable buffer | | | |
| [ ] | Round-trip test (G8) | | | |
| [ ] | V4-Flash token-0 logits ≤1e-2 vs HF reference (G9) | | | |
| [ ] | FP4 / FP8 expert weights documented as out-of-scope | | | |

## Phase 18 (v3) — Spec-system audit

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | Provider built once per builder call; threaded via `BuildContext` | | | fixes D1 |
| [ ] | `provider.activation_func()` returns callable instance | | | fixes D2 |
| [ ] | YAML `compress_ratios` becomes a list field | | | fixes D4 |
| [ ] | `tests/configs/test_deepseek_v4_yaml.py` (G1) | | | |
| [ ] | Audit: no spec-replaceable module instantiated outside `build_module` | | | |
| [ ] | Drop `_v4_token_ids` references everywhere | | | |

## Phase 19 (v3) — Distributed re-validation

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | Smoke 1×8 BF16 (TP=1 PP=1 EP=1) | | | |
| [ ] | Smoke 1×8 BF16 (TP=1 PP=2 EP=4) | | | |
| [ ] | Smoke 1×8 BF16 (TP=2 PP=2 EP=2) | | | |
| [ ] | Smoke 1×8 BF16 (TP=1 PP=4 EP=2) | | | |
| [ ] | Smoke 2×8 BF16 (DP=2 PP=2 EP=2 TP=2) | | | |
| [ ] | Routing snapshot diff = 0 across PP / EP changes (G11) | | | |
| [ ] | `c10d::allreduce_` warning gone | | | |

## Phase 20 (v3) — Numerical / convergence / perf gates

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | Numerical alignment report on V4-Flash (G9 extended) | | | |
| [ ] | 200-step convergence vs HF baseline (G12) | | | ±0.05 loss curve match |
| [ ] | TE on/off perf report (G13) | | | TFLOPS + HBM delta |
| [ ] | FP8 follow-up plan scoped | | | next plan |
| [ ] | Release checklist signed off | | | go/no-go |

## Phase 21 (v3) — Cleanup + docs + handover

| | Task | commit | date | note |
|---|---|---|---|---|
| [ ] | Remove `_RMSNorm` duplicates / `dual_rope.py` / `csa_attention.py` / `hca_attention.py` | | | |
| [ ] | Remove `DeepseekV4MTPBlock` (or move to `research/`) | | | |
| [ ] | Remove EP `all_reduce` fallback gate | | | |
| [ ] | Refresh techblog with as-built notes | | | |
| [ ] | Refresh `progress/` HTML + `ppt-template-amd.pptx` slides | | | |
| [ ] | Fix yaml comments (`compress_ratio` 4=CSA, 128=HCA) | | | |

## Blockers / Risks Log

| date | description | status | decision |
|---|---|---|---|
| 2026-04-28 | PyTorch warns `c10d::allreduce_` autograd kernel is not registered for the EP routed-output allreduce path in `v4_moe.py` | open | Plan-2 (P19) verifies the warning is gone after dispatcher migration; fallback gate retired in P21 |
|  | (example) HC 4-stream PP send/recv interface does not directly support 4D tensor | tracked in plan-2 | Plan-2 P15: lift `[S,B,K,D]` to `[S*K,B,D]` for PP P2P; revisit a 4D PP send path in P21 |
| 2026-05-01 | Current attention does not match real V4 (single-latent KV / q_norm / kv_norm / grouped O all missing) | open | Plan-2 P13 rebases on `MLASelfAttention` and lands the missing pieces |
| 2026-05-01 | HashRouter has no learnable gate weight; clamped SwiGLU clamps post-mul instead of pre-mul; `w1`/`w3` fused | open | Plan-2 P14 rewrites router + activation to match HF reference |
| 2026-05-01 | Custom V4 block / layer / MoE bypass `TransformerBlock` / `TransformerLayer` / `MoELayer` | open | Plan-2 P14–P15 rebase onto Megatron's standard parents |
| 2026-05-01 | Token-IDs propagation via `decoder._v4_token_ids` attribute | open | Plan-2 P14–P15 thread token_ids as a forward kwarg |
| 2026-05-01 | No state-dict adapter / V4-Flash safetensors cannot be loaded into the Primus model | open | Plan-2 P17 lands the adapter + numerical-alignment gate |
