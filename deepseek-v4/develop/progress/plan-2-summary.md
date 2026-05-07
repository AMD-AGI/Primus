# Plan-2 — Architecture-Faithful DeepSeek-V4 in Primus: Summary

> Snapshot date: **2026-05-07**.
> Companion docs: `[plan-2/README.md](../plan-2/README.md)`,
> `[plan-2/01-roadmap.md](../plan-2/01-roadmap.md)`,
> `[plan-2/03-phase-details.md](../plan-2/03-phase-details.md)`,
> `[plan-2/04-test-strategy.md](../plan-2/04-test-strategy.md)`,
> `[progress/status.md](./status.md)` (live tracker).

## 1. What Plan-2 Was

Plan-0 / plan-1 got DeepSeek-V4 into Primus as **runnable shells** — module
skeletons, a bring-up smoke (`PP=2 EP=4`), provider-driven spec wiring — but
the runtime model was **not faithful to V4-Flash / V4-Pro**. The attention
collapsed multi-latent KV into a stock GQA path, the MoE gate had no
learnable weight, the clamped SwiGLU clamped on the wrong side of the
multiplication, and the V4 block / layer / MoE bypassed the upstream
`TransformerBlock` / `TransformerLayer` / `MoELayer` contracts.

Plan-2 is the **architecture-faithful rewrite**, executed inside Megatron's
standard `spec + config + provider + submodule + build_module` pattern:

- **MLA-rooted attention.** `DeepseekV4Attention(MLASelfAttention)` adds
single-latent KV (`linear_kv` only — no `linear_k_proj` / `linear_v_proj`),
per-head parameter-less `q_rms`, learnable per-head `attn_sink`, grouped
low-rank O-projection, and CSA / HCA branches as spec submodules.
- **MoELayer-friendly MoE.** `DeepseekV4MoE(MegatronModule)` exposes the
`BaseMoELayer` surface (`local_expert_indices`, `set_layer_number`),
routes through Megatron's token dispatchers (`MoEAlltoAllTokenDispatcher`
/ `MoEFlexTokenDispatcher`), and ships two routers:
`DeepseekV4LearnedRouter` (sqrtsoftplus / sigmoid / softmax + selection-only
bias) and `DeepseekV4HashRouter` (learnable `weight` × frozen `tid2eid`
buffer).
- **TransformerLayer / TransformerBlock subclassing.**
`DeepseekV4HybridLayer(TransformerLayer)` overrides only the residual /
HC mixing and the attention-FFN selection;
`DeepseekV4TransformerBlock(TransformerBlock)` carries the V4 dispatcher
  - lift / lower path.
- **Native MTP via `MultiTokenPredictionBlock`.** The legacy primus
`DeepseekV4MTPBlock` is gone; `get_v4_mtp_block_spec(...)` returns a
spec-driven block built once on the post-process rank.
- **HC × PP packing as a Primus patch.** mHC's `K=hc_mult` streams ride the
PP wire packed into the sequence axis (`[S*K, B, D]`); the receive buffer
is sized correctly via a pair of patches against
`megatron.core.pipeline_parallel.schedules` (1F1B *and* interleaved-1F1B
/ VPP).
- **Pre-training-first scope.** The HF state-dict adapter +
V4-Flash safetensors numerical alignment (token-0 logits ≤ 1e-2 vs HF)
was **deferred to P22+**; activate it when an SFT or evaluation campaign
needs the released weights.

## 2. Phase-by-Phase Outcome

All commits below land on `dev/wenx/deepseek-v4`.


| #                | Phase                                         | Status                           | Key commits             | What shipped                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | --------------------------------------------- | -------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P12              | Plan-2 lockdown                               | done                             | `636ab3de`              | Architecture review (`plan-2/00-review-findings.md`), full plan-2 docs (`README` + `00`–`04`), tracking section opened in `status.md`, techblog pointer + day-by-day Gantt + roadmap PPT refreshed.                                                                                                                                                                                                                                                                                                                                                                          |
| P13              | Faithful attention                            | done                             | `cad0fb38` → `aa9929a0` | `DeepseekV4Attention(MLASelfAttention)` with single-latent KV, `q_rms`, `kv_norm`, learnable `attn_sink`, grouped low-rank O (`o_groups` / `o_lora_rank`); Compressor / Indexer folded into the same class as spec submodules; `linear_q_up_proj` and `linear_o_b` switched from duplicated mode to ColumnParallel / RowParallel; legacy `_LegacyDeepseekV4Attention` / `csa_attention.py` / `hca_attention.py` deleted; CPU fp32 unit tests vs inline HF reference.                                                                                                         |
| P14              | Faithful MoE + activation + router            | done                             | `1a8bf32e` → `5fe8bc3c` | Pre-multiplication clamped SwiGLU (`gate` clamp `max=α` one-sided, `up` clamp `(-α,+α)`); `ClampedSwiGLUMLP` exposes `w1` / `w2` / `w3`; `DeepseekV4LearnedRouter` (gate as `weight` Parameter, selection-only `expert_bias`, renormalization gated by score function); `DeepseekV4HashRouter` (learnable `weight` + frozen `tid2eid` buffer); `DeepseekV4MoE` switched to `MegatronModule` + Megatron token dispatchers; G3 / G4 / G5 unit tests green ≤ 1e-3 / 1e-6 fp32.                                                                                                  |
| P15              | Faithful layer + block + HC × PP              | done (CPU contract)              | `25ccdb5e`              | `DeepseekV4HybridLayer(TransformerLayer)` and `DeepseekV4TransformerBlock(TransformerBlock)`; `_lift_streams_in` / `_lower_streams_out` extracted as module-level helpers ( `[S, B, D] → [B, S, K, D]` on first stage, `[S*K, B, D]` for PP P2P, collapse + transpose on last stage); `HyperHead` only on `post_process`; `position_ids` plumbed end-to-end; `decoder._v4_token_ids` attribute stash retired (token IDs flow as a forward kwarg); CPU lift/lower round-trip test enforces the math contract.                                                                 |
| P16              | MTP integration                               | done                             | `6c5875d4` → `e591b893` | `get_v4_mtp_block_spec` returns `ModuleSpec(MultiTokenPredictionBlock, ...)` with V4 RMSNorm + ColumnParallel `eh_proj`; `DeepseekV4Model.__init__` builds the MTP block only when `mtp_num_layers > 0` *and* `mtp_on_this_rank`; `process_mtp_loss` adds the auxiliary MTP loss to the LM-loss path; legacy `DeepseekV4MTPBlock` deleted, `v4_use_custom_mtp_block` config flag dropped; V4 attention spec advertises `attn_mask_type=AttnMaskType.causal` so `MultiTokenPredictionLayer.__init__` accepts it.                                                              |
| P17              | Code cleanup (dead-code retirement)           | done                             | `e591b893`              | `_RMSNorm` duplicates dedup'd into shared `LocalRMSNorm`; `csa_attention.py` / `hca_attention.py` confirmed deleted; legacy `DeepseekV4MTPBlock` deleted; `v4_use_custom_mtp_block` / `mtp_compress_ratios` / `v4_enable_ep_allreduce_fallback` flags retired; YAML comments fixed (4 = CSA, 128 = HCA); `__init__.py` package surface trimmed; **G14** AST audit (`test_v4_p17_dead_code.py`) covers all of the above. `dual_rope.py` was intentionally **kept** — Megatron's rotary only supports a single base; V4's CSA / HCA dispatch needs the dual-base partial RoPE. |
| P18              | Spec-system audit                             | done                             | `b5832672`              | `BuildContext` (`build_context.py`) — provider built once per builder call and cached on the config; `provider.v4_mlp_activation_func()` returns `None` (eager clamped SwiGLU) by default, `TEActivationOp` only when `use_te_activation_func=True`; `compress_ratios` normalized to `tuple[int, ...]` in `__post_init__`; new YAML schema test `tests/unit_tests/configs/test_deepseek_v4_yaml.py`; AST audit forbids any direct `TENorm(...)` / `TEColumn/RowParallelLinear(...)` / `TELinear(...)` / `TEActivationOp(...)` outside `build_module`.                        |
| P19              | Distributed re-validation                     | core gates done; G11 deferred    | (see §3 below)          | All four target smokes — A `1×8 PP=1 EP=1`, B `1×8 PP=2 EP=4`, C `1×8 PP=4 EP=2`, D `1×8 PP=2 EP=4 VPP=2` — passed 10/10 iterations on `mi355-gpu-12` (BF16, MBS=1 GBS=16, seq=128, 8 layers / 3 hash layers / `hc_mult=4`). Two HC×PP / hash-router patches landed (see §3). `c10d::allreduce` autograd warning verified absent in stderr across smokes + profile runs. **Deferred**: G11 routing-snapshot diff (no dump tooling landed).                                                                                                                                   |
| P20 / P21 / P22+ | Convergence + perf, docs handover, HF adapter | **dropped from this status doc** | —                       | Phase 20 (200-step convergence vs Megatron-bridge baseline + TE on/off perf + FP8 follow-up plan), Phase 21 (techblog / timeline / PPT refresh), and Phase 22+ (HF state-dict adapter + V4-Flash safetensors round-trip) were originally tracked here but have been removed per the 2026-05-07 user direction. They live as documented intent in `plan-2/03-phase-details.md` and re-enter active work when their respective triggers (release campaign, downstream integration ask, SFT / eval need) fire.                                                                  |


## 3. Phase 19 in Detail

### 3.1 Smokes (mi355-gpu-12, container `dev_primus_wenx_693`)


| smoke | parallelism          | iters | notes                                                                                                                                                                                                                                                                                                                                                                  | log                                   |
| ----- | -------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| A     | TP=1 PP=1 EP=1       | 10/10 | dense-PP path, no V4 patches needed (HC stays in-stage; `pp_tensor_shape` and `pp_token_pre_broadcast` no-op when PP ≤ 1).                                                                                                                                                                                                                                             | `p19/smokeA*.log`                     |
| B     | TP=1 PP=2 EP=4       | 10/10 | required `megatron.deepseek_v4.pp_tensor_shape` — without it Megatron's `get_tensor_shapes` allocates `[S, B, D]` while V4 sends `[S*K, B, D]`, surfacing as a RoPE / hash-router mismatch.                                                                                                                                                                            | `p19/smokeB*.log`                     |
| C     | TP=1 PP=4 EP=2       | 10/10 | required both patches: hash-routed layer 2 sits on a middle PP rank, where `pretrain_gpt.get_batch` returns `None` for `input_ids`. The pre-broadcast patch fixes that.                                                                                                                                                                                                | `p19/smokeC_pp4_ep2_v2.log`           |
| D     | TP=1 PP=2 EP=4 VPP=2 | 10/10 | VPP-clean. The shape patch additionally wraps `forward_backward_pipelining_with_interleaving` to scale its `seq_length` kwarg by `hc_mult` (the interleaved schedule's inline `tensor_shape` bypasses `get_tensor_shapes`). The pre-broadcast patch runs all `num_microbatches × num_chunks` collectives upfront so they cannot race the warmup `recv_forward.wait()`. | `p19/smokeD_pp2_ep4_vpp2_v2_run3.log` |


### 3.2 Profile runs

Two `torch.profiler` Chrome-trace JSONs were captured on the same node, gated
by `PROFILE=True` + `disable_tensorboard=False` so the existing
`primus/backends/megatron/patches/torch_profiler_patches.py` hook fires:

- `output/amd/tas-mi355x-20260507/p19_profile_pp1_ep8/tensorboard/primus-megatron-exp[p19_profile_pp1_ep8]-rank[0].*.pt.trace.json` — TP=1 PP=1 **EP=8** (~99 MB).
- `output/amd/tas-mi355x-20260507/p19_profile_pp2_ep4/tensorboard/primus-megatron-exp[p19_profile_pp2_ep4]-rank[0].*.pt.trace.json` — TP=1 **PP=2 EP=4** (~105 MB).

Launchers live under `deepseek-v4/develop/progress/p19/` (`run_profile_ep8.sh`,
`run_profile_pp2_ep4.sh`).

### 3.3 Patches landed


| patch id                                      | file                                                                | what it does                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `megatron.deepseek_v4.pp_tensor_shape`        | `primus/backends/megatron/patches/deepseek_v4_pp_shape_patches.py`  | Wraps both `schedules.get_tensor_shapes` (1F1B path) **and** `schedules.forward_backward_pipelining_with_interleaving` (VPP path) so the PP recv buffer matches V4's mHC `[S*K, B, D]` send. Gated on `model_type == "deepseek_v4"` + `hc_mult > 1` + `PP > 1`; strict no-op otherwise.                                                                                                                                 |
| `megatron.deepseek_v4.pp_token_pre_broadcast` | `primus/backends/megatron/patches/deepseek_v4_get_batch_patches.py` | Wraps `pp_module.get_forward_backward_func` so each `train_step` first runs **all** `num_microbatches × num_chunks` `dist.broadcast` collectives upfront (before any send / recv), caching the resulting 6-tuples per `(vp_stage, microbatch)`. A companion wrapper around `pretrain_gpt.get_batch` consumes the cache. Cache reset in a `finally` after each schedule call. Gated on V4 + hash-routed layers + PP > 1. |


### 3.4 `c10d::allreduce` warning gone

The historical `UserWarning: An operator was called with autograd not registered for c10d::allreduce_` came from the early bring-up's
"local shard + `torch.distributed.all_reduce`" path for MoE routed-output
aggregation in `v4_moe.py`. P14 phase-2 migrated MoE to Megatron's token
dispatchers, P17 deleted the `v4_enable_ep_allreduce_fallback` debug gate,
and P19 confirms zero `c10d::allreduce` hits in stderr across all four
smokes + the EP=8 / PP=2 EP=4 profile runs.

## 4. Test Gates Achieved


| Gate    | Description                                               | Status                                                                                                                                                                                        |
| ------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **G1**  | YAML parses + dataclass round-trip                        | ✓ `tests/unit_tests/configs/test_deepseek_v4_yaml.py` (P18)                                                                                                                                   |
| **G3**  | Pre-mul clamped SwiGLU vs HF reference ≤ 1e-6 fp32        | ✓ `tests/unit_tests/megatron/transformer/deepseek_v4/test_clamped_swiglu.py` (P14 phase-1)                                                                                                    |
| **G4**  | Routers ≤ 1e-6 fp32 vs HF reference                       | ✓ `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_routers.py` (P14 phase-1)                                                                                                        |
| **G5**  | 1L MoE forward ≤ 1e-3 fp32 vs HF reference                | ✓ `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_moe.py` (P14 phase-2)                                                                                                            |
| **G6**  | PP=1 / 2 / 4 equivalence on a 4L V4 toy                   | ⚠ CPU sub-gate green (`test_v4_block_pp.py::test_lift_lower_multi_stream_intermediate_roundtrip`). Distributed bit-equality not run; P19 smokes A/B/C cover runtime stability of the patches. |
| **G7**  | MTP ablation (`mtp_num_layers=0` matches LM loss to 1e-6) | ⚠ Not run. CPU MTP block path covered by P16 unit tests; the loss-curve ablation is a follow-up convergence task.                                                                             |
| **G11** | Routing snapshot diff = 0 across PP / EP changes          | ⚠ **Deferred** — snapshot dump tooling never landed. P19 smokes verify runtime stability; the equality check is a future audit and does not block the pre-training path.                      |
| **G14** | Static dead-code audit                                    | ✓ `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p17_dead_code.py` (P17)                                                                                                          |
| Smokes  | A / B / C / D on `mi355-gpu-12`                           | ✓ All four 10/10 iters on 2026-05-07; logs under `progress/p19/`                                                                                                                              |


## 5. Architectural Shifts Forced by Plan-2


| Surface        | Plan-1 (before)                                                                                        | Plan-2 (now)                                                                                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Attention      | `_LegacyDeepseekV4Attention(nn.Module)`; flat GQA-style Q/K/V; CSA / HCA in separate files             | `DeepseekV4Attention(MLASelfAttention)`; single-latent KV; `q_rms` / `kv_norm` / sink / grouped low-rank O; Compressor / Indexer as spec submodules.                                                   |
| MoE            | `nn.Module`; routers without learnable gate; clamped SwiGLU clamps post-mul; fused `[gate, up]` weight | `MegatronModule` with `BaseMoELayer` surface; learned + hash routers both expose `weight` Parameter; pre-mul clamped SwiGLU with split `w1` / `w3` (HF state-dict parity); Megatron token dispatchers. |
| Layer / Block  | Standalone `nn.Module`s outside Megatron's hierarchy                                                   | `DeepseekV4HybridLayer(TransformerLayer)` + `DeepseekV4TransformerBlock(TransformerBlock)`; `isinstance` + sharded-state-dict integrations work.                                                       |
| MTP            | Custom `DeepseekV4MTPBlock` with ad-hoc layer build                                                    | `get_v4_mtp_block_spec()` → `MultiTokenPredictionBlock` + `MultiTokenPredictionLayer`; `process_mtp_loss` integrates with the LM-loss path.                                                            |
| Token-IDs path | `decoder._v4_token_ids` attribute stash                                                                | Forward kwarg `token_ids=` propagated layer → mlp → hash router; AST audit forbids the stash; mid-PP-stage `input_ids` arrive via the upfront PP broadcast patch.                                      |
| HC × PP        | Implicit `[S, B, D]` PP wire (mismatched the V4 mHC math)                                              | `[S, B, K, D]` lifted to `[S*K, B, D]` for PP P2P; recv buffer sized correctly via the `pp_tensor_shape` patch; `HyperHead` only on `post_process`.                                                    |
| TP             | `parallel_mode="duplicated"` on Q-up / O-b                                                             | `column_parallel` / `row_parallel`; bit-identical at TP=1, sharded at TP=2.                                                                                                                            |
| Spec hygiene   | Ad-hoc `TENorm(...)` / `TELinear(...)` calls inside `__init__`                                         | Every replaceable module emitted via `ModuleSpec(module=..., submodules=...)` and built through `build_module`; AST audit enforces it (P18).                                                           |


## 6. Deferred / Out-of-Scope (Not Blocking Pre-Training)

1. **G6 distributed PP equivalence** — the CPU lift / lower round-trip
  covers the math contract; a multi-rank token-0 hidden-state bit-equality
   run across PP=1 / 2 / 4 is a future audit.
2. **G7 MTP loss-curve ablation** — needs a longer convergence run; lives
  alongside the deferred Phase 20 convergence campaign.
3. **G11 routing-snapshot diff = 0 across PP / EP changes** — needs a
  router dump hook + cross-config diff script. Not landed; defer to a
   follow-up plan.
4. **Phase 20 release gates** — 200-step Megatron-bridge convergence,
  TE on/off TFLOPS / HBM perf report, FP8 follow-up plan scoping.
5. **Phase 21 docs / handover** — techblog refresh, progress timeline +
  PPT refresh, `develop_deepseek-v4-in-primus.md` final convention sweep.
6. **Phase 22+ HF state-dict adapter + V4-Flash checkpoint load** —
  `DeepSeekV4StateDictAdapter`, `scripts/load_v4_flash_check.py`,
   `tid2eid` as non-trainable buffer, G8 round-trip, G9 token-0 logits
   ≤ 1e-2 vs HF reference. Activate when an SFT or evaluation campaign
   needs the released V4 weights.

## 7. Pointers

- Live tracker: `[progress/status.md](./status.md)` — Phase 12 → Phase 19 +
Blockers / Risks log.
- Plan-2 source of truth: `[plan-2/](../plan-2/)` (README + 5 numbered docs).
- P19 launchers + logs: `[progress/p19/](./p19/)` (smoke A/B/C/D scripts +
profile launchers + matching `.log` outputs).
- P19 patches:
`primus/backends/megatron/patches/deepseek_v4_pp_shape_patches.py`,
`primus/backends/megatron/patches/deepseek_v4_get_batch_patches.py`.
- Profile traces: see §3.2 above (rank-0 Chrome-trace JSON in the
`output/amd/tas-mi355x-20260507/p19_profile_`* `tensorboard/` folders).
- Day-1 baseline (plan-0 / plan-1): `[progress/2026-04-28-day-1-summary.md](./2026-04-28-day-1-summary.md)`.

