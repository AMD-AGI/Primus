# DeepSeek-V4 Integration — Day-1 Summary (2026-04-28)

One-day snapshot of the DeepSeek-V4 training-support effort in Primus. All dates below are `2026-04-28` unless noted.

---

## 1. Scope Covered on Day-1


| Phase     | Scope                                                                               | Status                                            |
| --------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| P0        | Investigation + techblog + 4 architecture diagrams + 4-doc development plan         | shipped (docs)                                    |
| P1        | Configs & YAML (base / Flash / Pro + MI355X example + tokenizer)                    | committed                                         |
| P2        | `model_type=deepseek_v4` dispatch (trainer + import_utils + model package skeleton) | committed                                         |
| P3        | Layer specs + `DeepseekV4TransformerBlock` scaffolding                              | committed                                         |
| P4        | HC + Hybrid Attention (Dense / HCA / CSA + Indexer + Compressor + sink + dual-RoPE) | committed                                         |
| P5        | V4 MoE (hash + sqrtsoftplus) + clamped SwiGLU + V4 MTP block                        | committed                                         |
| P6        | End-to-end trainer wiring + PP local layer slicing + EP expert sharding             | implemented in working tree                       |
| P7        | Single-node bring-up with `PP=2, EP=4, MBS=1, GBS=16`                               | **passed on dev box**                             |
| P8        | ~~Convergence / FP8 / FP4~~                                                         | cancelled (out of current scope)                  |
| (dropped) | ~~Original P7 Muon integration~~                                                    | cancelled — Primus already ships distributed Muon |


---

## 2. Documentation Deliverables

All files live under `deepseek-v4/develop/`:

- `develop_deepseek-v4-in-primus.md` — original Chinese spec translated into conversational English (user request: "稍微润色一下，口语化").
- `techblog/01-deepseek-v4-architecture-deep-dive.md` — architecture deep-dive covering HC, Hybrid Attention (CSA / HCA / SWA), MoE (hash + sqrtsoftplus), clamped SwiGLU, dual-RoPE, MTP V4.
- `techblog/diagrams/*.png` — 4 architecture diagrams, rendered directly via Pillow (user explicitly rejected Mermaid/SVG because they wouldn't render in the markdown viewer).
- `techblog/render_diagrams.py` — reproducible PNG renderer with CJK font support (NotoSansSC).
- `plan/00-roadmap.md` — 8-phase roadmap.
- `plan/01-code-layout.md` — file-level landing list.
- `plan/02-phase-details.md` — per-phase task breakdown + risks.
- `plan/03-testing-strategy.md` — unit / integration / numeric-alignment plan.
- `progress/status.md` — per-task checklist with commit hashes + notes.

---

## 3. Code Deliverables

### 3.1 Committed on `dev/wenx/deepseek-v4` (draft PR #698)

Branch is 6 commits ahead of `origin/main`:


| commit     | phase  | scope                                                                                                                            |
| ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `e194e039` | (docs) | architecture deep-dive + 4-doc development plan                                                                                  |
| `d3383c02` | P1     | yaml configs + example training config + tokenizer registration                                                                  |
| `8ae10000` | P2     | `model_type=deepseek_v4` dispatch + V4 model package skeleton                                                                    |
| `a5d2a561` | P3     | layer specs + transformer block scaffolding                                                                                      |
| `3b7ad8c8` | P4     | HC (HyperMixer / HyperHead / Sinkhorn), Compressor, Indexer, SWA, attn sink, dual-RoPE, CSA / HCA attention, standalone V4 block |
| `5e4008dc` | P5     | `clamped_swiglu`, `HashRouter`, `V4TopKRouter`, `DeepseekV4MoE`, `DeepseekV4MTPBlock`, plus `status.md` commit-hash backfill     |


All 5 phase commits are independently importable (bisect-friendly).

### 3.2 New files added on Day-1 (not yet committed at time of writing)

- `run_deepseek_v4.sh` — the Phase-7 single-node launcher (`PP=2, EP=4, MBS=1, GBS=16`) referencing `run_qwen.bak.sh`.
- `deepseek-v4/develop/progress/2026-04-28-day-1-summary.md` — this file.

### 3.3 Day-1 P6/P7 edits in working tree (not yet committed)

Per user instruction ("中途不要 commit 和 push 代码"):

- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_builders.py` — align `model_provider()` signature with upstream Megatron (`config` / `pg_collection`), forward into `DeepseekV4Model`.
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py`
  - build only the layers that belong to this pipeline rank via `get_num_layers_to_build` + `get_transformer_layer_offset`;
  - `set_input_tensor` added so non-first PP stages consume P2P input;
  - robust parsing + length normalization for `compress_ratios` (string / decoder+MTP length / too short or too long);
  - final output wrapped with `make_viewless_tensor` to satisfy PP schedule's `deallocate_output_tensor` assertion.
- `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py` — the custom V4 MTP block becomes opt-in (`config.v4_use_custom_mtp_block`), default path keeps `GPTModel`'s native MTP to avoid extra parameter load in the smoke config.
- `primus/backends/megatron/core/transformer/moe/v4_moe.py` — shard routed experts by EP rank + EP all-reduce on routed output.
- `primus/backends/megatron/core/transformer/dual_rope.py` — rename `DualRoPE.apply()` to `apply_rope()` (clashed with `nn.Module.apply` during DDP init).
- `primus/backends/megatron/core/transformer/deepseek_v4_attention.py`, `attn_sink.py` — cast softmax probs to value's dtype so `matmul` is BF16-safe.
- `deepseek-v4/develop/progress/status.md` — per-task results filled in for P6 / P7.

---

## 4. Phase-7 Smoke Result

Environment:

- Host: `uswslocpm2m-106-2371`
- Container: `dev_primus_wenx_691`
- Command: `TRAIN_ITERS=3 ./run_deepseek_v4.sh`
- Parallelism: `TP=1, PP=2, EP=4, DP=4` across 8 GPUs
- Toy shape: 8 layers, 8 experts, `seq_len=1024`, `MBS=1`, `GBS=16`, `mtp_num_layers=0`

Outcome (3-iter run, log `/tmp/p7_run_final.log`):

- training reached `iteration        3/       3`
- `torchrun finished successfully (code 0)`
- no fatal traceback, no `[direct] torchrun exited with code 1`
- per-rank HIP memory usage around 33–42 GB / 288 GB (< 15 %), well within budget
- reported throughput ≈ 141 TFLOP/s/GPU at this toy shape

### Bugs found and fixed while bringing P7 up

The run was red 7 times before going green; the root causes, in order encountered, were:

1. `model_provider() got an unexpected keyword argument 'config'` — builder signature mismatch with upstream `model_provider`.
2. `Common attention only support rope_type="rope", but got yarn` — default V4 yaml sets YaRN; smoke overrides `rope_type=rope`.
3. `Expert bias for aux-loss-free routing only supports sigmoid score function` — smoke overrides `moe_router_enable_expert_bias=False` because the toy run uses sqrtsoftplus.
4. `DualRoPE.apply() missing 2 required keyword-only arguments` — our `apply()` shadowed `nn.Module.apply`, DDP init trips it; renamed to `apply_rope`.
5. `lr_warmup_steps < lr_decay_steps` assertion — smoke sets `lr_warmup_iters=0`, `lr_decay_iters=TRAIN_ITERS`.
6. `expected scalar type Float but found BFloat16` during `matmul(probs, v)` — cast softmax output to value dtype in both eager path and `attn_sink`.
7. `counter-productive to free a view of another tensor` — V4 block's final `transpose + contiguous` returned a view under certain layouts; wrap with `make_viewless_tensor`.

---

## 5. Key Technical Decisions Made on Day-1

1. **Standalone V4 block instead of a `TransformerBlock` subclass.** Megatron's `TransformerBlock` hard-codes a single residual stream + spec-driven `__init_`_; V4's multi-stream HC + per-layer attention dispatch fits cleanly as its own `nn.Module` that matches `GPTModel.forward`'s call contract.
2. **No custom `argparse` patch for V4.** Primus's `merge_namespace` step (`train_runtime.py:_initialize_trainer`) already copies yaml-only fields onto `backend_args` after `convert_config`, and `MegatronBaseTrainer._patch_parse_args` returns that namespace verbatim. V4 reads its fields via `getattr(config, ..., default)`.
3. **FP32 HC parameters.** `HyperMixer` / `HyperHead` linear weights + the Sinkhorn iterates stay fp32 so doubly-stochastic projection is stable under bf16 activation.
4. **Partial interleaved RoPE** (pairs `(2k, 2k+1)`, only last `rotary_dim` channels rotated) — matches the released V4 weights and is required for dual-RoPE compressed-pool alignment.
5. `**token_ids` piggy-backs on `input_ids` via a transient block attribute** instead of changing the upstream forward signature. This keeps the V4 model compatible with GPTModel's PP/DDP infrastructure; proper cross-PP propagation is a perf-phase concern.
6. **P6 MTP default path = GPTModel's native MTP.** The V4 custom MTP block (`DeepseekV4MTPBlock`) is kept behind `config.v4_use_custom_mtp_block` for experiments; default path integrates naturally with Megatron's loss wiring.
7. **EP-aware MoE with local shard + all-reduce**, rather than token dispatcher, for first-cut functional correctness. The token-dispatcher / grouped-GEMM integration is scheduled for the perf phase.
8. **P7 scope was redefined**: the original Muon phase was cancelled because Primus already ships distributed Muon; P7 is now "end-to-end single-node bring-up". P8 was cancelled entirely for current scope.

---

## 6. Known Follow-ups

Recorded in `status.md` risk log:

- PyTorch warns that the autograd kernel for `c10d::allreduce_` is not registered along the EP routed-output all-reduce path in `v4_moe.py`. Functional on Day-1, but needs a proper token dispatcher / autograd-safe EP path in optimization work.
- Compressor assertion (`seq_len must be divisible by ratio`) is still an asserting contract; when switching to non-toy sequence lengths we should guarantee divisibility or relax the check.
- TP partitioning of QKV / Compressor / Indexer is not yet wired — P6 leaves this as TODO.
- No 50-iter stability run yet — only a 3-iter functional smoke was executed for Day-1 closure.
- Numeric parity vs HF / DeepSeek reference is not attempted (requires loading reference checkpoint, perf-phase work).

---

## 7. Repo / PR Status at End of Day-1

- Branch: `dev/wenx/deepseek-v4` — 6 commits ahead of `origin/main`, pushed.
- PR: [https://github.com/AMD-AGI/Primus/pull/698](https://github.com/AMD-AGI/Primus/pull/698) — **Draft**, title `feat: deepseek-v4 model support`, description carries the P1–P5 walkthrough (P6 / P7 walkthrough not yet added because those commits have not been created yet).
- Working tree at close of Day-1 contains P6 / P7 edits plus this summary + `run_deepseek_v4.sh`, pending an explicit commit/push step.
