# 03 — Plan-2 Phase Details

> Format: Tasks → Exit Criteria → Risks/Notes.
> Progress lives in `deepseek-v4/develop/progress/status.md` (append-only,
> Phase 12+ section).

---

## P12 — Architecture Review and Lockdown

### Tasks

1. Land plan-2 documents under `deepseek-v4/develop/plan-2/`.
2. Open a Phase 12+ tracking section in `status.md`.
3. Refresh the techblog with as-built notes for plan-1 closing state +
   pointer to plan-2 review findings.

### Exit Criteria

- All four plan-2 docs present and reviewed.
- Phase 12+ section seeded in `status.md`.
- Stakeholders signed off that plan-1 phases 9 / 10 / 11 are paused (no
  more code changes outside plan-2 scope).

### Risks / Notes

- Plan churn risk if review finds additional architecture issues during
  P13. Allow up to 1 cycle of plan amendments before locking.

---

## P13 — Faithful Attention

### Tasks

1. Add `DeepseekV4Attention(MLASelfAttention)` and
   `DeepseekV4AttentionSubmodules(MLASelfAttentionSubmodules)`.
2. Override the KV branch so `K = V = kv` (single-latent). Replace
   MLA's two-tensor split with the V4 reuse pattern.
3. Add per-head `q_rms` (small RMSNorm on `head_dim`) after
   `linear_q_up_proj`. Provider chooses TENorm / FusedLayerNorm / torch.
4. Re-use MLA's `kv_layernorm` for V4's `kv_norm`.
5. Add learnable per-head `attn_sink` and inject into `core_attention`.
   Provider may select `TEDotProductAttention` with sink (when supported)
   or fall back to a `AttentionSinkWrapper` module for eager softmax.
6. Replace `linear_proj` with grouped low-rank O:
   - `linear_o_a`: `[n_heads * head_dim / o_groups → o_groups * o_lora_rank]`
     (column-parallel)
   - `linear_o_b`: `[o_groups * o_lora_rank → hidden]` (row-parallel)
   - `forward` reshapes `[B,S,H,Dh] → [B,S,G,(H/G)*Dh]`, einsums per
     group, folds back.
7. Compressor + Indexer become spec submodules:
   - When `compress_ratio == 0`: no compressor branch. (P13 just lands
     the dense path so Megatron's MLA core_attention takes over.)
   - When `compress_ratio == 4`: build `DeepseekV4Compressor(overlap=True)`
     + `DeepseekV4Indexer`; attention forward gathers per-query top-K and
     joins into joint logits.
   - When `compress_ratio == 128`: build
     `DeepseekV4Compressor(overlap=False)` (no Indexer); concat to SWA KV
     with the compressed-causal mask.
8. Move CSA / HCA classes onto a single `DeepseekV4Attention.forward`
   driven by spec, retiring `csa_attention.py` and `hca_attention.py`
   (their logic moves into `DeepseekV4Attention._compressor_branch`).
9. Drop `linear_q_a` / `linear_q_b` / `linear_k_proj` / `linear_v_proj`
   / `linear_o_proj` field names. Use MLA's standard names (with the
   grouped O exception).
10. Switch projection specs from `parallel_mode="duplicated"` to the TP
    layout above. Use the provider for TE/Turbo/local backends.
11. Numerical alignment scaffold: a 1-layer toy with `compress_ratio=0`
    that runs the V4 attention forward and compares against the HF
    reference attention forward (CPU, fp32, identical weights via the
    state-dict adapter prototype).

### Exit Criteria

- 1L toy V4 attention agrees with HF reference forward to ≤1e-3 absolute
  on token-0 hidden state, fp32 CPU.
- TP=2 sharding test: per-rank parameter shape reduces by 2 along the
  expected axes; gathered output matches TP=1 within 1e-5.
- All `compress_ratio ∈ {0, 4, 128}` paths run without NaN.
- `csa_attention.py` / `hca_attention.py` deprecated (replaced by spec
  submodule handling).

### Risks / Notes

- Reusing `MLASelfAttention` may pull in TE-only fields (e.g.
  `linear_kv_down_proj` for the BMM split). Validate the path that
  bypasses the down-projection step (V4 has no kv_down_proj — `wkv` is
  direct from hidden to head_dim).
- Some MLA fields (e.g. `q_layernorm`) expect `LayerNormBuilder`. The V4
  q_norm differs in that it operates per-head; we may need to subclass
  `LayerNormBuilder` to declare the right shape.

---

## P14 — Faithful MoE + Activation + Router

### Tasks

1. Add `clamped_swiglu_pre_mul(gate, up, alpha)` activation in
   `core/transformer/clamped_swiglu.py`. Update `ClampedSwiGLUMLP`
   semantics: keep `w1` (gate) and `w3` (up) as separate Linears so the
   parameter layout matches the released checkpoint. Provide a fused
   "gate concatenated up" build for grouped-gemm experts (where the
   activation is applied after the GEMM).
2. Add `DeepseekV4LearnedRouter(TopKRouter)` in `moe/v4_topk_router.py`.
   - Adds `score_function` ∈ {"sqrtsoftplus", "sigmoid", "softmax"}.
   - Adds `expert_bias` (used only for selection, not for weights).
   - Honors `moe_router_topk_scaling_factor`.
3. Rewrite `DeepseekV4HashRouter(TopKRouter)`:
   - Exposes a learnable `gate_linear` (same as the learned router).
   - `tid2eid` is an `nn.Buffer` (long, `[vocab_size, topk]`).
   - Routing takes `(hidden, token_ids)`. Returns `(probs, routing_map)`
     identical to the learned router (probs come from gathered learned
     score; expert ids come from `tid2eid`).
4. Rewrite `DeepseekV4MoE` to subclass `MoELayer`:
   - Inherits load-balance loss / z-loss / dispatcher lifecycle.
   - Override only the router selection: `layer_idx < num_hash_layers`
     uses HashRouter; otherwise the learned router.
   - Threading token_ids: the router's `forward` accepts `token_ids` as
     a kwarg; the V4 hybrid layer passes it explicitly (no decoder
     attribute stash).
5. Provider: `v4_grouped_mlp_spec(num_experts, swiglu_limit)` returns a
   `ModuleSpec` with `MLPSubmodules` whose `activation_func` is the
   pre-mul clamped SwiGLU. For TE / Turbo grouped paths that don't allow
   custom activation, the provider downgrades to local experts with
   explicit warning.
6. Provider: `v4_router_spec(learned=True/False)` returns a `ModuleSpec`
   for the learned / hash router with the right submodule wiring.
7. Spec wiring (`deepseek_v4_layer_specs.py`):
   - `mlp` spec uses the V4 MoE with the V4 router selection.
   - `shared_experts` spec uses `SharedExpertMLP` with the same V4
     pre-mul clamp activation.
8. Numerical alignment: 1L toy, MoE-only forward, compare against HF
   reference `DeepseekV4MoE.forward` to ≤1e-3.

### Exit Criteria

- Pre-mul clamped activation passes a unit test against the HF reference
  on randomized inputs (fp32, 1e-6 tolerance).
- Routers produce identical `(probs, routing_map)` to the HF reference
  on identical weights (HashRouter requires the same `tid2eid` table —
  derived deterministically from the seed).
- `DeepseekV4MoE` runs through Megatron's dispatcher with hash + learned
  routers. Loss balance counters update.
- 1L toy MoE forward: token-0 hidden ≤1e-3 vs HF reference.

### Risks / Notes

- TE grouped-gemm activation injection may not exist for V4's pre-mul
  clamp. If so, P14 lands the spec with the local expert path enforced
  for V4 grouped runs and revisits backend support in P19.
- HashRouter checkpoint compatibility requires the `tid2eid` shape to
  match release. Verify by loading from V4-Flash safetensors.

---

## P15 — Faithful Layer + Block + HC × PP

### Tasks

1. Add `DeepseekV4HybridLayer(TransformerLayer)` and
   `DeepseekV4HybridLayerSubmodules(TransformerLayerSubmodules)` with
   `attn_hc` / `ffn_hc` as optional spec submodules.
2. Override the residual path in `DeepseekV4HybridLayer.forward`:
   - Compute HC weights via `self.attn_hc.compute_weights(x_streams)`.
   - Collapse, run attention with `input_layernorm`, expand.
   - Same for `pre_mlp_layernorm` + MLP.
3. Drop the `_v4_token_ids` decoder attribute. Thread `token_ids` as a
   forward kwarg through `TransformerBlock.forward → TransformerLayer.forward →
   MoE.forward → router.forward`. Where Megatron's signature does not
   allow extra kwargs, define a `V4ForwardContext` dataclass and pass it
   in `kwargs` (Megatron passes through `**kwargs`).
4. Replace `DeepseekV4TransformerBlock(nn.Module)` with
   `DeepseekV4TransformerBlock(TransformerBlock)`. Override:
   - `_build_layers`: build `DeepseekV4HybridLayer` from spec.
   - `_lift_streams_in / _lower_streams_out`: helpers that lift/lower
     `[S, B, D]` to `[S, B, K, D]` only at the FIRST and LAST PP stage
     boundaries.
   - PP send/recv shape: between stages, K is flattened into the seq
     axis: `[S, B, K, D] → [S*K, B, D]`. The next stage reverses it
     before running its layers.
5. Apply `HyperHead` only on the post_process stage; otherwise pass
   `[S, B, K, D]` through (K folded into seq for P2P).
6. Update `DeepseekV4Model.forward` to pass `token_ids` explicitly to
   `decoder.forward` (replacing the attribute stash).
7. Update position_ids to use the caller-provided `position_ids`
   (forward kwarg). Drop the `arange(S)` shortcut.
8. PP equivalence test: a 4L V4 toy that runs PP=1 / PP=2 / PP=4 and
   produces matching token-0 hidden states.

### Exit Criteria

- All TransformerLayer / TransformerBlock features (recompute, CUDA
  graph, sequence-parallel) work without bespoke V4 plumbing.
- `decoder._v4_token_ids` removed; no module sets attributes on `decoder`.
- PP=1 vs PP=2 vs PP=4 equivalence on 4L V4 toy: token-0 hidden state
  matches within 1e-4; loss curve matches within 1e-4 over 50 iters.
- HC math is bit-exact across PP boundaries (via the lift/lower helper
  test).

### Risks / Notes

- The `[S, B, K, D] → [S*K, B, D]` packing may collide with sequence-
  parallel chunking. Validate by enabling `sequence_parallel=True` on
  the smoke runs.
- Recompute for HC sub-blocks: the `HyperMixer` math is fp32; verify
  recompute reproduces the same (pre, post, comb) under fp32 forward.

---

## P16 — MTP Integration

### Tasks

1. Add `get_v4_mtp_block_spec(config, *, transformer_layer_spec, vp_stage)`
   helper.
2. `DeepseekV4Model.__init__` builds `self.mtp` from
   `MultiTokenPredictionBlock` when `config.mtp_num_layers > 0` AND
   `mtp_on_this_rank(...)`.
3. `MultiTokenPredictionLayerSubmodules` for V4: `enorm`, `hnorm` =
   provider RMSNorm; `eh_proj` = provider column-parallel linear;
   `mtp_model_layer` = `DeepseekV4HybridLayer` spec; `layer_norm` =
   provider RMSNorm.
4. The MTP head's `HyperHead` is per-MTP-layer. Reuse
   `DeepseekV4TransformerBlockSubmodules.hyper_head` shape per MTP layer,
   stored under `submodules`.
5. Wire `process_mtp_loss` in `DeepseekV4Model.forward` (mirrors
   `MambaModel`).
6. Retire / move `DeepseekV4MTPBlock` to `research/` (or delete) once
   the `MultiTokenPredictionBlock` path is green.

### Exit Criteria

- `mtp_num_layers=1` runs end-to-end on a 4L V4 toy.
- MTP loss term appears in the train log (alongside main LM loss).
- Ablation: `mtp_num_layers=0` vs `mtp_num_layers=1` produces the same
  main-LM loss curve to 1e-6 (since MTP only adds an aux loss path).

### Risks / Notes

- `MultiTokenPredictionBlock` requires `pre_process` / `post_process`
  flags and a layer pattern. The pattern for V4 is just `M` (each MTP
  depth is one V4 hybrid layer); confirm with upstream Mamba example.

---

## P17 — Code Cleanup (dead-code retirement)

> Plan-2 reshuffle (Phase 16 follow-up): the original P17 (HF state-dict
> adapter + V4-Flash checkpoint load) is **deferred to P22+**. Plan-2
> is pre-training-only — no HF weights need to be loaded for the
> release. The slot is now used for the dead-code / hygiene work that
> previously sat in P21, so the lean release happens *before* P18's
> spec audit walks the tree.

### Tasks

1. **Remove `_RMSNorm` duplicates**:
   - `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py`
     keeps a private `_RMSNorm` for the no-spec fallback. Move it
     onto a single canonical implementation (or delete entirely once
     every call site gets a spec-built norm via the V4 provider).
   - `primus/backends/megatron/core/transformer/compressor.py` shadows
     the same class. Pull from a single source of truth.
2. **Retire the standalone `dual_rope.py`**:
   - V4 attention (P13) uses Megatron's rotary path internally; the
     standalone dual-RoPE module survives only as a fallback. Audit
     remaining call sites and remove the file once the spec-driven
     attention path is the only consumer.
3. **Retire `csa_attention.py` / `hca_attention.py`**:
   - All three V4 layer types (`compress_ratio in {0, 4, 128}`) now
     route through `DeepseekV4Attention.forward` (P13). The legacy
     CSA / HCA classes are dead code; delete the files and remove
     them from the package surface.
4. **Retire the legacy `DeepseekV4MTPBlock`**:
   - P16 wired the spec-based MTP path
     (`get_v4_mtp_block_spec` + upstream
     `MultiTokenPredictionBlock` + `process_mtp_loss`). The standalone
     `DeepseekV4MTPBlock` has been deprecation-warned since P16 commit
     `6c5875d4`. Move it under `research/` (or delete) and drop the
     `v4_use_custom_mtp_block` config flag.
5. **Drop the EP `all_reduce` fallback gate**:
   - `v4_enable_ep_allreduce_fallback` toggle and the corresponding
     `c10d::allreduce_` warning path in `v4_moe.py` go away once
     `MoEAlltoAllTokenDispatcher` (or `MoEFlexTokenDispatcher`) is the
     only routed-output reduction path.
6. **Drop the `_v4_token_ids` references everywhere** (was P18 task,
   front-loaded here so the audit phase finds a clean tree):
   - AST audit on the entire `primus/backends/megatron/...` subtree
     confirms zero matches for `_v4_token_ids`.
   - Status check belongs in CI as a static-analysis test.
7. **Fix yaml comments** (was P21 task):
   - `compress_ratios` comment in
     `primus/configs/models/megatron/deepseek_v4_*.yaml` correctly
     documents `4 = CSA` and `128 = HCA` (current comments invert
     this in some files).
8. **`__init__.py` package surface refresh**:
   - Re-export only the live classes; remove deprecated symbols from
     `__all__`.

### Exit Criteria

- No dead-code warnings on a fresh import audit (`isort` + `autoflake`
  both clean on the V4 subtree).
- AST audit confirms no `_v4_token_ids` references anywhere.
- `csa_attention.py`, `hca_attention.py`, standalone `dual_rope.py`,
  and `deepseek_v4_mtp.py` (legacy) are deleted (or moved under
  `research/`).
- `v4_enable_ep_allreduce_fallback` flag removed; YAML config files
  no longer reference it.
- yaml `compress_ratios` comment block matches the canonical
  4 / 128 mapping.
- Package `__init__` exports the active surface only; deprecated
  `DeepseekV4MTPBlock` and CSA / HCA classes removed.

### Risks / Notes

- Removing `_RMSNorm` duplicates without breaking the no-spec
  fallback path (used by tiny CPU smokes / unit tests) is the
  trickiest item — keep one private RMSNorm in `deepseek_v4_block.py`
  if necessary, but make sure the spec-driven path never reaches it.
- Yaml comment fix coordinates with downstream training scripts; do
  it in a single commit with a deprecation table for old field
  values.
- The deferred state-dict adapter (P22+) will need the parameter
  layouts that land here to stay stable; record any rename / remove
  decisions in `02-target-architecture.md` §7 so the future adapter
  knows where the keys went.

---

## P18 — Spec-System Audit

> Note: the `_v4_token_ids` removal moved to **P17** so the audit phase
> walks a clean tree. P15 already eliminated the runtime stash; P17
> deletes any remaining references / docs / tests.

### Tasks

1. Walk every V4 module and confirm:
   - All replaceable submodules are spec submodules (no `nn.Module()`
     constructions inside `__init__` for things the user might want to
     replace).
   - The provider is built **once** per builder call and threaded down.
2. Provider audit: `activation_func()` returns a callable instance, not
   a class. Add an explicit test.
3. YAML schema audit: `compress_ratios` becomes a list; aliases /
   defaults documented; new fields (`o_groups`, `o_lora_rank`,
   `n_shared_experts`, `swiglu_limit`, `attn_sliding_window`,
   `index_topk`, etc.) covered.
4. Add `tests/configs/test_deepseek_v4_yaml.py`:
   - All three V4 yamls (Flash, Pro, Base) parse to valid configs.
   - Schema mismatches surface clear errors.

### Exit Criteria

- `pytest tests/configs/test_deepseek_v4_yaml.py` green.
- Provider singleton test passes.
- No eager construction inside `__init__` for spec-replaceable
  components on a quick AST scan of the V4 subtree.

### Risks / Notes

- Provider threading may need a dataclass-style `BuildContext` to avoid
  passing `provider` through every helper signature; design it once.

---

## P19 — Distributed Re-Validation

### Tasks

1. Re-run the smoke matrix on the rewritten stack:
   - 1×8 BF16 (TP=1 PP=1 EP=1)
   - 1×8 BF16 (TP=1 PP=2 EP=4)
   - 1×8 BF16 (TP=2 PP=2 EP=2)
   - 1×8 BF16 (TP=1 PP=4 EP=2)
   - 2×8 BF16 (DP=2 PP=2 EP=2 TP=2)
2. Capture deterministic routing snapshots (hash router + learned
   router) and compare against a frozen reference.
3. Verify HC math equivalence across PP via the lift/lower path.
4. Validate dispatcher: alltoall + flex paths. If TP×EP grouped path is
   not validated, document and gate.

### Exit Criteria

- All 5 configurations reach `iteration 50` without hang.
- Loss decreases monotonically.
- Routing snapshot diff = 0 across PP / EP changes.
- `c10d::allreduce_` warning gone (verified absent in stderr).

### Risks / Notes

- HC × CP (context-parallel) interaction is left as a Phase 21 follow-up
  unless smoke shows clean behavior.
- Multi-node smokes depend on cluster availability; coordinate with
  ops.

---

## P20 — Convergence / Perf Gates

> Plan-2 reshuffle: full V4-Flash numerical alignment (token-0 logits
> ≤1e-2 vs HF) is **deferred to P22+** (it depends on the deferred HF
> state-dict adapter). The release gate now relies on (a) per-module
> numerical alignment that already lands in G2 / G3 / G4 / G5
> (P13 + P14), (b) PP / MTP equivalence in G6 / G7 (P19), and
> (c) Megatron-bridge training-baseline parity below.

### Tasks

1. **Short-run convergence**: 200-step training on a fixed data slice
   against a **Megatron-bridge baseline** (same data, same optimizer,
   same hparams, no DeepSeek-V4 specifics on the baseline side).
   Loss curves agree within ±0.05.
2. **TE on/off perf**: TE-backed forward + backward vs local-fallback
   on the same 4L config. Record TFLOPS and HBM use.
3. **FP8 follow-up**: scope the FP8 path against TE + Primus-Turbo and
   write a short proposal for the next plan.

### Exit Criteria

- Convergence report published under `deepseek-v4/develop/plan-2/`;
  loss curve plot attached.
- Perf comparison report published.
- Release checklist signed off (go/no-go matrix; HF numerical-
  alignment row marked **N/A — deferred to P22+**).

### Risks / Notes

- Convergence baseline must use exactly the same data slice + RNG
  seed as the V4 run; otherwise the ±0.05 band is meaningless.
- TE on/off perf gate is informational, not blocking, for the
  pre-training release; document the TFLOPS / HBM delta and call out
  any TE-specific regressions.

---

## P21 — Docs + Handover

> Plan-2 reshuffle: the dead-code / yaml-comment items moved to **P17**
> so cleanup happens before the spec audit (P18) walks the tree. P21
> is now docs-only.

### Tasks

1. Update the techblog (`deepseek-v4/develop/techblog/`) with as-built
   notes for P13–P20 (architecture decisions, deferred items, testing
   coverage, known limitations).
2. Refresh `deepseek-v4/develop/progress/` HTML timeline +
   `ppt-template-amd.pptx` slide deck to the final plan-2 state.
3. Refresh `deepseek-v4/develop_deepseek-v4-in-primus.md` with the
   final on-disk convention (modules, specs, yaml fields).
4. Cross-link the deferred **P22+** HF state-dict adapter section
   (this file + `02-target-architecture.md` §7) as the entry point
   for whoever picks up the SFT / evaluation track.

### Exit Criteria

- Tech blog reflects what shipped.
- Progress HTML + PPT updated to plan-2 final state.
- `develop_deepseek-v4-in-primus.md` matches the released module
  surface 1:1.
- Deferred-work index (P22+) is discoverable from both the roadmap
  and the techblog.

### Risks / Notes

- Keep the docs change in a separate commit from any code change so
  reviewers can verify "no behavior delta" easily.

---

## P22+ — HF State-Dict Adapter + V4-Flash Checkpoint Load *(deferred follow-up)*

> **Status: deferred, not on the pre-training release path.** The
> pre-training release ships without HF-weight loading. This section
> is preserved so a future SFT / evaluation campaign can pick it up
> without re-deriving the design.

### Trigger to activate

- A downstream consumer needs to fine-tune from `DeepSeek-V4-Flash`
  weights, OR
- An evaluation campaign needs token-0 logit parity against the HF
  reference checkpoint.

### Tasks (carried over from the original plan-2 P17 + plan-2 P20 numerical-alignment item)

1. Implement `DeepSeekV4StateDictAdapter` per the table in §7 of
   `02-target-architecture.md`.
2. Add `scripts/load_v4_flash_check.py`:
   - Loads the released `DeepSeek-V4-Flash` safetensors.
   - Builds the Primus model (`hc_mult=1` for the smoke; `hc_mult=4`
     once HC is verified).
   - Adapter applied to remap keys.
   - Runs a 64-token forward on CPU (fp32) and compares token-0 logits
     against an HF reference forward on the same prompt.
3. CI: a small (4-layer, 4-experts) deterministic-init checkpoint
   round-tripped through the adapter (Primus → HF dict → Primus).
4. Numerical-alignment harness on a 4L (or full 43L if compute
   permits) slice: token-0 logits ≤1e-2, top-100 mean ≤1e-1.

### Exit Criteria (when activated)

- V4-Flash safetensors load with no missing / unexpected keys.
- Token-0 logits match HF reference to ≤1e-2 in fp32.
- Round-trip preserves the state_dict bit-exact (modulo dtype casts).

### Risks / Notes

- The released checkpoint includes FP4-quantized expert weights for
  larger variants. The adapter's first cut commits to **BF16**
  loading; FP4 / FP8 unpacking is its own follow-up.
- Hash router's `tid2eid` is a checkpoint tensor — make sure the
  adapter loads it as a non-trainable buffer, not as a parameter.
- The HF reference uses dense causal attention as a fallback for
  compressed layers (per the `modeling_deepseek_v4.py` TODO). The
  alignment gate must use the same fallback, not the real Compressor
  / Indexer (otherwise we are comparing different math).
- If parameter layouts shift in P13 / P14 / P15 / P17 between now
  and the trigger, update the adapter table in
  `02-target-architecture.md` §7 in the same PR — that file is the
  contract for this deferred phase.
