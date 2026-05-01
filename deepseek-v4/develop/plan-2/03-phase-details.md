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

## P17 — State-Dict Adapter + Checkpoint Load

### Tasks

1. Implement `DeepSeekV4StateDictAdapter` per the table in §7 of
   `02-target-architecture.md`.
2. Add `scripts/load_v4_flash_check.py`:
   - Loads the released `DeepSeek-V4-Flash` safetensors.
   - Builds the Primus model (`hc_mult=1` for the smoke; `hc_mult=4`
     once HC is verified).
   - Adapter applied to remap keys.
   - Runs a 64-token forward on CPU (fp32) and compares token-0 logits
     against an HF reference forward on the same prompt.
3. CI: a small (4-layer, 4-experts) deterministic-init checkpoint is
   round-tripped through the adapter (Primus → HF dict → Primus).

### Exit Criteria

- V4-Flash safetensors load with no missing / unexpected keys.
- Token-0 logits match HF reference to ≤1e-2 in fp32.
- Round-trip preserves the state_dict bit-exact (modulo dtype casts).

### Risks / Notes

- The released checkpoint includes FP4-quantized expert weights for
  larger variants. P17 only commits to **BF16** loading; FP4 / FP8
  unpacking is deferred.
- Hash router's `tid2eid` is a checkpoint tensor — make sure the
  adapter loads it as a non-trainable buffer, not as a parameter.

---

## P18 — Spec-System Audit

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
5. Drop the `_v4_token_ids` stash everywhere.

### Exit Criteria

- `pytest tests/configs/test_deepseek_v4_yaml.py` green.
- A static check (or AST audit) confirms no `_v4_token_ids` left in the
  tree.
- Provider singleton test passes.

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

## P20 — Numerical / Convergence / Perf Gates

### Tasks

1. **Numerical alignment**: full V4-Flash reference forward → Primus
   forward agreement on a 4L slice (or full 43L if compute permits) at
   ≤1e-2 token-0 logits, ≤1e-1 mean-of-top-100 logits.
2. **Short-run convergence**: 200-step training on the same data slice
   as the HF reference (or a Megatron-bridge baseline). Loss curves
   match within ±0.05.
3. **TE on/off perf**: TE-backed forward + backward vs local-fallback
   on the same 4L config. Record TFLOPS and HBM use.
4. **FP8 follow-up**: scope the FP8 path against TE + Primus-Turbo and
   write a short proposal for the next plan.

### Exit Criteria

- Numerical-alignment report published under
  `deepseek-v4/develop/plan-2/`.
- Convergence report published; loss curve plot attached.
- Perf comparison report published.
- Release checklist signed off (go/no-go matrix).

### Risks / Notes

- Numerical alignment depends on P17's adapter being correct AND P14
  routers gradient-checked.
- The HF reference uses dense causal attention as a fallback for
  compressed layers (per the modeling_deepseek_v4.py TODO). The Primus
  alignment gate must use the same fallback, not the real Compressor /
  Indexer (otherwise we are comparing different math).

---

## P21 — Cleanup + Docs + Handover

### Tasks

1. Remove dead code:
   - `_RMSNorm` duplicates (`block.py`, `compressor.py`).
   - Standalone `dual_rope.py` (replaced by Megatron's rotary).
   - `csa_attention.py` / `hca_attention.py` (folded into
     `DeepseekV4Attention`).
   - `DeepseekV4MTPBlock` (if not moved to research/).
   - EP `all_reduce` fallback gate.
2. Update the techblog (`deepseek-v4/develop/techblog/`) with the
   as-built notes.
3. Refresh `deepseek-v4/develop/progress/` HTML timeline +
   `ppt-template-amd.pptx` slide deck.
4. Update yaml comments (fix the `4 / 128` HCA / CSA confusion).
5. Refresh `develop_deepseek-v4-in-primus.md` with the final convention.

### Exit Criteria

- No dead-code warnings on a fresh import audit.
- Tech blog reflects what shipped.
- Progress HTML + PPT updated to plan-2 final state.

### Risks / Notes

- Coordinating yaml field renames with downstream training scripts —
  do this in a single commit with a deprecation table.
