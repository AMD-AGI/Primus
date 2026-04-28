# 02 — Per-Phase Detailed Task List

> File locations live in [`01-code-layout.md`](01-code-layout.md).
> Each phase follows a **Tasks → Exit criteria → Risks / notes** triad.
> When a task is done, tick it in [`../progress/status.md`](../progress/status.md)
> and record the commit hash.

---

## Phase 1 — Configs & yaml

### Tasks

1. **Create base config**: `primus/configs/models/megatron/deepseek_v4_base.yaml`
   - `extends: llama_base.yaml` (cf. `deepseek_v3_base.yaml`)
   - Add the new V4 fields with their defaults (even if they're not yet used in v1):
     ```yaml
     model_type: deepseek_v4
     # HC
     hc_mult: 4
     hc_use_sinkhorn: true
     # Hybrid attention
     hybrid_attention_enabled: true
     compress_ratios: null            # provided by the per-variant yaml
     index_topk: 512
     compress_rope_theta: 160000.0
     attn_sliding_window: 128
     attn_sink: true
     # MoE
     num_hash_layers: 3
     moe_score_function: sqrtsoftplus  # ∈ {sqrtsoftplus, sigmoid}
     # MTP
     mtp_num_layers: 1
     mtp_use_separate_hc_head: true
     # Misc
     activation_func: clamped_swiglu
     ```
2. **V4-Flash main yaml**: `primus/configs/models/megatron/deepseek_v4_flash.yaml`
   - Strictly mirror `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/config.json`
     (hidden_size, num_hidden_layers, num_attention_heads, moe_intermediate_size,
     `compress_ratios` list, etc.).
3. **MI355X training yaml**: `examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml`
   - Use `deepseek_v3-BF16-pretrain.yaml` as the template; tune PP layout and
     recompute layer ids as needed.
4. **(Optional, end of Phase 1)** Provide V4-Pro / V4-Pro-Thinking / V4-40B base
   yamls; do not run them yet.

### Exit Criteria

- `python -c "from primus.core.config_loader import load_config; load_config('configs/models/megatron/deepseek_v4_flash.yaml')"`
  (or equivalent loader) loads without errors.
- New fields are not rejected by Megatron argparse — at minimum register them
  via an `extra_args_provider` (cf. `pretrain_mamba.py`).

### Risks / Notes

- New fields **must** be registered into Megatron CLI in Phase 1 — otherwise
  Phase 2 builder import will fail when args are missing. Add an
  `add_arguments` hook in `deepseek_v4_layer_specs_patches.py` that registers
  but doesn't yet consume them in Phase 1.

---

## Phase 2 — Register `model_type=deepseek_v4`

### Tasks

1. **Add `primus/pretrain_deepseek_v4.py`** (top-level builder shell).
   - Mirror `third_party/Megatron-Bridge/3rdparty/Megatron-LM/pretrain_mamba.py`.
   - Provides `model_provider`, `forward_step`, `train_valid_test_datasets_provider`, `extra_args_provider`.
   - In v1 the `model_provider` simply wraps `pretrain_gpt.model_provider` so
     trainer dispatch can be exercised end-to-end first.
2. **Modify `primus/backends/megatron/megatron_pretrain_trainer.py`**: add a
   branch in the `model_type` switch:
   ```python
   elif model_type == "deepseek_v4":
       from pretrain_deepseek_v4 import (
           forward_step,
           train_valid_test_datasets_provider,
       )
       log_rank_0("Using DeepSeek-V4 model provider and training components")
   ```
3. **Modify `primus/core/utils/import_utils.py:get_model_provider`**: add a
   `deepseek_v4` branch:
   ```python
   elif model_type == "deepseek_v4":
       model_provider = lazy_import(
           ["model_provider", "pretrain_deepseek_v4"], "model_provider", ...)
       deepseek_v4_builder = lazy_import(
           ["deepseek_v4_builders"], "deepseek_v4_builder", ...)
       return partial(model_provider, deepseek_v4_builder)
   ```
4. **Add `primus/deepseek_v4_builders.py`** (peer of `mamba_builders.py`):
   ```python
   def deepseek_v4_builder(args, pre_process, post_process, vp_stage=None, config=None):
       ...
   ```
   v1 just returns a `GPTModel` (Phase 3 will replace it with `DeepseekV4Model`).
5. **Add `primus/backends/megatron/core/models/deepseek_v4/__init__.py`** plus
   a stub `deepseek_v4_model.py` (`from megatron.core.models.gpt import GPTModel as DeepseekV4Model`).

### Exit Criteria

- A 1L tiny config (`hidden_size=64, num_layers=1, num_attention_heads=2, moe disabled`)
  starts with `model_type: deepseek_v4`, loss is not NaN, runs 10 iterations.

### Risks / Notes

- `pretrain_deepseek_v4.py` must be on `sys.path` (mamba is wired up the same
  way). Primus's lazy_import already tries `["model_provider", "pretrain_deepseek_v4"]`
  candidates, which matches existing behavior.
- If the Megatron version differs in `args.spec` handling (mamba_builders.py:17
  imports a custom LayerSpec via `args.spec`), we need our V4 yaml to set
  `spec: primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs.deepseek_v4_stack_spec`.

---

## Phase 3 — Layer Spec + Block Scaffolding

### Tasks

1. **`deepseek_v4_layer_specs.py`**: cf. `mamba_layer_specs.py`, define
   - Three layer kinds:
     - `attention_layer = TransformerLayer(self_attention=MLA-like, mlp=MoE)` — default layer
     - `dense_attention_layer` — special layer for `compress_ratio==0`
       (head/tail layers / hash layers)
     - `mtp_layer` — used only by MTP
   - Expose `deepseek_v4_stack_spec` so the builder can `import_module(args.spec)`.
2. **`deepseek_v4_block.py`**: implement the 1-stream version first (HC=1) so
   forward / backward both pass; the hidden-shape contract is already written
   for `[B, S, hc_mult, D]`, but with hc_mult=1 it degenerates to baseline.
3. **`deepseek_v4_model.py`**: cf. `mamba_model.py`, assemble embed / decoder /
   final_norm / lm_head; preserve `pre_process` / `post_process` for PP.
4. **Argument registration**: in `pretrain_deepseek_v4.py:extra_args_provider`,
   register every Phase-1 V4 field via `add_argument` and align defaults with
   the yaml.
5. **Builder swap**: change `deepseek_v4_builder` to construct `DeepseekV4Model`
   instead of `GPTModel`.

### Exit Criteria

- With the V4-Flash yaml (set `num_layers: 4` so the toy fits), forward + backward
  passes, 10 iterations of monotonically-decreasing loss without NaN.
- Parameter count is close to the reference (estimate from
  `num_layers, hidden_size`); ±1% tolerance.

### Risks / Notes

- **We do not implement the real V4 attention here**: attention can be Megatron's
  built-in `MLASelfAttention` (V3-style) or even ordinary MHA + RoPE. Semantic
  correctness doesn't matter; the goal is to land the scaffolding.
- PP partitioning: pin PP=1 in v1 to dodge the HC-across-PP issue; multi-PP is
  Phase 6's territory.

---

## Phase 4 — HC + Hybrid Attention (**Core 1**)

### Tasks

> Strongly recommend small commits — one file per PR plus a unit test.

1. **`hyper_connection.py`**
   - `class HyperConnection(nn.Module)`: implement `compute_weights / collapse / expand`.
   - `class HyperHead(nn.Module)`: sigmoid-only weighted sum, no Sinkhorn.
   - References: `inference/model.py:Block.attn_hc, ffn_hc` call chain + NeMo
     `DeepseekV4HyperConnection`.
   - **Mathematical highlights (must implement faithfully)**:
     - `pre = sigmoid(linear) + eps`, `post = 2 · sigmoid(linear)`
     - `comb_logit → softmax(-1) + eps → Sinkhorn-Knopp(20 iters)`
     - `flat = x.flatten(2).float()` then `rsqrt` normalization (avoids
       mixed-precision issues).
2. **`compressor.py`**
   - Two modes:
     - `overlap=True, ratio=4`: used for the CSA main path + Indexer main path
     - `overlap=False, ratio=128`: used for HCA
   - Internally: `wkv` + `wgate` + `softmax(score+ape, dim=ratio) · KV` → pooled.
   - APE = absolute positional embedding, computed against `compress_rope_theta`.
3. **`indexer.py`**
   - Contains `mini Compressor @ index_head_dim=128` + `wq_b: q_lora_rank → 64·128`
     + `scores = ReLU(Q · K^T) · weights_proj`.
   - Outputs `topk indices [B, S, K]`, with `-1` denoting causally invalid.
   - **Mask rule**: `p < (q + 1) // ratio`.
4. **`sliding_window_kv.py`**: keep the most recent `attn_sliding_window=128` of KV.
5. **`attn_sink.py`**: a per-head learnable scalar concatenated into the softmax
   tail column, then sliced via `[..., :-1]`.
6. **`csa_attention.py`** + **`hca_attention.py`**
   - Shared base: q_path (LoRA + per-head rsqrt + partial RoPE 64), output path
     (inverse RoPE + grouped low-rank wo_a + wo_b).
   - Differences:
     - CSA uses Indexer-derived top-K mask + SWA buffer concat compressed_KV.
     - HCA: all queries share the entire pool, only causally clamped via
       `p < (q+1)//ratio`.
7. **`dual_rope.py`**
   - Two freq tables: `rotary_base` (default 10000) and `compress_rope_theta`
     (default 160000).
   - **Important**: YaRN scaling is enabled **only** on layers with
     `compress_ratio != 0` (verified in the reference).
8. **`hyper_connection_patches.py`** at `before_train` injects: upgrade
   `deepseek_v4_block.py` to its full 4-stream HC implementation.
9. **Update `deepseek_v4_block.py`**:
   - Two sub-blocks: `[attn_hc, attention]` and `[ffn_hc, FFN]`.
   - Pick the attention implementation by `layer_idx`: `compress_ratio==0` →
     dense MLA, `==4` → CSA, `==128` → HCA.

### Exit Criteria

- **Unit test 1**: 1L toy model (hidden_size=128, hc_mult=4) with fixed weights;
  forward output max abs error < 1e-4 vs NeMo `DeepseekV4HyperConnection`.
- **Unit test 2**: with fixed weights, Compressor / Indexer outputs agree with
  the reference `inference/model.py` (< 1e-3).
- **Unit test 3**: full 1L block forward + backward; gradients flow without inf/nan.
- **Integration test**: V4-Flash yaml with 4L toy model runs 50 iterations with
  monotonic loss decrease.

### Risks / Notes

- **Sinkhorn numerics**: must reuse the reference `_hc_split_sinkhorn`
  algorithm with 20 fixed iterations, otherwise numerics drift.
- **partial RoPE**: rotates only `qk_pos_emb_head_dim=64` instead of the full
  `qk_head_dim=192`; NeMo's `_apply_partial_rope_interleaved` already corrects
  the HF mistake — follow NeMo.
- **Indexer's reuse of q_lora_rank**: the Indexer's `wq_b` follows the main
  attention's `wq_a + q_norm`, but **does not reuse** wq_b. This is easy to
  misread in the reference.
- **attn_sink position**: concatenated **before** softmax as an extra column,
  then sliced **after** softmax. Equivalent to an "opt-out" probability.
- **No sparse-attn kernel in v1**: just `torch.matmul + masked_fill + softmax`,
  perf-poor but correct; performance work via triton fusion lives in P8.

---

## Phase 5 — MoE / Activation / RoPE / MTP (**Core 2**)

### Tasks

> Phase 5 can run in parallel with Phase 4. Each item should be its own PR.

1. **`hash_router_patches.py`**
   - New `class HashRouter(BaseRouter)`: `tid2eid` static lookup (each token id
     has a fixed expert id per layer).
   - Active when `layer_idx < num_hash_layers` (V4-Flash = 3).
   - **EP compatibility**: when EP > 1, HashRouter either marks tokens not owned
     by the current rank as drop (re-route to shared expert) or forwards via
     alltoall. The exact strategy is documented in
     `notes/2026-XX-hash-routing-ep.md` (filled in during dev).
2. **`sqrtsoftplus_router_patches.py`**
   - Extend `PrimusTopKRouter` to support `score_function ∈ {sigmoid, sqrtsoftplus}`.
   - `sqrtsoftplus(x) = sqrt(softplus(x))`.
   - Coexists with `noaux_tc` (`moe_router_enable_expert_bias=true`).
3. **`clamped_swiglu.py`**
   - `swiglu(x) = clamp(silu(x[..., :d]) * x[..., d:], min=-α, max=α)` where α
     is sourced from config (V4 default verified against `inference/model.py`).
   - Register `clamped_swiglu` in Megatron's activation function registry.
4. **layer-aware `dual_rope.py`**: extend `RotaryEmbedding` to be
   layer-conditional: `if layer.compress_ratio != 0: theta=compress_rope_theta + YaRN; else: theta=rotary_base`.
5. **`mtp_v4_patches.py`**
   - V3 already provides an mtp path, but V4's MTP has its own HC head (see
     reference `MTPBlock`).
   - Implementation:
     - Share `embed` and `lm_head`.
     - **Do not** share the HC head (mtp owns its own `hc_head_fn`).
     - Reuse Megatron's `mtp_num_layers` hook, only swapping the head.

### Exit Criteria

- 4L V4-Flash yaml (hash routing on for the first 3 layers + sqrtsoftplus +
  clamped_swiglu + dual_rope + mtp_num_layers=1) runs forward + backward with
  monotonic loss decrease.
- Forward logits at token-0 produced by reference `inference/model.py` agree
  with Primus on the same yaml + same weights to max abs error < 1e-2.

### Risks / Notes

- **Hash routing token ids**: the `tid2eid` table must be consistent across
  PP / TP / EP. We recommend generating it once in `MegatronArgBuilder` from
  a deterministic seed + tokenizer vocab size, instead of generating per-rank.
- **clamped SwiGLU vs Megatron fused**: with `apply_swiglu_fusion=true`,
  Megatron uses the fused kernel which has no clamp hook. v1 should force the
  unfused path (`apply_swiglu_fusion=false`); the fused-with-clamp variant
  comes in the perf phase.
- **Separate MTP HC head**: do not let Phase-3 HyperHead get reused by mtp —
  state_dict naming must be split.

---

## Phase 6 — Trainer End-to-End + TP / PP / EP

### Tasks

1. **Replace stub model**: drop the Phase-2/Phase-3 stub; wire
   `deepseek_v4_builder` to the full `DeepseekV4Model` with HC + V4 attention +
   V4 MoE.
2. **PP layout design**:
   - HC's 4 streams must flow through one PP segment as a single unit; PP
     send/recv treats `[B, S, hc_mult, D]` as one tensor.
   - We forbid one layer with HC=4 and another with HC=1 — enforced via yaml
     check.
   - Provide a fixed v1 layout:
     `pipeline_model_parallel_layout: ".=Et*1|t*1|t*2|...|L"`, sized for V4-Flash's 43 layers.
3. **TP partitioning**:
   - QKV continues to use `TELayerNormColumnParallelLinear`.
   - Compressor / Indexer's `wkv`, `wq_b`, `weights_proj` go through
     `ColumnParallelLinear` / `RowParallelLinear`.
   - HC's `compute_weights` uses `linear` + `rsqrt` and is **not** TP-split
     (hc_mult=4 is too small to be worth splitting).
4. **EP**:
   - Inherit Megatron's MoE EP path entirely.
   - Add an `expert_to_rank` table only at the hash-router dispatch point so
     tokens go to the correct EP rank.
5. **Smoke test**:
   - 1×8 GPU BF16 V4-Flash full 43L + 1MTP runs.
   - 4×8 GPU BF16 runs.

### Exit Criteria

- 1×8 GPU BF16 50-iter run: no hang, loss decreases continuously.
- 4-node 32-GPU BF16 50-iter run: no hang.

### Risks / Notes

- **Critical**: HC streams' PP send/recv must keep dtype/shape consistent;
  any mismatch hangs the moment PP > 1. Add an early echo test in Phase 6:
  a toy `hc_mult=2` net forwarding through PP=2 to validate the pipe.
- recompute_layer_ids design: HC's 4-way expansion makes activations 4×, so V4
  needs more selective recompute than V3. Default to
  `recompute_granularity: full` for all CSA/HCA layers.

---

## Phase 7 — Muon Optimizer Integration

### Tasks

1. **Reuse `muon_optimizer_patches.py`** and only extend the V4 param-group split:
   - **AdamW group**: embedding, lm_head, all HC compute_weights parameters,
     HyperHead, MTP head.
   - **Muon group**: attention's wq_a / wq_b / wkv / wo_a / wo_b, all
     Compressor / Indexer weights, MoE expert + router weights.
   - Implement the rule in `deepseek_v4_builder` via parameter-name prefix
     filtering (`hc.`, `embed.`, `lm_head.`, `mtp.`).
2. **Hyperparameter alignment**: use the Hybrid Newton-Schulz coefficients from
   the RedNote slide 9 (`a=3.4445, b=-4.7750, c=2.0315`) and 5 iterations.

### Exit Criteria

- 50-iter BF16 run with muon enabled passes.
- Grad-norm is stable vs the AdamW baseline (no explosion / vanishing).
- Loss slope ≥ AdamW baseline.

### Risks / Notes

- Muon does NS iteration on **matrix parameters**, so 1D params (norm.weight,
  scale) must go to AdamW. Megatron's existing muon path already does this,
  but V4 introduces HC weights (4D-ish: linear + rsqrt stitched together)
  which can be misclassified — we need an explicit filter rule.

---

## Phase 8 — Convergence + FP8 + FP4

### Tasks

1. **Numerical alignment**: with fixed V4-Flash weights (loaded from the
   reference checkpoint), forward output max abs error < 1e-2 (BF16) vs
   `inference/model.py`. Record in `notes/<convergence-runs>.md`.
2. **Short runs**: mock_data / bookcorpus, 1B tokens; record loss curves and
   compare against V3 / V3.2.
3. **FP8**: reuse Megatron's `fp8_patches.py`;
   `deepseek_v4_flash-FP8-pretrain.yaml` adds `fp8: hybrid`,
   `fp8_recipe: e4m3`.
4. **(Optional) FP4**: reuse `fp4_patches.py`; verify wgrad / fwd don't crash;
   perf data is secondary.

### Exit Criteria

- BF16 1B-token convergence curve agrees with the paper / reference experiment
  to within ±0.05 nats.
- FP8 throughput is ≥ +30% over BF16 on MI355X.
- No NaN / Inf.

### Risks / Notes

- FP8 may have numerical issues with V4's sparse attn / HC paths; if needed,
  force HC compute_weights to fp32 (the reference itself does so via `.float()`).
- FP4 only does GEMM, not reduce / softmax, so attention softmax stays fp32.

---

## Cross-Phase Out-of-Scope (explicitly not doing)

- Anticipatory Routing (an optional optimization in the V4 paper, not required
  for convergence)
- A custom sparse-attn HIP kernel
- A custom Compressor fusion kernel
- Training validation for DeepSeek-V4-Pro / Pro-Thinking / V40B (yaml only)
