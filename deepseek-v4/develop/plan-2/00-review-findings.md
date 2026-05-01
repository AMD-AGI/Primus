# 00 — Plan-2 Code Review Findings

> Snapshot of the current `dev/wenx/deepseek-v4` branch on top of base
> commit `e194e03974115ba6344daf3296e50bbcfe543a93`.
> Commit walk: P1 (`d3383c02`) → P10 (`b38e83cf`) → smoke stabilization
> (`752b7534`).

Severity legend:

| Tag | Meaning |
|---|---|
| **CRIT** | Architecture diverges from real DeepSeek-V4 — the model trained today is not a real V4 |
| **HIGH** | Distributed / spec / parallelism correctness; deferred decisions that block release |
| **MED**  | Reuse / hygiene; landed code works but breaks Megatron conventions and limits future maintainability |
| **LOW**  | Local style / test-coverage / documentation drift |

---

## A. Architecture Faithfulness vs Real V4

> Reference sources cross-checked: `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/{config.json,inference/model.py}`,
> `deepseek-v4/transformers/src/transformers/models/deepseek_v4/{configuration_deepseek_v4.py,modeling_deepseek_v4.py}`,
> `deepseek-v4/NVIDIA-NeMo/Automodel/nemo_automodel/components/models/deepseek_v4/{model.py,layers.py}`.

| ID | Severity | Finding | Where |
|---|---|---|---|
| **A1** | CRIT | Attention uses **separate** `linear_k_proj` / `linear_v_proj`. Real V4 has a **single** `wkv` projection (single-latent MQA: K = V = kv). Wastes parameters AND blocks checkpoint loading. | `primus/backends/megatron/core/transformer/deepseek_v4_attention.py:171-180` |
| **A2** | CRIT | **Missing per-head `q_norm`** RMSNorm after `wq_b`. Real V4 reference does `q_rms = rsqrt(mean(q^2))` before partial RoPE. | same — `_project_q` / `forward` path has no norm |
| **A3** | CRIT | **Missing `kv_norm`** RMSNorm after `wkv`. Real V4 reference does `kv_norm(self.wkv(hidden))` before partial RoPE. | same |
| **A4** | CRIT | **Missing grouped low-rank O-projection** (`o_groups=8`, `o_lora_rank=1024`). Primus uses a flat `linear_o_proj`. Wrong parameter count vs released checkpoint. | `deepseek_v4_attention.py:183-187`; HF reference at `modeling_deepseek_v4.py:wo_a/wo_b` |
| **A5** | CRIT | **HashRouter is wrong**. Real V4 `Gate` has a learnable `weight = Parameter(n_routed_experts, hidden_size)` + `tid2eid` lookup; the **expert ids** come from the table, but the **routing weights are gathered from the (learned) score**. Primus instead emits `1/topk` uniform weights with no scoring at all. | `primus/backends/megatron/core/transformer/moe/v4_hash_router.py:96-141` |
| **A6** | CRIT | **Clamped SwiGLU semantics are different**. Real V4 clamps **gate and up separately, BEFORE the multiply**: `gate.clamp(max=α)`, `up.clamp(min=-α, max=α)`, then `silu(gate)*up`. Primus instead clamps the **post-multiply output**: `clamp(silu(gate)*up, min=-α, max=α)`. | `primus/backends/megatron/core/transformer/clamped_swiglu.py:45-66` |
| **A7** | CRIT | **Fused `gate_up` projection** in `ClampedSwiGLUMLP` does not match real V4's separate `w1` (gate) / `w3` (up) parameter layout. Blocks checkpoint loading even after A6 is fixed. | `clamped_swiglu.py:103-104` |
| **A8** | HIGH | `HCAAttention` applies compress-base RoPE with positions `arange(P)`, but real V4 uses `compressed_position * compress_ratio` so that the compressed position represents the END of its raw window. RoPE phase alignment differs. | `primus/backends/megatron/core/transformer/hca_attention.py:124-130` |
| **A9** | HIGH | **No state-dict adapter** between Primus parameter names and HF / inference checkpoint keys (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, `attn_sink`, `attn_norm`, `ffn_norm`, `hc_attn_*`, `hc_ffn_*`, `gate.weight`, `gate.bias`, `gate.tid2eid`, `experts.i.{w1,w2,w3}`, `embed`, `norm`, `head`). Primus model **cannot consume V4 weights** for finetuning or evaluation. | nothing under `primus/backends/.../models/deepseek_v4/` |
| **A10** | MED | Yaml comment claims `4 = HCA branch, 128 = CSA branch`, but the code routes `compress_ratio == 4` → `CSAAttention` (with Indexer) and `compress_ratio == 128` (or any other) → `HCAAttention`. Misleading documentation; verified the code matches the V4 convention (4 = sparse-with-indexer, 128 = full compressed pool). | `primus/configs/models/megatron/deepseek_v4_flash.yaml:35-38` |

## B. Megatron Reuse / Spec Pattern Violations

> Reference: `third_party/Megatron-LM/megatron/core/models/{gpt,mamba}/`, the
> mamba layer-spec convention in `mamba_layer_specs.py`, and the
> `MLASelfAttention` / `MoELayer` / `TransformerLayer` / `TransformerBlock`
> construction expected by the standard `model_provider`.

| ID | Severity | Finding |
|---|---|---|
| **B1** | CRIT | `DeepseekV4Attention` is a plain `nn.Module`, **not a subclass of `MLASelfAttention` / `Attention`**. V4 is fundamentally MLA + extras. We're re-implementing MLA from scratch and losing TE/Turbo backends, fused MLA-RoPE kernels, distributed checkpoint sharding, etc. |
| **B2** | CRIT | `DeepseekV4TransformerBlock(nn.Module)` does **not subclass `TransformerBlock`**. Re-implements PP local-layer construction, `set_input_tensor`, final-norm placement. Loses recompute, CUDA graph, sequence-parallel, distributed checkpoint topology. |
| **B3** | HIGH | `DeepseekV4HybridLayer` extends `GraphableMegatronModule` (recompute only), not `TransformerLayer`. Loses BDA fusion, native MLP / MoE replacement via spec. The HC mixing should be the residual function of a `TransformerLayer`. |
| **B4** | HIGH | `DeepseekV4MoE(nn.Module)` does **not subclass `MoELayer`**. Misses load-balance loss / z-loss tracking, MoE FSDP sync, MoE recompute, expert-capacity counters, dispatcher lifecycle hooks. Forces the dispatcher integration to be re-derived ad-hoc inside V4 MoE. |
| **B5** | HIGH | MTP doesn't go through `MultiTokenPredictionBlock`. Custom `DeepseekV4MTPBlock` is opt-in (`v4_use_custom_mtp_block`) and bypasses Megatron's MTP loss path entirely. The default training run has **no MTP** even though V4-Flash ships `num_nextn_predict_layers=1`. |
| **B6** | HIGH | `Compressor`, `Indexer`, `AttentionSink`, `DualRoPE`, `HyperMixer`, `HyperHead` are constructed inside module `__init__` paths, **not declared as spec submodules**. Cannot swap implementations (e.g., a fused compressor / TE attention sink) by editing the spec. |
| **B7** | MED | `MLATransformerConfig` is the parent, so V4 inherits MLA fields but never builds an MLA module. Either drop the dependency (and document) or actually use `MLASelfAttention`. Currently it's a mismatch. |
| **B8** | MED | The dataclass `DeepseekV4HybridLayerSubmodules` has positional defaults that include private classes (`_RMSNorm`, `_DenseSwiGLUMLP`) — these private symbols leak into the spec API. |

## C. Distributed Correctness

| ID | Severity | Finding |
|---|---|---|
| **C1** | CRIT | **HC × PP is broken**. `DeepseekV4TransformerBlock.forward` applies `hyper_head` (collapse `[B,S,K,D]→[B,S,D]`) at the END of every PP stage, then re-expands `[B,S,D]→[B,S,K,D]` from scratch on the next stage. The K-stream context is destroyed at every PP boundary — the HC math is mathematically equivalent to running with `hc_mult=1` across stages, and the only stage where HC is meaningful is the last one. |
| **C2** | HIGH | **Token-IDs stash anti-pattern**. `DeepseekV4Model.forward` writes `decoder._v4_token_ids = input_ids` and `DeepseekV4TransformerBlock.forward` reads it. This: (a) does not propagate across PP P2P, so any non-first PP stage that owns hash-routed layers cannot see `input_ids`; (b) leaves stale state if forward raises; (c) confuses `torch.compile` / CUDA graphs because the attribute mutates during forward. |
| **C3** | HIGH | **Position IDs faked inside block forward**. `position_ids = torch.arange(S, device=x.device)` ignores the caller-supplied `position_ids`. Breaks sequence packing, padded-data position offsets, context-parallel rank-relative positions. |
| **C4** | HIGH | **TP partitioning broken on attention projections**. `_build_linear_projection_spec` uses `parallel_mode="duplicated"` for **all** Q/K/V/O projections, so each TP rank holds a full copy of these matrices. With TP > 1 the model wastes parameters and the activations are not sharded along the head axis. |
| **C5** | MED | EP routed-output `all_reduce` fallback is gated by `v4_enable_ep_allreduce_fallback` but the helper still imports `torch.distributed`. With dispatcher integration in place, this fallback should be removed (or moved to a debug path). |
| **C6** | MED | `_v4_token_ids` is set on `decoder` but `DeepseekV4HybridLayer.forward` reads it via the `token_ids` argument that is passed by the block's loop. Inconsistent two-channel propagation; pick one. |

## D. Spec System / Builder Hygiene

| ID | Severity | Finding |
|---|---|---|
| **D1** | HIGH | `DeepSeekV4SpecProvider` is **re-instantiated** inside `_build_projection` (block) and inside the layer-specs builder. The provider should be passed in once via `submodules` / `params` and threaded through. |
| **D2** | HIGH | `provider.activation_func()` returns the **class** `TEActivationOp` and is plugged into `MLPSubmodules.activation_func`. Megatron expects either a callable function (e.g. `F.silu`) or a callable instance; passing a class can silently mis-fire when the MLP calls `self.activation_func(x)` and gets `TEActivationOp(x)` (a constructor call). Needs explicit instance / functional wrap. |
| **D3** | HIGH | `moe_router_topk_scaling_factor` is hard-coded to `1.0` in the V4 MoE router build path; the yaml field is silently ignored. |
| **D4** | MED | `compress_ratios` is stored as a YAML **string** `"[0, 0, 4, ...]"` and parsed with `ast.literal_eval` at runtime. Should be a YAML list / dataclass field. |
| **D5** | MED | `DeepseekV4MoESubmodules` defaults `token_dispatcher = MoEAlltoAllTokenDispatcher` (class), but the spec-driven path expects a `ModuleSpec`. Inconsistent. |
| **D6** | MED | `DeepseekV4MoE` shared expert is built by **deep-copying the config and mutating it** (`shared_cfg = copy(self.config); shared_cfg.activation_func = F.silu; ...`). This is fragile — any field added to `TransformerConfig` upstream that requires `__post_init__` re-run will silently drift. |
| **D7** | LOW | The yaml `compress_ratios` for V4-Flash has 44 entries (43 decoder + 1 MTP) but `_normalize_compress_ratios` truncates the MTP entry. The MTP layer's compress ratio (`4`) is currently dropped on the floor. |

## E. Code Quality / Reuse

| ID | Severity | Finding |
|---|---|---|
| **E1** | HIGH | Local `_RMSNorm` defined in **two** files (`deepseek_v4_block.py`, `compressor.py`) duplicates Megatron's `WrappedTorchNorm` / `TENorm` / `LayerNormBuilder`. |
| **E2** | HIGH | Local `_DenseSwiGLUMLP` re-implements `MLP` with provider-driven projections; should reuse `MLP` + provider-selected `linear_fc1` / `linear_fc2` like every other model. |
| **E3** | HIGH | `dual_rope.py` re-implements YaRN frequency scaling and the partial-RoPE application kernel. Megatron has `RotaryEmbedding(rotary_interleaved=True)` and `YarnRotaryEmbedding`. Net effect: Primus carries a parallel rotary stack. |
| **E4** | MED | `AttentionSink` is a tiny module that re-derives `softmax_with_sink`. TE's `TEDotProductAttention` supports `attention_sink` natively (in newer TE versions); should be used when available. |
| **E5** | LOW | `DeepseekV4MTPBlock` lives behind a feature flag and overlaps in scope with `MultiTokenPredictionBlock`. Either remove or document as a research path. |
| **E6** | LOW | `deepseek_v4_block.py` is 778 lines mixing block, layer, and helper code. Split into `block.py` / `layer.py` / `_helpers.py` for review-ability. |

## F. Testing

| ID | Severity | Finding |
|---|---|---|
| **F1** | HIGH | **No checkpoint-load test**. There is no path from V4-Flash safetensors to the Primus model state_dict. All distributed runs are random-init smokes. |
| **F2** | HIGH | **No numerical-alignment test** vs HF / NeMo reference forward on a 4L toy model. Convergence claims for plan-1 phase 9/10 are runtime-shape claims, not numerical claims. |
| **F3** | MED | HashRouter unit tests check shape + determinism but not gradient flow on the gate weight (because that path is missing). |
| **F4** | MED | No `tests/configs/test_deepseek_v4_yaml.py` (deferred to P3 in plan-0; never landed). |
| **F5** | LOW | No PP-correctness test for HC — the `iteration 3` smoke does not verify per-stage K-stream consistency. |

## Summary by File (top hits)

| File | CRIT/HIGH issues |
|---|---|
| `primus/backends/megatron/core/transformer/deepseek_v4_attention.py` | A1, A2, A3, A4, B1, C4 |
| `primus/backends/megatron/core/transformer/clamped_swiglu.py` | A6, A7 |
| `primus/backends/megatron/core/transformer/moe/v4_hash_router.py` | A5, F3 |
| `primus/backends/megatron/core/transformer/moe/v4_moe.py` | B4, D5, D6 |
| `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py` | B2, B3, C1, C2, C3, E1, E2 |
| `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py` | B6, C4, D1, D2, D3 |
| `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py` | C2, B5 |
| `primus/backends/megatron/core/transformer/dual_rope.py` | E3 |
| `primus/backends/megatron/core/transformer/hca_attention.py` | A8 |
| `primus/configs/models/megatron/deepseek_v4_flash.yaml` | A10, D4 |

The next file (`01-roadmap.md`) lays out how plan-2 sequences the fixes
across phases.
