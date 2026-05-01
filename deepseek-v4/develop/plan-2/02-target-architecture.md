# 02 — Plan-2 Target Architecture

> Target module layout for the rewritten DeepSeek-V4 in Primus, expressed
> in terms of Megatron's standard `spec + config + provider + submodule +
> build_module` pattern, with explicit reuse of `MLASelfAttention`,
> `MoELayer`, `TransformerLayer`, `TransformerBlock`,
> `MultiTokenPredictionBlock`, and `(Yarn)RotaryEmbedding`.

## 1. Module Map (V4 reference → Primus target)

| V4 reference (HF / inference) | Primus target | Megatron parent | Plan-2 phase |
|---|---|---|---|
| `DeepseekV4Config` | `DeepSeekV4TransformerConfig` (existing) | `MLATransformerConfig` | P12 (touched in P14 / P18 for new fields) |
| `DeepseekV4Model` (top level) | `DeepseekV4Model` | `LanguageModule` | P15 (small refactor) |
| `DeepseekV4DecoderLayer` (HC + attn + ffn) | `DeepseekV4HybridLayer` | `TransformerLayer` | P15 |
| (decoder list + final norm) | `DeepseekV4TransformerBlock` | `TransformerBlock` | P15 |
| `DeepseekV4Attention` (MLA + sink + grouped O) | `DeepseekV4Attention` | `MLASelfAttention` | P13 |
| `DeepseekV4Compressor` | `DeepseekV4Compressor` (no change to internal math) | `nn.Module` (spec submodule of attention) | P13 |
| `DeepseekV4Indexer` | `DeepseekV4Indexer` (no change to internal math) | `nn.Module` (spec submodule of attention) | P13 |
| `DeepseekV4Gate` (score + tid2eid) | `DeepseekV4HashRouter` and `DeepseekV4LearnedRouter` | `TopKRouter` (Megatron) | P14 |
| `DeepseekV4MoE` | `DeepseekV4MoE` | `MoELayer` | P14 |
| `DeepseekV4Expert` (clamp gate / clamp up / silu * mul / w2) | `clamped_swiglu_pre_mul` activation function + standard `MLP` / `GroupedMLP` | callable in `MLPSubmodules.activation_func` | P14 |
| `DeepseekV4HyperConnection` | `HyperMixer` (existing, retained) | `nn.Module` (spec submodule) | P15 |
| `DeepseekV4HyperHead` | `HyperHead` (existing, retained) | `nn.Module` (spec submodule, only on last PP stage) | P15 |
| Rotary (interleaved + dual base + YaRN) | `DeepseekV4DualRoPE` | wraps `RotaryEmbedding(rotary_interleaved=True)` + `YarnRotaryEmbedding` | P15 |
| MTP layers + heads | (none in Primus) | `MultiTokenPredictionBlock` + `MultiTokenPredictionLayer` | P16 |
| State-dict adapter | `DeepSeekV4StateDictAdapter` | new | P17 |

## 2. Faithful Attention (P13)

```python
@dataclass
class DeepseekV4AttentionSubmodules(MLASelfAttentionSubmodules):
    # Inherits MLA submodules (q_layernorm, kv_layernorm, linear_q_*,
    # linear_kv_*, core_attention, linear_proj).
    # V4 extras:
    linear_o_a: Optional[ModuleSpec] = None        # grouped low-rank O down
    linear_o_b: Optional[ModuleSpec] = None        # grouped low-rank O up
    attn_sink: Optional[ModuleSpec] = None         # AttentionSink module
    compressor: Optional[ModuleSpec] = None        # only when compress_ratio > 0
    indexer: Optional[ModuleSpec] = None           # only when compress_ratio == 4

class DeepseekV4Attention(MLASelfAttention):
    # Reuses MLA's: low-rank Q (wq_a / wq_b), single-latent KV (wkv),
    # q_layernorm + kv_layernorm, partial RoPE, core_attention.
    # Adds:
    # - per-head q_rms after wq_b (extra RMSNorm on heads dim)
    # - learnable attn_sink (per-head scalar) inserted into core_attention
    # - grouped low-rank O projection: replaces linear_proj
    # - optional Compressor / Indexer branch driven by compress_ratio
```

Key requirements:

1. **K = V = kv** (single latent). The HF reference uses `wkv` once; in
   Megatron MLA terms this means `linear_kv_up_proj` has output dim
   `head_dim` (not `2 * head_dim`) — V4 attention overrides the KV branch
   accordingly.
2. **q_norm per head**: applied as an additional RMSNorm with shape
   `[head_dim]` after `linear_q_up_proj`, BEFORE partial RoPE. Provider
   selects `TENorm` / Apex / torch RMSNorm.
3. **kv_norm**: standard MLA `kv_layernorm` is sufficient — same as HF
   reference's `kv_norm`.
4. **Attention sink**: implemented as an extra fp32 column added before
   softmax inside `core_attention`. Where TE supports `attention_sink`
   natively, the spec selects `TEDotProductAttention` with a sink
   parameter; otherwise, fall back to a thin `AttentionSink` wrapper that
   computes softmax-with-sink in fp32.
5. **Grouped low-rank O**: the spec exposes `linear_o_a` /  `linear_o_b`
   instead of MLA's `linear_proj`. The forward in `DeepseekV4Attention`
   reshapes `[B, S, n_heads, head_dim]` to
   `[B, S, n_groups, n_heads/n_groups * head_dim]`, applies `wo_a`
   per-group via einsum, then `wo_b` to fold back to `hidden`.
6. **Compressor branch** (compress_ratio > 0):
   - Compressor is a spec submodule built when the layer's
     `compress_ratio != 0`.
   - For `compress_ratio == 4` the Indexer is also built and the attention
     forward gathers per-query top-K and joins them with the SWA logits
     before softmax.
   - For `compress_ratio == 128` (no Indexer) the compressed pool is
     concatenated to SWA KV with the compressed-causal mask.
7. **TP shape contract**: `linear_q_up_proj`, `linear_o_b` are
   column-parallel; `linear_kv_up_proj`, `linear_o_a` are
   column-parallel; `linear_proj` (replaced by `linear_o_b` here) is
   row-parallel into `hidden`.

## 3. Faithful Activation (P14)

V4 expects:

```python
gate = clamp(w1(x), max=swiglu_limit)
up   = clamp(w3(x), min=-swiglu_limit, max=swiglu_limit)
y    = silu(gate) * up
```

- New activation function `clamped_swiglu_pre_mul(gate, up, *, alpha)` that
  matches the **pre-multiplication** clamp semantics. Returns `[..., I]`.
- Used as `MLPSubmodules.activation_func`, with the gate/up split happening
  before activation (Megatron's standard split-then-activate path with
  `gated_linear_unit=True` and the activation function called on
  `(gate, up)` rather than on the fused tensor).
- For the grouped expert path, the activation function lives inside
  `TEGroupedMLP` / `GroupedMLP` and is selected by the V4 provider:
  `provider.activation_func(swiglu_limit=α)` returns a callable that does
  the V4 pre-mul clamp. When the backend cannot inject a custom callable
  (legacy / `bias_swiglu_impl` paths), the provider downgrades to local
  experts with explicit warning.

## 4. Faithful Routers + MoE (P14)

```python
class DeepseekV4LearnedRouter(TopKRouter):
    """V4 learned router. Adds sqrtsoftplus / sigmoid / softmax score
    function and noaux_tc bias. Inherits TopKRouter for the dispatcher
    contract (probs, routing_map, etc.).
    """
    def routing(self, hidden):
        logits = self.gate_linear(hidden)
        scores = _v4_score(logits, self.score_function)  # sqrtsoftplus default
        sel_score = scores + self.expert_bias if self.expert_bias is not None else scores
        topk_idx = sel_score.topk(self.topk).indices
        topk_vals = scores.gather(-1, topk_idx) * self.route_scale
        if self.norm_topk_prob and self.score_function != "softmax":
            topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True).clamp_min(1e-20)
        return _build_probs_and_routing_map(topk_idx, topk_vals, self.num_experts)

class DeepseekV4HashRouter(TopKRouter):
    """V4 hash router. Same gate weight + score function as the learned
    router, but expert ids come from a static tid2eid lookup keyed on
    token id (NOT topk on score).
    """
    tid2eid: torch.LongTensor  # [vocab_size, topk] (loaded from checkpoint)

    def routing(self, hidden, *, token_ids):
        logits = self.gate_linear(hidden)
        scores = _v4_score(logits, self.score_function)
        idx = self.tid2eid[token_ids.flatten()]                # static
        weights = scores.gather(-1, idx) * self.route_scale     # learned weights
        if self.norm_topk_prob and self.score_function != "softmax":
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        return _build_probs_and_routing_map(idx, weights, self.num_experts)

class DeepseekV4MoE(MoELayer):
    # Inherits load-balance loss, z-loss, dispatcher lifecycle.
    # Overrides router selection: layer_idx < num_hash_layers -> HashRouter.
    # Threading token_ids: see C2 fix in P15 (use TransformerLayer kwarg, not
    # decoder._v4_token_ids).
```

The submodule spec becomes:

```python
DeepseekV4MoESubmodules(
    learned_router = ModuleSpec(module=DeepseekV4LearnedRouter, params={...}),
    hash_router    = ModuleSpec(module=DeepseekV4HashRouter,    params={...}),
    token_dispatcher = ModuleSpec(module=MoEAlltoAllTokenDispatcher),
    experts        = provider.v4_grouped_mlp_spec(num_experts=N, swiglu_limit=α),
    shared_experts = ModuleSpec(module=SharedExpertMLP, submodules=...),
)
```

## 5. Faithful Layer + Block + HC × PP (P15)

```python
@dataclass
class DeepseekV4HybridLayerSubmodules(TransformerLayerSubmodules):
    # All inherited fields (input_layernorm, self_attention, self_attn_bda,
    # pre_mlp_layernorm, mlp, mlp_bda) keep their roles.
    # V4 extras:
    attn_hc: Optional[ModuleSpec] = None   # HyperMixer for attention site
    ffn_hc:  Optional[ModuleSpec] = None   # HyperMixer for ffn site

class DeepseekV4HybridLayer(TransformerLayer):
    # Replaces _bias_dropout_add / standard residual with HC mixing:
    #   pre, post, comb = attn_hc.compute_weights(x_streams)
    #   collapsed = HyperMixer.collapse(x_streams, pre)
    #   attn_out  = self_attention(input_layernorm(collapsed), ...)
    #   x_streams = HyperMixer.expand(x_streams, attn_out, post, comb)
    # Same for FFN sub-block.
    #
    # The token_ids needed by HashRouter is threaded as a forward kwarg,
    # NOT stashed as an attribute.
```

Block:

```python
class DeepseekV4TransformerBlock(TransformerBlock):
    # Inherits PP local-layer construction, recompute, set_input_tensor,
    # final_layernorm placement.
    #
    # Overrides:
    #   - input/output transform: lift [S, B, D] -> [S, B, K, D] only on
    #     the FIRST PP stage; subsequent stages receive [S, B, K, D] from
    #     P2P send/recv directly.
    #   - HyperHead collapse only on the LAST PP stage (where final
    #     layernorm and output projection sit).
    #
    # PP send/recv carries [S, B, K, D]. Either:
    #   (a) extend pipeline_parallel send_forward to handle 4D, or
    #   (b) flatten K into the seq axis at stage boundary
    #       ([S, B, K, D] -> [S*K, B, D] -> [S, B, K, D]).
    # Plan-2 lands path (b) first (no third_party/ change), with (a) as a
    # follow-up.
```

The K-stream lift/lower lives in `deepseek_v4_block.py` as helpers; it
preserves Megatron's PP machinery.

## 6. MTP Integration (P16)

```python
def get_v4_mtp_block_spec(config, *, transformer_layer_spec, vp_stage):
    """Returns a ModuleSpec(module=MultiTokenPredictionBlock, ...)
    using the V4 hybrid layer spec for the inner mtp_model_layer."""
    ...
```

- `DeepseekV4Model.__init__` builds `self.mtp` via Megatron's
  `MultiTokenPredictionBlock` when `config.mtp_num_layers > 0`.
- The MTP layer's `mtp_model_layer` is a `DeepseekV4HybridLayer` spec,
  reusing the same attention / MoE / HC submodules as the main decoder.
- The MTP head's HC `HyperHead` is a separate per-MTP-layer instance, as
  specified in the V4 reference.
- Loss path: `process_mtp_loss` from upstream Megatron handles the MTP
  loss; remove `DeepseekV4MTPBlock` (or move under `research/`).

## 7. State-Dict Adapter (P17)

`primus/backends/megatron/core/models/deepseek_v4/state_dict_adapter.py`:

```python
class DeepSeekV4StateDictAdapter:
    """Map released V4-Flash safetensors keys -> Primus state_dict keys.

    Reference key layout: deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py
    Reference target layout: NeMo's nemo_automodel/components/models/deepseek_v4/state_dict_adapter.py
    """

    HF_TO_PRIMUS = {
        "embed.weight": "embedding.word_embeddings.weight",
        "norm.weight":  "decoder.final_layernorm.weight",
        "head.weight":  "output_layer.weight",
        # per-layer prefix `layers.<i>.`
        "layers.{i}.attn.wq_a.weight":  "decoder.layers.{i}.self_attention.linear_q_down_proj.weight",
        "layers.{i}.attn.q_norm.weight": "decoder.layers.{i}.self_attention.q_layernorm.weight",
        "layers.{i}.attn.wq_b.weight":  "decoder.layers.{i}.self_attention.linear_q_up_proj.weight",
        "layers.{i}.attn.wkv.weight":   "decoder.layers.{i}.self_attention.linear_kv_up_proj.weight",
        "layers.{i}.attn.kv_norm.weight": "decoder.layers.{i}.self_attention.kv_layernorm.weight",
        "layers.{i}.attn.wo_a.weight":  "decoder.layers.{i}.self_attention.linear_o_a.weight",
        "layers.{i}.attn.wo_b.weight":  "decoder.layers.{i}.self_attention.linear_o_b.weight",
        "layers.{i}.attn.attn_sink":    "decoder.layers.{i}.self_attention.attn_sink.sink",
        "layers.{i}.attn_norm.weight":  "decoder.layers.{i}.input_layernorm.weight",
        "layers.{i}.ffn_norm.weight":   "decoder.layers.{i}.pre_mlp_layernorm.weight",
        # HC params
        "layers.{i}.hc_attn_fn":    "decoder.layers.{i}.attn_hc.fn.weight",
        "layers.{i}.hc_attn_base":  "decoder.layers.{i}.attn_hc.base",
        "layers.{i}.hc_attn_scale": "decoder.layers.{i}.attn_hc.scale",
        "layers.{i}.hc_ffn_fn":     "decoder.layers.{i}.ffn_hc.fn.weight",
        "layers.{i}.hc_ffn_base":   "decoder.layers.{i}.ffn_hc.base",
        "layers.{i}.hc_ffn_scale":  "decoder.layers.{i}.ffn_hc.scale",
        # MoE
        "layers.{i}.ffn.gate.weight":   "decoder.layers.{i}.mlp.router.gate_linear.weight",
        "layers.{i}.ffn.gate.bias":     "decoder.layers.{i}.mlp.router.expert_bias",
        "layers.{i}.ffn.gate.tid2eid":  "decoder.layers.{i}.mlp.router.tid2eid",
        "layers.{i}.ffn.experts.{e}.w1.weight": "decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc1.weight",  # gate slice
        "layers.{i}.ffn.experts.{e}.w2.weight": "decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc2.weight",
        "layers.{i}.ffn.experts.{e}.w3.weight": "decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc1.weight",  # up slice
        "layers.{i}.ffn.shared_experts.w1.weight": "decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight",  # gate
        "layers.{i}.ffn.shared_experts.w2.weight": "decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight",
        "layers.{i}.ffn.shared_experts.w3.weight": "decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight",  # up
    }

    def to_primus(self, hf_state_dict): ...
    def to_hf(self, primus_state_dict): ...
```

`w1` (gate) and `w3` (up) are concatenated into Megatron's
`linear_fc1.weight` along the output axis (gate slot first, up slot
second). The adapter is responsible for that concat / split.

A CPU-only smoke (`scripts/load_v4_flash_check.py`) loads the BF16
safetensors, runs a 64-token prompt forward, and compares token-0 logits
against the HF reference forward.

## 8. Spec Tree (target)

```python
def get_deepseek_v4_runtime_decoder_spec(config, *, vp_stage):
    provider = DeepSeekV4SpecProvider(config=config)
    layer_specs = []
    for layer_idx in stage_layer_indices:
        compress_ratio = compress_ratios[layer_idx]
        attention = build_v4_attention_spec(config, provider,
                                            compress_ratio=compress_ratio)
        mlp = build_v4_mlp_spec(config, provider, layer_idx=layer_idx)
        hc_mult = config.hc_mult
        layer_specs.append(ModuleSpec(
            module=DeepseekV4HybridLayer,
            submodules=DeepseekV4HybridLayerSubmodules(
                input_layernorm   = provider.layer_norm(rms_norm=True),
                self_attention    = attention,
                self_attn_bda     = identity_or_v4_hc_bda,    # HC-aware BDA
                pre_mlp_layernorm = provider.layer_norm(rms_norm=True),
                mlp               = mlp,
                mlp_bda           = identity_or_v4_hc_bda,
                attn_hc           = ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
                ffn_hc            = ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
            ),
            params={"layer_idx": layer_idx, "compress_ratio": compress_ratio},
        ))
    block_submodules = DeepseekV4TransformerBlockSubmodules(
        layer_specs   = layer_specs,
        hyper_head    = ModuleSpec(module=HyperHead) if hc_mult > 1 else None,
        final_layernorm = provider.layer_norm(rms_norm=True),
        mtp_block_spec = get_v4_mtp_block_spec(config, ...) if config.mtp_num_layers > 0 else None,
    )
    return ModuleSpec(module=DeepseekV4TransformerBlock,
                      submodules=block_submodules)
```

The spec is a tree with NO surprise constructors — every replaceable piece
is a `ModuleSpec`.

## 9. Provider Surface (target)

```python
class DeepSeekV4SpecProvider(PrimusTurboSpecProvider):
    def v4_norm(self, *, for_qk: bool=False) -> ModuleSpec: ...
    def v4_attention_core(self) -> ModuleSpec: ...           # TEDotProductAttention/PrimusTurbo
    def v4_attention_sink(self) -> ModuleSpec: ...           # TE/local
    def v4_q_layernorm(self) -> ModuleSpec: ...
    def v4_kv_layernorm(self) -> ModuleSpec: ...
    def v4_grouped_mlp_spec(self, *, num_experts, swiglu_limit) -> ModuleSpec: ...
    def v4_shared_expert_spec(self, *, intermediate_size, swiglu_limit) -> ModuleSpec: ...
    def v4_compressor_spec(self, *, compress_ratio) -> ModuleSpec: ...
    def v4_indexer_spec(self) -> ModuleSpec: ...
    def v4_router_spec(self, *, learned: bool) -> ModuleSpec: ...
    def v4_token_dispatcher_spec(self, *, dispatcher_type) -> ModuleSpec: ...
    def v4_activation_func(self, *, swiglu_limit) -> Callable: ...
```

All of these are pure factories; the V4 model itself never builds
modules outside of `build_module(spec, ...)`.

## 10. Where reuse is expected

| Module | Reuse from upstream |
|---|---|
| `MLASelfAttention` | `megatron.core.transformer.multi_latent_attention` |
| `MLATransformerConfig` | same |
| `TransformerLayer`, `TransformerLayerSubmodules` | `megatron.core.transformer.transformer_layer` |
| `TransformerBlock`, `TransformerBlockSubmodules`, `get_num_layers_to_build` | `megatron.core.transformer.transformer_block` |
| `TopKRouter`, `MoELayer`, `SharedExpertMLP` | `megatron.core.transformer.moe.*` |
| `MoEAllGatherTokenDispatcher`, `MoEAlltoAllTokenDispatcher`, `MoEFlexTokenDispatcher` | same |
| `MultiTokenPredictionBlock`, `MultiTokenPredictionLayer` | `megatron.core.transformer.multi_token_prediction` |
| `RotaryEmbedding(rotary_interleaved=True)`, `YarnRotaryEmbedding`, `apply_rotary_pos_emb` | `megatron.core.models.common.embeddings` |
| `MLP`, `MLPSubmodules`, `GroupedMLP`, `TEGroupedMLP`, `SharedExpertMLP` | `megatron.core.transformer.{mlp,moe.experts,moe.shared_experts}` |
| `LanguageModule`, `LanguageModelEmbedding` | `megatron.core.models.common.*` |
| `ColumnParallelLinear` / `RowParallelLinear` (and TE variants via provider) | `megatron.core.tensor_parallel.layers` (+ TE) |

The next file (`03-phase-details.md`) breaks the work down into per-phase
tasks with exit criteria.
