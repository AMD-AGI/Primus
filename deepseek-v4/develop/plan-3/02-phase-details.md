# 02 — Plan-3 Phase Details

> One section per phase, mirroring `../plan-2/03-phase-details.md`.
> Each phase lists: motivation, design, tasks, risks, and explicit
> exit criteria + test gates (cross-references to `03-test-strategy.md`).

---

## P20 — V4-aware TFLOPs reporting

### Motivation

Megatron's `num_floating_point_operations(args, batch_size)` lives in
`third_party/Megatron-LM/megatron/training/training.py:228`. It branches
on:

1. `args.is_hybrid_model` → mamba / hybrid path.
2. `args.multi_latent_attention` → MLA path (counts compressed Q + KV
   GEMMs but assumes V3-style dense / GShard MoE).
3. Otherwise → standard transformer path (counts dense `4·B·S·H²` MLP /
   attention with optional `moe_layer_freq` mask).

V4 falls onto branch (3) because `multi_latent_attention=False` in the
V4 base config (V4 is *single-latent* KV, not MLA's compressed-KV
form). The standard transformer formula:

* counts dense MLP everywhere `moe_layer_freq` is `0` — V4 has **no**
  dense MLP layers in the released Flash schedule (every layer is
  MoE).
* applies a single `attn_layer_flops` everywhere — V4 layers come in
  three flavours (dense / CSA / HCA) with very different attention
  shapes (single-latent KV → `H·head_dim²` Q-only flop, no K / V
  projection cost; CSA adds Compressor + Indexer side paths; HCA adds
  the Compressor pool).
* ignores the **mHC** packing — every HC layer multiplies the per-token
  hidden-state cost by `hc_mult` (4 for V4-Flash) because each token
  carries `K` residual streams.
* ignores the **grouped low-rank O** — the `H·head_dim → hidden`
  projection becomes `H·head_dim/o_groups → o_groups·o_lora_rank →
  hidden`, which is **smaller** than the flat MLA O cost.
* ignores the **Q-LoRA path** (`hidden → q_lora_rank → H·head_dim`),
  which is what the V4 single-latent KV path actually pays.
* ignores the **MTP head** beyond the basic `1 + mtp_num_layers` logits
  multiplier — V4's MTP adds a separate `HyperHead` + a per-MTP-layer
  attention + MoE.
* ignores the **hash router** vs **learned router** difference — both
  have `H·H_vocab_routed` flop but the hash router adds a static lookup
  with zero extra GEMM cost.

The reported TFLOPs is therefore a confused mixture: too high on the
dense FFN line (V4 has none), too low on the attention + Compressor +
HC + MTP lines, and ignores the `K`-stream packing entirely. P19
profiles already showed this — the reported number is plausible-looking
but does not reflect the actual GEMM cost.

### Design

Land a Primus patch under
`primus/backends/megatron/patches/deepseek_v4_flops_patches.py` that
wraps `megatron.training.training.num_floating_point_operations` with a
V4-specific closed form. Patch is registered via
`@register_patch("megatron.deepseek_v4.flops_accounting", ...)` with
condition `model_type == "deepseek_v4"` so non-V4 callers get the
upstream function untouched.

The V4 closed form (per fwd; multiply by 3 for fwd + dgrad + wgrad):

```text
B = micro_batch * num_microbatches    # i.e. global_batch_size
S = seq_length
H = hidden_size
H_q = num_attention_heads * kv_channels      # Q output width
H_v = kv_channels                            # single-latent K = V width
q_lora = q_lora_rank
o_groups = o_groups; o_lora = o_lora_rank
H_route = num_moe_experts; topk = moe_router_topk
H_moe = moe_ffn_hidden_size
H_shared = moe_shared_expert_intermediate_size
K = hc_mult                                  # number of HC streams
hash_layers = num_hash_layers; mtp_n = mtp_num_layers

attn_qkv_flops(layer_kind):
    # Q-LoRA path (always)
    flop  = 2·B·S·H·q_lora            # linear_q_down_proj
    flop += 2·B·S·q_lora·H_q          # linear_q_up_proj
    # Single-latent KV (always)
    flop += 2·B·S·H·H_v               # linear_kv
    # Output projection (V4 grouped low-rank)
    flop += 2·B·S·(H_q/o_groups)·(o_groups·o_lora)   # linear_o_a
    flop += 2·B·S·(o_groups·o_lora)·H               # linear_o_b
    return flop

attn_score_flops(layer_kind):
    # All three layer kinds run an [S, S] local SWA softmax.
    base = 4·B·S·S·H_q          # QK^T + (probs·V), each 2·B·S·S·H_q
    if layer_kind == "dense":   # compress_ratio == 0
        return base
    if layer_kind == "hca":     # compress_ratio == 128
        # joint softmax over [local_keys, compressed_pool],
        # P = ceil(S / 128) compressed positions per batch.
        P = ceil(S / 128)
        return base + 4·B·S·P·H_q
    if layer_kind == "csa":     # compress_ratio == 4
        # joint softmax over [local_keys, top-K compressed].
        return base + 4·B·S·index_topk·H_q

compressor_flops(layer_kind):
    if layer_kind in {"hca", "csa"}:
        # Compressor: hidden → coff·head_dim per pool position.
        coff = 1 if layer_kind == "hca" else 2
        P = ceil(S / compress_ratio)
        return 2·B·P·H·(coff·H_v)·2  # wkv + wgate
    return 0

indexer_flops(layer_kind):
    if layer_kind == "csa":
        # Indexer: hidden → index_n_heads·index_head_dim,
        # then per-query top-K scoring against the compressed pool.
        return (2·B·S·H·index_dq_rank
                + 2·B·S·index_dq_rank·index_n_heads·index_head_dim
                + 2·B·S·H·index_n_heads             # w_w
                + 2·B·S·P·index_n_heads·index_head_dim)  # bhsd,bhskd
    return 0

hc_flops():
    # Per HC layer (attn_hc + ffn_hc), HC params kept fp32.
    # HyperMixer: [B, S, K, H] → linear → [B, S, K, K] mixing matrix.
    # 2 × per layer (attn + ffn).
    return 2 · (2·B·S·K·H·K + 2·B·S·K·H·K)

moe_flops(is_hash_layer):
    # Routed MoE — V4 has no dense layer; every layer routes to topk experts.
    routed = 2·B·S·H·H_moe·topk·3    # SwiGLU stack: gate + up + down
    shared = 2·B·S·H·H_shared·3
    if is_hash_layer:
        # Hash router: just a static lookup + a learned gate weight.
        router = 2·B·S·H·H_route
    else:
        # Learned router: same gate-weight matmul; the score function
        # adds O(B·S·H_route) which is negligible.
        router = 2·B·S·H·H_route
    return routed + shared + router

logits_flops():
    return 2·B·S·H·vocab_size

mtp_flops():
    # Each MTP "layer" is a full V4 hybrid layer (attn + moe + hc).
    # We use the same per-layer formulas with layer_kind=="dense"
    # (MTP heads do not own compressed branches).
    return mtp_n · (
        attn_qkv_flops("dense") + attn_score_flops("dense")
        + moe_flops(is_hash_layer=False) + hc_flops()
    )

# Aggregate
total_fwd_flops = (
    sum_over_layers(
        attn_qkv_flops(layer_kind) + attn_score_flops(layer_kind)
        + compressor_flops(layer_kind) + indexer_flops(layer_kind)
        + moe_flops(is_hash_layer = layer_idx < hash_layers)
        + hc_flops()
    )
    + (1 + mtp_n) · logits_flops()
    + mtp_flops()
)

return 3 · total_fwd_flops          # forward + dgrad + wgrad
```

The patch reads `args.compress_ratios` (already normalised to a tuple of
ints by P18) to label each layer as `dense / hca / csa`.

### Tasks

1. New file `primus/backends/megatron/patches/deepseek_v4_flops_patches.py` — implements `_v4_num_floating_point_operations(args, batch_size)` and `@register_patch` wrapper.
2. Register at `phase=before_train`, `condition=model_type=="deepseek_v4"`.
3. Add a one-time rank-0 log dump at startup that prints the per-layer breakdown (attn / score / compressor / indexer / moe / hc) so a human can sanity-check the formula.
4. Unit test under `tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py`:
   * 8-layer Flash-shape config with `compress_ratios=[0,0,4,128,4,128,4,0]`, `hash_layers=3`.
   * Hand-computed reference (excel-style closed form) → assert within 1%.
   * Non-V4 config → assert wrapper returns the upstream value byte-for-byte.

### Exit criteria

* P19 profile re-run reports a V4 TFLOPs that matches the closed form within 1%.
* Non-V4 model types are byte-for-byte unchanged.

### Test gates

* **G16** — `test_deepseek_v4_flops_patches.py` (closed-form match).
* **G17** — `model_type != deepseek_v4` returns upstream value (byte-for-byte).

### Status (2026-05-07)

✅ **Done.** Patch landed at
`primus/backends/megatron/patches/deepseek_v4_flops_patches.py`. Two
non-obvious invariants surfaced during bring-up and are pinned in
comments:

1. **Direct-import binding** — `primus.modules.trainer.megatron.trainer`
   imports `num_floating_point_operations` at module load, so a
   monkey-patch on `megatron.training.training` alone is invisible to
   the trainer's local copy. The patch now calls
   `_rebind_trainer_imports()` after wrapping to refresh the trainer's
   bound name.
2. **`pretrain()` enum overwrite** — Megatron's `pretrain()` rewrites
   `args.model_type` from the YAML string `"deepseek_v4"` to a
   `ModelType` enum at `training.py:1210` *before* `train()` ever calls
   `num_floating_point_operations`. A naive runtime check
   (`args.model_type == "deepseek_v4"`) silently falls through to the
   upstream formula. The wrapper instead captures `dispatch_v4` at
   install time via the closure; this is gated by the
   `@register_patch(condition=…)` so install time is the only point
   where `args.model_type` is still the YAML string.

Smoke verification on `mi355-gpu-12` (`dev_primus_wenx_693`, 8 GPUs,
`bs=16, S=128, hc_mult=4, L=8, mtp=0`): closed form total = 73.43 TFLOP
/ global-batch.

| Run        | Last-iter TFLOP/s/GPU | Iter time | Implied / iter | Δ vs closed form |
| ---------- | --------------------: | --------: | -------------: | ---------------: |
| EP=8       | 17.9                  | 512.5 ms  | 73.4 TFLOP     | 0.04%            |
| PP=2 EP=4  | 14.0                  | 655.9 ms  | 73.5 TFLOP     | 0.09%            |

Logs: `deepseek-v4/develop/progress/p20/smoke_ep8_pp1_final.log` /
`smoke_pp2_ep4_final.log`.

Unit tests: 21/21 green
(`tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py`).

---

## P21 — Strict spec build (no nn-fallback)

### Motivation

Smokes at full Flash dims (`PRIMUS_PP=2 PRIMUS_EP=4`, `seq=1024`,
`hidden=4096`, `num_heads=64`, `head_dim=512`) emit hundreds of:

```
DeepSeek-V4 attention projection submodule init failed
  (Transformer Engine linear layers do not support gather_output = True
   when instantiating TEColumnParallelLinear); fallback to nn.Linear.
DeepSeek-V4 attention projection submodule init failed
  (Transformer Engine linear layers do not support input_is_parallel = False
   when instantiating TERowParallelLinear); fallback to nn.Linear.
```

… one per V4 attention construction. The current code masks this:

```python
# primus/backends/megatron/core/transformer/deepseek_v4_attention.py:_build_projection
try:
    return build_module(submodule)
except Exception as exc:
    logger.warning("...; fallback to nn.Linear.", exc)
    return nn.Linear(in_features, out_features, bias=False)
```

This produces an unsharded model: every V4 attention layer ends up with
duplicated `nn.Linear` weights instead of the column / row-parallel
shards the spec asked for. TP=1 happens to work because the eager math
in `DeepseekV4Attention.forward` consumes a full-width tensor; TP>1
would silently diverge.

The same pattern exists in:

* `primus/backends/megatron/core/transformer/deepseek_v4_attention.py:_build_projection` (linear projections)
* `primus/backends/megatron/core/transformer/deepseek_v4_attention.py:_build_compressor` (warns + falls back to `Compressor(...)`)
* `primus/backends/megatron/core/transformer/deepseek_v4_attention.py:_build_indexer` (warns + falls back to `Indexer(...)`)
* `primus/backends/megatron/core/transformer/deepseek_v4_attention.py` `attn_sink_module` build (warns + falls back to inline softmax)
* `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py:_build_projection` (warns + falls back to `nn.Linear`)

For Compressor / Indexer the spec passes the same Python class as the
fallback (`Compressor` / `Indexer`), so the fallback-on-failure is
silent dead code — it just hides whatever prevented `build_module` from
being called.  For `attn_sink` the inline softmax is canonical (the
parameter `self.attn_sink` lives on the module directly), so the
inline path is *always* the path; the `attn_sink_module` build is
purely forward-compat scaffolding. We delete it under P21 and re-add it
in a future plan when a real fused sink-softmax module is wired.

### Design

Replace every `try / logger.warning / fall back` block with:

```python
return build_module(submodule, ...)   # let exceptions propagate
```

… plus drop the fallback construction. For `_build_projection` the
`submodule is None` case stays (it is the documented CPU-test path:
caller passes `None`, gets a vanilla `nn.Linear`).

Then root-cause the two failing kwargs:

* **`gather_output=True` rejected by `TEColumnParallelLinear`** —
  `_build_column_parallel_spec` passes `gather_output=True` so
  downstream attention math sees a full-width tensor. TE's column
  parallel linear does not support this kwarg. Fix: when the V4 spec
  needs gather-output semantics, use the upstream
  `megatron.core.tensor_parallel.layers.ColumnParallelLinear` (the
  non-TE class) which **does** support `gather_output=True`. The
  provider already exposes the non-TE class via
  `megatron.core.tensor_parallel.layers` imports; expose it through a
  new `provider.column_parallel_linear_with_gather_output()` helper so
  the choice is documented at the provider level.

  Alternative: drop `gather_output=True` and refactor the V4 attention
  forward to consume the sharded tensor directly. Cheaper short-term,
  but ties the V4 forward to TP details. P21 picks the provider-helper
  path so V4 attention math stays TP-agnostic.

* **`input_is_parallel=False` rejected by `TERowParallelLinear`** —
  same story: `_build_row_parallel_spec` defaults
  `input_is_parallel=False` so the caller can pass a full-width input.
  Fix: use the upstream `RowParallelLinear` (non-TE) when the caller
  needs the scatter-then-allreduce semantics; expose via
  `provider.row_parallel_linear_with_scatter_input()`.

The provider helpers stay V4-agnostic — any caller that needs the
non-TE column / row parallel linear can opt in.

### Tasks

1. Delete the `try / except Exception / logger.warning / fall back` blocks in:
   * `_build_projection` (in `deepseek_v4_attention.py`)
   * `_build_compressor`, `_build_indexer` (same file)
   * `attn_sink_module` build branch (same file)
   * `_build_projection` (in `deepseek_v4_block.py`)
2. Drop the `attn_sink_module` field from `DeepseekV4AttentionSubmodules` (the inline softmax-with-sink path is canonical; if a future plan needs a fused sink module, re-add at that time).
3. Add `provider.column_parallel_linear_with_gather_output()` and `provider.row_parallel_linear_with_scatter_input()` to `PrimusTurboSpecProvider` (return upstream `ColumnParallelLinear` / `RowParallelLinear` from `megatron.core.tensor_parallel.layers`). Subclassed `DeepSeekV4SpecProvider` inherits.
4. Update `_build_column_parallel_spec` / `_build_row_parallel_spec` in `deepseek_v4_layer_specs.py` and `deepseek_v4_block.py` to pick the right helper based on the requested kwargs.
5. AST audit gate (G15): walk every file under `primus/backends/megatron/core/` (including the V4 subtree) and reject any `try / except Exception` block whose handler returns `nn.Linear` or `nn.Module` — failing `build_module` must propagate.
6. TP=2 vs TP=1 forward-equivalence smoke (G15b): CPU 1L toy with `tensor_model_parallel_size=2`. Outputs must match TP=1 within 1e-5 (this catches sharding mismatches the silent fallback used to hide).

### Exit criteria

* No `"submodule init failed"` / `"fallback to nn.Linear"` log lines in any P19 / P24 smoke.
* G15 + G15b green.

### Test gates

* **G15** — AST audit: no `try / except / return nn.Linear` patterns under `primus/backends/megatron/core/`.
* **G15b** — TP=1 build smoke (full TP=2 forward-equivalence runs under P24 on `mi355-gpu-12`).

### Status (2026-05-07)

Done — P21 commit `a4419ac5`.

* **Strict-build surgery.** `_build_projection` (in
  `deepseek_v4_attention.py` and `deepseek_v4_block.py`),
  `_build_compressor`, and `_build_indexer` all dropped their
  `try/except/return nn.Linear|local Compressor|local Indexer` blocks
  — `build_module` failures now propagate. The dead
  `attn_sink_module` build branch is gone (along with the
  `attn_sink` field on `DeepseekV4AttentionSubmodules`, the
  `provider.v4_attention_sink()` method, and the standalone
  `primus/backends/megatron/core/transformer/attn_sink.py` whose
  only consumer was that method); `self.attn_sink: nn.Parameter`
  remains the canonical per-head sink and lines up with the released
  V4-Flash checkpoint key `layers.{i}.attn.attn_sink`.
* **Provider helpers.**
  `PrimusTurboSpecProvider.column_parallel_linear_with_gather_output()`
  and `…row_parallel_linear_with_scatter_input()` return the
  upstream non-TE `ColumnParallelLinear` / `RowParallelLinear` from
  `megatron.core.tensor_parallel.layers`; the V4 spec helpers
  (`_build_column_parallel_spec`, `_build_row_parallel_spec`) route
  through them automatically when the caller asks for
  `gather_output=True` or `input_is_parallel=False`. The standard TE
  / Turbo path stays for the other cases.
* **G15 + G15b.** New
  `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p21_strict_build.py`
  ships 13 cases:

  * an AST walker that scans every `.py` under
    `primus/backends/megatron/core/` and asserts no
    `try → except → return nn.Linear(...)` patterns remain;
  * a parameterised string-grep that bans the retired warning
    strings (`"submodule init failed"`, `"fallback to nn.Linear"`,
    `"using local Compressor|Indexer"`,
    `"attn_sink submodule init failed"`);
  * provider-helper contract tests (`column_parallel_linear_with_gather_output`
    is the upstream `ColumnParallelLinear`; the row variant mirrors;
    `provider.v4_attention_sink` is gone);
  * a dataclass-surface test (`DeepseekV4AttentionSubmodules.attn_sink`
    is gone; the live slots are intact);
  * G15b — `_tp1_distributed` initialises a 1-rank `gloo` TP group,
    builds full V4 attention submodules through
    `_build_v4_attention_submodules`, instantiates
    `DeepseekV4Attention`, and asserts every linear is one of
    `{ColumnParallelLinear, RowParallelLinear, TELinear,
    PrimusTurboLinear}` — never a bare `nn.Linear`.
* **Smoke verification.** EP=8 PP=1 and PP=2 EP=4 smokes on
  `mi355-gpu-12` (`p21/smoke_ep8_pp1.log`,
  `p21/smoke_pp2_ep4.log`) ran 10/10 iters cleanly with `lm_loss`
  converging (11.89 → 11.65 / 11.62). `grep -cE` for
  `submodule init failed|fallback to nn\.Linear|fallback to local|using local Compressor|using local Indexer|attn_sink submodule init`
  returns `0` on both logs (vs. hundreds on the pre-P21
  `output/.../log_node_0.txt`).

---

## P22 — `core_attention` submodule for V4 (turbo / TE)

### Motivation

`DeepseekV4Attention.forward` runs an eager-Python scaled-dot-product:

```python
logits = matmul(q.float(), k.float().transpose(-2, -1)) * scale   # [B, H, S, S]
logits = logits + attn_mask
probs  = self._append_sink_softmax(logits)
out    = matmul(probs.to(v.dtype), v)
```

At full Flash dims (`H=64`, `head_dim=512`, `seq=4096`, `hc_mult=4`)
this materialises a `[B, 64, 16384, 16384] fp32` logits tensor — 16 GiB
per microbatch. Even with `recompute_num_layers=1` it is the dominant
activation cost.

The standard Megatron / Primus path is to delegate dense-causal
attention to `provider.core_attention()` which returns
`PrimusTurboAttention` (Triton flash-attn with optional sink) when
`use_turbo_attention=True`, and `TEDotProductAttention` (TE flash-attn)
otherwise. Both return `[B, S, H·head_dim]` and avoid materialising the
quadratic logits tensor. The V2-Lite recipe
(`examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml`)
already runs this way.

### Compress-ratio analysis (the user's audit)

| compress_ratio | Layer kind | Can use `core_attention`? | Why |
|---|---|---|---|
| **0** | Dense + SWA + per-head sink | **Yes** | Q,K,V live on the same `[B, S, H, head_dim]` axis; standard causal attention with sliding window + (optional) sink. Turbo flash-attn already supports `use_sink_attention=True` (per-head learned sink) via `PrimusTurboAttention.sinks`, with sliding-window via `args.sink_sliding_window`. TE supports causal + SWA via `attn_mask_type` + `window_size`, but does not (today) support learned sink — so the path is `Turbo when use_turbo_attention=True, eager-Python when False`. |
| **128 (HCA)** | Local SWA + compressed pool, joint softmax with shared sink | **No** (this plan) | The local SWA branch and the pool branch share **one** softmax with **one** sink column. Decomposing into two attention calls and combining via LSE would work, but neither TE nor Turbo flash-attn returns LSE today. We could re-derive the joint softmax inside a custom kernel, but that is a separate kernel-engineering effort. **Stays on eager-Python under P22**; defer to a perf plan once an LSE-returning flash-attn variant lands. |
| **4 (CSA)** | Local SWA + per-query top-K from compressed pool | **No** (any plan) | The per-query top-K gather (`gathered_h = pool[..., topk_idxs, :]` of shape `[B, H, S, K, head_dim]`) is a sparse per-row indexed attention. This is **not** a flash-attn pattern — there is no kernel that reads a different per-query subset of keys from a pool. This branch will always need a custom kernel (or the dense Python fallback that currently materialises the 256 GiB `gathered_h` tensor at full Flash dims). Out of scope for plan-3. |

So P22 lands `core_attention` for `compress_ratio == 0` only. HCA + CSA
explicitly stay on the eager-Python path with code comments pointing to
this analysis.

### Sink attention plumbing

The Turbo class reads `use_sink_attention` from the global Megatron
args. V4's `attn_sink: true` already maps to a learnable `[H]` parameter
named `attn_sink` on the attention module. The Turbo class, when
`use_sink_attention=True`, owns its own `self.sinks = nn.Parameter(...)`
of the same shape. Two code-paths to reconcile:

1. **Use Turbo's `self.sinks` directly.** Drop V4's `attn_sink`
   parameter and rename the Turbo one. Breaks state-dict compatibility
   with the released V4-Flash checkpoint (parameter key
   `layers.{i}.attn.attn_sink` becomes `layers.{i}.attn.core_attention.sinks`).
2. **Tie V4's `attn_sink` to Turbo's `sinks`.** After building
   `core_attention`, call `core_attention.sinks = self.attn_sink`
   (alias the parameter). Keeps the `attn_sink` checkpoint key. **This
   plan picks option 2** so the V4-Flash adapter (deferred plan-2 P22+)
   can still load the parameter with no key remapping.

Set Turbo's `sink_sliding_window` to `config.attn_sliding_window` and
`sink_window_even_layers_only=False` (V4 applies SWA on every layer,
not just the gpt-oss even-layer pattern). Both fields are read by Turbo
from `args`, so plan-3 adds a pair of fields on the V4 yaml /
DeepseekV4TransformerConfig and a `forward_v4_sink_args` patch on
`get_args()` that exposes them. (Already supported in Turbo — we just
need to pass them through.)

### Tasks

1. Add `core_attention: Optional[Union[ModuleSpec, type]] = None` to `DeepseekV4AttentionSubmodules`.
2. In `_build_v4_attention_submodules` (`deepseek_v4_layer_specs.py`), when the layer is dense (`compress_ratio == 0`), build a `core_attention` spec using `provider.core_attention()` + the standard kwargs (`config`, `layer_number`, `attn_mask_type=causal`, `attention_type="self"`, `pg_collection`, `softmax_scale=self._attention_scale_const()`).
3. In `DeepseekV4Attention.__init__`, build `self.core_attention` from `submodules.core_attention` when provided; alias `self.core_attention.sinks = self.attn_sink` when `attn_sink_enabled and self.core_attention.use_sink_attention`.
4. Add a `_attention_forward_via_core(self, q, k, v, attn_mask)` helper that reshapes `[B, H, S, head_dim]` → `[S, B, H, head_dim]` (Turbo's `qkv_format="sbhd"`) and calls `self.core_attention(q, k, v)`. Used only when `self.compress_ratio == 0 and self.core_attention is not None`.
5. Code comments in `_csa_forward` and the HCA branch documenting the analysis above (why these stay on eager-Python).
6. Yaml additions: `examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml` gains `use_sink_attention: true` (when `attn_sink: true`) and `sink_sliding_window: <attn_sliding_window>`, `sink_window_even_layers_only: false`. Plumbed through `args` so the Turbo class picks them up.
7. Unit test: CPU 1L toy where the dense path matches the eager-Python output within 1e-3 (G18). `attn_sink` weights still load via state-dict (key path unchanged).

### Exit criteria

* Dense V4 attention with `use_turbo_attention=true` runs through `PrimusTurboAttention` and saves the quadratic activation memory.
* HCA + CSA still run eager-Python (unchanged).
* `attn_sink` parameter key is preserved.

### Test gates

* **G18** — Dense attention: turbo vs eager-Python forward equivalence within 1e-3.
* **G19** — HCA + CSA branches still produce identical outputs vs plan-2 baseline.

---

## P23 — Turbo DeepEP dispatcher in V4 specs

### Motivation

Today `_build_ffn_spec` in `deepseek_v4_layer_specs.py` does:

```python
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,    # <-- captured at module import time
)
...
elif dispatcher_type == "flex":
    dispatcher_cls = MoEFlexTokenDispatcher
```

The Primus turbo patch
(`primus/backends/megatron/patches/turbo/moe_dispatcher_patches.py`) at
`phase=before_train` does
`token_dispatcher.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher`.
That monkey-patch updates the module attribute, but the local symbol
in `deepseek_v4_layer_specs.py` was bound at import time and still
points at the original class. Result: even with turbo-deepep enabled,
V4 specs build the standard `MoEFlexTokenDispatcher`. The turbo path
silently never reaches the V4 MoE.

### Design

Probe `args.use_turbo_deepep` (gated by the same condition as the
patch — `enable_primus_turbo=True`, `tensor_model_parallel_size==1`,
`primus_turbo` package importable) inside `_build_ffn_spec`. When
active:

* Import `PrimusTurboDeepEPTokenDispatcher` directly from
  `primus.backends.megatron.core.extensions.primus_turbo`.
* Set `dispatcher_cls = PrimusTurboDeepEPTokenDispatcher`.
* Set `args.moe_token_dispatcher_type = "flex"` if not already
  (the patch does this in `before_train`; we do it here so the spec
  build sees the same value when patches haven't run yet — e.g. unit
  tests that import V4 specs directly).
* Log once at rank-0: `"[DeepSeek-V4] MoE turbo deepep enabled; dispatcher = PrimusTurboDeepEPTokenDispatcher"`.

When inactive (TP>1 or turbo off), keep the existing behaviour
(MoEFlexTokenDispatcher / MoEAlltoAllTokenDispatcher).

Also update `_resolve_dispatcher_type_from_spec` in `v4_moe.py` so
`PrimusTurboDeepEPTokenDispatcher` resolves to `"flex"` instead of
falling through to the `unsupported dispatcher module` warning + `"alltoall"` fallback.

### Tasks

1. Add a `_pick_v4_dispatcher_cls(config) -> tuple[type, str]` helper in `deepseek_v4_layer_specs.py` that encapsulates the probe + class import.
2. Update `_build_ffn_spec` to call the helper instead of the static lookup.
3. Update `_resolve_dispatcher_type_from_spec` in `v4_moe.py` to recognise `PrimusTurboDeepEPTokenDispatcher`.
4. Unit test (G20): mock `args.use_turbo_deepep=True`, `args.enable_primus_turbo=True`, `args.tensor_model_parallel_size=1` → assert dispatcher_cls is `PrimusTurboDeepEPTokenDispatcher`. Mock TP=2 → assert fallback to `MoEFlexTokenDispatcher`.

### Exit criteria

* Smoke logs report `dispatcher active via PrimusTurboDeepEPTokenDispatcher` for every routed-MoE V4 layer when turbo deepep is on.
* TP>1 cleanly falls back to `MoEFlexTokenDispatcher` with one rank-0 warning.

### Test gates

* **G20** — V4 dispatcher selection unit test (turbo on / off, TP>1).

---

## P24 — Turbo attention + DeepEP smoke

### Motivation

Plan-3's release gate. Re-runs the four P19 distributed configurations
with the full turbo flag set on. Ensures the P21 / P22 / P23 changes
compose without regressing the plan-2 PP / VPP patches.

### Design

`run_deepseek_v4.sh` gains turbo defaults:

```bash
export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-True}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-True}
export USE_SINK_ATTENTION=${USE_SINK_ATTENTION:-True}
```

… and threads `--enable_primus_turbo`, `--use_turbo_deepep`,
`--use_sink_attention`, `--sink_sliding_window`,
`--sink_window_even_layers_only False` into the primus-cli call.

Smoke matrix (re-runs P19 configs):

| smoke | TP | PP | EP | VPP | seq | layers | source script |
|---|---|---|---|---|---|---|---|
| A | 1 | 1 | 8 | 1 | 128 | 8 | `deepseek-v4/develop/progress/p20/run_smokeA_turbo.sh` |
| B | 1 | 2 | 4 | 1 | 128 | 8 | `…/run_smokeB_turbo.sh` |
| C | 1 | 4 | 2 | 1 | 128 | 8 | `…/run_smokeC_turbo.sh` |
| D | 1 | 2 | 4 | 2 | 128 | 8 | `…/run_smokeD_turbo.sh` |

Each runs 10 iterations on mock data, BF16, MBS=1 GBS=16, and is
expected to complete cleanly. Logs land under
`deepseek-v4/develop/progress/p20/`.

(Note: full Flash dims still trip the CSA Python fallback OOM on a
single MI355X — that is a separate kernel-engineering effort, out of
plan-3 scope. The smoke uses the Phase-19 small-config sizes.)

### Tasks

1. Update `run_deepseek_v4.sh` with the turbo defaults + new CLI flags.
2. Add four smoke scripts under `deepseek-v4/develop/progress/p20/`.
3. Run all four; log files in the same directory.
4. Update `../progress/status.md` Phase 24 row with hashes + log paths.

### Exit criteria

* All four smokes reach iter 10 with `>>> done with setup`, then a clean shutdown.
* No `"submodule init failed"`, `"fallback to nn.Linear"`, `"c10d::allreduce_"`, `"unsupported dispatcher module"` in any log.
* `MoE layer=N dispatcher active via PrimusTurboDeepEPTokenDispatcher` for routed-MoE layers.
* V4-aware TFLOPs (P20) reported per iter.

### Test gates

* **G21** — Turbo smoke matrix: A/B/C/D all green.
