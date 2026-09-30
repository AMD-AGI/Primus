# ZAYA1 feature map

Commit: `06269526c4d2b770e347bfa4e51fa78802c8452f` (`06269526c`) on `dev/clairlee/zaya1`.
Date: 2026-09-28 22:06:35 +0000.
The ZAYA1 sources described below match that commit.

Entry point: `model_type: zaya1` in `primus/configs/models/megatron/zaya1_base.yaml`.
`primus/core/utils/import_utils.py` dispatches that type to `zaya1_builder` in
`primus/backends/megatron/core/models/zaya1/zaya1_builders.py`. The mixer math lives in
`primus/backends/megatron/core/models/zaya1/zaya1_modules.py`. Config fields are declared on
`Zaya1TransformerConfig` in `zaya1_transformer_config.py`.

## CCA / Compressed Causal Attention

The source names this Compressed convolutional attention. There is no boolean switch.

`ZayaStack` builds one stage per decoder layer. `layer_kinds` returns `"a"` for CCA and `"m"` for MoE. With `zaya_layers` omitted, even stages are CCA and odd stages are MoE. The released shape is 80 stages (40 CCA + 40 MoE) via `num_layers: 80` in `primus/configs/models/megatron/zaya1.yaml`.

- Schedule: `layer_kinds`, `zaya1_modules.py`
- Construction: `ZayaStage.__init__` builds `CCA` when `kind == "a"`
- Implementation: `CCA` (`project` / `forward`) and `attention` in the same file
- Convolution widths: `cca_time0: 2`, `cca_time1: 2` in `zaya1_base.yaml`
- Partial RoPE: `partial_rotary_factor: 0.5`, `rotary_base: 10000`, applied inside `CCA._rope`

## ZAYA1 nonlinear router

Every MoE stage (`kind == "m"`) builds `ZayaMoE`, which always builds `ZayaRouter`. The router is a down-projection to `zaya_mlp_expansion` (256), then a three-layer MLP with two GELU nonlinearities and a final linear to the scored classes, followed by softmax.

- Class: `ZayaRouter` in `zaya1_modules.py`
- Width: `zaya_mlp_expansion: 256` in `zaya1_base.yaml`
- Experts and top-k: `num_experts: 16`, `moe_router_topk: 1`
- Previous-stage router state: `zaya_use_eda: true` adds `prev_router * router_states_scale` before the router RMSNorm
- fp32 softmax: `zaya_high_prec: true`
- Load-balance biases: microbatch AdamW on `p_e - 1/E`, step size `zaya_balance_lr: 1.0e-3`. `0` freezes `balancing_biases`.

## ZAYA1 residual scaling

Enabled by `scale_residual_merge: true` in `zaya1_base.yaml`.

`ResidualScaling` applies a per-hidden affine, `(x + bias) * scale`, separately to the mixer output and, from stage 1 onward, to the incoming residual. Stage 0 stores only the mixer-output affine. `ZayaStage` attaches one module per stage; `ZayaStack` attaches another before the final RMSNorm. Setting `scale_residual_merge` false skips both.

## ZAYA1 Mixture-of-Depths behavior

Enabled by `zaya_use_mod: true` in `zaya1_base.yaml`.

`ZayaRouter` then scores `num_experts + 1` classes. The extra class is the skip. Its balancing bias is initialized to `-1`. In `ZayaMoE.forward`, a token routed to that class is passed through unchanged and multiplied by the unbiased gathered routing probability. Setting `zaya_use_mod` false removes the skip class.

## Combined file and line reference

| Feature | Contribution | Enablement / configuration | Instantiation | Runtime implementation |
|---|---|---|---|---|
| ZAYA1 model | Selects the complete ZAYA1 architecture and its alternating CCA/MoE stack. | `primus/configs/models/megatron/zaya1_base.yaml:10-13` (`model_type: zaya1`) | `primus/core/utils/import_utils.py:62-74` dispatches to `zaya1_builder`; `primus/backends/megatron/core/models/zaya1/zaya1_builders.py:43-88` constructs `Zaya1Model` | `primus/backends/megatron/core/models/zaya1/zaya1_model.py:80-81` constructs `ZayaStack` |
| CCA / Compressed Causal Attention | Reduces attention cost by compressing the KV representation while preserving causal context, improving memory efficiency and inference/training scalability at long sequence lengths. | `primus/configs/models/megatron/zaya1_base.yaml:31-47` configures partial RoPE; `:81-84` configures convolution widths | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:25-44` selects even stages as `"a"` by default; `:430-437` constructs `CCA` | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:108-240` implements CCA projection; `:243-264` performs causal GQA |
| ZAYA1 nonlinear router | Makes MoE expert selection input-dependent and nonlinear, allowing more expressive routing and better specialization of experts than a simple linear router. | `primus/configs/models/megatron/zaya1_base.yaml:65-70` enables 16-expert top-1 routing; `:76-80` enables width 256, EDA, and fp32 softmax | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:385-395` constructs `ZayaRouter` and experts | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:278-350` implements the down-projection, RMSNorm, three-layer GELU MLP, EDA state, softmax, and top-k |
| ZAYA1 residual scaling | Applies learned/scaled residual contributions to stabilize the combination of CCA and MoE blocks, helping training stability and optimization in the deep architecture. | `primus/configs/models/megatron/zaya1_base.yaml:81` sets `scale_residual_merge: true` | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:430-431` constructs per-stage scaling; `:458-462` constructs final scaling | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:77-98` implements `(x + bias) * scale`; `:439-449` applies it at each stage; `:464-474` applies the final merge |
| ZAYA1 Mixture-of-Depths behavior | Dynamically varies how much computation different tokens receive, allowing less important tokens to skip or receive reduced computation, improving the compute/quality tradeoff. | `primus/configs/models/megatron/zaya1_base.yaml:77` sets `zaya_use_mod: true` | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:285-313` adds the extra scored skip class and initializes its balancing bias to `-1` | `primus/backends/megatron/core/models/zaya1/zaya1_modules.py:399-416` routes the skip class through the identity path, weighted by the gathered routing probability |

Implementation note: this port names `CCA` “Compressed convolutional attention” and executes full causal scaled-dot-product attention. Its current implementation does not sparsify or shorten the sequence axis, so the CCA contribution above should be read as reduced/structured KV representation rather than elimination of the quadratic attention matrix.
