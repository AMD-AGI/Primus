# ZAYA1 implementation vs. arXiv:2511.17127

Source: [Training Foundation Models on a Full-Stack AMD Platform](https://arxiv.org/pdf/2511.17127) (arXiv:2511.17127v2).

The architecture in the paper and this port agree on the ZAYA1-base widths. The differences are in the residual formula, an extra Mixture-of-Depths path, and the training system around the model.

## What matches

The matching shape is hidden size 2048, 40 CCA blocks and 40 MoE blocks, query heads 8, KV heads 2, head dimension 128, router width 256, 16 experts, top-1, SwiGLU width 2048, convolution widths 2 and 2, vocabulary 262272, and the Gemma 3 tokenizer.

The code stores this as 80 alternating stages in `primus/configs/models/megatron/zaya1.yaml`. Table II’s “40 transformer layers” is the same block count, one CCA stage plus one MoE stage per layer. `num_attention_heads: 8` is already the compressed query count. The paper’s uncompressed head count of 16 is not a separate tensor.

## Residual scaling

Section IV writes the residual affine as `αx + b`, then normalizes only the layer branch:

```text
x_{l+1} = (α x_l + b_α) + Layer(RMSNorm(β x_l + b_β))
```

`ResidualScaling` in `primus/backends/megatron/core/models/zaya1/zaya1_modules.py` computes `(x + b) * α` instead. It also keeps two streams: the accumulated residual and the previous mixer output. Both are scaled, added, and only then passed through RMSNorm and the next mixer. The paper applies both affines to the same vector and adds the mixer output after the layer.

## Mixture-of-Depths

Section IV specifies 16 experts, top-1 routing, and no residual expert. `zaya_use_mod: true` in `primus/configs/models/megatron/zaya1_base.yaml` adds a 17th router class. A token assigned to that class is copied through unchanged and multiplied by its routing probability (`zaya1_modules.py`). Its balancing bias starts at `-1`, and the load-balance target is `1/17` rather than `1/16`. This paper does not describe that skip class.

## Router

The router otherwise follows equations (2)–(5): down-project to 256, mix in the previous router state, RMSNorm, a 3-layer GELU MLP, softmax, then top-1 on the scores plus a balance bias. Two details differ:

- The paper’s EDA coefficient `γ` is written as one coefficient. The code uses a learned vector of length 256, `router_states_scale`.
- The paper says the balance loop is an AdamW controller inspired by PID, but it does not publish the step size or moments. The code uses AdamW on `p_e - 1/E` with learning rate `1e-3`, betas `0.9` and `0.999`, and zero weight decay, and it includes the Mixture-of-Depths class in that target.

## Attention and parallelism

Appendix C and Table III describe CCA as attention in a smaller latent space: low-rank Q/K/V, a depthwise convolution, a grouped convolution, a one-token delay on half the values, query/key skip mixing, and a key temperature. Those pieces are present, and the projection widths match Table III. Attention is still over the full sequence length. The paper’s FlashAttention kernel and the context-parallel exchange of the last two tokens are not. `zaya1_builder` rejects tensor, pipeline, context, and expert parallelism, and the model rejects packed sequences.

## Training recipe

Section IV-B and Section V train with Muon on the 2D weights and AdamW on the convolutions, embeddings, residual scales, norms, and temperatures, with five Newton–Schulz iterations. Phase 1 uses data parallel plus ZeRO-1, a cosine learning rate from `6e-4` to `2e-4`, sequence length 4096, and RoPE base 10,000, later extended to 32,768 and RoPE base 1,000,000.

The checked-in recipe in `examples/megatron/configs/MI355X/zaya1_8B-BF16-pretrain.yaml` is a 30-iteration AdamW smoke run on mock data with sequence length 128 and learning rate `1e-5`. No training corpus is pinned.
