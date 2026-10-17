# ZAYA1-8B validation

The checked-in recipes are `examples/megatron/configs/MI300X/zaya1_8B-BF16-pretrain.yaml` and `examples/megatron/configs/MI355X/zaya1_8B-BF16-pretrain.yaml`. Those two files are the same recipe. The run below used that recipe with `train_iters: 30000` on GSM8K.

## Hardware and software

| Item | Value |
|---|---|
| GPU | 8x AMD Instinct MI300X (gfx942) |
| Model | ZAYA1-8B: 80 stages (40 CCA + 40 MoE), hidden 2048, vocab 262272 |
| dtype | BF16 (`bf16: true`) |
| Parallelism | Data parallel 8. Tensor, pipeline, context, and expert parallel size 1. Sequence parallel off. |
| Batch | micro-batch 1, global batch 8, sequence length 128 |
| Optimizer | AdamW, lr `1e-5` cosine to 0, weight decay 0.1 |
| Data | GSM8K indexed with the Gemma tokenizer shipped in `Zyphra/ZAYA1-8B` |
| PyTorch | 2.12.0+rocm10.0.0 |
| HIP | 7.15.26333 |
| Megatron-LM | `d3528a213` |
| Primus commit of the run | `5c7dc4113` |

The MI355X example yaml was not launched on MI355X. It matches the MI300X yaml, so this run does not exercise a gfx950 kernel path.

## Training

`pretrain()` finished 30,000 steps. Skipped iterations 0, NaN iterations 0. Final grad norm 2.710. Final validation lm loss 3.017.

| Iteration | lm loss |
|---|---|
| 1 | 12.820 |
| 10 | 11.229 |
| 30 | 7.597 |
| 30000 | 0.190 |

Excerpt from the rank-7 training log:

```text
iteration        1/   30000 | ... | lm loss: 1.282030E+01 | loss scale: 1.0 | grad norm: 12.429 | number of skipped iterations:   0 | number of nan iterations:   0
iteration       10/   30000 | ... | lm loss: 1.122902E+01 | loss scale: 1.0 | grad norm: 18.906 | number of skipped iterations:   0 | number of nan iterations:   0
iteration       30/   30000 | ... | lm loss: 7.597235E+00 | loss scale: 1.0 | grad norm: 6.582 | number of skipped iterations:   0 | number of nan iterations:   0
iteration    30000/   30000 | ... | lm loss: 1.897044E-01 | loss scale: 1.0 | grad norm: 2.710 | number of skipped iterations:   0 | number of nan iterations:   0
```

## Forward parity against the released checkpoint

A separate prefill comparison scored Primus ZAYA1 log probabilities against an SGLang prefill of `Zyphra/ZAYA1-8B` on 4 GSM8K prompts (4096 completion tokens):

| Metric | Value |
|---|---|
| Pearson r | 0.9965 |
| KL divergence | 0.00102 |
| MAE of logprob | 0.0230 |
| RMSE of logprob | 0.0732 |
| Mean rollout probability difference | -0.000144 |

That comparison checks the loaded checkpoint forward (CCA, router, residual scaling, RMSNorm). It is not a second training run. The unit tests in `tests/unit_tests/megatron/transformer/zaya1/` lock the same formulas on a tiny config: residual affine `(x + bias) * scale`, raw key temperature, vector EDA, mixture-of-depths skip, RMSNorm, builder wiring, and one forward/backward.
