# ZAYA1-8B validation

The checked-in BF16 recipes are `zaya1_8B-BF16-pretrain.yaml` under `examples/megatron/configs/MI300X`, `MI325X`, and `MI355X`. They train on mock data (`mock_data: true`), so they need no dataset mount. They differ only in batch size, which is set to fill each GPU's memory (see [Throughput](#throughput)). The run below used micro-batch 1 and global batch 8 with `train_iters: 30000` on GSM8K instead of mock data.

The FP8 recipes are `zaya1_8B-FP8-pretrain.yaml` in the same three directories, also on mock data. They set `fp8: hybrid` with `fp8_recipe: tensorwise`. CCA Q/K/V/O projections and expert SwiGLU GEMMs are Transformer Engine FP8 linears. The router, depthwise convolutions, RMSNorm, attention, embeddings, and the tied logits projection stay in BF16. The training and parity numbers in this note are from the BF16 run.

Tensorwise (current) scaling takes each FP8 scale from the tensor being cast, so FP8 is accurate from the first step of a cold start. Delayed scaling starts with an empty amax history. On the released checkpoint, a cold delayed-scaling forward scored SGLang's greedy GSM8K rollouts at Pearson r 0.72 (BF16 gives 0.98) and stayed there until training-style backward passes filled the amax history. Keep `fp8_recipe: delayed` only for runs that warm the history before the steps that matter.

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

The only MI355X (gfx950) runs in this note are the 30-step recipes in [Throughput](#throughput).

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

That comparison checks the loaded checkpoint forward (CCA, router, residual scaling, RMSNorm). It is not a second training run. The unit tests in `tests/unit_tests/megatron/transformer/zaya1/` lock the same formulas on a tiny config: residual affine `(x + bias) * scale`, raw key temperature, vector EDA, mixture-of-depths skip, RMSNorm, builder wiring, one forward/backward, and a cold tensorwise FP8 forward checked against the same weights in high precision.

## Throughput

Each recipe was run as checked in on one 8-GPU node with `rocm/primus:v26.8` (PyTorch 2.14.0, Transformer Engine 2.17.1): 30 steps of mock data, sequence length 128, data parallel 8, one micro-batch per GPU per step. Each micro-batch size is the largest multiple of 8 that completes the recipe, including the end-of-training evaluation. Step time is the median, tokens/s/GPU the harmonic mean, and TFLOP/s/GPU the mean over steps 6 to 30.

| GPU | Precision | Micro / global batch | Step time | Tokens/s/GPU | TFLOP/s/GPU | Peak memory per GPU | Next size up |
|---|---|---|---|---|---|---|---|
| MI300X | BF16 | 128 / 1024 | 1635 ms | 10016 | 118.6 | 181 of 192 GB | 136 stalls in evaluation |
| MI300X | FP8 | 120 / 960 | 1814 ms | 8478 | 100.4 | 187 of 192 GB | 128 runs out of memory |
| MI325X | BF16 | 216 / 1728 | 2049 ms | 13426 | 159.0 | 249 of 256 GB | 224 runs out of memory |
| MI325X | FP8 | 184 / 1472 | 2118 ms | 11154 | 132.1 | 245 of 256 GB | 192 runs out of memory |
| MI355X | BF16 | 248 / 1984 | 1733 ms | 17363 | 209.7 | 282 of 288 GB | 256 runs out of memory |
| MI355X | FP8 | 208 / 1664 | 1890 ms | 12400 | 158.4 | 284 of 288 GB | 216 runs out of memory |

TFLOP/s/GPU is Megatron's analytic GPT estimate with ZAYA1's dimensions. It models neither CCA nor the alternation of attention and MoE stages, so compare it only between ZAYA1 runs.

Memory per GPU is a fixed 74 GB (BF16) or 87 GB (FP8, 91 GB on MI355X) for weights, gradients, sharded optimizer state, and runtime buffers, plus about 0.8 GB per sample, mostly fp32 logits over the 262k vocabulary. FP8 holds extra FP8 weight copies, so it fits fewer samples. It is also slower than BF16 here: with hidden size 2048 and top-1 routing over 16 experts, the expert GEMMs are small, and the FP8 casts cost more than the faster GEMMs save.

The FP8 recipes set `NVTE_USE_CAST_TRANSPOSE_TRITON: "0"` in their `env:` block. The runner turns on Transformer Engine's Triton cast-transpose by default, and that kernel autotunes for each new tensor shape. Expert token counts change every step, so with the Triton kernel the MI325X FP8 run at micro-batch 64 spent 307 s on its first step and still took 2 to 10 s per step at steps 11 to 15, against 1.8 s with the HIP cast kernel.
