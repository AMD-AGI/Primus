# ZAYA1-8B validation

The checked-in BF16 recipes are `zaya1_8B-BF16-pretrain.yaml` under `examples/megatron/configs/MI300X`, `MI325X`, and `MI355X`. They train on mock data (`mock_data: true`), so they need no dataset mount. They differ in batch size (see [Throughput](#throughput)), and the MI355X recipes also use Transformer Engine's fused cross-entropy, no dataloader attention mask, and overlapped gradient reduce-scatter and parameter all-gather (see [MI355X tuning](#mi355x-tuning)). The run below used micro-batch 1 and global batch 8 with `train_iters: 30000` on GSM8K instead of mock data.

The FP8 recipes are `zaya1_8B-FP8-pretrain.yaml` in the same three directories, also on mock data. They set `fp8: hybrid` with `fp8_recipe: tensorwise`. CCA Q/K/V/O projections and expert SwiGLU GEMMs are Transformer Engine FP8 linears. The router, depthwise convolutions, RMSNorm, attention, embeddings, and the tied logits projection stay in BF16. The training run and the first parity comparison in this note are BF16.

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

The MI355X (gfx950) runs in this note are the mock-data recipe runs and checks in [Throughput](#throughput) and [MI355X tuning](#mi355x-tuning).

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

On MI355X, the same comparison (SDPA attention, eval mode, one GPU per run) for the model code before and after the changes in [MI355X tuning](#mi355x-tuning):

| Precision | Model code | Pearson r | KL divergence | MAE of logprob | RMSE of logprob |
|---|---|---|---|---|---|
| BF16 | before | 0.9979 | 0.00101 | 0.0215 | 0.0562 |
| BF16 | current | 0.9974 | 0.00111 | 0.0230 | 0.0620 |
| FP8 | before | 0.9799 | 0.00943 | 0.0751 | 0.1812 |
| FP8 | current | 0.9796 | 0.00928 | 0.0745 | 0.1777 |

The fused functions keep intermediates in fp32 where eager mode rounds them to BF16 between steps, so the current BF16 forward rounds differently from the earlier one.

## Throughput

Each recipe ran on one 8-GPU node with `rocm/primus:v26.8` (PyTorch 2.14.0, Transformer Engine 2.17.1): mock data, sequence length 128, data parallel 8, one micro-batch per GPU per step. Step time is the median, tokens/s/GPU the harmonic mean, and TFLOP/s/GPU the mean over the measured steps.

The MI300X and MI325X rows are 30-step runs measured over steps 6 to 30, from before the model changes in [MI355X tuning](#mi355x-tuning). Each micro-batch size is the largest multiple of 8 that completes the recipe, including the end-of-training evaluation. The MI355X rows are measured over steps 30 to 50 with the current code, in 50-step runs (`--train_iters 50`) and in the 300-step runs of [MI355X tuning](#mi355x-tuning). Each value is the median of two BF16 or four FP8 runs. Their recipes keep the batch sizes found the same way before those changes, so their loss curves stay comparable with earlier runs.

| GPU | Precision | Micro / global batch | Step time | Tokens/s/GPU | TFLOP/s/GPU | Peak memory per GPU | Next size up |
|---|---|---|---|---|---|---|---|
| MI300X | BF16 | 128 / 1024 | 1635 ms | 10016 | 118.6 | 181 of 192 GB | 136 stalls in evaluation |
| MI300X | FP8 | 120 / 960 | 1814 ms | 8478 | 100.4 | 187 of 192 GB | 128 runs out of memory |
| MI325X | BF16 | 216 / 1728 | 2049 ms | 13426 | 159.0 | 249 of 256 GB | 224 runs out of memory |
| MI325X | FP8 | 184 / 1472 | 2118 ms | 11154 | 132.1 | 245 of 256 GB | 192 runs out of memory |
| MI355X | BF16 | 248 / 1984 | 688 ms | 46104 | 546.1 | 215 of 288 GB | 320 fits |
| MI355X | FP8 | 208 / 1664 | 849 ms | 31230 | 369.9 | 217 of 288 GB | 320 fits |

TFLOP/s/GPU is Megatron's analytic GPT estimate with ZAYA1's dimensions. It models neither CCA nor the alternation of attention and MoE stages, so compare it only between ZAYA1 runs.

On MI300X and MI325X, memory per GPU is a fixed 74 GB (BF16) or 87 GB (FP8) for weights, gradients, sharded optimizer state, and runtime buffers, plus about 0.8 GB per sample, mostly fp32 logits over the 262k vocabulary. FP8 holds extra FP8 weight copies, so it fits fewer samples. On MI355X, the fused cross-entropy works on the BF16 logits in place and the fused stage boundaries save fewer intermediates, so a sample costs about 0.6 GB (BF16) or 0.5 GB (FP8). At micro-batch 320 (global batch 2560), BF16 runs at 807 ms per step and 50,678 tokens/s/GPU with a 257 GB peak, and FP8 at 1002 ms and 40,881 tokens/s/GPU with 268 GB.

FP8 is slower than BF16 here: with hidden size 2048 and top-1 routing over 16 experts, the expert GEMMs are small, and the FP8 casts cost more than the faster GEMMs save. With current scaling, Transformer Engine runs separate amax and cast kernels for every cast, so an FP8 step launches about 49,000 kernels against 20,000 for BF16, and at micro-batch 208 much of the step goes to launching them. A larger micro-batch therefore helps FP8 more: at 320, FP8 gains 31% tokens/s/GPU and BF16 10%.

The FP8 recipes set `NVTE_USE_CAST_TRANSPOSE_TRITON: "0"` in their `env:` block. The runner turns on Transformer Engine's Triton cast-transpose by default, and that kernel autotunes for each new tensor shape. Expert token counts change every step, so with the Triton kernel the MI325X FP8 run at micro-batch 64 spent 307 s on its first step and still took 2 to 10 s per step at steps 11 to 15, against 1.8 s with the HIP cast kernel.

## MI355X tuning

Each row adds one change to the row above, on one MI355X node at the recipes' batch sizes (BF16 248 / 1984, FP8 208 / 1664), measured as in [Throughput](#throughput). Model changes apply on every GPU. Recipe changes are in the MI355X recipes only.

| Change | Where | BF16 step time | FP8 step time |
|---|---|---|---|
| Earlier code and recipes | | 1688 ms | 1886 ms |
| MoE tokens dispatched with one sort instead of a boolean mask per expert | model | 1285 ms | 1334 ms |
| Residual scaling, merge and RMSNorm fused at each stage boundary | model | 1059 ms | 1150 ms |
| CCA query/key chain fused, RoPE tables built once per forward | model | 930 ms | 1041 ms |
| CCA convolutions as elementwise math and a batched GEMM instead of MIOpen | model | 830 ms | 958 ms |
| Transformer Engine fused cross-entropy (`cross_entropy_fusion_impl: te`) | recipe | 768 ms | 906 ms |
| No dataloader attention mask (`create_attention_mask_in_dataloader: false`) | recipe | 743 ms | 878 ms |
| Overlapped gradient reduce-scatter and parameter all-gather, 1e9-parameter buckets | recipe | 688 ms | 849 ms |

The first and last rows are medians over runs (three of the earlier code, and the runs of [Throughput](#throughput)); the others are single runs. Identical runs differ in step time by up to 5% for FP8 (839 to 882 ms for the last row) and for the earlier code, and by under 0.1% for BF16 with the current code, so the FP8 gains of the last two rows are within that spread. Binding each rank to its GPU's NUMA node (`primus-cli direct --numa -- train pretrain ...`) gains about 1% for BF16, and 1 to 2% for FP8 against unbound runs alternating with the bound ones; the table is without it.

The fused functions run through Megatron's `jit_fuser` (`torch.compile`) on GPU. CPU runs, like the unit tests, stay eager. Without the dataloader mask, CCA's attention runs with `is_causal`; set `create_attention_mask_in_dataloader: true` again together with `reset_attention_mask`.

Overlap needs a gradient for every parameter at every step, or Megatron's DDP asserts at the second step. The sorted dispatch runs every expert, on an empty slice too, and the first MoE stage's EDA scale gets a zero gradient; `test_every_stack_parameter_gets_a_gradient` checks it. With Megatron's default 40M-parameter buckets (190 for ZAYA1), overlap is slower than none, because before each bucket's reduce-scatter DDP checks its gradients for NaN and Inf with two host syncs.

Over 300 steps of each recipe with the same seed (`--train_iters 300`), the current code follows the earlier code's training loss. The earlier code is not bit-reproducible on MI355X, so it ran twice, and the differences below are per step from its first run. The current code repeated bit for bit in every rerun.

| Precision | Run | Largest difference (step) | Mean difference | Mean loss, steps 201 to 300 |
|---|---|---|---|---|
| BF16 | earlier code, first run | | | 0.09291 |
| BF16 | earlier code, second run | 2.7e-4 (47) | 3.5e-5 | 0.09291 |
| BF16 | current code | 5.0e-4 (22) | 6.0e-5 | 0.09295 |
| FP8 | earlier code, first run | | | 0.09664 |
| FP8 | earlier code, second run | 8.0e-4 (28) | 1.0e-4 | 0.09664 |
| FP8 | current code | 2.9e-3 (19) | 2.2e-4 | 0.09660 |

The largest differences come in the first 50 steps, while the loss falls from 12.9 to 3.4. Over the last 100 steps the current code's mean loss is within 0.05% of the earlier code's, above it for BF16 and below it for FP8.
