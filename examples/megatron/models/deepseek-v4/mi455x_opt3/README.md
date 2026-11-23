# DeepSeek-V4 Flash proxy on one MI455X (gfx1250)

Single GPU, EP=1, 32 experts (one EP8 rank of the 256-expert proxy), MBS=GBS=1, sequence length 4096.
Precision follows the DeepSeek-V4 paper without Transformer Engine: FP8 E4M3 with 128x128 weight and 1x128
activation blocks and UE8M0 scales for the expert and dense linears; BF16 for the embedding, lm_head, router,
norms and the attention core.

## Run

Inside the Primus container (Primus at `/workspace/Primus`, Primus-Turbo at `/workspace/Primus-Turbo`):

```bash
export TRITON_MAIN=<triton checkout>/python   # optional, see below
bash examples/megatron/models/deepseek-v4/mi455x_opt3/run_opt3_1gpu.sh graph 20
```

`graph` runs the Megatron full-iteration CUDA graph (`--cuda_graph_impl full_iteration`); `eager` runs without it.
`opt3_best.env` is the preset: the recipe flags, the fused-Adam switches, the TunableOp setup and
`TRITON_HIP_USE_EXPERT_SCHEDULING=0`.

## Fast path and switches

The fused kernels and wiring are on by default when the device is gfx1250 (the registry is
`primus/backends/megatron/core/dsv4_features.py`; each feature keeps its own shape and parallelism guards and
falls back silently). User-facing switches:

| switch | effect |
|---|---|
| `PRIMUS_DSV4_FAST=0` | master off: the upstream Primus paths run (Primus side only) |
| `PRIMUS_DSV4_BITWISE=1` | turns the fusions that change floating-point summation order off, for numerics debugging |
| `PRIMUS_GLUON=0` | forces the non-Gluon fallbacks (`1` skips the Triton version check) |
| `PRIMUS_HC_FUSED=0`, `PRIMUS_OVERLAP=0`, `PRIMUS_FP8BLK_ADAM=0`, `PRIMUS_DSA_GFX1250_TURBO=0` | disable one component |
| `PRIMUS_FP8BLK_ADAM_SKIP_BF16=0` | keep the BF16 weights current (external readers, mid-run checkpoint load) |
| `PRIMUS_BUBBLE_LAG_LOG=0`, `PRIMUS_FUSE2_PREFETCH=0` | log without a one-step lag / do not prefetch the next batch |
| `PRIMUS_DEBUG_CHECKSUM_FILE=<path>` | write bit fingerprints of parameters, main_grads and optimizer state after the last iteration |

## Requirements

- Gluon kernels (the Gluon dense GEMM, the quantization kernels, the HyperConnection backward, the BF16 split-K
  GEMMs) need Triton >= 3.9 on gfx1250. With the image Triton 3.8 the gate (`dsv4_features.gluon_ok`) closes and the
  FlyDSL / Triton / hipBLASLt paths run instead. Point `TRITON_MAIN` at a Triton main checkout to enable them.
- `tunableop_dsv4_gfx1250_bf16_0.csv`: hipBLASLt TunableOp results for lm_head and the grouped o_a bmm. It is tied
  to the hipBLASLt of the image; re-run the BF16 GEMM tests after an image change.
- Primus-Turbo built for gfx1250.

## Checking a run

With the Gluon gate open (Triton >= 3.9), loss at iteration 1 is identical for every configuration of the fast path (the forward is bit-identical). With Gluon off (`PRIMUS_GLUON=0`, or Triton 3.8), the hipBLASLt / Triton fallbacks round differently: loss@1 moves by about 1e-4 (11.91716 / 11.91709 vs 11.91726) and loss@20 stays inside 9.868-9.872. Compare
`PRIMUS_DEBUG_CHECKSUM_FILE` outputs of two runs to confirm that parameters, main_grads and optimizer state match.
