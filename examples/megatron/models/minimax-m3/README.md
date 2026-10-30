# MiniMax-M3 proxy on MI455X

MiniMax-M3 BF16 pretraining on MI455X (gfx1250). The default is a 4-layer proxy of the real model:
the first three layers stay dense, and layer 4 is one MSA (MiniMax Sparse Attention) layer plus one
MoE layer (128 experts, top-4, hidden 6144, 64 query / 4 KV heads of dim 128). Sequence length 4096,
MBS 1, mock data, full recompute of the first layer. The config is
`examples/megatron/configs/MI455X/minimax_m3-BF16-pretrain.yaml` and the launcher is
`run_minimax_m3_mi455x.sh`: with `PRIMUS_LAUNCHER=container` (default) it starts a new container of
`DOCKER_IMAGE` from the host and re-runs itself inside it; with `PRIMUS_LAUNCHER=direct` it runs inside
the current container.

MSA is not Primus-Turbo attention. `msa_backend=flydsl` (the default) runs the kernels under
`primus/backends/megatron/core/transformer/minimax_m3/flydsl/`. On gfx950 those are the MFMA kernels.
On gfx1250 they are the wave32 WMMA kernels (`*_w32.py`); a tree without that port fails at compile
with `Cannot select: permlane32.swap`. `PRIMUS_MSA_BACKEND=eager` is the plain-PyTorch reference and
does not need those kernels. The experts use Primus-Turbo's grouped GEMM and RMSNorm uses
Primus-Turbo. DeepEP stays off (see Known issues).

The default image is `amdprimus/amdprimus:tas-20261019` (see the DeepSeek-V3 example's README),
`amdprimus/amdprimus@sha256:d653324d124e218f43c92eaac3cfc5d05059bd45fb4abf5bdfb78b987af4af87`: torch
2.14.0 on ROCm 10.2.0a20261011, Triton 3.8.0, Transformer Engine 2.19.0.dev0 (bfe94fbe5),
Primus-Turbo 0.6.1.dev2, FlyDSL 0.3.4.1 and AITER 0.1.24.post1. Every run below used that image and
its own Primus-Turbo.

## Run

```bash
# On the host, 1 GPU (the default: EP1, GBS 1). Pick a free device on a shared node.
HIP_VISIBLE_DEVICES=1 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# On the host, 2 GPUs (EP2, GBS 2)
HIP_VISIBLE_DEVICES=2,3 GPUS_PER_NODE=2 bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh

# Inside a container
PRIMUS_LAUNCHER=direct bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh
```

Two runs at once on one node need different `MASTER_PORT`s.

**Settings** are environment variables:

- Parallelism: `GPUS_PER_NODE` (default 1), `HIP_VISIBLE_DEVICES`, `PRIMUS_TP` (must be 1), `PRIMUS_PP`
  (default 1), `PRIMUS_EP` (default `GPUS_PER_NODE`). Expert parallel must divide both 128 and the
  data-parallel size. `GBS` defaults to `MBS` times that data-parallel size.
- Model depth: `PRIMUS_TOTAL_LAYERS` (default 4, minimum 4; 8 is the MI355X proxy, 60 is the full
  model). The launcher keeps the first three layers dense and sets `moe_layer_freq`,
  `sparse_attention_freq` and `sparse_disable_index_value` to the same length.
- Training: `TRAIN_ITERS` (default 10; at least 3, because the yaml warms the learning rate up over
  2), `MBS` (default 1), `GBS`, `PRIMUS_SEQ_LENGTH` (default 4096), `PRIMUS_MAX_POSITION_EMBEDDINGS`
  (default `PRIMUS_SEQ_LENGTH`; the preset would otherwise build the RoPE cache at 1,048,576),
  `PRIMUS_RECOMPUTE_NUM_LAYERS` (default 1). The yaml's `recompute_method: block` recomputes the
  first that many layers, so the default recomputes dense layer 1, not the MSA/MoE layer.
- MSA: `PRIMUS_MSA_BACKEND`, `flydsl` (default) or `eager`.
- Experts: `USE_TURBO_GROUPED_GEMM` (default `True`; `False` is TE's grouped GEMM, one hipBLASLt GEMM
  per expert) and `TURBO_GROUPED_GEMM_BACKEND` (default `default`, which is Triton here; `bf16:flydsl`
  takes at most 64 local experts, so EP 2 or more). `USE_TURBO_DEEPEP` (default `False`).
- Dense layers: `USE_TURBO_ATTENTION` (default `False`) runs Primus-Turbo attention in the three dense
  layers. The launcher then sets `PRIMUS_TURBO_ATTN_BACKEND=triton`, because the default backend fails
  with `invalid argument for fmha_bwd` here. MSA layers are unaffected.
- `MOCK_DATA` (default `True`).
- `PROFILE=True` profiles iterations 6–7 (trace under `output/<team>/<user>/<exp>/tensorboard/`).
- `PATCH_TRITON_EXPERT_SCHEDULING` (default `1`): see Triton expert scheduling below.
- `HIPBLASLT_TUNING_OVERRIDE_FILE` (default unset: hipBLASLt's own kernel choice). This example has no
  tuned file yet. A file may be given relative to the current directory; the launcher mounts it into
  the container. See hipBLASLt tuning in the DeepSeek-V3 README for how such a file is made.
- Optimizer: `PRIMUS_FUSED_ADAM_CLIP` (default `1`) runs Primus's HIP Adam, which applies the
  gradient-clip coefficient inside the kernel. `PRIMUS_FUSED_ADAM_BF16_WRITEBACK` and
  `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU` (default: the value of `PRIMUS_FUSED_ADAM_CLIP`) also have it
  write the BF16 parameters and keep the loss scale on the host. Megatron then logs no grad norm. Set
  `PRIMUS_FUSED_ADAM_CLIP=0` with `--use_precision_aware_optimizer True`: the BF16 writeback needs
  separate FP32 master parameters, and the run stops at the first step without them. With
  `PRIMUS_FUSED_ADAM_CLIP=0` Adam is Primus's Triton AdamW (`use_triton_fused_adam`, on by default);
  `--use_triton_fused_adam False` keeps TE's FusedAdam. With `PRIMUS_FUSED_ADAM_CLIP=1` the Triton
  patch logs that it keeps TE's FusedAdam, whose step the HIP kernel replaces.
- Config and output: `EXP` (the experiment yaml), and `PRIMUS_WORKSPACE` (default `output`), `PRIMUS_TEAM`,
  `PRIMUS_USER`, `PRIMUS_EXP_NAME`, which make up the output directory `<workspace>/<team>/<user>/<exp>`.

The defaults are in the script. Arguments to the script are appended to the Primus command line, for
example `--gradient_accumulation_fusion True`.

**Output.** The console log is written to `output/<team>/<user>/<exp>/train.log`. From the host,
training runs as root in the new container, so the files it creates are owned by root. Inside a
container it runs as the container's user, which has to be root for the Triton patch.

**Triton expert scheduling.** Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes
Triton kernels fault or return NaN at random. The launcher sets `TRITON_HIP_USE_EXPERT_SCHEDULING=0`,
which Triton applies whenever it compiles a kernel, and with `PATCH_TRITON_EXPERT_SCHEDULING=1` (the
default) also makes `is_expert_scheduling_enabled()` in the container's Triton return `False`, which
invalidates Inductor's cache of kernels compiled without the variable. The patch needs root and stays
in a long-lived container; set `PATCH_TRITON_EXPERT_SCHEDULING=0` to skip it. See the Llama-3.1-8B
README for the same note.

## Reference

Single runs on one shared 4x MI455X node (432 GB each) with `amdprimus/amdprimus:tas-20261019` and
the yaml as it is now. Not a repeated baseline; two runs of the default differed by 0.3%. ms/iter is
Primus's average over iterations 3–10 (`log_avg_skip_iterations=2`; iterations 1–2 compile the FlyDSL
and Triton kernels). Memory is the logged `hip mem usage`, which counts the whole device. Routing
was `moe_router_force_load_balancing: true` and `moe_router_dtype: fp32`. Every run finished 10
iterations with 0 NaN.

| GPUs | Layers | MBS / GBS | Change from the defaults | ms/iter | Tokens/s/GPU | TFLOP/s/GPU | Memory/GPU | lm loss, iteration 1 → 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 / 1 | none | 158.2 | 25,887 | 427.0 | 241.5 GB | 13.448 → 10.989 |
| 1 | 4 | 1 / 1 | `USE_TURBO_ATTENTION=True` | 151.3 | 27,077 | 446.7 | 233.8 GB | 13.448 → 10.989 |
| 1 | 4 | 1 / 1 | `PRIMUS_MSA_BACKEND=eager` | 189.8 | 21,579 | 355.9 | 257.9 GB | 13.448 → 10.988 |
| 1 | 4 | 2 / 2 | `MBS=2` | 230.0 | 35,623 | 587.6 | 248.5 GB | 13.435 → 10.570 |
| 2 (EP2) | 4 | 1 / 2 | none | 139.0 | 29,462 | 486.0 | 184.2 GB | 13.443 → 10.312 |
| 2 (EP2) | 4 | 1 / 2 | `TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl` | 136.3 | 30,046 | 495.6 | 184.2 GB | 13.443 → 10.312 |
| 2 (EP2) | 8 | 1 / 2 | `PRIMUS_TOTAL_LAYERS=8` | 337.4 | 12,141 | 322.0 | 432 GB (all) | 13.444 → 9.936 |
| 4 (EP4) | 4 | 1 / 4 | none | 126.5 | 32,370 | 534.0 | 155.9 GB | 13.437 → 9.669 |
| 4 (EP4) | 8 | 1 / 4 | `PRIMUS_TOTAL_LAYERS=8` | 270.6 | 15,134 | 401.3 | 288.6 GB | 13.429 → 9.119 |

Iteration 1 took 28–38 s with FlyDSL MSA (39 s with the FlyDSL grouped GEMM) and 18 s with eager,
mostly kernel compilation. EP2 and EP4 use Megatron's `alltoall` dispatcher. The 8-layer EP2 run has
22.4 B parameters per GPU, about 380 GB of weights, gradients and Adam state before activations, so
it fills the device; recomputing all 8 layers was 14% slower and did not make room. At EP4 it leaves
about 140 GB free.

The EP4 rows were measured on `dev/mi455x_v3` (a624bb95), the other rows on an older base (dad0a081).
Both have Megatron-LM d3528a21 (1042 commits after core_v0.15.0rc7) and no Triton AdamW. On
a624bb95 the defaults ran at 156.0 ms on one GPU and 138.2 ms on EP2, with the same losses and
memory.

The table and the sections below it predate the fused Adam defaults and
`check_for_nan_in_loss_and_grad: false`; the runs in it used TE's FusedAdam with the check on. With
the fused Adam defaults one GPU ran at 133.8 ms and EP2 at 126.8 ms on a624bb95 (see Optimizer).
EP4 has not been re-run.

This branch is now based on `dev/mi455x_v3` c19456f1, with Megatron-LM core_v0.19.2 and Primus's
Triton AdamW on by default. None of the numbers on this page has been re-measured on it.

**Optimizer.** Megatron's step reads and writes the FP32 gradients and parameters (10.9 B on the
1-GPU proxy) several times: a norm over every gradient for the NaN check, the 1/DP gradient scale,
the clipping l2norm, the clip scale, Adam, and the BF16 parameter copy. At EP4 with 8 layers, these
passes, the gradient accumulation and zeroing took about 77 of the 281 ms of kernel time.
Cumulative, one GPU, `dev/mi455x_v3` a624bb95:

| Change | ms/iter |
| --- | --- |
| none: TE FusedAdam (`PRIMUS_FUSED_ADAM_CLIP=0`), NaN check on | 156.0 |
| `check_for_nan_in_loss_and_grad: false` | 150.6 |
| `PRIMUS_FUSED_ADAM_CLIP=1` | 140.2 |
| `PRIMUS_FUSED_ADAM_BF16_WRITEBACK=1`, `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=1` (the defaults) | 133.8 |

EP2 went from 138.2 to 126.8 ms. A later pair on one GPU, with other jobs on the node, ran at 160.2
and 137.4 ms. Iteration-10 losses are 10.98863 and 10.98864 on one GPU and 10.31213 and 10.31194 on
EP2, because the clip coefficient is computed in FP32 on the device rather than in Python. Profiled,
Adam is one 27.6 ms kernel for both parameter groups and the clipping l2norm 2.7 ms; the scale
passes, the NaN-check norm and the parameter copy are gone. Before the fix in
`fused_adam_clip_patches.py` (no hard-coded `-std=c++17`), the HIP extension did not build against
this image's torch 2.14, whose headers need C++20, and Primus logged the error and trained without
the patch.

**Expert grouped GEMM.** `USE_TURBO_GROUPED_GEMM=False`, otherwise the same runs. Losses agree to
within 0.001.

| Run | Primus-Turbo (default) | TE grouped GEMM |
| --- | --- | --- |
| 1 GPU | 158.2 ms | 167.4 ms |
| 1 GPU, `USE_TURBO_ATTENTION=True` | 151.3 ms | 158.1 ms |
| 1 GPU, eager MSA | 189.8 ms | 197.6 ms |
| 1 GPU, MBS 2 | 230.0 ms | 234.3 ms |
| EP2 | 139.0 ms | 142.6 ms |
| EP2, 8 layers | 337.4 ms | 368.8 ms |

The Primus-Turbo grouped GEMM keeps a layer's experts in one weight, which needs the
`te_l2norm_large_tensor` patch described under Known issues. The FlyDSL backend at EP2 is
bit-identical to Triton over 10 iterations; a standalone run at the EP2 shapes shows it does run the
FlyDSL forward and weight-gradient kernels.

**Where the 1-GPU step goes.** GPU kernel time of profiled iteration 7 with the defaults of the time,
before the fused Adam (`PROFILE=True`), 165 ms on a 162 ms GPU span:

| Kernels | ms |
| --- | --- |
| optimizer: TE Adam, scale and l2norm over 10.9 B fp32 parameters | 43.4 |
| hipBLASLt GEMMs: dense FFN, attention projections, output layer | 38.3 |
| elementwise and reductions; 10.4 of it the fp32 += bf16 gradient accumulation into `main_grad` | 24.0 |
| copies and fills; 12.5 of it four host-to-device copies | 19.5 |
| TE unfused softmax in the three dense attention layers | 13.7 |
| Primus-Turbo grouped GEMM (Triton), all expert GEMMs | 12.1 |
| FlyDSL MSA and indexer kernels | 9.8 |
| norm, RoPE, cross entropy, MoE permute | 3.1 |
| other | 1.4 |

With TE's grouped GEMM the expert GEMMs are 128 hipBLASLt launches per GEMM and 35.9 ms. On one GPU
the optimizer is large because the proxy holds all 128 experts and the untied 200k-token embedding
and output layer with only 4096 tokens per step. `USE_TURBO_ATTENTION=True` replaces the unfused softmax.
The host-to-device copies are TE Adam's tensor-list metadata and the batch, and their time is mostly
spent waiting in the stream behind earlier kernels: with `HSA_ENABLE_SDMA=0` they take 0.1 ms in all
and the step is no faster (158.0 ms).

**Settings measured on the way.** One GPU unless noted, the defaults of the time (TE grouped GEMM
for the first two):

- `manual_gc_interval`. Megatron's `manual_gc` with interval 1 calls `gc.collect()` every step. With
  about 583,000 tracked Python objects after startup that takes about 280 ms, more than the step, and
  frees nothing. Interval 1 to 100: 463.9 to 183.6 ms on one GPU, 545.1 to 150.7 ms on EP2. The yaml
  uses 100. The MI355X M3 yamls also use 1; their effect there was not measured.
- `overlap_grad_reduce` and `overlap_param_gather` on to off: 183.6 to 167.4 ms on one GPU, 150.7 to
  142.6 ms on EP2. The yaml turns them off.
- `gradient_accumulation_fusion: true`: 183.6 to 209.8 ms with TE's grouped GEMM (overlap on), 158.2
  to 184.3 ms with Primus-Turbo's. The fusion is not what costs; the fp32 output is. With it, each
  dense weight gradient is a bf16 x bf16 GEMM written straight into the fp32 `main_grad` with beta 1.
  For that hipBLASLt on gfx1250 picks a `BSS` MT256x128x64 kernel: 35.4 ms a step, in place of about
  9.6 ms of the bf16-output kernels. The Triton expert weight-gradient GEMM goes from 7.0 to 16.5 ms
  with its fp32 accumulate epilogue. The fp32 `+=` it saves is 8.6 ms (profiled iteration 7, 163.9
  to 191.9 ms). Into a bf16 `main_grad` (`--main_grads_dtype bf16`, see Switches on one GPU)
  hipBLASLt keeps the same bf16 kernels (+0.6 ms for beta 1), the Triton GEMM adds 2.3 ms, the bf16
  `+=` saves 4.2 ms, and the step goes from 134.0 to 132.8 ms. The yaml leaves it off. Tuning
  hipBLASLt for the fp32-output shapes was not tried.

**MSA and indexer only**, at the proxy layer shape (S=4096, B=1, 64 query / 4 KV heads, top-16
blocks of 128, 4 index heads of dim 128). CUDA events, 5 iterations after 2 warmups, one GPU.
Eager backward is (forward+backward) minus a separately timed forward. Peak is allocator memory for
that call, not the training step's `hip mem usage`.

| Op | flydsl | eager |
| --- | --- | --- |
| attention forward | 1.26 ms, 0.36 GiB | 17.68 ms, 21.50 GiB |
| attention backward | 7.40 ms, 0.61 GiB | 20.30 ms, 24.89 GiB |
| indexer forward | 0.30 ms | 0.57 ms |
| indexer backward | 0.72 ms | 1.32 ms |

Attention forward+backward is 8.65 ms versus 37.98 ms. End to end, FlyDSL takes the step from 189.8 to
158.2 ms and saves 16 GB on the 4-layer proxy, where one of the four layers is MSA.

The FlyDSL unit tests (`tests/unit_tests/backends/megatron/test_minimax_m3_flydsl_*.py`) passed on
gfx1250 on the old base: 43 items, forward and backward against eager, indexer, and the
`MinimaxSparseAttention` module.

## Switches on one GPU

Each switch on top of the defaults of the time (TE FusedAdam, NaN check on), one GPU,
10 iterations, same image and node. The defaults ran at 157.7 ms again during this sweep, so
differences under 1% are noise. Every run below had 0 NaN.

| Switch | ms/iter | Memory | lm loss, iteration 10 |
| --- | --- | --- | --- |
| none (defaults) | 158.2 | 241.5 GB | 10.989 |
| `--use_precision_aware_optimizer True --main_grads_dtype bf16 --exp_avg_dtype bf16 --exp_avg_sq_dtype bf16` | 134.0 | 159.5 GB | 10.989 |
| the same and `--gradient_accumulation_fusion True` | 132.8 | 161.9 GB | 10.989 |
| `--use_precision_aware_optimizer True --exp_avg_dtype bf16 --exp_avg_sq_dtype bf16` | 146.0 | 179.8 GB | 10.989 |
| `PRIMUS_RECOMPUTE_NUM_LAYERS=0` | 151.2 | 241.5 GB | 10.989 |
| `USE_TURBO_ATTENTION=True` | 151.3 | 233.8 GB | 10.989 |
| `--use_turbo_autotune True` | 155.8 | 241.5 GB | 10.989 |
| `--use_turbo_norm_te_linear True` | 157.1 | 241.5 GB | 10.989 |
| `--use_turbo_cross_entropy True` | 157.2 | 240.7 GB | 10.989 |
| `--use_turbo_gemm True` | 157.4 | 241.5 GB | 10.989 |
| `--moe_shared_expert_overlap True` | 159.0 | 241.8 GB | 11.002 |
| `--fp8 hybrid` (delayed scaling) | 164.8 | 244.6 GB | 11.389 |
| `USE_TURBO_GROUPED_GEMM=False` | 167.4 | 245.5 GB | 10.989 |
| `--fp8 hybrid --fp8_recipe tensorwise` | 168.4 | 248.3 GB | 11.024 |
| `--gradient_accumulation_fusion True` | 184.3 | 243.0 GB | 10.989 |
| `PRIMUS_MSA_BACKEND=eager` | 189.8 | 257.9 GB | 10.988 |

- The precision-aware optimizer keeps exact FP32 master weights, stored as the BF16 parameter plus a
  16-bit remainder (Megatron's `store_param_remainders`). That and BF16 Adam moments save 62 GB and
  12 ms; a BF16 `main_grad` saves another 20 GB and 12 ms. Profiled with all three, the time comes
  out of TE's Adam (27.8 to 18.7 ms), the gradient `+=` (10.4 to 5.0 ms), the parameter copy after
  the step (5.0 to 0.5 ms) and zeroing the gradient buffer. The losses agree over 10 iterations,
  which says nothing about convergence, and a BF16 `main_grad` also accumulates micro-batches and
  reduces across data-parallel ranks in BF16. Not the default.
- `PRIMUS_RECOMPUTE_NUM_LAYERS=0` does not move the peak on the 4-layer proxy: the recomputed layer
  is dense layer 1, whose activations at sequence 4096 are small next to the weights and optimizer.
- `--use_turbo_autotune` takes iteration 1 from 28 to 35 s.
- `--moe_shared_expert_overlap` has no dispatch to overlap with on one GPU.
- FP8: with the default `delayed` recipe only TE's dense linears run FP8, and Primus-Turbo logs
  `Primus-Turbo FP8 delayed not work since Primus-Turbo not support delayed scaling`, so the experts
  stay BF16. With `tensorwise` that warning is gone and Primus-Turbo's FP8 path covers the experts too.
  Both are slower than BF16 here and start with a higher loss; neither was profiled.
- `overlap_grad_reduce`, `overlap_param_gather` and `manual_gc_interval: 1` run but are slower; see
  Settings measured on the way.
- Combined, on `dev/mi455x_v3` (defaults 156.0 ms there): `USE_TURBO_ATTENTION=True`,
  `PRIMUS_RECOMPUTE_NUM_LAYERS=0` and `--use_turbo_autotune True` ran at 145.2 ms, 233.8 GB, loss
  10.989. Adding the precision-aware optimizer with BF16 `main_grad` and moments and
  `--gradient_accumulation_fusion True` gave 122.0 ms and 172.3 GB, loss 10.988.

Accepted, but they do nothing on this model (loss identical to the defaults):

- `--use_turbo_fused_qk_rmsnorm_rope True` (158.6 ms): the fused op takes full, non-interleaved RoPE
  only, and M3's `rotary_percent: 0.5` falls back to Megatron's path.
- `--use_turbo_mega_moe True` (158.5 ms): applies only at EP 2 to 8, where it does not build (Known
  issues).
- `turbo_sync_free_moe_stage: 1`: sets `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`,
  which the yaml already sets.

Do not run:

- `--turbo_fused_grouped_gemm True` and `--use_turbo_fused_act_with_probs True` assert
  `glu_linear_offset == 0`; M3's `swigluoai` uses 1.0. The second also rules out sync-free stage 3.
- `TURBO_GROUPED_GEMM_BACKEND=bf16:flydsl` on one GPU: `User specified backend FLYDSL cannot handle
  the given inputs`, because 128 local experts are over its 64. It runs at EP2.
- `--fp8 e4m3 --fp8_recipe blockwise` and `--fp8_recipe mxfp8` stop at TE's arch check (`Device arch
  gfx94x or gfx95x is required for FP8 block scaling execution.`, `Device arch gfx95x or newer is
  required for MXFP8 execution.`). TE 2.19.0.dev0 leaves gfx1250 as a TODO until its GEMM is available.
- `USE_TURBO_DEEPEP=True`, sync-free stages 2 and 3, and MegaMoE: Known issues. On one GPU there is
  no dispatch for them to replace either.
- Primus-Turbo attention with its default backend: `invalid argument for fmha_bwd`, so
  `USE_TURBO_ATTENTION=True` makes the launcher use Triton.
- `moe_use_legacy_grouped_gemm`, `PRIMUS_TP` above 1, context parallel, a dataloader attention mask,
  and a BF16 router: Known issues.
- `PRIMUS_TOTAL_LAYERS=8`: does not fit on one GPU, and fills both at EP2.

## Not yet measured

- Sequence 16384 and 32768. MSA's cost stays on the selected blocks; dense attention does not.
- Pipeline parallel. The launcher allows `PRIMUS_PP`, but no PP run has been done. Tensor parallel
  and context parallel are not implemented for MSA.
- EP8 and more than one node.
- DeepEP and MegaMoE, which do not run on gfx1250 yet (see Known issues).
- A real dataset. These runs are mock data.
- The full 60-layer model. From the proxy's ~18 bytes/parameter (BF16 weights, FP32 gradients and
  Adam state), a training step at MBS 1, seq 4096 wants on the order of 40 MI455X GPUs (EP8 x PP5).
  That has not been run.
- The gfx950 MFMA kernels. This node cannot execute them.
- hipBLASLt GEMM tuning of the kind in the GPT-OSS example, including the fp32-output weight-gradient
  GEMMs behind the `gradient_accumulation_fusion` slowdown.

## Known issues

- **TE l2norm and tensors of 2^31 elements.** TE's `multi_tensor_l2norm` reads out of bounds when it
  is given more than one tensor and one of them has 2^31 or more elements. Megatron's grad norm
  passes all gradients in one call, and the Primus-Turbo grouped GEMM keeps a layer's experts in one
  weight: fc1 is 128 x 6144 x 6144 = 4.8e9 elements on one GPU and 2.4e9 at EP2. Without a fix the
  first optimizer step faults (`HSA_STATUS_ERROR_MEMORY_FAULT` in `custom_multi_tensor_l2norm_kernel`),
  or, when the stray address happens to be mapped, training runs with a wrong grad norm (11.365
  instead of 11.406 at iteration 1). The `megatron.optimizer.te_l2norm_large_tensor` patch
  (`primus/backends/megatron/patches/te_patches/l2norm_large_tensor_patches.py`) splits such tensors
  into views of 2^30 elements for the norm. TE's Adam and scale kernels handle these sizes. From EP4
  on, M3's stacked expert weights stay under 2^31.
- **DeepEP.** The DeepEP in this image is written for MI355X (wave64) and has not been ported to
  gfx1250's wave32. `USE_TURBO_DEEPEP=True` on EP2 stops in the first step with `DeepEP error: CPU
  recv timeout`; with TE's grouped GEMM it hung there instead. Sync-free MoE stages 2 and 3
  (`turbo_sync_free_moe_stage`) turn DeepEP on, so they do not run either. Stage 1's two settings,
  `moe_use_fused_router_with_aux_score` and `moe_permute_fusion`, are already in the yaml.
- **MegaMoE.** `--use_turbo_mega_moe True` takes effect only at EP 2 to 8. At EP2 its FlyDSL
  `dispatch_prologue_kernel` does not build for gfx1250 (`lld invocation failed`, then `All autotune
  configs failed`) in the first forward.
- **gfx1250 MSA kernels.** `msa_backend=flydsl` on gfx1250 needs the wave32 WMMA port. It does not
  use `v_permlane32_swap`, MFMA, or `ds_read_tr16`. gfx950 still uses the original kernels; only the
  dispatch changed. The kernels address buffers with 32-bit offsets, so each tensor must stay under
  2 GiB.
- **Eager MSA memory.** Eager materialises the `[batch, heads, seq, seq]` fp32 score matrix and the
  indexer's `[batch, 4, seq, seq]` scores. At seq 4096 that is about 21 GiB for attention forward
  alone. FlyDSL does not.
- **No tensor parallel, context parallel, or dataloader mask.** `PRIMUS_TP` must be 1. FlyDSL rejects
  a dataloader attention mask (`create_attention_mask_in_dataloader: false`). Packed sequences are
  not supported.
- **Router dtype.** With 128 experts, BF16 router logits NaN on the first step
  (`found NaN in local grad norm for bucket #0`). `moe_router_dtype: fp32` is set in the yaml. With
  `overlap_grad_reduce: false` all gradients are in one bucket, so that message does not name a
  parameter.
- **Layer pattern.** `num_layers`, `moe_layer_freq`, `sparse_attention_freq` and
  `sparse_disable_index_value` must be the same length. The launcher writes all four. The first three
  layers are dense attention and a dense FFN (intermediate 12288); later layers are MSA plus MoE
  (expert intermediate 3072, one shared expert). Do not set `moe_use_legacy_grouped_gemm`: the
  activation is `quick_geglu` (`swigluoai`), which the legacy grouped MLP refuses.
- **RoPE cache.** Leave `max_position_embeddings` at the run's sequence length.
