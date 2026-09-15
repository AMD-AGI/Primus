# Gemma 4 pre-training on MI455X (gfx1250)

Single-GPU Megatron-Bridge pre-training recipes for Gemma 4, on MI455X (gfx1250)
hardware. Training goes through `train pretrain` + `pretrain_trainer.yaml` on the
**megatron_bridge** backend, not the native Megatron trainer.

These configs exist because `examples/megatron_bridge/configs/MI300X` has Gemma 4
recipes that assume eight GPUs (TP=8 on the dense model, TP=2/EP=4 on the MoE),
and none of those parallel degrees are legal at world size 1.

## Hardware gate

**Required architecture: `gfx1250` (MI455X).**

| Item | Value |
|---|---|
| ISA | `gfx1250` (`rocminfo`, or `torch.cuda.get_device_properties(0).gcnArchName`) |
| GPUs | 1 (`HIP_VISIBLE_DEVICES=0`, `NNODES=1`, `GPUS_PER_NODE=1`) |
| Env | `PRIMUS_GPU_MODEL=MI455X`, or auto-detect via `runner/helpers/envs/primus-env.sh` |
| Image | `amdprimus/amdprimus:gfx1250-20260910` |
| Memory | 432 GiB HBM, ~464 GB usable |

Do not validate on gfx942 / gfx950 / MI300X / MI355X. The default Primus CI GPU
runners are not this architecture and are not a substitute — treat neither
`run-unittest-torch` nor the TAS GPU end-to-end jobs as gfx1250 coverage.

A passing environment load prints:

```
Detected GPU model: MI455X
Loading MI455X (gfx1250) specific settings...
```

Do not `source runner/helpers/envs/MI455X.sh` on its own; `LOG_INFO_RANK0` is
defined in `base_env.sh`.

## Configs

| config (`examples/megatron_bridge/configs/MI455X/`) | What it is | Weights |
|---|---|---|
| `gemma4_26b_pretrain_1gpu_proxy.yaml` | Gemma-4-26B-A4B MoE shape, **6 of 30 layers** | Random init, mock data, no download |
| `gemma4_31b_pretrain_1gpu_proxy.yaml` | Gemma-4-31B dense shape, **6 of 60 layers** | Random init, mock data, no download |

Depth comes from the model presets `primus/configs/models/megatron_bridge/gemma4_{26b,31b}_6layer_proxy.yaml`,
which inherit every other architectural dimension — hidden size, head dims, the
128 experts and top-k 8 on the MoE, and the sliding-window pattern — from the
canonical presets unchanged. 6 layers preserves the published 5 sliding : 1 full
attention ratio. Override with `PRIMUS_NUM_LAYERS`.

Shared overrides: `TP=EP=CP=PP=1`, sequence parallelism off, `bf16`,
`gradient_accumulation_fusion: false`, `recompute_granularity: none`, mock data,
checkpoint saving off.

**Why these are proxies.** Neither model fits at full depth on one card. The 26B
has a measured depth fit of `memory(GB) = 15.7 + 18.47 x layers`, so 30 layers
needs ~570 GB against ~464 available; the deepest that fits is 24 layers. This is
a machine limit, not unfinished work, and a second GPU relieves it.

## 1. Device verification (before anything else)

```bash
rocminfo | grep -m1 gfx            # must print gfx1250
rocm-smi --showuse --showmemuse    # must return promptly
```

If `rocm-smi` reports telemetry as `N/A`, or any HIP call hangs in
`InterruptSignal::WaitRelaxed`, the GPU is wedged. Reboot before drawing any
conclusion from a training failure — see [Known issues](#known-issues).

## 2. Required environment

```bash
export GEMMA4_CONVERSION_MODE=text
```

Gemma 4's HF architecture class is `Gemma4ForConditionalGeneration`, and
`AutoBridge` dispatches on it, so the text path has to be selected explicitly.

`HIPBLASLT_TENSILE_LIBPATH` **is set for you** by `base_env.sh`, which detects
the misplaced Tensile directory and exports it. You should see:

```
[WARN] hipBLASLt Tensile files are not in the directory hipBLASLt loads them from.
[WARN] Setting HIPBLASLT_TENSILE_LIBPATH=... to avoid the rocBLAS fallback.
```

This matters more than anything else in this document: it is worth **2.93x**
end-to-end. The ROCm SDK wheel installs the Tensile files in
`hipblaslt/library/gfx<arch>/` while hipBLASLt loads them from
`hipblaslt/library/`, so the solution index never loads and
`hipblasLtMatmulAlgoGetHeuristic` returns `HIPBLAS_STATUS_INVALID_VALUE` for
*every* shape. hipBLASLt then looks broken rather than untuned, which invites
the much worse workaround of `TORCH_BLAS_PREFER_HIPBLASLT=0` — that flag does
not keep the stack away from hipBLASLt, so the result is rocBLAS *plus* a
failing hipBLASLt. Set the variable by hand only to override the detector.

If you switch the dense model to TransformerEngine layers, also set:

```bash
export PRIMUS_GEMMA4_TORCH_OPTIM=1
```

TE's `FusedAdam` does not return on this device: `optimizer.step()` sits at 100%
GPU indefinitely while the driver logs `MES(0, 0) ring buffer is full`, and
recovery is a host reboot. This is handled by the `gemma4.local_spec` patch, and
it is independent of the layer spec so TE runs can keep the optimizer on torch.

## 3. Launch

From the Primus root, either start the container yourself and use `direct`:

```bash
export DOCKER_IMAGE=amdprimus/amdprimus:gfx1250-20260910

./primus-cli container --image "$DOCKER_IMAGE" \
  --env HIP_VISIBLE_DEVICES=0 --env NNODES=1 --env GPUS_PER_NODE=1 \
  --env PRIMUS_GPU_MODEL=MI455X \
  --env GEMMA4_CONVERSION_MODE=text \
  -- train pretrain \
  --config examples/megatron_bridge/configs/MI455X/gemma4_26b_pretrain_1gpu_proxy.yaml
```

or, from a shell already inside the image:

```bash
GPUS_PER_NODE=1 ./primus-cli direct -- train pretrain \
  --config examples/megatron_bridge/configs/MI455X/gemma4_31b_pretrain_1gpu_proxy.yaml
```

Useful overrides: `PRIMUS_MBS`, `PRIMUS_GBS`, `PRIMUS_NUM_LAYERS`,
`PRIMUS_TRAIN_ITERS`, `PRIMUS_SEQ_LENGTH`.

## 4. What a pass looks like

- `model.num_layers : 6` in the config dump, and
  `Set config_container.model.num_layers = 6` from the config-override patch
- the two `HIPBLASLT_TENSILE_LIBPATH` lines from §2
- **0 nan iterations and 0 skipped iterations**, and the full iteration count
  reached. This is the validity bar for any quotable number
- loss descending smoothly and reproducibly

Reference points, two clean runs each, 6 layers at sequence length 2048:

| model | mbs | layer impl | dense GEMM | ms/iter | tok/s/GPU | peak VRAM | loss |
|---|---|---|---|---|---|---|---|
| 26B MoE | 16 | TransformerEngine | hipBLASLt | | | 301.6 GB | 15.049 |
| 31B dense | 1 | local | hipBLASLt | | | 85.5 GB | 26.187 |
| 31B dense | 8 | local | Primus-Turbo FlyDSL | | | 220.0 GB | 26.004 |

The 31B rows differ by 3.13x from the GEMM backend alone. On gfx1250 hipBLASLt
has tuned exactly one of the four bf16 contraction layouts, so the dgrad and
wgrad layouts — which carry two thirds of training FLOPs — run at (withheld)
TFLOP/s against (withheld) for the tuned one. The FlyDSL routing that closes
this is not yet part of the repo, so the shipped configs produce the middle row.
It helps dense models only: on the 26B it costs 8%, because a MoE model issues
its expert FFNs through the grouped GEMM path, which never calls `torch.matmul`.

On the 26B, TransformerEngine is the fast path and is the default here, worth
1.69x over the local layers. On the 31B, `local` is 27% faster than TE and uses
11 GB less.

## Known issues

**Do not quote the harness `TFLOP/s` column.** It derives FLOPs from the
*configured* model depth while only 6 layers are built, which overstates the
dense model by roughly 5.3x. For the MoE it errs the other way, since top-k 8 of
128 experts leaves most parameters inactive. The column is meaningless here.

**The GPU wedges intermittently**, on configurations that also succeed. The
signature is `hipErrorLaunchFailure` or a hang, followed by
`MES(N, 0) failed to respond to msg=REMOVE_QUEUE` in `dmesg`; recovery is always
a host reboot, because once `kfd_open` hangs every GPU process blocks and
`/sys/module/amdgpu/refcnt` stays pinned. Two consequences for anyone reading
results here:

- **A single wedge is evidence about the boot, not about the configuration.**
  Four ceilings on the 31B that each rested on one wedge — micro-batch 8,
  12 layers, 18 layers, and TransformerEngine — all ran clean when re-probed.
  Quote rows with two clean runs.
- There is a window of roughly ten minutes in which a wedged GPU still reads as
  healthy, with a single driver message at onset and silence afterwards, so
  message counts and escalation are unreliable as liveness signals.

**Two missing gfx1250 builds need working around**, and both are already set in
these configs. `scaled_masked_softmax_cuda` is not compiled in the image, so
`masked_softmax_fusion: false` is mandatory on the dense model — without it the
run dies with `ModuleNotFoundError` on the first forward. The fused GEMM +
bias-grad path is also absent, hence `gradient_accumulation_fusion: false`. TE's
fused attention is not built either, so TE falls back to
`UnfusedDotProductAttention`; Gemma 4's sliding window and logit softcap are
still fused, through a Triton `score_mod` path.

**Initial loss sits above `ln(262144) = 12.49`** on both models (26B at 15.05
with TE, 31B at 26.0). Both descend smoothly and reproducibly, and the four-way
agreement across two layer impls and two GEMM backends is within 0.040 nats on
the 31B, so throughput comparisons are sound. Absolute convergence quality is
not established by these runs.
