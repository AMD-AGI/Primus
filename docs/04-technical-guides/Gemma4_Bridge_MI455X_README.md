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

Reference fingerprint for the shipped configs, 6 layers at sequence length 2048,
both measured on this revision. Check your run against these two columns rather
than against a step time, which varies with the container image.

| config | mbs | layer impl | dense GEMM | peak VRAM | loss at iteration 8 |
|---|---|---|---|---|---|
| 26B MoE proxy | 16 | TransformerEngine | hipBLASLt | 301.6 GB | 15.048 |
| 31B dense proxy | 4 | local | hipBLASLt | 149.3 GB | 26.056 |

**Peak memory is the reliable half of this fingerprint; treat the loss as good
to about three decimals, not exact.** Memory has reproduced to 0.01 GB through
repeats, a rebase onto a much newer `main`, and a change of transformers
version. Loss is stable across repeats of one revision, but the 26B moved from
15.049 to 15.048 across that rebase with the configuration untouched, so a
difference in the third decimal means the code moved underneath you, not that
your run is wrong. A difference in the *first* decimal does mean something is
wrong.

The versions these came from are worth stating, because they are not the ones
you will get by default. Both rows were measured with **transformers 5.12.1**,
which is what our container image ships, while the pretrain hook's
`requirements-megatron_bridge.txt` pins **5.10.1** — and `primus-cli direct`
bypasses the hook that installs the pin, which is why the image's version wins
in our runs. Both are inside Megatron-Bridge v0.6.1's `>=5.8,<=5.12.1` range.

We checked the pin rather than assuming: on an image identical except for
transformers downgraded to 5.10.1, the **31B reproduced both numbers exactly**
(149.3 GB, loss 26.056). So the 31B fingerprint is version-independent across
that range. The **26B is unconfirmed on 5.10.1**, and not for lack of trying:
both attempts lost the device to the wedge described below, on two different
boots, each time after the model had built and entered the training loop. Two
failures on one row is not proof that 5.10.1 is at fault — that row is also the
heaviest config here at 301.6 GB, and this host wedges on roughly half of all
attempts regardless of what is running — but it is the only row we could not
land, so treat the 26B fingerprint as established on 5.12.1 only.

**Absolute throughput for this part is deliberately not published here.** What
this document is for is the ratios below, and unlike a tokens/s figure they
survive a change of container image or host — we have one case where the same
configuration measured 27% apart purely from an image version.

| change | effect on step time | applies to |
|---|---|---|
| `HIPBLASLT_TENSILE_LIBPATH` pointed at the real directory | **2.93x** | both |
| TransformerEngine instead of local layers | 1.69x | 26B only |
| local layers instead of TransformerEngine | 1.27x, and 11 GB less | 31B only |
| recompute off | 1.13x | 26B |
| micro-batch 1 to 16 | 1.94x | 26B |

Every row above is reproducible with what this PR ships, which is why the
dense-GEMM routing described next has no row of its own.

There is a known headroom item that is deliberately left unquantified here.
hipBLASLt has tuned exactly one of the four bf16 contraction layouts on
gfx1250, so the dgrad and wgrad layouts that carry two thirds of training FLOPs
run roughly 15x slower than the tuned one, and routing them to a Primus-Turbo
FlyDSL kernel instead recovers a large multiple on the dense 31B. We are not
printing that multiple, because **the routing is not in this PR** and a number
you cannot reproduce from the tree in front of you is worse than no number: our
measurements of it predate a 119-commit rebase, and the figure moves with batch
size besides. It will be published alongside the change that implements it.

If you went looking for it, note that `PRIMUS_GEMMA4_TURBO_LINEAR` does nothing
on this tree — the patch module is not registered, so setting it is a silent
no-op rather than an error. We verified that directly: mbs 4 with the variable
set and unset came out within 2% of each other, which is this host's run-to-run
noise.

Two things about that routing are worth knowing before you invest in it. The
effect grows with batch size, as the step becomes more GEMM-bound. And it helps
dense models *only* — on the 26B it was a measurable regression, because a MoE
model issues its expert FFNs through the grouped GEMM path, which never calls
`torch.matmul`, so the wrapper pays dispatch cost on every matmul while seeing
only the attention projections.

Exclude the first iteration from any timing you do take: it is roughly 4.6x the
steady-state step on the 31B, because it includes Triton compilation.

**Do not raise the 31B micro-batch without re-measuring.** On the path this PR
ships, mbs 8 has wedged the device twice out of two attempts — once minutes
after two clean mbs 4 runs on the same boot, which is the only comparison in
this document where boot state is held fixed. It did run clean twice under the
out-of-tree GEMM routing described above, which is suggestive but not something
you can act on from here.

On the 26B, TransformerEngine is the fast path and is the default here, worth
1.69x over the local layers. On the 31B, `local` is 27% faster than TE and uses
11 GB less.

## Known issues

**Ignore the harness `TFLOP/s` column.** It derives FLOPs from the *configured*
model depth while only 6 layers are built, which overstates the dense model by
roughly 5.3x — it reports a figure for the 31B that is not physically achievable
on this part. For the MoE it errs the other way, since top-k 8 of 128 experts
leaves most parameters inactive and counting `8N` over total parameters is the
wrong yardstick. The column is meaningless for a layer-reduced proxy in both
directions.

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

This is **not specific to Gemma 4 or to this backend.** The DeepSeek-V4 gfx1250
bring-up in `examples/deepseek-v4/run_deepseek_v4_pro_muon_local.sh` carries
workarounds for the same unrecoverable-MES failure from four unrelated triggers:
a tuned hipBLASLt bundle deadlocking on a backward-FP8 split-K kernel, an SDMA
host-to-device copy that never signals completion, a Triton MoE permute autotune,
and an alignment-sensitive copy path. Worth reading if you hit this, for two
reasons. Its SDMA case presented with a **completely clean `dmesg`**, which is
further evidence that no driver-log signal is sufficient to detect the wedge. And
having first blamed the permute fusion, that script now records the SDMA defect
as the likely true cause with "permute fusion possibly innocent" — the same
misattribution this guide warns about, reached independently.

One untested lead from that work, if wedges are blocking you: `base_env.sh`
defaults `HSA_ENABLE_SDMA=1`, and blit-kernel copies (`HSA_ENABLE_SDMA=0`) are
slower but avoid the SDMA queues implicated above. We have not measured this on
Gemma 4 either way.

Do **not** set `AMD_SERIALIZE_COPY=3`. `MI455X.sh` sets it to 0 deliberately; the
DeepSeek-V4 measurements put its cost at 38% throughput, and the iteration-1
wedge it once worked around no longer reproduces.

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

## See also

Other MI455X (gfx1250) work in this repository, all single-GPU:

- [Native SFT LoRA on MI455X](LoRA_Native_Trainer_MI455_README.md) — LoRA/SFT on
  the native Megatron trainer, with Llama-3.2-1B plus 70B/72B/235B layer-reduced
  proxies. Uses the same MI455X environment detection and the same
  `gradient_accumulation_fusion: false` workaround. It requires
  `PRIMUS_TURBO_ATTN_BACKEND=triton` because its image has no aiter; the Bridge
  path here does not, since these configs leave Turbo attention off. That guide
  is deliberately free of throughput numbers, so do not look for a perf
  comparison between the two.
- `examples/deepseek-v4/run_deepseek_v4_pro_muon_local.sh` — DeepSeek-V4-Pro
  bring-up. Documentation is in the script comments rather than under `docs/`.
  Source of the wedge corroboration above, and of several gfx1250 environment
  settings that `base_env.sh` and `MI455X.sh` now apply for you
  (`HSA_NO_SCRATCH_RECLAIM=1`, single-GPU NCCL loopback, `AMD_SERIALIZE_COPY=0`).
- `examples/deepseek-v4/projection/` — trace-driven projection from measured
  MI355X runs to MI455X. Note its MI455X compute peak is an **estimate**
  (~10 PFLOP/s bf16, unpublished), so treat its MI455X tab as scaling intuition
  rather than measurement.

Two gfx1250 facts from that work that generalise: Primus-Turbo's gluon and
FlyDSL attention kernels are **gfx950-only**, and `torch.distributed.all_reduce`
with `op=AVG` hangs on some gfx1250 builds even at world size 1, which matters
for MoE auxiliary-loss reductions.
