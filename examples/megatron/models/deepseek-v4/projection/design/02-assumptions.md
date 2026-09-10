# 02 — Assumptions (single source of truth)

Every assumption baked into the projection. When a projection number looks off,
start here.

## Optimizer / DP

- **A1.** Production optimizer modeled = **AdamW + distributed optimizer (zero1)**.
  Muon is out of scope. NOTE on **capture**: the trace itself is taken with
  `use_distributed_optimizer=False` (+ fp32 states) because with dist-opt ON the
  ROCm Kineto profiler drops the compute GPU kernels for pure dense(cr=0)/HCA
  (cr=128) layers (CSA cr=4 is unaffected). dist-opt does not change the fwd/bwd
  compute, so this only affects which kernels Kineto records; the optimizer step
  is modeled analytically regardless (A3).
- **A2.** DP communication (param all-gather / grad reduce-scatter) is **fully
  hidden** behind compute at large GA. We do not add a DP comm term. *(Optimistic;
  primary calibration target.)*
- **A3.** The optimizer step is a per-iteration term, **not** multiplied by GA or
  replicated per PP microbatch. It scales with **per-rank optimizer parameter
  count**: full-model params are first averaged over PP/TP ownership, then
  sharded over DP under ZeRO-1. CP does not shard parameters. The modeled Adam
  traffic uses the full mixed-precision read/write cost per parameter and is
  treated as memory-bound.

## Pipeline / parallelism

- **A4.** PP point-to-point comm is **hidden**; only the pipeline **bubble**
  remains. CP/TP comm not modeled in v1 (TP=1, CP=1 in the V4 release configs).
- **A5.** Pipeline bubble fraction uses 1F1B: `(PP-1)/GA`; interleaved VPP divides
  it by the VPP degree: `(PP-1)/(GA*VPP)`.
- **A6.** Per-stage time is the **sum of the specific cr-type layers** mapped to
  that stage; the iteration critical path is driven by the **max** (slowest)
  stage, plus embedding on stage 0 and output/loss on the last stage.

## EP / MoE

- **A7.** EP dispatch/combine has **no overlap** with compute (current stack) and
  is counted in full, every microbatch.
- **A8.** No MoE deepep-comm + grouped-gemm overlap; MoE = dispatch + grouped_gemm
  + combine summed.
- **A9.** EP is intra-node only (e.g. EP=8 within an 8-GPU node). Cross-node EP is
  out of scope for v1; if EP spans nodes the dispatch/combine cost model must
  change (RDMA, different bandwidth).
- **A10.** MoE per-layer cost is `cr`-independent; the three single-cr traces
  must agree on it (cross-check). The site uses one MoE breakdown for all layers.

## Trace capture / attribution

- **A11.** One trace per cr (`0`, `4`, `128`), **1 layer**, `seq=4096`,
  `recompute off`, profiler window iter 6->7 (post warmup/autotune).
- **A12.** Capture with comm-overlap **off** (`num_layers=1` breaks Megatron's
  chained param sync, and compute is already clean without overlap). GA=2; clean
  per-kernel time = `min` over launches grouped by `(module, phase, shape)`,
  removing residual jitter.
- **A13.** Kernel -> nn.module attribution uses `with_stack=True`: GPU kernels are
  linked to their launching CPU op via trace flow events, and the module is read
  from the CPU op's python call stack. **fwd/bwd phase** is determined by, in
  priority order: (1) a `_fwd_`/`_bwd_` tag in the kernel name; (2) for linked
  kernels, whether the launching CPU op's timestamp lies inside an
  `autograd::engine::evaluate_function` interval (= backward); (3) for unlinked
  kernels (no `External id`), whether the kernel's GPU timestamp lies in the
  backward GPU-time window reconstructed from the linked backward kernels. The
  old rule (default-to-forward when unlinked) leaked backward compute -- incl.
  the MoE dgrad/wgrad grouped GEMMs -- into forward; see `design/06`. One-off
  device stalls billed to a compute kernel (> `_MAX_PLAUSIBLE_LAUNCH_US`) are
  dropped as artifacts and reported in `provenance.dropped_stall_us_per_mb`.
- **A14.** Only `gemm`, `grouped_gemm`, `attn` kernels get a TFLOPs number; all
  other kernels are memory-bound (TFLOPs = null) and contribute time only.
- **A15.** Embedding / output-logits / loss are taken **once** (from any single
  trace), not triple-counted across the three cr traces.

## Recompute

- **A16.** Traces are captured with recompute **off** (pure fwd, pure bwd). At
  projection time, a recomputed layer's backward gets `+1 forward` of that layer
  added back. Recompute selection (#layers / which) is a site control.

## MTP

- **A17.** MTP is modeled analytically when `mtp_num_layers > 0`. Each MTP
  depth uses `mtp_compress_ratios` (default cr=4, matching the current Flash
  Megatron FLOPs anchor), plus the per-depth `eh_proj`, extra logits/loss, and
  HyperHead FLOPs. Timing is an approximation: the MTP inner layer reuses the
  measured layer time for that cr, while `eh_proj` is scaled from output-GEMM
  throughput. A dedicated MTP trace remains the next calibration step.

## Scope exclusions (v1)

- **A18.** No cross-node EP; no TP; no CP comm modeling.
- **A19.** FP8 / MXFP8 not modeled; BF16 only.

## MI355X -> MI455X scaling

- **A20.** Compute-bound kernels (`gemm`, `grouped_gemm`, `attn`) scale by the
  **peak-TFLOPs ratio** (BF16) `t_mi355 / t_mi455`.
- **A21.** Memory-bound kernels (everything else, incl. optimizer) scale by the
  **HBM-bandwidth ratio** `bw_mi355 / bw_mi455`.
- **A22.** A single tunable **efficiency factor** (default 1.0) multiplies the
  scaled compute time to account for MFU differences on new HW (flat peak ratio
  is optimistic).
- **A23.** Comm (EP/DP/PP) is not rescaled in v1 (intra-node EP only; DP/PP
  hidden).

## Validation

- **A24.** Self-consistency: configured to the trace scenario (PP=1, EP=8,
  measured GA, single node) the model must reproduce the measured single-node
  iteration time. Multi-node calibration is deferred.
