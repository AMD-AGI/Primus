# FP8 HSTU

Two arms, both **on by default** for this config as of 2026-09-10, and both
independently switchable. They are gfx950-only and fail closed, the same
contract `HSTU_ATTN_BWD=flydsl` already carries here: a non-gfx950 launcher
turns them off by env rather than editing this AMD-default config.

**The forward** runs the HSTU linear layers and attention on `e4m3` operands
with per-tensor scales that are calibrated for four steps and then frozen.

**The backward** quantizes exactly one site — the output projection's dgrad —
to `e5m2`, under a scale controller that refreshes rather than freezes. It is a
separate switch because it is a separate measured unit, layered on top of the
forward arm; see [§12](#12-the-backward-arm-one-site-and-why-only-one).

Embeddings, the optimizer, the loss, the metrics and every collective stay in
their existing precision in both arms.

The backward has only ever been measured **on top of** the forward, so the
bootstrap raises on FP8-backward-with-a-bf16-forward rather than running a
combination nothing has benchmarked. Turning the forward off means turning
both off.

This document is the consolidated record of how they work and how they got
there. It replaces the Plan 08, Plan 09 and Plan 10 working documents, which
were written as campaigns ran and carried statuses (`NOT PROMOTED`,
`Draft for review`, `IN PROGRESS`) that no longer describe what shipped. The
measured performance of the shipped arms lives in
[`../perf_opt.md`](../perf_opt.md) — the MI350X evaluation row for the current
figure, the FP8 forward and FP8 backward optimization-log rows for the
per-change accounting. This file is the mechanism and the history behind them.

Deliberately, no measured delta is repeated here or in the source. A number is
a property of a build, a library version and a host; copied into a second place
it goes stale silently, and a reader cannot tell which copy is current.

`FP8_HSTU_FORWARD_REVIEW.html` in this directory is a rendered walkthrough of
the same code aimed at a reviewer reading the diff.

---

## 1. The one knob, and the seventeen behind it

`yambda_5b.gin` binds exactly one thing:

```
apply_env_bootstrap.DLRM_HSTU_FP8_FWD = False
```

Turning it on exports a named preset, `_FP8_FORWARD_ARM` in
`dlrm_v4/train/_env_bootstrap.py`, and exports it **only** when that switch
resolves on — so a bf16 run is byte-identical to one where the section does not
exist. The preset is:

| knob | value | what it does |
|---|---|---|
| `DLRM_HSTU_FP8_GEMM` | `1` | `e4m3` operands for the HSTU projections |
| `DLRM_HSTU_FP8_ATTN` | `1` | `e4m3` q/k/v into attention |
| `DLRM_HSTU_FP8_FUSE_LN` | `1` | input LayerNorm emits scaled `e4m3` |
| `DLRM_HSTU_FP8_FUSE_OUTLN` | `1` | output LayerNorm/dropout emits scaled `e4m3` |
| `DLRM_HSTU_FP8_FUSE_QKV` | `1` | uvqk projection emits bf16 `u` and `e4m3` q/k/v |
| `DLRM_HSTU_FP8_FUSE_SILU` | `1` | SiLU epilogue quantizes in place |
| `DLRM_HSTU_FP8_FUSE_RESIDUAL` | `1` | residual add quantizes in place |
| `DLRM_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM` | `1` | which residual spelling |
| `DLRM_HSTU_FP8_SAVE_UVQK` | `1` | backward re-reads the forward's `uvqk` pack |
| `DLRM_HSTU_FP8_SAVE_UVQK_LAYERS` | `0,1,2` | …for all three HSTU stack layers |
| `DLRM_HSTU_FP8_EARLY_RELEASE_UVQK` | `1` | free the pack right after attention backward |
| `DLRM_HSTU_FP8_SCALE_MODE` | `frozen` | calibrate, then never recompute |
| `DLRM_HSTU_FP8_CALIBRATION_STEPS` | `4` | length of the calibration window |
| `DLRM_HSTU_FP8_HOST_SCALE_VALUES` | `1` | pass frozen scales as host values |
| `DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN` | `1.5` | safety margin on the activation scale |
| `DLRM_HSTU_FP8_FAIL_CLOSED` | `1` | raise rather than serve bf16 under an FP8 label |
| `DLRM_HSTU_FP8_GEMM_FASTACC` | `0` | `use_fast_accum=False`, set up front, not by an A/B (see §3) |

Seventeen gin bindings would be seventeen chances to ship a partial arm, and a
partial arm is a configuration nothing has measured. So setting a compute knob
without the master switch **raises**. That gate is load-bearing rather than
tidy: `DLRM_HSTU_FP8_SAVE_UVQK_LAYERS` is parsed by
`STULayer._allow_saved_fp8_uvqk` on the **bf16** path too and raises on a layer
no `STUStack` owns, so an unconditionally-exported "inert" binding could break a
run that never asked for FP8. Per-knob environment override still wins, so an
A/B can move exactly one knob and hold the rest of the arm.

---

## 2. Why the design is fusion, not quantization

This is the single most important thing to understand, and it is the reason an
earlier piecemeal attempt was abandoned rather than tuned.

Measured on gfx950 before any of this was built:

- prequantized uvqk and output GEMMs were **1.50x and 1.82x** faster than bf16;
- prequantized manual dgrad/wgrad were **1.36x–1.99x** faster;
- but **standalone activation quantization dropped forward speed to 0.15x–0.51x**;
- and **standalone gradient quantization dropped backward speed to 0.03x–0.37x**.

The GEMMs get faster and the casts eat the gain several times over. At HSTU's
activation sizes, quantizing as its own pass means writing the tensor in bf16
and reading it back to convert, and that round trip costs more than the GEMM
saves. There is no version of this that works by adding cast kernels.

So every quantization is folded into the op that already holds the value in
registers: the input LayerNorm, the uvqk projection, SiLU, the output
LayerNorm/dropout, the residual add. `DLRM_HSTU_FP8_GEMM` on its own is
literally inert — it moves board peak by −0.009 GiB and allocated memory by
0.000. The fused path is what produces the win, and `FUSE_QKV` and `ATTN`
together recover 10.23 of the 14.23 GiB the fused path otherwise costs.
Dropping either one of that pair lands back at +14.23 GiB to within 2 MiB: they
are one optimization with two switches.

---

## 3. Precision contract

- **fp32** — canonical parameters, Adam moments, the optimizer update, loss,
  metrics, normalization statistics, control reductions.
- **e4m3** — eligible forward activations and weight shadows.
- **bf16/fp32** — scaled-GEMM outputs and the whole backward.
- `use_fast_accum=False`, set up front with this contract (Plan 08), not chosen
  by an A/B. On ROCm (torch 2.12) tensor-wise `torch._scaled_mm` ignores the
  flag and hipBLASLt accumulates in fp32 either way;
  `DLRM_HSTU_FP8_GEMM_FASTACC=1` only makes the K512 kernels decline.
- No global FP8 autocast, and no silent fallback in any measurement arm.

Weight shadows are keyed by the canonical parameter's identity, layout and
`Tensor._version` — **not** by `data_ptr()`, which is reused by the allocator
and would serve a stale shadow after an optimizer step. Packing of full and
targets-only uvqk layouts happens once per version rather than per call.

---

## 4. What runs where

| file | role |
|---|---|
| `ops/triton/triton_fp8.py` | the FP8 runtime: scaled-GEMM wrappers, dtype/alignment/layout checks, versioned column-major weight shadows, scale state, saturation telemetry, fail-closed dispatch, per-rank capability report |
| `ops/triton/hipblaslt_scaled_addmm.cpp` | the hipBLASLt `torch._scaled_mm` core |
| `ops/triton/triton_layer_norm.py` | input LN and output LN/dropout epilogues that store scaled `e4m3` directly |
| `ops/triton/triton_hstu_linear.py` | the projections, split uvqk production |
| `ops/triton/triton_hstu_attention.py` | FP8 attention forward with explicit descales |
| `ops/triton/triton_hstu_preprocess_and_attention.py` | the fused forward, and where the BLAS-pin hook must sit |
| `ops/hstu_compute.py`, `modules/stu.py` | arm resolution, layer masks, the backward's refusal |
| `dlrm_v4/checkpoint.py` | persists scale state so a resumed run reproduces the trajectory |

The bf16 path must remain byte-for-byte unchanged with the flags off, and
`ops/tests/hstu_compute_test.py`, `layer_norm_test.py` and
`triton_fp8_scale_test.py` are what hold that.

**Attention is asymmetric.** The gin default forward is the FlyDSL e4m3
kernel (`HSTU_ATTN_FWD=flydsl`) with the fused-silu epilogue. `HSTU_ATTN_FWD=triton`
is the opt-out to the Triton e4m3 forward. The backward is the gfx950 FlyDSL
kernel, which consumes scaled `e4m3` q/k/v directly. The qk GEMM defaults to
k128 fp8 MFMA when q and k are already e4m3 (`DLRM_HSTU_FP8_BWD_QK_MFMA=fp8`);
opt out with `=bf16`. So `DLRM_HSTU_FP8_BWD=0` refers to the **projection**
dgrad/wgrad only — the attention backward already reads FP8.

---

## 5. Frozen scales, and why freezing is a performance feature

Scales are calibrated over a bounded warmup, synchronized across ranks, given
the `1.5` margin, and then frozen. Freezing is not a numerical convenience. A
live scale costs an amax reduction per tensor per step **and** a
device-to-host read of the result, and that read is a blocking sync sitting in
the hot path.

Removing it was measured directly. With host scale values on, the control trace
contains exactly three `item -> _local_scalar_dense -> 4-byte D2H` chains per
bucket and the candidate contains **zero**, with bitwise-identical FP8 output:

| tokens (M) | before | after | reduction |
|---|---|---|---|
| 1,024 | 0.053604 ms | 0.034340 ms | 35.94% |
| 2,289,366 | 0.814278 ms | 0.044899 ms | 94.49% |
| 2,450,569 | 0.852508 ms | 0.044179 ms | 94.82% |
| 2,630,727 | 0.910587 ms | 0.043484 ms | 95.23% |

Those four token counts are the real production buckets: ordinary uvqk and
full-row targets-only k/v rows range 2,289,366–2,630,727 with 2,450,569 as the
median representative, while compact targets-only u/q and the final output
projection run at 1,024 rows.

Scale state is checkpointed — the frozen scale, the warmup count, and the
synchronized amax history — so a resumed run reproduces the trajectory. Legacy
checkpoints restart calibration rather than silently running uncalibrated.

---

## 6. Memory, and why the layer mask exists

Enabling FP8 costs **+4.00 GiB** on the worst rank (199.83 → 203.84 GiB against
a 260 GiB planner cap and a 288 GiB board). The whole cost belongs to enabling
FP8 rather than to the saved pack: `fp8_no_save` − control is +3.994 GiB, and
adding the pack is +0.010. It is the quantized copies that have no bf16 tensor
to replace, and it does not shrink with the fusions.

The saved `uvqk` pack has its own history, which is worth knowing because two
plausible fixes failed:

**Lifetime shortening alone does not work.** One FP8 v/q/k pack is 3,131,947,008
bytes at the production fixture, and putting it in `ctx.save_for_backward` keeps
it alive until the entire combined backward returns even though it is dead
right after attention backward. Ephemeral context ownership
(`DLRM_HSTU_FP8_EARLY_RELEASE_UVQK`) does release it — current allocation drops
by the full pack size — but the global high-water mark of 40,072,275,456 bytes
is **already established before backward entry**, while both ordinary layers'
forward packs are live. Early release moves that peak by 1,024 bytes. The fix is
correct and kept, but it is not the memory fix.

**Saving every layer was first rejected on an A2 HBM envelope, then re-measured
as free.** The pack is still a layer *list* rather than a boolean. Mask `1`
won the original A2 sweep (lowest worst-rank HBM, then lowest backward p50).
A later paired A/B of `0,1,2` vs `1` read **−0.612%** of step at **zero extra
HBM** (worst-rank alloc_peak identical). That is the gin default as of
2026-09-10; `$DLRM_HSTU_FP8_SAVE_UVQK_LAYERS=1` restores the single-layer mask.

**Allocator GC tuning was rejected outright.** A `garbage_collection_threshold`
of 0.6 put one candidate peak 243,290,112 bytes below its control, but the
recurring collection stalls regressed step p50 by 2.802%.

---

## 7. What is deliberately not FP8

**Projection wgrads.** The dgrad of the output projection now *does* run FP8 —
that is [§12](#12-the-backward-arm-one-site-and-why-only-one), and it is what
replaced the blanket rejection `stu.py` used to carry. The wgrads did not
follow it, and the reason is layout rather than numerics.

Two implementations were tried and both failed their latency kill gates at
2,450,576 padded rows:

| approach | p50 | bf16 reference | transformed-layout FP8 |
|---|---|---|---|
| custom Triton split-K wgrad | 11.519 ms | 3.935 ms | 2.485 ms |
| direct row-layout hipBLASLt TN | 5.171 ms | 3.973 ms | 2.439 ms |

The direct hipBLASLt result had relative L2 0.03736 and cosine 0.999302 — the
math is fine. Producing both row- and column-major gradient layouts was already
measured slower than bf16, and standalone boundary gradient casts are forbidden
by §2. Reopening this needs a new row-layout vendor solution or a genuinely
fused producer/consumer kernel.

**Communication.** Out of scope; the sparse communication path is not a
byte-sensitive first lever.

**Compact targets-only u/q** stays bf16 in both directions. In the forward,
only the full-row final-layer k/v path is eligible. In the backward its M is
about a thousand rows, where the pair costs four significant figures less than
everything around it — there is no win to have, so it is not implemented rather
than implemented and disabled.

---

## 8. It fails closed, and says so in its own log

The failure mode that matters for a quantization arm is not a crash. It is a run
that reports a bf16 rate under an FP8 label, which reads as "FP8 is barely
faster" and costs a campaign to discover. So:

- `DLRM_HSTU_FP8_FAIL_CLOSED=1` makes an unavailable FP8 path **raise**.
- `[hstu-fp8] resolved:` prints what the forward knobs actually resolved to,
  and `[hstu-fp8] engaged:` prints once per call site.
- Every measurement arm writes an atomic per-rank capability report.
- The runtime probes the backend's FP8 `scale_result` convention once and fails
  closed if output scaling is ignored.

This exists because an earlier round ran six "control" jobs that had silently
inherited the FP8 environment from the submitting shell and reported a −0.24%
A/B before anyone noticed. The check that separates the arms is now mechanical:
3 resolved/engaged lines in every FP8 arm, zero in every control.

---

## 9. Interaction with the bf16 BLAS solution pins

`GR_PIN_BF16_BLAS_SOLUTIONS` pins hipBLASLt solutions for five projection
GEMMs. Every key in that table is a **BFloat16** TunableOp signature, so with
`FUSE_QKV` and `FUSE_OUTLN` on, the uvqk and output projections become
`torch._scaled_mm` calls that no key can match — `uvqk_fwd` and `out_fwd` stop
applying. The pins are worth −3.66% with FP8 off and −2.86% with it on, so
leave them on in both arms and do not add the two gains.

**The backward arm takes a third pin out the same way.** With
`DLRM_HSTU_FP8_BWD` on, the output projection's dgrad also becomes a
`_scaled_mm` call, so `out_dgrad` stops applying and only `uvqk_wgrad` and
`out_wgrad` still carry value. This is visible in the paired traces behind the
FP8-backward row: the control's dgrad runs the pinned `Custom_Cijk_*` solution,
the FP8 arm's runs an `F8B8BS` one. It also explains why that arm's gain is
smaller than an offline probe predicts — a probe times FP8 against a *stock*
bf16 GEMM, but the baseline being displaced here is a pinned one, so the pin had
already banked part of the win.

In all three cases the launch log still reports the pins as installed. That line
says they were registered, not that a runtime key matched them.

There is an **ordering hazard** here that is silent. TunableOp fixes an
operator's solution the first time it resolves a key, and the key includes the
jagged token count, so `install_for_tokens` must run before the first GEMM at
every new token count. FP8 moves the projection earlier in
`_HSTUPreprocessAndAttentionFunction.forward`, into the `use_fp8` branch, so the
hook has to sit ahead of that branch. Getting it wrong produces no error and no
log line: the pins install, report active, and miss.

---

## 10. The `_scaled_mm` dispatch tail, and why it is left alone

Section 9's ordering hazard is a pinning problem one level up from a property of
hipBLASLt itself: its solution cache is keyed on the full problem shape, and on
this model `M` is the jagged token count. The consequence is visible in any
profile of the shipped arm and has already been reported as a regression, so it
is written down here.

**What it looks like.** In the step-52 capture (8 ranks x 5 steps, 40
rank-steps) `aten::_scaled_mm` is called 280 times for **785 ms** of host time,
**19.6 ms per rank-step**. The distribution is bimodal — median **135 us**, p90
**8.7 ms** — and the 38% of calls above 1 ms carry **96.7%** of the total.

**Why.** Splitting by shape resolves it completely:

| | calls | slow (>=1 ms) | host time |
|---|---|---|---|
| first use of an `(M, K, N)` on a rank | 120 | 107 (89%) | 766.4 ms |
| repeat at a shape already seen | 160 | 0 (0%) | 19.0 ms |

There were **40 distinct `M`** across those five steps, all in the 2.22–2.38 M
range and never repeating, because `M` is the token count. So nearly every call
is a first use, hipBLASLt runs its heuristic, and the run **never warms up** —
this is steady-state cost at step 52, not calibration or autotune warmup, and it
is flat across all five steps.

**Why it is not worth fixing.** The step is GPU-bound. Occupancy is **97.7%**
(303.0 ms busy of 310.2 ms), so the host runs far enough ahead that only
**0.76 ms per rank-step** of that stall is exposed as idle GPU — **0.25%** of the
step. Total GPU idle is **7.11 ms**, which bounds *every* host-side stall in the
step at 2.3%, not just this one. The weight-shadow refresh is the same story and
starker: **5.46 ms** per rank-step of host time, **0.00 ms** exposed.

Bucketing or padding `M` would collapse the distinct shapes and recover most of
the 19.6 ms of host time. It would also buy approximately nothing, and it would
put a padding contract on the hot path. Treating the 19.6 ms as recoverable is
the misreading to avoid.

**Reproduced.** Job **31774**, a fresh 80-step capture at this PR's head on the
same host, gives the same picture on an independent run — 8 ranks x 5 steps,
280 calls over 40 distinct `M`, identical to the original:

| | job 31774 | job 31617 |
|---|---|---|
| `_scaled_mm` host / rank-step | 20.22 ms | 19.64 ms |
| calls >= 1 ms | 36.8% | 38.2% |
| GPU occupancy | 97.6% | 97.7% |
| **exposed as GPU idle** | **0.95 ms (0.30%)** | **0.76 ms (0.25%)** |
| FP8 `F8F8` gemm share | 4.1% | 4.3% |
| `bwd_dkdv` share | 16.4% | 17.2% |

Exposure is uneven across ranks (0.00 ms on rank 5, 2.46 ms on rank 0, which
also carries the most collective wait), so read it as a per-rank mean rather
than a fixed cost. Either way it is under a third of a percent of the step.

**The mechanism, tested directly.** Calling `torch._scaled_mm` outside the
trainer on the same container and stack, at the HSTU `K`/`N` with a realistic
`M`, separates the cache from the work — one arm holds `M` fixed, one moves it
by a single row per call, and a third returns to the first arm's shape:

| arm | median host time |
|---|---|
| same `M` every call | **13.0 us** |
| `M + 1` each call (never-seen shape) | **741.2 us** |
| back to the first arm's `M` | **14.5 us** |

That is **57x** on an unseen shape at `N=512` and **60x** at `N=1536`, and the
third arm is the part that settles it: returning to a shape already resolved is
fast again, so this is a cache keyed on `M`, not a shape that is intrinsically
harder to solve. The trainer's own logs corroborate it from a third direction —
an 80-step run writes **~145 MB** of hipBLASLt solution chatter.

**Where the time actually is,** for anyone arriving here looking for step time —
of **326.7 ms** of GPU kernel work per rank-step:

| kernel | ms/rank-step | share | launches |
|---|---|---|---|
| `bwd_dkdv_kernel_0` (attention backward) | 56.3 | 17.2% | 2 |
| bf16 hipBLASLt GEMMs (mostly backward) | 62.2 | 19.1% | 41 |
| `_hstu_attn_fwd` | 23.0 | 7.0% | 3 |
| **FP8 `F8F8` GEMMs** | **14.0** | **4.3%** | 5 |

That last row is section 2's argument restated from the shipped arm: with every
fusion engaged, the FP8 matmuls are a twentieth of GPU time. The concentration
in `bwd_dkdv_kernel_0` is the most promising single target in the file, and it
is backward, which this change does not touch.

---

## 11. History: what the campaign measured, including what it rejected

The isolated kernel stack was always the strong result. At the fixed production
shape (2,039,028 tokens, three layers, targets-only final layer):

| | forward | backward | combined |
|---|---|---|---|
| bf16 control | 59.722 ms | 104.746 ms | 164.535 ms |
| FP8 selected | 37.956 ms | 100.072 ms | 139.448 ms |
| reduction | 36.4% | 4.5% | **15.2%** |

Turning that into whole-trainer step time is where the campaign spent its
effort, and it is why several attractive-looking variants are not here:

- **Saved forward `e4m3` v/q/k for both layers** improved the kernel stack
  further, to 134.006 ms combined (18.6%), and passed its isolated gates — but
  produced trainer p50 reductions of 0.327%, 9.559% and 3.717%, and exceeded
  the HBM envelope in two of three repetitions. Banked.
- **Predecoding bf16 Q into scratch** gave 143.359 ms combined / 103.417 ms
  backward, both worse than feeding FlyDSL directly.
- **The gfx950 packed FP8→bf16 intrinsic** is exponent-scaled only and cannot
  represent an arbitrary calibrated scale. It also pushed two attention backward
  calls to 47.396 ms against 40.475 ms for in-kernel conversion. The `iglp3`
  schedule is what is retained.
- **A 48-configuration Triton forward-attention search** reselected the pin
  already in place (`BLOCK_M=128`, `BLOCK_N=32`, `matrix_instr_nonkdim=16`,
  `num_warps=8`, `num_stages=2`): 6.921 ms against the pinned 6.908 ms.
  Removing sort-by-length scheduling moved p50 by −0.011 ms (−0.156%).

Trace attribution on the trainer confirmed HSTU itself was genuinely faster at
matched production shapes — 16.75% median reduction, passing in 24 of 24 trace
cells — so the gap between 15% isolated and low-single-digit whole-step was
overlap and exposure, not a phantom kernel win.

### An open item worth knowing

The campaign's own declared promotion gate was three interleaved 80-step pairs
with **every** pair positive and a median whole-step p50 reduction of at least
7%. That gate was never cleared: the pre-fusion candidate returned 3.451% /
5.717% / −2.823%, and the final three-pair attempt on the improved candidate was
lost to an unrelated host OOM (2.37 TiB consumed by other containers on the
node) before any measured pair ran, leaving no verdict.

What shipped was instead validated by the ordinary `perf_opt.md` protocol —
four arms, three interleaved 200-step rounds, one host, 57 matched steps — which
measures **−4.63%** on the current trunk, with wall clock to AUC 0.75 at 1.036×
and samples-to-threshold identical to the sample on all three convergence seeds.
That is a real, significant, reproduced result, and it is also **below the 7%
the campaign said it would require**. Whether the original gate should still
bind is a live question rather than a settled one; it is recorded here so nobody
re-derives the discrepancy from scratch.

### Numerical evidence carried forward

- Whole-trainer max absolute step-loss delta against bf16: **1.0e-5**, zero
  nonfinite tokens.
- Convergence: samples to AUC 0.75 identical seed for seed against the paired
  bf16 arm (68,812,800 / 75,694,080 / 61,931,520), AUC within 1.4e-3.
- All 40 AUC envelopes, per-step gradient 2σ envelopes and matched loss
  tolerances passed across six interleaved 200-step trajectory runs.

---

## 12. The backward arm: one site, and why only one

`yambda_5b.gin` binds a second switch, independent of the forward one:

```
apply_env_bootstrap.DLRM_HSTU_FP8_BWD = False
```

That switch is the arm. `ops/triton/bwd_fp8.py` reads it directly and carries
the tunables below; there is no preset to expand and no partial arm to
assemble. The forward keeps its two-tier preset because seventeen knobs can
be combined into something unmeasured, and one knob cannot.

The four below are tunables rather than gates -- moving one is an A/B against
the arm, not a way to half-enable it. The margin is kept at a generic value; see
[the scale subsection](#the-scale-refreshes-it-does-not-freeze):

| knob | value | what it does |
|---|---|---|
| `DLRM_HSTU_FP8_BWD_MARGIN` | `8` | generic power-of-two headroom over the previous window's amax |
| `DLRM_HSTU_FP8_BWD_REFRESH_CALLS` | `200` | backward calls between refreshes |
| `DLRM_HSTU_FP8_BWD_WARMUP_CALLS` | `4` | calls run bf16 before engaging |
| `DLRM_HSTU_FP8_BWD_GRAD_FORMAT` | `e5m2` | gradient format; weights stay `e4m3` |

The code is `ops/triton/bwd_fp8.py`, deliberately separate from
`triton_fp8.py`: that module is the promoted forward arm and the two should not
share a blast radius while this one is younger.

### Why only the dgrad

`hipblaslt_scaled_addmm` enforces TN — both operands contiguous along K. That
single constraint decides the whole shape of this arm.

- **dgrad** is `grad[M,N] @ weight.T[N,K]`, so K is the small feature dimension.
  The gradient is already row-major and the weight shadow is built once per
  optimizer step. **Nothing has to be transposed.**
- **wgrad** is `act.T[K,M] @ grad[M,N]`, so K is the multi-million-row dimension.
  Both operands are stored row-major over `[M, feature]`, contiguous along
  neither's K. **Both need re-laying out**, and a transposed quantization runs at
  a small fraction of the row-major rate — the conversion costs more than the
  GEMM saves.

So the plan's original premise, that one quantization feeds both consumers, is
half right: the dgrad wants the gradient row-major and the wgrad wants the same
tensor column-major, and those are different buffers. `torch._scaled_mm` adds a
wgrad-only obstacle on top — it rejects a `mat1` whose trailing dimension is not
a multiple of 16, and for the wgrad that dimension is the jagged row count.

### The scale refreshes; it does not freeze

The forward's calibrate-then-freeze policy does not transfer. The gradient amax
**grows by orders of magnitude** over a run, so a frozen scale drifts into large
relative error. This controller refreshes on a cadence at a margin: the scale
in use comes from earlier windows, so it cannot follow a jump inside the current
one, and the margin is what absorbs such a jump.

**Why the margin is 8.** 8 is not tuned to the data. On synthetic data the
shipped controller clips nothing anywhere from margin 4 to 32, so the value is
not load-bearing. It is a generic (power-of-two) safety margin, and the
pre-registered re-derivation's own variants span 4 to 64. The synthetic check
(r14 fp8ctl, job 70825) replays the shipped controller -- scale = margin × the
previous window's max, warm-up calls 1-4, then a refresh every 200 calls -- and
finds no clipped call at margins 4, 8, 16 or 32 for any site class; the largest
requirement is 3.47 (uvqk; 4 as a power of two). The pre-registered
point-sample rule itself returned 32 (R-MARGIN = DIFFERENT), driven by
vk_projection bursts on the synthetic generator's lightest batches; keeping 8 is
a decision made on the shipped controller semantics above, not that rule's
output. 8 was first picked during Plan 10 from gradient telemetry on the
training data, rounded up to a power of two; under the rule against offline
real-data constants that measurement is not its justification, and the value is
kept only as the generic margin described here.

Refreshing needs an amax, and a separate reduction pass over a multi-GB tensor
would cost about what the GEMM saves. It does not need one — the cast already
reads every element, so the reduction rides along for a negligible share of it.
The controller never reads a device value back to the host in the steady-state
step; the one host read happens on the refresh boundary, and it is what would
show the amax growth happening in production rather than in a simulation.

### `e5m2` for gradients, `e4m3` for weights

This split is the standard FP8 training recipe, arrived at here the long way
round. `e4m3` was chosen for gradients first, on spike data — evidence that only
describes the *top* of the range. In a float format the margin costs nothing in
mantissa, so its entire price is paid at the bottom, and scoring the bottom
against the gradient histograms showed `e4m3` flushing a large fraction of this
gradient to zero. Adam normalizes per parameter, so a flushed coordinate gets no
update at all, and paired seeds duly came in behind the control on
samples-to-AUC. `e5m2` spends one mantissa bit for four decades of range and
moves that underflow fraction by orders of magnitude. Weights keep `e4m3`: their
range is narrow, so for them the mantissa bit is the better buy.

### What this arm does not establish

**Convergence is measured, and the arm is neutral.** Three paired seeds to AUC
0.75, with the control being the FP8 forward arm and a bf16 backward, so only
this arm's `e5m2` gradient differs between the two. Samples to threshold are
identical seed for seed, and the same counts fall out of a bf16-trunk arm as
well. "Identical" here means within one eval boundary, which is the resolution
the forward arm's equivalent claim also carries -- not bit-identical
trajectories. Counts and crossing AUCs are in
[`../perf_opt.md`](../perf_opt.md).

**That the gain is large.** It is not. The site-level saving is a small
single-digit fraction of a percent of the step, close to the resolution of a
200-step harness — an earlier campaign on this same arm read the effect as
indistinguishable from zero across 23 pairs and withdrew a larger claim it had
made. What makes the current figure credible is the paired trace attribution,
not the rate on its own. Both live in
[`../perf_opt.md`](../perf_opt.md).

**Engagement is not provable from the log alone.** `[hstu-fp8-bwd] engaged:` is
emitted **once per site**, so it says the site engaged at least once, not that
it engages every step. Several paths return `None` and fall back to bf16 with no
log line: the warmup window, a scale not yet published, a weight without a
stable identity, a trailing dimension not divisible by 16. Counting
`_scaled_mm` calls in the backward phase of a trace is the only check that
establishes steady-state engagement.

---

## 13. Where the raw evidence lives

The campaign artifacts are on shared NFS, outside this repo:

```
/scratch/dlrmv4/plan08/job76322-fp8-delta-v20-cert
/scratch/dlrmv4/plan09/job76322-early-release-v1
```

The harnesses that produced them (`plan0N_*` probes, stage gates, container
entrypoints and analyzers) were single-use and are deliberately not tracked in
this repository. Reproducing a specific gate means recovering them from the
branch history rather than running them from a checkout.

The Plan 10 backward campaign followed the same convention: its ceiling
reprofile, offline shape probes, gradient telemetry, underflow scorer and A/B
drivers lived in `scripts/plan10_*` and in a `plan10_grad_telemetry` module
threaded through three backward sites. All of it is out of the tree. Two of
those sites — `ordinary_uvqk` and `targets_only_full_vk` — only ever carried
telemetry and never had an FP8 path, so removing it returned `triton_addmm.py`
and `triton_hstu_preprocess_and_attention.py` to their pre-campaign state; what
shipped is one call site in `triton_hstu_linear.py`.

The one artifact kept in-tree is the paired step-52 trace behind the
FP8-backward row, `traces/mi350x/1_n/trace_step52_fp8bwd.json.gz`, because the
evaluation row links to it and the kernel-level attribution is not re-derivable
without it.
