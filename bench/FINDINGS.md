# KDA `flydsl` vs `fla`: measurement, one optimisation pass, and what is left

Hardware: 1× MI355X (`gfx950:sramecc+:xnack-`) on `crsuse2-m2m-258`,
`docker.io/rocm/primus:v26.4`, torch `2.12.0+rocm7.14.0a20260608`,
flydsl `0.1.6`, **fla `0.5.2`** (the version `requirements.txt` pins; earlier
passes in `/shared_nfs/botao/kimi_k3/wp9_p*` used 0.4.2, and 0.5.2's backward is
faster, so ratios here are not directly comparable with theirs).

Harness: `bench/bench_kda_backends.py`. `torch.cuda.Event`, median of 20 after 5
warmups, forward under `no_grad`, backward by subtracting the forward from a
fwd+bwd. Launch counts come from `torch.profiler`, the backward's from a graph
replayed with `retain_graph=True` so the forward's kernels are not counted twice.

**Every speed number below is one benchmark process per (shape, dtype).** That is
how production runs — one KDA geometry per process — and it also removes Dynamo's
cross-shape recompiles as a confound, which a single sweeping process cannot: the
same code measured 19/67 launches per shape in isolation and 21–27/77–97 when
seven geometries shared one process.

---

## 1. Three backends at the geometries Kimi K3 actually trains at

bf16, one process per shape, on an otherwise-idle GPU. Shapes are
`[B, T, H, K, V]`; KDA is full-width with no GQA, so `H = num_attention_heads`
and `K = V = linear_key_head_dim = 128` (`primus/configs/models/megatron/`).

| shape | backend | fwd µs | bwd µs | fwd+bwd µs | launches fwd/bwd | peak GB |
| --- | --- | --- | --- | --- | --- | --- |
| `prod_T2048` `[1,2048,96,128,128]` | `eager` | 91063 | 270980 | 362043 | 13690 / 38515 | 27.19 |
| | `fla` | **714** | **1596** | **2310** | **8 / 13** | **1.32** |
| | `flydsl` | 1048 | 4366 | 5415 | 19 / 67 | 2.32 |
| `curve_mbs8` `[8,2048,16,128,128]` | `eager` | 87241 | 299072 | 386313 | 13754 / 38579 | 36.22 |
| | `fla` | **857** | **1997** | **2853** | **8 / 13** | **1.76** |
| | `flydsl` | 1332 | 5488 | 6820 | 19 / 67 | 3.09 |
| `curve_mbs16` `[16,2048,16,128,128]` | `eager` | 136797 | 388681 | 525478 | 13754 / 38579 | 72.44 |
| | `fla` | **1530** | **3524** | **5054** | **8 / 13** | **3.51** |
| | `flydsl` | 2524 | 10692 | 13217 | 19 / 67 | 6.18 |

`eager` is the reference, not a candidate: it is 70–100× slower than `fla` and
needs 27–72 GB for one layer's fwd+bwd, because it builds the intra-chunk score
matrices one column at a time and keeps every column alive through the backward.
Its value is as the numerical oracle in §3.

## 2. Numerical parity — `flydsl` is the more accurate of the two

Both fused backends compared against `eager_chunk_kda` evaluated in **fp32** on
the identical (bf16-rounded) inputs, so the difference is the kernel's own
arithmetic. `max_rel` is normalised by the tensor's own scale; `rel_rms` is the
norm-wise figure. "grad" is the worst of `dq, dk, dv, dg, dbeta`.

| case | `o` max_rel | `o` rel_rms | grad max_rel | grad rel_rms |
| --- | --- | --- | --- | --- |
| `[1,1024,96,128,128]` bf16, `fla` | 6.69e-03 | 3.33e-03 | 6.79e-03 | 3.35e-03 |
| `[1,1024,96,128,128]` bf16, **`flydsl`** | **2.67e-03** | **1.67e-03** | **4.13e-03** | **2.38e-03** |
| `[1,1024,96,128,128]` fp32, `fla` | 4.23e-06 | 2.26e-06 | 1.67e-05 | 5.42e-06 |
| `[1,1024,96,128,128]` fp32, **`flydsl`** | **5.22e-07** | **2.85e-07** | **8.67e-07** | **1.31e-06** |
| `[2,512,8,128,128]` bf16, `fla` | 7.39e-03 | 3.33e-03 | 4.78e-03 | 3.41e-03 |
| `[2,512,8,128,128]` bf16, **`flydsl`** | **1.97e-03** | **1.67e-03** | **3.82e-03** | **2.39e-03** |
| `[2,128,4,64,64]` bf16, `fla` | 6.25e-03 | 3.31e-03 | 6.73e-03 | 3.42e-03 |
| `[2,128,4,64,64]` bf16, **`flydsl`** | **2.35e-03** | **1.67e-03** | **4.21e-03** | **2.45e-03** |

At the production geometry `flydsl` is **2.5× closer to the oracle on the output
and 1.6× closer on the gradients** than `fla`, in bf16 and in fp32. That margin
is a real asset and it is the reason two otherwise-attractive speedups were
rejected in §5.

Unit tests: 87 of 88 pass in
`tests/unit_tests/megatron/transformer/kimi_k3/test_kda_{flydsl_kernel,vs_fla,eager_reference,backend_dispatch,collapse_to_gated_delta_rule}.py`.
The one failure, `test_fla_chunk_kda_does_not_activate_beta_itself`, **fails
identically on pristine `main`** and is a property of fla 0.5.2, not of any change
here: the test pins the fact that no released `fla` honours
`use_beta_sigmoid_in_kernel`, and 0.5.2 now does. The `_fla` adapter is still
correct — it never passes the flag and always hands over an already-activated
`beta` — but that test and the hazard note in `kda_kernels/_fla/__init__.py`
need updating for 0.5.2.

## 3. Where the time went before this pass

`torch.profiler`, `[1,4096,96,128,128]` bf16, on-device time.

### Forward: 2324 µs in 21 launches (`fla`: 1300 µs in 8)

| item | µs | share |
| --- | --- | --- |
| the four FlyDSL kernels (sweep 524, scores 294, prep 207, UT 54) | 1079 | 46 % |
| Tensile batched GEMMs ×3 (`ut@KΓ`, `ut@V`, the output `baddbmm`) | 338 | 15 % |
| **torch elementwise glue** | **907** | **39 %** |

### Backward: 10434 µs in **97** launches (`fla`: 3102 µs in **13**)

| item | µs | share |
| --- | --- | --- |
| FlyDSL kernels (`scores_bwd` 1739, sweep ×2 1225, recompute's scores 295 + prep 204 + UT 55) | 3518 | 34 % |
| Tensile batched GEMMs ×15 | 1759 | 17 % |
| **torch elementwise glue, ~80 launches** | **5083** | **49 %** |

Attributed by launching op, the glue's largest single entry was `aten::copy_`
(16 calls, 1076 µs) followed by the chunk-prep adjoint's 17-op elementwise chain
(~1400 µs) and five `aten::add` gradient accumulations (346 µs).

**This confirms the launch-count reading, and refines it.** The whole remaining
gap is 21/97 launches against 8/13, and it decomposes as ~50 % glue,
~20 % batched GEMMs that `fla` performs inside its kernels, and ~30 % the
kernels themselves. It is *not* true that the kernels are at parity: our
`kda_scores_bwd` is 1739 µs against fla 0.5.2's `chunk_kda_bwd_kernel_intra` at
1250 µs (1.39×), and our sweep is 524 µs against `chunk_gated_delta_rule_fwd_kernel_h`
at 241 µs. The "at parity" note in `wp9_p4/FINDINGS.md` was measured against fla
0.4.2, whose backward is slower.

## 4. What this pass changed, and what it bought

The glue is elementwise chains in one index space, which is what Inductor fuses,
so it goes through `torch.compile` (`_flydsl_v1/_compile.py`) rather than into
more hand-written FlyDSL kernels. Measured in isolation
(`bench/probe_torch_tricks.py`), the chunk-prep adjoint goes from **17 launches
to 3** with a worst output difference of 1.9e-06, which is fp32 reassociation.

| file | change |
| --- | --- |
| `_compile.py` (new) | Inductor wrapper: lazy `torch.compile(dynamic=False)`, the recompile limit raised (Dynamo recompiles per shape **and per grad mode**, because every region runs once under `no_grad` in the forward and once under `enable_grad` in the backward's recompute), and a `K3P_KDA_FLYDSL_COMPILE=0` kill switch. |
| `prep.py` | the chunk-prep adjoint extracted into one compiled region; the `.float()` before each multiply dropped, because a bf16 × fp32 multiply promotes in one launch; `qf*d_qf + kf*(a−b)` folded into `addcmul`. |
| `chunk.py` | the five transposed casts and the `cumsum` in one compiled region, so the 200 MB `gf` that exists only to be scanned never reaches memory; the two β products and the output layout compiled so their adjoints fuse. |
| `sweep.py` | `[dQG; dW]` written as two `baddbmm` into the halves of one buffer instead of a `cat` of two GEMMs plus a negation (bit-identical, 4 launches → 2, ~1.4 GB saved); `softmax_scale` folded into those GEMMs' `alpha` instead of a 400 MB pass over `do`; both halves of `qw` prepared in one region. |

One thing worth recording as a trap: the **first** version of this wrote the
compiled regions functionally (`.transpose().to().contiguous()`). Inductor fuses
that to one kernel, but ATen runs it as two passes, because `.to()` preserves
strides. When Dynamo fell back in a long multi-shape run the forward went from
21 launches to **26 and slower than the baseline**. Every compiled region now has
the best hand-written eager spelling as its *body* — `empty` + `copy_`, which
Inductor functionalises and fuses just as well — so a fallback can never be a
regression.

### Before / after, per-shape, bf16

`base` is pristine `main` (`df90fdd5`) in a `git worktree`; `opt` is this branch.
Same harness, same process protocol, same GPU, `fla` re-measured in each run.

| shape | metric | base | **opt** | `fla` | base ÷ fla | **opt ÷ fla** | opt ÷ base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prod_T2048` | fwd µs | 1244 | **1131** | 801 | 0.58× | **0.71×** | 1.10× |
| | bwd µs | 5348 | **4582** | 1569 | 0.29× | **0.34×** | 1.17× |
| | fwd+bwd µs | 6592 | **5713** | 2370 | 0.36× | **0.41×** | **1.15×** |
| `prod_T4096` | fwd µs | 2365 | **1979** | 1378 | 0.58× | **0.70×** | 1.20× |
| | bwd µs | 10419 | **8208** | 3126 | 0.30× | **0.38×** | 1.27× |
| | fwd+bwd µs | 12784 | **10187** | 4504 | 0.35× | **0.44×** | **1.25×** |
| `off8L_mbs2` | fwd µs | 7450 | **6473** | 4397 | 0.59× | **0.68×** | 1.15× |
| | bwd µs | 34441 | **27375** | 9876 | 0.29× | **0.36×** | 1.26× |
| | fwd+bwd µs | 41891 | **33848** | 14273 | 0.34× | **0.42×** | **1.24×** |
| `curve_mbs8` | fwd µs | 1530 | **1334** | 867 | 0.57× | **0.65×** | 1.15× |
| | bwd µs | 6854 | **5443** | 1922 | 0.28× | **0.35×** | 1.26× |
| | fwd+bwd µs | 8384 | **6777** | 2789 | 0.33× | **0.41×** | **1.24×** |
| `curve_mbs16` | fwd µs | 2882 | **2527** | 1532 | 0.53× | **0.61×** | 1.14× |
| | bwd µs | 13453 | **10591** | 3560 | 0.26× | **0.34×** | 1.27× |
| | fwd+bwd µs | 16335 | **13118** | 5092 | 0.31× | **0.39×** | **1.25×** |
| all shapes | launches fwd/bwd | 21 / 97 | **19 / 67** | 8 / 13 | | | |
| `prod_T4096` | peak GB | 4.90 | **4.62** | 2.63 | | | |
| `off8L_mbs2` | peak GB | 17.12 | **16.14** | 9.22 | | | |

**1.24–1.25× faster fwd+bwd than the previous FlyDSL backend at every real
geometry, 30 backward launches removed, slightly less memory, and parity
unchanged and still better than `fla`'s.** Against `fla` the fwd+bwd ratio moves
from 0.31–0.36× to 0.39–0.44×: **`flydsl` does not yet beat `fla`, it is still
2.3–2.6× slower.**

## 5. Why it does not reach parity, in arithmetic rather than in prospect

After this pass the production backward is 8208 µs against `fla`'s 3126, and the
5082 µs difference decomposes:

| remaining gap | µs | what closing it needs |
| --- | --- | --- |
| torch/Inductor glue still on the graph | ~2600 | the rest of the fusion, plus taking over the assembly's adjoint so the five `aten::add` gradient accumulations happen inside a fused region |
| 15 Tensile batched GEMMs `fla` never issues | ~1780 | folding `ut@KΓ`, `ut@V` and the six sweep-adjoint products **into** the kernels, i.e. what `fla` does |
| the kernels themselves | ~700 | `kda_scores_bwd` 1739 → 1250 (fla's), the sweep 524 → 241 |

The forward is the tighter constraint, and it is worth stating plainly: the four
FlyDSL kernels alone cost **1079 µs where `fla`'s entire forward is 1300 µs**. So
even with every microsecond of glue and every GEMM deleted, this decomposition
lands at ~0.9× of `fla` — **overtaking `fla` requires making the score and sweep
kernels faster, not only fusing around them.** Two concrete leads, both measured
but not implemented here:

- the forward score kernel is **exactly LDS-bandwidth bound**: its one-output-
  element-per-thread mapping reads 3 LDS words per 2 FMAs, which is 24 GB of LDS
  traffic per call and 310 µs at the hardware's 78 TB/s — against 294 µs
  measured. Register-blocking the off-diagonal sub-blocks (they can contract a
  whole row-block against every earlier column-block at once, since their
  reference row makes both decay factors ≤ 1) takes the ratio to ~0.8 and should
  give ~30 %.
- `kda_scores_bwd` is 5.9× the forward kernel for 2× the arithmetic.

### Two speedups measured and deliberately rejected

- **bf16 operands for `ut @ KΓ` / `ut @ V`.** 1.5× on the GEMMs, but it pushes
  the bf16 output error from 2.7e-03 to 5.7e-03 — spending almost all of the
  accuracy margin §2 documents, to buy ~110 µs of a 1979 µs forward.
- **CUDA graphs.** Already settled in `wp9_p5/FINDINGS.md` §4: hipBLASLt raises
  `operation not permitted when stream is capturing` in the backward on this
  image, and a failed capture does not unwind — it wedged a node for 34 minutes.

## 6. What any of this is worth end to end — the honest ceiling

From the real 8L-official run (`/home/botahu/primus_output/8L_1318_r%t.out`,
188.9 TFLOP/s/GPU, 56872 TFLOP per global batch, 4×8 MI355X): a step is
56872 / 32 / 188.9 = **9.41 s**. Per rank per step KDA runs 6 layers × 2
micro-batches, and `recompute_granularity=full` means the forward runs twice:

| | 2·fwd + bwd, per layer per micro-batch | × 12 | share of a 9.41 s step |
| --- | --- | --- | --- |
| `fla` | 18670 µs | 224 ms | **2.4 %** |
| `flydsl`, before | 49341 µs | 592 ms | 6.3 % |
| `flydsl`, after this pass | 40321 µs | 484 ms | 5.1 % |

So this pass recovers 108 ms of a 9.41 s step (**1.1 %**) relative to the old
FlyDSL backend, and **a hypothetically free KDA kernel would beat `fla` by at
most 2.4 %** — which is the ceiling the provenance analysis derives from the FLOP
breakdown (`kda_core` 290 TFLOP against `kda_proj`'s 14656, 50.5×), reached here
independently from measured time rather than from FLOPs. Switching production to
`flydsl` today would still *cost* about 2.8 % of a step.

`fla` therefore remains the correct production default, and the reason to work on
this kernel is the kernel, not the step time. The `curve` geometry is where a
FlyDSL module backend has ever paid for itself end to end, and that was the
attention-residual mixer (+18.3 % at MBS 16), not KDA.

## 7. Reproducing

```bash
WP=/shared_nfs/botao/kimi_k3_optimization_0812
# one process per shape, both trees, launch counts included
TREE=$WP/Primus_flydsl_opt   TAG=opt  bash $WP/tools/bench_per_shape.sh
TREE=$WP/Primus_baseline_main TAG=base bash $WP/tools/bench_per_shape.sh
# parity against the fp32 eager oracle
python bench/bench_kda_backends.py --tag parity --parity-only --dtypes bf16,fp32
# per-kernel and per-launching-op attribution
python bench/profile_kda.py --shapes prod_T4096 --dtypes bf16
# the individual fusion decisions, each checked numerically
python bench/probe_torch_tricks.py
```

Everything runs inside `rocm/primus:v26.4` on a compute node; `tools/run.sh` is
the login-node entry point. `fla` is not in the image and must be installed
(`pip install flash-linear-attention==0.5.2`, the `requirements.txt` pin).
