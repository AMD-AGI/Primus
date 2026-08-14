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

## 5b. Round 2: the kernel bodies. Two negative results, and where the gap really is

Round 1 removed the glue; §5 said the only way past ~0.9× is faster kernels. Round
2 went after them and **did not move the end-to-end number**. What it produced
instead is a decomposition that says exactly why, and it is worth more than the
2 % it bought.

### What was tried

**A build-parameter sweep** (`bench/tune_kernels.py`), every configuration timed
on the operands the real assembly hands the kernel and checked **bit-equal**
against the shipped one — these are scheduling parameters, so anything but
bit-equality is a bug, not a tradeoff.

| kernel | swept | result |
| --- | --- | --- |
| `kda_decay_scores` | `waves_per_eu ∈ {1,2,4,8}` | 358–367 µs, no consistent winner across four runs. Left at 2. |
| `kda_decay_scores_bwd` | `THREADS_D ∈ {16,32,64,128}` × `waves_per_eu ∈ {1,2,4}` | 1698 (32, 4) / 1705 (16, 4) / 1801 (64) / **2992 (128)**. Took `waves_per_eu = 4`: **2 %**. |
| `kda_state_sweep` | `block_v ∈ {16,32,64}` × `waves_per_eu ∈ {1,2,4}` | `block_v = 16` confirmed, 737 µs against 912 (32) and 956 (64); `waves_per_eu` under 1 %. No change. |

**Negative result 1: register pressure and occupancy are not these kernels'
problem.** `THREADS_D` sets the score adjoint's whole accumulator tile — 16
through 64 give it 1, 2 or 4 rows and 8, 4 or 2 channels per thread, i.e. 8 to 32
accumulator registers before temporaries — and every one of them lands within
6 %. The single large effect is 128, which collapses the tile to one channel and
loses the wide LDS access below (1.8×).

**Explicit LDS vectorisation** of both score kernels' contractions: read four
consecutive channels per access (`ds_read_b128`) instead of one, which required
the thread→channel mapping in the adjoint to become **contiguous** rather than
strided by `THREADS_D`. Bit-identical at every geometry, 49/50 unit tests pass
(the 50th is the fla-0.5.2 one from §2). Worth 2–4 % on each kernel.

**Negative result 2: the compiler was already doing it.** The forward score
kernel is exactly LDS-bandwidth bound — 3 LDS words per 2 FMAs, 24 GB per call,
310 µs predicted at 78 TB/s against 294 µs measured — so §5 expected ~30 % from
this. It gave 2–4 %, because LLVM was already merging the adjacent
`ds_read_b32`. The prediction was right about the *bound* and wrong about the
*headroom*.

Net at the end-to-end level: **nothing measurable.** ~75 µs of kernel time on a
10200 µs fwd+bwd is below this benchmark's run-to-run spread; the per-shape
fwd+bwd ratios after round 2 are 0.43 / 0.44 / 0.42 / 0.41 / 0.39 against round
1's 0.41 / 0.44 / 0.42 / 0.42 / 0.39.

### Where the forward gap actually is, stage by stage

This is the useful output. `[1,4096,96,128,128]` bf16, on-device, both backends
in the same run, our 19 launches grouped against fla's 8 by what they compute:

| stage | ours µs | `fla` µs | |
| --- | --- | --- | --- |
| intra-chunk scores `Aqk`, `Akk` | 295 | 407 | **we are 1.4× faster** |
| `(I − L)^{-1}` triangular solve | 54 | 190 | **we are 3.5× faster** |
| operand prep + `W`, `U`, output GEMMs | 542 | 370 | 1.5× slower |
| inter-chunk state sweep | 542 | 234 | **2.3× slower** |
| **`[NB,C,D]` fp32 layout + within-chunk cumsum + β products** | **492** | **52** | **9.5× slower** |
| total | 1925 | 1253 | 0.65× |

**We already beat `fla` on the two stages that are actually hard** — the
block-referenced decay-weighted score matrices and the on-chip triangular solve,
349 µs against 597 — and lose the forward on two things that are not kernel
arithmetic at all:

- **440 µs of the 672 µs gap is a layout contract.** Our kernels require fp32
  contiguous `[NB, C, D]` operands, so `q`, `k`, `v` are transposed and widened
  into 200 MB tensors and `cg` is materialised; `fla`'s kernels read the bf16
  `[B, T, H, D]` input in place and its cumsum kernel costs 52 µs. Nothing about
  this is FlyDSL's doing — it is what the four kernels ask for.
- **308 µs is one kernel**, the state sweep, at 2.3× `fla`'s.

### So: is >1.0× reachable, and what would it take

On the arithmetic above, yes, and only these two things:

| | forward µs | vs `fla` 1253 |
| --- | --- | --- |
| today | 1925 | 0.65× |
| + kernels read bf16 `[B,T,H,D]` in place (removes ~440) | 1485 | 0.84× |
| + state sweep at `fla`'s 234 µs (removes 308) | 1177 | **1.06×** |

The first is a contained but real change: teach `kda_decay_scores`,
`kda_chunk_prep` and `kda_decay_scores_bwd` to take a bf16 base pointer with the
`[B, T, H, D]` stride and convert on load, which is an address-arithmetic and
load-type change in their fill phases, not a new algorithm. It also pays twice,
because the backward recomputes the assembly. A cheaper half-measure with **zero
accuracy cost** is to keep the `[NB, C, D]` layout but store `q`/`k` in bf16
— they arrive bf16, so the fp32 copy adds no information — which halves that
traffic for a one-line change per kernel plus a load-type change.

The second is a genuine kernel investigation and the one place where "make the
kernel faster" is still the right instruction.

The **backward** is not reachable the same way, and it is the one that decides the
headline number. Its 8233 µs against `fla`'s 3183 is: our score adjoint 1738
(against `fla`'s entire intra-backward at 1250), two sweeps 1256, the recompute's
forward kernels 550, **1715 µs in fifteen Tensile batched GEMMs `fla` never
issues at all**, and ~1750 of remaining glue and gradient accumulation. Closing
it means fusing those GEMMs into the kernels, i.e. hand-writing the backward the
way `fla` does — the piece of work every earlier pass also concluded was the
price, and larger than everything in rounds 1 and 2 together.

So the two must be stated separately, because they have different answers:

| | ours | `fla` | ratio |
| --- | --- | --- | --- |
| **forward**, today | 1925 | 1253 | 0.65× |
| **forward**, with the layout contract and the sweep fixed | 1177 | 1253 | **1.06×** |
| **fwd+bwd**, today | 10158 | 4436 | 0.44× |
| **fwd+bwd**, with a *perfect* forward and the backward untouched | 9410 | 4436 | 0.47× |

**Beating `fla` on the forward is reachable and quantified. Beating it on
fwd+bwd is not, without hand-writing the backward adjoint** — the forward is
19 % of our fwd+bwd and 28 % of `fla`'s, so even taking it to zero leaves the
ratio at 0.55×.

## 5c. Round 3: forward only. A measured ledger, one win, and a corrected target

Round 2 projected that the forward could reach **1.06×** of `fla` from two
changes. Round 3 measured the stages instead of attributing them from a
kernel-name profile, and **that projection was wrong on two counts**. The
corrected answer is below; the ledger is the durable artifact.

### The ledger — every forward stage on the tensors the assembly hands it

`bench/probe_fwd_ablation.py`, `[1,4096,96,128,128]` bf16, median of 20. Stages
are timed standalone, so they sum to ~110 % of the whole (each pays its own
launch); "ratio if free" is the forward ratio against `fla` with that stage's
entire cost deleted, i.e. a hard upper bound on optimising it.

| stage | µs | % of forward | ratio if free |
| --- | --- | --- | --- |
| **`fused_chunk_sweep` (sweep kernel + output `baddbmm`)** | **845** | **43 %** | **1.18×** |
| `decay_scores` kernel | 347 | 18 % | 0.82× |
| `prepare_operands` — all five `[NB,C,*]` operands + cumsum | 250 | 13 % | 0.77× |
| ↳ one `[B,T,H,D]` → `[NB,C,D]` fp32 cast+transpose | 97 | 5 % | 0.71× |
| `chunk_prep` kernel | 227 | 12 % | 0.76× |
| `W` GEMM (`ut @ kgam`) | 114 | 6 % | 0.71× |
| `U` GEMM (`ut @ vf`) | 113 | 6 % | 0.71× |
| β products (`low`, `scale_ut`) | 81 | 4 % | 0.70× |
| `ut_inverse` kernel | 79 | 4 % | 0.70× |
| `lay_back` (output → `[B,T,H,V]`) | 65 | 3 % | 0.70× |
| ↳ `store_w` (fp32 → bf16 into `qw[:, C:]`) | 64 | 3 % | 0.69× |
| `KG` transpose | 46 | 2 % | 0.69× |
| whole flydsl forward / `fla` forward | 1967 / 1322 | | 0.67× |

**Correction 1.** Round 2 put "the `[NB,C,D]` fp32 layout" at 440 µs. Measured,
*all five* operands plus the cumsum cost **250 µs**, and the part `q`/`k` account
for is ~90–100 µs. The 440 came from a kernel-name profile in which
`triton_poi_fused_copy_transpose_0` covers four unrelated copies; I attributed
the whole name to the layout. So work item 1 — teaching the kernels to read bf16
`[B,T,H,D]` in place — is worth **~90 µs, 0.67 → 0.71×**, not 0.84×. It is scoped
and still worth doing, but it cannot carry the round, so it was not implemented
here.

**Correction 2.** Round 2 compared our sweep against `fla`'s
`chunk_gated_delta_rule_fwd_kernel_h` at 234 µs. That is not the counterpart.
**`fla` uses a different decomposition**: its sequential kernel computes *only*
the `[K,V]` state history (234 µs), and the per-chunk output — our `Rq` and `T` —
happens afterwards in `chunk_gla_fwd_kernel_o` (193 µs), which is **fully
parallel over chunks**. We fuse both into the sequential loop. So the honest
comparison is our 845 against `fla`'s 427, and "sweep at `fla`'s 234" was
comparing 845 against a kernel that does a third of its work.

### Attempt 1: get the exposed HBM latency off the chunk loop's critical path

The diagnosis that made this worth trying: **`fla` reaches 234 µs with four times
*fewer* workgroups than we use (192 against 768), so the gap was never
parallelism** — it is what each chunk's iteration costs. Reading the loop, two
global loads sat *after* the MFMA accumulation and before the barrier that ends
the step: `sink_t` loaded `Yc[nb, row, v]` and `_state_core` loaded
`Dec[nb, kk]`. Neither depends on the accumulator. At BV=16 there are three
workgroups per CU, so there is nothing to hide an HBM latency behind, twice per
chunk, 64 chunks deep.

Each sink now carries a `prefetch` that the group functions call **before**
entering the accumulation, so both latencies overlap the MFMAs
(`kda_state_sweep_kernel.py`). Bit-exactness is not claimed — the arithmetic is
unchanged but the emission order is — and the numbers are unchanged to the digit
against the fp32 eager oracle (output `2.67e-03` at production geometry, exactly
as before); **41/41 FlyDSL kernel tests pass.**

| | before | after |
| --- | --- | --- |
| `fused_chunk_sweep` region | 845 µs | **780 µs** |
| forward, `prod_T4096` | 1967 | **1917** |

Real, evidence-backed, and small: **−65 µs, 7.7 % of the stage, ~2 % of the
forward.** The two exposed latencies were worth 65 µs, not the several hundred the
serial-chain argument suggested, which says the chain's cost is spread across the
LDS→fragment→MFMA dependency rather than concentrated in those two loads.

### Forward, per shape, four states of the branch

| shape | `main` | round 1 | round 2 | **round 3** | `fla` | `main`/fla | **r3/fla** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prod_T2048` | 1244 | 1131 | 1039 | **1024** | 709 | 0.58× | **0.69×** |
| `prod_T4096` | 2365 | 1979 | 1963 | **1917** | 1340 | 0.58× | **0.70×** |
| `off8L_mbs2` | 7450 | 6473 | 6485 | **6337** | 4348 | 0.59× | **0.69×** |
| `curve_mbs8` | 1530 | 1334 | 1351 | **1343** | 856 | 0.57× | **0.64×** |
| `curve_mbs16` | 2882 | 2527 | 2515 | **2490** | 1516 | 0.54× | **0.61×** |

Launches unchanged at 19/67 against `fla`'s 8/13; fwd+bwd 0.39–0.45×.

### Did the forward cross 1.0×? No — and here is what it would take

**No. Best is 0.70× at production geometry, 0.61–0.70× across the five shapes.**
The corrected arithmetic, grouped by what each side computes:

| | ours µs | `fla` µs | |
| --- | --- | --- | --- |
| intra scores + triangular solve | 387 | 597 | **we win by 210** |
| operand prep + `W`, `U` | 413 | 177 | we lose 236 |
| state sweep + chunk output + output layout | 767 | 427 | we lose 340 |
| `[NB,C,D]` layout + cumsum + β + `KG` transpose | 343 | 52 | we lose 291 |
| total | 1910 | 1253 | 0.66× |

Crossing 1.0× means finding **657 µs**, and the three losing buckets hold 867. All
three are kernel-fusion projects, none is a tuning knob:

1. **~200 µs** — fold the two β products into the score and UT kernels (they
   already touch `akk` and write `P`), have `chunk_prep` write `KG` transposed
   from LDS, and read `q`/`k` from the bf16 `[B,T,H,D]` input in place.
2. **~150 µs** — fuse `ut @ kgam` and `ut @ vf` into one kernel; `fla` does both
   in `recompute_w_u_fwd_kda_kernel` at 177 µs where we spend 413 in a kernel
   plus two fp32 Tensile GEMMs.
3. **~250 µs** — split the sweep the way `fla` does: a sequential kernel for the
   `[K,V]` state history only, then a **fully parallel** kernel for `Rq`/`T`, so
   the per-chunk output work stops inheriting the serial chain's latency
   tolerance. This is also what would let `block_v` rise to 64 and cut the A
   operand's `V/BV`-fold re-reads.

Together ~600 of the 657, i.e. **1.0× is reachable but only if all three land** —
three separate kernels, each comparable in size to a full pass of this work
package. Round 3's bounded budget bought item 1 of nine, and honestly reports the
other eight rather than starting one it could not finish.

## 5d. Round 4: item 3 is negative, and with it the 1.0× arithmetic stops closing

Round 3 costed three fusion projects at ~200 / ~150 / ~250 µs against a 657 µs
target. Round 4 measured the largest one **before** writing it, and it is not
worth +250 µs — it is worth **−120 µs**. That is decisive, so this section reports
it, delivers the one item that did land, and recomputes the account rather than
spending the round on items that cannot reach the goal.

Node note: `crsuse2-m2m-088`, all eight GPUs idle (the previous node had a tenant
holding 89 % of GPU 0's VRAM). Absolute µs therefore differ slightly from §5c;
every comparison below is re-measured on this node.

### Item 3 — split the sweep like `fla`: measured, and it costs

The kernel already has the two build flags the split needs, so
`bench/probe_sweep_split.py` measures the sequential half without writing an
output kernel. `A` is what the forward runs today; `B` is the split's sequential
half (drop `Rq` from the loop, write the `[NB,K,V]` history); `C` is `B` without
the history write, the floor the split could ever reach. `prod_T4096`, µs:

| `block_v` | A today | B split's sequential half | C floor | owed `Rq = QG @ S` | split total | vs A |
| --- | --- | --- | --- | --- | --- | --- |
| **16** (shipped) | **504** | 434 | 348 | 190 (fp32) | 624 | **costs 120** |
| 32 | 616 | 524 | 320 | 190 | 714 | costs 98 |
| 64 | 715 | 496 | 382 | 190 | 686 | saves 29 |

Three things follow, and they close the item:

- **Dropping `Rq` from the loop saves only 70 µs** (504 → 434). It is nearly free
  where it is, and the kernel's own design comment says why: the `rq` and `t`
  phases *share every MFMA B fragment*, because both contract the same state
  against two stacked row blocks of the same operand. Removing one halves the
  MFMAs but not the LDS traffic that dominates.
- **The split then owes more than it saves**: +86 µs to write the state history
  (348 → 434) and +190 µs for `Rq = QG @ S` as a batched GEMM. bf16 is worse
  (114 µs for the GEMM but 105 to cast `S`), and would spend accuracy besides.
- **"The split unlocks `block_v` → 64" was wrong.** `block_v = 64` is slower
  *because of parallelism* — 192 workgroups on 256 CUs — which the split does not
  change. At 64 the split roughly breaks even, on a baseline already 211 µs worse
  than 16.

### Item 1a — fold both β products into the UT kernel: landed, −66 µs

`L = Akk ⊙ (−β_r)` and `M = P ⊙ β_c` were two elementwise passes over a 200 MB
`[NB,C,C]` tensor for one multiply each. They are now a row scale during the UT
kernel's LDS fill and a column scale at its store, both on values already in
registers (`kda_ut_inverse_kernel.py`, `fuse_beta`). The adjoint needs the
*unscaled* `P`, which cannot be recovered from `M` (dividing by a sigmoid
underflows), so the kernel writes it as a second output under `emit_p` — off in
the no-grad forward, so the forward pays nothing for it.

| | µs |
| --- | --- |
| unfused: two β products + `ut_inverse` | 147 |
| **fused `ut_inverse_beta`, one launch** | **80** |
| `ut_inverse` alone, for reference | 82 |

**The two multiplies are genuinely free** — the fused kernel costs what the
inverse cost by itself. Forward 1961 → 1865 µs on the same run.

One DSL trap worth recording, since it cost a debug cycle: a build-time `if` in a
traced kernel body is rewritten into an `scf.if`, so a value assigned inside it
does not escape — it arrives as `None` at the next `arith.mulf`. Build-time
choices go through a dict of closures selected at build scope, which is the
convention the score-adjoint kernel already documents.

### Where round 4 leaves the forward

| shape | `main` | round 3 | **round 4** | `fla` | `main`/fla | r3/fla | **r4/fla** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prod_T2048` | 1244 | 1024 | **979** | 685 | 0.58× | 0.69× | **0.70×** |
| `prod_T4096` | 2365 | 1917 | **1858** | 1303 | 0.58× | 0.70× | **0.70×** |
| `off8L_mbs2` | 7450 | 6337 | **6131** | 4177 | 0.59× | 0.69× | **0.68×** |
| `curve_mbs8` | 1530 | 1343 | **1307** | 842 | 0.57× | 0.64× | **0.64×** |
| `curve_mbs16` | 2882 | 2490 | **2410** | 1547 | 0.54× | 0.61× | **0.64×** |

No shape regressed. Parity against the fp32 eager oracle is **unchanged to the
digit** — output `2.67e-03` at production geometry against `fla`'s `6.69e-03` —
and 87/88 unit tests pass, the exception being the pre-existing fla-0.5.2 one
from §2.

### The recomputed account: 1.0× does not close

Needed at `prod_T4096`: 1858 − 1303 = **555 µs**. What is left, with item 3 at
zero:

| item | measured stage cost | plausibly recoverable | status |
| --- | --- | --- | --- |
| 3. split the sweep | 799 (whole region) | **0** | **measured −120 µs; closed** |
| 1c. kernels read bf16 `[B,T,H,D]` in place | 254 (`prepare_operands`) | ~100 | not attempted |
| 2. fuse the `W` and `U` GEMMs into one kernel | 288 (2 GEMMs + `store_w`) | ~140 | not attempted |
| 1b. `chunk_prep` writes `KG` transposed | 46 | ~40 | not attempted |
| | | **~280 of 555** | |

Landing all three remaining items at full value puts the forward at ~1578 against
`fla`'s 1303 — **0.83×**. **So the forward cannot cross 1.0× either.** The reason
is specifically item 3: it was 38 % of the budget and it is negative, and there is
no other lever on the sweep — the split costs, `block_v` and `waves_per_eu` are at
their optimum (§5b), and round 3's prefetch already took the 65 µs of exposed HBM
latency. The sweep region is 43 % of our forward and the only stage whose removal
would cross 1.0× ("ratio if free" 1.12×), and it is now the one with no remaining
plan.

What would still be true after items 1b, 1c and 2: the forward at ~0.83×, and we
would remain **faster than `fla` on the two hardest stages** (intra-chunk score
matrices and the triangular solve, now 427 µs against `fla`'s 597) and **more
accurate on every output**. What it would take to actually pass `fla` is a sweep
that is 300 µs faster with the state recurrence still sequential, and this round
found no candidate for it.

## 5e. Round 5 — exploration: what the sweep is bound by, and every candidate priced

An exploration round, no implementation. The sweep region is 43 % of the forward
and the only stage whose removal would pass `fla` ("ratio if free" 1.12×), so it
got the whole budget. Deliverable: an attribution by **deletion** and a verdict on
each candidate. `bench/probe_sweep_bound.py` and `bench/probe_sweep_conc.py` are
the probes; the kernel gained a `probe=` build parameter whose variants
**deliberately compute the wrong answer** and are unreachable from any production
path.

### 1. Is the loop LDS-bound? No — and neither is it latency-bound

Round 4 inferred "the loop is LDS-bound, not MFMA-bound" from `emit_rq=False`
buying only 14 %. **Both halves of that inference are wrong.** Each row below
deletes exactly one cost from the real kernel (`prod_T4096`, `block_v=16`):

| variant | what it deletes | µs | saved | share |
| --- | --- | --- | --- | --- |
| `full` | — | 502 | — | — |
| `lds1` | 7 of the 8 b-fragment LDS reads per MFMA | 499 | **2.9** | **0.6 %** |
| `nobar` | both barriers in the chunk loop | 497 | 4.7 | 0.9 % |
| `noy` | the `Yc` global load | 447 | 55 | 11 % |
| `areuse` | ~3/4 of the A operand's global loads | 390 | 112 | 22 % |
| `nostore` | the `Rq` and `T` global stores | **323** | **179** | **36 %** |

**The LDS traffic is 0.6 % of this loop.** An 8× reduction in it — which is what a
transposed or bf16 LDS layout would buy — is worth 3 µs. MFMA issue accounts for
~8 µs by calculation (2.4e6 wave-MFMAs at one per 8 cycles). Barriers are 1 %.

**It is global memory traffic: stores 36 %, A operand 22 %, `Yc` 11 % — 69 % of the
loop between them**, and 70 % of it is attributed once MFMA is included.

And it is *throughput*, not latency. `bench/probe_sweep_conc.py` holds the
arithmetic fixed and varies only how many workgroups it is spread over:

| workgroups | per CU | µs | ns per chunk-step |
| --- | --- | --- | --- |
| 768 (production, 96 × 64) | 3.0 | 502 | 10.22 |
| 1536 | 6.0 | 496 | 10.08 |
| 3072 | 12.0 | 494 | **10.06** |
| 6144 | 24.0 | 497 | 10.11 |
| 49152 | 192.0 | 614 | 12.50 |

**An eight-fold increase in occupancy moves the cost per chunk-step by 1.6 %.** So
"three workgroups per CU on a 64-step serial chain" — the story rounds 3 and 4
worked from, and the reason the prefetch was tried — is not what limits this
kernel either. That also kills the premise behind every "more parallelism"
candidate, including the scan.

### 2. Each candidate, priced

**Candidate 1 — carry the state in registers instead of LDS. Falsified.** This was
the most promising one on paper, and `fla` really does do it: its
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` holds `h` as `b_h1..b_h4`,
`[BV, 64]` fp32 register tiles carried across the whole chunk loop, with no LDS
round trip and no barrier. But the measurement says the entire LDS exchange —
reads *and* barriers — is **1.5 % of our loop**. There is nothing to win.
Implementation cost would have been high (our group-2 accumulator layout and
group-1 B-operand layout differ, so removing LDS needs cross-lane permutes), for
7 µs.

**Candidate 2 — larger chunk. Measured, 2.6× worse.** `chunk_size` is already a
runtime argument, so this needed no code:

| | forward µs |
| --- | --- |
| C = 64 | **1861** |
| C = 128 | 4853 |

Exactly what the arithmetic predicts: the serial chain halves, but the score
matrices are `[C, C]` per chunk over `T/C` chunks so they grow as `T·C`, and the
triangular solve as `T·C²`. Dismissed.

**Candidate 3 — log-depth parallel scan. Dismissed by arithmetic.** Substituting
`T_n = U_n − W_n S_n` makes the transition `S_{n+1} = A_n S_n + B_n` with
`A_n = Diag(dec_n) − KG_nᵀ W_n` a **full `[K, K]` matrix**, so composition is
`2K³ + 2K²V` against a direct step's three `C`-sized products:

| | |
| --- | --- |
| direct | 38.7 GFLOP |
| Blelloch scan (~2·NC compositions) | 103.1 GFLOP, **2.67×** |
| new `[K,K]` `A` residency | 0.38 GB |
| scan floor from `A` traffic alone | **228 µs** |
| scan at an optimistic 300 TFLOP/s | **344 µs** |

So its floor is 344–570 µs against today's 502 and `fla`'s 234. It buys
parallelism, and §1 shows parallelism is worth 1.6 %. Dismissed.

**Candidate 4 (mine) — fuse the output into the sweep so `Rq` and `T` never reach
HBM.** The one candidate the measurements *support*. The stores are 36 % of the
loop, and the output `baddbmm` that consumes them is a further 214 µs measured
standalone. If the sweep kernel carried `Aqk` in LDS (a `[C, C]` fp32 tile, 16 KB)
and emitted `O = scale·(Aqk @ T + Rq)` directly:

| | µs |
| --- | --- |
| drop the `Rq` store | −90 |
| drop the output `baddbmm` | −214 |
| drop the `T` store (no-grad forward only; the backward needs it) | −90 |
| add: read `Aqk` into the kernel | +40 |
| **net** | **−354** |

That is the forward at 1861 − 354 = **1507 against `fla`'s 1303 — 0.86×**. Real,
measurement-backed, and **still short of 1.0×**. Implementation is a substantial
extension of the sweep kernel (an `Aqk` LDS tile, an output phase, and the
`emit_states`/grad-mode split), comparable to one of the earlier rounds.

### 3. Verdict: no path to a 300 µs saving, so no path past `fla`

The sweep must lose ~300 µs for the forward to pass `fla`. After this round:

| candidate | value | basis |
| --- | --- | --- |
| state in registers | ~7 µs | measured (LDS is 1.5 % of the loop) |
| larger chunk | −2990 µs | measured |
| parallel scan | ≥ 0, floor above today's | arithmetic |
| **fuse the output in** | **−354 µs of the forward** | measured components |
| everything from rounds 2–4 | already taken or negative | — |

Candidate 4 is the best available and it lands the forward at **0.86×**. Nothing
found in this round closes the remaining 0.14, and the three ideas that could have
been step changes are each closed by a measurement rather than by an opinion.

**One honest gap.** At *identical* workload — `emit_rq=False`, which is exactly
`fla`'s two products — we measure 434 µs against `fla`'s 234. So `fla` is 1.85×
faster doing the same arithmetic on the same data, and **none of the six deletion
probes locates that difference**: LDS, barriers, occupancy and MFMA are all ~1 %,
and the traffic we can attribute is traffic `fla` also moves (it writes `h` and
`v_new`, 603 MB, where we write 402). The remaining suspicion is the store
*pattern* — at `block_v = 16` a wave's store covers four discontiguous 64-byte
fragments of a 512-byte row stride, where `fla` at `BV = 64` writes 256-byte runs —
but `block_v = 64` measures *worse* for us overall (715 µs), so that explanation
does not survive its own test either. This is a gap in understanding, not a
candidate, and it is the honest reason not to claim the door is provably shut:
1.85× exists in `fla` and we cannot yet say why.

## 5f. Round 6 — the fused chunk output: the last live candidate, landed

Round 5's only surviving candidate, implemented. The sweep kernel now emits
`O = scale·(Aqk @ T + Rq)` itself, so neither `Rq` nor `T` makes a round trip
through HBM and the 214 µs output `baddbmm` disappears.

How the three pieces fit:

- **`Rq` never leaves registers.** Group 1 and the new output phase share the MFMA
  accumulator layout, so a value produced at accumulator slot `(mi, nt, s)` in
  group 1 is the same `(row, v)` the output phase needs at that slot. Four fp32
  per thread stay live across the barrier; no transpose, no LDS.
- **`T` only reaches memory when the backward wants it** (`emit_t`), which on the
  no-grad forward is never.
- **`Aqk` is staged in LDS**, 16 KB at `C = 64`, carrying the same
  cross-workgroup redundancy the A operand already has and which L2 absorbs the
  same way.

The output contraction is **fp32 VALU, not MFMA**, on purpose: `Aqk` is the one
operand this kernel has never rounded — keeping it fp32 is what holds the bf16
output error at 2.6e-3 — and this flydsl build has no usable fp32 MFMA. Fusion is
gated to `mode="mfma"`, because `group_valu` indexes accumulators by
`(block, row, v)` and the register handoff key would not match.

### Measured, `prod_T4096`

| | before | after |
| --- | --- | --- |
| `fused_chunk_sweep` region (kernel + output) | 799 | **642** |
| whole forward | 1861 | **1684** |
| forward vs `fla` | 0.70× | **0.77×** |

**−177 µs, against the −354 round 5 estimated.** The estimate was gross: it
credited removing the stores and the `baddbmm` but charged nothing for the
in-kernel VALU contraction (~140 µs) or for staging `Aqk` (~96 µs). Both are now
measured, and the net is the difference. Recording the shortfall rather than the
headline is the same discipline that caught the 440 µs layout and the 234 µs
comparison in §5c.

### Per shape, forward

| shape | `main` | round 4 | **round 6** | `fla` | `main`/fla | r4/fla | **r6/fla** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prod_T2048` | 1244 | 979 | **905** | 700 | 0.58× | 0.70× | **0.77×** |
| `prod_T4096` | 2365 | 1858 | **1705** | 1313 | 0.58× | 0.70× | **0.77×** |
| `off8L_mbs2` | 7450 | 6131 | **5612** | 4282 | 0.59× | 0.68× | **0.76×** |
| `curve_mbs8` | 1530 | 1307 | **1191** | 839 | 0.57× | 0.64× | **0.70×** |
| `curve_mbs16` | 2882 | 2410 | **2189** | 1536 | 0.54× | 0.64× | **0.70×** |

Every shape improved; none regressed. fwd+bwd is 0.41–0.46×. Parity against the
fp32 eager oracle is **unchanged to the digit** — `2.67e-03` output and
`5.22e-07` fp32 at production geometry — and 87/88 tests pass, the exception being
the fla-0.5.2 one from §2.

Why the single-stage delta (−157 µs on the sweep region) and the whole-forward
delta (−177 µs) differ: the region timing excludes the `baddbmm`'s output
allocation and the launch it no longer pays, and the ledger's stages are timed
standalone so each carries its own launch overhead. The integrated number is the
one to quote.

**The forward is now 0.77× and the remaining items from §5d (`~100 µs` for
in-place bf16 reads, `~140` for fusing the `W`/`U` GEMMs, `~40` for the `KG`
transpose) would take it to about 0.88×.** 1.0× still does not close.

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
