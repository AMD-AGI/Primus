# Plan-8 P54 — CSA FWD tilelang (cr=4) — descoped at task-list refinement

> Phase summary written 2026-05-15 at P54 close-out.

**Status: descoped at task-list refinement per R9.1 / R9.3.  The
CSA FWD's structural shape envelope (head_dim=512, K_topk=512)
puts it in the same SMEM-budget regime as P50/P51; tilelang's
default `T.gemm` scheduling regressed 5-15× on those phases at
this dim.  Without a hand-tuned config or upstream tilelang SMEM
partitioning, a CSA FWD tilelang kernel would land at the same
regression band.  CSA call sites continue to route through the
plan-5 P32 final Triton CSA kernel (single-row sparse tile design,
already at the SMEM-budget optimum).**

---

## 1. Objective (originally scoped)

Implement joint local SWA + sparse top-K + sink fused FWD via
tilelang.  Borrow from `tilelang/examples/dsa_sparse_finetune/
sparse_mla_fwd.py` and `tilelang/examples/deepseek_v4/
sparse_attn_fwd_sm90.py` (DSV4-style MQA sparse FWD reference,
sm90).  Two wrapper APIs: pre-gathered (`v4_csa_attention`) and
in-kernel-gather-from-pool (`v4_csa_attention_from_pool`).

## 2. Why the descope

Two independent observations:

### 2.1 V4-Flash CSA SMEM-budget profile is no better than dense

Per-program SMEM at V4-Flash CSA widths (B=1, HQ=64, Sq=4096,
K_topk=512, D=512, bf16):

* Q tile (single row): 1 * 512 * 2 = 1 KiB
* K_local tile: `BLOCK_N * 512 * 2` = 32 KiB at BLOCK_N=32
* V_local tile: 32 KiB
* Gathered tile (sparse): `BLOCK_K * 512 * 2` = 32 KiB at BLOCK_K=32
* O tile: 1 KiB
* acc_s_local + acc_s_sparse + softmax fragments: ~10 KiB

Sum: ~110 KiB — close to the 160 KiB MI355 budget; same band
that hit P50 at BLOCK_M=block_N=32.  The plan-5 P31 Triton CSA
kernel ALREADY runs at this single-row design with careful tile
choices; it sits at the SMEM optimum.

Tilelang's MFMA schedule + LDS bank pattern at `dim=512` reaches
~13 TFLOP/s (per P50 microbench); the plan-5 P31 Triton CSA
kernel reaches ~28 TFLOP/s on the same envelope.  A naive CSA
FWD tilelang port would likely land in the 5-15× regression
band that P50 / P51 exhibited.

### 2.2 CSA FWD is more complex than dense FWD (joint softmax)

CSA's joint local + sparse fused softmax adds two K-loops
(local SWA chunks + sparse top-K gathered chunks) + a per-head
sink + a sparse-mask additive bias.  That's at least 2-3× the
kernel-code complexity of P50 + the same per-call MFMA scheduling
lottery.  Without a clear path to beating the plan-5 P31 Triton
baseline, the implementation cost is hard to justify under R9.1.

## 3. Code surface

No code change.  Documentation-only:

```
deepseek-v4/develop/progress/p54/
  + p54-summary.md (this file)
```

The plan-8 P54 status row in `progress/status.md` is marked
`[-]` per R2.2.

## 4. Follow-ups + commit pin

* Plan-9: revisit CSA tilelang once (a) the upstream SMEM
  partitioning lands or (b) we have a hand-tuned MFMA schedule
  that beats Triton at D=512.
* `tilelang/examples/deepseek_v4/sparse_attn_fwd_sm90.py` is
  a sm90-specific reference; an MI355X port would still need
  the SMEM partitioning fix to beat Triton.
* Feature commit SHA: TBD-p54.
