# Mamba2 loss-mismatch: diagnosis + proof experiment

## The question

The pre-push sanity table showed the 300M `mamba_hybrid` run **+10.56%** above
FLA's `mamba2_300M_hybrid` reference at step 700, while every other model was
within tolerance (pure GDN/KDA ≈ 0%, GDN hybrid +1.37%). Why?

## "How come we didn't see it in the normal experiment?"

We did — it was in the eval summary the whole time (`eval/summaries/summary.md`):

| step | Primus | FLA/8 | Δ% |
|---:|---:|---:|---:|
| 10 | 11.94 | 11.86 | +0.65% |
| 100 | 7.10 | 6.58 | +7.83% |
| 600 | 4.51 | 4.11 | +9.76% |

It just wasn't flagged, because **hybrids have no FLA-init converter**, so they
were always reported as INFO (curve/speed valid, exact loss not expected to
match). The number only jumped out once the sanity table put the % deltas of
all five models side by side.

## Root cause (static analysis)

The offset is **not a bug**. Our production `mamba_hybrid` and FLA's mamba2
reference are **different architectures**:

| param | production (GDN-matched) | FLA mamba2 ref |
|---|---:|---:|
| hidden_size | 1024 | **1216** |
| ffn_hidden_size | 4096 | **4864** |
| mamba_state_dim | 64 | **128** |
| mamba_num_groups | 8 | **1** |
| norm_eps | 1e-6 | **1e-5** |
| **MLPs** | **after every mixer (12)** | **only on MLA layers (3)** |
| params | **356.7M** | 301.3M |

Two things to note:

1. **Our model is actually bigger (356.7M vs 301.3M)** yet trains to a higher
   loss — so "FLA is a bigger model" is *false*. The params are in different
   places.
2. The decisive difference is **MLP placement**. FLA's `Mamba2Block`
   (`fla/models/mamba2/modeling_mamba2.py`) attaches a GatedMLP **only when
   `self.is_attn_layer`** — the 9 pure-Mamba2 blocks have no MLP (original
   Mamba design). Our Megatron `HybridStack` pattern
   `*-M-M-M-...` puts an MLP after *every* mixer (12 MLPs). Plus FLA uses a
   wider hidden (1216), 2× SSM state (128), and a single B/C group.

Why didn't the GDN hybrid show this? Because FLA's `gated_deltanet` hybrid
*does* use a transformer-style block (mixer + MLP every layer), which is
exactly what our GDN hybrid pattern reproduces → +1.37%. FLA's mamba2 does not,
and our config doesn't replicate it → +10.56%. The config header for
`zebra_llama_300M_mamba_hybrid.yaml` says it was sized to match the **GDN
hybrid 300M** for an apples-to-apples mamba-vs-GDN comparison — never to match
FLA's `mamba2_300M`. So the FLA mamba2 column was never a valid parity baseline
for that config.

## The proof experiment

`zebra_llama_300M_mamba_hybrid_flaexact.yaml` is a **byte-for-byte
architectural replica** of FLA's `mamba2_300M_hybrid.json`:
hidden 1216, ffn 4864, state 128, n_groups 1, eps 1e-5, and MLP **only** on the
3 MLA layers (pattern `*-MMM*-MMM*-MMM`). Everything else (LR schedule, data,
FLA runtime knobs, seed) is identical to the production mamba sanity run, so the
**only** changed variable is the architecture.

- **Hypothesis:** the +10% offset is architecture/hyperparameter, not a bug.
- **Prediction:** the replica lands on FLA's curve within the ~1-2% no-FLA-init
  band (same band the GDN hybrid sits in).
  - within band  → offset fully explained by config; case closed.
  - still ~10%   → real bug in the Mamba2 mixer; investigate further.

### RESULT (run on 8×MI300X, 700 iters)

The FLA-exact replica (301.3M params, matching FLA's 301.3M) ended at **+10.85%**
— essentially identical to production's +10.56%. **Architecture is NOT the cause.**

| step | FLA ref | production Δ% | FLA-exact Δ% |
|---:|---:|---:|---:|
| 100 | 6.5825 | +7.78% | +6.53% |
| 400 | 4.4923 | +8.62% | +8.42% |
| 700 | 3.9673 | +10.56% | +10.85% |

Since the GDN hybrid (same MLA + MLP scaffolding, same data, also no FLA-init)
matches FLA within +1.37%, the residual gap is specific to the **Mamba2 mixer**.

## Phase 2: init vs mixer-bug (FLA-init test)

We further verified data order (all FLA 300M runs use `batch=128, accum=1`),
weight decay (Megatron excludes all 1-D params: A_log/D/dt_bias), and init
schemes (special params identical; in_proj `normal(0,0.02)` identical; out_proj
scaling within ~8%). So the remaining suspect is **initialization sensitivity**
of the Mamba2 SSM vs a real forward bug.

`tools/convert_fla_mamba2_init_to_megatron.py` loads FLA's *exact* mamba2 weights
into the FLA-exact Megatron model (Mamba2 mixer = clean direct copy; MLA fuses
`k_rope` into `linear_kv_down_proj`; MLP `[gate,up]`→`fc1`). Then:

```bash
cd /home/vanbhati@amd.com/Primus
bash sanity_before_push/mamba_parity/run_mamba_parity.sh flainit   # builds ckpt + trains
python3 sanity_before_push/mamba_parity/compare_mamba_parity.py
```

- **iter-10 mapping check:** with correct FLA-init, iter-10 loss must ≈ FLA's
  (~11.86). If it's off, the converter (likely the MLA rope/ordering) needs a fix
  before trusting the curve.
- **step-700 verdict:**
  - within ~2% → the offset was **initialization**; Mamba2 mixer is correct.
  - still ~10% (with iter-10 matching) → a **real Mamba2 mixer/numerics bug**.

### Run it (8×MI300X, same container as the sanity run)

```bash
cd /home/vanbhati@amd.com/Primus
bash sanity_before_push/mamba_parity/run_mamba_parity.sh
python3 sanity_before_push/mamba_parity/compare_mamba_parity.py
```

~700 iters (warmup 200 + 500 steady), roughly the same wall time as the
production mamba sanity run.

### Optional ablations (attribute the gap to each factor)

To split the contribution of MLP-placement vs dims, clone the FLA-exact config
and revert one variable at a time (e.g. keep `*-M-M-M-...` but FLA dims; or FLA
MLP placement but hidden 1024). Not required for the proof, but useful if you
want a per-factor breakdown.

## Files

- `300M_mamba_hybrid_flaexact.yaml` — experiment run config (gitignored).
- `../../primus/configs/models/megatron/zebra_llama_300M_mamba_hybrid_flaexact.yaml`
  — the FLA-exact model (EXPERIMENT-ONLY; delete before merge).
- `run_mamba_parity.sh` — launcher.
- `compare_mamba_parity.py` — 3-way (FLA / production / FLA-exact) loss table.
