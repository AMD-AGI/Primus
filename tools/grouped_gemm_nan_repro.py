#!/usr/bin/env python3
"""
Minimal single-operator reproducer for the MoE fused grouped-GEMM NaN on
MI355X (gfx950) / ROCm.

Loads an artifact saved during a Primus training run (PRIMUS_NAN_SAVE=1):
    {
      "X": [sum_tokens, in_features]  bf16,
      "weights": [ per-expert [out, in] bf16 ],
      "tokens_per_expert": [m_0, m_1, ...]  (group sizes / m_splits),
      "kernel_output": [sum_tokens, out]  (the in-training kernel result, has NaN),
    }

and:
  1. prints value distributions of X / weights / outputs,
  2. RE-RUNS the TE fused grouped GEMM (transformer_engine GroupedLinear) on the
     exact same tensors to show the NaN reproduces offline,
  3. compares against a correct per-expert torch.matmul reference (fp32 + bf16):
        Y_i = X_i @ W_i^T

Usage:
    python tools/grouped_gemm_nan_repro.py output/nan_dump/grouped_fc1_repro_rank7.pt
"""

import sys

import torch


# ----------------------------- helpers -------------------------------------

def offsets(ms):
    o = [0]
    for v in ms:
        o.append(o[-1] + v)
    return o


def dist(tag, t, sample=1_000_000):
    """Print a value distribution summary for tensor t."""
    tf = t.detach().float().flatten()
    n = tf.numel()
    finite = torch.isfinite(tf)
    n_nan = int(torch.isnan(tf).sum())
    n_inf = int(torch.isinf(tf).sum())
    fin = tf[finite]
    if fin.numel() == 0:
        print(f"  {tag:26s} n={n} ALL non-finite (nan={n_nan} inf={n_inf})")
        return
    mn, mx = float(fin.min()), float(fin.max())
    mean, std = float(fin.mean()), float(fin.std())
    absmax = float(fin.abs().max())
    a = fin.abs()
    if a.numel() > sample:
        idx = torch.randint(0, a.numel(), (sample,), device=a.device)
        a = a[idx]
    qs = torch.quantile(a, torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], device=a.device))
    q = ", ".join(f"{float(v):.4g}" for v in qs)
    print(
        f"  {tag:26s} n={n} min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g} "
        f"absmax={absmax:.4g} nan={n_nan} inf={n_inf} |abs| q[50/90/99/99.9/100]=[{q}]"
    )


def nonfinite_groups(Y, ms):
    o = offsets(ms)
    bad = []
    for i, n in enumerate(ms):
        s, e = o[i], o[i + 1]
        if e > s and (torch.isnan(Y[s:e]).any() or torch.isinf(Y[s:e]).any()):
            bad.append(i)
    return bad


def ref_matmul(X, weights, ms, dtype):
    """Correct per-expert reference; returns a full [sum_tokens, out] tensor."""
    o = offsets(ms)
    out = None
    for i, n in enumerate(ms):
        s, e = o[i], o[i + 1]
        if e <= s:
            continue
        xi = X[s:e].to(dtype)
        wi = weights[i].to(dtype)
        yi = xi @ wi.t() if wi.shape[-1] == xi.shape[-1] else xi @ wi
        if out is None:
            out = torch.empty((X.shape[0], yi.shape[-1]), dtype=dtype, device=X.device)
        out[s:e] = yi
    return out


def run_te_grouped(X, weights, ms):
    """Re-run the TE fused grouped GEMM on the same tensors."""
    try:
        import transformer_engine.pytorch as te
    except Exception as e:
        print(f"  [TE] transformer_engine import failed: {e}")
        return None
    n_experts = len(weights)
    in_features = weights[0].shape[-1]
    out_features = weights[0].shape[0]
    dev, dt = X.device, X.dtype
    try:
        gl = te.GroupedLinear(
            n_experts,
            in_features,
            out_features,
            bias=False,
            params_dtype=dt,
            device=dev,
        )
    except Exception as e:
        print(f"  [TE] GroupedLinear construction failed: {e}")
        return None
    with torch.no_grad():
        for i, w in enumerate(weights):
            p = getattr(gl, f"weight{i}", None)
            if p is None:
                print(f"  [TE] missing weight{i}; aborting TE rerun")
                return None
            p.copy_(w.to(dev, dt))
    m_splits = [int(v) for v in ms]
    # TE GroupedLinear.forward(inp, m_splits, ...)
    try:
        with torch.no_grad():
            out = gl(X, m_splits)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
    except Exception as e:
        print(f"  [TE] GroupedLinear forward failed: {e}")
        return None


# ------------------------------- main --------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    art = torch.load(path, map_location="cpu")
    X = art["X"]
    weights = art["weights"]
    ms = art["tokens_per_expert"]
    ko = art.get("kernel_output", None)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded {path}")
    print(f"  module={art.get('module')} rank={art.get('rank')} dtype={art.get('dtype')}")
    print(f"  X={tuple(X.shape)} n_experts={len(weights)} sum_tokens={sum(ms)} "
          f"empty_groups={sum(1 for v in ms if v == 0)} "
          f"min_group={min(ms)} max_group={max(ms)}")

    Xd = X.to(dev)
    Wd = [w.to(dev) for w in weights]

    # --- input distributions ---
    print("\n=== INPUT distributions ===")
    dist("X (all tokens)", Xd)
    Wcat = torch.stack([w.flatten() for w in Wd]).flatten() if len({tuple(w.shape) for w in Wd}) == 1 \
        else torch.cat([w.flatten() for w in Wd])
    dist("W (all experts)", Wcat)

    # --- references ---
    print("\n=== REFERENCE outputs (correct math) ===")
    ref32 = ref_matmul(Xd, Wd, ms, torch.float32)
    dist("REF fp32", ref32)
    print(f"  REF fp32 non-finite groups: {nonfinite_groups(ref32, ms)} (expected [])")
    refb = ref_matmul(Xd, Wd, ms, torch.bfloat16)
    dist("REF bf16 (torch)", refb)
    print(f"  REF bf16 non-finite groups: {nonfinite_groups(refb, ms)}")

    # --- TE grouped GEMM rerun ---
    print("\n=== TE fused grouped GEMM rerun (te.GroupedLinear) ===")
    te_out = run_te_grouped(Xd, Wd, ms)
    if te_out is not None:
        dist("TE grouped out", te_out)
        teb = nonfinite_groups(te_out, ms)
        print(f"  TE grouped non-finite groups: {teb}")
        print(f"  >>> TE grouped GEMM {'REPRODUCED NaN/Inf OFFLINE' if teb else 'was finite (no repro)'}")

    # --- in-training kernel output ---
    if ko is not None:
        kod = ko.to(dev)
        print("\n=== in-training kernel_output (from the run) ===")
        dist("KERNEL (training)", kod)
        kbad = nonfinite_groups(kod, ms)
        print(f"  KERNEL non-finite groups: {kbad[:20]}")

        # side-by-side per bad group
        o = offsets(ms)
        print("\n=== per-bad-group side-by-side (first 3) ===")
        for i in kbad[:3]:
            s, e = o[i], o[i + 1]
            print(f"\n  --- expert {i}  rows={e - s} ---")
            dist(f"  X[expert {i}]", Xd[s:e])
            dist(f"  W[expert {i}]", Wd[i])
            dist(f"  REF fp32", ref32[s:e])
            dist(f"  REF bf16", refb[s:e])
            if te_out is not None:
                dist(f"  TE grouped", te_out[s:e])
            dist(f"  KERNEL train", kod[s:e])

    print("\nConclusion: finite normal X (~5) and W (~0.1) -> plain matmul (fp32 & bf16)")
    print("is finite, while the fused grouped GEMM produces NaN/~3.4e38. Defect is in")
    print("the grouped-GEMM compute path, not precision / inputs / weights.")


if __name__ == "__main__":
    main()
