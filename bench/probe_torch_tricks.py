"""Do the launch-saving torch spellings this work package wants actually work?

Each probe answers one yes/no question that decides whether a fusion is legal,
and checks the numbers rather than just the absence of an exception.

    python bench/probe_torch_tricks.py
"""

from __future__ import annotations

import json

import torch


def launches(fn, iters=3, warmup=2):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if getattr(e, "device_type", None) == DeviceType.CUDA) / iters


def main():
    dev = "cuda"
    out = {}
    nb, c, k, v = 4096, 64, 128, 128

    do = torch.randn(nb, c, v, device=dev)
    dvt = torch.randn(nb, c, v, device=dev)
    st = torch.randn(nb, v, k, device=dev)

    # 1. bmm into a strided (batched-strided) slice of a [nb, 2C, K] buffer,
    #    which is what replaces `torch.cat((do @ st, -(dvt @ st)))`.
    ref = torch.cat((do @ st, -(dvt @ st)), dim=-2)
    try:
        buf = torch.empty(nb, 2 * c, k, device=dev)
        torch.bmm(do, st, out=buf[:, :c])
        torch.baddbmm(buf[:, c:], dvt, st, beta=0.0, alpha=-1.0, out=buf[:, c:])
        out["bmm_into_slice"] = {
            "ok": True,
            "max_abs_vs_cat": (buf - ref).abs().amax().item(),
            "launches": launches(
                lambda: (
                    torch.bmm(do, st, out=buf[:, :c]),
                    torch.baddbmm(buf[:, c:], dvt, st, beta=0.0, alpha=-1.0, out=buf[:, c:]),
                )
            ),
        }
    except Exception as exc:  # noqa: BLE001
        out["bmm_into_slice"] = {"ok": False, "error": repr(exc)[:200]}
    out["cat_launches"] = launches(lambda: torch.cat((do @ st, -(dvt @ st)), dim=-2))

    # 2. `(a * b).sum(-1)` vs vecdot -- the sweep adjoint's `d_dec`.
    a = torch.randn(nb, k, v, device=dev)
    b = torch.randn(nb, k, v, device=dev)
    r0 = (a * b).sum(-1)
    try:
        r1 = torch.linalg.vecdot(a, b, dim=-1)
        out["vecdot"] = {
            "ok": True,
            "max_abs": (r1 - r0).abs().amax().item(),
            "launches": launches(lambda: torch.linalg.vecdot(a, b, dim=-1)),
        }
    except Exception as exc:  # noqa: BLE001
        out["vecdot"] = {"ok": False, "error": repr(exc)[:200]}
    out["mul_sum_launches"] = launches(lambda: (a * b).sum(-1))

    # 3. cumsum with the output aliasing the input -- would let the layout copy
    #    and the within-chunk cumsum share one buffer.
    x = torch.randn(nb, c, k, device=dev)
    ref_cs = x.cumsum(-2)
    y = x.clone()
    try:
        torch.cumsum(y, dim=-2, out=y)
        out["cumsum_aliased"] = {"ok": True, "max_abs": (y - ref_cs).abs().amax().item()}
    except Exception as exc:  # noqa: BLE001
        out["cumsum_aliased"] = {"ok": False, "error": repr(exc)[:200]}

    # 4. does a bf16 x fp32 multiply promote in one launch?
    hb = torch.randn(nb, c, k, device=dev, dtype=torch.bfloat16)
    f32 = torch.randn(nb, c, k, device=dev)
    out["promote_mul"] = {
        "dtype": str((hb * f32).dtype),
        "launches": launches(lambda: hb * f32),
        "launches_explicit_float": launches(lambda: hb.float() * f32),
        "max_abs": (hb * f32 - hb.float() * f32).abs().amax().item(),
    }

    # 5. can Inductor fuse the chunk-prep adjoint's ~14 elementwise ops into one
    #    kernel? That chain is the single largest block of glue in the backward,
    #    and if `torch.compile` collapses it there is no need to hand-write it.
    qf = torch.randn(nb, c, k, device=dev)
    kf = torch.randn(nb, c, k, device=dev)
    cg = -torch.rand(nb, c, k, device=dev).cumsum(-2)
    d_qw = torch.randn(nb, 2 * c, k, device=dev, dtype=torch.bfloat16)
    d_kgam = torch.randn(nb, c, k, device=dev)
    d_kg = torch.randn(nb, c, k, device=dev, dtype=torch.bfloat16)
    d_dec = torch.randn(nb, k, device=dev)

    def prep_bwd(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, chunk):
        gamma = cg.exp()
        chunk_total = cg[:, -1:, :]
        e_fac = (chunk_total - cg).exp()
        d_qf = d_qw[:, :chunk] * gamma
        a = d_kgam * gamma
        b = d_kg * e_fac
        d_kf = a + b
        d_cg = torch.addcmul(qf * d_qf, kf, a - b)
        d_ct = (kf * b).sum(dim=-2) + d_dec * chunk_total.reshape(d_dec.shape).exp()
        d_cg = torch.cat((d_cg[:, : chunk - 1], (d_cg[:, chunk - 1] + d_ct).unsqueeze(1)), dim=1)
        return d_qf, d_kf, d_cg

    ref = prep_bwd(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, c)
    out["prep_bwd_eager_launches"] = launches(
        lambda: prep_bwd(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, c)
    )
    try:
        compiled = torch.compile(prep_bwd, dynamic=False)
        got = compiled(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, c)
        out["prep_bwd_compiled"] = {
            "ok": True,
            "max_abs": max((g - r).abs().amax().item() for g, r in zip(got, ref)),
            "launches": launches(
                lambda: compiled(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, c), iters=3, warmup=1
            ),
        }
    except Exception as exc:  # noqa: BLE001
        out["prep_bwd_compiled"] = {"ok": False, "error": repr(exc)[:400]}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
