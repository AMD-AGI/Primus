"""Record / verify the per-stack BLAS first-use replay table (dlrm_v4/train/blas_replay_table_<stack>.jsonl).

WHY. The ROCm 10.2 hipBLASLt (1.5, 100500-3dc098eb) splits its gfx950 Tensile libraries into lazily loaded files
that depend on the problem size: ..._Contraction_l_Ailk_Bljk_Cijk_Dijk_CU256_ID75a0_gfx950.co (475.6 MB, 826
kernels), ..._Ailk_Bjlk_..._CU256_ID75a0 (227 MB), ..._ID75a3-75a2 (401 + 235 MB) and more. The heuristic picks a
solution from one of them the first time a GEMM's size falls into its region, and hipModuleLoad of that file then
costs 0.35-1.15 s inside the timed region (full-run probes 383256 / 383263 / 383264 / 383427 / 383472: the
PREPROC_DEDUP unique chain's FusedLinearNorm backward, ops/fused_linear_norm.py:291/292, at rank-specific steps in
windows 43-56 and on all ranks at window 63). blas_warmup's M=16 pass cannot reach them (M-independent first-use
costs only), and the v26.3 hipBLASLt (100202) keeps these solutions in files loaded at init, so the table is
per stack.

WHAT. The FusedLinearNorm GEMMs (dlrm_v4/train/blas_warmup._fused_linear_norm call forms, exactly the operand
layouts of ops/fused_linear_norm._LinearNormFunction: fwd addmm(b, x[M,K], W[N,K].t()) = GemmAndBias TN, dgrad
dh[M,N] @ W (NN), wgrad dh.t() @ x (NT); bf16; TunableOp on, tuning off, as in production) at EVERY row count the
dedup chain can produce: training R = 2 * M_pad, M_pad = max(256, round_up(min(U, T), 256)), U < 2^21
(ops/preproc_dedup.KEY_U_MAX) -> R in 512 * [1, 8192]; eval R = M_pad -> forward at 256 * [1, 8192]. Row counts are
the module's structural bounds, not data. A call whose host time (synchronized) exceeds --slow-ms is a first use
(library load, handle / operator construction); its (K, N, op, M) is a table row. Replaying the rows once before
RUN_START makes every library any reachable row count needs resident.

  scan   python gen_blas_replay_table.py scan <part> <nparts> <out.jsonl>
  merge  python gen_blas_replay_table.py merge <table.jsonl> <scan parts...>
  verify python gen_blas_replay_table.py verify <part> <nparts> <table.jsonl> <out.jsonl>
         (replays the table through blas_warmup's loader, then rescans with a shifted interleave: expect 0 slow)
"""

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")

# (K = Din, N = Dout) of the dedup unique chain's FusedLinearNorm pairs: content L1 512->256, additional L1
# 1024->256, both L2 256->512 (ops/preproc_dedup.py runs c and a through FusedLinearNormMLP on R rows).
PAIRS = ((512, 256), (1024, 256), (256, 512))
TRAIN_STEP, TRAIN_N = 512, 8192  # R = 2 * M_pad, M_pad <= 2^21
EVAL_STEP, EVAL_N = 256, 8192  # eval R = M_pad (forward only)


def grid(part, nparts, shift=0):
    """[(M, ops)] in increasing M; this process takes every nparts-th point (offset part + shift)."""
    pts = {}
    for k in range(1, TRAIN_N + 1):
        pts.setdefault(TRAIN_STEP * k, set()).update(("fwd", "dgrad", "wgrad"))
    for k in range(1, EVAL_N + 1):
        pts.setdefault(EVAL_STEP * k, set()).add("fwd")
    ms = sorted(pts)
    return [(m, sorted(pts[m])) for i, m in enumerate(ms) if i % nparts == (part + shift) % nparts]


def make_bufs(torch, dev, m_max):
    bf = torch.bfloat16
    kmax = max(k for k, _ in PAIRS)
    nmax = max(n for _, n in PAIRS)
    x = torch.full((m_max, kmax), 0.0625, dtype=bf, device=dev)
    dh = torch.full((m_max, nmax), 0.0625, dtype=bf, device=dev)
    ws = {(k, n): torch.full((n, k), 0.0625, dtype=bf, device=dev) for k, n in PAIRS}
    bs = {(k, n): torch.full((n,), 0.0625, dtype=bf, device=dev) for k, n in PAIRS}
    return x, dh, ws, bs


def run_op(torch, bufs, k, n, op, m):
    x_big, dh_big, ws, bs = bufs
    # contiguous [m, k] / [m, n] operands with natural leading dimensions (as _LinearNormFunction's saved x_c / dh)
    xv = x_big.narrow(0, 0, m)
    if k != x_big.shape[1]:
        xv = _cache_view(x_big, k, m)
    dhv = dh_big.narrow(0, 0, m)
    if n != dh_big.shape[1]:
        dhv = _cache_view(dh_big, n, m)
    w = ws[(k, n)]
    if op == "fwd":
        return torch.addmm(bs[(k, n)], xv, w.t())
    if op == "dgrad":
        return dhv @ w
    return dhv.t() @ xv


_views = {}


def _cache_view(big, width, m):
    """A contiguous [m, width] view on big's storage (row-major, ld = width): big is allocated with the widest
    row, so reinterpreting its flat storage as [rows * maxw / width, width] keeps every operand contiguous."""
    key = (big.data_ptr(), width)
    v = _views.get(key)
    if v is None:
        flat = big.view(-1)
        rows = flat.numel() // width
        v = flat.narrow(0, 0, rows * width).view(rows, width)
        _views[key] = v
    return v.narrow(0, 0, m)


def scan(part, nparts, out, shift=0, slow_ms=40.0, table=None):
    import torch

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    t_init = time.time()
    if table is not None:
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "..",
                "primus",
                "backends",
                "recommendation",
                "dlrm_v4",
                "src",
            ),
        )
        from generative_recommenders.dlrm_v4.train import blas_warmup

        rep = blas_warmup.replay_blas_table(dev, path=table)
        print("REPLAY " + json.dumps(rep), flush=True)
    pts = grid(part, nparts, shift)
    m_max = max(m for m, _ in pts)
    bufs = make_bufs(torch, dev, m_max)
    torch.cuda.synchronize()
    rows = []
    n_calls = 0
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for m, ops in pts:
            for k, n in PAIRS:
                for op in ops:
                    torch.cuda.synchronize()
                    a = time.perf_counter()
                    y = run_op(torch, bufs, k, n, op, m)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - a) * 1e3
                    del y
                    n_calls += 1
                    if ms > slow_ms:
                        r = dict(K=k, N=n, op=op, M=m, ms=round(ms, 1), part=part, t=round(time.time(), 3))
                        rows.append(r)
                        print("SLOW " + json.dumps(r), flush=True)
    import torch.cuda.tunable as tunable

    meta = dict(
        validators=dict(tunable.get_validators()),
        cu=torch.cuda.get_device_properties(dev).multi_processor_count,
        arch=getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "?"),
        calls=n_calls,
        points=len(pts),
        seconds=round(time.time() - t0, 1),
        init_s=round(t0 - t_init, 1),
        part=part,
        nparts=nparts,
        shift=shift,
        slow_ms=slow_ms,
        replayed=table is not None,
    )
    print("META " + json.dumps(meta), flush=True)
    with open(out, "w") as f:
        f.write(json.dumps(dict(meta=meta)) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def merge(table, parts):
    metas, rows = [], {}
    for p in parts:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if "meta" in d:
                    metas.append(d["meta"])
                    continue
                key = (d["K"], d["N"], d["op"], d["M"])
                rows.setdefault(key, []).append(d["ms"])
    v0 = metas[0]["validators"]
    assert all(
        m["validators"] == v0 and m["cu"] == metas[0]["cu"] for m in metas
    ), "parts from different stacks"
    hdr = dict(
        kind="blas_replay_table",
        stack=dict(validators=v0, cu=metas[0]["cu"], arch=metas[0]["arch"]),
        generator="tools/recommendation/dlrm_v4/gen_blas_replay_table.py",
        pairs=[list(p) for p in PAIRS],
        grid=dict(train=[TRAIN_STEP, TRAIN_N], eval=[EVAL_STEP, EVAL_N]),
        scan_parts=len(metas),
        slow_ms=metas[0]["slow_ms"],
        note="rows = (K, N, op, M) whose first call took > slow_ms in a dense scan; replay before RUN_START",
    )
    with open(table, "w") as f:
        f.write(json.dumps(hdr) + "\n")
        for key in sorted(rows, key=lambda r: (r[3], r[0], r[1], r[2])):
            k, n, op, m = key
            f.write(json.dumps(dict(K=k, N=n, op=op, M=m, ms=max(rows[key]))) + "\n")
    print(f"wrote {table}: {len(rows)} rows from {len(metas)} parts")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        scan(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif cmd == "merge":
        merge(sys.argv[2], sys.argv[3:])
    elif cmd == "verify":
        scan(int(sys.argv[2]), int(sys.argv[3]), sys.argv[5], shift=1, table=sys.argv[4])
    else:
        raise SystemExit(__doc__)
