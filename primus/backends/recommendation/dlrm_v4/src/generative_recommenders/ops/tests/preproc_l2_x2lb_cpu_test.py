# pyre-unsafe
"""X2L-B (gin item (93); ops/hip/preproc_l2_x2lb.py + x2lb_pass.cu) on CPU: no GPU, no hipcc, no FlyDSL
(DESIGN_stageC 8.1; the GPU half is preproc_l2_x2lb_test.py).

The model tests drive the REAL E-M3a autograd Functions of ops/preproc_fp8_l2.py:
_LinearSwishLNEmitFunction for the content and additional MLPs, and _GroupedLinearLNTailFP8Function for the three-MLP
tail with a BF16 action entry. Every kernel those Functions call is replaced by a recording CPU stand-in.

The X2L-B side runs the REAL plumbing: publish / attach / try_fused / take, and the REAL _CudaRuntime and _Lib. The
kernel library underneath is EmuLib, a CPU emulator of x2lb_pass.cu's extern "C" ABI. It reads the pointers it is
handed back as host tensors. It computes da1 and PR #7's swish backward with the same torch expressions as the
stand-ins of the shipped _scaled_mm and SwishLN backward, so the fused and the shipped paths are equal by
construction. A wiring error anywhere between try_fused and the C ABI therefore shows as a gradient difference.

Groups (DESIGN_stageC 8.1 #1-#7):
 1. Knob off, run in a child process. With the knob unset, "0", or the module's import made to fail, the module is
    never imported, and the launch log and every leaf gradient are today's. When X2LB_BASE_TREE names an unpatched
    recommendation/ directory, both are also compared to that tree. With the knob on but unarmed, the result is
    today's plus one NOT armed line.
 2. Plumbing: the production shape fuses both MLPs and the gradients equal the shipped run's. Also covered:
    - retain_graph x2;
    - every capture / partial-backward case declines (will_run and the live-proxy rule), with the leaf and captured
      gradients equal to the shipped run's;
    - the documented residual (a hook registered on the proxy, then the proxy dropped);
    - create_graph and anomaly mode decline;
    - saved-tensor hooks, eval and no_grad publish nothing;
    - one MLP only;
    - in-place on a saved tensor;
    - the handshake, the sentinel and the latch on a launch error;
    - a handle holds no tensor and forms no cycle.
 3. Declines: every tensor_reason / call_reason class, and one DECLINED line per class (digits normalized).
 4. Runtime and loader:
    - ctypes argument order by the C parameter names, rc handling, PR #7's finalize call form;
    - the ABI checks;
    - the cache key, trusted and private directories, rebuild-once, the build timeout;
    - the census, the ISA gate's pinned and full modes, the source / ISA-file / flag pins.
 5. Prebuild:
    - phase 1 IGNORED / FAILED CLOSED / armed;
    - phase 2's check points on the emulator (checks (a)-(f) catch each defect; the live-set peak is exactly 5U +
      canaries);
    - headroom, latch ownership (PR #7 is never latched), the arming preconditions;
    - the blas_warmup hooks (RNG restore, empty_cache, result token).
 6. Gin: the item (93) block and its binding False, the env bootstrap.

Child processes run with -B and PYTHONDONTWRITEBYTECODE=1.
Optional inputs:
- X2LB_BASE_TREE: an unpatched recommendation/ for the knob-off identity.
- X2LB_LAB_CACHE: the lab's x2lb_cache directory of job 70428, for the real ISA gate on the lab assembly.
"""

import contextlib
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch
from generative_recommenders.ops import preproc_fp8_l2 as L2
from generative_recommenders.ops.triton import triton_fp8 as TF8
from generative_recommenders.ops.triton import triton_layer_norm as TLN
from generative_recommenders.ops.triton import triton_preproc_fp8 as TPF
from generative_recommenders.ops.triton import triton_preproc_tail as TPT

ENV = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
MOD = "generative_recommenders.ops.hip.preproc_l2_x2lb"
_HERE = os.path.dirname(os.path.abspath(__file__))
_GR = os.path.dirname(os.path.dirname(_HERE))  # .../generative_recommenders
_TREE = os.path.dirname(_GR)  # .../recommendation
_GIN = os.path.join(_GR, "dlrm_v4", "train", "gin", "yambda_5b.gin")
_HIP = os.path.join(_GR, "ops", "hip")
# The integration patch keeps X2L-B off; the ship flip changes this one expectation with the gin binding.
_GIN_BINDING = "apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD = True"
E4, E5, BF, F32 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32
M_TEST = 384
NCU = 5  # the emulated device's CU count: plan grid min(ceil(M / 64), 5)

# The sealed kernel and ISA-gate files (labkit PRED_x2lb.md section 6).
_KIT_SRC_SHA = "a78f4aa3611f6b6211d6f197848cc40661572390c35c6d78853d3064e5eebb55"
_KIT_ISA_SHA = {
    "isa_check.py": "7680f8138aa0648ccd4ef385779ce3058406592d16870840b4532e593ffec7f2",
    "isa_dataflow.py": "08e7789e15300da4b64aa6ea3b377c944a2ba882c11dd62a060b19280d613384",
    "isa_exec.py": "fd2121ed32c6c2ea2dc25cb7bf3a0ad3f85baca2744537f076617b335df39641",
    "emu/x2lb_emu.py": "468a9abf44fed5f23c8841e2bd9f022a506fde867f0459685d2e2ce900beec1e",
}
# labkit x2lb.py _FLAGS (L11's production loader flags)
_KIT_FLAGS = ("-DKITTENS_CDNA4", "-std=c++20", "-w", "-O3", "-mllvm", "-amdgpu-sched-strategy=iterative-ilp",
              "-ffp-contract=off", "--save-temps", "-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage")
# kernel-lab job 70428's container build (ROCm 7.2.1): its -Rpass-analysis remarks, in the format of that build's
# log (labkit runs/run_70428/correct/g0/main/remarks; x2lb_cache/1c53548919fa9d49)
_LAB_KERNELS = (
    ("_ZN4x2lb8x2lb_finEPKfiPf", 20, 39, 8, 0),
    ("_ZN4x2lb9x2lb_passILi0ELi0EEEvNS_6ParamsE", 74, 220, 2, 99328),
    ("_ZN4x2lb9x2lb_passILi0ELi1EEEvNS_6ParamsE", 76, 224, 2, 99328),
    ("_ZN4x2lb9x2lb_passILi1ELi0EEEvNS_6ParamsE", 64, 140, 2, 99328),
)


def _remarks(override=None):
    """The lab's remark lines; override = {kernel: {field: value}} edits one kernel."""
    out = []
    for name, sgpr, vgpr, occ, lds in _LAB_KERNELS:
        loc = "x2lb_pass.cu:753:0:" if "fin" in name else "x2lb_pass.cu:401:0:"
        f = {"TotalSGPRs": sgpr, "VGPRs": vgpr, "AGPRs": 0, "ScratchSize [bytes/lane]": 0, "Dynamic Stack": "False",
             "Occupancy [waves/SIMD]": occ, "SGPRs Spill": 0, "VGPRs Spill": 0, "LDS Size [bytes/block]": lds}
        f.update((override or {}).get(name, {}))
        out.append(f"remark: {loc} Function Name: {name} [-Rpass-analysis=kernel-resource-usage]")
        for k, v in f.items():
            if v is not None:
                out.append(f"remark: {loc}     {k}: {v} [-Rpass-analysis=kernel-resource-usage]")
    return "\n".join(out) + "\n"


def _X():
    from generative_recommenders.ops.hip import preproc_l2_x2lb

    return preproc_l2_x2lb


# =================================================================================== CPU stand-ins (recording)
LOG = []


def _d(a):
    if isinstance(a, torch.Tensor):
        return f"T{tuple(a.shape)}:{str(a.dtype).replace('torch.', '')}:{tuple(a.stride())}"
    if isinstance(a, (list, tuple)):
        return "[" + ",".join(_d(x) for x in a) + "]"
    if isinstance(a, (bool, int, float, str, torch.dtype)) or a is None:
        return repr(a)
    return type(a).__name__


def rec(name, *args, **kw):
    LOG.append(name + "(" + ",".join([_d(a) for a in args] + [f"{k}={_d(v)}" for k, v in sorted(kw.items())]) + ")")


def _mm(a, b, scale_a, scale_b):
    """The dgrad's fp32 value, ONE expression for the stand-in and EmuLib (equal bits by construction)."""
    return (a.float() @ b.float()) * (scale_a * scale_b)


def _ln_bwd(dy, x, w, b, mean, rstd, swish):
    """PR #7's (and the Triton LN's) backward in fp32 torch: (dx, dW, dB, dx-sum). ONE function for the stand-in,
    the fake ln_bwd and EmuLib."""
    x, dy = x.float(), dy.float()
    xhat = (x - mean[:, None]) * rstd[:, None]
    if swish:
        s = torch.sigmoid(xhat * w.float() + b.float())
        t = dy * x * s * (1 - s)
        xu = w.float() * t
        dx = (xu - (xhat * (xhat * xu).mean(1, keepdim=True) + xu.mean(1, keepdim=True))) * rstd[:, None] + dy * s
        return dx, (t * xhat).sum(0), t.sum(0), dx.sum(0)
    g = w.float() * dy
    dx = (g - (xhat * (xhat * g).mean(1, keepdim=True) + g.mean(1, keepdim=True))) * rstd[:, None]
    return dx, (dy * xhat).sum(0), dy.sum(0), dx.sum(0)


def f_swishln256_fwd_emit(h1, w, b, eps, inv_scale, emu, tm=True, return_partials=False):
    rec("swishln256_fwd_emit", h1, w, b, inv_scale, tm=tm)
    x = h1.float()
    mean = x.mean(1)
    rstd = torch.rsqrt(x.var(1, unbiased=False) + eps)
    y = (x - mean[:, None]) * rstd[:, None] * w.float() + b.float()
    a1 = (x * torch.sigmoid(y)).to(BF).float()
    a1_rm = a1.to(E4)
    Mp = (h1.shape[0] + 127) // 128 * 128
    a1_tm = torch.zeros((256, Mp), dtype=torch.float32)
    a1_tm[:, : h1.shape[0]] = a1.t()
    a1_tm = a1_tm.to(E4) if tm else None
    return a1_rm, a1_tm, mean, rstd, a1.abs().amax().reshape(1)


def f_site_steady(partials, scale, pending, clipped, limit, snap):
    rec("site_steady", partials, scale, snap=snap)
    return scale.clone() if snap else None


def f_quantize_w2_dual(weight, bias=None, cast_bf16=False):
    rec("quantize_w2_dual", weight, cast_bf16=cast_bf16)
    w = weight.to(BF).float()
    s = (w.abs().amax() / 448.0).reshape(1)
    w8 = (w / s).to(E4)
    return w8, w8.t().contiguous(), s, (bias.to(BF) if cast_bf16 else None)


def f_scaled_mm(a, b, scale_a, scale_b, *, out_dtype, bias=None, scale_result=None):
    rec("_scaled_mm", a, b, scale_a, scale_b, out_dtype=out_dtype, bias=bias)
    o = _mm(a, b, scale_a, scale_b)
    if bias is not None:
        o = o + bias.float()
    return o.to(out_dtype)


def f_grouped_ln_residual_fwd(hs, ws, bs, eps, BLOCK_N=None, num_warps=None):
    rec("grouped_ln_residual_fwd", hs, ws, bs)
    out, means, rstds = None, [], []
    for h, w, b in zip(hs, ws, bs):
        x = h.float()
        m = x.mean(1)
        r = torch.rsqrt(x.var(1, unbiased=False) + eps)
        y = (x - m[:, None]) * r[:, None] * w.float() + b.float()
        out = y if out is None else out + y
        means.append(m)
        rstds.append(r)
    return out.to(hs[0].dtype), means, rstds


def f_ln512_bwd_dxsum_emit(dy, x, weight, mean, rstd, inv_scale, out=None, return_partials=False):
    rec("ln512_bwd_dxsum_emit", dy, x, weight, mean, rstd, inv_scale)
    dx, dw, db, dxs = _ln_bwd(dy, x, weight, torch.zeros_like(weight), mean, rstd, False)
    dh8 = (dx * inv_scale).clamp(-57344, 57344).to(E5)
    return dh8, dw.to(weight.dtype), db.to(weight.dtype), dxs, dx.abs().amax().reshape(1)


def f_wln_bwd_dxsum(dy, x, weight, bias, mean, rstd, eps, BLOCK_D, is_swish):
    rec("triton_weighted_layer_norm_bwd_dxsum", dy, x, weight, bias, mean, rstd, is_swish=is_swish)
    dx, dw, db, dxs = _ln_bwd(dy, x, weight, bias, mean, rstd, is_swish)
    return dx.to(x.dtype), dw.to(weight.dtype), db.to(weight.dtype), dxs


def f_wgrad_nn_b(dh8, a1_tm, dh_scale, a1_scale):
    rec("_wgrad_nn_b", dh8, a1_tm, dh_scale, a1_scale)
    M = dh8.shape[0]
    return ((dh8.float().t() @ a1_tm.float()[:, :M].t()) * (dh_scale * a1_scale)).contiguous(), "nn"


def f_grouped_bwd(ctx, dy, saved):
    rec("_grouped_bwd", dy)
    return None  # the stacked FlyDSL path needs a GPU: P3 per MLP


def standins():
    st = contextlib.ExitStack()
    for obj, name, fn in ((TPF, "swishln256_fwd_emit", f_swishln256_fwd_emit), (TPF, "site_steady", f_site_steady),
                          (TPF, "quantize_w2_dual", f_quantize_w2_dual),
                          (TPF, "ln512_bwd_dxsum_emit", f_ln512_bwd_dxsum_emit), (TF8, "_scaled_mm", f_scaled_mm),
                          (TPT, "grouped_ln_residual_fwd", f_grouped_ln_residual_fwd),
                          (L2, "triton_weighted_layer_norm_bwd_dxsum", f_wln_bwd_dxsum),
                          (L2, "_wgrad_nn_b", f_wgrad_nn_b), (L2, "_grouped_bwd", f_grouped_bwd)):
        st.enter_context(mock.patch.object(obj, name, fn))
    return st


# The shipped launch sequence of one train forward + backward (knob off): the two FP8 pair 0s, the tail (k = 0, 1
# FP8, k = 2 BF16), then the tail backward (k = 2, 1, 0) and the FP8 pair 0s (additional, then content).
_SHIPPED_OPS = [
    "swishln256_fwd_emit", "site_steady", "swishln256_fwd_emit", "site_steady",
    "quantize_w2_dual", "_scaled_mm", "quantize_w2_dual", "_scaled_mm", "grouped_ln_residual_fwd",
    "_grouped_bwd", "triton_weighted_layer_norm_bwd_dxsum",
    "ln512_bwd_dxsum_emit", "site_steady", "_scaled_mm", "_wgrad_nn_b",
    "ln512_bwd_dxsum_emit", "site_steady", "_scaled_mm", "_wgrad_nn_b",
    "triton_weighted_layer_norm_bwd_dxsum", "triton_weighted_layer_norm_bwd_dxsum",
]


def build(M=M_TEST, seed=7, train=True, keep_proxy=False, want_p0=False, ports=False, ix=False):
    """The E-M3a graph on CPU (the knob as the environment and preproc_fp8_l2 see it). Content (k = 0, x [M, 512])
    and additional (k = 1, x [M, 1024]) run _LinearSwishLNEmitFunction (their stride-0 proxies feed the tail); the
    action MLP (k = 2) is a BF16 entry. Like GroupedTail, the proxies are dropped after the tail's forward unless
    keep_proxy. ports: pair 0 gets an E-X2b port (a stride-0 bf16 [M, 256] leaf; the caller mocks _xs.bind / route).
    ix: x carries the Indexed-X tag (the caller mocks L2._ix with FakeIX)."""
    L2._sites.clear()
    g = torch.Generator().manual_seed(seed)

    def rn(*s):
        return torch.randn(*s, generator=g)

    leaves, plans, flat, proxies, p0s, portd = {}, [], [], {}, {}, {}
    for k, din in ((0, 512), (1, 1024), (2, 24)):
        x = rn(M, din).to(BF).requires_grad_()
        w1 = (rn(256, din) * 0.05).to(BF).requires_grad_()
        b1 = (rn(256) * 0.1).to(BF).requires_grad_()
        nw0 = (1 + 0.1 * rn(256)).requires_grad_()
        nb0 = (0.1 * rn(256)).requires_grad_()
        w2 = (rn(512, 256) * 0.05).to(BF).requires_grad_()
        b2 = (rn(512) * 0.1).to(BF).requires_grad_()
        nw1 = (1 + 0.1 * rn(512)).requires_grad_()
        nb1 = (0.1 * rn(512)).requires_grad_()
        leaves.update({f"{k}.x": x, f"{k}.w1": w1, f"{k}.b1": b1, f"{k}.nw0": nw0, f"{k}.nb0": nb0,
                       f"{k}.w2": w2, f"{k}.b2": b2, f"{k}.nw1": nw1, f"{k}.nb1": nb1})
        slot = L2.Slot(k)
        slot.train = train
        if k in L2.FP8_MLPS:
            slot.mode = "fp8"
            for role in ("a1", "dh2"):
                s = L2._site(role, k, x.device)
                s.scale.fill_(0.01)
                s.inv_scale.fill_(100.0)
                s.published = True
            port = None
            if ports:
                port = portd[k] = torch.empty_strided((M, 256), (0, 0), dtype=BF).requires_grad_()
            if ix:
                setattr(x, L2._IX_ATTR, ("fake-ix", k))
            y = L2._LinearSwishLNEmitFunction.apply(x, w1, b1, nw0, nb0, 1e-5, k, 2, train, slot,
                                                    types.SimpleNamespace(kind="port") if ports else None, port)
            if keep_proxy:
                proxies[k] = y
            if want_p0 and y.grad_fn is not None:
                p0s[k] = y.grad_fn
        else:
            h = torch.addmm(b1, x, w1.t())
            y = (h.float() * torch.sigmoid(h.float())).to(BF)  # stand-in pair 0 of the BF16 action MLP
        plans.append(slot)
        flat += [y, w2, b2, nw1, nb1]
        del y
    out = L2.grouped_tail_apply(1e-5, plans, flat)
    del flat  # GroupedTail drops _ys / flat when finish returns
    loss = (out.float() * torch.linspace(-1, 1, 512)).sum() if out.requires_grad else None
    return types.SimpleNamespace(leaves=leaves, plans=plans, out=out, loss=loss, proxies=proxies, p0s=p0s,
                                 ports=portd)


class FakeIX:
    """ops/ec_indexed_l1 on CPU for pair 0's Indexed-X branch: U = x itself, idx = arange(M); wgrad records the dh it
    receives and returns dh^T @ U[idx] in W1's dtype."""

    def __init__(self):
        self.dhs = []

    def fwd(self, x, w_c, b_c, w_dtype, name, train=True):
        M = x.shape[0]
        idx = torch.arange(M)
        h = torch.addmm(b_c, x, w_c.t())
        call = types.SimpleNamespace(indexed=True, ledger=types.SimpleNamespace(note_bwd=lambda routed: None))
        return types.SimpleNamespace(h=h, table=x.detach(), w_bwd=None, idx=idx, call=call, x=None)

    def wgrad(self, ix, dh, U, idx, dt):
        self.dhs.append(dh)
        return (dh.float().t() @ U[idx].float()).to(dt)


def _tensors(m, names):
    return [m.proxies[int(n[5:])] if n.startswith("proxy") else m.leaves[n] for n in names]


def run_case(case, m):
    """Run one backward variant on model m. Returns (leaf grads, captured grads)."""
    kind, names = case[0], case[1] if len(case) > 1 else ()
    caps = {}
    if kind == "plain":
        m.loss.backward()
    elif kind == "retain2":
        m.loss.backward(retain_graph=True)
        m.loss.backward()
    elif kind == "grad":
        caps.update(zip(names, torch.autograd.grad(m.loss, _tensors(m, names))))
    elif kind == "inputs":
        ts = _tensors(m, names)
        for n, t in zip(names, ts):
            if n.startswith("proxy"):
                t.retain_grad()
        m.loss.backward(inputs=ts)
        for n, t in zip(names, ts):
            if n.startswith("proxy"):
                caps[n] = t.grad
    elif kind == "anomaly":
        with torch.autograd.detect_anomaly(check_nan=True):
            m.loss.backward()
    elif kind == "create_graph":
        gs = torch.autograd.grad(m.loss, _tensors(m, names), create_graph=True)
        caps.update((n, g.detach()) for n, g in zip(names, gs))
    elif kind in ("hook_kept", "hook_dropped"):
        seen = {}
        for k in (0, 1):
            m.proxies[k].register_hook(lambda g, k=k: seen.__setitem__(f"hook{k}", g.detach().clone()))
        if kind == "hook_dropped":
            m.proxies.clear()
        m.loss.backward()
        caps.update(seen)
    else:
        raise ValueError(kind)
    leaves = {n: None if t.grad is None else t.grad.detach().clone() for n, t in m.leaves.items()}
    return leaves, caps


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def _sha(t):
    v = t.detach().contiguous()
    v = v.view(torch.int16) if v.element_size() == 2 else v.view(torch.int32) if v.element_size() == 4 else v
    return hashlib.sha256(v.numpy().tobytes()).hexdigest()[:16]


# =================================================================================== the emulated library
def _at(ptr, shape, dtype):
    """The host tensor at address ptr (no copy; writes go to the caller's tensor)."""
    n = int(math.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    raw = (ctypes.c_uint8 * n).from_address(int(ptr))
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).view(*shape)


class EmuLib:
    """x2lb_pass.cu's extern "C" ABI on host memory. S1 / S1+DA1 compute da1 = bf16(_mm(dh8, w8t^T, sa, sb)) and PR
    #7's swish row on it (_ln_bwd), the expressions of the shipped stand-ins. PART row 0 of each output o holds the
    column sums and the other rows 0, so a finalize that adds the rows returns exactly those sums. `defect` makes one
    named error (check_point tests); `rc_on` returns hip error 98 at that call number; `raise_on` raises there."""

    HIP_INVALID = 1

    def __init__(self, defect=None, rc_on=None, raise_on=None, variant=0, abi=None, impl_sha=None):
        X = _X()
        self.calls, self.fins = [], []
        self.defect, self.rc_on, self.raise_on = defect, rc_on, raise_on
        self.variant, self.abi = variant, X.ABI_VERSION if abi is None else abi
        self.impl_sha = X.SOURCE_SHA256 if impl_sha is None else impl_sha

        def mk(f):
            def fn(*a):
                return f(*a)
            return fn

        for name in X.SIGNATURES:  # plain functions: _Lib sets restype / argtypes on them
            setattr(self, name, mk(getattr(self, "_" + name)))

    def _x2lb_abi_version(self):
        return self.abi

    def _x2lb_variant(self):
        return self.variant

    def _x2lb_impl(self):
        return f"x2lb_pass emu src_sha256={self.impl_sha}".encode()

    def _x2lb_plan(self, M, num_cu):
        if M < 1 or M > (1 << 29) or num_cu < 1:
            return -self.HIP_INVALID
        return min(-(-M // 64), num_cu)

    def _x2lb_drain_last_error(self):
        return 0

    def _x2lb_launch(self, *a):
        return self._run("s1", *a)

    def _x2lb_launch_emit(self, *a):
        return self._run("emit", *a)

    def _x2lb_launch_s0(self, *a):
        return self._run("s0", *a)

    def _run(self, mode, dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream):
        self.calls.append(dict(mode=mode, M=M, grid=grid, part_rows=part_rows, stream=stream,
                               ptrs=dict(dh8=dh8, w8t=w8t, sa=sa, sb=sb, h1=h1, mean1=mean1, rstd1=rstd1, nw=nw,
                                         nb=nb, dh1=dh1, da1=da1, part=part)))
        n = len(self.calls)
        if self.raise_on == n:
            raise RuntimeError("emulated host-side failure")
        if self.rc_on == n:
            return 98
        a16 = (dh8, w8t, h1, nw, nb, dh1, part)
        if (mode == "s0" or part_rows != grid or not 1 <= M <= (1 << 29) or grid < 1
                or any(p is None or p % 16 for p in a16) or (mode == "emit" and (da1 is None or da1 % 16))):
            return self.HIP_INVALID
        a = _at(dh8, (M, 512), E5)
        w = _at(w8t, (256, 512), E4)
        s_a, s_b = _at(sa, (1,), F32), _at(sb, (1,), F32)
        da = _mm(a, w.t(), s_a, s_b).to(BF)
        dx, dw, db, dxs = _ln_bwd(da, _at(h1, (M, 256), BF), _at(nw, (256,), F32), _at(nb, (256,), F32),
                                  _at(mean1, (M,), F32), _at(rstd1, (M,), F32), True)
        dhv = dx.to(BF)
        d = self.defect
        if d == "dh1_vs_pr7" or (d == "s1_vs_emit" and mode == "s1") or (d == "grid7" and grid == 7):
            row = {"dh1_vs_pr7": 1, "s1_vs_emit": 2, "grid7": 3}[d]
            dhv.view(torch.int16)[row, 5] ^= 1
        if d == "canary" and mode == "emit":
            _at(dh1, (M + 1, 256), BF)[M, 0] = 1.0  # one bf16 past row M
        _at(dh1, (M, 256), BF).copy_(dhv)
        if mode == "emit":
            if d == "da1_bit":
                da.view(torch.int16)[0, 7] ^= 1
            _at(da1, (M, 256), BF).copy_(da)
        P = _at(part, (3, grid, 256), F32)
        last = P[2, grid - 1, 255].clone()
        P.zero_()
        P[0, 0], P[1, 0], P[2, 0] = dw, db, dxs
        if d == "part_hole" and mode == "emit":
            P[2, grid - 1, 255] = last  # this element is never written
        return 0

    def _x2lb_fin_launch(self, part, part_rows, out, stream):
        self.fins.append(dict(part=part, part_rows=part_rows, out=out, stream=stream))
        P, O = _at(part, (3, part_rows, 256), F32), _at(out, (3, 256), F32)
        for o in range(3):
            acc = P[o, 0].clone()
            for g in range(1, part_rows):
                acc += P[o, g]
            O[o] = acc
        return 0


def fake_lb(fail=False, fin_scale=1.0):
    """ops/flydsl/ln_bwd (PR #7) on CPU: build_finalize(256, 3, True) adds PART's rows in order, ln_bwd = _ln_bwd."""
    lb = types.ModuleType("generative_recommenders.ops.flydsl.ln_bwd")
    lb.fx = types.SimpleNamespace(Stream=lambda s: ("fx.Stream", s))
    lb.fin_calls, lb.ln_calls = [], []

    def build_finalize(D, nacc, w_f32):
        if (D, nacc, w_f32) != (256, 3, True):
            raise AssertionError(f"build_finalize{(D, nacc, w_f32)}: X2L-B must use PR #7's (256, 3, True)")

        def fin(part, dw, db, dxs, G, stream=None):
            lb.fin_calls.append(dict(G=G, stream=stream, part=part))
            for o, out in enumerate((dw, db, dxs)):
                acc = part[o, 0].clone()
                for g in range(1, int(G)):
                    acc += part[o, g]
                out.copy_(acc * fin_scale)
        return fin

    def ln_bwd(dy, x, weight, bias, mean, rstd, residual=None, *, is_swish=False, sum_dx=False, **kw):
        lb.ln_calls.append(dict(is_swish=is_swish, sum_dx=sum_dx, M=int(x.shape[0])))
        if fail:
            raise RuntimeError("injected PR #7 ln_bwd failure")
        dx, dw, db, dxs = _ln_bwd(dy, x, weight, bias, mean, rstd, is_swish)
        return dx.to(x.dtype), dw.to(weight.dtype), db.to(weight.dtype), dxs

    lb.build_finalize, lb.ln_bwd = build_finalize, ln_bwd
    return lb


class _Stream:
    def __init__(self, h):
        self.cuda_stream = h


STREAM0 = _Stream(0x51)


def make_lib(emu):
    X = _X()
    meta = dict(resources={X.KERNEL_S1: dict(unified_vgpr=220, vgpr_spill=0, scratch=0, lds=99328, occupancy=2)},
                isa=dict(ok=True, mode="pinned"), code_object=dict(status="ok", lab="identical",
                                                                   **X.LAB_CODE_OBJECT_SHA256))
    with mock.patch.object(ctypes, "CDLL", lambda path: emu):
        return X._Lib(Path("/emu/libx2lb_pass.so"), meta, 0.0, True)


# =================================================================================== knob-off child process
def _driver(mode, tree):
    """Child: one forward + backward of build() under `mode`; prints RESULT {json}."""
    assert os.path.realpath(L2.__file__).startswith(os.path.realpath(tree) + os.sep), L2.__file__
    os.environ.pop(ENV, None)
    if mode in ("zero", "zero_blocked"):
        os.environ[ENV] = "0"
    elif mode in ("unarmed", "on_blocked"):
        os.environ[ENV] = "1"
    if mode.endswith("blocked"):
        sys.modules[MOD] = None  # any import of the module raises
    with standins():
        m = build()
        out_sha = _sha(m.out)
        m.loss.backward()
        if mode == "on_blocked":  # a second step: the import is not retried, no second line
            m2 = build()
            m2.loss.backward()
            assert L2._X2LB_REQUESTED is False
    grads = {n: None if t.grad is None else _sha(t.grad) for n, t in sorted(m.leaves.items())}
    print("RESULT " + json.dumps(dict(mode=mode, imported=sys.modules.get(MOD) is not None, log=LOG, grads=grads,
                                      out=out_sha)), flush=True)


def _drive(mode, tree=_TREE):
    env = dict(os.environ, PYTHONPATH=tree, PYTHONDONTWRITEBYTECODE="1")
    env.pop(ENV, None)
    r = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--driver", mode, tree], capture_output=True,
                       text=True, env=env, cwd=tempfile.gettempdir(), timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        raise AssertionError(f"driver {mode} on {tree}: rc {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    res = json.loads(line[-1][len("RESULT "):])
    res["stdout"] = r.stdout
    return res


def _child(script, env_extra=None):
    env = dict(os.environ, PYTHONPATH=_TREE, PYTHONDONTWRITEBYTECODE="1")
    env.pop(ENV, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(script)], capture_output=True, text=True,
                          env=env, cwd=tempfile.gettempdir(), timeout=900)


class X2LBKnobOffTest(unittest.TestCase):
    """DESIGN_stageC 8.1 #1: knob off = today, byte for byte (child processes: this process imports the module)."""

    @classmethod
    def setUpClass(cls):
        cls.off = _drive("off")

    def test_knob_off_never_imports_and_runs_todays_launches(self):
        off = self.off
        self.assertFalse(off["imported"])
        self.assertEqual([e.split("(")[0] for e in off["log"]], _SHIPPED_OPS)
        self.assertNotIn("[preproc-l2-x2lb]", off["stdout"])
        for mode in ("zero", "blocked", "zero_blocked"):  # "0", and the module's import made to fail
            r = _drive(mode)
            self.assertFalse(r["imported"], mode)
            self.assertEqual(r["log"], off["log"], mode)
            self.assertEqual(r["grads"], off["grads"], mode)
            self.assertEqual(r["out"], off["out"], mode)
            self.assertNotIn("[preproc-l2-x2lb]", r["stdout"], mode)

    def test_knob_off_equals_the_unpatched_tree(self):
        base = os.environ.get("X2LB_BASE_TREE")
        if not base:
            self.skipTest("set X2LB_BASE_TREE to an unpatched recommendation/ directory")
        for mode in ("off", "zero"):
            b = _drive(mode, base)
            self.assertFalse(b["imported"])
            self.assertEqual(b["log"], self.off["log"], mode)  # op, shapes, dtypes, strides, scalar arguments
            self.assertEqual(b["grads"], self.off["grads"], mode)
            self.assertEqual(b["out"], self.off["out"], mode)

    def test_knob_on_unarmed_runs_todays_launches_with_one_not_armed_line(self):
        r = _drive("unarmed")
        self.assertTrue(r["imported"])
        self.assertEqual(r["log"], self.off["log"])
        self.assertEqual(r["grads"], self.off["grads"])
        lines = [ln for ln in r["stdout"].splitlines() if "[preproc-l2-x2lb]" in ln]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("[preproc-l2-x2lb] NOT armed on cuda:-1", lines[0])

    def test_knob_on_import_failure_fails_closed(self):
        """Review MAJOR-1 (both lenses; F1): knob "1" and ops/hip/preproc_l2_x2lb not importable -> today's launches
        and gradients, exactly one "FAILED CLOSED (import)" line, the knob off for the rest of the process."""
        r = _drive("on_blocked")
        self.assertFalse(r["imported"])
        self.assertEqual(r["log"][:len(self.off["log"])], self.off["log"])
        self.assertEqual(r["log"][len(self.off["log"]):], self.off["log"])  # the second step: today's too
        self.assertEqual(r["grads"], self.off["grads"])
        lines = [ln for ln in r["stdout"].splitlines() if "[preproc-l2-x2lb]" in ln]
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("[preproc-l2-x2lb] FAILED CLOSED (import): ModuleNotFoundError"), lines)
        self.assertIn("-> shipped dgrad + SwishLN bwd for the rest of the run", lines[0])

    def test_import_is_lazy(self):
        r = _child(r"""
            import os, sys
            os.environ["DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"] = "1"
            from generative_recommenders.ops import preproc_fp8_l2, preproc_grouped_tail  # noqa
            assert "generative_recommenders.ops.hip.preproc_l2_x2lb" not in sys.modules
            from generative_recommenders.ops.hip import preproc_l2_x2lb as X
            assert X._lib is None and not X._armed and not X._ZERO and X._LB is None  # importing builds nothing
            assert "generative_recommenders.ops.flydsl.ln_bwd" not in sys.modules
        """)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


# =================================================================================== in-process base
class _Base(unittest.TestCase):
    def setUp(self):
        self.X = X = _X()
        X.reset_for_testing()
        L2.reset_for_testing()
        self.said = []
        self.handles = []
        self.st = contextlib.ExitStack()
        self.st.enter_context(standins())
        self.st.enter_context(mock.patch.dict(os.environ, {}, clear=False))
        for k in (ENV, X.MLPS_ENV_VAR, X.FIN_ENV_VAR):
            os.environ.pop(k, None)
        self.st.enter_context(mock.patch.object(X, "_say", self.said.append))
        self.st.enter_context(mock.patch.object(torch.cuda, "current_stream", lambda device=None: STREAM0))
        self.st.enter_context(mock.patch.object(torch.cuda, "current_device", lambda: 0))
        # CPU: M_TEST < 65,536, and PR #7 cannot own a CPU call; both are checked by their own tests below
        self.st.enter_context(mock.patch.object(X, "M_MIN", 64))
        self.st.enter_context(mock.patch.object(TLN, "flydsl_ln_bwd_decline_reason", lambda *a, **k: None))
        real_publish = X.publish

        def publish(*a, **k):
            hd = real_publish(*a, **k)
            if hd is not None:
                self.handles.append(hd)
            return hd

        self.st.enter_context(mock.patch.object(X, "publish", publish))
        del LOG[:]

    def tearDown(self):
        self.st.close()
        self.X.reset_for_testing()
        L2.reset_for_testing()
        del LOG[:]

    def knob(self, value, mlps=None):
        X = self.X
        if value is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = value
        if mlps is None:
            os.environ.pop(X.MLPS_ENV_VAR, None)
        else:
            os.environ[X.MLPS_ENV_VAR] = mlps
        L2._X2LB_REQUESTED = None
        X._REQUESTED = X._MLPS = None

    def arm(self, emu=None, mlps=None):
        """knob on, library 'loaded' (the REAL _Lib over EmuLib), the CPU device (index -1) armed by a 'passed'
        phase 2, the REAL _CudaRuntime with a fixed CU count, PR #7's finalize faked."""
        X = self.X
        self.knob("1", mlps)
        self.emu = emu or EmuLib()
        X._lib = make_lib(self.emu)
        rt = X._CudaRuntime()
        rt._ncu[-1] = NCU
        X._RT = rt
        X._armed.add(-1)
        X._ZERO[-1] = torch.zeros((), dtype=BF)
        X._P1_STATE = X._P2_STATE = "ok"
        X._LB = self.lb = fake_lb()
        return self.emu

    def shipped(self, case, **kw):
        self.knob(None)
        del LOG[:]
        m = build(**kw)
        leaves, caps = run_case(case, m)
        return leaves, caps, list(LOG)

    def fused(self, case, emu=None, mlps=None, **kw):
        self.arm(emu, mlps)
        del LOG[:]
        m = build(**kw)
        leaves, caps = run_case(case, m)
        return leaves, caps, list(LOG)

    def lines(self, marker):
        return [s for s in self.said if marker in s]

    def assertSameGrads(self, a, b, what=""):
        self.assertEqual(set(a), set(b), what)
        bad = [n for n in a if not _same(a[n], b[n])]
        self.assertEqual(bad, [], f"{what}: gradients differ from the shipped run")


# =================================================================================== 2. plumbing
class X2LBPlumbingTest(_Base):
    def test_production_shape_fuses_both_mlps_and_equals_shipped(self):
        ls, _, log_s = self.shipped(("plain",))
        lf, _, log_f = self.fused(("plain",))
        self.assertSameGrads(lf, ls, "plain")
        X = self.X
        self.assertEqual(X.FUSED_CALLS, {0: 1, 1: 1})
        self.assertEqual(X.DECLINES, {})
        self.assertEqual(self.lines("DECLINED") + self.lines("FAILED CLOSED") + self.lines("handshake"), [])
        self.assertIsNone(X.broken())
        # S1 ran for k = 1, then k = 0, on the plan grid; PR #7's finalize reduced each PART on the same stream
        self.assertEqual([(c["mode"], c["M"], c["grid"], c["part_rows"], c["stream"]) for c in self.emu.calls],
                         [("s1", M_TEST, NCU, NCU, STREAM0.cuda_stream)] * 2)
        self.assertEqual([(f["G"], f["stream"]) for f in self.lb.fin_calls], [(NCU, ("fx.Stream", STREAM0))] * 2)
        # the fused run skips exactly the two FP8 dgrads and the two pair-0 SwishLN backwards
        dgrad = [e for e in log_s if e.startswith("_scaled_mm(T(384, 512):float8_e5m2")]
        swish = [e for e in log_s if e.startswith("triton_weighted_layer_norm_bwd_dxsum") and "is_swish=True" in e]
        self.assertEqual((len(dgrad), len(swish)), (2, 2))
        self.assertEqual(log_f, [e for e in log_s if e not in dgrad and e not in swish])
        armed = self.lines("armed OK: [preproc-l2-x2lb] fused")
        self.assertEqual(len(armed), 1, self.said)
        self.assertIn("fused L2 dgrad + SwishLN256 bwd on cuda:-1 mlp=1 M=384 grid=5 "
                      "fin=ln_bwd_flydsl_fin_d256_n3_wf32", armed[0])
        self.assertIn("kill switch DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=0", armed[0])
        self.assertEqual(len(self.handles), 2)
        for hd in self.handles:
            self.assertEqual((hd.state, hd.result, hd.sentinel), ("consumed", None, None))

    def test_retain_graph_twice(self):
        ls, _, _ = self.shipped(("retain2",))
        lf, _, _ = self.fused(("retain2",))
        self.assertSameGrads(lf, ls, "retain_graph x2")
        self.assertEqual(self.X.FUSED_CALLS, {0: 2, 1: 2})
        self.assertEqual(len(self.emu.calls), 4)

    def test_second_backward_without_retain_raises_as_shipped(self):
        errs = []
        for mode in ("shipped", "fused"):
            if mode == "fused":
                self.arm()
            else:
                self.knob(None)
            m = build()
            m.loss.backward()
            with self.assertRaises(RuntimeError) as cm:
                m.loss.backward()
            errs.append(str(cm.exception).splitlines()[0])
        self.assertEqual(errs[0], errs[1])
        self.assertIn("backward through the graph a second time", errs[0])

    # (case, keep the proxies, the decline class every call of that case gives)
    _WILL = "pair N does not run in this backward or is only captured (autograd.grad / backward(inputs=))"
    _LIVE = "proxy still referenced (capture, retain_grad or hook)"
    CAPTURE_CASES = (
        (("grad", ("0.w2", "1.w2")), False, _WILL),
        (("grad", ("proxy0", "proxy1")), True, _LIVE),
        (("grad", ("0.nw0", "1.nw0")), False, _WILL),
        (("grad", ("0.x", "1.x")), False, _WILL),
        (("grad", ("proxy0", "proxy1", "0.nw0", "1.nw0")), True, _LIVE),
        (("inputs", ("0.w2", "1.w2")), False, _WILL),
        (("inputs", ("proxy0", "proxy1")), True, _LIVE),
        (("inputs", ("0.x", "1.x")), False, _WILL),
        (("inputs", ("proxy0", "proxy1", "0.nw0", "1.nw0")), True, _LIVE),
        (("hook_kept",), True, _LIVE),
    )

    def test_capture_and_partial_backward_cases_decline(self):
        """The 15 cases of tools/resolution_probe.py rule v2 that must decline (DESIGN_stageC 3.5 / 8.1 #2; the two
        that fuse are the production-shape and retain_graph tests): leaf and captured gradients equal the shipped
        run, nothing stranded, no handshake error, no latch (the leaf query raises inside autograd.grad)."""
        for case, keep, why in self.CAPTURE_CASES:
            with self.subTest(case=case):
                self.X.reset_for_testing()
                self.handles.clear()
                self.said.clear()
                ls, cs, _ = self.shipped(case, keep_proxy=keep)
                lf, cf, _ = self.fused(case, keep_proxy=keep)
                self.assertSameGrads(lf, ls, str(case))
                self.assertEqual(set(cf), set(cs))
                self.assertTrue(all(_same(cf[n], cs[n]) for n in cs), f"{case}: captured gradients differ")
                self.assertEqual(self.X.FUSED_CALLS, {}, case)
                self.assertEqual(set(self.X.DECLINES), {why}, case)
                self.assertTrue(all(hd.result is None and hd.sentinel is None for hd in self.handles), case)
                self.assertEqual(self.lines("handshake") + self.lines("FAILED CLOSED"), [])
                self.assertIsNone(self.X.broken())

    def test_hook_registered_then_proxy_dropped_is_the_documented_residual(self):
        """Unsupported, documented (DESIGN_stageC 3.3): the hook lives on pair 0's grad_fn and sees the zero
        sentinel; every leaf gradient is still the shipped one."""
        ls, cs, _ = self.shipped(("hook_dropped",), keep_proxy=True)
        lf, cf, _ = self.fused(("hook_dropped",), keep_proxy=True)
        self.assertSameGrads(lf, ls, "hook then drop")
        self.assertEqual(self.X.FUSED_CALLS, {0: 1, 1: 1})
        for k in ("hook0", "hook1"):
            self.assertGreater(float(cs[k].float().abs().max()), 0.0)  # shipped: da1
            self.assertEqual(float(cf[k].float().abs().max()), 0.0)  # fused: the sentinel's zero
            self.assertEqual(tuple(cf[k].shape), (M_TEST, 256))

    def test_create_graph_declines(self):
        case = ("create_graph", ("0.nw0", "1.nw0"))
        try:
            ls, cs, _ = self.shipped(case)
            shipped_err = None
        except RuntimeError as exc:
            shipped_err = str(exc).splitlines()[0]
        try:
            lf, cf, _ = self.fused(case)
            fused_err = None
        except RuntimeError as exc:
            fused_err = str(exc).splitlines()[0]
        self.assertEqual(fused_err, shipped_err)
        if shipped_err is None:
            self.assertSameGrads(lf, ls)
            self.assertTrue(all(_same(cf[n], cs[n]) for n in cs))
        self.assertEqual(self.X.FUSED_CALLS, {})
        self.assertEqual(set(self.X.DECLINES), {"create_graph backward"})

    def test_anomaly_mode_declines_with_one_line(self):
        ls, _, _ = self.shipped(("anomaly",))
        lf, _, _ = self.fused(("anomaly",))
        self.assertSameGrads(lf, ls, "anomaly")
        m = build()  # a second anomaly backward: no second line
        run_case(("anomaly",), m)
        self.assertEqual(self.X.FUSED_CALLS, {})
        self.assertEqual(self.X.DECLINES, {"anomaly mode": 4})
        self.assertEqual(len(self.lines("DECLINED")), 1)
        self.assertIn("[preproc-l2-x2lb] DECLINED at call (anomaly mode) -> shipped dgrad + SwishLN bwd for such calls",
                      self.said)
        self.assertIsNone(self.X.broken())

    def test_saved_tensor_hooks_publish_nothing(self):
        ls, _, _ = self.shipped(("plain",))
        self.arm()
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            m = build()
        lf, _ = run_case(("plain",), m)
        self.assertSameGrads(lf, ls, "saved-tensor hooks")
        self.assertEqual(self.handles, [])
        self.assertEqual((self.X.FUSED_CALLS, self.X.DECLINES, self.said), ({}, {}, []))

    def test_eval_and_no_grad_publish_nothing(self):
        self.knob(None)
        with torch.no_grad():
            ref = build(train=False).out
        self.arm()
        calls = []
        with mock.patch.object(L2, "_x2lb", side_effect=lambda: calls.append(1) or self.X):
            with torch.no_grad():
                got = build(train=False).out
            got_grad_on = build(train=False).out  # train False with grad mode on: no handle either
        self.assertTrue(_same(got, ref))
        self.assertTrue(_same(got_grad_on, ref))
        self.assertEqual((calls, self.handles, self.said), ([], [], []))
        self.assertTrue(all(p.x2lb is None for p in build(train=False).plans))

    def test_one_mlp_armed(self):
        ls, _, _ = self.shipped(("plain",))
        for mlps, want in (("1", {1: 1}), ("0", {0: 1})):
            with self.subTest(mlps=mlps):
                self.X.reset_for_testing()
                self.handles.clear()
                lf, _, _ = self.fused(("plain",), mlps=mlps)
                self.assertSameGrads(lf, ls, mlps)
                self.assertEqual(self.X.FUSED_CALLS, want)
                self.assertEqual([hd.k for hd in self.handles], list(want))
                self.assertEqual(self.X.DECLINES, {})

    def test_inplace_on_a_saved_tensor_raises_as_shipped(self):
        errs = []
        for mode in ("shipped", "fused"):
            if mode == "fused":
                self.arm()
            else:
                self.knob(None)
            m = build()
            with torch.no_grad():
                m.leaves["1.nw0"].mul_(1.0)  # pair 0's saved norm0.weight of MLP 1, version bumped
            with self.assertRaises(RuntimeError) as cm:
                m.loss.backward()
            errs.append(str(cm.exception))
        self.assertEqual(errs[0], errs[1])
        self.assertIn("modified by an inplace operation", errs[0])

    def test_launch_error_latches_and_falls_back_for_that_k(self):
        ls, _, log_s = self.shipped(("plain",))
        # k = 1 is the first call: rc 98 -> latched; k = 0 is then not active: both shipped, same backward
        lf, _, log_f = self.fused(("plain",), emu=EmuLib(rc_on=1))
        self.assertSameGrads(lf, ls, "latch at k=1")
        self.assertEqual(log_f, log_s)
        self.assertEqual(self.X.FUSED_CALLS, {})
        fc = self.lines("FAILED CLOSED")
        self.assertEqual(len(fc), 1)
        self.assertIn("x2lb call: RuntimeError: x2lb_launch(s1) returned hip error 98", fc[0])
        self.assertTrue(all(hd.result is None for hd in self.handles))
        # the next forward publishes nothing
        self.handles.clear()
        lf2, _ = run_case(("plain",), build())
        self.assertSameGrads(lf2, ls)
        self.assertEqual(self.handles, [])
        # a failure at k = 0 only: k = 1 fused, k = 0 shipped, in one backward; a raising launch latches the same way
        for emu in (EmuLib(rc_on=2), EmuLib(raise_on=2)):
            self.X.reset_for_testing()
            self.said.clear()
            lf, _, _ = self.fused(("plain",), emu=emu)
            self.assertSameGrads(lf, ls, "latch at k=0")
            self.assertEqual(self.X.FUSED_CALLS, {1: 1})
            self.assertEqual(len(self.lines("FAILED CLOSED")), 1)

    def test_take_handshake(self):
        X = self.X

        class Ctx:
            pass

        ctx = Ctx()
        hd = X.X2LBHandle(0, ctx, STREAM0)
        dy = torch.randn(8, 256).to(BF)
        self.assertIsNone(hd.take(dy))  # no result, a real gradient: shipped
        self.assertIsNone(hd.take(None))
        self.assertIsNone(hd.take(torch.zeros((), dtype=BF).expand(8, 256)))  # stride 0, not a sentinel storage
        z = X._ZERO[-1] = torch.zeros((), dtype=BF)
        with self.assertRaisesRegex(RuntimeError, r"\[preproc-l2-x2lb\] handshake: .*sentinel gradient without"):
            hd.take(z.expand(8, 256))  # a sentinel without results
        res = (torch.ones(8, 256, dtype=BF), torch.ones(256), torch.ones(256), torch.ones(256))

        def pend():
            hd.result, hd.sentinel, hd.task, hd.state = res, z.expand(8, 256), X._graph_task_id(), "fused"

        for bad in (dy, torch.zeros((), dtype=BF).expand(8, 256), z.expand(8, 256).contiguous(), z.expand(4, 256)):
            pend()  # results, but a real gradient / another zero-stride tensor / a copy / another shape arrived
            with self.assertRaisesRegex(RuntimeError, r"handshake: .*not the sentinel"):
                hd.take(bad)
            self.assertEqual((hd.result, hd.state), (None, "declined"))  # dropped: nothing outlives the failure
        pend()
        hd.task = X._graph_task_id() + 1  # another backward's pending result (overlapping backwards)
        self.assertIsNone(hd.take(dy))  # this backward's own da1: the shipped statements
        with self.assertRaisesRegex(RuntimeError, r"handshake: .*in a backward other than the one that fused"):
            hd.take(z.expand(8, 256))
        self.assertIs(hd.result, res)  # still its owner's
        pend()
        got = hd.take(z.expand(8, 256))
        self.assertIs(got, res)
        self.assertEqual((hd.result, hd.sentinel, hd.state, hd.task), (None, None, "consumed", None))
        pend()
        z.expand(8, 256).fill_(0.0)  # an in-place write keeps ptr and strides but moves the version counter
        with self.assertRaisesRegex(RuntimeError, r"handshake: .*sentinel was written in place"):
            hd.take(z.expand(8, 256))
        self.assertIsNone(hd.result)

    def test_node_prehook_that_replaces_the_gradient_raises_handshake(self):
        """A node pre-hook on pair 0 (unsupported, documented) that changes dy: take raises, never a silent zero."""
        self.arm()
        m = build(want_p0=True)
        m.p0s[1].register_prehook(lambda grads: (grads[0].contiguous() + 0,))
        with self.assertRaisesRegex(RuntimeError, r"\[preproc-l2-x2lb\] handshake: mlp 1: fused results but the "
                                                  r"proxy gradient is not the sentinel"):
            m.loss.backward()

    def test_handle_holds_no_tensor_and_forms_no_cycle(self):
        self.arm()
        gc.collect()
        gc.disable()
        try:
            m = build()
            self.assertEqual(len(self.handles), 2)
            for hd in self.handles:
                held = [getattr(hd, s) for s in hd.__slots__ if s != "__weakref__"]
                self.assertFalse(any(isinstance(v, torch.Tensor) for v in held), held)
                self.assertEqual(hd.state, "attached")
                self.assertIsNone(hd.proxy())  # the proxy is dead before the backward (GroupedTail drops it)
            m.loss.backward()
            for hd in self.handles:
                self.assertEqual((hd.result, hd.sentinel), (None, None))
            del hd
            refs = [weakref.ref(h) for h in self.handles]
            self.handles.clear()
            del m
            self.assertEqual([r() for r in refs], [None, None])  # freed by refcount alone (gc is off)
        finally:
            gc.enable()

    def test_sentinel_is_the_per_device_zero_scalar(self):
        self.arm()
        X = self.X
        m = build()
        real = X.try_fused
        got = []
        with mock.patch.object(X, "try_fused", side_effect=lambda *a: got.append(real(*a)) or got[-1]), \
                mock.patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
            m.loss.backward()
        self.assertEqual(len(got), 2)
        z = X._ZERO[-1]
        for s in got:
            self.assertEqual((s.data_ptr(), s.stride(), tuple(s.shape), s.dtype),
                             (z.data_ptr(), (0, 0), (M_TEST, 256), BF))
        self.assertEqual(zeros.call_count, 0)  # no per-call allocation or fill

    def test_stranded_result_is_dropped_and_declined(self):
        self.arm()
        X = self.X
        m = build()
        hd = self.handles[0]
        hd.result = (torch.ones(1),)  # an earlier fused result that was never consumed
        lf, _ = run_case(("plain",), m)
        self.assertIn("an earlier fused result was never consumed", X.DECLINES)
        self.assertEqual(X.FUSED_CALLS, {1: 1})
        self.assertIsNone(hd.result)
        ls, _, _ = self.shipped(("plain",))
        self.assertSameGrads(lf, ls)

    def test_wrong_pair0_operand_wiring_changes_the_gradients(self):
        """Control of the comparison itself: the fused path with norm0's weight and bias swapped at the runtime
        boundary must differ from the shipped run (so 'equal' above is not vacuous)."""
        ls, _, _ = self.shipped(("plain",))
        self.arm()
        rt = self.X._RT
        real = rt.fused
        rt.fused = lambda lib, dh8, w8t, sa, sb, h1, m1, r1, nw, nb: real(lib, dh8, w8t, sa, sb, h1, m1, r1, nb, nw)
        lf, _ = run_case(("plain",), build())
        self.assertEqual(self.X.FUSED_CALLS, {0: 1, 1: 1})
        self.assertTrue(any(not _same(lf[n], ls[n]) for n in ls))

    def test_late_grouped_tail_decline_clears_the_handle(self):
        from generative_recommenders.ops import preproc_grouped_tail as pgt

        slot = L2.Slot(0)
        self.assertIsNone(slot.x2lb)
        slot.mode, slot.payload, slot.x2lb = "fp8", ("payload",), object()
        lin = types.SimpleNamespace(weight=1, bias=2)
        norm = types.SimpleNamespace(weight=3, bias=4, _eps=1e-5)
        mlp = types.SimpleNamespace(_pairs=lambda: [(lin, norm)])
        tail = types.SimpleNamespace(_mlps=[mlp], _ys=["proxy"], _pfs=[None], _fp8=True, _plans=[slot], _x0s=["x0"])
        with mock.patch.object(pgt, "_run_head", lambda mlp, x: "y"), \
                mock.patch.object(pgt._LinearNormFunction, "apply", lambda *a: "acc"):
            self.assertEqual(pgt.GroupedTail._complete_shipped(tail), "acc")
        self.assertEqual((slot.payload, slot.x2lb), (None, None))


    # ------------------------------------------------------------ review round 1 (stagec_impl/review, gpt-6 214601Z)
    def _taken(self):
        """Wrap X2LBHandle.take: the results it hands to pair 0 (identity checks)."""
        X, got = self.X, []
        real = X.X2LBHandle.take

        def take(hd, dy):
            r = real(hd, dy)
            if r is not None:
                got.append(r)
            return r

        self.st.enter_context(mock.patch.object(X.X2LBHandle, "take", take))
        return got

    def test_inplace_hook_on_the_sentinel_raises_and_latches(self):
        """gpt-6 failure mode 1 / review MINOR-1: a hook that writes the sentinel in place (fill_ keeps its pointer and
        strides) must not pass take() silently. It raises a handshake error, drops every result of that backward, and
        the next call latches X2L-B off (the per-device zero scalar was written); later steps equal shipped."""
        X = self.X
        self.arm()
        m = build(keep_proxy=True)
        for k in (0, 1):
            m.proxies[k].register_hook(lambda g: g.fill_(1.0))
        m.proxies.clear()
        with self.assertRaisesRegex(RuntimeError, r"\[preproc-l2-x2lb\] handshake: mlp 1: the sentinel was written "
                                                  r"in place"):
            m.loss.backward()
        self.assertEqual([hd.result for hd in self.handles], [None, None])  # the other MLP's result dropped too
        del m
        self.assertEqual(X._ZERO[-1]._version, 1)
        ls, _, _ = self.shipped(("plain",))
        self.knob("1")  # still armed, with the written zero scalar
        lf, _ = run_case(("plain",), build())
        self.assertSameGrads(lf, ls, "after the in-place hook")
        self.assertEqual(X.FUSED_CALLS, {0: 1, 1: 1})  # the first backward's; none since
        self.assertIn("sentinel was written in place", X.broken())
        self.assertEqual(len(self.lines("FAILED CLOSED")), 1)

    def test_stride0_gradient_without_results_is_not_a_sentinel(self):
        """Review MINOR-2: a backward through a reduction of a held proxy (SumBackward gives a stride-(0, 0) gradient)
        runs the shipped statements; only a gradient on a _ZERO storage counts as a sentinel."""
        res = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            m = build(keep_proxy=True)
            m.proxies[1].sum().backward()
            res.append({n: None if t.grad is None else t.grad.clone() for n, t in m.leaves.items()})
        self.assertSameGrads(res[1], res[0], "proxy.sum().backward()")
        self.assertEqual(self.X.FUSED_CALLS, {})
        self.assertEqual(self.lines("handshake"), [])

    def test_overlapping_backward_from_a_hook(self):
        """gpt-6 failure mode 2 / review MINOR-3: a second retain_graph backward started from a hook between the tail
        and pair 0 (task identity). The inner backward declines without touching the outer's pending results and
        runs the shipped path; the outer still consumes its own. Gradients equal the shipped run."""
        res = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            m = build()
            fired = []

            def hook(g, m=m, fired=fired):
                if not fired:
                    fired.append(1)
                    m.loss.backward(retain_graph=True)

            m.leaves["0.w2"].register_hook(hook)  # W2's AccumulateGrad runs right after the tail
            m.loss.backward(retain_graph=True)
            res.append({n: None if t.grad is None else t.grad.clone() for n, t in m.leaves.items()})
        self.assertSameGrads(res[1], res[0], "overlapping backwards")
        X = self.X
        self.assertEqual(X.FUSED_CALLS, {0: 1, 1: 1})  # the outer backward
        self.assertEqual(X.DECLINES, {"a fused result of another backward is pending (overlapping or aborted "
                                      "backward)": 2})  # the inner one, both MLPs
        self.assertEqual(self.lines("handshake") + self.lines("FAILED CLOSED"), [])
        self.assertTrue(all(hd.result is None and hd.state == "consumed" for hd in self.handles))

    def test_a_failing_tail_backward_releases_its_results(self):
        """gpt-6 failure mode 3 / review MINOR-4: an exception in the tail's backward after a fused k (k = 0's NN-b
        wgrad raises after k = 1 and k = 0 fused) drops both results before it propagates (about 1.29 GB per MLP at
        production M); the shipped run raises the same exception."""
        errs = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            m = build()
            calls = []

            def boom(*a, calls=calls):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected wgrad failure at k=0")
                return f_wgrad_nn_b(*a)

            with mock.patch.object(L2, "_wgrad_nn_b", boom), self.assertRaises(RuntimeError) as cm:
                m.loss.backward()
            errs.append(str(cm.exception))
        self.assertEqual(errs, ["injected wgrad failure at k=0"] * 2)
        self.assertEqual(self.X.FUSED_CALLS, {0: 1, 1: 1})
        self.assertEqual([(hd.state, hd.result, hd.sentinel) for hd in self.handles], [("declined", None, None)] * 2)
        self.assertEqual(len(self.X._PENDING), 0)

    def test_abort_after_the_tail_then_retry(self):
        """The documented residual of an abort OUTSIDE the tail (after it fused, before pair 0 ran): the results stay
        with the kept graph; a retain_graph retry of that graph declines ("another backward ... pending") and runs
        the shipped path, so its gradients equal the shipped run doing the same; dropping the graph frees them."""
        res = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            self.handles.clear()
            m = build()
            fired = []

            def hook(g, fired=fired):
                if not fired:
                    fired.append(1)
                    raise RuntimeError("injected abort after the tail")

            m.leaves["0.w2"].register_hook(hook)
            with self.assertRaisesRegex(RuntimeError, "injected abort after the tail"):
                m.loss.backward(retain_graph=True)
            if mode == "fused":
                self.assertTrue(all(hd.result is not None for hd in self.handles))  # held by the graph
            m.loss.backward(retain_graph=True)  # the retry
            res.append({n: None if t.grad is None else t.grad.clone() for n, t in m.leaves.items()})
        self.assertSameGrads(res[1], res[0], "retry after an abort")
        self.assertEqual(set(self.X.DECLINES), {"a fused result of another backward is pending (overlapping or "
                                                "aborted backward)"})
        refs = [weakref.ref(hd) for hd in self.handles]
        self.handles.clear()
        del m
        gc.collect()
        self.assertEqual([r() for r in refs], [None, None])
        self.assertEqual(len(self.X._PENDING), 0)

    def test_ex2b_port_receives_the_handle_dh1(self):
        """Review MINOR-5 (DESIGN 8.2 G-c / 5.7 coupling, CPU half): pair 0 with an E-X2b port (bind / route mocked):
        route receives the handle's dh1 tensor itself; every leaf and port gradient equals the shipped run's."""
        res = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            taken = self._taken() if mode == "fused" else []
            routed = []
            with mock.patch.object(L2._xs, "bind", lambda rec, x, w: types.SimpleNamespace(kind="port")), \
                    mock.patch.object(L2._xs, "route", lambda ctx, rec, dh, i: routed.append(dh) or True):
                m = build(ports=True)
                m.loss.backward()
            res.append(({n: None if t.grad is None else t.grad.clone() for n, t in m.leaves.items()},
                        {k: p.grad.clone() for k, p in m.ports.items()}, routed, taken))
        (ls, ps, rs, _), (lf, pf, rf, tf) = res
        self.assertSameGrads(lf, ls, "E-X2b port")
        self.assertEqual(set(pf), {0, 1})
        self.assertTrue(all(_same(pf[k], ps[k]) for k in ps), "port gradients differ")
        self.assertEqual(self.X.FUSED_CALLS, {0: 1, 1: 1})
        self.assertEqual(len(rf), 2)
        self.assertEqual(sorted(id(r[0]) for r in tf), sorted(id(d) for d in rf))  # route got the handle's dh1
        self.assertIsNone(lf["0.x"])  # routed: dX goes through the port

    def test_ix_wgrad_receives_the_handle_dh1(self):
        """Review MINOR-5, the IX twin: with Indexed-X on pair 0 (FakeIX), the IX wgrad receives the handle's dh1;
        every leaf gradient equals the shipped run's."""
        res = []
        for mode in ("shipped", "fused"):
            self.arm() if mode == "fused" else self.knob(None)
            taken = self._taken() if mode == "fused" else []
            fix = FakeIX()
            with mock.patch.object(L2, "_ix", lambda fix=fix: fix):
                m = build(ix=True)
                m.loss.backward()
            res.append(({n: None if t.grad is None else t.grad.clone() for n, t in m.leaves.items()}, fix, taken))
        (ls, fs, _), (lf, ff, tf) = res
        self.assertSameGrads(lf, ls, "Indexed-X")
        self.assertEqual((len(fs.dhs), len(ff.dhs)), (2, 2))
        self.assertEqual(self.X.FUSED_CALLS, {0: 1, 1: 1})
        self.assertEqual(sorted(id(r[0]) for r in tf), sorted(id(d) for d in ff.dhs))
        self.assertTrue(all(_same(a, b) for a, b in zip(fs.dhs, ff.dhs)))

    def test_attach_is_single_use(self):
        """Mutation d4 (DESIGN 8.1 #7 "attach without the single-use clear"): the tail's forward moves each handle out
        of its Slot, as it does the FP8 payload. A handle left on a reused Slot would meet the next forward's tail
        as a stale handle (a DECLINED "handle state" line whenever that forward's pair 0 does not publish)."""
        self.arm()
        m = build()
        self.assertEqual([p.x2lb for p in m.plans], [None, None, None])
        self.assertEqual([hd.state for hd in self.handles], ["attached", "attached"])

    def test_will_run_leaf_is_norm0_weight(self):
        """Review NIT-2: will_run's positional next_functions[3] is norm0.weight's AccumulateGrad (lin_b is always a
        tensor input of pair 0), also with an E-X2b port and with Indexed-X."""
        self.knob(None)
        for kw in ({}, {"ports": True}, {"ix": True}):
            with self.subTest(**kw), mock.patch.object(L2, "_ix", FakeIX), \
                    mock.patch.object(L2._xs, "bind", lambda rec, x, w: types.SimpleNamespace(kind="port")):
                m = build(want_p0=True, **kw)
                for k in (0, 1):
                    self.assertIs(m.p0s[k].next_functions[3][0].variable, m.leaves[f"{k}.nw0"])


# =================================================================================== 3. declines
def _good_ops(M=M_TEST):
    """In-contract operands (CPU), as the tail and pair 0 hold them."""
    return dict(dh8=torch.zeros(M, 512, dtype=E5), w8_t=torch.zeros(256, 512, dtype=E4),
                dh_scale=torch.ones(1), w_scale=torch.ones(1), h1=torch.zeros(M, 256, dtype=BF),
                nw=torch.ones(256), nb=torch.zeros(256), mean1=torch.zeros(M), rstd1=torch.ones(M), x_dt=BF)


class X2LBDeclineTest(_Base):
    def reason(self, **over):
        ops = _good_ops()
        ops.update(over)
        return self.X.tensor_reason(**ops)

    def test_tensor_reason_each_operand(self):
        X = self.X
        self.assertIsNone(self.reason())
        M = M_TEST
        mis = torch.zeros(M * 256 + 8, dtype=BF)[1:1 + M * 256].view(M, 256)  # a 2-byte misaligned view
        cases = {
            "dh8 ": dict(dh8=torch.zeros(M, 512, dtype=E5).as_strided((M, 512), (1, M))),
            "w8_t ": dict(w8_t=torch.zeros(256, 512, dtype=torch.float8_e5m2)),
            "dh_scale missing": dict(dh_scale=None),
            "dh_scale torch.float32 (2,)": dict(dh_scale=torch.ones(2)),
            "dh_scale torch.bfloat16 (1,)": dict(dh_scale=torch.ones(1, dtype=BF)),
            "w_scale missing": dict(w_scale=None),
            "h1 torch.bfloat16 (384, 256) stride (1, 384)": dict(h1=torch.zeros(256, M, dtype=BF).t()),
            "h1 torch.bfloat16 (384, 256) stride (256, 1)": dict(h1=mis),
            "mean1 ": dict(mean1=torch.zeros(M, dtype=torch.float64)),
            "rstd1 ": dict(rstd1=torch.ones(2 * M)[::2]),
            "norm weight torch.bfloat16": dict(nw=torch.ones(256, dtype=BF)),
            "norm bias ": dict(nb=torch.zeros(255)),
            "proxy dtype torch.float32": dict(x_dt=torch.float32),
        }
        for want, over in cases.items():
            with self.subTest(want=want):
                why = self.reason(**over)
                self.assertIsNotNone(why)
                self.assertTrue(why.startswith(want), why)
        with mock.patch.object(X, "M_MIN", 1 << 16):
            h = torch.zeros((), dtype=BF)
            self.assertEqual(self.reason(h1=h.expand(65535, 256)), "M 65535 outside [65536, 4200000]")
            self.assertEqual(self.reason(h1=h.expand(4200001, 256)), "M 4200001 outside [65536, 4200000]")
            self.assertEqual(X._cls(self.reason(h1=h.expand(65535, 256))),
                             X._cls(self.reason(h1=h.expand(4200001, 256))))

    def test_pr7_off_or_latched_declines(self):
        # setUp patches PR #7's predicate to "owns it" (CPU); here its real predicate decides
        with mock.patch.object(TLN, "flydsl_ln_bwd_decline_reason", _REAL_PRED):
            with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", False):
                self.assertEqual(self.reason(), "PR #7 would not own the shipped SwishLN backward: "
                                                "DLRM_HSTU_FP8_WLN_BWD_FLYDSL is off")
            with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", True), \
                    mock.patch.object(TLN, "_WLN_BWD_FLYDSL_BROKEN", "x"):
                self.assertEqual(self.reason(), "PR #7 would not own the shipped SwishLN backward: "
                                                "the FlyDSL LayerNorm backward is latched off")

    def test_call_reason_each(self):
        X = self.X
        self.arm()

        class Ctx:
            pass

        p0 = Ctx()
        hd = X.X2LBHandle(0, p0, STREAM0)
        proxy = torch.zeros(4, 256)
        hd.proxy = weakref.ref(proxy)
        hd.state = "attached"
        with mock.patch.object(X, "will_run", lambda p: True):
            with torch.enable_grad():
                self.assertEqual(X.call_reason(hd, p0), "create_graph backward")
            with torch.no_grad():
                self.assertEqual(X.call_reason(hd, p0), "proxy still referenced (capture, retain_grad or hook)")
                del proxy
                self.assertIsNone(X.call_reason(hd, p0))
                with torch.autograd.set_detect_anomaly(True):
                    self.assertEqual(X.call_reason(hd, p0), "anomaly mode")
                self.assertEqual(X.call_reason(hd, None), "pair 0's ctx gone")
                hd.state = "published"
                self.assertEqual(X.call_reason(hd, p0), "handle state published")
                hd.state = "attached"
                hd.proxy = None
                self.assertEqual(X.call_reason(hd, p0), "proxy still referenced (capture, retain_grad or hook)")
                hd.proxy = weakref.ref(Ctx())  # dead at once
                hd.result = (1,)
                self.assertEqual(X.call_reason(hd, p0), "an earlier fused result was never consumed")
                hd.state, hd.task = "fused", X._graph_task_id() + 1
                self.assertEqual(X.call_reason(hd, p0), "a fused result of another backward is pending (overlapping "
                                                        "or aborted backward)")
                hd.result, hd.task, hd.state = None, None, "attached"
                with mock.patch.object(X._RT, "capturing", lambda: True):
                    self.assertEqual(X.call_reason(hd, p0), "CUDA graph capture")
        with torch.no_grad():
            self.assertTrue(X.call_reason(hd, p0).startswith("pair 0 does not run"))  # a fake node: will_run False

    def test_will_run_missing_api_none_or_raising(self):
        X = self.X
        self.assertFalse(X.will_run(None))
        with mock.patch.object(torch._C, "_will_engine_execute_node", None, create=True):
            self.assertFalse(X.will_run(object()))
        node = types.SimpleNamespace(next_functions=((None, 0),) * 3 + ((object(), 0),))
        with mock.patch.object(torch._C, "_will_engine_execute_node", mock.Mock(side_effect=RuntimeError("x"))):
            self.assertFalse(X.will_run(node))
        with mock.patch.object(torch._C, "_will_engine_execute_node", lambda n: True):
            self.assertTrue(X.will_run(node))
            self.assertFalse(X.will_run(types.SimpleNamespace(next_functions=((None, 0),) * 4)))  # None node
            self.assertFalse(X.will_run(types.SimpleNamespace(next_functions=((None, 0),) * 3)))  # too few

    def test_try_fused_declines_once_per_class_and_runs_nothing(self):
        X = self.X
        emu = self.arm()

        class Ctx:
            saved_tensors = None

        def call(**over):
            ops = _good_ops()
            ops.update(over)
            p0 = Ctx()
            p0.saved_tensors = (None, None, None, ops["h1"], ops["nw"], ops["nb"], ops["mean1"], ops["rstd1"])
            hd = X.X2LBHandle(0, p0, STREAM0)
            hd.state, hd.proxy = "attached", weakref.ref(Ctx())
            with torch.no_grad(), mock.patch.object(X, "will_run", lambda p: True):
                got = X.try_fused(hd, ops["dh8"], ops["w8_t"], ops["dh_scale"], ops["w_scale"], ops["x_dt"])
            return got, hd

        h = torch.zeros((), dtype=BF)
        bad = (dict(dh_scale=None), dict(dh_scale=torch.ones(2)), dict(nw=torch.ones(256, dtype=BF)),
               dict(h1=torch.zeros(M_TEST * 256 + 8, dtype=BF)[1:1 + M_TEST * 256].view(M_TEST, 256)))
        for over in bad + bad:
            got, hd = call(**over)
            self.assertIsNone(got)
            self.assertEqual((hd.state, hd.result), ("declined", None))
        with mock.patch.object(X, "M_MIN", 1 << 16):
            for M in (65535, 4200001):
                got, hd = call(h1=h.expand(M, 256))
                self.assertIsNone(got)
        self.assertEqual(emu.calls, [])
        lines = self.lines("DECLINED")
        self.assertEqual(len(lines), 5, lines)  # 4 operand classes + one for both M values
        self.assertTrue(all(ln.startswith("[preproc-l2-x2lb] DECLINED at call (") for ln in lines))
        self.assertEqual(sum(X.DECLINES.values()), 10)
        self.assertEqual(X.DECLINES["M N outside [N, N]"], 2)
        self.assertIsNone(X.broken())
        got, hd = call()  # the same path with in-contract operands: fuses
        self.assertEqual(got.data_ptr(), X._ZERO[-1].data_ptr())
        self.assertEqual(hd.state, "fused")

    def test_not_armed_latched_and_ignored_lines(self):
        X = self.X
        self.knob("1")

        class Ctx:
            pass

        h = torch.zeros(4, 256, dtype=BF)
        for _ in range(2):
            self.assertIsNone(X.publish(Ctx(), 0, h, True))
        self.assertEqual(self.said, ["[preproc-l2-x2lb] NOT armed on cuda:-1: no pre-run_start prebuild passed on "
                                     "this device (BLAS_WARMUP off?) -> shipped dgrad + SwishLN bwd (nothing is built "
                                     "after RUN_START)"])
        self.said.clear()
        X._P1_STATE = "ok"  # phase 1 passed, phase 2 never ran (non-streaming mode): the default reason
        X.publish(Ctx(), 0, h, True)
        X.publish(Ctx(), 1, h, True)
        self.assertEqual(len(self.said), 1)
        self.assertIn("DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED (NOT armed): phase-2 self-check not run (the "
                      "startup preload did not run it (needs mode streaming-train-eval", self.said[0])
        self.said.clear()
        self.assertIsNone(X.publish(Ctx(), 0, h, False))  # not train: silent
        X.latch_off("test", RuntimeError("boom"))
        self.arm()
        X._BROKEN = "test: RuntimeError: boom"
        self.assertIsNone(X.publish(Ctx(), 0, h, True))  # latched: silent, no handle
        self.assertEqual(self.lines("FAILED CLOSED"), ["[preproc-l2-x2lb] FAILED CLOSED -> shipped dgrad + SwishLN "
                                                       "bwd for the rest of the run (test: RuntimeError: boom)"])

    def test_attach_declines(self):
        X = self.X
        self.arm()

        class Ctx:
            pass

        a1 = torch.zeros(4, 512, dtype=E4)
        proxy = torch.zeros(4, 256, dtype=BF)
        for setup, why in ((lambda hd: setattr(X._RT, "stream", lambda d: _Stream(0x99)),
                            "the tail's forward stream differs from pair 0's"),
                           (lambda hd: X._armed.clear(), "device not armed"),
                           (lambda hd: setattr(hd, "state", "attached"), "handle state attached")):
            with self.subTest(why=why):
                self.arm()
                ctx = Ctx()
                hd = X.X2LBHandle(0, ctx, STREAM0)
                setup(hd)
                self.assertIsNone(X.attach(hd, a1, proxy))
                self.assertEqual(hd.state, "declined")
                self.assertIn(f"[preproc-l2-x2lb] DECLINED at attach ({why}) -> shipped dgrad + SwishLN bwd for such "
                              "calls", self.said)
        self.arm()
        hd = X.X2LBHandle(0, Ctx(), STREAM0)  # its ctx is already gone (pair 0 built no node): silent
        self.said.clear()
        self.assertIsNone(X.attach(hd, a1, proxy))
        self.assertIn("pair N's ctx gone", X.DECLINES)
        self.assertEqual((hd.state, self.said), ("declined", []))
        ctx = Ctx()
        hd = X.X2LBHandle(0, ctx, STREAM0)
        self.assertIs(X.attach(hd, a1, proxy), hd)
        self.assertEqual((hd.state, hd.proxy()), ("attached", proxy))
        self.assertIsNone(X.attach(None, a1, proxy))

    def test_capture_declines_in_model(self):
        ls, _, _ = self.shipped(("plain",))
        self.arm()
        with mock.patch.object(self.X._RT, "capturing", lambda: True):
            lf, _ = run_case(("plain",), build())
        self.assertSameGrads(lf, ls)
        self.assertEqual(self.X.DECLINES, {"CUDA graph capture": 2})


_REAL_PRED = TLN.flydsl_ln_bwd_decline_reason  # captured before any test patches it


# =================================================================================== 4. runtime, loader, static
def _c_params(name):
    src = Path(_HIP, "x2lb_pass.cu").read_text()
    m = re.search(r'extern "C" ([\w ]+?\*?)\s*' + name + r"\(([^)]*)\)", src)
    assert m, name
    params = [p.strip() for p in m.group(2).replace("\n", " ").split(",") if p.strip() and p.strip() != "void"]
    return m.group(1).strip(), [re.match(r"(.*?)(\w+)$", p).groups() for p in params]


class X2LBRuntimeTest(_Base):
    def test_ctypes_arguments_by_c_name(self):
        """_CudaRuntime.launch / fin hand every pointer to the C parameter of the same name (x2lb_pass.cu)."""
        X = self.X
        emu = self.arm()
        M, G = 130, 3
        t = dict(dh8=torch.zeros(M, 512, dtype=E5), w8t=torch.zeros(256, 512, dtype=E4), sa=torch.ones(1),
                 sb=torch.ones(1), h1=torch.zeros(M, 256, dtype=BF), mean1=torch.zeros(M), rstd1=torch.ones(M),
                 nw=torch.ones(256), nb=torch.zeros(256), dh1=torch.empty(M, 256, dtype=BF),
                 da1=torch.empty(M, 256, dtype=BF), part=torch.empty(3, G, 256))
        for mode, cname in (("s1", "x2lb_launch"), ("emit", "x2lb_launch_emit")):
            emu.calls.clear()
            X._RT.launch(X._lib, mode, t["dh8"], t["w8t"], t["sa"], t["sb"], t["h1"], t["mean1"], t["rstd1"], t["nw"],
                         t["nb"], t["dh1"], t["da1"] if mode == "emit" else None, t["part"], M, G, STREAM0)
            (c,) = emu.calls
            _, params = _c_params(cname)
            names = [n for _, n in params]
            self.assertEqual(names[:12], list(c["ptrs"]))  # the emulator's own order == the C names
            for n in names[:12]:
                want = None if (n == "da1" and mode == "s1") else t[n].data_ptr()
                self.assertEqual(c["ptrs"][n], want, (mode, n))
            self.assertEqual((c["part_rows"], c["M"], c["grid"], c["stream"]), (G, M, G, STREAM0.cuda_stream))
            self.assertEqual(names[12:], ["part_rows", "M", "grid", "stream"])
        out = torch.empty(3, 256)
        X._RT.fin(X._lib, "x2lb", t["part"], G, STREAM0, out=out)
        (f,) = emu.fins
        self.assertEqual([n for _, n in _c_params("x2lb_fin_launch")[1]], ["part", "part_rows", "out", "stream"])
        self.assertEqual((f["part"], f["part_rows"], f["out"], f["stream"]),
                         (t["part"].data_ptr(), G, out.data_ptr(), STREAM0.cuda_stream))

    def test_rc_and_grid_checks(self):
        X = self.X
        self.arm()
        t = _good_ops(130)
        part = torch.empty(3, 3, 256)
        dh1 = torch.empty(130, 256, dtype=BF)
        args = (t["dh8"], t["w8_t"], t["dh_scale"], t["w_scale"], t["h1"], t["mean1"], t["rstd1"], t["nw"], t["nb"],
                dh1, None, part, 130)
        with self.assertRaisesRegex(ValueError, "PART rows 3 != grid 4"):
            X._RT.launch(X._lib, "s1", *args, 4, STREAM0)
        self.emu.rc_on = 2
        X._RT.launch(X._lib, "s1", *args, 3, STREAM0)
        with self.assertRaisesRegex(RuntimeError, r"x2lb_launch\(s1\) returned hip error 98"):
            X._RT.launch(X._lib, "s1", *args, 3, STREAM0)
        with mock.patch.object(X._lib, "x2lb_launch", lambda *a: -700):  # a stale error, drained: informational
            X._RT.launch(X._lib, "s1", *args, 3, STREAM0)
            X._RT.launch(X._lib, "s1", *args, 3, STREAM0)
        self.assertEqual(self.lines("stale HIP error"), ["[preproc-l2-x2lb] stale HIP error 700 drained before "
                                                        "x2lb_launch(s1) (left on this thread by an earlier call, not "
                                                        "ours)"])
        with self.assertRaisesRegex(ValueError, "rejected"):
            X._lib.plan(0, NCU)

    def test_fin_pr7_call_form(self):
        X = self.X
        self.arm()
        part = torch.randn(3, 4, 256)
        dw, db, dxs = X._RT.fin(X._lib, "pr7", part, 4, STREAM0)
        (c,) = self.lb.fin_calls
        self.assertEqual((c["G"], c["stream"]), (4, ("fx.Stream", STREAM0)))
        self.assertIs(c["part"], part)
        for o, got in enumerate((dw, db, dxs)):
            self.assertEqual((got.dtype, tuple(got.shape)), (F32, (256,)))
        X._LB = None
        with self.assertRaisesRegex(RuntimeError, "ln_bwd not loaded"):
            X._RT.fin(X._lib, "pr7", part, 4, STREAM0)

    def test_fused_wiring_on_the_real_runtime(self):
        X = self.X
        self.arm()
        d = _good_ops(200)
        d["h1"].normal_()
        d["rstd1"].fill_(1.0)
        with mock.patch.object(X, "fin_mode", lambda: "x2lb"):
            res, grid, st = X._RT.fused(X._lib, d["dh8"], d["w8_t"], d["dh_scale"], d["w_scale"], d["h1"], d["mean1"],
                                        d["rstd1"], d["nw"], d["nb"])
        self.assertEqual((grid, st), (4, STREAM0))
        self.assertEqual([tuple(t.shape) for t in res], [(200, 256), (256,), (256,), (256,)])
        self.assertEqual(len(self.emu.fins), 1)
        # review NIT-3: the diagnostic finalize's three rows come back as separate tensors (no aliasing .grad)
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in res[1:]}), 3)
        with self.assertRaisesRegex(RuntimeError, "library not loaded"):
            X._RT.fused(None, *[d[k] for k in ("dh8", "w8_t", "dh_scale", "w_scale", "h1", "mean1", "rstd1", "nw",
                                               "nb")])

    def test_lib_abi_checks(self):
        X = self.X
        self.assertEqual(make_lib(EmuLib()).impl.split("src_sha256=")[1], X.SOURCE_SHA256)
        for emu, msg in ((EmuLib(abi=1), "x2lb_abi_version\\(\\)=1, expected 2026101202"),
                         (EmuLib(variant=1), "x2lb_variant\\(\\)=1 \\(a lab negative-control build\\)"),
                         (EmuLib(impl_sha="0" * 64), "does not carry src_sha256")):
            with self.assertRaisesRegex(RuntimeError, msg):
                make_lib(emu)

    def test_load_rebuilds_a_bad_cache_entry_once(self):
        X = self.X
        good = object()
        for bad in ("raises", "abi"):
            X.reset_for_testing()
            calls = []

            def fake_build(fresh=False):
                calls.append(fresh)
                return (Path("/x/fresh.so" if fresh else "/x/cached.so"), {}, 0.0, not fresh)

            def fake_lib(so, meta, s, cached):
                if cached:
                    raise (OSError("truncated .so") if bad == "raises" else RuntimeError("x2lb_abi_version()=1"))
                return good

            with mock.patch.object(X, "build", side_effect=fake_build), mock.patch.object(X, "_Lib", fake_lib):
                self.assertIs(X.load(), good)
            self.assertEqual(calls, [False, True], bad)
        X.reset_for_testing()
        with mock.patch.object(X, "build", return_value=(Path("/x/new.so"), {}, 1.0, False)), \
                mock.patch.object(X, "_Lib", side_effect=RuntimeError("bad build")):
            with self.assertRaisesRegex(RuntimeError, "bad build"):
                X.load()  # a fresh build that fails is not retried


_FAKE_HIPCC = """#!{python}
import os, subprocess, sys, time
a = sys.argv[1:]
if "--version" in a:
    print(os.environ.get("FAKE_HIPCC_VERSION", "fake hipcc 1.0")); sys.exit(0)
if os.environ.get("FAKE_HIPCC_SLEEP"):
    subprocess.Popen(["sleep", "30"]); time.sleep(30)
if os.environ.get("FAKE_HIPCC_COUNT"):
    open(os.environ["FAKE_HIPCC_COUNT"], "a").write(str(os.getpid()) + "\\n")
    time.sleep(float(os.environ.get("FAKE_HIPCC_DELAY", "0")))
open(a[a.index("-o") + 1], "wb").write(b"not a real library")
if not os.environ.get("FAKE_HIPCC_NO_ASM"):
    open("x2lb_pass-hip-amdgcn-amd-amdhsa-gfx950.s", "w").write(
        "\\t.globl __hip_cuid_" + str(os.getpid()) + "\\n__hip_cuid_" + str(os.getpid()) + ":\\n\\t.byte 0\\nfake asm\\n")
sys.stderr.write({remarks!r} if not os.environ.get("FAKE_HIPCC_SPILL") else {spill!r})
"""


class X2LBBuildTest(unittest.TestCase):
    def setUp(self):
        self.X = X = _X()
        X.reset_for_testing()
        X._toolchains.clear()
        self.tmp = tempfile.mkdtemp(prefix="x2lb_buildtest_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.hipcc = os.path.join(self.tmp, "hipcc")
        spill = _remarks({X.KERNEL_EMIT: {"VGPRs Spill": 3}})
        with open(self.hipcc, "w") as fh:
            fh.write(_FAKE_HIPCC.format(python=sys.executable, remarks=_remarks(), spill=spill))
        os.chmod(self.hipcc, 0o755)
        self.root = os.path.join(self.tmp, "cache")
        os.mkdir(self.root)
        os.chmod(self.root, 0o755)
        self.env = mock.patch.dict(os.environ, {X.HIPCC_ENV_VAR: self.hipcc, X.CACHE_ENV_VAR: self.root,
                                                "PYTHONDONTWRITEBYTECODE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.isa_calls = []

        def fake_isa(asm):
            self.isa_calls.append(asm)
            return dict(ok=True, mode="full", asm_norm_sha256=X.asm_norm_sha256(Path(asm).read_bytes()), kernels={},
                        rc=0, checker=dict(X.ISA_FILES_SHA256), seconds=0.0, tail=None)

        self.isa = mock.patch.object(X, "isa_gate", side_effect=fake_isa)
        self.isa.start()
        self.addCleanup(self.isa.stop)
        self.addCleanup(X.reset_for_testing)

    def test_cache_key_components(self):
        X = self.X
        k0 = X.cache_dir()
        self.assertEqual(X.cache_dir(), k0)
        self.assertEqual(k0.parent, Path(self.root))
        src = Path(self.tmp, "x2lb_pass.cu")
        src.write_bytes(X._SOURCE.read_bytes() + b"\n")
        self.assertNotEqual(X.cache_dir(src), k0)
        with mock.patch.object(X, "_FLAGS", X._FLAGS + ("-DX",)):
            self.assertNotEqual(X.cache_dir(), k0)
        with mock.patch.object(X, "_ARCH", "gfx942"):
            self.assertNotEqual(X.cache_dir(), k0)
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_VERSION": "fake hipcc 2.0"}):
            X._toolchains.clear()
            self.assertNotEqual(X.cache_dir(), k0)
        X._toolchains.clear()
        for var in X._HIPCC_ENV:
            with mock.patch.dict(os.environ, {var: "-O1"}):
                self.assertNotEqual(X.cache_dir(), k0, var)
        self.assertIn('-DX2LB_SRC_SHA256="%s"' % X.SOURCE_SHA256, X.build_flags(X._SOURCE.read_bytes()))

    def test_built_cached_untrusted_fresh(self):
        X = self.X
        so, meta, _, cached = X.build()
        self.assertFalse(cached)
        self.assertTrue(str(so).startswith(self.root))
        self.assertEqual(sorted(p.name for p in so.parent.iterdir() if not p.name.startswith(".")),
                         ["libx2lb_pass.so", "meta.json", "x2lb_pass.gfx950.s"])  # no temporaries, no build dir
        self.assertEqual(meta["resources"][X.KERNEL_S1]["lds"], 99328)
        self.assertEqual(meta["src_sha256"], X.SOURCE_SHA256)
        self.assertEqual(len(self.isa_calls), 1)
        self.assertTrue(X.build()[3])  # a finished entry with a current verdict: no compile, no ISA gate
        self.assertEqual(len(self.isa_calls), 1)
        os.chmod(self.root, 0o777)  # OUR root, group / world writable (umask 000): tightened, the shared entry used
        self.assertTrue(X.build()[3])
        self.assertEqual(stat.S_IMODE(os.stat(self.root).st_mode), 0o755)
        uid = os.getuid()
        with mock.patch.object(X.os, "getuid", lambda: uid + 1):  # a root of another uid: untouched, build privately
            so2, _, _, cached2 = X.build()
        self.assertFalse(cached2)
        self.assertFalse(str(so2).startswith(self.root))
        self.assertTrue(so2.parent.name.startswith("x2lb_private_"))
        shutil.rmtree(so2.parent, True)
        so3 = X.build(fresh=True)[0]  # a rebuild: private path, the shared entry dropped
        self.assertFalse(str(so3).startswith(self.root))
        self.assertFalse((so.parent / X._LIB_NAME).exists())
        shutil.rmtree(so3.parent, True)

    def test_umask_000_root_is_created_0700_and_shared(self):
        """Review (robust) R3: launch_slurm.sh runs the workers under umask 000 and Path.mkdir(parents=True) ignores
        mode for parents. The root (not created in advance, two levels deep) must come out 0700, so the shared entry
        is trusted: built once, then a cache hit (no per-rank private build, no leaked directory)."""
        X = self.X
        root = os.path.join(self.tmp, "fresh", "cache")
        old = os.umask(0)
        try:
            with mock.patch.dict(os.environ, {X.CACHE_ENV_VAR: root}):
                so, _, _, cached = X.build()
                self.assertFalse(cached)
                self.assertTrue(str(so).startswith(root + os.sep), so)
                self.assertEqual(stat.S_IMODE(os.stat(root).st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(os.stat(so.parent).st_mode), 0o700)
                self.assertTrue(X.build()[3])
        finally:
            os.umask(old)

    def test_load_removes_a_private_build_dir(self):
        """Review (robust) R3: a private build directory is removed once its library is loaded (and after a refused
        load); a shared cache entry stays."""
        X = self.X
        seen = []

        def fake_lib(so, meta, s, cached):
            seen.append((so, so.is_file()))
            return types.SimpleNamespace(path=so, meta=meta)

        uid = os.getuid()
        with mock.patch.object(X, "_Lib", fake_lib), mock.patch.object(X.os, "getuid", lambda: uid + 1):
            X.load()
        (so, existed), = seen
        self.assertTrue(existed and so.parent.name.startswith("x2lb_private_"))
        self.assertFalse(so.parent.exists())
        X.reset_for_testing()
        seen.clear()
        with mock.patch.object(X, "_Lib", fake_lib):
            X.load()
        self.assertTrue(seen[0][0].is_file())  # the shared entry is kept

    def test_unreadable_meta_is_a_miss(self):
        """Review (robust) NIT: a cache hit whose meta cannot be read (a half-deleted entry) is a miss: rebuilt under
        the entry lock, never a FileNotFoundError / JSON error out of build()."""
        X = self.X
        so = X.build()[0]
        (so.parent / X._META_NAME).write_text("{not json")
        so2, meta, _, cached = X.build()
        self.assertFalse(cached)
        self.assertEqual(so2, so)
        self.assertEqual(meta["src_sha256"], X.SOURCE_SHA256)
        self.assertEqual(len(self.isa_calls), 2)

    def test_racing_builders_compile_once(self):
        """Review (robust) test gap 4: two processes building the same entry against one trusted root (fake hipcc
        that takes 3 s): exactly one compile; the other waits on the entry lock and hits the cache."""
        count = os.path.join(self.tmp, "compiles")
        script = textwrap.dedent(r"""
            import json, sys
            from pathlib import Path
            from unittest import mock
            from generative_recommenders.ops.hip import preproc_l2_x2lb as X
            def fake_isa(asm):
                return dict(ok=True, mode="full", asm_norm_sha256=X.asm_norm_sha256(Path(asm).read_bytes()),
                            kernels={}, rc=0, checker=dict(X.ISA_FILES_SHA256), seconds=0.0, tail=None)
            with mock.patch.object(X, "isa_gate", fake_isa):
                so, meta, s, cached = X.build()
            print("RESULT " + json.dumps(dict(so=str(so), cached=cached)), flush=True)
        """)
        env = dict(os.environ, PYTHONPATH=_TREE, PYTHONDONTWRITEBYTECODE="1", FAKE_HIPCC_COUNT=count,
                   FAKE_HIPCC_DELAY="3")
        procs = [subprocess.Popen([sys.executable, "-B", "-c", script], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env, cwd=self.tmp) for _ in range(2)]
        outs = [p.communicate(timeout=600) for p in procs]
        res = []
        for p, (o, e) in zip(procs, outs):
            self.assertEqual(p.returncode, 0, o + e)
            res.append(json.loads([ln for ln in o.splitlines() if ln.startswith("RESULT ")][-1][7:]))
        self.assertEqual(len(open(count).read().split()), 1)
        self.assertEqual(sorted(r["cached"] for r in res), [False, True])
        self.assertEqual(res[0]["so"], res[1]["so"])

    def test_stale_verdict_reruns_the_gate_and_a_refusal_fails_closed(self):
        X = self.X
        so = X.build()[0]
        meta_p = so.parent / X._META_NAME
        meta = json.loads(meta_p.read_text())
        meta["isa"]["asm_norm_sha256"] = "0" * 64
        meta_p.write_text(json.dumps(meta))
        self.assertTrue(X.build()[3])  # re-gated, still a hit
        self.assertEqual(len(self.isa_calls), 2)
        meta["isa"]["checker"] = {"isa_check.py": "0" * 64}
        meta_p.write_text(json.dumps(meta))
        with mock.patch.object(X, "isa_gate", return_value=dict(ok=False, mode="full", kernels={
                X.KERNEL_S1: dict(ok=False, failed=["C3"])}, rc=0, tail="C3 ...")):
            with self.assertRaisesRegex(RuntimeError, "ISA gate refuses the cached"):
                X.build()
            shutil.rmtree(so.parent, True)
            with self.assertRaisesRegex(RuntimeError, r"ISA gate refuses this build: .*'C3'"):
                X.build()

    def test_build_failures(self):
        X = self.X
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_SPILL": "1"}):
            with self.assertRaisesRegex(RuntimeError, "resource census: _ZN4x2lb9x2lb_passILi0ELi1EE"):
                X.build()
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_NO_ASM": "1"}):
            with self.assertRaisesRegex(RuntimeError, "no gfx950 device assembly"):
                X.build()
        import time
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_SLEEP": "1"}), mock.patch.object(X, "_BUILD_TIMEOUT_S", 1):
            t0 = time.time()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                X.build()
            self.assertLess(time.time() - t0, 20)
        d = X.cache_dir()
        self.assertEqual([p.name for p in d.iterdir() if not p.name.startswith(".")], [])  # nothing half-written


class X2LBIsaGateTest(unittest.TestCase):
    def setUp(self):
        self.X = _X()
        self.tmp = tempfile.mkdtemp(prefix="x2lb_isatest_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def asm(self, name, cuid):
        p = Path(self.tmp, name)
        p.write_text(f"\t.globl __hip_cuid_{cuid}\n__hip_cuid_{cuid}:\n\t.byte 0\n\t.text\nnot a kernel\n")
        return p

    def test_pinned_mode_keys_on_the_cuid_normalized_assembly(self):
        X = self.X
        a, b = self.asm("a.s", "1111"), self.asm("b.s", "2222")
        self.assertNotEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(X.asm_norm_sha256(a.read_bytes()), X.asm_norm_sha256(b.read_bytes()))
        verdict = {k: dict(ok=True, failed=[], errors=[]) for k in (X.KERNEL_S1, X.KERNEL_EMIT, X.KERNEL_S0,
                                                                    X.KERNEL_FIN)}
        cmds = []

        def run(cmd, timeout, cwd=None, env=None):
            cmds.append((cmd, env))
            return 0, "chatter\nX2LB_ISA_VERDICT " + json.dumps(verdict) + "\n"

        with mock.patch.object(X, "_run_group", run), \
                mock.patch.object(X, "PINNED_ASM_NORM_SHA256", X.asm_norm_sha256(a.read_bytes())):
            v = X.isa_gate(b)
            self.assertEqual((v["ok"], v["mode"]), (True, "pinned"))
            self.assertEqual(cmds[-1][0][-1], "1")
            b.write_text(b.read_text() + "\ts_nop 0\n")
            v = X.isa_gate(b)
            self.assertEqual((v["ok"], v["mode"]), (True, "full"))
            self.assertEqual(cmds[-1][0][-1], "0")
        cmd, env = cmds[-1]
        self.assertEqual(cmd[:3], [sys.executable, "-B", "-c"])
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(cmd[4], str(X._ISA_DIR))
        for out in ((1, "X2LB_ISA_VERDICT " + json.dumps(verdict)),  # rc != 0
                    (0, "no verdict line"),
                    (0, "X2LB_ISA_VERDICT " + json.dumps({k: v for k, v in verdict.items() if k != X.KERNEL_FIN})),
                    (0, "X2LB_ISA_VERDICT " + json.dumps(dict(verdict, **{X.KERNEL_S0: dict(ok=False)})))):
            with mock.patch.object(X, "_run_group", lambda *a, out=out, **k: out):
                self.assertFalse(X.isa_gate(a)["ok"], out)
        with mock.patch.dict(X.ISA_FILES_SHA256, {"isa_exec.py": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, r"ISA gate files differ from the sealed kit: \['isa_exec.py'\]"):
                X.isa_gate(a)

    def test_real_checker_refuses_a_foreign_assembly_in_both_modes(self):
        X = self.X
        a = self.asm("a.s", "33")
        before = set(Path(X._ISA_DIR).rglob("*"))
        v = X.isa_gate(a)
        self.assertEqual((v["ok"], v["mode"], v["rc"]), (False, "full", 0), v.get("tail"))
        self.assertEqual({k: x["failed"] for k, x in v["kernels"].items()},
                         {k: ["missing"] for k in (X.KERNEL_S1, X.KERNEL_EMIT, X.KERNEL_S0, X.KERNEL_FIN)})
        with mock.patch.object(X, "PINNED_ASM_NORM_SHA256", X.asm_norm_sha256(a.read_bytes())):
            v = X.isa_gate(a)
        self.assertEqual((v["ok"], v["mode"]), (False, "pinned"))
        self.assertEqual(set(Path(X._ISA_DIR).rglob("*")), before)  # no __pycache__ left in x2lb_isa

    def test_lab_assembly(self):
        """With X2LB_LAB_CACHE = the lab's x2lb_cache of job 70428: the production assembly passes in pinned mode,
        and the g3_kperm negative control (variant 1) is refused by the full C1-C18 gate."""
        root = os.environ.get("X2LB_LAB_CACHE")
        if not root:
            self.skipTest("set X2LB_LAB_CACHE to the lab's x2lb_cache directory")
        X = self.X
        prod = Path(root, "1c53548919fa9d49", "x2lb_pass.gfx950.s")
        self.assertEqual(X.asm_norm_sha256(prod.read_bytes()), X.PINNED_ASM_NORM_SHA256)
        v = X.isa_gate(prod)
        self.assertEqual((v["ok"], v["mode"]), (True, "pinned"), v.get("tail"))
        kperm = Path(root, "4a775a3c2367a035", "x2lb_pass.gfx950.s")
        if kperm.is_file():
            v = X.isa_gate(kperm)
            self.assertEqual((v["ok"], v["mode"]), (False, "full"))
            self.assertIn("C3", {f for k in v["kernels"].values() for f in k["failed"]})


class X2LBStaticTest(unittest.TestCase):
    def setUp(self):
        self.X = _X()

    def test_source_pin(self):
        X = self.X
        got = hashlib.sha256(X._SOURCE.read_bytes()).hexdigest()
        self.assertEqual(got, X.SOURCE_SHA256, "x2lb_pass.cu changed: rerun the lab / GPU gates before updating the pin")
        self.assertEqual(X.SOURCE_SHA256, _KIT_SRC_SHA)

    def test_isa_files_pinned_to_the_sealed_kit(self):
        X = self.X
        self.assertEqual(X.isa_files_sha256(), X.ISA_FILES_SHA256)
        self.assertEqual(X.ISA_FILES_SHA256, _KIT_ISA_SHA)

    def test_c_abi(self):
        X = self.X
        text = X._SOURCE.read_text()
        self.assertIn(f"#define X2LB_ABI_VERSION {X.ABI_VERSION}", text)
        cty = {"int": X._I, "long long": X._LL, "void*": X._VP, "const void*": X._VP, "const char*": X._CP}
        for name, (res, args) in X.SIGNATURES.items():
            ret, params = _c_params(name)
            self.assertEqual([cty.get(t.replace(" *", "*").strip()) for t, _ in params], list(args), name)
            self.assertIs(cty[ret.replace(" *", "*")], res, name)
        self.assertEqual(set(X.SIGNATURES), set(re.findall(r'extern "C" (?!__device__)[\w ]+?\*?\s*(x2lb_\w+)\(',
                                                            text)))

    def test_flags_are_the_kits(self):
        X = self.X
        self.assertEqual(X._FLAGS, _KIT_FLAGS)
        src = X._SOURCE.read_bytes()
        self.assertEqual(X.build_flags(src), ["--offload-arch=gfx950", *_KIT_FLAGS,
                                              f'-DX2LB_SRC_SHA256="{_KIT_SRC_SHA}"'])  # == the lab job's command

    def test_census(self):
        X = self.X
        ks = X.parse_resource_remarks("noise\n" + _remarks())
        self.assertEqual(len(ks), 4)
        res = X.check_resources(ks)
        self.assertEqual({k: (v.get("unified_vgpr"), v["lds"], v["occupancy"]) for k, v in res.items()},
                         {X.KERNEL_FIN: (None, 0, 8), X.KERNEL_S1: (220, 99328, 2), X.KERNEL_EMIT: (224, 99328, 2),
                          X.KERNEL_S0: (140, 99328, 2)})
        self.assertEqual(X.parse_resource_remarks("no remarks here"), [])
        bad = {"S1 LDS": {X.KERNEL_S1: {"LDS Size [bytes/block]": 99329}},
               "EMIT VGPR spill": {X.KERNEL_EMIT: {"VGPRs Spill": 1}},
               "S0 SGPR spill": {X.KERNEL_S0: {"SGPRs Spill": 2}},
               "S0 occupancy": {X.KERNEL_S0: {"Occupancy [waves/SIMD]": 1}},
               "S1 unified VGPR 252": {X.KERNEL_S1: {"VGPRs": 249}},
               "EMIT AGPRs": {X.KERNEL_EMIT: {"AGPRs": 32}},
               "S1 scratch": {X.KERNEL_S1: {"ScratchSize [bytes/lane]": 16}},
               "S0 LDS too big": {X.KERNEL_S0: {"LDS Size [bytes/block]": 163841}},
               "fin scratch": {X.KERNEL_FIN: {"ScratchSize [bytes/lane]": 4}},
               "missing field": {X.KERNEL_EMIT: {"VGPRs Spill": None}}}
        for what, over in bad.items():
            with self.subTest(what=what):
                with self.assertRaisesRegex(RuntimeError, "resource census"):
                    X.check_resources(X.parse_resource_remarks(_remarks(over)))
        missing = [k for k in ks if k["Function Name"] != X.KERNEL_FIN]
        with self.assertRaisesRegex(RuntimeError, "no remarks for _ZN4x2lb8x2lb_fin"):
            X.check_resources(missing)


# =================================================================================== 5. prebuild
class X2LBPrebuildTest(_Base):
    def setUp(self):
        super().setUp()
        self.cpu = torch.device("cpu")
        self.st.enter_context(mock.patch.object(self.X, "SPLIT_TAIL", 64))

    def test_cancel_data_structure_determinism_no_rng(self):
        X = self.X
        rng = torch.get_rng_state()
        M = 300
        with mock.patch.object(X, "_CH", 128):  # several generator chunks
            d = X.cancel_data(M, self.cpu, X._Live())
            again = X.cancel_data(M, self.cpu, X._Live())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k in d:
            a, b = d[k], again[k]
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)), k)
        u = d["dh8"].view(torch.uint8)
        self.assertTrue(torch.equal(u[:, 128:256], u[:, 0:128] ^ 0x80))  # K step 1 = -(K step 0)
        big, small = u[:, 0:128] & 0x7F, u[:, 256:512] & 0x7F
        self.assertEqual((int(big.min()), int(big.max())), (0x50, 0x7A))
        self.assertEqual((int(small.min()), int(small.max())), (0x01, 0x2F))
        w = d["w8t"].view(torch.uint8)
        self.assertTrue(torch.equal(w[:, 128:256], w[:, 0:128]))
        self.assertEqual((float(d["sa"]), float(d["sb"])), (2.0 ** -20, 2.0 ** -10))
        for k in ("h1", "mean1", "rstd1", "nw", "nb"):
            self.assertTrue(bool(torch.isfinite(d[k].float()).all()), k)
        self.assertTrue(bool(torch.isfinite(d["dh8"].float()).all()))
        hd = d["h1"].double()
        self.assertTrue(torch.equal(d["mean1"], hd.mean(1).float()))
        da = f_scaled_mm(d["dh8"], d["w8t"].t(), d["sa"], d["sb"], out_dtype=BF)
        self.assertTrue(bool(torch.isfinite(da.float()).all()))

    def check(self, rule, M=300, emu=None, lb=None):
        X = self.X
        self.arm(emu)
        X._LB = self.lb = lb or fake_lb()
        live = X._Live()
        rng = torch.get_rng_state()
        del LOG[:]
        out = X.check_point(X._lib, M, rule, self.cpu, live)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return out, live

    def test_check_points_pass_on_the_emulator(self):
        X = self.X
        for rule, M, gated in (("all", 300, 300), ("dp_region", 330, 330 - 64), ("report", 130, 0)):
            with self.subTest(rule=rule):
                out, live = self.check(rule, M)
                G = min(-(-M // 64), NCU)
                self.assertEqual((out["M"], out["grid"], out["rule"], out["da1_gated_rows"]), (M, G, rule, gated))
                self.assertEqual(out["torch_mm"], "triton_fp8._scaled_mm")
                # the order and frees of DESIGN_stageC 5.3 step 4: the peak is exactly 5U + stats + canaries + PART
                U = M * 512
                self.assertEqual(live.peak, 5 * U + 8 * M + 2 * X.CANARY_ROWS * 512 + 3 * G * 256 * 4)
                self.assertEqual(live.live, {})
                # (a) through the production callable, (b) through PR #7's ln_bwd directly, never TLN's dispatch
                mm = [e for e in LOG if e.startswith("_scaled_mm(")]
                self.assertEqual(mm, [f"_scaled_mm(T({M}, 512):float8_e5m2:(512, 1),T(512, 256):float8_e4m3fn:"
                                      "(1, 512),T(1,):float32:(1,),T(1,):float32:(1,),bias=None,"
                                      "out_dtype=torch.bfloat16)"])
                self.assertEqual(self.lb.ln_calls, [dict(is_swish=True, sum_dx=True, M=M)])
                self.assertFalse(any(e.startswith("triton_weighted_layer_norm") for e in LOG))
                self.assertEqual([(c["mode"], c["grid"]) for c in self.emu.calls], [("emit", G), ("s1", G), ("s1", 7)])
                if rule == "report":
                    self.assertLess(out["finals_use"], 1.0)
                    self.assertEqual(out["finals_vs_pr7_max_abs"], [0.0, 0.0, 0.0])

    def test_check_points_catch_each_defect(self):
        for defect, rule, msg in (("da1_bit", "all", r"\(a\) S1\+DA1 da1 != hipBLASLt"),
                                  ("dh1_vs_pr7", "all", r"\(b\) S1\+DA1 dh1 != PR #7"),
                                  ("s1_vs_emit", "all", r"\(c\) S1 dh1 != S1\+DA1 dh1"),
                                  ("grid7", "all", r"\(d\) S1 grid 7 dh1 != grid 5 dh1"),
                                  ("canary", "all", r"\(e\) dh1_x: .* canary words in rows >= M"),
                                  ("part_hole", "all", r"\(e\) S1\+DA1 PART: 1 of .* never written")):
            with self.subTest(defect=defect):
                with self.assertRaisesRegex(RuntimeError, msg):
                    self.check(rule, emu=EmuLib(defect=defect))
        with self.assertRaisesRegex(RuntimeError, r"finals dW outside \(gamma_"):
            self.check("report", M=130, lb=fake_lb(fin_scale=1.01))
        with self.assertRaisesRegex(RuntimeError, r"\(a\) .* at M 330: 1 of 266 gated rows differ \(first 0\)"):
            self.check("dp_region", M=330, emu=EmuLib(defect="da1_bit"))  # row 0 is in the gated region

    def test_dp_region_reports_rows_beyond_the_gate(self):
        def split_last_row(a, b, sa, sb, *, out_dtype, bias=None, scale_result=None):
            o = f_scaled_mm(a, b, sa, sb, out_dtype=out_dtype, bias=bias)
            o.view(torch.int16)[-1, 0] ^= 1  # hipBLASLt's da1 differs on one "split" row past the gate
            return o

        with mock.patch.object(TF8, "_scaled_mm", split_last_row):
            out, _ = self.check("dp_region", M=330)
            self.assertEqual((out["da1_gated_rows"], out["da1_reported_diff_rows"]), (330 - 64, 1))
            with self.assertRaisesRegex(RuntimeError, r"\(a\) .*1 of 330 gated rows differ \(first 329\)"):
                self.check("all", M=330)

    def prebuild(self, lb=None, mem_free=1 << 40, emu=None):
        """prebuild_check on cuda:0 with the device side mocked; each check point runs the REAL check_point on CPU
        at a small M (M_DP -> 300, M_A -> 330, M_B -> 390, M_FIN -> 130)."""
        X = self.X
        self.arm(emu)
        X._armed.clear()
        X._ZERO.clear()
        X._P2_STATE = None
        lb = lb or fake_lb()
        X._LB = None
        small = {X.M_DP: 300, X.M_A: 330, X.M_B: 390, X.M_FIN: 130}
        real_cp = X.check_point
        seen, heads = [], []

        def cp(lib, M, rule, device, live):
            seen.append((M, rule, device))
            return real_cp(lib, small[M], rule, self.cpu, live)

        class RT(X._CudaRuntime):
            @staticmethod
            def make_zero(dev):
                return torch.zeros((), dtype=BF)

        rt = RT()
        rt._ncu[-1] = NCU
        X._RT = rt
        import generative_recommenders.ops.flydsl as flypkg

        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(X, "check_point", cp))
            st.enter_context(mock.patch.object(torch.cuda, "device", lambda d: contextlib.nullcontext()))
            st.enter_context(mock.patch.object(torch.cuda, "synchronize", lambda d=None: None))
            st.enter_context(mock.patch.object(torch.cuda, "mem_get_info",
                                               lambda d=None: heads.append(d) or (mem_free, 1 << 41)))
            st.enter_context(mock.patch.object(torch.cuda, "memory_reserved", lambda d=None: 0))
            st.enter_context(mock.patch.object(torch.cuda, "memory_allocated", lambda d=None: 0))
            st.enter_context(mock.patch.dict(sys.modules, {"generative_recommenders.ops.flydsl.ln_bwd": lb}))
            st.enter_context(mock.patch.object(flypkg, "ln_bwd", lb, create=True))
            st.enter_context(mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", True))
            st.enter_context(mock.patch.object(TLN, "_WLN_BWD_FLYDSL_BROKEN", None))
            token = X.prebuild_check(torch.device("cuda", 0))
            pr7 = (TLN._wln_bwd_flydsl_enabled(), TLN._WLN_BWD_FLYDSL_BROKEN)
        return token, seen, heads, pr7

    def test_prebuild_check_arms_and_prints_line_2(self):
        X = self.X
        token, seen, heads, _ = self.prebuild()
        self.assertEqual(token, "x2lb=1")
        self.assertEqual([(M, r) for M, r, _ in seen], list(X.CHECK_POINTS))
        self.assertEqual(len(heads), len(X.CHECK_POINTS))  # a headroom check before every check point
        self.assertEqual((X._armed, X._P2_STATE), ({0}, "ok"))
        self.assertIn(0, X._ZERO)
        self.assertIs(X._LB.ln_bwd, X._LB.ln_bwd)
        (line,) = self.lines("self-check OK")
        self.assertTrue(line.startswith("armed OK: [preproc-l2-x2lb] pre-run_start self-check OK (cancel: M 2356560 "
                                        "da1 == hipBLASLt all rows; M 2424832 / 2492120 da1 == hipBLASLt on rows < "
                                        "M-64 ("), line)
        self.assertIn("dh1 == PR #7(da1) all rows; S1 == S1+DA1; grids 5/5/5/7; canaries; partials within gamma at "
                      "M 16383 (use ", line)
        self.assertIn("fin pr7); peak transient", line)
        self.assertTrue(line.endswith("device=cuda:0)"), line)
        self.assertEqual(X.PREBUILD_INFO[0]["peak_bytes"], 5 * 390 * 512 + 8 * 390 + 2 * 64 * 512 + 3 * 5 * 1024)

    def test_headroom_fails_closed_before_any_allocation(self):
        X = self.X
        token, seen, heads, pr7 = self.prebuild(mem_free=5 * X.M_DP * 512)  # < 5U + 1 GiB
        self.assertEqual(token, "x2lb=error")
        self.assertEqual(seen, [])  # no check point, no allocation
        (line,) = self.lines("FAILED CLOSED")
        self.assertIn("(phase-2 headroom: HeadroomError: phase-2 headroom: ", line)
        self.assertIn(f"at M {X.M_DP})", line)
        self.assertEqual((X._armed, X._P2_STATE), (set(), "failed"))
        self.assertEqual(pr7, (True, None))
        with mock.patch.object(torch.cuda, "mem_get_info", lambda d=None: (10, 0)), \
                mock.patch.object(torch.cuda, "memory_reserved", lambda d=None: 100), \
                mock.patch.object(torch.cuda, "memory_allocated", lambda d=None: 0):
            X._headroom(self.cpu, 110, 1)  # reserved-but-free bytes count as free
            with self.assertRaises(X.HeadroomError):
                X._headroom(self.cpu, 111, 1)

    def test_a_phase2_error_latches_only_x2lb(self):
        X = self.X
        for what, kw in (("ln_bwd", dict(lb=fake_lb(fail=True))), ("scaled_mm", {})):
            with self.subTest(what=what):
                X.reset_for_testing()
                self.said.clear()
                ctx = (mock.patch.object(TF8, "_scaled_mm", mock.Mock(side_effect=RuntimeError("injected hipBLASLt")))
                       if what == "scaled_mm" else contextlib.nullcontext())
                with ctx:
                    token, _, _, pr7 = self.prebuild(**kw)
                self.assertEqual(token, "x2lb=error")
                (line,) = self.lines("FAILED CLOSED")
                self.assertIn("(phase-2 self-check: RuntimeError: injected", line)
                self.assertEqual(pr7, (True, None))  # PR #7 enabled and not latched
                self.assertIsNotNone(X.broken())
                self.assertEqual(X._armed, set())
        X.reset_for_testing()
        self.said.clear()
        token, _, _, _ = self.prebuild(emu=EmuLib(defect="s1_vs_emit"))  # a failed check: the same
        self.assertEqual(token, "x2lb=error")
        self.assertIn("(c) S1 dh1 != S1+DA1 dh1", self.lines("FAILED CLOSED")[0])

    def test_phase2_preconditions(self):
        X = self.X
        self.knob(None)
        self.assertEqual(X.prebuild_check(torch.device("cuda", 0)), "")  # knob off
        self.knob("1")
        self.assertEqual(X.prebuild_check(torch.device("cuda", 0)), "x2lb=off")  # phase 1 did not run
        self.assertEqual(X._P2_SKIP, "phase 1 did not pass (not run)")
        X._P2_SKIP = None
        X._P1_STATE, X._lib = "ok", object()
        with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", True), \
                mock.patch.object(TLN, "_WLN_BWD_FLYDSL_BROKEN", "latched"):
            self.assertEqual(X.prebuild_check(torch.device("cuda", 0)), "x2lb=off")
        self.assertEqual(X._P2_SKIP, "PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) off or latched off")
        X._BROKEN = "x"
        self.assertEqual(X.prebuild_check(torch.device("cuda", 0)), "x2lb=off")
        self.assertEqual(self.said, [])
        X._BROKEN = None
        with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", True), \
                mock.patch.object(TLN, "_WLN_BWD_FLYDSL_BROKEN", None):
            self.assertEqual(X.prebuild_check(torch.device("cpu")), "x2lb=error")  # not a GPU: fails closed
        self.assertIn("device cpu is not a GPU", self.lines("FAILED CLOSED")[0])

    def test_phase1(self):
        X = self.X
        dev = torch.device("cuda", 0)
        self.knob(None)
        self.assertFalse(X.prebuild_build(dev))
        self.assertEqual(self.said, [])

        def run(patches=None, mlps=None):
            patches = dict(patches or {})
            X.reset_for_testing()
            self.said.clear()
            self.knob("1", mlps)
            with contextlib.ExitStack() as st:
                for (obj, name), val in patches.items():
                    st.enter_context(mock.patch.object(obj, name, val))
                ok = X.prebuild_build(dev)
            return ok, list(self.said)

        ok, said = run()
        self.assertFalse(ok)
        self.assertEqual(said, ["[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED (NOT armed): E-M3a "
                                "(DLRM_HSTU_FP8_PREPROC_L2) is off or IGNORED; shipped dgrad + SwishLN bwd for the "
                                "whole run"])
        self.assertEqual(X._P1_STATE, "ignored")
        on = {(L2, "enabled"): lambda: True, (TLN, "_WLN_BWD_FLYDSL_ENABLED"): True}
        ok, said = run({(L2, "enabled"): lambda: True, (TLN, "_WLN_BWD_FLYDSL_ENABLED"): False})
        self.assertIn("IGNORED (NOT armed): PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is off", said[0])
        ok, said = run(on | {(torch.cuda, "is_available"): lambda: False})
        self.assertIn("IGNORED (NOT armed): device cuda:0 is not a GPU", said[0])
        props = types.SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        gpu = on | {(torch.cuda, "is_available"): lambda: True, (torch.cuda, "get_device_properties"): lambda d: props}
        ok, said = run(gpu)
        self.assertIn("IGNORED (NOT armed): device arch gfx942 is not gfx950", said[0])
        props.gcnArchName = "gfx950:sramecc+:xnack-"
        ok, said = run(gpu, mlps="2")
        self.assertFalse(ok)
        self.assertIn("FAILED CLOSED", said[0])
        self.assertIn("phase 1 (configuration): ValueError: DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_MLPS='2'", said[0])
        self.assertEqual(X._P1_STATE, "failed")
        ok, said = run(gpu | {(X, "load"): mock.Mock(side_effect=RuntimeError("ISA gate refuses this build"))})
        self.assertFalse(ok)
        self.assertIn("FAILED CLOSED -> shipped dgrad + SwishLN bwd for the rest of the run (phase 1 (build / census "
                      "/ ISA gate / ABI): RuntimeError: ISA gate refuses this build)", said[0])
        lib = make_lib(EmuLib())
        ok, said = run(gpu | {(X, "load"): lambda: lib})
        self.assertTrue(ok)
        self.assertEqual(X._P1_STATE, "ok")
        self.assertEqual(len(said), 1)
        self.assertTrue(said[0].startswith(
            "armed OK: [preproc-l2-x2lb] pre-run_start build OK (x2lb_pass<0,0> VGPR 220 / 248, spill 0, scratch 0, "
            "LDS 99328, occupancy 2; isa_check C1-C18 PASS (pinned .s 57de20b6ef50); src a78f4aa3611f; code object "
            ".text 1873c0cbded3 (== lab); emu/libx2lb_pass.so cached; mlps 0,1; fin pr7; phase 1 "), said[0])
        self.assertFalse(X.prebuild_build(dev))  # once per process

    def test_finals_bound_check(self):
        X = self.X
        M, G = 130, 3
        d = X.cancel_data(M, self.cpu, X._Live())
        da1 = f_scaled_mm(d["dh8"], d["w8t"].t(), d["sa"], d["sb"], out_dtype=BF)
        _, dw, db, dxs = _ln_bwd(da1, d["h1"], d["nw"], d["nb"], d["mean1"], d["rstd1"], True)
        use = X.finals_bound_check(d, da1, (dw, db, dxs), G, M)
        self.assertLess(use, 1.0)
        with self.assertRaisesRegex(RuntimeError, "finals dB outside"):
            X.finals_bound_check(d, da1, (dw, db * 1.01, dxs), G, M)
        with self.assertRaisesRegex(RuntimeError, "finals dXS: non-finite"):
            X.finals_bound_check(d, da1, (dw, db, dxs * float("nan")), G, M)


# =================================================================================== 5b. blas_warmup hooks, 6. gin
class X2LBHooksTest(_Base):
    def setUp(self):
        super().setUp()
        try:
            from generative_recommenders.dlrm_v4.train import blas_warmup  # needs gin
            from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
        except ImportError as exc:
            self.skipTest(f"needs gin: {exc}")
        self.bw, self.apply = blas_warmup, apply_env_bootstrap

    def test_knob_off_hooks_import_and_log_nothing(self):
        for value in (None, "0"):
            r = _child(r"""
                import io, logging, sys, torch
                buf = io.StringIO()
                logging.basicConfig(stream=buf, level=logging.DEBUG)
                from generative_recommenders.dlrm_v4.train import blas_warmup as bw
                bw._prebuild_preproc_l2_x2lb_build(torch.device("cpu"))
                assert bw._prebuild_preproc_l2_x2lb_check(torch.device("cpu")) == ""
                bw._x2lb_phase2_not_run("anything")
                bw.run_startup_preload(False, 1)
                assert "generative_recommenders.ops.hip.preproc_l2_x2lb" not in sys.modules
                assert "x2lb" not in buf.getvalue(), buf.getvalue()
                print("CHILD OK")
            """, {} if value is None else {ENV: value})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("CHILD OK", r.stdout)
            self.assertNotIn("x2lb", r.stdout.replace("CHILD OK", ""))

    def test_phase1_hook(self):
        X = self.X
        self.knob("1")
        with mock.patch.object(X, "prebuild_build", return_value=True) as pb:
            self.bw._prebuild_preproc_l2_x2lb_build(torch.device("cpu"))
        pb.assert_called_once_with(torch.device("cpu"))
        out = io.StringIO()
        with _blocked_import(), contextlib.redirect_stdout(out):
            self.bw._prebuild_preproc_l2_x2lb_build(torch.device("cpu"))
        self.assertIn("[preproc-l2-x2lb] FAILED CLOSED at prebuild (import): ModuleNotFoundError", out.getvalue())
        self.assertIs(L2._X2LB_REQUESTED, False)  # the forward never retries the import (no second line)
        self.knob("1")
        with mock.patch.object(X, "prebuild_build", side_effect=RuntimeError("unexpected")):
            self.bw._prebuild_preproc_l2_x2lb_build(torch.device("cpu"))  # belt and braces: latched, not raised
        self.assertIn("phase 1: RuntimeError: unexpected", X.broken())

    def test_phase2_hook_restores_rng_releases_memory_and_returns_the_token(self):
        X = self.X
        self.knob("1")
        rng = torch.get_rng_state()

        def draws(dev):
            torch.rand(3)  # a (forbidden) RNG draw
            return "x2lb=1"

        with mock.patch.object(X, "prebuild_check", side_effect=draws), \
                mock.patch.object(torch.cuda, "empty_cache") as ec, \
                self.assertLogs(self.bw.logger, level="ERROR") as logs:
            self.assertEqual(self.bw._prebuild_preproc_l2_x2lb_check(torch.device("cpu")), " x2lb=1")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        ec.assert_called_once_with()
        self.assertIn("RNG state moved in the x2lb phase 2; restoring it", logs.output[0])
        with mock.patch.object(X, "prebuild_check", side_effect=RuntimeError("unexpected")), \
                mock.patch.object(torch.cuda, "empty_cache") as ec:
            self.assertEqual(self.bw._prebuild_preproc_l2_x2lb_check(torch.device("cpu")), " x2lb=error")
        ec.assert_called_once_with()
        self.assertIn("phase-2 self-check: RuntimeError: unexpected", X.broken())

    def preconditions(self, fn, reason):
        """fn() runs the startup-preload path that does not reach phase 2; with phase 1 passed, the first publish
        prints exactly one IGNORED line with `reason` and publishes nothing."""
        X = self.X
        X.reset_for_testing()
        self.said.clear()
        self.knob("1")
        X._P1_STATE, X._lib = "ok", object()
        fn()

        class Ctx:
            pass

        h = torch.zeros(4, 256, dtype=BF)
        self.assertIsNone(X.publish(Ctx(), 0, h, True))
        self.assertIsNone(X.publish(Ctx(), 1, h, True))
        self.assertEqual(len(self.said), 1, self.said)
        self.assertTrue(self.said[0].startswith("[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED (NOT "
                                                "armed): phase-2 self-check not run ("), self.said[0])
        self.assertIn(reason, self.said[0])

    def test_arming_preconditions(self):
        bw = self.bw
        self.preconditions(lambda: bw.run_startup_preload(False, 1),
                           "the DataLoader was not pre-started (DL_PRESTART, gin item (18))")
        self.preconditions(lambda: bw.run_startup_preload(True, 1, flydsl=0, gc_freeze=0),
                           "the FlyDSL preload / ln_bwd prebuild did not run (flydsl=0 ln_bwd=1 result off)")
        with mock.patch.object(bw, "_preload_flydsl", lambda: "error"):
            self.preconditions(lambda: bw.run_startup_preload(True, 1, gc_freeze=0),
                               "(flydsl=1 ln_bwd=1 result error)")
        with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", False):
            self.preconditions(lambda: self.assertEqual(bw._prebuild_flydsl_ln_bwd(), "ln_bwd=off"),
                               "PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is off")
        props = types.SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(TLN, "_WLN_BWD_FLYDSL_ENABLED", True), \
                mock.patch.object(torch.cuda, "get_device_properties", lambda d: props):
            self.preconditions(lambda: self.assertTrue(bw._prebuild_flydsl_ln_bwd().startswith("ln_bwd=skip(arch")),
                               "device arch gfx942:sramecc+:xnack- is not gfx950")
        self.preconditions(lambda: None, "needs mode streaming-train-eval")  # the startup preload never ran

    def test_env_bootstrap_binding(self):
        for value, want in ((False, "0"), (True, "1")):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.apply(DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=value)
                self.assertEqual(os.environ.get(ENV), want)
        with mock.patch.dict(os.environ, {ENV: "1"}, clear=True):
            self.apply(DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=False)  # a pre-set env var wins
            self.assertEqual(os.environ.get(ENV), "1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.apply()
            self.assertNotIn(ENV, os.environ)

    def test_gin_binding_and_item_block(self):
        with open(_GIN) as f:
            text = f.read()
        lines = [ln.strip() for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        live = [ln for ln in lines if ENV in ln]
        self.assertEqual(live, [_GIN_BINDING])
        items = [int(n) for n in re.findall(r"^# \((\d+)\) ", text, flags=re.M)]
        # (93)'s own block, not "highest item == 93": later items ((94)+) and other owners' blocks may follow it or
        # sit between item (52)'s binding and (93)'s header (ship17 rows bind items up to 109+).
        self.assertGreaterEqual(max(items), 93)
        self.assertEqual(items.count(93), 1)
        start = text.index("# (93) X2L-B")
        self.assertTrue(text[:start].endswith("# -------------------------------------------------------"
                                              "----------------------\n"))  # (93)'s block opens after a separator
        block = text[start:text.index("\n# ----", start)]
        b52 = text.index("apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_L2 = True\n")
        self.assertLess(b52, start)  # (93) comes after item (52)'s binding, which arms the FP8 L2 path it hooks
        for marker in ("DECLARED fp32 REDUCTION-ORDER NUMERICS CLASS", "ARMING PRECONDITIONS", "FALLBACKS",
                       '"armed OK: [preproc-l2-x2lb] pre-run_start build OK"',
                       '"armed OK: [preproc-l2-x2lb] pre-run_start self-check OK"',
                       '"armed OK: [preproc-l2-x2lb] fused L2 dgrad + SwishLN256 bwd"',
                       '"[preproc-l2-x2lb] FAILED CLOSED"', '"[preproc-l2-x2lb] DECLINED"',
                       '"[preproc-l2-x2lb] NOT armed"',
                       '"[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED"',
                       '"[preproc-l2-x2lb] handshake"', "Opt out (kill switch): DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=0."):
            self.assertIn(marker, block)
        self.assertIn(_GIN_BINDING, block)

    def test_production_gin_parses_and_binds(self):
        import gin

        want = _GIN_BINDING.split(" = ")[1] == "True"
        try:
            gin.clear_config()
            gin.parse_config_file(_GIN, skip_unknown=True)
            self.assertIs(gin.query_parameter("apply_env_bootstrap." + ENV), want)
            with mock.patch.dict(os.environ, {}, clear=True):
                self.apply()
                self.assertEqual(os.environ.get(ENV), "1" if want else "0")
            with mock.patch.dict(os.environ, {ENV: "0" if want else "1"}, clear=True):
                self.apply()
                self.assertEqual(os.environ.get(ENV), "0" if want else "1")  # a pre-set env var wins over gin
        finally:
            gin.clear_config()


@contextlib.contextmanager
def _blocked_import():
    """Any import of ops/hip/preproc_l2_x2lb raises ModuleNotFoundError (sys.modules entry None, the package's
    attribute removed) for the duration."""
    import generative_recommenders.ops.hip as hip_pkg

    had = getattr(hip_pkg, "preproc_l2_x2lb", None)
    if had is not None:
        delattr(hip_pkg, "preproc_l2_x2lb")
    try:
        with mock.patch.dict(sys.modules, {MOD: None}):
            yield
    finally:
        if had is not None:
            hip_pkg.preproc_l2_x2lb = had


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--driver":
        _driver(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
