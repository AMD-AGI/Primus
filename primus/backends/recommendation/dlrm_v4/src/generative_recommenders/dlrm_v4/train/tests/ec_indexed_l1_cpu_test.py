"""Indexed-X L1 (DLRM_HSTU_FP8_EC_INDEXED_L1, default off): the autograd protocol on CPU (no GPU, no launch).

Drives the REAL ec_bf16_output._merged_construct and _ECExpandGroup (and, with E-X2b on, ECExpandGroupPorts and
the real claim / bind / route of ops/triton/triton_ec_dgrad_segreduce.py), the REAL E-M3a FP8 head
preproc_fp8_l2._LinearSwishLNEmitFunction, the REAL preproc_grouped_tail._run_head + _LinearNormFunction, the REAL
FusedLinearNormMLP.forward (nn.Sequential on CPU) and the REAL protocol module ops/ec_indexed_l1.py. The GPU-only
pieces become torch stand-ins with the shipped arithmetic:
  * the Indexed-X kernels: K.check_fwd / K.check_wgrad (dtype, shape, stride, storage class) and then
    K.reference_fwd / K.reference_wgrad = the shipped addmm / dh.t() @ x on the gathered rows;
  * the EC segment reduce = fp32 index_add_, build_segments = stable argsort + bincount, E-X2b's fused kernel =
    index_add_ of torch.mm addends (as ec_dgrad_ports_cpu_test.py);
  * the LayerNorm kernels and E-M3a's P2 emit = deterministic torch surrogates; E-M3a's scale site = a stub.
So knob on must give the knob-off gradients BIT FOR BIT here. On the GPU the kernels are the declared fp32
reduction-order class (lab GPU tests, probes/iter83/ixint/tests).
"""

import contextlib
import gc
import inspect
import logging
import os
import re
import subprocess
import sys
import textwrap
import types
import unittest
import warnings
import weakref
from unittest import mock

import torch
import torch.nn.functional as F

try:
    from generative_recommenders.common import HammerKernel
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
    from generative_recommenders.dlrm_v4.train import ec_dgrad_ports as xp
    from generative_recommenders.ops import ec_indexed_l1 as ixp
    from generative_recommenders.ops import fused_linear_norm as fln
    from generative_recommenders.ops import preproc_fp8_l2 as pfl
    from generative_recommenders.ops import preproc_grouped_tail as pgt
    from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
    from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs
    from generative_recommenders.ops.triton import triton_ec_indexed_l1 as K
except (ImportError, OSError) as exc:  # e.g. gin or triton missing
    raise unittest.SkipTest(f"Indexed-X CPU test needs the EC / preprocessor modules: {exc}")

from torch.utils._python_dispatch import TorchDispatchMode

W, H = 512, 256
CAT_FIRST = ["artist_id", "album_id", "item_id", "uid", "artist_cand", "album_cand", "item_cand"]
ITEM_FIRST = ["item_id", "artist_id", "album_id", "uid", "artist_cand", "album_cand", "item_cand"]
PAIRS = [("item_id", "item_cand"), ("artist_id", "artist_cand"), ("album_id", "album_cand")]
CAT_GROUP = ["artist_id", "album_id"]
KEPT = ["uid", "artist_cand", "album_cand", "item_cand"]
UIH = [5, 7, 3, 9]
B = len(UIH)
M = sum(UIH) + B  # merged member rows (28)
# disjoint tables: item 0-15, artist 16-27, album 28-39, uid 40-43; 44-47 unused
TABLE = {"item_id": (0, 16), "artist_id": (16, 28), "album_id": (28, 40), "uid": (40, 44),
         "item_cand": (0, 16), "artist_cand": (16, 28), "album_cand": (28, 40)}
U = 48
XSHAPES = {(M, W), (2 * M, W), (M, 2 * W)}  # a materialized member X (item, cat flat, cat big)
_EBO_FLAGS = ("_CAT_LOGGED", "_MERGE_LOGGED", "_EC_GROUP_LOGGED", "_SEG_ARMED_LOGGED")
_ESR = "generative_recommenders.dlrm_v4.train.ec_segment_reduce"
_GF = "generative_recommenders.ops.glue_fusion"
_TPF = "generative_recommenders.ops.triton.triton_preproc_fp8"
_JT = "torchrec.sparse.jagged_tensor"
_MISSING = object()

# eval/eval_armed_census.py:166-170 (training_dlrmv4_arbor_r16), verbatim.
ARMED_RE = re.compile(r"\bARMED\b|\barmed OK:|(?<!['\"])\barmed\b(?!['\"])")
DECL_RE = re.compile(
    r"NOT used|NOT armed|\bINERT\b"
    r"|\b(?:[Dd]eclin\w*|[Ff]all(?:s|ing)?[ -]?[Bb]acks?|[Dd]isabled|DISABLED)\b"
    r"(?!\s*[=:]\s*(?:None|0|False|false|no|off)\b)")
ARMED_I = re.compile(ARMED_RE.pattern, re.I)
DECL_I = re.compile(DECL_RE.pattern, re.I)

SAID = []  # loud lines (K._say: Indexed-X, every rank)
XS_SAID = []  # loud lines of E-X2b (xs._say)
INFO = []  # INFO lines of ops/ec_indexed_l1.py
CALLS = {"fwd": 0, "wgrad": 0, "fused": 0}
IN_STANDIN = [0]  # > 0 while a kernel stand-in runs (it materializes X on purpose)


class _Capture(logging.Handler):
    def emit(self, record):
        INFO.append(record.getMessage())


@contextlib.contextmanager
def _standin():
    IN_STANDIN[0] += 1
    try:
        yield
    finally:
        IN_STANDIN[0] -= 1


# ---------------------------------------------------------------- stand-ins (the shipped arithmetic)
def _ecix_fwd_ref(table, idx, nh, w1, b1):
    with _standin():
        K.check_fwd(table, idx, nh, w1, b1)
        CALLS["fwd"] += 1
        return K.reference_fwd(table, idx, nh, w1, b1)


def _ecix_wgrad_ref(dh, table, idx, nh, out_dtype=torch.float32, workspace=None):
    with _standin():
        K.check_wgrad(dh, table, idx, nh, out_dtype)
        CALLS["wgrad"] += 1
        return K.reference_wgrad(dh, table, idx, nh, out_dtype)


def _seg_reduce_ref(g, idx, num_rows, out_dtype=torch.float32, order=None, offsets=None,
                    accumulate_into=None):
    out = accumulate_into
    if out is None:
        out = torch.zeros((num_rows, g.shape[1]), dtype=torch.float32)
    out.index_add_(0, idx, g.float())
    return out


def _build_segments_ref(idx, num_rows):
    idx = idx.reshape(-1)
    order = torch.argsort(idx, stable=True)
    off = torch.zeros(num_rows + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(torch.bincount(idx, minlength=num_rows), 0)
    return order, off


def _joint_ref(hoisted, num_rows):
    if len(hoisted) != 2:
        return None
    return [_build_segments_ref(idx, num_rows) for idx, _ in hoisted]


def _fused_ref(cat, item, num_rows, emit=None):
    """E-X2b's kernel contract, checked, with the shipped arithmetic (ec_dgrad_ports_cpu_test.py)."""
    with _standin():
        for m, kind in ((cat, "cat"), (item, "item")):
            xs._check_member(m, kind, m.dh.device, num_rows)
        CALLS["fused"] += 1
        acc = torch.zeros((num_rows, W), dtype=torch.float32)
        acc.index_add_(0, cat.idx, torch.mm(cat.dh, cat.w1).view(-1, W).float())
        acc.index_add_(0, item.idx, torch.mm(item.dh, item.w1).float())
        return acc


def _merged_index_ref(ri_u, ri_c, len_u, len_c, n_u, n_c):
    parts, a, b = [], 0, 0
    for x, y in zip(len_u.tolist(), len_c.tolist()):
        parts += [ri_u[a:a + x], ri_c[b:b + y]]
        a, b = a + x, b + y
    return torch.cat(parts)


def _norm_fwd_ref(x, weight, bias, eps, is_swish, residual=None):
    """fused_linear_norm._norm_fwd stand-in: -> (y like x, mean fp32 [N], rstd fp32 [N])."""
    xf = x.float()
    mean = xf.mean(1)
    rstd = torch.rsqrt(xf.var(1, unbiased=False) + eps)
    y = (xf - mean[:, None]) * rstd[:, None] * weight.float() + bias.float()
    if is_swish:
        y = xf * torch.sigmoid(y)
    y = y.to(x.dtype)
    if residual is not None:
        y = y + residual
    return y, mean, rstd


def _ln_bwd_ref(dy, x, weight, bias, mean, rstd, eps, BLOCK_D, is_swish):
    """triton_weighted_layer_norm_bwd_dxsum stand-in: -> (dx like x, contiguous; dweight, dbias, dxsum fp32)."""
    xf, dyf = x.float(), dy.float()
    gate = torch.sigmoid(xf) if is_swish else torch.ones_like(xf)
    dx = (dyf * weight.float() * gate * rstd[:, None]).to(x.dtype).contiguous()
    return dx, (dyf * xf).sum(0), dyf.sum(0), dx.float().sum(0)


def _emit_ref(h1, w, b, eps, inv_scale, emu, tm=True, return_partials=False):
    """triton_preproc_fp8.swishln256_fwd_emit stand-in: -> (a1_rm, a1_tm, mean, rstd, partials)."""
    y, mean, rstd = _norm_fwd_ref(h1, w, b, eps, True)
    return y, (y.t().contiguous() if tm else None), mean, rstd, y.float().abs().amax().reshape(1)


class _Site:
    """E-M3a's delayed-scale site (preproc_fp8_l2._Site) without its Triton bookkeeping launch."""

    def __init__(self):
        self.scale = torch.ones(1)
        self.inv_scale = torch.ones(1)

    def steady(self, partials, snapshot):
        return self.scale.clone()


class _JaggedTensor:
    def __init__(self, lengths, values, weights=None):
        self._l, self._v = lengths, values

    def values(self):
        return self._v

    def lengths(self):
        return self._l


class _Feats:
    def __init__(self, lengths, stride):
        self._l, self._s = lengths, stride

    def lengths(self):
        return self._l

    def stride(self):
        return self._s


class _Sink(torch.autograd.Function):
    """dL/dy = r for E-M3a's stride-0 a1 proxy WITHOUT reading it (its one storage element is uninitialized);
    the loss term's value is 0, so a non-finite loss can only come from another reader."""

    @staticmethod
    def forward(ctx, y, r):
        ctx.r, ctx.dt = r, y.dtype
        return torch.zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, g):
        return (g * ctx.r).to(ctx.dt), None


class _XDetect(TorchDispatchMode):
    """Records ops (outside the kernel stand-ins) whose output has a member-X shape ([M, 512], [2M, 512],
    [M, 1024]) with non-zero strides: a materialized X or dX."""

    def __init__(self):
        super().__init__()
        self.hits = []
        self.phase = "fwd"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if IN_STANDIN[0] == 0:
            for t in torch.utils._pytree.tree_leaves(out):
                if isinstance(t, torch.Tensor) and tuple(t.shape) in XSHAPES and any(s for s in t.stride()):
                    self.hits.append((self.phase, str(func), tuple(t.shape)))
        return out


# ---------------------------------------------------------------- model pieces
def _inputs(names, item_lens=None):
    g = torch.Generator().manual_seed(7)
    lens = {n: (list(UIH) if n in ("item_id", "artist_id", "album_id") else [1] * B) for n in names}
    if item_lens is not None:
        lens["item_id"] = list(item_lens)
    ri = []
    for n in names:
        lo, hi = TABLE[n]
        ri.append(torch.randint(lo, hi, (sum(lens[n]),), generator=g))
    lengths = torch.tensor([v for n in names for v in lens[n]], dtype=torch.int64)
    lpk = [sum(lens[n]) for n in names]
    offs = [sum(lpk[:i]) for i in range(len(names))]
    emb = torch.randn(U, W, generator=g)
    R = {"c": dict(h=torch.randn(M, H, generator=g), o=torch.randn(M, W, generator=g)),
         "a": dict(h=torch.randn(M, H, generator=g), o=torch.randn(M, W, generator=g)),
         "x": torch.randn(M, W, generator=g), "h2": torch.randn(M, H, generator=g)}
    for n in KEPT:
        R[n] = torch.randn(B, W, generator=g)
    return torch.cat(ri), _Feats(lengths, B), lpk, offs, emb, R


def _mlps(wdtype=torch.float32):
    """Content (512) and additional (1024) FusedLinearNormMLPs with seeded weights, PyTorch norms (CPU).
    wdtype: the dtype of pair 0's Linear (its W1 master weight)."""
    g = torch.Generator().manual_seed(11)
    out = []
    with torch.random.fork_rng():
        for din in (W, 2 * W):
            m = fln.FusedLinearNormMLP(torch.nn.Linear(din, H), SwishLayerNorm(H), torch.nn.Linear(H, W),
                                       LayerNorm(W))
            with torch.no_grad():
                for p in m.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) / 16)
            for mod in m.modules():
                if hasattr(mod, "_hammer_kernel"):
                    mod._hammer_kernel = HammerKernel.PYTORCH
            if wdtype != torch.float32:
                m[0].to(wdtype)
            out.append(m)
    return out


def fp8_head(mlp, x, k):
    """preproc_fp8_l2.run_head's FP8 branch: E-X2b claim at the call site, then the real FP8 head."""
    (lin0, norm0), _ = mlp._pairs()
    slot = pfl.Slot(k)
    slot.train = torch.is_grad_enabled()
    rec, port = xs.claim(x, lin0.weight)
    return pfl._LinearSwishLNEmitFunction.apply(x, lin0.weight, lin0.bias, norm0.weight, norm0.bias, norm0._eps,
                                                k, 2, slot.train, slot, rec, port)


def _consume(path, mlp, x, k, r, res, kind):
    if path == "head":  # E-M3a FP8 head (training steps >= 5): the indexed L1
        y = fp8_head(mlp, x, k)
        res["ys"][kind] = y
        if y.grad_fn is not None:
            res["saved"][kind] = y.grad_fn.saved_tensors  # before the backward frees them
        return _Sink.apply(y, r["h"])
    if path == "run_head":  # E-M3a warm-up / non-FP8 head / grouped-tail re-run: rebuild
        y = pgt._run_head(mlp, x)
        res["ys"][kind] = y
        return (y.float() * r["h"]).sum()
    if path == "fln":  # grouped tail declined or off: FusedLinearNormMLP.forward rebuilds
        y = mlp(x)
        res["ys"][kind] = y
        return (y.float() * r["o"]).sum()
    if path == "linear":  # an Indexed-X-UNAWARE reader (plain F.linear on the member output)
        return F.linear(x.float(), mlp[0].weight.float()).float().sum()
    raise ValueError(path)


def _extra(extra, values, ret, mlp_c, R, res):
    if extra is None:
        return None
    if extra == "masked_reader":
        z = values.float()
        return torch.where(torch.zeros_like(z, dtype=torch.bool), z, torch.zeros_like(z)).sum()
    if extra == "nan_to_num_reader":
        return torch.nan_to_num(values.float(), nan=0.0).sum()
    if extra == "colview_reader":  # a CATFUSED column view of `big` (the album half)
        return torch.nan_to_num(ret["album_id"].values().float(), nan=0.0).sum()
    if extra == "plain_reader":  # an unaware reader NEXT TO the indexed head
        return (values.float() * R["x"]).sum()
    if extra == "second_aware":  # a second Indexed-X-aware consumer of the item member (rebuild)
        return (pgt._run_head(mlp_c, values).float() * R["h2"]).sum()
    if extra == "bwd_hook_reader":  # a backward hook on the item head's output reads `values`
        y = res["ys"]["item"]
        y.register_hook(lambda g: (res.setdefault("hook_read", values.float().sum()), None)[1])
        return None
    raise ValueError(extra)


def _arm(ix, ex2b=False):
    """ix: True (knob on), False (unset), None (kill switch "0")."""
    if ix is None:
        os.environ[K.ENV_VAR] = "0"
    elif ix:
        os.environ[K.ENV_VAR] = "1"
    else:
        os.environ.pop(K.ENV_VAR, None)
    ebo._IX_ON = None  # re-read the knob
    K._ENABLED = bool(ix)  # bypasses the ROCm-build check on CPU
    K._BROKEN = None
    ixp.reset_for_testing()
    if ex2b:
        os.environ[xs.ENV_VAR] = "1"
    else:
        os.environ.pop(xs.ENV_VAR, None)
    ebo._EX2B_ON = None
    xs._ENABLED = bool(ex2b)
    xs._BROKEN = None
    xs._ONCE.clear()
    xs._BOUND_KINDS.clear()
    xs._PORTS_LOGGED = False
    xp._FUSED_LOGGED = False
    xp._CHECKS_LEFT = 2
    for k in CALLS:
        CALLS[k] = 0


def run(ix, names=CAT_FIRST, *, ex2b=False, item_path="head", cat_path="head", hooks=False, extra=None,
        mode="backward", wdtype=torch.float32, audit=None, detect=None, kept=None, item_lens=None, arm=True):
    """One grouped EC call + the two pair 0s (+ kept consumers) under autocast(bf16), then `mode`."""
    if arm:
        _arm(ix, ex2b)
    ri, feats, lpk, offs, emb0, R = _inputs(names, item_lens)
    emb = emb0.clone().requires_grad_(True)
    mlp_c, mlp_a = _mlps(wdtype)
    res = dict(fired={"values": [], "big": []}, ys={}, saved={})
    n_said, n_xs, n_info = len(SAID), len(XS_SAID), len(INFO)
    with contextlib.ExitStack() as stack:
        for mode_ctx in (audit, detect):
            if mode_ctx is not None:
                stack.enter_context(mode_ctx)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ret = ebo._merged_construct(emb, feats, names, ri, torch.bfloat16, lpk, offs,
                                        KEPT if kept is None else kept)
            values = ret["item_id"].values()
            big = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR)[0]
            res.update(values=values, big=big, ret=ret, emb_t=emb, mlps=(mlp_c, mlp_a))
            if hooks:
                for nm, t in (("values", values), ("big", big)):
                    t.register_hook(lambda g, nm=nm: res["fired"][nm].append(tuple(g.shape)))
                    t.retain_grad()
            loss = _consume(item_path, mlp_c, values, 0, R["c"], res, "item")
            loss = loss + _consume(cat_path, mlp_a, big, 1, R["a"], res, "cat")
            for n in KEPT:
                loss = loss + (ret[n].values() * R[n]).sum()
            more = _extra(extra, values, ret, mlp_c, R, res)
            if more is not None:
                loss = loss + more
        res["loss"] = loss
        if detect is not None:
            detect.phase = "bwd"
        if mode == "backward":
            loss.backward()
        elif mode == "twice":
            loss.backward(retain_graph=True)
            res["acc_after"] = [getattr(values.grad_fn, "ex2b_acc", "absent")]
            loss.backward()
            res["acc_after"].append(getattr(values.grad_fn, "ex2b_acc", "absent"))
        elif mode == "create_graph":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loss.backward(create_graph=True)
        elif mode == "grad_x":
            res["grad_x"] = torch.autograd.grad(loss, [values])[0].detach().clone()
        elif mode == "inputs_w":
            loss.backward(inputs=[mlp_c[0].weight, mlp_a[0].weight])
        elif mode != "none":
            raise ValueError(mode)
    grads = {"emb": emb.grad}
    for tag, m in (("c", mlp_c), ("a", mlp_a)):
        for pn, p in m.named_parameters():
            grads[f"{tag}.{pn}"] = p.grad
    if hooks:
        grads["values.grad"], grads["big.grad"] = values.grad, big.grad
    res["grads"] = {k: (None if v is None else v.detach().clone()) for k, v in grads.items()}
    res["said"], res["xs_said"], res["info"] = SAID[n_said:], XS_SAID[n_xs:], INFO[n_info:]
    res["ec_node"] = type(values.grad_fn).__name__ if values.grad_fn is not None else None
    return res


def _bits(t):
    t = t.detach().contiguous()
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _forward_ec(names=CAT_FIRST, kept=None):
    """One EC call with the current gate state (no re-arming); -> (values, big, ret)."""
    ri, feats, lpk, offs, emb0, _ = _inputs(names)
    emb = emb0.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ret = ebo._merged_construct(emb, feats, names, ri, torch.bfloat16, lpk, offs, KEPT if kept is None else kept)
    tag = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR, None)  # None: no CAT_FUSE member in this call
    return ret["item_id"].values(), None if tag is None else tag[0], ret


def _repo_root():
    # .../<root>/generative_recommenders/dlrm_v4/train/ec_bf16_output.py -> <root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ebo.__file__)))))


def _subprocess(script, env_extra):
    env = dict(os.environ)
    for k in (K.ENV_VAR, ixp.CHECK_ENV_VAR):
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join([_repo_root()] + [p for p in sys.path if p])
    # -B: inside the validator's container (root) a plain run would write root-owned __pycache__ into the tree.
    return subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env=env,
                          timeout=600)


class ECIndexedL1CPUTest(unittest.TestCase):
    def setUp(self):
        # Every patch is recorded by _set and undone by _restore, registered BEFORE the first patch (cleanups
        # also run when setUp fails part way), so nothing leaks into the other tests of the pytest process.
        self._undo = []
        self._stubs = []
        self._env = {k: os.environ.get(k) for k in (
            K.ENV_VAR, ixp.CHECK_ENV_VAR, xs.ENV_VAR, "DLRM_HSTU_FP8_EC_GROUP_BWD",
            "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL")}
        self._log = logging.getLogger(ixp.__name__)
        self._log_level = self._log.level
        self._handler = _Capture()
        self.addCleanup(self._restore)
        os.environ["DLRM_HSTU_FP8_EC_GROUP_BWD"] = "1"
        os.environ.pop(ixp.CHECK_ENV_VAR, None)
        self._log.addHandler(self._handler)
        self._log.setLevel(logging.INFO)
        s = self._set
        for n in _EBO_FLAGS:
            s(ebo, n, getattr(ebo, n))
        s(ebo, "_SEGMENT_REDUCE", True)
        s(ebo, "_MERGE_PAIRS", list(PAIRS))
        s(ebo, "_CAT_GROUP", list(CAT_GROUP))
        s(ebo, "_EX2B_ON", None)
        s(ebo, "_IX_ON", None)
        s(ebo, "_IX_CHAIN", "")
        s(ebo, "_IX_UIH_POST_ID", "item_id")
        s(ebo, "_merged_index", _merged_index_ref)
        s(ebo, "_ec_joint_segments", _joint_ref)
        if hasattr(ebo, "_ec_reduce_fused"):  # ship13 (rccl M4 (e)): the grouped reduce declines
            s(ebo, "_ec_reduce_fused", lambda hoisted, segs, U: (None, segs))
        s(K, "_ENABLED", True)
        s(K, "_BROKEN", None)
        s(K, "_check_device", lambda dev: None)  # K.check_* still validate dtype / shape / stride / storage
        s(K, "ecix_fwd", _ecix_fwd_ref)
        s(K, "ecix_wgrad", _ecix_wgrad_ref)
        s(K, "_say", lambda msg: SAID.append(msg))
        s(ixp, "_DEVICES", ("cuda", "cpu"))
        s(xs, "_ENABLED", False)
        s(xs, "_BROKEN", None)
        s(xs, "dgrad_segreduce", _fused_ref)
        s(xs, "_say", lambda msg: XS_SAID.append(msg))
        s(xs, "_PORTS_LOGGED", xs._PORTS_LOGGED)
        s(xp, "_DEVICES", ("cuda", "cpu"))
        s(xp, "_CHECKS_LEFT", 2)
        s(xp, "_FUSED_LOGGED", False)
        s(pfl, "_ENABLED", True)  # G10: E-M3a on
        s(pfl, "_MIN_ROWS", 1)  # G7 at M = 28
        s(pfl, "_site", lambda role, k, device: _Site())
        s(pfl, "triton_weighted_layer_norm_bwd_dxsum", _ln_bwd_ref)
        s(fln, "_norm_fwd", _norm_fwd_ref)
        s(fln, "triton_weighted_layer_norm_bwd_dxsum", _ln_bwd_ref)
        ixp.reset_for_testing()
        # GPU-only modules: patch the real ones, or stand in where they cannot be imported.
        try:
            import generative_recommenders.ops.triton.triton_preproc_fp8 as tpf
        except Exception:
            tpf = types.ModuleType(_TPF)
            sys.modules[_TPF] = tpf
            self._stubs.append(_TPF)
        s(tpf, "swishln256_fwd_emit", _emit_ref)
        try:
            import generative_recommenders.dlrm_v4.train.ec_segment_reduce as esr
        except Exception:
            esr = types.ModuleType(_ESR)
            sys.modules[_ESR] = esr
            self._stubs.append(_ESR)
        s(esr, "segment_reduce_rows", _seg_reduce_ref)
        s(esr, "build_segments", _build_segments_ref)
        try:
            import generative_recommenders.ops.glue_fusion  # noqa: F401
        except Exception:
            gf = types.ModuleType(_GF)
            gf.merged_index_fusable = lambda *a: False
            sys.modules[_GF] = gf
            self._stubs.append(_GF)
        try:
            import torchrec.sparse.jagged_tensor  # noqa: F401
        except Exception:  # ImportError, or OSError when fbgemm_gpu's .so cannot load
            saved = {n: sys.modules.get(n, _MISSING) for n in ("torchrec", "torchrec.sparse", _JT)}
            self._torchrec = saved
            for name in saved:
                sys.modules[name] = types.ModuleType(name)
            sys.modules[_JT].JaggedTensor = _JaggedTensor
        for k in CALLS:
            CALLS[k] = 0

    def _set(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name, _MISSING)))
        setattr(obj, name, value)

    def _restore(self):
        for obj, name, old in reversed(self._undo):
            if old is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, old)
        for name in self._stubs:
            sys.modules.pop(name, None)
        for name, old in getattr(self, "_torchrec", {}).items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        self._log.removeHandler(self._handler)
        self._log.setLevel(self._log_level)
        ixp.reset_for_testing()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---------------------------------------------------------------- helpers
    def assertBits(self, a, b, what=""):
        self.assertIsNotNone(a, what)
        self.assertIsNotNone(b, what)
        self.assertEqual(a.shape, b.shape, what)
        self.assertEqual(a.dtype, b.dtype, what)
        self.assertTrue(torch.equal(_bits(a), _bits(b)), f"{what}: not bitwise equal")

    def assertSameGrads(self, on, off, keys=None):
        a, b = on["grads"], off["grads"]
        self.assertEqual(sorted(a), sorted(b))
        for k in (keys or sorted(b)):
            if b[k] is None:
                self.assertIsNone(a[k], k)
            else:
                self.assertBits(a[k], b[k], k)

    def assertPlaceholder(self, t, what):
        self.assertEqual(t.stride(), (0, 0), what)
        self.assertEqual(t.untyped_storage().nbytes(), 2, what)
        self.assertTrue(bool(torch.isnan(t.as_strided((1,), (1,))).all()), what)

    def assertNoIX(self, r):
        for k in ("values", "big"):
            self.assertIsNone(getattr(r[k], ixp.IX_ATTR, None), k)
            self.assertNotEqual(r[k].stride(), (0, 0), k)

    def _issued(self):
        return ixp.STATS["issued_item"] + ixp.STATS["issued_cat"]

    # ---------------------------------------------------------------- T1
    def test_import_graph_knob_off(self):
        script = textwrap.dedent(r"""
            import sys
            import torch
            from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
            from generative_recommenders.ops import fused_linear_norm, preproc_fp8_l2, preproc_grouped_tail  # noqa
            from generative_recommenders.modules import preprocessors  # noqa
            from generative_recommenders.dlrm_v4.train import blas_warmup as bw
            bw._prebuild_ec_indexed_l1(torch.device("cpu"))
            bw._prebuild_ec_indexed_l1(torch.device("cuda", 0))
            assert ebo._ix_plan(None, None, [], [], -1, 512, [], []) is None
            assert ebo._IX_ON is False, ebo._IX_ON
            lazy = ['generative_recommenders.ops.ec_indexed_l1',
                    'generative_recommenders.ops.triton.triton_ec_indexed_l1']
            assert not any(n in sys.modules for n in lazy), [n for n in lazy if n in sys.modules]
            print("IMPORT_GRAPH_OK")
        """)
        for env in ({}, {K.ENV_VAR: "0"}):
            r = _subprocess(script, env)
            self.assertEqual(r.returncode, 0, f"{env}: {r.stdout}\n{r.stderr}")
            self.assertIn("IMPORT_GRAPH_OK", r.stdout)

    # ---------------------------------------------------------------- T2
    def test_bootstrap_plumbing(self):
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        p = inspect.signature(eb.apply_env_bootstrap).parameters
        self.assertIn(K.ENV_VAR, p)
        self.assertIsNone(p[K.ENV_VAR].default)
        src = inspect.getsource(eb)
        self.assertIn(f'_bind_bool("{K.ENV_VAR}", {K.ENV_VAR})', src)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(K.ENV_VAR, None)
            eb._bind_bool(K.ENV_VAR, None)  # absent binding: off in code, nothing set
            self.assertNotIn(K.ENV_VAR, os.environ)
            eb._bind_bool(K.ENV_VAR, True)
            self.assertEqual(os.environ[K.ENV_VAR], "1")
            os.environ[K.ENV_VAR] = "0"  # a pre-set env var wins over a binding (the kill switch)
            eb._bind_bool(K.ENV_VAR, True)
            self.assertEqual(os.environ[K.ENV_VAR], "0")
        sh = os.path.join(_repo_root(), "scripts", "launch_slurm.sh")
        if os.path.exists(sh):  # the launcher forwards every non-empty DLRM_HSTU_FP8_* variable
            with open(sh, encoding="utf-8") as f:
                self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"', f.read())

    # ---------------------------------------------------------------- T3
    def test_attr_literals_agree(self):
        self.assertEqual(ixp.IX_ATTR, "_dlrmv4_ec_indexed_l1")
        for mod in (fln, pfl, pgt):
            self.assertEqual(mod._IX_ATTR, ixp.IX_ATTR, mod.__name__)
        self.assertNotEqual(ixp.IX_ATTR, xs.PORT_ATTR)
        self.assertEqual((ebo._IX_ENV, K.ENV_VAR), (K.ENV_VAR, "DLRM_HSTU_FP8_EC_INDEXED_L1"))

    # ---------------------------------------------------------------- T4
    def test_chain_registration_and_plan_gates(self):
        reg = dict(fuse_linear_norm=True, fuse_ln_residual=True, bf16_training=True, transducer_bf16_dtype=True,
                   uih_post_id="item_id")
        os.environ["DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL"] = "1"
        ebo.note_ec_indexed_l1_chain(**reg)
        self.assertEqual((ebo._IX_CHAIN, ebo._IX_UIH_POST_ID), ("", "item_id"))
        ebo.note_ec_indexed_l1_chain(**dict(reg, fuse_linear_norm=False))
        self.assertEqual(ebo._IX_CHAIN, "fuse_linear_norm off")
        ebo._MERGE_PAIRS, ebo._CAT_GROUP = [p for p in PAIRS if p[0] != "item_id"], ["artist_id"]
        os.environ.pop("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL")
        ebo.note_ec_indexed_l1_chain(**reg)
        for piece in ("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", "ec_merge_uih_candidate (item_id)",
                      "DLRM_HSTU_FP8_EC_CAT_FUSE (2-key group)"):
            self.assertIn(piece, ebo._IX_CHAIN)
        ebo._MERGE_PAIRS, ebo._CAT_GROUP = list(PAIRS), list(CAT_GROUP)

        def gate(names=CAT_FIRST, chain="", uih="item_id", **kw):
            _arm(True)
            ebo._IX_CHAIN, ebo._IX_UIH_POST_ID = chain, uih
            n_said, n_info = len(SAID), len(INFO)
            v, big, _ = _forward_ec(names, kept=kw.get("kept"))
            loud = [s for s in SAID[n_said:] if s.startswith(K.TAG)]
            quiet = [s for s in INFO[n_info:] if s.startswith(K.TAG) and "not eligible" in s]
            return v, big, loud, quiet

        # loud: unregistered (before any structural check), incapable chain
        ebo._CAT_GROUP = []
        v, big, loud, quiet = gate(chain=None)
        self.assertEqual(len(loud), 1, loud)
        self.assertIn("declined at ec: consumer chain not registered", loud[0])
        self.assertEqual(quiet, [])
        ebo._CAT_GROUP = list(CAT_GROUP)
        v, big, loud, quiet = gate(chain="fuse_linear_norm off")
        self.assertEqual(len(loud), 1, loud)
        self.assertIn("consumer chain is not Indexed-X-aware (fuse_linear_norm off)", loud[0])
        self.assertEqual(self._issued(), 0)
        # INFO only (no census token, nothing issued): no cat member; 3 / 1 hoisted members; unmerged member;
        # premerged member with another name
        ebo._CAT_GROUP = []
        v, big, loud, quiet = gate()
        self.assertEqual((loud, len(quiet)), ([], 1), (loud, quiet))
        self.assertIn("no CAT_FUSE artist|album member", quiet[0])
        ebo._CAT_GROUP = list(CAT_GROUP)
        for why, kw, setup in (
                ("3 hoisted members", dict(kept=["artist_cand", "album_cand", "item_cand"]), None),
                ("1 hoisted members", dict(kept=["uid", "artist_cand", "album_cand", "item_id", "item_cand"]),
                 [p for p in PAIRS if p[0] != "item_id"]),
                ("'item_id' premerged=False", dict(), [p for p in PAIRS if p[0] != "item_id"]),
                ("'item_id' premerged=True, need the merged 'other_id'", dict(uih="other_id"), None)):
            ebo._MERGE_PAIRS = list(PAIRS) if setup is None else setup
            for names in (CAT_FIRST, ITEM_FIRST):
                v, big, loud, quiet = gate(names, **kw)
                self.assertEqual(loud, [], (why, loud))
                self.assertEqual(len(quiet), 1, (why, quiet))
                self.assertIn(why, quiet[0])
                self.assertEqual(self._issued(), 0, why)
                self.assertIsNone(getattr(v, ixp.IX_ATTR, None), why)
            ebo._MERGE_PAIRS = list(PAIRS)
        for s in INFO:
            self.assertFalse(ARMED_I.search(s) or DECL_I.search(s), s)
        # G7: unequal member rows (the grouped tail would decline) -> loud, nothing issued
        _arm(True)
        ebo._IX_CHAIN, ebo._IX_UIH_POST_ID = "", "item_id"
        n_said = len(SAID)
        ri, feats, lpk, offs, emb0, _ = _inputs(CAT_FIRST, item_lens=[6, 8, 4, 10])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ret = ebo._merged_construct(emb0.clone().requires_grad_(True), feats, CAT_FIRST, ri, torch.bfloat16,
                                        lpk, offs, KEPT)
        loud = SAID[n_said:]
        self.assertEqual(len(loud), 1, loud)
        self.assertIn(f"declined at ec: cat rows {M} != item rows {M + B}", loud[0])
        self.assertIsNone(getattr(ret["item_id"].values(), ixp.IX_ATTR, None))
        self.assertEqual(self._issued(), 0)
        # mapping: both member orders engage with the right slots
        for names, (cs, it) in ((CAT_FIRST, (0, 1)), (ITEM_FIRST, (1, 0))):
            _arm(True)
            ebo._IX_CHAIN, ebo._IX_UIH_POST_ID = "", "item_id"
            v, big, _ = _forward_ec(names)
            rv, rc = getattr(v, ixp.IX_ATTR), getattr(big, ixp.IX_ATTR)
            self.assertEqual((rc.kind, rc.member, rc.rows, rc.nh), ("cat", cs, M, 2), names[0])
            self.assertEqual((rv.kind, rv.member, rv.rows, rv.nh), ("item", it, M, 1), names[0])

    # ---------------------------------------------------------------- T5
    def test_knob_off_and_kill_switch_inert(self):
        calls = []
        real_plan = ixp.plan
        self._set(ixp, "plan", lambda *a, **k: (calls.append(1), real_plan(*a, **k))[1])
        for ex2b in (False, True):
            for names in (CAT_FIRST, ITEM_FIRST):
                ref = None
                for knob in (False, None):
                    r = run(knob, names, ex2b=ex2b)
                    self.assertEqual(r["ec_node"], "ECExpandGroupPortsBackward" if ex2b else "_ECExpandGroupBackward")
                    self.assertNoIX(r)
                    ec = r["values"].grad_fn
                    self.assertTrue(all(type(c) is bool for c in ec.casts), ec.casts)
                    self.assertFalse(hasattr(ec, "ix_ledgers"))
                    for y in r["ys"].values():
                        self.assertIsNone(y.grad_fn.ix)  # pair 0 got a plain tensor
                    self.assertEqual(CALLS["fwd"] + CALLS["wgrad"], 0)
                    self.assertEqual([s for s in r["said"] if s.startswith(K.TAG)], [])
                    if ref is None:
                        ref = r
                    else:
                        self.assertSameGrads(r, ref)
        self.assertEqual(calls, [])
        self.assertIs(ebo._IX_ON, False)

    # ---------------------------------------------------------------- T6
    def test_fp8_head_bitwise_vs_knob_off(self):
        for ex2b in (False, True):
            for names in (CAT_FIRST, ITEM_FIRST):
                what = f"ex2b={ex2b} {names[0]}"
                d_off = _XDetect()
                off = run(False, names, ex2b=ex2b, detect=d_off)
                self.assertTrue(any(p == "fwd" for p, _, _ in d_off.hits), "the X detector sees the shipped X")
                d_on = _XDetect()
                on = run(True, names, ex2b=ex2b, detect=d_on)  # _arm resets CALLS
                self.assertSameGrads(on, off)
                self.assertEqual(on["ec_node"], "ECExpandGroupPortsBackward" if ex2b else "_ECExpandGroupBackward")
                # placeholders and tags
                v, big = on["values"], on["big"]
                self.assertPlaceholder(v, "values")
                self.assertPlaceholder(big, "big")
                self.assertIsNot(getattr(v, ixp.IX_ATTR, None), None)
                self.assertIsNot(getattr(big, ixp.IX_ATTR, None), None)
                self.assertIsNone(getattr(big._base, ixp.IX_ATTR, None))
                for n in ("artist_id", "album_id"):
                    self.assertIsNone(getattr(on["ret"][n].values(), ixp.IX_ATTR, None))
                for kind, x in (("item", v), ("cat", big)):
                    rec = getattr(x, ixp.IX_ATTR)
                    node = on["ys"][kind].grad_fn
                    self.assertTrue(node.ix.indexed, what)
                    saved = on["saved"][kind]
                    self.assertEqual(len(saved), 9)
                    self.assertEqual(len(off["saved"][kind]), 8)
                    self.assertIs(saved[0], rec.emb_c, what)
                    self.assertIs(saved[8], rec.idx, what)
                self.assertEqual((CALLS["fwd"], CALLS["wgrad"]), (2, 2), what)
                # no X: not in the EC + FP8-head forward, and with E-X2b on not in the backward either
                self.assertEqual([h for h in d_on.hits if h[0] == "fwd"], [], what)
                if ex2b:
                    self.assertEqual(d_on.hits, [], what)
                    self.assertEqual(CALLS["fused"], 1, what)
                # census
                text = "\n".join(on["said"])
                self.assertIn(f"{K.ARMED_PREFIX} fwd", text)
                self.assertIn(f"{K.ARMED_PREFIX} wgrad", text)
                self.assertIn(f"{K.TAG} invariants OK", text)
                self.assertNotIn("declined", text)
                self.assertNotIn("FAILED", text)
                self.assertFalse([s for s in on["xs_said"] if "DECLINED" in s or "FALLBACK" in s], on["xs_said"])
        # storage audit over the EC + both heads, forward and backward
        for ex2b in (False, True):
            au = ixp.StorageAudit()
            on = run(True, ex2b=ex2b, audit=au)
            self.assertEqual(au.violations, [], ex2b)
            self.assertEqual(len(au.watched), 2)
        self.assertEqual(ixp._AUDITS, [])
        # positive controls: each IX-unaware reader is flagged, with a finite loss
        for extra in ("masked_reader", "nan_to_num_reader", "colview_reader", "bwd_hook_reader"):
            au = ixp.StorageAudit()
            r = run(True, extra=extra, audit=au)
            self.assertGreaterEqual(len(au.violations), 1, extra)
            self.assertTrue(bool(torch.isfinite(r["loss"])), extra)
        # the debug check (DLRM_HSTU_FP8_EC_INDEXED_L1_CHECK=1) logs only
        off = run(False)
        os.environ[ixp.CHECK_ENV_VAR] = "1"
        on = run(True)
        self.assertSameGrads(on, off)
        checks = [s for s in on["said"] if " check " in s]
        self.assertEqual(len(checks), 4, checks)
        self.assertTrue(all(" 0 of " in s for s in checks), checks)

    # ---------------------------------------------------------------- T7
    def test_rebuild_paths_bitwise_and_hooks(self):
        for ex2b in (False, True):
            for path in ("run_head", "fln"):
                for names in (CAT_FIRST, ITEM_FIRST):
                    off = run(False, names, ex2b=ex2b, item_path=path, cat_path=path)
                    on = run(True, names, ex2b=ex2b, item_path=path, cat_path=path)
                    self.assertSameGrads(on, off)
                    self.assertEqual(CALLS["fwd"] + CALLS["wgrad"], 0)
                    site = "_run_head" if path == "run_head" else "FusedLinearNormMLP"
                    self.assertEqual((ixp.STATS[f"rebuilt_{site}_item"], ixp.STATS[f"rebuilt_{site}_cat"]), (1, 1))
                    if path == "run_head":
                        lines = [s for s in on["info"] if "X rebuilt at _run_head" in s]
                        self.assertEqual(len(lines), 2, on["info"])  # item and cat, INFO
                        for s in lines:
                            self.assertFalse(ARMED_I.search(s) or DECL_I.search(s), s)
                        self.assertEqual([s for s in on["said"] if s.startswith(K.TAG)], [])
                    else:
                        lines = [s for s in on["said"] if "declined at FusedLinearNormMLP" in s]
                        self.assertEqual(len(lines), 2, on["said"])
                        self.assertTrue(all(DECL_RE.search(s) for s in lines))
                    if ex2b:  # claim is silent on the rebuilt X
                        self.assertFalse([s for s in on["xs_said"] if "DECLINED" in s or "FALLBACK" in s],
                                         on["xs_said"])
        # graph: the cat rebuild's input is `big` (ViewBackward0 of the EC output `flat`)
        _arm(True)
        v, big, _ = _forward_ec()
        xr = ixp.rebuild(big, "_run_head")
        gf = xr.grad_fn
        self.assertEqual(type(gf).__name__, "_IXMaterializeBackward")
        self.assertIs(gf.next_functions[0][0], big.grad_fn)
        self.assertEqual(type(big.grad_fn).__name__, "ViewBackward0")
        self.assertEqual(tuple(xr.shape), (M, 2 * W))
        self.assertIsNotNone(xr._base)
        self.assertNotEqual(xr.untyped_storage()._cdata, big.untyped_storage()._cdata)
        with self.assertRaises(RuntimeError):  # only the tagged object itself may be rebuilt
            ixp.rebuild(_TaggedCopy.of(big), "_run_head")
        # hooks / retain_grad on values and big: fire once with the shipped shapes, .grad raw-bit vs knob off
        for ex2b in (False, True):
            for path in ("head", "run_head", "fln"):
                off = run(False, ex2b=ex2b, item_path=path, cat_path=path, hooks=True)
                on = run(True, ex2b=ex2b, item_path=path, cat_path=path, hooks=True)
                want = {"values": [(M, W)], "big": [(M, 2 * W)]}
                self.assertEqual(off["fired"], want)
                self.assertEqual(on["fired"], want, (ex2b, path))
                self.assertSameGrads(on, off)
                if ex2b and path == "head":
                    for r in (on, off):
                        self.assertTrue(any("DECLINED at claim" in s and "hooks or retain_grad" in s
                                            for s in r["xs_said"]), r["xs_said"])

    # ---------------------------------------------------------------- T8
    def test_rerun_second_consumer(self):
        """GroupedTail._complete_shipped: every FP8 head already ran (node A, output dropped), then _run_head
        re-runs pair 0 on the same member output (node B, in the loss)."""
        real_ecb, real_chk = xp.ec_backward, ixp.ec_backward_check
        for ex2b in (False, True):
            res = {}
            for knob in (False, True):
                _arm(knob, ex2b)
                seen = dict(ports=[], ledgers=[], fwd=None)

                def ecb(ctx, member_grads, port_grads):
                    seen["ports"].append([g is None for g in port_grads])
                    return real_ecb(ctx, member_grads, port_grads)

                def chk(ledgers, grads):
                    seen["ledgers"].append({led.kind: dict(led.bwd) for _, led in ledgers})
                    return real_chk(ledgers, grads)

                xp.ec_backward, ixp.ec_backward_check = ecb, chk
                try:
                    ri, feats, lpk, offs, emb0, R = _inputs(CAT_FIRST)
                    emb = emb0.clone().requires_grad_(True)
                    mlp_c, mlp_a = _mlps()
                    n_xs = len(XS_SAID)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        ret = ebo._merged_construct(emb, feats, CAT_FIRST, ri, torch.bfloat16, lpk, offs, KEPT)
                        values = ret["item_id"].values()
                        big = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR)[0]
                        for mlp, x, k in ((mlp_c, values, 0), (mlp_a, big, 1)):
                            fp8_head(mlp, x, k)  # node A: its output is dropped, autograd never reaches it
                        loss = (pgt._run_head(mlp_c, values).float() * R["c"]["h"]).sum()  # node B
                        loss = loss + (pgt._run_head(mlp_a, big).float() * R["a"]["h"]).sum()
                        for n in KEPT:
                            loss = loss + (ret[n].values() * R[n]).sum()
                    if knob:
                        seen["fwd"] = (getattr(values, ixp.IX_ATTR).ledger.fwd, getattr(big, ixp.IX_ATTR).ledger.fwd)
                    loss.backward()
                finally:
                    xp.ec_backward, ixp.ec_backward_check = real_ecb, real_chk
                g = {"emb": emb.grad.clone()}
                for tag, m in (("c", mlp_c), ("a", mlp_a)):
                    for pn, p in m.named_parameters():
                        g[f"{tag}.{pn}"] = None if p.grad is None else p.grad.clone()
                res[knob] = dict(grads=g, xs_said=XS_SAID[n_xs:], seen=seen)
            self.assertSameGrads(res[True], res[False])
            seen = res[True]["seen"]
            self.assertEqual(seen["fwd"], (2, 2))  # two Indexed-X-aware consumers in the forward ...
            self.assertEqual(len(seen["ledgers"]), 1)
            for kind in ("item", "cat"):  # ... but exactly one of them in the backward pass: no raise
                self.assertEqual(list(seen["ledgers"][0][kind].values()), [(1, False)], kind)
            if ex2b:
                # IX on: the rebuild hides the second consumer from claim, so E-X2b prints nothing ...
                self.assertFalse([s for s in res[True]["xs_said"] if "DECLINED" in s or "FALLBACK" in s],
                                 res[True]["xs_said"])
                # ... and both of its ports got no gradient (node A never ran): the shipped EC reduce
                self.assertEqual(seen["ports"], [[True, True]])
                # IX off (sanity of the scenario): claim sees the second consumer of the tagged member output
                self.assertTrue(any("DECLINED at claim" in s and "second consumer" in s
                                    for s in res[False]["xs_said"]), res[False]["xs_said"])

    # ---------------------------------------------------------------- T9
    def test_in_function_declines_and_latch(self):
        off = run(False)
        casts = []
        real_casts = pfl._autocast_casts
        self._set(pfl, "_autocast_casts", lambda x, w, b: (casts.append((w, b)), real_casts(x, w, b))[1])
        weight_casts = []
        real_wcasts = pfl._autocast_weight_casts
        self._set(pfl, "_autocast_weight_casts",
                  lambda dev, w, b: (lambda r: (weight_casts.append(r), r)[1])(real_wcasts(dev, w, b)))

        def unsupported(*a, **k):
            raise K.IndexedL1Unsupported("forced")

        def boom(*a, **k):
            raise RuntimeError("forced launch failure")

        # forward wrapper raises IndexedL1Unsupported: in-Function decline, exact, the casts are reused
        self._set(K, "ecix_fwd", unsupported)
        on = run(True)
        self.assertSameGrads(on, off)
        self.assertIn(("fwd item", "class"), ixp._ONCE)
        self.assertTrue(any("declined at fwd (_LinearSwishLNEmitFunction, item)" in s for s in on["said"]))
        for kind in ("item", "cat"):
            self.assertFalse(on["ys"][kind].grad_fn.ix.indexed)
            self.assertEqual(len(on["saved"][kind]), 8)
        self.assertIsNone(K._BROKEN)
        self.assertEqual(len(weight_casts), 2)
        for (w, b), (wc, bc) in zip(casts[-2:], weight_casts):
            self.assertIs(w, wc)
            self.assertIs(b, bc)
        for kind, (wc, bc) in zip(("item", "cat"), weight_casts):
            self.assertIs(on["saved"][kind][1], wc)  # lin_w_c is w_c: no second cast
        # forward raises anything else: latch, this call exact, the next grouped call issues nothing
        self._set(K, "ecix_fwd", boom)
        on = run(True)
        self.assertSameGrads(on, off)
        self.assertIn("forced launch failure", K._BROKEN or "")
        self.assertTrue(any("FAILED CLOSED (DISABLED for the rest of the run)" in s for s in on["said"]))
        v, big, _ = _forward_ec()
        self.assertIsNone(getattr(v, ixp.IX_ATTR, None))
        self.assertEqual(type(v.grad_fn).__name__, "_ECExpandGroupBackward")
        self.assertFalse(hasattr(v.grad_fn, "ix_ledgers"))
        self._set(K, "ecix_fwd", _ecix_fwd_ref)
        # wgrad raises (both kinds): exact
        for exc, broken in ((unsupported, False), (boom, True)):
            self._set(K, "ecix_wgrad", exc)
            on = run(True)
            self.assertSameGrads(on, off)
            self.assertEqual(K._BROKEN is not None, broken)
            self.assertTrue(any("declined at wgrad (item)" in s for s in on["said"]) or broken)
        self._set(K, "ecix_wgrad", _ecix_wgrad_ref)
        # latched between the forward and the backward: the wgrad rebuilds X, exact
        _arm(True)
        r = run(True, mode="none", arm=False)
        K.latch_off("test", RuntimeError("latched between forward and backward"))
        r["loss"].backward()
        on_grads = {"emb": r["emb_t"].grad}
        for tag, m in zip(("c", "a"), r["mlps"]):
            for pn, p in m.named_parameters():
                on_grads[f"{tag}.{pn}"] = p.grad
        self.assertSameGrads(dict(grads=on_grads), off)
        self.assertEqual(CALLS["wgrad"], 0)
        self.assertEqual(ixp.STATS["rebuilt_wgrad_item"] + ixp.STATS["rebuilt_wgrad_cat"], 2)
        # W1 parameter dtype gate: bf16 master W1 -> the forward declines, no indexed wgrad, exact
        off16 = run(False, wdtype=torch.bfloat16)
        on16 = run(True, wdtype=torch.bfloat16)
        self.assertSameGrads(on16, off16)
        self.assertIn(("fwd item", "wgrad"), ixp._ONCE)
        self.assertEqual((CALLS["fwd"], CALLS["wgrad"]), (0, 0))
        self.assertTrue(any("declined at fwd (_LinearSwishLNEmitFunction, item): W1 parameter dtype" in s
                            for s in on16["said"]))

    # ---------------------------------------------------------------- T10
    def test_ec_forward_fail_closed(self):
        off = run(False)
        calls = []
        real = ixp._placeholder

        def fail_first(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("forced placeholder failure")
            return real(*a, **k)

        self._set(ixp, "_placeholder", fail_first)
        on = run(True)
        self.assertEqual(len(calls), 1)  # the second member saw the latch in issue() and expanded
        self.assertSameGrads(on, off)
        self.assertNoIX(on)
        self.assertEqual(self._issued(), 0)
        self.assertFalse(hasattr(on["values"].grad_fn, "ix_ledgers"))
        self.assertTrue(all(type(c) is bool for c in on["values"].grad_fn.casts))
        self.assertIn("forced placeholder failure", K._BROKEN or "")
        self.assertTrue(any("FAILED CLOSED" in s and "ec forward" in s for s in on["said"]))

    # ---------------------------------------------------------------- T11
    def test_plan_fail_closed(self):
        off = run(False)

        def boom(*a, **k):
            raise RuntimeError("forced plan failure")

        self._set(ixp, "plan", boom)
        on = run(True)
        self.assertSameGrads(on, off)
        self.assertNoIX(on)
        self.assertIn("ec plan", K._BROKEN or "")
        self.assertTrue(any("FAILED CLOSED (DISABLED for the rest of the run)" in s for s in on["said"]))
        self.assertEqual(on["ec_node"], "_ECExpandGroupBackward")

    # ---------------------------------------------------------------- T12
    def test_backward_check(self):
        for ex2b in (False, True):
            # an IX-unaware consumer only: the EC backward raises
            with self.assertRaises(RuntimeError) as cm:
                run(True, ex2b=ex2b, item_path="linear")
            self.assertIn(f"{K.TAG} INVARIANT (item member)", str(cm.exception))
            # an unaware reader NEXT TO the indexed head: no raise, NaN loss (the storage audit flags it, T6)
            au = ixp.StorageAudit()
            r = run(True, ex2b=ex2b, extra="plain_reader", audit=au)
            self.assertFalse(bool(torch.isfinite(r["loss"])))
            self.assertGreaterEqual(len(au.violations), 1)
            # two Indexed-X-aware consumers (indexed head + rebuild): exact, one census line, no raise
            off = run(False, ex2b=ex2b, extra="second_aware")
            on = run(True, ex2b=ex2b, extra="second_aware")
            self.assertSameGrads(on, off)
            lines = [s for s in on["said"] if "declined the single-consumer check" in s]
            self.assertEqual(len(lines), 1, on["said"])
            self.assertTrue(DECL_RE.search(lines[0]))
            # retain_graph twice: one ledger entry per pass, grads 2x knob off
            snaps = []
            real_chk = ixp.ec_backward_check

            def chk(ledgers, grads, real=real_chk):
                snaps.append({led.kind: dict(led.bwd) for _, led in ledgers})
                return real(ledgers, grads)

            self._set(ixp, "ec_backward_check", chk)
            off = run(False, ex2b=ex2b, mode="twice")
            on = run(True, ex2b=ex2b, mode="twice")
            self.assertSameGrads(on, off)
            self.assertEqual(len(snaps), 2)
            tasks = set()
            for snap in snaps:
                for kind in ("item", "cat"):
                    self.assertEqual(len(snap[kind]), 1, snap)
                    (t, (n, via_port)), = snap[kind].items()
                    tasks.add(t)
                    self.assertEqual((n, via_port), (1, ex2b), snap)
            self.assertEqual(len(tasks), 2)  # a distinct graph task per pass
            if ex2b:
                self.assertEqual(on["acc_after"], [None, None])  # E-X2b never keeps [U, 512] across passes
            ixp.ec_backward_check = real_chk
            # autograd.grad(loss, [values]): the EC node does not run; its ledger entry stays (dies with the step)
            off = run(False, ex2b=ex2b, mode="grad_x")
            on = run(True, ex2b=ex2b, mode="grad_x")
            self.assertBits(on["grad_x"], off["grad_x"], "autograd.grad(loss, [values])")
            led = getattr(on["values"], ixp.IX_ATTR).ledger
            self.assertEqual(len(led.bwd), 1)
            if ex2b:
                self.assertTrue(any("DECLINED at route" in s and "the EC backward does not run" in s
                                    for s in on["xs_said"]), on["xs_said"])
            self.assertSameGrads(run(True, ex2b=ex2b), run(False, ex2b=ex2b))  # then a fresh step works
            # backward(inputs=[W1]): exact
            off = run(False, ex2b=ex2b, mode="inputs_w")
            on = run(True, ex2b=ex2b, mode="inputs_w")
            self.assertSameGrads(on, off)
            # create_graph: the wgrad declines to the shipped GEMM; first-order grads exact
            off = run(False, ex2b=ex2b, mode="create_graph")
            on = run(True, ex2b=ex2b, mode="create_graph")
            self.assertSameGrads(on, off)
            for kind in ("item", "cat"):
                lines = [s for s in on["said"] if f"declined at wgrad ({kind}): create_graph backward" in s]
                self.assertEqual(len(lines), 1, on["said"])
            self.assertEqual(CALLS["wgrad"], 0)
            if ex2b:
                self.assertTrue(any("DECLINED at route" in s and "create_graph backward" in s
                                    for s in on["xs_said"]), on["xs_said"])

    # ---------------------------------------------------------------- T13
    def test_eval_no_grad_untouched(self):
        outs = {}
        for knob in (False, True):
            _arm(knob)
            with torch.no_grad():
                v, big, ret = _forward_ec()
            outs[knob] = (v, big, {n: ret[n].values() for n in KEPT})
            self.assertIsNone(getattr(v, ixp.IX_ATTR, None))
            self.assertIsNone(getattr(big, ixp.IX_ATTR, None))
        self.assertEqual(self._issued(), 0)
        self.assertEqual(ixp._LOGGED, set())
        (v0, b0, k0), (v1, b1, k1) = outs[False], outs[True]
        self.assertBits(v1, v0, "values")
        self.assertBits(b1, b0, "big")
        for n in KEPT:
            self.assertBits(k1[n], k0[n], n)

    # ---------------------------------------------------------------- T14
    def test_lifetimes_no_cycles(self):
        def forward_only():
            _arm(True)
            ri, feats, lpk, offs, emb0, R = _inputs(CAT_FIRST)
            emb = emb0.clone().requires_grad_(True)
            mlp_c, mlp_a = _mlps()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                ret = ebo._merged_construct(emb, feats, CAT_FIRST, ri, torch.bfloat16, lpk, offs, KEPT)
                values = ret["item_id"].values()
                big = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR)[0]
                rec_i, rec_c = getattr(values, ixp.IX_ATTR), getattr(big, ixp.IX_ATTR)
                self.assertIs(rec_i.emb_c, rec_c.emb_c)
                ec = values.grad_fn
                self.assertTrue(all(type(c) is bool for c in ec.casts))
                self.assertTrue(all(type(m) is int and isinstance(led, ixp.Ledger) for m, led in ec.ix_ledgers))
                self.assertFalse(any(r is big._base for r in gc.get_referents(rec_c)))  # no reference to `flat`
                self.assertFalse(hasattr(rec_c, "base_ref"))
                loss = _Sink.apply(fp8_head(mlp_c, values, 0), R["c"]["h"])
                loss = loss + _Sink.apply(fp8_head(mlp_a, big, 1), R["a"]["h"])
                for n in KEPT:
                    loss = loss + (ret[n].values() * R[n]).sum()
            refs = dict(rec_i=weakref.ref(rec_i), rec_c=weakref.ref(rec_c), emb_c=weakref.ref(rec_i.emb_c),
                        led_i=weakref.ref(rec_i.ledger), led_c=weakref.ref(rec_c.ledger), values=weakref.ref(values))
            return loss, refs

        gc.collect()
        gc.disable()
        try:
            loss, refs = forward_only()  # every local (records, placeholders, the JaggedTensor dict) is gone
            self.assertIsNone(refs["rec_i"]())
            self.assertIsNone(refs["rec_c"]())
            self.assertIsNone(refs["values"]())
            self.assertIsNotNone(refs["emb_c"]())  # held by the two FP8 heads' saved tensors only
            self.assertIsNotNone(refs["led_i"]())  # the EC ctx and the heads' IXCall hold the ledgers
            loss.backward()
            self.assertIsNone(refs["emb_c"]())  # freed by the heads' backward
            del loss
            self.assertEqual({k: r() is None for k, r in refs.items()}, {k: True for k in refs})
        finally:
            gc.enable()

    # ---------------------------------------------------------------- T15
    def test_prebuild_hook_cpu(self):
        script = textwrap.dedent(r"""
            import sys
            import torch
            from generative_recommenders.dlrm_v4.train import blas_warmup as bw
            rng = torch.random.get_rng_state()
            bw._prebuild_ec_indexed_l1(torch.device("cpu"))
            assert torch.equal(rng, torch.random.get_rng_state())
            k = 'generative_recommenders.ops.triton.triton_ec_indexed_l1'
            assert k not in sys.modules, k
            assert 'generative_recommenders.ops.ec_indexed_l1' not in sys.modules
            print("PREBUILD_CPU_OK")
        """)
        r = _subprocess(script, {K.ENV_VAR: "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PREBUILD_CPU_OK", r.stdout)

    # ---------------------------------------------------------------- T16
    def test_activation_checkpointing(self):
        try:
            from torch.utils.checkpoint import checkpoint
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"torch.utils.checkpoint unavailable: {exc}")

        def ck_run(knob, reentrant):
            _arm(knob)
            ri, feats, lpk, offs, emb0, R = _inputs(CAT_FIRST)
            emb = emb0.clone().requires_grad_(True)
            mlp_c, mlp_a = _mlps()

            def region(e):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    ret = ebo._merged_construct(e, feats, CAT_FIRST, ri, torch.bfloat16, lpk, offs, KEPT)
                    values = ret["item_id"].values()
                    big = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR)[0]
                    loss = _Sink.apply(fp8_head(mlp_c, values, 0), R["c"]["h"])
                    loss = loss + _Sink.apply(fp8_head(mlp_a, big, 1), R["a"]["h"])
                    for n in KEPT:
                        loss = loss + (ret[n].values() * R[n]).sum()
                return loss

            n_said = len(SAID)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint(region, emb, use_reentrant=reentrant).backward()
            g = {"emb": emb.grad.clone()}
            for tag, m in (("c", mlp_c), ("a", mlp_a)):
                for pn, p in m.named_parameters():
                    g[f"{tag}.{pn}"] = None if p.grad is None else p.grad.clone()
            return dict(grads=g, said=SAID[n_said:])

        for reentrant in (False, True):
            off = ck_run(False, reentrant)
            on = ck_run(True, reentrant)
            self.assertSameGrads(on, off)
            self.assertEqual(self._issued(), 0)
            hooks = [s for s in on["said"] if "declined at ec: saved-tensor hooks are active" in s]
            self.assertEqual(len(hooks), 0 if reentrant else 1, on["said"])

    # ---------------------------------------------------------------- T17
    def test_census_tokens(self):
        loud, info = [], []
        n_said, n_info = len(SAID), len(INFO)
        # a full indexed step (arming lines, the placeholders INFO line) and the rebuild INFO line
        run(True)
        run(True, item_path="run_head", cat_path="run_head")
        run(True, item_path="fln", cat_path="fln")
        run(True, extra="second_aware")
        # plan declines through the real EC: INFO (not the Indexed-X call) and loud (configuration / layout)
        ebo._CAT_GROUP = []
        _arm(True)
        _forward_ec()
        ebo._CAT_GROUP = list(CAT_GROUP)
        _arm(True)
        _forward_ec(kept=["artist_cand", "album_cand", "item_cand"])
        for chain in (None, "fuse_linear_norm off"):
            _arm(True)
            ebo._IX_CHAIN = chain
            _forward_ec()
        ebo._IX_CHAIN = ""
        for attr, value in (("_ENABLED", False), ("_MIN_ROWS", 1 << 16)):
            _arm(True)
            old = getattr(pfl, attr)
            setattr(pfl, attr, value)
            try:
                _forward_ec()
            finally:
                setattr(pfl, attr, old)
        # in-Function declines, the latch and the IGNORED line (Appendix C wording)
        self._set(K, "ecix_fwd", lambda *a, **k: (_ for _ in ()).throw(K.IndexedL1Unsupported("forced")))
        run(True)
        self._set(K, "ecix_fwd", _ecix_fwd_ref)
        run(True, mode="create_graph")
        K.latch_off("census test", RuntimeError("x"))
        self._set(K, "_ENABLED", None)
        self._set(K, "_KERNELS_ERROR", "census test")
        self.assertFalse(K.enabled())
        loud, info = SAID[n_said:], [s for s in INFO[n_info:] if s.startswith(K.TAG)]
        self.assertTrue(info)
        for s in info:  # INFO lines carry no census token, even case-insensitively
            self.assertFalse(ARMED_I.search(s) or DECL_I.search(s), s)
        decl = [s for s in loud if "declined" in s]
        self.assertGreaterEqual(len(decl), 6, loud)
        for s in decl:
            self.assertTrue(DECL_RE.search(s), s)
        for key in ("FAILED CLOSED (DISABLED", "IGNORED (NOT armed)"):
            lines = [s for s in loud if key in s]
            self.assertTrue(lines, key)
            self.assertTrue(all(DECL_RE.search(s) for s in lines), lines)
        armed = [s for s in loud if s.startswith(K.ARMED_PREFIX)]
        self.assertEqual(sorted({s.split(" -- ")[0] for s in armed}),
                         [f"{K.ARMED_PREFIX} fwd", f"{K.ARMED_PREFIX} wgrad"])
        for s in armed:
            self.assertTrue(ARMED_RE.search(s) and not DECL_RE.search(s), s)
        # the prebuild and invariants lines carry no ARMED token: the A/B greps them separately
        inv = [s for s in loud if "invariants OK" in s]
        self.assertTrue(inv)
        self.assertFalse(any(ARMED_RE.search(s) or DECL_RE.search(s) for s in inv))


class _TaggedCopy:
    @staticmethod
    def of(x):
        """A different tensor object carrying the same tag (e.g. a copy that kept the attribute)."""
        y = x.detach().clone()
        setattr(y, ixp.IX_ATTR, getattr(x, ixp.IX_ATTR))
        return y


if __name__ == "__main__":
    unittest.main()
