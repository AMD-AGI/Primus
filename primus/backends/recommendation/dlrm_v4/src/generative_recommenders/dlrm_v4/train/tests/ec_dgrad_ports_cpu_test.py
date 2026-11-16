"""E-X2b dh ports: the autograd protocol on CPU (no GPU, no Triton needed).

Drives the real ec_bf16_output._merged_construct (hook 1), the real
ECExpandGroupPorts and shipped _ECExpandGroup.backward (hook 2) and the real
claim / bind / route of ops/triton/triton_ec_dgrad_segreduce.py. A toy pair 0
(_Pair0) carries _LinearNormFunction's port protocol around torch math. The
GPU-only pieces are replaced by torch references with the shipped arithmetic:
the segment reduce (fp32 index_add_ of the incoming bf16 rows), the segment
build, and the fused kernel (fp32 index_add_ of torch.mm addends). Hence the
fused path must give the knob-off gradients BIT FOR BIT here; on the GPU the
kernel is a declared fp32-order change (ec_dgrad_segreduce GPU test).
"""

import contextlib
import gc
import io
import os
import subprocess
import sys
import textwrap
import types
import unittest
import weakref

import torch

try:
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
    from generative_recommenders.dlrm_v4.train import ec_dgrad_ports as xp
    from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs
except ImportError as exc:  # e.g. gin missing
    raise unittest.SkipTest(f"E-X2b CPU test needs ec_bf16_output: {exc}")

from torch.utils._python_dispatch import TorchDispatchMode

W, H = 512, 256
HO = 128  # toy pair-0 output width (!= H: the zero-gradient detector keys on [M, H])
CAT_FIRST = ["artist_id", "album_id", "item_id", "uid", "artist_cand", "album_cand", "item_cand"]
ITEM_FIRST = ["item_id", "artist_id", "album_id", "uid", "artist_cand", "album_cand", "item_cand"]
PAIRS = [("item_id", "item_cand"), ("artist_id", "artist_cand"), ("album_id", "album_cand")]
UIH = [5, 7, 3, 9]
B = len(UIH)
M = sum(UIH) + B  # merged member rows
# disjoint tables: item 0-15, artist 16-27, album 28-39, uid 40-43; 44-47 unused
TABLE = {"item_id": (0, 16), "artist_id": (16, 28), "album_id": (28, 40), "uid": (40, 44),
         "item_cand": (0, 16), "artist_cand": (16, 28), "album_cand": (28, 40)}
U = 48
# ec_bf16_output's log-once flags this test flips (restored in tearDown).
_EBO_FLAGS = ("_CAT_LOGGED", "_MERGE_LOGGED", "_EC_GROUP_LOGGED", "_SEG_ARMED_LOGGED")
_ESR = "generative_recommenders.dlrm_v4.train.ec_segment_reduce"
_GF = "generative_recommenders.ops.glue_fusion"
_JT = "torchrec.sparse.jagged_tensor"


# ---------------------------------------------------------------- references
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


CALLS = []


def _fused_ref(cat, item, num_rows, emit=None):
    """The kernel's contract, checked, with the shipped arithmetic."""
    for m, kind in ((cat, "cat"), (item, "item")):
        xs._check_member(m, kind, m.dh.device, num_rows)
        assert torch.equal(m.idx[m.order], torch.sort(m.idx, stable=True).values)
        assert torch.equal(m.order, _build_segments_ref(m.idx, num_rows)[0])
    CALLS.append((cat.idx.numel(), item.idx.numel(), num_rows))
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


# ---------------------------------------------------------------- toy pair 0
class _Pair0(torch.autograd.Function):
    """_LinearNormFunction's E-X2b protocol around torch math:
    y = (x W^T + b) P, so dh = dy P^T. claim at the call site (pair0 below),
    bind in forward, route in backward, outputs sliced to the inputs."""

    @staticmethod
    def forward(ctx, x, lin_w, lin_b, proj, ex2b_rec=None, ex2b_port=None):
        x = x.contiguous()
        x_c, lin_w_c = x.to(torch.bfloat16), lin_w.to(torch.bfloat16)
        h = torch.addmm(lin_b.to(torch.bfloat16), x_c, lin_w_c.t())
        ctx.save_for_backward(x_c, lin_w_c, proj)
        ctx.ex2b = None if ex2b_port is None else xs.bind(ex2b_rec, x_c, lin_w_c)
        ctx.grad_dtypes = (x.dtype, lin_w.dtype, lin_b.dtype)
        return h @ proj

    @staticmethod
    def backward(ctx, dy):
        x, lin_w, proj = ctx.saved_tensors
        dh = (dy @ proj.t()).contiguous()
        needs = ctx.needs_input_grad
        routed = ctx.ex2b is not None and xs.route(ctx, ctx.ex2b, dh, len(needs) - 1)
        dx = dh @ lin_w if needs[0] and not routed else None
        x_dt, w_dt, b_dt = ctx.grad_dtypes
        out = (
            None if dx is None else dx.to(x_dt),
            (dh.t() @ x).to(w_dt) if needs[1] else None,
            dh.float().sum(0).to(b_dt) if needs[2] else None,
            None,
            None,
            dh if routed else None,
        )
        return out[: len(needs)]


SEEN = []


def pair0(x, lin_w, lin_b, proj):
    rec, port = xs.claim(x, lin_w)
    SEEN.append(rec is not None)
    return _Pair0.apply(x, lin_w, lin_b, proj, rec, port)


def _inputs(names, overlap=None):
    g = torch.Generator().manual_seed(7)
    lens = {n: (UIH if n in ("item_id", "artist_id", "album_id") else [1] * B) for n in names}
    ri = []
    for n in names:
        lo, hi = TABLE[n]
        k = torch.randint(lo, hi, (sum(lens[n]),), generator=g)
        if overlap is not None and n == overlap[0]:
            k[0] = overlap[1]
        ri.append(k)
    lengths = torch.tensor([v for n in names for v in lens[n]], dtype=torch.int64)
    lpk = [sum(lens[n]) for n in names]
    offs = [sum(lpk[:i]) for i in range(len(names))]
    emb = torch.randn(U, W, generator=g)
    params = [torch.randn(H, W, generator=g) / 16, torch.randn(H, generator=g),
              torch.randn(H, 2 * W, generator=g) / 16, torch.randn(H, generator=g)]
    proj = [(torch.randn(H, HO, generator=g) / 16).to(torch.bfloat16) for _ in range(2)]
    R = [torch.randn(M, HO, generator=g) for _ in range(2)] + [torch.randn(M, W, generator=g)]
    Rk = {n: torch.randn(B, W, generator=g) for n in names}
    return torch.cat(ri), _Feats(lengths, B), lpk, offs, emb, params, proj, R, Rk


def run(knob, names=CAT_FIRST, *, x_item=None, x_cat=None, hook=None, retain=False,
        extra=False, mode="backward", grad=True, overlap=None, out=None):
    """One EC call + both pair 0s + kept consumers; returns grads (and more)."""
    _arm(knob)
    ri, feats, lpk, offs, emb0, params, proj, R, Rk = _inputs(names, overlap)
    emb = emb0.clone().requires_grad_(True)
    w_c, b_c, w_a, b_a = [p.clone().requires_grad_(True) for p in params]
    kept = ["uid", "artist_cand", "album_cand", "item_cand"]
    caps = []
    with torch.set_grad_enabled(grad):
        ret = ebo._merged_construct(emb, feats, names, ri, torch.bfloat16, lpk, offs, kept)
        values = ret["item_id"].values()
        big = getattr(ret["artist_id"].values(), ebo.CATFUSED_ATTR)[0]
        flat = big._base
        if hook == "before":
            values.register_hook(lambda g: caps.append(g.clone()))
        y_c = pair0(values if x_item is None else x_item(values), w_c, b_c, proj[0])
        y_a = pair0(big if x_cat is None else x_cat(big), w_a, b_a, proj[1])
        loss = (y_c.float() * R[0]).sum() + (y_a.float() * R[1]).sum()
        for n in kept:
            loss = loss + (ret[n].values() * Rk[n]).sum()
        if extra:
            loss = loss + (values.float() * R[2]).sum()
    res = dict(values=values, big=big, kept={n: ret[n].values() for n in kept},
               ec_node=type(values.grad_fn).__name__ if grad else None)
    if out is not None:
        out.update(res)
    if not grad:
        return res
    if hook == "after":
        values.register_hook(lambda g: caps.append(g.clone()))
    if hook == "flat":
        flat.register_hook(lambda g: caps.append(g.clone()))
    if retain:
        values.retain_grad()
    if mode == "backward":
        loss.backward()
    elif mode == "twice":
        loss.backward(retain_graph=True)
        loss.backward()
    elif mode == "create_graph":
        loss.backward(create_graph=True)
    elif mode == "grad_x":
        res["grad_x"] = torch.autograd.grad(loss, [values])[0]
        return res
    elif mode == "inputs_w":
        loss.backward(inputs=[w_c, w_a])
    for k, t in (("emb", emb), ("w_c", w_c), ("b_c", b_c), ("w_a", w_a), ("b_a", b_a)):
        res[k] = None if t.grad is None else t.grad.detach().clone()
    res["caps"] = caps
    res["x_grad"] = values.grad if retain else None
    return res


def _forward_ec(names=CAT_FIRST):
    """One EC call with the current gate state (no re-arming)."""
    ri, feats, lpk, offs, emb0, *_ = _inputs(names)
    emb = emb0.clone().requires_grad_(True)
    ret = ebo._merged_construct(emb, feats, names, ri, torch.bfloat16, lpk, offs,
                                ["uid", "artist_cand", "album_cand", "item_cand"])
    return ret["item_id"].values()


def _arm(knob):
    if knob is None:  # explicit kill switch
        os.environ[xs.ENV_VAR] = "0"
    elif knob:
        os.environ[xs.ENV_VAR] = "1"
    else:
        os.environ.pop(xs.ENV_VAR, None)
    ebo._EX2B_ON = None  # re-read the knob
    xs._ENABLED = bool(knob)  # bypass the ROCm-build requirement on CPU
    xs._BROKEN = None


def _bits(t):
    return t.detach().contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


class _Zeros(TorchDispatchMode):
    """Records zero/fill/expand ops producing an [M, 256] tensor."""

    NAMES = ("zeros", "zeros_like", "new_zeros", "fill_", "full", "full_like", "new_full",
             "expand", "expand_copy")

    def __init__(self):
        super().__init__()
        self.hits = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func.overloadpacket.__name__ in self.NAMES and isinstance(out, torch.Tensor):
            if tuple(out.shape) == (M, H):
                self.hits.append(str(func))
        return out


class ECDgradPortsCPUTest(unittest.TestCase):
    def setUp(self):
        # Everything a test patches is saved first and restored by _restore, which is
        # registered with addCleanup BEFORE the first patch: cleanups also run when
        # setUp fails part way (unittest skips tearDown then), so no patch can leak
        # into later tests.
        self._env = {k: os.environ.get(k) for k in (xs.ENV_VAR, "DLRM_HSTU_FP8_EC_GROUP_BWD")}
        self._saved = dict(
            seg=ebo._SEGMENT_REDUCE, pairs=ebo._MERGE_PAIRS, grp=ebo._CAT_GROUP,
            on=ebo._EX2B_ON, mi=ebo._merged_index, js=ebo._ec_joint_segments,
            en=xs._ENABLED, br=xs._BROKEN, fused=xs.dgrad_segreduce, dev=xp._DEVICES,
        )
        self._flags = {n: getattr(ebo, n) for n in _EBO_FLAGS}
        self._m4 = getattr(ebo, "_ec_reduce_fused", None)
        self._esr = None
        self._stubs = []
        self.addCleanup(self._restore)
        os.environ["DLRM_HSTU_FP8_EC_GROUP_BWD"] = "1"
        xp._DEVICES = ("cuda", "cpu")
        ebo._SEGMENT_REDUCE = True
        ebo._MERGE_PAIRS = list(PAIRS)
        ebo._CAT_GROUP = ["artist_id", "album_id"]
        ebo._merged_index = _merged_index_ref
        ebo._ec_joint_segments = _joint_ref
        # ship13 (rccl M4 (e)): the grouped reduce declines -> the per-member reference.
        if self._m4 is not None:
            ebo._ec_reduce_fused = lambda hoisted, segs, U: (None, segs)
        xs.dgrad_segreduce = _fused_ref
        xs._ONCE.clear()
        xs._BOUND_KINDS.clear()
        xs._PORTS_LOGGED = False
        xp._FUSED_LOGGED = False
        xp._CHECKS_LEFT = 2
        CALLS.clear()
        SEEN.clear()
        # GPU-only modules: patch the real ones, or stand in where Triton is absent.
        try:
            import generative_recommenders.dlrm_v4.train.ec_segment_reduce as esr
        except ImportError:
            esr = types.ModuleType(_ESR)
            sys.modules[_ESR] = esr
            self._stubs.append(_ESR)
        self._esr = (esr, getattr(esr, "segment_reduce_rows", None),
                     getattr(esr, "build_segments", None))
        esr.segment_reduce_rows = _seg_reduce_ref
        esr.build_segments = _build_segments_ref
        try:
            import generative_recommenders.ops.glue_fusion  # noqa: F401
        except ImportError:
            gf = types.ModuleType(_GF)
            gf.merged_index_fusable = lambda *a: False
            sys.modules[_GF] = gf
            self._stubs.append(_GF)
        try:
            import torchrec.sparse.jagged_tensor  # noqa: F401
        except ImportError:
            for name in ("torchrec", "torchrec.sparse", _JT):
                sys.modules[name] = types.ModuleType(name)
                self._stubs.append(name)
            sys.modules[_JT].JaggedTensor = _JaggedTensor

    def _restore(self):
        s = self._saved
        ebo._SEGMENT_REDUCE, ebo._MERGE_PAIRS, ebo._CAT_GROUP = s["seg"], s["pairs"], s["grp"]
        ebo._EX2B_ON, ebo._merged_index, ebo._ec_joint_segments = s["on"], s["mi"], s["js"]
        xs._ENABLED, xs._BROKEN, xs.dgrad_segreduce = s["en"], s["br"], s["fused"]
        xp._DEVICES = s["dev"]
        if self._m4 is not None:
            ebo._ec_reduce_fused = self._m4
        for n, v in self._flags.items():
            setattr(ebo, n, v)
        if self._esr is not None:
            esr, f, b = self._esr
            if f is not None:
                esr.segment_reduce_rows, esr.build_segments = f, b
        for name in self._stubs:
            sys.modules.pop(name, None)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def assertBits(self, a, b, what=""):
        self.assertIsNotNone(a, what)
        self.assertIsNotNone(b, what)
        self.assertEqual(a.shape, b.shape, what)
        self.assertTrue(torch.equal(_bits(a), _bits(b)), f"{what}: not bitwise equal")

    def assertSameGrads(self, on, off, keys=("emb", "w_c", "b_c", "w_a", "b_a")):
        for k in keys:
            self.assertBits(on[k], off[k], k)

    # (b), (k), arming lines, member order
    def test_fused_bitwise_vs_knob_off_both_member_orders(self):
        buf = io.StringIO()
        for names in (CAT_FIRST, ITEM_FIRST):
            off = run(False, names)
            self.assertEqual(off["ec_node"], "_ECExpandGroupBackward")
            with contextlib.redirect_stdout(buf):
                on = run(True, names)
            self.assertEqual(on["ec_node"], "ECExpandGroupPortsBackward")
            self.assertSameGrads(on, off)
            self.assertEqual(CALLS[-1], (2 * M, M, U))
            self.assertEqual(SEEN[-2:], [True, True])
        text = buf.getvalue()
        for line in ("armed OK: [ec-dgrad-segreduce] ports", "armed OK: [ec-dgrad-segreduce] fused",
                     "[ec-dgrad-segreduce] invariants OK"):
            self.assertIn(line, text)
        self.assertNotIn("DECLINED", text)
        self.assertNotIn("FALLBACK", text)

    def test_knob_off_and_kill_switch_are_inert(self):
        for knob in (False, None):
            SEEN.clear()
            r = run(knob)
            self.assertEqual(r["ec_node"], "_ECExpandGroupBackward")
            self.assertFalse(hasattr(r["values"], xs.PORT_ATTR))
            self.assertFalse(hasattr(r["big"], xs.PORT_ATTR))
            self.assertEqual(SEEN, [False, False])
            self.assertEqual(xs.claim(r["values"], torch.zeros(H, W)), (None, None))
        self.assertEqual(CALLS, [])

    # (h)
    def test_no_grad_forward_identical_and_untagged(self):
        off = run(False, grad=False)
        on = run(True, grad=False)
        for k in ("values", "big"):
            self.assertBits(on[k], off[k], k)
            self.assertFalse(hasattr(on[k], xs.PORT_ATTR), k)
        for n, v in on["kept"].items():
            self.assertBits(v, off["kept"][n], n)

    # (c)
    def test_views_and_copies_do_not_claim(self):
        off = run(False)
        on = run(True, x_item=lambda t: t.view(t.shape), x_cat=lambda t: t.clone())
        self.assertEqual(SEEN[-2:], [False, False])
        self.assertSameGrads(on, off)
        self.assertEqual(CALLS, [])

    # (d)
    def test_extra_consumer_falls_back_exactly(self):
        off = run(False, extra=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            on = run(True, extra=True)
        self.assertIn("FALLBACK (a member output has another consumer)", buf.getvalue())
        self.assertSameGrads(on, off)
        self.assertEqual(CALLS, [])

    # (e), (f)
    def test_hooks_and_retain_grad(self):
        for kw in (dict(hook="before"), dict(hook="after"), dict(hook="flat"),
                   dict(retain=True)):
            off = run(False, **kw)
            on = run(True, **kw)
            self.assertSameGrads(on, off)
            if kw.get("hook"):
                self.assertEqual(len(on["caps"]), 1, kw)
                self.assertBits(on["caps"][0], off["caps"][0], f"hook {kw}")
            if kw.get("retain"):
                self.assertBits(on["x_grad"], off["x_grad"], "retain_grad")
        self.assertEqual(SEEN[-2:], [True, True])  # retain_grad after forward: claimed, route declined

    # (g), create_graph, autograd.grad, backward(inputs=)
    def test_backward_modes(self):
        for mode in ("twice", "create_graph", "inputs_w"):
            off = run(False, mode=mode)
            n0 = len(CALLS)
            on = run(True, mode=mode)
            keys = ("w_c", "w_a") if mode == "inputs_w" else (
                "emb", "w_c", "b_c", "w_a", "b_a")
            self.assertSameGrads(on, off, keys)
            self.assertEqual(len(CALLS) - n0, 2 if mode == "twice" else 0, mode)
        off = run(False, mode="grad_x")
        on = run(True, mode="grad_x")
        self.assertBits(on["grad_x"], off["grad_x"], "autograd.grad(loss, [x])")

    # (l) only the content MLP claims
    def test_one_member_routed_falls_back(self):
        off = run(False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            on = run(True, x_cat=lambda t: t.clone())
        self.assertIn("FALLBACK (only one member routed dh)", buf.getvalue())
        self.assertSameGrads(on, off)

    # (i) latch
    def test_fused_exception_latches_off(self):
        off = run(False)

        def boom(*a, **k):
            raise RuntimeError("injected")

        xs.dgrad_segreduce = boom
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            on = run(True)
        self.assertIn("FAILED CLOSED", buf.getvalue())
        self.assertIn("injected", xs._BROKEN or "")
        self.assertSameGrads(on, off)
        # later steps of the process: no ports at all
        v = _forward_ec()
        self.assertEqual(type(v.grad_fn).__name__, "_ECExpandGroupBackward")
        self.assertFalse(hasattr(v, xs.PORT_ATTR))

    # plan 2.6 / VERIFY C3: overlapping member rows and mixed-half cat rows
    def test_invariant_violations_fail_closed(self):
        for overlap in (("item_id", 20), ("album_id", 20)):
            off = run(False, overlap=overlap)
            xp._CHECKS_LEFT = 2
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                on = run(True, overlap=overlap)
            self.assertIn("INVARIANT violated", buf.getvalue())
            self.assertIn("FAILED CLOSED", buf.getvalue())
            self.assertSameGrads(on, off)
            self.assertEqual(CALLS, [])

    # (a) placeholder semantics: no [M, 256] zero gradient, routed or not
    def test_no_port_gradient_materialized(self):
        for kw in (dict(), dict(hook="after"), dict(x_cat=lambda t: t.clone())):
            _arm(True)
            mode = _Zeros()
            with mode:
                run(True, **kw)
            self.assertEqual(mode.hits, [], kw)

    # (j) no reference cycles: refcounting alone frees records, ports, EC node
    def test_no_reference_cycles(self):
        def once():
            out = {}
            run(True, out=out)
            v = out["values"]
            rec, port = getattr(v, xs.PORT_ATTR)
            crec, cport = getattr(out["big"], xs.PORT_ATTR)
            refs = [weakref.ref(rec), weakref.ref(crec), weakref.ref(port), weakref.ref(cport),
                    weakref.ref(v)]
            self.assertEqual(rec.state, "routed")
            return refs

        gc.collect()
        gc.disable()
        try:
            refs = once()
            self.assertEqual([r() is None for r in refs], [True] * len(refs))
        finally:
            gc.enable()

    def test_import_lightness(self):
        script = textwrap.dedent(r"""
            import os, sys
            os.environ["DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE"] = "1"
            from generative_recommenders.dlrm_v4.train import ec_bf16_output  # noqa
            p = 'generative_recommenders.'
            heavy = [p + 'dlrm_v4.train.' + n for n in ('ec_segment_reduce', 'ec_gpu_plan', 'ec_offsets')]
            lazy = [p + 'dlrm_v4.train.ec_dgrad_ports', p + 'ops.triton.triton_ec_dgrad_segreduce']
            assert not any(n in sys.modules for n in heavy + lazy), sorted(sys.modules)
            from generative_recommenders.dlrm_v4.train import ec_dgrad_ports  # noqa
            assert not any(n in sys.modules for n in heavy), [n for n in heavy if n in sys.modules]
        """)
        # -B: inside the validator's container (root) a plain run would write root-owned
        # __pycache__ into the tree under test.
        r = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
