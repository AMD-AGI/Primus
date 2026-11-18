#!/usr/bin/env python3
# pyre-unsafe
"""GPU test (gfx950) of the EDITED call sites of DLRM_HSTU_FP8_FWD_MXFP6 (yambda_5b.gin item (92)).

Drives ops/triton/triton_hstu_preprocess_and_attention.py's two autograd Functions -- the ordinary uvqk layer
(two layers, two weights) and the targets-only layer -- at D=512 (U 512, V/Q/K 1536, V/K 1024) with an M tail
(M = 3000, Mp = 3072): forward, a backward with retain_graph=True, and the retained-graph second backward (which
rebuilds the pack through the MX6 arms 5 / 9).

* The REAL r14 producer (generative_recommenders.ops.hip.ln_store_mx6): its ln_store_mx6 is wrapped only to bind
  every call against its own signature, record the keywords, and stash the shipped store_t's e4m3 y8 of the same
  x for the stand-in below (the outputs are the producer's, untouched).
* An IDENTITY stand-in for pr7's GEMM module with the API the adapter assumes (SUPPORTED_N, kernel, mx6_weight,
  k512_mx6(..., out=)): it computes the SHIPPED FP8 math -- the same triton_fp8 entry points, scales, folded bias
  and weight shadows as the knob-off call -- from that y8. MX6 numerics are not under test here (pr7's kernels
  and the offline fp64 check); the plumbing is: with the knob on, every output, saved tensor, ctx-handed tensor
  and gradient must be BITWISE the knob-off one, and the MX6 path must have served every call.

Each scenario runs in a child process (the knobs are module constants read at import) with
DLRM_HSTU_FP8_FWD_MXFP6=1 and ..._STRICT=1 (any decline or rejection raises):
  flavor "min":  the FP8 chain the MX6 arms sit in, set explicitly (Triton attention);
  flavor "prod": the full yambda_5b.gin bootstrap env (FlyDSL attention and every shipped item);
  x U on MX6 (stand-in SUPPORTED_N has 512: steady producer calls run the no-x8 kernel) or U on FP8 (x8 kernel).
Child protocol: triton_fp8.prebuild_extensions() (the production prebuild; the TM_NONEXP pool must be untouched
by mx6_fwd.prebuild and print no "pool created lazily"), fp8_tm_alloc.prewarm(), knob-off warm-up steps until
the fp8t wgrad scales are published, then runs A1 (off), B1 (on: producer check and guards), B2 (on: steady
state), A2, A3 (off), B3 (on), each of the three layers. A tensor that differs between A1 / A2 / A3 is
non-deterministic in the shipped code (lock-ordered accumulations); the knob-on runs must then stay inside the
knob-off spread; every other tensor must be bitwise equal across all six runs.

Run (kernel lab): lab_py.sh <tree root> -m pytest -q -s generative_recommenders/ops/tests/mx6_fwd_callsite_test.py
(MX6_CALLSITE_ECHO=1 prints each child's whole log, including its JSON result line)."""

import importlib
import inspect
import io
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

_REPO = pathlib.Path(__file__).resolve().parents[2].parent  # recommendation/
_KNOB = "DLRM_HSTU_FP8_FWD_MXFP6"
_IDENTITY = "_mx6_callsite_identity_gemm"
_RESULT = "MX6_CALLSITE_RESULT "
_MIN_ENV: Dict[str, str] = {
    "DLRM_HSTU_FP8_FWD": "1",
    "DLRM_HSTU_FP8_GEMM": "1",
    "DLRM_HSTU_FP8_FUSE_LN": "1",
    "DLRM_HSTU_FP8_FUSE_QKV": "1",
    "DLRM_HSTU_FP8_ATTN": "1",
    "DLRM_HSTU_FP8_FAIL_CLOSED": "1",
    "DLRM_HSTU_FP8_HOST_SCALE_VALUES": "1",
    "DLRM_HSTU_FP8_SCALE_MODE": "frozen",
    "DLRM_HSTU_FP8_BIAS_CACHE_STRICT": "1",
    "DLRM_HSTU_FP8_SAVE_UVQK": "1",
    "DLRM_HSTU_FP8_EARLY_RELEASE_UVQK": "1",
    "DLRM_HSTU_FP8_BWD": "1",
    "DLRM_HSTU_FP8_BWD_UVQK": "1",
    "DLRM_HSTU_FP8_BWD_UVQK_WGRAD": "fp8t",
    "DLRM_HSTU_FP8_BWD_UVQK_STORE_T": "1",
    "DLRM_HSTU_FP8_BWD_STORE_T_V2": "1",
    "DLRM_HSTU_FP8_BWD_WGRAD_CK": "1",
    "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED": "1",
    "DLRM_HSTU_FP8_TM_NONEXP": "1",
    "DLRM_HSTU_FP8_TM_NONEXP_TILED": "1",
    "DLRM_HSTU_FP8_FWD_K512": "1",
    "HSTU_ATTN_FWD": "triton",
    "HSTU_ATTN_BWD": "triton",
}
_DROP_PREFIXES = ("DLRM_HSTU", "HSTU_", "TRITON_HSTU", "GR_PIN", "TRITON_SPEC", "PYTORCH_TUNABLEOP",
                  "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


# ================================================================================================ parent
def _skip_reason() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return "needs torch"
    if not torch.cuda.is_available() or torch.version.hip is None:
        return "needs a ROCm GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists("/opt/rocm/bin/hipcc"):
        return "needs /opt/rocm/bin/hipcc"
    try:
        importlib.import_module("generative_recommenders.ops.hip.ln_store_mx6")
    except ImportError as error:
        return f"the r14 producer is not in this tree ({error})"
    return None


def _prod_env() -> Optional[Dict[str, str]]:
    """The environment yambda_5b.gin + apply_env_bootstrap export (as train_ranker), or None without gin."""
    try:
        import gin

        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb
    except ImportError:
        return None
    gin_file = _REPO / "generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
    with mock.patch.dict(os.environ, {}, clear=True):
        gin.clear_config()
        try:
            gin.parse_config_file(str(gin_file), skip_unknown=True)
            eb.apply_env_bootstrap()
            env = dict(os.environ)
        finally:
            gin.clear_config()
    return env


def _child_env(flavor_env: Dict[str, str], u_mode: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_DROP_PREFIXES)}
    env.update(flavor_env)
    env.update({
        _KNOB: "1",
        _KNOB + "_STRICT": "1",
        _KNOB + "_GEMM_MODULE": _IDENTITY,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DLRM_HSTU_FP8_PREBUILD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MX6_CALLSITE_U": u_mode,
    })
    env.pop(_KNOB + "_PRODUCER_MODULE", None)
    env.pop(_KNOB + "_SHADOW_AUDIT", None)
    return env


_SKIP = _skip_reason()


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class Mx6CallSiteTest(unittest.TestCase):
    def _scenario(self, flavor: str, u_mode: str) -> Dict[str, Any]:
        if flavor == "min":
            flavor_env = dict(_MIN_ENV)
        else:
            prod = _prod_env()
            if prod is None:
                self.skipTest("gin-config not importable: no production env")
            flavor_env = prod
        env = _child_env(flavor_env, u_mode)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", flavor, u_mode],
                              env=env, cwd=str(_REPO), capture_output=True, text=True, timeout=2400)
        log = f"$ child {flavor} {u_mode} rc={proc.returncode} secs={time.time() - t0:.0f}\n" + proc.stdout + \
            "\n----- stderr -----\n" + proc.stderr
        if os.environ.get("MX6_CALLSITE_ECHO") == "1":  # the whole child log on this process's stdout (pytest -s)
            print(f"\n========== child {flavor} {u_mode} ==========\n{log}\n========== end child ==========",
                  flush=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_RESULT)]
        self.assertEqual(proc.returncode, 0, log[-6000:])
        self.assertEqual(len(lines), 1, log[-6000:])
        res = json.loads(lines[0][len(_RESULT):])
        print(f"[callsite {flavor} {u_mode}] " + json.dumps(res["summary"]), flush=True)
        self._check(res, u_mode)
        return res

    def _check(self, res: Dict[str, Any], u_mode: str) -> None:
        self.assertEqual(res["gates"], [True, True, True, True])  # call site, triton_fp8, adapter, STRICT
        # REVIEW M3: mx6_fwd.prebuild maps no TM_NONEXP pool block and prints no pool line
        pre = res["prebuild"]
        self.assertTrue(res["tm_active"], "TM_NONEXP inactive: the M3 check would be vacuous")
        self.assertTrue(pre["ok"])
        self.assertEqual(pre["before"], pre["after"])  # (another prebuild may have mapped it earlier: not M3's)
        self.assertFalse(pre["lazy_line"])
        self.assertFalse(pre["tm_armed_line"])
        self.assertTrue(pre["prebuilt_line"])
        self.assertTrue(pre["producer_prebuilt_line"])
        self.assertTrue(res["tiled_active"])
        self.assertTrue(res["published"], "the fp8t wgrad scales never published: the arms would decline")
        # engagement: every knob-on run served every call through MX6; the knob-off runs never entered it
        u_served = 2 if u_mode == "mx6" else 0
        want = {"planned_uvqk": 2, "planned_vk": 1, "served_VQK": 2, "served_VK": 1, "served_U": u_served,
                "recompute_VQK": 2, "recompute_VK": 1, "declined": 0, "skipped": 0}
        for run in ("B1", "B2", "B3"):
            got = {k: res["counters"][run].get(k, 0) for k in want}
            self.assertEqual(got, want, run)
            self.assertEqual(res["markers"][run], [True, True, True], run)
        for run in ("A1", "A2", "A3"):
            self.assertEqual(res["counters"][run], {}, run)
            self.assertEqual(res["markers"][run], [False, False, False], run)
        lines = res["lines"]
        self.assertEqual(lines["B1"]["producer check ok"], 1)
        self.assertEqual(lines["B1"]["guard VQK armed"], 2)  # one per uvqk layer weight
        self.assertEqual(lines["B1"]["guard VK armed"], 1)
        self.assertEqual(lines["B1"]["guard U armed"], u_served)
        self.assertEqual(lines["B1"]["device ARMED"], 1)
        for run in ("B1", "B2", "B3"):
            for bad in ("DECLINED", "REJECTED", "ERROR", "NOT ARMED"):
                self.assertEqual(lines[run][bad], 0, (run, bad))
        for run in ("B2", "B3"):
            self.assertEqual(lines[run]["guard any"], 0, run)
            self.assertEqual(lines[run]["producer check any"], 0, run)
        # the real producer's calls: the adapter's exact keywords; the steady state runs the no-x8 kernel
        # once U is MX6 (x8 kernel while U stays FP8); the check call is the other instantiation
        steady = [c for c in res["producer_calls"]["B2"] if c["kw"] == ["want_x8", "xt_out"]]
        self.assertEqual(len(steady), 3)
        self.assertEqual([c["want_x8"] for c in steady], [u_mode == "fp8"] * 2 + [False])
        recompute = [c for c in res["producer_calls"]["B2"] if c["kw"] == ["want_x8"]]
        self.assertEqual([c["want_x8"] for c in recompute], [False] * 3)
        checks = [c for c in res["producer_calls"]["B1"] if "x6_out" in c["kw"]]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["want_x8"], False)  # the first call wrote x8 (guards pending)
        # bitwise: knob on == knob off
        cmp_ = res["compare"]
        self.assertGreater(cmp_["names"], 50)
        self.assertEqual(cmp_["missing"], [])
        self.assertEqual(cmp_["meta_mismatch"], [])
        self.assertEqual(cmp_["mismatch_deterministic"], [])
        self.assertEqual(cmp_["outside_envelope"], [])
        for name in cmp_["nondeterministic"]:  # only gradients may be order-nondeterministic
            self.assertIn(".g", name, name)

    def test_min_u_mx6(self) -> None:
        self._scenario("min", "mx6")

    def test_min_u_fp8(self) -> None:
        self._scenario("min", "fp8")

    def test_prod_u_mx6(self) -> None:
        self._scenario("prod", "mx6")

    def test_prod_u_fp8(self) -> None:
        self._scenario("prod", "fp8")


# ================================================================================================ child
class _Tee(io.TextIOBase):
    def __init__(self, stream: Any) -> None:  # pyre-ignore[2]
        self.stream = stream
        self.buf = io.StringIO()

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()

    def mark(self) -> int:
        return len(self.buf.getvalue())

    def since(self, mark: int) -> str:
        return self.buf.getvalue()[mark:]


def _identity_gemm_module(supported: Tuple[int, ...]) -> types.ModuleType:
    """pr7's assumed API computing the SHIPPED FP8 math (see the module docstring)."""
    import torch

    from generative_recommenders.ops.triton import triton_fp8 as f8

    mod = types.ModuleType(_IDENTITY)
    mod.SUPPORTED_N = tuple(supported)  # pyre-ignore[16]
    mod.activations = {}  # pyre-ignore[16]  x6.data_ptr() -> the shipped e4m3 y8 of the same x
    mod.weights = {}  # pyre-ignore[16]  w6.data_ptr() -> (weight, layout)
    mod.calls = []  # pyre-ignore[16]

    def kernel(n: int) -> str:
        if int(n) not in mod.SUPPORTED_N:
            raise RuntimeError(f"identity stand-in: N={n} not built")
        return f"identity-N{n}"

    def mx6_weight(weight, layout):  # pyre-ignore[2,3]
        layout = tuple((int(a), int(b)) for a, b in layout)
        _, scale_b = f8._get_weight_shadow(weight, layout)
        n = sum(b - a for a, b in layout)
        w6 = torch.zeros((n // 256, 2, 4, 12288), dtype=torch.uint8, device=weight.device)
        ws = torch.full((n // 32, 4, 4), 127, dtype=torch.uint8, device=weight.device)
        mod.weights[w6.data_ptr()] = (weight, layout)
        return w6, ws, scale_b

    def k512_mx6(x6, xs, w6, ws, bias, sa, sb, m, n, out=None):  # pyre-ignore[2,3]
        mod.calls.append((int(m), int(n), str(out.dtype)))
        entry = mod.weights.get(w6.data_ptr())
        if entry is None:  # the prebuild self-check: synthetic zero operands, zero bias -> +0
            if int(torch.count_nonzero(x6)) or int(torch.count_nonzero(w6)) or int(torch.count_nonzero(bias.float())):
                raise RuntimeError("identity stand-in: unregistered non-zero operands")
            out.view(torch.uint8).zero_()
            return out
        weight, layout = entry
        y8 = mod.activations[x6.data_ptr()]
        if out.dtype == torch.bfloat16:  # U: the shipped scaled_addmm_fp8_preq (bias already bf16)
            res = f8.scaled_addmm_fp8_preq(activation_fp8=y8, scale_a=sa, weight=weight, bias_or_residual=bias,
                                           out_dtype=out.dtype, weight_layout=layout, data_parallel=True)
        else:  # V/Q/K, V/K: the shipped inner call on the folded scale and the scaled bias
            res = f8.scaled_mm_fp8_out_preq(activation_fp8=y8, scale_a=sa, weight=weight, bias=bias,
                                            weight_layout=layout, output_scale=None)
        out.copy_(res)
        return out

    mod.kernel = kernel  # pyre-ignore[16]
    mod.mx6_weight = mx6_weight  # pyre-ignore[16]
    mod.k512_mx6 = k512_mx6  # pyre-ignore[16]
    return mod


def _bits(t: Any) -> Any:  # pyre-ignore[2,3]
    import torch

    return t.detach().reshape(-1).contiguous().view(torch.uint8)


def _same(a: Any, b: Any) -> bool:  # pyre-ignore[2]
    import torch

    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(_bits(a), _bits(b))


def _max_abs(a: Any, b: Any) -> float:  # pyre-ignore[2]
    d = (a.detach().float() - b.detach().float()).abs()
    d = d.nan_to_num(nan=float("inf"))
    return float(d.max()) if d.numel() else 0.0


def _child(flavor: str, u_mode: str) -> int:
    import torch

    tee = _Tee(sys.stdout)
    sys.stdout = tee
    res: Dict[str, Any] = {"flavor": flavor, "u_mode": u_mode}
    gemm_mod = _identity_gemm_module((512, 1024, 1536) if u_mode == "mx6" else (1024, 1536))
    sys.modules[_IDENTITY] = gemm_mod

    import generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention as cs
    from generative_recommenders.ops.hip import ln_store_mx6 as L6
    from generative_recommenders.ops.hip import mx6_fwd
    from generative_recommenders.ops.triton import bwd_fp8, fp8_tm_alloc, wgrad_ck
    from generative_recommenders.ops.triton import triton_fp8 as f8
    from generative_recommenders.ops.triton.triton_layer_norm import (
        triton_weighted_layer_norm_fwd,
        triton_weighted_layer_norm_fwd_fp8_store_t,
    )

    res["gates"] = [bool(cs._MX6_FWD), bool(f8._HSTU_FP8_FWD_MXFP6), bool(mx6_fwd.ENABLED), bool(mx6_fwd.STRICT)]
    dev = torch.device("cuda", torch.cuda.current_device())

    # -- the REAL producer, wrapped only to bind / record / stash the shipped y8 for the stand-in ------------
    real = L6.ln_store_mx6
    sig = inspect.signature(real)
    calls: List[Dict[str, Any]] = []
    stash = {"on": False}  # off during the prebuild: its zero GEMM self-check needs no y8, and the store_t
    # below would map a TM_NONEXP block inside the very window the M3 check watches

    def producer(*args, **kwargs):  # pyre-ignore[2,3]
        sig.bind(*args, **kwargs)
        calls.append({"kw": sorted(kwargs), "want_x8": bool(kwargs.get("want_x8")), "m": int(args[0].shape[0])})
        out = real(*args, **kwargs)
        if stash["on"]:
            x, w, b, eps, inv = args[:5]
            y8 = triton_weighted_layer_norm_fwd_fp8_store_t(x=x, weight=w, bias=b, eps=eps, inv_scale=inv,
                                                            pad_to=128, tiled_xt=True)[0]
            gemm_mod.activations[out[0].data_ptr()] = y8
        return out

    L6.ln_store_mx6 = producer

    # -- the production prebuild; mx6_fwd.prebuild must not touch the TM_NONEXP pool (REVIEW M3) ------------
    def snap() -> Dict[str, Any]:
        return {"pools": sorted(int(k) for k in fp8_tm_alloc._pools),
                "blocks": {str(k): len(v) for k, v in fp8_tm_alloc._block_ptrs.items()},
                "armed": sorted(fp8_tm_alloc._armed_logged), "lazy": sorted(fp8_tm_alloc._lazy_logged),
                "prewarmed": sorted(int(k) for k in fp8_tm_alloc._prewarmed)}

    real_prebuild = mx6_fwd.prebuild

    def checked_prebuild(*a, **k):  # pyre-ignore[2,3]
        before = snap()
        mark = tee.mark()
        ok = real_prebuild(*a, **k)
        text = tee.since(mark)
        res["prebuild"] = {"ok": ok, "before": before, "after": snap(),
                           "lazy_line": "pool created lazily" in text, "tm_armed_line": "ARMED tm_nonexp" in text,
                           "prebuilt_line": "[mx6-fwd] prebuilt:" in text,
                           "producer_prebuilt_line": "[ln-store-mx6] prebuilt in" in text}
        return ok

    mx6_fwd.prebuild = checked_prebuild
    try:
        f8.prebuild_extensions()
    finally:
        mx6_fwd.prebuild = real_prebuild
    stash["on"] = True
    res["producer_sha256"] = L6.kernel().source_sha256
    res["tm_active"] = bool(fp8_tm_alloc.active())
    if fp8_tm_alloc.active():
        fp8_tm_alloc.prewarm(fp8_tm_alloc.default_blocks(), dev)
    res["tiled_active"] = bool(wgrad_ck.tiled_active(dev))

    # -- three layers: two ordinary uvqk layers (two weights) and the targets-only layer --------------------
    H, A, HD, D = 4, 128, 128, 512
    lengths = [1100, 900, 700, 300]  # M = 3000: a tail tile (Mp = 3072)
    offsets = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int64, device=dev)
    m_rows = int(offsets[-1])
    max_len = max(lengths)
    alpha = 1.0 / A ** 0.5
    num_targets = torch.ones(len(lengths), dtype=torch.int64, device=dev)
    fuse_silu = bool(f8.hstu_fp8_fuse_silu_enabled())
    g = torch.Generator(device=dev).manual_seed(5858)

    def frozen_state(value):  # pyre-ignore[2,3]
        state = torch.zeros(f8.fp8_scale_state_size(), dtype=torch.float32, device=dev)
        state[0] = f8.calibrated_scale(value)
        state[1] = state.numel() - 2
        state[2:].fill_(float(value.detach().abs().max()))
        return state

    layers: List[Dict[str, Any]] = []
    for idx, kind in enumerate(("ordinary", "ordinary", "targets")):
        x = (torch.randn((m_rows, D), device=dev, generator=g) * (1.0 + 0.5 * idx) + 0.1).to(torch.bfloat16)
        nw = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
        nb = (0.05 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
        w = torch.nn.Parameter(torch.randn((D, 2048), device=dev, generator=g) / D ** 0.5)
        b = torch.nn.Parameter(0.1 * torch.randn(2048, device=dev, generator=g))
        with torch.no_grad():
            normalized, _, _ = triton_weighted_layer_norm_fwd(x, nw, nb, 1.0e-6)
            if kind == "ordinary":
                proj = normalized @ w[:, 512:].to(torch.bfloat16) + b[512:].to(torch.bfloat16)
            else:
                cols = torch.cat((w[:, 512:1024], w[:, 1536:]), dim=1)
                proj = normalized @ cols.to(torch.bfloat16) + torch.cat((b[512:1024], b[1536:])).to(torch.bfloat16)
        layers.append(dict(kind=kind, x=x, nw=nw, nb=nb, W=w, B=b, s_in=frozen_state(normalized),
                           s_qkv=frozen_state(proj)))
    cot_cache: Dict[Tuple[int, int], Any] = {}

    def cotangent(layer: int, i: int, out):  # pyre-ignore[2,3]
        key = (layer, i)
        if key not in cot_cache:
            gen = torch.Generator(device=dev).manual_seed(1000 * layer + i)
            cot_cache[key] = torch.randn(tuple(out.shape), device=dev, generator=gen).to(out.dtype)
        return cot_cache[key]

    def run(knob: bool) -> Tuple[Dict[str, Any], Dict[str, Any], List[bool]]:
        cs._MX6_FWD = knob
        tensors: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        markers: List[bool] = []
        outs_all: List[Any] = []
        cots_all: List[Any] = []
        leaves: List[Tuple[Any, ...]] = []
        for li, spec in enumerate(layers):
            x = spec["x"].detach().clone().requires_grad_(True)
            nw = spec["nw"].detach().clone().requires_grad_(True)
            nb = spec["nb"].detach().clone().requires_grad_(True)
            spec["W"].grad = None
            spec["B"].grad = None
            if spec["kind"] == "ordinary":
                outs = cs.triton_hstu_preprocess_and_attention(
                    x, nw, nb, 1.0e-6, H, A, HD, spec["W"], spec["B"], max_len, offsets, alpha, None,
                    max_attn_len=0, contextual_seq_len=0, recompute_uvqk_in_backward=True,
                    recompute_normed_x_in_backward=True, sort_by_length=True, enable_tma=None,
                    fp8_input_scale=spec["s_in"], fp8_qkv_scale=spec["s_qkv"], defer_silu=fuse_silu,
                    allow_save_fp8_uvqk=True)
            else:
                outs = cs.triton_hstu_preprocess_and_attention_targets_only(
                    x, nw, nb, 1.0e-6, H, A, HD, spec["W"], spec["B"], max_len, offsets, alpha, num_targets,
                    enable_tma=None, fp8_input_scale=spec["s_in"], fp8_qkv_scale=spec["s_qkv"],
                    allow_save_fp8_vk=True)
            ctx = outs[0].grad_fn
            for i, o in enumerate(outs):
                tensors[f"L{li}.out{i}"] = o.detach().clone()
            for j, t in enumerate(ctx.saved_tensors):
                tensors[f"L{li}.saved{j}"] = t.detach().clone()
                meta[f"L{li}.saved{j}"] = [str(t.dtype), list(t.shape), list(t.stride())]
            for attr in ("_stored_x_fp8_t", "_ephemeral_fp8_vqk", "_stored_vk_x_fp8_t", "_ephemeral_fp8_vk"):
                t = getattr(ctx, attr, None)
                if isinstance(t, torch.Tensor):
                    tensors[f"L{li}.ctx{attr}"] = t.detach().clone()
                    meta[f"L{li}.ctx{attr}"] = [str(t.dtype), list(t.shape), list(t.stride())]
            markers.append(bool(getattr(ctx, "_mx6_vqk", False) or getattr(ctx, "_mx6_vk", False)))
            for i, o in enumerate(outs):
                if o.requires_grad:
                    outs_all.append(o)
                    cots_all.append(cotangent(li, i, o))
            leaves.append((x, nw, nb, spec["W"], spec["B"]))
        torch.autograd.backward(outs_all, cots_all, retain_graph=True)
        for phase in ("g1", "g2"):
            if phase == "g2":
                torch.autograd.backward(outs_all, cots_all)  # the retained-graph second backward
            for li, lv in enumerate(leaves):
                for name, t in zip(("x", "nw", "nb", "W", "B"), lv):
                    if t.grad is not None:
                        tensors[f"L{li}.{phase}.{name}"] = t.grad.detach().clone()
                    t.grad = None
        torch.cuda.synchronize()
        cs._MX6_FWD = False
        return tensors, meta, markers

    def published() -> bool:
        ok = True
        for spec in layers:
            if spec["kind"] == "ordinary":
                ok &= bwd_fp8.uvqk_fp8t_will_run("uvqk_projection", spec["W"], rows=m_rows, dtype=torch.bfloat16,
                                                 input_scale_value=f8.activation_scale_value(spec["s_in"]))
            else:
                ok &= bwd_fp8.uvqk_fp8t_will_run("vk_projection", spec["W"].narrow(1, 0, 1024), rows=m_rows,
                                                 dtype=torch.bfloat16,
                                                 input_scale_value=f8.activation_scale_value(spec["s_in"]),
                                                 per_layer=False)
        return bool(ok)

    warm = 0
    while warm < 12 and not published():
        run(False)
        warm += 1
    run(False)  # one more step past the publication
    res["warmup_steps"] = warm + 1
    res["published"] = published()

    runs: Dict[str, Dict[str, Any]] = {}
    metas: Dict[str, Dict[str, Any]] = {}
    res["counters"] = {}
    res["markers"] = {}
    res["lines"] = {}
    res["producer_calls"] = {}
    for name, knob in (("A1", False), ("B1", True), ("B2", True), ("A2", False), ("A3", False), ("B3", True)):
        c0 = mx6_fwd.counters()
        mark = tee.mark()
        ncalls = len(calls)
        runs[name], metas[name], res["markers"][name] = run(knob)
        c1 = mx6_fwd.counters()
        res["counters"][name] = {k: c1.get(k, 0) - c0.get(k, 0) for k in set(c0) | set(c1)
                                 if c1.get(k, 0) != c0.get(k, 0)}
        text = "\n".join(ln for ln in tee.since(mark).splitlines() if "[mx6-fwd]" in ln)
        res["lines"][name] = {
            "producer check ok": sum("producer check on cuda:" in ln and ln.endswith("-> ok")
                                     for ln in text.splitlines()),
            "producer check any": text.count("producer check on cuda:"),
            "guard VQK armed": sum("guard VQK on cuda:" in ln and ln.endswith("-> armed") for ln in text.splitlines()),
            "guard VK armed": sum("guard VK on cuda:" in ln and ln.endswith("-> armed") for ln in text.splitlines()),
            "guard U armed": sum("guard U on cuda:" in ln and ln.endswith("-> armed") for ln in text.splitlines()),
            "guard any": text.count("] guard "),
            "device ARMED": text.count("[mx6-fwd] ARMED on cuda:"),
            "DECLINED": text.count("DECLINED"),
            "REJECTED": text.count("REJECTED"),
            "ERROR": text.count("[mx6-fwd] ERROR"),
            "NOT ARMED": text.count("NOT ARMED"),
            "recompute": text.count("retained-graph backward:"),
        }
        res["producer_calls"][name] = calls[ncalls:]

    # -- bitwise comparison ---------------------------------------------------------------------------------
    names = sorted(set().union(*(set(r) for r in runs.values())))
    missing = [f"{n}@{r}" for n in names for r in runs if n not in runs[r]]
    meta_mismatch = [n for n in sorted(metas["A1"]) if any(metas[r].get(n) != metas["A1"][n] for r in metas)]
    nondet, mismatch, outside, envelope = [], [], [], {}
    for n in names:
        if any(n not in runs[r] for r in runs):
            continue
        a1, a2, a3 = runs["A1"][n], runs["A2"][n], runs["A3"][n]
        bs = [runs[r][n] for r in ("B1", "B2", "B3")]
        if _same(a1, a2) and _same(a1, a3):
            if not all(_same(a1, b) for b in bs):
                mismatch.append(n)
            continue
        nondet.append(n)
        spread = max(_max_abs(a1, a2), _max_abs(a1, a3), _max_abs(a2, a3))
        worst = max(_max_abs(a1, b) for b in bs)
        envelope[n] = [spread, worst]
        if not worst <= 2.0 * spread:
            outside.append(n)
    res["compare"] = {"names": len(names), "missing": missing, "meta_mismatch": meta_mismatch,
                      "mismatch_deterministic": mismatch, "nondeterministic": nondet, "envelope": envelope,
                      "outside_envelope": outside}
    res["summary"] = {
        "warmup_steps": res["warmup_steps"], "published": res["published"], "tm_active": res["tm_active"],
        "tiled_active": res["tiled_active"], "prebuild_ok": res.get("prebuild", {}).get("ok"),
        "pool_untouched": res.get("prebuild", {}).get("before") == res.get("prebuild", {}).get("after"),
        "pool_before_mx6_prebuild": res.get("prebuild", {}).get("before"),
        "names": len(names), "bitwise_equal": len(names) - len(nondet) - len(mismatch),
        "nondeterministic": nondet, "mismatch_deterministic": mismatch, "outside_envelope": outside,
        "B1_counters": res["counters"]["B1"], "B1_lines": res["lines"]["B1"],
        "producer_sha256": res["producer_sha256"], "gemm_calls": len(gemm_mod.calls),
    }
    sys.stdout = tee.stream
    print(_RESULT + json.dumps(res, default=str), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2], sys.argv[3]))
    unittest.main()
