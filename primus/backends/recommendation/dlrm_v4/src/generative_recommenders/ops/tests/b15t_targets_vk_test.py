#!/usr/bin/env python3

# pyre-unsafe

"""B15T(L2), yambda_5b.gin item B15T(L2), DLRM_HSTU_FP8_FWD_K512_TILED_A: the
targets-only HSTU layer with the knob on and off, bit for bit.

With the knob on, the targets-only forward's Triton v2 store_t runs with
store_y=False (the unmodified kernel still writes y's bytes, but into a
dedicated 1/32-size discarded scratch): no row-major e4m3 y [M, 512] is
returned, and
the V/K projection reads y's bytes from the step-tiled yT on the K512 library's
tiled-A entry. Everything that leaves the autograd function must be
byte-identical to the knob-off path.

GinKnobTest (no GPU; needs gin-config):
- the shipped yambda_5b.gin parses with skip_unknown=True (train_ranker phase 1:
  apply_env_bootstrap is registered, so an unknown parameter of it raises) and
  binds apply_env_bootstrap.DLRM_HSTU_FP8_FWD_K512_TILED_A = True;
- apply_env_bootstrap() exports "1", and a pre-set "0" wins (the kill switch);
  triton_fp8 then reads the knob off (needs torch);
- the full strict parse (train_ranker phase 3, after its phase-2 imports) accepts
  the file (skipped if those imports fail here).

TargetsVKParityTest (gfx950 + /opt/rocm/bin/hipcc + gin-config): a worker
subprocess runs the PRODUCTION environment: the shipped gin's apply_env_bootstrap
(FP8 forward arm, fp8t V/K wgrad with the store_t v2, K512, wgrad_ck step-tiled
xT, B11 pool, ...), expandable segments, the pinned Triton configs and
triton_fp8.prebuild_extensions(). It warms the vk_projection backward site up
(bf16 warmup calls until the site publishes, as in training), then at each M runs
triton_hstu_preprocess_and_attention_targets_only forward + backward with the
knob off and on, in the same process on the same inputs (the arm is the knob's
module constant). Per arm it hashes (SHA-256 of the bytes, plus shape / dtype /
stride) every forward output, every ctx.saved_tensors entry, the stored yT and
the saved e4m3 V/K pack, the ctx scalars, and every gradient. Before each arm
the caching allocator is filled with 0xFF and released, so an output byte that
is not written reads as NaN, and every yT (fp8_tm_alloc.empty_xt: a B11 pool
block in production, which the caching-allocator fill does not reach) is
filled with a per-arm sentinel byte before the store_t writes it, so a yT byte
the store_t does not write differs between the arms. Shapes: D=512, 4 heads,
128/128 (production), max_seq_len 4096, one target per sequence; M in {1, 127,
128, 129, 1000, 1987050, 2376002, 2650962, 2900001} (B15T_TEST_M overrides).
FP8 scales: frozen scale states calibrated on the data with the production rule
(amax * 1.5 / 448), i.e. non-power-of-two values.
Checks: on == off for every hash; the on arm took the tiled path at every M
(store_t asked for store_y=False, returned y=None and a [KT, 512, 128] yT; the
tiled-A V/K call ran; no row-major V/K call), the off arm never; the on arm
compiled nothing (no new Triton cache entry, no K512 build: the store_y=False
call runs the binary the off arm's store_y=True call compiled); an in-process A/A of the
on arm at two M; D9 rows (r16: x rows of +-Inf, NaN, huge and constant values,
and +-Inf LayerNorm weight columns) on == off; forced declines in the on arm,
each bitwise the off arm with its "[b15t] declined" line: after the y skip (y
rebuilt from yT, B15T then LATCHED OFF): a K512 operand decline, a tiled-A
launch error (hipError 98 injected), a store_t that wrote y anyway; the latched
state (the next call writes y: decided before the LayerNorm); before the
LayerNorm: the store_t pre-check, the tiled-A entry not ready. One "[b15t]
ARMED" line; the y rebuild (b15t_row_major_y) equals the store_t's y for 2-D
and step-tiled yT. A second worker with DLRM_HSTU_FP8_FWD_K512_TILED_A=0 in the
environment (the kill switch over the gin binding) must run no B15T path, print
no B15T line, skip the tiled-A self-check, and give the same hashes.

Run (kernel lab, gin-config installed): lab_py.sh <tree root> -B -m pytest -q -s
generative_recommenders/ops/tests/b15t_targets_vk_test.py
(B15T_TEST_OUT=<dir> keeps the workers' JSON and stdout.)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[2]  # .../generative_recommenders
_GIN = _PKG / "dlrm_v4" / "train" / "gin" / "yambda_5b.gin"
KNOB = "DLRM_HSTU_FP8_FWD_K512_TILED_A"
HIPCC = "/opt/rocm/bin/hipcc"

D, HEADS, DQK, DV, MAX_SEQ_LEN = 512, 4, 128, 128, 4096
EDGE_M: Tuple[int, ...] = (1, 127, 128, 129, 1000)
PROD_M: Tuple[int, ...] = (1987050, 2376002, 2650962, 2900001)
AA_M: Tuple[int, ...] = (129, 2376002)  # on arm run twice (in-process determinism)
FORCED_M: Tuple[int, ...] = (129, 1000)
D9_M: Tuple[int, ...] = (1000, 2376002)
KILL_M: Tuple[int, ...] = (129, 1000, 2376002)
WARM_M = 6000
MEAN_SEQ_LEN = 2320  # ~1024 sequences at the in-model ~2.38M tokens per rank
SCALE_MARGIN = 1.5  # production DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN
WORKER_TIMEOUT_S = 5400
_CTX_SCALARS = (
    "saved_bf16_normed_x",
    "attn_alpha",
    "max_seq_len",
    "hidden_dim",
    "attn_dim",
    "num_heads",
    "norm_eps",
    "norm_BLOCK_D",
    "fp8_contract",
    "fp8_input_forward",
    "fp8_qkv_forward",
    "fp8_input_scale_value",
    "fp8_qkv_scale_value",
    "residual_rows_only",
)
_REPORT_ENV = (
    KNOB,
    "DLRM_HSTU_FP8_FWD",
    "DLRM_HSTU_FP8_FWD_K512",
    "DLRM_HSTU_FP8_BWD_STORE_T_V2",
    "DLRM_HSTU_FP8_BWD_UVQK",
    "DLRM_HSTU_FP8_BWD_UVQK_WGRAD",
    "DLRM_HSTU_FP8_BWD_UVQK_STORE_T",
    "DLRM_HSTU_FP8_BWD_VK_PRODUCER",
    "DLRM_HSTU_FP8_BWD_WGRAD_CK",
    "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED",
    "DLRM_HSTU_FP8_TM_NONEXP",
    "DLRM_HSTU_FP8_SAVE_TARGETS_VK",
    "DLRM_HSTU_FP8_TGT_SPARSE_RESID",
    "DLRM_HSTU_FP8_SAVE_UVQK",
    "DLRM_HSTU_FP8_EARLY_RELEASE_UVQK",
    "DLRM_HSTU_FP8_HOST_SCALE_VALUES",
    "DLRM_HSTU_FP8_FAIL_CLOSED",
    "DLRM_HSTU_FP8_TELEMETRY_INTERVAL",
    "DLRM_HSTU_FP8_PREBUILD",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The caller's environment without any knob the gin bootstrap binds (so the
    shipped gin decides them), plus `extra`."""
    drop_prefix = ("DLRM_HSTU_FP8_", "TRITON_HSTU_", "HSTU_ATTN_")
    drop = {"TRITON_FULL_AUTOTUNE", "GR_PIN_BF16_BLAS_SOLUTIONS"}
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(drop_prefix) and k not in drop
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # ship15: yambda_5b.gin binds DLRM_HSTU_FP8_FWD_MXFP6 = True, and MX6 serves the
    # targets V/K forward, bypassing B15T. B15T stays the path under the MX6 kill
    # switch, so this test pins MX6 off to keep exercising it (test-only).
    env["DLRM_HSTU_FP8_FWD_MXFP6"] = "0"
    env.update(extra or {})
    return env


def _py(code: str, env: Dict[str, str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _last_json(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("B15T_JSON "):
            return json.loads(line[len("B15T_JSON ") :])
    raise AssertionError("no B15T_JSON line in:\n" + stdout[-4000:])


def _have_gin() -> bool:
    import importlib.util

    return importlib.util.find_spec("gin") is not None


# --------------------------------------------------------------------------- gin
_GIN_PROBE = r"""
import json, os, sys
import gin
from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
gin.parse_config_file(sys.argv[1] if len(sys.argv) > 1 else %(gin)r, skip_unknown=True)
bound = gin.query_parameter("apply_env_bootstrap.%(knob)s")
pre = os.environ.get("%(knob)s")
apply_env_bootstrap()
out = {"bound": bound, "pre": pre, "env": os.environ.get("%(knob)s")}
if %(import_fp8)s:
    from generative_recommenders.ops.triton import triton_fp8
    out["requested"] = triton_fp8.k512_tiled_a_requested()
print("B15T_JSON " + json.dumps(out))
"""

_STRICT_PROBE = r"""
import json, os
import gin
from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
gin.parse_config_file(%(gin)r, skip_unknown=True)
apply_env_bootstrap()
try:
    # train_ranker's phase-2 imports (they register the configurables the strict
    # phase-3 parse binds)
    import generative_recommenders.dlrm_v4.train.mlperf_logging_utils  # noqa
    import generative_recommenders.dlrm_v4.checkpoint  # noqa
    import generative_recommenders.dlrm_v4.train.utils  # noqa
    import generative_recommenders.dlrm_v4.utils  # noqa
    import generative_recommenders.dlrm_v4.train.blas_warmup  # noqa
    import generative_recommenders.dlrm_v4.train.fp8_warmup  # noqa
    import generative_recommenders.ops.triton.autotune_pin_table  # noqa
    import generative_recommenders.modules.stu  # noqa
    import generative_recommenders.ops.position  # noqa
    import generative_recommenders.ops.input_dropout_fold  # noqa
except Exception as error:
    print("B15T_JSON " + json.dumps({"skip": f"{type(error).__name__}: {error}"[:300]}))
    raise SystemExit(0)
gin.parse_config_file(%(gin)r)
print("B15T_JSON " + json.dumps({"bound": gin.query_parameter("apply_env_bootstrap.%(knob)s"),
                                  "env": os.environ.get("%(knob)s")}))
"""


def _torch_importable() -> bool:
    r = _py("import torch", _clean_env(), timeout=300)
    return r.returncode == 0


@unittest.skipIf(not _have_gin(), "needs gin-config")
class GinKnobTest(unittest.TestCase):
    def _probe(self, extra: Dict[str, str], import_fp8: bool) -> Dict[str, Any]:
        code = _GIN_PROBE % {"gin": str(_GIN), "knob": KNOB, "import_fp8": import_fp8}
        r = _py(code, _clean_env(extra))
        self.assertEqual(r.returncode, 0, r.stdout[-4000:])
        return _last_json(r.stdout)

    def test_gin_binds_knob_true_and_bootstrap_exports_it(self) -> None:
        out = self._probe({}, import_fp8=False)
        self.assertIs(out["bound"], True)
        self.assertIsNone(out["pre"])
        self.assertEqual(out["env"], "1")

    def test_env_kill_switch_wins(self) -> None:
        out = self._probe({KNOB: "0"}, import_fp8=False)
        self.assertIs(out["bound"], True)
        self.assertEqual((out["pre"], out["env"]), ("0", "0"))
        if not _torch_importable():
            self.skipTest("torch not importable: module-constant half skipped")
        on = self._probe({}, import_fp8=True)
        off = self._probe({KNOB: "0"}, import_fp8=True)
        self.assertIs(on["requested"], True)
        self.assertIs(off["requested"], False)

    def test_full_strict_parse(self) -> None:
        code = _STRICT_PROBE % {"gin": str(_GIN), "knob": KNOB}
        r = _py(code, _clean_env(), timeout=1800)
        self.assertEqual(r.returncode, 0, r.stdout[-4000:])
        out = _last_json(r.stdout)
        if "skip" in out:
            self.skipTest("train_ranker phase-2 imports unavailable: " + out["skip"])
        self.assertIs(out["bound"], True)
        self.assertEqual(out["env"], "1")


# ------------------------------------------------------------------------ worker
def _seq_lengths(m: int, seed: int) -> List[int]:
    """Deterministic sequence lengths in [1, MAX_SEQ_LEN] summing to m."""
    rng = random.Random(seed)
    b = max(1, -(-m // MEAN_SEQ_LEN))
    raw = [rng.randint(MEAN_SEQ_LEN // 4, MAX_SEQ_LEN) for _ in range(b)]
    total = sum(raw)
    lens = [max(1, min(MAX_SEQ_LEN, r * m // total)) for r in raw]
    diff = m - sum(lens)
    i = 0
    while diff:
        j = i % b
        if diff > 0 and lens[j] < MAX_SEQ_LEN:
            lens[j] += 1
            diff -= 1
        elif diff < 0 and lens[j] > 1:
            lens[j] -= 1
            diff += 1
        i += 1
    assert sum(lens) == m and all(1 <= v <= MAX_SEQ_LEN for v in lens)
    return lens


def _digest(t: Any) -> Optional[Dict[str, Any]]:
    import torch

    if t is None:
        return None
    t = t.detach()
    flat = (t if t.is_contiguous() else t.contiguous()).reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    chunk = 1 << 28
    for i in range(0, flat.numel(), chunk):
        h.update(flat[i : i + chunk].cpu().numpy())
    return {
        "sha256": h.hexdigest(),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "stride": list(t.stride()),
    }


def _worker(cfg: Dict[str, Any]) -> None:
    # Production allocator before torch initializes CUDA; then the gin bootstrap
    # (train_ranker phase 1) before any kernel module is imported.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import gin

    from generative_recommenders.dlrm_v4.train._env_bootstrap import (
        apply_env_bootstrap,
    )

    gin.parse_config_file(str(_GIN), skip_unknown=True)
    apply_env_bootstrap()
    import torch

    from generative_recommenders.ops.hip import k512_fp8_gemm as k512
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import fp8_tm_alloc
    from generative_recommenders.ops.triton import triton_fp8 as tf8
    from generative_recommenders.ops.triton import (
        triton_hstu_preprocess_and_attention as tpa,
    )
    from generative_recommenders.ops.triton import triton_layer_norm as tln
    from generative_recommenders.ops.triton import wgrad_ck
    from generative_recommenders.ops.triton.autotune_pin_table import (
        apply_autotune_pins,
    )

    # As train_ranker: the pinned Triton configs before any launch.
    pins = apply_autotune_pins()
    # Compile instruments: K512 hipcc builds (k512._build calls) and the Triton
    # cache directory (one new entry per compiled kernel specialization).
    k512_builds = [0]
    orig_build = k512._build

    def counting_build(*a: Any, **k: Any) -> Any:
        k512_builds[0] += 1
        return orig_build(*a, **k)

    k512._build = counting_build
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".triton", "cache"
    )

    def compile_state() -> Tuple[int, frozenset]:
        names = frozenset(os.listdir(triton_cache)) if os.path.isdir(triton_cache) else frozenset()
        return k512_builds[0], names

    # F4: every yT starts as a per-arm sentinel byte (the B11 pool blocks are not
    # reached by the caching-allocator fill below).
    sentinel = [0xA5]
    orig_empty_xt = fp8_tm_alloc.empty_xt

    def sentinel_empty_xt(*a: Any, **k: Any) -> Any:
        t = orig_empty_xt(*a, **k)
        t.view(torch.uint8).fill_(sentinel[0])
        return t

    fp8_tm_alloc.empty_xt = sentinel_empty_xt
    report: Dict[str, Any] = {
        "mode": cfg["mode"],
        "env": {k: os.environ.get(k) for k in _REPORT_ENV},
        "knob_requested": tf8.k512_tiled_a_requested(),
        "autotune_pins": list(pins),
    }
    print(f"[b15t-test] worker mode={cfg['mode']} env={report['env']}", flush=True)
    report["prebuilt"] = bool(tf8.prebuild_extensions())
    report["tiled_a_ok"] = sorted(list(k) for k in k512._tiled_a_ok)
    dev = torch.device("cuda", torch.cuda.current_device())
    report["device"] = {
        "name": torch.cuda.get_device_name(dev),
        "arch": str(getattr(torch.cuda.get_device_properties(dev), "gcnArchName", "")),
    }
    bf16 = torch.bfloat16

    # Parameters as STULayer holds them for the FP8 targets path: fp32 uvqk
    # weight/bias (passed uncast), bf16 LayerNorm weight/bias.
    gp = torch.Generator(device=dev).manual_seed(20261010)
    n_uvqk = 2 * HEADS * DV + 2 * HEADS * DQK
    bound = (6.0 / (D + n_uvqk)) ** 0.5
    ln_w = (1.0 + 0.1 * torch.randn(D, generator=gp, device=dev)).to(bf16).requires_grad_()
    ln_b = (0.1 * torch.randn(D, generator=gp, device=dev)).to(bf16).requires_grad_()
    uvqk_w = ((torch.rand((D, n_uvqk), generator=gp, device=dev) * 2 - 1) * bound).requires_grad_()
    uvqk_b = (0.02 * torch.randn(n_uvqk, generator=gp, device=dev)).requires_grad_()
    params = {"ln_w": ln_w, "ln_b": ln_b, "uvqk_w": uvqk_w, "uvqk_b": uvqk_b}
    u_dim = v_dim = HEADS * DV
    q_dim = HEADS * DQK

    def make_x(m: int) -> torch.Tensor:
        g = torch.Generator(device=dev).manual_seed(1_000_003 + m)
        out = torch.empty((m, D), dtype=bf16, device=dev)
        step = 1 << 20
        for i in range(0, m, step):
            j = min(m, i + step)
            out[i:j] = torch.randn((j - i, D), generator=g, device=dev).to(bf16)
        return out

    def offsets_for(m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lens = _seq_lengths(m, seed=m)
        offs = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device=dev)
        return offs, torch.ones(len(lens), dtype=torch.int64, device=dev)

    # Frozen FP8 scale states, calibrated on the warmup data with the production
    # rule (scale = amax * ASCALE_MARGIN / 448; [1] = completed steps).
    def scale_state(amax: torch.Tensor) -> torch.Tensor:
        st = torch.zeros(tf8.fp8_scale_state_size(), dtype=torch.float32, device=dev)
        st[2:].fill_(float(amax))
        st[1] = st.numel() - 2
        st[0] = float(amax) * SCALE_MARGIN / 448.0
        return st

    x_warm = make_x(WARM_M)
    with torch.no_grad():
        nx, _, _ = tln.triton_weighted_layer_norm_fwd(x_warm, ln_w, ln_b, 1e-6)
        w_vk = torch.cat(
            (uvqk_w[:, u_dim : u_dim + v_dim], uvqk_w[:, u_dim + v_dim + q_dim :]), dim=1
        )
        b_vk = torch.cat((uvqk_b[u_dim : u_dim + v_dim], uvqk_b[u_dim + v_dim + q_dim :]))
        vk_ref = nx @ w_vk.to(bf16) + b_vk.to(bf16)
        in_state = scale_state(nx.float().abs().amax())
        qkv_state = scale_state(vk_ref.float().abs().amax())
    report["scales"] = {"input": float(in_state[0]), "qkv": float(qkv_state[0])}
    del nx, vk_ref

    save_targets_vk = os.environ.get("DLRM_HSTU_FP8_SAVE_TARGETS_VK", "0") != "0"
    tgt_sparse = os.environ.get("DLRM_HSTU_FP8_TGT_SPARSE_RESID", "0") != "0"
    fuse_residual = tf8.hstu_fp8_fuse_residual_enabled()

    # Pass-through recorders on the targets module's globals.
    events: List[List[Any]] = []
    force_write_y = [False]
    orig_store_t = tpa.triton_weighted_layer_norm_fwd_fp8_store_t
    orig_tiled = tpa.scaled_mm_fp8_out_preq_tiled_a
    orig_rm = tpa.scaled_mm_fp8_out_preq
    orig_rebuild = tpa.b15t_row_major_y

    def rec_store_t(*a: Any, **k: Any) -> Any:
        asked = bool(k.get("store_y", True))
        if force_write_y[0]:
            k = dict(k, store_y=True)  # forced: a store_t that writes y anyway
        r = orig_store_t(*a, **k)
        events.append(["store_t", asked, r[0] is None, list(r[3].shape)])
        return r

    def rec_tiled(*a: Any, **k: Any) -> Any:
        r = orig_tiled(*a, **k)
        events.append(["vk_tiled", r is not None])
        return r

    def rec_rm(*a: Any, **k: Any) -> Any:
        events.append(["vk_row_major"])
        return orig_rm(*a, **k)

    def rec_rebuild(*a: Any, **k: Any) -> Any:
        events.append(["y_rebuilt"])
        return orig_rebuild(*a, **k)

    tpa.triton_weighted_layer_norm_fwd_fp8_store_t = rec_store_t
    tpa.scaled_mm_fp8_out_preq_tiled_a = rec_tiled
    tpa.scaled_mm_fp8_out_preq = rec_rm
    tpa.b15t_row_major_y = rec_rebuild

    def dirty_allocator() -> None:
        # Fill most of the reusable memory (device-free plus cached-unused) with
        # 0xFF through the caching allocator and release it to the cache: freshly
        # allocated outputs then start as e4m3/bf16 NaN bytes.
        free, _ = torch.cuda.mem_get_info(dev)
        cached = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        n = int(min((free + cached) * 0.6, 96 << 30)) >> 20 << 20
        if n > 0:
            junk = torch.empty(n, dtype=torch.uint8, device=dev)
            junk.fill_(0xFF)
            del junk

    def call(x: torch.Tensor, offs: torch.Tensor, nt: torch.Tensor,
             w_ln: torch.Tensor) -> Any:
        return tpa.triton_hstu_preprocess_and_attention_targets_only(
            x=x,
            norm_weight=w_ln,
            norm_bias=ln_b,
            norm_eps=1e-6,
            num_heads=HEADS,
            attn_dim=DQK,
            hidden_dim=DV,
            uvqk_weight=uvqk_w,
            uvqk_bias=uvqk_b,
            max_seq_len=MAX_SEQ_LEN,
            seq_offsets=offs,
            attn_alpha=1.0 / (DQK**0.5),
            num_targets=nt,
            fp8_input_scale=in_state,
            fp8_qkv_scale=qkv_state,
            return_residual_proxy=fuse_residual,
            residual_rows_only=tgt_sparse,
            allow_save_fp8_vk=save_targets_vk,
        )

    def run(m: int, x_base: torch.Tensor, offs: torch.Tensor, nt: torch.Tensor,
            on: Optional[bool], digest: bool = True, sent: int = 0xA5,
            w_ln: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        # on: the arm (the knob's module constant); None keeps the value the
        # environment gave it at import (the kill-switch worker). sent: the yT
        # sentinel byte of this arm. w_ln: a LayerNorm weight override (D9).
        if on is not None:
            tf8._HSTU_FP8_FWD_K512_TILED_A = on
        w_ln = ln_w if w_ln is None else w_ln
        for p in list(params.values()) + [w_ln]:
            p.grad = None
        events.clear()
        dirty_allocator()
        sentinel[0] = sent
        x = x_base.detach().requires_grad_()
        c0 = compile_state()
        outs = call(x, offs, nt, w_ln)
        rec: Dict[str, Any] = {}
        ctx = outs[0].grad_fn
        if digest:
            for i, o in enumerate(outs):
                rec[f"fwd.out{i}"] = _digest(o)
            # Inputs saved as themselves are recorded by name (same storage).
            input_ptrs = {
                t.data_ptr(): name
                for name, t in (("x", x), ("ln_w", w_ln), ("ln_b", ln_b),
                                ("uvqk_w", uvqk_w), ("uvqk_b", uvqk_b), ("offsets", offs))
            }
            for i, t in enumerate(ctx.saved_tensors):
                name = input_ptrs.get(t.data_ptr())
                rec[f"saved.{i}"] = f"input:{name}" if name else _digest(t)
            rec["ctx.stored_vk_x_fp8_t"] = _digest(getattr(ctx, "_stored_vk_x_fp8_t", None))
            rec["ctx.ephemeral_fp8_vk"] = _digest(getattr(ctx, "_ephemeral_fp8_vk", None))
            rec["ctx.scalars"] = {k: repr(getattr(ctx, k, None)) for k in _CTX_SCALARS}
        g = torch.Generator(device=dev).manual_seed(7_000_003 + m)
        grads = [
            (torch.randn(o.shape, generator=g, device=dev) * 0.05).to(o.dtype) for o in outs
        ]
        fwd_events = [list(e) for e in events]
        torch.autograd.backward(list(outs), grads)
        torch.cuda.synchronize()
        c1 = compile_state()
        if digest:
            rec["grad.x"] = _digest(x.grad)
            for name, p in params.items():
                rec[f"grad.{name}"] = _digest(w_ln.grad if name == "ln_w" else p.grad)
        rec["events"] = fwd_events
        rec["compiles"] = {"k512_builds": c1[0] - c0[0],
                           "triton_new": sorted(c1[1] - c0[1])}
        del outs, ctx, grads, x
        return rec

    # bf16 warmup until the vk_projection backward site publishes (training's
    # first steps), so the forward then takes the fp8t store_t branch.
    offs_w, nt_w = offsets_for(WARM_M)
    warm_calls = 0
    vk_w_view = uvqk_w.narrow(1, 0, v_dim + DQK * HEADS)

    def fp8t_will_run(rows: int) -> bool:
        return bwd_fp8.uvqk_fp8t_will_run(
            "vk_projection", vk_w_view, rows=rows, dtype=bf16,
            input_scale_value=float(in_state[0]), per_layer=False,
        )

    warm_arm = None if cfg["mode"] == "kill" else False
    while not fp8t_will_run(WARM_M) and warm_calls < 16:
        run(WARM_M, x_warm, offs_w, nt_w, on=warm_arm, digest=False)
        warm_calls += 1
    report["warm_calls"] = warm_calls
    report["fp8t_will_run"] = fp8t_will_run(WARM_M)
    report["tiled_active"] = wgrad_ck.tiled_active(dev)
    if not report["fp8t_will_run"]:
        print("B15T_JSON " + json.dumps(report), flush=True)
        raise SystemExit("vk_projection fp8t never armed: the store_t branch is unreachable")
    del x_warm

    # b15t_row_major_y == the store_t's own y, 2-D and step-tiled yT.
    rebuild = []
    for m in (1, 129, 1000):
        xx = make_x(m)
        for tiled in (False, True):
            y, _, _, yt = orig_store_t(
                xx, ln_w.detach(), ln_b.detach(), 1e-6, 1.0 / float(in_state[0]),
                pad_to=bwd_fp8.uvqk_tm_pad(), tiled_xt=tiled,
            )
            ry = orig_rebuild(yt, m)
            rebuild.append([m, tiled, yt.dim(), bool(torch.equal(ry.view(torch.uint8), y.view(torch.uint8)))
                            and tuple(ry.shape) == tuple(y.shape) and ry.is_contiguous()])
    report["rebuild"] = rebuild

    results: Dict[str, Dict[str, Any]] = {}
    m_list = [int(v) for v in cfg["m"]]
    for m in m_list:
        x_base = make_x(m)
        offs, nt = offsets_for(m)
        res: Dict[str, Any] = {"B": int(nt.numel())}
        arms = cfg["arms"]
        for arm in arms:
            on = {"off": False, "on": True, "on2": True, "env": None}[arm]
            if arm == "on2" and m not in cfg.get("aa_m", []):
                continue
            sent = {"off": 0xA5, "on": 0x5A, "on2": 0x3C, "env": 0x96}[arm]
            res[arm] = run(m, x_base, offs, nt, on=on, sent=sent)
            print(f"[b15t-test] M={m} arm={arm} events={res[arm]['events']}", flush=True)
        results[str(m)] = res
        del x_base, offs, nt
        torch.cuda.empty_cache()
    report["results"] = results

    forced: Dict[str, Any] = {}
    latches: Dict[str, Any] = {}
    if cfg.get("forced"):
        print("[b15t-test] === forced declines from here on", flush=True)

        def force_attr(mod: Any, attr: str, fn: Any) -> Any:
            def enter() -> Any:
                saved = getattr(mod, attr)
                setattr(mod, attr, fn)
                return lambda: setattr(mod, attr, saved)

            return enter

        def force_write_y_on() -> Any:
            force_write_y[0] = True
            return lambda: force_write_y.__setitem__(0, False)

        # name -> (enter() returning restore(), latches B15T)
        cases = {
            # after the y skip: y rebuilt from yT, B15T latched off
            "k512_operand": (force_attr(k512, "ineligible_reason_tiled_a",
                                        lambda *a, **k: ("forced", "test: K512 tiled-A operand decline")),
                             True),
            "launch": (force_attr(k512, "_tiled_a_call", lambda *a, **k: 98), True),
            "store_t_wrote_y": (force_write_y_on, True),
            # before the LayerNorm: y written, no latch
            "storet_precheck": (force_attr(tpa, "store_t_skip_y_decline_reason",
                                           lambda *a, **k: ("forced_storet", "test: store_t pre-check decline")),
                                False),
            "not_ready": (force_attr(k512, "tiled_a_ready", lambda *a, **k: False), False),
        }
        for m in [int(v) for v in cfg["forced_m"]]:
            x_base = make_x(m)
            offs, nt = offsets_for(m)
            for name, (enter, latches_b15t) in cases.items():
                restore = enter()
                try:
                    forced[f"{name}:{m}"] = run(m, x_base, offs, nt, on=True, sent=0x69)
                finally:
                    restore()
                latched = tf8.b15t_latched()
                latches[f"{name}:{m}"] = latched
                if latches_b15t:
                    # B-4: the next call (nothing forced) is decided before the
                    # LayerNorm: y written, row-major V/K.
                    forced[f"latched_after_{name}:{m}"] = run(m, x_base, offs, nt, on=True, sent=0x6A)
                tf8._B15T_LATCHED = None
                print(f"[b15t-test] forced {name} M={m} latched={latched!r} "
                      f"events={forced[f'{name}:{m}']['events']}", flush=True)
            del x_base, offs, nt
    report["latches"] = latches
    # D9 rows (r16; r14 B-6): +-Inf / NaN / huge / constant x rows and +-Inf
    # LayerNorm weight columns, off vs on (after the forced section: the B15T
    # latch is clear; any error must be the same in both arms).
    d9: Dict[str, Any] = {}
    if cfg.get("d9_m"):
        w_inf = ln_w.detach().clone()
        w_inf[3] = float("inf")
        w_inf[300] = float("-inf")
        w_inf.requires_grad_()
        big = torch.finfo(bf16).max
        for m in [int(v) for v in cfg["d9_m"]]:
            x_base = make_x(m)
            rows = [r for r in (0, 1, 2, 3, 4, 5, 6, m // 2, m - 2, m - 1) if 0 <= r < m]
            fills = [float("inf"), float("-inf"), float("nan"), big, 5.0, None, None, -big, 0.0, float("-inf")]
            for r, f in zip(rows, fills):
                if r == 5:
                    x_base[r, 7] = float("inf")
                elif r == 6:
                    x_base[r, 11] = float("nan")
                elif f is not None:
                    x_base[r] = f
            if m > 3:
                x_base[3, ::2] = -big
            offs, nt = offsets_for(m)
            for case, w in (("rows", None), ("rows_infw", w_inf)):
                for arm, on, sent in (("off", False, 0xA5), ("on", True, 0x5A), ("off2", False, 0x3C)):
                    key = f"{case}:{m}:{arm}"
                    try:
                        d9[key] = run(m, x_base, offs, nt, on=on, sent=sent, w_ln=w)
                    except Exception as error:  # noqa: BLE001 - recorded, compared
                        d9[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}",
                                   "events": [list(e) for e in events]}
                    print(f"[b15t-test] D9 {key} events={d9[key]['events']} "
                          f"error={d9[key].get('error')}", flush=True)
            del x_base, offs, nt
            torch.cuda.empty_cache()
    report["d9"] = d9
    report["forced"] = forced
    print("B15T_JSON " + json.dumps(report), flush=True)


def _run_worker(cfg: Dict[str, Any], env_extra: Dict[str, str], out_dir: Path, tag: str
                ) -> Tuple[Dict[str, Any], str]:
    cfg_path = out_dir / f"{tag}_cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    r = subprocess.run(
        [sys.executable, "-B", str(_HERE), "--b15t-worker", str(cfg_path)],
        env=_clean_env(env_extra),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=WORKER_TIMEOUT_S,
    )
    (out_dir / f"{tag}_stdout.txt").write_text(r.stdout)
    if r.returncode != 0:
        raise AssertionError(f"worker {tag} rc={r.returncode}:\n{r.stdout[-6000:]}")
    report = _last_json(r.stdout)
    (out_dir / f"{tag}_report.json").write_text(json.dumps(report, indent=1))
    return report, r.stdout


def _gpu_skip_reason() -> Optional[str]:
    if not _have_gin():
        return "needs gin-config (the worker runs the shipped gin's bootstrap)"
    try:
        import torch
    except Exception as error:  # pragma: no cover - environment specific
        return f"needs torch ({error})"
    if torch.version.hip is None or not torch.cuda.is_available():
        return "needs a ROCm GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists(HIPCC):
        return f"needs {HIPCC}"
    return None


_HASH_KEYS_SKIP = ("events", "compiles")


def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = sorted((set(a) | set(b)) - set(_HASH_KEYS_SKIP))
    return [k for k in keys if a.get(k) != b.get(k)]


class TargetsVKParityTest(unittest.TestCase):
    report: Dict[str, Any] = {}
    stdout: str = ""
    kill: Dict[str, Any] = {}
    kill_stdout: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        reason = _gpu_skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        keep = os.environ.get("B15T_TEST_OUT")
        cls._tmp = None if keep else tempfile.mkdtemp(prefix="b15t_test_")
        out_dir = Path(keep or cls._tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        m_env = os.environ.get("B15T_TEST_M")
        m_list = [int(v) for v in m_env.split(",")] if m_env else list(EDGE_M + PROD_M)
        ab_cfg = {"mode": "ab", "m": m_list, "arms": ["off", "on", "on2"],
                  "aa_m": [m for m in AA_M if m in m_list], "forced": True,
                  "forced_m": [m for m in FORCED_M if m in m_list] or [m_list[0]],
                  "d9_m": [m for m in D9_M if m in m_list] or [m_list[-1]]}
        cls.report, cls.stdout = _run_worker(ab_cfg, {}, out_dir, "ab")
        kill_m = [m for m in KILL_M if m in m_list] or [m_list[0]]
        kill_cfg = {"mode": "kill", "m": kill_m, "arms": ["env"], "forced": False}
        cls.kill, cls.kill_stdout = _run_worker(kill_cfg, {KNOB: "0"}, out_dir, "kill")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_tmp", None):
            shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_production_env_and_arming(self) -> None:
        env = self.report["env"]
        self.assertEqual(env[KNOB], "1")
        for k in ("DLRM_HSTU_FP8_FWD_K512", "DLRM_HSTU_FP8_BWD_STORE_T_V2",
                  "DLRM_HSTU_FP8_BWD_UVQK_STORE_T", "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED"):
            self.assertEqual(env[k], "1", k)
        self.assertTrue(self.report["knob_requested"])
        self.assertTrue(self.report["fp8t_will_run"])
        self.assertTrue(self.report["tiled_active"])
        # prebuild_extensions ran the tiled-A self-check (the worker sees one GPU).
        self.assertEqual(self.report["tiled_a_ok"], [[0, 1024]])
        self.assertEqual(self.stdout.count("[k512-fwd] tiled-A self-check OK: N=1024"), 1)
        self.assertEqual(self.stdout.count("[b15t] ARMED targets VK reads tiled yT; store_t skips y"), 1)
        plain = self.stdout.split("[b15t-test] === forced declines")[0]
        self.assertNotIn("[b15t] declined", plain)
        self.assertNotIn("[b15t] LATCHED OFF", plain)
        self.assertNotIn("[k512-fwd] tiled-A OFF", self.stdout)
        self.assertNotIn("tiled-A prebuild FAILED", self.stdout)

    def test_on_arm_compiles_nothing(self) -> None:
        # The store_y=False store_t call reuses the binary the off arm's
        # store_y=True call compiled at the same M (same kernel, same pointer
        # classes and stride specializations); the tiled-A entry was built by
        # prebuild_extensions.
        for m, res in self.report["results"].items():
            for arm in ("on", "on2"):
                if arm in res:
                    self.assertEqual(res[arm]["compiles"], {"k512_builds": 0, "triton_new": []},
                                     f"M={m} {arm}")

    def test_on_matches_off_bitwise(self) -> None:
        for m, res in self.report["results"].items():
            off, on = res["off"], res["on"]
            self.assertEqual(_diff(off, on), [], f"M={m}: on != off")
            if "on2" in res:
                self.assertEqual(_diff(on, res["on2"]), [], f"M={m}: on A/A")
            # The comparison covers the B15T-relevant tensors.
            for k in ("ctx.stored_vk_x_fp8_t", "ctx.ephemeral_fp8_vk", "grad.x", "grad.uvqk_w",
                      "grad.uvqk_b", "grad.ln_w", "grad.ln_b", "fwd.out0", "fwd.out1"):
                self.assertIsNotNone(on.get(k), f"M={m}: {k} missing")

    def test_arms_took_their_paths(self) -> None:
        for m, res in self.report["results"].items():
            mp_blocks = (int(m) + 127) // 128
            self.assertEqual(
                res["on"]["events"],
                [["store_t", False, True, [mp_blocks, 512, 128]], ["vk_tiled", True]],
                f"M={m} on",
            )
            self.assertEqual(
                res["off"]["events"],
                [["store_t", True, False, [mp_blocks, 512, 128]], ["vk_row_major"]],
                f"M={m} off",
            )

    def test_forced_declines_bitwise(self) -> None:
        forced = self.report["forced"]
        self.assertTrue(forced)
        for key, rec in forced.items():
            name, m = key.split(":")
            off = self.report["results"][m]["off"]
            self.assertEqual(_diff(off, rec), [], f"forced {key} != off")
            mp_blocks = (int(m) + 127) // 128
            shape = [mp_blocks, 512, 128]
            post_skip = [["store_t", False, True, shape], ["vk_tiled", False], ["y_rebuilt"],
                         ["vk_row_major"]]
            pre_ln = [["store_t", True, False, shape], ["vk_row_major"]]
            expected = {
                "k512_operand": post_skip,
                "launch": post_skip,
                "store_t_wrote_y": [["store_t", False, False, shape], ["vk_row_major"]],
                "storet_precheck": pre_ln,
                "not_ready": pre_ln,
            }.get(name, pre_ln)  # latched_after_*: decided before the LayerNorm
            self.assertEqual(rec["events"], expected, key)
        latches = self.report["latches"]
        for key, latched in latches.items():
            name = key.split(":")[0]
            if name in ("k512_operand", "launch", "store_t_wrote_y"):
                self.assertIsNotNone(latched, key)
                self.assertTrue(latched.startswith({"k512_operand": "forced:", "launch": "launch:",
                                                    "store_t_wrote_y": "store_t:"}[name]), latched)
            else:
                self.assertIsNone(latched, key)
        self.assertIn("hipError 98", latches[next(k for k in latches if k.startswith("launch:"))])
        # N1: one line per (kind, detail); the forced details are constant.
        for kind in ("(forced)", "(forced_storet)", "(not_ready)", "(store_t)"):
            self.assertEqual(
                len(re.findall(r"\[b15t\] declined -> row-major y \+ V/K projection " + re.escape(kind),
                               self.stdout)), 1, kind)
        for kind in ("(launch)", "(latched)"):
            self.assertGreaterEqual(
                len(re.findall(r"\[b15t\] declined -> row-major y \+ V/K projection " + re.escape(kind),
                               self.stdout)), 1, kind)
        self.assertGreaterEqual(self.stdout.count("[b15t] LATCHED OFF"), 1)

    def test_d9_rows_bitwise(self) -> None:
        d9 = self.report["d9"]
        self.assertTrue(d9)
        cases = sorted({k.rsplit(":", 1)[0] for k in d9})
        for case in cases:
            off, on, off2 = d9[f"{case}:off"], d9[f"{case}:on"], d9[f"{case}:off2"]
            m = int(case.split(":")[1])
            self.assertEqual(off.get("error"), on.get("error"), case)
            if "error" in on:
                continue
            mp_blocks = (m + 127) // 128
            self.assertEqual(on["events"], [["store_t", False, True, [mp_blocks, 512, 128]],
                                            ["vk_tiled", True]], case)
            fwd = [k for k in off if k.startswith(("fwd.", "saved.", "ctx."))]
            self.assertTrue(fwd)
            self.assertEqual([k for k in fwd if off.get(k) != on.get(k)], [], f"D9 {case}: forward")
            # Gradients: on == off, unless the off arms disagree with each other
            # (process state moved by non-finite data; then on == off2).
            ref = off if _diff(off, off2) == [] else off2
            self.assertEqual(_diff(ref, on), [], f"D9 {case}: on != off (off A/A clean: {ref is off})")

    def test_row_major_rebuild(self) -> None:
        self.assertTrue(self.report["rebuild"])
        for m, tiled, dim, ok in self.report["rebuild"]:
            self.assertEqual(dim, 3 if tiled else 2)
            self.assertTrue(ok, f"M={m} tiled={tiled}")

    def test_kill_switch(self) -> None:
        kill = self.kill
        self.assertEqual(kill["env"][KNOB], "0")
        self.assertFalse(kill["knob_requested"])
        self.assertEqual(kill["tiled_a_ok"], [])
        self.assertNotIn("[b15t]", self.kill_stdout)
        self.assertNotIn("tiled-A self-check", self.kill_stdout)
        for m, res in kill["results"].items():
            mp_blocks = (int(m) + 127) // 128
            self.assertEqual(res["env"]["events"],
                             [["store_t", True, False, [mp_blocks, 512, 128]], ["vk_row_major"]], m)
            ab = self.report["results"][m]
            self.assertEqual(_diff(ab["off"], res["env"]), [], f"M={m}: kill switch != off (cross-process)")
            self.assertEqual(_diff(ab["on"], res["env"]), [], f"M={m}: kill switch != on (cross-process)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--b15t-worker":
        _worker(json.loads(Path(sys.argv[2]).read_text()))
    else:
        unittest.main()
