# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""CPU tests of r14's MX6 dY producer seam (r14 iteration 66, DESIGN v1 sec. 11 / 13; r16's S1 contract sec. 9): the
EOF append of ops/triton/bwd_fp8.py (S1 output_projection_dgrad_operands_mx6, R2 output_projection_g8_rescue, S1-E
output_projection_mx6_encode, the R2 stash and the helpers) and the W1 cache of ops/hip/outproj_dual_mx6.py, checked
against the FP8 path they replace and against r16's MX6 rows (ops/hip/l11_do8.py: the result types, mx6_produce,
mx6_rescue, the field checks, the prebuild INERT rule). No GPU, no hipcc, no Triton launch.

REAL code under test: S1 / R2 / S1-E and the helpers; the out-proj _SiteScale state machine (publish incl.
DFMT_SHIFT=auto); the FP8 path S1 mirrors (output_projection_dgrad_operands -> _quantize_gradient_and_weight_fused,
the warm-up in quantize_gradient, quantize_weight, record_engaged); the Python of _fused_site_launch and
_fused_weight_quant; fp8_tm_alloc.empty_tm (a plain torch.empty off CUDA); the loader's launch, _check_rc,
ineligible_reason, not_armed_reason, w6_shadow (the W1 cache), w6_encode, w6_ineligible_reason and latch; r16's
checkers and mx6_produce / mx6_rescue / prebuild_mx6.

STUBBED, device work only (the FP8 path and S1 run on the SAME stubs, so any state difference is a seam difference):
  * the two ctypes entries of the HIP .so (a fake library behind the real loader): the producer reads INV through its
    pointer and writes deterministic G6 / dout_t / PARTIAL bytes (dout_t = the torch reference of the dual cast's
    DST_T at the INV bits and T_INV_MUL it was given; PARTIAL = the scripted one) through the given pointers, INV_USED
    when non-null; W1 writes bytes derived from the weight values. The loader's metadata checks run first on the same
    tensors, so every write stays inside them;
  * the Triton launches behind quantize_row_major_dual and quantize_row_major (recording torch references; the dual
    cast returns the same scripted PARTIAL);
  * fp8_helpers_fused.weight_amax_with_site (a torch reference of the site program, program 0 of
    _fp8h_amax_tiles_kernel, plus the |w| tile maxima) and weight_quant_from_partials (_fp8h_weight_quant_kernel);
  * torch.cuda.current_device / device / current_stream; the loader's two metadata checks and bwd_fp8's
    _helpers_fused_ok see the CPU tensors through a view whose .device / .is_cuda say cuda:0 (every other check runs as
    written).

  T1   line neutrality: every @triton.jit line and kernel source of bwd_fp8.py / fp8_helpers_fused.py equals the base;
       bwd_fp8.py is the base file plus a pure EOF append (S-b landing); the API sits after the last kernel and imports
       l11_do8 / outproj_dual_mx6 lazily (S1 contract item 7). v1.2: the base is one of two allow-listed files, r16
       commit 7c0bbd7's (b5bb287a) or ship16 544e691's (e2655b82, E-RM rows inside BF); T3's FP8 reference call then
       carries ship16's erm=None.
  T2   API presence: l11_do8._mx6_producer_missing() == []; a fresh `import bwd_fp8` imports no ops.hip module; r16's
       prebuild_mx6 no longer goes INERT on the producer API and latches at the build step on a CPU host (item 1).
  T3   bookkeeping: two published sites, the FP8 path on one and S1 on the other with the same PARTIAL, 26 calls and
       5-9 refresh windows per configuration (SHIFT auto with up / down steps, a first refresh at t_calls 0; NaN / Inf /
       zero partials, e5m2 and e4m3 clipping, W1 refreshes; fixed shifts 5 / 7; no t_clipped_windows): all 11 site
       fields bitwise after every call, the snapshots, the site-launch call records, the INV bits each cast read, dout_t
       bytes, the device order (items 2, 3).
  T4   decline matrix: every per-call row -> None twice, nothing allocated, cloned, launched or changed, one line per
       reason class; a missing site mints nothing; a warm-up site is not accumulated (item 3). v1.1 (F1): one row with
       row stride 1024 (is_contiguous() for a size-1 dim) declines, r16 releases it without a latch, S1-E copies it.
  T5   persistent raises: the .so not prebuilt / latched, the configuration, the weight, r16's types, t_shift -> raise
       OutprojMx6Unavailable before anything (items 3, 7b).
  T6   commit boundary: pre-commit failures (producer rc > 0 or raise, W1 rc > 0, allocation, clone, empty_tm, launch
       arguments) re-raise unchanged with the state untouched; a stale rc < 0 proceeds; a raise inside the site launch
       or publish has calls delta 1 (r16: Mx6ContractError) (items 3, 4).
  T7   Mx6Operands form: r16's types and checkers, its own inv_scale_used with the pre-publish bits (item 2).
  T8   R2: the FP8 path's dual-cast call at the CAPTURED inv bits and cast-time shift (after AUTO steps), no state
       change, ops6.dout_fp8 untouched, through l11_do8.mx6_rescue (item 5).
  T9   TSPEC key model on R2's / S1's recorded calls (AMD 'D' / 'S' rules at production M).
  T10  W1 cache: hit, _version / data_ptr / identity / dead-weakref refreshes into NEW buffers, pruning, dtypes, latch;
       S1-E leaves it (and every other state) untouched.
  T17  R2 stash: an identity hit is _fused_weight_quant(W, S1's w_partial) with the same objects; any miss is
       quantize_weight plus one line; reset_mx6_for_testing() clears it. v1.2 (r16 iteration 59 R-1 C3, S1 contract
       v3 note): the stash also keys on W._version and W.data_ptr(), captured in S1's prepare block after W1 (a raise
       there is pre-commit); the key rows: nothing between S1 and R2 -> hit; torch.no_grad() W.mul_(1.0001), a raw
       write + increment_version(W), increment_version(W) alone, and W.data = ... (new storage at another address)
       -> miss + the line, w8 the cast of the current values; two documented limits -> hit with S1's stale partials
       (no key sees them; W1 has the same limit): a raw write without a version bump, and W.data swapped away and
       back to S1's storage with a write through a .data alias (set_data keeps _version, data_ptr is S1's again).
       Every row pins S1's storage, so no freed weight address is reused by a later row's weight.
  T18  (seam part, v1.2) stash-key mutants built from R2's own source (an AST edit): a key without _version is caught
       by the (a) / (b) / (b') rows, one without data_ptr() by (c), v1.1's identity-only key by all four.
  r16 item 4 (mx6_produce) and item 5 (mx6_rescue) run through r16's real code. T11 / T12 are the loader's own test
  file; item 6 (r16's suites) is run separately."""

import ast
import collections
import contextlib
import ctypes
import gc
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import torch

from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.hip import outproj_dual_mx6 as odm
from generative_recommenders.ops.triton import bwd_fp8, fp8_tm_alloc

try:  # imports triton (the stateful tests stub its launches; without triton they are skipped)
    from generative_recommenders.ops.triton import fp8_helpers_fused as fhf
except Exception as _fhf_error:  # noqa: BLE001
    fhf = None
    _FHF_SKIP = f"fp8_helpers_fused not importable: {type(_fhf_error).__name__}: {_fhf_error}"
else:
    _FHF_SKIP = ""

E4, E5, BF, F32, U8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8
CPU = torch.device("cpu")
D, NOUT, G6_ROW, W6_BYTES, W6S_BYTES = 512, 1536, 400, 589824, 768
API = ("output_projection_dgrad_operands_mx6", "output_projection_g8_rescue", "output_projection_mx6_encode")
FIELDS = ("calls", "t_calls", "scale", "inv_scale", "pending", "clipped_windows", "t_clipped_windows", "published",
          "t_shift", "_t_clipped_seen", "_t_clean_windows")
ABSENT = ("MX6 DISABLED", "MX6 INERT", "MX6 release", "MX6 rescue", "DEVICE FAULT")  # r16's must-be-ABSENT tokens
_BF_PATH = Path(bwd_fp8.__file__).resolve()
_OPS = _BF_PATH.parents[1]  # generative_recommenders/ops
_ROOT = _OPS.parents[1]  # the recommendation/ tree (the import root)
_FHF_PATH = _OPS / "triton" / "fp8_helpers_fused.py"
_TSP_PATH = _OPS / "triton" / "triton_spec_table.jsonl"
_LOADER = "generative_recommenders.ops.hip.outproj_dual_mx6"


def _m_pad(m: int) -> int:
    return (m + 127) // 128 * 128


def _n_partial(m: int) -> int:
    return 4 * ((_m_pad(m) + 255) // 256)


def _bits(t: Any) -> Any:
    """Bit-level value of a small tensor (fp32 / int64 / bool / bytes), or the object itself."""
    if not isinstance(t, torch.Tensor):
        return t
    t = t.detach().reshape(-1).contiguous()
    if t.dtype in (torch.float32, torch.int32):
        return t.view(torch.int32).tolist()
    if t.dtype == torch.int64:
        return t.tolist()
    return t.view(torch.uint8).tolist()


def _fields(state: Any) -> Dict[str, Any]:
    """The 11 per-site fields SPEC_SEAM 2.2 / the S1 contract compare, bit for bit."""
    return {f: _bits(getattr(state, f)) for f in FIELDS}


def _peek(weight: torch.Tensor) -> Any:
    return bwd_fp8._peek_site_state("output_projection", weight.data_ptr())


def _snapshot() -> Dict[str, Any]:
    """Everything an S1 decline / raise, R2 or S1-E must leave as it was: every site's 11 fields, the site labels,
    record_engaged's and _log_armed's once-sets, the R2 stash, the W1 cache and the loader's latch."""
    stash = bwd_fp8._MX6_RESCUE_STASH
    return {
        "sites": {k: _fields(v) for k, v in sorted(bwd_fp8._sites.items())},
        "layer_index": sorted((k[0], k[1], v) for k, v in bwd_fp8._layer_index.items()),
        "engaged": sorted(bwd_fp8._engaged_sites),
        "armed_logged": sorted(repr(k) for k in bwd_fp8._uvqk_armed_logged),
        "logged_resolved": bwd_fp8._logged_resolved,
        # the stash's tensors by identity, its key ints (v1.2: W._version, W.data_ptr()) by value
        "stash": None if stash is None else tuple(id(x) if isinstance(x, torch.Tensor) else x for x in stash),
        "w6_cache": sorted((repr(k), id(e.w6), id(e.w6s), e.version, e.data_ptr) for k, e in odm._w6_cache.items()),
        "w6_stats": odm.w6_cache_stats(),
        "loader": (odm.failure(), odm.prebuilt()),
    }


class _Capture(contextlib.AbstractContextManager):
    def __enter__(self):
        self.buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self.buf)
        self._cm.__enter__()
        return self

    def __exit__(self, *a):
        self._cm.__exit__(*a)
        return False

    @property
    def text(self) -> str:
        return self.buf.getvalue()


class _CudaView(torch.Tensor):
    """A view of a CPU tensor whose .device / .is_cuda say cuda:0 (same storage, data_ptr, dtype, shape, strides): the
    loader's metadata checks and bwd_fp8._helpers_fused_ok run as written on the CPU tensors of these tests."""

    @property
    def device(self) -> torch.device:  # pyre-ignore[3]
        return torch.device("cuda", 0)

    @property
    def is_cuda(self) -> bool:  # pyre-ignore[3]
        return True


def _cuda_view(t: Any) -> Any:
    return t.detach().as_subclass(_CudaView) if isinstance(t, torch.Tensor) else t


@contextlib.contextmanager
def _alloc_spy(counts: Dict[str, int]):  # pyre-ignore[3]
    """Count every torch allocation entry point and Tensor.clone while the block runs."""
    names = ("empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like", "empty_strided",
             "tensor")
    with contextlib.ExitStack() as st:
        for name in names:
            real = getattr(torch, name)

            def wrap(*a: Any, _real: Callable[..., Any] = real, _name: str = name, **k: Any) -> Any:
                counts[_name] = counts.get(_name, 0) + 1
                return _real(*a, **k)

            st.enter_context(mock.patch.object(torch, name, wrap))
        real_clone = torch.Tensor.clone

        def clone(self_: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
            counts["clone"] = counts.get("clone", 0) + 1
            return real_clone(self_, *a, **k)

        st.enter_context(mock.patch.object(torch.Tensor, "clone", clone))
        yield counts


def _outproj_lines(text: str) -> List[str]:
    """The [outproj-mx6] lines printed (arm_print_all_ranks' " [rank N]" tag dropped)."""
    return [re.sub(r" \[rank \d+\]$", "", ln.strip()) for ln in text.splitlines() if "outproj-mx6" in ln]


# ------------------------------------------------------------------------------------------------ torch references
def _f32_of_bits(bits_u32: int) -> torch.Tensor:
    return torch.from_numpy(np.array([bits_u32], dtype=np.uint32).view(np.float32).copy())


def _ref_cast(value: torch.Tensor, inv: torch.Tensor, dtype: torch.dtype, limit: float) -> torch.Tensor:
    """The row-major cast: dtype(clamp(fp32(x) * inv, +-limit))."""
    return (value.detach().float() * inv.detach().float().reshape(())).clamp(-limit, limit).to(dtype)


def _ref_dout_t(value: torch.Tensor, inv: torch.Tensor, t_inv_mul: float, m_pad: int) -> torch.Tensor:
    """The dual cast's step-tiled DST_T: e4m3(clamp((fp32(x) * inv) * 2^-t, +-448)), pad tokens through the same
    arithmetic on +0.0, [m_pad/128, 512, 128] (step k = tokens [128k, 128k + 128) of every column)."""
    m = int(value.shape[0])
    x = torch.zeros((m_pad, D), dtype=F32)
    x[:m] = value.detach().float()
    st = (x * inv.detach().float().reshape(())) * float(t_inv_mul)
    q = st.clamp(-448.0, 448.0).to(E4)
    return q.view(m_pad // 128, 128, D).permute(0, 2, 1).contiguous()


def _ref_site_program(site: Dict[str, Any]) -> None:
    """Program 0 of _fp8h_amax_tiles_kernel (fp8_helpers_fused): amax = NaN if any PARTIAL entry is NaN, else the max
    with 0; note_clipping; note_t_clipping iff T_CLIPPED; accumulate (NaN-sticky); the two snapshots."""
    p = site["partial"].reshape(-1).float()
    if bool(torch.isnan(p).any()):
        amax = torch.tensor(float("nan"), dtype=F32)
    else:
        amax = torch.clamp(p.max(), min=0.0)
    s = site["scale"].reshape(())
    site["clipped"].add_((amax > s * float(site["grad_max"])).to(torch.int64))
    if site["t_clipped"] is not None:
        site["t_clipped"].add_((amax > s * float(site["t_factor"])).to(torch.int64))
    pend = site["pending"]
    newp = torch.where(torch.isnan(pend), pend, torch.where(torch.isnan(amax), amax, torch.maximum(pend, amax)))
    pend.copy_(newp.reshape(pend.shape))
    if site["snap"] is not None:
        site["snap"].copy_(s.reshape(1))
    if site["snap_t"] is not None:
        site["snap_t"].copy_((s * float(site["t_mul"])).reshape(1))


def _ref_weight_tiles(weight: torch.Tensor) -> torch.Tensor:
    """The |w| maxima of the 32 x 256 tiles (NaN-propagating), in the kernel's tile order."""
    w = weight.detach()
    if w.dim() != 2:
        w = w.reshape(w.shape[0], -1)
    r, c = int(w.shape[0]), int(w.shape[1])
    nr, nc = -(-r // 32), -(-c // 256)
    a = torch.zeros((nr * 32, nc * 256), dtype=F32)
    a[:r, :c] = w.float().abs()
    return a.view(nr, 32, nc, 256).amax(dim=(1, 3)).reshape(-1).contiguous()


_RECIP_448 = float(torch.div(torch.tensor([1.0]), torch.tensor([448.0])).item())


def _ref_weight_quant(weight: torch.Tensor, partial: torch.Tensor, fp8_dtype: torch.dtype
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """_fp8h_weight_quant_kernel: scale = clamp_min(amax(partials), 1e-30) * fp32(1/448), x * (1 / scale) clamped."""
    src = weight.detach()
    assert src.is_contiguous()
    p = partial.reshape(-1).float()
    amax = torch.tensor(float("nan")) if bool(torch.isnan(p).any()) else torch.clamp(p.max(), min=0.0)
    lo = torch.tensor(1.0e-30, dtype=F32)
    scale = torch.where(torch.isnan(amax), amax, torch.maximum(amax, lo)) * _RECIP_448
    inv = 1.0 / scale
    out = (src.float() * inv).clamp(-448.0, 448.0).to(fp8_dtype)
    return out.view_as(src), scale.reshape(1)


def _w6_bytes_of(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The fake W1 kernel's output: bytes derived from the fp32-widened weight values (deterministic in the values)."""
    b = w.detach().float().contiguous().view(U8).reshape(-1)
    return torch.bitwise_xor(b[:W6_BYTES], 0x5A).contiguous(), b[-W6S_BYTES:].clone()


def _partial(m: int, kind: str, amax: float, seed: int) -> torch.Tensor:
    """A scripted PARTIAL of `m` rows (4 ceil(m_pad / 256) entries): max `amax`; kind nan / inf adds one such entry,
    zero is all +0.0."""
    n = _n_partial(m)
    if kind == "zero":
        return torch.zeros(n, dtype=F32)
    g = torch.Generator().manual_seed(1000 + seed)
    p = torch.rand(n, generator=g, dtype=F32) * float(amax) * 0.5
    i = int(torch.randint(n, (1,), generator=g))
    p[i] = float(amax)
    if kind == "nan":
        p[(i + 1) % n] = float("nan")
    if kind == "inf":
        p[(i + 2) % n] = float("inf")
    return p


# ------------------------------------------------------------------------------------------------ T1 static
# The base = r16 commit 7c0bbd7's ops/triton/bwd_fp8.py = b23945a's (ship15.1), byte for byte (SPEC_SEAM sec. 5), or
# (v1.2; the tester's F-1) the pushed ship16 commit 544e691's file: E-RM's rows edit BF inside existing functions and
# move no kernel line, and the EOF append is the same bytes on both. Per base: (sha256, bytes, lines, the extra
# keyword arguments its FP8 path passes to _quantize_gradient_and_weight_fused; ship16: erm=None with E-RM off).
_BASES: Dict[str, Tuple[str, int, int, Dict[str, Any]]] = {
    "7c0bbd7": ("b5bb287aa19c5f2ff62ca168b907a8029af5dc04b3bf02b46454cbb6084008cb", 112715, 2760, {}),
    "544e691": ("e2655b82e3c6d08c31a8f0176adc2689a3975aa5e75d077600d1afc4bbae3641", 115338, 2824, {"erm": None}),
    # ROCm 10.2 port (fp8-gemm-pins): 544e691's file with its two FP8 dgrad call sites (uvqk / targets-only V/K) as
    # blas_solution_pins.scaled_mm( -- same lines, so no kernel moves; the module import is appended at EOF.
    "544e691+fp8pins": ("6c52aed95ad7dddeb21b15064390a45be0baf7ab30a6ea53430c4a713a85ece1", 115456, 2824, {"erm": None}),
}


def _base_of(raw: bytes) -> Optional[str]:
    """The allow-listed base whose bytes `raw` starts with (an append must follow), else None."""
    for name, (sha, nbytes, _, _) in _BASES.items():
        if len(raw) > nbytes and hashlib.sha256(raw[:nbytes]).hexdigest() == sha:
            return name
    return None


_BASE = _base_of(_BF_PATH.read_bytes())
# an unknown base keeps 7c0bbd7's pins: the T1 append test then fails on the hash, and T3 on the FP8 call's kwargs
_BASE_BF_SHA256, _BASE_BF_BYTES, _BASE_BF_LINES, _FP8_REF_EXTRA_KWARGS = _BASES[_BASE or "7c0bbd7"]
_APPEND_MARK = "# r14 iteration 66: the MX6 dY producer seam of r16's MX6 row"
# (decorator line, def line, end line, name, sha256[:16] of the lines decorator..end): seam_evidence.py's jit_lines,
# the baseline of SPEC_SEAM sec. 5 (b23945a; FHF 3248bb3f).
_BF_KERNELS = (
    (226, 227, 249, "_quantize_amax_kernel", "98bc1e48636956f4"),
    (254, 255, 320, "_quantize_amax_dual_kernel", "c7b240a4c2425778"),
    (411, 412, 467, "_quantize_amax_colsum_kernel", "3c91245191122ba5"),
    (472, 473, 574, "_quantize_amax_colsum_v2_tile", "21c1ad1556301c89"),
    (576, 577, 627, "_quantize_amax_colsum_v2_kernel", "43e871de26ca7a54"),
    (745, 746, 800, "_quantize_amax_colsum_sel_kernel", "94ed6706842e83f5"),
)
_FHF_KERNELS = (
    (74, 75, 79, "_nanmax_vec", "7c2528ef310098b3"),
    (82, 83, 93, "_packed_to_src_col", "f08dbc0200ee6906"),
    (96, 97, 187, "_fp8h_amax_tiles_kernel", "c44218f4c577f7f3"),
    (190, 191, 194, "_reduce_partials", "7e9f8cfb538aa769"),
    (197, 198, 200, "_clamp_min_nan", "7ac6e239995499ed"),
    (203, 204, 222, "_e4m3fn_bits_from_fp32", "bc5cb7cda0e9eaa4"),
    (225, 226, 279, "_fp8h_shadow_quant_t_kernel", "d0f04ca573c19bac"),
    (282, 283, 311, "_fp8h_weight_quant_kernel", "1b55f6e80f662ff2"),
)
_FHF_LINES = 501
# DESIGN 3.3: the three r16 names and the helpers r14 adds to bwd_fp8.py
_SEAM_DEFS = ("OutprojMx6Unavailable", "output_projection_dgrad_operands_mx6", "output_projection_g8_rescue",
              "output_projection_mx6_encode", "prebuild_outproj_mx6", "reset_mx6_for_testing")


def _jit_kernels(path: Path) -> List[Tuple[int, int, int, str, str]]:
    """Every function with a triton decorator: (decorator line, def line, end line, name, source sha256[:16])."""
    src = path.read_text()
    lines = src.splitlines()
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if "triton" in (ast.get_source_segment(src, dec) or ""):
                    body = "\n".join(lines[dec.lineno - 1:node.end_lineno])
                    out.append((dec.lineno, node.lineno, node.end_lineno, node.name,
                                hashlib.sha256(body.encode()).hexdigest()[:16]))
    return sorted(out)


class LineNeutralityTest(unittest.TestCase):
    """T1 / S1 contract item 7: no @triton.jit line or kernel source of bwd_fp8.py (or fp8_helpers_fused.py) moved."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _BF_PATH.read_bytes()
        cls.src = cls.raw.decode()
        cls.tree = ast.parse(cls.src)

    def test_bwd_fp8_kernels_keep_line_and_source(self) -> None:
        self.assertEqual(_jit_kernels(_BF_PATH), list(_BF_KERNELS))  # nothing moved, nothing added
        decorated = [i + 1 for i, ln in enumerate(self.src.splitlines()) if re.match(r"^\s*@triton\.", ln)]
        self.assertEqual(decorated, [k[0] for k in _BF_KERNELS])  # no @triton.autotune / heuristics anywhere

    def test_fp8_helpers_fused_untouched(self) -> None:
        self.assertEqual(_jit_kernels(_FHF_PATH), list(_FHF_KERNELS))
        self.assertEqual(len(_FHF_PATH.read_text().splitlines()), _FHF_LINES)

    def test_pure_eof_append_of_the_base_file(self) -> None:
        """S-b landing: the file starts with an allow-listed base byte for byte (7c0bbd7's 2,760 lines = b23945a's, or
        ship16 544e691's 2,824 lines with E-RM inside BF; v1.2), and the r14 block follows. The kernel checks above
        hold on both."""
        self.assertIsNotNone(_BASE, f"bwd_fp8.py starts with none of the allow-listed bases {sorted(_BASES)}")
        head = self.raw[:_BASE_BF_BYTES]
        self.assertEqual(hashlib.sha256(head).hexdigest(), _BASE_BF_SHA256)
        self.assertEqual(head.count(b"\n"), _BASE_BF_LINES)
        tail = self.raw[_BASE_BF_BYTES:].decode()
        lines = tail.splitlines()
        mark = next(i for i, ln in enumerate(lines) if ln.startswith(_APPEND_MARK))
        self.assertTrue(all(not ln.strip() or ln.startswith("#") for ln in lines[:mark]), lines[:mark])
        self.assertLess(mark, 4)

    def test_api_follows_the_last_kernel_and_imports_lazily(self) -> None:
        top = {n.name: n for n in self.tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in _SEAM_DEFS:
            self.assertIn(name, top)
            self.assertGreater(top[name].lineno, _BASE_BF_LINES, name)
        for name in API:
            self.assertTrue(callable(getattr(bwd_fp8, name)), name)
        s1 = inspect.signature(bwd_fp8.output_projection_dgrad_operands_mx6)
        self.assertEqual([(p.name, p.kind) for p in s1.parameters.values()],
                         [("dout", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                          ("output_weight", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                          ("token_major", inspect.Parameter.KEYWORD_ONLY),
                          ("t_tiled", inspect.Parameter.KEYWORD_ONLY)])
        self.assertEqual(list(inspect.signature(bwd_fp8.output_projection_g8_rescue).parameters),
                         ["dout", "output_weight", "ops6"])
        self.assertEqual(list(inspect.signature(bwd_fp8.output_projection_mx6_encode).parameters), ["dout", "weight"])
        # l11_do8 / outproj_dual_mx6 are imported inside functions only (importing bwd_fp8 imports nothing new)
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node)
                self.assertNotIn("l11_do8", text)
                self.assertNotIn("outproj_dual_mx6", text)
        lazy = [n for n in ast.walk(self.tree)
                if isinstance(n, ast.ImportFrom) and n.module == "generative_recommenders.ops.hip"]
        self.assertTrue(lazy)
        self.assertTrue(all(n.lineno > _BASE_BF_LINES for n in lazy))
        self.assertTrue(all(n.col_offset > 0 for n in lazy))  # inside a function body


# ------------------------------------------------------------------------------------------------ T2 API presence
class ApiPresenceTest(unittest.TestCase):
    """T2 / S1 contract item 1."""

    def test_producer_api_present(self) -> None:
        self.assertEqual(do8.MX6_PRODUCER_API, API)
        self.assertEqual(do8._mx6_producer_missing(), [])

    def test_import_bwd_fp8_imports_no_hip_module(self) -> None:
        code = ("import json, sys\n"
                "import generative_recommenders.ops.triton.bwd_fp8\n"
                "print(json.dumps(sorted(m for m in sys.modules if m.startswith('generative_recommenders'))))\n")
        base = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8")}
        base["PYTHONPATH"] = str(_ROOT) + (os.pathsep + base["PYTHONPATH"] if base.get("PYTHONPATH") else "")
        base["PYTHONDONTWRITEBYTECODE"] = "1"
        armed = {"DLRM_HSTU_FP8_BWD": "1", "DLRM_HSTU_FP8_BWD_OUT_WGRAD": "1", "DLRM_HSTU_FP8_BWD_HELPERS_FUSED": "1",
                 do8.MX6_KNOB: "1"}
        for name, extra in (("knob off", {}), ("MX6 armed", armed)):
            env = dict(base)
            env.update(extra)
            run = subprocess.run([sys.executable, "-B", "-c", code], env=env, cwd=str(_ROOT), capture_output=True,
                                 text=True, timeout=600)
            self.assertEqual(run.returncode, 0, (name, run.stderr[-2000:]))
            mods = json.loads(run.stdout.strip().splitlines()[-1])
            self.assertIn("generative_recommenders.ops.triton.bwd_fp8", mods, name)
            self.assertEqual([m for m in mods if m.startswith("generative_recommenders.ops.hip")], [], name)

    def _r16_fixture(self) -> None:
        """r16's Mx6PrebuildTest fixture: DO8 and DO8-ONLY prebuilt (int8 'only' .so), DO8-VD, DIET, the MX6 knob."""
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        do8.set_mx6_sites_for_testing(None)
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        do8.set_diet_for_testing(True)
        do8.set_mx6_for_testing(True)

    def test_prebuild_mx6_passes_the_api_rule_and_latches_at_the_build_on_cpu(self) -> None:
        keys = ("DLRM_HSTU_FP8_L11_DO8_ONLY", "DLRM_HSTU_FP8_L11_VD", "DLRM_HSTU_FP8_L11_DO8_SRC",
                "DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED", "DLRM_HSTU_FP8_L11_DO8_CACHE", "DLRM_HSTU_FP8_L11_DO8_SCHED",
                "DLRM_HSTU_FP8_LNMD_MASK_NIB", do8.MX6_KNOB, do8.MX6_SITES_KNOB)
        with mock.patch.dict(os.environ):
            for k in keys:
                os.environ.pop(k, None)
            os.environ.update({"DLRM_HSTU_FP8_L11_DO8_ONLY": "1", "DLRM_HSTU_FP8_L11_VD": "1"})
            try:
                self._r16_fixture()
                self.assertIsNone(do8._mx6_inert_reason())  # every INERT rule passes, the producer API included
                load = mock.MagicMock(side_effect=RuntimeError("no ROCm GPU in this process (CPU host)"))
                check = mock.MagicMock(return_value=(True, "unused"))
                with mock.patch.object(do8, "_load_mx6", load), mock.patch.object(torch.cuda, "synchronize"), \
                        _Capture() as cap:
                    ok = do8.prebuild_mx6(self_check_fn=check)
                self.assertFalse(ok)
                load.assert_called_once_with()
                check.assert_not_called()
                self.assertNotIn("INERT", cap.text)
                self.assertNotIn("producer API is not in this tree", cap.text)
                want = do8.FAMILY_MX6 + " DISABLED: prebuild build/load: RuntimeError: no ROCm GPU in this process"
                self.assertEqual(cap.text.count(want), 1, cap.text)
                self.assertIsNotNone(do8.mx6_broken())
                self.assertFalse(do8.mx6_enabled())
                # positive control: with the names gone the rule fires (it is live in this tree)
                self._r16_fixture()
                with contextlib.ExitStack() as st:
                    for name in API:
                        st.enter_context(mock.patch.object(bwd_fp8, name, None))
                    self.assertIn("r14's MX6 producer API is not in this tree", do8._mx6_inert_reason() or "")
            finally:
                do8.set_requested_for_testing(None)
                do8.set_mx6_for_testing(None)
                do8.set_mx6_sites_for_testing(None)
                do8.set_diet_for_testing(None)
                do8.reset_for_testing(unload=True)
                l11.set_requested_for_testing(None)
                l11.reset_for_testing()


# ------------------------------------------------------------------------------------------------ the CPU harness
class _SeamCase(unittest.TestCase):
    """The accepted configuration (S1 contract sec. 1) on CPU, the device work stubbed (module docstring)."""

    CONFIG: Dict[str, Any] = {"auto": True, "shift": 5, "floor": 5, "relax": 2, "armed": True, "refresh": 3,
                              "warmup": 4}
    _CONFIG_ATTRS = ("_BWD_ENABLED", "_TRITON_AVAILABLE", "_GRAD_FORMAT", "_GRAD_DTYPE", "_GRAD_MAX",
                     "_OUT_WGRAD_ARMED", "_OUT_WGRAD_T_E4M3", "_OUT_WGRAD_DFMT_SHIFT_AUTO", "_OUT_WGRAD_DFMT_SHIFT",
                     "_OUT_WGRAD_DFMT_SHIFT_FLOOR", "_OUT_WGRAD_DFMT_SHIFT_RELAX", "_BWD_HELPERS_FUSED",
                     "_BWD_REFRESH_CALLS", "_BWD_WARMUP_CALLS", "_BWD_MARGIN", "_BWD_SCALE_LOG")

    def setUp(self) -> None:
        if fhf is None:
            self.skipTest(_FHF_SKIP)
        self._reset_modules()
        st = contextlib.ExitStack()
        self.addCleanup(self._cleanup, st)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)  # small CPU ops only; restored after the test
        st.enter_context(mock.patch.dict(os.environ))
        for k in (odm.FAIL_CLOSED_ENV, odm.AUDIT_ENV, "DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED"):
            os.environ.pop(k, None)
        for attr in self._CONFIG_ATTRS:
            st.enter_context(mock.patch.object(bwd_fp8, attr, getattr(bwd_fp8, attr)))
        self.configure()
        # recorders / knobs of the stand-ins
        self.events: List[str] = []
        self.dual_calls: List[Dict[str, Any]] = []
        self.qrm_calls: List[Dict[str, Any]] = []
        self.qgw_calls: List[Dict[str, Any]] = []
        self.empty_tm_calls: List[Tuple[Any, ...]] = []
        self.site_calls: List[Dict[str, Any]] = []
        self.fwq_calls: List[Tuple[Any, Any]] = []
        self.qw_calls: List[Any] = []
        self.producer_calls: List[Dict[str, Any]] = []
        self.w6_calls: List[Dict[str, Any]] = []
        self.next_partial: Optional[torch.Tensor] = None
        self.producer_rc = 0
        self.producer_exc: Optional[BaseException] = None
        self.w6_rc = 0
        self.site_exc: Optional[BaseException] = None
        self.publish_exc: Optional[BaseException] = None
        self.ineligible_override: Optional[str] = None
        # bwd_fp8: device launches stubbed, bookkeeping real (recorded)
        real_hfo = bwd_fp8._helpers_fused_ok
        st.enter_context(mock.patch.object(bwd_fp8, "_helpers_fused_ok", lambda w: real_hfo(_cuda_view(w))))
        st.enter_context(mock.patch.object(bwd_fp8, "quantize_row_major_dual", self._dual))
        st.enter_context(mock.patch.object(bwd_fp8, "quantize_row_major", self._qrm))
        self._real_empty_tm = fp8_tm_alloc.empty_tm
        st.enter_context(mock.patch.object(fp8_tm_alloc, "empty_tm", self._empty_tm))
        self._real_site = bwd_fp8._fused_site_launch
        st.enter_context(mock.patch.object(bwd_fp8, "_fused_site_launch", self._site))
        self._real_fwq = bwd_fp8._fused_weight_quant
        st.enter_context(mock.patch.object(bwd_fp8, "_fused_weight_quant", self._fwq))
        self._real_qw = bwd_fp8.quantize_weight
        st.enter_context(mock.patch.object(bwd_fp8, "quantize_weight", self._qw))
        self._real_qgw = bwd_fp8._quantize_gradient_and_weight_fused
        st.enter_context(mock.patch.object(bwd_fp8, "_quantize_gradient_and_weight_fused", self._qgw))
        real_engaged = bwd_fp8.record_engaged
        st.enter_context(mock.patch.object(bwd_fp8, "record_engaged",
                                           lambda site: (self.events.append("engaged"), real_engaged(site))[1]))
        real_publish = bwd_fp8._SiteScale.publish

        def publish(state: Any) -> None:
            self.events.append("publish")
            if self.publish_exc is not None:
                raise self.publish_exc
            return real_publish(state)

        st.enter_context(mock.patch.object(bwd_fp8._SiteScale, "publish", publish))
        st.enter_context(mock.patch.object(fhf, "weight_amax_with_site", self._wamax))
        st.enter_context(mock.patch.object(fhf, "weight_quant_from_partials", _ref_weight_quant))
        # the loader: a fake .so behind the real launch / W1 code
        st.enter_context(mock.patch.object(odm, "_lib", SimpleNamespace(
            fn={"outproj_dual_mx6_launch": self._producer, "outproj_w6_shadow_launch": self._w6_kernel})))
        real_inel, real_w6_inel = odm.ineligible_reason, odm.w6_ineligible_reason

        def inel(*a: Any, **k: Any) -> Optional[str]:
            if self.ineligible_override is not None:
                return self.ineligible_override
            return real_inel(*[_cuda_view(x) for x in a], **{n: _cuda_view(v) for n, v in k.items()})

        st.enter_context(mock.patch.object(odm, "ineligible_reason", inel))
        st.enter_context(mock.patch.object(odm, "w6_ineligible_reason", lambda w: real_w6_inel(_cuda_view(w))))
        st.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
        st.enter_context(mock.patch.object(torch.cuda, "device", new=lambda i=None: contextlib.nullcontext()))
        st.enter_context(mock.patch.object(torch.cuda, "current_stream",
                                           new=lambda d=None: SimpleNamespace(cuda_stream=0)))
        odm.set_prebuilt_for_testing(True)
        self.cap = st.enter_context(_Capture())

    def _cleanup(self, st: contextlib.ExitStack) -> None:
        st.close()
        self._reset_modules()

    @staticmethod
    def _reset_modules() -> None:
        bwd_fp8.reset_for_testing()
        bwd_fp8.reset_mx6_for_testing()
        odm.reset_for_testing()
        do8.reset_for_testing()

    def configure(self, **over: Any) -> None:
        """The accepted configuration (CONFIG, overridden by `over`): the scale controller on, e5m2 grads, the e4m3
        W3 copy, HELPERS_FUSED, SHIFT auto / fixed, REFRESH and WARMUP small."""
        c = dict(self.CONFIG)
        c.update(over)
        b = bwd_fp8
        b._BWD_ENABLED, b._TRITON_AVAILABLE, b._BWD_HELPERS_FUSED = True, True, True
        b._GRAD_FORMAT, b._GRAD_DTYPE, b._GRAD_MAX = "e5m2", E5, 57344.0
        b._OUT_WGRAD_ARMED, b._OUT_WGRAD_T_E4M3 = bool(c["armed"]), True
        b._OUT_WGRAD_DFMT_SHIFT_AUTO, b._OUT_WGRAD_DFMT_SHIFT = bool(c["auto"]), int(c["shift"])
        b._OUT_WGRAD_DFMT_SHIFT_FLOOR, b._OUT_WGRAD_DFMT_SHIFT_RELAX = int(c["floor"]), int(c["relax"])
        b._BWD_REFRESH_CALLS, b._BWD_WARMUP_CALLS = int(c["refresh"]), int(c["warmup"])
        b._BWD_MARGIN, b._BWD_SCALE_LOG = 8.0, False

    # ---------------------------------------------------------------- stand-ins: bwd_fp8 / fp8_helpers_fused
    def _partial_for(self, m: int) -> torch.Tensor:
        n = _n_partial(m)
        p = self.next_partial if self.next_partial is not None else torch.arange(1, n + 1, dtype=F32)
        assert p.numel() == n, (p.numel(), n)
        return p.contiguous()

    def _dual(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.events.append("dual")
        b = inspect.signature(self._real_dual_sig).bind(*args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        value, inv = a["value"], a["inv_scale"]
        self.dual_calls.append({"value": value, "inv": inv, "inv_bits": _bits(inv), "pos": tuple(args[2:]),
                                "kwargs": dict(kwargs)})
        dtype = E5 if a["dtype"] is None else a["dtype"]
        limit = 57344.0 if a["limit"] is None else a["limit"]
        t_dtype, t_inv_mul = a["t_dtype"], a["t_inv_mul"]
        assert a["t_tiled"] and t_dtype is not None and (t_dtype != dtype or t_inv_mul != 1.0), a
        m = int(value.shape[0])
        out_t = _ref_dout_t(value, inv, t_inv_mul, _m_pad(m))
        part = self._partial_for(m).clone()
        g8 = _ref_cast(value, inv, dtype, limit)
        return (g8, out_t, part) if a["return_partial"] else (g8, out_t, part.amax())

    _real_dual_sig = staticmethod(bwd_fp8.quantize_row_major_dual)

    def _qrm(self, value: torch.Tensor, inv_scale: torch.Tensor, dtype: Optional[torch.dtype] = None,
             limit: Optional[float] = None, return_partial: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        self.events.append("qrm")
        self.qrm_calls.append({"value": value, "inv": inv_scale, "dtype": dtype, "limit": limit})
        out = _ref_cast(value, inv_scale, E5 if dtype is None else dtype, 57344.0 if limit is None else limit)
        amax = value.detach().abs().amax().float()
        return out.view_as(value), (amax.reshape(1) if return_partial else amax)

    def _empty_tm(self, shape: Any, dtype: torch.dtype, device: Any) -> torch.Tensor:
        self.events.append("empty_tm")
        self.empty_tm_calls.append((tuple(int(s) for s in shape), dtype, torch.device(device)))
        return self._real_empty_tm(shape, dtype, device)

    def _site(self, *args: Any, **kwargs: Any) -> Any:
        self.events.append("site")
        b = inspect.signature(self._real_site).bind(*args, **kwargs).arguments
        state = b["state"]
        rec = {"n_pos": len(args), "keywords": sorted(kwargs), "state": state, "partial": b["partial"],
               "weight": b["weight"], "partial_bits": _bits(b["partial"]), "snapshot": b["snapshot"],
               "t_note": b["t_note"], "t_shift": b["t_shift"], "snapshot_t": b["snapshot_t"], "calls": state.calls,
               "t_calls": state.t_calls}
        self.site_calls.append(rec)
        rec["ret"] = self._real_site(*args, **kwargs)
        return rec["ret"]

    def _wamax(self, weight: torch.Tensor, site: Optional[Dict[str, Any]]) -> torch.Tensor:
        if self.site_exc is not None:
            raise self.site_exc  # e.g. a launch hook raising after the kernel was enqueued
        if site is not None:
            _ref_site_program(site)
        return _ref_weight_tiles(weight)

    def _fwq(self, weight: torch.Tensor, w_partial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.events.append("fwq")
        self.fwq_calls.append((weight, w_partial))
        return self._real_fwq(weight, w_partial)

    def _qw(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.events.append("qw")
        self.qw_calls.append(weight)
        return self._real_qw(weight)

    def _qgw(self, *args: Any, **kwargs: Any) -> Any:
        self.qgw_calls.append({"args": args, "kwargs": dict(kwargs)})
        return self._real_qgw(*args, **kwargs)

    # ---------------------------------------------------------------- stand-ins: the .so's two C entries
    def _producer(self, dout_p: int, inv_p: int, t_inv_mul: float, m: int, m_pad: int, g6_p: int, dt_p: int,
                  part_p: int, used_p: Optional[int], stream: int) -> int:
        self.events.append("producer")
        if not (dout_p and inv_p and g6_p and dt_p and part_p) or used_p == 0:
            self.producer_calls.append({"null": True})
            return 1  # the C launcher's argument check: nothing launched
        inv_bits = int(ctypes.c_uint32.from_address(inv_p).value)
        self.producer_calls.append({"dout": dout_p, "inv": inv_p, "inv_bits": inv_bits, "t_inv_mul": t_inv_mul, "m": m,
                                    "m_pad": m_pad, "g6": g6_p, "dout_t": dt_p, "partial": part_p, "inv_used": used_p,
                                    "stream": stream})
        if self.producer_exc is not None:
            raise self.producer_exc
        if self.producer_rc > 0:
            return self.producer_rc  # nothing launched, nothing written
        dout = torch.frombuffer(bytearray(ctypes.string_at(dout_p, m * D * 2)), dtype=BF).reshape(m, D)
        g6 = dout.view(U8).reshape(m, 2 * D)[:, :G6_ROW].contiguous()  # a row-local function of the dout bits
        dt = _ref_dout_t(dout, _f32_of_bits(inv_bits), t_inv_mul, m_pad)
        part = self._partial_for(m)
        ctypes.memmove(g6_p, g6.data_ptr(), m * G6_ROW)
        ctypes.memmove(dt_p, dt.data_ptr(), dt.numel())
        ctypes.memmove(part_p, part.data_ptr(), part.numel() * 4)
        if used_p is not None:
            ctypes.c_uint32.from_address(used_p).value = inv_bits
        return self.producer_rc  # 0, or < 0: a stale error drained, launched

    def _w6_kernel(self, w_p: int, is_bf16: int, w6_p: int, w6s_p: int, stream: int) -> int:
        self.events.append("w6")
        self.w6_calls.append({"w": w_p, "bf16": is_bf16, "w6": w6_p, "w6s": w6s_p})
        if not (w_p and w6_p and w6s_p):
            return 1
        if self.w6_rc > 0:
            return self.w6_rc
        n = NOUT * D * (2 if is_bf16 else 4)
        w = torch.frombuffer(bytearray(ctypes.string_at(w_p, n)), dtype=BF if is_bf16 else F32)
        w6, w6s = _w6_bytes_of(w)
        ctypes.memmove(w6_p, w6.data_ptr(), W6_BYTES)
        ctypes.memmove(w6s_p, w6s.data_ptr(), W6S_BYTES)
        return self.w6_rc

    # ---------------------------------------------------------------- operands and calls
    @staticmethod
    def param(seed: int = 7, dtype: torch.dtype = F32) -> torch.nn.Parameter:
        g = torch.Generator().manual_seed(seed)
        return torch.nn.Parameter((torch.randn((NOUT, D), generator=g) * 0.02).to(dtype))

    @staticmethod
    def dout(m: int, seed: int, amax: Optional[float] = None) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((m, D), generator=g, dtype=F32)
        if amax is not None:
            x = x.clamp(-1.0, 1.0) * (0.5 * amax)
            x[0, 0] = amax  # exact in bf16 for the amax values used here
        return x.to(BF)

    def fp8(self, dout: torch.Tensor, weight: torch.Tensor) -> Any:
        return bwd_fp8.output_projection_dgrad_operands(dout, weight, token_major=True, t_tiled=True)

    def s1(self, dout: torch.Tensor, weight: torch.Tensor, token_major: bool = True, t_tiled: bool = True) -> Any:
        return bwd_fp8.output_projection_dgrad_operands_mx6(dout, weight, token_major=token_major, t_tiled=t_tiled)

    def warm(self, weight: torch.Tensor, amax: float = 143.0, calls: Optional[int] = None) -> Any:
        """The site's warm-up through the FP8 path (as in training: S1 declines a warm-up site): `calls` calls (default
        the whole warm-up, which publishes at amax 143: inv bits 0x424880e5)."""
        n = bwd_fp8._BWD_WARMUP_CALLS if calls is None else calls
        for i in range(n):
            self.assertIsNone(self.fp8(self.dout(64, 9000 + i, amax=amax), weight))
        return _peek(weight)

    def assert_lines_clean(self, text: Optional[str] = None) -> None:
        """T19 on every line this module family printed: the prefix and none of r16's must-be-ABSENT tokens."""
        for ln in _outproj_lines(self.cap.text if text is None else text):
            self.assertTrue(ln.startswith("[outproj-mx6]"), ln)
            for tok in ABSENT:
                self.assertNotIn(tok.lower(), ln.lower(), ln)


# ------------------------------------------------------------------------------------------------ T3 / T7 / T8
# (M, PARTIAL kind, PARTIAL max) per call after the warm-up (published at amax 143 after 4 calls; REFRESH 3 publishes at
# calls 6, 9, 12, ...). Windows: e4m3 clipping (steps up), a NaN window (NaN scale), zero partials, clean windows
# (steps down), an Inf partial (Inf scale, inv 0), a window cast at inv 0.
_T3_SEQ: Tuple[Tuple[int, str, float], ...] = (
    (300, "max", 200.0), (257, "max", 400.0),
    (1, "max", 100.0), (128, "max", 2000.0), (769, "max", 500.0),
    (1000, "max", 20000.0), (255, "nan", 50.0), (256, "max", 10.0),
    (384, "max", 50.0), (129, "zero", 0.0), (513, "max", 60.0),
    (64, "max", 30.0), (100, "zero", 0.0), (200, "max", 40.0),
    (300, "max", 50.0), (300, "max", 60.0), (300, "max", 70.0),
    (512, "max", 100.0), (511, "max", 100.0), (4097, "max", 100.0),
    (300, "max", 150.0), (383, "inf", 100.0), (2, "max", 1.0),
    (130, "max", 5.0), (700, "max", 6.0), (127, "max", 7.0),
)
_T3_BUMP = (8, 17)  # dense Adam's step before these calls: both weights updated in place (W1 refreshes)
_T3_CONFIGS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("SHIFT auto floor 5 relax 2", {"auto": True, "shift": 5, "floor": 5, "relax": 2, "armed": True}),
    ("SHIFT auto, REFRESH 5: the first refresh is the first post-warm-up call (t_calls 0 before it)",
     {"auto": True, "shift": 5, "floor": 5, "relax": 1, "armed": True, "refresh": 5}),
    ("SHIFT auto floor 4 relax 1", {"auto": True, "shift": 4, "floor": 4, "relax": 1, "armed": True}),
    ("SHIFT 7 fixed", {"auto": False, "shift": 7, "floor": 5, "relax": 4, "armed": True}),
    ("SHIFT 5 fixed", {"auto": False, "shift": 5, "floor": 5, "relax": 4, "armed": True}),
    ("SHIFT auto, no t_clipped_windows (OUT_WGRAD unarmed)", {"auto": True, "shift": 5, "floor": 5, "relax": 2,
                                                               "armed": False}),
    ("SHIFT auto, bf16 master weight", {"auto": True, "shift": 5, "floor": 5, "relax": 2, "armed": True,
                                        "wdtype": BF}),
)


class BookkeepingTest(_SeamCase):
    """T3 (+ T7, T8 on every call): S1 vs `_quantize_gradient_and_weight_fused` for the same PARTIAL."""

    def check_form(self, ops6: Any, m: int, state: Any, inv0: Any, shift0: int, dout: torch.Tensor) -> None:
        """T7 / S1 contract item 2."""
        self.assertIs(type(ops6), do8.Mx6Operands)
        self.assertIs(type(ops6.rescue_args), do8.Mx6RescueArgs)
        self.assertEqual(tuple(ops6.rescue_args), (True, True, E4, 2.0 ** -shift0, 448.0, shift0, True))
        self.assertIsNone(do8._mx6_contract_mismatch(ops6, m, dout.device))
        self.assertIsNone(do8._mx6_operand_mismatch(ops6.g6, ops6.w6, ops6.w6s, m, dout.device))
        inv = ops6.inv_scale_used
        self.assertIsNot(inv, state.inv_scale)
        self.assertNotEqual(inv.untyped_storage().data_ptr(), state.inv_scale.untyped_storage().data_ptr())
        self.assertEqual((inv.dtype, inv.numel(), inv.untyped_storage().nbytes(), inv.data_ptr() % 16),
                         (F32, 1, 4, 0))  # its own 4-B allocation, 16-B aligned
        self.assertEqual(_bits(inv), inv0)  # the bits the cast read: taken before any publish
        self.assertEqual((ops6.g6.dtype, tuple(ops6.g6.shape), ops6.g6.stride()), (U8, (m, G6_ROW), (G6_ROW, 1)))
        self.assertEqual((ops6.dout_fp8.dtype, tuple(ops6.dout_fp8.shape)), (E4, (_m_pad(m) // 128, D, 128)))
        for t in (ops6.dout_fp8_scale, ops6.g_decode_scale):
            self.assertEqual((t.dtype, t.numel()), (F32, 1))

    def run_sequence(self, cfg_name: str, wdtype: torch.dtype = F32) -> Dict[str, int]:
        for rec in (self.dual_calls, self.qgw_calls, self.empty_tm_calls, self.site_calls, self.fwq_calls,
                    self.qw_calls, self.producer_calls, self.w6_calls):
            rec.clear()
        g = torch.Generator().manual_seed(11)
        base = (torch.randn((NOUT, D), generator=g) * 0.02).to(wdtype)
        wa, wb = torch.nn.Parameter(base.clone()), torch.nn.Parameter(base.clone())
        sta, stb = self.warm(wa), self.warm(wb)
        self.assertTrue(sta.published and stb.published)
        self.assertIsNot(sta, stb)
        self.assertEqual(_fields(sta), _fields(stb))
        cov = {"refresh": 0, "up": 0, "down": 0, "r2_cast_shift": 0, "nan_scale": 0, "clip": 0, "t_clip": 0,
               "w1_refresh": 0, "calls": 0}
        refresh = bwd_fp8._BWD_REFRESH_CALLS
        for i, (m, kind, amax) in enumerate(_T3_SEQ):
            msg = f"{cfg_name}: call {i} M={m} {kind} {amax}"
            dout = self.dout(m, 100 + i)
            self.next_partial = _partial(m, kind, amax, i)
            if i in _T3_BUMP:
                with torch.no_grad():
                    for w in (wa, wb):
                        w.mul_(1.03125)  # same values on both sides; bumps _version (W1 refresh at the next S1)
            inv0, shift0 = _bits(stb.inv_scale), stb.t_shift
            self.assertEqual(_bits(sta.inv_scale), inv0, msg)
            calls0, w6_0 = stb.calls, len(self.w6_calls)
            nd, ns, nq = len(self.dual_calls), len(self.site_calls), len(self.qgw_calls)
            nf, nw = len(self.fwq_calls), len(self.qw_calls)
            self.events.clear()
            res = self.fp8(dout, wa)
            ev_fp8 = list(self.events)
            self.events.clear()
            n_tm, n_prod = len(self.empty_tm_calls), len(self.producer_calls)
            dout_v, dout_c = dout._version, dout.clone()
            ops6 = self.s1(dout, wb)
            ev_s1 = list(self.events)
            self.assertTrue(dout._version == dout_v and torch.equal(dout.view(torch.int16), dout_c.view(torch.int16)),
                            msg)  # S1 reads dout only (R2 casts it again)
            # ---- T3: every field bitwise, the snapshots, calls moved by exactly one
            self.assertEqual(_fields(sta), _fields(stb), msg)
            self.assertEqual(stb.calls, calls0 + 1, msg)
            g8a, gsa, opa, opsa, w8a, w8sa = res
            self.assertEqual(_bits(ops6.g_decode_scale), _bits(gsa), msg)
            self.assertEqual(_bits(ops6.dout_fp8_scale), _bits(opsa), msg)
            # the FP8 reference went through _quantize_gradient_and_weight_fused with the contract's arguments
            self.assertEqual(len(self.qgw_calls), nq + 1, msg)
            q = self.qgw_calls[-1]
            self.assertEqual((q["args"][:3], q["kwargs"]),
                             (("output_projection", dout, wa), {"discriminator": wa.data_ptr(),
                                                               "snapshot_decode_scale": True, "token_major": True,
                                                               "token_major_e4m3": True, "token_major_tiled": True,
                                                               **_FP8_REF_EXTRA_KWARGS}),
                             msg)
            # ---- the casts: one dual cast on the FP8 path, none in S1; both read the pre-call inv bits
            self.assertEqual(len(self.dual_calls), nd + 1, msg)
            dc = self.dual_calls[-1]
            self.assertIs(dc["inv"], sta.inv_scale, msg)
            self.assertEqual(dc["inv_bits"], inv0, msg)
            self.assertEqual(len(self.producer_calls), n_prod + 1, msg)
            pc = self.producer_calls[-1]
            self.assertEqual(_bits(_f32_of_bits(pc["inv_bits"])), inv0, msg)
            self.assertEqual((pc["dout"], pc["m"], pc["m_pad"], pc["t_inv_mul"]),
                             (dout.data_ptr(), m, _m_pad(m), 2.0 ** -shift0), msg)
            self.assertEqual(pc["t_inv_mul"], dc["kwargs"]["t_inv_mul"], msg)
            # the captured bits: INV = its own clone and no INV_USED (S1 contract 2.2; v1.1-1), or INV = the live
            # state.inv_scale with the kernel writing INV_USED (DESIGN D10); never anything else
            used = ops6.inv_scale_used.data_ptr()
            self.assertIn((pc["inv"], pc["inv_used"]), ((used, None), (stb.inv_scale.data_ptr(), used)), msg)
            self.assertEqual((pc["g6"], pc["dout_t"]), (ops6.g6.data_ptr(), ops6.dout_fp8.data_ptr()), msg)
            self.assertTrue(torch.equal(ops6.dout_fp8.view(U8), opa.view(U8)), msg)  # dout_t bytes == the FP8 call's
            self.assertEqual(self.empty_tm_calls[n_tm:], [((_m_pad(m) // 128, D, 128), E4, CPU)], msg)
            # ---- the site launch: same call on both paths, S1's on the producer's own PARTIAL buffer
            self.assertEqual(len(self.site_calls), ns + 2, msg)
            sa, sb = self.site_calls[-2], self.site_calls[-1]
            keys = ("n_pos", "keywords", "partial_bits", "snapshot", "t_note", "t_shift", "snapshot_t", "calls",
                    "t_calls")
            self.assertEqual({k: sa[k] for k in keys}, {k: sb[k] for k in keys}, msg)
            self.assertEqual((sa["n_pos"], sa["keywords"]), (3, ["snapshot", "snapshot_t", "t_note", "t_shift"]), msg)
            self.assertIs(sb["state"], stb, msg)
            self.assertIs(sb["weight"], wb, msg)
            self.assertEqual((sb["partial"].data_ptr(), sb["partial"].numel(), sb["partial"].dtype),
                             (pc["partial"], _n_partial(m), F32), msg)
            self.assertEqual(sb["partial"].untyped_storage().nbytes(), 4 * _n_partial(m), msg)  # its own allocation
            self.assertEqual(sb["t_shift"], shift0, msg)
            # ---- S1 runs no weight cast (no _fused_weight_quant / quantize_weight) and the FP8 path's per-call one
            self.assertEqual((len(self.fwq_calls), len(self.qw_calls)), (nf + 1, nw), msg)
            # ---- device order: fresh buffers, the producer and the W1 refresh if due (either order: W1 touches no
            # controller tensor; DESIGN 11.1 / v1.1-2), then the commit's site program, publish on a refresh, the line
            refreshed = stb.calls % refresh == 0
            w1_due = len(self.w6_calls) > w6_0
            k = ev_s1.index("site")
            self.assertEqual(ev_s1[0], "empty_tm", msg)
            self.assertEqual(sorted(ev_s1[1:k]), sorted(["producer"] + (["w6"] if w1_due else [])), msg)
            self.assertEqual(ev_s1[k:], ["site"] + (["publish"] if refreshed else []) + ["engaged"], msg)
            self.assertEqual(ev_fp8, ["dual", "site"] + (["publish"] if refreshed else []) + ["fwq", "engaged"], msg)
            self.assertEqual(w1_due, i == 0 or i in _T3_BUMP, msg)
            entry = odm._w6_cache[(id(wb), 0, (NOUT, D), (D, 1), wdtype, "L11")]
            self.assertTrue(ops6.w6 is entry.w6 and ops6.w6s is entry.w6s, msg)  # the cached shadow, no copy
            # ---- T7
            self.check_form(ops6, m, stb, inv0, shift0, dout)
            # ---- T8: R2 is the FP8 path's dual cast at the captured bits and the cast-time shift; no state change
            before = _snapshot()
            dt_before = ops6.dout_fp8.view(U8).clone()
            nd = len(self.dual_calls)
            out = bwd_fp8.output_projection_g8_rescue(dout, wb, ops6)
            self.assertEqual(len(self.dual_calls), nd + 1, msg)
            rc = self.dual_calls[-1]
            self.assertTrue(rc["value"] is dout and rc["inv"] is ops6.inv_scale_used and rc["pos"] == (), msg)
            self.assertEqual(rc["kwargs"], dc["kwargs"], msg)  # t_dtype, t_inv_mul, t_limit, return_partial, t_tiled
            self.assertEqual(rc["inv_bits"], dc["inv_bits"], msg)
            self.assertEqual(rc["kwargs"]["t_inv_mul"], 2.0 ** -ops6.rescue_args.t_shift, msg)
            self.assertEqual(_snapshot(), before, msg)
            self.assertTrue(torch.equal(ops6.dout_fp8.view(U8), dt_before), msg)
            g8, g8s, w8, w8s = out
            self.assertTrue(torch.equal(g8.view(U8), g8a.view(U8)), msg)  # the FP8 call's g8 bytes
            self.assertIs(g8s, ops6.g_decode_scale, msg)
            self.assertTrue(torch.equal(w8.view(U8), w8a.view(U8)) and _bits(w8s) == _bits(w8sa), msg)
            self.assertIs(self.fwq_calls[-1][0], wb, msg)
            self.assertIs(self.fwq_calls[-1][1], sb["ret"][0], msg)  # S1's own w_partial (stash hit)
            self.assertIs(do8.mx6_rescue(dout, wb, ops6)[1], ops6.g_decode_scale, msg)  # r16's checker passes
            self.assertIsNone(do8.mx6_broken(), msg)
            self.assertEqual(_snapshot(), before, msg)
            # ---- coverage
            cov["calls"] += 1
            if refreshed:
                cov["refresh"] += 1
                cov["up"] += int(stb.t_shift > shift0)
                cov["down"] += int(stb.t_shift < shift0)
                cov["nan_scale"] += int(bool(torch.isnan(stb.scale).any()))
                if stb.t_shift != shift0:
                    self.assertEqual(ops6.rescue_args.t_shift, shift0, msg)  # the cast-time shift, not the new one
                    cov["r2_cast_shift"] += 1
                if _bits(stb.inv_scale) != inv0:
                    self.assertNotEqual(_bits(ops6.inv_scale_used), _bits(stb.inv_scale), msg)
            cov["w1_refresh"] += int(w1_due)
        cov["clip"] = int(sta.clipped_windows)
        cov["t_clip"] = int(sta.t_clipped_windows) if sta.t_clipped_windows is not None else -1
        self.assertEqual(odm.w6_cache_stats()["refreshes"], 1 + len(_T3_BUMP), cfg_name)
        self.assert_lines_clean()
        self.assertEqual(_outproj_lines(self.cap.text), [], cfg_name)  # S1 / R2 print nothing on these paths
        return cov

    def test_bookkeeping_bitwise_over_refresh_windows(self) -> None:
        for name, cfg in _T3_CONFIGS:
            with self.subTest(name):
                self._reset_modules()
                odm.set_prebuilt_for_testing(True)
                self.configure(**{k: v for k, v in cfg.items() if k != "wdtype"})
                cov = self.run_sequence(name, cfg.get("wdtype", F32))
                self.assertEqual(cov["calls"], len(_T3_SEQ))
                self.assertGreaterEqual(cov["refresh"], len(_T3_SEQ) // cfg.get("refresh", 3) - 1, cov)
                self.assertGreaterEqual(cov["nan_scale"], 1, cov)
                self.assertGreaterEqual(cov["clip"], 1, cov)
                if cfg["auto"] and cfg["armed"]:
                    self.assertGreaterEqual(cov["up"], 1, cov)
                    self.assertGreaterEqual(cov["down"], 1, cov)
                    self.assertGreaterEqual(cov["r2_cast_shift"], 2, cov)
                    self.assertGreaterEqual(cov["t_clip"], 2, cov)
                else:
                    self.assertEqual((cov["up"], cov["down"], cov["r2_cast_shift"]), (0, 0, 0), cov)
                if not cfg["armed"]:
                    self.assertEqual(cov["t_clip"], -1)
                    self.assertTrue(all(not c["t_note"] for c in self.site_calls))
                if not cfg["auto"] and cfg["shift"] == 7:
                    self.assertTrue(all(c["t_calls"] == 0 and not c["t_note"] for c in self.site_calls))


# ------------------------------------------------------------------------------------------------ T4 declines
class DeclineTest(_SeamCase):
    """T4 / S1 contract item 3: a condition of THIS call -> None with nothing touched, one line per reason class."""

    # what each reason class's line says (SEAM_NOTES 5.1)
    CLASS_TEXT = {"layout": "the W3 dout_t is not token-major and step-tiled",
                  "dout": "(need bf16 [M, 512], contiguous", "rows": "outside 1..",
                  "warm-up": "missing or not published yet (warm-up)"}

    def assert_declines(self, name: str, call: Callable[[], Any], cls: str, first: bool) -> None:
        before, n_lines = _snapshot(), len(_outproj_lines(self.cap.text))
        recs = (len(self.producer_calls), len(self.w6_calls), len(self.empty_tm_calls), len(self.dual_calls),
                len(self.site_calls))
        for _ in range(2):
            counts: Dict[str, int] = {}
            with _alloc_spy(counts):
                out = call()
            self.assertIsNone(out, name)
            self.assertEqual(counts, {}, name)  # no allocation, no clone
            self.assertEqual(_snapshot(), before, name)
        self.assertEqual((len(self.producer_calls), len(self.w6_calls), len(self.empty_tm_calls),
                          len(self.dual_calls), len(self.site_calls)), recs, name)
        new = _outproj_lines(self.cap.text)[n_lines:]
        self.assertEqual(len(new), 1 if first else 0, (name, new))
        for ln in new:
            self.assertTrue(ln.startswith("[outproj-mx6] S1 decline: "), ln)
            self.assertIn("(this call runs the FP8 dual cast)", ln)
            self.assertIn(self.CLASS_TEXT[cls], ln, cls)

    def test_per_call_rows(self) -> None:
        w = self.param()
        st = self.warm(w)
        self.assertTrue(st.published)
        good = self.dout(300, 1)
        mis = torch.empty(300 * D + 8, dtype=BF)[1:1 + 300 * D].view(300, D)  # contiguous, 2 B off 16
        self.assertTrue(mis.is_contiguous() and mis.data_ptr() % 16 != 0)
        nc = torch.zeros((D, 300), dtype=BF).t()
        self.assertFalse(nc.is_contiguous())
        # F1 (v1.1): one row with row stride 1024 is_contiguous() (a size-1 dim) but not the producer's layout
        one = torch.zeros((2, 2 * D), dtype=BF)[:1, :D]
        self.assertTrue(one.is_contiguous() and one.stride() == (2 * D, 1) and one.data_ptr() % 16 == 0)
        rows: Tuple[Tuple[str, str, Any, Dict[str, bool]], ...] = (  # every input built before the spy
            ("token_major False", "layout", good, {"token_major": False}),
            ("t_tiled False", "layout", good, {"t_tiled": False}),
            ("both False", "layout", good, {"token_major": False, "t_tiled": False}),
            ("fp32 dout", "dout", good.float(), {}),
            ("fp16 dout", "dout", good.half(), {}),
            ("e5m2 dout", "dout", good.to(E5), {}),
            ("3-D dout", "dout", good.view(3, 100, D), {}),
            ("1-D dout", "dout", good.reshape(-1), {}),
            ("width 256", "dout", good[:, :256].contiguous(), {}),
            ("width 1024", "dout", torch.zeros((150, 1024), dtype=BF), {}),
            ("non-contiguous", "dout", nc, {}),
            ("misaligned", "dout", mis, {}),
            ("M = 1, row stride 1024 (F1)", "dout", one, {}),
            ("another device (meta)", "dout", torch.empty((300, D), dtype=BF, device="meta"), {}),
            ("not a tensor", "dout", good.float().numpy(), {}),
            ("M = 0", "rows", torch.empty((0, D), dtype=BF), {}),
        )
        seen = set()
        for name, cls, dout, kw in rows:
            with self.subTest(name):
                self.assert_declines(name, lambda: self.s1(dout, w, **kw), cls, cls not in seen)
                seen.add(cls)
        big, edge = self.dout(1001, 3), self.dout(1000, 3)
        with mock.patch.object(odm, "M_MAX", 1000):
            self.assert_declines("M > M_MAX", lambda: self.s1(big, w), "rows", False)
            self.assertIsNotNone(self.s1(edge, w))  # M == M_MAX is taken
        self.assertEqual(sorted(seen), ["dout", "layout", "rows"])
        self.assert_lines_clean()

    def test_missing_site_mints_nothing(self) -> None:
        w0 = self.param(1)
        self.warm(w0)  # another layer's site exists
        w = self.param(2)
        self.assertIsNone(_peek(w))
        dout = self.dout(300, 1)
        self.assert_declines("site missing", lambda: self.s1(dout, w), "warm-up", True)
        self.assertIsNone(_peek(w))
        self.assertEqual(sorted(bwd_fp8._sites), ["output_projection.L0"])
        self.assertEqual(len(bwd_fp8._layer_index), 1)
        self.assert_lines_clean()

    def test_warm_up_site_is_not_accumulated(self) -> None:
        w = self.param()
        st = self.warm(w, amax=143.0, calls=2)
        self.assertFalse(st.published)
        f0 = _fields(st)
        dout = self.dout(300, 1, amax=1000.0)  # exact in bf16
        self.assert_declines("site not published", lambda: self.s1(dout, w), "warm-up", True)
        self.assertEqual(_fields(st), f0)  # neither calls nor pending moved (no warm-up accumulate)
        # the FP8 cast that follows runs this call's warm-up bookkeeping, once
        self.assertIsNone(self.fp8(dout, w))
        self.assertEqual(st.calls, 3)
        self.assertEqual(_bits(st.pending), _bits(torch.tensor(1000.0)))
        self.assert_lines_clean()


# ------------------------------------------------------------------------------------------------ T5 / T6 raises
class RaiseTest(_SeamCase):
    """T5 (S1 contract 7b): persistent conditions raise OutprojMx6Unavailable before anything is touched. T6: the commit
    boundary (pre-commit failures leave the state, post-commit raises move calls by one)."""

    def published(self, seed: int = 7) -> Tuple[torch.nn.Parameter, Any]:
        w = self.param(seed)
        st = self.warm(w)
        self.assertTrue(st.published)
        return w, st

    def assert_raises_untouched(self, name: str, call: Callable[[], Any], exc: Any, want: str,
                                prepared: bool) -> BaseException:
        """`call` raises `exc` (message containing `want`) with the state untouched. prepared=False: a persistent
        condition, raised before anything (no allocation, clone or launch). prepared=True: a failure while preparing
        (SPEC_SEAM 2.3: only fresh buffers written, and at most a W1 entry built from the current weight values)."""
        before = _snapshot()
        recs = (len(self.producer_calls), len(self.w6_calls), len(self.empty_tm_calls), len(self.site_calls))
        counts: Dict[str, int] = {}
        with _alloc_spy(counts):
            with self.assertRaises(exc, msg=name) as ctx:
                call()
        self.assertIn(want, str(ctx.exception), name)
        after = _snapshot()
        if prepared:
            for key in ("w6_cache", "w6_stats"):
                before.pop(key)
                after.pop(key)
            for e in odm._w6_cache.values():  # a W1 entry made before the failure: the current weight's image
                w = e.ref()
                self.assertTrue(w is not None and e.version == w._version and e.data_ptr == w.data_ptr(), name)
                self.assertTrue(torch.equal(e.w6, _w6_bytes_of(w)[0]), name)
        self.assertEqual(after, before, name)  # state, stash, labels, loader latch (+ W1 cache unless prepared)
        if not prepared:
            self.assertEqual(counts, {}, name)
            self.assertEqual((len(self.producer_calls), len(self.w6_calls), len(self.empty_tm_calls),
                              len(self.site_calls)), recs, name)
        self.assertEqual(len(self.site_calls), recs[3], name)
        return ctx.exception

    def test_persistent_rows_raise_before_anything(self) -> None:
        w, st = self.published()
        dout = self.dout(300, 1)
        b = bwd_fp8
        Unavailable = bwd_fp8.OutprojMx6Unavailable
        self.assertTrue(issubclass(Unavailable, RuntimeError))
        # r16 adopting the w_partial amendment (DESIGN 16.2) without the seam's field tuples changing with it
        args_v2 = collections.namedtuple("Mx6RescueArgs", do8.Mx6RescueArgs._fields + ("w_partial",))
        rows: Tuple[Tuple[str, Tuple[Tuple[Any, str, Any], ...], str], ...] = (
            ("loader not prebuilt", ((odm, "_prebuilt", False),), "not prebuilt in this process"),
            ("loader latched", ((odm, "_failure", "injected latch"),), "DISABLED in this process (injected latch)"),
            ("loader latched, r16's tokens in the reason",
             ((odm, "_failure", "MX6 DISABLED by a DEVICE FAULT; mx6 inert, MX6 Release, MX6 rescue"),),
             "DISABLED in this process (MX6 DISABLED by a DEVICE FAULT"),
            ("controller off", ((b, "_BWD_ENABLED", False),), "scale controller is off"),
            ("grads e4m3", ((b, "_GRAD_DTYPE", E4), (b, "_GRAD_MAX", 448.0)), "gradient format is not e5m2"),
            ("DFMT e5m2", ((b, "_OUT_WGRAD_T_E4M3", False),), "dout_t copy is not e4m3"),
            ("HELPERS_FUSED off", ((b, "_BWD_HELPERS_FUSED", False),), "fused scale bookkeeping is off"),
            ("r16 Mx6RescueArgs changed", ((do8, "Mx6RescueArgs", args_v2),), "are not the S1 contract's fields"),
        )
        for name, patches, want in rows:
            with self.subTest(name):
                with contextlib.ExitStack() as ps, _Capture() as cap:
                    for target, attr, value in patches:
                        ps.enter_context(mock.patch.object(target, attr, value))
                    self.assert_raises_untouched(name, lambda: self.s1(dout, w), Unavailable, want, False)
                    self.assert_raises_untouched(name, lambda: self.s1(dout, w), Unavailable, want, False)
                    # persistent beats per-call: a call that also fails a per-call row still raises
                    self.assert_raises_untouched(name, lambda: self.s1(dout, w, token_major=False), Unavailable, want,
                                                 False)
                lines = _outproj_lines(cap.text)
                mine = [ln for ln in lines if "S1 unavailable" in ln]
                self.assertEqual(len(mine), 1, (name, lines))  # once per reason per process
                self.assertTrue(mine[0].startswith("[outproj-mx6] S1 unavailable: "), mine[0])
                self.assertIn("(raised before any scale-state change)", mine[0])
                self.assert_lines_clean(cap.text)
        self.assertEqual(st.calls, 4)

    def test_weight_rows_raise_before_anything(self) -> None:
        w, st = self.published()
        dout = self.dout(300, 1)
        Unavailable = bwd_fp8.OutprojMx6Unavailable
        no_grad = torch.zeros((NOUT, D))  # a leaf that does not require grad
        rows = (
            ("not a leaf", lambda: w * 1.0, "not the leaf parameter"),
            ("no grad", lambda: no_grad, "not the leaf parameter"),
            ("shape (1536, 256)", lambda: torch.nn.Parameter(torch.zeros((NOUT, 256))), "shape (1536, 256)"),
            ("shape (512, 1536)", lambda: torch.nn.Parameter(torch.zeros((D, NOUT))), "shape (512, 1536)"),
            ("fp16 weight", lambda: torch.nn.Parameter(torch.zeros((NOUT, D), dtype=torch.float16)),
             "fused scale bookkeeping is off"),
            ("non-contiguous", lambda: torch.nn.Parameter(torch.zeros((D, NOUT))).t(),
             "fused scale bookkeeping is off"),
            ("3-D weight", lambda: torch.nn.Parameter(torch.zeros((3, NOUT, D))), "fused scale bookkeeping is off"),
        )
        for name, make, want in rows:
            with self.subTest(name):
                bad = make()
                with _Capture():
                    self.assert_raises_untouched(name, lambda: self.s1(dout, bad), Unavailable, want, False)
        self.assertEqual(st.calls, 4)

    def test_t_shift_outside_4_7_raises(self) -> None:
        w, st = self.published()
        dout = self.dout(300, 1)
        for shift in (3, 8):
            with self.subTest(shift):
                st.t_shift = shift
                with _Capture():
                    self.assert_raises_untouched(f"t_shift {shift}", lambda: self.s1(dout, w),
                                                 bwd_fp8.OutprojMx6Unavailable, f"t_shift {shift} is not in 4..7",
                                                 False)
        st.t_shift = 5

    def test_precommit_failures_reraise_unchanged(self) -> None:
        w, st = self.published()
        dout = self.dout(300, 1)
        kernel = RuntimeError("injected producer failure")
        real_empty = torch.empty

        # (name, arrange, exception type, message part, the C entry that fails: "producer" / "w6" / None = none runs)
        rows = (
            ("producer rc 1", lambda: setattr(self, "producer_rc", 1), odm.OutprojMx6Error,
             "outproj_dual_mx6_launch rc=1 (hipErrorInvalidValue) M=300", "producer"),
            ("producer rc 700", lambda: setattr(self, "producer_rc", 700), odm.OutprojMx6Error,
             "rc=700 (hipErrorIllegalAddress)", "producer"),
            ("producer raises", lambda: setattr(self, "producer_exc", kernel), RuntimeError, "injected producer",
             "producer"),
            ("launch arguments", lambda: setattr(self, "ineligible_override", "injected argument mismatch"), ValueError,
             "launch: injected argument mismatch", None),
            ("W1 rc 1", lambda: setattr(self, "w6_rc", 1), odm.OutprojMx6Error, "outproj_w6_shadow_launch rc=1", "w6"),
        )
        for name, arrange, exc, want, failing in rows:
            with self.subTest(name):
                odm.clear_w6_cache()  # a W1 refresh is due on this call
                arrange()
                n_prod, n_w6 = len(self.producer_calls), len(self.w6_calls)
                try:
                    with _Capture() as cap:
                        err = self.assert_raises_untouched(name, lambda: self.s1(dout, w), exc, want, True)
                finally:
                    self.producer_rc, self.producer_exc, self.w6_rc, self.ineligible_override = 0, None, 0, None
                if exc is RuntimeError:
                    self.assertIs(err, kernel)
                # each C entry at most once and the failing one ran (W1 and the producer may come in either order:
                # DESIGN 11.1 runs W1 first, v1.1-2 the producer first; neither touches a controller tensor)
                ran = {"producer": len(self.producer_calls) - n_prod, "w6": len(self.w6_calls) - n_w6}
                self.assertLessEqual(max(ran.values()), 1, (name, ran))
                if failing is not None:
                    self.assertEqual(ran[failing], 1, (name, ran))
                if failing == "w6":
                    self.assertEqual(odm.w6_cache_stats()["entries"], 0, name)  # a failed refresh caches nothing
                lines = [ln for ln in _outproj_lines(cap.text) if "S1 raised before its commit" in ln]
                self.assertEqual(len(lines), 1, (name, cap.text))
                self.assertTrue(lines[0].startswith("[outproj-mx6] S1 raised before its commit (scale state "
                                                    "untouched): " + type(err).__name__ + ": "), lines[0])
                self.assert_lines_clean(cap.text)
        real_clone = torch.Tensor.clone

        def clone_of_inv_fails(self_: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
            if self_ is st.inv_scale:
                raise RuntimeError("HIP out of memory (injected at the inv_scale clone)")
            return real_clone(self_, *a, **k)

        d10_commits = 0
        for name, target, attr, new in (("allocation", torch, "empty", None),
                                        ("empty_tm", fp8_tm_alloc, "empty_tm", None),
                                        ("inv_scale clone", torch.Tensor, "clone", clone_of_inv_fails)):
            with self.subTest(name):
                oom = RuntimeError(f"HIP out of memory (injected at the {name})")
                n_prod, n_w6 = len(self.producer_calls), len(self.w6_calls)
                before = _snapshot()
                err: Optional[BaseException] = None
                with _Capture() as cap, mock.patch.object(target, attr, new or mock.Mock(side_effect=oom)):
                    try:
                        self.s1(dout, w)
                    except RuntimeError as error:  # noqa: PERF203
                        err = error
                if err is None and new is not None:
                    # DESIGN D10's form takes no clone (the kernel writes INV_USED): this row does not apply
                    self.assertIsNotNone(self.producer_calls[-1]["inv_used"], name)
                    d10_commits += 1
                    continue
                self.assertIsNotNone(err, name)
                self.assertEqual(str(err), str(oom), name)
                if new is None:
                    self.assertIs(err, oom, name)
                self.assertEqual(_snapshot(), before, name)
                self.assertEqual((len(self.producer_calls), len(self.w6_calls)), (n_prod, n_w6), name)  # no launch
                self.assertEqual(_outproj_lines(cap.text),
                                 [f"[outproj-mx6] S1 raised before its commit (scale state untouched): RuntimeError: "
                                  f"HIP out of memory (injected at the {name})"], name)
        self.assertIs(torch.empty, real_empty)
        self.assertEqual(st.calls, 4 + d10_commits)

    def test_stale_error_rc_proceeds(self) -> None:
        w, st = self.published()
        self.producer_rc = -53
        with _Capture() as cap:
            ops6 = self.s1(self.dout(300, 1), w)
            ops6b = self.s1(self.dout(300, 2), w)
        self.assertIsInstance(ops6, do8.Mx6Operands)
        self.assertIsInstance(ops6b, do8.Mx6Operands)
        self.assertEqual(st.calls, 6)
        lines = _outproj_lines(cap.text)
        self.assertEqual(len(lines), 1, lines)  # once per process and code
        self.assertTrue(lines[0].startswith("[outproj-mx6] stale HIP error 53 (hipErrorPriorLaunchFailure) drained at "
                                            "outproj_dual_mx6_launch"), lines[0])
        self.assert_lines_clean(cap.text)

    def test_post_commit_raise_is_calls_delta_one(self) -> None:
        w, st = self.published()
        stash0 = bwd_fp8._MX6_RESCUE_STASH
        self.site_exc = RuntimeError("injected launch-exit hook")
        f0 = _fields(st)
        with self.assertRaises(RuntimeError) as ctx:
            self.s1(self.dout(300, 1), w)
        self.assertIs(ctx.exception, self.site_exc)
        self.assertEqual((st.calls, st.t_calls), (f0["calls"] + 1, f0["t_calls"] + 1))  # committed before the launch
        self.assertEqual({k: v for k, v in _fields(st).items() if k not in ("calls", "t_calls")},
                         {k: v for k, v in f0.items() if k not in ("calls", "t_calls")})
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH, stash0)  # the stash names successful calls only
        self.assertNotIn("output_projection_dgrad", bwd_fp8._engaged_sites)
        self.site_exc = None
        # a raise from publish on a refresh call: calls 5 (above), now 6 = a refresh
        self.publish_exc = RuntimeError("injected publish failure")
        with self.assertRaises(RuntimeError):
            self.s1(self.dout(300, 2), w)
        self.assertEqual(st.calls, 6)
        self.assertEqual(st.calls % bwd_fp8._BWD_REFRESH_CALLS, 0)
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH, stash0)


# ------------------------------------------------------------------------------------------------ r16's checkers
class R16ContractTest(_SeamCase):
    """S1 contract items 4 and 5 through r16's real mx6_produce / mx6_rescue (the registry's cancel stubbed)."""

    def setUp(self) -> None:
        super().setUp()
        self.cancel = mock.Mock(return_value=True)
        p = mock.patch.object(do8, "registry", return_value=SimpleNamespace(cancel=self.cancel))
        p.start()
        self.addCleanup(p.stop)
        self.w = self.param()
        self.state = self.warm(self.w)

    def decision(self, m: int) -> Any:
        return do8.Mx6Decision("SITE", m, 1, True, None, open_=True)

    def test_success_keeps_the_plan_open(self) -> None:
        dec = self.decision(300)
        ops6 = do8.mx6_produce(dec, self.dout(300, 1), self.w, token_major=True, t_tiled=True)
        self.assertIsInstance(ops6, do8.Mx6Operands)
        self.assertTrue(dec.open)
        self.assertEqual(self.state.calls, 5)
        c = do8.mx6_counters()
        self.assertEqual((c["released"], c["latched"]), (0, 0))
        self.cancel.assert_not_called()
        # item 5: the rescue through r16's checker returns R2's tuple and moves nothing
        f0 = _fields(self.state)
        out = do8.mx6_rescue(self.dout(300, 1), self.w, ops6)
        self.assertIsNone(do8._mx6_rescue_mismatch(out, 300, CPU))
        self.assertEqual(_fields(self.state), f0)
        self.assertIsNone(do8.mx6_broken())

    def test_decline_releases_without_latch(self) -> None:
        dec = self.decision(300)
        self.assertIsNone(do8.mx6_produce(dec, self.dout(300, 1), self.w, token_major=True, t_tiled=False))
        self.assertFalse(dec.open)
        c = do8.mx6_counters()
        self.assertEqual((c["released"], c["latched"]), (1, 0))
        self.cancel.assert_called_once_with("SITE")
        self.assertEqual(self.state.calls, 4)
        self.assertIn("release: producer declined (scale state untouched)", self.cap.text)

    def test_single_row_padded_stride_declines_without_latch(self) -> None:
        """F1 (v1.1): an M = 1 dout with row stride 1024 passes r16's preflight (is_contiguous()) and is a condition of
        THIS call (S1 contract 2.3): S1 declines, r16 releases without a latch, no state moves, nothing is launched."""
        one = torch.zeros((2, 2 * D), dtype=BF)[:1, :D]
        one.copy_(self.dout(1, 3))
        self.assertTrue(one.is_contiguous() and one.stride() == (2 * D, 1))
        dec = do8.Mx6Decision("SITE", 1, 1, True, None, open_=True)
        f0, n = _fields(self.state), len(self.producer_calls)
        self.assertIsNone(do8.mx6_produce(dec, one, self.w, token_major=True, t_tiled=True))
        self.assertFalse(dec.open)
        c = do8.mx6_counters()
        self.assertEqual((c["released"], c["latched"]), (1, 0))
        self.assertIsNone(do8.mx6_broken())
        self.assertEqual(_fields(self.state), f0)
        self.assertEqual(len(self.producer_calls), n)
        self.assertIn("S1 decline: dout torch.bfloat16 (1, 512) stride (1024, 1)", self.cap.text)
        # the dense copy of the same row is taken
        dense = one.clone(memory_format=torch.contiguous_format)
        self.assertEqual(dense.stride(), (D, 1))
        self.assertIsInstance(self.s1(dense, self.w), do8.Mx6Operands)

    def test_persistent_raise_releases_and_latches(self) -> None:
        dec = self.decision(300)
        odm.set_prebuilt_for_testing(False)
        self.assertIsNone(do8.mx6_produce(dec, self.dout(300, 1), self.w, token_major=True, t_tiled=True))
        self.assertFalse(dec.open)
        c = do8.mx6_counters()
        self.assertEqual((c["released"], c["latched"]), (1, 1))
        self.assertIn("producer raised: OutprojMx6Unavailable: [outproj-mx6] S1 unavailable: not prebuilt",
                      do8.mx6_broken() or "")
        self.assertEqual(self.state.calls, 4)

    def test_precommit_failure_releases_and_latches(self) -> None:
        dec = self.decision(300)
        self.producer_rc = 1
        self.assertIsNone(do8.mx6_produce(dec, self.dout(300, 1), self.w, token_major=True, t_tiled=True))
        self.assertFalse(dec.open)
        self.assertIn("producer raised: OutprojMx6Error", do8.mx6_broken() or "")
        self.assertEqual(self.state.calls, 4)

    def test_post_commit_raise_is_a_contract_break(self) -> None:
        for name in ("site launch", "publish"):
            with self.subTest(name):
                do8.reset_for_testing()
                self.cancel.reset_mock()
                dec = self.decision(300)
                calls0 = self.state.calls
                if name == "publish":
                    while (self.state.calls + 1) % bwd_fp8._BWD_REFRESH_CALLS:
                        self.assertIsNotNone(self.s1(self.dout(300, 5), self.w))
                    calls0 = self.state.calls
                    self.publish_exc = RuntimeError("injected publish failure")
                else:
                    self.site_exc = RuntimeError("injected launch-exit hook")
                try:
                    with self.assertRaisesRegex(do8.Mx6ContractError, "bookkeeping ran 1 times"):
                        do8.mx6_produce(dec, self.dout(300, 1), self.w, token_major=True, t_tiled=True)
                finally:
                    self.site_exc = self.publish_exc = None
                self.assertEqual(self.state.calls, calls0 + 1)
                self.assertFalse(dec.open)
                self.cancel.assert_called_once_with("SITE")
                self.assertIn("broke the S1 contract", do8.mx6_broken() or "")


# ------------------------------------------------------------------------------------------------ T8 / T17 rescue
# T17 key rows (v1.2; r16 iteration 59 R-1 condition C3): a write to the weight between S1 and R2.
def _w_nothing(w: torch.nn.Parameter) -> None:
    pass


def _w_no_grad_mul(w: torch.nn.Parameter) -> None:
    with torch.no_grad():
        w.mul_(1.0001)  # in place: _version moves, same storage


def _w_raw_write_bump(w: torch.nn.Parameter) -> None:
    w.data.mul_(1.5)  # a raw write the autograd counter does not see (FusedDenseAdam's kernel) ...
    torch.autograd.graph.increment_version(w)  # ... then its explicit bump (dense_adam_fused.py)


def _w_bump_only(w: torch.nn.Parameter) -> None:
    torch.autograd.graph.increment_version(w)  # values unchanged


def _w_new_storage(w: torch.nn.Parameter) -> None:
    w.data = w.detach() * 2.0  # the same Parameter on new storage: data_ptr moves, _version does not


def _w_raw_write(w: torch.nn.Parameter) -> None:
    w.data.mul_(1.5)  # bypasses _version: no key can see it (W1 has the same limit, DESIGN R7)


def _w_swap_back(w: torch.nn.Parameter) -> None:
    """v1.2 review: a storage swap chain that ends at S1's address. data_ptr() compares addresses, not storage
    identity, and set_data never bumps _version, so the key cannot see it (the same class as a raw write)."""
    s0 = w.data  # an alias of S1's storage with its own version counter
    w.data = s0 * 2.0  # away: new storage at another address, _version unchanged
    s0.mul_(1.5)  # new values through the alias: W's _version does not move
    w.data = s0  # back: S1's storage at S1's address, _version still S1's


# (row, the write between S1 and R2, R2's expected w8 path, expected "w8 is the path's cast of the CURRENT values")
_T17_ROWS: Tuple[Tuple[str, Callable[[torch.nn.Parameter], None], str, bool], ...] = (
    ("hit: nothing between S1 and R2", _w_nothing, "stash", True),
    ("(a) torch.no_grad() W.mul_(1.0001)", _w_no_grad_mul, "miss", True),
    ("(b) raw write W.data.mul_(1.5) + increment_version(W)", _w_raw_write_bump, "miss", True),
    ("(b') increment_version(W) only, values unchanged", _w_bump_only, "miss", True),
    ("(c) W.data = W * 2: the same Parameter on new storage", _w_new_storage, "miss", True),
    ("limit: raw write W.data.mul_(1.5) without a version bump", _w_raw_write, "stash", False),
    ("limit': W.data swapped away and back to S1's storage, written through a .data alias", _w_swap_back, "stash",
     False),
)


def _t17_want(path: str, current: bool) -> Dict[str, Any]:
    """The real R2's observation for a T17 row (RescueStashTest.key_row)."""
    return {"path": path, "lines": 0 if path == "stash" else 1, "own_partials": path == "stash",
            "w8_current": current, "untouched": True}


def _r2_key_mutant(*drop: str) -> Callable[..., Any]:
    """T18 (seam part, v1.2): R2 with every clause of its stash test that mentions one of `drop` removed. An AST edit of
    R2's own source, compiled against bwd_fp8's globals, so the mutant sees the same stash, stubs and line sink."""
    tree = ast.parse(inspect.getsource(bwd_fp8.output_projection_g8_rescue))
    test = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp))
    kept = [v for v in test.values if not any(d in ast.unparse(v) for d in drop)]
    assert len(test.values) - len(kept) == len(drop), (drop, [ast.unparse(v) for v in test.values])
    test.values = kept
    ns: Dict[str, Any] = {}
    code = compile(ast.fix_missing_locations(tree), f"<T18 R2 mutant: key without {' / '.join(drop)}>", "exec")
    exec(code, bwd_fp8.__dict__, ns)  # noqa: S102 -- a test mutant of the code under test
    return ns["output_projection_g8_rescue"]


class RescueStashTest(_SeamCase):
    """T8 details and T17: R2's arguments, its w8 from S1's stash, the fallback and the reset. T18 (seam part): the
    stash-key mutants."""

    def setUp(self) -> None:
        super().setUp()
        self.w = self.param()
        self.state = self.warm(self.w)
        self._keep: List[torch.nn.Parameter] = []  # key_row's weights stay alive for the test

    def test_r2_is_the_fp8_dual_cast_at_the_captured_bits(self) -> None:
        dout = self.dout(769, 3)
        # take the site to a refresh call whose AUTO publish steps the shift up (an e4m3-clipping window)
        while (self.state.calls + 1) % bwd_fp8._BWD_REFRESH_CALLS:
            self.next_partial = _partial(769, "max", 50.0, self.state.calls)
            self.assertIsNotNone(self.s1(dout, self.w))
        shift0, inv0 = self.state.t_shift, _bits(self.state.inv_scale)
        self.next_partial = _partial(769, "max", 1.0e6, 0)
        ops6 = self.s1(dout, self.w)
        self.assertEqual(self.state.t_shift, shift0 + 1)  # published, stepped up
        self.assertNotEqual(_bits(self.state.inv_scale), inv0)
        self.assertEqual(_bits(ops6.inv_scale_used), inv0)
        before, dt0, dout0 = _snapshot(), ops6.dout_fp8.view(U8).clone(), (dout.clone(), dout._version)
        nd = len(self.dual_calls)
        out = bwd_fp8.output_projection_g8_rescue(dout, self.w, ops6)
        self.assertEqual(len(self.dual_calls), nd + 1)
        rc = self.dual_calls[-1]
        self.assertTrue(rc["value"] is dout and rc["inv"] is ops6.inv_scale_used and rc["pos"] == ())
        self.assertEqual(rc["kwargs"], {"t_dtype": E4, "t_inv_mul": 2.0 ** -shift0, "t_limit": 448.0,
                                        "return_partial": True, "t_tiled": True})
        self.assertEqual(rc["inv_bits"], inv0)
        self.assertEqual(_snapshot(), before)
        self.assertTrue(torch.equal(ops6.dout_fp8.view(U8), dt0))
        self.assertTrue(torch.equal(dout, dout0[0]) and dout._version == dout0[1])
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        self.assertIs(out[1], ops6.g_decode_scale)
        self.assertIsNone(do8._mx6_rescue_mismatch(out, 769, CPU))
        self.assertTrue(torch.equal(out[0].view(U8), _ref_cast(dout, ops6.inv_scale_used, E5, 57344.0).view(U8)))

    def test_stash_hit_uses_s1s_partials_with_the_same_objects(self) -> None:
        dout = self.dout(300, 1)
        ops6 = self.s1(dout, self.w)
        w_partial = self.site_calls[-1]["ret"][0]
        stash = bwd_fp8._MX6_RESCUE_STASH
        self.assertTrue(stash[0] is ops6.inv_scale_used and stash[1] is w_partial and stash[2] is self.w)
        self.assertEqual(len(stash), 5)  # v1.2 (C3): + the weight's _version and data_ptr() at S1
        self.assertEqual(stash[3:], (int(self.w._version), self.w.data_ptr()))
        self.assertTrue(type(stash[3]) is int and type(stash[4]) is int)
        nf, nq = len(self.fwq_calls), len(self.qw_calls)
        out = bwd_fp8.output_projection_g8_rescue(dout, self.w, ops6)
        self.assertEqual((len(self.fwq_calls), len(self.qw_calls)), (nf + 1, nq))
        self.assertTrue(self.fwq_calls[-1][0] is self.w and self.fwq_calls[-1][1] is w_partial)
        ref = _ref_weight_quant(self.w, w_partial, E4)
        self.assertTrue(torch.equal(out[2].view(U8), ref[0].view(U8)) and _bits(out[3]) == _bits(ref[1]))
        self.assertNotIn("rescue w8", self.cap.text)

    def test_misses_use_quantize_weight_with_one_line(self) -> None:
        dout = self.dout(300, 1)
        ops_a = self.s1(dout, self.w)
        ops_b = self.s1(dout, self.w)  # the stash now names call b
        twin = torch.nn.Parameter(self.w.detach().clone())  # same values, another object
        for name, weight, ops6, fused in (("older call", self.w, ops_a, False), ("other weight", twin, ops_b, False),
                                          ("hit", self.w, ops_b, True)):
            with self.subTest(name):
                nf, nq = len(self.fwq_calls), len(self.qw_calls)
                f0 = _fields(self.state)
                out = bwd_fp8.output_projection_g8_rescue(dout, weight, ops6)
                self.assertEqual((len(self.fwq_calls) - nf, len(self.qw_calls) - nq), (1, 0) if fused else (0, 1))
                if not fused:
                    self.assertIs(self.qw_calls[-1], weight)
                self.assertEqual(_fields(self.state), f0)
                self.assertIsNone(do8._mx6_rescue_mismatch(out, 300, CPU))
        lines = [ln for ln in _outproj_lines(self.cap.text) if "rescue w8" in ln]
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("[outproj-mx6] rescue w8 via quantize_weight (no matching S1 stash)"))
        # reset: the stash and the lines are cleared (the next miss prints again)
        bwd_fp8.reset_mx6_for_testing()
        self.assertIsNone(bwd_fp8._MX6_RESCUE_STASH)
        nq = len(self.qw_calls)
        bwd_fp8.output_projection_g8_rescue(dout, self.w, ops_b)
        self.assertEqual(len(self.qw_calls), nq + 1)
        self.assertEqual(len([ln for ln in _outproj_lines(self.cap.text) if "rescue w8" in ln]), 2)
        self.assert_lines_clean()

    def test_reset_calls_the_loader_reset_only_when_imported(self) -> None:
        with mock.patch.object(odm, "reset_for_testing") as reset:
            bwd_fp8.reset_mx6_for_testing()
        reset.assert_called_once_with()
        # not imported: any import attempt now fails (None in sys.modules, no attribute on the package)
        pkg = sys.modules["generative_recommenders.ops.hip"]
        attr = pkg.__dict__.pop("outproj_dual_mx6", None)
        try:
            with mock.patch.dict(sys.modules, {_LOADER: None}):
                bwd_fp8.reset_mx6_for_testing()  # must neither import the loader nor fail
                self.assertIsNone(sys.modules[_LOADER])
                self.assertIsNone(bwd_fp8._MX6_RESCUE_STASH)
        finally:
            if attr is not None:
                setattr(pkg, "outproj_dual_mx6", attr)

    def test_prebuild_hook_is_the_loaders_prebuild(self) -> None:
        """bwd_fp8.prebuild_outproj_mx6() (DESIGN 12.6; OPTIONAL and not called by r16, r16 iteration 59 R-3):
        outproj_dual_mx6.prebuild(), every exception unchanged; no scale state, stash or TM pool block. v1.2 (R-3, text
        and this test only): a device-side error in r14's self-check arrives as the ORIGINAL exception, and a launcher
        rc 700 / 719 there is an OutprojMx6Error whose text matches l11_outproj_lnmd._DEVICE_FAULT. So a caller must
        classify by that text in its own try (re-raise on a match, latch otherwise), never by isinstance. v1.2 review:
        the docstring also gives the try's placement (after r16's INERT rules, before `try: _load_mx6()`, never in
        prebuild_extensions / prebuild_l11_do8) and the scope of the text rule. v1.3 (test-only): r16's C3-2 v2 row
        (ROW_c3_2_device_fault_v2.patch, sha256 50ce60095a9036ed6b81ff4845c016a6207876e470ac5394eb6bcb9203914ff7)
        adds the five hipError names of rc 53 / 214 / 702 / 709 / 710 to _DEVICE_FAULT, so those loader texts match
        too and the rows below pin them True (v1.2 pinned 53 / 214 / 702 / 710 False, the documented gap; 709 is new).
        The hook docstring's scope sentence (rc 700 / 719 match; rc 53, 214, 702 and 710 latch) is v1.2 source text
        that this test-only build does not change: the part check below still pins it as written, and it now
        describes the pre-C3-2 state."""

        def rc_error(rc: int) -> BaseException:
            try:
                odm._check_rc(rc, "outproj_dual_mx6_launch", 769)  # the loader's own rc > 0 exception
            except odm.OutprojMx6Error as error:
                return error
            raise AssertionError(rc)

        latched = odm.OutprojMx6Error("[outproj-mx6] unavailable: injected")
        rows = ((latched, False), (RuntimeError("HIP error: an illegal memory access was encountered"), True),
                (rc_error(700), True), (rc_error(719), True),
                # device-fatal codes whose loader text matches _DEVICE_FAULT since r16's C3-2 v2 row (50ce6009; they
                # re-raise). v1.2 pinned 53 / 214 / 702 / 710 False (the documented gap); 709 is the fifth C3-2 name
                (rc_error(53), True), (rc_error(214), True), (rc_error(702), True), (rc_error(709), True),
                (rc_error(710), True))
        before, n_tm = _snapshot(), len(self.empty_tm_calls)
        with mock.patch.object(odm, "prebuild") as pre:
            self.assertIsNone(bwd_fp8.prebuild_outproj_mx6())
            pre.assert_called_once_with()
            for exc, fault in rows:
                with self.subTest(str(exc)[:60]):
                    pre.side_effect = exc
                    with self.assertRaises(type(exc)) as ctx:
                        bwd_fp8.prebuild_outproj_mx6()
                    self.assertIs(ctx.exception, exc)  # unchanged: the caller sees what prebuild raised
                    self.assertEqual(bool(l11._DEVICE_FAULT.search(f"{type(exc).__name__}: {exc}")), fault)
        self.assertIsInstance(rows[2][0], odm.OutprojMx6Error)  # rc 700: an OutprojMx6Error that must re-raise
        self.assertIn("hipErrorIllegalAddress", str(rows[2][0]))
        self.assertIn("hipErrorLaunchFailure", str(rows[3][0]))
        self.assertEqual((_snapshot(), len(self.empty_tm_calls)), (before, n_tm))
        doc = " ".join((bwd_fp8.prebuild_outproj_mx6.__doc__ or "").split())
        for part in ("OPTIONAL; r16 does not call it", "OWN try", "_DEVICE_FAULT", "try: _load_mx6()",
                     "isinstance(error, OutprojMx6Error)", "rc 700 / 719", "after r16's INERT rules",
                     "triton_fp8.prebuild_extensions", "prebuild_l11_do8", "rc 53, 214, 702 and 710"):
            self.assertIn(part, doc)
        self.assertNotIn("requested of r16", doc)

    def test_only_successful_calls_set_the_stash(self) -> None:
        dout = self.dout(300, 1)
        ops6 = self.s1(dout, self.w)
        stash = bwd_fp8._MX6_RESCUE_STASH
        self.assertIsNone(self.s1(dout, self.w, t_tiled=False))  # a decline
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH, stash)
        self.producer_rc = 1
        with self.assertRaises(odm.OutprojMx6Error):  # a pre-commit raise
            self.s1(dout, self.w)
        self.producer_rc = 0
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH, stash)
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH[0], ops6.inv_scale_used)

    # ---------------------------------------------------------------- T17 key rows / T18 key mutants (v1.2, C3)
    def _w8_lines(self) -> List[str]:
        return [ln for ln in _outproj_lines(self.cap.text) if "rescue w8 via quantize_weight" in ln]

    def key_row(self, r2: Callable[..., Any], write: Callable[[torch.nn.Parameter], None], seed: int
                ) -> Dict[str, Any]:
        """One T17 row: S1 on a fresh published weight (the stash names this call), `write(w)` between S1 and R2, then
        `r2` (R2, or a T18 mutant). What R2 did: its w8 path ("stash" = _fused_weight_quant, "miss" = quantize_weight),
        the miss lines it printed, whether a hit used THIS call's w_partial, whether (w8, w8_scale) are the path's own
        cast of the CURRENT weight values, and whether every state and r16's form check stayed as they were."""
        w = self.param(seed)
        self._keep.append(w)  # the weight objects stay alive for the test
        self.warm(w)
        dout = self.dout(300, seed)
        ops6 = self.s1(dout, w)
        self.assertIsNotNone(ops6)
        w_partial = self.site_calls[-1]["ret"][0]
        stash = bwd_fp8._MX6_RESCUE_STASH
        self.assertTrue(stash[0] is ops6.inv_scale_used and stash[1] is w_partial and stash[2] is w)
        self.assertEqual(stash[3:], (int(w._version), w.data_ptr()))  # the key as S1 captured it
        # pin S1's storage too: a row that swaps W's storage (c) would otherwise free an address whose site entry is
        # live, and a later row's weight landing there would find an already-published site (allocator-dependent)
        self._keep.append(w.detach())
        write(w)
        bwd_fp8._MX6_LINES.discard(("rescue-w8", ""))  # each row may print the once-per-process miss line again
        n_lines, nf, nq = len(self._w8_lines()), len(self.fwq_calls), len(self.qw_calls)
        before = _snapshot()
        out = r2(dout, w, ops6)
        untouched = _snapshot() == before and do8._mx6_rescue_mismatch(out, 300, CPU) is None
        path = {(1, 0): "stash", (0, 1): "miss"}.get((len(self.fwq_calls) - nf, len(self.qw_calls) - nq), "?")
        ref = _ref_weight_quant(w, _ref_weight_tiles(w), E4) if path == "stash" else self._real_qw(w)
        return {"path": path, "lines": len(self._w8_lines()) - n_lines,
                "own_partials": path == "stash" and self.fwq_calls[-1][1] is w_partial,
                "w8_current": bool(torch.equal(out[2].view(U8), ref[0].view(U8)) and _bits(out[3]) == _bits(ref[1])),
                "untouched": untouched}

    def test_t17_key_rows(self) -> None:
        """T17 (v1.2; r16 R-1 C3, S1 contract v3 note): R2 hits only if the stash names this call's inv_scale_used and
        weight objects AND the weight's _version and data_ptr() are the ones S1 captured. A _version-visible write, or
        a storage swap that leaves W at another address, between S1 and R2 is a miss: quantize_weight of the current
        values plus the one line. A write that bypasses _version, or a swap chain that ends at S1's storage and
        address, stays a hit with S1's stale partials (the documented limits; r16's seam invariant C2 keeps the
        window empty)."""
        for i, (name, write, path, current) in enumerate(_T17_ROWS):
            with self.subTest(name):
                self.assertEqual(self.key_row(bwd_fp8.output_projection_g8_rescue, write, 60 + i),
                                 _t17_want(path, current))
        for ln in self._w8_lines():
            self.assertEqual(ln, "[outproj-mx6] rescue w8 via quantize_weight (no matching S1 stash)")
        self.assertEqual(len(self._w8_lines()), sum(p == "miss" for _, _, p, _ in _T17_ROWS))
        self.assert_lines_clean()

    def test_t17_key_is_captured_before_the_commit(self) -> None:
        """C3's placement: the key is taken inside S1's prepare try, after W1 (odm.w6_shadow) and before the commit,
        and the stash is (inv_used, w_partial, weight, w_version, w_ptr). A failure to read it is therefore a pre-commit
        raise: the state is untouched (calls delta 0), one "raised before its commit" line, the stash kept."""
        s1 = ast.parse(inspect.getsource(bwd_fp8.output_projection_dgrad_operands_mx6)).body[0]
        prep = next(n for n in ast.walk(s1) if isinstance(n, ast.Try)
                    and any("w6_shadow" in ast.unparse(s) for s in n.body))
        body = [ast.unparse(s) for s in prep.body]
        i_w1 = next(i for i, s in enumerate(body) if "w6_shadow" in s)
        self.assertEqual(body[i_w1 + 1:], ["w_version, w_ptr = (int(weight._version), int(weight.data_ptr()))"])
        stash = [n for n in ast.walk(s1) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_MX6_RESCUE_STASH" for t in n.targets)]
        self.assertEqual([ast.unparse(n.value) for n in stash], ["(inv_used, w_partial, weight, w_version, w_ptr)"])
        commit = next(n for n in ast.walk(s1)
                      if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "state.calls")
        self.assertLess(prep.end_lineno, commit.lineno)
        self.assertLess(commit.lineno, stash[0].lineno)

        class _NoVersion(torch.nn.Parameter):  # W1 reads getattr(w, "_version", 0); S1's key reads w._version
            @property
            def _version(self) -> int:  # pyre-ignore[3]
                raise AttributeError("injected: _version unreadable")

        w = _NoVersion(self.param(71).detach().clone())
        st = self.warm(w)
        stash0, f0 = bwd_fp8._MX6_RESCUE_STASH, _fields(st)
        with _Capture() as cap:
            with self.assertRaisesRegex(AttributeError, "injected: _version unreadable"):
                self.s1(self.dout(300, 71), w)
        self.assertEqual(_fields(st), f0)  # pre-commit: calls delta 0
        self.assertIs(bwd_fp8._MX6_RESCUE_STASH, stash0)
        self.assertEqual(_outproj_lines(cap.text),
                         ["[outproj-mx6] S1 raised before its commit (scale state untouched): AttributeError: injected: "
                          "_version unreadable"])

    def test_t18_key_mutants_are_caught_by_the_t17_rows(self) -> None:
        """T18 (seam part, v1.2): mutants of R2 with a weaker stash key. A row catches a mutant when the mutant's
        observation differs from the real R2's. The key without _version is caught by (a), (b) and (b'); the key
        without data_ptr() by (c); v1.1's identity-only key by all four. No mutant changes the hit or the two limit
        rows."""
        want = {name: _t17_want(path, current) for name, _, path, current in _T17_ROWS}
        for mutant, drop, must in (
            ("stash ignores _version", ("_version",), {"(a)", "(b)", "(b')"}),
            ("stash ignores data_ptr()", ("data_ptr",), {"(c)"}),
            ("v1.1 identity-only key", ("_version", "data_ptr"), {"(a)", "(b)", "(b')", "(c)"}),
        ):
            with self.subTest(mutant):
                r2m = _r2_key_mutant(*drop)
                self.assertIsNot(r2m, bwd_fp8.output_projection_g8_rescue)
                caught = set()
                for i, (name, write, _, _) in enumerate(_T17_ROWS):
                    if self.key_row(r2m, write, 60 + i) != want[name]:
                        caught.add(name.split()[0])
                self.assertEqual(caught, must, mutant)
        # the builder edits the real source: without dropping anything it is R2 itself, and it passes every row
        same = _r2_key_mutant()
        for i, (name, write, path, current) in enumerate(_T17_ROWS):
            self.assertEqual(self.key_row(same, write, 80 + i), want[name], name)


# ------------------------------------------------------------------------------------------------ T9 TSPEC key model
_I32MAX = 2 ** 31 - 1
_OPTS = "{'num_warps': 8, 'debug': False, 'instrumentation_mode': ''}"
_PROD_M = (("lab", 2342736), ("ts73", 2546840), ("census15", 2499798), ("1M", 1000000))


def _ptr(nbytes: int, aligned: bool = True) -> str:
    """AMD Triton 3.6 pointer specialization: 'D' = 16-B aligned, 'S' = storage <= 2^31 - 1 B (use_buffer_ops)."""
    return ("D" if aligned else "") + ("S" if nbytes <= _I32MAX else "")


def _ival(v: int) -> Tuple[str, Any]:
    return ("constexpr", 1) if v == 1 else ("i32", "D" if v % 16 == 0 else "")


def _dual_key(m: int, kw: Dict[str, Any], src_bytes: int, out_t_bytes: int, inv_aligned: bool,
              inv_bytes: int) -> str:
    """quantize_row_major_dual(dout bf16 [m, 512], inv, **kw) -> _quantize_amax_dual_kernel's key (BF:327-406)."""
    t_dtype, t_inv_mul, t_limit = kw.get("t_dtype"), float(kw.get("t_inv_mul", 1.0)), kw.get("t_limit")
    t_tiled = bool(kw.get("t_tiled", False))
    sep = t_dtype is not None and (t_dtype != E5 or t_inv_mul != 1.0)
    m_pad = (m + (128 if t_tiled else 16) - 1) // (128 if t_tiled else 16) * (128 if t_tiled else 16)
    t_name = {E4: "*fp8e4nv", E5: "*fp8e5"}[t_dtype if sep else E5]
    parts = [("*bf16", _ptr(src_bytes)), ("*fp8e5", _ptr(m * 512)), (t_name, _ptr(out_t_bytes)),
             ("*fp32", _ptr(4 * 4 * ((m_pad + 255) // 256))), ("*fp32", _ptr(inv_bytes, inv_aligned)),
             _ival(m), _ival(m_pad), ("constexpr", 512), _ival(512), _ival(m_pad), ("constexpr", 256),
             ("constexpr", 128), ("constexpr", 57344.0), ("constexpr", sep), ("constexpr", t_inv_mul if sep else 1.0),
             ("constexpr", float(t_limit if t_limit is not None else 57344.0) if sep else 1.0), ("constexpr", t_tiled)]
    return repr(parts) + _OPTS


def _g8_only_key(m: int) -> str:
    """quantize_row_major(dout, inv) -> _quantize_amax_kernel bf16 -> e5m2 (BF:861-898)."""
    n = m * 512
    parts = [("*bf16", _ptr(2 * n)), ("*fp8e5", _ptr(n)), ("*fp32", _ptr(4 * ((n + 8191) // 8192))), ("*fp32", _ptr(4)),
             _ival(n), ("constexpr", 8192), ("constexpr", 57344.0)]
    return repr(parts) + _OPTS


def _quantize_weight_key() -> str:
    n = NOUT * D
    parts = [("*fp32", _ptr(4 * n)), ("*fp8e4nv", _ptr(n)), ("*fp32", _ptr(4 * ((n + 8191) // 8192))),
             ("*fp32", _ptr(4)), _ival(n), ("constexpr", 8192), ("constexpr", 448.0)]
    return repr(parts) + _OPTS


def _weight_quant_key(weight: torch.Tensor, w_partial: torch.Tensor) -> str:
    """_fused_weight_quant(W, w_partial) -> _fp8h_weight_quant_kernel (FHF:476-501)."""
    n, p = weight.numel(), w_partial.numel()
    w_name = {F32: "*fp32", BF: "*bf16"}[weight.dtype]
    parts = [(w_name, _ptr(weight.element_size() * n)), ("*fp8e4nv", _ptr(n)), ("*fp32", _ptr(4)),
             ("*fp32", _ptr(4 * p)), _ival(p), _ival(n), ("fp32", None), ("fp32", None), ("constexpr", 8192),
             ("constexpr", max(16, 1 << (p - 1).bit_length())), ("constexpr", 448.0)]
    return repr(parts) + _OPTS


def _site_key(weight: torch.Tensor, n_site: int, t_note: bool, snapshot: bool, snapshot_t: bool) -> str:
    """_fused_site_launch -> _fp8h_amax_tiles_kernel with the site program (FHF:338-404), fp32 / bf16 [1536, 512]."""
    r, c = int(weight.shape[0]), int(weight.shape[1])
    nc, n_tiles = -(-c // 256), -(-r // 32) * -(-c // 256)
    w_name = {F32: "*fp32", BF: "*bf16"}[weight.dtype]
    big = 1 << 30
    parts = [(w_name, _ptr(weight.element_size() * r * c)), ("*fp32", _ptr(4 * n_tiles)), _ival(r), _ival(c),
             _ival(c), ("constexpr", 1), _ival(big), _ival(big), _ival(big), _ival(0), _ival(0), _ival(0), _ival(0),
             _ival(nc), _ival(n_tiles), ("*fp32", _ptr(4 * n_site)), _ival(n_site), ("*fp32", "DS"), ("*i64", "DS"),
             ("*i64", "DS"), ("*fp32", "DS"), ("*fp32", "DS"), ("*fp32", "DS"), ("fp32", None), ("fp32", None),
             ("fp32", None), ("constexpr", 1), ("constexpr", 32), ("constexpr", 256), ("constexpr", True),
             ("constexpr", bool(t_note)), ("constexpr", bool(snapshot)), ("constexpr", bool(snapshot_t)),
             ("constexpr", 16384)]
    return repr(parts) + _OPTS


class TspecKeyTest(_SeamCase):
    """T9: the specializations R2 and S1's commit launch, built from their RECORDED calls with production-M storage
    sizes, are rows of ops/triton/triton_spec_table.jsonl (no JIT after RUN_START); the deviations SPEC_SEAM 4.1 names
    are not (pins critique C7 against future table re-records)."""

    def setUp(self) -> None:
        super().setUp()
        rows = [json.loads(line) for line in _TSP_PATH.read_text().splitlines() if line.strip()]
        self.keys = {r["key"] for r in rows[1:]}
        self.w = self.param()
        self.state = self.warm(self.w)
        self.assertEqual(self.state.t_shift, 5)  # the production AUTO floor
        self.dout_ = self.dout(300, 1)
        self.ops6 = self.s1(self.dout_, self.w)
        self.site = self.site_calls[-1]
        self.r2 = bwd_fp8.output_projection_g8_rescue(self.dout_, self.w, self.ops6)
        self.r2_call = self.dual_calls[-1]
        self.fwq = self.fwq_calls[-1]

    def test_r2_dual_cast_key_is_a_table_row(self) -> None:
        inv = self.r2_call["inv"]
        self.assertIs(inv, self.ops6.inv_scale_used)
        own = (inv.data_ptr() % 16 == 0, inv.untyped_storage().nbytes())
        self.assertEqual(own, (True, 4))
        for label, m in _PROD_M:
            with self.subTest(label):
                self.assertIn(_dual_key(m, self.r2_call["kwargs"], m * 1024, 2 ** 31, *own), self.keys)  # pool out_t
                self.assertNotIn(_dual_key(m, self.r2_call["kwargs"], m * 1024, _m_pad(m) * 512, *own), self.keys)
                self.assertNotIn(_dual_key(m, self.r2_call["kwargs"], m * 1024, 2 ** 31, False, 4), self.keys)
        for shift in (6, 7):  # the FP8 path's own coverage at these shifts: no post T_TILED row either
            kw = dict(self.r2_call["kwargs"], t_inv_mul=2.0 ** -shift)
            self.assertNotIn(_dual_key(2546840, kw, 2546840 * 1024, 2 ** 31, *own), self.keys)

    def test_g8_only_cast_is_not_a_row_at_production_m(self) -> None:
        for label, m in _PROD_M[:3]:
            self.assertNotIn(_g8_only_key(m), self.keys, label)
        self.assertIn(_g8_only_key(1000000), self.keys)  # the model reproduces the one row it has

    def test_w8_casts_are_rows(self) -> None:
        w, p = self.fwq
        self.assertTrue(w is self.w and p.numel() == 96)
        self.assertIn(_weight_quant_key(w, p), self.keys)  # the stash path = the FP8 path's per-call w8 cast
        self.assertIn(_quantize_weight_key(), self.keys)  # the fallback (item (35) Q2)

    def test_s1_site_program_key_is_a_row(self) -> None:
        s = self.site
        for label, m in _PROD_M:
            with self.subTest(label):
                self.assertIn(_site_key(self.w, _n_partial(m), s["t_note"], s["snapshot"], s["snapshot_t"]), self.keys)


# ------------------------------------------------------------------------------------------------ T10 W1 cache, S1-E
class W6CacheTest(_SeamCase):
    """T10: outproj_dual_mx6.w6_shadow follows k512_mx6_gemm.mx6_weight's protocol (DESIGN 10), kernel mocked."""

    def shadow(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        n = len(self.w6_calls)
        w6, w6s = odm.w6_shadow(w)
        return w6, w6s, len(self.w6_calls) - n

    def assert_image(self, w6: torch.Tensor, w6s: torch.Tensor, w: torch.Tensor) -> None:
        r6, r6s = _w6_bytes_of(w)
        self.assertTrue(torch.equal(w6, r6) and torch.equal(w6s, r6s))
        self.assertEqual((w6.dtype, w6.numel(), w6s.numel()), (U8, W6_BYTES, W6S_BYTES))

    def test_hit_and_refresh_rules(self) -> None:
        w = self.param(1)
        a6, a6s, n = self.shadow(w)
        self.assertEqual(n, 1)
        self.assert_image(a6, a6s, w)
        b6, b6s, n = self.shadow(w)
        self.assertEqual(n, 0)
        self.assertTrue(b6 is a6 and b6s is a6s)
        self.assertEqual(odm.w6_cache_stats(), {"entries": 1, "hits": 1, "refreshes": 1, "pruned": 0})
        old = (a6.clone(), a6s.clone())
        with torch.no_grad():
            w.mul_(2.0)  # an optimizer step in place: _version moves
        c6, c6s, n = self.shadow(w)
        self.assertEqual(n, 1)
        self.assertTrue(c6 is not a6 and c6s is not a6s)  # NEW buffers: never written under a queued reader
        self.assertTrue(torch.equal(a6, old[0]) and torch.equal(a6s, old[1]))
        self.assert_image(c6, c6s, w)
        torch.autograd.graph.increment_version(w)  # dense Adam's explicit bump, values unchanged
        d6, _, n = self.shadow(w)
        self.assertEqual(n, 1)
        self.assertIsNot(d6, c6)
        w.data = w.data.clone()  # the same object on new storage: data_ptr moves
        e6, _, n = self.shadow(w)
        self.assertEqual(n, 1)
        self.assertIsNot(e6, d6)
        self.assertEqual(self.shadow(w)[2], 0)
        self.assertEqual(odm.w6_cache_stats()["entries"], 1)

    def test_identity_rules(self) -> None:
        w = self.param(1)
        self.shadow(w)
        twin = torch.nn.Parameter(w.detach().clone())  # same values, another tensor
        alias = torch.nn.Parameter(w.detach())  # same storage and data_ptr, another object
        self.assertEqual(alias.data_ptr(), w.data_ptr())
        for other in (twin, alias):
            o6, o6s, n = self.shadow(other)
            self.assertEqual(n, 1)
            self.assert_image(o6, o6s, w)
        self.assertEqual(odm.w6_cache_stats()["entries"], 3)
        self.assertEqual(self.shadow(w)[2], 0)
        # a new tensor at a freed weight's id and data_ptr (a dead weakref under the same key) never hits
        tmp = torch.nn.Parameter(torch.zeros(1))
        dead = weakref.ref(tmp)
        del tmp
        gc.collect()
        self.assertIsNone(dead())
        w5 = self.param(5)
        key = (id(w5), 0, (NOUT, D), (D, 1), F32, "L11")
        stale = (torch.full((W6_BYTES,), 7, dtype=U8), torch.full((W6S_BYTES,), 7, dtype=U8))
        odm._w6_cache[key] = odm._W6Entry(dead, int(w5._version), w5.data_ptr(), *stale)
        f6, f6s, n = self.shadow(w5)
        self.assertEqual(n, 1)
        self.assertTrue(f6 is not stale[0] and f6s is not stale[1])
        self.assert_image(f6, f6s, w5)

    def test_freed_weights_are_pruned_on_refresh(self) -> None:
        w = self.param(1)
        self.shadow(w)
        tmp = self.param(2)
        self.shadow(tmp)
        self.assertEqual(odm.w6_cache_stats()["entries"], 2)
        del tmp
        gc.collect()
        self.assertEqual(self.shadow(w)[2], 0)  # a hit prunes nothing
        self.assertEqual(odm.w6_cache_stats()["pruned"], 0)
        w3 = self.param(3)
        self.assertEqual(self.shadow(w3)[2], 1)
        st = odm.w6_cache_stats()
        self.assertEqual((st["entries"], st["pruned"]), (2, 1))
        self.assertTrue(all(e.ref() is not None for e in odm._w6_cache.values()))

    def test_dtypes_and_latch(self) -> None:
        wb = self.param(1, dtype=BF)
        b6, b6s, n = self.shadow(wb)
        self.assertEqual((n, self.w6_calls[-1]["bf16"]), (1, 1))
        self.assert_image(b6, b6s, wb)  # widened exactly
        for name, bad, exc, want in (
            ("fp16", torch.nn.Parameter(torch.zeros((NOUT, D), dtype=torch.float16)), ValueError, "dtype"),
            ("shape", torch.nn.Parameter(torch.zeros((D, NOUT))), ValueError, "shape"),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(exc, want):
                    odm.w6_shadow(bad)
        odm.set_prebuilt_for_testing(False)
        with self.assertRaisesRegex(odm.OutprojMx6Error, "not prebuilt"):
            odm.w6_shadow(wb)
        odm.set_prebuilt_for_testing(True)
        with _Capture():
            odm._fail("injected latch")
        with self.assertRaisesRegex(odm.OutprojMx6Error, "injected latch"):
            odm.w6_shadow(wb)

    def test_s1_e_is_stateless(self) -> None:
        """S1-E (S1 contract sec. 7): no site, scale state, stash, W1 entry or TM pool block; G6 / W6 / W6S identical to
        S1's / W1's for the same operands; INV 1.0, t 2^-5, no INV_USED, plain scratch."""
        w = self.param(1)
        self.warm(w)
        dout = self.dout(769, 4)
        ops6 = self.s1(dout, w)
        synthetic = torch.randn((NOUT, D), generator=torch.Generator().manual_seed(3)) * 0.05  # not a Parameter
        before, n_tm = _snapshot(), len(self.empty_tm_calls)
        real_prebuild = odm.prebuild
        order: List[str] = []
        self.events.clear()
        with mock.patch.object(odm, "prebuild", lambda: (order.append("prebuild"), real_prebuild())[1]):
            g6, w6, w6s = bwd_fp8.output_projection_mx6_encode(dout, synthetic)
        self.assertEqual(_snapshot(), before)
        self.assertEqual(len(self.empty_tm_calls), n_tm)
        self.assertEqual(order + self.events, ["prebuild", "producer", "w6"])
        p = self.producer_calls[-1]
        self.assertEqual((p["inv_bits"], p["t_inv_mul"], p["inv_used"], p["m"], p["m_pad"]),
                         (0x3F800000, 2.0 ** -5, None, 769, 896))
        self.assertTrue(torch.equal(g6, ops6.g6))  # the same producer on the same dout bits
        self.assertEqual((g6.dtype, tuple(g6.shape), g6.stride()), (U8, (769, G6_ROW), (G6_ROW, 1)))
        twin = torch.nn.Parameter(synthetic.clone())
        t6, t6s = odm.w6_shadow(twin)
        self.assertTrue(torch.equal(w6, t6) and torch.equal(w6s, t6s))  # W1's bytes for the same values
        self.assertIsNone(do8._mx6_operand_mismatch(g6, w6, w6s, 769, CPU))
        # the training weight's entry is not touched either
        self.assertIs(odm.w6_shadow(w)[0], ops6.w6)

    def test_s1_e_copies_and_checks_its_inputs(self) -> None:
        nc = torch.zeros((D, 300), dtype=BF).t()
        mis = torch.empty(300 * D + 8, dtype=BF)[1:1 + 300 * D].view(300, D)
        one = torch.zeros((2, 2 * D), dtype=BF)[:1, :D]  # F1 (v1.1): is_contiguous(), row stride 1024
        for name, dout in (("non-contiguous", nc), ("misaligned", mis), ("M = 1, row stride 1024", one)):
            with self.subTest(name):
                g6, _, _ = bwd_fp8.output_projection_mx6_encode(dout, self.param(1))
                p = self.producer_calls[-1]
                self.assertNotEqual(p["dout"], dout.data_ptr())
                self.assertEqual(p["dout"] % 16, 0)
                self.assertEqual(tuple(g6.shape), (int(dout.shape[0]), G6_ROW))
        w = self.param(1)
        for name, dout, weight in (
            ("fp32 dout", torch.zeros((4, D)), w), ("width", torch.zeros((4, 256), dtype=BF), w),
            ("M = 0", torch.zeros((0, D), dtype=BF), w), ("3-D", torch.zeros((1, 4, D), dtype=BF), w),
            ("weight shape", torch.zeros((4, D), dtype=BF), torch.zeros((D, NOUT))),
            ("weight fp16", torch.zeros((4, D), dtype=BF), torch.zeros((NOUT, D), dtype=torch.float16)),
        ):
            with self.subTest(name):
                n = len(self.producer_calls)
                with self.assertRaisesRegex(ValueError, r"^\[outproj-mx6\] S1-E needs"):
                    bwd_fp8.output_projection_mx6_encode(dout, weight)
                self.assertEqual(len(self.producer_calls), n)
        odm.set_prebuilt_for_testing(False)
        with _Capture():
            odm._fail("injected latch")
        n = len(self.producer_calls)
        with self.assertRaisesRegex(odm.OutprojMx6Error, "injected latch"):
            bwd_fp8.output_projection_mx6_encode(self.dout(4, 1), w)
        self.assertEqual(len(self.producer_calls), n)


if __name__ == "__main__":
    unittest.main()
