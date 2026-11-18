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

"""DLRM_HSTU_FP8_TM_NONEXP (ops/triton/fp8_tm_alloc.py): token-major FP8 operands from a
private non-expandable pool. The knob and the allocator conf are read at import, so the
armed cases run in a subprocess with the production allocator setting."""

import json
import os
import subprocess
import sys
import unittest

import torch

_CHILD = r"""
import json, torch
from generative_recommenders.ops.triton import fp8_tm_alloc as TM
dev = torch.device("cuda", 0)
GIB = 1 << 30

def seg(t):
    p = t.data_ptr()
    for s in torch.cuda.memory_snapshot():
        a = s.get("address", 0)
        if a <= p < a + s.get("total_size", 0):
            return bool(s.get("is_expandable")), str(s.get("segment_pool_id")), s.get("total_size")
    return None, None, None

r = {"active": TM.active()}
r["prewarm_bytes"] = TM.prewarm(2, dev)
a = TM.empty_tm((512, 1000064), torch.float8_e4m3fn, dev)
r["shape"] = list(a.shape); r["stride"] = list(a.stride()); r["offset"] = a.storage_offset()
r["dtype"] = str(a.dtype); r["contig"] = a.is_contiguous()
exp, pool, size = seg(a)
r["a_expandable"] = exp; r["a_pool_private"] = pool != "(0, 0)"; r["a_block"] = size
pa = a.data_ptr()
a.view(torch.uint8).fill_(7)
r["a_ok"] = bool((a.view(torch.uint8) == 7).all())
del a
torch.cuda.synchronize()
c = torch.empty((512, 1000064), dtype=torch.float8_e4m3fn, device=dev)  # outside the pool
r["outside_took_pool_block"] = c.data_ptr() == pa
r["outside_expandable"] = seg(c)[0]
b = TM.empty_tm((512, 2000000), torch.float8_e4m3fn, dev)
r["reused_whole_block"] = b.data_ptr() == pa
z = TM.empty_tm((512, 0), torch.float8_e4m3fn, dev)
r["zero_ok"] = list(z.shape) == [512, 0]
big = TM.empty_tm((512, 4200000), torch.float8_e4m3fn, dev)  # above the 2 GiB class
r["big_block"] = seg(big)[2]
print("RESULT " + json.dumps(r))
"""


_CHILD_XT = r"""
import hashlib, json, torch
from generative_recommenders.ops.triton import fp8_tm_alloc as TM
from generative_recommenders.ops.triton import triton_layer_norm as LN
dev = torch.device("cuda", 0)

def private(t):
    p = t.data_ptr()
    for s in torch.cuda.memory_snapshot():
        a = s.get("address", 0)
        if a <= p < a + s.get("total_size", 0):
            return str(s.get("segment_pool_id")) != "(0, 0)" and not s.get("is_expandable")
    return None

def md5(t):
    return hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

r = {"active": TM.active(), "xt_in_pool": TM.xt_in_pool(), "default_blocks": TM.default_blocks()}
r["xt_private"] = private(TM.empty_xt((512, 1024), torch.float8_e4m3fn, dev))
r["dout_t_private"] = private(TM.empty_tm((512, 1024), torch.float8_e4m3fn, dev))
g = torch.Generator(device=dev).manual_seed(5)
x = torch.randn((1000, 512), device=dev, generator=g).to(torch.bfloat16)
w = (1.0 + 0.1 * torch.randn((512,), device=dev, generator=g)).to(torch.bfloat16)
b = (0.1 * torch.randn((512,), device=dev, generator=g)).to(torch.bfloat16)
y, mean, rstd, yt = LN.triton_weighted_layer_norm_fwd_fp8_store_t(x=x, weight=w, bias=b, eps=1e-6, inv_scale=4.0, pad_to=128)
_, yt2 = LN.triton_weighted_layer_norm_recompute_fp8t(x=x, weight=w, bias=b, eps=1e-6, mean=mean, rstd=rstd, inv_scale=4.0, pad_to=128)
torch.cuda.synchronize()
r["store_t_private"] = private(yt)
r["recompute_private"] = private(yt2)
r["shape"] = list(yt.shape) + list(yt2.shape)
r["md5"] = [md5(yt), md5(yt2)]
print("RESULT " + json.dumps(r))
"""


_CHILD_TILED = r"""
import hashlib, io, json, contextlib, os, torch
from generative_recommenders.ops.triton import fp8_tm_alloc as TM
from generative_recommenders.ops.triton import bwd_fp8 as BW
from generative_recommenders.ops.triton import triton_layer_norm as LN
from generative_recommenders.dlrm_v4 import alloc_pregrow as AP
dev = torch.device("cuda", 0)
E4 = torch.float8_e4m3fn

def private(t):
    p = t.data_ptr()
    for s in torch.cuda.memory_snapshot():
        a = s.get("address", 0)
        if a <= p < a + s.get("total_size", 0):
            return str(s.get("segment_pool_id")) != "(0, 0)" and not s.get("is_expandable")
    return None

def md5(t):
    return hashlib.md5(t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

r = {"active": TM.active(), "tiled_pool": TM.tiled_pool(), "all_tiled": TM.pooled_all_tiled()}
# the alloc_pregrow prewarm decision (margin larger than the GPU: no stream pool grows)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    AP.pregrow_before_run_start()
r["pregrow_lines"] = [ln for ln in buf.getvalue().splitlines() if "hstu-fp8-tm" in ln]
r["prewarmed"] = sorted(TM._prewarmed)
# an untiled copy before any pool block exists: the pool only if the prewarm mapped it
r["copy_first_private"] = private(TM.empty_copy((512, 8, 128), E4, dev))
r["tiled_tm_private"] = private(TM.empty_tm((8, 512, 128), E4, dev))
r["tiled_xt_private"] = private(TM.empty_xt((8, 512, 128), E4, dev))
r["rm_tm_private"] = private(TM.empty_tm((512, 1024), E4, dev))
r["rm_xt_private"] = private(TM.empty_xt((512, 1024), E4, dev))
r["copy_private"] = private(TM.empty_copy((512, 8, 128), E4, dev))
if os.environ.get("CHILD_PRODUCERS") == "1":
    # the tiled producers: LN store_t v2 / recompute_fp8t (yT) and the out-proj dual cast (dout_t)
    LN._STORE_T_V2 = True
    g = torch.Generator(device=dev).manual_seed(7)
    x = (torch.randn((1000, 512), device=dev, generator=g) * 1.7 + 0.3).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn((512,), device=dev, generator=g)).to(torch.bfloat16)
    b = (0.1 * torch.randn((512,), device=dev, generator=g)).to(torch.bfloat16)
    y, mean, rstd, yt = LN.triton_weighted_layer_norm_fwd_fp8_store_t(
        x=x, weight=w, bias=b, eps=1e-6, inv_scale=40.0, pad_to=128, tiled_xt=True)
    _, yt2 = LN.triton_weighted_layer_norm_recompute_fp8t(
        x=x, weight=w, bias=b, eps=1e-6, mean=mean, rstd=rstd, inv_scale=40.0, pad_to=128, tiled_xt=True)
    src = (torch.randn((1000, 512), device=dev, generator=g) * 1e-2).to(torch.bfloat16)
    inv = torch.full((1,), 2.0 ** 10, dtype=torch.float32, device=dev)
    q, dt, amax = BW.quantize_row_major_dual(src, inv, t_dtype=E4, t_inv_mul=2.0 ** -5, t_limit=448.0, t_tiled=True)
    torch.cuda.synchronize()
    r["prod_shapes"] = [list(yt.shape), list(yt2.shape), list(dt.shape)]
    r["prod_private"] = [private(yt), private(yt2), private(dt)]
    r["prod_md5"] = [md5(y), md5(mean), md5(rstd), md5(yt), md5(yt2), md5(q), md5(dt), md5(amax)]
# empty_copy: out of memory propagates and does not latch the pool; empty_tm latches (unchanged)
real = TM._alloc_block
def boom(*a, **k):
    raise torch.OutOfMemoryError("HIP out of memory (injected)")
TM._alloc_block = boom
try:
    TM.empty_copy((512, 8, 128), E4, dev)
    r["copy_oom"] = "returned"
except torch.OutOfMemoryError:
    r["copy_oom"] = "raised"
r["broken_after_copy_oom"] = TM._broken
r["active_after_copy_oom"] = TM.active()
TM._alloc_block = real
print("RESULT " + json.dumps(r))
"""


class Fp8TmAllocTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
    def test_tiled_placement_knob_and_prewarm_skip(self) -> None:
        # r14 iter40 B11T: DLRM_HSTU_FP8_TM_NONEXP_TILED=0 moves only the step-tiled (3-D)
        # operands out of the private pool and skips the prewarm iff every pooled operand
        # is step-tiled; row-major operands and the untiled copies stay in the pool; the
        # producers' bytes do not change; "1" / unset = before this knob.
        ck4 = {
            "DLRM_HSTU_FP8_BWD_WGRAD_CK": "1",
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED": "1",
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3": "1",
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED": "1",
        }
        k0 = {"DLRM_HSTU_FP8_TM_NONEXP_TILED": "0"}
        # name: (extra env, tiled_pool, all_tiled, prewarm blocks or None (skipped), producers)
        cases = {
            "unset_ck4": (dict(ck4), True, False, 5, True),
            "k1_ck4": ({**ck4, "DLRM_HSTU_FP8_TM_NONEXP_TILED": "1"}, True, False, 5, False),
            "k0_ck4": ({**ck4, **k0}, False, True, None, True),
            "k0_w3_rowmajor": (
                {**ck4, **k0, "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED": "0"}, False, False, 5, False),
            "k0_xt_rowmajor": (
                {**ck4, **k0, "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED": "0"}, False, False, 5, False),
            "k0_xt0_dout_tiled": (
                {**k0, "DLRM_HSTU_FP8_TM_NONEXP_XT": "0", "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3": "1",
                 "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED": "1"}, False, True, None, False),
            "k0_ck4_blocks2": ({**ck4, **k0, "DLRM_HSTU_FP8_TM_NONEXP_BLOCKS": "2"}, False, True, 2, False),
        }
        md5s = {}
        for name, (extra, tiled_pool, all_tiled, blocks, producers) in cases.items():
            with self.subTest(case=name):
                env = dict(os.environ)
                for k in list(env):
                    if k.startswith("DLRM_HSTU_FP8_TM_NONEXP_") or k.startswith(
                        "DLRM_HSTU_FP8_BWD_WGRAD_CK"
                    ):
                        env.pop(k)
                env.pop("PYTORCH_ALLOC_CONF", None)
                env.update(
                    {
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "DLRM_HSTU_FP8_TM_NONEXP": "1",
                        "DLRM_HSTU_FP8_ALLOC_PREGROW": "memcpy=0.01,margin=100000",
                        "CHILD_PRODUCERS": "1" if producers else "0",
                        **extra,
                    }
                )
                out = subprocess.run(
                    [sys.executable, "-B", "-c", _CHILD_TILED],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                self.assertEqual(out.returncode, 0, out.stderr[-3000:])
                line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                self.assertTrue(line, out.stdout[-3000:])
                r = json.loads(line[-1][len("RESULT ") :])
                self.assertTrue(r["active"])
                self.assertEqual(r["tiled_pool"], tiled_pool)
                self.assertEqual(r["all_tiled"], all_tiled)
                self.assertIn(
                    "[hstu-fp8-tm] tiled operands: B11 pool"
                    if tiled_pool
                    else "[hstu-fp8-tm] tiled operands: ordinary allocator "
                    "(DLRM_HSTU_FP8_TM_NONEXP_TILED=0)",
                    out.stdout,
                )
                skipped = "[hstu-fp8-tm] prewarm skipped: every pooled operand is step-tiled"
                if blocks is None:
                    self.assertIn(skipped, r["pregrow_lines"])
                    self.assertEqual(r["prewarmed"], [])
                    self.assertIn("pool created lazily on cuda:0 (not prewarmed)", out.stdout)
                else:
                    self.assertNotIn(skipped, r["pregrow_lines"])
                    self.assertEqual(r["prewarmed"], [0])
                    self.assertTrue(
                        any(f"{blocks} x 2.00 GiB blocks mapped" in ln for ln in r["pregrow_lines"]),
                        r["pregrow_lines"],
                    )
                    self.assertNotIn("pool created lazily", out.stdout)
                # the first untiled copy: the pool iff it was prewarmed (not created for it)
                self.assertEqual(r["copy_first_private"], blocks is not None)
                # 3-D (tiled) requests: the pool iff the knob is 1; 2-D and copies: the pool
                self.assertEqual(r["tiled_tm_private"], tiled_pool)
                self.assertEqual(
                    r["tiled_xt_private"],
                    tiled_pool and extra.get("DLRM_HSTU_FP8_TM_NONEXP_XT") != "0",
                )
                self.assertTrue(r["rm_tm_private"])
                self.assertEqual(r["rm_xt_private"], extra.get("DLRM_HSTU_FP8_TM_NONEXP_XT") != "0")
                self.assertTrue(r["copy_private"])
                # an OOM in empty_copy propagates and does not latch the pool off
                self.assertEqual(r["copy_oom"], "raised")
                self.assertIsNone(r["broken_after_copy_oom"])
                self.assertTrue(r["active_after_copy_oom"])
                if producers:
                    kt = (1000 + 127) // 128
                    self.assertEqual(r["prod_shapes"], [[kt, 512, 128]] * 3)
                    self.assertEqual(r["prod_private"], [tiled_pool] * 3)
                    md5s[name] = r["prod_md5"]
        self.assertEqual(len({tuple(v) for v in md5s.values()}), 1, md5s)
        self.assertEqual(len(md5s), 2)

    def test_is_oom(self) -> None:
        from generative_recommenders.ops.triton import fp8_tm_alloc as TM

        self.assertTrue(TM.is_oom(torch.OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(TM.is_oom(RuntimeError("HIP out of memory. Tried to allocate 1.23 GiB")))
        self.assertFalse(TM.is_oom(RuntimeError("invalid argument")))
        self.assertFalse(TM.is_oom(ValueError("out of memory")))


    @unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
    def test_xt_placement_knob(self) -> None:
        # r14 iter38 review S3: DLRM_HSTU_FP8_TM_NONEXP_XT moves only the LN xT
        # (store_t / recompute_fp8t) out of the private pool; the default is the
        # private pool, also with DLRM_HSTU_FP8_BWD_WGRAD_CK=1 (r14 iter38 M sweep);
        # dout_t stays; the bytes do not change.
        cases = {
            "default": ({}, True),
            "ck": ({"DLRM_HSTU_FP8_BWD_WGRAD_CK": "1"}, True),
            "ck_xt0": ({"DLRM_HSTU_FP8_BWD_WGRAD_CK": "1", "DLRM_HSTU_FP8_TM_NONEXP_XT": "0"}, False),
            "ck_xt1": ({"DLRM_HSTU_FP8_BWD_WGRAD_CK": "1", "DLRM_HSTU_FP8_TM_NONEXP_XT": "1"}, True),
            "xt0": ({"DLRM_HSTU_FP8_TM_NONEXP_XT": "0"}, False),
        }
        hashes = {}
        for name, (extra, in_pool) in cases.items():
            with self.subTest(case=name):
                env = dict(os.environ)
                for k in ("DLRM_HSTU_FP8_BWD_WGRAD_CK", "DLRM_HSTU_FP8_TM_NONEXP_XT", "PYTORCH_ALLOC_CONF"):
                    env.pop(k, None)
                env.update(
                    {
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "DLRM_HSTU_FP8_TM_NONEXP": "1",
                        **extra,
                    }
                )
                out = subprocess.run(
                    [sys.executable, "-c", _CHILD_XT],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                self.assertEqual(out.returncode, 0, out.stderr[-3000:])
                line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                self.assertTrue(line, out.stdout[-3000:])
                r = json.loads(line[-1][len("RESULT ") :])
                self.assertTrue(r["active"])
                self.assertEqual(r["xt_in_pool"], in_pool)
                self.assertEqual(r["default_blocks"], 5 if in_pool else 2)
                self.assertEqual(r["xt_private"], in_pool)
                self.assertEqual(r["store_t_private"], in_pool)
                self.assertEqual(r["recompute_private"], in_pool)
                self.assertTrue(r["dout_t_private"])
                self.assertEqual(r["shape"], [512, 1024, 512, 1024])
                if not in_pool:
                    self.assertIn("LN xT from the main pool", out.stdout)
                hashes[name] = r["md5"]
        self.assertEqual(len({tuple(v) for v in hashes.values()}), 1, hashes)


    def test_inert_without_knob(self) -> None:
        from generative_recommenders.ops.triton import fp8_tm_alloc as TM

        if TM.requested():
            self.skipTest("DLRM_HSTU_FP8_TM_NONEXP set in this process")
        self.assertFalse(TM.active())
        self.assertEqual(TM.prewarm(2), 0)
        t = TM.empty_tm((4, 8), torch.float8_e4m3fn, torch.device("cpu"))
        self.assertEqual(list(t.shape), [4, 8])

    @unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
    def test_private_nonexpandable_pool(self) -> None:
        env = dict(os.environ)
        env.update(
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "DLRM_HSTU_FP8_TM_NONEXP": "1",
            }
        )
        env.pop("PYTORCH_ALLOC_CONF", None)
        out = subprocess.run(
            [sys.executable, "-c", _CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        self.assertTrue(line, out.stdout[-3000:])
        r = json.loads(line[-1][len("RESULT ") :])
        self.assertTrue(r["active"])
        self.assertEqual(r["prewarm_bytes"], 2 * (2 << 30))
        self.assertEqual(r["shape"], [512, 1000064])
        self.assertEqual(r["stride"], [1000064, 1])
        self.assertEqual(r["offset"], 0)
        self.assertTrue(r["contig"])
        self.assertEqual(r["dtype"], "torch.float8_e4m3fn")
        self.assertFalse(r["a_expandable"])
        self.assertTrue(r["a_pool_private"])
        self.assertEqual(r["a_block"], 2 << 30)
        self.assertTrue(r["a_ok"])
        self.assertFalse(r["outside_took_pool_block"])
        self.assertTrue(r["outside_expandable"])
        self.assertTrue(r["reused_whole_block"])
        self.assertTrue(r["zero_ok"])
        self.assertEqual(r["big_block"], 5 * (512 << 20))


if __name__ == "__main__":
    unittest.main()
