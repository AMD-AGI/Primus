"""gfx950 packed-FP32 op_sel guard for the FlyDSL attention backward (opsel-fix).

FlyDSL 0.2.4's bundled LLVM selects `v_pk_fma_f32 vD, vA, s[x:x+1], vC
op_sel:[0,1,0]` (the low lane reads the HIGH dword of src1: invalid on gfx950,
llvm/llvm-project#232016) when a splat of an SGPR pair's high half is the
SECOND multiplicand. hstu_bwd_dkdv's vf3p / vf4 constants (pk_cx2, pk_nn2,
pk_ncxn2) are therefore passed as the first multiplicand (see pk_cx2 there).

This test compiles bwd_dkdv_kernel_0 with the production compile kwargs of the
C8 FP8 dq-split backward (F2) and its DkdvE5m2Sink variant (F2e; recorded on
2026-11-03 from the production path by the flydsl lane's record_cfg.py) on the
CPU (COMPILE_ONLY=1, FLYDSL_DUMP_IR=1) and checks the final ISA for the
gfx950-invalid packed-FP32 op_sel forms. Runs in its own process (the FlyDSL
environment switches are read at import); skipped without FlyDSL >= 0.2.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from generative_recommenders.ops.triton.gfx950_pk_opsel import invalid_pk_f32_opsel

_KW_F2 = {
    "BN": 128, "do_blk": True, "do_dyn": False, "dq16": "bf16p", "dqsplit": True, "ds8_pad": 16,
    "epilogue": "reuse_pk", "f8_sat": True, "f8b": "all", "fp8_mfma": "k128", "fused": True,
    "input_fp8": True,
    "ms_sched": "mid,sb,wsl,sgv5,la,xt,rs2,lc,ek,nk,wh,mc,dzr,bst,ob0,obj,gsp,xtd,osn,osnb",
    "msplit": 1, "mu2": "k128", "mu2_dsc": 128.0, "mu2_ssc": 32768.0, "q_bf16": False,
    "rs_in": 1536, "rs_out": 2048, "rs_q": 1536, "sched": "iglp2",
    "valu": "ilp,msat,nobr,rfl,tsc,undef,vf3p,vf4", "ws_aux": 2, "ws_l128t": True,
}
_KW_F2E = dict(_KW_F2, e8emit="only")

# One CPU compile per config, in a child process (COMPILE_ONLY / FLYDSL_DUMP_IR are
# read when flydsl is imported). Inputs follow fly_harness_v4's compile mode.
_CHILD = textwrap.dedent(r'''
    import json, os, sys
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = sys.argv[2]
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
    os.environ["FLYDSL_COMPILE_ARCH"] = "gfx950"
    os.environ["DLRM_HSTU_FP8_BWD_DQ_WS_PERSIST"] = "0"
    import torch
    import flydsl.expr as fx
    from generative_recommenders.ops.flydsl import hstu_bwd_dkdv as kb
    from generative_recommenders.ops.flydsl import hstu_dq_split
    from generative_recommenders.ops.flydsl.hstu_dkdv_e5m2 import partial_rows_bound
    kw = json.loads(sys.argv[1])
    H, D, ALPHA, QS = 4, 128, 1.0 / (128 ** 0.5), 0.02187788113951683
    lengths = torch.tensor([300, 41, 77, 128, 129, 255, 256, 40], dtype=torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    T, B, N = int(offs[-1]), int(lengths.numel()), int(lengths.max())
    vqk = torch.zeros((T, 3 * H * D), dtype=torch.float8_e4m3fn)
    v, q, k = (x.view(T, H, D) for x in vqk.split(H * D, dim=1))
    dout = torch.zeros((T, H, D), dtype=torch.float8_e4m3fn)
    do_scale = torch.full((T, H), 127, dtype=torch.uint8)
    g = torch.zeros((T, 4 * H * D), dtype=torch.bfloat16)
    dv, dq, dk = (g[:, a:a + H * D].view(T, H, D) for a in (512, 1024, 1536))
    ws = hstu_dq_split.build_workspace(offs, H, T, N)
    nb = (N + kw["BN"] - 1) // kw["BN"]
    fn = kb.compile_hstu_bwd_dkdv(H, D, ALPHA, **kw)
    a = (q, k, v, dout, dk, dv, kb._dq_ws_2d(ws.view(torch.bfloat16)), offs.to(torch.int32), B, nb,
         fx.Float32(1.0 / N), fx.Float32(QS))
    kws = {"do_scale": do_scale}
    if kw.get("e8emit"):
        Mp = (T + 255) // 256 * 256
        rb = partial_rows_bound(T, B)
        pfx = torch.zeros(B + 1, dtype=torch.int32)
        pfx[1:] = ((lengths + 127) // 128).cumsum(0).to(torch.int32)
        kws.update(dz8=torch.zeros((Mp, 4 * H * D), dtype=torch.uint8),
                   e8inv=torch.full((1,), 64.0, dtype=torch.float32), e8pfx=pfx,
                   e8pam=torch.zeros((rb, H * 32), dtype=torch.float32),
                   e8pcs=torch.zeros((rb, 2 * H * D), dtype=torch.float32))
    fn.launch(*a, **kws)
    print("compiled", flush=True)
''')


def _flydsl_02() -> bool:
    try:
        import flydsl  # noqa: F401
        from generative_recommenders.ops.flydsl import hstu_bwd_dkdv as kb
        return hasattr(kb, "_dq_ws_2d") and hasattr(kb, "compile_hstu_bwd_dkdv")
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_flydsl_02(), "needs FlyDSL 0.2.x and the dq-split bwd kernel")
class FlyDSLBwdOpselTest(unittest.TestCase):
    def _check(self, kw) -> None:
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, HIP_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(kw), d], env=env,
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-4000:])
            isa = glob.glob(os.path.join(d, "**", "*_final_isa.s"), recursive=True)
            self.assertTrue(isa, "no final ISA dumped")
            for p in isa:
                with open(p) as f:
                    asm = f.read()
                self.assertRegex(asm, r"\bv_pk_fma_f32\b")  # the packed path is still used
                self.assertEqual(invalid_pk_f32_opsel(asm), [], p)

    def test_f2(self) -> None:
        self._check(_KW_F2)

    def test_f2e(self) -> None:
        self._check(_KW_F2E)


if __name__ == "__main__":
    unittest.main()
