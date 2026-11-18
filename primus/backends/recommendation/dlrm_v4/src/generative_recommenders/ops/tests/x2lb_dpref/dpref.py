# pyre-unsafe
"""Test-only data-parallel hipBLASLt reference for ops/tests/preproc_l2_x2lb_test.py (DESIGN_stageC 5.1 and 8.2 G-a).

dp32 is hipBLASLt's top-1 solution for torch's _scaled_mm problem, run with a 32 MiB workspace. That workspace forces
the data-parallel grid [S labkit PRED_x2lb.md section 2 "References"; review_measure 2.1]. The output goes into a
NaN-sentinel tensor.

This is a port of the lab kit's x2lb_ab.py run_into / dp_refs / contract_kernel / replica search (find_replica) over
the VERBATIM copies e0_hblt.py and e0_hblt_ext.cpp next to this file (sha256 f3c8f6ff... / 75699111..., labkit
SHA256SUMS.x2lb). It is imported only by the GPU test, never by model code, and loaded by path (ops/tests is not a
package).
"""

import importlib.util
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
E0_SHA256 = {"e0_hblt.py": "f3c8f6ff3700fe989eabb40534cfdad6f41f9174efcf2fb6828bb7b333f63bfa",
             "e0_hblt_ext.cpp": "756991113b7d5cc24b01c6f9da442601e458e866c0f9f7aff1a5a966c7179437"}
MiB = 1 << 20
DP_WS = 32 * MiB  # x2lb_ab.DP_WS
TORCH_WS = 76 * MiB  # torch's process-wide hipBLASLt workspace on ROCm (e0_hblt.TORCH_WS)
REPLICA_WS = (76, 64, 128, 256)  # x2lb_ab.REPLICA_WS (MiB)
# CONTRACT_dgrad section 1 (x2lb_ab.CONTRACT_TOKENS): A = MT256x256x128 MIWT8_8, B = MT256x240x128 MIWT4_15
CONTRACT_TOKENS = {"256x256x128": ("_MI16x16x1_", "_SK3_", "_MIWT8_8_"),
                   "256x240x128": ("_MI16x16x1_", "_SK3_", "_MIWT4_15_")}
MT_RE = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")
SENT16 = 0x7FC1  # x2lb_ab.run_into's NaN sentinel (bf16 bits)


def tile_of(name: Optional[str]) -> Optional[str]:
    m = MT_RE.search(name or "")
    return f"{m.group(1)}x{m.group(2)}x{m.group(3)}" if m else None


def contract_kernel(name: Optional[str]) -> bool:
    """x2lb_ab.contract_kernel: CONTRACT_dgrad's solution A or B (any other F8B8BS variant is not covered)."""
    t = tile_of(name)
    return bool(name and "_F8B8BS_" in name and t in CONTRACT_TOKENS and all(x in name for x in CONTRACT_TOKENS[t]))


def _load_e0() -> Any:
    import hashlib

    for n, want in E0_SHA256.items():
        with open(os.path.join(HERE, n), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            raise RuntimeError(f"x2lb_dpref/{n} is not the sealed E0 copy ({got[:12]} != {want[:12]})")
    name = "x2lb_dpref_e0_hblt"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "e0_hblt.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kernel_name(torch: Any, fn: Any) -> Optional[str]:
    """The Tensile kernel one call of fn launches (kineto), or None."""
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = [e.key for e in p.key_averages() if e.key.startswith("Cijk_")]
    return names[0] if names else None


class DpRef:
    """dp32 / replica checks on the libhipblaslt torch loaded (e0_hblt.HipBLASLt, production's library)."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.e0 = _load_e0()
        self.hb = self.e0.HipBLASLt(torch)
        self.c_type = self.e0.R_16BF

    def _problem(self, dh8: Any, w8t: Any, sa: Any, sb: Any, c_type: Optional[int] = None) -> Any:
        return self.hb.smm(dh8, w8t.t(), sa, sb, c_type=self.c_type if c_type is None else c_type)

    def run_into(self, prob: Any, rec: Dict[str, Any], ws: int) -> Any:
        """x2lb_ab.run_into: one hipblasLtMatmul with `rec` into a fresh NaN-sentinel output."""
        torch = self.torch
        o = torch.empty_like(prob.D)
        o.view(torch.int16).fill_(SENT16)
        keep = prob.D
        prob.D = o
        try:
            st = self.hb.run(prob, rec, ws)
            torch.cuda.synchronize()
            if st != 0:
                raise RuntimeError(f"hipblasLtMatmul status {st}")
        finally:
            prob.D = keep
        return o

    def find_replica(self, dh8: Any, w8t: Any, sa: Any, sb: Any, out_torch: Any) -> Dict[str, Any]:
        """x2lb_ab.find_replica (bytes only): the C type / workspace whose top-1 reproduces torch's output bitwise at
        this M. Sets self.c_type to it (default bf16 when none does)."""
        torch = self.torch
        tried = []
        for c_type in (self.e0.R_16BF, self.e0.R_16F):
            for ws_mib in REPLICA_WS:
                prob = self._problem(dh8, w8t, sa, sb, c_type)
                try:
                    st, recs, _ = self.hb.heuristic(prob, ws_mib * MiB, 1)
                    ok = False
                    if recs:
                        o = self.run_into(prob, recs[0], ws_mib * MiB)
                        ok = bool((o.view(torch.int16) == out_torch.view(torch.int16)).all().item())
                        del o
                    tried.append(dict(c_type=c_type, ws_mib=ws_mib, status=st, bitwise=ok))
                finally:
                    prob.close()
                if ok:
                    self.c_type = c_type
                    return dict(ok=True, c_type=c_type, ws_mib=ws_mib, tried=tried)
        return dict(ok=False, tried=tried)

    def dp32(self, dh8: Any, w8t: Any, sa: Any, sb: Any) -> Tuple[Optional[Any], Dict[str, Any]]:
        """(dp32 da1 bf16 [M, 256] or None, info): the replica's top-1 with a 32 MiB workspace (x2lb_ab.dp_refs's
        dp32), its kernel name and whether that is a contract solution."""
        prob = self._problem(dh8, w8t, sa, sb)
        try:
            st, recs, _ = self.hb.heuristic(prob, DP_WS, 1)
            if st != 0 or not recs:
                return None, dict(status=st, kernel=None, contract=False)
            rec = recs[0]
            o = self.run_into(prob, rec, DP_WS)
            kn = kernel_name(self.torch, lambda: self.hb.run(prob, rec, DP_WS))
            return o, dict(status=st, index=rec.get("index"), kernel=kn, contract=contract_kernel(kn))
        finally:
            prob.close()
