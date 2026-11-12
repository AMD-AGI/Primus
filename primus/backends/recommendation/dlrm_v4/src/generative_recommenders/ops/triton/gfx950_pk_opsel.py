"""gfx950 packed-FP32 op_sel hazard: launch option + ISA rule (ROCm 10.2 port, opsel-fix).

Triton 3.9's bundled LLVM (3.9.0+gitfc8d31a5.rocm10.2) can select a packed-FP32
VALU instruction whose LOW lane reads the HIGH dword of src1 (or src2) while src0
reads its natural dword:

    v_pk_add_f32 / v_pk_mul_f32 vD, vA, v[s:s+1] op_sel:[0,1]
    v_pk_fma_f32               vD, vA, v[s:s+1], vC op_sel:[0,1,0]   (or op_sel[2] set)

That encoding is invalid on gfx950 (MI355X): AMD LLVM 24 (ROCm 10.2's hipcc) rejects
it with "invalid op_sel operand", gfx942 / gfx90a accept it, and upstream LLVM fixed
the selection in llvm/llvm-project#232016 (reland of #231506) and #233970 (the fixed
compiler commutes the operands, giving the valid op_sel:[1,0]). On hardware the low
result can come out as a signed zero while an MFMA runs in the same wave
(_dq8_sched_kernel, hstu_dq_e5m2._no_packed_fp32_launch_opts).

no_packed_fp32_launch_opts() is the per-kernel remedy: the launch option
llvm_fn_attrs="target-features=-packed-fp32-ops" makes the kernel use the scalar
v_add/v_mul/v_fma_f32 (the same IEEE fp32 operations, so outputs are bit-identical;
the compiler still inserts every hazard wait state). It is {} on Triton builds whose
HIP backend has no llvm_fn_attrs option (3.6) and on other targets, so those launches
and binaries are unchanged. The option enters the kernel's L1 cache key (a call
kwarg), so TSPEC table rows of such a launch carry it in call_opts / key
(triton_spec_table.jsonl).

invalid_pk_f32_opsel(asm) is the exact rule (= AMD LLVM 24 llvm-mc on all 4,704
op_sel x op_sel_hi x neg x VGPR / SGPR / constant forms; independent of op_sel_hi,
neg_lo / neg_hi and the source kind). Tests use it on the compiled ISA.
"""

import re
from typing import Dict, List, Optional

NO_PACKED_FP32_FN_ATTRS = "target-features=-packed-fp32-ops"

_OPTS: Optional[Dict[str, str]] = None


def no_packed_fp32_launch_opts() -> Dict[str, str]:
    """{"llvm_fn_attrs": "target-features=-packed-fp32-ops"} for a gfx950 target on a
    Triton whose HIP backend takes LLVM function attributes (3.9), {} otherwise.
    Resolved once per process (current device's target)."""
    global _OPTS
    if _OPTS is None:
        opts: Dict[str, str] = {}
        try:
            import triton
            from triton.backends.amd.compiler import HIPOptions

            target = triton.runtime.driver.active.get_current_target()
            if (target.backend == "hip" and str(target.arch).startswith("gfx950")
                    and "llvm_fn_attrs" in getattr(HIPOptions, "__dataclass_fields__", {})):
                opts = {"llvm_fn_attrs": NO_PACKED_FP32_FN_ATTRS}
        except Exception:  # noqa: BLE001 (no HIP backend / driver: leave the launch as is)
            opts = {}
        _OPTS = opts
    return dict(_OPTS)


_PK = re.compile(r"^\s*v_pk_(add|mul|fma)_f32\b(.*)$")
_OPSEL = re.compile(r"\bop_sel:\[([01,]+)\]")


def invalid_pk_f32_opsel(asm: str) -> List[str]:
    """The gfx950-invalid packed-FP32 op_sel instructions of an AMDGPU assembly text
    (compiler .amdgcn / .s output or llvm-objdump disassembly): v_pk_add_f32 /
    v_pk_mul_f32 with op_sel == [0,1], v_pk_fma_f32 with op_sel[1] or op_sel[2] set.
    An absent op_sel prints as all zeros."""
    bad = []
    for line in asm.splitlines():
        code = line.split(";")[0].split("//")[0]
        m = _PK.match(code)
        if not m:
            continue
        sm = _OPSEL.search(m.group(2))
        sel = [int(x) for x in sm.group(1).split(",")] if sm else []
        n = 3 if m.group(1) == "fma" else 2
        sel = (sel + [0] * n)[:n]
        if (m.group(1) == "fma" and (sel[1] or sel[2])) or (m.group(1) != "fma" and sel == [0, 1]):
            bad.append(line.strip())
    return bad
