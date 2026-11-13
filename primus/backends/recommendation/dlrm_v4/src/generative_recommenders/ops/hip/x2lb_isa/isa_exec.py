#!/usr/bin/env python3
"""isa_exec.py -- concrete multi-wave executor of a compiled x2lb_pass instance (CPU, numpy) and the ISA clauses C16 / C17
of isa_check.py (fix pass, 2026-10-12: race review F1, contract review F3).

The executor runs the kernel's OWN instruction stream (the .s text isa_check parses) for one workgroup: 8 waves of 64
lanes, v0 = threadIdx.x, s2 = blockIdx.x, s[0:1] = a kernarg segment holding fake, far-apart tensor bases plus M,
ntiles and the hidden block count G. It follows the real control flow (SCC / EXEC branches, EXEC-masked VALU writes,
s_and_saveexec), steps the 8 waves from barrier to barrier, models the LDS byte by byte, and tags every byte a load
brings into a VGPR or an LDS-DMA brings into LDS with its global byte address (zero for an out-of-range lane, which
reads 0). Integer / address arithmetic is evaluated exactly (32-bit lanes); floating-point values are not modelled
(the dataflow clause C15 owns the arithmetic). LDS-DMA lanes out of range of their resource write NOTHING (the
adversarial reading of FACTS (f)), so only the kernel's own zero fill can make the padded rows zero.

  C16 (zero fill, every compute instance): for every A tile the kernel DMAs (prologue and loop), with rows = its valid
      rows: if 0 < rows < 64, the wave issues, before its next barrier, exactly 4 ds_write_b128 into the A region whose
      active lanes are EXACTLY the lanes whose DMA voffset row is >= rows (= zfill_lane), whose data are zero, and whose
      address + immediate equals that lane's own DMA LDS destination M0 + 16*lane of DMA i (i = imm / 8192); otherwise
      no A-region ds_write with an active lane. Semantic (not a text pattern), so it holds for any guard encoding.
  C17 (concrete addressing = the index functions of x2lb_pass.cu section 1, mirrored by emu/x2lb_emu.py and drift-
      checked by idx_harness.cpp): per wave w and lane l
        a. LDS-DMA i of a tile: voffset = a_dma_voff(w, l), soffset = a_dma_soff(i), LDS dest = a_dma_lds(buf, w, i) +
           16 l (buf = 0 in the prologue, (j + 1) % 2 in iteration j), resource = dh8 + 32768 * rsrc_tile(T, rows),
           num_records = 512 * rows;
        b. every MFMA (chain link k, column block n, row group m from C3): src1 lane bytes = dh8[64T + 16m + l%16]
           [128k + 16(l/16) + {0..15, 64..79}] or 0 for rows >= M, src0 lane bytes = w8_t[w_col(w, n, l)][same K]
           (byte provenance through DMA, zero fill, buffer alternation and fragment reads); the fragment-read addresses
           are frag_lp0 / frag_lp1(l) + frag_off(buf, k, m) for the lo / hi quad (F3: a lo / hi base swap fails here);
           the W load addresses are w_voff(w, l) + w_imm(n, k, h) from resource w8t, num_records 131072;
        c. R writes = rw_base(w, l) + m * 16 * 528, m = 0..3 once per iteration; R reads = rr_base(w, l) + u * 16 * 528;
        d. h1 (and S0 da1) loads: voffset = c_voff(w, u, l), the prefetch issued after the u-th dh1 store is slot u's,
           the prologue load into the same registers is slot u's too; resource h1 + 32768 * rsrc_tile(T, rows),
           num_records 512 * rows; dh1 stores (u-th of the iteration): c_voff(w, u, l), resource dh1 + 32768 T;
           EMIT da1 stores: e_voff(w, m, l); mean / rstd loads: voffset 4 l, resource + 256 * rsrc_tile(T, rows),
           num_records 4 * rows; nw / nb loads: 4 * c_col0(l) (+16);
        e. the 16 v_readlane lane indices per iteration = {16u + 2w + h}; the consumer's half select (v_cndmask on vcc)
           sees vcc = lanes 32..63 (hi_half).
Scenarios (M, G, blocks) cover: a partial tile reached in the loop (rows 37, 63) and in the prologue (rows 20, 1), full
tiles only, and the one-past-the-end prefetch tile (rows 0) of every WG.
"""
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "emu"))
import x2lb_emu as IDX  # noqa: E402  (the index-function mirror, drift-checked against the .cu by idx_harness.cpp)

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
UNK, UNWR, ZERO, CONST = -9, -1, 0, -2  # byte tags; positive = global byte address
NL = 64

# fake tensor bases (bits 47:0 of a buffer base), one per Params pointer, plus the kernarg segment
BASES = dict(dh8=0x1000_0000_0000, w8t=0x1100_0000_0000, sa=0x1200_0000_0000, sb=0x1300_0000_0000,
             h1=0x1400_0000_0000, mean1=0x1500_0000_0000, rstd1=0x1600_0000_0000, nw=0x1700_0000_0000,
             nb=0x1800_0000_0000, dh1=0x1900_0000_0000, da1=0x1A00_0000_0000, part=0x1B00_0000_0000)
PARAM_ORDER = ("dh8", "w8t", "sa", "sb", "h1", "mean1", "rstd1", "nw", "nb", "dh1", "da1", "part")
KARG = 0x1F00_0000_0000
INLINE_F = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000, "2.0": 0x40000000,
            "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000}
REG = re.compile(r"^([vas])\[(\d+):(\d+)\]$|^([vas])(\d+)$")


class ExecError(Exception):
    pass


def tensor_of(addr):
    for k, b in BASES.items():
        if b <= addr < b + (1 << 40):
            return k, addr - b
    return None, None


def preg(tok):
    m = REG.match(tok.strip())
    if not m:
        return None
    if m.group(1):
        return m.group(1), int(m.group(2)), int(m.group(3))
    return m.group(4), int(m.group(5)), int(m.group(5))


def lit(tok):
    t = tok.strip()
    if t in INLINE_F:
        return INLINE_F[t]
    try:
        return int(t, 0) & M32
    except ValueError:
        return None


def sx32(v):
    return v - (1 << 32) if v & 0x80000000 else v


class Vec:
    """one VGPR: 64 lane values (uint64 holding 32 bits), per-lane known flags and per-byte tags"""
    __slots__ = ("val", "kn", "tag")

    def __init__(self, val=None, kn=None, tag=None):
        self.val = np.zeros(NL, np.uint64) if val is None else val
        self.kn = np.zeros(NL, bool) if kn is None else kn
        self.tag = np.full((NL, 4), UNK, np.int64) if tag is None else tag


def const_tags(val, kn):
    t = np.full((NL, 4), UNK, np.int64)
    for b in range(4):
        byte = (val >> np.uint64(8 * b)) & np.uint64(0xFF)
        t[:, b] = np.where(kn, np.where(byte == 0, ZERO, CONST), UNK)
    return t


class Wave:
    def __init__(self, ex, w):
        self.ex, self.w = ex, w
        self.s = {}
        self.v = {}
        self.exec = M64
        self.vcc = 0
        self.scc = None
        self.m0 = None
        self.it = -1          # -1 prologue, j >= 0 loop iteration, -2 after the loop
        self.events = []
        self.vwriter = {}     # vgpr -> index of the load event that last wrote it (or None)

    # ---------------------------------------------------------------- operands
    def active(self):
        return np.array([(self.exec >> l) & 1 for l in range(NL)], bool)

    def sget(self, tok):
        t = tok.strip()
        if t in ("vcc", "vcc_lo"):
            return self.vcc & M32 if t == "vcc_lo" else None  # 64-bit vcc is read through s64get
        if t == "vcc_hi":
            return (self.vcc >> 32) & M32
        if t == "exec_lo":
            return self.exec & M32
        if t == "exec_hi":
            return (self.exec >> 32) & M32
        if t == "m0":
            return self.m0
        if t == "scc":
            return self.scc
        r = preg(t)
        if r and r[0] == "s":
            return self.s.get(r[1])
        return lit(t)

    def s64get(self, tok):
        t = tok.strip()
        if t == "vcc":
            return self.vcc
        if t == "exec":
            return self.exec
        r = preg(t)
        if r and r[0] == "s":
            lo, hi = self.s.get(r[1]), self.s.get(r[1] + 1)
            return None if lo is None or hi is None else lo | (hi << 32)
        v = lit(t)
        if v is None:
            return None
        return v | (M32 << 32) if v & 0x80000000 and t.startswith("-") else v

    def s64set(self, tok, val):
        t = tok.strip()
        if t == "vcc":
            self.vcc = val
            return
        if t == "exec":
            if val is None:
                raise ExecError("EXEC became unknown")
            self.exec = val
            return
        r = preg(t)
        self.s[r[1]] = None if val is None else val & M32
        self.s[r[1] + 1] = None if val is None else (val >> 32) & M32

    def sset(self, tok, val):
        t = tok.strip()
        if t == "m0":
            self.m0 = val
            return
        if t in ("vcc_lo", "vcc_hi", "exec_lo", "exec_hi"):
            raise ExecError("write of " + t + " not modelled")
        r = preg(t)
        if r is None or r[0] != "s":
            raise ExecError("SALU destination " + t)
        for i in range(r[1], r[2] + 1):
            self.s[i] = None if val is None else (val >> (32 * (i - r[1]))) & M32

    def vsrc(self, tok):
        """VALU source as a Vec (broadcast for scalars / constants)"""
        t = tok.strip()
        neg = False
        r = preg(t)
        if r and r[0] in "va":
            return self.v.get(r[1]) or Vec()
        val = self.sget(t)
        if val is None and t not in ("vcc", "exec") and not (r and r[0] == "s"):
            val = lit(t)
        if val is None:
            return Vec()
        vv = np.full(NL, val & M32, np.uint64)
        kn = np.ones(NL, bool)
        return Vec(vv, kn, const_tags(vv, kn))

    def vset(self, idx, vec, act=None):
        act = self.active() if act is None else act
        old = self.v.get(idx) or Vec()
        self.v[idx] = Vec(np.where(act, vec.val, old.val), np.where(act, vec.kn, old.kn),
                          np.where(act[:, None], vec.tag, old.tag))
        self.vwriter[idx] = None

    # ---------------------------------------------------------------- execution
    def run(self):
        ex = self.ex
        pc, steps = 0, 0
        insts = ex.insts
        while True:
            steps += 1
            if steps > 400000:
                raise ExecError("step limit")
            if pc >= len(insts):
                raise ExecError("ran off the end")
            x = insts[pc]
            if x.idx == ex.hdr_idx:
                self.it += 1
            elif self.it >= 0 and x.blk not in ex.loop_blocks and self.it != -2:
                self.it = -2
            op = x.op
            nxt = pc + 1
            if op == "s_endpgm":
                return
            if op == "s_barrier":
                self.events.append(dict(kind="barrier", idx=x.idx, it=self.it))
                yield pc
                pc = nxt
                continue
            if op.startswith(("s_waitcnt", "s_nop", "s_setprio", "s_sleep", "s_dcache", "s_sendmsg")):
                pc = nxt
                continue
            if op == "s_branch":
                pc = ex.labels[x.ops[0]]
                continue
            if op.startswith("s_cbranch"):
                cond = op[len("s_cbranch_"):]
                if cond in ("scc0", "scc1"):
                    if self.scc is None:
                        raise ExecError(f"branch on unknown SCC at line {x.line + 1}")
                    take = self.scc == (1 if cond == "scc1" else 0)
                elif cond in ("execz", "execnz"):
                    take = (self.exec == 0) == (cond == "execz")
                elif cond in ("vccz", "vccnz"):
                    if self.vcc is None:
                        raise ExecError(f"branch on unknown VCC at line {x.line + 1}")
                    take = ((self.vcc & self.exec) == 0) == (cond == "vccz")
                else:
                    raise ExecError("branch " + op)
                pc = ex.labels[x.ops[0]] if take else nxt
                continue
            try:
                if op.startswith("s_"):
                    self.salu(x)
                elif op.startswith(("buffer_", "global_", "ds_", "flat_", "scratch_")):
                    self.mem(x)
                elif op.startswith("v_"):
                    self.valu(x)
                else:
                    raise ExecError("unknown instruction")
            except ExecError as e:
                raise ExecError(f"line {x.line + 1} `{x.text}`: {e}") from None
            pc = nxt

    # ---------------------------------------------------------------- SALU
    def salu(self, x):
        op, o = x.op, x.ops
        g = self.sget
        if op.startswith("s_load_dword"):
            n = {"s_load_dword": 1, "s_load_dwordx2": 2, "s_load_dwordx4": 4, "s_load_dwordx8": 8,
                 "s_load_dwordx16": 16}[op]
            base = self.s64get(o[1])
            off = lit(o[2]) if len(o) > 2 else 0
            r = preg(o[0])
            for i in range(n):
                self.s[r[1] + i] = None if base is None else self.ex.kmem.get(base + off + 4 * i)
            return
        if op in ("s_mov_b32", "s_movk_i32"):
            v = g(o[1])
            if op == "s_movk_i32" and v is not None:
                v = (sx32(v << 16 & M32) >> 16) & M32 if v < 0x10000 else v
            self.sset(o[0], v)
            return
        if op == "s_mov_b64":
            self.s64set(o[0], self.s64get(o[1]))
            return
        if op == "s_cselect_b32":
            if self.scc is None:
                raise ExecError("s_cselect on unknown SCC")
            self.sset(o[0], g(o[1]) if self.scc else g(o[2]))
            return
        if op == "s_cselect_b64":
            if self.scc is None:
                raise ExecError("s_cselect on unknown SCC")
            self.s64set(o[0], self.s64get(o[1]) if self.scc else self.s64get(o[2]))
            return
        if op.startswith("s_cmp_"):
            a, b = g(o[0]), g(o[1])
            if a is None or b is None:
                self.scc = None
                return
            kind, ty = op[6:].split("_")
            if ty == "i32":
                a, b = sx32(a), sx32(b)
            self.scc = int({"eq": a == b, "lg": a != b, "gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[kind])
            return
        if op.endswith("_saveexec_b64"):
            src = self.s64get(o[1])
            if src is None:
                raise ExecError("saveexec of an unknown mask")
            old = self.exec
            self.s64set(o[0], old)
            new = {"s_and_saveexec_b64": old & src, "s_or_saveexec_b64": old | src,
                   "s_andn2_saveexec_b64": src & ~old & M64, "s_xor_saveexec_b64": old ^ src}.get(op)
            if new is None:
                raise ExecError(op)
            self.exec = new
            self.scc = int(new != 0)
            return
        if op.endswith("_b64") and op[:-4] in ("s_and", "s_or", "s_xor", "s_andn2", "s_orn2", "s_nand", "s_nor"):
            a, b = self.s64get(o[1]), self.s64get(o[2])
            if a is None or b is None:
                r = None
            else:
                r = {"s_and": a & b, "s_or": a | b, "s_xor": a ^ b, "s_andn2": a & ~b & M64,
                     "s_orn2": (a | ~b) & M64, "s_nand": ~(a & b) & M64, "s_nor": ~(a | b) & M64}[op[:-4]]
            self.s64set(o[0], r)
            self.scc = None if r is None else int(r != 0)
            return
        if op in ("s_lshl_b64", "s_lshr_b64", "s_ashr_i64"):
            a, b = self.s64get(o[1]), g(o[2])
            if a is None or b is None:
                r = None
            else:
                sh = b & 63
                if op == "s_lshl_b64":
                    r = (a << sh) & M64
                elif op == "s_lshr_b64":
                    r = a >> sh
                else:
                    r = ((a - (1 << 64) if a >> 63 else a) >> sh) & M64
            self.s64set(o[0], r)
            self.scc = None if r is None else int(r != 0)
            return
        if op == "s_not_b32":
            a = g(o[1])
            self.sset(o[0], None if a is None else ~a & M32)
            return
        if op in ("s_addc_u32", "s_subb_u32"):
            a, b = g(o[1]), g(o[2])
            if a is None or b is None or self.scc is None:
                self.sset(o[0], None)
                self.scc = None
                return
            if op == "s_addc_u32":
                t = a + b + self.scc
                self.sset(o[0], t & M32)
                self.scc = int(t > M32)
            else:
                t = a - b - self.scc
                self.sset(o[0], t & M32)
                self.scc = int(t < 0)
            return
        a, b = g(o[1]) if len(o) > 1 else None, g(o[2]) if len(o) > 2 else None
        if a is None or b is None:
            self.sset(o[0], None)
            self.scc = None
            return
        if op == "s_add_u32":
            t = a + b
            r, c = t & M32, int(t > M32)
        elif op == "s_sub_u32":
            r, c = (a - b) & M32, int(a < b)
        elif op == "s_add_i32":
            t = sx32(a) + sx32(b)
            r, c = t & M32, int(not (-(1 << 31) <= t < (1 << 31)))
        elif op == "s_sub_i32":
            t = sx32(a) - sx32(b)
            r, c = t & M32, int(not (-(1 << 31) <= t < (1 << 31)))
        elif op == "s_mul_i32":
            r, c = (sx32(a) * sx32(b)) & M32, self.scc
        elif op == "s_lshl_b32":
            r = (a << (b & 31)) & M32
            c = int(r != 0)
        elif op == "s_lshr_b32":
            r = a >> (b & 31)
            c = int(r != 0)
        elif op == "s_ashr_i32":
            r = (sx32(a) >> (b & 31)) & M32
            c = int(r != 0)
        elif op in ("s_and_b32", "s_or_b32", "s_xor_b32", "s_andn2_b32", "s_orn2_b32"):
            r = {"s_and_b32": a & b, "s_or_b32": a | b, "s_xor_b32": a ^ b, "s_andn2_b32": a & ~b & M32,
                 "s_orn2_b32": (a | ~b) & M32}[op]
            c = int(r != 0)
        elif op in ("s_min_i32", "s_max_i32", "s_min_u32", "s_max_u32"):
            aa, bb = (sx32(a), sx32(b)) if op.endswith("i32") else (a, b)
            pick_a = aa < bb if "min" in op else aa > bb
            r, c = (a if pick_a else b), int(pick_a)
        elif op.startswith("s_lshl") and op.endswith("_add_u32"):
            k = int(op[6])
            t = (a << k) + b
            r, c = t & M32, int(t > M32)
        elif op == "s_bfe_u32":
            off, wd = b & 31, (b >> 16) & 0x7F
            r = (a >> off) & ((1 << wd) - 1) if wd else 0
            c = int(r != 0)
        else:
            raise ExecError("SALU op not modelled: " + op)
        self.sset(o[0], r)
        self.scc = c

    # ---------------------------------------------------------------- VALU
    def valu(self, x):
        op, o = x.op, x.ops
        base = re.sub(r"_e(32|64)$", "", op)
        act = self.active()
        if op.startswith("v_readfirstlane_b32"):
            s = self.vsrc(o[1])
            ls = [l for l in range(NL) if act[l]]
            l0 = ls[0] if ls else 0
            self.sset(o[0], int(s.val[l0]) if s.kn[l0] else None)
            return
        if op.startswith("v_readlane_b32"):
            s = self.vsrc(o[1])
            lane = self.sget(o[2])
            if lane is None:
                self.sset(o[0], None)
                self.events.append(dict(kind="readlane", idx=x.idx, it=self.it, src=preg(o[1])[1], lane=None))
                return
            lane &= 63
            self.sset(o[0], int(s.val[lane]) if s.kn[lane] else None)
            self.events.append(dict(kind="readlane", idx=x.idx, it=self.it, src=preg(o[1])[1], lane=lane,
                                    tag=s.tag[lane].copy()))
            return
        if base.startswith("v_cmp_"):
            m = re.match(r"v_cmp_(\w+?)_([iuf])(16|32|64)$", base)
            dst = "vcc" if op.endswith("_e32") else o[0]
            srcs = o[0:2] if op.endswith("_e32") else o[1:3]
            if op.endswith("_e32"):
                dst, srcs = "vcc", (o[1:3] if o[0] == "vcc" else o[0:2])
            if not m or m.group(2) == "f":
                self.s64set(dst, None)
                return
            a, b = self.vsrc(srcs[0]), self.vsrc(srcs[1])
            if not (a.kn[act].all() and b.kn[act].all()):
                self.s64set(dst, None)
                return
            av, bv = a.val.astype(np.int64), b.val.astype(np.int64)
            if m.group(2) == "i":
                av = np.where(av >= 1 << 31, av - (1 << 32), av)
                bv = np.where(bv >= 1 << 31, bv - (1 << 32), bv)
            res = {"lt": av < bv, "le": av <= bv, "gt": av > bv, "ge": av >= bv, "eq": av == bv,
                   "ne": av != bv, "lg": av != bv}[m.group(1)]
            mask = 0
            for l in range(NL):
                if act[l] and res[l]:
                    mask |= 1 << l
            self.s64set(dst, mask)
            return
        if base == "v_cndmask_b32":
            cond = self.s64get(o[3]) if len(o) > 3 else self.vcc
            self.events.append(dict(kind="cndmask", idx=x.idx, it=self.it, cond=cond, cond_tok=o[3] if len(o) > 3
                                    else "vcc"))
            a, b = self.vsrc(o[1]), self.vsrc(o[2])
            if cond is None:
                self.vset(preg(o[0])[1], Vec(), act)
                return
            c = np.array([(cond >> l) & 1 for l in range(NL)], bool)
            self.vset(preg(o[0])[1], Vec(np.where(c, b.val, a.val), np.where(c, b.kn, a.kn),
                                         np.where(c[:, None], b.tag, a.tag)), act)
            return
        d = preg(o[0]) if o else None
        if d is None or d[0] == "s":
            if d is not None:  # VALU writing an SGPR (v_cmp handled above): unknown
                for i in range(d[1], d[2] + 1):
                    self.s[i] = None
            return
        if base in ("v_mov_b32",) and "_dpp" not in op and "_sdwa" not in op:
            s = self.vsrc(o[1])
            self.vset(d[1], Vec(s.val.copy(), s.kn.copy(), s.tag.copy()), act)
            return
        if base == "v_mov_b64":
            r = preg(o[1])
            for i in range(2):
                if r and r[0] in "va":
                    s = self.v.get(r[1] + i) or Vec()
                elif r and r[0] == "s":
                    s = self.vsrc("s%d" % (r[1] + i))
                else:
                    v64 = self.s64get(o[1])
                    s = self.vsrc(str((v64 >> (32 * i)) & M32)) if v64 is not None else Vec()
                self.vset(d[1] + i, Vec(s.val.copy(), s.kn.copy(), s.tag.copy()), act)
            return
        if base.startswith("v_mfma"):
            self.events.append(dict(kind="mfma", idx=x.idx, it=self.it, src0=self.octet_tags(o[1]),
                                    src1=self.octet_tags(o[2]), src0_reg=preg(o[1])[1], src1_reg=preg(o[2])[1]))
            for i in range(d[1], d[2] + 1):
                self.vset(i, Vec(), act)
            return
        fn = INT_OPS.get(base)
        if fn is None or "_dpp" in op or "_sdwa" in op:
            for i in range(d[1], d[2] + 1):  # floating point, DPP, SDWA, 64-bit, ...: value not modelled
                self.vset(i, Vec(), act)
            return
        nsrc, f = fn
        srcs = [self.vsrc(t) for t in o[1:1 + nsrc]]
        kn = np.logical_and.reduce([s.kn for s in srcs])
        vals = [s.val.astype(np.uint64) for s in srcs]
        res = f(x, *vals) & np.uint64(M32)
        res = np.where(kn, res, np.uint64(0))
        self.vset(d[1], Vec(res, kn, const_tags(res, kn)), act)

    def octet_tags(self, tok):
        r = preg(tok)
        out = np.full((NL, 32), UNK, np.int64)
        for q in range(8):
            v = self.v.get(r[1] + q) or Vec()
            out[:, 4 * q:4 * q + 4] = v.tag
        return out

    # ---------------------------------------------------------------- memory
    def rsrc(self, tok):
        r = preg(tok)
        w = [self.s.get(r[1] + i) for i in range(4)]
        if any(v is None for v in w[:3]):
            return None
        return dict(base=w[0] | ((w[1] & 0xFFFF) << 32), stride=(w[1] >> 16) & 0x3FFF, nrec=w[2], w3=w[3])

    def mem(self, x):
        op, o, ex = x.op, x.ops, self.ex
        act = self.active()
        imm = 0
        for md in x.mods:
            if re.match(r"^offset:(0x[0-9a-fA-F]+|\d+)$", md):
                imm = int(md[7:], 0)
        if op.startswith("buffer_"):
            lds = "lds" in x.mods
            ndw = {"dword": 1, "dwordx2": 2, "dwordx3": 3, "dwordx4": 4}[op.split("_", 2)[2]]
            store = op.startswith("buffer_store")
            # operands: [vdata,] vaddr, srsrc, soffset (an LDS-DMA load has no vdata)
            oo = o if lds else o[1:]
            vofs = self.vsrc(oo[0]) if "offen" in x.mods else Vec(np.zeros(NL, np.uint64), np.ones(NL, bool))
            soff = self.sget(oo[2])
            rs = self.rsrc(oo[1])
            if rs is None or soff is None or not vofs.kn[act].all():
                raise ExecError("buffer address not known")
            if rs["stride"] != 0:
                raise ExecError("structured buffer resource")
            vo = vofs.val.astype(np.int64)
            addr = rs["base"] + soff + vo + imm
            inr = np.stack([(vo + imm + 4 * d) < rs["nrec"] for d in range(ndw)], 1)
            ev = dict(kind="dma" if lds else ("bstore" if store else "bload"), idx=x.idx, it=self.it, rsrc=rs,
                      soff=soff, voff=vo.copy(), imm=imm, act=act.copy(), inr=inr, ndw=ndw, addr=addr.copy())
            if lds:
                if self.m0 is None:
                    raise ExecError("M0 unknown at an LDS-DMA")
                ldsa = self.m0 + imm + 16 * np.arange(NL)
                ev["lds"] = ldsa
                for l in range(NL):
                    if not act[l]:
                        continue
                    for d in range(ndw):
                        if inr[l, d]:  # out-of-range lanes write NOTHING (adversarial reading of FACTS f)
                            a0 = int(ldsa[l]) + 4 * d
                            ex.lds[a0:a0 + 4] = int(addr[l]) + 4 * d + np.arange(4)
            elif store:
                ev["data"] = preg(o[0])
            else:
                r = preg(o[0])
                ev["dst"] = r[1]
                for d in range(ndw):
                    tg = np.where(inr[:, d][:, None], (addr + 4 * d)[:, None] + np.arange(4)[None, :], ZERO)
                    self.vset(r[1] + d, Vec(np.zeros(NL, np.uint64), np.zeros(NL, bool), tg.astype(np.int64)), act)
                    self.vwriter[r[1] + d] = len(self.events)
            self.events.append(ev)
            return
        if op.startswith("global_"):
            store = op.startswith("global_store")
            ndw = {"dword": 1, "dwordx2": 2, "dwordx4": 4}.get(op.split("_", 2)[2], 1)
            if store:
                vaddr = o[0]
                if o[2] == "off":
                    lo, hi = self.vsrc("v%d" % preg(vaddr)[1]), self.vsrc("v%d" % (preg(vaddr)[1] + 1))
                    addr = (hi.val.astype(np.int64) << 32) | lo.val.astype(np.int64)
                    kn = lo.kn & hi.kn
                else:
                    sb = self.s64get(o[2])
                    vo = self.vsrc(vaddr)
                    addr = (sb or 0) + vo.val.astype(np.int64)
                    kn = vo.kn & (sb is not None)
                self.events.append(dict(kind="gstore", idx=x.idx, it=self.it, addr=addr + imm, kn=kn, act=act.copy()))
                return
            r = preg(o[0])
            if o[2] == "off":
                lo, hi = self.vsrc("v%d" % preg(o[1])[1]), self.vsrc("v%d" % (preg(o[1])[1] + 1))
                addr = (hi.val.astype(np.int64) << 32) | lo.val.astype(np.int64)
                kn = lo.kn & hi.kn
            else:
                sb = self.s64get(o[2])
                vo = self.vsrc(o[1])
                if sb is None:
                    raise ExecError("global base unknown")
                addr = sb + vo.val.astype(np.int64)
                kn = vo.kn
            addr = addr + imm
            if not kn[act].all():
                raise ExecError("global address not known")
            self.events.append(dict(kind="gload", idx=x.idx, it=self.it, addr=addr.copy(), act=act.copy(), dst=r[1],
                                    ndw=ndw))
            for d in range(ndw):
                tg = (addr + 4 * d)[:, None] + np.arange(4)[None, :]
                self.vset(r[1] + d, Vec(np.zeros(NL, np.uint64), np.zeros(NL, bool), tg.astype(np.int64)), act)
                self.vwriter[r[1] + d] = len(self.events) - 1
            return
        if op.startswith("ds_"):
            if op in ("ds_read_b128", "ds_write_b128", "ds_read_b32", "ds_write_b32", "ds_read_b64", "ds_write_b64"):
                n = {"b32": 1, "b64": 2, "b128": 4}[op.split("_")[2]]
                is_w = op.startswith("ds_write")
                va = self.vsrc(o[0] if is_w else o[1])
                if not va.kn[act].all():
                    raise ExecError("LDS address not known")
                addr = va.val.astype(np.int64) + imm
                ev = dict(kind="dswrite" if is_w else "dsread", idx=x.idx, it=self.it, addr=addr.copy(), imm=imm,
                          act=act.copy(), asm=x.asm, n=n)
                if is_w:
                    r = preg(o[1])
                    tags = np.concatenate([(self.v.get(r[1] + q) or Vec()).tag for q in range(n)], 1)
                    vals = [(self.v.get(r[1] + q) or Vec()) for q in range(n)]
                    ev["data"] = r[1]
                    ev["zero"] = np.logical_and.reduce([v.kn & (v.val == 0) for v in vals])
                    for l in range(NL):
                        if act[l]:
                            a0 = int(addr[l])
                            ex.lds[a0:a0 + 4 * n] = np.where(tags[l] == UNK, UNK, tags[l])
                else:
                    r = preg(o[0])
                    ev["dst"] = r[1]
                    for q in range(n):
                        tg = np.full((NL, 4), UNK, np.int64)
                        for l in range(NL):
                            if act[l]:
                                a0 = int(addr[l]) + 4 * q
                                tg[l] = ex.lds[a0:a0 + 4]
                        self.vset(r[1] + q, Vec(np.zeros(NL, np.uint64), np.zeros(NL, bool), tg), act)
                        self.vwriter[r[1] + q] = len(self.events)
                self.events.append(ev)
                return
            # ds_swizzle, ds_read2st64, ...: data not modelled
            d = preg(o[0]) if o and not op.startswith("ds_write") else None
            if d is not None and d[0] == "v":
                for i in range(d[1], d[2] + 1):
                    self.vset(i, Vec(), act)
            return
        raise ExecError("memory op not modelled: " + op)


def _med3(a, b, c):
    s = np.sort(np.stack([a, b, c], 1), 1)
    return s[:, 1]


def _s(a):
    a = a.astype(np.int64)
    return np.where(a >= 1 << 31, a - (1 << 32), a)


def _bitop3(x, a, b, c):
    imm = 0
    for md in x.mods:
        if md.startswith("bitop3:"):
            imm = int(md[7:], 0)
    r = np.zeros(NL, np.uint64)
    for bit in range(32):
        ab = (a >> np.uint64(bit)) & np.uint64(1)
        bb = (b >> np.uint64(bit)) & np.uint64(1)
        cb = (c >> np.uint64(bit)) & np.uint64(1)
        idx = (ab << np.uint64(2)) | (bb << np.uint64(1)) | cb
        r |= ((np.uint64(imm) >> idx) & np.uint64(1)) << np.uint64(bit)
    return r


U = np.uint64
INT_OPS = {
    "v_and_b32": (2, lambda x, a, b: a & b),
    "v_or_b32": (2, lambda x, a, b: a | b),
    "v_xor_b32": (2, lambda x, a, b: a ^ b),
    "v_not_b32": (1, lambda x, a: ~a),
    "v_add_u32": (2, lambda x, a, b: a + b),
    "v_sub_u32": (2, lambda x, a, b: (a - b) & U(M32)),
    "v_subrev_u32": (2, lambda x, a, b: (b - a) & U(M32)),
    "v_mul_lo_u32": (2, lambda x, a, b: (a * b) & U(M32)),
    "v_mul_u32_u24": (2, lambda x, a, b: (a & U(0xFFFFFF)) * (b & U(0xFFFFFF))),
    "v_mad_u32_u24": (3, lambda x, a, b, c: (a & U(0xFFFFFF)) * (b & U(0xFFFFFF)) + c),
    "v_lshlrev_b32": (2, lambda x, a, b: b << (a & U(31))),
    "v_lshrrev_b32": (2, lambda x, a, b: b >> (a & U(31))),
    "v_ashrrev_i32": (2, lambda x, a, b: (_s(b) >> (a & U(31)).astype(np.int64)).astype(np.uint64)),
    "v_lshl_or_b32": (3, lambda x, a, b, c: (a << (b & U(31))) | c),
    "v_and_or_b32": (3, lambda x, a, b, c: (a & b) | c),
    "v_or3_b32": (3, lambda x, a, b, c: a | b | c),
    "v_xor3_b32": (3, lambda x, a, b, c: a ^ b ^ c),
    "v_add3_u32": (3, lambda x, a, b, c: a + b + c),
    "v_lshl_add_u32": (3, lambda x, a, b, c: ((a << (b & U(31))) & U(M32)) + c),
    "v_add_lshl_u32": (3, lambda x, a, b, c: ((a + b) & U(M32)) << (c & U(31))),
    "v_xad_u32": (3, lambda x, a, b, c: (a ^ b) + c),
    "v_bfe_u32": (3, lambda x, a, b, c: (a >> (b & U(31))) & ((U(1) << (c & U(31))) - U(1))),
    "v_bfi_b32": (3, lambda x, a, b, c: (a & b) | (~a & c)),
    "v_bitop3_b32": (3, _bitop3),
    "v_med3_i32": (3, lambda x, a, b, c: _med3(_s(a), _s(b), _s(c)).astype(np.uint64)),
    "v_med3_u32": (3, lambda x, a, b, c: _med3(a.astype(np.int64), b.astype(np.int64), c.astype(np.int64)).astype(
        np.uint64)),
    "v_max_i32": (2, lambda x, a, b: np.maximum(_s(a), _s(b)).astype(np.uint64)),
    "v_min_i32": (2, lambda x, a, b: np.minimum(_s(a), _s(b)).astype(np.uint64)),
    "v_max_u32": (2, lambda x, a, b: np.maximum(a, b)),
    "v_min_u32": (2, lambda x, a, b: np.minimum(a, b)),
    "v_bfrev_b32": (1, lambda x, a: np.array([int(f"{int(v):032b}"[::-1], 2) for v in a], np.uint64)),
}


class Executor:
    def __init__(self, insts, labels, blocks, loop_blocks, hdr_idx):
        self.insts, self.labels, self.blocks = insts, labels, blocks
        self.loop_blocks, self.hdr_idx = loop_blocks, hdr_idx

    def run_wg(self, M, G, B):
        nt = (M + 63) // 64
        km = {}
        for i, k in enumerate(PARAM_ORDER):
            km[KARG + 8 * i] = BASES[k] & M32
            km[KARG + 8 * i + 4] = BASES[k] >> 32
        km[KARG + 0x60] = M & M32
        km[KARG + 0x64] = nt
        km[KARG + 0x68] = G          # hidden_block_count_x (gridDim.x)
        km[KARG + 0x6C] = 1
        km[KARG + 0x70] = 1
        km[KARG + 0x74] = 512 | (1 << 16)
        km[KARG + 0x78] = 1
        self.kmem = km
        self.lds = np.full(99328, UNWR, np.int64)
        waves = []
        for w in range(8):
            wv = Wave(self, w)
            wv.s[0], wv.s[1], wv.s[2] = KARG & M32, KARG >> 32, B
            tid = 64 * w + np.arange(NL)
            wv.v[0] = Vec(tid.astype(np.uint64), np.ones(NL, bool), const_tags(tid.astype(np.uint64), np.ones(NL, bool)))
            waves.append(wv)
        gens = [wv.run() for wv in waves]
        alive = [True] * 8
        while any(alive):
            hit = []
            for i, gg in enumerate(gens):
                if not alive[i]:
                    continue
                try:
                    next(gg)
                    hit.append(True)
                except StopIteration:
                    alive[i] = False
                    hit.append(False)
            if any(hit) and not all(hit):
                raise ExecError("waves disagree on the number of barriers (divergent barrier)")
        return waves


# ======================================================================================================================
# C16 / C17
# ======================================================================================================================
SCENARIOS = [  # (M, G, blocks, what)
    (357, 2, (0, 1), "loop zero fill rows 37 (WG 1, tile 5), full tiles, one-past-the-end"),
    (148, 3, (2,), "prologue zero fill rows 20 (WG 2's only tile)"),
    (1, 1, (0,), "prologue zero fill rows 1, M = 1"),
    (127, 1, (0,), "loop zero fill rows 63"),
    (128, 1, (0,), "full tiles only (no zero fill)"),
]


def tile_rows(t, M):
    return IDX.tile_rows(t, M)


def check(label, s0, emit, insts, labels, blocks, loop_blocks, hdr_idx, mf_info, err, note, scenarios=SCENARIOS):
    """mf_info: {mfma inst idx: (chain link k, column block n, row group m)} from C3 (compute instances).
    Returns a summary dict."""
    exr = Executor(insts, labels, blocks, loop_blocks, hdr_idx)
    summ = dict(scenarios=0, waves=0, dma=0, zfill_groups=0, mfma=0, rwrite=0, rread=0, h1=0, dh1=0, stats=0,
                readlane=0, da1=0, errors=0)
    nerr = {"C16": 0, "C17": 0}  # separate report budgets: one clause's flood must not hide the other's verdict

    def e16(msg):
        nerr["C16"] += 1
        if nerr["C16"] <= 12:
            err("C16 " + msg)

    def e17(msg):
        nerr["C17"] += 1
        if nerr["C17"] <= 24:
            err("C17 " + msg)

    lanes = np.arange(NL)
    for (M, G, blocks_, what) in scenarios:
        nt = (M + 63) // 64
        for B in blocks_:
            try:
                waves = exr.run_wg(M, G, B)
            except ExecError as e:
                err(f"C16/C17 executor (M {M} G {G} B {B}): {e}")
                return summ
            summ["scenarios"] += 1
            for wv in waves:
                w = wv.w
                summ["waves"] += 1
                ev = wv.events
                # split events by region: prologue (it -1) / iteration j / exit (-2); barrier positions per iteration
                iters = sorted({e["it"] for e in ev if e["it"] >= 0})
                tiles = {j: B + j * G for j in iters}
                if [B + j * G for j in iters] != [t for t in range(B, nt, G)]:
                    e17(f"M {M} G {G} B {B} w {w}: loop iterations {iters} do not run tiles {list(range(B, nt, G))}")
                # ---- (a) + C16: DMA groups and zero fill
                def dma_group(it_sel, T, buf):
                    nonlocal summ
                    rows = tile_rows(T, M)
                    dm = [e for e in ev if e["kind"] == "dma" and e["it"] == it_sel]
                    if s0:
                        if dm:
                            e17(f"S0 issues {len(dm)} LDS-DMA")
                        return
                    if len(dm) != 4:
                        e17(f"M {M} B {B} w {w} it {it_sel}: {len(dm)} LDS-DMA (want 4)")
                        return
                    seen_i = set()
                    for e in dm:
                        summ["dma"] += 1
                        i = e["soff"] // 128 if e["soff"] % 128 == 0 else -1
                        seen_i.add(i)
                        want_v = np.array([IDX.a_dma_voff(w, l) for l in lanes])
                        want_lds = IDX.a_dma_lds(buf, w, i) + 16 * lanes
                        rs = e["rsrc"]
                        want_base = BASES["dh8"] + 32768 * IDX.rsrc_tile(T, rows)
                        if not (0 <= i < 4 and e["imm"] == 0 and np.array_equal(e["voff"], want_v)
                                and np.array_equal(e["lds"], want_lds) and rs["base"] == want_base
                                and rs["nrec"] == 512 * rows and e["act"].all()):
                            e17(f"M {M} B {B} w {w} it {it_sel} DMA line {insts[e['idx']].line + 1}: soffset "
                                f"{e['soff']} imm {e['imm']} base {hex(rs['base'])} (want {hex(want_base)}) nrec "
                                f"{rs['nrec']} (want {512 * rows}) voffset ok {np.array_equal(e['voff'], want_v)} "
                                f"lds ok {np.array_equal(e['lds'], want_lds)}")
                    if seen_i != {0, 1, 2, 3}:
                        e17(f"M {M} B {B} w {w} it {it_sel}: DMA K steps {sorted(seen_i)}")
                    # C16: zero-fill writes of this group = A-region ds_writes of this wave in the same region
                    # before the next barrier
                    first = min(ev.index(e) for e in dm)
                    nxt_bar = next((k for k in range(first, len(ev)) if ev[k]["kind"] == "barrier"), len(ev))
                    zw = [e for e in ev[first:nxt_bar] if e["kind"] == "dswrite" and
                          (e["addr"][e["act"]] < 65536).any()]
                    oob = np.array([IDX.a_dma_voff(w, l) // 512 >= rows for l in lanes])  # DMA lanes out of range
                    want_act = np.array([IDX.zfill_lane(rows, w, l) for l in lanes])
                    if not np.array_equal(oob & (0 < rows < 64), want_act):
                        e16(f"zfill_lane != out-of-range DMA lanes (rows {rows}, w {w})")
                    if want_act.any():  # this wave owns rows >= rows of a partial tile
                        summ["zfill_groups"] += 1
                        if len(zw) != 4:
                            e16(f"M {M} B {B} w {w} it {it_sel} rows {rows}: {len(zw)} zero-fill ds_write (want 4)")
                        dl = {e["soff"] // 128: e["lds"] for e in dm}
                        imms = set()
                        for z in zw:
                            i = z["imm"] // 8192 if z["imm"] % 8192 == 0 else -1
                            imms.add(i)
                            if not (z["asm"] and z["n"] == 4 and np.array_equal(z["act"], want_act)
                                    and z["zero"][z["act"]].all() and i in dl
                                    and np.array_equal(z["addr"][z["act"]], dl[i][z["act"]])):
                                e16(f"M {M} B {B} w {w} it {it_sel} rows {rows}: zero-fill write line "
                                    f"{insts[z['idx']].line + 1}: active lanes {int(z['act'].sum())} (want "
                                    f"{int(want_act.sum())}, exact {np.array_equal(z['act'], want_act)}), data zero "
                                    f"{bool(z['zero'][z['act']].all())}, address == own DMA dest "
                                    f"{i in dl and np.array_equal(z['addr'][z['act']], dl[i][z['act']])}")
                        if imms != {0, 1, 2, 3}:
                            e16(f"M {M} B {B} w {w} it {it_sel}: zero-fill immediates cover K steps {sorted(imms)}")
                    elif zw:
                        e16(f"M {M} B {B} w {w} it {it_sel} rows {rows}: {len(zw)} A-region ds_write with active "
                            f"lanes, but this wave owns no padded row of the tile")

                dma_group(-1, B, 0)
                for j in iters:
                    dma_group(j, tiles[j] + G, (j + 1) % 2)
                # ---- (b) MFMA operand bytes, fragment-read and W-load addresses
                for j in iters:
                    T = tiles[j]
                    buf = j % 2
                    mfs = [e for e in ev if e["kind"] == "mfma" and e["it"] == j]
                    if s0:
                        continue
                    if len(mfs) != 32:
                        e17(f"M {M} B {B} w {w} it {j}: {len(mfs)} MFMA executed (want 32)")
                    for e in mfs:
                        summ["mfma"] += 1
                        k, n, m = mf_info.get(e["idx"], (None, None, None))
                        if k is None:
                            e17(f"MFMA line {insts[e['idx']].line + 1} has no C3 chain position")
                            continue
                        q4 = lanes // 16
                        r16 = lanes % 16
                        Kb = np.concatenate([np.arange(16), 64 + np.arange(16)])
                        K = 128 * k + 16 * q4[:, None] + Kb[None, :]
                        row = 64 * T + 16 * m + r16
                        want1 = np.where((row < M)[:, None], BASES["dh8"] + row[:, None] * 512 + K, ZERO)
                        col = np.array([IDX.w_col(w, n, l) for l in lanes])
                        want0 = BASES["w8t"] + col[:, None] * 512 + K
                        if not np.array_equal(e["src1"], want1):
                            bad = np.argwhere(e["src1"] != want1)
                            l, b = bad[0]
                            e17(f"M {M} B {B} w {w} it {j} MFMA line {insts[e['idx']].line + 1} (k {k} m {m}): src1 "
                                f"lane {l} byte {b} = {_tagstr(e['src1'][l, b])} (want {_tagstr(want1[l, b])}); "
                                f"{len(bad)} bytes differ")
                        if not np.array_equal(e["src0"], want0):
                            bad = np.argwhere(e["src0"] != want0)
                            l, b = bad[0]
                            e17(f"M {M} B {B} w {w} it {j} MFMA line {insts[e['idx']].line + 1} (k {k} n {n}): src0 "
                                f"lane {l} byte {b} = {_tagstr(e['src0'][l, b])} (want {_tagstr(want0[l, b])})")
                    # the fragment-read addresses (lo quad from lp0, hi quad from lp1) are checked on the read events below
                    reads = [e for e in ev if e["kind"] == "dsread" and e["it"] == j and e["asm"]
                             and (e["addr"] < 65536).all()]
                    want = {}
                    for kk in range(4):
                        for mm in range(4):
                            fo = IDX.frag_off(buf, kk, mm)
                            want[(0, kk, mm)] = np.array([IDX.frag_lp0(l) for l in lanes]) + fo
                            want[(1, kk, mm)] = np.array([IDX.frag_lp1(l) for l in lanes]) + fo
                    got = set()
                    for e in reads:
                        hit = [key for key, a in want.items() if np.array_equal(e["addr"], a)]
                        if len(hit) != 1:
                            e17(f"M {M} B {B} w {w} it {j}: fragment read line {insts[e['idx']].line + 1} address "
                                f"matches no frag_lp0/lp1 + frag_off(buf {buf}, k, m)")
                        else:
                            got.add(hit[0])
                    if got != set(want):
                        e17(f"M {M} B {B} w {w} it {j}: fragment reads cover {len(got)} of 32 (half, k, m)")
                # W loads (prologue): every buffer_load from the w8t resource
                if not s0:
                    wl = [e for e in ev if e["kind"] == "bload" and e["rsrc"]["base"] == BASES["w8t"]]
                    seen = set()
                    for e in wl:
                        hit = [(n, k, h) for n in range(2) for k in range(4) for h in range(2)
                               if np.array_equal(e["voff"] + e["imm"],
                                                 np.array([IDX.w_voff(w, l) + IDX.w_imm(n, k, h) for l in lanes]))]
                        if len(hit) != 1 or e["rsrc"]["nrec"] != 256 * 512 or e["soff"] != 0 or e["it"] != -1:
                            e17(f"w {w}: W load line {insts[e['idx']].line + 1} matches {hit} (nrec "
                                f"{e['rsrc']['nrec']}, soffset {e['soff']}, region {e['it']})")
                        else:
                            seen.add(hit[0])
                    if len(seen) != 16:
                        e17(f"w {w}: W loads cover {len(seen)} of 16 (n, k, h) halves")
                # ---- (c) R writes / reads
                for j in iters:
                    if not s0:
                        rw = [e for e in ev if e["kind"] == "dswrite" and e["it"] == j and
                              (e["addr"][e["act"]] >= 65536).any()]
                        ms = set()
                        for e in rw:
                            summ["rwrite"] += 1
                            hit = [mm for mm in range(4) if np.array_equal(
                                e["addr"], np.array([IDX.rw_base(w, l) for l in lanes]) + mm * 16 * 528)]
                            if len(hit) != 1 or not e["act"].all():
                                e17(f"w {w} it {j}: R write line {insts[e['idx']].line + 1} is not rw_base + m*8448")
                            else:
                                ms.add(hit[0])
                        if ms != {0, 1, 2, 3}:
                            e17(f"w {w} it {j}: R writes cover row groups {sorted(ms)}")
                        rr = [e for e in ev if e["kind"] == "dsread" and e["it"] == j and (e["addr"] >= 65536).all()]
                        us = set()
                        for e in rr:
                            summ["rread"] += 1
                            hit = [u for u in range(4) if np.array_equal(
                                e["addr"], np.array([IDX.rr_base(w, l) for l in lanes]) + u * 16 * 528)]
                            if len(hit) != 1:
                                e17(f"w {w} it {j}: R read line {insts[e['idx']].line + 1} is not rr_base + u*8448")
                            else:
                                us.add(hit[0])
                        if us != {0, 1, 2, 3}:
                            e17(f"w {w} it {j}: R reads cover slots {sorted(us)}")
                # ---- (d) consumer global traffic
                cv = {u: np.array([IDX.c_voff(w, u, l) for l in lanes]) for u in range(4)}

                def slot_of(e):
                    hit = [u for u in range(4) if np.array_equal(e["voff"] + e["imm"], cv[u])]
                    return hit[0] if len(hit) == 1 else None

                def check_rs(e, tensor, T, rowb):
                    rows = tile_rows(T, M)
                    want_base = BASES[tensor] + 64 * rowb * IDX.rsrc_tile(T, rows)
                    return e["rsrc"]["base"] == want_base and e["rsrc"]["nrec"] == rows * rowb and e["soff"] == 0

                tens_loads = ("h1", "da1") if s0 else ("h1",)
                pro_reg = {}
                for tensor in tens_loads:
                    pl = [e for e in ev if e["kind"] == "bload" and e["it"] == -1
                          and tensor_of(e["rsrc"]["base"])[0] == tensor]
                    for e in pl:
                        u = slot_of(e)
                        pro_reg[(tensor, e["dst"])] = u
                        if u is None or not check_rs(e, tensor, B, 512):
                            e17(f"w {w}: prologue {tensor} load line {insts[e['idx']].line + 1}: slot {u}, resource ok "
                                f"{check_rs(e, tensor, B, 512)}")
                        summ["h1" if tensor == "h1" else "da1"] += 1
                    if sorted(u for (tt, _), u in pro_reg.items() if tt == tensor) != [0, 1, 2, 3]:
                        e17(f"w {w}: prologue {tensor} loads cover slots "
                            f"{sorted(str(u) for (tt, _), u in pro_reg.items() if tt == tensor)}")
                for j in iters:
                    T = tiles[j]
                    seg = [e for e in ev if e["it"] == j]
                    st = [e for e in seg if e["kind"] == "bstore" and tensor_of(e["rsrc"]["base"])[0] == "dh1"]
                    if len(st) != 4:
                        e17(f"w {w} it {j}: {len(st)} dh1 stores (want 4)")
                        continue
                    for u, e in enumerate(st):
                        summ["dh1"] += 1
                        if not (np.array_equal(e["voff"] + e["imm"], cv[u]) and check_rs(e, "dh1", T, 512)):
                            e17(f"w {w} it {j}: dh1 store {u} line {insts[e['idx']].line + 1}: voffset == c_voff(w, {u}) "
                                f"{np.array_equal(e['voff'] + e['imm'], cv[u])}, resource ok {check_rs(e, 'dh1', T, 512)}")
                    pos = [seg.index(e) for e in st] + [len(seg)]
                    for tensor in tens_loads:
                        for u in range(4):
                            pf = [e for e in seg[pos[u]:pos[u + 1]] if e["kind"] == "bload"
                                  and tensor_of(e["rsrc"]["base"])[0] == tensor]
                            if len(pf) != 1:
                                e17(f"w {w} it {j}: {len(pf)} {tensor} prefetches after dh1 store {u} (want 1)")
                                continue
                            e = pf[0]
                            summ["h1" if tensor == "h1" else "da1"] += 1
                            ok = np.array_equal(e["voff"] + e["imm"], cv[u]) and check_rs(e, tensor, T + G, 512)
                            if not ok or pro_reg.get((tensor, e["dst"])) != u:
                                e17(f"w {w} it {j}: {tensor} prefetch of slot {u} line {insts[e['idx']].line + 1}: "
                                    f"address/resource ok {ok}, its registers v{e['dst']} were loaded in the prologue "
                                    f"for slot {pro_reg.get((tensor, e['dst']))}")
                    if emit:
                        es = [e for e in seg if e["kind"] == "bstore" and tensor_of(e["rsrc"]["base"])[0] == "da1"]
                        mm_ = set()
                        for e in es:
                            summ["da1"] += 1
                            hit = [mm for mm in range(4) if np.array_equal(
                                e["voff"] + e["imm"], np.array([IDX.e_voff(w, mm, l) for l in lanes]))]
                            if len(hit) != 1 or not check_rs(e, "da1", T, 512):
                                e17(f"w {w} it {j}: EMIT da1 store line {insts[e['idx']].line + 1}: e_voff match {hit}")
                            else:
                                mm_.add(hit[0])
                        if mm_ != {0, 1, 2, 3}:
                            e17(f"w {w} it {j}: EMIT stores cover row groups {sorted(mm_)}")
                    # mean / rstd loads of this iteration (tile T + G) and of the prologue (tile B)
                for (it_sel, T) in [(-1, B)] + [(j, tiles[j] + G) for j in iters]:
                    for tensor in ("mean1", "rstd1"):
                        sl = [e for e in ev if e["kind"] == "bload" and e["it"] == it_sel
                              and tensor_of(e["rsrc"]["base"])[0] == tensor]
                        summ["stats"] += len(sl)
                        if len(sl) != 1 or not (np.array_equal(sl[0]["voff"] + sl[0]["imm"], 4 * lanes)
                                                and check_rs(sl[0], tensor, T, 4)):
                            e17(f"w {w} region {it_sel}: {tensor} loads {len(sl)} (want 1 at voffset 4*lane, tile {T})")
                # ---- (e) readlane lanes and the half select
                for j in iters:
                    rl = [e for e in ev if e["kind"] == "readlane" and e["it"] == j]
                    summ["readlane"] += len(rl)
                    lanes_set = sorted(e["lane"] for e in rl if e["lane"] is not None)
                    want_l = sorted([16 * u + 2 * w + h for u in range(4) for h in range(2)] * 2)
                    if lanes_set != want_l:
                        e17(f"w {w} it {j}: v_readlane lanes {lanes_set} (want 2 x {{16u + 2w + h}})")
                    cm = [e for e in ev if e["kind"] == "cndmask" and e["it"] == j]
                    badc = [e for e in cm if e["cond"] != 0xFFFFFFFF00000000]
                    if not cm or badc:
                        e17(f"w {w} it {j}: {len(badc)} of {len(cm)} consumer v_cndmask see a condition other than "
                            f"lanes 32..63")
                # nw / nb loads (prologue)
                for tensor in ("nw", "nb"):
                    gl = [e for e in ev if e["kind"] == "gload" and tensor_of(int(e["addr"][0]))[0] == tensor]
                    offs = sorted(set(tuple((e["addr"] - BASES[tensor]).tolist()) for e in gl))
                    want_o = sorted({tuple((4 * np.array([IDX.c_col0(l) for l in lanes]) + 16 * h).tolist())
                                     for h in range(2)})
                    if offs != want_o or any(e["it"] != -1 for e in gl):
                        e17(f"w {w}: {tensor} loads are not 4*c_col0(lane) (+16) in the prologue")
    summ["errors"] = dict(nerr)
    for k, lim in (("C16", 12), ("C17", 24)):
        if nerr[k] > lim:
            err(f"{k} ... {nerr[k]} violations in total (first {lim} shown)")
    return summ


def _tagstr(t):
    t = int(t)
    if t == ZERO:
        return "0"
    if t in (UNK, UNWR, CONST):
        return {UNK: "unknown", UNWR: "never-written LDS", CONST: "a constant"}[t]
    k, off = tensor_of(t)
    if k in ("dh8", "w8t"):
        return f"{k}[{off // 512}][{off % 512}]"
    return f"{k}+{off}"
