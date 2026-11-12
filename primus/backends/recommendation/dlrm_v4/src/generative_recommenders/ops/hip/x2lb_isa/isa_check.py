#!/usr/bin/env python3
"""isa_check.py -- ISA contract gate of X2L-B v0 on the compiled gfx950 assembly (CPU only, stdlib).

Usage:  PYTHONDONTWRITEBYTECODE=1 python -B isa_check.py [out/<cfg>/x2lb_pass.s ...] [--json out/isa_check.json]
        (default: out/{r720,r720_ilp,sdk10,sdk10_ilp}/x2lb_pass.s; the gate is r720, r720_ilp, sdk10)

Adapted from X2H's probes/iter68/x2h/build/isa_check.py (C1-C12); C8-C12 are ported from it (the CFG, the wait-coverage
dataflow incl. inline asm, the asm-vs-MFMA hazard walk, the EXEC dataflow and the exit drain) with the EXEC-sensitive
set extended by DPP and ds_swizzle. Kernels: x2lb_pass<S0, EMIT> (S1 = <0,0>, S1 + DA1 = <0,1>, S0 = <1,0>) and x2lb_fin.
Regions of x2lb_pass: prologue (before the persistent loop header) | loop body = top wait + s_barrier, LDS-DMA of the
next A tile, MFMA phase (first .. last MFMA), epilogue, mid s_barrier, consumer (4 slots, each ending at its dh1
buffer_store_dwordx4) | exit (partials).

  C1  MFMAs (compute instances): exactly 32 per loop body, all `v_mfma_f32_16x16x128_f8f6f4 ... blgp:1` (opcode 45,
      plain form; src0 = W e4m3 with cbsz 0, src1 = dh8 e5m2), in 8 chains of exactly 4 links (srcC = previous dst,
      followed through renames) whose head srcC is the inline constant 0; no other instruction writes a chain's
      accumulator between its links. S0: no MFMA and no LDS-DMA at all.
  C2  no cbsz / abid / neg_lo / neg_hi / clamp / op_sel modifiers on any MFMA (blgp:1 only).
  C3  operand roles = the contract's per-lane K bytes [CD 0, 6.2]: src0 of chain link k = a resident W octet whose two
      quads were loaded in the prologue by buffer_load_dwordx4 from one voffset with immediates 2048n + 128k and
      2048n + 128k + 64 (n = column block, the same for the whole chain), never written inside the loop; src1 = an A
      fragment whose quads come from two inline-asm ds_read_b128 at immediate k*8192 + m*2048 from the two fragment
      base registers (lo from one, hi from the other, the same pair for every MFMA), m the same along the chain;
      every (n, m) pair occurs once per loop body.
  C4  EPI2 dataflow [CD 6.3]: each of the 32 accumulator floats of the 8 chains is next read by a fp32 multiply with
      the scale register S (written in the prologue by v_mul_f32 = sa*sb, possibly via v_mov), whose result is next read
      by an add of the inline constant 0, whose result is next read by v_cvt_pk_bf16_f32; no FMA/MAC between the first
      MFMA and the mid barrier. bf16 rounding is v_cvt_pk_bf16_f32 only (16 epilogue + 16 consumer per loop body; S0:
      16) and no software-RNE pattern (v_cmp_u_f32, literal 0x7fff / 0x400000, v_bfe_u32 .., 16, 1) anywhere.
  C5  kernel descriptor of every kernel: float_round_mode_32 0, float_denorm_mode_32 3, ieee_mode 1, dx10_clamp 1,
      fp16_overflow 0 (= hipBLASLt's and PR #7's MODE [CD 5.1, CS 2]).
  C6  no s_waitcnt with a vmcnt field between the first and the last MFMA; every vmcnt wait classified (prologue /
      top = immediately before the loop's first s_barrier, with vmcnt == the design count 10 (S1) / 14 (S1+DA1) /
      consumer = after the mid barrier / exit); anything else = UNEXPECTED (fail).
  C7  consumer census per slot (the 4 slots split at the dh1 stores): 8 v_exp_f32, 8 v_rcp_f32, 2 ds_swizzle_b32
      (offset swizzle(SWAP,16)), 8 v_add_f32_dpp (quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror, twice),
      4 v_cvt_pk_bf16_f32, exactly 56 fused fp32 FMA lanes (v_pk_fma_f32 = 2, v_fma_f32 / v_fmac_f32 = 1: the seven
      fused steps E1 E2 E6 E10 E13 E14 E15 x 8 columns, [CS 2]) and no v_fma_mix* / v_mad* / v_mac* / v_fma_legacy;
      FMA lanes outside the consumer: 0. Plus the exact constants: 0xbfb8aa3b (-log2 e, prologue) and 0x3b800000 (2^-8,
      2 per slot).
  C8  register-level wait coverage (X2H C8): every VGPR written by a load (ds_read*, ds_swizzle / bpermute, VMEM loads,
      INCLUDING inline asm) is retired by an s_waitcnt before any read (RAW) or other-counter overwrite (WAW), on every
      CFG path (LGKM in order unless SMEM is outstanding; VM in order).
  C9  every s_barrier is preceded (s_nop skipped) by an s_waitcnt with lgkmcnt(0).
  C10 inline-asm DS instructions vs in-flight MFMA (vDst RAW/WAW, SrcC WAR) within 20 wait states (X2H C10).
  C11 inside the persistent loop every v_mfma*, DPP op, ds_swizzle, v_permlane*, s_barrier, v_readfirstlane,
      v_mbcnt* and ballot runs under full EXEC (X2H C11 dataflow, + DPP / swizzle).
  C12 from every LDS-DMA, every path to s_endpgm passes an s_waitcnt vmcnt(0) (X2H C12).
  C13 LDS-DMA retirement (new): on every CFG path, an LDS-DMA is retired by a vmcnt wait (>= N VMEM instructions issued
      after it) before the SECOND s_barrier that follows it -- the A tile prefetched in iteration j (after its top
      barrier) is complete in every wave before the top barrier of iteration j+1, after which it is read.
  C14 x2lb_fin: no FMA/MAC, no packed add (one sequential scalar add chain over g), the 32-deep load batch present,
      descriptor as C5.
Fix pass (2026-10-12, task F; IMPLEMENTATION.md "fix pass"):
  C10 the asm-vs-MFMA walk keeps a separate srcC-WAR set that only a compiler-visible WRITE prunes (race review F4).
  C15 symbolic dataflow of the compiled kernel against CONTRACT_dgrad 6.3 / CONTRACT_swish 2-3 (isa_dataflow.py, the
      contract reviewer's dfcheck D1-D3): EPI2 per bf16 half incl. the item / word packing, the MFMA chains' W / A
      operands per link, every consumer dh1 word = PR #7's expression with its leaves bound to the slot's h1 quad, R
      read, readlane rows and nw / nb columns, and the in-order 4-slot accumulator folds (contract review F1). C4 and
      C7 remain censuses; C15 is the order / role / sign / constant check.
  C16 zero fill of partial tiles, semantically, by concrete execution (isa_exec.py; race review F1).
  C17 concrete per-lane addressing (DMA, fragment reads incl. lo / hi, W loads, R writes / reads, h1 / dh1 / da1 /
      mean / rstd / nw / nb, readlane lanes, half select) and byte provenance of every MFMA operand (contract review F3).
  C18 LDS-DMA census (S1: 4 prologue + 4 loop body; S0: none) and every inline-asm s_waitcnt is exactly lgkmcnt(0)
      (race review F4 optional item, F6).
  W1  (warning, not a gate) MFMA operand-overlap census; a vDst that partially overlaps srcC is reported (race F5).
Fail closed: a missing ISA file or kernel is a FAIL; an exception inside C15 / C16 / C17 is a FAIL.
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = {
    "_ZN4x2lb9x2lb_passILi0ELi0EEEvNS_6ParamsE": ("x2lb_pass<0,0> S1", 0, 0),
    "_ZN4x2lb9x2lb_passILi0ELi1EEEvNS_6ParamsE": ("x2lb_pass<0,1> S1+DA1", 0, 1),
    "_ZN4x2lb9x2lb_passILi1ELi0EEEvNS_6ParamsE": ("x2lb_pass<1,0> S0", 1, 0),
}
FIN = "_ZN4x2lb8x2lb_finEPKfiPf"
MF = "v_mfma_f32_16x16x128_f8f6f4"
TOP_VMCNT = {(0, 0): 10, (0, 1): 14}
REG = re.compile(r"^([vas])\[(\d+):(\d+)\]$|^([vas])(\d+)$")
FMA_RE = re.compile(r"^v_(pk_)?(fma|fmac|mac|mad|fma_mix|fma_legacy|mad_legacy|fmamk|fmaak|madmk|madak)\w*_f(16|32|64)")
FMA_BAD = re.compile(r"^v_(fma_mix|mad|mac|fma_legacy|mad_legacy|madmk|madak|fmamk|fmaak)")


# =====================================================================================================================
# parsing (X2H isa_check.py: reg, regs_of, split_ops; s_parse / s_cfg ported verbatim below)
# =====================================================================================================================
def reg(tok):
    tok = tok.strip()
    m = REG.match(tok)
    if not m:
        return None
    if m.group(1):
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    return (m.group(4), int(m.group(5)), int(m.group(5)))


def regs_of(r):
    return [(r[0], i) for i in range(r[1], r[2] + 1)] if r else []


def split_ops(rest):
    ops, cur, depth = [], "", 0
    for ch in rest:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append(cur.strip())
            cur = ""
        else:
            cur += ch
    ops.append(cur.strip())
    last = ops[-1].split()
    mods = ""
    if len(last) > 1:
        ops[-1] = last[0]
        mods = " ".join(last[1:])
    return ops, mods


class SInst:
    __slots__ = ("idx", "line", "op", "ops", "mods", "text", "asm", "blk", "inloop")

    def __init__(self, idx, line, op, ops, mods, text, asm):
        self.idx, self.line, self.op, self.ops, self.mods, self.text, self.asm = idx, line, op, ops, mods, text, asm
        self.blk = None
        self.inloop = False


def _sregs(tok):
    r = reg(tok)
    return [(r[0], i) for i in range(r[1], r[2] + 1)] if r else []


def _vregs(tok):
    return [r for r in _sregs(tok) if r[0] in "va"]


def s_parse(lines):
    """instructions (inline-asm flag from ;;#ASMSTART/;;#ASMEND, loop membership from LLVM's block annotations) and
    block-start labels (X2H)"""
    insts, labels, blk_starts, pending = [], {}, [], []
    asm, inloop = False, False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith(";;#ASMSTART"):
            asm = True
            continue
        if s.startswith(";;#ASMEND"):
            asm = False
            continue
        m = re.match(r"^(\.LBB\d+_\d+):", ln)
        m2 = re.match(r"^; %bb\.(\d+):", ln)
        if m or m2:
            pending.append(m.group(1) if m else "%bb." + m2.group(1))
            inloop = any(k in ln for k in ("in Loop:", "Loop Header", "Parent Loop"))
            continue
        t = ln.split(";")[0].strip()
        if not t or t.startswith(".") or t.endswith(":"):
            continue
        parts = t.split(None, 1)
        ops, mods = split_ops(parts[1]) if len(parts) > 1 else ([], "")
        x = SInst(len(insts), i, parts[0], ops, mods.split(), t, asm)
        x.inloop = inloop
        if pending:
            for lb in pending:
                labels[lb] = x.idx
            blk_starts.append(x.idx)
            pending = []
        insts.append(x)
    return insts, labels, blk_starts


def s_cfg(insts, labels, blk_starts):
    starts = set(blk_starts) | {0}
    for x in insts:
        if x.op.startswith(("s_branch", "s_cbranch", "s_endpgm", "s_setpc")) and x.idx + 1 < len(insts):
            starts.add(x.idx + 1)
    starts = sorted(starts)
    blocks = []
    for bi, st in enumerate(starts):
        en = starts[bi + 1] if bi + 1 < len(starts) else len(insts)
        blocks.append(list(range(st, en)))
        for k in range(st, en):
            insts[k].blk = bi
    s2b = {st: bi for bi, st in enumerate(starts)}
    succ = {}
    for bi, b in enumerate(blocks):
        last = insts[b[-1]]
        if last.op == "s_branch":
            s = [s2b[labels[last.ops[0]]]]
        elif last.op.startswith("s_cbranch"):
            s = [s2b[labels[last.ops[0]]]] + ([bi + 1] if bi + 1 < len(blocks) else [])
        elif last.op.startswith(("s_endpgm", "s_setpc")):
            s = []
        else:
            s = [bi + 1] if bi + 1 < len(blocks) else []
        succ[bi] = s
    pred = {bi: [] for bi in range(len(blocks))}
    for bi, ss in succ.items():
        for t in ss:
            pred[t].append(bi)
    return blocks, succ, pred


def _is_smem(x):
    return x.op.startswith(("s_load", "s_buffer_load", "s_scratch_load", "s_atomic", "s_buffer_atomic", "s_dcache",
                            "s_memtime", "s_memrealtime", "s_store", "s_buffer_store"))


def _is_vmem(x):
    return x.op.startswith(("buffer_", "global_", "scratch_", "flat_", "tbuffer_")) and not x.op.startswith(
        ("buffer_wbl2", "buffer_inv", "buffer_wbinvl1"))


def _is_lds_dma(x):
    return x.op.startswith("global_load_lds") or (x.op.startswith("buffer_load") and "lds" in x.mods)


def _lgkm_inc(x):
    return x.op.startswith("ds_") or _is_smem(x) or x.op.startswith(("s_sendmsg", "flat_"))


def _load_dst(x):
    op = x.op
    if op.startswith("ds_"):
        if op.startswith(("ds_read", "ds_load")) or "_rtn" in op or op.startswith(
                ("ds_swizzle", "ds_bpermute", "ds_permute", "ds_consume", "ds_append")):
            return ("lgkm", _vregs(x.ops[0]))
        return None
    if _is_vmem(x) and not _is_lds_dma(x):
        if op.startswith(("buffer_load", "global_load", "scratch_load", "flat_load", "tbuffer_load")):
            return ("vm", _vregs(x.ops[0]))
        if "atomic" in op and "sc0" in x.mods:
            return ("vm", _vregs(x.ops[0]))
    return None


def _vreads(x):
    op = x.op
    if op.startswith(("v_permlane16_swap", "v_permlane32_swap")):
        srcs = x.ops[0:2]
    elif op.startswith(("ds_write", "ds_store", "global_store", "buffer_store", "scratch_store", "flat_store")):
        srcs = x.ops
    elif op.startswith("ds_") and not _load_dst(x):
        srcs = x.ops
    elif op.startswith(("s_waitcnt", "s_nop", "s_barrier", "s_branch", "s_cbranch", "s_endpgm", "s_setprio")):
        srcs = []
    elif op.startswith("v_writelane"):
        srcs = x.ops[1:] + [x.ops[0]]
    else:
        srcs = x.ops[1:]
    return [r for t in srcs for r in _vregs(t)]


def _vwrites(x):
    op = x.op
    if op.startswith(("v_permlane16_swap", "v_permlane32_swap")):
        return [r for t in x.ops[0:2] for r in _vregs(t)]
    if op.startswith(("ds_write", "ds_store", "global_store", "buffer_store", "scratch_store", "flat_store", "s_")):
        return []
    if _is_lds_dma(x) or (op.startswith("ds_") and not _load_dst(x)):
        return []
    if "atomic" in op and "sc0" not in x.mods and _is_vmem(x):
        return []
    if not x.ops or not x.ops[0]:
        return []
    return _vregs(x.ops[0])


def _waitcnt(x):
    w = {}
    for k in ("vmcnt", "lgkmcnt", "expcnt"):
        m = re.search(k + r"\((\d+)\)", x.text)
        if m:
            w[k] = int(m.group(1))
    m = re.match(r"s_waitcnt\s+(0x[0-9a-fA-F]+|\d+)$", x.text)
    if m:
        v = int(m.group(1), 0)
        w = dict(vmcnt=(v & 15) | ((v >> 14) & 3) << 4, expcnt=(v >> 4) & 7, lgkmcnt=(v >> 8) & 15)
        if w["vmcnt"] == 63:
            del w["vmcnt"]
        if w["lgkmcnt"] == 15:
            del w["lgkmcnt"]
    return w


# =====================================================================================================================
# C8-C12 (ported from X2H isa_check.py; C11 sensitive set + DPP / ds_swizzle)
# =====================================================================================================================
def c8_wait_coverage(insts, blocks, succ):
    loads = {x.idx: _load_dst(x) for x in insts if _load_dst(x)}
    IN = {bi: None for bi in range(len(blocks))}
    IN[0] = ({}, False)
    viol, work, it = {}, [0], 0
    while work:
        it += 1
        if it > 400000:
            raise RuntimeError("C8 dataflow: no fixed point")
        bi = work.pop(0)
        pend, smem = dict(IN[bi][0]), IN[bi][1]
        for k in blocks[bi]:
            x = insts[k]
            if x.op == "s_waitcnt":
                w = _waitcnt(x)
                for li in [li for li, (kind, after) in pend.items()
                           if (kind == "lgkm" and "lgkmcnt" in w and (w["lgkmcnt"] == 0 or (not smem and after >= w["lgkmcnt"])))
                           or (kind == "vm" and "vmcnt" in w and (w["vmcnt"] == 0 or after >= w["vmcnt"]))]:
                    del pend[li]
                if w.get("lgkmcnt") == 0:
                    smem = False
                continue
            rd, wr, dl = set(_vreads(x)), set(_vwrites(x)), _load_dst(x)
            for li, (kind, _) in pend.items():
                lr = set(loads[li][1])
                if rd & lr:
                    viol.setdefault((li, k, "RAW"), sorted(rd & lr))
                if wr & lr and not (dl is not None and dl[0] == kind):
                    viol.setdefault((li, k, "WAW"), sorted(wr & lr))
            if _lgkm_inc(x):
                pend = {li: (kd, a + (kd == "lgkm")) for li, (kd, a) in pend.items()}
            if _is_smem(x):
                smem = True
            if _is_vmem(x):
                pend = {li: (kd, a + (kd == "vm")) for li, (kd, a) in pend.items()}
            if dl:
                for li in [li for li in pend if pend[li][0] == dl[0] and set(loads[li][1]) <= set(dl[1])]:
                    del pend[li]
                pend[k] = (dl[0], 0)
        for t in succ[bi]:
            old = IN[t]
            if old is None:
                new = (dict(pend), smem)
            else:
                m = dict(old[0])
                for li, (kd, a) in pend.items():
                    m[li] = (kd, min(a, m[li][1])) if li in m else (kd, a)
                new = (m, old[1] or smem)
            if old is None or new != old:
                IN[t] = new
                if t not in work:
                    work.append(t)
    return viol


def c10_asm_mfma_hazards(insts, blocks, pred, window=20):
    """Backward walk from every inline-asm DS instruction over <= `window` wait states. Fix pass (race review F4): three
    tracked register sets instead of two. rd (asm reads vs MFMA vDst, RAW) and ww (asm writes vs MFMA vDst, WAW) are
    pruned by any compiler-visible read or write of the register (a compiler-visible read means the compiler already
    waited for the MFMA result; a write supersedes it). wc (asm writes vs MFMA srcC, WAR) is pruned ONLY by a
    compiler-visible WRITE (which the compiler's own SrcC-WAR handling orders after the MFMA's read): a compiler-visible
    READ of the same register is a read-after-read that needs no wait, so it proves nothing about the MFMA's pending
    srcC read (the old code pruned it, which could hide an asm overwrite of a live srcC)."""
    out = []

    def ws(x):
        return int(x.ops[0], 0) + 1 if x.op == "s_nop" else 1

    for x in insts:
        if not x.asm or not x.op.startswith("ds_"):
            continue
        rd0, wr0 = frozenset(_vreads(x)), frozenset(_vwrites(x))
        if not rd0 and not wr0:
            continue
        stack, seen = [(x.blk, x.idx - 1, 0, 0, rd0, wr0, wr0)], set()
        while stack:
            bi, k, acc, depth, rd, ww, wc = stack.pop()
            if (bi, k, rd, ww, wc) in seen or depth > 32 or (not rd and not ww and not wc):
                continue
            seen.add((bi, k, rd, ww, wc))
            stop = False
            while k >= blocks[bi][0]:
                y = insts[k]
                if y.op.startswith("v_mfma"):
                    d = set(_vregs(y.ops[0]))
                    c = set(_vregs(y.ops[3])) if len(y.ops) > 3 else set()
                    for kind, hit in (("MFMA vDst -> asm read (RAW)", rd & d), ("MFMA vDst / asm write (WAW)", ww & d),
                                      ("MFMA SrcC / asm write (WAR)", wc & c)):
                        if hit:
                            out.append(dict(asm_line=x.line, asm=x.text, mfma_line=y.line, mfma=y.text, ws=acc,
                                            kind=kind))
                elif not y.asm:
                    yr, yw = set(_vreads(y)), set(_vwrites(y))
                    rd, ww, wc = frozenset(rd - (yr | yw)), frozenset(ww - (yr | yw)), frozenset(wc - yw)
                    if not rd and not ww and not wc:
                        stop = True
                        break
                acc += ws(y)
                if acc > window:
                    stop = True
                    break
                k -= 1
            if not stop:
                for p in pred[bi]:
                    stack.append((p, blocks[p][-1], acc, depth + 1, rd, ww, wc))
    return out


def c11_exec_state(insts, blocks, succ):
    def sg(tok):
        r = reg(tok)
        return ["s%d" % i for i in range(r[1], r[2] + 1)] if r and r[0] == "s" else []

    IN = {bi: None for bi in range(len(blocks))}
    IN[0] = (True, frozenset())
    work, at = [0], {}
    while work:
        bi = work.pop(0)
        full, saved = IN[bi][0], dict(IN[bi][1])
        for k in blocks[bi]:
            x = insts[k]
            at[k] = full
            op = x.op
            restored = (lambda tok: bool(sg(tok)) and all(saved.get(n, False) for n in sg(tok)))
            if op.startswith(("s_and_saveexec", "s_andn2_saveexec", "s_or_saveexec", "s_xor_saveexec")):
                for n in sg(x.ops[0]):
                    saved[n] = full
                full = op.startswith("s_or_saveexec") and x.ops[1] == "-1"
            elif x.ops and x.ops[0] == "exec" and op.startswith("s_"):
                if op.startswith("s_or_b64") and len(x.ops) == 3:
                    full = full or restored(x.ops[2] if x.ops[1] == "exec" else x.ops[1])
                elif op.startswith("s_mov_b64"):
                    full = x.ops[1] == "-1" or restored(x.ops[1])
                else:
                    full = False
            elif op == "v_writelane_b32" and len(x.ops) >= 3:
                names, key = sg(x.ops[1]), ("lane", x.ops[0], x.ops[2])
                if names and names[0] in saved:
                    saved[key] = saved[names[0]]
                else:
                    saved.pop(key, None)
            elif op == "v_readlane_b32" and len(x.ops) >= 3:
                key = ("lane", x.ops[1], x.ops[2])
                for n in sg(x.ops[0]):
                    if key in saved:
                        saved[n] = saved[key]
                    else:
                        saved.pop(n, None)
            elif op == "s_mov_b64" and x.ops and sg(x.ops[0]) and x.ops[1] == "exec":
                for n in sg(x.ops[0]):
                    saved[n] = full
            elif op == "s_mov_b64" and x.ops and sg(x.ops[0]) and sg(x.ops[1]):
                for d, s_ in zip(sg(x.ops[0]), sg(x.ops[1])):
                    if s_ in saved:
                        saved[d] = saved[s_]
                    else:
                        saved.pop(d, None)
            elif op.startswith("s_") and x.ops and sg(x.ops[0]) and not op.startswith(
                    ("s_cmp", "s_bitcmp", "s_waitcnt", "s_nop", "s_cbranch", "s_branch", "s_barrier", "s_setprio",
                     "s_store", "s_buffer_store", "s_sendmsg")):
                for n in sg(x.ops[0]):
                    saved.pop(n, None)
            elif op.startswith("v_") and x.ops and sg(x.ops[0]):
                for n in sg(x.ops[0]):
                    saved.pop(n, None)
        out = (full, frozenset(saved.items()))
        for t in succ[bi]:
            old = IN[t]
            if old is None:
                new = out
            else:
                od, nd = dict(old[1]), dict(out[1])
                new = (old[0] and out[0], frozenset((kk, od[kk] and nd[kk]) for kk in set(od) & set(nd)))
            if new != old:
                IN[t] = new
                if t not in work:
                    work.append(t)
    return at


def c12_undrained_at_exit(insts, blocks, succ):
    bad = []
    for x in insts:
        if not (_is_lds_dma(x) or ("atomic" in x.op and _is_vmem(x))):
            continue
        stack, seen, hit = [(x.blk, x.idx + 1)], set(), False
        while stack and not hit:
            bi, k = stack.pop()
            if (bi, k) in seen:
                continue
            seen.add((bi, k))
            drained = False
            while k <= blocks[bi][-1]:
                y = insts[k]
                if y.op == "s_waitcnt" and _waitcnt(y).get("vmcnt") == 0:
                    drained = True
                    break
                if y.op == "s_endpgm":
                    hit = True
                    break
                k += 1
            if not (drained or hit):
                stack += [(t, blocks[t][0]) for t in succ[bi]]
        if hit:
            bad.append(x)
    return bad


def c13_dma_retirement(insts, blocks, succ):
    """forward dataflow; state = {dma idx: (VMEM ops issued after it, s_barriers passed)}; merge = union, min ops,
    max barriers. A DMA still pending when a SECOND barrier after it is reached is a violation."""
    IN = {bi: None for bi in range(len(blocks))}
    IN[0] = {}
    viol, work, it = {}, [0], 0
    while work:
        it += 1
        if it > 400000:
            raise RuntimeError("C13 dataflow: no fixed point")
        bi = work.pop(0)
        pend = dict(IN[bi])
        for k in blocks[bi]:
            x = insts[k]
            if x.op == "s_waitcnt":
                w = _waitcnt(x)
                if "vmcnt" in w:
                    pend = {d: (a, b) for d, (a, b) in pend.items() if not (w["vmcnt"] == 0 or a >= w["vmcnt"])}
                continue
            if x.op == "s_barrier":
                for d, (a, b) in pend.items():
                    if b >= 1:
                        viol.setdefault(d, x.idx)
                pend = {d: (a, b + 1) for d, (a, b) in pend.items()}
                continue
            if _is_vmem(x):
                pend = {d: (a + 1, b) for d, (a, b) in pend.items()}
            if _is_lds_dma(x):
                pend[k] = (0, 0)
        for t in succ[bi]:
            old = IN[t]
            if old is None:
                new = dict(pend)
            else:
                new = dict(old)
                for d, (a, b) in pend.items():
                    new[d] = (min(a, new[d][0]), max(b, new[d][1])) if d in new else (a, b)
            if old is None or new != old:
                IN[t] = new
                if t not in work:
                    work.append(t)
    return viol


# =====================================================================================================================
# kernel-level checks
# =====================================================================================================================
def split_asm(text):
    lines = text.splitlines()
    kern = {}
    for name in list(KERNELS) + [FIN]:
        try:
            st = next(i for i, l in enumerate(lines) if l.startswith(name + ":"))
        except StopIteration:
            continue
        en = next(i for i in range(st, len(lines)) if lines[i].startswith(".Lfunc_end"))
        kern[name] = lines[st:en]
    desc = {}
    for m in re.finditer(r"\.amdhsa_kernel (\S+)\n(.*?)\.end_amdhsa_kernel", text, re.S):
        desc[m.group(1)] = dict(re.findall(r"\.amdhsa_(\w+) (\S+)", m.group(2)))
    return kern, desc


def check_desc(name, desc, rep, err):
    d = desc.get(name, {})
    want = dict(float_round_mode_32="0", float_denorm_mode_32="3", ieee_mode="1", dx10_clamp="1", fp16_overflow="0")
    rep["descriptor"] = {k: d.get(k) for k in want}
    for k, v in want.items():
        if d.get(k) != v:
            err(f"C5 descriptor .amdhsa_{k} = {d.get(k)} (want {v})")


def fma_lanes(x):
    if not FMA_RE.match(x.op):
        return 0
    return 2 if x.op.startswith("v_pk_") else 1


def check_kernel(name, lines, desc):
    label, s0, emit = KERNELS[name]
    rep = dict(kernel=label, errors=[], notes=[])
    err, note = rep["errors"].append, rep["notes"].append
    insts, labels, bstarts = s_parse(lines)
    blocks, succ, pred = s_cfg(insts, labels, bstarts)
    check_desc(name, desc, rep, err)
    # ---- regions, from the CFG (a build may lay the loop out rotated, header block after the body: SDK10):
    # the persistent loop = the natural loop of the first "Loop Header: Depth=1" block; its body is linearized in
    # execution order (reverse post-order of the loop's blocks with the back edges removed, from the header).
    hdr_line = next((i for i, l in enumerate(lines) if "Loop Header" in l and "Depth=1" in l), None)
    if hdr_line is None:
        err("no persistent loop header")
        rep["ok"] = False
        return rep
    hb = insts[labels[lines[hdr_line].split(":")[0]]].blk
    fwd, stack = {hb}, [hb]
    while stack:
        u = stack.pop()
        for v in succ[u]:
            if v not in fwd:
                fwd.add(v)
                stack.append(v)
    tails = [u for u in pred[hb] if u in fwd]
    loop, stack = {hb}, list(tails)
    while stack:
        u = stack.pop()
        if u in loop:
            continue
        loop.add(u)
        stack += [p_ for p_ in pred[u] if p_ not in loop]
    order, seen = [], set()

    def dfs(u):
        seen.add(u)
        for v in succ[u]:
            if v in loop and v != hb and v not in seen:
                dfs(v)
        order.append(u)
    dfs(hb)
    lin_blocks = order[::-1]
    body = [k for bi in lin_blocks for k in blocks[bi]]
    pos = {k: i for i, k in enumerate(body)}
    # prologue = blocks on a path entry -> loop header outside the loop (the loop guard's skip edge makes the exit
    # blocks reachable from the entry too; they cannot reach the header)
    ent, stack = set(), [0]
    while stack:
        u = stack.pop()
        if u in ent or u in loop:
            continue
        ent.add(u)
        stack += succ[u]
    to_hdr, stack = set(), [p_ for p_ in pred[hb] if p_ not in loop]
    while stack:
        u = stack.pop()
        if u in to_hdr or u in loop:
            continue
        to_hdr.add(u)
        stack += pred[u]
    pro_blocks = ent & to_hdr
    prologue = [x.idx for x in insts if x.blk in pro_blocks]
    h_idx = blocks[hb][0]
    rep["loop_blocks"] = len(loop)
    rep["loop_rotated"] = any(blocks[bi][0] < h_idx for bi in loop)
    bars = [k for k in body if insts[k].op == "s_barrier"]
    rep["loop_barriers"] = len(bars)
    if len(bars) != 2:
        err(f"expected 2 s_barrier in the loop body (top, mid), found {len(bars)}")
        rep["ok"] = False
        return rep
    top_bar, mid_bar = bars
    mf = [k for k in body if insts[k].op.startswith("v_mfma")]
    all_mf = [x.idx for x in insts if x.op.startswith("v_mfma")]
    dmas = [x.idx for x in insts if _is_lds_dma(x)]
    stores = [k for k in body if pos[k] > pos[mid_bar] and insts[k].op.startswith("buffer_store")]
    rep["mfma_in_body"] = len(mf)
    rep["lds_dma"] = len(dmas)
    mf_info = {}  # MFMA instruction index -> (chain link k, column block n, row group m), filled by C3, used by C17
    # consumer slots: (mid_bar, store0], (store0, store1], ...
    if len(stores) != 4:
        err(f"C7 expected 4 dh1 buffer_store_dwordx4 after the mid barrier in the loop body, found {len(stores)}")
    slots = []
    lo = pos[mid_bar]
    for st in stores:
        slots.append(body[lo + 1:pos[st] + 1])
        lo = pos[st]
    # ---- C1 / C2 / C3 / C4 (compute instances)
    if s0:
        if all_mf or dmas:
            err(f"C1 S0 instance has {len(all_mf)} MFMA / {len(dmas)} LDS-DMA instructions (want 0 / 0)")
    else:
        if len(all_mf) != 32 or len(mf) != 32:
            err(f"C1 expected 32 MFMAs, all in the loop body: {len(all_mf)} total, {len(mf)} in the body")
        bad_form = [insts[k] for k in all_mf if insts[k].op != MF or insts[k].mods != ["blgp:1"]]
        for x in bad_form[:4]:
            err(f"C1/C2 MFMA form/modifiers at line {x.line + 1}: {x.text}")
        # chains
        chains, open_by_dst = [], {}
        for k in mf:
            x = insts[k]
            dst, c2 = reg(x.ops[0]), x.ops[3]
            if c2 == "0":
                chains.append([k])
                open_by_dst[dst] = len(chains) - 1
                continue
            ci = open_by_dst.get(reg(c2))
            if ci is None:
                err(f"C1 MFMA at line {x.line + 1} reads srcC {c2} no open chain produced")
                continue
            prev = chains[ci][-1]
            dregs = set(regs_of(reg(insts[prev].ops[0])))
            for j in body[pos[prev] + 1:pos[k]]:  # no other writer of the accumulator between the links
                if j in all_mf:
                    continue
                if set(_vwrites(insts[j])) & dregs:
                    err(f"C1 accumulator {c2} rewritten at line {insts[j].line + 1} inside a chain: {insts[j].text}")
            chains[ci].append(k)
            del open_by_dst[reg(c2)]
            open_by_dst[dst] = ci
        lens = sorted(len(c) for c in chains)
        rep["chains"] = dict(n=len(chains), lengths=sorted(set(lens)))
        if len(chains) != 8 or set(lens) != {4}:
            err(f"C1 expected 8 chains of exactly 4 links, got {len(chains)} with lengths {lens}")
        # C3: provenance by last writer in program order (W registers are never written in the loop: checked)
        last_writer, prov = {}, {}
        for x in [insts[k] for k in prologue] + [insts[k] for k in body]:
            if x.op.startswith("v_mfma"):
                prov[x.idx] = {rr: last_writer.get(rr) for t in x.ops[1:3] for rr in regs_of(reg(t))}
            for rr in _vwrites(x):
                last_writer[rr] = x.idx
        wset, nm_pairs, abases, wfrag = set(), set(), set(), {}
        for ci, c in enumerate(chains):
            ns, ms = set(), set()
            for lk, k in enumerate(c):
                x = insts[k]
                wr, ar = reg(x.ops[1]), reg(x.ops[2])
                n_k = m_k = None
                # src0 = resident W octet: two prologue buffer_load_dwordx4 quads at voff + 2048n + 128k (+64)
                quads = []
                for q in range(2):
                    qr = ("v", wr[1] + 4 * q, wr[1] + 4 * q + 3)
                    ws = {prov[k].get(rr) for rr in regs_of(qr)}
                    if len(ws) != 1 or None in ws:
                        err(f"C3 MFMA line {x.line + 1}: src0 quad {q} has no single writer")
                        quads = None
                        break
                    wk = ws.pop()
                    y = insts[wk]
                    mo = re.search(r"offset:(\d+)", y.text)
                    if not (y.op == "buffer_load_dwordx4" and "lds" not in y.mods and y.blk in pro_blocks and reg(y.ops[0]) == qr
                            and y.ops[2] != "" and y.ops[3] == "0"):
                        err(f"C3 MFMA line {x.line + 1}: src0 quad {q} not a prologue buffer_load_dwordx4 "
                            f"(soffset 0): {y.text}")
                        quads = None
                        break
                    quads.append((int(mo.group(1)) if mo else 0, y.ops[1], y.ops[2]))
                if quads:
                    (o0, v0, r0), (o1, v1, r1) = quads
                    if not (o1 == o0 + 64 and v0 == v1 and r0 == r1 and (o0 % 2048) == 128 * lk):
                        err(f"C3 MFMA line {x.line + 1} (chain link {lk}): W quads at offsets {o0}/{o1} (want "
                            f"2048n + {128 * lk} and +64 from one voffset/resource)")
                    ns.add(o0 // 2048)
                    n_k = o0 // 2048
                    wset.update(regs_of(wr))
                    wfrag[(o0 // 2048, lk)] = wfrag.get((o0 // 2048, lk), set()) | {wr}
                # src1 = A fragment: two inline-asm ds_read_b128 at k*8192 + m*2048 from the two fragment bases
                aq = []
                for q in range(2):
                    qr = ("v", ar[1] + 4 * q, ar[1] + 4 * q + 3)
                    ws = {prov[k].get(rr) for rr in regs_of(qr)}
                    if len(ws) != 1 or None in ws:
                        err(f"C3 MFMA line {x.line + 1}: src1 quad {q} has no single writer")
                        aq = None
                        break
                    y = insts[ws.pop()]
                    mo = re.search(r"offset:(0x[0-9a-fA-F]+|\d+)", y.text)
                    if not (y.op == "ds_read_b128" and y.asm and reg(y.ops[0]) == qr):
                        err(f"C3 MFMA line {x.line + 1}: src1 quad {q} not an inline-asm ds_read_b128: {y.text}")
                        aq = None
                        break
                    aq.append((int(mo.group(1), 0) if mo else 0, y.ops[1]))
                if aq:
                    (o0, b0), (o1, b1) = aq
                    if not (o0 == o1 and b0 != b1 and (o0 // 8192) == lk and o0 % 2048 == 0 and o0 < 32768):
                        err(f"C3 MFMA line {x.line + 1} (chain link {lk}): A quads at {o0}/{o1} from {b0}/{b1} "
                            f"(want k*8192 + m*2048 with k = {lk}, two bases)")
                    ms.add((o0 % 8192) // 2048)
                    m_k = (o0 % 8192) // 2048
                    abases.add((b0, b1))
                if n_k is not None and m_k is not None:
                    mf_info[k] = (lk, n_k, m_k)
            if len(ns) != 1 or len(ms) != 1:
                err(f"C3 chain {ci}: column block {sorted(ns)} / row group {sorted(ms)} not constant along the chain")
            else:
                nm_pairs.add((ns.pop(), ms.pop()))
        rep["C3"] = dict(nm_pairs=sorted(nm_pairs), fragment_base_pairs=sorted(abases), w_registers=len(wset))
        if nm_pairs != {(n, m) for n in range(2) for m in range(4)}:
            err(f"C3 (column block, row group) pairs {sorted(nm_pairs)} != all 8")
        if len(abases) != 1:
            err(f"C3 A fragments read from {len(abases)} base-register pairs (want 1)")
        if len(wset) != 64 or any(len(v) != 1 for v in wfrag.values()) or len(wfrag) != 8:
            err(f"C3 resident W: {len(wset)} registers, {len(wfrag)} (n, k) fragments (want 64 / 8 distinct)")
        wr_in_loop = [insts[k] for k in body if set(_vwrites(insts[k])) & wset]
        for x in wr_in_loop[:3]:
            err(f"C3 resident W register overwritten inside the loop at line {x.line + 1}: {x.text}")
        # C4 EPI2 dataflow: acc float -> mul by S -> add 0 -> cvt
        s_regs, bad_epi = set(), 0
        for c in chains:
            k_last = c[-1]
            for rr in regs_of(reg(insts[k_last].ops[0])):
                cur, stage, j = rr, 0, pos[k_last] + 1
                ok = False
                while j < pos[mid_bar]:
                    y = insts[body[j]]
                    if cur in _vreads(y):
                        if stage == 0:
                            if y.op not in ("v_mul_f32_e32", "v_mul_f32_e64", "v_pk_mul_f32"):
                                break
                            # the multiplier = the source operand that does not carry the tracked value
                            others = [r for t in y.ops[1:3] if cur not in _vregs(t) for r in _vregs(t)]
                            s_regs.update(others)
                            cur = _out_reg(y, cur)
                            stage = 1
                        elif stage == 1:
                            if not (y.op in ("v_add_f32_e32", "v_add_f32_e64", "v_pk_add_f32") and "0" in y.ops[1:3]):
                                break
                            cur = _out_reg(y, cur)
                            stage = 2
                        else:
                            ok = y.op == "v_cvt_pk_bf16_f32"
                            break
                    elif cur in _vwrites(y):
                        break
                    j += 1
                bad_epi += 0 if ok else 1
        rep["C4_epi2"] = dict(acc_floats=32, bad=bad_epi, scale_regs=sorted("%s%d" % r for r in s_regs))
        if bad_epi:
            err(f"C4 {bad_epi} of 32 accumulator floats do not flow acc*S -> +0 -> v_cvt_pk_bf16_f32")
        # S = prologue v_mul_f32 (sa*sb), possibly through v_mov copies
        s_src = set()
        pro_insts = [insts[k] for k in prologue]
        for r in s_regs:
            if any(r in _vwrites(insts[k]) for k in body):
                s_src.add("written inside the loop")
                continue
            w = None
            for x in pro_insts:
                if r in _vwrites(x):
                    w = x
            hops = 0
            while w is not None and w.op.startswith("v_mov") and hops < 4:
                src = _vregs(w.ops[1])
                if not src:
                    break
                w2 = None
                for x in pro_insts:
                    if x.idx < w.idx and src[0] in _vwrites(x):
                        w2 = x
                w, hops = w2, hops + 1
            s_src.add(w.text if w is not None else None)
        rep["C4_scale_writer"] = sorted(map(str, s_src))
        if len(s_src) != 1 or not all(t and t.startswith("v_mul_f32") for t in s_src):
            err(f"C4 epilogue multiplier is not the single prologue v_mul_f32 (sa*sb): {sorted(map(str, s_src))}")
        fma_mfma = sum(fma_lanes(insts[k]) for k in body[pos[mf[0]]:pos[mid_bar]]) if mf else 0
        if fma_mfma:
            err(f"C4 {fma_mfma} FMA lanes between the first MFMA and the mid barrier (EPI2 must not contract)")
    # ---- C4 conversions (all instances)
    cvt_body = sum(1 for k in body if insts[k].op == "v_cvt_pk_bf16_f32")
    want_cvt = 16 if s0 else 32
    rep["cvt_pk_bf16_in_body"] = cvt_body
    if cvt_body != want_cvt:
        err(f"C4 {cvt_body} v_cvt_pk_bf16_f32 in the loop body (want {want_cvt})")
    rne = [x for x in insts if x.op.startswith("v_cmp_u_f32") or
           re.search(r"(?<![0-9a-fx])0x7fff(?![0-9a-f])|(?<![0-9a-fx])0x400000(?![0-9a-f])", x.text) or
           (x.op == "v_bfe_u32" and x.ops[-2:] == ["16", "1"])]
    if rne:
        err(f"C4 software-RNE-like instructions: {[(x.line + 1, x.text) for x in rne[:4]]}")
    # ---- C6 waits
    waits = []
    for x in insts:
        if x.op != "s_waitcnt" or "vmcnt" not in _waitcnt(x):
            continue
        k = x.idx
        if x.blk in pro_blocks:
            cls = "prologue"
        elif x.blk not in loop:
            cls = "exit"
        elif mf and pos[mf[0]] <= pos[k] <= pos[mf[-1]]:
            cls = "INSIDE-MFMA-PHASE"
        elif pos[k] < pos[top_bar] and all(insts[j].op == "s_nop" for j in body[pos[k] + 1:pos[top_bar]]):
            cls = "top"
        elif pos[k] > pos[mid_bar]:
            cls = "consumer"
        else:
            cls = "UNEXPECTED"
        waits.append(dict(line=x.line + 1, text=x.text, cls=cls, asm=x.asm))
    rep["vmcnt_waits"] = waits
    for w in waits:
        if w["cls"] in ("UNEXPECTED", "INSIDE-MFMA-PHASE"):
            err(f"C6 vmcnt wait at line {w['line']} ({w['text']}): {w['cls']}")
    tops = [w for w in waits if w["cls"] == "top"]
    if not s0:
        want = TOP_VMCNT[(s0, emit)]
        if len(tops) != 1 or f"vmcnt({want})" not in tops[0]["text"] or "lgkmcnt(0)" not in tops[0]["text"]:
            err(f"C6 top-of-loop wait {[w['text'] for w in tops]} (want one `s_waitcnt vmcnt({want}) lgkmcnt(0)` "
                f"immediately before the top s_barrier)")
    # ---- C7 consumer census
    cons = []
    for si, sl in enumerate(slots):
        c = dict(exp=0, rcp=0, swz=0, dpp=0, cvt=0, fma_lanes=0, inv_d=0, readlane=0)
        dpp_ctrl = []
        for k in sl:
            x = insts[k]
            if x.op.startswith("v_exp_f32"):
                c["exp"] += 1
            elif x.op.startswith("v_rcp_f32"):
                c["rcp"] += 1
            elif x.op == "ds_swizzle_b32":
                c["swz"] += 1
                if "swizzle(SWAP,16)" not in x.text:
                    err(f"C7 ds_swizzle at line {x.line + 1} is not SWAP16: {x.text}")
            elif "_dpp" in x.op:
                c["dpp"] += 1
                dpp_ctrl.append(" ".join(m for m in x.mods if m.startswith(("quad_perm", "row_")) and
                                         not m.startswith(("row_mask", "row_bcast"))))
                # the move+add of a butterfly step: folded (v_add_f32_dpp) or a DPP move feeding an IEEE add
                if not x.op.startswith(("v_add_f32_dpp", "v_mov_b32_dpp")) or not all(
                        m in x.mods for m in ("row_mask:0xf", "bank_mask:0xf")):
                    err(f"C7 DPP op at line {x.line + 1}: {x.text}")
            elif x.op == "v_cvt_pk_bf16_f32":
                c["cvt"] += 1
            if x.op.startswith("v_readlane"):
                c["readlane"] += 1
            if "0x3b800000" in x.text:
                c["inv_d"] += 1
            c["fma_lanes"] += fma_lanes(x)
            if FMA_BAD.match(x.op):
                err(f"C7 non-contract fused/mixed op in the consumer at line {x.line + 1}: {x.text}")
        c["dpp_ctrl"] = dpp_ctrl
        cons.append(c)
        want = dict(exp=8, rcp=8, swz=2, dpp=8, cvt=4, fma_lanes=56, inv_d=2)
        for kk, v in want.items():
            if c[kk] != v:
                err(f"C7 consumer slot {si}: {kk} = {c[kk]} (want {v})")
        # the 8 DPP steps must split into two subsequences, each exactly distances 1, 2, 4, 8 in order [CS 3.3]
        seq = ["quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]", "row_half_mirror", "row_mirror"]
        nxt, okd = [0, 0], len(dpp_ctrl) == 8
        for ctl in dpp_ctrl:
            for j in range(2):
                if nxt[j] < 4 and seq[nxt[j]] == ctl:
                    nxt[j] += 1
                    break
            else:
                okd = False
        if not okd or nxt != [4, 4]:
            err(f"C7 consumer slot {si}: DPP controls {dpp_ctrl} are not two in-order 1,2,4,8 trees")
    rep["C7_consumer_slots"] = cons
    fma_outside = sum(fma_lanes(insts[k]) for k in range(len(insts)) if not any(k in sl for sl in slots))
    rep["fma_lanes_outside_consumer"] = fma_outside
    if fma_outside:
        err(f"C7 {fma_outside} FMA lanes outside the consumer slots")
    if not any("0xbfb8aa3b" in insts[k].text for k in prologue):
        err("C7 the -log2(e) constant 0xbfb8aa3b is not in the prologue")
    # ---- C8-C12 (X2H), C13
    viol = c8_wait_coverage(insts, blocks, succ)
    asm_loads = [x for x in insts if x.asm and _load_dst(x)]
    rep["C8_wait_coverage"] = dict(loads=sum(1 for x in insts if _load_dst(x)), asm_loads=len(asm_loads),
                                   violations=len(viol))
    for (li, k, kind), rr in sorted(viol.items(), key=lambda t: (insts[t[0][1]].line, t[0][0]))[:12]:
        err(f"C8 {kind}: {'asm ' if insts[li].asm else ''}load at line {insts[li].line + 1} `{insts[li].text}` not "
            f"retired before line {insts[k].line + 1} `{insts[k].text}` (regs {['%s%d' % r for r in rr]})")
    bars_all = []
    for x in insts:
        if x.op != "s_barrier":
            continue
        k = x.idx - 1
        while k >= 0 and insts[k].op == "s_nop":
            k -= 1
        w = _waitcnt(insts[k]) if k >= 0 and insts[k].op == "s_waitcnt" else {}
        kind = ("vm%d+lgkm0" % w["vmcnt"] if "vmcnt" in w and w.get("lgkmcnt") == 0 else
                "lgkm0" if w.get("lgkmcnt") == 0 else "NO lgkmcnt(0)")
        bars_all.append(kind)
        if kind == "NO lgkmcnt(0)":
            err(f"C9 s_barrier at line {x.line + 1} is not immediately preceded by s_waitcnt ... lgkmcnt(0)")
    rep["C9_barriers"] = bars_all
    hz = c10_asm_mfma_hazards(insts, blocks, pred)
    rep["C10_asm_mfma_hazards"] = len(hz)
    for h in hz[:6]:
        err(f"C10 {h['kind']}: MFMA at line {h['mfma_line'] + 1} {h['ws']} wait states before asm line "
            f"{h['asm_line'] + 1} `{h['asm']}`")
    ex = c11_exec_state(insts, blocks, succ)
    part = []
    for x in insts:
        if not x.inloop:
            continue
        r0 = reg(x.ops[0]) if x.ops else None
        sens = (x.op.startswith(("v_mfma", "v_permlane", "s_barrier", "v_readfirstlane", "v_mbcnt", "ds_swizzle",
                                 "ds_bpermute")) or "_dpp" in x.op or
                (x.op.startswith("v_cmp") and r0 is not None and r0[0] == "s" and r0[2] - r0[1] == 1))
        if sens and not ex.get(x.idx, True):
            part.append(x)
    rep["C11_partial_exec_in_loop"] = len(part)
    for x in part[:6]:
        err(f"C11 EXEC-sensitive instruction under partial EXEC at line {x.line + 1}: `{x.text}`")
    und = c12_undrained_at_exit(insts, blocks, succ)
    rep["C12_undrained_at_exit"] = len(und)
    for x in und[:4]:
        err(f"C12 LDS-DMA at line {x.line + 1} `{x.text}` can reach s_endpgm without vmcnt(0)")
    v13 = c13_dma_retirement(insts, blocks, succ)
    rep["C13_dma_not_retired_by_2nd_barrier"] = len(v13)
    for d, b in sorted(v13.items())[:6]:
        err(f"C13 LDS-DMA at line {insts[d].line + 1} `{insts[d].text}` still pending at the 2nd barrier after it "
            f"(line {insts[b].line + 1})")
    # ---- C18 (fix pass): LDS-DMA census and the hand-written-wait rule (race review F4 optional, F6)
    dma_body = [k for k in body if _is_lds_dma(insts[k])]
    dma_pro = [k for k in prologue if _is_lds_dma(insts[k])]
    want_dma = (0, 0) if s0 else (4, 4)
    rep["C18_lds_dma"] = dict(prologue=len(dma_pro), loop_body=len(dma_body), total=len(dmas))
    if (len(dma_pro), len(dma_body)) != want_dma or len(dmas) != sum(want_dma):
        err(f"C18 LDS-DMA count: prologue {len(dma_pro)}, loop body {len(dma_body)}, total {len(dmas)} "
            f"(want {want_dma[0]} + {want_dma[1]})")
    asm_waits = [x for x in insts if x.asm and x.op == "s_waitcnt"]
    bad_aw = [x for x in asm_waits if _waitcnt(x) != {"lgkmcnt": 0}]
    rep["C18_asm_waits"] = dict(n=len(asm_waits), not_lgkmcnt0=len(bad_aw))
    for x in bad_aw[:4]:
        err(f"C18 hand-written wait at line {x.line + 1} `{x.text}` is not exactly `s_waitcnt lgkmcnt(0)` (SMEM may "
            f"be in flight: an lgkmcnt(N > 0) under-waits; x2lb_pass.cu rule at the prologue s_waitcnt, cu:420-424)")
    # ---- W1 (fix pass, race review F5; warning, not a gate): MFMA operand-overlap census
    ov = dict(dst_srcA=[], dst_srcB=[], dst_partial_srcC=[])
    for k in all_mf:
        y = insts[k]
        d, a_, b_ = set(regs_of(reg(y.ops[0]))), set(regs_of(reg(y.ops[1]))), set(regs_of(reg(y.ops[2])))
        c_ = set(regs_of(reg(y.ops[3]))) if len(y.ops) > 3 and reg(y.ops[3]) else set()
        if d & a_:
            ov["dst_srcA"].append(y.line + 1)
        if d & b_:
            ov["dst_srcB"].append(y.line + 1)
        if c_ and d != c_ and d & c_:
            ov["dst_partial_srcC"].append(y.line + 1)
    rep["W1_mfma_overlap"] = {k: len(v) for k, v in ov.items()}
    if ov["dst_partial_srcC"]:
        note(f"W1 WARNING: {len(ov['dst_partial_srcC'])} of {len(all_mf)} MFMAs have a vDst that PARTIALLY overlaps "
             f"their srcC (lines {ov['dst_partial_srcC'][:4]}): no GPU-verified L11 build has this pattern (race "
             f"review F5). If the lab's container build shows it, G2 must run that very build.")
    # ---- C15 (fix pass, contract review F1): symbolic dataflow D1-D3 of the compiled ISA (isa_dataflow.py)
    import isa_dataflow as DF  # noqa: E402  (lazy: it imports this module's parser)
    try:
        df = DF.check(name, lines)
    except Exception as ex:  # fail closed
        df = dict(errors=[f"dataflow checker raised {type(ex).__name__}: {ex}"], info=[])
    rep["C15_dataflow"] = dict(errors=len(df["errors"]), info=df["info"])
    for e in df["errors"][:12]:
        err("C15 " + e)
    # ---- C16 / C17 (fix pass, race review F1, contract review F3): concrete multi-wave execution (isa_exec.py)
    import isa_exec as EX  # noqa: E402
    try:
        rep["C16_C17"] = EX.check(label, s0, emit, insts, labels, blocks, loop, blocks[hb][0], mf_info, err, note)
    except Exception as ex:  # fail closed
        err(f"C16/C17 executor raised {type(ex).__name__}: {ex}")
    # ---- counts (informational)
    rep["counts_loop_body"] = _census([insts[k] for k in body])
    rep["linearized_loop"] = dict(blocks=len(lin_blocks), instructions=len(body), rotated=rep["loop_rotated"])
    rep["cfg_blocks"] = len(blocks)
    rep["ok"] = not rep["errors"]
    return rep


def _out_reg(y, cur):
    """the destination 32-bit register that carries `cur`'s lane through an elementwise (packed or scalar) op"""
    d = _vregs(y.ops[0])
    if y.op.startswith("v_pk_"):
        # packed: element e of the source pair goes to element e of the destination pair; op_sel_hi:[..,0] on a
        # broadcast operand does not move the tracked lane (the tracked value is never the broadcast scalar here)
        for t in y.ops[1:3]:
            src = _vregs(t)
            if cur in src:
                return d[src.index(cur)] if len(src) == len(d) else d[0]
    return d[0]


def _census(xs):
    c = {}
    for x in xs:
        if x.op.startswith("v_mfma"):
            k = "mfma"
        elif x.op.startswith("ds_read"):
            k = "ds_read"
        elif x.op.startswith("ds_write"):
            k = "ds_write"
        elif x.op.startswith("ds_"):
            k = "ds_other"
        elif _is_lds_dma(x):
            k = "lds_dma"
        elif x.op.startswith(("buffer_load", "global_load")):
            k = "vmem_load"
        elif x.op.startswith(("buffer_store", "global_store")):
            k = "vmem_store"
        elif x.op in ("s_waitcnt", "s_barrier", "s_nop"):
            k = x.op
        elif x.op.startswith("v_exp") or x.op.startswith("v_rcp"):
            k = "transcendental"
        elif x.op.startswith("v_"):
            k = "valu"
        elif x.op.startswith("s_"):
            k = "salu"
        else:
            k = "other"
        c[k] = c.get(k, 0) + 1
    c["total"] = len(xs)
    return c


def check_fin(lines, desc):
    rep = dict(kernel="x2lb_fin", errors=[], notes=[])
    insts, _, _ = s_parse(lines)
    check_desc(FIN, desc, rep, rep["errors"].append)
    f = [x.text for x in insts if fma_lanes(x) or FMA_BAD.match(x.op)]
    if f:
        rep["errors"].append(f"C14 FMA/MAC in x2lb_fin: {f[:3]}")
    rep["adds"] = sum(1 for x in insts if x.op.startswith(("v_add_f32", "v_pk_add_f32")))
    rep["global_loads"] = sum(1 for x in insts if x.op.startswith(("global_load", "buffer_load")))
    if any(x.op.startswith("v_pk_add_f32") for x in insts):
        rep["errors"].append("C14 packed add in x2lb_fin (the G-sum must be one sequential scalar chain)")
    if rep["adds"] < 32 or rep["global_loads"] < 32:
        rep["errors"].append(f"C14 x2lb_fin: {rep['adds']} adds / {rep['global_loads']} loads (want the 32-deep batch)")
    rep["ok"] = not rep["errors"]
    return rep


def main(argv):
    paths, jpath = [], None
    it = iter(argv)
    for a in it:
        if a == "--json":
            jpath = next(it)
        else:
            paths.append(a)
    if not paths:
        paths = [os.path.join(HERE, "out", c, "x2lb_pass.s") for c in ("r720", "r720_ilp", "sdk10", "sdk10_ilp")]
    gate_cfgs = ("r720", "r720_ilp", "sdk10")
    if argv and not any(os.path.basename(os.path.dirname(p_)) in gate_cfgs + ("sdk10_ilp",) for p_ in paths):
        gate_cfgs = tuple(os.path.basename(os.path.dirname(p_)) for p_ in paths)  # explicit files (lab): all gated
    allrep, ok = {}, True
    for p in paths:
        cfg = os.path.basename(os.path.dirname(p))
        if not os.path.exists(p):
            print(f"== {p}: MISSING (build failed?) -> {'FAIL' if cfg in gate_cfgs else 'skip (informational)'}")
            ok &= cfg not in gate_cfgs
            allrep[cfg] = dict(missing=p)
            continue
        kern, desc = split_asm(open(p).read())
        allrep[cfg] = {}
        for name in list(KERNELS) + [FIN]:
            if name not in kern:
                print(f"== {cfg} {name}: kernel not found -> FAIL")
                ok &= cfg not in gate_cfgs
                continue
            r = check_fin(kern[name], desc) if name == FIN else check_kernel(name, kern[name], desc)
            allrep[cfg][r["kernel"]] = r
            if cfg in gate_cfgs:
                ok &= r["ok"]
            tag = "" if cfg in gate_cfgs else " (informational)"
            print(f"== {cfg} {r['kernel']}: {'PASS' if r['ok'] else 'FAIL'}{tag}")
            if name != FIN:
                print(f"   mfma {r.get('mfma_in_body')} chains {r.get('chains')} lds_dma {r.get('lds_dma')} "
                      f"cvt_pk_bf16/body {r.get('cvt_pk_bf16_in_body')} desc {r.get('descriptor')}")
                if "C3" in r:
                    print(f"   C3 (n, m) {r['C3']['nm_pairs']} A bases {r['C3']['fragment_base_pairs']} W regs "
                          f"{r['C3']['w_registers']}; C4 EPI2 bad {r['C4_epi2']['bad']}/32 scale {r['C4_scale_writer']}")
                print("   vmcnt waits: " + ", ".join(f"{w['line']}:{w['cls']}:{w['text'][10:]}" for w in r.get("vmcnt_waits", [])))
                cs = r.get("C7_consumer_slots", [])
                print("   C7 slots: " + "; ".join(f"exp {c['exp']} rcp {c['rcp']} swz {c['swz']} dpp {c['dpp']} cvt "
                                               f"{c['cvt']} fma {c['fma_lanes']} 2^-8 {c['inv_d']} rl {c['readlane']}"
                                               for c in cs) + f"; FMA lanes outside {r.get('fma_lanes_outside_consumer')}")
                c8 = r.get("C8_wait_coverage", {})
                print(f"   C8 loads {c8.get('loads')} (asm {c8.get('asm_loads')}) violations {c8.get('violations')}  "
                      f"C9 {r.get('C9_barriers')}  C10 {r.get('C10_asm_mfma_hazards')}  C11 "
                      f"{r.get('C11_partial_exec_in_loop')}  C12 {r.get('C12_undrained_at_exit')}  C13 "
                      f"{r.get('C13_dma_not_retired_by_2nd_barrier')}  blocks {r.get('cfg_blocks')}")
                c15 = r.get("C15_dataflow", {})
                print(f"   C15 dataflow errors {c15.get('errors')}; " + "; ".join(i for i in c15.get("info", [])
                                                                          if i.startswith(("S =", "A fragment",
                                                                                           "EMIT"))))
                print(f"   C16/C17 executed: {r.get('C16_C17')}")
                print(f"   C18 LDS-DMA {r.get('C18_lds_dma')} asm waits {r.get('C18_asm_waits')}  W1 MFMA overlaps "
                      f"{r.get('W1_mfma_overlap')}")
                print(f"   loop body census: {r.get('counts_loop_body')}")
            for e in r["errors"]:
                print("   ERROR", e)
            for n in r["notes"]:
                print("   note", n)
    if jpath:
        with open(jpath, "w") as f:
            json.dump(allrep, f, indent=1)
    print(f"X2L-B ISA CHECK ({', '.join(gate_cfgs)}):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
