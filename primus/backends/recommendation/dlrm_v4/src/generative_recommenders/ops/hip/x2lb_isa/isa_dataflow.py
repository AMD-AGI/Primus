#!/usr/bin/env python3
"""isa_dataflow.py -- isa_check.py clause C15 (fix pass 2026-10-12, contract review F1 MAJOR): symbolic dataflow check of
X2L-B's compiled ISA against CONTRACT_dgrad 6.3 and CONTRACT_swish 2-3. CPU only, stdlib.

PROVENANCE: adopted from the contract reviewer's probes/iter70/x2lb/review_contract/dfcheck.py (sha256 recorded in
out/sha256.txt), checks D1-D3 unchanged; changes: module paths relative to this directory and main() defaults to this build's
outputs. It reuses only the PARSER / CFG helpers of isa_check.py
(s_parse, s_cfg, reg, split_asm); every semantic check here is independent of C1-C14.

For every x2lb_pass instance it symbolically executes the prologue (layout order) and ONE linearized iteration of the
persistent loop (hash-consed expression DAG over every VGPR / SGPR), then requires:
  D1  (S1, S1+DA1) every R-buffer ds_write_b128 of the MFMA phase writes, per bf16 half, cvt(add(mul(S, ACC), +0)) with
      S = the prologue product of the two scale loads and ACC a 4-link MFMA chain (inline-0 head, srcC = previous link,
      modifiers exactly blgp:1, the same D item i along the chain) whose link k reads W octet (n, k) (two prologue
      buffer_load_dwordx4 quads at imm 2048n + 128k and +64, lo quad first, one voffset / resource) as src0 and the A
      fragment (two inline-asm ds_read_b128 quads at k*8192 + m*2048, lo quad from base L0, hi from base L1, the SAME
      (L0, L1) for every link of every chain) as src1; item i / column block n / row group m consistent with the word
      position (word wd = cvt(acc[n][2p], acc[n][2p+1]), n = wd // 2). The S1+DA1 stores must store R quads.
  D2  every consumer dh1 buffer_store_dwordx4 (slot u) stores, per dword i, cvt(dx_2i, dx_2i+1) where dx is EXACTLY the
      PR #7 expression of CONTRACT_swish 2 (E1-E16, R1-R4, hsum8 pairing, DPP 1,2,4,8 then ds_swizzle SWAP16), with the
      leaves bound to: x = bf16 halves of dword i of ONE loop-carried quad that this slot's own prefetch reloads;
      dy = bf16 halves of dword i of the slot's R ds_read at u*0x2100 (S1) / of one loop-carried quad the slot reloads
      (S0); mean / rstd = v_cndmask(vcc, readlane(R, 16u+2w+1), readlane(R, 16u+2w)); w / w2 / b2 = the prologue nw /
      nb loads of column c0 + j (w2, b2 = RN(. * 0xbfb8aa3b)); 1/D = 0x3b800000.
  D3  each of the 24 accumulators (dW, dB, dXS x 8 columns) ends the iteration in ONE register r holding the in-order
      fold of the 4 slots (fma(t, xhat, acc) / acc + t / acc + dx) started from that register's own loop-carried value.
Canonicalization: add / mul operands and fma multiplicands are unordered (IEEE commutative bit for bit); a negation of a
multiplicand is pulled out (RN is sign-symmetric); x + (-0.0) is x (exact for every non-signaling x; SDK10 pads a packed
add with -0.0). Nothing else is normalized: a reassociated sum, a contracted mul+add, an extra or missing rounding, a
different constant or a different input binding is a mismatch. The concrete addresses behind the bindings (which
tensor row / column a load, R access or readlane lane reaches) are clause C17 (isa_exec.py).
Usage: python -B isa_dataflow.py [file.s ...]   (default: the build's 4 configs)
"""
import os
import re
import sys

sys.dont_write_bytecode = True
BUILD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BUILD)
import isa_check as IC  # noqa: E402  (parser only)

NL2E = 0xBFB8AA3B
INV_D = 0x3B800000
KERNELS = dict(IC.KERNELS)

# ----------------------------------------------------------------------------------------------------------------------
# hash-consed expression nodes: an expression is an int id; NODES[id] is a shallow tuple (op, fields / child ids)
# ----------------------------------------------------------------------------------------------------------------------
NODES = []
TABLE = {}
CH = {"neg": (1,), "mul": (1, 2), "add": (1, 2), "fma": (2, 3, 4), "exp2": (1,), "rcp": (1,), "cvt": (1, 2),
      "dpp": (2,), "swz16": (1,), "sel": (1, 2, 3), "bf_lo": (1,), "bf_hi": (1,), "rl": (1,)}


def mk(*node):
    i = TABLE.get(node)
    if i is None:
        i = len(NODES)
        NODES.append(node)
        TABLE[node] = i
    return i


def op_of(e):
    return NODES[e][0]


def K(v):
    return mk("K", v)


def NEG(a):
    n = NODES[a]
    return n[1] if n[0] == "neg" else mk("neg", a)


def strip_neg(a):
    s = 1
    while NODES[a][0] == "neg":
        a, s = NODES[a][1], -s
    return a, s


def MUL(a, b):
    a, sa = strip_neg(a)
    b, sb = strip_neg(b)
    x, y = sorted((a, b))
    r = mk("mul", x, y)
    return r if sa * sb > 0 else mk("neg", r)


NEGZERO_BITS = 0x80000000


def ADD(a, b):
    # x + (-0.0) == x exactly for every non-signaling x (RNE: +0 + -0 = +0, -0 + -0 = -0); compilers may pad packed
    # adds with -0.0 (SDK10 does in the exit tree). +0.0 is NOT an identity (it maps -0 to +0) and is never folded.
    if NODES[b] == ("K", NEGZERO_BITS):
        return a
    if NODES[a] == ("K", NEGZERO_BITS):
        return b
    x, y = sorted((a, b))
    return mk("add", x, y)


def FMA(a, b, c):
    a, sa = strip_neg(a)
    b, sb = strip_neg(b)
    x, y = sorted((a, b))
    return mk("fma", sa * sb, x, y, c)


def EXP(a):
    return mk("exp2", a)


def RCP(a):
    return mk("rcp", a)


def CVT(lo, hi):
    return mk("cvt", lo, hi)


def DPP(ctrl, a):
    return mk("dpp", ctrl, a)


def SWZ(a):
    return mk("swz16", a)


def SEL(c, t, f):
    return mk("sel", c, t, f)


FONE = K(0x3F800000)
FZERO = K(0)


def subst(e, ph, val, memo=None):
    memo = {} if memo is None else memo
    if e == ph:
        return val
    if e in memo:
        return memo[e]
    n = NODES[e]
    if n[0] not in CH:
        memo[e] = e
        return e
    kids = {i: subst(n[i], ph, val, memo) for i in CH[n[0]]}
    if all(kids[i] == n[i] for i in kids):
        memo[e] = e
        return e
    if n[0] == "mul":
        r = MUL(kids[1], kids[2])
    elif n[0] == "add":
        r = ADD(kids[1], kids[2])
    elif n[0] == "fma":
        r = FMA(kids[2] if n[1] > 0 else NEG(kids[2]), kids[3], kids[4])
    elif n[0] == "neg":
        r = NEG(kids[1])
    else:
        r = mk(*[kids.get(i, f) for i, f in enumerate(n)])
    memo[e] = r
    return r


def show(e, depth=6):
    n = NODES[e]
    if depth <= 0:
        return "%s#%d" % (n[0], e)
    parts = []
    for i, f in enumerate(n[1:], 1):
        if n[0] in CH and i in CH[n[0]]:
            parts.append(show(f, depth - 1))
        else:
            parts.append(repr(f) if not isinstance(f, tuple) else "(..%d)" % len(f))
    return "%s(%s)" % (n[0], ", ".join(parts))


# ----------------------------------------------------------------------------------------------------------------------
# operands
# ----------------------------------------------------------------------------------------------------------------------
INLINE_F = {"0": 0, "1.0": 0x3F800000, "-1.0": 0xBF800000, "0.5": 0x3F000000, "-0.5": 0xBF000000, "2.0": 0x40000000,
            "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000}


def parse_lit(tok):
    tok = tok.strip()
    if tok in INLINE_F:
        return INLINE_F[tok]
    try:
        return int(tok, 0) & 0xFFFFFFFF
    except ValueError:
        return None


class Machine:
    def __init__(self):
        self.v, self.s = {}, {}

    def getv(self, i):
        return self.v[i] if i in self.v else mk("IN", "v%d" % i)

    def gets(self, i):
        return self.s[i] if i in self.s else mk("INS", "s%s" % i)

    def operand(self, tok):
        tok = tok.strip()
        neg = False
        if tok.startswith("-") and tok not in INLINE_F and not re.match(r"^-\d", tok):
            neg, tok = True, tok[1:]
        if tok.startswith("|"):
            return mk("UNK", "abs " + tok)
        r = IC.reg(tok)
        if r:
            val = self.getv(r[1]) if r[0] in "va" else self.gets(r[1])
        elif tok in ("vcc", "vcc_lo", "m0"):
            val = self.gets("vcc" if tok.startswith("vcc") else "m0")
        else:
            lit = parse_lit(tok)
            if lit is None:
                return mk("UNK", tok)
            val = K(lit)
        return NEG(val) if neg else val


def dst_regs(tok):
    r = IC.reg(tok)
    return (None, []) if not r else (r[0], list(range(r[1], r[2] + 1)))


def mods_of(x, key):
    for m in x.mods:
        if m.startswith(key + ":["):
            return [int(t) for t in m[len(key) + 2:-1].split(",")]
    return None


def lin(cw, c):
    return mk("lin", cw, c)


def as_lin(e):
    n = NODES[e]
    if n[0] == "lin":
        return n
    if n[0] == "K":
        v = n[1] if n[1] < 0x80000000 else n[1] - (1 << 32)
        return ("lin", 0, v)
    return None


def s_exec(m, x, line):
    op, ops = x.op, x.ops
    if not ops:
        return
    if ops[0] in ("vcc", "m0", "exec"):
        dkey = [ops[0]]
    else:
        d = IC.reg(ops[0])
        if d is None or d[0] != "s":
            return
        dkey = list(range(d[1], d[2] + 1))

    def val(tok):
        r = IC.reg(tok)
        if r and r[0] == "s":
            return m.gets(r[1])
        if tok in ("vcc", "m0"):
            return m.gets(tok)
        lit = parse_lit(tok)
        return K(lit) if lit is not None else mk("UNK", tok)

    if op == "s_mov_b64" and IC.reg(ops[1]) and IC.reg(ops[1])[0] == "s":
        r = IC.reg(ops[1])
        for k, di in enumerate(dkey):
            m.s[di] = m.gets(r[1] + k)
        return
    if op.startswith("s_load_dword"):
        base = IC.reg(ops[1])
        off = parse_lit(ops[2]) or 0
        bv = m.gets(base[1]) if base else mk("UNK", "base")
        for k, di in enumerate(dkey):
            m.s[di] = mk("karg", off + 4 * k) if bv == mk("INS", "s0") else mk("SLD", bv, off + 4 * k)
        return
    res = None
    if op in ("s_mov_b32", "s_movk_i32"):
        res = val(ops[1])
    elif op in ("s_lshl_b32", "s_lshr_b32", "s_or_b32", "s_add_i32", "s_add_u32", "s_sub_i32", "s_andn2_b32"):
        a, b = as_lin(val(ops[1])), as_lin(val(ops[2]))
        if a and b and b[1] == 0:
            if op == "s_lshl_b32":
                res = lin(a[1] << b[2], a[2] << b[2])
            elif op == "s_lshr_b32" and a == ("lin", 64, 0) and b[2] == 6:
                res = lin(1, 0)
            elif op in ("s_add_i32", "s_add_u32"):
                res = lin(a[1], a[2] + b[2])
            elif op == "s_sub_i32":
                res = lin(a[1], a[2] - b[2])
            elif op == "s_or_b32" and a[1] % 2 == 0 and a[2] % 2 == 0 and b[2] == 1:
                res = lin(a[1], a[2] + 1)
            elif op == "s_andn2_b32" and a == ("lin", 64, 0) and b[2] == 63:
                res = lin(64, 0)
        elif a and b and op in ("s_add_i32", "s_add_u32") and a[1] == 0:
            res = lin(b[1], a[2] + b[2])
    if res is None:
        res = mk("S", op, line)
    for k, di in enumerate(dkey):
        m.s[di] = res if len(dkey) == 1 else mk("part", k, res)


def pk_operand(m, tok, sel_bit):
    t = tok.strip()
    neg = False
    if t.startswith("-") and t not in INLINE_F and not re.match(r"^-\d", t):
        neg, t = True, t[1:]
    r = IC.reg(t)
    if r:
        idx = r[1] + sel_bit
        v = m.getv(idx) if r[0] in "va" else m.gets(idx)
    else:
        lit = parse_lit(t)
        if lit is None:
            return mk("UNK", tok)
        v = K(lit) if sel_bit == 0 else mk("KHI", lit)
    return NEG(v) if neg else v


def ctl_of(x):
    return " ".join(mm for mm in x.mods if mm.startswith(("quad_perm", "row_")) and
                    not mm.startswith(("row_mask", "row_bcast")))


def v_exec(m, x, line, events):
    op, ops = x.op, x.ops
    if op.startswith("s_"):
        s_exec(m, x, line)
        return
    if op.startswith("v_readlane_b32"):
        d = IC.reg(ops[0])
        r = IC.reg(ops[2])
        ln = as_lin(m.gets(r[1])) if r else ("lin", 0, parse_lit(ops[2]))
        m.s[d[1]] = mk("rl", m.operand(ops[1]), ln)
        return
    if op.startswith("v_readfirstlane_b32"):
        d = IC.reg(ops[0])
        src = m.operand(ops[1])
        m.s[d[1]] = lin(64, 0) if src == mk("IN", "v0") else mk("rfl", src)
        return
    if op.startswith("v_cmp"):
        val = mk("cmp", op, m.operand(ops[1]), m.operand(ops[2]))
        if ops[0] == "vcc":
            m.s["vcc"] = val
        else:
            r = IC.reg(ops[0])
            if r:
                m.s[r[1]] = mk("cmpm", val)
                m.s[r[1] + 1] = mk("cmph", val)
        return
    if op.startswith("v_mfma"):
        dk, dl = dst_regs(ops[0])
        r0, r1 = IC.reg(ops[1]), IC.reg(ops[2])
        a = tuple(m.getv(r0[1] + i) for i in range(r0[2] - r0[1] + 1))
        b = tuple(m.getv(r1[1] + i) for i in range(r1[2] - r1[1] + 1))
        if ops[3] == "0":
            c = (FZERO,) * 4
        else:
            rc = IC.reg(ops[3])
            c = tuple(m.getv(rc[1] + i) for i in range(4))
        for i, di in enumerate(dl):
            m.v[di] = mk("MF", i, " ".join(x.mods), a, b, c[i])
        return
    if op.startswith(("ds_", "buffer_", "global_")):
        mo = re.search(r"offset:(0x[0-9a-fA-F]+|\d+)", x.text)
        off = int(mo.group(1), 0) if mo else 0
        if op.startswith(("ds_read", "ds_swizzle", "buffer_load", "global_load")):
            if "lds" in x.mods:
                events.append(("dma", line, ops))
                return
            dk, dl = dst_regs(ops[0])
            if op.startswith("ds_swizzle"):
                src = m.operand(ops[1].split()[0])  # "v163 offset:swizzle(SWAP" after the parser's comma split
                # the operand parser splits "swizzle(SWAP,16)" at its comma: match the raw text
                m.v[dl[0]] = SWZ(src) if re.search(r"offset:swizzle\(SWAP,\s*16\)", x.text) else mk("swz?", x.text, src)
                return
            if op.startswith("ds_read"):
                for i, di in enumerate(dl):
                    m.v[di] = mk("DS", ops[1], m.operand(ops[1]), off, i, line, x.asm)
                events.append(("ldsread", line, op, ops[0], ops[1], off, x.asm))
                return
            sb = IC.reg(ops[2]) if len(ops) > 2 else None
            res = m.gets(sb[1]) if sb else mk("UNK", "rsrc")
            for i, di in enumerate(dl):
                m.v[di] = mk("LD", op, res, ops[1], off, i, line)
            events.append(("load", line, op, ops[0], res, ops[1], off))
            return
        if op.startswith(("ds_write", "buffer_store", "global_store")):
            data_tok = ops[1] if op.startswith("ds_write") else (ops[0] if op.startswith("buffer") else ops[1])
            r = IC.reg(data_tok)
            vals = [m.getv(i) for i in range(r[1], r[2] + 1)] if r else []
            events.append(("store", line, op, ops, vals, off, x.asm))
        return
    if not op.startswith("v_"):
        return
    dk, dl = dst_regs(ops[0]) if ops else (None, [])
    if dk is None:
        return
    if dk == "s":
        for di in dl:
            m.s[di] = mk("S", op, line)
        return
    base = re.sub(r"_e(32|64)$", "", op)
    if op.startswith("v_pk_"):
        osel = mods_of(x, "op_sel") or [0, 0, 0]
        oshi = mods_of(x, "op_sel_hi") or [1, 1, 1]
        nlo = mods_of(x, "neg_lo") or [0, 0, 0]
        nhi = mods_of(x, "neg_hi") or [0, 0, 0]
        outs = []
        for lane in range(2):
            sel = osel if lane == 0 else oshi
            ng = nlo if lane == 0 else nhi
            v = []
            for k, t in enumerate(ops[1:]):
                val = pk_operand(m, t, sel[k] if k < len(sel) else lane)
                v.append(NEG(val) if k < len(ng) and ng[k] else val)
            if base == "v_pk_fma_f32":
                outs.append(FMA(v[0], v[1], v[2]))
            elif base == "v_pk_mul_f32":
                outs.append(MUL(v[0], v[1]))
            elif base == "v_pk_add_f32":
                outs.append(ADD(v[0], v[1]))
            elif base == "v_pk_mov_b32":
                outs.append(v[0] if lane == 0 else v[1])
            else:
                outs.append(mk("UNK", op, line))
        for i, di in enumerate(dl[:2]):
            m.v[di] = outs[i]
        return
    if op.startswith("v_mov_b32_dpp"):
        m.v[dl[0]] = DPP(ctl_of(x), m.operand(ops[1]))
        return
    if op.startswith("v_add_f32_dpp"):
        m.v[dl[0]] = ADD(DPP(ctl_of(x), m.operand(ops[1])), m.operand(ops[2]))
        return
    if "_dpp" in op or "_sdwa" in op:
        for di in dl:
            m.v[di] = mk("I", op, line)
        return
    if base == "v_mov_b32":
        m.v[dl[0]] = m.operand(ops[1])
        return
    if base == "v_mov_b64":
        r = IC.reg(ops[1])
        for i, di in enumerate(dl):
            if r:
                m.v[di] = m.getv(r[1] + i) if r[0] in "va" else m.gets(r[1] + i)
            else:
                m.v[di] = K(parse_lit(ops[1]) or 0) if i == 0 else mk("KHI", ops[1])
        return
    if base == "v_cndmask_b32":
        cond = m.gets("vcc") if (len(ops) < 4 or ops[3] == "vcc") else m.gets(IC.reg(ops[3])[1])
        m.v[dl[0]] = SEL(cond, m.operand(ops[2]), m.operand(ops[1]))
        return
    simple = {"v_mul_f32": lambda: MUL(m.operand(ops[1]), m.operand(ops[2])),
              "v_add_f32": lambda: ADD(m.operand(ops[1]), m.operand(ops[2])),
              "v_sub_f32": lambda: ADD(m.operand(ops[1]), NEG(m.operand(ops[2]))),
              "v_fma_f32": lambda: FMA(m.operand(ops[1]), m.operand(ops[2]), m.operand(ops[3])),
              "v_fmac_f32": lambda: FMA(m.operand(ops[1]), m.operand(ops[2]), m.getv(dl[0])),
              "v_exp_f32": lambda: EXP(m.operand(ops[1])),
              "v_rcp_f32": lambda: RCP(m.operand(ops[1])),
              "v_cvt_pk_bf16_f32": lambda: CVT(m.operand(ops[1]), m.operand(ops[2]))}
    if base in simple:
        m.v[dl[0]] = simple[base]()
        return
    if base == "v_lshlrev_b32" and ops[1] == "16":
        m.v[dl[0]] = mk("bf_lo", m.operand(ops[2]))
        return
    if base == "v_and_b32" and parse_lit(ops[1]) == 0xFFFF0000:
        m.v[dl[0]] = mk("bf_hi", m.operand(ops[2]))
        return
    if base == "v_xor_b32" and parse_lit(ops[1]) == 0x80000000:
        m.v[dl[0]] = NEG(m.operand(ops[2]))
        return
    if base == "v_bfrev_b32" and ops[1] == "1":
        m.v[dl[0]] = K(0x80000000)
        return
    for di in dl:  # integer / address arithmetic and anything else: opaque
        m.v[di] = mk("I", op, line)


# ----------------------------------------------------------------------------------------------------------------------
# regions (the natural-loop linearization of isa_check.check_kernel)
# ----------------------------------------------------------------------------------------------------------------------
def regions(lines):
    insts, labels, bstarts = IC.s_parse(lines)
    blocks, succ, pred = IC.s_cfg(insts, labels, bstarts)
    hdr_line = next(i for i, l in enumerate(lines) if "Loop Header" in l and "Depth=1" in l)
    hb = insts[labels[lines[hdr_line].split(":")[0]]].blk
    fwd, stack = {hb}, [hb]
    while stack:
        u = stack.pop()
        for v in succ[u]:
            if v not in fwd:
                fwd.add(v)
                stack.append(v)
    loop, stack = {hb}, [u for u in pred[hb] if u in fwd]
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
    body = [k for bi in order[::-1] for k in blocks[bi]]
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
    prologue = [k for bi in sorted(ent & to_hdr) for k in blocks[bi]]
    return insts, prologue, body


# ----------------------------------------------------------------------------------------------------------------------
# the contract (CONTRACT_swish 2-3), symbolic
# ----------------------------------------------------------------------------------------------------------------------
TREE = ["quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]", "row_half_mirror", "row_mirror"]


def row_tree(h):
    s = h
    for c in TREE:
        s = ADD(DPP(c, s), s)
    return ADD(s, SWZ(s))


def hsum8(v):
    return ADD(ADD(ADD(v[0], v[4]), ADD(v[2], v[6])), ADD(ADD(v[1], v[5]), ADD(v[3], v[7])))


def slot_contract(X, DY, mean, rstd, W, W2, B2, ADW, ADB, ADX):
    nmr = NEG(MUL(mean, rstd))                            # S1
    xh, sg, g, q, adw, adb = [], [], [], [], [], []
    for j in range(8):
        xh_j = FMA(X[j], rstd, nmr)                       # E1
        z = FMA(xh_j, W2[j], B2[j])                       # E2
        s_ = RCP(ADD(EXP(z), FONE))                       # E3 E4 E5
        der = FMA(NEG(s_), s_, s_)                        # E6
        t_ = MUL(MUL(DY[j], X[j]), der)                   # E7 E8
        g_ = MUL(W[j], t_)                                # E9
        adw.append(FMA(t_, xh_j, ADW[j]))                 # E10
        adb.append(ADD(ADB[j], t_))                       # E11
        q.append(MUL(xh_j, g_))                           # E12
        xh.append(xh_j)
        sg.append(s_)
        g.append(g_)
    a1 = MUL(MUL(row_tree(hsum8(q)), K(INV_D)), rstd)     # R1 R3 R4
    a2 = MUL(MUL(row_tree(hsum8(g)), K(INV_D)), rstd)     # R2 R3 R4
    dx = [FMA(DY[j], sg[j], FMA(g[j], rstd, NEG(FMA(xh[j], a1, a2)))) for j in range(8)]  # E13 E14 E15
    words = [CVT(dx[2 * i], dx[2 * i + 1]) for i in range(4)]  # E16
    adx = [ADD(ADX[j], dx[j]) for j in range(8)]          # E17
    return words, adw, adb, adx


def discover(words):
    """bind (X, DY, mean, rstd) of one slot by walking E15 -> E5 -> E2 -> E1 in each stored dx"""
    X, DY = [None] * 8, [None] * 8
    mean = rstd = None
    for i in range(4):
        w = NODES[words[i]]
        if w[0] != "cvt":
            return None, "word %d is not a v_cvt_pk_bf16_f32: %s" % (i, show(words[i], 3)), None, None
        for h in range(2):
            dx = NODES[w[1 + h]]
            if dx[0] != "fma" or dx[1] != 1:
                return None, "dx is not an fma: %s" % show(w[1 + h], 3), None, None
            sig = [t for t in dx[2:4] if op_of(t) == "rcp"]
            if len(sig) != 1:
                return None, "dx multiplicands: %s" % show(w[1 + h], 3), None, None
            DY[2 * i + h] = dx[2] if dx[3] == sig[0] else dx[3]
            den = NODES[NODES[sig[0]][1]]
            if den[0] != "add":
                return None, "den: %s" % show(NODES[sig[0]][1], 3), None, None
            ex = den[1] if den[2] == FONE else den[2]
            z = NODES[NODES[ex][1]] if op_of(ex) == "exp2" else None
            if z is None or z[0] != "fma":
                return None, "z: %s" % show(ex, 3), None, None
            xhs = [t for t in z[2:4] if op_of(t) == "fma"]
            if not xhs:
                return None, "xhat: %s" % show(NODES[ex][1], 3), None, None
            xh = NODES[xhs[0]]
            nmr = NODES[xh[4]]
            if nmr[0] != "neg" or op_of(nmr[1]) != "mul":
                return None, "nmr: %s" % show(xh[4], 3), None, None
            mr = NODES[nmr[1]][1:]
            rs = [t for t in xh[2:4] if t in mr]
            if not rs:
                return None, "rstd not shared between E1 and S1", None, None
            r_ = rs[0]
            X[2 * i + h] = xh[2] if xh[3] == r_ else xh[3]
            me = mr[0] if mr[1] == r_ else mr[1]
            if mean is None:
                mean, rstd = me, r_
            elif (mean, rstd) != (me, r_):
                return None, "row statistics differ between columns of one slot", None, None
    return X, DY, mean, rstd


def bf_binding(V, nm, u, err):
    out = []
    for i in range(4):
        lo, hi = NODES[V[2 * i]], NODES[V[2 * i + 1]]
        if lo[0] != "bf_lo" or hi[0] != "bf_hi" or lo[1] != hi[1]:
            err("D2 slot %d: %s pair %d is not (bf_lo, bf_hi) of one dword: %s / %s" % (u, nm, i, show(V[2 * i], 2),
                                                                                       show(V[2 * i + 1], 2)))
            return None
        out.append(lo[1])
    return out


def first_diff(a, b, path="", depth=0):
    if a == b:
        return None
    na, nb = NODES[a], NODES[b]
    if depth > 60 or na[0] != nb[0] or len(na) != len(nb) or na[0] not in CH:
        return "%s: got %s | want %s" % (path or ".", show(a, 3), show(b, 3))
    for i in CH[na[0]]:
        if na[i] != nb[i]:
            return first_diff(na[i], nb[i], "%s/%s[%d]" % (path, na[0], i), depth + 1)
    return "%s: fields differ: got %s | want %s" % (path, na, nb)


def epi_check(half, S, n, item, mrow, abase):
    h = NODES[half]
    if h[0] != "add" or FZERO not in h[1:]:
        return "not an add of +0: %s" % show(half, 3)
    mul = h[2] if h[1] == FZERO else h[1]
    mn = NODES[mul]
    if mn[0] != "mul" or S not in mn[1:]:
        return "not a multiply by S: %s" % show(mul, 3)
    cur = mn[2] if mn[1] == S else mn[1]
    links = []
    while op_of(cur) == "MF":
        links.append(NODES[cur])
        cur = NODES[cur][5]
    if cur != FZERO or len(links) != 4:
        return "chain of %d links ending in %s" % (len(links), show(cur, 2))
    links.reverse()
    for k, l in enumerate(links):
        if l[1] != item or l[2] != "blgp:1":
            return "link %d item %d mods %s (want item %d, blgp:1)" % (k, l[1], l[2], item)
        wa, aa = [NODES[t] for t in l[3]], [NODES[t] for t in l[4]]
        if len(wa) != 8 or any(t[0] != "LD" for t in wa) or [t[5] for t in wa] != [0, 1, 2, 3] * 2:
            return "link %d src0 is not two loaded quads" % k
        if len({t[6] for t in wa[:4]}) != 1 or len({t[6] for t in wa[4:]}) != 1 or wa[0][4] != 2048 * n + 128 * k \
                or wa[4][4] != 2048 * n + 128 * k + 64 or wa[0][3] != wa[4][3] or wa[0][2] != wa[4][2] \
                or wa[0][1] != "buffer_load_dwordx4":
            return "link %d W quads at %d/%d via %s (want %d/+64, one voffset/resource)" % (
                k, wa[0][4], wa[4][4], wa[0][1], 2048 * n + 128 * k)
        if NODES[wa[0][2]] != ("part", 0, wa[0][2]) and NODES[wa[0][2]][0] not in ("part", "karg", "S", "INS"):
            pass
        if len(aa) != 8 or any(t[0] != "DS" for t in aa) or [t[4] for t in aa] != [0, 1, 2, 3] * 2:
            return "link %d src1 is not two ds_read quads" % k
        if len({t[5] for t in aa[:4]}) != 1 or len({t[5] for t in aa[4:]}) != 1:
            return "link %d src1 quads not single reads" % k
        if aa[0][3] != 8192 * k + 2048 * mrow or aa[4][3] != aa[0][3]:
            return "link %d A quads at %d/%d (want %d)" % (k, aa[0][3], aa[4][3], 8192 * k + 2048 * mrow)
        abase.add((aa[0][1], aa[4][1]))
    return None


def check(name, lines):
    label, s0, emit = KERNELS[name]
    out = dict(kernel=label, errors=[], info=[])
    err, info = out["errors"].append, out["info"].append
    insts, prologue, body = regions(lines)
    m = Machine()
    ev_pro = []
    for k in prologue:
        v_exec(m, insts[k], insts[k].line + 1, ev_pro)
    wv, ws = set(), set()
    for k in body:
        x = insts[k]
        if not x.ops or x.op.startswith(("ds_write", "buffer_store", "global_store", "s_waitcnt", "s_cbranch",
                                         "s_branch", "s_barrier", "s_nop")) or "lds" in x.mods:
            continue
        if x.ops[0] == "vcc" or x.op.startswith("v_cmp") and x.op.endswith("_e32"):
            ws.add("vcc")
        d = IC.reg(x.ops[0])
        if d:
            (wv if d[0] in "va" else ws).update(range(d[1], d[2] + 1))
    mb = Machine()
    mb.v = {r: v for r, v in m.v.items() if r not in wv}
    mb.s = {r: v for r, v in m.s.items() if r not in ws}
    events = []
    for k in body:
        x = insts[k]
        if x.op == "s_barrier":
            events.append(("barrier", x.line + 1))
            continue
        v_exec(mb, x, x.line + 1, events)
    bars = [i for i, e in enumerate(events) if e[0] == "barrier"]
    if len(bars) != 2:
        err("expected 2 barriers in the loop body, got %d" % len(bars))
        return out
    pre_mid, post_mid = events[bars[0]:bars[1]], events[bars[1]:]
    # ---- prologue facts: S = product of the two scale loads; nw / nb column loads
    S = None
    for k in prologue:
        x = insts[k]
        if x.op.startswith("v_mul_f32"):
            S = m.v.get(IC.reg(x.ops[0])[1])
    if S is not None:
        sn = NODES[S]
        info("S = %s" % show(S, 3))
        kargs = sorted(NODES[t][2] if NODES[t][0] == "SLD" else None for t in sn[1:3]
                       if NODES[t][0] in ("SLD",)) if sn[0] == "mul" else []
    nwl = [e for e in ev_pro if e[0] == "load" and NODES[e[4]] == ("karg", 0x38)]
    nbl = [e for e in ev_pro if e[0] == "load" and NODES[e[4]] == ("karg", 0x40)]

    def col(loads, j):
        for e in loads:
            if e[6] == 16 * (j // 4):
                return mk("LD", e[2], e[4], e[5], e[6], j % 4, e[1])
        return mk("MISSING", j)
    NW = [col(nwl, j) for j in range(8)]
    NB = [col(nbl, j) for j in range(8)]
    if len(nwl) != 2 or len(nbl) != 2 or len({e[5] for e in nwl + nbl}) != 1:
        err("nw / nb prologue loads: %s" % [(e[1], e[3], e[5], e[6]) for e in nwl + nbl])
    W2 = [MUL(NW[j], K(NL2E)) for j in range(8)]
    B2 = [MUL(NB[j], K(NL2E)) for j in range(8)]
    vcc = mb.gets("vcc")
    info("vcc in the consumer = %s; nw/nb voffset %s" % (show(vcc, 3), nwl and nwl[0][5]))
    # ---- D1
    if not s0:
        if S is None or NODES[S][0] != "mul" or any(NODES[t][0] != "SLD" for t in NODES[S][1:3]) or \
                sorted(NODES[t][2] for t in NODES[S][1:3]) != [0, 0]:
            err("D1 scale S is not the product of two scalar loads at offset 0: %s" % (S is not None and show(S, 3)))
        else:
            srcs = sorted(NODES[NODES[t][1]] for t in NODES[S][1:3])
            if srcs != [("karg", 0x10), ("karg", 0x18)]:
                err("D1 scale loads from %s (want kernarg 0x10 sa, 0x18 sb)" % srcs)
        rw = [e for e in pre_mid if e[0] == "store" and e[2] == "ds_write_b128" and e[6] and e[5] % 0x2100 == 0
              and e[5] < 4 * 0x2100 and all(op_of(v) == "cvt" for v in e[4])]
        other = [e for e in pre_mid if e[0] == "store" and e[2] == "ds_write_b128" and e not in rw]
        info("R writes at %s; other ds_write_b128 in the MFMA phase: %d (%s)" % (
            [hex(e[5]) for e in rw], len(other), sorted({show(v, 1) for e in other for v in e[4]})))
        if sorted(e[5] // 0x2100 for e in rw) != [0, 1, 2, 3]:
            err("D1 R writes per row group: %s" % [e[5] for e in rw])
        if len({e[3][0] for e in rw}) != 1:
            err("D1 R writes from different base registers")
        abase = set()
        for e in rw:
            mrow = e[5] // 0x2100
            for wd, v in enumerate(e[4]):
                for h, half in enumerate(NODES[v][1:3]):
                    why = epi_check(half, S, wd // 2, 2 * (wd % 2) + h, mrow, abase)
                    if why:
                        err("D1 m=%d word %d half %d: %s" % (mrow, wd, h, why))
        if len(abase) != 1:
            err("D1 A fragments from %d (lo, hi) base pairs: %s" % (len(abase), abase))
        else:
            info("A fragment bases (lo, hi) = %s" % (abase,))
        if emit:
            ems = [e for e in pre_mid if e[0] == "store" and e[2].startswith("buffer_store")]
            for e in ems:
                if not any(e[4] == r[4] for r in rw):
                    err("D1 EMIT store at line %d does not store an R quad" % e[1])
            info("EMIT da1 stores: %d, all R quads" % len(ems))
    # ---- D2 / D3
    stores = [e for e in post_mid if e[0] == "store" and e[2] == "buffer_store_dwordx4"]
    if len(stores) != 4:
        err("D2 expected 4 dh1 stores after the mid barrier, got %d" % len(stores))
        return out
    loads_post = [e for e in post_mid if e[0] == "load"]
    PH = {(nm, j): mk("ACC", nm, j) for nm in ("adw", "adb", "adx") for j in range(8)}
    folds = dict(PH)
    for u, st in enumerate(stores):
        words = st[4]
        X, DY, mean, rstd = discover(words)
        if X is None:
            err("D2 slot %d: could not bind the inputs: %s" % (u, DY))
            continue
        exp_words, adw, adb, adx = slot_contract(X, DY, mean, rstd, NW, W2, B2,
                                                 [PH[("adw", j)] for j in range(8)],
                                                 [PH[("adb", j)] for j in range(8)],
                                                 [PH[("adx", j)] for j in range(8)])
        for i in range(4):
            if words[i] != exp_words[i]:
                err("D2 slot %d word %d differs from the contract: %s" % (u, i, first_diff(words[i], exp_words[i])))
        for nm, outv in (("adw", adw), ("adb", adb), ("adx", adx)):
            for j in range(8):
                folds[(nm, j)] = subst(outv[j], PH[(nm, j)], folds[(nm, j)])
        bx = bf_binding(X, "x", u, err)
        bd = bf_binding(DY, "dy", u, err)
        lo_line, hi_line = st[1], stores[u + 1][1] if u + 1 < 4 else 10 ** 9
        for nm2, bb in (("x", bx), ("dy", bd)):
            if not bb:
                continue
            nn = [NODES[b] for b in bb]
            if nm2 == "dy" and not s0:
                if not (all(t[0] == "DS" for t in nn) and len({t[5] for t in nn}) == 1 and nn[0][3] == u * 0x2100
                        and [t[4] for t in nn] == [0, 1, 2, 3]):
                    err("D2 slot %d: dy from %s (want dwords 0..3 of one R ds_read at %s)" % (
                        u, [show(b, 1) for b in bb], hex(u * 0x2100)))
                else:
                    info("slot %d: dy = R ds_read_b128 %s offset %s (line %d)" % (u, nn[0][1], hex(nn[0][3]), nn[0][5]))
                continue
            if not all(t[0] == "IN" for t in nn):
                err("D2 slot %d: %s not loop-carried registers: %s" % (u, nm2, [show(b, 1) for b in bb]))
                continue
            regs = [int(t[1][1:]) for t in nn]
            if regs != list(range(regs[0], regs[0] + 4)):
                err("D2 slot %d: %s dwords %s not one quad in order" % (u, nm2, regs))
                continue
            pf = [e for e in loads_post if lo_line < e[1] < hi_line and e[2] == "buffer_load_dwordx4"
                  and IC.reg(e[3])[1] == regs[0]]
            if len(pf) != 1:
                err("D2 slot %d: %s quad v%d is not reloaded exactly once by this slot's prefetch (%d)" % (
                    u, nm2, regs[0], len(pf)))
            else:
                rs = NODES[pf[0][4]]
                info("slot %d: %s quad v[%d:%d], reloaded at line %d (voffset %s imm %d rsrc %s)" % (
                    u, nm2, regs[0], regs[0] + 3, pf[0][1], pf[0][5], pf[0][6], show(pf[0][4], 2)))
        for nm, val in (("mean", mean), ("rstd", rstd)):
            n = NODES[val]
            if not (n[0] == "sel" and n[1] == vcc and op_of(n[2]) == "rl" and op_of(n[3]) == "rl"):
                err("D2 slot %d: %s = %s" % (u, nm, show(val, 3)))
                continue
            hi_n, lo_n = NODES[n[2]], NODES[n[3]]
            if lo_n[2] != ("lin", 2, 16 * u) or hi_n[2] != ("lin", 2, 16 * u + 1) or lo_n[1] != hi_n[1]:
                err("D2 slot %d: %s lanes lo %s hi %s (want 2w+%d / 2w+%d) from %s / %s" % (
                    u, nm, lo_n[2], hi_n[2], 16 * u, 16 * u + 1, show(lo_n[1], 1), show(hi_n[1], 1)))
            else:
                info("slot %d: %s = sel(vcc, readlane(%s, 2w+%d), readlane(., 2w+%d))" % (
                    u, nm, show(lo_n[1], 1), 16 * u + 1, 16 * u))
    # the stats registers must be reloaded (mean1 / rstd1 kernargs 0x28 / 0x30) after their readlanes
    # D3
    used = {}
    for key, f in folds.items():
        hits = [r for r, endv in mb.v.items() if endv == subst(f, PH[key], mk("IN", "v%d" % r))]
        if len(hits) != 1:
            err("D3 %s[%d]: in-order 4-slot fold found in %d registers (want 1)" % (key[0], key[1], len(hits)))
        else:
            used[key] = hits[0]
    if len(set(used.values())) != len(used):
        err("D3 two accumulators share a register")
    info("accumulators: " + " ".join("%s%d=v%d" % (k[0], k[1], v) for k, v in sorted(used.items())))
    info("dh1 stores: " + "; ".join("line %d voffset %s rsrc %s imm %d" % (st[1], st[3][1], st[3][2], st[5])
                                     for st in stores))
    return out


def main(argv):
    paths = argv or [os.path.join(BUILD, "out", c, "x2lb_pass.s") for c in ("r720", "r720_ilp", "sdk10", "sdk10_ilp")]
    ok = True
    for p in paths:
        kern, _ = IC.split_asm(open(p).read())
        for name in KERNELS:
            if name not in kern:
                print("== %s: %s missing -> FAIL" % (p, name))
                ok = False
                continue
            r = check(name, kern[name])
            ok &= not r["errors"]
            print("== %s %s: %s" % (os.path.relpath(p, BUILD), r["kernel"], "PASS" if not r["errors"] else "FAIL"))
            for e in r["errors"][:40]:
                print("   ERROR", e)
            for i in r["info"]:
                print("   info", i)
    print("DFCHECK:", "PASS" if ok else "FAIL", "(%d expression nodes)" % len(NODES))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
