#!/usr/bin/env python3
"""x2lb_emu.py -- lane-level CPU model of the X2L-B v0 kernel's data movement (actckpt iter70, x2lb/build). CPU only.

Mirrors the index functions of ../x2lb_pass.cu section 1 ([cu] = that file) 1:1 and replays, lane by lane, every byte
the kernel moves, with symbolic data, to prove on the CPU what the GPU lab would otherwise find out the hard way:

  E1  A tile LDS-DMA (L11 st_16x128 swizzle, 4 instructions x 8 waves x 64 lanes x 16 B) covers each A buffer exactly
      once, and the fragment reads (lp0/lp1 + frag_off) give lane (q4, r16) of row group m, K step k the bytes
      dh8[16m + r16][128k + 16 q4, +16) ++ [128k + 64 + 16 q4, +16)  ([CD 0] per-lane K bytes).
  E2  resident W fragments gathered from w8_t give lane (q4, r16) of (w, n, k) the bytes
      w8_t[w_col][128k + 16 q4, +16) ++ [128k + 64 + 16 q4, +16) ([CD 6.2]); every w8_t byte is used exactly once per WG.
  E3  MFMA pairing: for every output (i, j) of every v_mfma_f32_16x16x128_f8f6f4 (src0 = W frag, src1 = dh8 frag),
      the 128 (src0 byte, src1 byte) pairs of each K step are (w8_t[n][K], dh8[t][K]) with the SAME K, and the 4 steps
      cover K = 0..511 exactly once, in ascending 128-blocks: acc = sum_K dh8[t][K] * w8_t[n][K] for t = 16m + j,
      n = w_col(w, n, s = i); D lane (q4, r16) item r = da1[16m + r16][32w + 8 q4 + 4n + r].
  E4  EPI2 packing + R write + consumer read: the consumer lane (w, u, lane) receives exactly da1[c_row][c_col0..+8];
      every R byte (64 rows x 512 B) is written once per tile and read once per tile.
  E5  global offsets: consumer h1 / dh1 / S0-da1 (c_voff + u*SLOT_STRIDE), EMIT da1 (e_voff + m*SLOT_STRIDE),
      mean/rstd (lane*4 + v_readlane rows) address exactly the row/columns the lane works on; with the per-tile
      num_records = rows*512 the range check drops exactly the rows >= M.
  E6  partial tiles: for M = 64q + r (r = 1..63) the A buffer holds zeros on rows >= M (DMA lanes of those rows are
      out of range; the ds_write zero lanes cover exactly those rows' 4 x 128 B, and no lane both DMAs and writes).
      Fix pass (race review F1): the zero-fill guard is the kernel's own index function zfill_lane (mirrored, drift-
      checked by idx_harness.cpp), the out-of-range DMA lanes write nothing, and E6 also checks zfill_lane == the
      out-of-range DMA lanes for rows 0..64 (before, E6 wrote the zeros itself and could not fail).
  E7  static schedule: tiles b + jG over b < G cover [0, ntiles) exactly once for every (M, G) tested; the one-past-
      the-end prefetch tile has rows = 0.
  E8  LDS map: A0 [0, 32768), A1 [32768, 65536), R [65536, 99328) disjoint; ds offsets fit 16 bits; partial staging
      [3][16][256] fp32 fits the A region; the end-of-kernel reduction feeds each (o, c) the 16 slot partials once.
  E9  bank-conflict census (informational) of the fragment reads, R writes, R reads (b128 phases of 16 lanes).
  E10 INFORMATIONAL since the fix pass (robustness review R12): the fp64 accuracy of the documented partial order and
      the padded-row identity of the contract reference. Neither line can fail for a wrong kernel order (any fp32
      order meets 1e-5; the padded-row line runs swish_ref, not the kernel); the bitwise evidence is the lane-level
      emulator (build/emu_lane) and isa_check C15 D3.
  --dump-index prints the section-1 index table in idx_harness.cpp's format (build_cpu.sh diffs the two: R12).

Usage: PYTHONDONTWRITEBYTECODE=1 /scratch/projection_20261005/venv/bin/python -B x2lb_emu.py [--quick]
Exit 0 iff every check passes.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SWISH = os.path.join(HERE, "..", "..", "contract_swish")

# ---------------------------------------------------------------- [cu] section 0 constants
KD, NOUT, BM, NWAVE, NSLOT = 512, 256, 64, 8, 4
ABYTES, RBASE, RPITCH = 32768, 65536, 528
RBYTES = BM * RPITCH
LDS_BYTES = RBASE + RBYTES
ROWB, SLOT_STRIDE = 512, 16 * 512


# ---------------------------------------------------------------- [cu] section 1 index functions (1:1)
def static_tile(b, j, G): return b + j * G
def tile_rows(t, M): return 0 if M - t * BM < 0 else (BM if M - t * BM > BM else M - t * BM)
def a_dma_row(w, lane): return 8 * w + (lane >> 3)
def a_dma_kswz(w, lane): return 16 * ((lane & 7) ^ ((a_dma_row(w, lane) >> 1) & 7))
def a_dma_voff(w, lane): return a_dma_row(w, lane) * KD + a_dma_kswz(w, lane)
def a_dma_lds(buf, w, i): return buf * ABYTES + i * 8192 + w * 1024
def a_dma_soff(i): return i * 128
def zfill_lane(rows, w, lane): return 0 < rows < BM and a_dma_row(w, lane) >= rows   # [cu] zfill_lane (fix pass)
def rsrc_tile(t, rows): return t if rows > 0 else 0                                     # [cu] rsrc_tile (fix pass, R9)
def frag_lp0(lane): return (lane & 15) * 128 + ((16 * (lane >> 4)) ^ (((lane & 15) >> 1) << 4))
def frag_lp1(lane): return (lane & 15) * 128 + ((16 * (lane >> 4) + 64) ^ (((lane & 15) >> 1) << 4))
def frag_off(buf, k, m): return buf * ABYTES + k * 8192 + m * 2048
def w_col(w, n, lane): return 32 * w + 8 * ((lane & 15) >> 2) + 4 * n + (lane & 3)
def w_voff(w, lane): return w_col(w, 0, lane) * KD + 16 * (lane >> 4)
def w_imm(n, k, h): return 4 * n * KD + 128 * k + 64 * h
def d_row(m, lane): return 16 * m + (lane & 15)
def d_col(w, n, i, lane): return 32 * w + 8 * (lane >> 4) + 4 * n + i
def rw_base(w, lane): return RBASE + (lane & 15) * RPITCH + 64 * w + 16 * (lane >> 4)
def c_row(w, u, lane): return 16 * u + 2 * w + (lane >> 5)
def c_col0(lane): return 8 * (lane & 31)
def rr_base(w, lane): return RBASE + (2 * w + (lane >> 5)) * RPITCH + 2 * c_col0(lane)
def c_voff(w, u, lane): return c_row(w, u, lane) * ROWB + 2 * c_col0(lane)
def e_voff(w, m, lane): return d_row(m, lane) * ROWB + 2 * (32 * w + 8 * (lane >> 4))
def top_vmcnt(emit): return (4 if emit else 0) + NSLOT + NSLOT + 2                   # [cu] top_vmcnt
def acc_slot(w, lane): return 2 * w + (lane >> 5)
def red_index(o, slot, c): return (o * 2 * NWAVE + slot) * NOUT + c


FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- symbolic tensors
# A byte of dh8 is tagged by (row, K) -> code = row * 512 + K + 1 (0 = zero); W byte (n, K) -> -(n * 512 + K + 1).
def lds_after_dma(rows, buf):
    """LDS byte map (code per byte, int64) of one A buffer after the kernel's DMA + zero fill of a tile with `rows`
    valid rows (dh8 row r of the tile -> code). Asserts single coverage.
    Fix pass (race review F1): the two writers are modelled separately and the zero fill's guard is the kernel's own
    index function zfill_lane (mirrored above, drift-checked against the .cu by idx_harness.cpp). An out-of-range DMA
    lane writes NOTHING (the adversarial reading of FACTS f), so a padded row is zero only if zfill_lane fills it; a
    regressed guard leaves the sentinel there and E1/E6 fail."""
    lds = np.full(LDS_BYTES, -(10 ** 12), dtype=np.int64)  # sentinel = never written
    writes = np.zeros(LDS_BYTES, dtype=np.int64)
    for w in range(NWAVE):
        for i in range(4):
            base = a_dma_lds(buf, w, i)
            for lane in range(64):
                dst = base + 16 * lane  # hardware: M0 + 16 * lane
                rr = a_dma_row(w, lane)
                g0 = a_dma_voff(w, lane) + a_dma_soff(i)  # byte offset within the tile (row rr * 512 + K)
                assert g0 // KD == rr
                k0 = g0 % KD
                if g0 - a_dma_soff(i) < rows * KD:  # range check on voffset (+ inst offset 0) < num_records
                    lds[dst:dst + 16] = rr * KD + k0 + np.arange(16) + 1
                    writes[dst:dst + 16] += 1
                if zfill_lane(rows, w, lane):       # the kernel's zero-fill ds_write of this lane (issue_A)
                    za = (a_dma_lds(0, w, 0) + 16 * lane) + buf * ABYTES + i * 8192  # ldsz + bofs + imm
                    lds[za:za + 16] = 0
                    writes[za:za + 16] += 1
    return lds, writes


def e1_e6_a_tile():
    print("E1/E6 A tile DMA + fragment reads")
    # E6 (fix pass): the kernel's zero-fill guard zfill_lane selects EXACTLY the out-of-range DMA lanes of a partial
    # tile, and nothing on a full or empty tile, for every rows in 0..64 and every (wave, lane)
    bad = sum(1 for rows in range(0, BM + 1) for w in range(NWAVE) for lane in range(64)
              if zfill_lane(rows, w, lane) != (0 < rows < BM and a_dma_voff(w, lane) >= rows * KD))
    check("E6 zfill_lane(rows, w, lane) == (0 < rows < 64 and the lane's DMA voffset >= rows*512), rows 0..64", bad == 0,
          f"bad {bad}")
    for buf in (0, 1):
        for rows in (64, 1, 17, 40, 63):
            lds, writes = lds_after_dma(rows, buf)
            reg = slice(buf * ABYTES, (buf + 1) * ABYTES)
            cover = bool(np.all(writes[reg] == 1)) and int(writes.sum()) == ABYTES
            bad = 0
            for k in range(4):
                for m in range(4):
                    for lane in range(64):
                        q4, r16 = lane >> 4, lane & 15
                        a0 = frag_lp0(lane) + frag_off(buf, k, m)
                        a1 = frag_lp1(lane) + frag_off(buf, k, m)
                        assert a0 % 16 == 0 and a1 % 16 == 0 and frag_off(buf, k, m) < 65536
                        got = np.concatenate([lds[a0:a0 + 16], lds[a1:a1 + 16]])
                        r = 16 * m + r16
                        K = np.concatenate([128 * k + 16 * q4 + np.arange(16), 128 * k + 64 + 16 * q4 + np.arange(16)])
                        want = (r * KD + K + 1) if r < rows else np.zeros(32, dtype=np.int64)
                        bad += int(np.any(got != want))
            check(f"buf {buf} rows {rows}: buffer covered exactly once, 1024 fragment reads = contract bytes",
                  cover and bad == 0, f"bad fragments {bad}")


def w_frag_bytes(w, n, k, lane):
    """the 32 W bytes lane `lane` of wave w holds for (n, k): two 16-B loads at w_voff + w_imm(n, k, h)"""
    out = []
    for h in range(2):
        off = w_voff(w, lane) + w_imm(n, k, h)
        col, kb = off // KD, off % KD
        out.append(-(col * KD + kb + np.arange(16) + 1))
    return np.concatenate(out)


def e2_e3_w_and_mfma():
    print("E2/E3 resident W fragments and MFMA K pairing")
    used = np.zeros(NOUT * KD, dtype=np.int64)
    bad_w = 0
    for w in range(NWAVE):
        for n in range(2):
            for k in range(4):
                for lane in range(64):
                    q4, r16 = lane >> 4, lane & 15
                    got = w_frag_bytes(w, n, k, lane)
                    col = w_col(w, n, lane)
                    K = np.concatenate([128 * k + 16 * q4 + np.arange(16), 128 * k + 64 + 16 * q4 + np.arange(16)])
                    want = -(col * KD + K + 1)
                    bad_w += int(np.any(got != want))
                    np.add.at(used, (-got - 1), 1)
    check("W fragment bytes = w8_t[w_col][128k+16q4..] ++ [128k+64+16q4..] for 8 waves x 2 n x 4 k x 64 lanes",
          bad_w == 0, f"bad {bad_w}")
    check("every w8_t byte is held exactly once per WG", bool(np.all(used == 1)))
    # MFMA operand model (CDNA4 16x16x128 f8f6f4, standard layout [CD 4]): operand lane l = 16g + s carries row s of its
    # 16-row block, bytes b = 0..31 at K position kpos(g, b) = 16g + b (b < 16) / 64 + 16g + (b - 16) within the step.
    lds, _ = lds_after_dma(64, 0)
    bad, cover_bad = 0, 0
    for w in (0, 3, 7):
        for m in range(4):
            for n in range(2):
                # chain over k: collect, per output (i, j), the K indices of the pairs
                pairs = {}
                for k in range(4):
                    for g in range(4):
                        for s_i in range(16):
                            la = 16 * g + s_i  # src0 lane holding W row i = s_i
                            wb = w_frag_bytes(w, n, k, la)
                            for s_j in range(16):
                                lb = 16 * g + s_j  # src1 lane holding dh8 row j = s_j
                                a0 = frag_lp0(lb) + frag_off(0, k, m)
                                a1 = frag_lp1(lb) + frag_off(0, k, m)
                                db = np.concatenate([lds[a0:a0 + 16], lds[a1:a1 + 16]])
                                Kw = (-wb - 1) % KD
                                Kd = (db - 1) % KD
                                if np.any(Kw != Kd):
                                    bad += 1
                                pairs.setdefault((s_i, s_j), []).append(Kw)
                for (s_i, s_j), lst in pairs.items():
                    ks = np.concatenate(lst)
                    if not np.array_equal(np.sort(ks), np.arange(KD)):
                        cover_bad += 1
                    # step k's pairs lie in [128k, 128k + 128): ascending K blocks along the chain
                    if [int(x.min()) // 128 for x in lst[::4]] != [0, 1, 2, 3]:
                        cover_bad += 1
    check("MFMA pairs: src0 W byte and src1 dh8 byte always share the same K", bad == 0, f"bad lane-pairs {bad}")
    check("each output's 4-step chain covers K = 0..511 once, step k = K block k", cover_bad == 0, f"bad {cover_bad}")
    # D layout: lane (q4, r16) item r -> D row i = 4 q4 + r (= src0 row = W row s), D col j = r16 (= src1 row = token)
    bad_d = 0
    for w in range(NWAVE):
        for n in range(2):
            for lane in range(64):
                q4, r16 = lane >> 4, lane & 15
                for r in range(4):
                    s_w = 4 * q4 + r
                    col = w_col(w, n, 16 * 0 + s_w)  # column of W row s_w (w_col depends on lane & 15 only)
                    if col != d_col(w, n, r, lane):
                        bad_d += 1
                for m in range(4):
                    if d_row(m, lane) != 16 * m + r16:
                        bad_d += 1
    check("D lane (q4, r16) item r = da1[16m + r16][32w + 8 q4 + 4n + r] (d_row / d_col)", bad_d == 0)


def e4_redistribution():
    print("E4 EPI2 packing, R write, consumer read")
    R = np.full(RBYTES, -1, dtype=np.int64)
    wcount = np.zeros(RBYTES, dtype=np.int64)
    for w in range(NWAVE):
        for m in range(4):
            for lane in range(64):
                # q = {epi2(acc[0].xy), epi2(acc[0].zw), epi2(acc[1].xy), epi2(acc[1].zw)}: bf16 element e of q is
                # acc[e >> 2] item e & 3 -> da1[d_row][d_col(w, e >> 2, e & 3)]
                cols = [d_col(w, e >> 2, e & 3, lane) for e in range(8)]
                assert cols == list(range(32 * w + 8 * (lane >> 4), 32 * w + 8 * (lane >> 4) + 8))
                a = rw_base(w, lane) + m * 16 * RPITCH - RBASE
                assert a % 16 == 0 and m * 16 * RPITCH < 65536
                row = d_row(m, lane)
                for e in range(8):
                    R[a + 2 * e:a + 2 * e + 2] = row * NOUT + cols[e]
                    wcount[a + 2 * e:a + 2 * e + 2] += 1
    data = np.zeros(RBYTES, dtype=bool)
    for r in range(BM):
        data[r * RPITCH:r * RPITCH + 512] = True
    check("R: each of the 64 x 256 da1 elements written exactly once per tile, pad bytes untouched",
          bool(np.all(wcount[data] == 1)) and bool(np.all(wcount[~data] == 0)))
    bad, rcount = 0, np.zeros(RBYTES, dtype=np.int64)
    for w in range(NWAVE):
        for u in range(NSLOT):
            for lane in range(64):
                a = rr_base(w, lane) + u * 16 * RPITCH - RBASE
                assert a % 16 == 0 and u * 16 * RPITCH < 65536
                got = R[a:a + 16:2]
                want = c_row(w, u, lane) * NOUT + c_col0(lane) + np.arange(8)
                bad += int(np.any(got != want))
                rcount[a:a + 16] += 1
    check("consumer lane (w, u, lane) reads da1[c_row][c_col0 .. +8] (8 waves x 4 slots x 64 lanes)", bad == 0)
    check("every da1 byte of R read exactly once per tile", bool(np.all(rcount[data] == 1)))
    rows = sorted(c_row(w, u, lane) for w in range(NWAVE) for u in range(NSLOT) for lane in (0, 32))
    check("consumer rows: each tile row by exactly one (wave, slot, half)", rows == list(range(BM)))


def e5_global_offsets():
    print("E5 global offsets and range checks")
    bad = 0
    for w in range(NWAVE):
        for lane in range(64):
            for u in range(NSLOT):
                v = c_voff(w, 0, lane) + u * SLOT_STRIDE
                if v != c_row(w, u, lane) * ROWB + 2 * c_col0(lane) or v != c_voff(w, u, lane):
                    bad += 1
            for m in range(4):
                v = e_voff(w, 0, lane) + m * SLOT_STRIDE
                if v != d_row(m, lane) * ROWB + 2 * (32 * w + 8 * (lane >> 4)):
                    bad += 1
    check("slot / row-group offsets = row * 512 + 2 * first column (whole offset in voffset)", bad == 0)
    # range check: num_records = rows * 512; a 16-B access at voffset v is in range iff v + 16 <= nrec (the kernel's
    # offsets are 16-B aligned and rows are 512 B, so 'v < nrec' and 'v + 16 <= nrec' agree)
    bad = 0
    for rows in range(0, 65):
        nrec = rows * ROWB
        for w in range(NWAVE):
            for lane in range(64):
                for u in range(NSLOT):
                    v = c_voff(w, u, lane)
                    if (v < nrec) != (c_row(w, u, lane) < rows):
                        bad += 1
                for m in range(4):
                    v = e_voff(w, m, lane)
                    if (v < nrec) != (d_row(m, lane) < rows):
                        bad += 1
                if (lane * 4 < rows * 4) != (lane < rows):
                    bad += 1
    check("per-tile num_records = rows*512 (rows*4 for mean/rstd) drops exactly the rows >= M", bad == 0)
    # v_readlane rows for the slot statistics: lanes 0-31 of slot u use row 16u + 2w, lanes 32-63 row 16u + 2w + 1
    bad = sum(1 for w in range(NWAVE) for u in range(NSLOT) for lane in range(64)
              if (16 * u + 2 * w + (1 if lane >= 32 else 0)) != c_row(w, u, lane))
    check("v_readlane(mean/rstd, 16u + 2w (+1)) = the lane's consumer row", bad == 0)


def e7_schedule():
    print("E7 static tile schedule")
    bad = 0
    for M in (1, 63, 64, 65, 4095, 65536, 2_424_832, 2_492_120, 2_650_962):
        nt = (M + BM - 1) // BM
        for ncu in (1, 7, 80, 256, 304):
            G = min(nt, ncu)
            seen = np.zeros(nt, dtype=np.int64)
            for b in range(G):
                j = 0
                while static_tile(b, j, G) < nt:
                    seen[static_tile(b, j, G)] += 1
                    j += 1
                if tile_rows(static_tile(b, j, G), M) != 0:
                    bad += 1
            if not np.all(seen == 1) or sum(tile_rows(t, M) for t in range(nt)) != M:
                bad += 1
            if M <= (1 << 29) and 64 * (nt - 1 + G) >= 2 ** 31:
                bad += 1
    check("tiles b + jG cover [0, ntiles) once; one-past-the-end tile has 0 rows; 64(t+G) < 2^31", bad == 0)


def e8_lds_map():
    print("E8 LDS map")
    regions = [(0, ABYTES), (ABYTES, 2 * ABYTES), (RBASE, RBASE + RBYTES)]
    ok = all(regions[i][1] <= regions[i + 1][0] for i in range(2)) and LDS_BYTES == 99328 and LDS_BYTES <= 163840
    check("A0 / A1 / R disjoint, LDS 99,328 B <= 163,840 B", ok)
    check("partial staging [3][16][256] fp32 = 49,152 B fits the A region", 3 * 16 * 256 * 4 <= 2 * ABYTES)
    got = np.zeros(3 * 16 * 256, dtype=np.int64)
    for w in range(NWAVE):
        for lane in range(64):
            for o in range(3):
                for e in range(8):
                    got[red_index(o, acc_slot(w, lane), c_col0(lane) + e)] += 1
    check("each (o, slot, column) staged by exactly one lane; the tree reads all 16 slots per (o, c)",
          bool(np.all(got == 1)))


def b128_conflicts(addrs):
    """extra passes of one ds_read/write_b128 (64 lane byte addresses) under the 16-consecutive-lane phase model and the
    measured MI350X phase grouping [FACTS c]: quad = (addr / 16) % 16; distinct addresses on one quad serialize"""
    phases_seq = [list(range(16 * p, 16 * p + 16)) for p in range(4)]
    P = lambda a, b: list(range(a, b + 1))
    phases_meas = [P(0, 3) + P(12, 15) + P(20, 23) + P(24, 27), P(32, 35) + P(44, 47) + P(52, 55) + P(56, 59),
                   P(4, 7) + P(8, 11) + P(16, 19) + P(28, 31), P(36, 39) + P(40, 43) + P(48, 51) + P(60, 63)]
    out = []
    for phases in (phases_seq, phases_meas):
        worst = 0
        for ph in phases:
            byq = {}
            for l in ph:
                byq.setdefault((addrs[l] // 16) % 16, set()).add(addrs[l])
            worst = max(worst, max(len(v) for v in byq.values()))
        out.append(worst)
    return out


def e9_banks():
    print("E9 LDS bank conflicts (informational: worst ways per phase, [16-lane model, measured model])")
    fr = [b128_conflicts([frag_lp0(l) + frag_off(0, 0, 0) for l in range(64)]),
          b128_conflicts([frag_lp1(l) + frag_off(0, 0, 0) for l in range(64)])]
    rw = [b128_conflicts([rw_base(w, l) for l in range(64)]) for w in range(NWAVE)]
    rr = [b128_conflicts([rr_base(w, l) for l in range(64)]) for w in range(NWAVE)]
    print(f"    fragment reads lp0/lp1: {fr}; R writes per wave: {sorted(set(map(tuple, rw)))}; "
          f"R reads per wave: {sorted(set(map(tuple, rr)))}")
    check("fragment reads conflict-free (L11 swizzle)", all(max(x) == 1 for x in fr))
    check("consumer R reads conflict-free", all(max(x) == 1 for x in rr))


# ---------------------------------------------------------------- E10 numerics model
def e10_numerics(quick):
    print("E10 consumer numerics and partial order (numpy fp32 model; swish_ref = the contract's reference)")
    sys.path.insert(0, os.path.abspath(SWISH))
    import swish_ref as sr
    F32 = np.float32
    rng = np.random.default_rng(70)
    for M in ((70, 1000) if quick else (70, 1000, 9001)):
        ncu = 3
        nt = (M + BM - 1) // BM
        G = min(nt, ncu)
        x = sr.bf16_to_f32(sr.bf16_rne_bits(rng.standard_normal((M, NOUT)).astype(F32)))
        dy = sr.bf16_to_f32(sr.bf16_rne_bits((rng.standard_normal((M, NOUT)) * 1e-2).astype(F32)))
        mean = x.mean(axis=1, dtype=np.float64).astype(F32)
        var = ((x.astype(np.float64) - mean[:, None]) ** 2).mean(axis=1)
        rstd = (1.0 / np.sqrt(var + 1e-5)).astype(F32)
        w = (1.0 + 0.2 * rng.standard_normal(NOUT)).astype(F32)
        b = (0.3 * rng.standard_normal(NOUT)).astype(F32)
        R = sr.rows(x, dy, mean, rstd, w, b)  # dict of per-row intermediates (fp32), the contract's op list
        t_, xhat, dxf = R["t"], R["xhat"], R["dxf"]
        # kernel order: lane slot (w, sub) of WG b folds rows row0 + 16u + 2w + sub for tiles b, b+G, ... and u = 0..3
        P = np.zeros((3, G, NOUT), dtype=F32)
        for bb in range(G):
            acc = np.zeros((16, 3, NOUT), dtype=F32)
            j = 0
            while static_tile(bb, j, G) < nt:
                t = static_tile(bb, j, G)
                for u in range(NSLOT):
                    for wv in range(NWAVE):
                        for sub in range(2):
                            r = t * BM + 16 * u + 2 * wv + sub
                            sl = 2 * wv + sub
                            if r < M:
                                acc[sl, 0] = sr.fma32(t_[r], xhat[r], acc[sl, 0])
                                acc[sl, 1] = (acc[sl, 1] + t_[r]).astype(F32)
                                acc[sl, 2] = (acc[sl, 2] + dxf[r]).astype(F32)
                            else:  # padded row: the kernel computes it with x = dy = mean = rstd = 0 -> +0 terms
                                z = sr.rows(np.zeros((1, NOUT), F32), np.zeros((1, NOUT), F32), np.zeros(1, F32),
                                            np.zeros(1, F32), w, b)
                                acc[sl, 0] = sr.fma32(z["t"][0], z["xhat"][0], acc[sl, 0])
                                acc[sl, 1] = (acc[sl, 1] + z["t"][0]).astype(F32)
                                acc[sl, 2] = (acc[sl, 2] + z["dxf"][0]).astype(F32)
                j += 1
            v = [acc[k] for k in range(16)]
            h = 1
            while h < 16:
                for k in range(0, 16, 2 * h):
                    v[k] = (v[k] + v[k + h]).astype(F32)
                h *= 2
            P[:, bb] = v[0]
        out = np.zeros((3, NOUT), dtype=F32)
        for g in range(G):
            out = (out + P[:, g]).astype(F32)
        ex = sr.fp64_exact_sums(t_, xhat, dxf)
        err = [float(np.max(np.abs(out[o].astype(np.float64) - ex[o]))) for o in range(3)]
        scale = [float(np.max(np.abs(ex[o]))) for o in range(3)]
        rel = [e / max(s, 1e-30) for e, s in zip(err, scale)]
        # INFORMATIONAL (fix pass, robustness review R12): any fp32 summation order passes a 1e-5 bound, so this line
        # is accuracy context, not evidence of the kernel's order. The bitwise evidence of the order is the lane-level
        # emulator's PART comparison (../emu, re-run in build/emu_lane) and isa_check C15 D3 on the compiled code.
        print(f"  [INFO] M {M} (G {G}, {nt} tiles, last tile {tile_rows(nt - 1, M)} rows): partials in the documented "
              f"order vs fp64: max|err|/max|exact| dW %.2e dB %.2e dXS %.2e" % tuple(rel))
    z = sr.rows(np.zeros((1, NOUT), F32), np.zeros((1, NOUT), F32), np.zeros(1, F32), np.zeros(1, F32), w, b)
    zero_bits = all(int(np.sum(sr.bits(z[k]) != 0)) == 0 for k in ("t", "dxf")) and \
        int(np.sum(sr.bits(sr.fma32(z["t"][0], z["xhat"][0], np.zeros(NOUT, F32))) != 0)) == 0
    # INFORMATIONAL (R12): this exercises the contract reference swish_ref on a zero row, not the kernel; the kernel's
    # padded rows are checked bitwise by the lane-level emulator (cases with M % 64 != 0).
    print(f"  [INFO] contract reference on a padded row (x = dy = mean = rstd = 0): t, t*xhat, dx all +0: {zero_bits}")


def dump_index():
    """the index table of section 1, in the exact format of idx_harness.cpp (the host build of the .cu's functions):
    build_cpu.sh diffs the two, so a .cu edit that this mirror does not follow fails the CPU gate (R12)"""
    out = []
    for w in range(8):
        for l in range(64):
            s = (f"L {w} {l} {a_dma_row(w, l)} {a_dma_kswz(w, l)} {a_dma_voff(w, l)} {frag_lp0(l)} {frag_lp1(l)} "
                 f"{w_col(w, 1, l)} {w_voff(w, l)}")
            s += f" {rw_base(w, l)} {rr_base(w, l)} {e_voff(w, 0, l)} {acc_slot(w, l)} {c_col0(l)}"
            for u in range(4):
                s += f" {c_row(w, u, l)} {c_voff(w, u, l)} {e_voff(w, u, l)}"
            for n in range(2):
                for i in range(4):
                    s += f" {d_col(w, n, i, l)}"
            s += " " + "".join("1" if zfill_lane(r, w, l) else "0" for r in range(0, 65))
            out.append(s)
    for b in range(2):
        for k in range(4):
            for m in range(4):
                out.append(f"F {b} {k} {m} {frag_off(b, k, m)} {a_dma_lds(b, m, k)} {w_imm(m & 1, k, b)} {a_dma_soff(k)}")
    for M in (1, 63, 64, 65, 4095, 2492120, 4200000, 536870912):
        for t in (0, 1, (M - 1) // 64, (M - 1) // 64 + 1, (M - 1) // 64 + 256):
            out.append(f"T {M} {t} {tile_rows(t, M)} {rsrc_tile(t, tile_rows(t, M))} {static_tile(t % 7, t // 7, 7)}")
    out.append(f"K {top_vmcnt(0)} {top_vmcnt(1)} {LDS_BYTES} {red_index(2, 15, 255)} {red_index(1, 3, 7)}")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dump-index", action="store_true", help="print the index table for the idx_harness diff")
    a = ap.parse_args()
    if a.dump_index:
        dump_index()
        return 0
    t0 = time.time()
    e1_e6_a_tile()
    e2_e3_w_and_mfma()
    e4_redistribution()
    e5_global_offsets()
    e7_schedule()
    e8_lds_map()
    e9_banks()
    e10_numerics(a.quick)
    print(f"x2lb_emu: {'ALL PASS' if not FAILS else 'FAIL ' + str(FAILS)} ({time.time() - t0:.1f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
