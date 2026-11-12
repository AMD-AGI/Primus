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

# pyre-strict
"""DO8: the C8 attention-backward dO quantization (component A item A11) folded into L11's e5m2-du mode.

Gin item (DO8) (yambda_5b.gin, DLRM_HSTU_FP8_L11_DO8; only "1" arms it, "0" or any other value is off). Producer side
of R15's A11 contract (ops/flydsl/hstu_do8_registry.py, contract v1.2, 'plus' only); r16 iteration 36, track DO8E5.

WHAT. On an ordinary-layer backward call that item (39) L11 runs in its e5m2-du (B1a, item (38)(a)) mode, and whose
attention backward will quantize dO on the C8 do_blk path (the attention forward tagged a dO8 site on `out`; R15's
forward gate), ONE launch of a separate kernel, l11_outproj_lnmd_bwd_do8 (l11_do8.cu, its own .so), replaces L11's E5
launch. It writes everything L11's E5 launch writes (bf16 d(attn), B1a's e5m2 u block + pad rows, amax / u-colsum /
dW / dB partials), on sequence-aligned 64-token tiles, and in addition the attention backward's e4m3 dO (dO8 [M, 512])
and E8M0 group scales (E8 [M, 4] = 127 - e_g), byte-identical to hstu_do_fp8.quantize_do_e4m3_blk(d(attn),
seq_offsets, N) on every row that quantizer writes. The tiles come from a device work list (l11_do8_wl_build) built
from the attention call's own seq_offsets tensor (the site's), once per step: both ordinary layers receive the same
x_offsets object, so the second layer reuses the first layer's list (memo keyed on that object, its _version, numel,
dtype, device, M and the stream). After the last write to d(attn) the backward publishes (dO8, E8) on the site
(hstu_do8_registry.publish, bf16_written=True); the C8 attention backward takes them instead of running the quantizer
on the bf16 d(attn), or -- any failed check -- quantizes as before (same bytes).

WHY A SEPARATE .so (iter-35 review C-I1). L11's loader latches ALL of L11 off unless every instantiation in its .so has
0 VGPR spill / 0 scratch, and the E5 + emission instantiation crashes clang's greedy RA under L11's iterative-ilp
scheduler. So DO8 is its own translation unit with its own flags (no -amdgpu-sched-strategy: the default scheduler,
255 VGPR / 0 spill / 0 scratch in the iter-35 lab), cache (/tmp/dlrm_hstu_fp8_l11_do8/<key>), resource gate, prebuild
self-check and latch. Any DO8 failure (build, gate, self-check, launch, registry) latches only DO8; L11 keeps running
its own launches (and the attention backward its quantizer). l11_outproj_lnmd.cu / .py are untouched.

DO8-VD (r16 iteration 41, track do8vd). L11-VD's VALU diet (l11_outproj_lnmd_vd.cu: C1 bf16-domain keep masks, C2
one Newton step, C3 unsigned-max fallback predicate, C4 packed dW/dB chains, LANEFIX, PIN) is carried into DO8's
kernel body: l11_do8.cu is the diet source (impl tag "+vd(C1,C2,C3,C4,LANEFIX,PIN)"), l11_do8_novd.cu the pre-diet
source, byte for byte the DO8 source before the diet. The source follows L11's own selector
(l11_outproj_lnmd.vd_requested(), DLRM_HSTU_FP8_L11_VD == "1"), so DO8 and L11 always build the same consumer
variant; each source has its own cache key (the key covers the source bytes). The diet launch rejects a dscale
outside (0, FLT_MAX], as L11-VD's launches do.

DECIDE BEFORE EMISSION. `emission_site` is evaluated on host metadata only (no sync, no allocation, no launch) right
before the launch: knob on, prebuilt, not latched, L11 enabled and prebuilt, the call is in the e5m2-du mode (B1a's
plan), a site on attn, the consumer knob on (registry.take_enabled()), site T == M and (H, D) == (4, 128), seq_offsets
int32/int64, contiguous, on the device, B <= 8192. Otherwise L11's own E5 launch runs (no emission). The bf16-du mode
(B1a declined, i.e. its warm-up calls) has no DO8 variant: counted, never printed.

LINES (every rank, "[rank N]", ops.utils.arm_print_all_ranks, once each per process; only with the knob on). All start
with FAMILY. "armed OK: <FAMILY> prebuild before run_start ...", "armed OK: <FAMILY> ACTIVE mode=e5 plus ..." at the
first published record, "<FAMILY> NOT armed: <reason> ..." once per unexpected reason, "<FAMILY> DISABLED: ...",
"<FAMILY> INERT: ...", "<FAMILY> prebuild DEVICE FAULT: ...". Counters ("[l11-do8] counters ...") at events 1, 2, 4,
8, ... and once at exit.

DO8-ONLY (r16 iteration 42, track do8only; gin item (88) (DO8-ONLY), DLRM_HSTU_FP8_L11_DO8_ONLY == "1", read per call; R15
contract v2 'only', ops/flydsl/hstu_do8_registry.py). On a call DO8 emits for, after every decline check and the dz8 /
dO8 / E8 allocations (host only; the placeholder / partials allocations and the launch's checks follow it and are
covered by cancel), registry.plan(site, attn) decides: "only" -> the EM = 2 launch from a second .so (l11_do8.cu built
with -DL11DO8_ONLY=1: the same outputs minus the bf16 d(attn) store); autograd receives the launch's untouched d(attn)
buffer as a placeholder (NaN-filled with DLRM_HSTU_FP8_L11_DO8_ONLY_POISON=1), and publish() hands the record over with
bf16_written=False, re-raising any registry error (no exact fallback exists without the bf16 d(attn)). "plus" (and the
knob unset or "0": then the registry is not even asked) -> exactly the DO8 path above. A planned call that fails after
the plan is cancelled (registry.cancel) and falls back like any failed DO8 call (L11's own E5 launch writes the bf16
d(attn)); a failed EM = 2 kernel launch (rc > 0) latches only 'only', any other failure latches DO8 as in 'plus'. The
'only' .so has its own build, cache key, resource gate (EM = 2 instantiation, 0 VGPR spill, 0 scratch, LDS_BYTES) and
first-use self-check (EM = 2 vs EM = 1 on self_check's synthetic inputs: dO8, E8, dz8 and every partial byte-identical,
the EM = 2 d(attn) buffer's sentinel bytes untouched), built at prebuild after DO8's own self-check; any failure there
(or of an EM = 2 kernel launch) latches only 'only' (DO8 keeps its 'plus' launches). The pre-diet source
(l11_do8_novd.cu, DLRM_HSTU_FP8_L11_VD != "1") has no 'only' build: 'plus' (one INERT line). Lines start with
FAMILY_ONLY: "armed OK: ... only prebuild before run_start ...", "armed OK: ... only ACTIVE mode=e5 only ..." at the
first 'only' record, "... only DISABLED: ...", "... only INERT: ...", "... only prebuild DEVICE FAULT: ..."; counters
"[l11-do8] only counters: planned= only_launched= published= cancelled= fallback= plan_plus= latched=" at the arming
line, at the first cancel / fallback / plan_plus, at a latch and once at exit.

DO8 DIET (r16 iteration 52; gin item (89), DLRM_HSTU_FP8_L11_DO8_DIET == "1", read once, lazily). l11_do8.cu carries
the r16 iter52 VALU/SGPR diet of the emission tail (O1 packed u16 group maxima, O7(i) one SGPR row count per tile, O11,
v3b, O2 + E8 window, combine v6x) behind its preprocessor switch L11DO8_DIET (default 1). Knob "1": the build flags are
exactly the ones before this switch existed (no define; the .cu's default 1). Absent, "0" or any other value: the
flags gain -DL11DO8_DIET=0, which compiles ship14's (c68395d) kernel body in every guarded region, for the 'plus' and
the 'only' .so alike; the define is part of the flags, so each setting has its own cache key and .so. Same output
bytes either way (only the ISA differs). The pre-diet l11_do8_novd.cu has no such switch (the define is inert there).
The prebuild lines gain "; diet OFF (...)" only when the knob is not "1".

NIB (r16 iteration 54, gin item (94); knob DLRM_HSTU_FP8_LNMD_MASK_NIB, exactly "1" arms, default off; contract
ops/lnmd_mask_nib.py). NIB v2: the NIB keep-mask reader exists ONLY in the DO8-ONLY ('only', EM 2) build; the 'plus'
build is ALWAYS today's int8 build (its flags, cache key and .so are the knob-off build's). Why: the Leg L lab job 70100
(production ROCm 7.2.1) measured the NIB 'plus' build at SGPR spill 41 vs the int8 build's 38, which fails the
per-variant gate (iter53 VERIFY B2: <= base + 2); the NIB 'only' build passed (34 vs 33), and with DO8-ONLY every
steady-state DO8 call runs the 'only' .so. With the knob on, a source that has the L11_MKB_NIB switch (the text
"#if L11_MKB_NIB": l11_do8.cu; the pre-diet l11_do8_novd.cu has none, and a source that only names the macro does not
count) and NIB writers that can arm in this process (_nib_writers_off: item (28)(e)'s maskgen
DLRM_HSTU_FP8_LNMD_FWD_MASKGEN != "0", DLRM_HSTU_FP8_MASK16 == "1", DLRM_HSTU_FP8_LNMD_FWD_V2 != "0" and
DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL != "producer", read as triton_hstu_linear reads them; otherwise every keep mask is
int8), _flags adds -DL11_MKB_NIB=1 to the 'only' build (its own cache key): that kernel reads the LN*U*dropout keep
mask as uint8 [M, 256] (layout A, one dword per row-lane) instead of int8 [M, 512]; same keep decisions, same outputs
byte for byte; its impl string carries "+mkb=nib". mask_format() is the format of the LOADED 'plus' .so (always "int8":
_load raises on a "+mkb=nib" impl); only_mask_format() is that of the LOADED 'only' .so ("nib" | "int8"; "int8" when
nothing is loaded), which _load_only checks against the build flags (raise). nib_ready() (knob on; DO8 requested,
enabled and prebuilt; DO8-ONLY requested, its 'only' .so prebuilt, not latched and reading NIB; L11 enabled and
prebuilt) is what the NIB writers ask before they write a NIB mask.
ROUTING in run_e5_plus (never a latch on a format question: r16 iter53 VERIFY B1):
  - a NIB mask (lnmd_mask_nib.is_nib) is a counted DECLINE ("E5 mask format", one NOT armed line per process) before
    any allocation unless nib_ready() and its layout is uint8 [M, 256] row-contiguous, 16-B aligned; after the dz8 /
    dO8 / E8 allocations _plan_only asks the dO8 registry as for any DO8-ONLY call: "only" -> the NIB 'only' .so
    reads the mask as is; "plus" (or the 'only' fallback: no 'only' plan is open then) -> the same DECLINE (the 'plus'
    .so reads int8): L11's own E5 launch serves the call with the exact int8 unpack (lnmd_mask_nib.as_int8), without
    DO8's dO8/E8 emission;
  - an int8 mask while the loaded 'only' .so reads NIB (after a NIB writer latch, a writer launch that raised or the
    FP8 warmup's startup byte check, for the rest of the process; or per call wherever item (28)(e)'s maskgen does not
    arm on a call that reaches DO8, e.g. fewer than 2^20 rows: LNMD fwd v2's floor): no plan (_plan_only is not
    called); the int8 'plus' launch (dO8/E8 emitted, the bf16 d(attn) written), counted (counters() mask_int8_plus),
    one line per process;
  - any other mask (the knob off, or an int8 'only' .so): exactly the path without NIB.
The raw launch_e5_plus raises on a mask the .so it launches does not read. Calibration steps (bf16 LN, no FP8 y saved)
never reach DO8: L11 declines them at D4. The self-checks: self_check ('plus', int8) is ship15's; self_check_only
feeds a NIB 'only' launch lnmd_mask_nib.pack_ref(mask) and its EM = 1 reference ('plus', int8) the int8 mask, and also
requires unpack(pack_ref(mask)) == mask. The prebuild "armed OK" lines end with " mask=int8" ('plus', always) and
" mask=nib(uint8[M,256])" or " mask=int8" ('only'). At exit, knob on only, "[l11-do8] nib counters: only_mask_format=
int8_plus= nib_plus= mask_declined= fallback= final=1" follows the DO8 counters line (nib_summary_line; review N2).
PREBUILD FALLBACK (review R1-F4, v2): a NIB 'only' build that fails at prebuild (hipcc, DO8's resource gate, the
impl-format check, self_check_only) does not latch DO8-ONLY: one "<FAMILY>: LN*U*dropout keep mask NIB DISABLED"
line, the NIB 'only' .so is dropped, nib_build_requested() is False for the rest of the process (_nib_fallback), and
the int8 'only' build + self-check run instead (DO8 'plus' untouched; nib_ready() stays False, so the writers decline
N6). A self-check device fault still aborts; with DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 the failure raises as before. A
'plus' failure behaves exactly as without NIB (no NIB involved).

MX6 (r16 code row; gin item (104), DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6: exactly "1" arms it, read once at the
prebuild; code default OFF and bound OFF; site set DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6_SITES = top | bottom | all, code
default "top"). On a DO8 call the dO8 registry plans 'only', at a STUStack layer of the static site set, the out-proj
dgrad runs as E2M3 x E2M3 with E8M0 per-32 block scales: r14's producer (bwd_fp8.output_projection_dgrad_operands_mx6,
"S1": the G6 [M, 400] row records, the W6 / W6S weight shadow, the W3 dout_t operand and the e5m2 gradient site's
scale bookkeeping exactly once, returned as Mx6Operands) replaces the FP8 dual cast, and ONE launch of
l11_outproj_lnmd_bwd_do8_mx6 (EM 2) replaces the FP8 'only' launch; the record, its publish, the d(attn) placeholder,
B1a's record and the W3 wgrad are the 'only' path's. The kernel comes from a THIRD .so, libl11_do8_only_mx6.so:
l11_do8.cu with the 'only' build's exact flags plus -DL11DO8_MX6=1 LAST (its own cache key; the 'plus' / 'only'
commands and keys are unchanged), built only with the knob on.
SITES (static, per layer; critique C1): the forward (triton_hstu_linear._l11_mx6_forward_tag) captures this compute
output's STUStack layer (the LN*U*dropout layer tag modules/stu.py sets) on ctx when mx6_site_selected(layer): top =
layers >= 1 (the shipped stack: layer 1, whose L11 call is the first of the backward; E-RM, a separate r14 row, owns
the bottom), bottom = layer 0, all = every layer; untagged calls (warm-up, direct) never. Any other site value: INERT.
E-RM (r14's row, ops/triton/dy_erm.py; its site is layer 0's out-proj): MX6 never takes an E-RM site. A site set
holding layer 0 (bottom, all) while E-RM is REQUESTED (dy_erm.requested(), its knob; never its runtime state) is INERT
(fail closed), so beside E-RM MX6 runs the top set only; and the seam declines a call E-RM's forward tagged
(ctx.erm_site; unreachable under that rule; one NOT armed line while MX6 is armed, quiet otherwise), so the shipped
block takes the E-RM record, once. The W3 operand S1 returns must pass x21's own dout_t checks (step-tiled, contiguous,
16-B aligned): one x21 declined would turn W3's step tiling, and with it E-RM, off for the process.
INERT (one line each, nothing built): an invalid site value; a site set holding layer 0 while E-RM is requested; DO8 or
DO8-ONLY not prebuilt or latched; a source without the L11DO8_MX6 switch (the pre-diet l11_do8_novd.cu: DO8-VD off);
DIET off (the MX6 kernel is gated with the diet body only); the 'only' .so reading NIB while MX6_NIB_GATED is False
(the switch is True: MX6 + NIB passed its production resource gate and its GPU correctness ties, so a NIB 'only' .so
gets a NIB MX6 build); r14's producer API absent from bwd_fp8 (MX6_PRODUCER_API).
PREBUILD (after the 'only' prebuild): build + check_resources_mx6 (by mangled prefix: FP8 'only' twin _Z24 and MX6
'only' _Z28 at LDS 163,840, MX6 SGPR spill <= 63 and <= 36 for a NIB build, the two dy debug kernels at LDS 98,304,
0 VGPR spill / 0 scratch everywhere incl. the list builders; the FP8 check_resources never gates this .so) + load
(the twin's VGPRs / SGPR spill equal to the loaded 'only' .so's kernel; RTLD_LOCAL; l11do8_abi_version() == 2 is
authoritative, the impl text keeps "abi=1"; impl "EM 2 only" + the MX6 tag, "+mkb=nib" exactly when the 'only' .so
has it) + self_check_mx6 (the lattice chain tie: MX6 'only' vs the FP8 'only' twin of the same .so byte for byte,
plus a determinism twin). Any failure latches MX6 only.
DECIDE BEFORE THE CAST (the seam, triton_hstu_linear._l11_mx6_backward): mx6_preflight (host metadata only, incl.
critique C3's accepted configuration), then B1a's plan, DO8's emission site and the dO8 registry plan, each evaluated
ONCE (mx6_decide; critique C10) and handed to whichever path runs. 'plus' (calibration, debug readers) -> the FP8 path.
FALLBACK (R2): S1 declines or raises -> mx6_release (the 'only' plan cancelled; a raise also latches MX6; then the
line and the count), then the FP8 dual cast and today's chain. Any failure after S1 committed the scale state (MX6
operand checks, the MX6 launch rc > 0, the shared machinery, host work after the launch) -> run_e5_only_mx6 cancels
the plan, latches MX6 (shared-machinery failures also latch DO8, today's policy), then prints one rescue line and
counts it; the seam re-casts g8 with the DUAL kernel at the CAPTURED inv_scale (mx6_rescue ->
bwd_fp8.output_projection_g8_rescue, checked on the host like S1: no scale-state update, the R2 operands) and joins
today's chain at "DO8 returned None" (L11's own E5 launch with the same B1a plan, else the dgrad GEMM): never a
second DO8 attempt in the call (bitwise (DO8) off, the declared (39) order class). TRANSACTION: from registry.plan's
'only' answer to the hand-over, the Mx6Decision owns the open plan; every exit publishes it (success) or cancels it
(release, rescue, contract break, registry failure, any exception: mx6_decide's own guard, then the seam's
mx6_abandon), and the plan is cancelled before any latch, line or count.
LINES (knob on only; FAMILY_MX6 = FAMILY + " MX6"). At prebuild_mx6's success, before RUN_START, once per process
and in this order: "armed OK: <FAMILY_MX6> prebuild before run_start ...", "armed OK: <FAMILY_MX6> ACTIVE mode=e5 only
..." (ending "impl=<impl> device=<device> sites=<set> poison=<0|1>") and the counters line. An MX6 record prints
nothing (_mx6_published only counts it): no MX6 line in the timed region on the normal path. Other lines:
"<FAMILY_MX6> INERT: ...", "<FAMILY_MX6> DISABLED: <reason> -> FP8 dual cast + (DO8-ONLY) for this process",
"<FAMILY_MX6> prebuild DEVICE FAULT: ...", "<FAMILY_MX6> NOT armed: ..." (one per reason), "<FAMILY_MX6> release:
...", "<FAMILY_MX6> rescue: ..."; counters "[l11-do8] mx6 counters: decided= launched= published= rescued= released=
latched=" at arming, at the first rescue / release, at a latch and at exit (after the 'only' line, from _only_final).
Knob off: nothing is read but the knob, nothing is built, nothing is printed, and every existing function behaves
exactly as without this row.
"""

import atexit
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from generative_recommenders.ops import lnmd_mask_nib
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.utils import arm_print_all_ranks

KNOB: str = "DLRM_HSTU_FP8_L11_DO8"
SCHED_ENV: str = KNOB + "_SCHED"  # "default" (code default, the lab-measured flags) | "iterative-ilp" (diagnostic)
KERNEL_NAME: str = "l11_outproj_lnmd_bwd_do8"
BUILD_KERNEL_NAME: str = "l11_do8_wl_build"
FAMILY: str = "L11 DO8 dO8/E8 emission"
ITEM: str = "(DO8)"

_SOURCE: Path = Path(__file__).with_name("l11_do8.cu")  # DO8-VD: L11-VD's VALU diet in DO8's kernel body
_SOURCE_NOVD: Path = Path(__file__).with_name("l11_do8_novd.cu")  # the pre-diet DO8 source (L11-VD off)
_ARCH: str = "gfx950"
# L11's flags (l11_outproj_lnmd._FLAGS) WITHOUT "-mllvm -amdgpu-sched-strategy=iterative-ilp": the iter-35 lab built the
# E5 + emission instantiation only with the default scheduler (iterative-ilp crashed clang-22's greedy RA, e5a..e5f).
_BASE_FLAGS: Tuple[str, ...] = (
    "-DKITTENS_CDNA4",
    "-std=c++20",
    "-w",
    "-O3",
    "-ffp-contract=off",
    "--save-temps",
    "-shared",
    "-fPIC",
    "-Rpass-analysis=kernel-resource-usage",
)
_SCHED_FLAGS: Dict[str, Tuple[str, ...]] = {
    "default": (),
    "iterative-ilp": ("-mllvm", "-amdgpu-sched-strategy=iterative-ilp"),
}
# ROCm 10.2 (T14): flags for the compiler `hipcc --version` names, added by build_command (and hashed by cache_key)
# after _flags, which stays toolchain-free. AMD clang 24 (TheRock 10.2) schedules the three DO8 kernels with the
# generic register-pressure trackers and ends at 256 VGPR + VGPR spills ('plus' 8 VGPR / 36 B per lane of scratch,
# 'only' and the MX6 .so's FP8 twin 1 / 8 B), which the 0-spill / 0-scratch gate below rejects ("DISABLED"). With
# the AMDGPU-specific trackers the same source builds spill-free ('plus', 'only' and both MX6-.so FP8 / MX6 kernels
# 255-256 VGPR, 0 spill, 0 scratch, occupancy 2); a scheduling change only (same source: byte-identical outputs to
# the clang 22 build). AMD clang 22 (ROCm 7.2) builds clean without it and gets no extra flag: its commands, cache
# keys and code objects are exactly as before.
_TOOLCHAIN_FLAGS_MIN_CLANG: int = 24
_TOOLCHAIN_FLAGS: Tuple[str, ...] = ("-mllvm", "-amdgpu-use-amdgpu-trackers")
# ... and for the pre-diet kernel bodies (the l11_do8_novd.cu source: DLRM_HSTU_FP8_L11_VD != "1", the L11-VD opt-out
# and the unit tests' default; or l11_do8.cu with DLRM_HSTU_FP8_L11_DO8_DIET != "1"), which still spill 1-3 VGPRs with
# the trackers alone: MachineSink's cycle sinking moves their hoisted row offsets back into the persistent loop (248-249
# VGPR, 0 spill, 0 scratch). Not for the production diet build, which is clean without it (it would cost VALU). The
# diet-off MX6 .so still fails its gate (36 B scratch; with clang 22 its NIB SGPR-spill rule), as before: off on both.
_TOOLCHAIN_FLAGS_PREDIET: Tuple[str, ...] = ("-mllvm", "-sink-insts-to-avoid-spills")
_CACHE_ROOT: str = "/tmp/dlrm_hstu_fp8_l11_do8"
_LIB_NAME: str = "libl11_do8.so"
_BUILD_TIMEOUT_S: int = 300
_ABI_VERSION: int = 1
LDS_BYTES: int = 163840  # the 'plus' instantiation: L11's 96 KiB + the 64 KiB d(attn) tile staging
H: int = 4  # heads: lane >> 4 of the kernel's 64 lanes x 8 columns
HD: int = 128  # head dim (H * HD = 512 = L11's D)
B_MAX: int = 8192  # sequences per list build (1024 threads x 8)
# The production instantiation's template arguments in the mangled name (E5 1, WLIST 2, EM 1, RET 0, EMV 2, TM 0).
_PROD_MANGLE: str = "ILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi1ELi2ELi1ELi0ELi2ELi0EE"
# DO8-ONLY (r16 iter42): the producer knob of R15's contract v2 (== hstu_do8_registry.ONLY_ENV; the registry reads the
# same variable per call), its line family, the 'only' build of l11_do8.cu and its instantiation (EM 2).
ONLY_KNOB: str = KNOB + "_ONLY"
FAMILY_ONLY: str = FAMILY + " only"
ITEM_ONLY: str = "(DO8-ONLY)"
_ONLY_DEFINE: str = "-DL11DO8_ONLY=1"
_LIB_NAME_ONLY: str = "libl11_do8_only.so"
_ONLY_MANGLE: str = "ILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi1ELi2ELi2ELi0ELi2ELi0EE"
_POISON_BF16: int = 0x7FC0  # contract v2 debug knob (registry.poison_requested): a bf16 NaN in every placeholder word
_SENTINEL_BF16: int = 0x7FA5  # the 'only' self-check's prefill of the EM = 2 d(attn) buffer (a NaN payload)
# review_do8only P1: the attribute an EM = 2 placeholder carries (its site), so publish() can never send it out as 'plus'
_ONLY_PLACEHOLDER_ATTR: str = "_l11_do8_only_placeholder_site"
# DO8 DIET (r16 iter52, gin item (89)): the knob, and the define its "off" adds to every l11_do8 build ('plus' and
# 'only'); "1" adds nothing (the .cu defaults L11DO8_DIET to 1), so the default build command is unchanged.
DIET_KNOB: str = KNOB + "_DIET"
_DIET_OFF_DEFINE: str = "-DL11DO8_DIET=0"
# NIB (r16 iter54): the build switch of l11_do8.cu's packed keep-mask reader (NIB v2: the 'only' .so only), the text
# that marks a source as NIB-capable, the impl tag of a NIB build and the format names of the prebuild lines.
_NIB_DEFINE: str = "-DL11_MKB_NIB=1"
_NIB_MARKER: bytes = b"#if L11_MKB_NIB"  # the switch itself (review R2-F8): naming the macro is not enough
_NIB_IMPL_TAG: str = "+mkb=nib"
_MASK_DESC: Dict[str, str] = {"nib": "nib(uint8[M,256])", "int8": "int8"}
_MASK_DECLINE: str = "E5 mask format"  # run_e5_plus's decline reason prefix (decline key "E5 mask")
# MX6 (r16 code row, gin item (104); module docstring "MX6"): the knob, its static site set, the line family,
# the third .so (the 'only' build + -DL11DO8_MX6=1 LAST), its gate's mangled prefixes / limits, the operand sizes of
# l11_do8.cu's MX6 entry (G6 row records, W6 image, W6S scale plane) and r14's producer API in bwd_fp8.
MX6_KNOB: str = "DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6"
MX6_SITES_KNOB: str = MX6_KNOB + "_SITES"
MX6_SITES: Tuple[str, ...] = ("top", "bottom", "all")
MX6_SITES_DEFAULT: str = "top"  # the shipping form: the top site MX6, the bottom site E-RM's (a separate r14 row)
FAMILY_MX6: str = FAMILY + " MX6"
ITEM_MX6: str = "(104)"
# MX6 + NIB: the production resource gate of the MX6 + NIB 'only' build passed (SGPR spill within the NIB rule), and
# so did its GPU correctness ties: kitN job 72655 (production ROCm 7.2.1 builds of l11_do8.cu 04942aa1; kit seal
# SHA256SUMS 68af4496; PREREG fc3e9776 + AMEND1 781177c8 + AMEND2 902ae4e2) returned N_PASS, G-N0..G-N4 all PASS:
# G-N0 resources and FP8-twin identity; G-N1 MX6 + NIB on pack_ref(mask) == MX6 int8 byte for byte (dz8, dO8, E8,
# AMAXP / DWP / DBP / DUCS, guards; 25 cases); G-N2 the lattice tie inside MX6 + NIB; G-N3 FP8 + NIB == FP8 int8;
# G-N4 the pack round trip. True: MX6 builds with the 'only' .so's NIB decision. The constant stays as the switch:
# False makes MX6 INERT again whenever the 'only' .so reads the NIB keep mask (_mx6_inert_reason).
MX6_NIB_GATED: bool = True
MX6_KERNEL_NAME: str = KERNEL_NAME + "_mx6"
_MX6_DEFINE: str = "-DL11DO8_MX6=1"
_MX6_MARKER: bytes = b"#if L11DO8_MX6"  # the switch itself (a source that only names the macro has no MX6 kernel)
_LIB_NAME_MX6: str = "libl11_do8_only_mx6.so"
_ABI_VERSION_MX6: int = 2  # authoritative (the impl text of the MX6 build still reads "abi=1")
_MX6_IMPL_TAG: str = "+mx6(e2m3xe2m3,row400,opsel_k)"
_FP8_SYM: str = "_Z24" + KERNEL_NAME  # the FP8 kernels' mangled prefix (14 template arguments, the base's names)
_MX6_SYM: str = "_Z28" + MX6_KERNEL_NAME  # the MX6 kernels' (KERNEL_NAME is a substring: classify by prefix)
_DBG_MANGLE: str = "ILi0ELi2ELi0ELi0ELi1ELi0ELi1ELi0ELi0ELi2ELi0ELi0ELi0ELi0EE"  # the dy debug instantiation
LDS_BYTES_DBG: int = 98304  # the dy debug kernels: L11's 96 KiB (no d(attn) staging)
_MX6_SGPR_SPILL_MAX: int = 63  # one spill VGPR (64 lanes) at most
# NIB per-variant rule (the NIB build within 2 of its int8 twin): the int8 MX6 'only' build's production SGPR spill + 2
_MX6_NIB_SGPR_SPILL_MAX: int = 36
G6_ROW: int = 400  # bytes per G6 row record: 256 B heads | 128 B tails | 16 E8M0 bytes [q][k]
W6_BYTES: int = 589824  # the pre-swizzled W6 image of the [1536, 512] weight
W6S_BYTES: int = 768  # the W6 E8M0 scale plane [48][q][k]
# r14's producer API in ops/triton/bwd_fp8.py (S1 / W1; r16's S1 contract for r14): the stateful cast (S1), the FP8
# rescue (R2) and the stateless encode of the self-check; MX6 is INERT unless all three exist.
MX6_PRODUCER_API: Tuple[str, ...] = (
    "output_projection_dgrad_operands_mx6",
    "output_projection_g8_rescue",
    "output_projection_mx6_encode",
)


class OnlyLaunchError(RuntimeError):
    """DO8-ONLY (r15 review N1): the EM = 2 kernel launch itself failed (l11do8_e5_only_launch rc > 0). This is the one
    failure of an 'only' call that latches 'only' alone; any other failure of such a call (launch_e5_plus's checks,
    plan(cap), the drain, the placeholder / partials allocations, the host work after the launch) can happen to a
    'plus' call too and latches DO8, as it does there."""


class Mx6OperandError(RuntimeError):
    """MX6: an operand of the MX6 launch (G6 / W6 / W6S / the W3 operand / the scales r14's producer returned, or the
    keep-mask format of the MX6 .so) fails its check. MX6-specific: latches MX6 only (DO8 and DO8-ONLY keep running)."""


class Mx6LaunchError(RuntimeError):
    """MX6: the MX6 kernel launch itself failed (l11do8_e5_only_mx6_launch rc > 0). Latches MX6 only."""


class Mx6ContractError(RuntimeError):
    """MX6: r14's producer broke its state contract (the e5m2 gradient site's bookkeeping must happen exactly once on
    success and not at all on a decline or a raise; the result must be an Mx6Operands). Raised, never swallowed: the
    alternative is a silent second (or missing) scale-state update."""


class Mx6RescueArgs(NamedTuple):
    """MX6 (S1 contract): what bwd_fp8.output_projection_g8_rescue needs to repeat THIS call's FP8 dual cast byte for
    byte: quantize_row_major_dual(dout, inv_scale_used, t_dtype=t_dtype, t_inv_mul=t_inv_mul, t_limit=t_limit,
    return_partial=helpers_fused, t_tiled=t_tiled) with token_major the FP8 path's flag and t_shift the cast-time shift
    (t_inv_mul == 2^-t_shift for an e4m3 copy). Filled by r14's producer; opaque to the seam (it only hands it back)."""

    token_major: bool
    t_tiled: bool
    t_dtype: Optional[torch.dtype]
    t_inv_mul: float
    t_limit: Optional[float]
    t_shift: int
    helpers_fused: bool


class Mx6Operands(NamedTuple):
    """MX6 (S1 contract): r14's producer result for one call, after the e5m2 gradient site's bookkeeping.
    g6 uint8 [M, 400] row records (row stride 400, 16-B aligned, no padding rows); w6 uint8 [589824] and w6s uint8
    [768] (W1's shadow of the out-proj weight, 16-B aligned); dout_fp8 / dout_fp8_scale: the W3 wgrad operand and its
    decode scale, exactly what the FP8 dual cast hands W3 (step-tiled e4m3 [ceil(M/128), 512, 128]; valid on a rescue
    too); g_decode_scale fp32 [1]: the decode scale the FP8 dgrad would have used (the snapshot taken before any
    publish); inv_scale_used fp32 [1], its own 16-B-aligned allocation: the CAPTURED inv_scale bits the cast read,
    taken before any publish (the rescue casts with them); rescue_args: Mx6RescueArgs (opaque to the seam)."""

    g6: torch.Tensor
    w6: torch.Tensor
    w6s: torch.Tensor
    dout_fp8: torch.Tensor
    dout_fp8_scale: torch.Tensor
    g_decode_scale: torch.Tensor
    inv_scale_used: torch.Tensor
    rescue_args: Any


class Mx6Decision:
    """MX6: mx6_decide's answer for ONE backward call (critique C10: B1a's plan, DO8's emission site and the dO8
    registry plan are evaluated once and every path consumes this object). site: DO8's emission site (None: DO8 did
    not take the call, or it was not asked: no B1a plan). only_plan: None (the registry was not asked, or MX6's plan
    was closed again: it may be asked again), True (an 'only' plan is open for MX6) or False (the registry answered
    'plus': not asked again for this call). reason: None = MX6 takes the call, else why not.
    open: an 'only' plan owned by this decision is open (mx6_release / run_e5_only_mx6's failure path close it; a
    successful run_e5_only_mx6 hands it to publish()). layer: the STUStack layer the forward captured; rows: M."""

    __slots__ = ("site", "rows", "layer", "only_plan", "reason", "open")

    def __init__(self, site: Any, rows: int, layer: int, only_plan: Optional[bool], reason: Optional[str],
                 open_: bool = False) -> None:
        self.site = site
        self.rows = int(rows)
        self.layer = int(layer)
        self.only_plan = only_plan
        self.reason = reason
        self.open = bool(open_)

    @property
    def taken(self) -> bool:
        """MX6 takes the call: an open 'only' plan and no reason against it."""
        return self.only_plan is True and self.reason is None and self.open

    def __repr__(self) -> str:
        return (f"Mx6Decision(layer={self.layer}, rows={self.rows}, only_plan={self.only_plan}, open={self.open}, "
                f"reason={self.reason!r})")


_lock = threading.Lock()
_requested: Optional[bool] = None
_override: Optional[bool] = None
_diet: Optional[bool] = None  # DO8 DIET knob, read once (lazily, at the first build)
_diet_override: Optional[bool] = None
_broken: Optional[str] = None
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_fn_launch: Optional[Callable[..., int]] = None
_fn_wl_build: Optional[Callable[..., int]] = None
_fn_wl_cap: Optional[Callable[..., int]] = None
_fn_plan: Optional[Callable[..., int]] = None
_fn_drain: Optional[Callable[[], int]] = None
_impl: str = ""
_resources: Dict[str, Any] = {}
_build_secs: float = 0.0
_prebuilt: bool = False
_active_logged: bool = False
_once: set = set()
_plan_cache: Dict[int, Tuple[int, int, int, int]] = {}
_registry_mod: Any = None
_registry_tried: bool = False
# Work-list memo (one entry): the list of the latest seq_offsets object (weakref) + its key.
_WL_MEMO: Dict[str, Any] = {}
# Per-call engagement counters (per process).
_COUNTS: Dict[str, int] = {
    "calls": 0,
    "emitted": 0,
    "published": 0,
    "refused": 0,
    "failed": 0,
    "wl_built": 0,
    "wl_reused": 0,
    "bf16_mode": 0,
}
_DECLINED: Dict[str, int] = {}
_STATE: Dict[str, Any] = {"events": 0, "atexit": False}
# DO8-ONLY state: latch, prebuild flag, the 'only' .so, the in-flight 'only' call (its site and a weakref to its
# placeholder, from the launch to publish) and the producer-side counters.
_only_broken: Optional[str] = None
_only_prebuilt: bool = False
_only_lib: Optional[ctypes.CDLL] = None
_only_lib_path: Optional[Path] = None
_fn_launch_only: Optional[Callable[..., int]] = None
_impl_only: str = ""
_only_resources: Dict[str, Any] = {}
_only_logged: bool = False
_ONLY_CALL: Dict[str, Any] = {}
_ONLY_COUNTS: Dict[str, int] = {
    "planned": 0,
    "only_launched": 0,
    "published": 0,
    "cancelled": 0,
    "fallback": 0,
    "plan_plus": 0,
}
_ONLY_STATE: Dict[str, Any] = {"atexit": False}
# NIB (r16 iter54, review R1-F4): the reason the NIB 'only' build failed at prebuild (_nib_fall_back). Set: the int8
# 'only' build runs instead and nib_build_requested() is False for the rest of the process (nothing is latched).
_nib_fallback: Optional[str] = None
# NIB v2 (run_e5_plus): int8_plus = int8 keep masks run as DO8 'plus' (no plan) because the loaded 'only' .so reads NIB;
# nib_plus = NIB keep masks declined because DO8-ONLY ran the call as 'plus' (registry plan "plus" / 'only' fallback).
_NIB_COUNTS: Dict[str, int] = {"int8_plus": 0, "nib_plus": 0}
# MX6 state (module docstring "MX6"): the knobs (read once), the latch, the INERT reason, the prebuild flag, the third
# .so and its entries (the MX6 'only' launch and, for the self-check, the FP8 'only' twin of the same .so), the device
# 1.0 scales of the MX6 launch and the counters (decided: plans taken for MX6; launched: MX6 launches that returned;
# published: MX6 records published; rescued: calls rescued after the producer committed; released: plans released
# before it did). No once-flag for the ACTIVE line: prebuild_mx6 prints it, before RUN_START, once per process.
# _MX6_DECLINED counts the seam's preflight declines per reason key.
_mx6_requested: Optional[bool] = None
_mx6_override: Optional[bool] = None
_mx6_sites_read: bool = False
_mx6_sites_raw: Optional[str] = None
_mx6_sites_override: Optional[str] = None
_mx6_broken: Optional[str] = None
_mx6_inert: Optional[str] = None
_mx6_prebuilt: bool = False
_mx6_lib: Optional[ctypes.CDLL] = None
_mx6_lib_path: Optional[Path] = None
_fn_launch_mx6: Optional[Callable[..., int]] = None
_fn_launch_mx6_fp8: Optional[Callable[..., int]] = None
_impl_mx6: str = ""
_mx6_resources: Dict[str, Any] = {}
_MX6_ONE: Dict[str, torch.Tensor] = {}
_MX6_COUNTS: Dict[str, int] = {"decided": 0, "launched": 0, "published": 0, "rescued": 0, "released": 0}
_MX6_DECLINED: Dict[str, int] = {}


# ---------------------------------------------------------------------------- knob / latch / lines
def requested() -> bool:
    """DLRM_HSTU_FP8_L11_DO8 == "1" (read once, lazily: the gin env bootstrap runs after import)."""
    global _requested
    if _override is not None:
        return _override
    if _requested is None:
        _requested = os.environ.get(KNOB, "0") == "1"
    return _requested


def enabled() -> bool:
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def prebuilt() -> bool:
    return _prebuilt


def resources() -> Dict[str, Any]:
    return dict(_resources)


def impl() -> str:
    return _impl


def library_path() -> Optional[Path]:
    return _lib_path


def counters() -> Dict[str, Any]:
    out: Dict[str, Any] = dict(_COUNTS)
    out["declined"] = dict(_DECLINED)
    out["events"] = int(_STATE["events"])
    # NIB (r16 iter54): the loaded 'plus' / 'only' .so's keep-mask formats and the run_e5_plus "E5 mask format"
    # declines (also in "declined" under their key; the "[l11-do8] counters" line prints them there)
    out["mask_format"] = mask_format()
    out["mask_only_format"] = only_mask_format()
    out["mask_declined"] = int(_DECLINED.get(decline_key(_MASK_DECLINE), 0))
    out["mask_fallback"] = int(_nib_fallback is not None)  # the NIB 'only' build failed at prebuild -> int8 'only'
    out["mask_int8_plus"] = int(_NIB_COUNTS["int8_plus"])  # v2: int8 masks run as 'plus' (NIB 'only' .so loaded)
    out["mask_nib_plus"] = int(_NIB_COUNTS["nib_plus"])  # v2: NIB masks declined on a call run as 'plus'
    return out


def set_requested_for_testing(value: Optional[bool]) -> None:
    global _override, _requested
    _override = value
    _requested = None


def diet_requested() -> bool:
    """DLRM_HSTU_FP8_L11_DO8_DIET == "1" (surrounding whitespace ignored; read once, lazily: the first build runs after
    the gin env bootstrap). Absent, "", "0" or any other value: False -> every l11_do8 build adds -DL11DO8_DIET=0 (ship14's
    c68395d kernel body). The gin file binds it True (item (89))."""
    global _diet
    if _diet_override is not None:
        return _diet_override
    if _diet is None:
        _diet = os.environ.get(DIET_KNOB, "").strip() == "1"
    return _diet


def set_diet_for_testing(value: Optional[bool]) -> None:
    """Tests only: force the DIET knob (None = read the environment again). A library already loaded is not rebuilt:
    use reset_for_testing(unload=True) before the next _load()."""
    global _diet_override, _diet
    _diet_override = value
    _diet = None


def _diet_tail() -> str:
    """The prebuild lines' DIET note: empty with the knob on (the lines are unchanged), else names the OFF build."""
    if diet_requested():
        return ""
    return f"; diet OFF ({DIET_KNOB} != 1: {_DIET_OFF_DEFINE}, ship14 c68395d kernel body)"


def set_prebuilt_for_testing(value: bool) -> None:
    global _prebuilt
    _prebuilt = value


def reset_for_testing(unload: bool = False) -> None:
    """Tests only: clear the latch, once-lines, counters, memo, prebuilt flag; unload=True also forgets the library."""
    global _broken, _prebuilt, _active_logged, _registry_mod, _registry_tried, _nib_fallback
    global _lib, _lib_path, _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain, _impl
    _broken = None
    _nib_fallback = None
    _prebuilt = False
    _active_logged = False
    _registry_mod = None
    _registry_tried = False
    _once.clear()
    _plan_cache.clear()
    _WL_MEMO.clear()
    for k in _COUNTS:
        _COUNTS[k] = 0
    for k in _NIB_COUNTS:
        _NIB_COUNTS[k] = 0
    _DECLINED.clear()
    _STATE["events"] = 0
    _reset_only_for_testing(unload)
    _reset_mx6_for_testing(unload)
    if unload:
        with _lock:
            _lib = None
            _lib_path = None
            _fn_launch = _fn_wl_build = _fn_wl_cap = _fn_plan = _fn_drain = None
            _impl = ""
            _resources.clear()


def _reset_only_for_testing(unload: bool) -> None:
    global _only_broken, _only_prebuilt, _only_logged, _only_lib, _only_lib_path, _fn_launch_only, _impl_only
    _only_broken = None
    _only_prebuilt = False
    _only_logged = False
    _ONLY_CALL.clear()
    for k in _ONLY_COUNTS:
        _ONLY_COUNTS[k] = 0
    if unload:
        with _lock:
            _only_lib = None
            _only_lib_path = None
            _fn_launch_only = None
            _impl_only = ""
            _only_resources.clear()


def _fail_closed() -> bool:
    """DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 (smoke/debug only): raise instead of latching off."""
    return os.environ.get(KNOB + "_FAIL_CLOSED", "0") == "1"


def _print_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    arm_print_all_ranks(message)


def summary_line(final: bool = False) -> str:
    dec = ",".join(f"{k}:{v}" for k, v in sorted(_DECLINED.items()))
    c = _COUNTS
    return (
        f"[l11-do8] counters: events={_STATE['events']} calls={c['calls']} emitted={c['emitted']} "
        f"published={c['published']} refused={c['refused']} failed={c['failed']} wl_built={c['wl_built']} "
        f"wl_reused={c['wl_reused']} bf16_mode={c['bf16_mode']} declined={{{dec}}} "
        f"latched={int(_broken is not None)}" + (" final=1" if final else "")
    )


def nib_summary_line(final: bool = False) -> str:
    """NIB v2 (review N2): this process's run_e5_plus keep-mask routing counts. only_mask_format = the loaded 'only'
    .so's format; int8_plus = int8 masks run as DO8 'plus' (the 'only' .so reads NIB); nib_plus = NIB masks declined on
    a call DO8-ONLY ran as 'plus'; mask_declined = every "E5 mask format" decline (nib_plus included); fallback = the
    NIB 'only' build fell back to int8 at prebuild. Printed by _final after the DO8 counters line, knob on only."""
    return (
        f"[l11-do8] nib counters: only_mask_format={only_mask_format()} int8_plus={_NIB_COUNTS['int8_plus']} "
        f"nib_plus={_NIB_COUNTS['nib_plus']} mask_declined={int(_DECLINED.get(decline_key(_MASK_DECLINE), 0))} "
        f"fallback={int(_nib_fallback is not None)}" + (" final=1" if final else "")
    )


def _final() -> None:
    if _STATE["events"]:
        arm_print_all_ranks(summary_line(final=True))
        if lnmd_mask_nib.requested():  # NIB v2 (review N2): the routing counts at exit; knob off: nothing more
            arm_print_all_ranks(nib_summary_line(final=True))


def _event() -> None:
    """One per-call event (emitted / published / refused / declined / failed); log-spaced counter lines."""
    _STATE["events"] += 1
    n = int(_STATE["events"])
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_final)
    if n & (n - 1) == 0:
        arm_print_all_ranks(summary_line())


def _count(key: str, event: bool = True) -> None:
    _COUNTS[key] += 1
    if event:
        _event()


def _disable(reason: str) -> None:
    """Latch DO8 (only DO8) off for this process and print one DISABLED line."""
    global _broken
    if _fail_closed():
        raise RuntimeError(f"{FAMILY} failed with {KNOB}_FAIL_CLOSED=1: {reason}")
    if _broken is not None:
        return
    _broken = reason
    _COUNTS["failed"] += 1
    _event()
    arm_print_all_ranks(
        f"{FAMILY} DISABLED: {reason} -> item (39) L11's own launches + quantize_do_e4m3_blk for this process "
        "(L11 unaffected)"
    )


def decline_key(reason: str) -> str:
    tokens = reason.split()
    key = tokens[0] if tokens else reason
    if len(tokens) > 1:
        key += " " + re.split(r"[=:(\d]", tokens[1], maxsplit=1)[0]
    return key


def note_decline(reason: str, quiet: bool = False) -> None:
    """Count a declined call; one 'NOT armed' line per (reason key, process) unless quiet (expected states)."""
    k = decline_key(reason)
    _DECLINED[k] = _DECLINED.get(k, 0) + 1
    _event()
    if not quiet:
        _print_once(
            "decline:" + k,
            f"{FAMILY} NOT armed: {reason} -> item (39) L11's own launch; the C8 attention backward quantizes dO",
        )


def note_bf16_mode(ctx: Any) -> None:
    """The seam's bf16-du branch (B1a declined this call): DO8 has no variant there. Counted only when a dO8 site
    exists (an emission was possible in principle); never printed (an expected state: B1a's warm-up calls)."""
    if requested() and getattr(ctx, "do8_site", None) is not None:
        _COUNTS["bf16_mode"] += 1


# ---------------------------------------------------------------------------- DO8-ONLY knob / latch / lines
def only_requested() -> bool:
    """DLRM_HSTU_FP8_L11_DO8_ONLY == "1" (surrounding whitespace ignored), read per call like the registry's own read of
    the same variable (hstu_do8_registry.only_requested, which prints the REQUESTED / invalid-value lines). Absent, "",
    "0" or any other value: False, and DO8 runs exactly its 'plus' path."""
    return os.environ.get(ONLY_KNOB, "").strip() == "1"


def only_enabled() -> bool:
    """An 'only' launch is possible: the knob on, DO8 enabled, the 'only' .so prebuilt and not latched."""
    return only_requested() and enabled() and _only_prebuilt and _only_broken is None


def only_broken() -> Optional[str]:
    return _only_broken


def only_prebuilt() -> bool:
    return _only_prebuilt


def only_resources() -> Dict[str, Any]:
    return dict(_only_resources)


def only_impl() -> str:
    return _impl_only


def only_library_path() -> Optional[Path]:
    return _only_lib_path


def only_counters() -> Dict[str, Any]:
    out: Dict[str, Any] = dict(_ONLY_COUNTS)
    out["latched"] = int(_only_broken is not None)
    return out


def set_only_prebuilt_for_testing(value: bool) -> None:
    global _only_prebuilt
    _only_prebuilt = value


def only_summary_line(final: bool = False) -> str:
    c = _ONLY_COUNTS
    return (
        f"[l11-do8] only counters: planned={c['planned']} only_launched={c['only_launched']} "
        f"published={c['published']} cancelled={c['cancelled']} fallback={c['fallback']} plan_plus={c['plan_plus']} "
        f"latched={int(_only_broken is not None)}" + (" final=1" if final else "")
    )


def _only_final() -> None:
    if any(_ONLY_COUNTS.values()) or _only_broken is not None:
        arm_print_all_ranks(only_summary_line(final=True))
    _mx6_final()  # MX6 (item (104)): its own exit line after the 'only' one; knob off: nothing


def _only_atexit() -> None:
    if not _ONLY_STATE["atexit"]:
        _ONLY_STATE["atexit"] = True
        atexit.register(_only_final)


def _only_event(key: str) -> None:
    """Count one 'only' event; the counter line at the first cancel / fallback / plan_plus of the process."""
    _ONLY_COUNTS[key] += 1
    _only_atexit()
    if key in ("cancelled", "fallback", "plan_plus") and _ONLY_COUNTS[key] == 1:
        arm_print_all_ranks(only_summary_line())


def _disable_only(reason: str) -> None:
    """Latch DO8-ONLY (only 'only') off for this process and print one DISABLED line: DO8 keeps its 'plus' launches
    (EM = 1, the bf16 d(attn) written) and its own latch."""
    global _only_broken
    if _fail_closed():
        raise RuntimeError(f"{FAMILY_ONLY} failed with {KNOB}_FAIL_CLOSED=1: {reason}")
    if _only_broken is not None:
        return
    _only_broken = reason
    _only_atexit()
    arm_print_all_ranks(
        f"{FAMILY_ONLY} DISABLED: {reason} -> item {ITEM} 'plus' for this process (the bf16 d(attn) is written; "
        "DO8 and L11 unaffected)"
    )
    arm_print_all_ranks(only_summary_line())


# ---------------------------------------------------------------------------- registry (R15 hstu_do8_registry)
def registry() -> Any:
    """The consumer's registry module, or None when this tree has no R15 hook (one INERT line, knob on only)."""
    global _registry_mod, _registry_tried
    if _registry_tried:
        return _registry_mod
    _registry_tried = True
    try:
        from generative_recommenders.ops.flydsl import hstu_do8_registry as reg  # R15 A11 hook (contract v1.2)

        _registry_mod = reg
    except ImportError as error:
        _registry_mod = None
        if requested():
            _print_once(
                "inert-registry",
                f"{FAMILY} INERT: no ops/flydsl/hstu_do8_registry ({error}); R15's A11 consumer hook is not in this "
                "tree, so no dO8 site exists and nothing is emitted",
            )
    return _registry_mod


def site_of(attn: torch.Tensor) -> Any:
    """FORWARD (HSTUComputeOutputFunction.forward only, R15 contract rule (a)): the dO8 site tagged on `attn`, or None.
    Knob off: None, nothing imported, nothing printed."""
    if not requested():
        return None
    reg = registry()
    if reg is None:
        return None
    return reg.site_of(attn)


# ---------------------------------------------------------------------------- build / load
def sched() -> str:
    raw = os.environ.get(SCHED_ENV, "").strip() or "default"
    if raw not in _SCHED_FLAGS:
        _print_once("sched", f"{FAMILY}: {SCHED_ENV}={raw!r} is not one of {sorted(_SCHED_FLAGS)} -> 'default'")
        return "default"
    return raw


def _source() -> Path:
    """DLRM_HSTU_FP8_L11_DO8_SRC overrides the source (A/B of a candidate .cu only). Otherwise L11's own L11-VD
    selector decides: l11_do8.cu (the diet) when DLRM_HSTU_FP8_L11_VD == "1", else l11_do8_novd.cu (pre-diet)."""
    override = os.environ.get(KNOB + "_SRC", "").strip()  # A/B of a candidate .cu only
    if override:
        return Path(override)
    return _SOURCE if l11.vd_requested() else _SOURCE_NOVD


# ---------------------------------------------------------------------------- NIB keep-mask format (r16 iter54)
def _nib_writers_off() -> Optional[str]:
    """None = the NIB keep-mask writers can arm in this process; otherwise the knob that keeps every keep mask int8,
    in which case a NIB 'only' .so would never get a NIB mask (review R1-F3; NIB v2: every call would run 'plus'), so
    the int8 'only' .so is built. Host env only,
    read exactly as ops/triton/triton_hstu_linear.py reads it (names, defaults, tests): item (28)(e)'s maskgen
    (_use_lnmd_fwd_maskgen: DLRM_HSTU_FP8_LNMD_FWD_MASKGEN != "0"), mask16 (_use_mask16: DLRM_HSTU_FP8_MASK16 == "1"),
    the LN*U*dropout forward v2 (_use_lnmd_fwd_v2: DLRM_HSTU_FP8_LNMD_FWD_V2 != "0"), and the FP8 out-proj wgrad not in
    its "producer" mode (_FP8_OUT_WGRAD_IMPL: that forward writes its own int8 mask)."""
    env = os.environ
    if env.get("DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", "0") == "0":
        return "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN is off (item (28)(e)'s maskgen writes every NIB mask)"
    if env.get("DLRM_HSTU_FP8_MASK16", "0") != "1":
        return "DLRM_HSTU_FP8_MASK16 is not 1 (the maskgen needs mask16)"
    if env.get("DLRM_HSTU_FP8_LNMD_FWD_V2", "0") == "0":
        return "DLRM_HSTU_FP8_LNMD_FWD_V2 is off (the maskgen needs the v2 forward)"
    if env.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL", "nn") == "producer":
        return "DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL=producer (its forward writes the int8 mask)"
    return None


def _nib_src(src: bytes) -> bool:
    """The NIB reader 'only' build for THIS source: the knob DLRM_HSTU_FP8_LNMD_MASK_NIB == "1", no NIB fallback at
    this process's prebuild (_nib_fallback), the source has the L11_MKB_NIB switch ("#if L11_MKB_NIB": l11_do8.cu; not
    the pre-diet l11_do8_novd.cu) and the NIB writers can arm (_nib_writers_off). Knob off: False before any other
    read. NIB v2: _flags asks this for the 'only' build only."""
    return (
        lnmd_mask_nib.requested()
        and _nib_fallback is None
        and _NIB_MARKER in src
        and _nib_writers_off() is None
    )


def nib_build_requested(source: Optional[Path] = None) -> bool:
    """True = the next 'only' build of the selected source (`_source()`, or `source`) is the NIB reader build: the knob
    on, no NIB fallback in this process, the source text contains the switch "#if L11_MKB_NIB", and the NIB writers can
    arm (_nib_writers_off). Then _flags adds -DL11_MKB_NIB=1 to the 'only' build (the cache key covers the flags, so
    NIB and int8 builds never share a cache entry). NIB v2: the 'plus' build is int8 whatever this says. Knob off:
    False, no read."""
    if not lnmd_mask_nib.requested():
        return False
    try:
        return _nib_src((source or _source()).read_bytes())
    except OSError:
        return False


def _mask_format_of(impl_: str) -> str:
    return "nib" if _NIB_IMPL_TAG in impl_ else "int8"


def mask_format() -> str:
    """The keep-mask format of the LOADED 'plus' .so: "int8" (int8 [M, 512]; also when nothing is loaded). NIB v2: the
    'plus' build never gets -DL11_MKB_NIB=1 and _load raises on a "+mkb=nib" impl, so a loaded 'plus' .so never reads
    "nib" (uint8 [M, 256]). The 'only' .so's format: only_mask_format()."""
    if _lib is None:
        return "int8"
    return _mask_format_of(_impl)


def only_mask_format() -> str:
    """NIB v2: the keep-mask format of the LOADED 'only' .so: "nib" (uint8 [M, 256]; its impl contains "+mkb=nib":
    built while nib_build_requested(), checked in _load_only) or "int8" (int8 [M, 512]; also when nothing is loaded).
    The only DO8 .so that may read NIB, together with its MX6 build (item (104): the 'only' build's flags plus
    -DL11DO8_MX6=1, so the same format; _load_mx6 checks that mx6_mask_format() equals this one)."""
    if _only_lib is None:
        return "int8"
    return _mask_format_of(_impl_only)


def _nib_not_ready() -> Optional[str]:
    """None = nib_ready(); otherwise its first unmet condition (run_e5_plus's NIB decline names it). Host state only,
    in nib_ready()'s order; the knob first (off: nothing else is read)."""
    if not lnmd_mask_nib.requested():
        return f"{lnmd_mask_nib.KNOB} is not 1"
    if not (requested() and enabled() and prebuilt() and broken() is None):
        return "DO8 off, latched or not prebuilt"
    if not only_requested():
        return f"DO8-ONLY off ({ONLY_KNOB} != 1): only its 'only' .so reads NIB"
    if not (only_prebuilt() and only_broken() is None):
        return "the DO8-ONLY 'only' .so is not prebuilt or latched"
    if only_mask_format() != "nib":
        return "the loaded 'only' .so reads the int8 keep mask"
    if not (l11.enabled() and l11.prebuilt()):
        return "L11 item (39) not enabled/prebuilt"
    return None


def nib_ready() -> bool:
    """The NIB writers may write a NIB keep mask for a DO8 call (NIB v2: only the DO8-ONLY 'only' .so reads it, and
    with item (104) armed its MX6 build, whose format is the 'only' .so's: MX6_NIB_GATED): the knob on; DO8
    requested, enabled (not latched) and prebuilt (build + gate + self-check before RUN_START); DO8-ONLY requested
    (DLRM_HSTU_FP8_L11_DO8_ONLY == "1", read per call), its 'only' .so prebuilt, not latched and reading NIB
    (only_mask_format() == "nib"); L11 enabled and prebuilt (DO8 never runs without it). Host state only (cheap; the
    writers ask per maskgen call)."""
    return _nib_not_ready() is None


def _mask_mismatch(random_mask: Any, m: int, fmt: str) -> Optional[str]:
    """None = `random_mask` is what a DO8 .so of keep-mask format `fmt` reads for m rows: "nib" ->
    lnmd_mask_nib.layout_ok (uint8 [m, 256], row stride 256, 16-B aligned); "int8" -> L11's D8 layout (int8 [m, 512],
    row stride 512, 16-B aligned). Otherwise a description of the mismatch. Host metadata only (no sync, no
    allocation, no unpack)."""
    if fmt == "nib":
        if lnmd_mask_nib.layout_ok(random_mask, m):
            return None
        want = f"uint8 [{m}, {lnmd_mask_nib.ROW_BYTES}] (NIB)"
    else:
        if isinstance(random_mask, torch.Tensor) and l11._rows512(random_mask, torch.int8, m):
            return None
        want = f"int8 [{m}, {l11.D}]"
    if isinstance(random_mask, torch.Tensor):
        got = (f"{random_mask.dtype} {tuple(random_mask.shape)} stride {tuple(random_mask.stride())} "
               f"a16={int(random_mask.data_ptr() % 16 == 0)}")
    else:
        got = type(random_mask).__name__
    return f"keep mask {got}; the loaded {fmt} .so reads {want}, row-contiguous, 16-B aligned"


def _mask_decline(random_mask: Any, m: int, mask_nib: bool, only_nib: bool) -> Optional[str]:
    """NIB v2, run_e5_plus before any allocation: None = the .so this call may launch reads `random_mask`; otherwise
    the decline reason. mask_nib (lnmd_mask_nib.is_nib): only the NIB 'only' .so reads it -> nib_ready() and the NIB
    layout (uint8 [m, 256], row stride 256, 16-B aligned). Any other mask is for the int8 'plus' .so (or an int8 'only'
    .so): int8 [m, 512], row stride 512, 16-B aligned -- checked for every tensor mask, and for any mask while the NIB
    'only' .so is loaded (only_nib); with int8 .so only, a non-tensor stand-in is not checked (today's path: a failing
    launch latches DO8, as before). Host metadata only (no sync, no allocation, no unpack)."""
    if mask_nib:
        why = _nib_not_ready()
        if why is not None:
            return f"keep mask NIB uint8 {tuple(random_mask.shape)}; no NIB 'only' .so is ready ({why})"
        return _mask_mismatch(random_mask, m, "nib")
    if isinstance(random_mask, torch.Tensor) or only_nib:
        return _mask_mismatch(random_mask, m, "int8")
    return None


def _note_int8_plus(m: int) -> None:
    """NIB v2, run_e5_plus: an int8 keep mask while the loaded 'only' .so reads NIB (after a NIB writer latch, or on a
    call the NIB writers did not arm): the call runs DO8 'plus' (the int8 .so; _plan_only is not called, so the
    registry is not asked and cannot answer "only"). Counted (counters() mask_int8_plus), one line per process; not a
    decline (DO8 emits dO8/E8 as for any 'plus' call)."""
    _NIB_COUNTS["int8_plus"] += 1
    _print_once(
        "nib-int8-plus",
        f"{FAMILY}: {lnmd_mask_nib.FAMILY}: an int8 keep mask reached DO8 while its 'only' .so reads NIB (M={m}; a NIB "
        "writer latch, or a call the NIB writers did not arm) -> DO8 'plus' for such calls (the int8 .so: dO8/E8 "
        "emitted, the bf16 d(attn) written; no 'only' plan); counted (mask_int8_plus), one line per process",
    )


def _flags(src: bytes, only: bool = False, mx6: bool = False) -> List[str]:
    only = only or mx6  # MX6: always the 'only' build plus the switch (build(mx6=True)'s .so; no MX6 'plus' .so)
    sha = hashlib.sha256(src).hexdigest()
    # DO8 DIET (item (89)): knob off -> -DL11DO8_DIET=0 (c68395d body; before the 'only' define); on -> no define
    diet = [] if diet_requested() else [_DIET_OFF_DEFINE]
    flags = [f"--offload-arch={_ARCH}", *_BASE_FLAGS, *_SCHED_FLAGS[sched()], f'-DL11DO8_SRC_SHA256="{sha}"'] + diet + (
        [_ONLY_DEFINE] if only else []
    ) + ([_NIB_DEFINE] if only and _nib_src(src) else [])  # NIB v2: the 'only' build only, knob on ('plus' unchanged)
    # MX6 (item (104)): the third .so = these exact flags + -DL11DO8_MX6=1 LAST ('plus' / 'only' lists unchanged)
    return flags + [_MX6_DEFINE] if mx6 else flags


def toolchain_flags(toolchain: Optional[str] = None, source: Optional[Path] = None) -> List[str]:
    """ROCm 10.2 (T14): the extra hipcc flags of the compiler `toolchain` names (default: `hipcc --version` of the
    hipcc builds use): _TOOLCHAIN_FLAGS for AMD clang >= _TOOLCHAIN_FLAGS_MIN_CLANG (+ _TOOLCHAIN_FLAGS_PREDIET for a
    pre-diet kernel body: `source` is l11_do8_novd.cu, or the DIET knob is off), else none (AMD clang 22 / ROCm 7.2,
    or a version text without "AMD clang version N")."""
    if toolchain is None:
        toolchain = l11._toolchain(l11._hipcc())
    m = re.search(r"AMD clang version (\d+)\.", toolchain)
    if m is not None and int(m.group(1)) >= _TOOLCHAIN_FLAGS_MIN_CLANG:
        prediet = (source is not None and Path(source).name == _SOURCE_NOVD.name) or not diet_requested()
        return list(_TOOLCHAIN_FLAGS) + (list(_TOOLCHAIN_FLAGS_PREDIET) if prediet else [])
    return []


def stack_flags(toolchain: Optional[str] = None, source: Optional[Path] = None) -> List[str]:
    """ROCm 10.2: the flags the software stack (not the knobs) adds after _flags, for a clang hipcc (`hipcc --version`
    names "clang version"; a stand-in toolchain text gets none): toolchain_flags (T14, by hipcc) and L11's in-lane
    row-sum define (T14b, by Triton: l11_outproj_lnmd.tlsum_tree(), as in L11's own flags; DO8's self-check ties DO8 to
    L11's launch). Empty on the ROCm 7.2 / Triton 3.6 stack."""
    if toolchain is None:
        toolchain = l11._toolchain(l11._hipcc())
    if "clang version" not in toolchain:
        return []
    return toolchain_flags(toolchain, source) + ([l11.TLSUM_DEFINE] if l11.tlsum_tree() else [])


def build_command(source: Path, out: Path, only: bool = False, mx6: bool = False) -> List[str]:
    """The exact hipcc command of `build` (the lab build task reuses it to keep the --save-temps ISA). only=True: the
    DO8-ONLY .so (the same flags + -DL11DO8_ONLY=1). mx6=True (implies only=True, as in _flags, cache_key and build):
    the MX6 .so (the 'only' flags + -DL11DO8_MX6=1 last). ROCm 10.2: + stack_flags() after them."""
    return [l11._hipcc(), str(source.resolve()), *_flags(source.read_bytes(), only, mx6), *stack_flags(source=source),
            "-o", str(out.resolve())]


def cache_key(source: Optional[Path] = None, only: bool = False, mx6: bool = False) -> str:
    source = source or _source()
    src = source.read_bytes()
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(b"\0" + "\0".join(_flags(src, only, mx6)).encode())
    digest.update(b"\0" + _ARCH.encode())
    toolchain = l11._toolchain(l11._hipcc())
    digest.update(b"\0" + toolchain.encode())
    # ROCm 10.2: the stack flags too (only when there are any: every key without them is unchanged)
    extra = stack_flags(toolchain, source)
    if extra:
        digest.update(b"\0" + "\0".join(extra).encode())
    return digest.hexdigest()[:16]


def _cache_dir(source: Optional[Path] = None, only: bool = False, mx6: bool = False) -> Path:
    root = os.environ.get(KNOB + "_CACHE", "").strip() or _CACHE_ROOT
    return Path(root) / cache_key(source, only, mx6)


def check_resources(kernels: List[Dict[str, str]], mangle: str = _PROD_MANGLE) -> Dict[str, Any]:
    """DO8's own gate (L11's is untouched): the 'plus' kernel and the list builder must report 0 VGPR spill and 0
    scratch, and the 'plus' kernel exactly LDS_BYTES of LDS; raises otherwise (the caller latches DO8 off). DO8-ONLY:
    mangle=_ONLY_MANGLE gates the 'only' .so the same way (its single DO8 kernel must be the EM = 2 instantiation)."""
    mine = [k for k in kernels if KERNEL_NAME in k.get("Function Name", "")]
    wlb = [k for k in kernels if BUILD_KERNEL_NAME in k.get("Function Name", "")]
    if len(mine) != 1:
        raise RuntimeError(f"{len(mine)} {KERNEL_NAME} resource remarks in the build log (need exactly 1)")
    if len(wlb) < 1:
        raise RuntimeError(f"no {BUILD_KERNEL_NAME} resource remarks in the build log")
    for k in mine + wlb:
        spill = int(k.get("VGPRs Spill", "-1"))
        scratch = int(k.get("ScratchSize [bytes/lane]", "-1"))
        if spill != 0 or scratch != 0:
            raise RuntimeError(
                f"resource check: {k.get('Function Name')} VGPR spill {spill}, scratch {scratch} B/lane (need 0/0)"
            )
    p = mine[0]
    if mangle not in p.get("Function Name", ""):
        raise RuntimeError(f"resource check: unexpected instantiation {p.get('Function Name')}")
    lds = int(p.get("LDS Size [bytes/block]", "-1"))
    if lds != LDS_BYTES:
        raise RuntimeError(f"resource check: {KERNEL_NAME} LDS {lds} B (expected {LDS_BYTES})")
    return {
        "vgpr": int(p.get("VGPRs", "-1")),
        "sgpr": int(p.get("TotalSGPRs", "-1")),
        "vgpr_spill": int(p.get("VGPRs Spill", "-1")),
        "sgpr_spill": int(p.get("SGPRs Spill", "-1")),
        "scratch": int(p.get("ScratchSize [bytes/lane]", "-1")),
        "lds": lds,
        "wl_build_vgpr": max(int(k.get("VGPRs", "-1")) for k in wlb),
        "sched": sched(),
        "kernels": len(mine) + len(wlb),
    }


def build(source: Optional[Path] = None, only: bool = False, mx6: bool = False) -> Tuple[Path, Dict[str, Any]]:
    """Compile (cache hit: no compile) and return (library path, resource summary). Raises on any failure (missing
    hipcc, compile error or compiler crash, timeout, VGPR spill, scratch, LDS mismatch). Same race-safe cache protocol
    as L11's loader (per-host/pid/thread temp file, atomic renames, sidecar before the .so). only=True: the DO8-ONLY
    .so (-DL11DO8_ONLY=1, its own cache key, file name and gate on the EM = 2 instantiation). mx6=True: the MX6 .so
    (the 'only' build + -DL11DO8_MX6=1, its own cache key and file name, gated by check_resources_mx6 only; a NIB
    'only' build makes it a NIB build with the NIB SGPR rule)."""
    source = source or _source()
    if mx6:
        only = True  # the MX6 .so is the 'only' build plus the MX6 switch (no MX6 'plus' .so in production)
    name = _LIB_NAME_MX6 if mx6 else (_LIB_NAME_ONLY if only else _LIB_NAME)
    mangle = _ONLY_MANGLE if only else _PROD_MANGLE
    so = _cache_dir(source, only, mx6) / name
    meta = so.with_name(name + ".resources.json")

    def gate(kernels: List[Dict[str, str]]) -> Dict[str, Any]:
        if mx6:
            return check_resources_mx6(kernels, nib=_NIB_DEFINE in _flags(source.read_bytes(), True, True))
        return check_resources(kernels, mangle)

    if so.is_file() and meta.is_file():
        summary = json.loads(meta.read_text())
        gate(summary["kernels_raw"])
        return so, summary
    so.parent.mkdir(parents=True, exist_ok=True)
    tag = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    tmp = so.parent / f"{tag}-{name}"
    tmp_meta = so.parent / f"{tag}-{name}.resources.json"
    work = so.parent / f"{tag}-build"
    src = source.read_bytes()
    cmd = build_command(source, tmp, only, mx6)
    # ROCm 10.2: the stack flags, named in the error below and in the summary's (prebuild lines') sched=
    extra = "+".join(f.lstrip("-") for f in stack_flags(source=source) if f != "-mllvm")
    try:
        work.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, cwd=str(work))
        log = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode != 0 or not tmp.is_file():
            raise RuntimeError(f"hipcc rc={proc.returncode} (sched={sched()}{'+' + extra if extra else ''}): "
                               f"{log.strip()[-1200:]}")
        kernels = l11.parse_resource_remarks(log)
        summary = gate(kernels)
        if extra:
            summary["sched"] = f"{summary.get('sched')}+{extra}"
        summary["kernels_raw"] = kernels
        summary["src_sha256"] = hashlib.sha256(src).hexdigest()
        tmp_meta.write_text(json.dumps(summary))
        os.replace(tmp_meta, meta)
        os.replace(tmp, so)
    finally:
        for t in (tmp, tmp_meta):
            try:
                t.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(work, ignore_errors=True)
    return so, summary


def _bind(lib: ctypes.CDLL) -> None:
    global _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain, _impl
    abi = lib.l11do8_abi_version
    abi.restype = ctypes.c_int
    abi.argtypes = []
    got = int(abi())
    if got != _ABI_VERSION:
        raise RuntimeError(f"l11do8_abi_version()={got}, expected {_ABI_VERSION}")
    launch = lib.l11do8_e5_plus_launch
    launch.restype = ctypes.c_int
    launch.argtypes = (
        [ctypes.c_void_p] * 4  # G8, W8, sa, sb
        + [ctypes.c_int]  # M
        + [ctypes.c_void_p] * 9  # X, U, MK, MEAN, RSTD, WN, BN, DX, DZ8
        + [ctypes.c_int]  # rows_out
        + [ctypes.c_void_p] * 5  # INVS, AMAXP, DWP, DBP, DUCS
        + [ctypes.c_longlong]  # P
        + [ctypes.c_float, ctypes.c_float]  # dscale, invD
        + [ctypes.c_void_p] * 2  # WL, CNT
        + [ctypes.c_longlong]  # cap
        + [ctypes.c_void_p] * 2  # DO8, E8
        + [ctypes.c_void_p]  # stream
    )
    wlb = lib.l11do8_wl_build_launch
    wlb.restype = ctypes.c_int
    wlb.argtypes = [
        ctypes.c_void_p,  # OFFS
        ctypes.c_int,  # offs_i64
        ctypes.c_int,  # B
        ctypes.c_int,  # M
        ctypes.c_int,  # nprog
        ctypes.c_void_p,  # WL
        ctypes.c_longlong,  # cap
        ctypes.c_void_p,  # CNT
        ctypes.c_void_p,  # stream
    ]
    cap = lib.l11do8_wl_cap
    cap.restype = ctypes.c_longlong
    cap.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
    plan_ = lib.l11do8_plan
    plan_.restype = ctypes.c_int
    plan_.argtypes = [ctypes.c_longlong, ctypes.c_void_p]
    drain = lib.l11do8_drain_last_error
    drain.restype = ctypes.c_int
    drain.argtypes = []
    tag = lib.l11do8_impl
    tag.restype = ctypes.c_char_p
    tag.argtypes = []
    _impl = (tag() or b"").decode()
    _fn_launch, _fn_wl_build, _fn_wl_cap, _fn_plan, _fn_drain = launch, wlb, cap, plan_, drain


def _load() -> None:
    global _lib, _lib_path, _build_secs
    if _lib is not None:
        return
    with _lock:
        if _lib is not None:
            return
        t0 = time.perf_counter()
        so, summary = build()
        lib = ctypes.CDLL(str(so))
        _bind(lib)
        if _mask_format_of(_impl) != "int8":  # NIB v2: the 'plus' .so always reads the int8 keep mask
            raise RuntimeError(f"'plus' .so reads keep mask format {_mask_format_of(_impl)}; the 'plus' build is "
                               f"always int8 (impl {_impl[:200]!r})")
        _resources.clear()
        _resources.update({k: v for k, v in summary.items() if k != "kernels_raw"})
        _build_secs = time.perf_counter() - t0
        _lib_path = so
        _lib = lib


def _load_only() -> None:
    """DO8-ONLY: build (or reuse) + gate + load the 'only' .so and bind l11do8_e5_only_launch, whose C signature is
    l11do8_e5_plus_launch's (same argtypes, copied from the bound 'plus' entry: _load() runs first). The work list,
    plan and drain stay the 'plus' .so's. Raises on any failure (the caller latches 'only' only). NIB v2: the .so must
    read the keep-mask format its build flags requested (nib_build_requested(): "nib", else "int8"); a mismatch raises
    (prebuild_only's NIB fallback then rebuilds 'only' as int8)."""
    global _only_lib, _only_lib_path, _fn_launch_only, _impl_only
    if _only_lib is not None:
        return
    _load()
    with _lock:
        if _only_lib is not None:
            return
        assert _fn_launch is not None
        want = "nib" if nib_build_requested() else "int8"  # NIB v2: the format the 'only' build flags select
        so, summary = build(only=True)
        lib = ctypes.CDLL(str(so))
        abi = lib.l11do8_abi_version
        abi.restype = ctypes.c_int
        abi.argtypes = []
        got = int(abi())
        if got != _ABI_VERSION:
            raise RuntimeError(f"'only' .so: l11do8_abi_version()={got}, expected {_ABI_VERSION}")
        launch = lib.l11do8_e5_only_launch
        launch.restype = ctypes.c_int
        launch.argtypes = list(_fn_launch.argtypes)
        tag = lib.l11do8_impl
        tag.restype = ctypes.c_char_p
        tag.argtypes = []
        impl_ = (tag() or b"").decode()
        if "EM 2 only" not in impl_:
            raise RuntimeError(f"'only' .so impl {impl_[:200]!r} is not the EM 2 build")
        if _mask_format_of(impl_) != want:  # NIB v2: only_mask_format() must be what the build requested
            raise RuntimeError(f"'only' .so reads keep mask format {_mask_format_of(impl_)}, the build requested "
                               f"{want} (impl {impl_[:200]!r})")
        _only_resources.clear()
        _only_resources.update({k: v for k, v in summary.items() if k != "kernels_raw"})
        _impl_only = impl_
        _fn_launch_only = launch
        _only_lib_path = so
        _only_lib = lib


def wl_cap(m: int, b: int) -> int:
    """List capacity (entries) for m rows and b sequences, from the .so (cdiv(m, 64) + b + 2)."""
    assert _fn_wl_cap is not None
    cap = int(_fn_wl_cap(int(m), int(b)))
    if cap <= 0:
        raise RuntimeError(f"l11do8_wl_cap(M={m}, B={b}) = {cap}")
    return cap


def plan(cap: int) -> Tuple[int, int, int, int]:
    """(G, fl, nf, P) for a list of at most cap tiles, from the .so."""
    got = _plan_cache.get(cap)
    if got is not None:
        return got
    assert _fn_plan is not None
    out = (ctypes.c_longlong * 4)()
    rc = int(_fn_plan(int(cap), ctypes.addressof(out)))
    if rc != 0:
        raise RuntimeError(f"l11do8_plan(cap={cap}) rc={rc}")
    got = (int(out[0]), int(out[1]), int(out[2]), int(out[3]))
    if len(_plan_cache) < 4096:
        _plan_cache[cap] = got
    return got


# ---------------------------------------------------------------------------- decide before emission
def _offsets_ok(offs: Any, device: torch.device) -> Optional[str]:
    if not isinstance(offs, torch.Tensor):
        return "seq_offsets not a tensor"
    if offs.dim() != 1 or offs.dtype not in (torch.int32, torch.int64) or not offs.is_contiguous():
        return f"seq_offsets {offs.dtype} {tuple(offs.shape)} contiguous={offs.is_contiguous()}"
    if offs.device != device:
        return f"seq_offsets on {offs.device}, d(attn) on {device}"
    if not (1 <= offs.numel() <= B_MAX + 1):
        return f"seq_offsets B={offs.numel() - 1} outside [0, {B_MAX}]"
    if offs.data_ptr() % offs.element_size() != 0:
        return "seq_offsets misaligned"
    return None


def emission_site(ctx: Any, attn: torch.Tensor) -> Any:
    """DECIDE BEFORE EMISSION (host metadata only: no sync, no allocation, no launch), called by the seam right before
    L11's e5m2-du launch. Returns the dO8 site to emit for, or None (L11's own launch runs; nothing is published)."""
    if not requested():
        return None
    site = getattr(ctx, "do8_site", None)
    if site is None:
        # no site: the attention backward does not plan the C8 do_blk quantization for this call (or the consumer
        # knob was off at the forward) -- an expected state, not counted as a call
        return None
    _COUNTS["calls"] += 1
    if _broken is not None:
        note_decline(f"latched off: {_broken}", quiet=True)
        return None
    if not _prebuilt:
        note_decline("D11 not prebuilt before RUN_START (no build/self-check inside the timed run)")
        return None
    if not (l11.enabled() and l11.prebuilt()):
        note_decline("L11 item (39) not enabled/prebuilt", quiet=True)
        return None
    reg = registry()
    if reg is None:
        note_decline("no hstu_do8_registry", quiet=True)
        return None
    if not reg.take_enabled():
        note_decline("consumer knob DLRM_HSTU_FP8_BWD_DO8 off")
        return None
    m = int(attn.shape[0])
    if int(site.T) != m or (int(site.H), int(site.D)) != (H, HD) or int(attn.shape[-1]) != H * HD:
        note_decline(f"site T={site.T} H={site.H} D={site.D} vs d(attn) [{m}, {int(attn.shape[-1])}] (kernel: H=4, D=128)")
        return None
    if site.device != attn.device:
        note_decline(f"site device {site.device} vs d(attn) {attn.device}")
        return None
    why = _offsets_ok(site.seq_offsets, attn.device)
    if why is not None:
        note_decline(why)
        return None
    return site


# ---------------------------------------------------------------------------- device work list
def work_list(offs: torch.Tensor, m: int, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """(WL int32 [cap, 2], CNT int32 [2], cap) of the sequence-aligned 64-token tile list for m rows, built on the
    current stream by l11_do8_wl_build from `offs` (the attention's own seq_offsets tensor), or reused: one entry,
    keyed on the offs OBJECT (weakref identity: a freed and re-allocated offsets tensor never matches) and its
    data_ptr, _version, numel, dtype, device, m and the stream. Both ordinary layers pass the same x_offsets object,
    so the list is built once per step."""
    assert _fn_wl_build is not None
    stream = torch.cuda.current_stream(device)
    key = (offs.data_ptr(), offs._version, int(offs.numel()), str(offs.dtype), str(offs.device), int(m),
           int(stream.cuda_stream))
    ref = _WL_MEMO.get("ref")
    if ref is not None and ref() is offs and _WL_MEMO.get("key") == key:
        _COUNTS["wl_reused"] += 1
        return _WL_MEMO["wl"], _WL_MEMO["cnt"], int(_WL_MEMO["cap"])
    b = int(offs.numel()) - 1
    cap = wl_cap(m, b)
    wl = torch.empty((cap, 2), dtype=torch.int32, device=device)
    cnt = torch.empty((2,), dtype=torch.int32, device=device)
    nprog = max(1, min(128, (int(n) + 63) // 64))  # x4c19's grid (cdiv(N, 64)); longer sequences: grid-stride
    rc = int(
        _fn_wl_build(
            offs.data_ptr(), int(offs.dtype == torch.int64), b, int(m), nprog, wl.data_ptr(), cap, cnt.data_ptr(),
            stream.cuda_stream,
        )
    )
    if rc > 0:
        raise RuntimeError(f"l11do8_wl_build_launch rc={rc} (hipError) M={m} B={b}")
    _WL_MEMO.clear()
    _WL_MEMO.update(ref=weakref.ref(offs), key=key, wl=wl, cnt=cnt, cap=cap)
    _COUNTS["wl_built"] += 1
    return wl, cnt, cap


# ---------------------------------------------------------------------------- launch / run
def launch_e5_plus(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    inv_scale: torch.Tensor,
    dz8: torch.Tensor,
    wl: torch.Tensor,
    cnt: torch.Tensor,
    cap: int,
    do8: torch.Tensor,
    e8: torch.Tensor,
    only: bool = False,
    dx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One DO8 launch on the current stream into the caller's dz8 / do8 / e8 (raw: tests and the self-check keep the
    partials). Returns (dx, DWP, DBP, DUCS, AMAXP, P); raises on a launch error (rc > 0). `run_e5_plus` is the
    production entry. DO8-ONLY: only=True launches the EM = 2 instantiation of the 'only' .so (same outputs, dx NOT
    written: the returned dx is the untouched placeholder); dx = a caller buffer of attn's shape / dtype (the 'only'
    self-check prefills a sentinel), else a fresh torch.empty_like(attn) in both modes."""
    _load()
    if only:
        _load_only()
    fn = _fn_launch_only if only else _fn_launch
    assert fn is not None and _fn_drain is not None
    m = int(attn.shape[0])
    if dx is not None and not (dx.dtype == attn.dtype and tuple(dx.shape) == tuple(attn.shape) and dx.is_contiguous()
                               and l11._a16(dx) and dx.device == attn.device):
        raise RuntimeError(f"dx layout {dx.dtype} {tuple(dx.shape)} {dx.stride()} (attn {tuple(attn.shape)})")
    rows_out = int(dz8.shape[0])
    if not (dz8.dtype == torch.float8_e5m2 and dz8.dim() == 2 and int(dz8.shape[1]) == l11.DU_ROW
            and dz8.stride() == (l11.DU_ROW, 1) and l11._a16(dz8) and m <= rows_out < m + l11.E5_MAX_PAD):
        raise RuntimeError(f"dz8 layout {dz8.dtype} {tuple(dz8.shape)} {dz8.stride()} (M={m})")
    if not (do8.dtype == torch.float8_e4m3fn and tuple(do8.shape) == (m, l11.D) and do8.is_contiguous()
            and l11._a16(do8)):
        raise RuntimeError(f"do8 layout {do8.dtype} {tuple(do8.shape)} {do8.stride()} (M={m})")
    if not (e8.dtype == torch.uint8 and tuple(e8.shape) == (m, H) and e8.is_contiguous() and l11._a16(e8)):
        raise RuntimeError(f"e8 layout {e8.dtype} {tuple(e8.shape)} {e8.stride()} (M={m})")
    if not (wl.dtype == torch.int32 and tuple(wl.shape) == (cap, 2) and wl.is_contiguous()
            and cnt.dtype == torch.int32 and cnt.numel() == 2 and cnt.is_contiguous()):
        raise RuntimeError(f"work list layout {tuple(wl.shape)} {wl.dtype} / {tuple(cnt.shape)} {cnt.dtype}")
    # NIB (r16 iter54): the keep mask must be the format of the .so this launch uses (raw entry: raise; the production
    # run_e5_plus routes a NIB mask only to the NIB 'only' .so and declines any mismatch first, so this never latches)
    mm = _mask_mismatch(random_mask, m, _mask_format_of(_impl_only if only else _impl))
    if mm is not None:
        raise RuntimeError(f"mask format: {mm} (M={m})")
    G, fl, nf, P = plan(cap)
    device = attn.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stale = int(_fn_drain())
        if stale != 0:
            _print_once(f"stale:{stale}", f"{FAMILY}: stale HIP error {stale} drained before the launch (not ours)")
        stream = torch.cuda.current_stream(device).cuda_stream
        if dx is None:
            dx = torch.empty_like(attn)  # 'plus': the bf16 d(attn); 'only': the placeholder (not written)
        dwp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        dbp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        ducs = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        amaxp = torch.empty((G * 8,), dtype=torch.float32, device=device)
        rc = int(
            fn(
                g8.data_ptr(), w8.data_ptr(), sa.data_ptr(), sb.data_ptr(), m,
                attn.data_ptr(), u.data_ptr(), random_mask.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                norm_weight.data_ptr(), norm_bias.data_ptr(), dx.data_ptr(), dz8.data_ptr(), rows_out,
                inv_scale.data_ptr(), amaxp.data_ptr(), dwp.data_ptr(), dbp.data_ptr(), ducs.data_ptr(), int(P),
                l11.dropout_scale(dropout_ratio), 1.0 / l11.D, wl.data_ptr(), cnt.data_ptr(), int(cap),
                do8.data_ptr(), e8.data_ptr(), stream,
            )
        )
    if rc > 0:
        msg = f"l11do8_e5_{'only' if only else 'plus'}_launch rc={rc} (hipError) M={m}"
        raise OnlyLaunchError(msg) if only else RuntimeError(msg)  # N1: only the EM = 2 kernel's own rc latches 'only'
    if rc < 0:
        _print_once(f"stale:{-rc}", f"{FAMILY}: stale HIP error {-rc} drained at the launch (not ours; result used)")
    return dx, dwp, dbp, ducs, amaxp, P


def run_e5_plus(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
    site: Any,
    only_plan: Optional[bool] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]]:
    """PRODUCTION: l11_outproj_lnmd.run_e5's exact contract and return (dx, placeholder du, dweight, dbias,
    DuE5m2Emit) -- the same checks (l11._operands_ok, l11.e5_decline_reason), the same dz8 / record construction --
    plus (dO8 [M, 512] e4m3fn, E8 [M, 4] uint8) for `site` (from emission_site). None = declined (its own line) or
    failed (DO8 latched, DISABLED line): the caller then runs L11's own run_e5 on the SAME operands, which writes only
    fresh buffers too, so nothing of this attempt is visible.
    DO8-ONLY: after every decline check and the dz8 / dO8 / E8 allocations, `_plan_only` asks registry.plan(site, x)
    (x = the attn the forward received); the placeholder / partials allocations and launch_e5_plus's checks (layouts,
    plan(cap), the drain) follow it and are covered by cancel. "only" -> the EM = 2 launch, and dx is the untouched
    placeholder (poisoned on request) that publish() then hands over with bf16_written=False. A failure after an "only"
    plan (or with a plan left open on the site) cancels it (registry.cancel) before returning None (L11's own run_e5
    then writes the bf16 d(attn)). Only the EM = 2 kernel's own failure (rc > 0: OnlyLaunchError) latches 'only' alone;
    any other failure of the call can happen to a 'plus' call too and latches DO8, as there (r15 review N1).
    NIB v2 (module docstring, ROUTING): a NIB keep mask runs only as 'only' on the NIB 'only' .so (else a counted
    "E5 mask format" decline, never a latch); an int8 mask while the loaded 'only' .so reads NIB runs 'plus' without
    _plan_only (counted mask_int8_plus); any other mask: exactly the path without NIB.
    MX6 (item (104)): only_plan=False = the MX6 seam already asked the dO8 registry for this call (mx6_decide) and it
    answered 'plus': run 'plus' without asking again (critique C10: one plan, one count). None (the default and every
    other caller): _plan_only as above."""
    if not (enabled() and _prebuilt and l11.enabled() and l11.prebuilt()):
        return None
    m = int(x.shape[0])
    why = l11._operands_ok(g8, w8, sa, sb, m)
    only = launched = False
    try:
        from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

        grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
        pad = int(bwd_fp8.uvqk_tm_pad())
        if why is None:
            why = l11.e5_decline_reason(x, u, BLOCK_D, num_warps, silu_u, du_row_width, grad_dtype, limit, pad,
                                        inv_scale)
        if why is not None:
            note_decline("E5 " + why, quiet=True)  # L11's own run_e5 runs the same checks and prints its line
            return None
        # NIB v2 (r16 iter54; iter53 VERIFY B1: never a latch on a format question). Only the NIB 'only' .so reads a
        # NIB uint8 [M, 256] mask; the 'plus' .so (and an int8 'only' .so) reads the int8 [M, 512] one. A mask the .so
        # this call may launch cannot read is a counted DECLINE here, before the work list and every allocation --
        # never launch_e5_plus's raise, which the except below would turn into a DO8 latch. L11's own run_e5 then
        # serves the call (it unpacks a NIB mask exactly: lnmd_mask_nib.as_int8). See _mask_decline.
        mask_nib = lnmd_mask_nib.is_nib(random_mask)
        only_nib = only_mask_format() == "nib"
        mm = _mask_decline(random_mask, m, mask_nib, only_nib)
        if mm is not None:
            note_decline(f"{_MASK_DECLINE}: {mm}")
            return None
        if _lib is None:
            raise RuntimeError("library not loaded (no prebuild)")
        wl, cnt, cap = work_list(site.seq_offsets, m, int(site.N), x.device)
        rows_out = (m + pad - 1) // pad * pad
        dz8 = torch.empty((rows_out, du_row_width), dtype=grad_dtype, device=x.device)
        do8 = torch.empty((m, l11.D), dtype=torch.float8_e4m3fn, device=x.device)
        e8 = torch.empty((m, H), dtype=torch.uint8, device=x.device)
        if only_nib and not mask_nib:
            # NIB v2: an int8 mask while the loaded 'only' .so reads NIB -> DO8 'plus' (the int8 .so; only stays False)
            # without asking the registry (it could answer "only"); counted, one line per process. Review N1: drop a
            # stale 'only' call entry (a call that raised before its publish) as _plan_only does at every DO8-ONLY call
            if only_requested():
                _ONLY_CALL.clear()
            _note_int8_plus(m)
        else:
            # DO8-ONLY: contract v2 plan(); False with its knob off (N1: cancel covers ...); MX6: the seam's answer
            only = _plan_only(site, x) if only_plan is None else False
            if mask_nib and not only:
                # NIB v2: the registry answered "plus" (or the 'only' fallback): no 'only' plan is open, and the 'plus'
                # .so reads int8 -> the same counted DECLINE (nothing to cancel); L11's own run_e5 serves the call
                _NIB_COUNTS["nib_plus"] += 1
                note_decline(f"{_MASK_DECLINE}: keep mask NIB on a call DO8-ONLY runs as 'plus' (dO8 registry plan "
                             "'plus' or the 'only' fallback); only the 'only' .so reads NIB")
                return None
        dx, dwp, dbp, ducs, amaxp, P = launch_e5_plus(
            g8, w8, sa, sb, x, u, random_mask, mean, rstd, weight, bias, dropout_ratio, inv_scale, dz8, wl, cnt, cap,
            do8, e8, only=only,
        )
        launched = True
        if only and _registry_mod.poison_requested():
            dx.view(torch.int16).fill_(_POISON_BF16)  # contract v2 evidence knob: any hidden reader sees NaN
        dweight = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        l11.dwdb_finalize(dwp, dbp, dweight, dbias, P)
        du = hstu_du_e5m2.placeholder_du(u, m, l11.D)
        emit = hstu_du_e5m2.DuE5m2Emit(
            dz8=dz8,
            u_cols=(0, l11.D),
            u_amax=amaxp.amax(),
            u_colsum=ducs.sum(dim=0),
            inv_scale=inv_scale,
            cast_scale=cast_scale,
            rows=m,
            width=du_row_width,
            site_key=site_key,
            generation=generation,
            buffer_ptr=0,  # no d(uvqk) buffer handed over (placeholder du), as B1a and run_e5
        )
    except Exception as error:  # noqa: BLE001 -- latch DO8 (or 'only' only); the caller runs L11's own launch
        if only or getattr(site, "only_plan", None) is not None:
            # DO8-ONLY: this call will not publish 'only' (L11's own run_e5 writes the bf16 d(attn)): close the plan,
            # also one left open on the site by a raise inside _plan_only after registry.plan answered "only" (N1)
            _cancel_only(site)
        if only and isinstance(error, OnlyLaunchError):
            # the EM = 2 kernel launch itself failed (rc > 0): latch 'only' only; DO8 'plus' keeps running
            _disable_only(f"e5 only launch: {type(error).__name__}: {str(error)[:600]}")
            return None
        if only:
            # r15 review N1: launch_e5_plus's checks, plan(cap), the drain, the placeholder / partials allocations
            # (incl. OOM) or the host work after the launch: failures a 'plus' call can have too -> latch DO8, as 'plus'
            stage = "after" if launched else "before"
            _disable(f"e5 only call ({stage} the EM = 2 launch): {type(error).__name__}: {str(error)[:600]}")
            return None
        _disable(f"e5 plus: {type(error).__name__}: {str(error)[:600]}")
        return None
    _count("emitted")
    if only:
        _ONLY_CALL.update(site=site, dx=weakref.ref(dx))  # publish() hands this placeholder over as 'only'
        setattr(dx, _ONLY_PLACEHOLDER_ATTR, site)  # review_do8only P1: publish() never sends it out as 'plus'
        _only_event("only_launched")
    if not l11._active_e5_logged:
        # Item (39)'s own e5 ACTIVE line (its census marker), once per process, printed here because this call ran
        # L11's e5m2-du mode through the DO8 kernel (l11.run_e5, which prints it, did not run). Same text as
        # l11_outproj_lnmd.run_e5 plus a DO8 tail; impl= names the kernel that ran (DO8's own impl, "+vd(...)" only
        # for the DO8-VD build: r16 iter39 PLAN C4), not L11's .so. DO8-ONLY: an 'only' first call adds its build.
        l11._active_e5_logged = True
        only_tail = f"; item {ITEM_ONLY}: EM 2 build, no bf16 d(attn) store, impl={_impl_only}" if only else ""
        arm_print_all_ranks(
            f"armed OK: {l11.FAMILY} ACTIVE mode=e5 (B1a contract) ({l11.KNOB}, item (39), with item (38)(a) "
            f"DLRM_HSTU_FP8_BWD_DU_E5M2): one {l11.KERNEL_NAME} E5 launch replaces hipBLASLt F8B8BS NTB0 out-proj "
            f"dgrad + _ln_mul_dropout_bwd_dx_du_rng_v3 (B1a EMIT: e5m2 u block + pad rows of dz8, amax/colsum "
            f"partials; no bf16 du, dy never in HBM); impl={_impl} M={m} Mp={int(dz8.shape[0])} P={P} "
            f"device={x.device}; launched as {KERNEL_NAME} (item {ITEM}, + dO8/E8{only_tail})"
        )
    return dx, du, dweight, dbias, emit, do8, e8


def _plan_only(site: Any, attn: torch.Tensor) -> bool:
    """DO8-ONLY, contract v2 rule (1): True = this call runs the EM = 2 launch and publishes 'only'. Host only, after
    every decline check and the dz8 / dO8 / E8 allocations; the placeholder / partials allocations and launch_e5_plus's
    checks follow it and are covered by cancel (r15 review N1). Knob unset / "0" / other: False without touching the
    registry or any state (B3 + v2.1 exactly).
    Knob on: 'plus' without asking when the 'only' .so is not prebuilt / latched (counted fallback); else
    registry.plan(site, attn) ("plus" reasons are counted and printed by the registry: plan_plus here)."""
    if not only_requested():
        return False
    _ONLY_CALL.clear()  # a stale entry of a call that never published (it raised) is not this call's
    reg = registry()
    if reg is None or not (_only_prebuilt and _only_broken is None):
        _only_event("fallback")
        return False
    if reg.plan(site, attn) != "only":  # never raises
        _only_event("plan_plus")
        return False
    _only_event("planned")
    return True


def _cancel_only(site: Any) -> None:
    """Contract v2 rule (4): an "only" plan whose 'only' call did not complete (L11's own launch rewrites the call,
    bf16 d(attn) included): close it, so the attention backward binds as for any call without a record."""
    _ONLY_CALL.clear()
    reg = registry()
    if reg is not None and reg.cancel(site):
        _only_event("cancelled")


def publish(site: Any, dattn: torch.Tensor, do8: torch.Tensor, e8: torch.Tensor) -> bool:
    """BACKWARD, after the LAST write to dattn (the exact tensor returned as d(attn)), on the current stream: hand the
    record to R15's registry ('plus': bf16_written=True). True = published. A refusal is counted (the consumer then
    quantizes the bf16 d(attn): dO8/E8 exact) and never raises. A refused (or publish-latched) call is not
    bit-identical to (DO8) off: DO8's launch already ran, so its dW/dB/u-colsum keep DO8's tile-walk fp32 order
    (the declared (39) summation-order class); d(attn), dz8 and amax are bitwise.
    DO8-ONLY: when this site's call ran the EM = 2 launch (run_e5_plus), the record goes out as 'only' instead
    (_publish_only: bf16_written=False, every error re-raised)."""
    reg = registry()
    if reg is None or site is None:
        return False
    if _ONLY_CALL and _ONLY_CALL.get("site") is site:
        return _publish_only(reg, site, dattn, do8, e8)
    if getattr(dattn, _ONLY_PLACEHOLDER_ATTR, None) is not None:
        # review_do8only P1: an EM = 2 placeholder whose call is no longer the in-flight 'only' call (another DO8 call
        # planned between its launch and this publish). Fail closed: its bf16 values were never written, so a 'plus'
        # record (bf16_written=True) would let an M-E1 re-run or a debug reader quantize garbage. The raise aborts the
        # backward before the attention backward runs; were it swallowed, the still-open plan would make bind raise.
        raise RuntimeError(
            f"{FAMILY_ONLY}: publish got an EM = 2 placeholder that is not the in-flight 'only' call's; its bf16 values "
            "were never written, so it cannot be published as 'plus' (contract v2)"
        )
    try:
        rec = reg.publish(site, dattn, do8, e8, producer="L11", bf16_written=True)
    except Exception as error:  # noqa: BLE001 -- the registry never raises for 'plus'; be fail-safe anyway
        _disable(f"publish: {type(error).__name__}: {str(error)[:300]}")
        return False
    if rec is None:
        _count("refused")
        _print_once(
            "refused",
            f"{FAMILY} NOT armed: hstu_do8_registry.publish refused a record (its refused{{...}} counter names the "
            "check) -> the C8 attention backward quantizes the bf16 d(attn)",
        )
        return False
    _count("published")
    global _active_logged
    if not _active_logged:
        _active_logged = True
        G, _, _, P = plan(int(_WL_MEMO.get("cap", 0))) if _WL_MEMO.get("cap") else (0, 0, 0, 0)
        arm_print_all_ranks(
            f"armed OK: {FAMILY} ACTIVE mode=e5 plus ({KNOB}, item {ITEM}, with items (39) and (38)(a) and R15's "
            f"hstu_do8_registry): one {KERNEL_NAME} launch replaces l11_outproj_lnmd_bwd (E5) and also emits the C8 "
            f"attention backward's e4m3 dO + E8 (quantize_do_e4m3_blk skipped when taken; bf16 d(attn) still "
            f"written); impl={_impl} M={int(dattn.shape[0])} cap={_WL_MEMO.get('cap')} G={G} P={P} "
            f"device={dattn.device}"
        )
    return True


def _publish_only(reg: Any, site: Any, dattn: torch.Tensor, do8: torch.Tensor, e8: torch.Tensor) -> bool:
    """DO8-ONLY, contract v2 rules (2) and (5): after the LAST write (the EM = 2 launch emitted dO8/E8; nothing writes
    the placeholder), publish(..., bf16_written=False). Never swallowed: the bf16 d(attn) does not exist, so a failed
    'only' publish has no exact fallback; the registry's ValueError / Do8OnlyError (and a d(attn) that is not this
    call's placeholder) propagate out of the backward, which aborts before the attention backward runs (were one
    swallowed, the still-open plan would make bind raise)."""
    ref = _ONLY_CALL.get("dx")
    mx6 = _ONLY_CALL.get("mx6")  # MX6: run_e5_only_mx6's launch facts (never set with the MX6 knob off)
    _ONLY_CALL.clear()
    if ref is None or ref() is not dattn:
        raise RuntimeError(
            f"{FAMILY_ONLY}: publish got a d(attn) that is not the placeholder of this site's EM = 2 launch; its bf16 "
            "values were never written, so it cannot be published as 'plus' (contract v2)"
        )
    reg.publish(site, dattn, do8, e8, producer="L11", bf16_written=False)
    _count("published")
    _only_event("published")
    if mx6 is not None:
        _mx6_published()  # counted only: MX6's ACTIVE line printed at its prebuild (the FP8 'only' line is not MX6's)
        return True
    global _only_logged
    if not _only_logged:
        _only_logged = True
        G, _, _, P = plan(int(_WL_MEMO.get("cap", 0))) if _WL_MEMO.get("cap") else (0, 0, 0, 0)
        arm_print_all_ranks(
            f"armed OK: {FAMILY_ONLY} ACTIVE mode=e5 only ({ONLY_KNOB}, item {ITEM_ONLY}, with item {ITEM} and R15's "
            f"hstu_do8_registry contract v2): one {KERNEL_NAME} EM=2 launch emits the C8 attention backward's e4m3 "
            f"dO + E8 and SKIPS the bf16 d(attn) store (autograd receives a placeholder; poison="
            f"{int(bool(reg.poison_requested()))}); impl={_impl_only} M={int(dattn.shape[0])} cap={_WL_MEMO.get('cap')} "
            f"G={G} P={P} device={dattn.device}"
        )
        arm_print_all_ranks(only_summary_line())
    return True


# ---------------------------------------------------------------------------- self-check / prebuild
_SELF_CHECK_SEED = 20261010
# Sequence lengths of the self-check (cycled to >= 32,768 rows): every in-tile group boundary (1, 7, 8, 9, 15, 16, 17,
# 63, 64, 65), multi-tile sequences, a zero-length sequence, and lengths beyond one tile row of a WG.
_SELF_CHECK_LENS = (1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 0, 200, 640, 1000, 4096, 2288, 333)


def self_check() -> Tuple[bool, str]:
    """DO8 vs L11's own E5 launch and the stock quantizer, on seeded SYNTHETIC tensors (a private generator: no
    training data, no global RNG draw, no FP8 scale state, no registry). Gates: the work list (count, status, entries
    vs a host list); DO8's dx, dz8 u block + pad rows, guard columns (prefilled), amax bitwise vs L11's launch_e5 on
    the same operands; dO8 / E8 (prefilled) bitwise vs quantize_do_e4m3_blk(L11's dx) on every row; u colsum, dW, dB
    relL2 vs L11's <= 1e-5 (declared fp32 order); all-zero token groups included (E8 127); finite data (as L11's own
    self-check: a NaN row would make every dW/dB column NaN and the relL2 gates vacuous -- the NaN / Inf / saturation /
    subnormal matrix runs in the lab gate, staged/lab)."""
    from generative_recommenders.ops.flydsl.hstu_do_fp8 import quantize_do_e4m3_blk
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    dev = torch.device("cuda", torch.cuda.current_device())
    lens: List[int] = []
    while sum(lens) < 32768:
        lens.extend(_SELF_CHECK_LENS)
    M, Dm, p = sum(lens), l11.D, 0.1
    offs_h = [0]
    for n_ in lens:
        offs_h.append(offs_h[-1] + n_)
    offs = torch.tensor(offs_h, dtype=torch.int64, device=dev)
    N = max(lens)
    g = torch.Generator(device=dev)
    g.manual_seed(_SELF_CHECK_SEED)
    g8f = torch.randn((M, Dm), generator=g, device=dev) * 1024.0
    # zero groups: sequence 5's first 8 rows (G8 zero -> dy zero -> d(attn) zero rows -> e = 0, E8 127)
    s5 = offs_h[5]
    g8f[s5:s5 + min(8, lens[5])] = 0.0
    g8 = g8f.clamp_(-57344.0, 57344.0).to(torch.float8_e5m2)
    del g8f
    w8 = (torch.randn((l11.NOUT, Dm), generator=g, device=dev) * 90.0).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    sa = torch.full((1,), 2.0**-19, dtype=torch.float32, device=dev)
    sb = torch.full((1,), 0.0613 / 448.0, dtype=torch.float32, device=dev)
    x = (torch.randn((M, Dm), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    u = torch.randn((M, Dm), generator=g, device=dev).to(torch.bfloat16)
    wn = (1.0 + 0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    bn = (0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    xf = x.float()
    mean = xf.mean(dim=1)
    rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6)
    del xf
    mask = thl._create_dropout_mask(
        N=M, D=Dm, BLOCK_D=Dm, concat_u=True, concat_x=True, dropout_ratio=p, seed=_SELF_CHECK_SEED, device=dev,
    )
    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    pad = int(bwd_fp8.uvqk_tm_pad())
    rows_out = (M + pad - 1) // pad * pad
    inv = torch.full((1,), 0.7371 * 2.0**-3, dtype=torch.float32, device=dev)  # not a power of two: rounds
    # L11's own E5 launch (its .so, its flags), into a prefilled dz8
    dz8_s = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
    dx_s, dwp_s, dbp_s, ducs_s, amaxp_s, P_s = l11.launch_e5(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8_s.view(grad_dtype)
    )
    # DO8: the device list from this offsets tensor (raw: the production memo stays untouched), prefilled outputs
    b = len(lens)
    cap = wl_cap(M, b)
    wl = torch.full((cap, 2), -7, dtype=torch.int32, device=dev)
    cnt = torch.full((2,), -7, dtype=torch.int32, device=dev)
    assert _fn_wl_build is not None
    rc = int(_fn_wl_build(offs.data_ptr(), 1, b, M, max(1, min(128, (N + 63) // 64)), wl.data_ptr(), cap,
                          cnt.data_ptr(), torch.cuda.current_stream(dev).cuda_stream))
    if rc > 0:
        return False, f"l11do8_wl_build_launch rc={rc}"
    dz8_d = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
    do8 = torch.full((M, Dm), 0x5A, dtype=torch.uint8, device=dev)
    e8 = torch.full((M, H), 0xA5, dtype=torch.uint8, device=dev)
    dx_d, dwp_d, dbp_d, ducs_d, amaxp_d, P_d = launch_e5_plus(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8_d.view(grad_dtype), wl, cnt, cap,
        do8.view(torch.float8_e4m3fn), e8,
    )
    q8, qe8 = quantize_do_e4m3_blk(dx_s.view(M, H, HD), offs, N)
    dw_s = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    db_s = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    dw_d = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    db_d = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    l11.dwdb_finalize(dwp_s, dbp_s, dw_s, db_s, P_s)
    l11.dwdb_finalize(dwp_d, dbp_d, dw_d, db_d, P_d)
    torch.cuda.synchronize()

    def nbytes(a: torch.Tensor, c: torch.Tensor) -> int:
        a8 = a.contiguous().view(torch.uint8).reshape(-1)
        c8 = c.contiguous().view(torch.uint8).reshape(-1)
        if a8.numel() != c8.numel():
            return -1
        return int((a8 != c8).sum())

    def rel(a: torch.Tensor, c: torch.Tensor) -> float:
        a64, c64 = a.double(), c.double()
        fin = torch.isfinite(a64) & torch.isfinite(c64)
        return float((a64[fin] - c64[fin]).norm() / c64[fin].norm().clamp_min(1e-300))

    # host reference list (sequence-major, every tile, no gaps: offs[0] == 0, offs[B] == M)
    ref: List[Tuple[int, int]] = []
    for i, n_ in enumerate(lens):
        for t in range(0, n_, 64):
            ref.append((offs_h[i] + t, min(64, n_ - t)))
    cnt_h = cnt.tolist()
    wl_ok = cnt_h == [len(ref), 0] and wl[: len(ref)].tolist() == [list(e) for e in ref]
    dx_m = nbytes(dx_d, dx_s)
    dz_m = nbytes(dz8_d[:, :Dm], dz8_s[:, :Dm])
    guard = int((dz8_d[:, Dm:] != 0x5A).sum())
    amax_eq = bool(torch.equal(amaxp_d.amax().reshape(1).view(torch.int32), amaxp_s.amax().reshape(1).view(torch.int32)))
    do8_m = nbytes(do8, q8)
    e8_m = nbytes(e8, qe8)
    r_cs, r_dw, r_db = rel(ducs_d.sum(0), ducs_s.sum(0)), rel(dw_d, dw_s), rel(db_d, db_s)
    e8_127 = int((qe8 == 127).sum())
    finite = all(bool(torch.isfinite(t).all()) for t in (dx_d.float(), dw_d.float(), db_d.float(), ducs_d))
    detail = (
        f"M={M} B={b} cap={cap} tiles={cnt_h[0]} status={cnt_h[1]} P={P_d}: list ok {int(wl_ok)}; vs L11 E5 dx mism "
        f"{dx_m}, dz8 u+pad mism {dz_m}, guard bytes changed {guard}, amax eq {int(amax_eq)}; vs quantize_do_e4m3_blk "
        f"dO8 mism {do8_m}, E8 mism {e8_m} (E8=127 bytes {e8_127}); relL2 colsum {r_cs:.2e} dW {r_dw:.2e} dB "
        f"{r_db:.2e}; finite {int(finite)}"
    )
    ok = (
        wl_ok
        and dx_m == 0
        and dz_m == 0
        and guard == 0
        and amax_eq
        and do8_m == 0
        and e8_m == 0
        and r_cs <= 1e-5
        and r_dw <= 1e-5
        and r_db <= 1e-5
        and e8_127 > 0
        and finite
    )
    return ok, detail


def _self_check_mask(mask: torch.Tensor, fmt: str) -> Tuple[bool, torch.Tensor, bool]:
    """NIB (r16 iter54): (nib, the mask a DO8 .so of keep-mask format `fmt` reads, round trip ok). A NIB .so gets
    lnmd_mask_nib.pack_ref(mask) and the round trip unpack(pack_ref(mask)) == mask is checked (it also proves the
    synthetic mask has bits 3-7 clear, the writers' invariant); an int8 .so gets `mask` itself (no extra work). NIB v2:
    self_check_only asks it for the 'only' .so (only_mask_format()); its 'plus' reference keeps the int8 mask."""
    if fmt != "nib":
        return False, mask, True
    packed = lnmd_mask_nib.pack_ref(mask)
    return True, packed, bool(torch.equal(lnmd_mask_nib.unpack(packed), mask))


def _self_check_mask_detail(nib: bool, nib_rt: bool) -> str:
    """The self-check detail's NIB segment (appended; empty for an int8 .so, so its detail text is unchanged)."""
    return f"; mask {_MASK_DESC['nib']} unpack(pack_ref) exact {int(nib_rt)}" if nib else ""


def _nib_fall_back(why: str) -> bool:
    """NIB (r16 iter54, review R1-F4; v2), prebuild_only only: a failure of the NIB 'only' build (hipcc, DO8's resource
    gate, the impl-format check, self_check_only; a device fault re-raises before this is asked) does not latch
    DO8-ONLY. True = fall back: one "<FAMILY>: LN*U*dropout keep mask NIB DISABLED" line on every rank, the NIB 'only'
    .so state dropped, and nib_build_requested() False for the rest of the process (_nib_fallback), so the caller
    re-runs prebuild_only(), which builds and self-checks the int8 'only' .so: DO8-ONLY stays on, DO8 'plus' (int8 in
    v2, never NIB) is untouched, nib_ready() stays False and the NIB writers decline (N6). False = the failed build was
    not the NIB build (or it already fell back once), or DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1: the caller latches (or
    raises) exactly as before."""
    global _nib_fallback, _only_prebuilt, _only_lib, _only_lib_path, _fn_launch_only, _impl_only
    if _nib_fallback is not None or _fail_closed() or not nib_build_requested():
        return False
    _nib_fallback = why
    _only_prebuilt = False
    with _lock:
        _only_lib = None
        _only_lib_path = None
        _fn_launch_only = None
        _impl_only = ""
        _only_resources.clear()
    arm_print_all_ranks(
        f"{FAMILY}: {lnmd_mask_nib.FAMILY} DISABLED ({lnmd_mask_nib.KNOB}): the NIB reader build failed at prebuild "
        f"({why[:700]}) -> DO8-ONLY builds and self-checks its int8 keep-mask 'only' .so instead (DO8 'plus' reads "
        "int8 anyway; nothing is latched); nib_ready() stays False, so the NIB writers stay off for this process"
    )
    return True


def prebuild(l11_ready: bool, e5_expected: bool, self_check_fn: Optional[Callable[[], Tuple[bool, str]]] = None) -> bool:
    """Before RUN_START (triton_fp8.prebuild_extensions -> triton_hstu_linear.prebuild_l11_do8, AFTER L11's prebuild):
    arch check, build + load + DO8's resource gate, then the self-check. Knob off: no-op, no line. Without a prebuilt
    L11 or without B1a's e5m2-du mode: one INERT line, nothing built. Any build / load / gate / self-check failure
    latches DO8 off (DISABLED line; L11 untouched). A self-check exception whose text is a device fault re-raises
    (abort before RUN_START, L11's policy). NIB v2 (r16 iter54): the 'plus' .so is always the int8 build, so a 'plus'
    failure latches DO8 exactly as without NIB; its armed line ends " mask=int8"; a NIB 'only' build failure falls
    back to the int8 'only' build inside prebuild_only (review R1-F4).
    MX6 (item (104)): prebuild_mx6 runs after DO8's and DO8-ONLY's prebuild on every path that returns (its knob off:
    it returns at its first test, nothing read but the knob, nothing printed); it never changes DO8's result."""
    ok = _prebuild_do8(l11_ready, e5_expected, self_check_fn)
    prebuild_mx6()
    return ok


def _prebuild_do8(l11_ready: bool, e5_expected: bool,
                  self_check_fn: Optional[Callable[[], Tuple[bool, str]]] = None) -> bool:
    """prebuild()'s DO8 and DO8-ONLY part (the function body before the MX6 row, unchanged)."""
    global _prebuilt
    if not requested():
        return False
    if not l11_ready:
        _print_once(
            "inert-l11",
            f"{FAMILY} INERT: needs item (39) L11 prebuilt and enabled in this process "
            "(DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=1, prebuild OK); nothing built",
        )
        return False
    if not e5_expected:
        _print_once(
            "inert-e5",
            f"{FAMILY} INERT: needs item (38)(a) B1a (DLRM_HSTU_FP8_BWD_DU_E5M2=1, not latched): DO8 exists only for "
            "L11's e5m2-du mode; nothing built",
        )
        return False
    if _broken is not None:
        return False
    t0 = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("no ROCm GPU")
        arch = l11.device_arch(torch.cuda.current_device())
        if arch != _ARCH:
            raise RuntimeError(f"arch {arch or '<unknown>'} is not {_ARCH}")
        _load()
    except Exception as error:  # noqa: BLE001 -- build / gate / load: never a device fault, always latch DO8
        _disable(f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}")
        return False
    try:
        ok, detail = (self_check_fn or self_check)()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        msg = f"{type(error).__name__}: {error}"
        if l11._DEVICE_FAULT.search(msg):
            arm_print_all_ranks(f"{FAMILY} prebuild DEVICE FAULT: {msg[:600]}")
            raise
        _disable(f"prebuild self-check: {msg[:600]}")
        return False
    if not ok:
        _disable(f"prebuild self-check mismatch: {detail}")
        return False
    _prebuilt = True
    so_key = _lib_path.parent.name if _lib_path is not None else "?"
    arm_print_all_ranks(
        f"armed OK: {FAMILY} prebuild before run_start: built+loaded+self-check OK {time.perf_counter() - t0:.1f}s "
        f"(VGPR {_resources.get('vgpr')}, spill {_resources.get('vgpr_spill')}, scratch {_resources.get('scratch')}, "
        f"LDS {_resources.get('lds')}, SGPR spill {_resources.get('sgpr_spill')}, sched={_resources.get('sched')}; "
        f"so {so_key}; {detail}{_diet_tail()})"
        f" mask={_MASK_DESC[mask_format()]}"  # NIB (r16 iter54): appended segment ("int8": NIB v2's 'plus' is int8)
    )
    _note_nib_inert()  # NIB: requested, but nothing in this process will read a NIB mask -> one line naming why
    if only_requested():
        prebuild_only()  # DO8-ONLY: its own build / gate / self-check / latch; never changes DO8's own result
    return True


def _note_nib_inert() -> None:
    """NIB (r16 iter54; v2), prebuild after DO8's own armed line: the knob on (and no NIB fallback, which printed its
    own DISABLED line) but no DO8 .so of this process will read a NIB mask -> one line naming why; nib_ready() stays
    False and the NIB writers stay off. Knob off: nothing read, nothing printed."""
    if not lnmd_mask_nib.requested() or _nib_fallback is not None:
        return
    try:
        has_switch = _NIB_MARKER in _source().read_bytes()
    except OSError:
        has_switch = False
    off = _nib_writers_off()
    if not has_switch:
        # the selected source has no NIB reader (the pre-diet l11_do8_novd.cu: L11-VD off, or a _SRC override without
        # the switch): every DO8 .so reads int8, nib_ready() stays False, the writers stay off
        _print_once(
            "nib-inert",
            f"{FAMILY}: {lnmd_mask_nib.FAMILY} requested ({lnmd_mask_nib.KNOB}=1) but {_source().name} has no "
            "L11_MKB_NIB reader (needs DLRM_HSTU_FP8_L11_VD=1, ops/hip/l11_do8.cu) -> this .so reads the int8 "
            "keep mask; nib_ready() False, the NIB writers stay off",
        )
    elif off is not None:
        # review R1-F3: the NIB writers can never arm in this process, so the 'only' .so is built int8 as well
        _print_once(
            "nib-inert",
            f"{FAMILY}: {lnmd_mask_nib.FAMILY} requested ({lnmd_mask_nib.KNOB}=1) but the NIB writers cannot arm "
            f"in this process ({off}) -> this .so reads the int8 keep mask; nib_ready() False, the NIB writers "
            "stay off",
        )
    elif not only_requested():
        # NIB v2: only the DO8-ONLY 'only' .so reads NIB; with DO8-ONLY off nothing is built that could read it
        _print_once(
            "nib-inert",
            f"{FAMILY}: {lnmd_mask_nib.FAMILY} requested ({lnmd_mask_nib.KNOB}=1) but DO8-ONLY is off ({ONLY_KNOB} "
            "!= 1): only its 'only' .so reads NIB (NIB v2) -> this .so reads the int8 keep mask; nib_ready() False, "
            "the NIB writers stay off",
        )


# ---------------------------------------------------------------------------- DO8-ONLY self-check / prebuild
def self_check_only() -> Tuple[bool, str]:
    """DO8-ONLY first-use check: the EM = 2 launch vs the EM = 1 launch on self_check's seeded SYNTHETIC inputs (same
    recipe, seed and device list; no training data, no global RNG draw, no FP8 scale state, no registry). Both write
    into prefilled dz8 / dO8 / E8. Gates: dO8, E8, the WHOLE dz8 (u block, pad rows, untouched guard columns), the
    AMAXP / DWP / DBP / DUCS partials and P byte-identical; the EM = 2 d(attn) buffer, prefilled with a NaN sentinel,
    unchanged in every byte (no DX store); EM = 1's d(attn) finite and written, its dO8 written, E8 = 127 groups
    present. EM = 1 itself was just gated against L11's E5 launch and quantize_do_e4m3_blk by self_check on these
    inputs, so a pass makes EM = 2's dO8 / E8 the quantizer's bytes of the d(attn) it no longer stores. NIB v2: a NIB
    'only' .so (only_mask_format() == "nib") gets lnmd_mask_nib.pack_ref(mask), EM = 1 (the int8 'plus' .so) the int8
    mask, every comparison above unchanged, plus the gate unpack(pack_ref(mask)) == mask (detail segment
    "; mask nib(uint8[M,256]) unpack(pack_ref) exact 1")."""
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    dev = torch.device("cuda", torch.cuda.current_device())
    # ---- self_check's inputs, the same recipe and seed (keep in step with self_check)
    lens: List[int] = []
    while sum(lens) < 32768:
        lens.extend(_SELF_CHECK_LENS)
    M, Dm, p = sum(lens), l11.D, 0.1
    offs_h = [0]
    for n_ in lens:
        offs_h.append(offs_h[-1] + n_)
    offs = torch.tensor(offs_h, dtype=torch.int64, device=dev)
    N = max(lens)
    g = torch.Generator(device=dev)
    g.manual_seed(_SELF_CHECK_SEED)
    g8f = torch.randn((M, Dm), generator=g, device=dev) * 1024.0
    s5 = offs_h[5]
    g8f[s5:s5 + min(8, lens[5])] = 0.0
    g8 = g8f.clamp_(-57344.0, 57344.0).to(torch.float8_e5m2)
    del g8f
    w8 = (torch.randn((l11.NOUT, Dm), generator=g, device=dev) * 90.0).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    sa = torch.full((1,), 2.0**-19, dtype=torch.float32, device=dev)
    sb = torch.full((1,), 0.0613 / 448.0, dtype=torch.float32, device=dev)
    x = (torch.randn((M, Dm), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    u = torch.randn((M, Dm), generator=g, device=dev).to(torch.bfloat16)
    wn = (1.0 + 0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    bn = (0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    xf = x.float()
    mean = xf.mean(dim=1)
    rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6)
    del xf
    mask = thl._create_dropout_mask(
        N=M, D=Dm, BLOCK_D=Dm, concat_u=True, concat_x=True, dropout_ratio=p, seed=_SELF_CHECK_SEED, device=dev,
    )
    # NIB v2 (r16 iter54): a NIB 'only' .so reads the packed mask (exact torch reference); its EM = 1 reference (the
    # 'plus' .so, int8) keeps the int8 mask, so every EM = 2 output is gated against the int8 path;
    # unpack(pack_ref(mask)) == mask is gated as well
    nib, mask_only, nib_rt = _self_check_mask(mask, only_mask_format())
    grad_dtype, _ = bwd_fp8.grad_dtype_and_limit()
    pad = int(bwd_fp8.uvqk_tm_pad())
    rows_out = (M + pad - 1) // pad * pad
    inv = torch.full((1,), 0.7371 * 2.0**-3, dtype=torch.float32, device=dev)
    b = len(lens)
    cap = wl_cap(M, b)
    wl = torch.full((cap, 2), -7, dtype=torch.int32, device=dev)
    cnt = torch.full((2,), -7, dtype=torch.int32, device=dev)
    assert _fn_wl_build is not None
    rc = int(_fn_wl_build(offs.data_ptr(), 1, b, M, max(1, min(128, (N + 63) // 64)), wl.data_ptr(), cap,
                          cnt.data_ptr(), torch.cuda.current_stream(dev).cuda_stream))
    if rc > 0:
        return False, f"l11do8_wl_build_launch rc={rc}"

    def launch(only: bool, dx: Optional[torch.Tensor]) -> Tuple[Any, ...]:
        dz8 = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
        d8 = torch.full((M, Dm), 0x5A, dtype=torch.uint8, device=dev)
        e8 = torch.full((M, H), 0xA5, dtype=torch.uint8, device=dev)
        mk = mask_only if only else mask  # NIB v2: the 'only' .so's format; the EM = 1 'plus' reference reads int8
        out = launch_e5_plus(g8, w8, sa, sb, x, u, mk, mean, rstd, wn, bn, p, inv, dz8.view(grad_dtype), wl, cnt, cap,
                             d8.view(torch.float8_e4m3fn), e8, only=only, dx=dx)
        return out + (dz8, d8, e8)

    dx_p, dwp_p, dbp_p, ducs_p, amaxp_p, P_p, dz_p, d8_p, e8_p = launch(False, None)
    sentinel = torch.full((M, Dm), _SENTINEL_BF16, dtype=torch.int16, device=dev)
    dx_buf = sentinel.clone().view(torch.bfloat16)
    dx_o, dwp_o, dbp_o, ducs_o, amaxp_o, P_o, dz_o, d8_o, e8_o = launch(True, dx_buf)
    torch.cuda.synchronize()

    def mism(a: torch.Tensor, c: torch.Tensor) -> int:
        a8 = a.contiguous().view(torch.uint8).reshape(-1)
        c8 = c.contiguous().view(torch.uint8).reshape(-1)
        return -1 if a8.numel() != c8.numel() else int((a8 != c8).sum())

    m_d8, m_e8, m_dz = mism(d8_o, d8_p), mism(e8_o, e8_p), mism(dz_o, dz_p)
    m_dw, m_db, m_cs, m_am = mism(dwp_o, dwp_p), mism(dbp_o, dbp_p), mism(ducs_o, ducs_p), mism(amaxp_o, amaxp_p)
    dx_changed = mism(dx_buf, sentinel.view(torch.bfloat16))
    same_buf = dx_o.data_ptr() == dx_buf.data_ptr()
    p_written = int((d8_p != 0x5A).sum())
    dx_p_ok = bool(torch.isfinite(dx_p.float()).all())
    e8_127 = int((e8_p == 127).sum())
    detail = (
        f"M={M} B={b} cap={cap} P={P_o}/{P_p}: EM=2 vs EM=1 dO8 mism {m_d8}, E8 mism {m_e8}, dz8 mism {m_dz} "
        f"(u+pad+guard), DWP mism {m_dw}, DBP mism {m_db}, DUCS mism {m_cs}, AMAXP mism {m_am}; EM=2 d(attn) "
        f"sentinel bytes changed {dx_changed} of {2 * M * Dm} (buffer returned {int(same_buf)}); EM=1 dO8 bytes "
        f"written {p_written}, d(attn) finite {int(dx_p_ok)}, E8=127 bytes {e8_127}"
    ) + _self_check_mask_detail(nib, nib_rt)
    ok = (
        P_o == P_p
        and m_d8 == 0
        and m_e8 == 0
        and m_dz == 0
        and m_dw == 0
        and m_db == 0
        and m_cs == 0
        and m_am == 0
        and dx_changed == 0
        and same_buf
        and p_written > 0
        and dx_p_ok
        and e8_127 > 0
        and nib_rt
    )
    return ok, detail


def prebuild_only(self_check_fn: Optional[Callable[[], Tuple[bool, str]]] = None) -> bool:
    """DO8-ONLY prebuild (prebuild() calls it after DO8's own prebuild passed, knob DLRM_HSTU_FP8_L11_DO8_ONLY == "1"):
    the 'only' .so (l11_do8.cu with -DL11DO8_ONLY=1), its resource gate (the EM = 2 instantiation, 0 VGPR spill, 0
    scratch, LDS_BYTES; the list builders 0 / 0) and self_check_only. Any failure latches 'only' only (one DISABLED
    line; DO8 keeps its 'plus' launches). The pre-diet source has no 'only' build: one INERT line, nothing built. A
    self-check device fault re-raises (DO8's policy: abort before RUN_START). NIB v2 (r16 iter54, review R1-F4): a
    failure of the NIB 'only' build (build / gate / impl format / self-check exception / mismatch) does not latch
    'only': _nib_fall_back prints one line, drops the NIB 'only' .so and makes nib_build_requested() False, and this
    function runs once more with the int8 'only' build (whose failure latches 'only' as before). DO8 'plus' (int8) is
    untouched either way."""
    global _only_prebuilt
    if not (only_requested() and _prebuilt and _broken is None) or _only_broken is not None:
        return False
    if _source() == _SOURCE_NOVD:
        _print_once(
            "only-inert-novd",
            f"{FAMILY_ONLY} INERT: needs DO8-VD (DLRM_HSTU_FP8_L11_VD=1, ops/hip/l11_do8.cu); the pre-diet "
            f"{_SOURCE_NOVD.name} has no 'only' build -> item {ITEM} 'plus' (the bf16 d(attn) is written); nothing built",
        )
        return False
    t0 = time.perf_counter()
    try:
        _load_only()
    except Exception as error:  # noqa: BLE001 -- build / gate / load: latch 'only' only
        why = f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}"
        if _nib_fall_back(f"'only' {why}"):  # NIB v2 (review R1-F4): the NIB 'only' build failed -> its int8 build
            return prebuild_only(self_check_fn)
        _disable_only(why)
        return False
    try:
        ok, detail = (self_check_fn or self_check_only)()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        msg = f"{type(error).__name__}: {error}"
        if l11._DEVICE_FAULT.search(msg):
            arm_print_all_ranks(f"{FAMILY_ONLY} prebuild DEVICE FAULT: {msg[:600]}")
            raise
        why = f"prebuild self-check: {msg[:600]}"
        if _nib_fall_back(f"'only' {why}"):  # NIB v2 (review R1-F4): the NIB 'only' check failed -> its int8 build
            return prebuild_only(self_check_fn)
        _disable_only(why)
        return False
    if not ok:
        why = f"prebuild self-check mismatch: {detail}"
        if _nib_fall_back(f"'only' {why}"):  # NIB v2 (review R1-F4): the NIB 'only' check failed -> its int8 build
            return prebuild_only(self_check_fn)
        _disable_only(why)
        return False
    _only_prebuilt = True
    _only_atexit()
    r = _only_resources
    so_key = _only_lib_path.parent.name if _only_lib_path is not None else "?"
    arm_print_all_ranks(
        f"armed OK: {FAMILY_ONLY} prebuild before run_start: built+loaded+self-check OK {time.perf_counter() - t0:.1f}s "
        f"(EM 2: VGPR {r.get('vgpr')}, spill {r.get('vgpr_spill')}, scratch {r.get('scratch')}, LDS {r.get('lds')}, "
        f"SGPR spill {r.get('sgpr_spill')}, sched={r.get('sched')}; so {so_key}; {detail}{_diet_tail()})"
        f" mask={_MASK_DESC[only_mask_format()]}"  # NIB (r16 iter54): appended segment, the 'only' .so's format
    )
    return True


# ============================================================================ MX6 (r16 code row, item (104))
# Module docstring "MX6". With the knob off none of this runs: prebuild_mx6 returns at its first test, and the seam's
# forward tag and backward dispatch (ops/triton/triton_hstu_linear.py, module end) ask mx6_enabled(), which stays False
# without an MX6 prebuild.


# ---------------------------------------------------------------------------- MX6 knobs / site set
def mx6_requested() -> bool:
    """DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6 == "1" (surrounding whitespace ignored; read once, lazily: the first read is
    prebuild_mx6's, after the gin env bootstrap, as diet_requested()). Absent, "", "0" or any other value: False. The
    gin binds it False (item (104)); a pre-set environment value wins (the kill switch and the A/B arm)."""
    global _mx6_requested
    if _mx6_override is not None:
        return _mx6_override
    if _mx6_requested is None:
        _mx6_requested = os.environ.get(MX6_KNOB, "").strip() == "1"
    return _mx6_requested


def set_mx6_for_testing(value: Optional[bool]) -> None:
    """Tests only: force the MX6 knob (None = read the environment again at the next ask)."""
    global _mx6_override, _mx6_requested
    _mx6_override = value
    _mx6_requested = None


def _mx6_sites_value() -> Optional[str]:
    """The raw DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6_SITES (None = unset), read once (with the knob, at the prebuild)."""
    global _mx6_sites_read, _mx6_sites_raw
    if _mx6_sites_override is not None:
        return _mx6_sites_override
    if not _mx6_sites_read:
        _mx6_sites_read = True
        _mx6_sites_raw = os.environ.get(MX6_SITES_KNOB)
    return _mx6_sites_raw


def set_mx6_sites_for_testing(value: Optional[str]) -> None:
    """Tests only: force the site-set value (None = read the environment again at the next ask)."""
    global _mx6_sites_override, _mx6_sites_read, _mx6_sites_raw
    _mx6_sites_override = value
    _mx6_sites_read = False
    _mx6_sites_raw = None


def mx6_sites() -> Optional[str]:
    """MX6's static site set: "top" (the code default, the variable unset), "bottom" or "all" (surrounding
    whitespace and case ignored); None for any other value, "" included (MX6 INERT: fail closed)."""
    raw = _mx6_sites_value()
    value = MX6_SITES_DEFAULT if raw is None else raw.strip().lower()
    return value if value in MX6_SITES else None


def mx6_site_selected(layer: int) -> bool:
    """The static per-layer site predicate (critique C1). `layer` is the STUStack layer index the FORWARD captured for
    this compute output (triton_hstu_linear._l11_mx6_forward_tag: the layer tag modules/stu.py STULayer._tag_lnmd_layer
    sets right before the call; -1 = untagged: a direct call or the FP8 first-use warmup). top = every stacked layer
    above the bottom one (layer >= 1; in the shipped stack that is layer 1, whose L11 call is the first one of the
    backward: the last layer runs targets-only and has no dO8 site), bottom = layer 0, all = every layer; an untagged
    call never. Never a per-call order and never another row's runtime state (an E-RM latch cannot move MX6)."""
    sites = mx6_sites()
    layer = int(layer)
    if sites is None or layer < 0:
        return False
    if sites == "top":
        return layer >= 1
    if sites == "bottom":
        return layer == 0
    return True


# ---------------------------------------------------------------------------- MX6 state / latch / lines
def mx6_enabled() -> bool:
    """An MX6 call is possible: built, gated, loaded and self-checked before RUN_START in this process (prebuild_mx6:
    the knob on, a valid site set and r14's producer API present) and not latched. Host state only: the forward's site
    tag and the backward's seam ask it; with the knob off (or INERT, or latched) it is False."""
    return _mx6_prebuilt and _mx6_broken is None


def mx6_broken() -> Optional[str]:
    return _mx6_broken


def mx6_prebuilt() -> bool:
    return _mx6_prebuilt


def mx6_inert() -> Optional[str]:
    """The INERT reason prebuild_mx6 found (None: not INERT, or not asked)."""
    return _mx6_inert


def mx6_impl() -> str:
    return _impl_mx6


def mx6_resources() -> Dict[str, Any]:
    return dict(_mx6_resources)


def mx6_library_path() -> Optional[Path]:
    return _mx6_lib_path


def mx6_mask_format() -> str:
    """The keep-mask format of the LOADED MX6 .so ("int8" when nothing is loaded). Its build's NIB decision is the
    'only' build's (the same flags plus the MX6 switch), checked against the loaded 'only' .so by _load_mx6."""
    if _mx6_lib is None:
        return "int8"
    return _mask_format_of(_impl_mx6)


def set_mx6_prebuilt_for_testing(value: bool) -> None:
    """Tests only: mark MX6 prebuilt WITHOUT prebuild_mx6's INERT rules (the site set, E-RM, DO8 / DO8-ONLY, source,
    DIET, NIB and producer rules). Beside E-RM the INERT rule is the load-bearing guard (the seam's per-call E-RM
    decline keys on ctx.erm_site, which E-RM stops making once it latches): never arm MX6 through this in a kit or a
    harness."""
    global _mx6_prebuilt
    _mx6_prebuilt = value


def mx6_counters() -> Dict[str, Any]:
    out: Dict[str, Any] = dict(_MX6_COUNTS)
    out["latched"] = int(_mx6_broken is not None)
    out["declined"] = dict(_MX6_DECLINED)
    out["inert"] = _mx6_inert
    return out


def mx6_summary_line(final: bool = False) -> str:
    c = _MX6_COUNTS
    return (
        f"[l11-do8] mx6 counters: decided={c['decided']} launched={c['launched']} published={c['published']} "
        f"rescued={c['rescued']} released={c['released']} latched={int(_mx6_broken is not None)}"
        + (" final=1" if final else "")
    )


def _mx6_final() -> None:
    """MX6's exit line ("final=1"), run by _only_final (atexit or clean_exit's EXIT_HOOKS) after the 'only' line when
    MX6 armed, latched or counted anything in this process; nothing otherwise (the knob off included)."""
    if _mx6_prebuilt or _mx6_broken is not None or any(_MX6_COUNTS.values()):
        arm_print_all_ranks(mx6_summary_line(final=True))


def _mx6_event(key: str) -> None:
    """Count one MX6 event; the counters line at the first rescue / release of the process."""
    _MX6_COUNTS[key] += 1
    _only_atexit()  # MX6's exit line rides on _only_final (no exit-time registration of its own)
    if key in ("rescued", "released") and _MX6_COUNTS[key] == 1:
        arm_print_all_ranks(mx6_summary_line())


def _disable_mx6(reason: str) -> None:
    """Latch MX6 (only MX6) off for this process: one DISABLED line and the counters line. DO8, DO8-ONLY and L11 keep
    running (later calls take the FP8 dual cast and the 'only' launch). DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 raises
    instead."""
    global _mx6_broken
    if _fail_closed():
        raise RuntimeError(f"{FAMILY_MX6} failed with {KNOB}_FAIL_CLOSED=1: {reason}")
    if _mx6_broken is not None:
        return
    _mx6_broken = reason
    _only_atexit()
    arm_print_all_ranks(f"{FAMILY_MX6} DISABLED: {reason} -> FP8 dual cast + {ITEM_ONLY} for this process")
    arm_print_all_ranks(mx6_summary_line())


def _mx6_note_inert(reason: str) -> None:
    """One INERT line per process; nothing is built (the knob stays on, every call takes the FP8 path)."""
    global _mx6_inert
    _mx6_inert = reason
    _print_once("mx6-inert", f"{FAMILY_MX6} INERT: {reason}; nothing built -> FP8 dual cast + {ITEM_ONLY}")


def mx6_note_decline(reason: str, quiet: bool = False) -> None:
    """Count a call the MX6 seam does not take before anything else was evaluated (the shipped block then runs
    exactly); one "NOT armed" line per (reason key, process) unless quiet (expected states: warm-up, latches)."""
    k = decline_key(reason)
    _MX6_DECLINED[k] = _MX6_DECLINED.get(k, 0) + 1
    if not quiet:
        _print_once("mx6-decline:" + k,
                    f"{FAMILY_MX6} NOT armed: {reason} -> FP8 dual cast + {ITEM_ONLY} for this call")


def _reset_mx6_for_testing(unload: bool) -> None:
    global _mx6_broken, _mx6_inert, _mx6_prebuilt
    global _mx6_lib, _mx6_lib_path, _fn_launch_mx6, _fn_launch_mx6_fp8, _impl_mx6
    _mx6_broken = None
    _mx6_inert = None
    _mx6_prebuilt = False
    for k in _MX6_COUNTS:
        _MX6_COUNTS[k] = 0
    _MX6_DECLINED.clear()
    if unload:
        with _lock:
            _mx6_lib = None
            _mx6_lib_path = None
            _fn_launch_mx6 = None
            _fn_launch_mx6_fp8 = None
            _impl_mx6 = ""
            _mx6_resources.clear()
            _MX6_ONE.clear()


# ---------------------------------------------------------------------------- MX6 INERT rules / gate / load
def _mx6_producer_missing() -> List[str]:
    """The MX6_PRODUCER_API names ops/triton/bwd_fp8.py does not define as callables (looked up, never called)."""
    from generative_recommenders.ops.triton import bwd_fp8

    return [n for n in MX6_PRODUCER_API if not callable(getattr(bwd_fp8, n, None))]


def _mx6_erm_request() -> Optional[str]:
    """The knob of r14's E-RM row (ops/triton/dy_erm.py, gin item (97)) when E-RM is REQUESTED in this process
    (dy_erm.requested(): its knob, read once at import, or its test override), else None. Looked up only: nothing is
    armed, built or latched, and E-RM's runtime state is never read (an E-RM latch cannot move MX6; critique C1)."""
    from generative_recommenders.ops.triton import dy_erm

    return dy_erm.KNOB if dy_erm.requested() else None


def _mx6_inert_reason() -> Optional[str]:
    """None = MX6 may be built in this process; else the first INERT rule that holds (module docstring "MX6"), in this
    order: the site set; a site set holding layer 0 (bottom, all) while r14's E-RM row is requested (E-RM owns layer
    0's out-proj: fail closed, never another row's runtime state); DO8 and DO8-ONLY prebuilt and not latched; a source
    with the L11DO8_MX6 switch (DO8-VD); DIET on; the 'only' .so reading NIB while MX6_NIB_GATED is False (the switch
    is True: then this rule never holds and a NIB 'only' .so gets a NIB MX6 build); r14's producer API. Host state
    only."""
    if mx6_sites() is None:
        return (f"{MX6_SITES_KNOB}={_mx6_sites_value()!r} is not one of {' / '.join(MX6_SITES)} (fail closed: no "
                "MX6 site)")
    erm = _mx6_erm_request() if mx6_site_selected(0) else None
    if erm is not None:
        return (f"{MX6_SITES_KNOB}={mx6_sites()} holds layer 0, the out-proj site r14's E-RM row owns ({erm}=1 "
                "requested): fail closed, MX6 never takes an E-RM site (top only beside E-RM)")
    if not (requested() and _prebuilt and _broken is None):
        return f"needs item {ITEM} prebuilt and not latched in this process ({KNOB}=1)"
    if not (only_requested() and _only_prebuilt and _only_broken is None):
        return f"needs item {ITEM_ONLY} prebuilt and not latched ({ONLY_KNOB}=1): MX6 replaces its 'only' launch"
    source = _source()
    if source == _SOURCE_NOVD:
        return (f"needs DO8-VD (DLRM_HSTU_FP8_L11_VD=1, ops/hip/l11_do8.cu): the pre-diet {_SOURCE_NOVD.name} has no "
                "MX6 kernel")
    try:
        has_switch = _MX6_MARKER in source.read_bytes()
    except OSError:
        has_switch = False
    if not has_switch:
        return f"{source.name} has no L11DO8_MX6 switch (no MX6 kernel in this source)"
    if not diet_requested():
        return f"needs item (89) DIET ({DIET_KNOB}=1): the MX6 kernel is resource-gated with the diet body only"
    if only_mask_format() == "nib" and not MX6_NIB_GATED:
        return ("the 'only' .so reads the NIB keep mask and MX6 + NIB is not gated (MX6_NIB_GATED False: no NIB MX6 "
                "build is allowed)")
    missing = _mx6_producer_missing()
    if missing:
        return (f"r14's MX6 producer API is not in this tree (ops/triton/bwd_fp8.py has no {', '.join(missing)}): "
                "no G6 / W6 operands exist")
    return None


def check_resources_mx6(kernels: List[Dict[str, str]], nib: bool = False) -> Dict[str, Any]:
    """MX6's own resource gate (the FP8 check_resources never gates this .so: KERNEL_NAME is a substring of the MX6
    kernel's name and that gate wants exactly one remark). The MX6 .so's DO8 remarks must be EXACTLY these four roles,
    each once, classified by mangled prefix (never by substring alone), plus >= 1 list builder:
      fp8_only  _Z24l11_outproj_lnmd_bwd_do8     + _ONLY_MANGLE  LDS LDS_BYTES (the FP8 'only' twin, same flags)
      mx6_only  _Z28l11_outproj_lnmd_bwd_do8_mx6 + _ONLY_MANGLE  LDS LDS_BYTES, SGPR spill <= 63 (<= 36 for a NIB
                build: the NIB per-variant rule, the int8 MX6 'only' build's production spill + 2)
      fp8_dbg / mx6_dbg  the dy debug instantiations (_DBG_MANGLE)  LDS LDS_BYTES_DBG
    every one of them and every list builder at 0 VGPR spill and 0 scratch. Raises otherwise (the caller latches MX6
    only). Returns the MX6 'only' kernel's summary plus its FP8 twin's VGPR / SGPR spill (_load_mx6 requires them to
    equal the loaded 'only' .so's kernel)."""
    roles: Dict[str, Tuple[str, int]] = {
        "fp8_only": (_FP8_SYM + _ONLY_MANGLE, LDS_BYTES),
        "mx6_only": (_MX6_SYM + _ONLY_MANGLE, LDS_BYTES),
        "fp8_dbg": (_FP8_SYM + _DBG_MANGLE, LDS_BYTES_DBG),
        "mx6_dbg": (_MX6_SYM + _DBG_MANGLE, LDS_BYTES_DBG),
    }
    do8 = [k for k in kernels if KERNEL_NAME in k.get("Function Name", "")]
    wlb = [k for k in kernels if BUILD_KERNEL_NAME in k.get("Function Name", "")]
    found: Dict[str, Dict[str, str]] = {}
    for k in do8:
        name = k.get("Function Name", "")
        hit = [r for r, (pfx, _) in roles.items() if name.startswith(pfx)]
        if len(hit) != 1:
            raise RuntimeError(f"MX6 resource check: unexpected instantiation {name}")
        if hit[0] in found:
            raise RuntimeError(f"MX6 resource check: {hit[0]} instantiated twice")
        found[hit[0]] = k
    missing = sorted(set(roles) - set(found))
    if missing:
        raise RuntimeError(f"MX6 resource check: missing {missing} ({len(do8)} {KERNEL_NAME}* remarks)")
    if not wlb:
        raise RuntimeError(f"no {BUILD_KERNEL_NAME} resource remarks in the MX6 build log")
    for k in list(found.values()) + wlb:
        spill = int(k.get("VGPRs Spill", "-1"))
        scratch = int(k.get("ScratchSize [bytes/lane]", "-1"))
        if spill != 0 or scratch != 0:
            raise RuntimeError(f"MX6 resource check: {k.get('Function Name')} VGPR spill {spill}, scratch {scratch} "
                               "B/lane (need 0/0)")
    for role, (_, want) in roles.items():
        lds = int(found[role].get("LDS Size [bytes/block]", "-1"))
        if lds != want:
            raise RuntimeError(f"MX6 resource check: {role} LDS {lds} B (expected {want})")
    p, f = found["mx6_only"], found["fp8_only"]
    bound = _MX6_NIB_SGPR_SPILL_MAX if nib else _MX6_SGPR_SPILL_MAX
    ssp = int(p.get("SGPRs Spill", "-1"))
    if not 0 <= ssp <= bound:
        raise RuntimeError(f"MX6 resource check: {p.get('Function Name')} SGPR spill {ssp} (need <= {bound}"
                           + (": the NIB per-variant rule)" if nib else ")"))
    return {
        "vgpr": int(p.get("VGPRs", "-1")),
        "sgpr": int(p.get("TotalSGPRs", "-1")),
        "vgpr_spill": int(p.get("VGPRs Spill", "-1")),
        "sgpr_spill": ssp,
        "scratch": int(p.get("ScratchSize [bytes/lane]", "-1")),
        "lds": int(p.get("LDS Size [bytes/block]", "-1")),
        "fp8_twin_vgpr": int(f.get("VGPRs", "-1")),
        "fp8_twin_sgpr_spill": int(f.get("SGPRs Spill", "-1")),
        "dbg_lds": LDS_BYTES_DBG,
        "wl_build_vgpr": max(int(k.get("VGPRs", "-1")) for k in wlb),
        "nib": int(bool(nib)),
        "sched": sched(),
        "kernels": len(do8) + len(wlb),
    }


def _check_abi_impl_mx6(abi: int, impl_: str, want_nib: bool) -> None:
    """_load_mx6: l11do8_abi_version() is authoritative (the impl text keeps "abi=1"); the impl must be the EM 2 'only'
    build with the MX6 tag and read the keep-mask format the loaded 'only' .so reads (want_nib: "+mkb=nib")."""
    if abi != _ABI_VERSION_MX6:
        raise RuntimeError(f"MX6 .so: l11do8_abi_version()={abi}, expected {_ABI_VERSION_MX6}")
    if "EM 2 only" not in impl_ or _MX6_IMPL_TAG not in impl_:
        raise RuntimeError(f"MX6 .so impl {impl_[:240]!r} is not the EM 2 + MX6 build")
    if (_NIB_IMPL_TAG in impl_) != bool(want_nib):
        raise RuntimeError(f"MX6 .so reads keep mask format {_mask_format_of(impl_)}, the 'only' .so reads "
                           f"{'nib' if want_nib else 'int8'} (impl {impl_[:200]!r})")


# l11do8_e5_only_mx6_launch's 30 parameters (l11_do8.cu, -DL11DO8_MX6=1): G6, W6, W6S, sa, sb, M, X, U, MK, MEAN, RSTD,
# WN, BN, DX, DZ8, rows_out, INVS, AMAXP, DWP, DBP, DUCS, P, dscale, invD, WL, CNT, cap, DO8, E8, stream.
_MX6_LAUNCH_ARGTYPES: Tuple[Any, ...] = tuple(
    [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_void_p] * 9 + [ctypes.c_int] + [ctypes.c_void_p] * 5
    + [ctypes.c_longlong] + [ctypes.c_float, ctypes.c_float] + [ctypes.c_void_p] * 2 + [ctypes.c_longlong]
    + [ctypes.c_void_p] * 2 + [ctypes.c_void_p]
)


def _load_mx6() -> None:
    """MX6: build (or reuse) + gate (check_resources_mx6) + load the third .so and bind l11do8_e5_only_mx6_launch (30
    argtypes) and, for self_check_mx6, the same .so's FP8 'only' entry l11do8_e5_only_launch (the 'plus' entry's
    argtypes). ctypes.CDLL's default RTLD_LOCAL keeps its exports out of the global namespace (the 'plus' and 'only'
    .so export the same names). The work list, plan(cap) and drain stay the 'plus' .so's (one source, one plan). ABI 2
    and the impl ("EM 2 only", the MX6 tag, "+mkb=nib" exactly when the 'only' .so has it) are checked, and the FP8
    'only' twin of this .so must report the loaded 'only' .so's kernel resources (VGPRs, SGPR spill): the self-check's
    tie vouches for the production 'only' kernel only while the twin is that kernel. Raises on any failure (the caller
    latches MX6 only)."""
    global _mx6_lib, _mx6_lib_path, _fn_launch_mx6, _fn_launch_mx6_fp8, _impl_mx6
    if _mx6_lib is not None:
        return
    _load_only()
    with _lock:
        if _mx6_lib is not None:
            return
        assert _fn_launch is not None
        want_nib = only_mask_format() == "nib"
        so, summary = build(only=True, mx6=True)
        twin = (summary.get("fp8_twin_vgpr"), summary.get("fp8_twin_sgpr_spill"))
        prod = (_only_resources.get("vgpr"), _only_resources.get("sgpr_spill"))
        if None in prod or twin != prod:
            raise RuntimeError(f"MX6 .so: its FP8 'only' twin (VGPRs, SGPR spill) {twin} != the loaded 'only' .so's "
                               f"{prod} (the lattice tie would not vouch for the production 'only' kernel)")
        lib = ctypes.CDLL(str(so))
        abi = lib.l11do8_abi_version
        abi.restype = ctypes.c_int
        abi.argtypes = []
        tag = lib.l11do8_impl
        tag.restype = ctypes.c_char_p
        tag.argtypes = []
        impl_ = (tag() or b"").decode()
        _check_abi_impl_mx6(int(abi()), impl_, want_nib)
        launch = lib.l11do8_e5_only_mx6_launch
        launch.restype = ctypes.c_int
        launch.argtypes = list(_MX6_LAUNCH_ARGTYPES)
        twin = lib.l11do8_e5_only_launch
        twin.restype = ctypes.c_int
        twin.argtypes = list(_fn_launch.argtypes)
        _mx6_resources.clear()
        _mx6_resources.update({k: v for k, v in summary.items() if k != "kernels_raw"})
        _impl_mx6 = impl_
        _fn_launch_mx6 = launch
        _fn_launch_mx6_fp8 = twin
        _mx6_lib_path = so
        _mx6_lib = lib


def _mx6_one(device: torch.device) -> torch.Tensor:
    """sa = sb of the MX6 launch: a device fp32 1.0 (the EPI2 scale stays in the epilogue, the E8M0 bytes carry every
    scale); one cached tensor per device, allocated at its first use (the prebuild's self-check)."""
    key = str(device)
    one = _MX6_ONE.get(key)
    if one is None:
        one = torch.ones((1,), dtype=torch.float32, device=device)
        _MX6_ONE[key] = one
    return one


# ---------------------------------------------------------------------------- MX6 operand checks / raw launch
def _desc(t: Any) -> str:
    if not isinstance(t, torch.Tensor):
        return type(t).__name__
    return f"{t.dtype} {tuple(t.shape)} stride {tuple(t.stride())} a16={int(l11._a16(t))} {t.device}"


def _mx6_operand_mismatch(g6: Any, w6: Any, w6s: Any, m: int, device: torch.device) -> Optional[str]:
    """None = G6 / W6 / W6S are what l11do8_e5_only_mx6_launch reads for m rows on `device`: G6 uint8 [m, 400] row
    records (row stride 400, 16-B aligned, no padding rows); W6 uint8 589,824 B and W6S uint8 768 B (contiguous, 16-B
    aligned). Else the mismatch. Host metadata only."""
    if not (isinstance(g6, torch.Tensor) and g6.dtype == torch.uint8 and g6.dim() == 2
            and tuple(g6.shape) == (m, G6_ROW) and g6.stride() == (G6_ROW, 1) and l11._a16(g6)
            and g6.device == device):
        return f"G6 {_desc(g6)} (need uint8 [{m}, {G6_ROW}], row stride {G6_ROW}, 16-B aligned, on {device})"
    for name, t, n in (("W6", w6, W6_BYTES), ("W6S", w6s, W6S_BYTES)):
        if not (isinstance(t, torch.Tensor) and t.dtype == torch.uint8 and t.numel() == n and t.is_contiguous()
                and l11._a16(t) and t.device == device):
            return f"{name} {_desc(t)} (need uint8, {n} B, contiguous, 16-B aligned, on {device})"
    return None


def _mx6_contract_mismatch(ops6: Any, m: int, device: torch.device) -> Optional[str]:
    """None = the parts of r14's result the rescue and the W3 wgrad depend on are well formed: an Mx6Operands whose
    dout_fp8 is the step-tiled e4m3 [ceil(m/128), 512, 128] W3 operand (contiguous, 16-B aligned: every dout_t
    property x21's wgrad_ck_w3._decline_reason checks; an operand x21 declined would turn W3's step tiling off for the
    process, and with it E-RM's row-major entry at the bottom), dout_fp8_scale / g_decode_scale fp32 [1] and
    inv_scale_used fp32 [1] in its own 16-B-aligned allocation, all on `device`. A mismatch here cannot be rescued (the
    rescue and W3 would read it): the seam treats it as a contract break. G6 / W6 / W6S are checked by the launch
    (Mx6OperandError: rescued)."""
    if not isinstance(ops6, Mx6Operands):
        return f"the producer returned {type(ops6).__name__}, not l11_do8.Mx6Operands"
    want = ((m + 127) // 128, l11.D, 128)
    t = ops6.dout_fp8
    if not (isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn and tuple(t.shape) == want
            and t.is_contiguous() and l11._a16(t) and t.device == device):
        return (f"dout_fp8 {_desc(t)} (need float8_e4m3fn {list(want)}, step-tiled, contiguous, 16-B aligned, on "
                f"{device})")
    for name, s in (("dout_fp8_scale", ops6.dout_fp8_scale), ("g_decode_scale", ops6.g_decode_scale)):
        if not (isinstance(s, torch.Tensor) and s.dtype == torch.float32 and s.numel() == 1 and s.device == device):
            return f"{name} {_desc(s)} (need fp32 [1] on {device})"
    s = ops6.inv_scale_used
    if not (isinstance(s, torch.Tensor) and s.dtype == torch.float32 and s.numel() == 1 and s.is_contiguous()
            and l11._a16(s) and s.device == device):
        return f"inv_scale_used {_desc(s)} (need fp32 [1], its own 16-B-aligned allocation, on {device})"
    return None


def _launch_mx6_so(
    fn: Optional[Callable[..., int]],
    head: List[int],
    entry: str,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    inv_scale: torch.Tensor,
    dz8: torch.Tensor,
    wl: torch.Tensor,
    cnt: torch.Tensor,
    cap: int,
    do8: torch.Tensor,
    e8: torch.Tensor,
    dx: Optional[torch.Tensor],
    mx6: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One EM 2 launch of the MX6 .so (its MX6 entry, or for the self-check its FP8 'only' twin) on the current
    stream: launch_e5_plus's layout checks (dz8 / dO8 / E8 / work list / the d(attn) buffer), the keep mask in the .so's
    format, plan(cap), the drain, the d(attn) placeholder (EM 2: never written) and the partials, then the entry with
    `head` (MX6: G6, W6, W6S, sa, sb; twin: G8, W8, sa, sb) and the 'plus' entry's tail. Returns (dx, DWP, DBP, DUCS,
    AMAXP, P). rc > 0: Mx6LaunchError for the MX6 entry (MX6's own failure), RuntimeError for the twin."""
    assert fn is not None and _fn_drain is not None
    m = int(attn.shape[0])
    if dx is not None and not (dx.dtype == attn.dtype and tuple(dx.shape) == tuple(attn.shape) and dx.is_contiguous()
                               and l11._a16(dx) and dx.device == attn.device):
        raise RuntimeError(f"dx layout {dx.dtype} {tuple(dx.shape)} {dx.stride()} (attn {tuple(attn.shape)})")
    rows_out = int(dz8.shape[0])
    if not (dz8.dtype == torch.float8_e5m2 and dz8.dim() == 2 and int(dz8.shape[1]) == l11.DU_ROW
            and dz8.stride() == (l11.DU_ROW, 1) and l11._a16(dz8) and m <= rows_out < m + l11.E5_MAX_PAD):
        raise RuntimeError(f"dz8 layout {dz8.dtype} {tuple(dz8.shape)} {dz8.stride()} (M={m})")
    if not (do8.dtype == torch.float8_e4m3fn and tuple(do8.shape) == (m, l11.D) and do8.is_contiguous()
            and l11._a16(do8)):
        raise RuntimeError(f"do8 layout {do8.dtype} {tuple(do8.shape)} {do8.stride()} (M={m})")
    if not (e8.dtype == torch.uint8 and tuple(e8.shape) == (m, H) and e8.is_contiguous() and l11._a16(e8)):
        raise RuntimeError(f"e8 layout {e8.dtype} {tuple(e8.shape)} {e8.stride()} (M={m})")
    if not (wl.dtype == torch.int32 and tuple(wl.shape) == (cap, 2) and wl.is_contiguous()
            and cnt.dtype == torch.int32 and cnt.numel() == 2 and cnt.is_contiguous()):
        raise RuntimeError(f"work list layout {tuple(wl.shape)} {wl.dtype} / {tuple(cnt.shape)} {cnt.dtype}")
    mm = _mask_mismatch(random_mask, m, mx6_mask_format())
    if mm is not None:
        raise (Mx6OperandError if mx6 else RuntimeError)(f"mask format: {mm} (M={m})")
    G, fl, nf, P = plan(cap)
    device = attn.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stale = int(_fn_drain())
        if stale != 0:
            _print_once(f"stale:{stale}", f"{FAMILY}: stale HIP error {stale} drained before the launch (not ours)")
        stream = torch.cuda.current_stream(device).cuda_stream
        if dx is None:
            dx = torch.empty_like(attn)  # EM 2: the placeholder (not written)
        dwp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        dbp = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        ducs = torch.empty((P, l11.D), dtype=torch.float32, device=device)
        amaxp = torch.empty((G * 8,), dtype=torch.float32, device=device)
        rc = int(
            fn(
                *head, m,
                attn.data_ptr(), u.data_ptr(), random_mask.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                norm_weight.data_ptr(), norm_bias.data_ptr(), dx.data_ptr(), dz8.data_ptr(), rows_out,
                inv_scale.data_ptr(), amaxp.data_ptr(), dwp.data_ptr(), dbp.data_ptr(), ducs.data_ptr(), int(P),
                l11.dropout_scale(dropout_ratio), 1.0 / l11.D, wl.data_ptr(), cnt.data_ptr(), int(cap),
                do8.data_ptr(), e8.data_ptr(), stream,
            )
        )
    if rc > 0:
        msg = f"{entry} rc={rc} (hipError) M={m}"
        raise Mx6LaunchError(msg) if mx6 else RuntimeError(msg)
    if rc < 0:
        _print_once(f"stale:{-rc}", f"{FAMILY}: stale HIP error {-rc} drained at the launch (not ours; result used)")
    return dx, dwp, dbp, ducs, amaxp, P


def launch_e5_only_mx6(
    g6: torch.Tensor,
    w6: torch.Tensor,
    w6s: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    inv_scale: torch.Tensor,
    dz8: torch.Tensor,
    wl: torch.Tensor,
    cnt: torch.Tensor,
    cap: int,
    do8: torch.Tensor,
    e8: torch.Tensor,
    dx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One MX6 'only' launch on the current stream into the caller's dz8 / do8 / e8 (raw: tests and the self-check
    keep the partials): l11do8_e5_only_mx6_launch of the MX6 .so (EM 2, MX 1: E2M3 x E2M3 with E8M0 per-32 scales; the
    d(attn) is NOT written: the returned dx is the untouched placeholder) with sa = sb = the device 1.0. Returns (dx,
    DWP, DBP, DUCS, AMAXP, P). G6 / W6 / W6S failing their checks: Mx6OperandError, the kernel's rc > 0:
    Mx6LaunchError (both MX6's own); any other failure (layouts, plan(cap), the drain, an allocation) raises as in
    launch_e5_plus. run_e5_only_mx6 is the production entry."""
    _load_mx6()
    why = _mx6_operand_mismatch(g6, w6, w6s, int(attn.shape[0]), attn.device)
    if why is not None:
        raise Mx6OperandError(why)
    one = _mx6_one(attn.device)
    head = [g6.data_ptr(), w6.data_ptr(), w6s.data_ptr(), one.data_ptr(), one.data_ptr()]
    return _launch_mx6_so(_fn_launch_mx6, head, "l11do8_e5_only_mx6_launch", attn, u, random_mask, mean, rstd,
                          norm_weight, norm_bias, dropout_ratio, inv_scale, dz8, wl, cnt, cap, do8, e8, dx, True)


# ---------------------------------------------------------------------------- MX6 seam API (decide / produce / run)
def _mx6_preflight_reason(
    ctx: Any, attn: torch.Tensor, dout: torch.Tensor, weight: torch.Tensor, random_mask: Any, token_major: bool,
    snr: bool,
) -> Tuple[Optional[str], bool]:
    """mx6_preflight's checks: (reason or None, quiet)."""
    if not mx6_enabled():
        return "MX6 latched or not prebuilt", True
    layer = getattr(ctx, "l11_mx6_layer", None)
    if layer is None or not mx6_site_selected(int(layer)):
        return "not an MX6 site (the forward's layer tag)", True
    if not (enabled() and _prebuilt and l11.enabled() and l11.prebuilt()):
        return "DO8 or L11 latched or not prebuilt", True
    if not only_enabled():
        return f"{ITEM_ONLY} off or latched", True
    if getattr(ctx, "do8_site", None) is None:
        return "no dO8 site (the attention backward does not plan its C8 do_blk dO quantization)", True
    if snr:
        return "C3 the out-proj wgrad SNR probe is on", False
    if not token_major:
        return "C3 dout_t is not token-major (the producer covers the NN-mode W3 operand only)", False
    from generative_recommenders.ops.triton import bwd_fp8

    if not bwd_fp8.scale_controller_enabled():
        return "C3 the FP8 backward or its scale controller is off", False
    if bwd_fp8.grad_dtype_and_limit()[0] is not torch.float8_e5m2:
        return "C3 the gradient format is not e5m2", False
    if not getattr(bwd_fp8, "_OUT_WGRAD_T_E4M3", False):
        return "C3 the W3 dout_t copy is not e4m3 (DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT)", False
    if not bwd_fp8._helpers_fused_ok(weight):
        return "C3 the fused scale bookkeeping is off (DLRM_HSTU_FP8_BWD_HELPERS_FUSED) or not for this weight", False
    if not bwd_fp8._has_stable_identity(weight):
        return "C3 the out-proj weight is not the parameter (no stable scale-site identity)", False
    m = int(dout.shape[0]) if dout.dim() == 2 else 0
    if not (dout.dtype == torch.bfloat16 and dout.dim() == 2 and m >= 1 and int(dout.shape[1]) == l11.D
            and dout.is_contiguous() and l11._a16(dout) and int(attn.shape[0]) == m):
        return f"dout {_desc(dout)} (need bf16 [M, {l11.D}], contiguous, 16-B aligned, M = d(attn) rows)", False
    if tuple(weight.shape) != (l11.NOUT, l11.D):
        return f"out-proj weight {tuple(weight.shape)} (need ({l11.NOUT}, {l11.D}))", False
    state = bwd_fp8._peek_site_state("output_projection", weight.data_ptr())
    if state is None or not state.published:
        return "the out-proj dgrad scale is not published yet (the scale controller's warm-up)", True
    fmt = mx6_mask_format()
    if lnmd_mask_nib.is_nib(random_mask) != (fmt == "nib"):
        # an int8 mask while the 'only' .so reads NIB (after a NIB writer latch, or a call the NIB writers did not
        # write): the shipped block's run_e5_plus serves it as DO8 'plus' (int8_plus), not as an 'only' launch
        plus = only_mask_format() == "nib" and not lnmd_mask_nib.is_nib(random_mask)
        route = ": with the NIB 'only' .so, DO8 takes an int8 mask as 'plus' (run_e5_plus's int8_plus route)"
        return f"keep mask format (the MX6 .so reads {fmt}){route if plus else ''}", False
    mm = _mask_mismatch(random_mask, m, fmt)
    if mm is not None:
        return mm, False
    return None, True


def mx6_preflight(ctx: Any, attn: torch.Tensor, dout: torch.Tensor, weight: torch.Tensor, random_mask: Any, *,
                  token_major: bool, snr: bool) -> Optional[str]:
    """(1) of the MX6 seam (triton_hstu_linear._l11_mx6_backward), on a call whose forward tagged an MX6 site: None =
    MX6 may take the call as far as host metadata tells before B1a's plan, DO8's emission site and the dO8 registry are
    asked; else the reason (counted; one NOT armed line per reason unless an expected state) and the seam runs the
    shipped block exactly (nothing else was evaluated). Host metadata only: no registry call, no allocation, no device
    access, no scale-state change (one peek of the out-proj site's `published` flag). The checks: MX6 armed and the
    layer in the site set; DO8 enabled and prebuilt, DO8-ONLY enabled (its knob read per call), L11 enabled and
    prebuilt, a dO8 site; critique C3's accepted configuration of r14's producer: a token-major dout_t (the seam then
    also requires it step-tiled), the e4m3 W3 copy with e5m2 grads, the fused scale bookkeeping for this weight, the
    wgrad SNR probe off, the FP8 backward and its controller on, a stable weight identity and a PUBLISHED out-proj
    scale (so the FP8 cast always casts: the seam's FP8 fallback never meets the warm-up); dout bf16 [M, 512]
    contiguous and 16-B aligned, the weight [1536, 512]; the keep mask in the MX6 .so's format."""
    why, quiet = _mx6_preflight_reason(ctx, attn, dout, weight, random_mask, token_major, snr)
    if why is not None:
        mx6_note_decline(why, quiet=quiet)
    return why


def mx6_decide(ctx: Any, attn: torch.Tensor, du_plan: Any, *, u: torch.Tensor, BLOCK_D: int, num_warps: int,
               silu_u: bool, du_row_width: int) -> Mx6Decision:
    """(2) of the MX6 seam, after mx6_preflight, BEFORE the cast; host only. With B1a's plan `du_plan` (the seam
    evaluated it ONCE, with the arguments the shipped block uses after the cast), DO8's emission site is evaluated ONCE
    (emission_site counts the call and prints its own declines, exactly as on the FP8 path), then run_e5_plus's E5
    checks on metadata (l11.e5_decline_reason with B1a's inv_scale; pure), then the dO8 registry ONCE (_plan_only).
    The decision is consumed by whichever path then runs (critique C10): taken (an open 'only' plan, counted
    "decided") -> r14's producer and the MX6 launch; not taken -> the shipped block with this site and this registry
    answer (only_plan False: run_e5_plus does not ask again; None: the registry was not asked). B1a declined (du_plan
    None): L11's bf16-du mode; the emission site is not evaluated (the shipped block does not evaluate it there
    either). A raise inside the registry call is DO8's shared machinery: an open plan is cancelled and DO8 latched,
    as run_e5_plus does. From the 'only' answer on the returned decision owns the open plan: it exists before anything
    else runs here, and a raise while counting it closes the plan (mx6_abandon) before propagating."""
    m = int(attn.shape[0])
    layer = int(getattr(ctx, "l11_mx6_layer", -1))
    if du_plan is None:
        return Mx6Decision(None, m, layer, None, "B1a declined the call: L11's bf16-du mode has no MX6 variant")
    site = emission_site(ctx, attn)
    if site is None:
        return Mx6Decision(None, m, layer, None, "DO8 declined the call (emission_site)")
    from generative_recommenders.ops.triton import bwd_fp8

    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    pad = int(bwd_fp8.uvqk_tm_pad())
    why = l11.e5_decline_reason(attn, u, BLOCK_D, num_warps, silu_u, du_row_width, grad_dtype, limit, pad, du_plan[0])
    if why is not None:
        return Mx6Decision(site, m, layer, None, "E5 " + why)
    try:
        only = _plan_only(site, attn)
    except Exception as error:  # noqa: BLE001 -- the registry is DO8's shared machinery: today's policy
        if getattr(site, "only_plan", None) is not None:
            _cancel_only(site)
        _disable(f"MX6 decide (dO8 registry plan): {type(error).__name__}: {str(error)[:600]}")
        return Mx6Decision(site, m, layer, None, "the dO8 registry plan failed (DO8 latched)")
    if not only:
        return Mx6Decision(site, m, layer, False, "the dO8 registry planned 'plus' (calibration or a debug reader)")
    dec = Mx6Decision(site, m, layer, True, None, open_=True)  # owns the open 'only' plan from here
    try:
        _mx6_event("decided")
    except BaseException:
        mx6_abandon(dec)  # the seam's guard does not cover this frame: no 'only' plan may dangle
        raise
    return dec


def _mx6_close(plan: Optional[Mx6Decision]) -> None:
    """Close the 'only' plan an MX6 decision still owns (registry cancel through _cancel_only: the 'only' counters
    stay balanced); no-op for a decision without an open plan. The decision lets go of the plan only once the cancel
    went through (a raise inside it leaves `open` set, so mx6_abandon tries again; a second cancel is a no-op)."""
    if plan is not None and plan.open:
        _cancel_only(plan.site)
        plan.open = False
        plan.only_plan = None  # closed: the registry may be asked again (today's chain plans for itself)


def mx6_release(plan: Optional[Mx6Decision], why: str, latch: bool = False) -> None:
    """Close an MX6 decision's open 'only' plan BEFORE r14's producer committed the scale state (it declined, or it
    raised: latch=True), with latch=True latch MX6 (DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 raises there: nothing printed
    or counted), then one line per reason per process and the count ("released"). The caller then runs the FP8 dual
    cast (its bookkeeping is the call's only one) and today's chain (DO8 may plan 'only' again: legal after a cancel).
    No-op without an open plan. Not a latch, so not raised under FAIL_CLOSED: "MX6 release" is a must-be-ABSENT line
    of an MX6 arm (grep for it); a condition that persists must make r14's S1 raise (S1 contract sec. 2.3)."""
    if plan is None or not plan.open:
        return
    _mx6_close(plan)
    if latch:
        _disable_mx6(why)
    _print_once("mx6-release:" + decline_key(why),
                f"{FAMILY_MX6} release: {why} -> FP8 dual cast + {ITEM_ONLY} for this call (its 'only' plan closed; "
                + ("MX6 latched)" if latch else "nothing latched)"))
    _mx6_event("released")


def mx6_abandon(plan: Optional[Mx6Decision]) -> None:
    """The seam's exception guard: a plan an MX6 decision still owns when an exception leaves the seam is closed and
    counted as released (no 'only' plan dangles); a plan already handed to publish() is not touched."""
    if plan is not None and plan.open:
        _mx6_close(plan)
        _mx6_event("released")


def _mx6_state_calls(weight: torch.Tensor) -> Optional[int]:
    """The out-proj e5m2 gradient site's host call counter (a peek: nothing created or changed), None without one."""
    from generative_recommenders.ops.triton import bwd_fp8

    state = bwd_fp8._peek_site_state("output_projection", weight.data_ptr())
    return None if state is None else int(state.calls)


def _mx6_contract_break(plan: Optional[Mx6Decision], why: str, contract: str = "S1") -> None:
    """r14's producer broke its S1 (or R2: the rescue) contract: close the plan, latch MX6 and raise Mx6ContractError
    (fail closed)."""
    msg = f"r14's MX6 producer broke the {contract} contract: {why}"
    _mx6_close(plan)
    _disable_mx6(msg)
    raise Mx6ContractError(msg)


def _mx6_require_delta(plan: Optional[Mx6Decision], weight: torch.Tensor, calls0: Optional[int], want: int,
                       what: str, contract: str = "S1") -> None:
    """The state contract on the host: the out-proj e5m2 site's `calls` moved by exactly `want` around the producer
    call (S1: 0 for a decline or a raise, 1 for a result; R2, the rescue: 0); otherwise _mx6_contract_break."""
    calls1 = _mx6_state_calls(weight)
    if calls0 is None or calls1 is None or calls1 - calls0 != want:
        ran = "?" if calls0 is None or calls1 is None else calls1 - calls0
        _mx6_contract_break(plan, f"the out-proj e5m2 site's bookkeeping ran {ran} times (calls {calls0} -> "
                                  f"{calls1}) when the producer {what}; the contract is {want}", contract)


def mx6_produce(plan: Mx6Decision, dout: torch.Tensor, weight: torch.Tensor, *, token_major: bool,
                t_tiled: bool) -> Optional[Mx6Operands]:
    """(3) of the MX6 seam, on a taken decision: r14's S1 producer bwd_fp8.output_projection_dgrad_operands_mx6(dout,
    weight, token_major=, t_tiled=). Returns its Mx6Operands (the e5m2 gradient site's bookkeeping done exactly once:
    from here the scale state is committed and every exit publishes the 'only' record or rescues), or None after
    mx6_release: S1 declined (state untouched; released and counted, nothing latched) or raised (state untouched by
    contract; released, MX6 latched); the caller then runs the FP8 dual cast, whose bookkeeping is the call's only
    one. The contract is checked on the host around the call: the out-proj site's `calls` counter moves by 0 on a
    decline or raise and by 1 on success, and the result is an Mx6Operands whose W3 operand and scales are well formed
    (_mx6_contract_mismatch); a break closes the plan, latches MX6 and raises Mx6ContractError (fail closed: never a
    silent second or missing scale-state update, never a W3 operand nobody checked)."""
    from generative_recommenders.ops.triton import bwd_fp8

    calls0 = _mx6_state_calls(weight)
    try:
        ops6 = bwd_fp8.output_projection_dgrad_operands_mx6(dout, weight, token_major=token_major, t_tiled=t_tiled)
    except Exception as error:  # noqa: BLE001 -- S1 raised (state untouched: checked) -> the FP8 path
        _mx6_require_delta(plan, weight, calls0, 0, f"raised {type(error).__name__}")
        mx6_release(plan, f"producer raised: {type(error).__name__}: {str(error)[:400]}", latch=True)
        return None
    if ops6 is None:
        _mx6_require_delta(plan, weight, calls0, 0, "declined")
        mx6_release(plan, "producer declined (scale state untouched)")
        return None
    _mx6_require_delta(plan, weight, calls0, 1, "returned its operands")
    why = _mx6_contract_mismatch(ops6, int(dout.shape[0]), dout.device)
    if why is not None:
        _mx6_contract_break(plan, why)
    return ops6


def run_e5_only_mx6(
    ops6: Mx6Operands,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
    site: Any,
    plan: Mx6Decision,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]]:
    """(4) of the MX6 seam: the PRODUCTION MX6 launch, run_e5_plus's 'only' branch on r14's operands. Same keywords
    (BLOCK_D / num_warps / silu_u are mx6_decide's E5 inputs, accepted for the shape of the call) and EXACTLY
    run_e5_plus's (and the seam's _l11_do8_e5_or_none's) return: (dx = the EM 2 d(attn) placeholder, poisoned on
    request; placeholder du; dweight; dbias; DuE5m2Emit; dO8 [M, 512] e4m3fn; E8 [M, 4] uint8). The call becomes the
    in-flight 'only' call (_ONLY_CALL, the placeholder attribute) that the seam's unchanged publish() hands over with
    bf16_written=False (DO8 / 'only' / MX6 counted); the plan is handed over with it.
    None = a failure after the producer committed the scale state, in this order: the 'only' plan is cancelled, MX6
    latched (an MX6-specific failure: Mx6OperandError, Mx6LaunchError) or DO8 and MX6 latched (the shared machinery:
    the work list, an allocation incl. OOM, the launch's layout checks, plan(cap), the drain, the host work after the
    launch; today's 'only' policy; DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 raises there, so nothing is rescued, printed or
    counted), then one rescue line and the count. The seam then rescues (mx6_rescue: the FP8 dual cast again at the
    captured inv_scale, no second scale-state update) and joins today's chain at "DO8 returned None". The plan is
    handed over LAST on success (after the record and the once-per-process line): until then an exception leaves it
    to the seam's mx6_abandon."""
    m = int(x.shape[0])
    launched = False
    try:
        if not (isinstance(plan, Mx6Decision) and plan.taken and plan.site is site):
            raise Mx6OperandError(f"no open MX6 'only' plan for this site ({plan!r})")
        from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

        grad_dtype, _ = bwd_fp8.grad_dtype_and_limit()
        pad = int(bwd_fp8.uvqk_tm_pad())
        wl, cnt, cap = work_list(site.seq_offsets, m, int(site.N), x.device)
        rows_out = (m + pad - 1) // pad * pad
        dz8 = torch.empty((rows_out, du_row_width), dtype=grad_dtype, device=x.device)
        do8 = torch.empty((m, l11.D), dtype=torch.float8_e4m3fn, device=x.device)
        e8 = torch.empty((m, H), dtype=torch.uint8, device=x.device)
        dx, dwp, dbp, ducs, amaxp, P = launch_e5_only_mx6(
            ops6.g6, ops6.w6, ops6.w6s, x, u, random_mask, mean, rstd, weight, bias, dropout_ratio, inv_scale, dz8,
            wl, cnt, cap, do8, e8,
        )
        launched = True
        _mx6_event("launched")
        if _registry_mod is not None and _registry_mod.poison_requested():
            dx.view(torch.int16).fill_(_POISON_BF16)  # contract v2 evidence knob: any hidden reader sees NaN
        dweight = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((l11.D,), dtype=weight.dtype, device=x.device)
        l11.dwdb_finalize(dwp, dbp, dweight, dbias, P)
        du = hstu_du_e5m2.placeholder_du(u, m, l11.D)
        emit = hstu_du_e5m2.DuE5m2Emit(
            dz8=dz8,
            u_cols=(0, l11.D),
            u_amax=amaxp.amax(),
            u_colsum=ducs.sum(dim=0),
            inv_scale=inv_scale,
            cast_scale=cast_scale,
            rows=m,
            width=du_row_width,
            site_key=site_key,
            generation=generation,
            buffer_ptr=0,  # no d(uvqk) buffer handed over (placeholder du), as B1a and run_e5
        )
    except Exception as error:  # noqa: BLE001 -- cancel, latch, then line and count; the seam rescues
        own = isinstance(error, (Mx6OperandError, Mx6LaunchError))
        _cancel_only(site)  # first: this call will not publish 'only' (L11's own launch writes the bf16 d(attn))
        if isinstance(plan, Mx6Decision):
            plan.open = False  # only after the cancel went through (else mx6_abandon tries again)
            plan.only_plan = None
        when = "after" if launched else "before"
        why = f"{type(error).__name__}: {str(error)[:600]}"
        if not own:
            _disable(f"e5 only call ({when} the MX6 EM = 2 launch): {why}")  # today's DO8 policy (shared machinery)
        _disable_mx6(f"rescue ({when} the MX6 EM = 2 launch): {why}")  # FAIL_CLOSED: raised above or here
        arm_print_all_ranks(
            f"{FAMILY_MX6} rescue: {why} ({'MX6' if own else 'shared machinery'}, {when} the MX6 EM = 2 launch) -> "
            "FP8 rescue cast (the dual kernel at the captured inv_scale: no second scale-state update) + item (39) "
            "L11's own launch for this call"
        )
        _mx6_event("rescued")
        return None
    _count("emitted")
    _ONLY_CALL.update(site=site, dx=weakref.ref(dx),
                      mx6={"layer": plan.layer, "rows": m, "cap": cap, "device": str(x.device)})
    setattr(dx, _ONLY_PLACEHOLDER_ATTR, site)  # review_do8only P1: publish() never sends it out as 'plus'
    _only_event("only_launched")
    if not l11._active_e5_logged:
        # Item (39)'s own e5 ACTIVE line (its census marker), once per process, as run_e5_plus prints it, naming the
        # MX6 build that ran
        l11._active_e5_logged = True
        arm_print_all_ranks(
            f"armed OK: {l11.FAMILY} ACTIVE mode=e5 (B1a contract) ({l11.KNOB}, item (39), with item (38)(a) "
            f"DLRM_HSTU_FP8_BWD_DU_E5M2): one {l11.KERNEL_NAME} E5 launch replaces hipBLASLt F8B8BS NTB0 out-proj "
            f"dgrad + _ln_mul_dropout_bwd_dx_du_rng_v3 (B1a EMIT: e5m2 u block + pad rows of dz8, amax/colsum "
            f"partials; no bf16 du, dy never in HBM); impl={_impl_mx6} M={m} Mp={int(dz8.shape[0])} P={P} "
            f"device={x.device}; launched as {MX6_KERNEL_NAME} (item {ITEM_MX6}, + dO8/E8; item {ITEM_ONLY}: EM 2 "
            "build, no bf16 d(attn) store)"
        )
    plan.open = False  # LAST: handed over with the record (publish() closes the plan; mx6_abandon leaves it alone)
    return dx, du, dweight, dbias, emit, do8, e8


def _mx6_rescue_mismatch(out: Any, m: int, device: torch.device) -> Optional[str]:
    """None = R2's result is the rescued FP8 dgrad operands the seam hands to L11's E5 launch or the dgrad GEMM: four
    tensors on `device`, g8 float8_e5m2 [m, 512] and w8 float8_e4m3fn [1536, 512] (contiguous, 16-B aligned), and the
    fp32 [1] scales g8_scale / w8_scale; else the mismatch. Host metadata only."""
    if not (isinstance(out, tuple) and len(out) == 4 and all(isinstance(t, torch.Tensor) for t in out)):
        return f"the rescue returned {type(out).__name__}, not (g8, g8_scale, w8, w8_scale)"
    g8, g8s, w8, w8s = out
    for name, t, dtype, shape in (("g8", g8, torch.float8_e5m2, (m, l11.D)),
                                  ("w8", w8, torch.float8_e4m3fn, (l11.NOUT, l11.D))):
        if not (t.dtype == dtype and tuple(t.shape) == shape and t.is_contiguous() and l11._a16(t)
                and t.device == device):
            return f"{name} {_desc(t)} (need {dtype} {list(shape)}, contiguous, 16-B aligned, on {device})"
    for name, s in (("g8_scale", g8s), ("w8_scale", w8s)):
        if not (s.dtype == torch.float32 and s.numel() == 1 and s.device == device):
            return f"{name} {_desc(s)} (need fp32 [1] on {device})"
    return None


def mx6_rescue(dout: torch.Tensor, weight: torch.Tensor, ops6: Mx6Operands
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(5) of the MX6 seam, R2, once per rescued call (run_e5_only_mx6 returned None: its plan closed, MX6 latched):
    r14's bwd_fp8.output_projection_g8_rescue(dout, weight, ops6) re-casts g8 with the FP8 dual kernel at the CAPTURED
    inv_scale (ops6.inv_scale_used) and returns (g8, g8_scale, w8, w8_scale). Its contract is checked on the host as
    S1's is: the out-proj e5m2 site's `calls` counter must not move (the call's scale bookkeeping was S1's; a rescue
    that ran it again would be a silent second update) and the result must be the FP8 dgrad operands
    (_mx6_rescue_mismatch). A break raises Mx6ContractError (fail closed: the plan is already closed and MX6 already
    latched; nothing else to undo). A raise of the rescue itself propagates (no further fallback)."""
    from generative_recommenders.ops.triton import bwd_fp8

    calls0 = _mx6_state_calls(weight)
    out = bwd_fp8.output_projection_g8_rescue(dout, weight, ops6)
    _mx6_require_delta(None, weight, calls0, 0, "re-cast g8 (output_projection_g8_rescue)", contract="R2")
    why = _mx6_rescue_mismatch(out, int(dout.shape[0]), dout.device)
    if why is not None:
        _mx6_contract_break(None, why, contract="R2")
    return out


def _mx6_published() -> None:
    """_publish_only, for a record run_e5_only_mx6 launched: count it, print nothing. MX6's ACTIVE line and the
    counters line print once per process at prebuild_mx6's success, before RUN_START, so the normal backward path
    prints no MX6 line (none in the timed region); the count reaches the counters line at exit (and at a rescue /
    release / latch). The FP8 'only' ACTIVE line names the FP8 'only' .so and is left to an FP8 'only' record."""
    _mx6_event("published")


# ---------------------------------------------------------------------------- MX6 self-check / prebuild
_MX6_LAT_G: int = -20  # self_check_mx6's lattice exponents (G-C4): dout = n * 2^(LAT_G + t_r), W = m * 2^(LAT_W + u)
_MX6_LAT_W: int = -9
_MX6_SELF_CHECK_SEED: int = 20261013
_MX6_G_SCALE: float = 2.0**-24  # the controller's publish (amax * 8 / 57344) for the lattice amax 7 * 2^-14, exact


def mx6_lattice_operands(m: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """self_check_mx6's lattice (G-C4): dout bf16 [m, 512] = n * 2^(-20 + t_r), n uniform integer in [-7, 7], t_r
    uniform in {0..6} per row (row 0: t = 6, dout[0, 0] = 7: the amax 7 * 2^-14); W fp32 [1536, 512] = k * 2^(-9 + u),
    k uniform integer in [-7, 7], u uniform in {0..3} per 32-row block (block 0: u = 3, W[0, 0] = 7: the amax 7 * 2^-6).
    Every value is exact in E2M3 under its E8M0 block scale, in e5m2 at the gradient scale 2^-24 and in e4m3 at
    quantize_weight's scale amax / 448 = 2^-12 (both powers of two), and every dot product is an integer below 2^15
    times one power of two, so every MFMA path computes it exactly: the FP8 and the MX6 kernels see the same dy. A
    private generator (no global RNG draw, no training data)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    n = torch.randint(-7, 8, (m, l11.D), generator=g, device=device).to(torch.float32)
    t = torch.randint(0, 7, (m, 1), generator=g, device=device).to(torch.float32)
    t[0, 0] = 6.0
    n[0, 0] = 7.0
    dout = (n * torch.pow(2.0, t + _MX6_LAT_G)).to(torch.bfloat16)
    k = torch.randint(-7, 8, (l11.NOUT, l11.D), generator=g, device=device).to(torch.float32)
    u = torch.randint(0, 4, (l11.NOUT // 32, 1, 1), generator=g, device=device).to(torch.float32)
    u[0, 0, 0] = 3.0
    k[0, 0] = 7.0
    w = (k.reshape(l11.NOUT // 32, 32, l11.D) * torch.pow(2.0, u + _MX6_LAT_W)).reshape(l11.NOUT, l11.D)
    return dout, w.contiguous()


def self_check_mx6() -> Tuple[bool, str]:
    """MX6 first-use check (prebuild_mx6, after self_check_only): the lattice chain tie of the lab's G-C4, in process.
    On mx6_lattice_operands (sequence 5's first 8 rows zeroed: E8 127 groups) the MX6 'only' launch (G6 / W6 / W6S
    from r14's stateless encode bwd_fp8.output_projection_mx6_encode; sa = sb = 1.0) and the FP8 'only' twin of the
    SAME .so (g8 from the production gradient cast bwd_fp8.quantize_row_major at the exact scale 2^-24, w8 / sb from
    bwd_fp8.quantize_weight) see the same dy, so dO8, E8, the WHOLE dz8 (u block, pad rows, untouched guard columns)
    and the AMAXP / DWP / DBP / DUCS partials and P must be byte-identical; both prefilled d(attn) sentinels stay
    untouched (EM 2: no d(attn) store) and the given buffers come back; the twin's dO8 is written and E8 127 groups
    exist; the lattice premise holds (g8 * sa == dout, w8 * sb == W exactly, sb a power of two). Plus a determinism
    twin: two MX6 launches on random (non-lattice) operands byte-identical with finite partials. Consumer inputs and
    work list: self_check_only's recipe and seed (no training data, no global RNG draw, no FP8 scale state, no
    registry); a NIB .so (MX6_NIB_GATED) gets lnmd_mask_nib.pack_ref(mask) for both launches and the round trip is
    gated."""
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    assert _fn_launch_mx6 is not None and _fn_launch_mx6_fp8 is not None and _fn_wl_build is not None
    dev = torch.device("cuda", torch.cuda.current_device())
    lens: List[int] = []
    while sum(lens) < 32768:
        lens.extend(_SELF_CHECK_LENS)
    M, Dm, p = sum(lens), l11.D, 0.1
    offs_h = [0]
    for n_ in lens:
        offs_h.append(offs_h[-1] + n_)
    offs = torch.tensor(offs_h, dtype=torch.int64, device=dev)
    N = max(lens)
    g = torch.Generator(device=dev)
    g.manual_seed(_SELF_CHECK_SEED)
    x = (torch.randn((M, Dm), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    u = torch.randn((M, Dm), generator=g, device=dev).to(torch.bfloat16)
    wn = (1.0 + 0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    bn = (0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    xf = x.float()
    mean = xf.mean(dim=1)
    rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6)
    del xf
    mask = thl._create_dropout_mask(
        N=M, D=Dm, BLOCK_D=Dm, concat_u=True, concat_x=True, dropout_ratio=p, seed=_SELF_CHECK_SEED, device=dev,
    )
    nib, mask_mx, nib_rt = _self_check_mask(mask, mx6_mask_format())  # both entries of the MX6 .so read its format
    grad_dtype, _ = bwd_fp8.grad_dtype_and_limit()
    pad = int(bwd_fp8.uvqk_tm_pad())
    rows_out = (M + pad - 1) // pad * pad
    inv = torch.full((1,), 0.7371 * 2.0**-3, dtype=torch.float32, device=dev)
    b = len(lens)
    cap = wl_cap(M, b)
    wl = torch.full((cap, 2), -7, dtype=torch.int32, device=dev)
    cnt = torch.full((2,), -7, dtype=torch.int32, device=dev)
    rc = int(_fn_wl_build(offs.data_ptr(), 1, b, M, max(1, min(128, (N + 63) // 64)), wl.data_ptr(), cap,
                          cnt.data_ptr(), torch.cuda.current_stream(dev).cuda_stream))
    if rc > 0:
        return False, f"l11do8_wl_build_launch rc={rc}"
    # the lattice and both encodings of it
    dout, w = mx6_lattice_operands(M, _MX6_SELF_CHECK_SEED, dev)
    s5 = offs_h[5]
    dout[s5:s5 + min(8, lens[5])] = 0.0  # zero token groups -> E8 = 127
    sa = torch.full((1,), _MX6_G_SCALE, dtype=torch.float32, device=dev)
    g8, _ = bwd_fp8.quantize_row_major(dout, torch.full((1,), 1.0 / _MX6_G_SCALE, dtype=torch.float32, device=dev))
    w8, sb = bwd_fp8.quantize_weight(w)
    g6, w6, w6s = bwd_fp8.output_projection_mx6_encode(dout, w)
    sbv = float(sb.reshape(-1)[0].item())
    premise = (bool(torch.equal(g8.float() * sa, dout.float())) and bool(torch.equal(w8.float() * sb, w))
               and sbv > 0.0 and math.frexp(sbv)[0] == 0.5)

    def launch(mx6: bool, ops: Tuple[torch.Tensor, ...], dx: Optional[torch.Tensor]) -> Tuple[Any, ...]:
        dz8 = torch.full((rows_out, l11.DU_ROW), 0x5A, dtype=torch.uint8, device=dev)
        d8 = torch.full((M, Dm), 0x5A, dtype=torch.uint8, device=dev)
        e8 = torch.full((M, H), 0xA5, dtype=torch.uint8, device=dev)
        if mx6:
            out = launch_e5_only_mx6(ops[0], ops[1], ops[2], x, u, mask_mx, mean, rstd, wn, bn, p, inv,
                                     dz8.view(grad_dtype), wl, cnt, cap, d8.view(torch.float8_e4m3fn), e8, dx=dx)
        else:
            head = [ops[0].data_ptr(), ops[1].data_ptr(), ops[2].data_ptr(), ops[3].data_ptr()]
            out = _launch_mx6_so(_fn_launch_mx6_fp8, head, "l11do8_e5_only_launch (the MX6 .so's FP8 twin)", x, u,
                                 mask_mx, mean, rstd, wn, bn, p, inv, dz8.view(grad_dtype), wl, cnt, cap,
                                 d8.view(torch.float8_e4m3fn), e8, dx, False)
        return out + (dz8, d8, e8)

    sentinel = torch.full((M, Dm), _SENTINEL_BF16, dtype=torch.int16, device=dev)
    dx_f = sentinel.clone().view(torch.bfloat16)
    dx_m = sentinel.clone().view(torch.bfloat16)
    f8 = launch(False, (g8, w8, sa, sb), dx_f)
    m6 = launch(True, (g6, w6, w6s), dx_m)
    # determinism twin on random operands (no lattice: E8M0 scales and E2M3 rounding everywhere)
    dr = (torch.randn((M, Dm), generator=g, device=dev) * 2.0**-10).to(torch.bfloat16)
    wr = torch.randn((l11.NOUT, Dm), generator=g, device=dev) * 0.05
    g6r, w6r, w6sr = bwd_fp8.output_projection_mx6_encode(dr, wr)
    r1 = launch(True, (g6r, w6r, w6sr), None)
    r2 = launch(True, (g6r, w6r, w6sr), None)
    torch.cuda.synchronize()

    def mism(a: torch.Tensor, c: torch.Tensor) -> int:
        a8 = a.contiguous().view(torch.uint8).reshape(-1)
        c8 = c.contiguous().view(torch.uint8).reshape(-1)
        return -1 if a8.numel() != c8.numel() else int((a8 != c8).sum())

    def outputs(o: Tuple[Any, ...]) -> List[torch.Tensor]:  # dz8, dO8, E8, DWP, DBP, DUCS, AMAXP
        return [o[6], o[7], o[8], o[1], o[2], o[3], o[4]]

    tie = [mism(a, c) for a, c in zip(outputs(m6), outputs(f8))]
    twin = [mism(a, c) for a, c in zip(outputs(r1), outputs(r2))]
    finite = all(bool(torch.isfinite(t).all()) for t in (r1[1], r1[2], r1[3], r1[4], m6[1], m6[2], m6[3]))
    sent_f = mism(dx_f, sentinel.view(torch.bfloat16))
    sent_m = mism(dx_m, sentinel.view(torch.bfloat16))
    same_buf = f8[0].data_ptr() == dx_f.data_ptr() and m6[0].data_ptr() == dx_m.data_ptr()
    f8_written = int((f8[7] != 0x5A).sum())
    e8_127 = int((f8[8] == 127).sum())
    names = ("dz8", "dO8", "E8", "DWP", "DBP", "DUCS", "AMAXP")
    detail = (
        f"M={M} B={b} cap={cap} P={m6[5]}/{f8[5]}: lattice MX6 vs FP8 twin "
        + ", ".join(f"{k} mism {v}" for k, v in zip(names, tie))
        + f"; d(attn) sentinel bytes changed {sent_m}/{sent_f} (buffers returned {int(same_buf)}); premise "
        f"{int(premise)} (sb={sbv:.3e}); twin dO8 bytes written {f8_written}, E8=127 bytes {e8_127}; determinism twin "
        + ", ".join(f"{k} mism {v}" for k, v in zip(names, twin))
        + f"; finite {int(finite)}"
    ) + _self_check_mask_detail(nib, nib_rt)
    ok = (
        m6[5] == f8[5]
        and r1[5] == r2[5]
        and all(v == 0 for v in tie)
        and all(v == 0 for v in twin)
        and sent_f == 0
        and sent_m == 0
        and same_buf
        and premise
        and f8_written > 0
        and e8_127 > 0
        and finite
        and nib_rt
    )
    return ok, detail


def prebuild_mx6(self_check_fn: Optional[Callable[[], Tuple[bool, str]]] = None) -> bool:
    """MX6 prebuild: prebuild() runs it after DO8's and DO8-ONLY's prebuild (triton_hstu_linear.prebuild_l11_do8 also
    with DO8 off, so a lone MX6 knob says it is INERT). Knob off: False at once (nothing else read, nothing printed).
    An INERT rule (_mx6_inert_reason): one INERT line, nothing built. Else build + check_resources_mx6 + load
    (_load_mx6), then self_check_mx6 (or `self_check_fn`); any failure latches MX6 only (one DISABLED line; DO8,
    DO8-ONLY and L11 untouched); a self-check device fault re-raises (abort before RUN_START, DO8's policy). Success,
    before RUN_START, in this order: the prebuild armed line (resources, ABI, cache key, self-check detail, keep-mask
    format, site set), MX6's ACTIVE line (what every MX6-site call runs; impl, device, site set and the registry's
    poison knob) and the counters line. An MX6 record prints nothing (_mx6_published), so no MX6 line falls in the
    timed region on the normal path. Runs once per process: a later call prints nothing."""
    global _mx6_prebuilt
    if not mx6_requested():
        return False
    if _mx6_prebuilt or _mx6_broken is not None or _mx6_inert is not None:
        return mx6_enabled()
    why = _mx6_inert_reason()
    if why is not None:
        _mx6_note_inert(why)
        return False
    t0 = time.perf_counter()
    try:
        _load_mx6()
    except Exception as error:  # noqa: BLE001 -- build / gate / load: latch MX6 only
        _disable_mx6(f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}")
        return False
    try:
        ok, detail = (self_check_fn or self_check_mx6)()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        msg = f"{type(error).__name__}: {error}"
        if l11._DEVICE_FAULT.search(msg):
            arm_print_all_ranks(f"{FAMILY_MX6} prebuild DEVICE FAULT: {msg[:600]}")
            raise
        _disable_mx6(f"prebuild self-check: {msg[:600]}")
        return False
    if not ok:
        _disable_mx6(f"prebuild self-check mismatch: {detail}")
        return False
    _mx6_prebuilt = True
    _only_atexit()
    r = _mx6_resources
    so_key = _mx6_lib_path.parent.name if _mx6_lib_path is not None else "?"
    arm_print_all_ranks(
        f"armed OK: {FAMILY_MX6} prebuild before run_start: built+loaded+self-check OK {time.perf_counter() - t0:.1f}s "
        f"(MX6 'only': VGPR {r.get('vgpr')}, spill {r.get('vgpr_spill')}, scratch {r.get('scratch')}, LDS "
        f"{r.get('lds')}, SGPR spill {r.get('sgpr_spill')}; FP8 twin VGPR {r.get('fp8_twin_vgpr')}; debug LDS "
        f"{r.get('dbg_lds')}; ABI {_ABI_VERSION_MX6}; so {so_key}; {detail}) mask={_MASK_DESC[mx6_mask_format()]} "
        f"sites={mx6_sites()}"
    )
    # MX6's ACTIVE line: here, right after the prebuild line (before RUN_START, once per process; an MX6 record prints
    # nothing). poison = the registry's DLRM_HSTU_FP8_L11_DO8_ONLY_POISON (registry(): the lazy import, as everywhere).
    reg = registry()
    dev = torch.device("cuda", torch.cuda.current_device())
    arm_print_all_ranks(
        f"armed OK: {FAMILY_MX6} ACTIVE mode=e5 only ({MX6_KNOB}, item {ITEM_MX6}, with items {ITEM}, {ITEM_ONLY}, "
        f"(39), (38)(a) and r14's G6/W6 producer): printed before run_start (prebuild, once per process); each "
        f"MX6-site call runs one {MX6_KERNEL_NAME} EM=2 launch (E2M3 x E2M3, E8M0 per 32) that replaces the FP8 "
        f"'only' launch, counted in the mx6 counters line (printed again at exit); impl={_impl_mx6} device={dev} "
        f"sites={mx6_sites()} poison={int(reg is not None and bool(reg.poison_requested()))}"
    )
    arm_print_all_ranks(mx6_summary_line())
    return True
