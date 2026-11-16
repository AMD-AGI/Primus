# pyre-unsafe
"""X2H (gin item (87)): E-X2b's two segment-reduce passes as one persistent HIP kernel (x2h_pass.cu), bit-identical to
the Triton pass.

DEFAULT OFF. Gate: DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=1 (1/true/on/yes; gin
apply_env_bootstrap.DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP, a pre-set env var wins; forwarded by
launch_slurm.sh's DLRM_HSTU_FP8_ prefix), read once per process. "0" is the kill switch. It acts only
while E-X2b (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE, ops/triton/triton_ec_dgrad_segreduce.py) is armed, and
on a device only after this module's pre-RUN_START prebuild passed there.

WHAT. triton_ec_dgrad_segreduce.dgrad_segreduce runs its two passes (cat: artist|album interleaved,
item) on x2h_pass instead of the Triton _ecx_dgrad_pass. Its allocations (acc [U, 512] fp32, frag_u /
frag_n [nt] int32, frag_v [nt, 512] fp32), its _ecx_combine and _ecx_zero_uncovered launches and the
rest of E-X2b are unchanged; each pass adds one small x2h_wpack launch (W1 into fragment order).
x2h_pass is one 512-thread workgroup per CU and persistent: a workgroup takes 6 tiles in a fixed static
order, then claims tiles from a per-stream, self-resetting atomic counter.
- Each of its 8 waves owns 64 of the 512 output columns. That wave keeps its W1 slice in registers
  (128 VGPR), prepacked per call by x2h_wpack.
- The dh rows of a 128-position tile are gathered into LDS by LDS-DMA (double-buffered, 528-B pitch;
  invalid positions read a 512-B zero row).
- The order / index / offset rows run 2-4 tiles ahead. Wave 7 derives each tile's segment slots and
  the 0/1 indicator tile one tile ahead.
- There is one barrier per tile, plus 2 per extra slot or half pass of a slow-path tile. Stores are
  deferred past the barrier.
The arithmetic is Triton's, operation for operation:
- the main dot: v_mfma_f32_32x32x16_bf16, 16 K-steps ascending from +0.0;
- xb = v_cvt_pk_bf16_f32 (RNE);
- the indicator dot: v_mfma_f32_16x16x32_bf16, 4 position steps ascending from +0.0;
- the same slots per pass, the same 2nd/3rd... passes for > 16 segments or both W1_a halves in a
  tile, and the same STORE / fragment rules.
Role swap (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP_ROLE, diagnostic only, default 1): 1 feeds the main
dot with swapped MFMA operand roles, so the 32x32 result block is transposed into the indicator
operands in registers (permlane swaps). 0 keeps Triton's roles and transposes through wave-private
LDS. MFMA results are bitwise invariant under the role swap: actckpt iteration 68, GPU gate G1, 0
mismatches over >= 131,072 adversarial operand sets per class, inf / NaN payloads included.

NUMERICS: none relative to E-X2b. Bit-identical to the Triton pass; item (77)'s declared fp32
reduction-order class is unchanged. Evidence (actckpt iteration 68, gfx950, ROCm 7.2.1 hipcc of
rocm/primus:v26.3, kernel-lab job 69883): G2 compared acc, FRAG_U / FRAG_N / FRAG_V and the
emitted addends with ship14's Triton pass (module sha256 801bca83), both roles. 0 elements differ on
the prebuild input, a structural set (153,637 positions: > 32 segments per tile, all-mixed halves,
hot rows), real early / late / high-M ids (M 2,356,560 / 2,507,492 / 2,787,229), a rounding-heavy
adversarial set and injected inf / NaN. G4: grids 1 / 7 / 64 / #CU / 2 #CU and reruns are bitwise
identical. Each run repeats a narrower version of this check before RUN_START (PREBUILD: one role,
finite inputs, three grids, sentinel-filled outputs) and stays on Triton on any mismatch.
MLPerf: the same FLOPs as the Triton pass. Every token's dX row is computed and every segment
addition is made; only the schedule differs.

PERFORMANCE [lab, M 2,356,560, one GPU, job 69883]: the two passes take 2.990 ms with Triton
(V0) and 1.976 ms with x2h_pass role 1 (2.094 ms role 0). That saves 0.986 ms per rank-step
scaled to M 2.29M. At the 1000 W cap (sustained) it saves 0.906 ms; x2h_pass then clocks about
1610 MHz against V0's 1850 MHz. In-model saving and neighbour-kernel give-back: the pre-registered
rung proposals/ship15/x2h/PRED_x2h.md (b9b373a1).

BUILD (prebuild only; never after RUN_START): hipcc (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP_HIPCC, else
$ROCM_PATH/bin/hipcc, else /opt/rocm/bin/hipcc) compiles x2h_pass.cu (--offload-arch=gfx950 -std=c++20
-O3 -ffp-contract=off -shared -fPIC; -Rpass-analysis=kernel-resource-usage only adds the remarks the
census reads, the gfx950 code is byte-identical without it) into /tmp/dlrm_hstu_ec_dgrad_x2h/<sha16 of
source + flags + arch + `hipcc --version` + the HIPCC_* / HIP_CLANG_PATH / HIP_DEVICE_LIB_PATH env>/
libx2h.so (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP_CACHE overrides the root). It is the wgrad_ck.build
pattern: a finished entry (the .so and its build.log) skips the compile, and racing ranks build to
unique temporary files and os.replace them (about 2 s in the container). The cache root and the
entry are created 0700 whatever the umask (launch_slurm.sh runs the workers under umask 000, and
Path.mkdir(parents=True) applies its mode to the last component only); an existing directory is never
chmod-ed. A cache directory that is not ours or is group/world-writable is not trusted: the process
then builds into a private mkdtemp directory, removed once its library is loaded (the mapping stays
valid; the build log is kept in memory). An entry that fails to load, fails the ABI check or lacks its
census is deleted and rebuilt once. A build past the timeout kills hipcc's whole process group. The
.so is loaded with ctypes and its kernels launch on the caller's current stream.

PREBUILD (blas_warmup._prebuild_ec_dgrad_x2h, before RUN_START, inside E-X2b's prebuild bracket,
after E-X2b's own prebuild passed). On the rank's device it runs these steps:
  1. builds and loads the library; requires gfx950 and a resource census of x2h_pass<role, 0> showing
     no VGPR spill and no scratch (lab: role 1 214 VGPR, role 0 218, LDS 152,992 / 163,232 B,
     occupancy 2); no census = fail;
  2. exact_check(): dgrad_segreduce through x2h_pass on E-X2b's exact synthetic input (prebuild()'s
     _synthetic), bitwise against the fp64 reference;
  3. cross_check(): the BITWISE CROSS-CHECK. The Triton passes and x2h_pass (grids #CU, 7, #CU again)
     run on crosscheck_input(), and every acc word must agree.
Before every one of these calls, acc and the fragment buffers are filled with sentinels
(dgrad_segreduce _prefill): NaN bits in acc / frag_v, "no open run" (-1 / 0) in frag_u / frag_n. A
skipped write therefore cannot pass on memory recycled from an earlier, correct call. After every
x2h_pass call the tile counters must be back at 0 and the zero row still zero.
crosscheck_input() is a deterministic rounding-heavy input. It touches no RNG state: a private
random.Random lays out the segments and integer hashes give the values. Each member has
> 6 x #CU tiles (so every workgroup claims tiles) and a partial last tile. It holds
single-position segments (tiles with > 32 segments), cat tiles holding both halves, hot rows
spanning several tiles (the combine), segments ending on a tile end, uncovered rows, exactly
cancelling addend pairs, -0.0 dh rows / W1 columns, and |values| over 2^-8 .. 2^8.
Then the device is armed. Any failure latches X2H off (one FAILED CLOSED line) and E-X2b keeps the
Triton passes. A device fault surfaces at blas_warmup's synchronize and fails the run there. The
transient (~0.8 GB live at 256 CUs: inputs 0.19 GB, two 0.25 GB acc, a compare mask) is released
(torch.cuda.empty_cache) before alloc_pregrow.

FALLBACKS (every one leaves E-X2b on, with the Triton passes):
- knob off; E-X2b off; emit (debug dX) requested;
- no prebuild passed on the device (NOT armed: nothing is built after RUN_START);
- an input outside x2h_pass's contract, or CUDA graph capture (DECLINED, for that call; logged once
  per reason class):
  - dh rows >= 2^22;
  - dh row stride not a multiple of 8 elements or < 256;
  - dh larger than 2^31 bytes;
  - a member tensor not 16-byte aligned (production passes fresh or 16-B aligned views;
    ec_offsets / onesweep_sort);
- a raising launch, or any error in this module's call path: latches X2H off; the Triton passes rerun
  into the same buffers, which they overwrite as on any call.

ARMED (every rank, once, tagged "[rank N]"):
  "armed OK: [ec-dgrad-x2h] pre-run_start prebuild OK (..."
  "armed OK: [ec-dgrad-x2h] fused passes on cuda:N ..."
  and E-X2b's "armed OK: [ec-dgrad-segreduce] fused" line names "x2h_pass x2".
Must be ABSENT: "[ec-dgrad-x2h] FAILED CLOSED", "[ec-dgrad-x2h] DECLINED",
"[ec-dgrad-x2h] NOT armed", "[ec-dgrad-x2h] DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=1 IGNORED",
"[ec-dgrad-segreduce] X2H unavailable". Informational: "[ec-dgrad-x2h] stale HIP error ... drained"
(an earlier call's error left on the thread, not ours).

OWNERSHIP: actckpt (item (87) X2H). Imported lazily: only by triton_ec_dgrad_segreduce and blas_warmup,
and only with the knob on. Importing it builds and loads nothing.
"""

import ctypes
import hashlib
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from generative_recommenders.ops.utils import arm_print_all_ranks

ENV_VAR: str = "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP"
ROLE_ENV_VAR: str = ENV_VAR + "_ROLE"  # diagnostic: "1" (default) | "0"
HIPCC_ENV_VAR: str = ENV_VAR + "_HIPCC"
CACHE_ENV_VAR: str = ENV_VAR + "_CACHE"
TAG: str = "[ec-dgrad-x2h]"
ARMED_PREFIX: str = "armed OK: " + TAG
PASS_NAME: str = "x2h_pass"

ABI_VERSION: int = 2026101202  # x2h_pass.cu X2H_ABI_VERSION
# x2h_pass.cu as verified on gfx950 (actckpt iteration 68, kernel-lab job 69883). ops/tests pins the
# file to it: an edited kernel needs the GPU gates again (proposals/ship15/x2h).
SOURCE_SHA256: str = "d50b03f34bf9db0824c8cd1bd91281a8944ed5c9aca2c16b7633ddeb05dc98b8"
HID: int = 256  # dh width = K of the main dot
WIDTH: int = 512  # EC row width = columns per W1 half
BM: int = 128  # positions per tile (Triton BM)
NSTATIC: int = 6  # static tile-sequence entries per workgroup (x2h_pass.cu NSTATIC)
MAX_DH_ROWS: int = 1 << 22  # 32-bit dh byte offsets
STRESS_GRID: int = 7  # cross-check grid: long claim chains through the tile counters

_ARCH: str = "gfx950"
_SOURCE: Path = Path(__file__).with_name("x2h_pass.cu")
_FLAGS: Tuple[str, ...] = ("-std=c++20", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                           "-Rpass-analysis=kernel-resource-usage")
# hipcc reads these; they change its output, so they are part of the cache key
_HIPCC_ENV: Tuple[str, ...] = ("HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "HIP_CLANG_PATH",
                               "HIP_DEVICE_LIB_PATH")
_CACHE_ROOT: str = "/tmp/dlrm_hstu_ec_dgrad_x2h"
_LIB_NAME: str = "libx2h.so"
_PRIVATE_PREFIX: str = "x2h_private_"  # mkdtemp prefix of a private (untrusted dir or rebuild) build
_BUILD_TIMEOUT_S: int = 600

_REQUESTED: Optional[bool] = None
_ROLE: Optional[int] = None
_BROKEN: Optional[str] = None
_lock = threading.Lock()
_lib: Optional["_Lib"] = None
_toolchains: Dict[str, str] = {}
_armed: Set[int] = set()  # device indices whose prebuild passed
_ONCE: Set[Tuple[str, str]] = set()
# prebuild summary per device index (tests / diagnostics)
PREBUILD_INFO: Dict[int, Dict[str, Any]] = {}


def _say(msg: str) -> None:
    arm_print_all_ranks(msg)


def _once(site: str, reason: str, msg: str) -> None:
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        _say(msg)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ------------------------------------------------------------------------------------------- gate
def requested() -> bool:
    """The knob (read once per process)."""
    global _REQUESTED
    if _REQUESTED is None:
        _REQUESTED = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
    return _REQUESTED


def role_swap() -> int:
    """1 (default) or 0 (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP_ROLE, read once); another value raises
    (prebuild then fails closed)."""
    global _ROLE
    if _ROLE is None:
        raw = os.environ.get(ROLE_ENV_VAR, "").strip()
        if raw not in ("", "0", "1"):
            raise ValueError(f"{ROLE_ENV_VAR}={raw!r}: must be 1 (default) or 0")
        _ROLE = 0 if raw == "0" else 1
    return _ROLE


def broken() -> Optional[str]:
    return _BROKEN


def latch_off(where: str, exc: BaseException) -> None:
    """X2H off for the rest of the process; E-X2b keeps running its Triton passes."""
    global _BROKEN
    if _BROKEN is None:
        _BROKEN = f"{where}: {type(exc).__name__}: {str(exc)[:300]}"
        try:
            _say(f"{TAG} FAILED CLOSED -> Triton _ecx_dgrad_pass for the rest of the run ({_BROKEN})")
        except Exception:  # pragma: no cover - a failing print must not escape into E-X2b
            pass


def _dev_index(dev: torch.device) -> int:
    if dev.index is not None:
        return int(dev.index)
    return int(torch.cuda.current_device()) if dev.type == "cuda" else -1


def active(dev: torch.device) -> bool:
    """dgrad_segreduce runs x2h_pass on `dev`: requested, not latched off, prebuilt on this device.
    Never builds and never raises."""
    try:
        if not requested() or _BROKEN is not None:
            return False
        idx = _dev_index(dev)
        if idx not in _armed:
            _once("not_armed", str(idx),
                  f"{TAG} NOT armed on cuda:{idx}: no pre-run_start prebuild passed on this device "
                  "(BLAS_WARMUP off?) -> Triton _ecx_dgrad_pass (nothing is built after RUN_START)")
            return False
        return True
    except Exception:  # pragma: no cover - defensive
        return False


# ------------------------------------------------------------------------------------------ build
def _hipcc() -> str:
    override = os.environ.get(HIPCC_ENV_VAR, "").strip()
    if override:
        return override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        cand = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(cand):
            return cand
    return "/opt/rocm/bin/hipcc"


def _toolchain(hipcc: str) -> str:
    """`hipcc --version` (cached per hipcc path; raises if hipcc cannot run)."""
    v = _toolchains.get(hipcc)
    if v is None:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{hipcc} --version rc={proc.returncode}: "
                               f"{((proc.stderr or '') + (proc.stdout or '')).strip()[-400:]}")
        v = proc.stdout
        _toolchains[hipcc] = v
    return v


def cache_dir(source: Path = _SOURCE) -> Path:
    root = os.environ.get(CACHE_ENV_VAR, "").strip() or _CACHE_ROOT
    h = hashlib.sha256()
    h.update(source.read_bytes())
    h.update(b"\0" + "\0".join(_FLAGS).encode())
    h.update(b"\0" + _ARCH.encode())
    # The toolchain too: a .so another hipcc built is not reused (the census holds for this build).
    h.update(b"\0" + _toolchain(_hipcc()).encode())
    h.update(b"\0" + "\0".join(f"{k}={os.environ.get(k, '')}" for k in _HIPCC_ENV).encode())
    return Path(root) / h.hexdigest()[:16]


def _trusted(p: Path) -> bool:
    """Ours and not writable by group / others (a planted .so must not be loaded)."""
    st = p.stat()
    return st.st_uid == os.getuid() and not st.st_mode & 0o022


def _mkdir_0700(p: Path) -> None:
    """mkdir `p` 0700 whatever the umask. Only a directory this call created is chmod-ed: an existing one
    is left as it is (_trusted then decides whether it is used)."""
    try:
        p.mkdir(mode=0o700)
    except FileExistsError:
        return
    os.chmod(p, 0o700)  # mkdir's mode is masked by the umask; this sets it exactly


def _rm_private(so: Path) -> None:
    """Remove a private build directory (nobody else can reuse it; a loaded .so stays mapped)."""
    if so.parent.name.startswith(_PRIVATE_PREFIX):
        shutil.rmtree(so.parent, ignore_errors=True)


def _hipcc_run(cmd: List[str]) -> Tuple[int, str]:
    """Run hipcc in its own process group; on the timeout the whole group (clang children too) is killed."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=_BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s: {' '.join(cmd)}")
    return proc.returncode, out or ""


def build(source: Path = _SOURCE, fresh: bool = False) -> Tuple[Path, str, float, bool]:
    """(library path, build log, build seconds, cache hit). Compiles x2h_pass.cu unless a finished cache
    entry (.so + build.log) exists. fresh=True (a bad entry): deletes the shared entry (no cross-process
    lock: a racing rank may still read or replace it) and builds into a private directory, so the new
    library has a path this process has not dlopen()ed before (load() removes it once loaded). Raises on
    any failure (a failed private build leaves no directory behind)."""
    d = cache_dir(source)
    d.parent.parent.mkdir(parents=True, exist_ok=True)
    _mkdir_0700(d.parent)
    _mkdir_0700(d)
    if fresh:
        for p in (d / _LIB_NAME, d / "build.log"):
            try:
                p.unlink()
            except OSError:
                pass
    if fresh or not (_trusted(d.parent) and _trusted(d)):
        d = Path(tempfile.mkdtemp(prefix=_PRIVATE_PREFIX))  # a rebuild, or an untrusted shared dir
    so, log = d / _LIB_NAME, d / "build.log"
    if so.is_file() and log.is_file():
        return so, log.read_text(errors="replace"), 0.0, True
    uniq = f"{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    tmp, tmp_log = d / f"tmp-{uniq}-{_LIB_NAME}", d / f"tmp-{uniq}-build.log"
    cmd = [_hipcc(), str(source), f"--offload-arch={_ARCH}", *_FLAGS, "-o", str(tmp)]
    t0 = time.perf_counter()
    try:
        rc, out = _hipcc_run(cmd)
        text = " ".join(cmd) + "\n" + out
        if rc != 0 or not tmp.is_file():
            raise RuntimeError(f"hipcc rc={rc}: {' '.join(cmd)}: {text.strip()[-1200:]}")
        tmp_log.write_text(text)
        os.replace(tmp_log, log)  # the log first: whoever sees the .so also finds its census
        os.replace(tmp, so)
    except BaseException:
        _rm_private(so)
        raise
    finally:
        for p in (tmp, tmp_log):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return so, text, time.perf_counter() - t0, False


_CENSUS_KEYS = ("VGPRs Spill", "SGPRs Spill", "VGPRs", "AGPRs", "ScratchSize [bytes/lane]",
                "Occupancy [waves/SIMD]", "LDS Size [bytes/block]")
# ROCm 10.2 (no "remark:" / [-Rpass...]): "<file>:<line>:0: Function Name: <mangled>", "<file>:<line>:0:  <Key>: <v>"
_LOC_ROCM102: str = r"[^\s:][^:]*:\d+:\d+:"


def census(log: str) -> Dict[str, Dict[str, int]]:
    """{mangled kernel name: {remark key: value}} from -Rpass-analysis=kernel-resource-usage remarks."""
    out: Dict[str, Dict[str, int]] = {}
    cur: Optional[Dict[str, int]] = None
    keys = "|".join(re.escape(k) for k in _CENSUS_KEYS)
    for line in log.splitlines():
        m = (re.search(r"remark:\s+Function Name:\s+(\S+)", line)
             or re.match(_LOC_ROCM102 + r"\s+Function Name:\s+(\S+)\s*$", line))
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = (re.search(r"remark:\s+(" + keys + r"):\s+(\d+)", line)
             or re.match(_LOC_ROCM102 + r"\s+(" + keys + r"):\s+(\d+)\s*$", line))
        if m and cur is not None:
            cur.setdefault(m.group(1), int(m.group(2)))
    return out


def pass_kernel(role: int) -> str:
    """Mangled name of x2h_pass<role, EMIT=0> (the production instantiation)."""
    return f"_ZN3x2h8x2h_passILi{int(role)}ELi0EEEvNS_6ParamsE"


def census_ok(res: Optional[Dict[str, int]]) -> bool:
    """A census entry exists and shows no VGPR spill and no scratch (a missing key counts as a failure)."""
    return bool(res) and res.get("VGPRs Spill", 1) == 0 and res.get("ScratchSize [bytes/lane]", 1) == 0


# ctypes signatures of the C ABI (x2h_pass.cu section 5; ops/tests checks them against the source).
_VP, _I, _LL = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
SIGNATURES: Dict[str, Tuple[Any, Tuple[Any, ...]]] = {
    "x2h_abi_version": (_I, ()),
    "x2h_lds_bytes": (_I, (_I,)),
    "x2h_wp_elems": (_I, (_I,)),
    "x2h_drain_last_error": (_I, ()),
    "x2h_wpack_launch": (_I, (_VP, _I, _VP, _I, _VP)),
    "x2h_pass_grid": (_I, (_I, _I, _I)),
    "x2h_pass_launch": (_I, (_I, _I, _VP, _LL, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _LL, _VP,
                             _VP, _I, _I, _I, _I, _I, _VP)),
}


class _Lib:
    def __init__(self, so: Path, log: str, build_s: float, cached: bool) -> None:
        self.path, self.log, self.build_s, self.cached = so, log, build_s, cached
        self.census = census(log)
        self.cdll = ctypes.CDLL(str(so))
        for name, (res, args) in SIGNATURES.items():
            fn = getattr(self.cdll, name)
            fn.restype = res
            fn.argtypes = list(args)
            setattr(self, name, fn)
        got = int(self.x2h_abi_version())
        if got != ABI_VERSION:
            raise RuntimeError(f"{so}: x2h_abi_version()={got}, expected {ABI_VERSION}")


def load(role: Optional[int] = None) -> "_Lib":
    """Build (or find in the cache) and load the library. Called by prebuild() only. A cache entry that
    fails to load, fails the ABI check, or (role given) lacks the census of x2h_pass<role, 0> is deleted
    and rebuilt once (a stale entry must not disable X2H on the node for every later job). A private build
    directory is removed once its library is loaded or refused."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                built = build()
                try:
                    lib = _Lib(*built)
                    if role is not None and built[3] and lib.census.get(pass_kernel(role)) is None:
                        raise RuntimeError("cached entry without a census")
                except Exception:
                    _rm_private(built[0])
                    if not built[3]:
                        raise
                    built = build(fresh=True)
                    try:
                        lib = _Lib(*built)
                    finally:
                        _rm_private(built[0])
                _rm_private(built[0])
                _lib = lib
    return _lib


def _check_rc(rc: int, what: str) -> None:
    if rc < 0:  # launched; -rc is a stale error an earlier call had left on this thread
        _once("stale", str(-rc), f"{TAG} stale HIP error {-rc} drained before {what} "
              "(left on this thread by an earlier call, not ours)")
    elif rc != 0:
        raise RuntimeError(f"{what} returned hip error {rc}")


# ---------------------------------------------------------------------------------------- runtime
class _CudaRuntime:
    """The device side (tests substitute a mock): streams, grid, the zero row, the tile counters,
    the W prepack and the pass launch."""

    def __init__(self) -> None:
        self._zrow: Dict[int, torch.Tensor] = {}
        self._counters: Dict[Tuple[int, int], torch.Tensor] = {}
        self._grid: Dict[int, int] = {}

    @staticmethod
    def stream(dev: torch.device) -> int:
        # _ecx_combine / _ecx_zero_uncovered launch on the CURRENT device's stream; x2h_pass must
        # run on the same stream, so tensors on another device fail (the caller sets the device).
        cur = torch.cuda.current_device()
        if dev.index is not None and dev.index != cur:
            raise ValueError(f"tensors on {dev} but the current device is cuda:{cur}")
        return int(torch.cuda.current_stream(dev).cuda_stream)

    @staticmethod
    def capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def grid(self, dev: torch.device) -> int:
        idx = _dev_index(dev)
        g = self._grid.get(idx)
        if g is None:
            g = int(torch.cuda.get_device_properties(dev).multi_processor_count)
            self._grid[idx] = g
        return g

    def zero_row(self, dev: torch.device) -> torch.Tensor:
        idx = _dev_index(dev)
        z = self._zrow.get(idx)
        if z is None:
            z = torch.zeros(HID, dtype=torch.bfloat16, device=dev)  # 512 B of +0.0
            # created at prebuild: wait for the fill once, so a launch on any stream reads zeros
            torch.cuda.current_stream(dev).synchronize()
            self._zrow[idx] = z
        return z

    def counters(self, dev: torch.device, stream: int) -> torch.Tensor:
        """8 int32 per (device, stream): cat [0:2], item [4:6]. Zero once (stream-ordered fill),
        reset by each pass's last workgroup."""
        key = (_dev_index(dev), int(stream))
        c = self._counters.get(key)
        if c is None:
            c = torch.zeros(8, dtype=torch.int32, device=dev)
            self._counters[key] = c
        return c

    @staticmethod
    def wpack(lib: "_Lib", w1: torch.Tensor, nhalf: int, stream: int) -> torch.Tensor:
        """W1 [256, 512 nhalf] bf16 -> fragment-order Wp (x2h_wpack), on `stream`."""
        wp = torch.empty(int(lib.x2h_wp_elems(nhalf)), dtype=torch.bfloat16, device=w1.device)
        _check_rc(int(lib.x2h_wpack_launch(w1.data_ptr(), int(w1.shape[1]), wp.data_ptr(), nhalf,
                                           stream)), "x2h_wpack")
        return wp

    @staticmethod
    def launch(lib: "_Lib", a: Dict[str, Any]) -> None:
        g = int(lib.x2h_pass_grid(a["npos"], a["grid"], 0))
        if g != min(a["grid"], _cdiv(a["npos"], BM)):
            raise RuntimeError(f"x2h_pass_grid {g} != min(grid {a['grid']}, tiles) (stale library?)")
        rc = lib.x2h_pass_launch(a["role"], 0, a["dh"], a["sdh"], a["wp"], a["idx"], a["order"],
                                 a["off"], a["acc"], a["frag_u"], a["frag_n"], a["frag_v"], None, 0,
                                 a["zrow"], a["counters"], a["npos"], a["frag_base"], a["is_cat"],
                                 a["grid"], 0, a["stream"])
        _check_rc(int(rc), f"x2h_pass({'cat' if a['is_cat'] else 'item'})")


_RT: Any = _CudaRuntime()
# the arguments of the last launches (tests / diagnostics)
LAST_LAUNCHES: Dict[str, Dict[str, Any]] = {}


def decline_reason(m: Any, kind: str) -> Optional[str]:
    """Why member `m` is outside x2h_pass's contract (beyond E-X2b's _check_member), or None."""
    rows = int(m.dh.shape[0])
    if rows >= MAX_DH_ROWS:
        return f"{kind} dh has {rows} rows >= 2^22"
    sdh = int(m.dh.stride(0)) if rows > 1 else HID
    if sdh % 8 != 0 or sdh < HID:
        return f"{kind} dh row stride {sdh} (needs a multiple of 8, >= {HID})"
    if max(rows, 1) * sdh * 2 >= 2**31:
        return f"{kind} dh larger than 2^31 bytes"
    for name in ("dh", "w1", "idx", "order", "offsets"):
        if int(getattr(m, name).data_ptr()) % 16:
            return f"{kind} {name} not 16-byte aligned"
    return None


def run_passes(cat: Any, item: Any, acc: torch.Tensor, frag_u: torch.Tensor, frag_n: torch.Tensor,
               frag_v: torch.Tensor, n_a: int, grid: Optional[int] = None, strict: bool = False) -> bool:
    """E-X2b's cat and item passes on x2h_pass into dgrad_segreduce's buffers (cat fragment rows
    0 .. n_a-1, item n_a ..). True: both launched (the caller adds combine + zero). False: the caller
    launches the Triton passes, after a decline (logged once per reason class) or after any error in
    this path (X2H latched off). strict (prebuild): raise instead. The members passed E-X2b's
    _check_member. Nothing in here raises into E-X2b unless strict."""
    try:
        reasons = [decline_reason(m, k) for m, k in ((cat, "cat"), (item, "item"))]
        reasons += [f"{n} not 16-byte aligned" for n, t in (("acc", acc), ("frag_u", frag_u),
                                                             ("frag_n", frag_n), ("frag_v", frag_v))
                    if int(t.data_ptr()) % 16]
        if _RT.capturing():
            reasons.append("CUDA graph capture (the tile counters are per eager stream)")
        why = next((r for r in reasons if r is not None), None)
        if why is not None:
            if strict:
                raise ValueError(f"{TAG} declined: {why}")
            _once("decline", re.sub(r"\d+", "N", why),
                  f"{TAG} DECLINED ({why}) -> Triton _ecx_dgrad_pass for such calls")
            return False
        dev = acc.device
        lib = _lib
        if lib is None:
            raise RuntimeError("library not loaded (prebuild did not run)")
        role, rt = role_swap(), _RT
        stream = rt.stream(dev)
        g = int(grid) if grid is not None else rt.grid(dev)
        if g <= 0:
            raise ValueError(f"grid {g}")
        zrow, ctr = rt.zero_row(dev), rt.counters(dev, stream)
        launches: Dict[str, Dict[str, Any]] = {}
        keep: List[torch.Tensor] = []  # W packs live until both passes are queued (stream-ordered)
        for kind, m, fbase, is_cat, nhalf, ctr_off in (("cat", cat, 0, 1, 2, 0),
                                                       ("item", item, int(n_a), 0, 1, 16)):
            npos = int(m.idx.numel())
            if not npos:
                continue
            wp = rt.wpack(lib, m.w1, nhalf, stream)
            keep.append(wp)
            rows = int(m.dh.shape[0])
            a = dict(role=role, dh=m.dh.data_ptr(), sdh=int(m.dh.stride(0)) if rows > 1 else HID,
                     wp=wp.data_ptr(), idx=m.idx.data_ptr(), order=m.order.data_ptr(),
                     off=m.offsets.data_ptr(), acc=acc.data_ptr(), frag_u=frag_u.data_ptr(),
                     frag_n=frag_n.data_ptr(), frag_v=frag_v.data_ptr(), zrow=zrow.data_ptr(),
                     counters=ctr.data_ptr() + ctr_off, npos=npos, frag_base=fbase, is_cat=is_cat,
                     grid=g, stream=stream)
            rt.launch(lib, a)
            launches[kind] = a
        del keep
        LAST_LAUNCHES.clear()
        LAST_LAUNCHES.update(launches)
        if not strict:  # a training call (prebuild's calls are strict)
            idx = _dev_index(dev)
            if ("fused", str(idx)) not in _ONCE:
                _once("fused", str(idx),
                      f"{ARMED_PREFIX} fused passes on cuda:{idx} -- {PASS_NAME} role_swap={role} x2 + "
                      f"x2h_wpack x2 (cat T={int(cat.idx.numel())}, item T={int(item.idx.numel())}, "
                      f"U={int(acc.shape[0])}, grid {g}) + _ecx_combine + _ecx_zero_uncovered, "
                      f"bit-identical to _ecx_dgrad_pass (prebuild cross-check); kill switch {ENV_VAR}=0")
        return True
    except Exception as exc:
        if strict:
            raise
        latch_off("x2h_pass call", exc)
        return False


# ------------------------------------------------------------------------------- cross-check input
_M32 = 0xFFFFFFFF
_REGIMES = (("single", 2), ("short", 3), ("mid", 3), ("hot", 1), ("aligned", 1), ("geo", 3))


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer hash of int64 x (low 32 bits); stays below 2^59 in int64."""
    x = x & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    return x ^ (x >> 16)


def _hashed(n: int, salt: int, device: torch.device) -> torch.Tensor:
    return _mix32(torch.arange(n, dtype=torch.int64, device=device) * 0x9E3779B1 + salt)


def _bf16_values(rows: int, cols: int, salt: int, device: torch.device) -> torch.Tensor:
    """[rows, cols] bf16 from hashes: random sign, exponent 2^-8 .. 2^8, all 7 mantissa bits,
    ~0.8 % exact +0.0 (rounding-heavy: K=256 products span 2^32 in magnitude)."""
    h = _hashed(rows * cols, salt, device)
    bits = ((h >> 31) << 15) | ((119 + ((h >> 16) & 0x7FFF) % 17) << 7) | (h & 0x7F)
    bits = torch.where(((h >> 8) & 0xFF) < 2, torch.zeros_like(bits), bits)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return bits.view(torch.bfloat16).view(rows, cols).contiguous()


def _design(rnd: random.Random, totals: List[int], start_id: int,
            big: int = 9000) -> Tuple[List[Tuple[int, int, int]], int]:
    """Sorted-order segment design [(half, row id, length)] in row-id order (the lab's struct
    design): regimes of single-position, short (1-5), mid (5-12), hot (300-2500), tile-end-aligned
    and geometric (mean ~20) segments, one big row at 1/3, ~3 % of row ids skipped (uncovered).
    Two totals (cat): consecutive segments alternate halves, so most tiles hold both."""
    halves, rem = len(totals), list(totals)
    out: List[Tuple[int, int, int]] = []
    rid, pos, i, left, regime, big_done = start_id, 0, 0, 0, "geo", False
    total = remaining = sum(totals)
    names, weights = [r for r, _ in _REGIMES], [w for _, w in _REGIMES]
    while remaining > 0:
        if left <= 0:
            regime = rnd.choices(names, weights)[0]
            left = rnd.randint(256, 1024)
        h = i % halves
        if rem[h] == 0:
            h = 1 - h
        if not big_done and pos >= total // 3:
            length, big_done = big, True
        elif regime == "single":
            length = 1
        elif regime == "short":
            length = rnd.randint(1, 5)
        elif regime == "mid":
            length = rnd.randint(5, 12)
        elif regime == "hot":
            length = rnd.randint(300, 2500)
        elif regime == "aligned":
            length = BM - pos % BM + BM * rnd.randint(0, 2)
        else:
            length = min(1 + int(rnd.expovariate(1 / 20.0)), 200)
        length = max(1, min(length, rem[h]))
        if rnd.random() < 0.03:
            rid += rnd.randint(1, 3)
        out.append((h, rid, length))
        rid, rem[h], remaining, pos, i, left = rid + 1, rem[h] - length, remaining - length, \
            pos + length, i + 1, left - length
    return out, rid


def _segments(idx: torch.Tensor, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(idx, stable=True)
    off = torch.zeros(num_rows + 1, dtype=torch.int64, device=idx.device)
    off[1:] = torch.cumsum(torch.bincount(idx, minlength=num_rows), 0)
    return order.contiguous(), off


def _cancel_pairs(sel: torch.Tensor, neg: torch.Tensor, idx: torch.Tensor, order: torch.Tensor,
                  offsets: torch.Tensor, is_cat: bool, salt: int) -> int:
    """For ~1/3 of the segments with >= 2 positions: the token at the 2nd position takes the 1st
    token's dh row negated, so their addends cancel exactly (RNE is sign-symmetric). Returns the
    number of pairs (duplicate 2nd tokens dropped, deterministically)."""
    ln = offsets[1:] - offsets[:-1]
    rows = torch.arange(ln.numel(), device=ln.device)
    pick = (ln >= 2) & (_mix32(rows + salt) % 3 == 0)
    s = offsets[:-1][pick]
    o1, o2 = order[s], order[s + 1]
    t1, t2 = (o1 >> 1, o2 >> 1) if is_cat else (o1, o2)
    srt = torch.argsort(t2, stable=True)
    t2s = t2[srt]
    first = torch.ones_like(t2s, dtype=torch.bool)
    first[1:] = t2s[1:] != t2s[:-1]
    t1k, t2k = t1[srt][first], t2s[first]
    sel[t2k] = sel[t1k]
    neg[t2k] = ~neg[t1k]
    return int(t2k.numel())


def crosscheck_input(device: torch.device, grid: int, member: Any,
                     seed: int = 20261012) -> Tuple[Any, Any, int, Dict[str, Any]]:
    """(cat, item, U, info): the prebuild cross-check input (module docstring, PREBUILD 3). No RNG
    state is read or written. member: the E-X2b Member namedtuple."""
    ti = BM * (NSTATIC * grid + 259) + 77  # item positions: partial last tile (77)
    mc = (BM * (NSTATIC * grid + 261) - 58) // 2  # cat tokens: 2 mc positions, partial last tile
    rnd = random.Random(seed)
    item_rows, nxt = _design(rnd, [ti], 0)
    cat_rows, nxt2 = _design(rnd, [mc, mc], nxt + 5)
    num_rows = nxt2 + 7  # trailing uncovered rows

    def tokens(rows: List[Tuple[int, int, int]], h: int, salt: int) -> torch.Tensor:
        ids = torch.tensor([r for hh, r, _ in rows if hh == h], dtype=torch.int64)
        lens = torch.tensor([n for hh, _, n in rows if hh == h], dtype=torch.int64)
        v = torch.repeat_interleave(ids, lens).to(device)
        return v[torch.argsort(_hashed(v.numel(), salt, device), stable=True)].contiguous()

    item_idx = tokens(item_rows, 0, 11)
    cat_idx = torch.stack([tokens(cat_rows, 0, 12), tokens(cat_rows, 1, 13)], 1).reshape(-1).contiguous()
    (o_c, f_c), (o_i, f_i) = _segments(cat_idx, num_rows), _segments(item_idx, num_rows)

    bank = _bf16_values(4096, HID, 21, device)
    info: Dict[str, Any] = {"cat_segments": len(cat_rows), "item_segments": len(item_rows)}
    dhs = []
    for kind, n, idx, order, off, salt in (("cat", mc, cat_idx, o_c, f_c, 31),
                                           ("item", ti, item_idx, o_i, f_i, 41)):
        sel = _hashed(n, salt, device) % 4096
        neg = torch.zeros(n, dtype=torch.bool, device=device)
        info[f"{kind}_cancel_pairs"] = _cancel_pairs(sel, neg, idx, order, off, kind == "cat", salt + 1)
        dh = bank[sel]
        dh = torch.where(neg[:, None], -dh, dh)
        nz = _hashed(n, salt + 2, device) % 997 == 0  # -0.0 rows
        dh.view(torch.int16)[nz] = -32768
        info[f"{kind}_negzero_rows"] = int(nz.sum())
        dhs.append(dh.contiguous())
    w1s = []
    for kind, cols, salt in (("cat", 2 * WIDTH, 51), ("item", WIDTH, 61)):
        w = _bf16_values(HID, cols, salt, device)
        nzc = _hashed(cols, salt + 1, device) % 61 == 0  # -0.0 columns
        w.view(torch.int16)[:, nzc] = -32768
        info[f"{kind}_negzero_cols"] = int(nzc.sum())
        w1s.append(w.contiguous())
    cat = member(dhs[0], w1s[0], cat_idx, o_c, f_c)
    item = member(dhs[1], w1s[1], item_idx, o_i, f_i)
    return cat, item, num_rows, info


def structure_stats(m: Any, is_cat: bool) -> Dict[str, int]:
    """Coverage of one member: positions, tiles, last-tile size, max segments in a tile, tiles with
    > 16 / > 32 segments, cat tiles holding both halves, max tiles a row spans."""
    T = int(m.idx.numel())
    dev = m.idx.device
    tiles = _cdiv(T, BM)
    s = torch.arange(T, device=dev)
    tile_of = torch.div(s, BM, rounding_mode="floor")
    start = ((s % BM) == 0) | (m.offsets[m.idx[m.order]] == s)
    nseg = torch.zeros(tiles, dtype=torch.int64, device=dev).index_add_(0, tile_of, start.to(torch.int64))
    ln = m.offsets[1:] - m.offsets[:-1]
    span = torch.where(ln > 0, torch.div(m.offsets[1:] - 1, BM, rounding_mode="floor")
                       - torch.div(m.offsets[:-1], BM, rounding_mode="floor") + 1, torch.zeros_like(ln))
    mixed = 0
    if is_cat:
        h1 = torch.zeros(tiles, dtype=torch.int64, device=dev).index_add_(0, tile_of, m.order & 1)
        size = torch.bincount(tile_of, minlength=tiles)
        mixed = int(((h1 > 0) & (h1 < size)).sum())
    return dict(T=T, tiles=tiles, last=T - (tiles - 1) * BM if tiles else 0,
                nseg_max=int(nseg.max()) if tiles else 0, tiles_gt16=int((nseg > 16).sum()),
                tiles_gt32=int((nseg > 32).sum()), mixed_tiles=mixed,
                span_max=int(span.max()) if span.numel() else 0)


def _nbad(got: torch.Tensor, ref: torch.Tensor) -> int:
    if got.shape != ref.shape or got.dtype != ref.dtype:
        raise RuntimeError(f"shape/dtype {tuple(got.shape)} {got.dtype} vs {tuple(ref.shape)} {ref.dtype}")
    return int((got.view(torch.int32) != ref.view(torch.int32)).sum().item())


# ---------------------------------------------------------------------------------------- prebuild
def _after_call(device: torch.device, what: str) -> None:
    """After every prebuild x2h_pass call (the lab gates' rule): the tile counters are back at 0 and
    the zero row is still zero."""
    left = int(_RT.counters(device, _RT.stream(device)).abs().sum().item())
    if left:
        raise RuntimeError(f"{what}: tile counters not reset (sum |c| = {left})")
    nz = int(_RT.zero_row(device).view(torch.int16).ne(0).sum().item())
    if nz:
        raise RuntimeError(f"{what}: zero row modified ({nz} words)")


def exact_check(xs: Any, device: torch.device) -> str:
    """PREBUILD 2: x2h_pass on E-X2b's exact synthetic input, bitwise against fp64 (sentinel prefill)."""
    cat, item, num_rows = xs._synthetic(device)
    got = xs.dgrad_segreduce(cat, item, num_rows, _impl="x2h", _prefill=True)
    bad = _nbad(got, xs._reference(cat, item, num_rows))
    if bad:
        raise RuntimeError(f"exact check: {bad} acc words differ from the fp64 reference")
    _after_call(device, "exact check")
    return f"{num_rows}x{WIDTH}"


def cross_check(xs: Any, device: torch.device, grid: int) -> Dict[str, Any]:
    """PREBUILD 3: the Triton passes vs x2h_pass at grids (#CU, 7, #CU) on crosscheck_input(), every
    call on sentinel-filled buffers, counters and zero row checked after every x2h_pass call."""
    cat, item, num_rows, info = crosscheck_input(device, grid, xs.Member)
    inv = xs.check_invariants(cat, item, num_rows)
    if inv is not None:
        raise RuntimeError(f"cross-check input: {inv}")
    ref = xs.dgrad_segreduce(cat, item, num_rows, _impl="triton", _prefill=True)
    grids = (grid, STRESS_GRID, grid)
    for g in grids:
        got = xs.dgrad_segreduce(cat, item, num_rows, _impl="x2h", _grid=g, _prefill=True)
        bad = _nbad(got, ref)
        del got
        if bad:
            raise RuntimeError(f"cross-check vs _ecx_dgrad_pass: {bad} of {ref.numel()} acc words differ "
                               f"(grid {g})")
        _after_call(device, f"cross-check grid {g}")
    return dict(grids=grids, cat=structure_stats(cat, True), item=structure_stats(item, False), input=info,
                num_rows=num_rows)


def prebuild(device: torch.device) -> bool:
    """Before RUN_START (module docstring, PREBUILD): build + load, census, exact check, bitwise
    cross-check against the Triton pass; arms `device` on success. Never raises: a failure latches
    X2H off (one FAILED CLOSED line) and E-X2b keeps its Triton passes."""
    if not requested() or _BROKEN is not None:
        return False
    t0 = time.perf_counter()
    try:
        from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

        role = role_swap()
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"device {device} is not a GPU")
        with torch.cuda.device(device):
            arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
            if arch != _ARCH:
                raise RuntimeError(f"device arch {arch} is not {_ARCH} (x2h_pass.cu is {_ARCH}-only)")
            lib = load(role)
            res = lib.census.get(pass_kernel(role))
            if not census_ok(res):
                raise RuntimeError(f"resource census of x2h_pass<{role},0>: {res} (needs an entry with no "
                                   "VGPR spill and no scratch)")
            idx = _dev_index(device)
            grid = _RT.grid(device)
            _RT.zero_row(device)
            exact = exact_check(xs, device)
            cc = cross_check(xs, device, grid)
    except Exception as exc:
        latch_off("prebuild", exc)
        return False
    _armed.add(idx)
    sc, si, info = cc["cat"], cc["item"], cc["input"]
    cen = (f"VGPR {res.get('VGPRs')} AGPR {res.get('AGPRs')} spill {res.get('VGPRs Spill')} scratch "
           f"{res.get('ScratchSize [bytes/lane]')} LDS {res.get('LDS Size [bytes/block]')} occupancy "
           f"{res.get('Occupancy [waves/SIMD]')}")
    PREBUILD_INFO[idx] = dict(role=role, lib=str(lib.path), cached=lib.cached, build_s=lib.build_s,
                              census=res, grid=grid, exact=exact, seconds=time.perf_counter() - t0, **cc)
    _say(f"{ARMED_PREFIX} pre-run_start prebuild OK ({PASS_NAME} role_swap={role}; {lib.path.parent.name}/"
         f"{_LIB_NAME} {'cached' if lib.cached else f'built in {lib.build_s:.1f}s'}, src sha256 "
         f"{hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:12]}; x2h_pass<{role},0> {cen}; exact check "
         f"{exact} bitwise; cross-check vs _ecx_dgrad_pass bitwise on sentinel-filled buffers: cat T={sc['T']} "
         f"item T={si['T']} (tiles {sc['tiles']}/{si['tiles']}, last {sc['last']}/{si['last']}), "
         f"U={cc['num_rows']}, max {max(sc['nseg_max'], si['nseg_max'])} segments/tile, "
         f"{sc['tiles_gt32'] + si['tiles_gt32']} tiles > 32 segments, {sc['mixed_tiles']} both-half cat tiles, "
         f"rows spanning <= {max(sc['span_max'], si['span_max'])} tiles, "
         f"{info['cat_cancel_pairs'] + info['item_cancel_pairs']} cancelling pairs, grids "
         f"{'/'.join(str(g) for g in cc['grids'])}, counters and zero row checked after every call; "
         f"{time.perf_counter() - t0:.1f}s; device=cuda:{idx})")
    return True


def reset_for_testing() -> None:
    """Test-only: forget the gate, the latch, the arming, the once-lines and the loaded library."""
    global _REQUESTED, _ROLE, _BROKEN, _lib, _RT
    _REQUESTED = _ROLE = _BROKEN = None
    _lib = None
    _RT = _CudaRuntime()
    _armed.clear()
    _ONCE.clear()
    PREBUILD_INFO.clear()
    LAST_LAUNCHES.clear()
