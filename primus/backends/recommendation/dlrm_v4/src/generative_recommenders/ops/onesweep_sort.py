"""Stable LSD radix sort pairs for bounded integer keys (yambda_5b.gin item (49); HIP extension).

onesweep_sort_kernels.hip runs rocPRIM's onesweep device code (the code torch.sort runs) under a leaner host
orchestration: the bit range is the caller's static key bound, the digit width is chosen per caller, the values are
a counting iterator (no iota kernel), and one fill per sort zeroes what rocPRIM zeroes with 1 + 2 fills per pass.
An LSD radix sort is stable and the stable sorting permutation is unique, so for keys in [0, 2^end_bit) the result
equals torch.sort(stable=True), which is what torch.sort runs for every 1-D CUDA input above 4096 elements.

Users (each behind its own default-off switch, see the callers):
  * DLRM_HSTU_FP8_POSENC_TS_ONESWEEP -- ops/triton/triton_position.py: the posenc timestamp-bucket sort (int16 keys
    <= ts table rows - 1, 6-bit digits, 2 passes).
  * DLRM_HSTU_FP8_EC_JOINT_ONESWEEP -- dlrm_v4/train/ec_offsets.py: the grouped EC backward's joint argsort (int32
    keys < 2U read from the two members' int64 ids, 9-bit digits, 2 passes at U <= 2^17) plus both members'
    segment offsets from the sorted keys.

The extension is built by torch.utils.cpp_extension into /tmp/dlrm_hstu_onesweep_sort (sources copied there first,
since the ROCm build hipifies next to its inputs) for the current device's arch only. train_ranker builds it before
RUN_START (DLRM_HSTU_FP8_PREBUILD). A build or load failure returns None from every entry point (the callers keep
torch.sort and each logs its own fail-closed line) and logs one warning here.

Build lock: the ranks of a node serialize the build/load on an flock of <dir>/build.flock, which the kernel releases
when its holder dies (SIGKILL included), with a timeout. torch's own <dir>/lock (FileBaton: an O_EXCL file polled
without a timeout) survives a process killed mid-build and would hang every later build in that container; since
every user of this dir takes the flock first, a torch lock found while holding it is stale and is removed.

One sort is one onesweep batch: rocPRIM's lookback state keeps 30 bits of count (its own host code splits sorts
into batches of < 2^30 items), so inputs of 2^30 or more items return None (the incumbent sorts them).

DLRM_HSTU_FP8_EC_JOINT_ONESWEEP_RB10 (yambda_5b.gin item (102); bound True there, code default off, "0" is the kill switch) builds a
separate variant of the extension (module dlrm_hstu_onesweep_sort_rb10 in /tmp/dlrm_hstu_onesweep_sort_rb10, the same
sources compiled with -DDLRM_ONESWEEP_EC_RB10, for the current device's arch only: PYTORCH_ROCM_ARCH is set to it
during that build, since the training image's gfx942;gfx950 would add gfx942, whose 64 KiB LDS limit rejects the
10-bit kernels) that also has the EC joint sort with 10-bit digits. ec_joint_segments uses it only for 19- and 20-bit
joint keys (2^17 < U <= 2^19: two passes instead of three 9-bit ones); every other call runs the default build's code
((b)'s 6-bit sort16, the 9-bit ec_joint), whose kernels in the variant are instruction-identical to the default
build's. The variant is built only when (c) DLRM_HSTU_FP8_EC_JOINT_ONESWEEP is on as well. With the switch off nothing
here changes: the default build, its directory and its module name are untouched. The variant fails closed to the
default build (build/load failure: one warning, plus the prebuild's "NOT armed ... onesweep_sort_ext_rb10=0" line,
then the default build; a raising wide call or a raw-bit mismatch in its first two calls against the 9-bit build: one
log line, then the 9-bit build for the rest of the process).
"""
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

_SOURCES = ("onesweep_sort_kernels.hip", "onesweep_sort_bind.cpp")
_BUILD_DIR = Path("/tmp/dlrm_hstu_onesweep_sort")
_BUILD_LOCK_TIMEOUT_S = 900.0  # a cold 8-rank build takes ~15 s; a hung builder fails the waiters closed
_MAX_ITEMS = 1 << 30  # exclusive: one onesweep batch (30-bit lookback counts)
_EXT = None
_FAILED: Optional[str] = None
_SWITCHES = ("DLRM_HSTU_FP8_POSENC_TS_ONESWEEP", "DLRM_HSTU_FP8_EC_JOINT_ONESWEEP")
# item (102): the 10-bit-digit EC joint sort lives only in a separately built variant of the extension
_RB10_SWITCH = "DLRM_HSTU_FP8_EC_JOINT_ONESWEEP_RB10"
_RB10_BUILD_DIR = Path("/tmp/dlrm_hstu_onesweep_sort_rb10")
_RB10_DEFINE = "-DDLRM_ONESWEEP_EC_RB10"
_RB10_FAILED: Optional[str] = None  # variant build/load failure (the default build is then used)
_RB10_OFF: bool = False  # latched off for the process (raising call or self-check mismatch)
_RB10_CHECKS: int = 2  # first wide calls of the process also run the 9-bit build and compare raw bits
_RB10_LOGGED: bool = False
_RB10_MIN_BITS, _RB10_MAX_BITS = 19, 20  # 2 passes of <= 10 bits; <= 18 bits already take 2 passes of 9


def _switch_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def _rb10_wanted() -> bool:
    """Item (102) builds its variant only when its own switch AND (c) DLRM_HSTU_FP8_EC_JOINT_ONESWEEP are on (it is
    inert without (c), so the default build serves (b) alone)."""
    return _switch_on(_RB10_SWITCH) and _switch_on("DLRM_HSTU_FP8_EC_JOINT_ONESWEEP")


def wanted() -> bool:
    """Whether any switch that uses the extension is on (prebuild gate)."""
    return any(_switch_on(n) for n in _SWITCHES)


def _arch() -> Optional[str]:
    try:
        name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        return name.split(":")[0] or None
    except Exception:
        return None


def _stage_sources(build_dir: Optional[Path] = None) -> List[str]:
    # None = _BUILD_DIR looked up at call time (callers and tests may rebind it)
    src_dir = (_BUILD_DIR if build_dir is None else build_dir) / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    out = []
    for f in _SOURCES:
        data = (here / f).read_bytes()
        dst = src_dir / f
        if not dst.exists() or dst.read_bytes() != data:
            tmp = src_dir / f"{f}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, dst)  # atomic: concurrent ranks stage identical bytes
        out.append(str(dst))
    return out


def _locked(fn, build_dir: Optional[Path] = None):
    """fn() under an exclusive flock on <dir>/build.flock (see the module docstring), after removing a stale
    torch build lock. Raises TimeoutError after _BUILD_LOCK_TIMEOUT_S. build_dir None = _BUILD_DIR at call time."""
    import fcntl

    if build_dir is None:
        build_dir = _BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(build_dir / "build.flock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > _BUILD_LOCK_TIMEOUT_S:
                    raise TimeoutError(
                        f"{build_dir}/build.flock still held after {_BUILD_LOCK_TIMEOUT_S:.0f} s"
                    ) from None
                time.sleep(0.05)
        stale = build_dir / "lock"
        if stale.exists():
            # no live user of this dir can hold it now: left by a process killed mid-build
            logger.warning("DLRMV4 onesweep sort: removed a stale torch build lock %s", stale)
            stale.unlink(missing_ok=True)
        return fn()
    finally:
        os.close(fd)  # releases the flock


def _cuda_flags(*defines: str) -> List[str]:
    """The extension's HIP flags (both builds; item (102)'s variant adds its define before the arch flag)."""
    arch = _arch()
    cuda_flags = ["-O3", "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__", *defines]
    if arch:
        cuda_flags.append(f"--offload-arch={arch}")
    return cuda_flags


def _with_arch(arch: Optional[str], fn):
    """fn() with PYTORCH_ROCM_ARCH = arch, restored afterwards (also after a raise); arch None: fn() unchanged.
    Used only by item (102)'s variant build (_load_rb10); the default build leaves PYTORCH_ROCM_ARCH alone.

    torch 2.12's cpp_extension adds the archs of $PYTORCH_ROCM_ARCH before extra_cuda_cflags (its arch check runs
    before they are appended), so our --offload-arch alone cannot keep the training image's gfx942;gfx950 out."""
    if not arch:
        return fn()
    old = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = old


def _load_rb10():
    """Item (102): the variant build (same sources + _RB10_DEFINE, own module name and directory), or None after
    one warning (the caller then builds the default extension)."""
    global _RB10_FAILED
    try:
        from torch.utils.cpp_extension import load

        cuda_flags = _cuda_flags(_RB10_DEFINE)
        arch = _arch()

        def _build():
            return load(
                name="dlrm_hstu_onesweep_sort_rb10",
                sources=_stage_sources(_RB10_BUILD_DIR),
                extra_cflags=["-O3", _RB10_DEFINE],
                extra_cuda_cflags=cuda_flags,
                build_directory=str(_RB10_BUILD_DIR),
                verbose=False,
            )

        # Always for the current device only: the training image's gfx942 (64 KiB LDS limit) rejects the 10-bit
        # pass kernels (69,704 B), lab 73242.
        ext = _locked(lambda: _with_arch(arch, _build), _RB10_BUILD_DIR)
        if not hasattr(ext, "ec_joint_w") or int(ext.radix_bits32w()) != 10:
            raise RuntimeError("variant build lacks ec_joint_w / radix_bits32w() == 10")
        return ext
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # head (the command) + tail (the compiler's error lines, which ninja prints last)
        _RB10_FAILED = msg if len(msg) <= 2400 else msg[:400] + " ... " + msg[-2000:]
        logger.warning(
            "DLRMV4 ec joint onesweep rb10 build FAILED -> default onesweep build (9-bit EC joint sort) for the "
            "rest of the run (%s)",
            _RB10_FAILED,
        )
        return None


def extension():
    """The built module, or None (build/load failed: callers keep torch.sort)."""
    global _EXT, _FAILED
    if _EXT is not None or _FAILED is not None:
        return _EXT
    if _rb10_wanted() and _RB10_FAILED is None:
        _EXT = _load_rb10()  # item (102); None -> the default build below
        if _EXT is not None:
            return _EXT
    try:
        from torch.utils.cpp_extension import load

        cuda_flags = _cuda_flags()

        def _build():
            return load(
                name="dlrm_hstu_onesweep_sort",
                sources=_stage_sources(),
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                build_directory=str(_BUILD_DIR),
                verbose=False,
            )

        _EXT = _locked(_build)
    except Exception as e:  # fail closed: torch.sort everywhere
        _FAILED = f"{type(e).__name__}: {e}"[:400]
        logger.warning(
            "DLRMV4 onesweep sort extension unavailable -> torch.sort / incumbent "
            "sort paths for the rest of the run (%s)",
            _FAILED,
        )
    return _EXT


def prebuild() -> bool:
    """Build/load the extension when a switch that uses it is on (train_ranker, before RUN_START)."""
    if not wanted():
        return False
    t0 = time.perf_counter()
    ok = extension() is not None
    if ok and _rb10_wanted():  # item (102)
        if hasattr(_EXT, "ec_joint_w"):
            logger.info("armed OK: prebuild before run_start: onesweep_sort_ext_rb10=1")
        else:
            logger.warning(
                "DLRMV4 NOT armed: prebuild before run_start: onesweep_sort_ext_rb10=0 (default build in use: %s)",
                _RB10_FAILED,
            )
    if ok:
        logger.info(
            "armed OK: prebuild before run_start: onesweep_sort_ext=1 %.1fs", time.perf_counter() - t0
        )
    else:
        logger.warning(
            "DLRMV4 NOT armed: prebuild before run_start: onesweep_sort_ext=0 %.1fs (%s)",
            time.perf_counter() - t0,
            _FAILED,
        )
    return ok


def _on_current_device(t: torch.Tensor) -> bool:
    return t.is_cuda and t.device.index == torch.cuda.current_device()


def sort16_pairs(keys: torch.Tensor, max_key: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(sorted keys, int64 permutation) == torch.sort(keys) for contiguous 1-D int16 keys in [0, max_key] with
    4096 < numel < 2^30, or None (extension unavailable / out of scope)."""
    n = keys.numel()
    if keys.dtype != torch.int16 or keys.dim() != 1 or not keys.is_contiguous() or not _on_current_device(keys):
        return None
    if n <= 4096 or n >= _MAX_ITEMS or max_key < 0:
        return None
    ext = extension()
    if ext is None:
        return None
    end_bit = max(int(max_key).bit_length(), int(ext.radix_bits16()) + 1)
    if end_bit > 15:
        return None
    dev = keys.device
    kout = torch.empty_like(keys)
    vout = torch.empty(n, dtype=torch.int64, device=dev)
    ktmp = torch.empty_like(keys)
    vtmp = torch.empty(n, dtype=torch.int64, device=dev)
    ws = torch.empty(int(ext.ws16(n, end_bit)), dtype=torch.uint8, device=dev)
    ext.sort16(keys, end_bit, kout, vout, ktmp, vtmp, ws)
    return kout, vout


def sort64_pairs(keys: torch.Tensor, max_key: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """zero-fills: (sorted keys, int64 permutation) == torch.sort(keys, stable=True) for contiguous 1-D int64 keys in
    [0, max_key] with 4096 < numel < 2^30, or None (extension not prebuilt / lacks sort64 / out of scope). Only when
    a switch prebuilt the extension before RUN_START (wanted()): a first use never builds it inside the timed region.
    rocPRIM's gfx950 <int64_t, int64_t> onesweep config over bits [0, bit_length(max_key)) and one fill per sort,
    where torch.sort runs all 64 bits (8 passes) with 17 fills."""
    n = keys.numel()
    if keys.dtype != torch.int64 or keys.dim() != 1 or not keys.is_contiguous() or not _on_current_device(keys):
        return None
    if n <= 4096 or n >= _MAX_ITEMS or max_key < 0 or not wanted():
        return None
    ext = extension()
    if ext is None or not hasattr(ext, "sort64"):
        return None
    end_bit = max(int(max_key).bit_length(), int(ext.radix_bits64()) + 1)
    if end_bit > 62:
        return None
    dev = keys.device
    kout = torch.empty_like(keys)
    vout = torch.empty(n, dtype=torch.int64, device=dev)
    ktmp = torch.empty_like(keys)
    vtmp = torch.empty(n, dtype=torch.int64, device=dev)
    ws = torch.empty(int(ext.ws64(n, end_bit)), dtype=torch.uint8, device=dev)
    ext.sort64(keys, end_bit, kout, vout, ktmp, vtmp, ws)
    return kout, vout


def _raw_eq64(x: torch.Tensor, y: torch.Tensor) -> bool:
    return (
        x.dtype == y.dtype == torch.int64
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.data_ptr() % 16 == y.data_ptr() % 16 == 0
        and torch.equal(x, y)
    )


def _rb10_engages(ext, num_rows: int) -> bool:
    """Item (102): the variant build is loaded, not latched off, and the joint keys need 19 or 20 bits."""
    if _RB10_OFF or not hasattr(ext, "ec_joint_w"):
        return False
    return _RB10_MIN_BITS <= int(2 * num_rows - 1).bit_length() <= _RB10_MAX_BITS


def _ec_joint_build(ext, a, b, num_rows: int, wide: bool) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    ta, tb = a.numel(), b.numel()
    n = ta + tb
    rb = int(ext.radix_bits32w()) if wide else int(ext.radix_bits32())
    end_bit = max(int(2 * num_rows - 1).bit_length(), rb + 1)
    sh = ta & 1
    dev = a.device
    kout = torch.empty(n, dtype=torch.int32, device=dev)
    vout = torch.empty(n + sh, dtype=torch.int64, device=dev)
    ktmp = torch.empty(n, dtype=torch.int32, device=dev)
    vtmp = torch.empty(n, dtype=torch.int64, device=dev)
    ws = torch.empty(int(ext.ws32w(n, end_bit) if wide else ext.ws32(n, end_bit)), dtype=torch.uint8, device=dev)
    off_a = torch.empty(num_rows + 1, dtype=torch.int64, device=dev)
    off_b = torch.empty(num_rows + 1, dtype=torch.int64, device=dev)
    gaps = torch.empty(3 * int(ext.gap_capacity(num_rows)), dtype=torch.int64, device=dev)
    joint = ext.ec_joint_w if wide else ext.ec_joint
    joint(a, b, num_rows, end_bit, kout, vout, ktmp, vtmp, ws, off_a, off_b, gaps)
    return [(vout[:ta], off_a), (vout[ta + sh:], off_b)]


def _ec_joint_wide(ext, a, b, num_rows: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Item (102): the 10-bit-digit joint build, or None (latched off: the caller runs the 9-bit build). Its first
    _RB10_CHECKS calls also run the 9-bit build and compare raw bits; a mismatch returns the 9-bit result. A raise of
    that 9-bit reference is not the variant's: it propagates to the caller's fail-closed handler (ec_offsets)."""
    global _RB10_OFF, _RB10_CHECKS, _RB10_LOGGED
    try:
        segs = _ec_joint_build(ext, a, b, num_rows, wide=True)
    except Exception as exc:  # fail closed: the 9-bit build from now on
        _RB10_OFF = True
        logger.warning(
            "DLRMV4 ec joint onesweep rb10 FAILED CLOSED -> 9-bit EC joint sort for the rest of the run (%s: %s)",
            type(exc).__name__, str(exc)[:300],
        )
        return None
    if _RB10_CHECKS > 0:
        ref = _ec_joint_build(ext, a, b, num_rows, wide=False)
        bad = [
            f"member {m} {what}"
            for m, (got_m, ref_m) in enumerate(zip(segs, ref))
            for what, x, y in (("order", got_m[0], ref_m[0]), ("offsets", got_m[1], ref_m[1]))
            if not _raw_eq64(x, y)
        ]
        if bad:
            _RB10_OFF = True
            logger.error(
                "DLRMV4 ec joint onesweep rb10 SELF-CHECK FAILED (T=%d+%d, U=%d): %s differ -- 9-bit result used, "
                "9-bit EC joint sort for the rest of the run", a.numel(), b.numel(), num_rows, ", ".join(bad),
            )
            return ref
        _RB10_CHECKS -= 1
    if not _RB10_LOGGED:
        _RB10_LOGGED = True
        logger.warning(
            "DLRMV4 armed OK: ec joint onesweep rb10 -- EC joint sort in 10-bit digit passes (%d key bits, 2 passes "
            "instead of 3; T=%d+%d, U=%d; first builds == the 9-bit build; kill switch %s=0)",
            int(2 * num_rows - 1).bit_length(), a.numel(), b.numel(), num_rows, _RB10_SWITCH,
        )
    return segs


def ec_joint_segments(
    a: torch.Tensor, b: torch.Tensor, num_rows: int
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """[(order_a, offsets_a), (order_b, offsets_b)] == ec_offsets' incumbent joint build for valid int64 ids
    (0 <= id < num_rows) with fewer than 2^30 ids in total, or None (extension unavailable / out of scope). The
    two orders are 16-byte aligned views of one int64 buffer; the offsets are int64 [num_rows + 1]."""
    if a.dtype != torch.int64 or b.dtype != torch.int64 or a.dim() != 1 or b.dim() != 1:
        return None
    if not (_on_current_device(a) and a.device == b.device):
        return None
    ta, tb = a.numel(), b.numel()
    n = ta + tb
    if ta == 0 or tb == 0 or n >= _MAX_ITEMS or num_rows <= 0 or 2 * num_rows > 2**31 - 1:
        return None
    ext = extension()
    if ext is None:
        return None
    if _rb10_engages(ext, num_rows):  # item (102); only the variant build has ec_joint_w
        segs = _ec_joint_wide(ext, a, b, num_rows)
        if segs is not None:
            return segs
    return _ec_joint_build(ext, a, b, num_rows, wide=False)
