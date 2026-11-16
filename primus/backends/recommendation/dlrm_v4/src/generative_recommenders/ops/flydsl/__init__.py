"""FlyDSL HSTU attention backward for gfx950 (MI350X).

Deliberately empty of imports: `hstu_bwd_dkdv` pulls in the `flydsl` package at
module scope, which is not a dependency of the trainer. Importing this package
must stay free so a build without flydsl installed can still resolve
`generative_recommenders.ops.flydsl.hstu_flydsl_mha` -- which is how a host that
cannot run the kernel produces a useful error instead of an ImportError.

Hence the shared default lives here rather than in either dispatch site: both
`ops/hstu_attention.py` and `hstu_flydsl_mha.py` need it to interpret an unset
$HSTU_ATTN_BWD, and two copies of the string would eventually disagree.

It is also where FlyDSL's target arch is pinned (FLYDSL_ARCH_PIN below): this
package is the parent of every module here that imports flydsl, so it runs
before any FlyDSL kernel build in the process. The pin reads KFD sysfs with the
stdlib only and may set $FLYDSL_GPU_ARCH in os.environ at import.
"""

# Backward kernel used when $HSTU_ATTN_BWD is unset. In a trainer run the gin
# binding (apply_env_bootstrap.HSTU_ATTN_BWD) sets the variable explicitly, so
# this governs direct library use and tests.
DEFAULT_BWD_BACKEND = "flydsl"

# Forward kernel used when $HSTU_ATTN_FWD is unset. The trainer gin default is
# flydsl; this library default stays triton so tests and non-gin callers do
# not raise off gfx950. DLRM_HSTU_FP8_FWD plus HSTU_ATTN_FWD=flydsl selects
# the FlyDSL e4m3 forward. The backward default remains flydsl.
DEFAULT_FWD_BACKEND = "triton"

# Stdlib only, so the package import stays free (see the module docstring).
import logging as _logging  # noqa: E402
import os as _os  # noqa: E402

_logger = _logging.getLogger(__name__)
_stream_logged = set()

# FlyDSL arch pin (ship13 K10). FlyDSL 0.1.1's get_rocm_arch()
# (flydsl/runtime/device.py) takes $FLYDSL_GPU_ARCH or
# $HSA_OVERRIDE_GFX_VERSION, else the first gfx name from
# `rocm_agent_enumerator -name` (timeout=5, every exception swallowed), else
# "gfx942". It is lru_cached, so the first call fixes the arch for the life of
# the process. Under full-node load the enumerator can fail or time out; that
# process then builds every FlyDSL kernel for gfx942 and aborts (rc 139, no
# Python traceback) at the first gfx950-only MFMA intrinsic. Reproduced and
# fixed bitwise in R15 iteration 45 (proposals/flydsl_gpu_arch_pin.md).
#
# The launcher exports FLYDSL_GPU_ARCH (scripts/launch_slurm.sh worker). For
# every other entry point (eval tools, lab probes, tests) this sets it once at
# import from the KFD topology, the file rocm_agent_enumerator's own sysfs
# path reads, with no HSA init and no subprocess. An explicit
# FLYDSL_GPU_ARCH or HSA_OVERRIDE_GFX_VERSION always wins; with no KFD sysfs,
# or with GPUs of different arches, nothing is set. hstu_flydsl_mha's
# require_flydsl*() refuse to run if FlyDSL still resolved another arch.
# No logging here: logging is configured later; the arming lines live in
# require_flydsl*().
_KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _kfd_gfx_arch(root=_KFD_TOPOLOGY_NODES):
    """gfx name from <root>/*/properties, or None. Never raises.

    Only when exactly one distinct positive gfx_target_version is found
    (CPU nodes read 0): 90500 -> gfx950, 90402 -> gfx942, 90010 -> gfx90a,
    110000 -> gfx1100 (major*10000 + minor*100 + stepping; minor and stepping
    in hex, as rocm_agent_enumerator formats them). A missing root, or an
    unreadable or malformed file, contributes nothing.
    """
    found = set()
    try:
        nodes = _os.listdir(root)
    except (OSError, TypeError, ValueError):
        return None
    for node in nodes:
        try:
            with open(_os.path.join(root, node, "properties"),
                      encoding="ascii") as f:
                lines = f.read().splitlines()
        except (OSError, TypeError, ValueError, UnicodeError):
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 2 or parts[0] != "gfx_target_version":
                continue
            try:
                v = int(parts[1])
            except ValueError:
                continue
            if v > 0:
                found.add(v)
    if len(found) != 1:
        return None
    v = found.pop()
    return "gfx%d%x%x" % (v // 10000, (v // 100) % 100, v % 100)


def _pin_flydsl_arch(root=_KFD_TOPOLOGY_NODES):
    """Set $FLYDSL_GPU_ARCH from KFD sysfs unless an arch is already given.

    Returns (value or None, source); source is "env" (FLYDSL_GPU_ARCH was
    set: by the launcher, the caller, or a parent process), "hsa_override"
    (HSA_OVERRIDE_GFX_VERSION was set), "kfd_sysfs" (set here) or "none"
    (left unset: FlyDSL falls back to rocm_agent_enumerator, then gfx942).
    Empty values count as unset, as in FlyDSL's own check.
    """
    if _os.environ.get("FLYDSL_GPU_ARCH"):
        return _os.environ["FLYDSL_GPU_ARCH"], "env"
    if _os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return _os.environ["HSA_OVERRIDE_GFX_VERSION"], "hsa_override"
    arch = _kfd_gfx_arch(root)
    if arch is None:
        return None, "none"
    _os.environ["FLYDSL_GPU_ARCH"] = arch
    return arch, "kfd_sysfs"


FLYDSL_ARCH_PIN = _pin_flydsl_arch()


def current_stream_launch() -> bool:
    """$DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM (default "1"; gin:
    apply_env_bootstrap.DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM).

    True: the FlyDSL attention fwd/bwd call() wrappers launch on
    torch.cuda.current_stream(q.device) (Arbor iter 6, 5ae3127). False: they
    pass no stream, so launch() uses its default fx.Stream(None) -- HIP's null
    stream, the exact pre-5ae3127 launch arguments. Not an FP8 knob; the
    prefix is only what launch_slurm.sh forwards into the container.

    The host-sync-free step ($DLRM_HSTU_FP8_SYNCFREE=1) requires True: it runs
    FlyDSL while torchrec side streams are current, where the null-stream
    launch raced its producers (20/20 wrong outputs, Arbor iter 6 job 53750).
    Read per call (cheap) because the gin env bootstrap may run after import.
    """
    raw = _os.environ.get("DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM", "1")
    on = raw.strip().lower() not in ("0", "false", "off", "no")
    if not on and _os.environ.get("DLRM_HSTU_FP8_SYNCFREE", "0") == "1":
        raise RuntimeError(
            "DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=%r with "
            "DLRM_HSTU_FP8_SYNCFREE=1: the host-sync-free step needs FlyDSL "
            "launches on the current stream (the legacy null-stream launch "
            "raced side-stream producers, 20/20 wrong outputs). Set "
            "DLRM_HSTU_FP8_SYNCFREE=0 to test the legacy launch stream." % raw
        )
    if on not in _stream_logged:
        _stream_logged.add(on)
        _logger.warning("[flydsl] launch stream: %s",
                        "current" if on else "legacy")
    return on
