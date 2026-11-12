"""FlyDSL 0.2.4 compile time: memoized per-op source-location capture.

FlyDSL 0.2.4 (ROCm 10.2 UTD image) gives every DSL op it builds a source
location: flydsl/expr/meta.py capture_user_location() walks the Python stack
and, for each user frame it records (up to $FLYDSL_DEBUG_MAX_LOC_DEPTH, default
5, plus the kernel boundary frame), calls inspect.getframeinfo(frame,
context=0). On CPython 3.12 that resolves the frame's column through
inspect._get_code_position(), which iterates code.co_positions() from the
first instruction up to f_lasti on every call: O(size of the function) per
frame. The trainer's FlyDSL kernels are emitted by large builder functions
(hstu_bwd_dkdv.py emit_body is ~3000 lines), so the walk dominates the
compile: the production F2 attention backward (bwd_dkdv, dq-split msplit
build) spent 3.3 s of its 4.8 s FlyDSL call in it (5708 captures, 24594
getframeinfo calls; cProfile, compile-only, NATIVE image). FlyDSL 0.1.1
(rocm/primus:v26.3) resolves one frame per traced op (meta._caller_location,
the op's immediate caller, usually a small helper); 0.2.4 also resolves the
large builder frames above it on nearly every op. That walk is most of 0.2.4's
first-call compile time over 0.1.1's (13 production specs, one process, quiet
node: 18.8 s, 6.1 s with this module).

install() replaces capture_user_location -- in flydsl.expr.meta and in the two
modules that import it by name (flydsl.compiler.kernel_function,
flydsl.compiler.ast_rewriter) -- with the same walk where each frame's
(filename, line, column) comes from a per-code-object table: the
co_positions() tuple is built once per code object and indexed by
f_lasti // 2, exactly the entry inspect._get_code_position() returns, and the
filename is inspect.getsourcefile(frame) or inspect.getfile(frame), as
getframeinfo computes it, once per code object. Same frames, same depth rule,
same boundary frame, same Location objects: the IR, debug locations included,
is byte-identical, and so is every kernel binary. Only host time changes.

Installed only when the three functions it reproduces are the verified ones
(sha256 of their source: FlyDSL 0.2.4's capture_user_location, CPython 3.12's
inspect.getframeinfo and inspect._get_code_position); anything else (FlyDSL
0.1.1 has no flydsl.expr.meta; a different FlyDSL or CPython) keeps the
library's own function. $DLRM_FLYDSL_LOC_MEMO=0 is the kill switch (read once,
at install). STATE says what happened: "on", "off", "absent", "unverified
(...)" or "error (...)".
"""

import hashlib
import inspect
import os
import sys

KILL_SWITCH = "DLRM_FLYDSL_LOC_MEMO"

# sha256 of inspect.getsource() of the functions the memo reproduces, as
# verified on dlrmv4_utd_rocm10.2.sif (FlyDSL 0.2.4, CPython 3.12.3) and
# rocm/primus:v26.9 (FlyDSL 0.3.4.1: its tracing stack holds (code, options)
# tuples and it counts dropped frames, otherwise the same walk).
_VERIFIED = {
    "flydsl.expr.meta.capture_user_location": (
        "28c3c0c004601ea50c944218fc1ffc6433ee7a2718110a8ac8113bfb94089d65",  # 0.2.4
        "8a3492a2bde978847037b4a54e61a4c7bfa89e96e316b10a4f52b695fbed0c64",  # 0.3.4.1
    ),
    "inspect.getframeinfo": (
        "a63d872482280afddf1eea9dc12a712234adde3f4e0b890ac95755ad76181762",
    ),
    "inspect._get_code_position": (
        "da0a300e7dde76a4ce86ac7ec705390c688c995bded486af9ca2dac63a7bba8a",
    ),
}

STATE = "not installed"


def _source_sha(fn) -> str:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    except (OSError, TypeError):
        return "?"


def _make_capture(meta):
    """meta.capture_user_location, statement for statement, with getframeinfo's
    (filename, positions) served from per-code-object tables."""
    _tls = meta._tls
    env = meta.env
    ir = meta.ir
    is_framework_file = meta._is_framework_file
    file_location = meta.file_location
    getframe = sys._getframe
    getsourcefile = inspect.getsourcefile
    getfile = inspect.getfile
    # id(code) -> (code, tuple(code.co_positions()), getframeinfo's filename).
    # Keyed by identity: hashing a code object hashes its contents (~16 us per
    # lookup for these builder functions). The entry holds the code object, so
    # its id cannot be reused while the entry exists; the `is` check guards it.
    codes = {}

    def _where(frame):
        # inspect.getframeinfo(frame, context=0): positions =
        # _get_code_position(code, f_lasti) (None x4 when f_lasti < 0), its
        # lineno replaced by f_lineno when None; capture_user_location then
        # takes line = lineno or f_lineno, col = col_offset or 0.
        code = frame.f_code
        ent = codes.get(id(code))
        if ent is None or ent[0] is not code:
            ent = codes[id(code)] = (code, tuple(code.co_positions()),
                                     getsourcefile(frame) or getfile(frame))
        lasti = frame.f_lasti
        if lasti < 0:
            line, col = None, None
        else:
            p = ent[1][lasti // 2]
            line, col = p[0], p[2]
        if line is None:
            line = frame.f_lineno
        if col is None:
            col = 0
        return ent[2], line, col

    def capture_user_location():
        stack = getattr(_tls, "stack", None)
        boundary = stack[-1] if stack else None
        if isinstance(boundary, tuple):  # FlyDSL >= 0.3: (code, options)
            boundary = boundary[0]
        max_depth = env.debug.max_loc_depth
        ctx = ir.Context.current
        locs = []
        boundary_loc = None

        frame = getframe(1)   # == inspect.currentframe().f_back in the original
        try:
            while frame is not None:
                code = frame.f_code
                is_boundary = boundary is not None and code is boundary
                if not is_framework_file(code.co_filename):
                    keep = len(locs) < max_depth
                    if keep or is_boundary:
                        filename, line, col = _where(frame)
                        floc = file_location(filename, line, col, context=ctx)
                        if keep:
                            locs.append(floc)
                        else:
                            # Always keep the kernel frame: it is the top of the stack.
                            boundary_loc = floc
                if is_boundary:
                    break
                frame = frame.f_back
        finally:
            del frame

        if boundary_loc is not None:
            locs.append(boundary_loc)

        if not locs:
            return ir.Location.unknown()
        callee, callers = locs[0], locs[1:]
        if not callers:
            return callee
        return ir.Location.callsite(callee, callers)

    capture_user_location._dlrm_loc_memo = True
    capture_user_location.__wrapped__ = meta.capture_user_location
    return capture_user_location


def install() -> str:
    """Install the memoized capture once per process (idempotent); returns and
    records STATE. Never raises."""
    global STATE
    if STATE == "on":
        return STATE
    try:
        if os.environ.get(KILL_SWITCH, "1").strip() == "0":
            STATE = "off"
            return STATE
        try:
            from flydsl.expr import meta
        except ImportError:
            STATE = "absent"
            return STATE
        orig = getattr(meta, "capture_user_location", None)
        if orig is None:
            STATE = "absent"
            return STATE
        if getattr(orig, "_dlrm_loc_memo", False):
            STATE = "on"
            return STATE
        got = {
            "flydsl.expr.meta.capture_user_location": _source_sha(orig),
            "inspect.getframeinfo": _source_sha(inspect.getframeinfo),
            "inspect._get_code_position": _source_sha(getattr(inspect, "_get_code_position", None)),
        }
        bad = [k for k, v in got.items() if v not in _VERIFIED[k]]
        if bad:
            STATE = "unverified (" + ",".join(bad) + ")"
            return STATE
        from flydsl.compiler import ast_rewriter, kernel_function

        fast = _make_capture(meta)
        for mod in (meta, kernel_function, ast_rewriter):
            if getattr(mod, "capture_user_location", None) is orig:
                setattr(mod, "capture_user_location", fast)
        STATE = "on"
    except Exception as error:  # never break a FlyDSL import over this
        STATE = f"error ({type(error).__name__}: {error})"
    return STATE
