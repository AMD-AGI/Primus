"""TSPEC: Triton specialization table (SHIP-13 CANDIDATE, dlstart).

Record the exact specialization of every Triton JIT compile in a non-scored run,
merge the per-rank files into one table, and in later runs compile + load those
specializations BEFORE RUN_START, so the first production call of each kernel is
an in-process (L1) cache hit instead of a JIT compile inside the timed region.
Unlike a binary kernel cache (item (34)) the table holds no binary: every run
compiles the CURRENT kernel source (an edited kernel body is compiled as it is
now, counted as src_changed). A row is skipped, and compiles at first use
exactly as without TSPEC (fail-open), when its kernel is gone, or its parameter
list, a module global it reads, the GPU target or a key-changing Triton setting
changed.

Standard library only at import time. triton is imported inside the functions
that need it, so the merge CLI runs on a login host without torch / triton:

    python3 <tree>/recommendation/generative_recommenders/dlrm_v4/train/triton_spec.py \
        merge OUT.jsonl IN1.jsonl [IN2 ...]
    -> one line "merged rows=N post=P pre=Q kernels=K files=F"
    python3 .../triton_spec.py merge --replace-version 3.9.0 OUT.jsonl TABLE.jsonl REC...
    -> drops TABLE's 3.9.0 rows (a re-recorded stack), keeps every other version's rows
    python3 .../triton_spec.py summary TABLE.jsonl    (per-kernel rows / recorded seconds)

Environment (read in every training process; apply_env_bootstrap binds them from
gin as well, a pre-set env var wins; "unset" / "" / "0" = off):
  TRITON_SPEC_RECORD=<dir>         write <dir>/rec.r<RANK>.p<PID>.jsonl, one row per
                                   JITFunction._do_compile call (phase pre/post RUN_START).
  TRITON_SPEC_REPLAY=<table>       compile + load the table rows before RUN_START.
  TRITON_SPEC_REPLAY_PHASES=post|all   post (default): only rows first compiled after
                                   RUN_START in the record run; all: every row.
  TRITON_SPEC_REPLAY_XVER=1|0      1 (default): also replay rows recorded on another
                                   verified Triton version, translated to this one
                                   (see "One table, rows per Triton version" below);
                                   0: only rows recorded on this Triton version.
  TRITON_SPEC_MONITOR=1            count JITFunction._do_compile calls (L1 misses) after
                                   RUN_START; per-rank summary at RUN_STOP and at exit.
  TRITON_SPEC_BARRIER_S=<s>        phase-A file barrier timeout (default 120 s).
  TRITON_SPEC_BARRIER_DIR=<dir>    barrier directory (default: tempfile.gettempdir(),
                                   i.e. container-local /tmp shared by the node's ranks).

Log lines (every rank, " rank=R" suffix):
  armed OK: [triton-spec] record -> <file> rank=R
  armed OK: [triton-spec] replay table=<path> rows=N resolved=a unresolved=b compiled=c
            primed=d errors=e in X.XXs (A X.XXs, B X.XXs) rank=R
  [triton-spec] NOT armed: <reason> rank=R          (any failure; the run continues as without TSPEC)
  [triton-spec] monitor post-run_start: compiles=N kernels=<name:count,...top 12> rank=R
  plus "[triton-spec] replay detail: ..." / "[triton-spec] monitor detail: ..." diagnostics.

One table, rows per Triton version (ROCm 10.2 port, T11). Every row carries the
triton_version it was recorded on, and row identity (merge dedup) includes it, so
one table file holds the rows of several stacks side by side (v26.3 = Triton 3.6,
UTD ROCm 10.2 = Triton 3.9) and re-recording one stack never touches the other's
rows. A replay takes this version's rows first. With TRITON_SPEC_REPLAY_XVER=1 (the
default) it adds the rows of the other verified versions (_XVER) that this version
has no row for, translated to this version's kernel-cache key: the
specialization part of the L1 key (argument types, %16 / pointer-range classes,
constexpr values) is the same in 3.6 and 3.9 (same native specializer and backend
rules; checked on the CPU with both builds' binders, tools/probe_keys.py of the
T11 work); what differs is (a) the call kwargs JITFunction.run appends to every
key (_RUN_KWARGS: 3.9 adds fpsan_homomorphic_casts, I:696-697 vs N:734-736) and
(b) the attrs _pack_args derives (3.9 gives every top-level constexpr parameter an
empty entry, N:727; 3.6 only a constexpr whose value is a str, I:689). Both are
reproduced exactly (translate_row, translate_attrs), the options are re-derived
from the call kwargs by this build's parse_options (as run() does), so a
translated row compiles the binary the run would compile at first use. A
translated row whose source moved, whose parameters changed or whose kernel is
gone is skipped as any row is. Kernel hashes are compared only between rows of the
same version (hash_same). Rows of a version without verified rules are never
translated (counted as other_triton). Re-recording a stack (record run on that
stack + merge with the current table as one input) gives it its own rows; the
translated rows then only fill what its record runs did not see (e.g. node counts
not recorded on that stack).

Re-recording for a stack (e.g. the UTD ROCm 10.2 image, Triton 3.9), as for the
3.6 rows: the SYNTHETIC dataset (dlrm_v4/train/tspec_synth_data.py --world-size
<ranks>, then tools/recommendation/dlrm_v4/ts_index_tool.py prebuild on its cache; no training-data file
is read), the unmodified trainer on that stack's image with DLRM_DATA_PATH=<synthetic
root>, DL_WINDOW_INDEX=1, TRITON_SPEC_RECORD=<dir>, TRITON_SPEC_REPLAY=0, the two
record arms (START_TS=0 NUM_TRAIN_TS=8 + final eval of window 8; START_TS=9
NUM_TRAIN_TS=2 + final eval of window 11; NUM_TRAIN_BATCHES=100,
NUM_EVAL_BATCHES=40, EVAL_EVERY_N_WINDOWS=<NUM_TRAIN_TS>, EVAL_EVERY_DATA_PCT=0) plus
an eval-heavy arm (START_TS=0 NUM_TRAIN_TS=8, EVAL_EVERY_N_WINDOWS=1,
SKIP_EVAL_EPOCH_PCT=0, NUM_EVAL_BATCHES=20: its eval / train interleave compiles
fallback-path keys the other two never see) at every node count available, then
    triton_spec.py merge --replace-version <that stack's triton.__version__> NEW.jsonl \
        ops/triton/triton_spec_table.jsonl [PROVENANCE.jsonl] <dir>/rec.*.jsonl
(the current table is one input: that version's old rows are replaced, the rows of
the other versions and the provenance of their record campaigns are kept unchanged;
PROVENANCE.jsonl, a row-less {"kind": "table", "triton_versions": [...], ...} line,
documents the new campaign). On cluster A: scripts/cluster_a/tspec_record.sbatch (both steps).
Check with a same-node run, TRITON_SPEC_MONITOR=1: "monitor post-run_start:
compiles=0" on every rank (UTD ROCm 10.2, 2026-11-03: 0 with the re-recorded rows,
2 per rank with the 3.6 rows translated, 67-73 per rank without replay).
Re-record a stack after anything listed in gin item (57) changes on it (kernel
parameters, call-site constexprs / launch options, autotune pins, the Triton build,
a TRITON_* / AMDGCN_* variable that enters the key).

Replay (per rank, after torch.cuda.set_device + process group, before the BLAS
warmup / FP8 first-use warmup (35) / startup preload; skipped on a resume
relaunch, where the MLPerf clock is already running):
  phase A: this local rank's LPT shard (deterministic, weights = recorded compile_s)
           is compiled (JITFunction._do_compile(key, ..., warmup=True), the recipe of
           JITFunction.preload) into the container-local Triton disk cache that all
           local ranks share, and primed (CompiledKernel._init_handles: launcher +
           hsaco load on this GPU; on 3.6 this also builds the per-signature launcher
           .so with gcc into the shared cache);
  barrier: file barrier over the local ranks (timeout; no collective; a timeout only
           costs duplicate compiles);
  phase B: every other selected row: _do_compile (now a disk hit) + _init_handles.
  No kernel is launched and no tensor is allocated; no model / optimizer / RNG
  state is read or written.
  globals guard: recheck_globals() (train_ranker: before the BLAS warmup and just
           before the MLPerf run_start is logged) evicts the replayed entries of a
           kernel, helper or constexpr callable whose module globals changed after
           the replay froze them, so it compiles at first use as without TSPEC.

Triton internals relied on. Verified in the THREE builds named below (identical
semantics except where noted; line numbers differ):
  "I" = 3.6.0, the image's /opt/venv copy that production and the gate container
        import (production log /scratch/ship12_20261009/ladder_early_1n/t6a.log
        warns from .../triton/backends/amd/compiler.py:86 = the 3.6.0 kpack line;
        the image's python sources are byte-identical to GitHub v3.6.0);
  "H" = 3.8.0, /scratch/opt21/turbo/packages_380_024/triton (host copy).
  "N" = 3.9.0, the UTD ROCm 10.2 image's /opt/venv copy (dlrmv4_utd_rocm10.2.sif,
        primus_rocm10.2_20261016). Differences that matter here: run() adds a third
        key kwarg fpsan_homomorphic_casts (N:736); _pack_args gives constexpr
        parameters an attrs entry (N:727); HIPOptions gains fpsan_homomorphic_casts /
        llvm_fn_attrs / wgp_cu_mode and defaults enable_fp_fusion to the knob
        (N:261-383); CompiledKernel.__del__ unloads its module (N:439-450).
  runtime/jit.py
    JITFunction._do_compile(self,key,signature,device,constexprs,options,attrs,warmup)
        H:877-906  I:826-853  N:887-921; stores kernel_cache[key]  H:903  I:850  N:918
    JITFunction.run L1 path: kwargs debug/instrumentation_mode H:727-728 I:696-697
        (+ fpsan_homomorphic_casts N:734-736); device_caches[device] -> (kernel_cache,
        kernel_key_cache, target, backend, binder) H:738 I:707 N:746;
        key = compute_cache_key(...) H:749 I:712 N:757; hit check H:750 I:713 N:758;
        miss -> _pack_args + _do_compile H:753-757 I:716-720 N:761-767;
        used_global_vals check (RuntimeError) H:762-766 I:725-729 N:770-774
    compute_cache_key: key = str(specialization with JITCallables -> .cache_key) +
        str(non-parameter call kwargs) H:592-613 (string H:611) I:563-584 (I:582)
        N:598-620 (N:618)
    _pack_args (options = backend.parse_options(kwargs); signature; constexprs by
        path; attrs) H:702-724 I:671-693 N:709-731 (attrvals: I:689 x[1]; N:727
        '' for a constexpr parameter)
    create_binder (sets self.compile / self.ASTSource used by _do_compile)
        H:689-700 I:658-669 N:696-707; device_caches = defaultdict(create_binder)
        H:808 I:771 N:818
    preload (decode + backend.parse_options + _do_compile(warmup=True)) H:824-875
        I:787-824 N:834-885
    JITCallable.cache_key (placeholder H:523 I:502 N:530; hash = source + starting
        line H:528 I:507; used_global_vals captured once H:529 I:508 N:536) H:515-538
        I:494-514 N:522-545
    KernelParam.is_constexpr H:335 I:331 N:336; annotation_type H:324 I:320 N:325
  runtime/autotuner.py  Autotuner.fn H:92 I:87 N:107; Heuristics.fn H:476 I:452 N:496;
    Autotuner.run -> fn.run(*args, **kwargs, **config.all_kwargs()) H:276 I:252 N:291
    (Config.all_kwargs identical in I and N)
  compiler/compiler.py  compile() H:226-363 I:226-360 N:226-363 (disk key H:246-250
    I:246-247 N:246-250; listener on a cache hit H:272-278 I:269-275 N:271-278, on a
    miss H:360-361 I:357-358 N:359-361); CompiledKernel.__init__ reads every IR file
    H:424-429 I:421-426 N:424-429; _init_handles H:448-485 I:436-471 N:452-490
    (launcher created BEFORE the load H:464 I:452 N:468, load_binary H:479-480
    I:465-466 N:483-484); run property H:487-491 I:473-477 N:492-496
  backends/amd/compiler.py  HIPOptions H:77-135 I:31-99 N:261-332; parse_options
    H:154-182 I:118-141 N:356-383 (dataclass-field filter H:181 I:140 N:381);
    get_tensor_specialization ('S' pointer-range class) H:236-240 I:182-186 N:436-440;
    parse_attr H:229-233 I:175-179 N:429-433
  backends/amd/driver.py  launcher: 3.6 builds one gcc .so per signature
    (HIPLauncher I:800-812, compile_module_from_src I:809); 3.8 / 3.9 use the generic
    hip_utils launch (HIPLauncher H:315-341 N:290-347)
  runtime/cache.py  FileCacheManager.put is tmp + os.replace (atomic, no lock)
    H:103-127 I:98-122 N:103-127; triton_key H:284 I:272 N:284
  knobs.py  compilation.listener H:372 I:371 N:373; compilation.instrumentation_mode and
    runtime.debug are read ONCE at import H:371/H:474 I:370/I:465 N:371/N:475;
    compilation.fpsan_homomorphic_casts N:372 (3.9 only);
    runtime.kernel_load_start_hook is a HookChain H:479 I:470 N:480 (add H:423 I:414
    N:424); language.default_fp_fusion H:497 I:485 N:498 (3.9 default False);
    runtime.override_arch H:475 I:466 N:476; amd.use_buffer_ops H:517 I:505 N:523
  runtime/_async_compile.py  active_mode / FutureKernel H:6/H:9 I:6/I:9 N:6/N:9
"""

import ast
import atexit
import gc
import hashlib
import importlib
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger: logging.Logger = logging.getLogger(__name__)

TAG = "[triton-spec]"
SCHEMA = 2
# Triton major.minor whose internals (listed in the module docstring) were checked.
_VERIFIED_TRITON = ("3.6", "3.8", "3.9")
# The call kwargs JITFunction.run adds to EVERY call, in this order, after the
# launch / autotune-config kwargs; they end the dict suffix of every L1 key
# (jit.py H:727-728 I:696-697 N:734-736). check_triton() re-reads them from the
# running build's run() source and refuses a build that disagrees.
_RUN_KWARGS = {
    "3.6": ("debug", "instrumentation_mode"),
    "3.8": ("debug", "instrumentation_mode"),
    "3.9": ("debug", "instrumentation_mode", "fpsan_homomorphic_casts"),
}
# _pack_args: True = every top-level constexpr parameter gets an attrs entry
# parse_attr("") (N:727); False = only a constexpr whose value is a str gets
# parse_attr(value) (I:689). attrs enter ASTSource.hash (the disk key and the
# kernel hash), not the L1 key.
_CONSTEXPR_ATTRS = {"3.6": False, "3.9": True}
# Versions whose rows replay may translate into each other (both rules above
# verified in both builds; 3.8 rows are replayed on 3.8 only).
_XVER = ("3.6", "3.9")
# Launch options that exist only in some versions' HIPOptions and that the call
# sites pass only when the running build has them (opsel-fix: llvm_fn_attrs =
# "target-features=-packed-fp32-ops" on gfx950 + Triton 3.9 for kernels whose
# 3.9 build has the gfx950-invalid packed-FP32 op_sel form, see
# ops/triton/gfx950_pk_opsel.py). translate_row drops such a kwarg when the
# destination version lacks it (the call site does not pass it there). The other
# direction cannot be derived from a row (only the call site knows), so
# select_rows_xver counts a translated row as covered when this version has a
# row of the same kernel whose key differs only by such kwargs.
_VERSION_ONLY_KWARGS = {"3.6": (), "3.9": ("llvm_fn_attrs",)}
_DO_COMPILE_PARAMS = (
    "self", "key", "signature", "device", "constexprs", "options", "attrs", "warmup",
)
# Modules the replay may import to resolve a row (anything else must already be
# imported). Importing a kernel module is what the run does anyway; none of the
# tree's Triton kernel modules imports FlyDSL at module level.
_IMPORT_PREFIXES = ("generative_recommenders.", "fbgemm_gpu.")
# Per-run fields dropped from the merged table ("world_size" is folded into the
# sorted union "world_sizes").
_VOLATILE = ("rank", "local_rank", "pid", "t_rel", "disk_hit", "stage_us", "via", "host",
             "world_size")
# Settings that enter EVERY L1 key: run() adds debug / instrumentation_mode (and on
# 3.9 fpsan_homomorphic_casts) to the call kwargs (jit.py H:727-728 I:696-697
# N:734-736) and use_buffer_ops decides the 'S' pointer class of every tensor
# argument (amd/compiler.py H:236-240 I:182-186 N:436-440). A table recorded with
# other values can never be hit: replay is not armed. (A knob a record run's build
# did not have is None in its header and not compared.)
_KEY_KNOBS = ("debug", "instrumentation_mode", "use_buffer_ops", "fpsan_homomorphic_casts")
_OFF = ("", "0", "off", "false", "no", "none")
_ON = ("1", "on", "true", "yes")
_MISSING = object()


# =============================================================================
# Environment
# =============================================================================
def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def env_record_dir() -> Optional[str]:
    """TRITON_SPEC_RECORD: a directory, or None when off (unset/""/"0"/off)."""
    v = _env("TRITON_SPEC_RECORD")
    return None if v.lower() in _OFF else v


def env_replay_table() -> Optional[str]:
    """TRITON_SPEC_REPLAY: a table path, or None when off (unset/""/"0"/off)."""
    v = _env("TRITON_SPEC_REPLAY")
    return None if v.lower() in _OFF else v


def env_replay_phases() -> Tuple[str, bool]:
    """TRITON_SPEC_REPLAY_PHASES -> ("post"|"all", value_was_valid)."""
    v = _env("TRITON_SPEC_REPLAY_PHASES").lower()
    if v == "all":
        return "all", True
    return "post", v in ("", "post")


def env_monitor() -> bool:
    """TRITON_SPEC_MONITOR: on for 1/true/on/yes (any case)."""
    return _env("TRITON_SPEC_MONITOR").lower() in _ON


def env_replay_xver() -> bool:
    """TRITON_SPEC_REPLAY_XVER: cross-version rows on unless 0/off/false/no/none."""
    v = _env("TRITON_SPEC_REPLAY_XVER").lower()
    return not v or v not in _OFF


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _resolve_rank(rank: Optional[int]) -> int:
    return int(rank) if rank is not None else _int_env("RANK", 0)


def _resolve_local_rank(local_rank: Optional[int]) -> int:
    return int(local_rank) if local_rank is not None else _int_env("LOCAL_RANK", 0)


def _resolve_local_world(local_world: Optional[int]) -> int:
    if local_world is not None:
        return max(1, int(local_world))
    return max(1, _int_env("LOCAL_WORLD_SIZE", 0) or _int_env("GPUS_PER_NODE", 0) or 1)


# =============================================================================
# Codec (stdlib only). A decoded constexpr must reproduce str(value) (it enters
# ASTSource.hash, compiler.py H:71-76) and the same object for code generation.
# Exact types only: bool / int / float / str / None / tuple, tl.constexpr,
# tl.dtype (not pointer/block types), torch.dtype, JIT callables by reference.
# =============================================================================
class Unsupported(TypeError):
    """A constexpr value the codec cannot reproduce exactly (row is skipped)."""


def _is_jit_callable(v: Any) -> bool:
    try:
        from triton.runtime.jit import JITCallable
    except Exception:  # no triton (login host / stub tests): duck-type it
        return hasattr(v, "cache_key") and hasattr(v, "fn")
    return isinstance(v, JITCallable)


def encode_value(v: Any) -> Dict[str, Any]:
    t = type(v)
    if v is None:
        return {"t": "none"}
    if t is bool:
        return {"t": "bool", "v": v}
    if t is int:
        return {"t": "int", "v": v}
    if t is float:
        return {"t": "float", "v": repr(v)}  # repr round-trips (inf/nan too)
    if t is str:
        return {"t": "str", "v": v}
    if t is tuple:
        return {"t": "tuple", "v": [encode_value(x) for x in v]}
    mod = getattr(t, "__module__", "") or ""
    name = getattr(t, "__name__", "")
    if name == "constexpr" and mod.startswith("triton.language"):
        return {"t": "tl.constexpr", "v": encode_value(v.value)}
    if name == "dtype" and mod.startswith("triton.language"):
        return {"t": "tl.dtype", "v": str(v.name)}
    if name == "dtype" and mod == "torch":
        return {"t": "torch.dtype", "v": str(v).split(".", 1)[-1]}
    fn = getattr(v, "fn", None)
    if fn is not None and hasattr(fn, "__qualname__") and _is_jit_callable(v):
        # "ck": the callable's cache_key, which is what the L1 key embeds
        # (compute_cache_key replace_callables, jit.py H:599-609 I:570-580).
        return {"t": "jit", "module": fn.__module__, "qualname": fn.__qualname__,
                "ck": str(v.cache_key)}
    raise Unsupported(f"constexpr of type {mod}.{name}")


def decode_value(d: Any, resolve_jit: Optional[Callable[[str, str], Any]] = None,
                 tl: Any = None, torch: Any = None,
                 on_jit: Optional[Callable[[Dict[str, Any], Any], None]] = None) -> Any:
    """Strict inverse of encode_value (a malformed value raises ValueError)."""
    if not isinstance(d, dict):
        raise ValueError(f"codec value is {type(d).__name__}, not an object")
    t = d.get("t")
    v = d.get("v")
    if t == "none":
        return None
    if t == "bool" and type(v) is bool:
        return v
    if t == "int" and type(v) is int:
        return v
    if t == "float" and type(v) is str:
        return float(v)
    if t == "str" and type(v) is str:
        return v
    if t == "tuple" and type(v) is list:
        return tuple(decode_value(x, resolve_jit, tl, torch, on_jit) for x in v)
    if t == "tl.constexpr":
        tl = tl if tl is not None else importlib.import_module("triton.language")
        return tl.constexpr(decode_value(v, resolve_jit, tl, torch, on_jit))
    if t == "tl.dtype" and type(v) is str:
        tl = tl if tl is not None else importlib.import_module("triton.language")
        return tl.dtype(v)  # the JITFunction.preload recipe (jit.py H:842-843 I:799)
    if t == "torch.dtype" and type(v) is str:
        torch = torch if torch is not None else importlib.import_module("torch")
        obj = getattr(torch, v)
        if type(obj).__name__ != "dtype":
            raise ValueError(f"torch.{v} is not a dtype")
        return obj
    if t == "jit" and isinstance(d.get("module"), str) and isinstance(d.get("qualname"), str):
        if resolve_jit is None:
            raise ValueError("JIT-callable constexpr needs a resolver")
        obj = resolve_jit(d["module"], d["qualname"])
        if on_jit is not None:
            on_jit(d, obj)
        return obj
    raise ValueError(f"bad codec value {d!r:.80}")


def _jsonable(v: Any) -> Any:
    """Tuples -> lists, recursively (JSON)."""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def _tuplify(v: Any) -> Any:
    """Lists -> tuples, recursively (jit.py convert_to_tuple_if_list H:616 I:587, deep)."""
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuplify(x) for k, x in v.items()}
    return v


def encode_compile_args(key: str, signature: Dict[str, Any], constexprs: Dict[Any, Any],
                        attrs: Dict[Any, Any], options_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The JITFunction._do_compile inputs as JSON. Order-carrying dicts become
    lists of pairs (json sort_keys would reorder dict keys; str(attrs) enters
    ASTSource.hash)."""
    for p in list(constexprs) + list(attrs):
        if not isinstance(p, tuple):
            raise Unsupported(f"non-tuple constexpr/attr path {p!r}")
    return {
        "key": key,
        "signature": [[str(k), _jsonable(v)] for k, v in signature.items()],
        "constexprs": [[list(p), encode_value(v)] for p, v in constexprs.items()],
        "attrs": [[list(p), _jsonable(v)] for p, v in attrs.items()],
        "options": {str(k): _jsonable(v) for k, v in options_dict.items()},
    }


def split_key_options(key: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """L1 key = str(specialization list) + str(call-kwargs dict) (jit.py H:611
    I:582). Returns (specialization part, kwargs dict) when the suffix is a dict
    literal that round-trips exactly through str(), else None."""
    if not isinstance(key, str) or not key.endswith("}"):
        return None
    end = len(key)
    while True:
        i = key.rfind("]{", 0, end)
        if i < 0:
            return None
        suffix = key[i + 1:]
        try:
            d = ast.literal_eval(suffix)
        except Exception:
            d = None
        if isinstance(d, dict) and str(d) == suffix:
            return key[:i + 1], d
        end = i


# =============================================================================
# Merge / table / shard / barrier (stdlib only)
# =============================================================================
def read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Objects of a JSON-lines file and the number of torn / non-object lines."""
    objs: List[Dict[str, Any]] = []
    torn = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                torn += 1
                continue
            if isinstance(o, dict):
                objs.append(o)
            else:
                torn += 1
    return objs, torn


def _row_ok(o: Dict[str, Any]) -> bool:
    return all(isinstance(o.get(k), str) and o.get(k) for k in ("module", "qualname", "key"))


def row_id(o: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (str(o.get("triton_version") or ""), str(o.get("module")), str(o.get("qualname")),
            str(o.get("key")))


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and x >= 0 else None  # drop nan / negative


def _ws_of(o: Dict[str, Any]) -> set:
    """World sizes a row was recorded at (record rows: "world_size"; tables:
    "world_sizes"). Diagnostics only (replay detail "other_ws")."""
    ws = {w for w in (o.get("world_sizes") or []) if type(w) is int and w > 0}
    w = o.get("world_size")
    if type(w) is int and w > 0:
        ws.add(w)
    return ws


def _fold_rows(sources: Iterable[Tuple[Optional[str], Dict[str, Any]]]) -> Tuple[
        Dict[Tuple[str, str, str, str], Dict[str, Any]], Dict[str, int]]:
    """Dedup compile rows by (triton_version, module, qualname, key). Content:
    last occurrence wins; phase = post if any occurrence is post; n_seen summed;
    compile_s = max over cold (disk-miss) compiles, else max over all."""
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    st = {"in": 0, "unsupported": 0, "replay": 0, "bad": 0, "dup": 0}
    for file_version, o in sources:
        if o.get("kind") != "compile":
            continue
        st["in"] += 1
        if o.get("unsupported"):
            st["unsupported"] += 1
            continue
        if o.get("via") == "replay":  # compiled by a replay: carries no phase information
            st["replay"] += 1
            continue
        if not _row_ok(o):
            st["bad"] += 1
            continue
        o = dict(o)
        if not o.get("triton_version") and file_version:
            o["triton_version"] = file_version
        s = _num(o.get("compile_s"))
        cold = (o.get("disk_hit") is False) or bool(o.get("compile_cold"))
        rid = row_id(o)
        prev = out.get(rid)
        n_seen = o.get("n_seen")
        # a table row keeps its count (0 = class-completed, never seen in a record run)
        n_seen = n_seen if type(n_seen) is int and n_seen >= 0 else 1
        if prev is None:
            o["n_seen"] = n_seen
            o["_cold_s"] = s if cold else None
            o["_any_s"] = s
            o["_warm_only"] = bool(o.get("warmup"))
            o["_ws"] = _ws_of(o)
            out[rid] = o
            continue
        st["dup"] += 1
        merged = dict(o)
        merged["n_seen"] = int(prev["n_seen"]) + n_seen
        merged["phase"] = "post" if "post" in (prev.get("phase"), o.get("phase")) else "pre"
        merged["_warm_only"] = bool(prev.get("_warm_only")) and bool(o.get("warmup"))
        merged["_ws"] = set(prev.get("_ws") or ()) | _ws_of(o)
        cs = [x for x in (prev.get("_cold_s"), s if cold else None) if x is not None]
        merged["_cold_s"] = max(cs) if cs else None
        al = [x for x in (prev.get("_any_s"), s) if x is not None]
        merged["_any_s"] = max(al) if al else None
        out[rid] = merged
    for o in out.values():
        cold_s, any_s = o.pop("_cold_s", None), o.pop("_any_s", None)
        o["compile_s"] = round(cold_s if cold_s is not None else (any_s or 0.0), 4)
        o["compile_cold"] = cold_s is not None
        o["warmup"] = bool(o.pop("_warm_only", False))
        ws = o.pop("_ws", None)
        if ws:
            o["world_sizes"] = sorted(ws)
        else:
            o.pop("world_sizes", None)
        if o.get("phase") not in ("pre", "post"):
            o["phase"] = "pre"
        for k in _VOLATILE:
            o.pop(k, None)
    return out, st


def _iter_sources(paths: Sequence[str], st: Dict[str, Any], headers: List[Dict[str, Any]],
                  replace: Sequence[str] = ()):
    """(file version, row) of every input; rows of a version in `replace` that come
    from an input TABLE are dropped (a re-recorded stack's record files replace its
    old rows; other versions' rows are kept)."""
    for p in paths:
        try:
            objs, torn = read_jsonl(p)
        except OSError as ex:
            st["unreadable"].append(f"{p}: {ex}")
            continue
        st["files"] += 1
        st["torn"] += torn
        file_version = None
        is_table = any(o.get("kind") == "table" for o in objs)
        # a table header without rows only carries provenance for rows recorded now
        carrier = is_table and not any(o.get("kind") == "compile" for o in objs)
        for o in objs:
            if o.get("kind") in ("header", "table"):
                if carrier and o.get("kind") == "table":
                    o["_carrier"] = True
                headers.append(o)
                if o.get("kind") == "header" and o.get("triton_version"):
                    file_version = str(o["triton_version"])
                continue
            if is_table and replace and str(o.get("triton_version") or file_version) in replace:
                st["replaced"] = st.get("replaced", 0) + 1
                continue
            yield file_version, o


def _variants(headers: Sequence[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    """Distinct `field` dicts (env / knobs) over record headers and earlier tables
    (a table carries all of them in "<field>_seen"), sorted."""
    seen: Dict[str, Dict[str, Any]] = {}
    for h in headers:
        vs = h.get(field + "_seen")
        if not isinstance(vs, list):
            vs = [h.get(field)] if field in h else []
        for v in vs:
            if isinstance(v, dict):
                seen.setdefault(json.dumps(v, sort_keys=True, default=str), v)
    return [seen[k] for k in sorted(seen)]


# Header fields merge() writes itself; any other field of an input TABLE header
# (how its rows were produced: source, recorded_runs, synthetic_specs,
# class_completion, ...) is kept under "provenance", so merging a new stack's record
# files into an existing table does not drop how the old rows were made.
_HDR_FIELDS = ("kind", "schema", "n", "n_post", "n_pre", "kernels", "files", "triton_versions",
               "triton_key_sha", "env", "env_variants", "env_seen", "knobs", "knob_variants",
               "knobs_seen", "targets", "world_sizes", "note", "provenance", "_carrier")


def _provenance(headers: Sequence[Dict[str, Any]], replace: Sequence[str] = ()) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for h in headers:
        if h.get("kind") != "table":
            continue
        items = [p for p in (h.get("provenance") or []) if isinstance(p, dict)]
        extra = {k: v for k, v in h.items() if k not in _HDR_FIELDS}
        if extra:
            extra["triton_versions"] = h.get("triton_versions")
            items.append(extra)
        for p in items:
            vs = p.get("triton_versions")
            if (replace and not h.get("_carrier") and isinstance(vs, list) and vs
                    and all(str(v) in replace for v in vs)):
                continue  # the campaign whose rows were replaced
            out.setdefault(json.dumps(p, sort_keys=True, default=str), p)
    return [out[k] for k in sorted(out)]


def merge(out_path: str, in_paths: Sequence[str],
          replace_versions: Sequence[str] = ()) -> Dict[str, Any]:
    """Merge record files and/or earlier tables into one deterministic table.
    replace_versions: Triton versions whose rows in the input TABLES are dropped
    (re-recording a stack: its new record files replace its old rows). OUT is left
    untouched (st["written"] False) when no input was readable or no row survived,
    so a mistyped glob cannot wipe an existing table."""
    st: Dict[str, Any] = {"files": 0, "torn": 0, "unreadable": [], "written": False,
                          "replaced": 0}
    headers: List[Dict[str, Any]] = []
    replace = tuple(str(v) for v in replace_versions)
    rows_by, fst = _fold_rows(_iter_sources(in_paths, st, headers, replace))
    st.update(fst)
    rows = [rows_by[k] for k in sorted(rows_by)]
    n_post = sum(1 for r in rows if r.get("phase") == "post")
    kernels = len({(r["module"], r["qualname"]) for r in rows})
    envs = _variants(headers, "env")
    knobs = _variants(headers, "knobs")
    st.update({"rows": len(rows), "post": n_post, "pre": len(rows) - n_post, "kernels": kernels,
               "env_variants": len(envs), "knob_variants": len(knobs),
               "cold": sum(1 for r in rows if r.get("compile_cold"))})
    if st["files"] == 0 or not rows:
        return st
    key_shas = set()
    for h in headers:
        v = h.get("triton_key_sha")
        for s in (v if isinstance(v, list) else [v]):
            if isinstance(s, str) and s:
                key_shas.add(s)
    h0 = next((h for h in headers if h.get("kind") == "header"), headers[0] if headers else {})
    hdr = {
        "kind": "table", "schema": SCHEMA, "n": len(rows), "n_post": n_post,
        "n_pre": len(rows) - n_post, "kernels": kernels, "files": st["files"],
        "triton_versions": sorted({str(r.get("triton_version") or "") for r in rows}),
        "triton_key_sha": sorted(key_shas),
        "env": h0.get("env", {}), "env_variants": len(envs), "env_seen": envs,
        "knobs": h0.get("knobs", {}), "knob_variants": len(knobs), "knobs_seen": knobs,
        "targets": sorted({json.dumps(r["target"], sort_keys=True) for r in rows
                           if isinstance(r.get("target"), dict)}),
        "world_sizes": sorted({w for r in rows for w in _ws_of(r)}),
        "note": "kernel names, dtypes, integer-specialization classes, constexpr values and "
                "launch options only; no sample values and no binaries. Regenerate with "
                "TRITON_SPEC_RECORD=<dir> + triton_spec.py merge.",
    }
    prov = _provenance(headers, replace)
    if prov:
        hdr["provenance"] = prov
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(hdr, sort_keys=True, separators=(",", ":")) + "\n")
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, out_path)
    st["written"] = True
    return st


def load_table(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    """(header, deduplicated rows, torn lines). Accepts a merged table or a raw
    record file."""
    st: Dict[str, Any] = {"files": 0, "torn": 0, "unreadable": []}
    headers: List[Dict[str, Any]] = []
    rows_by, _ = _fold_rows(_iter_sources([path], st, headers))
    if st["unreadable"]:
        raise OSError(st["unreadable"][0])
    hdr = next((h for h in headers if h.get("kind") == "table"), headers[0] if headers else {})
    return hdr, [rows_by[k] for k in sorted(rows_by)], int(st["torn"])


def select_rows(rows: Sequence[Dict[str, Any]], phases: str, version: Optional[str]) -> Tuple[
        List[Dict[str, Any]], int]:
    """Rows of this Triton version (and phase "post" unless phases == "all");
    returns (selected, rows skipped for another Triton version)."""
    sel, other = [], 0
    for r in rows:
        if version is not None and str(r.get("triton_version") or "") != version:
            other += 1
            continue
        if phases != "all" and r.get("phase") != "post":
            continue
        sel.append(r)
    return sel, other


def _mm(version: Any) -> str:
    """'3.9.0' -> '3.9' (major.minor; '' for a missing version)."""
    return ".".join(str(version or "").split(".")[:2])


def translate_row(row: Dict[str, Any], version: str, knobs: Dict[str, Any]) -> Dict[str, Any]:
    """A row recorded on another verified Triton version (_XVER), keyed the way
    THIS version's run() keys the same call: the specialization part of the L1
    key is kept, the call kwargs (key suffix and call_opts) get this version's
    run() kwargs (_RUN_KWARGS: a kwarg only the source version adds is dropped,
    one only this version adds is appended with the current knob value, exactly
    where run() appends it). The recorded HIPOptions and kernel hash belong to the
    other build and are dropped (options are re-derived from the call kwargs, as
    run() does). attrs are translated in _Replay.prepare (translate_attrs).
    Raises ValueError when the row cannot be translated exactly."""
    src, dst = _mm(row.get("triton_version")), _mm(version)
    if src not in _XVER or dst not in _XVER:
        raise ValueError(f"no verified translation {src or '?'} -> {dst or '?'}")
    sp = split_key_options(row.get("key"))
    if sp is None:
        raise ValueError("key suffix is not a call-kwargs dict")
    spec, kw = sp
    co = row.get("call_opts")
    if not isinstance(co, list) or not co or [p[0] for p in co if isinstance(p, list)] != list(kw):
        raise ValueError("call_opts missing or not the key's call kwargs")
    out = dict(kw)
    for name in _RUN_KWARGS[src]:
        if name not in _RUN_KWARGS[dst] and name in out:
            v = out.pop(name)
            if v not in (False, "", None):  # recorded with a mode this build cannot key
                raise ValueError(f"{name}={v!r} has no equivalent on Triton {dst}")
    for name in _VERSION_ONLY_KWARGS.get(src, ()):
        if name not in _VERSION_ONLY_KWARGS.get(dst, ()):
            out.pop(name, None)  # the call site passes it only where the build has it
    for name in _RUN_KWARGS[dst]:
        if name not in out:
            if knobs.get(name) is None:
                raise ValueError(f"current value of knob {name} unknown")
            out[name] = knobs[name]
    key = spec + str(out)
    if split_key_options(key) != (spec, out):
        raise ValueError("translated key does not round-trip")
    t = dict(row)
    t["key"] = key
    t["call_opts"] = [[str(k), _jsonable(v)] for k, v in out.items()]
    t["xver_from"] = str(row.get("triton_version"))
    t["triton_version"] = version
    for k in ("kernel_hash", "options"):
        t.pop(k, None)
    return t


def translate_attrs(signature: Sequence[Sequence[Any]], attrs: Sequence[Tuple[Tuple[int, ...], Any]],
                    constexprs: Dict[Tuple[int, ...], Any], src: str, dst: str,
                    parse_attr: Callable[[str], Any]) -> List[Tuple[Tuple[int, ...], Any]]:
    """attrs (path, value) pairs as `dst`'s _pack_args derives them from the call
    `src`'s _pack_args derived `attrs` from: only top-level constexpr parameters
    differ (_CONSTEXPR_ATTRS); every other entry is kept, in _pack_args' order
    (find_paths_if: parameter order, depth first)."""
    s_rule, d_rule = _CONSTEXPR_ATTRS.get(_mm(src)), _CONSTEXPR_ATTRS.get(_mm(dst))
    if s_rule is None or d_rule is None:
        raise ValueError(f"no attrs rule for {src} -> {dst}")
    if s_rule == d_rule:
        return list(attrs)
    const_top = {i for i, p in enumerate(signature) if p[1] == "constexpr"}
    by_param: Dict[int, List[Tuple[Tuple[int, ...], Any]]] = {}
    for path, val in attrs:
        by_param.setdefault(path[0], []).append((path, val))
    out: List[Tuple[Tuple[int, ...], Any]] = []
    for i in range(len(signature)):
        if i not in const_top:
            out.extend(by_param.get(i, []))
        elif d_rule:
            out.append(((i,), parse_attr("")))
        else:
            v = constexprs.get((i,), _MISSING)
            if isinstance(v, str):
                out.append(((i,), parse_attr(v)))
    extra = [p for p in by_param if p >= len(signature)]
    if extra:
        raise ValueError(f"attrs for parameters {extra} beyond the signature")
    return out


def select_rows_xver(rows: Sequence[Dict[str, Any]], phases: str, version: str,
                     knobs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """This version's selected rows (select_rows) plus, translated (translate_row),
    the selected rows of the other verified versions whose translated key this
    version has no row for (any phase: an own "pre" row means the run compiles it
    before RUN_START itself). Deterministic (table order); the first of several
    versions' rows with one translated key wins."""
    sel, other = select_rows(rows, phases, version)
    st = {"other": other, "xver": 0, "xver_dup": 0, "xver_bad": 0, "xver_phase": 0}
    only = _VERSION_ONLY_KWARGS.get(_mm(version), ())

    def _cover(key: Any) -> Any:
        # the key without this version's version-only call kwargs (see
        # _VERSION_ONLY_KWARGS): an own row that differs only by them covers it
        sp = split_key_options(key) if only else None
        if sp is None:
            return key
        return sp[0] + str({k: v for k, v in sp[1].items() if k not in only})

    seen = {(r.get("module"), r.get("qualname"), r.get("key")) for r in rows
            if str(r.get("triton_version") or "") == version}
    covered = {(m, q, _cover(k)) for (m, q, k) in seen}
    for r in rows:
        if str(r.get("triton_version") or "") == version:
            continue
        if phases != "all" and r.get("phase") != "post":
            st["xver_phase"] += 1
            continue
        try:
            t = translate_row(r, version, knobs)
        except ValueError:
            st["xver_bad"] += 1
            continue
        k = (t.get("module"), t.get("qualname"), t.get("key"))
        if k in seen or (k[0], k[1], _cover(k[2])) in covered:
            st["xver_dup"] += 1
            continue
        seen.add(k)
        covered.add((k[0], k[1], _cover(k[2])))
        sel.append(t)
        st["xver"] += 1
    return sel, st


def lpt_shard(rows: Sequence[Dict[str, Any]], n: int, default_s: float = 1.0) -> List[List[int]]:
    """Longest-processing-time-first assignment of row indices to n local ranks.
    Deterministic for a given row list (every rank computes the same shards
    without communication). Weight = the recorded cold compile time; a row only
    ever seen as a disk hit (compile_cold false: warm record run) gets the median
    cold time instead (or default_s), since its recorded time is not a compile."""
    n = max(1, int(n))
    cold = sorted(c for c in (_num(r.get("compile_s")) for r in rows
                              if r.get("compile_cold") is not False) if c)
    warm_s = cold[len(cold) // 2] if cold else default_s
    cost = []
    for r in rows:
        c = warm_s if r.get("compile_cold") is False else _num(r.get("compile_s"))
        cost.append(c if c else default_s)
    order = sorted(range(len(rows)), key=lambda i: (-cost[i], row_id(rows[i])))
    loads = [0.0] * n
    shards: List[List[int]] = [[] for _ in range(n)]
    for i in order:
        j = min(range(n), key=lambda k: (loads[k], k))
        shards[j].append(i)
        loads[j] += cost[i]
    return shards


def launch_token() -> str:
    """Same on every local rank of one launch, without communication: job id,
    node rank, rendezvous port and the spawning parent's pid + start time."""
    parts = [_env("SLURM_JOB_ID"), _env("NODE_RANK"), _env("MASTER_PORT")]
    ppid = os.getppid()
    parts.append(str(ppid))
    try:
        with open(f"/proc/{ppid}/stat", "r") as f:
            parts.append(f.read().rsplit(")", 1)[1].split()[19])  # field 22: starttime
    except Exception:
        pass
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _barrier_path(directory: str, token: str) -> str:
    return os.path.join(directory, "triton_spec_barrier", token)


def barrier_arrive(directory: str, token: str, member: int) -> None:
    d = _barrier_path(directory, token)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{member}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(str(os.getpid()))
    os.replace(tmp, os.path.join(d, f"{member}.arrived"))


def file_barrier(directory: str, token: str, member: int, members: int,
                 timeout_s: float, poll_s: float = 0.05) -> str:
    """Local (same filesystem) barrier, no torch.distributed. Returns "single",
    "ok" or "timeout" (the caller proceeds either way: a timeout only costs
    duplicate compiles)."""
    if members <= 1:
        return "single"
    barrier_arrive(directory, token, member)
    d = _barrier_path(directory, token)
    want = {f"{i}.arrived" for i in range(members)}
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        try:
            have = want & set(os.listdir(d))
        except OSError:
            have = set()
        if len(have) >= members:
            return "ok"
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(poll_s)


# =============================================================================
# Triton-dependent state
# =============================================================================
class _State:

    def __init__(self) -> None:
        self.run_start: Optional[float] = None
        self.wrapper_installed = False
        self.orig_do_compile: Any = None
        self.prev_listener: Any = None
        self.listener: Any = None
        self.load_hook: Any = None
        self.recorder: Optional["_Recorder"] = None
        self.monitor: Optional["_Monitor"] = None
        self.tls = threading.local()
        # id(JITFunction) -> [fn, hash_was_none_before_replay, [(kernel_cache, key, kernel)]]
        self.replayed: Dict[int, List[Any]] = {}
        # Every JIT callable (kernel, @triton.jit helper, constexpr callable) whose
        # cache_key the replay computed first: [(obj, frozen cache_key)]
        # (recheck_globals). final_recheck: the pre-RUN_START check ran.
        self.guarded: List[Tuple[Any, str]] = []
        self.final_recheck = False
        self.rank: Optional[int] = None
        self.hook_errors = 0
        self.foreign = 0
        self.globals_changed = 0
        self.atexit_registered = False
        self.pid = os.getpid()


_STATE = _State()


def _not_armed(reason: str, rank: int) -> None:
    logger.warning("%s NOT armed: %s rank=%d", TAG, " ".join(str(reason).split())[:400], rank)


def check_triton() -> Tuple[bool, str, Optional[str]]:
    """(ok, reason, version): triton importable, a verified version, and the
    internals this module calls still have the expected shape."""
    try:
        import inspect

        import triton
        from triton import knobs
        from triton.compiler import compiler as tcomp
        from triton.runtime import jit as tjit
    except Exception as ex:
        return False, f"triton import failed: {type(ex).__name__}: {ex}", None
    version = str(getattr(triton, "__version__", "?"))
    if ".".join(version.split(".")[:2]) not in _VERIFIED_TRITON:
        return False, f"triton {version} not verified (verified {', '.join(_VERIFIED_TRITON)})", version
    try:
        f = tjit.JITFunction._do_compile
        f = getattr(f, "_tspec_orig", f)
        params = inspect.signature(f).parameters
        names = tuple(params)
        if names != _DO_COMPILE_PARAMS and not any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            return False, f"JITFunction._do_compile{names} changed", version
        for owner, attr in ((tcomp.CompiledKernel, "_init_handles"), (knobs.compilation, "listener"),
                            (knobs.runtime, "kernel_load_start_hook"), (tjit, "JITCallable")):
            if not hasattr(owner, attr):
                return False, f"triton has no {attr}", version
        why = _build_rules_mismatch(_mm(version))
        if why:
            return False, why, version
    except Exception as ex:
        return False, f"internals check failed: {type(ex).__name__}: {ex}", version
    return True, "", version


def _jit_method_source(name: str) -> Optional[str]:
    """Source of JITFunction.<name> as SHIPPED (read from runtime/jit.py, so a test
    or tool that patches the attribute does not change what is checked); None
    when the source is not available."""
    import inspect
    import re

    from triton.runtime import jit as tjit

    try:
        src = inspect.getsource(tjit)
    except Exception:  # no source shipped: the version map is all there is
        return None
    i = src.find("\nclass JITFunction(")
    if i < 0:
        return None
    j = src.find("\nclass ", i + 1)
    body = src[i:j if j > 0 else len(src)]
    m = re.search(r"\n    def " + re.escape(name) + r"\(self.*?(?=\n    (?:def |@)|\Z)", body, re.S)
    return m.group(0) if m else None


def build_rules() -> Dict[str, Any]:
    """What this build's JITFunction.run / _pack_args do that the key and attrs
    rules depend on, read from their shipped source (None = not available)."""
    import re

    out: Dict[str, Any] = {"run_kwargs": None, "constexpr_attrs": None}
    src = _jit_method_source("run")
    if src is not None:
        out["run_kwargs"] = tuple(re.findall(r"""kwargs\[\s*["'](\w+)["']\s*\]\s*=""", src))
    src = _jit_method_source("_pack_args")
    if src is not None:
        out["constexpr_attrs"] = re.search(
            r"""x\[0\]\s*==\s*["']constexpr["']""", src) is not None
    return out


def _build_rules_mismatch(mm: str) -> str:
    """'' when the running build's run() kwargs and _pack_args constexpr rule are
    the ones recorded for its version in _RUN_KWARGS / _CONSTEXPR_ATTRS."""
    rules = build_rules()
    exp_kw = _RUN_KWARGS.get(mm)
    got_kw = rules["run_kwargs"]
    if exp_kw is not None and got_kw is not None and got_kw != exp_kw:
        return f"JITFunction.run adds call kwargs {got_kw}, expected {exp_kw} for {mm}"
    exp_ca = _CONSTEXPR_ATTRS.get(mm)
    got_ca = rules["constexpr_attrs"]
    if exp_ca is not None and got_ca is not None and got_ca != exp_ca:
        return f"JITFunction._pack_args constexpr attrs rule {got_ca}, expected {exp_ca} for {mm}"
    return ""


def _knob_snapshot() -> Dict[str, Any]:
    from triton import knobs

    out: Dict[str, Any] = {}
    for name, get in (("debug", lambda: knobs.runtime.debug),
                      ("instrumentation_mode", lambda: knobs.compilation.instrumentation_mode),
                      ("use_buffer_ops", lambda: knobs.amd.use_buffer_ops),
                      ("default_fp_fusion", lambda: knobs.language.default_fp_fusion),
                      ("override_arch", lambda: knobs.runtime.override_arch),
                      ("always_compile", lambda: knobs.compilation.always_compile),
                      # 3.9+ (N:372): a third run() key kwarg; left out on 3.6 / 3.8
                      # (record headers of those builds stay as before)
                      ("fpsan_homomorphic_casts",
                       lambda: knobs.compilation.fpsan_homomorphic_casts)):
        try:
            out[name] = get()
        except Exception:
            out[name] = None
    if out.get("fpsan_homomorphic_casts") is None:
        out.pop("fpsan_homomorphic_casts", None)
    return out


def _cache_env() -> Dict[str, str]:
    try:
        from triton._C.libtriton import get_cache_invalidating_env_vars

        return {str(k): str(v) for k, v in dict(get_cache_invalidating_env_vars()).items()}
    except Exception:
        return {}


def _triton_key_sha() -> Optional[str]:
    try:
        from triton.runtime.cache import triton_key

        return hashlib.sha256(triton_key().encode("utf-8")).hexdigest()
    except Exception:
        return None


def _lookup_attr(module: str, qualname: str, allow_import: bool) -> Any:
    m = sys.modules.get(module)
    if m is None:
        if not (allow_import and module.startswith(_IMPORT_PREFIXES)):
            raise LookupError(f"module {module} not imported")
        m = importlib.import_module(module)
    obj: Any = m
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _unwrap(obj: Any, want: Any) -> Any:
    """Follow Autotuner / Heuristics .fn chains (autotuner.py H:92/H:476 I:87/I:452)
    and __wrapped__ down to an instance of `want`, or None."""
    for _ in range(8):
        if isinstance(obj, want):
            return obj
        nxt = getattr(obj, "fn", None)
        if nxt is None:
            nxt = getattr(obj, "__wrapped__", None)
        if nxt is None or nxt is obj:
            return None
        obj = nxt
    return None


def _jit_types() -> Tuple[Any, Any]:
    from triton.runtime.jit import JITCallable, JITFunction

    return JITFunction, JITCallable


def resolve_kernel(module: str, qualname: str, allow_import: bool = True) -> Any:
    """module.qualname -> the JITFunction the runtime compiles (unwrapping
    Autotuner / Heuristics); LookupError when not found."""
    JITFunction, _ = _jit_types()
    obj = _lookup_attr(module, qualname, allow_import)
    fn = _unwrap(obj, JITFunction)
    if fn is None:
        raise LookupError(f"{module}.{qualname} is {type(obj).__name__}, no JITFunction under .fn")
    if (getattr(fn.fn, "__module__", None), getattr(fn.fn, "__qualname__", None)) != (module, qualname):
        raise LookupError(f"{module}.{qualname} resolves to another kernel ({fn!r})")
    return fn


def _jit_index() -> Dict[Tuple[str, str], List[Any]]:
    """Every live JITFunction by (module, qualname) (gc scan; nested / dynamic kernels)."""
    JITFunction, _ = _jit_types()
    idx: Dict[Tuple[str, str], List[Any]] = {}
    for o in _live_of_type(JITFunction):
        try:
            idx.setdefault((o.fn.__module__, o.fn.__qualname__), []).append(o)
        except Exception:
            continue
    return idx


# =============================================================================
# Wrapper around JITFunction._do_compile (recorder + monitor). Not the official
# jit_post_compile_hook: _call_hook json-serializes the constexprs first
# (serialize_specialization_data, jit.py H:378-393 I:374-388), which raises for a
# JITFunction constexpr (a class named "dtype", torch.dtype included, becomes its
# str(); I:376) and would break a recording run. The own codec is needed anyway:
# preload decodes a dtype string back only as a tl.dtype, so "torch.float16"
# would come back as a str (I:797-801).
# =============================================================================
def _wrapped_do_compile(orig: Callable[..., Any]) -> Callable[..., Any]:

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        tls = _STATE.tls
        tls.depth = getattr(tls, "depth", 0) + 1
        tls.last = None
        t0 = time.perf_counter()
        try:
            kernel = orig(self, key, signature, device, constexprs, options, attrs, warmup)
        finally:
            tls.depth -= 1
        dt = time.perf_counter() - t0
        try:
            mon = _STATE.monitor
            if mon is not None:
                mon.on_compile(self, dt)
            rec = _STATE.recorder
            if rec is not None and kernel is not None:
                rec.record(self, key, signature, device, constexprs, options, attrs, kernel, dt,
                           warmup, tls)
        except Exception as ex:  # observation must never break a compile
            _STATE.hook_errors += 1
            if _STATE.hook_errors <= 3:
                logger.warning("%s record/monitor failed for %s: %r", TAG,
                               getattr(self, "_fn_name", "?"), ex)
        return kernel

    _do_compile._tspec_orig = orig  # type: ignore[attr-defined]
    return _do_compile


def _ensure_wrapper() -> None:
    if _STATE.wrapper_installed:
        return
    from triton import knobs
    from triton.runtime import jit as tjit

    orig = tjit.JITFunction._do_compile
    prev = knobs.compilation.listener

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        try:
            tls = _STATE.tls
            if getattr(tls, "depth", 0) > 0:
                tls.last = (metadata.get("hash"), bool(cache_hit), times)
            else:
                _STATE.foreign += 1  # compile() not reached through JITFunction (e.g. Inductor)
            mon = _STATE.monitor
            if mon is not None and _STATE.run_start is not None:
                mon.on_listener(bool(cache_hit), getattr(tls, "depth", 0) > 0)
        except Exception:
            _STATE.hook_errors += 1
        finally:
            if prev is not None:
                prev(src=src, metadata=metadata, metadata_group=metadata_group, times=times,
                     cache_hit=cache_hit)

    def on_load(module, function, name, metadata_group, hash):
        mon = _STATE.monitor
        if mon is not None and _STATE.run_start is not None:
            mon.loads += 1

    tjit.JITFunction._do_compile = _wrapped_do_compile(orig)
    knobs.compilation.listener = listener
    knobs.runtime.kernel_load_start_hook.add(on_load)
    _STATE.orig_do_compile, _STATE.prev_listener = orig, prev
    _STATE.listener, _STATE.load_hook = listener, on_load
    _STATE.wrapper_installed = True


def _uninstall() -> None:
    """Undo _ensure_wrapper (tests)."""
    if not _STATE.wrapper_installed:
        return
    from triton import knobs
    from triton.runtime import jit as tjit

    tjit.JITFunction._do_compile = _STATE.orig_do_compile
    if knobs.compilation.listener is _STATE.listener:
        knobs.compilation.listener = _STATE.prev_listener
    try:
        knobs.runtime.kernel_load_start_hook.remove(_STATE.load_hook)
    except Exception:
        pass
    _STATE.wrapper_installed = False


def _reset_for_tests() -> None:
    global _STATE
    _uninstall()
    rec = _STATE.recorder
    if rec is not None:
        rec.close()
    _STATE = _State()


# =============================================================================
# RECORD
# =============================================================================
def _call_opts_for(jf: Any, device: Any, key: str, options: Any) -> Optional[List[List[Any]]]:
    """The non-parameter call kwargs (the L1 key's dict suffix), kept only when
    backend.parse_options(kwargs) reproduces the compiled options exactly, so a
    replay can re-derive the options the way run() does in its own environment
    (_pack_args, jit.py H:704 I:673)."""
    sp = split_key_options(key)
    if sp is None:
        return None
    _, kw = sp
    backend = jf.device_caches[device][3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = backend.parse_options(dict(kw))
    if again.__dict__ != options.__dict__:
        return None
    return [[str(k), _jsonable(v)] for k, v in kw.items()]


def _target_of(jf: Any, device: Any) -> Optional[Dict[str, Any]]:
    """The GPUTarget the kernel compiles for on `device` (device_caches[device][2],
    jit.py H:738 I:707) as JSON: backend, arch, warp_size."""
    try:
        t = jf.device_caches[device][2]
        return {"backend": str(t.backend), "arch": _jsonable(t.arch), "warp_size": int(t.warp_size)}
    except Exception:
        return None


def _globals_snapshot(jf: Any) -> List[List[str]]:
    """[module, name, repr] of the globals Triton re-checks on every launch
    (used_global_vals, jit.py H:762-766 I:725-729). Module "" = not a module
    dict (closure nonlocals)."""
    out: List[List[str]] = []
    for (name, _gid), (val, gdict) in list(getattr(jf, "used_global_vals", {}).items()):
        mod = gdict.get("__name__", "") if isinstance(gdict, dict) else ""
        m = sys.modules.get(mod) if isinstance(mod, str) and mod else None
        if m is None or getattr(m, "__dict__", None) is not gdict:
            mod = ""
        try:
            rep = repr(val)[:300]
        except Exception:
            rep = "<unrepr>"
        out.append([str(mod), str(name), rep])
    return out


class _Recorder:

    def __init__(self, directory: str, rank: int, local_rank: int, version: str,
                 world_size: Optional[int] = None) -> None:
        self.dir = os.path.abspath(directory)
        os.makedirs(self.dir, exist_ok=True)
        self.rank, self.local_rank, self.version = rank, local_rank, version
        self.world_size = world_size
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.n = self.n_unsup = self.n_err = 0
        self.disabled = False
        self.f: Any = None
        self.pid = -1
        self._open()
        self.write({"kind": "header", "schema": SCHEMA, "triton_version": version,
                    "rank": rank, "local_rank": local_rank, "pid": os.getpid(),
                    "world_size": world_size,
                    "host": socket.gethostname(), "t0": round(self.t0, 3),
                    "python": sys.version.split()[0], "knobs": _knob_snapshot(),
                    "env": _cache_env(), "triton_key_sha": _triton_key_sha()})

    def _open(self) -> None:
        self.pid = os.getpid()
        self.path = os.path.join(self.dir, f"rec.r{self.rank}.p{self.pid}.jsonl")
        self.f = open(self.path, "a", buffering=1, encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        if self.disabled:
            return
        line = json.dumps(obj, separators=(",", ":"), default=str)
        with self.lock:
            try:
                if os.getpid() != self.pid:  # forked child: its own file
                    self._open()
                self.f.write(line + "\n")
            except Exception as ex:
                self.n_err += 1
                if self.n_err >= 3:
                    self.disabled = True
                logger.warning("%s record write failed (%s)%s rank=%d", TAG, ex,
                               "; recorder disabled" if self.disabled else "", self.rank)

    def close(self) -> None:
        try:
            if self.f is not None and os.getpid() == self.pid:
                self.write({"kind": "summary", "rows": self.n, "unsupported": self.n_unsup,
                            "write_errors": self.n_err, "foreign_compiles": _STATE.foreign,
                            "hook_errors": _STATE.hook_errors})
                self.f.close()
        except Exception:
            pass
        self.f = None

    def record(self, jf: Any, key: str, signature: Dict[str, Any], device: Any,
               constexprs: Dict[Any, Any], options: Any, attrs: Dict[Any, Any], kernel: Any,
               dt: float, warmup: bool, tls: Any) -> None:
        fn = jf.fn
        row: Dict[str, Any] = {
            "schema": SCHEMA, "kind": "compile", "module": fn.__module__,
            "qualname": fn.__qualname__, "name": fn.__name__, "triton_version": self.version,
            "phase": "post" if _STATE.run_start is not None else "pre",
            "warmup": bool(warmup), "compile_s": round(dt, 4), "rank": self.rank,
            "local_rank": self.local_rank, "pid": os.getpid(),
            "t_rel": round(time.time() - (_STATE.run_start or self.t0), 3),
        }
        if self.world_size:
            row["world_size"] = int(self.world_size)
        tgt = _target_of(jf, device)
        if tgt is not None:
            row["target"] = tgt
        if getattr(tls, "replaying", False):
            row["via"] = "replay"
        try:
            row.update(encode_compile_args(key, signature, constexprs, attrs, options.__dict__))
        except Unsupported as ex:
            self.n_unsup += 1
            row.update({"unsupported": str(ex)[:200], "key": key})
            self.write(row)
            return
        row["fn_cache_key"] = str(jf.cache_key)  # already computed by compile()
        try:
            row["call_opts"] = _call_opts_for(jf, device, key, options)
        except Exception:
            row["call_opts"] = None
        row["globals"] = _globals_snapshot(jf)
        try:
            row["same_instance"] = resolve_kernel(fn.__module__, fn.__qualname__,
                                                  allow_import=False) is jf
        except Exception:
            row["same_instance"] = False
        if type(kernel).__name__ != "FutureKernel":  # never force an async compile here
            kh = getattr(kernel, "hash", None)
            if isinstance(kh, str):
                row["kernel_hash"] = kh
                last = getattr(tls, "last", None)
                if last is not None and last[0] == kh:
                    row["disk_hit"] = bool(last[1])
        self.n += 1
        self.write(row)


def _resolve_world_size(world_size: Optional[int]) -> Optional[int]:
    if world_size is not None:
        return int(world_size)
    return _int_env("WORLD_SIZE", 0) or None


def install_recorder(record_dir: Optional[str] = None, rank: Optional[int] = None,
                     local_rank: Optional[int] = None, world_size: Optional[int] = None) -> bool:
    """Wrap JITFunction._do_compile and write every compile to
    <record_dir>/rec.r<RANK>.p<PID>.jsonl (record_dir defaults to
    TRITON_SPEC_RECORD). Call before anything compiles. Never raises."""
    r = _resolve_rank(rank)
    if record_dir is None:
        record_dir = env_record_dir()
    if not record_dir:
        return False
    try:
        if _STATE.recorder is not None:
            return True
        ok, reason, version = check_triton()
        if not ok:
            _not_armed(f"record: {reason}", r)
            return False
        rec = _Recorder(record_dir, r, _resolve_local_rank(local_rank), str(version),
                        _resolve_world_size(world_size))
        _ensure_wrapper()
        _STATE.recorder = rec
        _register_atexit()
        logger.info("armed OK: %s record -> %s rank=%d", TAG, rec.path, r)
        return True
    except Exception as ex:
        _not_armed(f"record: {type(ex).__name__}: {ex}", r)
        return False


# =============================================================================
# MONITOR
# =============================================================================
class _Monitor:

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.lock = threading.Lock()
        self.pre = 0
        self.post = 0
        self.post_s = 0.0
        self.names: Dict[str, int] = {}
        self.disk_hit = self.disk_miss = self.loads = self.foreign_post = 0
        self.reported: set = set()

    def on_compile(self, jf: Any, dt: float) -> None:
        if _STATE.run_start is None:
            self.pre += 1
            return
        name = str(getattr(jf, "__name__", None) or getattr(jf, "_fn_name", "?"))
        with self.lock:
            self.post += 1
            self.post_s += dt
            self.names[name] = self.names.get(name, 0) + 1

    def on_listener(self, cache_hit: bool, via_jit: bool) -> None:
        if not via_jit:
            self.foreign_post += 1
        elif cache_hit:
            self.disk_hit += 1
        else:
            self.disk_miss += 1


def install_monitor(rank: Optional[int] = None) -> bool:
    """Count JITFunction._do_compile calls after mark_run_start() (explicit
    call = on; train_ranker goes through install_from_env). Never raises."""
    r = _resolve_rank(rank)
    try:
        if _STATE.monitor is not None:
            return True
        ok, reason, _ = check_triton()
        if not ok:
            _not_armed(f"monitor: {reason}", r)
            return False
        _ensure_wrapper()
        _STATE.monitor = _Monitor(r)
        _register_atexit()
        logger.info("%s monitor on (post-RUN_START compile count at RUN_STOP and exit) rank=%d",
                    TAG, r)
        return True
    except Exception as ex:
        _not_armed(f"monitor: {type(ex).__name__}: {ex}", r)
        return False


def monitor_report(at: str = "report", log: bool = True) -> str:
    """The contract line "[triton-spec] monitor post-run_start: compiles=N
    kernels=<name:count,...top 12> rank=R" (+ a detail line when logged)."""
    m = _STATE.monitor
    if m is None:
        return f"{TAG} monitor off"
    with m.lock:
        top = sorted(m.names.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        post, post_s = m.post, m.post_s
    line = (f"{TAG} monitor post-run_start: compiles={post} "
            f"kernels={','.join(f'{k}:{v}' for k, v in top)} rank={m.rank}")
    if log:
        logger.info("%s", line)
        logger.info("%s monitor detail: at=%s run_start=%s compile_s=%.2f disk_hit=%d "
                    "disk_miss=%d hsaco_loads=%d foreign=%d pre_compiles=%d "
                    "globals_changed=%d hook_errors=%d rank=%d", TAG, at,
                    "yes" if _STATE.run_start is not None else "no", post_s, m.disk_hit,
                    m.disk_miss, m.loads, m.foreign_post, m.pre, _STATE.globals_changed,
                    _STATE.hook_errors, m.rank)
    return line


def on_run_stop() -> None:
    """MLPerf RUN_STOP hook (mlperf_logging_utils.MLPerfRunTracker.run_stop)."""
    try:
        m = _STATE.monitor
        if m is not None and "run_stop" not in m.reported:
            m.reported.add("run_stop")
            monitor_report("run_stop")
    except Exception as ex:
        logger.warning("%s monitor report failed: %r", TAG, ex)


def _at_exit() -> None:
    if os.getpid() != _STATE.pid:  # forked child (DataLoader worker)
        return
    try:
        m = _STATE.monitor
        if m is not None and "exit" not in m.reported:
            m.reported.add("exit")
            monitor_report("exit")
        if _STATE.recorder is not None:
            _STATE.recorder.close()
    except Exception:
        pass


def _register_atexit() -> None:
    if not _STATE.atexit_registered:
        atexit.register(_at_exit)
        _STATE.atexit_registered = True
        _STATE.pid = os.getpid()


def install_from_env(rank: Optional[int] = None, local_rank: Optional[int] = None,
                     world_size: Optional[int] = None) -> Dict[str, bool]:
    """train_ranker entry: recorder (TRITON_SPEC_RECORD) and monitor
    (TRITON_SPEC_MONITOR). Imports triton only when one of them is on."""
    out = {"record": False, "monitor": False}
    rec = env_record_dir()
    if rec:
        out["record"] = install_recorder(rec, rank=rank, local_rank=local_rank,
                                         world_size=world_size)
    if env_monitor():
        out["monitor"] = install_monitor(rank=rank)
    return out


def mark_run_start() -> None:
    """Called right after the MLPerf run_start is logged: compiles after this
    are "post" (recorder) and counted (monitor). Only a timestamp: train_ranker
    runs recheck_globals("run_start") just BEFORE run_start is logged. A driver
    that never called it after a replay gets that check here (fallback)."""
    if _STATE.run_start is not None:
        return
    _STATE.run_start = time.time()
    if _STATE.guarded and not _STATE.final_recheck:
        recheck_globals("mark_run_start")


def _live_of_type(cls: Any) -> List[Any]:
    """Every live gc-tracked instance of cls. Checks type(o), never o.__class__:
    some module-level proxies (e.g. torch.distributed.reduce_op) warn or run code
    on attribute access."""
    hit: Dict[type, bool] = {}
    out: List[Any] = []
    for o in gc.get_objects():
        t = type(o)
        ok = hit.get(t)
        if ok is None:
            try:
                ok = issubclass(t, cls)
            except Exception:
                ok = False
            hit[t] = ok
        if ok:
            out.append(o)
    return out


def _snapshot_hashed() -> Dict[int, Any]:
    """id -> object for every live JIT callable whose cache_key is computed (gc scan)."""
    _, JITCallable = _jit_types()
    return {id(o): o for o in _live_of_type(JITCallable) if getattr(o, "hash", None) is not None}


def _changed_globals(obj: Any) -> List[str]:
    """Names in obj.used_global_vals whose current value differs from the frozen one
    (the comparison run() makes, jit.py H:762-766 I:725-729)."""
    absent = object()
    changed = []
    for (name, _), (val, gdict) in list((getattr(obj, "used_global_vals", None) or {}).items()):
        try:
            if gdict.get(name, absent) != val:
                changed.append(name)
        except Exception:
            changed.append(name)
    return changed


def recheck_globals(at: str = "check") -> int:
    """Globals guard for replayed kernels. Never raises; returns the number of
    JIT callables whose cache_key moved (their stale entries were dropped).

    Triton freezes the globals a JIT callable reads (used_global_vals) when its
    cache_key is first computed; constexpr global values also enter the
    cache_key (jit.py H:530-537 I:510-514). A launch re-checks only the
    kernel's own frozen set (H:762-766 I:725-729). 3.8 merges a called helper's
    globals into its caller's set (H:98-114); 3.6 merges them BEFORE the helper's
    cache_key is computed (I:99-111), so a helper first hashed through its caller
    adds nothing and a changed helper global goes unnoticed at launch (a stale
    binary, no RuntimeError). Replay computes these cache_keys earlier than the
    run would, for the kernels, their helpers and JIT-callable constexprs
    (_STATE.guarded). If any of them reads a global that changed since: recompute
    the cache_key of every guarded callable (all reset first, so helpers are
    recomputed before their callers use them), drop ALL in-process cache entries
    of every kernel whose cache_key moved (the L1 key does not contain the
    kernel's own hash), and drop replayed entries whose key embeds a moved
    callable's old cache_key (a JIT-callable constexpr: unreachable now). Those
    kernels compile at first use with the current values, exactly as without
    TSPEC; unchanged callables end up in the same state as before. train_ranker
    calls this before the BLAS warmup and just before run_start is logged, so it
    never runs inside the timed region."""
    if at == "run_start":
        _STATE.final_recheck = True
    if not _STATE.guarded:
        return 0
    try:
        changed = {}
        for obj, _ in _STATE.guarded:
            if getattr(obj, "hash", None) is not None:
                names = _changed_globals(obj)
                if names:
                    changed[id(obj)] = names
        if not changed:
            return 0
        JITFunction, _ = _jit_types()
        rank = _STATE.rank if _STATE.rank is not None else _resolve_rank(None)
        live = [(o, o.hash) for o, _ in _STATE.guarded if getattr(o, "hash", None) is not None]
        for o, _ in live:
            o.hash = None
            o.used_global_vals = {}
        moved: List[Tuple[Any, str]] = []
        for o, old in live:
            try:
                new = o.cache_key
            except Exception:  # left unhashed: computed (and raised) at first use, as without TSPEC
                o.hash, o.used_global_vals, new = None, {}, None
            if new != old:
                moved.append((o, old))
        stale = [old for _, old in moved if isinstance(old, str) and len(old) >= 32]
        moved_ids = {id(o) for o, _ in moved}
        kernels = {id(o): o for o, _ in _STATE.guarded if isinstance(o, JITFunction)}
        for ent in _STATE.replayed.values():
            kernels.setdefault(id(ent[0]), ent[0])
        evicted = 0
        for k in kernels.values():
            for dc in list(k.device_caches.values()):
                if id(k) in moved_ids:
                    evicted += len(dc[0])
                    dc[0].clear()
                    continue
                for key in [x for x in list(dc[0]) if any(c in str(x) for c in stale)]:
                    dc[0].pop(key, None)
                    evicted += 1
        for ent in _STATE.replayed.values():
            ent[2] = [(kc, key, kern) for kc, key, kern in ent[2] if kc.get(key) is kern]
        _STATE.guarded = [(o, str(o.hash) if getattr(o, "hash", None) is not None else ck)
                          for o, ck in _STATE.guarded]
        _STATE.globals_changed += len(moved)
        for o, _ in moved[:8]:
            names = changed.get(id(o))
            logger.warning("%s globals guard (%s): %s %s; its cache entries were dropped "
                           "(compiles at first use) rank=%d", TAG, at,
                           getattr(o, "_fn_name", "?"),
                           f"global(s) {','.join(names[:5])} changed after the replay" if names
                           else "depends on a changed global", rank)
        logger.warning("%s globals guard (%s): changed=%d moved=%d evicted_entries=%d rank=%d",
                       TAG, at, len(changed), len(moved), evicted, rank)
        return len(moved)
    except Exception as ex:
        logger.warning("%s globals guard (%s) failed: %r", TAG, at, ex)
        return 0


# =============================================================================
# REPLAY
# =============================================================================
class _Skip(Exception):

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _track(fn: Any) -> List[Any]:
    """Remember, before the replay first computes fn.cache_key, whether it was
    still unhashed (for the error path in _Replay.build)."""
    ent = _STATE.replayed.get(id(fn))
    if ent is None:
        ent = _STATE.replayed[id(fn)] = [fn, getattr(fn, "hash", None) is None, []]
    return ent


def _prime(kernel: Any) -> None:
    """CompiledKernel._init_handles (compiler.py H:448-485 I:436-471): launcher +
    hsaco load on the current device, no launch. _init_handles sets _run before
    loading (H:464 I:452), so on a failure with nothing loaded restore the
    pristine state: the first launch then retries exactly as without TSPEC."""
    if getattr(kernel, "module", None) is not None:
        return
    try:
        kernel._init_handles()
    except BaseException:
        if getattr(kernel, "module", None) is None:
            kernel._run = None
        raise


class _Replay:

    def __init__(self, rows: List[Dict[str, Any]], hdr: Dict[str, Any], device: Any,
                 prime: bool, c: Dict[str, Any]) -> None:
        from triton import knobs

        self.knobs = knobs
        self.cur_knobs = _knob_snapshot()
        self.rows, self.hdr, self.device, self.prime, self.c = rows, hdr, device, prime, c
        self.state: List[Optional[str]] = [None] * len(rows)
        self.rcache: Dict[Tuple[str, str, Any, bool], Any] = {}
        self.index: Optional[Dict[Tuple[str, str], List[Any]]] = None
        self.JITFunction, self.JITCallable = _jit_types()

    # -- bookkeeping ---------------------------------------------------------
    def _note(self, kind: str, row: Dict[str, Any], what: Any) -> None:
        lst = self.c["first_errors"]
        if len(lst) < 5:
            lst.append(f"{kind} {row.get('module')}.{row.get('qualname')}: {what}"[:300])

    def skip(self, i: int, reason: str, detail: str) -> None:
        self.state[i] = "skip"
        self.c["unresolved"] += 1
        by = self.c["unresolved_by"]
        by[reason] = by.get(reason, 0) + 1
        if reason not in ("missing",):
            self._note(f"skip[{reason}]", self.rows[i], detail)
        elif by[reason] <= 2:
            self._note("skip[missing]", self.rows[i], detail)

    def error(self, i: int, stage: str, ex: BaseException) -> None:
        if self.state[i] is None and stage == "decode":
            self.c["unresolved"] += 1
            by = self.c["unresolved_by"]
            by["decode"] = by.get("decode", 0) + 1
        self.state[i] = "error"
        self.c["errors"] += 1
        self._note(f"error[{stage}]", self.rows[i], f"{type(ex).__name__}: {ex}")

    # -- resolution ----------------------------------------------------------
    def _gc_find(self, module: str, qualname: str, ck: Any) -> List[Any]:
        if self.index is None:
            self.index = _jit_index()
        cands = self.index.get((module, qualname), [])
        if len(cands) <= 1:
            return list(cands)
        if isinstance(ck, str) and ck:
            for f in cands:
                _track(f)  # before cache_key is computed below
            same = [f for f in cands if str(f.cache_key) == ck]
            if same:
                return same[:8]
        return []  # ambiguous

    def resolve(self, row: Dict[str, Any]) -> List[Any]:
        module, qualname = row.get("module"), row.get("qualname")
        if not (isinstance(module, str) and isinstance(qualname, str)):
            raise _Skip("missing", "malformed module/qualname")
        same = row.get("same_instance", True) is not False
        k = (module, qualname, row.get("fn_cache_key"), same)
        if k in self.rcache:
            got = self.rcache[k]
            if isinstance(got, _Skip):
                raise got
            return got
        fns: List[Any] = []
        why = ""
        if same and "<locals>" not in qualname:
            try:
                fns = [resolve_kernel(module, qualname, allow_import=True)]
            except Exception as ex:
                why = f"{type(ex).__name__}: {ex}"
        if not fns:
            fns = self._gc_find(module, qualname, row.get("fn_cache_key"))
        if not fns:
            err = _Skip("missing", why or f"{module}.{qualname} not found")
            self.rcache[k] = err
            raise err
        for fn in fns:
            _track(fn)
        self.rcache[k] = fns
        return fns

    def resolve_jit_value(self, module: str, qualname: str) -> Any:
        try:
            obj = _unwrap(_lookup_attr(module, qualname, allow_import=True), self.JITCallable)
        except Exception as ex:
            obj, why = None, f"{type(ex).__name__}: {ex}"
        else:
            why = "not a JIT callable"
        if obj is None:  # a renamed / removed callee: the row is stale, not broken
            raise _Skip("missing", f"constexpr callable {module}.{qualname}: {why}")
        return obj

    # -- validation + decode ---------------------------------------------------
    def check_globals(self, row: Dict[str, Any]) -> None:
        for ent in row.get("globals") or []:
            if not (isinstance(ent, list) and len(ent) == 3 and all(isinstance(x, str) for x in ent)):
                continue
            mod, name, rep = ent
            if not mod or " at 0x" in rep or rep == "<unrepr>":
                continue
            m = sys.modules.get(mod)
            if m is None:
                continue
            cur = getattr(m, "__dict__", {}).get(name, _MISSING)
            if cur is _MISSING:
                raise _Skip("globals", f"{mod}.{name} missing")
            try:
                crep = repr(cur)[:300]
            except Exception:
                continue
            if crep != rep:
                raise _Skip("globals", f"{mod}.{name} {rep[:40]} -> {crep[:40]}")

    def options_for(self, row: Dict[str, Any], key: str, backend: Any) -> Any:
        knobs = self.knobs
        co = row.get("call_opts")
        if isinstance(co, list) and co:
            kw: Dict[str, Any] = {}
            for pair in co:
                if not (isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)):
                    raise ValueError("malformed call_opts")
                kw[pair[0]] = _tuplify(pair[1])
            sp = split_key_options(key)
            if sp is not None and sp[1] == kw:
                # run() fills these two from the knobs (jit.py H:727-728 I:696-697): a
                # different value means another key, so the row would never be hit.
                if knobs.runtime.debug and not kw.get("debug", False):
                    raise _Skip("knobs", "TRITON_DEBUG")
                if kw.get("instrumentation_mode", "") != knobs.compilation.instrumentation_mode:
                    raise _Skip("knobs", "TRITON_INSTRUMENTATION_MODE")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    opts = backend.parse_options(dict(kw))  # what _pack_args does (H:704 I:673)
                    rec = row.get("options")
                    if isinstance(rec, dict):
                        try:
                            if backend.parse_options(_tuplify(rec)).__dict__ != opts.__dict__:
                                self.c["opts_drift"] += 1
                        except Exception:
                            self.c["opts_drift"] += 1
                return opts
        # Fallback: the recorded HIPOptions fields (the preload recipe, jit.py
        # H:861-866 I:809-815), only while the knobs parse_options and run() read
        # match the record run's. "arch" is dropped: in the options it would override
        # this device's target (amd/compiler.py H:160/H:181 I:119/I:140).
        hk = self.hdr.get("knobs") if isinstance(self.hdr.get("knobs"), dict) else {}
        cur = self.cur_knobs
        for name in ("debug", "instrumentation_mode", "default_fp_fusion", "override_arch"):
            if name in hk and hk[name] != cur.get(name):
                raise _Skip("knobs", name)
        rec = row.get("options")
        if not isinstance(rec, dict):
            raise ValueError("no usable options")
        rec = {k: v for k, v in rec.items() if k != "arch"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opts = backend.parse_options(_tuplify(rec))
        self.c["fallback_opts"] += 1
        return opts

    def prepare(self, row: Dict[str, Any], fn: Any) -> Tuple[str, Dict[str, Any], Dict[Any, Any],
                                                             Dict[Any, Any], Any]:
        sig = row.get("signature")
        if not (isinstance(sig, list) and all(isinstance(p, list) and len(p) == 2
                                              and isinstance(p[0], str) for p in sig)):
            raise ValueError("malformed signature")
        names = [p.name for p in fn.params]
        if [p[0] for p in sig] != names:
            raise _Skip("signature", f"params {[p[0] for p in sig]} -> {names}")
        for kp, (pname, ty) in zip(fn.params, sig):
            if kp.is_constexpr and ty != "constexpr":
                raise _Skip("signature", f"{pname} is constexpr now")
            ann = kp.annotation_type
            if ann and not kp.is_constexpr and ty not in (ann, "constexpr"):
                raise _Skip("signature", f"{pname} annotated {ann}, recorded {ty}")
        signature = {pname: _tuplify(ty) for pname, ty in sig}
        tgt = row.get("target")  # absent in rows recorded before the field existed
        if isinstance(tgt, dict):
            now = _target_of(fn, self.device)
            if now is not None and now != tgt:
                raise _Skip("target", f"recorded {tgt} now {now}")
        self.check_globals(row)
        key = row.get("key")
        if not isinstance(key, str):
            raise ValueError("malformed key")
        rekeys: List[Tuple[str, str]] = []

        def on_jit(d: Dict[str, Any], obj: Any) -> None:
            ck = d.get("ck")
            if isinstance(ck, str) and len(ck) >= 32:
                _track(obj)  # the run computes this cache_key for its L1 key anyway
                cur = str(obj.cache_key)
                if cur != ck:
                    rekeys.append((ck, cur))

        cx_raw, at_raw = row.get("constexprs"), row.get("attrs")
        if not (isinstance(cx_raw, list) and isinstance(at_raw, list)):
            raise ValueError("malformed constexprs/attrs")
        constexprs: Dict[Any, Any] = {}
        for item in cx_raw:
            path, enc = item
            if not (isinstance(path, list) and all(type(x) is int for x in path)):
                raise ValueError("malformed constexpr path")
            constexprs[tuple(path)] = decode_value(enc, resolve_jit=self.resolve_jit_value,
                                                   on_jit=on_jit)
        for old, new in rekeys:  # a constexpr JIT callable whose source moved
            if old in key:
                key = key.replace(old, new)
                self.c["rekeyed"] += 1
        attrs: Dict[Any, Any] = {}
        for item in at_raw:
            path, val = item
            if not (isinstance(path, list) and all(type(x) is int for x in path)
                    and isinstance(val, list)):
                raise ValueError("malformed attrs")
            attrs[tuple(path)] = val
        backend = fn.device_caches[self.device][3]
        src = row.get("xver_from")
        if isinstance(src, str):  # translate_row did the key; the attrs need the backend
            try:
                attrs = dict(translate_attrs(sig, list(attrs.items()), constexprs, src,
                                             str(row.get("triton_version")), backend.parse_attr))
            except ValueError as ex:
                raise _Skip("xver", str(ex))
        options = self.options_for(row, key, backend)
        return key, signature, constexprs, attrs, options

    # -- compile + prime ---------------------------------------------------------
    def build(self, i: int, row: Dict[str, Any], fn: Any, prep: Tuple[Any, ...]) -> bool:
        key, signature, constexprs, attrs, options = prep
        ent = _track(fn)
        stage = "compile"
        try:
            kc = fn.device_caches[self.device][0]
            kernel = kc.get(key)
            if kernel is None:
                # == JITFunction.preload (jit.py H:867-875 I:816-824)
                kernel = fn._do_compile(key, signature, self.device, constexprs, options, attrs,
                                        warmup=True)
                if kernel is None:
                    raise RuntimeError("jit_cache_hook skipped the compile")
                if type(kernel).__name__ == "FutureKernel":
                    kernel = kernel.result()
                ent[2].append((kc, key, kernel))
                self.c["compiled"] += 1
                if row.get("xver_from"):
                    # another build's cache_key algorithm: no source comparison
                    self.c["xver_compiled"] += 1
                elif str(fn.cache_key) != str(row.get("fn_cache_key")):
                    self.c["src_changed"] += 1
                if isinstance(row.get("kernel_hash"), str):
                    self.c["hash_known"] += 1
                    if getattr(kernel, "hash", None) == row["kernel_hash"]:
                        self.c["hash_same"] += 1
            else:
                self.c["already"] += 1
            if self.prime:
                stage = "prime"
                try:
                    _prime(kernel)
                except Exception as ex:
                    if type(ex).__name__ != "OutOfResources":
                        raise
                    # An autotune config that compiles but exceeds this GPU's shared
                    # memory / thread limit: the first launch raises the same
                    # OutOfResources (Autotuner._bench: inf timing, autotuner.py H:170
                    # I:165), exactly as without TSPEC. Not a replay error.
                    self.c["prime_oor"] += 1
                    return True
                self.c["primed"] += 1
            return True
        except Exception as ex:
            if ent[1] and not ent[2] and getattr(fn, "hash", None) is not None:
                # nothing of this kernel was compiled here: back to "never hashed",
                # as if the replay had not touched it (cache_key, jit.py H:515-538)
                fn.hash = None
                fn.used_global_vals = {}
            self.error(i, stage, ex)
            return False

    def process(self, i: int) -> None:
        if self.state[i] is not None:
            return
        t0 = time.perf_counter()
        try:
            self._process(i)
        finally:  # per-row time: the slowest rows go into the counts / detail line
            dt = time.perf_counter() - t0
            slow = self.c["slowest"]
            slow.append([str(self.rows[i].get("qualname")), round(dt, 3)])
            slow.sort(key=lambda x: -x[1])
            del slow[5:]

    def _process(self, i: int) -> None:
        row = self.rows[i]
        try:
            fns = self.resolve(row)
        except _Skip as s:
            self.skip(i, s.reason, s.detail)
            return
        except Exception as ex:
            self.skip(i, "missing", f"{type(ex).__name__}: {ex}")
            return
        preps = []
        try:
            for fn in fns:
                preps.append(self.prepare(row, fn))
        except _Skip as s:
            self.skip(i, s.reason, s.detail)
            return
        except Exception as ex:
            self.error(i, "decode", ex)
            return
        self.c["resolved"] += 1
        self.c["instances"] += len(fns)
        for fn, prep in zip(fns, preps):
            if not self.build(i, row, fn, prep):
                return
        self.state[i] = "done"


def _resolve_table_path(table: str) -> Optional[str]:
    cands = [table]
    if not os.path.isabs(table):
        # relative paths: the current directory, then the recommendation/ root
        here = os.path.dirname(os.path.abspath(__file__))
        cands.append(os.path.join(os.path.abspath(os.path.join(here, "..", "..", "..")), table))
    for p in cands:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _new_counts(table: str, phases: str, rank: int) -> Dict[str, Any]:
    return {"armed": False, "reason": "", "table": table, "phases": phases, "rank": rank,
            "table_rows": 0, "rows": 0, "resolved": 0, "unresolved": 0, "compiled": 0,
            "primed": 0, "prime_oor": 0, "errors": 0, "already": 0, "instances": 0,
            "src_changed": 0, "hash_known": 0, "hash_same": 0, "rekeyed": 0, "opts_drift": 0,
            "fallback_opts": 0, "other_version": 0, "other_ws": 0, "torn": 0, "env_drift": [],
            "guarded": 0, "unresolved_by": {}, "first_errors": [], "slowest": [], "shard": 0,
            "local_world": 1, "barrier": "none", "t_a_s": 0.0, "t_b_s": 0.0, "t_wait_s": 0.0,
            "t_guard_s": 0.0, "t_total_s": 0.0, "xver": 0, "xver_from": [], "xver_dup": 0,
            "xver_bad": 0, "xver_compiled": 0}


def _header_variants(hdr: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    """The record runs' env / knobs dicts a table header carries (a merged table:
    "<field>_seen", older tables and raw record files: "<field>")."""
    vs = hdr.get(field + "_seen")
    if not isinstance(vs, list):
        vs = [hdr.get(field)]
    return [v for v in vs if isinstance(v, dict)]


def key_knob_drift(hdr: Dict[str, Any], cur: Dict[str, Any]) -> str:
    """'' when the current debug / instrumentation_mode / use_buffer_ops match
    (at least one of) the record runs' values, else the differences. These enter
    every L1 key (_KEY_KNOBS), so with a difference no row can ever be hit."""
    diffs = []
    for v in _header_variants(hdr, "knobs"):
        d = [f"{k}:{v[k]!r}->{cur.get(k)!r}" for k in _KEY_KNOBS
             if v.get(k) is not None and cur.get(k) is not None and v[k] != cur.get(k)]
        if not d:
            return ""
        diffs.append(d)
    return ",".join(diffs[0]) if diffs else ""


def env_drift(hdr: Dict[str, Any], cur: Dict[str, str]) -> List[str]:
    """Names of cache-invalidating env vars (get_cache_invalidating_env_vars) that
    differ from the closest record run. They change the disk key and possibly the
    code, not the L1 key: replay compiles with the current env, like the run would
    (only hash_same drops)."""
    best: Optional[List[str]] = None
    for v in _header_variants(hdr, "env"):
        names = sorted(k for k in set(v) | set(cur) if str(v.get(k)) != str(cur.get(k)))
        if best is None or len(names) < len(best):
            best = names
    return best or []


def replay(table: str, *, phases: str = "post", rank: Optional[int] = None,
           local_rank: Optional[int] = None, local_world: Optional[int] = None,
           device: Any = None, prime: bool = True, barrier_dir: Optional[str] = None,
           barrier_token: Optional[str] = None, barrier_timeout_s: Optional[float] = None,
           world_size: Optional[int] = None, log: bool = True,
           xver: Optional[bool] = None) -> Dict[str, Any]:
    """Compile + prime the table's selected rows on the current device. Must run
    after torch.cuda.set_device (kernel caches are per device, jit.py H:738 I:707
    N:746). xver: also the translated rows of the other verified Triton version
    (None = TRITON_SPEC_REPLAY_XVER, default on). Never raises; returns the counts
    (also logged)."""
    r = _resolve_rank(rank)
    lr = _resolve_local_rank(local_rank)
    lw = _resolve_local_world(local_world)
    ws = _resolve_world_size(world_size)
    _STATE.rank = r
    phases = "all" if phases == "all" else "post"
    c = _new_counts(str(table), phases, r)
    c["local_world"] = lw
    t0 = time.perf_counter()
    token = barrier_token or launch_token()
    bdir = barrier_dir or _env("TRITON_SPEC_BARRIER_DIR") or tempfile.gettempdir()
    try:
        tmo = float(barrier_timeout_s if barrier_timeout_s is not None
                    else (_env("TRITON_SPEC_BARRIER_S") or 120.0))
    except ValueError:
        tmo = 120.0
    arrived = False

    def fail(reason: str) -> Dict[str, Any]:
        c["reason"] = reason
        c["t_total_s"] = round(time.perf_counter() - t0, 3)
        if log:
            _not_armed(f"replay: {reason}", r)
        return c

    try:
        path = _resolve_table_path(str(table))
        if path is None:
            return fail(f"table not found: {table}")
        c["table"] = path
        ok, reason, version = check_triton()
        c["triton"] = version or "?"
        if not ok:
            return fail(reason)
        try:
            hdr, rows, torn = load_table(path)
        except Exception as ex:
            return fail(f"table unreadable: {type(ex).__name__}: {ex}")
        c["table_rows"], c["torn"] = len(rows), torn
        schema = hdr.get("schema", SCHEMA)
        if schema != SCHEMA:
            return fail(f"table schema {schema!r} != {SCHEMA}")
        drift = key_knob_drift(hdr, _knob_snapshot())
        if drift:
            return fail(f"Triton settings in every kernel-cache key differ from the record run "
                        f"({drift}); no row could be hit")
        c["env_drift"] = env_drift(hdr, _cache_env())
        if xver is None:
            xver = env_replay_xver()
        if xver and _mm(version) in _XVER:
            sel, xst = select_rows_xver(rows, phases, str(version), _knob_snapshot())
            other = xst["other"]
            c["xver"], c["xver_dup"], c["xver_bad"] = xst["xver"], xst["xver_dup"], xst["xver_bad"]
            c["xver_from"] = sorted({str(x.get("xver_from")) for x in sel if x.get("xver_from")})
        else:
            sel, other = select_rows(rows, phases, version)
        c["rows"], c["other_version"] = len(sel), other
        if not sel:
            return fail(f"no rows selected (table rows={len(rows)} phases={phases} "
                        f"triton={version} rows_for_other_triton={other} xver="
                        f"{'on' if xver else 'off'} xver_bad={c['xver_bad']})")
        if ws:
            c["other_ws"] = sum(1 for x in sel if _ws_of(x) and ws not in _ws_of(x))
        if device is None:
            from triton.runtime.driver import driver

            device = driver.active.get_current_device()
        rp = _Replay(sel, hdr, device, prime, c)
        shards = lpt_shard(sel, lw)
        mine = shards[lr % lw]
        c["shard"] = len(mine)
        tg = time.perf_counter()
        try:  # globals guard: which JIT callables were hashed before the replay
            pre: Optional[Dict[int, Any]] = _snapshot_hashed()
        except Exception:
            pre = None
        c["t_guard_s"] = time.perf_counter() - tg
        _STATE.tls.replaying = True
        try:
            ta = time.perf_counter()
            for i in mine:  # phase A: own shard -> shared disk cache (+ launcher .so on 3.6)
                rp.process(i)
            c["t_a_s"] = round(time.perf_counter() - ta, 3)
            tw = time.perf_counter()
            try:
                c["barrier"] = file_barrier(bdir, token, lr, lw, tmo)
                arrived = True
            except Exception as ex:  # no barrier = duplicate compiles in phase B only
                c["barrier"] = f"error:{type(ex).__name__}"
            c["t_wait_s"] = round(time.perf_counter() - tw, 3)
            tb = time.perf_counter()
            mine_set = set(mine)
            for i in range(len(sel)):  # phase B: everything else (disk hits)
                if i not in mine_set:
                    rp.process(i)
            c["t_b_s"] = round(time.perf_counter() - tb, 3)
        finally:
            _STATE.tls.replaying = False
            tg = time.perf_counter()
            try:  # ... and which the replay hashed: kernels, helpers, constexpr callables
                if pre is not None:
                    now = _snapshot_hashed()
                    new = [(o, str(o.hash)) for i, o in now.items() if i not in pre]
                    _STATE.guarded.extend(new)
                    c["guarded"] = len(new)
                else:
                    c["guarded"] = -1
            except Exception:
                c["guarded"] = -1
            c["t_guard_s"] = round(c["t_guard_s"] + time.perf_counter() - tg, 3)
        c["armed"] = c["compiled"] + c["already"] > 0
    except Exception as ex:  # anything the per-row guards did not catch
        c["errors"] += 1
        return fail(f"unexpected {type(ex).__name__}: {ex}")
    finally:
        if not arrived and lw > 1:
            try:  # never leave the other local ranks waiting for the full timeout
                barrier_arrive(bdir, token, lr)
            except Exception:
                pass
    c["t_total_s"] = round(time.perf_counter() - t0, 3)
    first = (c["first_errors"][0] if c["first_errors"] else "-").replace(" ", "_")[:200]
    if not c["armed"]:
        c["reason"] = (f"no row compiled (rows={c['rows']} unresolved={c['unresolved']} "
                       f"errors={c['errors']} first_error={first[:120]})")
    if log:
        if c["armed"]:
            logger.info(
                "armed OK: %s replay table=%s rows=%d resolved=%d unresolved=%d compiled=%d "
                "primed=%d errors=%d in %.2fs (A %.2fs, B %.2fs) rank=%d", TAG, c["table"],
                c["rows"], c["resolved"], c["unresolved"], c["compiled"], c["primed"],
                c["errors"], c["t_total_s"], c["t_a_s"], c["t_b_s"], r)
        else:
            _not_armed(f"replay: {c['reason']}", r)
        by = ",".join(f"{k}:{v}" for k, v in sorted(c["unresolved_by"].items())) or "-"
        slow = ",".join(f"{q}:{s:.2f}s" for q, s in c["slowest"][:3]) or "-"
        logger.info(
            "%s replay detail: phases=%s table_rows=%d other_triton=%d torn=%d already=%d "
            "instances=%d src_changed=%d hash_same=%d/%d rekeyed=%d opts_drift=%d "
            "fallback_opts=%d prime_oor=%d other_ws=%d env_drift=%s guarded=%d guard_scan=%.2fs "
            "shard=%d/%d barrier=%s wait=%.2fs triton=%s xver=%d(from %s compiled=%d) "
            "xver_dup=%d xver_bad=%d slowest=%s unresolved_by=%s first_error=%s rank=%d",
            TAG, phases, c["table_rows"], c["other_version"], c["torn"], c["already"],
            c["instances"], c["src_changed"], c["hash_same"], c["hash_known"], c["rekeyed"],
            c["opts_drift"], c["fallback_opts"], c["prime_oor"], c["other_ws"],
            ",".join(c["env_drift"]) or "-", c["guarded"], c["t_guard_s"], c["shard"], lw,
            c["barrier"], c["t_wait_s"], c.get("triton", "?"), c["xver"],
            ",".join(c["xver_from"]) or "-",
            c["xver_compiled"], c["xver_dup"], c["xver_bad"], slow, by, first, r)
    return c


def replay_from_env(rank: Optional[int] = None, local_rank: Optional[int] = None,
                    local_world: Optional[int] = None, device: Any = None,
                    world_size: Optional[int] = None, resume: bool = False) -> Dict[str, Any]:
    """train_ranker entry (TRITON_SPEC_REPLAY / TRITON_SPEC_REPLAY_PHASES). Off ->
    returns {"armed": False, "reason": "off"} without importing triton or
    logging. resume=True (an MLPerf resume relaunch: no RUN_START, the clock is
    already running) -> "NOT armed: replay: resume ...": every kernel compiles at
    first use exactly as without TSPEC. Never raises."""
    try:
        table = env_replay_table()
        if not table:
            return {"armed": False, "reason": "off"}
        if resume:
            _not_armed("replay: resume relaunch (the MLPerf clock is already running, so the "
                       "table is not compiled here; kernels compile at first use as without "
                       "TSPEC)", _resolve_rank(rank))
            return {"armed": False, "reason": "resume"}
        phases, valid = env_replay_phases()
        if not valid:
            logger.warning("%s TRITON_SPEC_REPLAY_PHASES=%r is not post|all; using post rank=%d",
                           TAG, _env("TRITON_SPEC_REPLAY_PHASES"), _resolve_rank(rank))
        return replay(table, phases=phases, rank=rank, local_rank=local_rank,
                      local_world=local_world, device=device, world_size=world_size)
    except Exception as ex:  # replay() never raises; this guards the env parsing
        _not_armed(f"replay: unexpected {type(ex).__name__}: {ex}", _resolve_rank(rank))
        return {"armed": False, "reason": repr(ex)}


# =============================================================================
# CLI (stdlib only)
# =============================================================================
def _summary(path: str) -> int:
    hdr, rows, torn = load_table(path)
    by: Dict[str, List[float]] = {}
    for r in rows:
        by.setdefault(f"{r['module']}.{r['qualname']}", []).append(_num(r.get("compile_s")) or 0.0)
    for name, v in sorted(by.items(), key=lambda kv: (-sum(kv[1]), kv[0])):
        print(f"{len(v):4d} {sum(v):8.2f}s  {name}")
    n_post = sum(1 for r in rows if r.get("phase") == "post")
    print(f"total rows={len(rows)} post={n_post} pre={len(rows) - n_post} kernels={len(by)} "
          f"recorded_compile_s={sum(sum(v) for v in by.values()):.1f} torn={torn} "
          f"triton={','.join(sorted({str(r.get('triton_version')) for r in rows}))}")
    return 0


def main(argv: Sequence[str]) -> int:
    argv = list(argv)
    replace: List[str] = []
    while len(argv) >= 3 and argv[0] == "merge" and argv[1] == "--replace-version":
        replace.append(argv[2])
        del argv[1:3]
    if len(argv) >= 3 and argv[0] == "merge":
        st = merge(argv[1], argv[2:], replace)
        if replace:
            print(f"replaced: {st['replaced']} rows of Triton {','.join(replace)} from input tables",
                  file=sys.stderr)
        for msg in st["unreadable"]:
            print(f"warning: unreadable input {msg}", file=sys.stderr)
        if (st["torn"] or st["bad"] or st["unsupported"] or st["env_variants"] > 1
                or st["knob_variants"] > 1 or st["cold"] < st["rows"]):
            # cold_rows < rows: some rows were only seen as disk hits (a warm record
            # run); their replay shard weight falls back to the median cold time.
            print(f"note: torn_lines={st['torn']} bad_rows={st['bad']} "
                  f"unsupported_rows={st['unsupported']} replay_rows_dropped={st['replay']} "
                  f"env_variants={st['env_variants']} knob_variants={st['knob_variants']} "
                  f"cold_rows={st['cold']}/{st['rows']}", file=sys.stderr)
        print(f"merged rows={st['rows']} post={st['post']} pre={st['pre']} "
              f"kernels={st['kernels']} files={st['files']}")
        if not st["written"]:
            print(f"warning: {argv[1]} NOT written (no readable input or no rows); an existing "
                  f"table there is left unchanged", file=sys.stderr)
            return 1
        return 0
    if len(argv) == 2 and argv[0] == "summary":
        return _summary(argv[1])
    print(__doc__)
    print("usage: triton_spec.py merge [--replace-version V ...] OUT.jsonl IN.jsonl [IN ...] | "
          "summary TABLE.jsonl",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
