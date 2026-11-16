"""R15 A11 consumer hook, contract v1.2: the C8 attention backward takes the
e4m3 dO and E8 block scales that its d(attn) producer already emitted.

Arbor R15 iterations 46-47 (component A, item A11). Producer: r16's L11
(`l11_outproj_lnmd_bwd`, 'plus' mode, r16 iteration 35 do8). Imports torch and
the stdlib only; nothing here launches a kernel, syncs the device or
allocates device memory.

WHAT IT REPLACES. On the C8 path (DLRM_HSTU_FP8_BWD_FULL, e4m3 q and k, k128
qk MFMA, fused dq16=bf16p, do_blk shape) `flydsl_hstu_attention_bwd` runs
    dO8, E8 = hstu_do_fp8.quantize_do_e4m3_blk(dout, seq_offsets, N)
    (dO8 [T, H, D] float8_e4m3fn, E8 [T, H] uint8 = 127 - e_g,
     groups G(p) = (p // 64, (p % 16) // 8) on sequence-relative p)
which re-reads the whole bf16 dO of each ordinary layer. L11 'plus' holds
every bf16 d(attn) row in registers, still writes the bf16 dx, and also
emits dO8/E8 byte-identical to quantize_do_e4m3_blk(dx) (r16 bitwise gate).
This module carries those bytes from the producer to the consumer. It never
computes, converts or decodes: a record is taken whole, or the quantizer runs
exactly as before.

OBJECTS.
  Do8Site    one per ordinary-layer TRAINING attention forward call (its
             activation input x needs a gradient; eval / no_grad forwards get
             none) whose backward will take the C8 do_blk dO quantization
             (decided in the forward with the backward's own predicates,
             hstu_flydsl_mha.c8_do_blk_planned). Holds the seq_offsets tensor
             that backward uses, N, T, H, D. Tagged on the attention output
             `out` (HSTU_DO8_SITE_ATTR) and kept in the attention autograd
             ctx; `carry_site` copies the tag to the [T, H*D] view that
             HSTUComputeOutputFunction actually receives (ops/hstu_compute.py
             views `out` before handing it over).
  Do8Record  the producer's (dO8, E8) for one backward of one site. Tagged on
             the exact dx object the producer returns to autograd
             (HSTU_DO8_RECORD_ATTR) and on the site. Autograd's view backward
             hands the attention backward a NEW [T, H, D] view of that dx, so
             the attention backward `bind`s the site's pending record to the
             dO object it received; take() then checks that this dO really is
             dx or an autograd view of it.

KNOB (since v1.1). DLRM_HSTU_FP8_BWD_DO8, read per call, default OFF. Gin
binding apply_env_bootstrap.DLRM_HSTU_FP8_BWD_DO8 (dlrm_v4/train/
_env_bootstrap.py, whose _bind_bool writes "1" for True and "0" for False); a
pre-set environment variable wins over it, so DLRM_HSTU_FP8_BWD_DO8=0 in the
environment is the kill switch of a gin-armed run. Only the exact value "1"
(surrounding whitespace ignored) arms, as for the sibling DLRM_HSTU_FP8_*
knobs compared with == "1" (B1a DU_E5M2, DQ_E5M2, E1 DKDV_E5M2, DOBLK_WL);
"true", "on" or "yes" do NOT arm it (fail closed, one line):
  "1"              the ordinary-layer attention forward creates sites on
                   training calls whose backward plans the C8 do_blk dO
                   quantization, and that backward may take a published
                   record;
  absent, "", "0"  off: no sites, no takes. A producer that follows the
                   contract (emit only for a site, and only when
                   take_enabled()) then never calls publish(), so nothing is
                   printed or counted and a tree that carries this hook
                   prints, launches and computes exactly what it did before.
                   A publish() call made anyway is refused (kill_switch, or
                   site for a None site), counted and printed;
  anything else    off, with one line.
One knob arms the pair: a producer emits only for a site (and take_enabled()).
The v1 switch DLRM_HSTU_FP8_BWD_DO8_TAKE is retired. It can no longer turn
anything on, but any value other than "" or "1" still forces off (a v1 A/B
control stays a control), with one line.

PRODUCER API (r16):  site_of(attn) in the forward; in the backward, emit only
  when a site is present and take_enabled(), and call publish(site, dx, do8,
  e8) after the LAST write to dx, on the stream the consumer will run on (the
  current stream). dx is the [T, H*D] tensor returned as d(attn) (attn's
  shape); do8/e8 are allocated, written and published with that same stream
  current (the record frees them after the consumer kernel is enqueued on it).
CONSUMER API (R15):  take(dout, seq_offsets, N) right before the quantizer;
  discard(dout_or_site, reason) on every path that will not take.

FAIL-CLOSED. take() returns None -- and the caller quantizes as today -- unless
every check holds: a pending (unconsumed) record is on dout; the knob is on;
the producer's dx is still alive; with root = dx if dx._base is None else
dx._base, dout is dx itself, or dout._base is root with dx's data_ptr and
numel (only such an autograd view shares dx's version counter; an alias with
its own counter -- x.data, t.set_(storage), a dtype view, a gradient hook
that returns g.data -- declines even with dx's data_ptr, numel, shape and
dtype); dx's and dout's _version equal the version recorded at publish;
dtypes bf16 / float8_e4m3fn / uint8; dout is [T, H, D] of the site, dO8 has
dout's numel and is contiguous, E8 is a contiguous (T, H), both 16-byte
aligned; one device, and on GPU the current stream is the publish stream;
seq_offsets is the producer's tensor (data_ptr, _version, numel, device) and
N matches. A declined record is released at once (its T*H*D-byte dO8 is not
pinned), and so is one discarded by a path that will not take. An UNTRACKED
write to dx after publish (a raw kernel, or g.data.mul_() in a hook that
returns None) is invisible to the host and breaks the contract.

ONE-SHOT. The first take consumes the record. Any later take of the same dO --
the R15 M-E1 exact re-run of the attention backward, which re-runs it with
identical arguments and no sinks -- declines ("consumed") and quantizes the
same bf16 dO, i.e. the same bytes again.

'ONLY' MODE (contract v2, R15 iteration 52; bf16_written=False). The
producer may skip the bf16 d(attn) store (autograd still gets a placeholder
of the right shape) ONLY for a call for which plan(site, attn) returned
"only" BEFORE its launch; any other bf16_written=False publish raises
ValueError (refused{only}), as in v1.2. plan() answers "only" only when the
producer knob DLRM_HSTU_FP8_L11_DO8_ONLY is "1" (default off), the consumer
knob is on, the forward stored its C8 plan inputs on the site
(hstu_flydsl_mha.c8_do_blk_planned, under the same knob) and re-evaluating
them now still selects the C8 do_blk branch (FP8 calibration steps have
BF16 q/k: "plus"), no debug reader of the bf16 dO is armed (FULL_STATS,
DUMP, HSTU_ATTN_BWD_CHECK), and nothing in autograd can read the d(attn)
gradient (grad mode, anomaly mode, tensor hooks or retain_grad on attn or
out, attn not the carried view of out); else "plus" (one line per reason).
An 'only' record (bf16_valid False) is never declined: every failed check
RAISES Do8OnlyError before any kernel reads the dO (take, discard on a path
that will read the bf16 dO, a planned call without a publish at bind, the
attention backward's C8-branch guard in hstu_flydsl_mha). Its first take
moves it to "retained": dO8/E8 stay referenced and every later take of the
SAME dO in the same layer call (the M-E1 exact re-runs) re-checks identity,
version, seq_offsets, N, stream and returns the identical tensors; release()
(the attention Function's backward, finally) ends the layer call. No
cross-call or cross-step reuse: a released record raises. 'plus' records keep
every v1.2 behaviour (one-shot, declines, lines).

PLAN LIFECYCLE (registry v2.2 and v2.3, R15 iteration 55; the rest of
contract v2 is unchanged). Each "only" answer of plan() opens one plan on its
site and records the autograd graph task (backward pass) that asked, -1 for
a call outside autograd. Exactly one event closes it: an 'only' publish
(published); cancel() (cancelled: the producer wrote the bf16 d(attn) after
all, and only then may it publish 'plus'); a later plan() of the same site
from ANOTHER backward pass or from outside autograd, or (v2.3) a 'plus'
publish for it from another pass (replanned, one line: the registry ASSUMES
that the pass that opened it ended without publish or cancel before its
consumer ran, so no reader got its placeholder -- not checked, see TOMBSTONE;
a direct call has no consumer); or release() at the end of the
layer call (abandoned: no publish and no cancel, and the attention backward
raised before reading its dO, at bind() or earlier). Hence planned ==
published + cancelled + replanned + abandoned + the plans still open (in
flight, or left by a backward that raised between its plan and its
consumer). These RAISE Do8OnlyError and publish nothing (an open plan stays
open, so bind() raises as well; a pending record stays pending):
  - plan() of a site whose plan from the SAME backward pass is still open
    (the producer planned twice for one d(attn));
  - a 'plus' publish (bf16_written=True) while a plan of the same pass is
    open (refused{plus_over_only}; cancel() first);
  - v2.3: a 'plus' publish that would supersede a pending 'only' record
    published in the same pass (refused{plus_over_only_record}: its dx may
    be that record's unwritten placeholder);
  - v2.3: an 'only' publish on a plan that another pass opened
    (refused{only_stale_plan}: plan() must answer "only" in the pass that
    skips the store).
For the publish checks a call outside autograd counts as one more pass (-1);
plan() itself never raises outside autograd. Without the torch lookup every
call counts as outside autograd: a re-plan is then only counted, a 'plus'
publish raises over any open plan and over any pending 'only' record, and an
'only' publish is never refused as stale. replanned= and abandoned= appear in
the only line and in stats()["only"] only when nonzero, so with the 'only'
knob off (no plan ever opens) every line and snapshot is v2.1's.

TOMBSTONE (registry v2.4, R15 iteration 56; contract v2 is unchanged). The
per-pass checks above treat every OTHER graph task as a pass that has ended. A
live one breaks that: a backward nested in this pass (torch.autograd.grad from
a node pre- or post-hook) or a concurrent backward of the same graph on
another thread runs the producer again (forced 'plus': a record is pending),
and its 'plus' publish supersedes this pass's pending 'only' record. Up to
v2.3 this pass's attention backward then found no pending record of its own
and quantized the never-written placeholder silently. v2.4: bind() RAISES
Do8OnlyError (refused{only_tombstone}) before anything reads the dO when the
dO's root (dout._base, else dout: the producer's dx, which carries the record
of its publish) is tagged with an 'only' record that is not the site's record;
that dO is a placeholder nothing can serve. 'plus' records never trigger it,
so with the 'only' knob off nothing changes. Not covered: a producer dx that
is itself a view (its root carries no tag); and the SITE'S OWN 'only' record
ended by another live pass while this pass's consumer has not run (a
concurrent pass whose producer publishes nothing, or whose 'plus' publish
v2.3's check (B) refuses without the graph-task lookup: its consumer binds
this record, its take raises and its release() ends it), where bind() still
finds no pending record (unbound), as in v2.3. bind() does not raise for an
ended own record because a second backward of the site that returns the same
dx without a new publish is unbound by contract (v2). A live plan closed as
replanned needs no tombstone: that pass's 'only' publish then raises
ValueError (refused{only}), as any 'only' publish without a plan does.

LOGS (stdout, flush; per process). Creating a site only bumps the quiet
`sites` counter, and a site that reaches the attention backward without a
pending record (the producer did not publish for that backward: no producer,
or L11 declined or latched on that call) only bumps the quiet `unbound`
counter (v1.2). Every other counter increment (published, taken, declined,
discarded, refused) is an event. With the knob on, one line
at the first knob read that finds it on (the first ordinary-layer attention
forward, since the site guard reads the knob first):
    [hstu-attn] dO8 registry REQUESTED: rank=R pid=P dev=D DLRM_HSTU_FP8_BWD_DO8=1 ...
The summary line
    [hstu-attn] dO8 registry: rank=R pid=P dev=D events=E sites=S
        published=P taken=T unbound=U declined={reason:n,...}
        discarded={...} refused={...} take=on|off contract=v1.2 [final=1]
is printed at events 1, 2, 4, 8, ... (i.e. from the first publish or refusal
on), at the first occurrence of each declined/discarded/refused reason, right
after the arming line, and once at interpreter exit (final=1) when the
process published or refused a record. One arming line on the first take:
    [hstu-attn] dO8 registry ARMED: rank=R pid=P dev=D producer=... ...
eval/eval_r15_do8_arming.py audits these lines (v1.2 tool: ARMED needs
unbound=0, i.e. every planned site that reached its backward got a record).
"""

import atexit
import itertools
import os
import weakref
from typing import Dict, Optional, Tuple

import torch

HSTU_DO8_SITE_ATTR: str = "_hstu_do8_site"
HSTU_DO8_RECORD_ATTR: str = "_hstu_do8_record"
# The A11 knob: "1" = on; absent, "" or "0" = off (the default).
KNOB_ENV: str = "DLRM_HSTU_FP8_BWD_DO8"
# The retired v1 switch: it can only force off (any value but "" and "1").
TAKE_ENV: str = "DLRM_HSTU_FP8_BWD_DO8_TAKE"
CONTRACT: str = "v1.2"
LOG_PREFIX: str = "[hstu-attn] dO8 registry"
# dO8 / E8 base alignment (bytes). The quantizer's outputs are fresh caching-
# allocator blocks; a producer slice must keep 16-byte (dwordx4) alignment.
ALIGN: int = 16
# Contract v2 ('only'): the producer's knob ("1" = may skip the bf16 d(attn)
# store when plan() says "only"; absent, "" or "0" = never, the default), the
# producer's optional debug knob (fill the placeholder with NaN), the version
# string of the 'only' lines, and the attribute that carries the forward's C8
# plan inputs from hstu_flydsl_mha.c8_do_blk_planned to new_site.
ONLY_ENV: str = "DLRM_HSTU_FP8_L11_DO8_ONLY"
POISON_ENV: str = "DLRM_HSTU_FP8_L11_DO8_ONLY_POISON"
CONTRACT_ONLY: str = "v2"
ONLY_PREFIX: str = "[hstu-attn] dO8 registry only"
HSTU_DO8_PLAN_ATTR: str = "_hstu_do8_plan"

_FP8 = getattr(torch, "float8_e4m3fn", None)
_tokens = itertools.count(1)

_COUNTS: Dict[str, int] = {"sites": 0, "published": 0, "taken": 0, "unbound": 0}
_REASONS: Dict[str, Dict[str, int]] = {"declined": {}, "discarded": {}, "refused": {}}
_STATE: Dict[str, object] = {"events": 0, "dev": "", "atexit": False}
_SEEN: set = set()
_WARNED: set = set()
# v2 'only' counters (quiet; their own line, never in the v1.2 summary line).
_ONLY: Dict[str, int] = {"planned": 0, "cancelled": 0, "published": 0, "taken": 0,
                         "retaken": 0, "released": 0, "released_untaken": 0}
_FORCED: Dict[str, int] = {}
# v2.2: 'only' plans the registry closed itself (no publish, no cancel), see
# PLAN LIFECYCLE. Shown in the only line and stats()["only"] only when nonzero.
_ONLY_CLOSED: Dict[str, int] = {"replanned": 0, "abandoned": 0}


class Do8OnlyError(RuntimeError):
    """Contract v2: an 'only' dO8 record (its producer did not write the bf16
    d(attn)) cannot be served here. Raised before any kernel reads that dO:
    the bf16 values it stands for do not exist, so no fallback is exact.
    v2.2: also a producer's plan-lifecycle violation (a second plan() of a
    site in one backward pass, a 'plus' publish over an open "only" plan).
    v2.3: also a 'plus' publish over a pending 'only' record of the same
    pass, and an 'only' publish on another pass's plan (PLAN LIFECYCLE).
    v2.4: also bind() of a dO that is an 'only' placeholder its site no
    longer serves (TOMBSTONE)."""


# --------------------------------------------------------------------------
# Knob


def _knob() -> Tuple[str, Optional[str]]:
    """(state, value) of the A11 knob, without side effects. state: "on"
    (DLRM_HSTU_FP8_BWD_DO8=1), "off" (absent, "" or "0"), "invalid" (any other
    value; value = that value) or "retired" (=1, but the retired
    DLRM_HSTU_FP8_BWD_DO8_TAKE holds neither "" nor "1"; value = its value)."""
    raw = os.environ.get(KNOB_ENV)
    if raw is None:
        return "off", None
    v = raw.strip()
    if v in ("", "0"):
        return "off", raw
    if v != "1":
        return "invalid", raw
    old = os.environ.get(TAKE_ENV)
    if old is not None and old.strip() not in ("", "1"):
        return "retired", old
    return "on", raw


def take_enabled() -> bool:
    """The A11 knob, read per call (the gin env bootstrap may run after
    import): True only for DLRM_HSTU_FP8_BWD_DO8=1. The default (absent, ""
    or "0") is False and prints nothing. An invalid value, or the retired
    DLRM_HSTU_FP8_BWD_DO8_TAKE forcing off, is False with one line; the first
    True prints one REQUESTED line."""
    state, raw = _knob()
    if state == "off":
        return False
    key = ("on",) if state == "on" else (state, raw)
    if key not in _WARNED:
        _WARNED.add(key)
        if state == "on":
            _emit(
                f"{LOG_PREFIX} REQUESTED: {_who()} {KNOB_ENV}=1: the ordinary-"
                "layer attention forward creates dO8 sites on calls whose "
                "backward plans the C8 do_blk dO quantization, and that "
                "backward takes a d(attn) producer's dO8/E8 when every check "
                f"holds (contract {CONTRACT} 'plus'; summary lines start at "
                f"the first publish; kill switch {KNOB_ENV}=0)"
            )
        elif state == "invalid":
            _emit(
                f"{LOG_PREFIX}: {KNOB_ENV}={raw!r} is neither 0 nor 1 -> "
                "treated as 0 (off: no sites, no takes)"
            )
        else:
            _emit(
                f"{LOG_PREFIX}: {TAKE_ENV}={raw!r} (retired v1 kill switch) -> "
                f"off: no sites, no takes; set {KNOB_ENV}=0 or 1 instead"
            )
    return state == "on"


def sites_enabled() -> bool:
    """Whether the attention forward may create sites: the knob itself
    (take_enabled()), so a producer that keys its emission on the site emits
    nothing while the consumer is off."""
    return take_enabled()


def _only_knob() -> Tuple[str, Optional[str]]:
    """(state, value) of the producer's v2 knob, without side effects: "on"
    (exactly "1", surrounding whitespace ignored), "off" (absent, "" or "0")
    or "invalid" (any other value)."""
    raw = os.environ.get(ONLY_ENV)
    if raw is None:
        return "off", None
    v = raw.strip()
    if v in ("", "0"):
        return "off", raw
    return ("on", raw) if v == "1" else ("invalid", raw)


def only_requested() -> bool:
    """Contract v2: DLRM_HSTU_FP8_L11_DO8_ONLY == "1", read per call. The
    default (absent, "" or "0") is False and prints nothing; an invalid value
    is False with one line; the first True prints one REQUESTED line."""
    state, raw = _only_knob()
    if state == "off":
        return False
    key = ("only",) if state == "on" else ("only_invalid", raw)
    if key not in _WARNED:
        _WARNED.add(key)
        if state == "on":
            _emit(
                f"{ONLY_PREFIX} REQUESTED: {_who()} {ONLY_ENV}=1: the producer "
                "may skip the bf16 d(attn) store of a call for which plan() "
                "answers 'only'; the C8 attention backward then takes the "
                "dO8/E8 record, re-takes it in the M-E1 re-runs of the same "
                "layer call and raises on any failed check (contract "
                f"{CONTRACT_ONLY} 'only'; kill switch {ONLY_ENV}=0)"
            )
        else:
            _emit(
                f"{ONLY_PREFIX}: {ONLY_ENV}={raw!r} is neither 0 nor 1 -> "
                "treated as 0 ('plus': the bf16 d(attn) is written)"
            )
    return state == "on"


def poison_requested() -> bool:
    """PRODUCER debug knob (contract v2 evidence, default off): with
    DLRM_HSTU_FP8_L11_DO8_ONLY_POISON=1 an 'only' producer fills its
    placeholder d(attn) with NaN (bf16 0x7FC0) instead of leaving it
    uninitialized, so any hidden reader of the bf16 dO shows up as NaN."""
    return os.environ.get(POISON_ENV, "").strip() == "1"


# --------------------------------------------------------------------------
# Objects


class Do8Site:
    """Forward-to-backward handle of one ordinary-layer attention call.

    seq_offsets: the exact tensor object the backward will pass to the
    quantizer (the producer builds its 64-token tile list from it); N: the
    attention's max_seq_len (the quantizer's N); T, H, D: out's shape;
    record: the latest Do8Record published on this site (None until then).
    v2 ('only'; all None unless the forward ran with the 'only' knob on):
    facts, the forward's C8 plan inputs (hstu_flydsl_mha._Do8PlanFacts,
    re-evaluated by plan()); only_plan, the token of plan()'s pending "only"
    answer; out_ref / view_ref, weakrefs to `out` and to the carried view.
    v2.2: only_task, the autograd graph task (backward pass) whose plan()
    opened only_plan (-1: a call outside autograd; None: no plan open)."""

    __slots__ = ("token", "seq_offsets", "N", "T", "H", "D", "device", "record",
                 "facts", "only_plan", "out_ref", "view_ref", "only_task")

    def __init__(
        self,
        seq_offsets: torch.Tensor,
        N: int,
        T: int,
        H: int,
        D: int,
        device: torch.device,
    ) -> None:
        self.token = next(_tokens)
        self.seq_offsets = seq_offsets
        self.N = int(N)
        self.T = int(T)
        self.H = int(H)
        self.D = int(D)
        self.device = device
        self.record: Optional["Do8Record"] = None
        self.facts: Optional[object] = None
        self.only_plan: Optional[int] = None
        self.out_ref: Optional[weakref.ref] = None
        self.view_ref: Optional[weakref.ref] = None
        self.only_task: Optional[int] = None


class Do8Record:
    """The producer's dO8/E8 for one backward of one site (see publish).

    Holds dO8 and E8 until it is taken, declined or discarded, then drops
    both. Holds NO strong reference to dx (a weakref only): dx carries this
    record in its __dict__, and a strong back-reference would make a tensor
    reference cycle.

    v2: bf16_valid is False for an 'only' record (dx is a placeholder, the
    bf16 dO does not exist); such a record is never declined, its first take
    moves it to "retained" (dO8/E8 kept, dO8 as the [T, H, D] view every take
    returns) and release() ends it. target_rows_bf16 is the TRS4 slot (the
    bf16 dO of each sequence's target row, for a target-row split): always
    None today and read by nothing. v2.3: task is the autograd graph task
    (backward pass) of an 'only' publish (-1: outside autograd); None for a
    'plus' record."""

    __slots__ = (
        "token", "site_token", "producer", "do8", "e8", "dx_ref", "dx_ptr",
        "dx_numel", "version", "device", "stream", "offs_key", "T", "H", "D",
        "N", "state", "bf16_valid", "target_rows_bf16", "plan_token", "task",
    )

    def __init__(
        self,
        site: Do8Site,
        dx: torch.Tensor,
        do8: torch.Tensor,
        e8: torch.Tensor,
        producer: str,
        bf16_valid: bool = True,
        plan_token: Optional[int] = None,
    ) -> None:
        self.token = next(_tokens)
        self.site_token = site.token
        self.producer = str(producer)
        self.do8: Optional[torch.Tensor] = do8
        self.e8: Optional[torch.Tensor] = e8
        self.dx_ref = weakref.ref(dx)
        self.dx_ptr = dx.data_ptr()
        self.dx_numel = dx.numel()
        self.version = dx._version
        self.device = dx.device
        self.stream = _stream_key(dx.device)
        self.offs_key = _offs_key(site.seq_offsets)
        self.T, self.H, self.D, self.N = site.T, site.H, site.D, site.N
        # 'plus': pending -> consumed | declined | discarded
        # 'only' (v2): pending -> retained -> released (or pending -> released)
        self.state = "pending"
        self.bf16_valid = bool(bf16_valid)
        self.target_rows_bf16: Optional[torch.Tensor] = None
        self.plan_token = plan_token
        self.task: Optional[int] = None   # v2.3: set by _publish_only


def _stream_key(device: torch.device) -> Optional[int]:
    """Handle of the current stream on a GPU device; None elsewhere. A host
    query only (no sync). Tests replace it to emulate a stream change."""
    if device.type != "cuda":
        return None
    return int(torch.cuda.current_stream(device).cuda_stream)


def _offs_key(t: torch.Tensor) -> Tuple[int, int, int, str]:
    return (t.data_ptr(), t._version, t.numel(), str(t.device))


def _graph_task() -> int:
    """v2.2: id of the autograd backward pass (graph task) this thread is
    running, -1 outside any backward or when this torch has no
    torch._C._current_graph_task_id (torch.utils.checkpoint's own lookup).
    A host query only; read by plan() with the 'only' knob on, never else."""
    fn = getattr(torch._C, "_current_graph_task_id", None)
    if fn is None:
        return -1
    try:
        return int(fn())
    except Exception:  # noqa: BLE001 - a lookup never breaks a backward
        return -1


# --------------------------------------------------------------------------
# Forward side (consumer's attention forward, producer's forward)


def new_site(
    out: torch.Tensor, seq_offsets: torch.Tensor, N: int
) -> Optional[Do8Site]:
    """Create the site of one attention forward call and tag `out` with it.

    The caller (the ordinary-layer attention Function's forward) calls this
    only when sites_enabled(), ctx.needs_input_grad[0] (a training forward)
    and hstu_flydsl_mha.c8_do_blk_planned(...) hold, and keeps the returned
    site in its ctx. None (no site) for a malformed call: then no producer
    can find a site and nothing changes. Counted in the quiet `sites`
    counter only: a site is no event and prints nothing."""
    if (
        not isinstance(out, torch.Tensor)
        or out.dim() != 3
        or not isinstance(seq_offsets, torch.Tensor)
        or seq_offsets.dim() != 1
        or seq_offsets.numel() < 1
        or seq_offsets.device != out.device
    ):
        return None
    T, H, D = (int(s) for s in out.shape)
    site = Do8Site(seq_offsets, int(N), T, H, D, out.device)
    setattr(out, HSTU_DO8_SITE_ATTR, site)
    # v2: the C8 plan inputs c8_do_blk_planned stored on `out` (only with
    # the 'only' knob on; absent otherwise, and then nothing below runs).
    facts = out.__dict__.pop(HSTU_DO8_PLAN_ATTR, None)
    if facts is not None:
        site.facts = facts
        site.out_ref = weakref.ref(out)
    _COUNTS["sites"] += 1
    if not _STATE["dev"]:
        _STATE["dev"] = str(out.device)
    return site


def stash_plan(out: torch.Tensor, facts: object) -> None:
    """v2, CONSUMER (hstu_flydsl_mha.c8_do_blk_planned, 'only' knob on): keep
    the forward's C8 plan inputs on `out` for new_site (same forward call).
    `facts.reason()` re-evaluates the C8 decision: None = C8 do_blk, else why
    not."""
    if isinstance(out, torch.Tensor) and facts is not None:
        setattr(out, HSTU_DO8_PLAN_ATTR, facts)


def carry_site(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return `dst` (a view of the attention output `src`) carrying src's
    site tag, if any. For the caller that views `out` before handing it to
    HSTUComputeOutputFunction. Anything that is not a plain tensor pair (an
    fx Proxy during tracing) passes through untouched."""
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        site = getattr(src, HSTU_DO8_SITE_ATTR, None)
        if isinstance(site, Do8Site):
            setattr(dst, HSTU_DO8_SITE_ATTR, site)
            if site.facts is not None:      # v2: plan() checks this view
                site.view_ref = weakref.ref(dst)
    return dst


def site_of(attn: torch.Tensor) -> Optional[Do8Site]:
    """PRODUCER: the site tagged on the `attn` argument of
    HSTUComputeOutputFunction.forward, or None (no emission for this call:
    the targets-only layer, a non-C8 call, the knob off, or any other
    composition). Call it in the forward and keep the result in ctx."""
    if not isinstance(attn, torch.Tensor):
        return None
    site = getattr(attn, HSTU_DO8_SITE_ATTR, None)
    if not isinstance(site, Do8Site):
        return None
    if attn.numel() != site.T * site.H * site.D or attn.device != site.device:
        return None
    return site


# --------------------------------------------------------------------------
# Producer backward


def _autograd_reason(site: Do8Site, attn: Optional[torch.Tensor]) -> Optional[str]:
    """v2: why something other than the attention backward could read (or
    replace) the d(attn) gradient, or None. Host only. Static part [R ship13
    + B3]: the attention output `out` has ONE consumer, the [T, H*D] view
    hstu_compute.py carries the site to, and that view has ONE consumer,
    HSTUComputeOutputFunction (stu.py ordinary layer -> hstu_compute_output
    -> triton_hstu_compute_output). Dynamic part, checked here: grad mode
    (create_graph: the placeholder would enter a graph), anomaly mode (the
    engine reads every gradient for NaN), retain_grad or tensor hooks on the
    view or on out, the producer's attn not being the carried view, and the
    view not being a direct view of out in the graph. Not visible to any host
    check: node pre-hooks (grad_fn.register_prehook) and a second consumer
    of attn. A pre-hook that replaces or writes the gradient makes take()'s
    identity/version check raise (an 'only' record is never declined); one
    that only reads it is not caught. A second consumer that is not a d(attn)
    producer sums its gradient into the dO, and take() raises on that sum
    (identity or version). A second consumer that is itself a DO8 producer of
    the site is forced 'plus' at its plan() (the first producer's 'only'
    record is pending); up to v2.2 its 'plus' record superseded the 'only'
    one and take() DECLINED it on the summed dO (producer_freed), so the
    quantizer read the placeholder sum. v2.3 refuses that 'plus' publish
    (refused{plus_over_only_record}); if the producer swallows the error,
    take() of the 'only' record raises on the summed dO."""
    if torch.is_grad_enabled():
        return "autograd: grad mode on in the producer's backward (create_graph)"
    if torch.is_anomaly_enabled():
        return "autograd: anomaly detection reads every gradient"
    out = site.out_ref() if site.out_ref is not None else None
    view = site.view_ref() if site.view_ref is not None else None
    if out is None or view is None:
        return "autograd: the attention output or its carried view is gone"
    if attn is not None and attn is not view:
        return "autograd: the producer's attn is not the carried view of out"
    for name, t in (("attn", view), ("out", out)):
        if getattr(t, "retains_grad", False):
            return f"autograd: retain_grad on {name}"
        if getattr(t, "_backward_hooks", None):
            return f"autograd: tensor hook on {name}"
    gf, ogf = view.grad_fn, out.grad_fn
    if gf is None or ogf is None:
        return "autograd: no graph above the attention output"
    nxt = gf.next_functions
    if len(nxt) != 1 or nxt[0][0] is not ogf:
        return "autograd: attn is not a direct view of the attention output"
    return None


def plan(site: Optional[Do8Site], attn: Optional[torch.Tensor] = None) -> str:
    """v2, PRODUCER (backward, BEFORE the launch that would skip the bf16
    d(attn) store; host only: no sync, no allocation, no kernel): "only" or
    "plus" for this call of `site` (the ctx.do8_site of the emitting call).

    "only": the producer may leave the bf16 d(attn) unwritten (it still
    returns a placeholder of attn's shape and dtype to autograd) and MUST then,
    in the same backward pass (v2.3), either publish(..., bf16_written=False)
    its dO8/E8 or, if it wrote the bf16 d(attn) after all, call cancel(site)
    (after which it may publish 'plus'; v2.2: a 'plus' publish while the plan
    is open raises; v2.3: in the pass that opened it). The plan is kept on
    the site; the attention backward raises at bind() if it is still open.
    Conditions, all evaluated now: DLRM_HSTU_FP8_L11_DO8_ONLY == "1";
    take_enabled(); no pending record on the site; the forward stored its C8
    plan inputs (site.facts) and facts.reason() is None (the C8 do_blk branch
    with the environment and num_targets memo of this moment; FP8 calibration
    calls have BF16 q/k and answer "plus"; no debug reader of the bf16 dO);
    _autograd_reason() is None (attn: the producer's own attn input).
    "plus" otherwise (write the bf16 d(attn) as in v1.2): silently with the
    knob off, else counted in forced{reason} with one line per reason.
    v2.2: a plan of this site that is still open is handled first
    (_replace_open_plan): opened by this same backward pass -> Do8OnlyError
    (the only raise of plan(); the plan stays open); opened by another pass
    or outside autograd -> closed and counted (replanned), then decided
    afresh. Never raises otherwise."""
    if not isinstance(site, Do8Site) or not only_requested():
        return "plus"
    if site.only_plan is not None:    # v2.2 (was: replaced silently below)
        _replace_open_plan(site)
    try:
        why = None
        if not take_enabled():
            why = f"consumer knob {KNOB_ENV} off"
        elif site.record is not None and site.record.state == "pending":
            why = "a record is already pending on this site"
        elif site.facts is None:
            why = ("no C8 plan inputs from the forward (it ran without "
                   f"{ONLY_ENV}=1)")
        else:
            why = site.facts.reason() or _autograd_reason(site, attn)
    except Exception as error:  # noqa: BLE001 - a plan never breaks a backward
        why = f"plan error {type(error).__name__}: {str(error)[:120]}"
    if why is not None:
        _forced(why, site.device)
        return "plus"
    site.only_plan = next(_tokens)
    site.only_task = _graph_task()
    _ONLY["planned"] += 1
    _ensure_atexit()
    return "only"


def _plan_counts() -> str:
    """v2.2: the plan counters of this process (planned == published +
    cancelled + replanned + abandoned + plans still open)."""
    return (f"planned={_ONLY['planned']} published={_ONLY['published']} "
            f"cancelled={_ONLY['cancelled']} replanned={_ONLY_CLOSED['replanned']} "
            f"abandoned={_ONLY_CLOSED['abandoned']}")


def _pass_name(task: Optional[int]) -> str:
    return "a call outside autograd" if task is None or task < 0 else f"backward pass {task}"


def _replace_open_plan(site: Do8Site, by: str = "plan()") -> None:
    """v2.2, plan() of a site whose "only" plan is still open (no publish, no
    cancel since it was opened). Opened by THIS backward pass: the producer
    planned twice for one d(attn), and the first plan's placeholder may
    already be on its way to the attention backward -> Do8OnlyError; the plan
    stays open, so bind() raises too unless the producer cancels it after
    writing the bf16 d(attn). Opened by another pass (assumed, not checked:
    that pass raised between its plan and its consumer, so no reader got that
    placeholder; v2.4 TOMBSTONE for a live one) or by a call
    outside autograd -> closed and counted in replanned (one line per
    process); plan() then decides afresh, as v2.1 did silently. v2.3: a
    'plus' publish (by="a 'plus' publish") calls this only for a plan that
    another pass opened, and closes it the same way."""
    task = _graph_task()
    if task >= 0 and task == site.only_task:
        raise Do8OnlyError(
            f"hstu_do8_registry contract {CONTRACT_ONLY} (registry v2.2): "
            f"plan() for site {site.token} (T={site.T} H={site.H} D={site.D}) "
            f"while its 'only' plan {site.only_plan} from this same backward "
            f"pass (graph task {task}) is still open: the producer planned twice "
            "for one d(attn) (publish or cancel each 'only' plan before planning "
            "again). The plan stays open, so the attention backward raises at "
            "bind unless the producer cancels it after writing the bf16 d(attn) "
            f"({_plan_counts()})"
        )
    old, old_task = site.only_plan, site.only_task
    site.only_plan = None
    site.only_task = None
    _ONLY_CLOSED["replanned"] += 1
    if ("replanned",) not in _SEEN:
        _SEEN.add(("replanned",))
        _emit(
            f"{ONLY_PREFIX} REPLANNED: {_who()} site {site.token}: {by} closed "
            f"the 'only' plan {old} left open by {_pass_name(old_task)} (no "
            f"publish, no cancel; {_plan_counts()}; logged once per process)"
        )
        _emit(only_line())


def cancel(site: Optional[Do8Site]) -> bool:
    """v2, PRODUCER: forget an open "only" plan because the bf16 d(attn) WAS
    written after all (e.g. the 'only' launch did not run and a fallback
    wrote dx). True if a plan was open. v2.2: a 'plus' publish of that call
    must come after this (publish(bf16_written=True) raises while the plan
    is open)."""
    if isinstance(site, Do8Site) and site.only_plan is not None:
        site.only_plan = None
        site.only_task = None
        _ONLY["cancelled"] += 1
        return True
    return False


def _publish_check(
    site: Do8Site, dx: torch.Tensor, do8: torch.Tensor, e8: torch.Tensor
) -> Optional[str]:
    if not isinstance(site, Do8Site):
        return "site"
    if not take_enabled():
        return "kill_switch"
    if not all(isinstance(t, torch.Tensor) for t in (dx, do8, e8)):
        return "type"
    if (
        dx.dtype is not torch.bfloat16
        or _FP8 is None
        or do8.dtype is not _FP8
        or e8.dtype is not torch.uint8
    ):
        return "dtype"
    n = site.T * site.H * site.D
    # v1.2: dx is the d(attn) tensor itself, i.e. attn's [T, H*D] shape (a
    # 3-D gradient for the 2-D attn is rejected by autograd anyway).
    if (
        tuple(dx.shape) != (site.T, site.H * site.D)
        or do8.numel() != n
        or tuple(e8.shape) != (site.T, site.H)
    ):
        return "shape"
    if not (dx.is_contiguous() and do8.is_contiguous() and e8.is_contiguous()):
        return "layout"
    if do8.data_ptr() % ALIGN or e8.data_ptr() % ALIGN:
        return "align"
    if not (dx.device == do8.device == e8.device == site.device):
        return "device"
    return None


def publish(
    site: Do8Site,
    dx: torch.Tensor,
    do8: torch.Tensor,
    e8: torch.Tensor,
    producer: str = "L11",
    bf16_written: bool = True,
) -> Optional[Do8Record]:
    """PRODUCER: hand this backward's dO8/E8 to the C8 attention backward.

    dx  the bf16 d(attn) the producer returns to autograd -- the exact
        object, shape (T, H*D) == attn.shape (anything else: refused
        "shape"); contiguous; ALREADY fully written (publish records
        dx._version, and any later tracked write makes the consumer decline;
        an untracked late write breaks the contract).
    do8 float8_e4m3fn with T*H*D elements ([T, H*D] or [T, H, D]),
        contiguous, == the bytes of quantize_do_e4m3_blk(dx.view(T, H, D),
        site.seq_offsets, site.N)[0] on every row the quantizer covers, i.e.
        computed from the bf16 values stored in dx (not from wider
        intermediates).
    e8  uint8 (T, H), contiguous, == quantize_do_e4m3_blk(...)[1] likewise.
        dO8 and E8 start 16-byte aligned (fresh allocations always do).
    Allocate do8/e8, launch the emission and call this with
    torch.cuda.current_stream(dx.device) current: the record stores that
    stream, take() requires it, and the consumer's free of do8/e8 is ordered
    on it (a block allocated under another stream could be recycled there
    while the consumer kernel still reads it).

    Returns the record (tagged on dx and on the site), or None when refused
    (counted in refused{reason}; nothing is tagged, the consumer quantizes the
    bf16 dx as today).

    v2: bf16_written=False ('only': dx is the placeholder, its bf16 values
    were never written) needs the site's open plan() "only" (else ValueError,
    refused{only}, as in v1.2). The same checks then RAISE Do8OnlyError
    instead of refusing (refused{only_<reason>}), and the plan stays open, so
    the attention backward raises at bind() even if the caller swallows this
    error: the consumer can never quantize the placeholder. v2.2: a 'plus'
    publish (bf16_written=True) while the site's "only" plan is open RAISES
    Do8OnlyError (refused{plus_over_only}; nothing is published and the plan
    stays open, so bind() raises as well): dx may be that plan's unwritten
    placeholder. A producer that wrote the bf16 d(attn) after all calls
    cancel(site) first (v2.1 closed the plan here silently).
    v2.3: these checks are per backward pass (PLAN LIFECYCLE; a call outside
    autograd counts as one pass of its own). A 'plus' publish over a plan
    that ANOTHER pass opened closes it (replanned, as plan() would) and goes
    on: the registry assumes that pass ended before its consumer ran, so its
    placeholder reached no reader (not checked: if that pass is still live,
    its own 'only' publish raises ValueError, refused{only}). A 'plus'
    publish over a pending 'only' record of ANOTHER pass supersedes it (the
    same assumption; v2.4: if that pass is still live, its consumer raises
    at bind, TOMBSTONE). A 'plus' publish that would supersede a pending
    'only' record of the SAME pass RAISES Do8OnlyError
    (refused{plus_over_only_record}; nothing is published and the 'only'
    record stays pending, so the attention backward takes it or raises). An
    'only' publish on a plan that another pass opened RAISES Do8OnlyError
    (refused{only_stale_plan}; the plan stays open, so bind() raises)."""
    if not bf16_written:
        token = site.only_plan if isinstance(site, Do8Site) else None
        if token is None:
            _bump_reason("refused", "only", getattr(site, "device", None))
            raise ValueError(
                f"hstu_do8_registry: an 'only' record (bf16_written=False) "
                f"needs contract {CONTRACT_ONLY}'s plan(site) == 'only' "
                "before the launch: the HSTU_ATTN_BWD_CHECK reference, the "
                "R15 M-E1 exact re-run and the Triton fallback read the bf16 "
                f"dO; contract {CONTRACT} accepts 'plus' (bf16_written=True)."
            )
        if site.only_task != _graph_task():    # v2.3 (C): another pass's plan
            _bump_reason("refused", "only_stale_plan", getattr(dx, "device", None))
            raise Do8OnlyError(
                f"hstu_do8_registry contract {CONTRACT_ONLY} (registry v2.3): an "
                f"'only' publish (bf16_written=False, producer={producer!s}) for "
                f"site {site.token} (T={site.T} H={site.H} D={site.D}) from "
                f"{_pass_name(_graph_task())} on the 'only' plan {token} that "
                f"{_pass_name(site.only_task)} opened: plan() must answer 'only' in "
                "the pass that skips the bf16 store (its autograd and debug-reader "
                "checks hold for that pass only). Nothing is published and the "
                "plan stays open, so the attention backward raises at bind "
                f"({_plan_counts()})"
            )
        why = _publish_check(site, dx, do8, e8)
        if why is not None:
            _bump_reason("refused", "only_" + why, getattr(dx, "device", None))
            raise Do8OnlyError(
                f"hstu_do8_registry contract {CONTRACT_ONLY}: the 'only' "
                f"record fails the publish check {why!r}; its bf16 d(attn) "
                "was not written, so nothing can serve this call (the "
                "attention backward also raises at bind)"
            )
        return _publish_only(site, dx, do8, e8, producer, token)
    if isinstance(site, Do8Site) and site.only_plan is not None and site.only_task != _graph_task():
        _replace_open_plan(site, "a 'plus' publish")   # v2.3 (A): another pass's plan
    if isinstance(site, Do8Site) and site.only_plan is not None:
        _bump_reason("refused", "plus_over_only", getattr(dx, "device", None))
        raise Do8OnlyError(
            f"hstu_do8_registry contract {CONTRACT_ONLY} (registry v2.2): a "
            f"'plus' publish (bf16_written=True, producer={producer!s}) for site "
            f"{site.token} (T={site.T} H={site.H} D={site.D}) while its 'only' "
            f"plan {site.only_plan} ({_pass_name(site.only_task)}) is open: dx "
            "may be that plan's unwritten placeholder. If the bf16 d(attn) was "
            "written after all, call cancel(site) first; else publish it with "
            "bf16_written=False. Nothing is published and the plan stays open, "
            f"so the attention backward raises at bind ({_plan_counts()})"
        )
    why = _publish_check(site, dx, do8, e8)
    if why is not None:
        _bump_reason("refused", why, getattr(dx, "device", None))
        return None
    old = site.record
    if old is not None and old.state == "pending":
        if not old.bf16_valid and old.task == _graph_task():    # v2.3 (B)
            _bump_reason("refused", "plus_over_only_record", getattr(dx, "device", None))
            raise Do8OnlyError(
                f"hstu_do8_registry contract {CONTRACT_ONLY} (registry v2.3): a "
                f"'plus' publish (bf16_written=True, producer={producer!s}) for "
                f"site {site.token} (T={site.T} H={site.H} D={site.D}) would "
                f"supersede its pending 'only' record {old.token} (plan "
                f"{old.plan_token}) published by this same pass "
                f"({_pass_name(old.task)}): dx may be that record's unwritten "
                "placeholder, which the M-E1 re-run or a declined take would then "
                "quantize. Nothing is published and the 'only' record stays "
                "pending, so the attention backward takes it or raises "
                f"({_plan_counts()})"
            )
        _release(old, "discarded", "superseded")
    rec = Do8Record(site, dx, do8, e8, producer)
    setattr(dx, HSTU_DO8_RECORD_ATTR, rec)
    site.record = rec
    _bump("published", dx.device)
    return rec


def _publish_only(
    site: Do8Site,
    dx: torch.Tensor,
    do8: torch.Tensor,
    e8: torch.Tensor,
    producer: str,
    token: int,
) -> Do8Record:
    old = site.record
    if old is not None and old.state == "pending":
        _release(old, "discarded", "superseded")
    rec = Do8Record(site, dx, do8, e8, producer, bf16_valid=False, plan_token=token)
    rec.task = _graph_task()                # v2.3 (B): the pass of this publish
    setattr(dx, HSTU_DO8_RECORD_ATTR, rec)
    site.record = rec
    site.only_plan = None
    site.only_task = None
    _ONLY["published"] += 1
    _bump("published", dx.device)
    return rec


# --------------------------------------------------------------------------
# Consumer backward


def bind(site: Optional[Do8Site], dout: torch.Tensor) -> bool:
    """CONSUMER (attention Function backward, before the attention backward
    call): tag the dO object it received with the site's pending record.
    Autograd delivers a fresh view of the producer's dx (the [T, H*D] ->
    [T, H, D] view backward), so the producer's own tag is not on it; take()
    re-checks that this dO is dx or an autograd view of it.

    v1.2: a site that arrives here without a pending record (the producer did
    not publish for this backward) bumps the quiet `unbound` counter; no
    event, no line, but every summary line carries it, so the arming audit
    sees partial arming. Only training forwards create sites, and only a
    backward calls bind(), so the count is exact.

    v2: an "only" plan still open here (the producer planned 'only' and
    published nothing, so this dO may be the unwritten placeholder) raises
    Do8OnlyError before anything reads the dO.

    v2.4 (TOMBSTONE): so does a dO whose root (dout._base, else dout) is
    tagged with an 'only' record that is not the site's record
    (refused{only_tombstone}): another live backward pass superseded it, and
    its bf16 values were never written."""
    if not isinstance(site, Do8Site) or not isinstance(dout, torch.Tensor):
        return False
    if site.only_plan is not None:
        raise Do8OnlyError(
            f"hstu_do8_registry contract {CONTRACT_ONLY}: the producer "
            "planned 'only' for this call (no bf16 d(attn)) but published no "
            "record and did not cancel the plan; the attention backward "
            "cannot read its dO"
        )
    rec = site.record
    root = dout if dout._base is None else dout._base      # v2.4 (TOMBSTONE)
    tag = getattr(root, HSTU_DO8_RECORD_ATTR, None)
    if isinstance(tag, Do8Record) and not tag.bf16_valid and tag is not rec:
        _bump_reason("refused", "only_tombstone", dout.device)
        raise Do8OnlyError(
            f"hstu_do8_registry contract {CONTRACT_ONLY} (registry v2.4): the dO "
            f"of site {site.token} (T={site.T} H={site.H} D={site.D}) is the "
            f"placeholder of 'only' record {tag.token} ({tag.state}, site "
            f"{tag.site_token}, {_pass_name(tag.task)}), but the site's record "
            f"is {'none' if rec is None else f'{rec.token} ({rec.state})'}: "
            "another live backward pass (nested in this one, or concurrent on "
            "the same graph) superseded it before this consumer ran, or it "
            "belongs to another site. Its bf16 d(attn) was never written, so "
            "nothing can read this dO"
        )
    if rec is None or rec.state != "pending":
        _COUNTS["unbound"] += 1
        return False
    if getattr(dout, HSTU_DO8_RECORD_ATTR, None) is not rec:
        setattr(dout, HSTU_DO8_RECORD_ATTR, rec)
    return True


def _take_check(
    rec: Do8Record,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: Optional[int],
    state: str = "pending",
) -> Optional[str]:
    if rec.state == "consumed":
        return "consumed"
    if rec.state != state or rec.do8 is None or rec.e8 is None:
        return "released"
    if not take_enabled():
        return "kill_switch"
    prod = rec.dx_ref()
    if prod is None:
        return "producer_freed"
    # Identity: dout must be dx itself or an autograd view of dx's root
    # tensor. Only those share dx's version counter, which is what makes the
    # version clause below see every tracked write. An alias with its own
    # counter (x.data, t.set_(storage), a dtype view, a gradient hook that
    # returns g.data) declines here even when its data_ptr, numel, shape and
    # dtype -- and, by coincidence, its _version number -- match dx's.
    root = prod if prod._base is None else prod._base
    if prod.data_ptr() != rec.dx_ptr or (
        dout is not prod
        and (
            dout._base is not root
            or dout.data_ptr() != rec.dx_ptr
            or dout.numel() != rec.dx_numel
        )
    ):
        return "storage"
    if prod._version != rec.version or dout._version != rec.version:
        return "version"
    if (
        dout.dtype is not torch.bfloat16
        or rec.do8.dtype is not _FP8
        or rec.e8.dtype is not torch.uint8
    ):
        return "dtype"
    if (
        dout.dim() != 3
        or tuple(dout.shape) != (rec.T, rec.H, rec.D)
        or rec.do8.numel() != dout.numel()
        or tuple(rec.e8.shape) != (rec.T, rec.H)
    ):
        return "shape"
    if not (
        dout.is_contiguous() and rec.do8.is_contiguous() and rec.e8.is_contiguous()
    ):
        return "layout"
    if rec.do8.data_ptr() % ALIGN or rec.e8.data_ptr() % ALIGN:
        return "align"
    if not (dout.device == rec.device == rec.do8.device == rec.e8.device):
        return "device"
    if _stream_key(dout.device) != rec.stream:
        return "stream"
    if not isinstance(seq_offsets, torch.Tensor) or _offs_key(seq_offsets) != rec.offs_key:
        return "seq_offsets"
    if N is not None and int(N) != rec.N:
        return "n"
    return None


def take(
    dout: torch.Tensor, seq_offsets: torch.Tensor, N: Optional[int] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """CONSUMER (C8 branch of flydsl_hstu_attention_bwd, in place of
    quantize_do_e4m3_blk): (dO8 [T, H, D] float8_e4m3fn, E8 [T, H] uint8) of
    the record on `dout`, or None (no record, or any failed check: declined,
    counted, record released; the caller runs the quantizer). One-shot.
    v2: an 'only' record is never declined (see _take_only)."""
    rec = getattr(dout, HSTU_DO8_RECORD_ATTR, None) if isinstance(
        dout, torch.Tensor) else None
    if not isinstance(rec, Do8Record):
        return None
    if not rec.bf16_valid:
        return _take_only(rec, dout, seq_offsets, N)
    why = _take_check(rec, dout, seq_offsets, N)
    if why is not None:
        if rec.state == "pending":
            _release(rec, "declined", why)
        else:
            _bump_reason("declined", why, rec.device)
        return None
    do8, e8 = rec.do8, rec.e8
    rec.do8 = None
    rec.e8 = None
    rec.state = "consumed"
    shape = (rec.T, rec.H, rec.D)
    if tuple(do8.shape) != shape:
        do8 = do8.view(shape)
    first = _COUNTS["taken"] == 0
    _bump("taken", rec.device, quiet=first)
    if first:
        _emit(
            f"{LOG_PREFIX} ARMED: {_who()} producer={rec.producer} first take: "
            f"dO8 [T, H, D] float8_e4m3fn + E8 [T, H] uint8 handed to the C8 "
            f"attention backward instead of quantize_do_e4m3_blk (T={rec.T} "
            f"H={rec.H} D={rec.D} N={rec.N}; contract {CONTRACT} 'plus'; kill "
            f"switch {KNOB_ENV}=0)"
        )
        _emit(summary_line())
    return do8, e8


def _take_only(
    rec: Do8Record,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v2 take of an 'only' record (bf16_valid False). The first take checks
    exactly what a 'plus' take checks and moves the record to "retained";
    every later take of the same layer call (an M-E1 re-run with the same
    dO) re-runs those checks (identity, version, dtype, shape, layout,
    alignment, device, stream, seq_offsets, N) and returns the IDENTICAL
    tensors. Any failed check, the knob off, or a record released at the end
    of its layer call RAISES Do8OnlyError: no fallback exists."""
    if rec.state == "retained":
        why = _take_check(rec, dout, seq_offsets, N, state="retained")
        if why is None:
            _ONLY["retaken"] += 1
            return rec.do8, rec.e8
    elif rec.state == "pending":
        why = _take_check(rec, dout, seq_offsets, N)
    else:
        why = rec.state
    if why is not None:
        raise Do8OnlyError(
            f"hstu_do8_registry contract {CONTRACT_ONLY}: take of an 'only' "
            f"dO8 record ({rec.state}) fails: {why}. Its bf16 d(attn) was "
            "never written, so neither the quantizer nor any other backward "
            "can run on this dO"
        )
    shape = (rec.T, rec.H, rec.D)
    if tuple(rec.do8.shape) != shape:
        rec.do8 = rec.do8.view(shape)
    rec.state = "retained"
    _ONLY["taken"] += 1
    first = _COUNTS["taken"] == 0
    _bump("taken", rec.device, quiet=first)
    if first:
        _emit(
            f"{LOG_PREFIX} ARMED: {_who()} producer={rec.producer} first take: "
            f"dO8 [T, H, D] float8_e4m3fn + E8 [T, H] uint8 handed to the C8 "
            f"attention backward instead of quantize_do_e4m3_blk (T={rec.T} "
            f"H={rec.H} D={rec.D} N={rec.N}; contract {CONTRACT_ONLY} 'only': "
            f"no bf16 d(attn); kill switches {ONLY_ENV}=0, {KNOB_ENV}=0)"
        )
        _emit(summary_line())
    if _ONLY["taken"] == 1:
        _emit(
            f"{ONLY_PREFIX} ARMED: {_who()} producer={rec.producer} first "
            f"'only' take (T={rec.T} N={rec.N}): the bf16 d(attn) store is "
            "skipped; the record is retained for the M-E1 re-runs of its "
            f"layer call and released when that backward returns (contract "
            f"{CONTRACT_ONLY} 'only')"
        )
        _emit(only_line())
    return rec.do8, rec.e8


def discard(dout_or_site: object, reason: str) -> bool:
    """CONSUMER: release a pending record that this path will not take (a
    non-C8 call, a FlyDSL early return, the attention backward's safety net),
    so its dO8/E8 are not pinned. No-op (False) without a pending record.
    v2: such a path reads the bf16 dO, so an 'only' record raises
    Do8OnlyError there, pending or retained; the one exception is the
    safety net after the first call ("not_taken") on a retained record,
    which keeps it for the re-runs of the same layer call (no-op)."""
    if isinstance(dout_or_site, Do8Site):
        rec = dout_or_site.record
    elif isinstance(dout_or_site, torch.Tensor):
        rec = getattr(dout_or_site, HSTU_DO8_RECORD_ATTR, None)
    else:
        rec = None
    if isinstance(rec, Do8Record) and not rec.bf16_valid:
        if rec.state == "pending" or (rec.state == "retained" and reason != "not_taken"):
            raise Do8OnlyError(
                f"hstu_do8_registry contract {CONTRACT_ONLY}: the path "
                f"{reason!r} would read the bf16 dO of an 'only' record "
                f"({rec.state}), which was never written"
            )
        return False
    if not isinstance(rec, Do8Record) or rec.state != "pending":
        return False
    _release(rec, "discarded", str(reason))
    return True


def release(site: Optional[Do8Site]) -> bool:
    """v2, CONSUMER (the attention Function's backward, in a finally: the
    end of this layer call). Drops a retained (or never taken) 'only'
    record's dO8/E8 and marks it "released" (a later take raises), and closes
    an "only" plan left open (v2.2: counted in abandoned). Such a plan was
    neither published nor cancelled, and this layer call's attention backward
    raised before reading its dO: at bind(), or earlier (tpa's _backward_impl
    can raise before bind(), e.g. on missing FP8 scales, and reads no dO
    before it). No-op (False) otherwise, i.e. always with the 'only' knob off
    ('plus' records are one-shot and never retained)."""
    if not isinstance(site, Do8Site):
        return False
    done = site.only_plan is not None
    if done:
        _ONLY_CLOSED["abandoned"] += 1
    site.only_plan = None
    site.only_task = None
    rec = site.record
    if isinstance(rec, Do8Record) and not rec.bf16_valid and rec.state in (
            "pending", "retained"):
        _ONLY["released" if rec.state == "retained" else "released_untaken"] += 1
        rec.do8 = None
        rec.e8 = None
        rec.state = "released"
        done = True
    return done


def only_record(dout: object) -> Optional[Do8Record]:
    """v2: the 'only' record (bf16_valid False) tagged on `dout`, else None.
    One attribute read: the attention backward's C8-branch guard."""
    rec = getattr(dout, HSTU_DO8_RECORD_ATTR, None) if isinstance(
        dout, torch.Tensor) else None
    if isinstance(rec, Do8Record) and not rec.bf16_valid:
        return rec
    return None


def _release(rec: Do8Record, kind: str, reason: str) -> None:
    rec.do8 = None
    rec.e8 = None
    rec.state = kind
    _bump_reason(kind, reason, rec.device)


# --------------------------------------------------------------------------
# Counters and log lines


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - a log field must never raise
        pass
    raw = os.environ.get("RANK", "").strip()
    return int(raw) if raw.lstrip("-").isdigit() else -1


def _who() -> str:
    return f"rank={_rank()} pid={os.getpid()} dev={_STATE['dev'] or '?'}"


def _fmt(d: Dict[str, int]) -> str:
    return "{" + ",".join(f"{k}:{v}" for k, v in sorted(d.items())) + "}"


def summary_line(final: bool = False) -> str:
    """The '[hstu-attn] dO8 registry:' counter line of this process."""
    return (
        f"{LOG_PREFIX}: {_who()} events={_STATE['events']} "
        f"sites={_COUNTS['sites']} published={_COUNTS['published']} "
        f"taken={_COUNTS['taken']} unbound={_COUNTS['unbound']} "
        f"declined={_fmt(_REASONS['declined'])} "
        f"discarded={_fmt(_REASONS['discarded'])} "
        f"refused={_fmt(_REASONS['refused'])} "
        f"take={'on' if _knob()[0] == 'on' else 'off'} contract={CONTRACT}"
        + (" final=1" if final else "")
    )


def _emit(line: str) -> None:
    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001 - closed stdout at exit, etc.
        pass


def only_line(final: bool = False) -> str:
    """v2: the '[hstu-attn] dO8 registry only:' counter line of this process
    (the v1.2 summary line is unchanged: 'only' takes count in its taken=).
    v2.2: replanned= / abandoned= follow released_untaken= when nonzero."""
    return (
        f"{ONLY_PREFIX}: {_who()} "
        + " ".join(f"{k}={v}" for k, v in _ONLY.items())
        + "".join(f" {k}={v}" for k, v in _ONLY_CLOSED.items() if v)
        + f" forced_plus={_fmt(_FORCED)} only={'on' if _only_knob()[0] == 'on' else 'off'}"
        + f" contract={CONTRACT_ONLY}" + (" final=1" if final else "")
    )


def _forced(why: str, device: Optional[torch.device]) -> None:
    """v2: plan() answered "plus" with the 'only' knob on: count the reason,
    one line (+ the only line) at its first occurrence."""
    _FORCED[why] = _FORCED.get(why, 0) + 1
    if device is not None and not _STATE["dev"]:
        _STATE["dev"] = str(device)
    _ensure_atexit()
    if ("forced", why) not in _SEEN:
        _SEEN.add(("forced", why))
        _emit(f"{ONLY_PREFIX} FORCED 'plus': {_who()} {why} (the bf16 d(attn) is "
              "written; logged once per reason)")
        _emit(only_line())


def _ensure_atexit() -> None:
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_final)


def _final() -> None:
    # Only a process that published or refused a record prints the final
    # line (a site alone is quiet).
    if _COUNTS["published"] + sum(_REASONS["refused"].values()) > 0:
        _emit(summary_line(final=True))
    if _ONLY["planned"] + sum(_FORCED.values()) > 0:     # v2
        _emit(only_line(final=True))


def _event(device: Optional[torch.device], force: bool, quiet: bool) -> None:
    if device is not None:
        _STATE["dev"] = str(device)
    n = int(_STATE["events"]) + 1
    _STATE["events"] = n
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_final)
    if not quiet and (force or (n & (n - 1)) == 0):
        _emit(summary_line())


def _bump(key: str, device: Optional[torch.device], quiet: bool = False) -> None:
    _COUNTS[key] += 1
    _event(device, False, quiet)


def _bump_reason(kind: str, reason: str, device: Optional[torch.device]) -> None:
    d = _REASONS[kind]
    d[reason] = d.get(reason, 0) + 1
    new = (kind, reason) not in _SEEN
    _SEEN.add((kind, reason))
    _event(device, new, False)


def stats() -> Dict[str, object]:
    """Snapshot of this process's counters. v2.2: "only" carries replanned /
    abandoned only when nonzero (read them with .get(key, 0))."""
    only = dict(_ONLY)
    only.update((k, v) for k, v in _ONLY_CLOSED.items() if v)
    only["forced_plus"] = dict(_FORCED)
    return {
        "sites": _COUNTS["sites"],
        "published": _COUNTS["published"],
        "taken": _COUNTS["taken"],
        "unbound": _COUNTS["unbound"],
        "declined": dict(_REASONS["declined"]),
        "discarded": dict(_REASONS["discarded"]),
        "refused": dict(_REASONS["refused"]),
        "events": int(_STATE["events"]),
        "only": only,
    }


def reset_for_testing() -> None:
    """Zero the counters and the once-per-reason / REQUESTED / arming state."""
    for k in _COUNTS:
        _COUNTS[k] = 0
    for k in _ONLY:
        _ONLY[k] = 0
    for k in _ONLY_CLOSED:
        _ONLY_CLOSED[k] = 0
    _FORCED.clear()
    for d in _REASONS.values():
        d.clear()
    _STATE["events"] = 0
    _STATE["dev"] = ""
    _SEEN.clear()
    _WARNED.clear()
