"""Label-safety tests for (119) ACTION_DEDUP (ops/action_dedup.py, DLRM_HSTU_FP8_ACTION_DEDUP).

Compliance ruling: a target (candidate) row's code is the constant 2**T, and
the code computation never reads the candidate's action payload. On yambda the
candidate's action payload `item_action_weight` carries the listen-plus label
(dlrm_v4/datasets/yambda.py: candidate_action_weight = LP_BIT if is_lp else
0), and DlrmHSTU.preprocess puts it in seq_payloads next to the UIH actions
`action_weight` (modules/dlrm_hstu.py merge loop). Proved here:

  (1) every target row's code == 2**T for random UIH payloads (T = 3, 5, 6,
      0..3 targets per sample, empty UIH); UIH rows carry exactly their bits;
  (2) changing any candidate-side payload -- item_action_weight with every LP
      label value and every action bit pattern, the other candidate keys,
      extra keys, or removing them -- leaves codes(), table_inputs(), plan()'s
      table, the dedup forward output and every backward gradient (action MLP
      parameters, action tables, all other parameters, inputs) bitwise
      unchanged (fp32 and bf16 autocast, inline plan and (65) prefill);
  (3) codes() / decline_reason() / plan() / try_plan() / take_or_plan() read
      only the encoder's own action feature key (and its watchtime key iff
      watchtime thresholds are configured): they run on a Mapping that raises
      on any other key access, membership test, enumeration or len;
  (4) the same end to end through ContextualPreprocessor.forward with the knob
      on (inline plan and the (65) prefill carrier): the dedup entry points
      (take_or_plan, action_prefill.issue) only ever see the strict mapping,
      and the whole forward never touches a candidate-side or extra key.

Payload key names come from the tree: the yambda config of
dlrm_v4/configs.get_hstu_configs (dataset from gin/yambda_5b.gin), checked
against the yambda dataset's label encoding and the DlrmHSTU merge loop.

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_label_safety_test.py
"""

import os
import re
import unittest
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm
import torch

from generative_recommenders.common import HammerKernel
from generative_recommenders.dlrm_v4.configs import get_hstu_configs
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.action_encoder import ActionEncoder
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.ops import action_dedup

ENV = action_dedup.ENV_VAR
_PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIN = os.path.join(_PKG, "dlrm_v4", "train", "gin", "yambda_5b.gin")
YAMBDA_PY = os.path.join(_PKG, "dlrm_v4", "datasets", "yambda.py")
DLRM_HSTU_PY = os.path.join(_PKG, "modules", "dlrm_hstu.py")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _gin_dataset() -> str:
    m = re.search(r'^dataset\s*=\s*"([^"]+)"', _read(GIN), re.M)
    return m.group(1) if m is not None else "yambda-5b"


DATASET: str = _gin_dataset()
CFG = get_hstu_configs(dataset=DATASET)
UIH_ACTION: str = CFG.uih_weight_feature_name  # "action_weight"
CAND_ACTION: str = CFG.candidates_weight_feature_name  # "item_action_weight" (LP label)
CAND_WATCH: str = CFG.candidates_watchtime_feature_name
MERGE: List[Tuple[str, str]] = list(CFG.merge_uih_candidate_feature_mapping)
UIH_KEYS: List[str] = [u for u, _ in MERGE]
CAND_KEYS: List[str] = [c for _, c in MERGE]
UIH_WATCH: str = [u for u, c in MERGE if c == CAND_WATCH][0]
# DlrmHSTU's additional embedding features (its own selection rule).
ADDITIONAL: List[str] = [
    u
    for u, c in MERGE
    if c in CFG.item_embedding_feature_names
    and u in CFG.user_embedding_feature_names
    and u != CFG.uih_post_id_feature_name
]
# Candidate-side payload keys DlrmHSTU.preprocess adds (plus its offsets).
CANDIDATE_SIDE: FrozenSet[str] = frozenset(CAND_KEYS + ["candidate_offsets"])
EXTRA_KEYS: Tuple[str, ...] = (
    "listen_plus_label",
    "item_action_weight_label",
    "item_target_watchtime",
    "labels",
    "item_action_weights",
)


def _lp_bit() -> int:
    src = _read(YAMBDA_PY)
    m = re.search(r"^LP_BIT\s*=\s*(\d+)", src, re.M)
    if m is None:
        raise AssertionError("LP_BIT not found in datasets/yambda.py")
    return int(m.group(1))


LP_BIT: int = _lp_bit()
LP_LABEL_VALUES: Tuple[int, ...] = (0, LP_BIT)  # candidate_action_weight = LP_BIT if is_lp else 0


class Forbidden(BaseException):
    """Raised on a forbidden payload access. A BaseException, so the dedup's
    fail-closed `except Exception` (try_plan) cannot swallow it."""


class StrictPayloads(MutableMapping):
    """seq_payloads stand-in: `allowed` None = every key except `forbidden`;
    else only `allowed`. Any other key access, membership test, .get,
    enumeration, len or copy raises Forbidden and is recorded."""

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        allowed: Optional[FrozenSet[str]] = None,
        forbidden: FrozenSet[str] = frozenset(),
        log: Optional[List[Tuple[str, str]]] = None,
        violations: Optional[List[str]] = None,
    ) -> None:
        self._d = data
        self._allowed = allowed
        self._forbidden = forbidden
        self.log: List[Tuple[str, str]] = [] if log is None else log
        self.violations: List[str] = [] if violations is None else violations

    def _check(self, op: str, key: Any) -> None:
        self.log.append((op, key))
        ok = key not in self._forbidden and (self._allowed is None or key in self._allowed)
        if not ok:
            self.violations.append(f"{op}({key!r})")
            raise Forbidden(f"payload key {key!r} touched by {op}")

    def _whole(self, op: str) -> None:
        if self._allowed is not None or self._forbidden:
            self.violations.append(op)
            raise Forbidden(f"payload mapping {op}")

    def __getitem__(self, key: str) -> torch.Tensor:
        self._check("getitem", key)
        return self._d[key]

    def __contains__(self, key: object) -> bool:
        self._check("contains", key)
        return key in self._d

    def get(self, key: str, default: Any = None) -> Any:  # pyre-ignore[14]
        self._check("get", key)
        return self._d.get(key, default)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._check("setitem", key)
        self._d[key] = value

    def __delitem__(self, key: str) -> None:
        self._check("delitem", key)
        del self._d[key]

    def __iter__(self) -> Iterator[str]:
        self._whole("iter")
        return iter(self._d)

    def __len__(self) -> int:
        self._whole("len")
        return len(self._d)

    def copy(self) -> "StrictPayloads":
        return StrictPayloads(
            dict(self._d), self._allowed, self._forbidden, self.log, self.violations
        )

    def touched(self) -> FrozenSet[str]:
        return frozenset(k for _, k in self.log)


def _own_keys(enc: torch.nn.Module) -> FrozenSet[str]:
    names = [enc._action_feature_name]
    if len(enc._watchtime_to_action_thresholds_and_weights) > 0:
        names.append(enc._watchtime_feature_name)
    return frozenset(names)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(
        {torch.bfloat16: torch.int16, torch.float32: torch.int32,
         torch.float64: torch.int64, torch.int64: torch.int64}[t.dtype]
    )


def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _model(seed: int, weights: Optional[List[int]] = None) -> ContextualPreprocessor:
    torch.manual_seed(seed)
    m = ContextualPreprocessor(
        input_embedding_dim=512,
        hidden_dim=256,
        output_embedding_dim=512,
        contextual_feature_to_max_length={},
        contextual_feature_to_min_uih_length={},
        action_embedding_dim=8,
        action_feature_name=UIH_ACTION,
        action_weights=list(CFG.action_weights) if weights is None else weights,
        additional_embedding_features=ADDITIONAL,
        action_embedding_init_std=0.1,
        is_inference=False,
        fuse_linear_norm=False,
        fuse_ln_residual=False,
    )
    m.train()
    m.set_hammer_kernel(HammerKernel.PYTORCH)
    return m


def _inputs(seed: int, B: int = 12, max_uih: int = 48, uih_zero: bool = False) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    uih = torch.randint(1, max_uih + 1, (B,), generator=g)
    if uih_zero:
        uih = torch.zeros(B, dtype=torch.int64)
    nt = torch.ones(B, dtype=torch.int64)
    T_uih = int(uih.sum())
    M = T_uih + B
    feats = {"seq_embeddings": torch.randn(M, 512, generator=g)}
    for n in ADDITIONAL:
        feats[n] = torch.randn(M, 512, generator=g)
    uih_ints: Dict[str, torch.Tensor] = {}
    for u in UIH_KEYS:
        if u not in ADDITIONAL:
            uih_ints[u] = torch.randint(0, 1 << 20, (T_uih,), generator=g)
    # Random UIH actions, bits above the T weights included.
    uih_ints[UIH_ACTION] = torch.randint(0, 1 << 12, (T_uih,), generator=g)
    cand = {c: torch.randint(0, 1 << 20, (B,), generator=g) for c in CAND_KEYS}
    lp = torch.tensor(LP_LABEL_VALUES, dtype=torch.int64)
    cand[CAND_ACTION] = lp[torch.randint(0, len(LP_LABEL_VALUES), (B,), generator=g)]
    return {
        "uih": uih, "nt": nt, "B": B, "T_uih": T_uih, "max_uih": max(int(uih.max()), 0),
        "ts": torch.randint(0, 1 << 30, (M,), generator=g), "feats": feats,
        "uih_ints": uih_ints, "cand": cand,
    }


def _payloads(
    inp: Dict[str, Any], leaves: Dict[str, torch.Tensor], variant: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    p: Dict[str, torch.Tensor] = dict(inp["uih_ints"])
    for k, v in leaves.items():
        if k != "seq_embeddings":
            p[k] = v
    if not variant.get("drop_candidate", False):
        p.update(inp["cand"])
        p.update(variant.get("cand", {}))
        p["candidate_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(inp["nt"])
    p["uih_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(inp["uih"])
    p.update(variant.get("extra", {}))
    return p


def _variants(inp: Dict[str, Any], seed: int = 11) -> List[Tuple[str, Dict[str, Any]]]:
    """Candidate-side changes; [0] is the baseline."""
    B = inp["B"]
    g = torch.Generator().manual_seed(seed)
    T = len(CFG.action_weights)
    out: List[Tuple[str, Dict[str, Any]]] = [("baseline", {})]
    for v in LP_LABEL_VALUES:  # every LP label value, on every candidate
        out.append((f"{CAND_ACTION}=={v} (all)", {"cand": {CAND_ACTION: torch.full((B,), v)}}))
    out.append((f"{CAND_ACTION} LP labels flipped", {
        "cand": {CAND_ACTION: LP_BIT - inp["cand"][CAND_ACTION]}}))
    for bits in range(1 << T):  # every candidate action bit pattern
        out.append((f"{CAND_ACTION} bits={bits}", {"cand": {CAND_ACTION: torch.full((B,), bits)}}))
    out.append((f"{CAND_ACTION} random 64-bit", {
        "cand": {CAND_ACTION: torch.randint(-(1 << 62), 1 << 62, (B,), generator=g)}}))
    out.append(("every candidate key randomized", {
        "cand": {c: torch.randint(-(1 << 40), 1 << 40, (B,), generator=g) for c in CAND_KEYS}}))
    out.append(("extra label-like keys", {
        "extra": {k: torch.randint(0, 1 << 8, (B,), generator=g) for k in EXTRA_KEYS}}))
    out.append(("candidate keys removed", {"drop_candidate": True}))
    return out


def _run(
    m: ContextualPreprocessor, inp: Dict[str, Any], variant: Dict[str, Any],
    prefill: bool, autocast: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Knob on: forward + backward of the ContextualPreprocessor. The full
    seq_payloads is a StrictPayloads forbidding candidate-side and extra keys;
    the dedup entry points (take_or_plan, action_prefill.issue) receive a
    StrictPayloads that allows only the encoder's own key(s)."""
    os.environ[ENV] = "1"
    os.environ[action_prefill.ENV] = "1" if prefill else "0"
    own = _own_keys(m._action_encoder)
    seen: Dict[str, Any] = {"strict": [], "flags": [], "codes": []}
    orig_top, orig_issue = action_dedup.take_or_plan, action_prefill.issue

    def strict_of(sp: Any) -> StrictPayloads:
        s = StrictPayloads(sp._d, allowed=own)
        seen["strict"].append(s)
        return s

    def spy_top(pre_mod, pre, seq_embeddings, seq_lengths, num_targets, seq_payloads):  # pyre-ignore
        got = orig_top(pre_mod, pre, seq_embeddings, seq_lengths, num_targets,
                       strict_of(seq_payloads))
        if got is not None:
            seen["flags"].append(got[2])
            seen["codes"].append(got[0].clone())
        return got

    def spy_issue(pre_mod, **kw):  # pyre-ignore
        kw["seq_payloads"] = strict_of(kw["seq_payloads"])
        return orig_issue(pre_mod, **kw)

    for p in m.parameters():
        p.grad = None
    leaves = {k: v.detach().clone().requires_grad_(True) for k, v in inp["feats"].items()}
    sp = StrictPayloads(
        _payloads(inp, leaves, variant), forbidden=CANDIDATE_SIDE | frozenset(EXTRA_KEYS)
    )
    uih, nt = inp["uih"], inp["nt"]
    fw0 = action_dedup.stats["forwards"]
    action_dedup.take_or_plan = spy_top
    action_prefill.issue = spy_issue
    try:
        with torch.autocast("cpu", dtype=autocast or torch.bfloat16, enabled=autocast is not None):
            if prefill:
                m.prefill_action(
                    max_uih_len=inp["max_uih"], max_targets=1, total_targets=inp["B"],
                    uih_seq_lengths=uih, num_targets=nt, seq_payloads=sp,
                )
            out = m(
                max_uih_len=inp["max_uih"], max_targets=1, total_uih_len=inp["T_uih"],
                total_targets=inp["B"], seq_lengths=uih + nt, seq_timestamps=inp["ts"],
                seq_embeddings=leaves["seq_embeddings"], num_targets=nt, seq_payloads=sp,
            )
        g = torch.Generator().manual_seed(7)
        w = torch.randn(tuple(out[6].shape), generator=g).to(torch.float64)
        (out[6].to(torch.float64) * w).sum().backward()
    finally:
        action_dedup.take_or_plan = orig_top
        action_prefill.issue = orig_issue
        os.environ.pop(ENV, None)
        os.environ.pop(action_prefill.ENV, None)
    res: Dict[str, Any] = {"out": out[6].detach()}
    for n, p in m.named_parameters():
        res[f"param:{n}"] = None if p.grad is None else p.grad.detach().clone()
    for k, v in leaves.items():
        res[f"input:{k}"] = None if v.grad is None else v.grad.detach().clone()
    res["_forwards"] = action_dedup.stats["forwards"] - fw0
    res["_seen"] = seen
    res["_outer"] = sp
    return res


class ActionDedupLabelSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop(ENV, None)
        os.environ.pop(action_prefill.ENV, None)
        action_dedup._latched = None
        action_prefill._pending = None

    def tearDown(self) -> None:
        self.assertIsNone(action_dedup._latched, "the dedup latched off during the test")

    # ------------------------------------------------------------ key names
    def test_real_payload_key_names(self) -> None:
        """The keys exercised are the tree's: the yambda config's UIH action key
        and candidate action key, the dataset's LP label encoding, the merge
        loop that puts both into seq_payloads, and the model's encoder key."""
        self.assertEqual(DATASET, "yambda-5b")
        self.assertEqual(UIH_ACTION, "action_weight")
        self.assertEqual(CAND_ACTION, "item_action_weight")
        self.assertIn((UIH_ACTION, CAND_ACTION), MERGE)
        self.assertEqual(list(CFG.action_weights), [1, 2, 4])
        self.assertEqual(LP_BIT, CFG.action_weights[0])
        src = _read(YAMBDA_PY)
        self.assertRegex(src, r"candidate_action_weight\s*=\s*LP_BIT if is_lp else 0")
        hstu = _read(DLRM_HSTU_PY)
        self.assertIn("payload_features[uih_feature_name] = values_left", hstu)
        self.assertIn("payload_features[candidate_feature_name] = values_right", hstu)
        self.assertRegex(
            hstu, r"action_feature_name=self\._hstu_configs\.uih_weight_feature_name"
        )
        m = _model(0)
        self.assertEqual(m._action_encoder._action_feature_name, UIH_ACTION)
        self.assertEqual(len(m._action_encoder._watchtime_to_action_thresholds_and_weights), 0)
        self.assertEqual(_own_keys(m._action_encoder), frozenset([UIH_ACTION]))
        self.assertTrue(set(ADDITIONAL) and not set(ADDITIONAL) & CANDIDATE_SIDE)

    def test_strict_mapping_guard_works(self) -> None:
        """Negative control: the guard catches a label read, a membership
        test, .get, enumeration and len (so (3) / (4) cannot pass vacuously)."""
        inp = _inputs(1, B=4, max_uih=5)
        d = _payloads(inp, {}, {})
        s = StrictPayloads(d, allowed=frozenset([UIH_ACTION]))
        self.assertTrue(torch.equal(s[UIH_ACTION], d[UIH_ACTION]))
        for f in (
            lambda: s[CAND_ACTION], lambda: CAND_ACTION in s, lambda: s.get(CAND_ACTION),
            lambda: s[UIH_WATCH], lambda: list(s), lambda: len(s), lambda: dict(s),
            lambda: s.items(), lambda: s.copy()[CAND_ACTION],
        ):
            with self.assertRaises(Forbidden):
                f()
                list(s.items())
        self.assertGreaterEqual(len(s.violations), 8)
        with self.assertRaises(Forbidden):  # try_plan's `except Exception` cannot hide it
            try:
                s[CAND_ACTION]
            except Exception:
                pass

    # ------------------------------------------------------------ (1)
    def test_target_codes_are_constant_2_pow_T(self) -> None:
        """(1) Every target row's code == 2**T, for random UIH payloads; UIH rows
        carry exactly their own bits (an independent reference); codes() runs
        on the strict mapping (own key only) with the LP label present."""
        for weights in ([1, 2, 4], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32]):
            T = len(weights)
            enc = ActionEncoder(action_embedding_dim=8, action_feature_name=UIH_ACTION,
                                action_weights=weights)
            for seed in range(6):
                g = torch.Generator().manual_seed(100 * T + seed)
                B = 37
                uih = torch.randint(0, 50, (B,), generator=g)
                if seed == 0:
                    uih = torch.zeros(B, dtype=torch.int64)  # empty UIH
                nt = torch.ones(B, dtype=torch.int64) if seed % 2 else torch.randint(
                    0, 4, (B,), generator=g)
                n_uih, n_tgt = int(uih.sum()), int(nt.sum())
                acts = torch.randint(0, 1 << 12, (n_uih,), generator=g)
                lp = torch.tensor(LP_LABEL_VALUES)[torch.randint(0, 2, (n_tgt,), generator=g)]
                d = {UIH_ACTION: acts, CAND_ACTION: lp, UIH_WATCH: torch.randint(
                    0, 1 << 20, (n_uih,), generator=g), CAND_WATCH: torch.randint(
                    0, 1 << 20, (n_tgt,), generator=g)}
                s = StrictPayloads(d, allowed=frozenset([UIH_ACTION]))
                code = action_dedup.codes(enc, s, uih + nt, nt, n_uih + n_tgt)
                self.assertEqual(s.violations, [])
                self.assertEqual(s.touched(), frozenset([UIH_ACTION]))
                # Independent reference layout: per sample, its UIH rows, then its targets.
                ref: List[int] = []
                pos = 0
                for b in range(B):
                    for a in acts[pos:pos + int(uih[b])].tolist():
                        ref.append(sum(1 << j for j, w in enumerate(weights) if a & w))
                    pos += int(uih[b])
                    ref.extend([1 << T] * int(nt[b]))
                self.assertEqual(code.dtype, torch.int64)
                self.assertEqual(code.tolist(), ref, f"T={T} seed={seed}")
                is_tgt = torch.tensor([
                    x for b in range(B) for x in ([False] * int(uih[b]) + [True] * int(nt[b]))
                ], dtype=torch.bool)
                self.assertTrue(bool((code[is_tgt] == (1 << T)).all()))
                self.assertTrue(bool((code[~is_tgt] < (1 << T)).all()))
                self.assertEqual(int((code == (1 << T)).sum()), n_tgt)

    # ------------------------------------------------------------ (2) + (3) direct
    def test_codes_table_plan_ignore_candidate_payloads(self) -> None:
        """(2)+(3) codes(), table_inputs(), decline_reason(), plan(), try_plan()
        and take_or_plan() on the strict mapping (own key only): bitwise the
        same for every candidate-side variant, and only the own key touched."""
        m = _model(3)
        inp = _inputs(4)
        enc = m._action_encoder
        own = _own_keys(enc)
        uih, nt = inp["uih"], inp["nt"]
        N = inp["T_uih"] + inp["B"]
        emb = inp["feats"]["seq_embeddings"]
        os.environ[ENV] = "1"
        try:
            base = None
            for name, var in _variants(inp):
                d = _payloads(inp, {}, var)
                s = StrictPayloads(d, allowed=own)
                self.assertIsNone(action_dedup.decline_reason(m, emb, s), name)
                code = action_dedup.codes(enc, s, uih + nt, nt, N)
                tab = action_dedup.table_inputs(enc).detach()
                with torch.no_grad():
                    pc, atab2 = action_dedup.plan(m, s, uih + nt, nt, N)
                    tp = action_dedup.try_plan(m, s, uih + nt, nt, N)
                    top = action_dedup.take_or_plan(m, None, emb, uih + nt, nt, s)
                self.assertIsNotNone(tp, name)
                self.assertIsNotNone(top, name)
                self.assertEqual(s.violations, [], name)
                self.assertEqual(s.touched(), own, name)
                cur = (code, tab, pc, atab2, tp[0], tp[1], top[0], top[1])
                self.assertTrue(bool((code[-1:] == (1 << len(CFG.action_weights))).all()))
                if base is None:
                    base = cur
                    continue
                for i, (a, b) in enumerate(zip(base, cur)):
                    self.assertTrue(_same(a, b), f"{name}: item {i} differs")
        finally:
            os.environ.pop(ENV, None)

    def test_watchtime_key_read_iff_configured(self) -> None:
        """(3) With watchtime thresholds, codes() / decline_reason() read the
        action key and the encoder's own (UIH) watchtime key, never the
        candidate's; without thresholds the watchtime key is never touched."""
        g = torch.Generator().manual_seed(9)
        B = 21
        uih = torch.randint(0, 40, (B,), generator=g)
        nt = torch.ones(B, dtype=torch.int64)
        n_uih = int(uih.sum())
        d = {
            UIH_ACTION: torch.randint(0, 8, (n_uih,), generator=g),
            UIH_WATCH: torch.randint(0, 100, (n_uih,), generator=g),
            CAND_ACTION: torch.full((B,), LP_BIT),
            CAND_WATCH: torch.randint(0, 100, (B,), generator=g),
        }
        for thresholds in ([], [(30, 8)], [(30, 8), (60, 16)]):
            enc = ActionEncoder(
                action_embedding_dim=8, action_feature_name=UIH_ACTION,
                action_weights=[1, 2, 4], watchtime_feature_name=UIH_WATCH,
                watchtime_to_action_thresholds_and_weights=thresholds,
            )
            holder = torch.nn.Module()
            holder._action_encoder = enc
            holder._action_embedding_mlp = torch.nn.Identity()
            own = _own_keys(enc)
            self.assertEqual(UIH_WATCH in own, len(thresholds) > 0)
            T = 3 + len(thresholds)
            base = None
            for lp in LP_LABEL_VALUES + (7, 1 << 20):
                for cw in (0, 1 << 30):
                    dd = dict(d)
                    dd[CAND_ACTION] = torch.full((B,), lp)
                    dd[CAND_WATCH] = torch.full((B,), cw)
                    s = StrictPayloads(dd, allowed=own)
                    self.assertIsNone(action_dedup.decline_reason(holder, dd[UIH_ACTION], s))
                    code = action_dedup.codes(enc, s, uih + nt, nt, n_uih + B)
                    self.assertEqual(s.violations, [])
                    self.assertEqual(s.touched(), own)
                    self.assertEqual(int((code == (1 << T)).sum()), B)
                    if base is None:
                        base = code
                    self.assertTrue(_same(base, code))

    # ------------------------------------------------------------ (2) + (4) end to end
    def _end_to_end(self, prefill: bool, autocast: Optional[torch.dtype], names: Optional[List[str]] = None) -> None:
        m = _model(5)
        inp = _inputs(6)
        T = len(CFG.action_weights)
        own = _own_keys(m._action_encoder)
        # Target rows of the merged layout (per sample: UIH rows, then its 1 target).
        tgt = torch.ops.fbgemm.asynchronous_complete_cumsum(inp["uih"] + inp["nt"])[1:] - 1
        base = None
        base_touched = None
        for name, var in _variants(inp):
            if names is not None and name not in names and name != "baseline":
                continue
            r = _run(m, inp, var, prefill=prefill, autocast=autocast)
            tag = f"prefill={prefill} autocast={autocast} {name}"
            outer = r["_outer"]
            self.assertEqual(outer.violations, [], tag)
            self.assertFalse(outer.touched() & (CANDIDATE_SIDE | frozenset(EXTRA_KEYS)), tag)
            self.assertEqual(r["_forwards"], 1, f"{tag}: dedup forward not taken")
            seen = r["_seen"]
            self.assertEqual(seen["flags"], ["yes" if prefill else "no"], tag)
            self.assertTrue(len(seen["strict"]) >= 1, tag)
            for s in seen["strict"]:
                self.assertEqual(s.violations, [], tag)
                self.assertTrue(s.touched() <= own, f"{tag}: {sorted(s.touched())}")
            code = seen["codes"][0]
            self.assertTrue(bool((code.index_select(0, tgt) == (1 << T)).all()), tag)
            self.assertEqual(int((code == (1 << T)).sum()), inp["B"], tag)
            if base is None:
                base, base_touched = r, outer.touched()
                for k, v in r.items():
                    if k.startswith("param:_action"):
                        self.assertIsNotNone(v, f"{tag}: no grad for {k}")
                continue
            self.assertEqual(outer.touched(), base_touched, tag)
            keys = [k for k in base if not k.startswith("_")]
            self.assertEqual(sorted(keys), sorted(k for k in r if not k.startswith("_")))
            for k in keys:
                self.assertTrue(_same(base[k], r[k]), f"{tag}: {k} differs")

    def test_end_to_end_inline_plan_fp32(self) -> None:
        """(2)+(4) ContextualPreprocessor.forward/backward, knob on, inline
        plan (no prefill), fp32: every variant bitwise identical."""
        self._end_to_end(prefill=False, autocast=None)

    def test_end_to_end_prefill_fp32(self) -> None:
        """(2)+(4) Same through the (65) prefill carrier (plan issued by
        action_prefill.issue, consumed by take_or_plan)."""
        self._end_to_end(prefill=True, autocast=None)

    def test_end_to_end_bf16_autocast(self) -> None:
        """(2)+(4) bf16 autocast (the training dtype), LP label variants and
        the randomized / removed candidate keys, inline and prefill."""
        names = [n for n, _ in _variants(_inputs(6)) if "bits=" not in n]
        for prefill in (False, True):
            self._end_to_end(prefill=prefill, autocast=torch.bfloat16, names=names)


if __name__ == "__main__":
    unittest.main()
