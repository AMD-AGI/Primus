"""Bitwise gate for DLRM_HSTU_FP8_ACTION_PREFILL (modules/action_prefill.py).

Prefill ON vs OFF must agree BIT FOR BIT on every forward output, every
parameter gradient and every input gradient, and the backward must issue the
same ops in the same order (the prefill graph is re-attached in place by a
bridge node). A control-vs-control pair first proves the baseline itself is
deterministic, so an equality here is evidence and not luck.

  * GPU (decisive): ContextualPreprocessor at the yambda-5b production dims
    (D=512, hidden 256, action_weights [1,2,4] -> 24 wide, 8 contextual
    features, artist/album additional features, ~2.3M rows like a production
    rank), bf16 autocast, Triton kernels, F1+F2 / F1 / plain containers, pair-2
    Linear prefill on and off. Backward checked at the dispatcher level and as
    the ordered list of GPU kernel names from the profiler.
  * CPU: the same preprocessor at small dims (PyTorch kernels, so the F1
    container takes its nn.Sequential fallback) and a whole DlrmHSTU
    main_forward + backward (prefill issued by DlrmHSTU._issue_action_prefill,
    i.e. the hook DlrmHSTU.preprocess calls before touching the EC output).

Run:  python generative_recommenders/dlrm_v4/train/tests/action_prefill_test.py
With ACTION_PREFILL_PROD_GIN=<.../gin/yambda_5b.gin> the production env
bootstrap (gin apply_env_bootstrap) and the Triton autotune pins are applied
first, as train_ranker.py does. ACTION_PREFILL_ROWS overrides the GPU row
target (default 2_300_000).
That branch pre-sets DLRM_HSTU_FP8_PREPROC_L2=0 (unless already set): E-M3a's
delayed-scale sites are process-global and switch from BF16 warm-up to FP8
after DLRM_HSTU_FP8_BWD_WARMUP_CALLS training calls, so the gate's successive
arms would not be comparable call to call. E-M3a x prefill is the xtree's
check (DLRM_HSTU_FP8_BWD_WARMUP_CALLS=1).
"""

import collections
import copy
import os
import sys
import unittest
from typing import Dict, List, Optional, Tuple

if os.environ.get("ACTION_PREFILL_PROD_GIN"):
    # Phase-1 bootstrap of train_ranker.py: env bindings before heavy imports.
    import gin
    from generative_recommenders.dlrm_v4.train._env_bootstrap import (
        apply_env_bootstrap,
    )

    gin.parse_config_file(os.environ["ACTION_PREFILL_PROD_GIN"], skip_unknown=True)
    # E-M3a off for the gate (module docstring); a pre-set value wins.
    os.environ.setdefault("DLRM_HSTU_FP8_PREPROC_L2", "0")
    apply_env_bootstrap()

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm (torchrec does in training)
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from generative_recommenders.common import HammerKernel
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.preprocessors import (
    CONTEXTUAL_FOLD_KEY,
    ContextualPreprocessor,
)

ENV = action_prefill.ENV
L2_ENV = action_prefill.L2_ENV
N_CONTEXTUAL = 8  # uid + 7 yambda-5b cross features
ADDITIONAL = ["artist_id", "album_id"]
ACTION = "action_weight"


class _OpLog(TorchDispatchMode):
    """Every aten op the dispatcher sees, in order (forward or backward)."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        shapes = tuple(
            tuple(a.shape) for a in args if isinstance(a, torch.Tensor)
        )
        self.ops.append(f"{func}{shapes}")
        return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def _same_bits(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _make_preprocessor(
    D: int, hidden: int, f1: bool, f2: bool, device: torch.device, seed: int
) -> ContextualPreprocessor:
    torch.manual_seed(seed)
    ctx_keys = [f"ctx{i}" for i in range(N_CONTEXTUAL)]
    m = ContextualPreprocessor(
        input_embedding_dim=D,
        hidden_dim=hidden,
        output_embedding_dim=D,
        contextual_feature_to_max_length={k: 1 for k in ctx_keys},
        contextual_feature_to_min_uih_length={k: 0 for k in ctx_keys},
        action_embedding_dim=8,
        action_feature_name=ACTION,
        action_weights=[1, 2, 4],
        additional_embedding_features=ADDITIONAL,
        action_embedding_init_std=0.1,
        is_inference=False,
        fuse_linear_norm=f1,
        fuse_ln_residual=f2,
    ).to(device)
    m.train()
    if device.type != "cuda":
        m.set_hammer_kernel(HammerKernel.PYTORCH)
    return m


def _make_inputs(
    B: int, max_uih: int, D: int, device: torch.device, emb_dtype: torch.dtype,
    seed: int, len_dtype: torch.dtype = torch.int64,
) -> Dict[str, object]:
    # Lengths on the host generator (they fix the shapes); bulk data drawn on
    # the target device (2.3M x 512 rows would take ~30 s on the host).
    g = torch.Generator().manual_seed(seed)
    uih = torch.randint(1, max_uih + 1, (B,), generator=g).to(len_dtype)
    num_targets = torch.ones(B, dtype=len_dtype)
    T_uih = int(uih.sum())
    M = T_uih + B
    gd = torch.Generator(device=device).manual_seed(seed + 1)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gd, device=device)

    def randint(hi: int, n: int) -> torch.Tensor:
        return torch.randint(0, hi, (n,), generator=gd, device=device)

    feats: Dict[str, torch.Tensor] = {"seq_embeddings": randn(M, D).to(emb_dtype)}
    for name in ADDITIONAL:
        feats[name] = randn(M, D).to(emb_dtype)
    for i in range(N_CONTEXTUAL):
        feats[f"ctx{i}"] = randn(B, D)  # EC kept (small) keys stay fp32
    ints: Dict[str, torch.Tensor] = {
        ACTION: randint(8, T_uih),
        "item_action_weight": randint(8, B),
    }
    for i in range(N_CONTEXTUAL):
        ints[f"ctx{i}_offsets"] = torch.arange(B + 1, dtype=torch.int64, device=device)
    return {
        "uih": uih.to(device),
        "num_targets": num_targets.to(device),
        "max_uih_len": int(uih.max()),
        "max_targets": 1,
        "total_uih_len": T_uih,
        "total_targets": B,
        "timestamps": randint(1 << 30, M),
        "feats": feats,
        "ints": ints,
    }


def _run(
    m: ContextualPreprocessor,
    inp: Dict[str, object],
    prefill: bool,
    l2: bool,
    fold: bool,
    loss_w: Dict[str, torch.Tensor],
    profile_bwd: bool = False,
) -> Dict[str, object]:
    """One forward + backward; returns outputs, grads and backward op logs."""
    device = inp["uih"].device
    on_gpu = device.type == "cuda"
    os.environ[ENV] = "1" if prefill else "0"
    os.environ[L2_ENV] = "1" if l2 else "0"
    used0 = action_prefill.stats["used"]
    for p in m.parameters():
        p.grad = None
    leaves = {
        k: v.detach().clone().requires_grad_(True) for k, v in inp["feats"].items()
    }
    payloads: Dict[str, torch.Tensor] = dict(inp["ints"])
    for k, v in leaves.items():
        if k != "seq_embeddings":
            payloads[k] = v
    uih, num_targets = inp["uih"], inp["num_targets"]
    # DlrmHSTU.main_forward builds the merged lengths it hands the transducer.
    seq_lengths = uih + num_targets
    fwd_prof = None
    if profile_bwd:
        torch.cuda.synchronize()
        fwd_prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        fwd_prof.__enter__()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu):
        if prefill:
            m.prefill_action(
                max_uih_len=inp["max_uih_len"],
                max_targets=inp["max_targets"],
                total_targets=inp["total_targets"],
                uih_seq_lengths=uih,
                num_targets=num_targets,
                seq_payloads=payloads,
            )
        out = m(
            max_uih_len=inp["max_uih_len"],
            max_targets=inp["max_targets"],
            total_uih_len=inp["total_uih_len"],
            total_targets=inp["total_targets"],
            seq_lengths=seq_lengths,
            seq_timestamps=inp["timestamps"],
            seq_embeddings=leaves["seq_embeddings"],
            num_targets=num_targets,
            seq_payloads=payloads,
            fold_contextual_concat=fold,
        )
    fwd_kernels: List[str] = []
    if fwd_prof is not None:
        torch.cuda.synchronize()
        fwd_prof.__exit__(None, None, None)
        fwd_kernels = _gpu_kernels(fwd_prof)
    if action_prefill._pending is not None:
        raise AssertionError("issued prefill was not consumed")
    outs = {
        "max_seq_len": torch.tensor(out[0]),
        "total_uih_len": torch.tensor(out[1]),
        "total_targets": torch.tensor(out[2]),
        "seq_lengths": out[3],
        "seq_offsets": out[4],
        "seq_timestamps": out[5],
        "seq_embeddings": out[6],
        "num_targets": out[7],
    }
    if fold:
        outs["contextual_fold"] = out[8][CONTEXTUAL_FOLD_KEY]
    loss = None
    for k in ("seq_embeddings", "contextual_fold"):
        if k not in outs:
            continue
        o = outs[k]
        if k not in loss_w:  # first run fixes the weights; later runs reuse
            g = torch.Generator().manual_seed(7 + len(loss_w))
            loss_w[k] = torch.randn(tuple(o.shape), generator=g).to(o.device)
        term = (o.float() * loss_w[k]).sum()
        loss = term if loss is None else loss + term
    # Order in which parameter gradients land (the DDP reducer hooks here).
    acc_order: List[str] = []
    hooks = [
        p.register_post_accumulate_grad_hook(lambda _p, n=n: acc_order.append(n))
        for n, p in m.named_parameters()
    ]
    log = _OpLog()
    kernels: List[str] = []
    if profile_bwd:
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with log:
                loss.backward()
            torch.cuda.synchronize()
        kernels = _gpu_kernels(prof)
    else:
        with log:
            loss.backward()
    for h in hooks:
        h.remove()
    grads = {f"param:{n}": p.grad for n, p in m.named_parameters()}
    grads.update({f"input:{k}": v.grad for k, v in leaves.items()})
    return {
        "outs": {k: v.detach() for k, v in outs.items()},
        "grads": grads,
        "bwd_ops": log.ops,
        "bwd_kernels": kernels,
        "fwd_kernels": fwd_kernels,
        "acc_order": acc_order,
        "used": action_prefill.stats["used"] - used0,
    }


def _gpu_kernels(prof: "torch.profiler.profile") -> List[str]:
    evs = [
        e for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    evs.sort(key=lambda e: e.time_range.start)
    return [e.name for e in evs]


def _missing_or_zero_grads(res: Dict[str, object]) -> List[str]:
    """Gradients that are None or all-zero. A dropped dgrad (e.g. a shifted
    ctx.needs_input_grad index, MNC iter 11 ledger mnc-xtree-lesson) shows as
    ALL-ZERO gradients once autograd materializes them, not as None, and a
    prefill-on vs prefill-off comparison cannot see it when both arms run the
    same wrong code: require every gradient to exist and be nonzero."""
    return [
        k for k, g in res["grads"].items()
        if g is None or not bool((g.float().abs().amax() > 0).item())
    ]


def _compare(a: Dict[str, object], b: Dict[str, object]) -> List[str]:
    bad = []
    for k in a["outs"]:
        if not _same_bits(a["outs"][k], b["outs"][k]):
            bad.append(f"out:{k}")
    for k in a["grads"]:
        if not _same_bits(a["grads"][k], b["grads"][k]):
            bad.append(f"grad:{k}")
    return bad


class _Gate:
    """Shared driver: control, control again, then each prefill arm."""

    def gate(
        self, tc: unittest.TestCase, m: ContextualPreprocessor,
        inp: Dict[str, object], fold: bool, l2_arms: Tuple[bool, ...],
        profile: bool,
    ) -> Dict[str, object]:
        M = inp["feats"]["seq_embeddings"].shape[0]
        w: Dict[str, torch.Tensor] = {}
        # Warm-up: first-call autotune / TunableOp / WLN-v2 mode recording
        # happen here, so the gated runs all see the same kernel choices.
        _run(m, inp, False, False, fold, w)
        _run(m, inp, True, True, fold, w)
        ctl = _run(m, inp, False, False, fold, w, profile)
        ctl2 = _run(m, inp, False, False, fold, w, profile)
        nondet = _compare(ctl, ctl2)
        tc.assertEqual(nondet, [], f"baseline is not deterministic: {nondet}")
        zero = _missing_or_zero_grads(ctl)
        tc.assertEqual(zero, [], f"control has missing/all-zero gradients: {zero}")
        tc.assertEqual(ctl["bwd_ops"], ctl2["bwd_ops"])
        tc.assertEqual(ctl["acc_order"], ctl2["acc_order"])
        tc.assertEqual(ctl["used"], 0)
        tc.assertEqual(ctl["bwd_kernels"], ctl2["bwd_kernels"])
        report = {"rows": M, "outputs": len(ctl["outs"]), "grads": len(ctl["grads"]),
                  "bwd_ops": len(ctl["bwd_ops"]), "bwd_kernels": len(ctl["bwd_kernels"])}
        for l2 in l2_arms:
            action_prefill._logged.discard("active")
            pre = _run(m, inp, True, l2, fold, w, profile)
            bad = _compare(ctl, pre)
            tc.assertEqual(bad, [], f"prefill (l2={l2}) differs from control: {bad}")
            zero = _missing_or_zero_grads(pre)
            tc.assertEqual(zero, [], f"prefill (l2={l2}) missing/all-zero gradients: {zero}")
            tc.assertEqual(pre["used"], 1, "prefill was not consumed")
            tc.assertEqual(
                ctl["bwd_ops"], pre["bwd_ops"],
                f"backward dispatcher op order differs (l2={l2})",
            )
            tc.assertEqual(
                ctl["acc_order"], pre["acc_order"],
                f"parameter grad-ready (DDP hook) order differs (l2={l2})",
            )
            if profile and not ctl["bwd_kernels"]:
                print("[action-prefill-test] WARNING profiler returned no GPU "
                      "kernel events; kernel-order check skipped", flush=True)
            if profile and ctl["bwd_kernels"]:
                if ctl["bwd_kernels"] != pre["bwd_kernels"]:
                    diff = [
                        (i, x, y) for i, (x, y) in enumerate(
                            zip(ctl["bwd_kernels"], pre["bwd_kernels"])
                        ) if x != y
                    ][:5]
                    tc.fail(
                        f"backward GPU kernel order differs (l2={l2}): "
                        f"{len(ctl['bwd_kernels'])} vs {len(pre['bwd_kernels'])}, "
                        f"first diffs {diff}"
                    )
            extra = collections.Counter(pre["fwd_kernels"]) - collections.Counter(
                ctl["fwd_kernels"])
            missing = collections.Counter(ctl["fwd_kernels"]) - collections.Counter(
                pre["fwd_kernels"])
            # Expected: the prefill's own uih + num_targets (one tiny int add)
            # and its "## action_prefill ##" range (a GPU annotation, not a
            # kernel). Nothing else may be added, nothing may go missing.
            tc.assertTrue(
                all("action_prefill" in k or "CUDAFunctor_add<" in k for k in extra)
                and sum(extra.values()) <= 2,
                f"extra fwd kernels {extra}",
            )
            tc.assertEqual(sum(missing.values()), 0, f"missing fwd kernels {missing}")
            report[f"l2={l2}"] = (
                f"bitwise+order OK; fwd kernels {len(pre['fwd_kernels'])} vs "
                f"{len(ctl['fwd_kernels'])}, extra={dict(extra)}"
            )
        return report


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class ActionPrefillGPUTest(unittest.TestCase, _Gate):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("DLRM_HSTU_FP8_AUTOTUNE_PINS") == "1":
            from generative_recommenders.ops.triton.autotune_pin_table import (
                apply_autotune_pins,
            )

            apply_autotune_pins()
        rows = int(os.environ.get("ACTION_PREFILL_ROWS", "2300000"))
        # mean UIH ~2250 like a production rank; B chosen for the row target
        cls.B = max(8, rows // 2250)
        cls.inp = _make_inputs(
            cls.B, 4499, 512, torch.device("cuda"), torch.bfloat16, seed=230923
        )

    def setUp(self) -> None:
        self._env = {k: os.environ.get(k) for k in (ENV, L2_ENV)}

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        action_prefill.clear()
        torch.cuda.synchronize()

    def _case(self, f1: bool, f2: bool, fold: bool, l2_arms: Tuple[bool, ...]):
        m = _make_preprocessor(512, 256, f1, f2, torch.device("cuda"), seed=17)
        rep = self.gate(self, m, self.inp, fold, l2_arms, profile=True)
        print(f"[action-prefill-test] gpu f1={f1} f2={f2} fold={fold}: {rep}",
              flush=True)
        del m
        torch.cuda.empty_cache()

    def test_f1_f2_production(self) -> None:
        self._case(True, True, True, (True, False))

    def test_f1_only(self) -> None:
        self._case(True, False, False, (True, False))

    def test_plain_sequential(self) -> None:
        self._case(False, False, False, (True,))

    def test_fallback_on_row_mismatch(self) -> None:
        """A prefill issued for other inputs is discarded, not used."""
        inp = _make_inputs(
            16, 300, 512, torch.device("cuda"), torch.bfloat16, seed=11
        )
        m = _make_preprocessor(512, 256, True, True, torch.device("cuda"), seed=3)
        os.environ[ENV] = "1"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.prefill_action(
                max_uih_len=inp["max_uih_len"], max_targets=1,
                total_targets=inp["total_targets"] + 1,  # wrong row count
                uih_seq_lengths=inp["uih"], num_targets=inp["num_targets"],
                seq_payloads=dict(inp["ints"]),
            )
        pre = action_prefill._pending
        self.assertIsNotNone(pre)
        got = m._take_action_prefill(
            max_uih_len=inp["max_uih_len"], max_targets=1,
            seq_embeddings=inp["feats"]["seq_embeddings"],
            num_targets=inp["num_targets"], seq_payloads=dict(inp["ints"]),
        )
        self.assertIsNone(got)
        self.assertIsNone(action_prefill._pending)

    def _grouped(self, on: bool) -> bool:
        """Force the module-level grouped-tail gate (read at import time)."""
        from generative_recommenders.modules import preprocessors as pp

        old = pp._PREPROC_GROUPED_TAIL
        pp._PREPROC_GROUPED_TAIL = on
        return old

    def test_f1_f2_grouped_tail(self) -> None:
        """ship11 item (30) grouped LayerNorm tail armed: the prefilled action
        head joins the group and the grouped node reuses the prefilled last
        Linear. Bitwise + backward order vs the grouped tail without prefill;
        the grouped forward kernel must run in both arms (no decline)."""
        from generative_recommenders.ops import preproc_grouped_tail as pgt

        old = self._grouped(True)
        seen: List[bool] = []
        orig_finish = pgt.GroupedTail.finish

        def finish(tail):  # record whether the chain stayed grouped
            out = orig_finish(tail)
            seen.append(not tail.declined)
            return out

        pgt.GroupedTail.finish = finish
        try:
            m = _make_preprocessor(512, 256, True, True, torch.device("cuda"), seed=17)
            rep = self.gate(self, m, self.inp, True, (True, False), profile=True)
            self.assertTrue(seen and all(seen), f"grouped tail declined: {seen}")
            # The gate compares the port with itself; also pin it to the
            # SHIPPED per-MLP chain. A needs_input_grad slip in the grouped
            # node shows as ALL-ZERO gradients (materialize_grads hands the
            # head node zeros), not as None: require every gradient to exist
            # and be nonzero, and compare with the shipped chain -- bitwise
            # under the item (10) LN512 pins (item (30)'s contract), else to a
            # tight relative norm (a dropped dgrad is a 100% error).
            w: Dict[str, torch.Tensor] = {}
            grp = _run(m, self.inp, True, True, True, w)
            self._grouped(False)
            shp = _run(m, self.inp, False, False, True, w)
            self._grouped(True)
            for name, res in (("grouped+prefill", grp), ("shipped", shp)):
                bad = [k for k, g in res["grads"].items()
                       if g is None or not bool((g.float().abs().amax() > 0).item())]
                self.assertEqual(bad, [], f"missing/all-zero gradients ({name}): {bad}")
            _pins, bitwise = pgt.ln512_pin_state()
            if bitwise:
                self.assertEqual(_compare(shp, grp), [],
                                 "grouped+prefill differs from the shipped chain")
                rep["vs_shipped"] = "bitwise (LN512 pins " + _pins + ")"
            else:
                worst = 0.0
                for part in ("outs", "grads"):
                    for k, ref in shp[part].items():
                        got = grp[part][k]
                        if not torch.is_floating_point(ref):
                            self.assertTrue(torch.equal(ref, got), k)
                            continue
                        den = ref.float().norm().item() or 1.0
                        rel = (got.float() - ref.float()).norm().item() / den
                        worst = max(worst, rel)
                        self.assertLess(rel, 2e-2, f"{part}:{k} rel {rel:.3g} vs shipped")
                rep["vs_shipped"] = (
                    f"rel-norm <= {worst:.2e} (LN512 pins {_pins}, not bitwise-pinned)"
                )
            print(f"[action-prefill-test] gpu grouped-tail f1=f2=1 fold=1: {rep}",
                  flush=True)
            del m
            torch.cuda.empty_cache()
        finally:
            pgt.GroupedTail.finish = orig_finish
            self._grouped(old)

    def test_grouped_tail_decline_from_prefill(self) -> None:
        """Forced declines of the grouped tail with a prefill issued, keyed by
        MLP (not by call count, since the prefill path checks the action MLP
        once more). "additional": the group declines at the additional MLP, so
        the prefilled action head completes the already-declined shipped chain
        (prefill used). "action": the group would decline AT the action MLP,
        so the prefill falls back (FALLBACK line) to the original order (not
        used). Each arm must be bitwise, with the same backward op order and
        grad-ready order, to the no-prefill run with the same forced decline."""
        from generative_recommenders.ops import preproc_grouped_tail as pgt

        old = self._grouped(True)
        orig = pgt._last_pair_reason
        try:
            m = _make_preprocessor(512, 256, True, True, torch.device("cuda"), seed=17)
            inp = _make_inputs(
                64, 600, 512, torch.device("cuda"), torch.bfloat16, seed=5
            )
            w: Dict[str, torch.Tensor] = {}
            _run(m, inp, False, False, True, w)
            targets = {
                "additional": m._additional_embedding_mlp,
                "action": m._action_embedding_mlp,
            }
            for name, target in targets.items():
                with self.subTest(decline_at=name):
                    hits: List[int] = []

                    def reason(mlp, eps, target=target, hits=hits):
                        if mlp is target:
                            hits.append(1)
                            return "forced (test)"
                        return orig(mlp, eps)

                    pgt._last_pair_reason = reason
                    pgt._DECLINE_WARNED = True  # keep the log quiet
                    ctl_d = _run(m, inp, False, False, True, w)
                    self.assertTrue(hits, "forced decline never reached")
                    hits.clear()
                    action_prefill._logged.discard("fallback:grouped")
                    pre = _run(m, inp, True, True, True, w)
                    self.assertTrue(hits, "forced decline never reached")
                    pgt._last_pair_reason = orig
                    self.assertEqual(_compare(ctl_d, pre), [])
                    self.assertEqual(pre["used"], 1 if name == "additional" else 0)
                    self.assertEqual(ctl_d["bwd_ops"], pre["bwd_ops"])
                    self.assertEqual(ctl_d["acc_order"], pre["acc_order"])
                    print(f"[action-prefill-test] gpu grouped-tail forced decline "
                          f"at the {name} MLP: bitwise+order OK (prefill used="
                          f"{pre['used']})", flush=True)
        finally:
            pgt._last_pair_reason = orig
            self._grouped(old)


class ActionPrefillCPUTest(unittest.TestCase, _Gate):
    def setUp(self) -> None:
        self._env = {k: os.environ.get(k) for k in (ENV, L2_ENV)}

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        action_prefill.clear()

    def test_preprocessor_cpu(self) -> None:
        inp = _make_inputs(6, 40, 32, torch.device("cpu"), torch.float32, seed=1)
        for f1, f2, fold in ((True, True, True), (True, False, False),
                             (False, False, False)):
            with self.subTest(f1=f1, f2=f2):
                m = _make_preprocessor(32, 16, f1, f2, torch.device("cpu"), seed=2)
                rep = self.gate(self, m, inp, fold, (True,), profile=False)
                print(f"[action-prefill-test] cpu f1={f1} f2={f2}: {rep}",
                      flush=True)

    def test_unbridged_order_differs(self) -> None:
        """Why the bridge exists: without it the branch's backward moves."""
        inp = _make_inputs(6, 40, 32, torch.device("cpu"), torch.float32, seed=1)
        m = _make_preprocessor(32, 16, False, False, torch.device("cpu"), seed=2)
        w: Dict[str, torch.Tensor] = {}
        ctl = _run(m, inp, False, False, False, w)
        orig = action_prefill.ActionPrefill.attach
        try:
            action_prefill.ActionPrefill.attach = lambda self: self.y
            raw = _run(m, inp, True, False, False, w)
        finally:
            action_prefill.ActionPrefill.attach = orig
        self.assertEqual(_compare(ctl, raw), [])
        self.assertEqual(raw["used"], 1)
        self.assertNotEqual(ctl["bwd_ops"], raw["bwd_ops"])
        self.assertNotEqual(ctl["acc_order"], raw["acc_order"])
        self.assertEqual(
            collections.Counter(ctl["bwd_ops"]), collections.Counter(raw["bwd_ops"])
        )

    def test_dlrm_hstu_main_forward_cpu(self) -> None:
        """Whole DlrmHSTU main_forward + backward, prefill issued by the hook."""
        from generative_recommenders.dlrm_v4.configs import get_hstu_configs
        from generative_recommenders.modules import dlrm_hstu as dh

        cfg = get_hstu_configs("yambda-5b")
        cfg.hstu_embedding_table_dim = 32
        cfg.hstu_transducer_embedding_dim = 32
        cfg.hstu_preprocessor_hidden_dim = 16
        cfg.hstu_num_heads = 2
        cfg.hstu_attn_linear_dim = 16
        cfg.hstu_attn_qk_dim = 16
        cfg.hstu_attn_num_layers = 1
        cfg.max_seq_len = 256
        for f1, f2 in ((True, True), (False, False)):
            with self.subTest(f1=f1, f2=f2):
                torch.manual_seed(5)
                model = dh.DlrmHSTU(
                    hstu_configs=cfg, embedding_tables={}, is_inference=False,
                    is_dense=True, bf16_training=False,
                    fuse_linear_norm=f1, fuse_ln_residual=f2,
                )
                model.set_hammer_kernel(HammerKernel.PYTORCH)
                model.train()
                ctl = self._dlrm_run(dh, model, cfg, prefill=False)
                ctl2 = self._dlrm_run(dh, model, cfg, prefill=False)
                self.assertEqual(_compare(ctl, ctl2), [])
                pre = self._dlrm_run(dh, model, cfg, prefill=True)
                self.assertTrue(pre["consumed"], "hook did not issue a prefill")
                self.assertEqual(_compare(ctl, pre), [])
                self.assertEqual(ctl["bwd_ops"], pre["bwd_ops"])
                self.assertEqual(ctl["acc_order"], pre["acc_order"])
                print(f"[action-prefill-test] cpu DlrmHSTU f1={f1} f2={f2}: "
                      f"{len(ctl['grads'])} grads, {len(ctl['bwd_ops'])} bwd ops "
                      "bitwise+order OK", flush=True)

    def _dlrm_run(self, dh, model, cfg, prefill: bool) -> Dict[str, object]:
        os.environ[ENV] = "1" if prefill else "0"
        g = torch.Generator().manual_seed(9)
        B, D = 5, cfg.hstu_embedding_table_dim
        uih = torch.randint(3, 30, (B,), generator=g)
        num_c = torch.ones(B, dtype=torch.int64)
        T = int(uih.sum())
        for p in model.parameters():
            p.grad = None
        leaves: Dict[str, torch.Tensor] = {}
        seq_embeddings = {}
        for name in cfg.user_embedding_feature_names:
            if name in ("item_id", "artist_id", "album_id"):
                lengths, rows = uih, T
            else:
                lengths, rows = torch.ones(B, dtype=torch.int64), B
            leaves[name] = torch.randn(rows, D, generator=g).requires_grad_(True)
            seq_embeddings[name] = dh.SequenceEmbedding(lengths=lengths,
                                                        embedding=leaves[name])
        for name in cfg.item_embedding_feature_names:
            leaves[name] = torch.randn(B, D, generator=g).requires_grad_(True)
            seq_embeddings[name] = dh.SequenceEmbedding(lengths=num_c,
                                                        embedding=leaves[name])
        payload = {
            "action_weight": torch.randint(0, 8, (T,), generator=g),
            "item_action_weight": torch.randint(0, 8, (B,), generator=g),
            "action_timestamp": torch.randint(0, 1 << 30, (T,), generator=g),
            "item_query_time": torch.randint(0, 1 << 30, (B,), generator=g),
            "dummy_watch_time": torch.zeros(T, dtype=torch.int64),
            "item_dummy_watchtime": torch.zeros(B, dtype=torch.int64),
            "uih_offsets": torch.ops.fbgemm.asynchronous_complete_cumsum(uih),
            "candidate_offsets": torch.ops.fbgemm.asynchronous_complete_cumsum(num_c),
        }
        max_c, max_u = dh.fx_length_stats(num_c, uih)  # seeds the total memo
        # Seeded BEFORE the hook: an RNG draw inside the prefill would shift
        # every dropout mask after it and break the bitwise comparison.
        torch.manual_seed(1234)
        model._issue_action_prefill(
            max_uih_len=max_u, max_num_candidates=max_c, uih_seq_lengths=uih,
            num_candidates=num_c, payload_features=payload,
        )
        issued = action_prefill._pending is not None
        used0 = action_prefill.stats["used"]
        out = model.main_forward(
            seq_embeddings=seq_embeddings, payload_features=payload,
            max_uih_len=max_u, uih_seq_lengths=uih, max_num_candidates=max_c,
            num_candidates=num_c,
        )
        consumed = (
            issued and action_prefill._pending is None
            and action_prefill.stats["used"] == used0 + 1
        )
        loss = sum(out[2].values())
        acc_order: List[str] = []
        hooks = [
            p.register_post_accumulate_grad_hook(lambda _p, n=n: acc_order.append(n))
            for n, p in model.named_parameters()
        ]
        log = _OpLog()
        with log:
            loss.backward()
        for h in hooks:
            h.remove()
        grads = {f"param:{n}": p.grad for n, p in model.named_parameters()}
        grads.update({f"input:{k}": v.grad for k, v in leaves.items()})
        outs = {"user": out[0], "item": out[1], "preds": out[3], "loss": loss}
        return {"outs": {k: v.detach() for k, v in outs.items() if v is not None},
                "grads": grads, "bwd_ops": log.ops, "consumed": consumed,
                "acc_order": acc_order}


if __name__ == "__main__":
    unittest.main(verbosity=2)
