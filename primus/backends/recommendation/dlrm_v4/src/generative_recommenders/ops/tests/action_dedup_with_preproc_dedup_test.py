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

"""(119) ACTION_DEDUP stacked on (118) PREPROC_DEDUP: the composed CPU test (main's ruling (b); ROW_119 md 12.D / 14).

The four states OFF / (118) / (119) / both, on the real ContextualPreprocessor.forward (bf16 autocast, the real
ActionEncoder with targets, F1 + F2 on, the E-M1 grouped tail requested):
  - every state runs forward + backward;
  - with (118) engaged no grouped tail is begun (so GroupedTail.finish()'s empty-chain assert is unreachable) and
    (119) adds table[code] to (118)'s content + additional sum;
  - every output / gradient is within the bf16 reduction-order band of the all-OFF reference;
  - both-ON is deterministic;
  - the stacked forward's empty-tail guard: a chain no MLP went through is not finished (finish() would assert).

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_with_preproc_dedup_test.py
"""

import inspect
import os
import unittest
from typing import Any, Dict, List, Tuple

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm
import torch

from generative_recommenders.common import HammerKernel
from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules import preprocessors as pp
from generative_recommenders.ops import action_dedup, preproc_dedup as pd

ENV = action_dedup.ENV_VAR
U, B = 500, 4
UIH = [700, 900, 1100, 1296]  # + 1 target per sequence -> 4000 rows


def _arm_118() -> None:
    pd.reset_for_testing()
    pd._DEVICES = ("cuda", "cpu")
    pd._STATE = "armed"
    pd._ARMED_DEVS.add(("cpu", -1))


def _pre() -> pp.ContextualPreprocessor:
    torch.manual_seed(1234)
    pre = pp.ContextualPreprocessor(
        input_embedding_dim=512, hidden_dim=256, output_embedding_dim=512, contextual_feature_to_max_length={},
        contextual_feature_to_min_uih_length={}, action_embedding_dim=8, action_feature_name="act",
        action_weights=[1, 2, 4], additional_embedding_features=["art", "alb"], action_embedding_init_std=0.1,
        is_inference=False, fuse_linear_norm=True, fuse_ln_residual=True)
    pre.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    for m in pre.modules():
        m._f1_reported = True  # quiet the CPU INERT line
    with torch.no_grad():  # nonzero LN affine / biases so every gradient is exercised
        for n, p in pre.named_parameters():
            if p.dim() == 1:
                p.copy_(torch.linspace(-0.5, 0.5, p.numel()) + (1.0 if n.endswith("weight") else 0.0))
    pre.train()
    return pre


def _data(seed: int = 3) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    T = sum(UIH) + B
    n_items = U // 2
    item = torch.randint(0, 40, (T,), generator=g)  # heavy duplication, as in a real batch
    span = (U - n_items) // 2
    cat = torch.stack([n_items + (item * 31) % span, n_items + span + (item * 17) % span], 1).reshape(-1)
    return {
        "emb": torch.randn(U, 512, generator=g),
        "item": item.to(torch.int64),
        "cat": cat.to(torch.int64),
        "act": torch.randint(0, 8, (sum(UIH),), generator=g),
        "ts": torch.randint(0, 1 << 30, (T,), generator=g),
        "w": torch.randn(T, 512, generator=g),
    }


def _run(pre: pp.ContextualPreprocessor, d: Dict[str, torch.Tensor], on118: bool, on119: bool,
         calls: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    os.environ[ENV] = "1" if on119 else "0"
    os.environ[action_prefill.ENV] = "0"
    emb = d["emb"].clone().requires_grad_(True)
    item, cat = d["item"], d["cat"]
    rec = pd.PPDDRecord(emb, item, cat, (), ebo._expand, ebo._CastExpandFp32Bwd.apply, torch.bfloat16) if on118 else None
    x = ebo._CastExpandFp32Bwd.apply(emb, item, torch.bfloat16)
    big = ebo._CastExpandFp32Bwd.apply(emb, cat, torch.bfloat16).view(item.numel(), 1024)
    if rec is not None:
        setattr(x, pd.PPDD_ATTR, rec)
        setattr(big, pd.PPDD_ATTR, rec)
    payloads = {"art": big[:, :512], "alb": big[:, 512:], "act": d["act"]}
    setattr(payloads["art"], ebo.CATFUSED_ATTR, (big, 0, 2))
    setattr(payloads["alb"], ebo.CATFUSED_ATTR, (big, 1, 2))
    uih = torch.tensor(UIH, dtype=torch.int64)
    nt = torch.ones(B, dtype=torch.int64)
    old_flag, old_begin = pp._PREPROC_GROUPED_TAIL, pre._grouped_tail_begin
    pp._PREPROC_GROUPED_TAIL = True  # E-M1 requested; on CPU it would decline, so record the begin calls instead
    pre._grouped_tail_begin = lambda *a, **k: calls.append("tail") or None  # type: ignore[assignment]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = pre(max_uih_len=max(UIH), max_targets=1, total_uih_len=sum(UIH), total_targets=B,
                      seq_lengths=uih + nt, seq_timestamps=d["ts"], seq_embeddings=x, num_targets=nt,
                      seq_payloads=payloads)[6]
    finally:
        pp._PREPROC_GROUPED_TAIL = old_flag
        pre._grouped_tail_begin = old_begin  # type: ignore[assignment]
    params = [emb] + list(pre.parameters())
    grads = torch.autograd.grad((out.float() * d["w"]).sum(), params, allow_unused=True)
    return out.detach(), [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, grads)]


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    n = ref.double().norm()
    d = (a.double() - ref.double()).norm()
    return float(d / n) if n > 0 else float(d)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    iv = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(iv[a.dtype]),
                                                                     b.contiguous().view(iv[b.dtype]))


class ActionDedupWithPreprocDedupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._env = {k: os.environ.get(k) for k in (ENV, action_prefill.ENV)}
        _arm_118()
        cls.pre = _pre()
        cls.d = _data()
        cls.res: Dict[Tuple[bool, bool], Any] = {}
        cls.calls: Dict[Tuple[bool, bool], List[str]] = {}
        cls.counts: Dict[Tuple[bool, bool], Tuple[int, int]] = {}
        for on118 in (False, True):
            for on119 in (False, True):
                calls: List[str] = []
                d0, f0 = pd.COUNTS.get("dedup", 0), action_dedup.stats["forwards"]
                cls.res[(on118, on119)] = _run(cls.pre, cls.d, on118, on119, calls)
                cls.calls[(on118, on119)] = calls
                cls.counts[(on118, on119)] = (pd.COUNTS.get("dedup", 0) - d0, action_dedup.stats["forwards"] - f0)

    @classmethod
    def tearDownClass(cls) -> None:
        for k, v in cls._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        pd.reset_for_testing()

    def test_each_state_engages_what_it_should(self) -> None:
        for (on118, on119), (n118, n119) in self.counts.items():
            self.assertEqual(n118, 1 if on118 else 0, (on118, on119))
            self.assertEqual(n119, 1 if on119 else 0, (on118, on119))
            # (118) engaged: no grouped tail is begun, so finish() never meets an empty chain
            self.assertEqual(self.calls[(on118, on119)], [] if on118 else ["tail"], (on118, on119))

    def test_outputs_and_grads_within_the_band_of_all_off(self) -> None:
        ref_out, ref_g = self.res[(False, False)]
        names = ["emb"] + [n for n, _ in self.pre.named_parameters()]
        for key in ((True, False), (False, True), (True, True)):
            out, gs = self.res[key]
            self.assertEqual((out.dtype, out.shape), (ref_out.dtype, ref_out.shape))
            self.assertTrue(bool(torch.isfinite(out.float()).all()), key)
            self.assertLessEqual(_rel(out, ref_out), 4e-3, key)
            for n, a, b in zip(names, gs, ref_g):
                self.assertTrue(bool(torch.isfinite(a).all()), (key, n))
                self.assertLessEqual(_rel(a, b), 2e-2, (key, n))

    def test_both_on_is_the_118_sum_plus_the_119_table(self) -> None:
        # (119) only changes how the action contribution is formed: vs (118) alone the difference is rounding-level
        out_both, _ = self.res[(True, True)]
        out_118, _ = self.res[(True, False)]
        self.assertLessEqual(_rel(out_both, out_118), 4e-3)

    def test_both_on_is_deterministic(self) -> None:
        out1, g1 = self.res[(True, True)]
        out2, g2 = _run(self.pre, self.d, True, True, [])
        self.assertTrue(_same(out1, out2))
        for a, b in zip(g1, g2):
            self.assertTrue(_same(a, b))

    def test_empty_tail_guard(self) -> None:
        from generative_recommenders.ops.preproc_grouped_tail import GroupedTail

        tail = GroupedTail(self.pre._mlp_add, 1e-6, 8, torch.device("cpu"))
        # the hazard: a chain no MLP went through has nothing to return
        self.assertFalse(tail.declined)
        self.assertEqual(len(tail._mlps), 0)
        with self.assertRaises(AssertionError):
            tail.finish()
        # the stacked forward finishes a tail only when an MLP went through it
        src = inspect.getsource(pp.ContextualPreprocessor.forward)
        guard = src.index("if tail.declined or len(tail._mlps) > 0:")
        self.assertLess(src.index("if tail is not None:", guard - 900), guard)
        self.assertLess(guard, src.index("output_seq_embeddings = tail.finish()"))


if __name__ == "__main__":
    unittest.main()
