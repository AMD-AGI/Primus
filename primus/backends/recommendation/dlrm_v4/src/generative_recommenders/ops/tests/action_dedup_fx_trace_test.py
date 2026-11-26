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

"""(119) ACTION_DEDUP under torch.fx symbolic tracing (WBND r119_v0 M1).

TorchRec's model rewrite fx-traces ContextualPreprocessor.forward once per rank. With the knob on, v0 reached
action_dedup.take_or_plan inside that trace and printed '[action-dedup] DECLINED (scripting / fx tracing /
torch.compile)' once per rank, a must-be-ABSENT marker. v1 gates the (119) block on `not is_fx_tracing()`: the trace
records the shipped branch (the same graph as with the knob off) and the dedup module is never consulted, so nothing
prints. Eager forwards still take the dedup. (The (65) prefill issue site, DlrmHSTU._issue_action_prefill, already
returns before issuing under fx tracing.)

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_fx_trace_test.py
"""

import contextlib
import io
import os
import unittest
from typing import Dict, List, Tuple

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm
import torch

from generative_recommenders.common import HammerKernel
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.ops import action_dedup

ENV = action_dedup.ENV_VAR
ADDITIONAL = ["artist_id", "album_id"]
B, MAX_UIH = 6, 9


def _model() -> ContextualPreprocessor:
    torch.manual_seed(5)
    m = ContextualPreprocessor(
        input_embedding_dim=512,
        hidden_dim=256,
        output_embedding_dim=512,
        contextual_feature_to_max_length={},
        contextual_feature_to_min_uih_length={},
        action_embedding_dim=8,
        action_feature_name="action_weight",
        action_weights=[1, 2, 4],
        additional_embedding_features=ADDITIONAL,
        action_embedding_init_std=0.1,
        is_inference=False,
    )
    m.train()
    m.set_hammer_kernel(HammerKernel.PYTORCH)
    return m


class _Wrap(torch.nn.Module):
    """The preprocessor's forward with tensor inputs only (host ints fixed), as a traced call site sees it."""

    def __init__(self, pre: ContextualPreprocessor, t_uih: int) -> None:
        super().__init__()
        self.pre = pre
        self.t_uih = t_uih

    def forward(self, seq_embeddings: torch.Tensor, acts: torch.Tensor, artist: torch.Tensor, album: torch.Tensor,
                seq_lengths: torch.Tensor, num_targets: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        payloads = {"action_weight": acts, "artist_id": artist, "album_id": album}
        out = self.pre(max_uih_len=MAX_UIH, max_targets=1, total_uih_len=self.t_uih, total_targets=B,
                       seq_lengths=seq_lengths, seq_timestamps=ts, seq_embeddings=seq_embeddings,
                       num_targets=num_targets, seq_payloads=payloads)
        return out[6]


def _inputs() -> Tuple[int, List[torch.Tensor]]:
    g = torch.Generator().manual_seed(11)
    uih = torch.randint(1, MAX_UIH + 1, (B,), generator=g)
    nt = torch.ones(B, dtype=torch.int64)
    t_uih = int(uih.sum())
    rows = t_uih + B
    return t_uih, [
        torch.randn(rows, 512, generator=g),
        torch.randint(0, 8, (t_uih,), generator=g),
        torch.randn(rows, 512, generator=g),
        torch.randn(rows, 512, generator=g),
        uih + nt,
        nt,
        torch.randint(0, 1 << 30, (rows,), generator=g),
    ]


class ActionDedupFxTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env: Dict[str, object] = {k: os.environ.get(k) for k in (ENV, action_prefill.ENV)}
        self._orig = (action_dedup.take_or_plan, action_dedup.note_decline, action_dedup.decline_reason)
        self.calls: Dict[str, int] = {"take_or_plan": 0, "note_decline": 0, "decline_reason": 0}

        def counted(name, fn):  # pyre-ignore[2, 3]
            def wrapper(*a, **k):  # pyre-ignore[2, 3]
                self.calls[name] += 1
                return fn(*a, **k)
            return wrapper

        action_dedup.take_or_plan = counted("take_or_plan", self._orig[0])
        action_dedup.note_decline = counted("note_decline", self._orig[1])
        action_dedup.decline_reason = counted("decline_reason", self._orig[2])
        action_dedup._latched = None

    def tearDown(self) -> None:
        action_dedup.take_or_plan, action_dedup.note_decline, action_dedup.decline_reason = self._orig
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # pyre-ignore[6]

    def _trace(self, knob: str, prefill: str) -> Tuple[torch.fx.GraphModule, str]:
        os.environ[ENV] = knob
        os.environ[action_prefill.ENV] = prefill
        t_uih, _ = _inputs()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gm = torch.fx.symbolic_trace(_Wrap(_model(), t_uih))
        return gm, buf.getvalue()

    def test_trace_with_knob_on_never_consults_the_dedup_and_prints_nothing(self) -> None:
        declined0 = action_dedup.stats["declined"]
        for prefill in ("0", "1"):
            _, printed = self._trace("1", prefill)
            self.assertNotIn("[action-dedup]", printed, prefill)
        self.assertEqual(self.calls, {"take_or_plan": 0, "note_decline": 0, "decline_reason": 0})
        self.assertEqual(action_dedup.stats["declined"], declined0)

    def test_traced_graph_is_the_knob_off_graph(self) -> None:
        on, _ = self._trace("1", "0")
        off, _ = self._trace("0", "0")
        self.assertEqual(on.code, off.code)

    def test_eager_forward_still_takes_the_dedup(self) -> None:
        os.environ[ENV] = "1"
        os.environ[action_prefill.ENV] = "0"
        t_uih, xs = _inputs()
        w = _Wrap(_model(), t_uih)
        forwards0 = action_dedup.stats["forwards"]
        out_on = w(*xs)
        self.assertEqual(self.calls["take_or_plan"], 1)
        self.assertEqual(action_dedup.stats["forwards"], forwards0 + 1)
        os.environ[ENV] = "0"
        out_off = w(*xs)
        torch.testing.assert_close(out_on, out_off, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
