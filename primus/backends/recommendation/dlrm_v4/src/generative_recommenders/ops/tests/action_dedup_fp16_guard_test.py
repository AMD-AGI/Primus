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

# pyre-unsafe
"""(119) ACTION_DEDUP: the fp16-autocast guard.

Under fp16 autocast the fp32 per-code gradient sum, once rounded for the fp16
MLP backward, could exceed the fp16 range (65504) and turn into inf / NaN with
no guard (the old IT32 FP16 latent). action_dedup.decline_reason therefore
declines every forward under fp16 autocast, and the shipped per-token action
branch runs. These CPU tests pin that behaviour."""

import contextlib
import io
import unittest

import torch

from generative_recommenders.ops import action_dedup


class _Enc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._num_action_types = 3
        self._action_feature_name = "action_weight"
        self._watchtime_feature_name = "dummy_watch_time"
        self._watchtime_to_action_thresholds_and_weights = []


class _Pre(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._action_encoder = _Enc()
        self._action_embedding_mlp = torch.nn.Linear(24, 8)


def _payloads() -> dict:
    return {"action_weight": torch.tensor([1, 2, 4, 7], dtype=torch.int64)}


class ActionDedupFp16GuardTest(unittest.TestCase):
    def test_declines_under_cpu_fp16_autocast(self) -> None:
        pre, ref = _Pre(), torch.zeros(1)
        with torch.autocast("cpu", dtype=torch.float16):
            self.assertEqual(
                action_dedup.decline_reason(pre, ref, _payloads()), "fp16 autocast"
            )

    def test_runs_under_cpu_bf16_autocast_and_without_autocast(self) -> None:
        pre, ref = _Pre(), torch.zeros(1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertIsNone(action_dedup.decline_reason(pre, ref, _payloads()))
        self.assertIsNone(action_dedup.decline_reason(pre, ref, _payloads()))

    def test_fp16_guard_precedes_other_checks_and_logs_decline(self) -> None:
        pre, ref = _Pre(), torch.zeros(1)
        with torch.autocast("cpu", dtype=torch.float16):
            reason = action_dedup.decline_reason(pre, ref, _payloads())
        self.assertEqual(reason, "fp16 autocast")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            action_dedup.note_decline(reason)
        text = buf.getvalue()
        self.assertIn("[action-dedup] DECLINED (fp16 autocast)", text)
        self.assertIn("runs the action branch per token", text)


if __name__ == "__main__":
    unittest.main()
