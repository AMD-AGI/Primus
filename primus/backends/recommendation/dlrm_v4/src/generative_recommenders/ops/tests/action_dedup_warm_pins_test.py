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

"""(119) ACTION_DEDUP prebuild warm-up vs the Triton autotune pins (WBND r119 kill-switch lens K1) and the knob
spellings (K3).

The table MLP's Triton norm forwards are autotuned on a key without the row count. With the pins on (production), the
pin table seeds every key before the BLAS warm-up, so _warm_mlp's 18-row norm launches only compile. With the pins off
(DLRM_HSTU_FP8_AUTOTUNE_PINS != "1" or TRITON_FULL_AUTOTUNE=1), an 18-row first launch would pick the config that
the ~2.3M-row launches then reuse, so _warm_mlp skips the norm launches and warms only the GEMMs. CPU only.

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_warm_pins_test.py
"""

import os
import unittest
from typing import Dict, List, Optional
from unittest import mock

import torch

from generative_recommenders.ops import action_dedup

PINS, FULL = "DLRM_HSTU_FP8_AUTOTUNE_PINS", "TRITON_FULL_AUTOTUNE"


class WarmPinsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in (PINS, FULL, action_dedup.ENV_VAR)}

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set(self, pins: Optional[str], full: Optional[str]) -> None:
        for k, v in ((PINS, pins), (FULL, full)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_norm_warm_only_with_the_pins_on(self) -> None:
        for pins, full, want in (("1", None, True), ("1", "0", True), ("1", "1", False), ("0", None, False),
                                 (None, None, False), ("true", None, False)):
            self._set(pins, full)
            self.assertEqual(action_dedup._norm_warm_ok(), want, (pins, full))

    def _warm(self) -> List[int]:
        calls: List[int] = []

        def fake(h_in, w, b, nw, nb, eps, swish, extra):  # pyre-ignore[2, 3]
            calls.append(int(h_in.shape[0]))
            return torch.zeros((h_in.shape[0], w.shape[0]), dtype=torch.bfloat16)

        from generative_recommenders.ops import fused_linear_norm

        with mock.patch.object(fused_linear_norm._LinearNormFunction, "apply", side_effect=fake):
            ran = action_dedup._warm_mlp(torch.device("cpu"), 18)
        self.assertEqual(ran, bool(calls))
        return calls

    def test_warm_mlp_launches_the_norms_only_with_the_pins_on(self) -> None:
        self._set("1", None)
        self.assertEqual(self._warm(), [18, 18])  # both pairs, at the table rows
        self._set("0", None)
        self.assertEqual(self._warm(), [])  # GEMMs only
        self._set("1", "1")
        self.assertEqual(self._warm(), [])

    def test_knob_spellings(self) -> None:
        for v, want in (("1", True), ("true", True), ("ON", True), (" yes ", True), ("0", False), ("false", False),
                        ("off", False), ("no", False), ("2", False), ("", False)):
            os.environ[action_dedup.ENV_VAR] = v
            self.assertEqual(action_dedup.enabled(), want, v)
        os.environ.pop(action_dedup.ENV_VAR, None)
        self.assertFalse(action_dedup.enabled())


if __name__ == "__main__":
    unittest.main()
