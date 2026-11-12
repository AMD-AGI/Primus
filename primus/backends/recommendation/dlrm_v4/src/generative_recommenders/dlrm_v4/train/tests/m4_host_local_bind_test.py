# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""M4 v3 (comm iteration 53; from the 'tests' review lens): DLRM_HSTU_FP8_M4_HOST_LOCAL's gin binding,
modelled on cold_prefetch_bind_test.py: the binding reaches the env, a pre-set env wins, the gin ships "0"
(global dealing: the item (82) M4-ON form, gin flip comm_m4_gin_on_c68395d.diff; was 'auto' while M4 was default OFF)."""
import os
import unittest
from unittest import mock

from generative_recommenders.dlrm_v4.train import _env_bootstrap
from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

_GIN = os.path.join(os.path.dirname(os.path.abspath(_env_bootstrap.__file__)), "gin", "yambda_5b.gin")
_K = "DLRM_HSTU_FP8_M4_HOST_LOCAL"


class M4HostLocalBindingTest(unittest.TestCase):
    def test_binding_sets_the_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_env_bootstrap(DLRM_HSTU_FP8_M4_HOST_LOCAL="0")
            self.assertEqual(os.environ.get(_K), "0")

    def test_preset_env_wins(self) -> None:
        with mock.patch.dict("os.environ", {_K: "1"}, clear=True):
            apply_env_bootstrap(DLRM_HSTU_FP8_M4_HOST_LOCAL="0")
            self.assertEqual(os.environ.get(_K), "1")

    def test_gin_binds_ship_form(self) -> None:
        with open(_GIN) as f:
            live = [ln.strip() for ln in f if not ln.lstrip().startswith("#") and _K in ln]
        self.assertEqual(live, ['apply_env_bootstrap.DLRM_HSTU_FP8_M4_HOST_LOCAL = "0"'])


if __name__ == "__main__":
    unittest.main()
