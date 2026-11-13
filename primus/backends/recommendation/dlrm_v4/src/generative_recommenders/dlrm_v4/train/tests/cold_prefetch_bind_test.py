# pyre-strict
"""Item (36)'s gin binding: apply_env_bootstrap.DL_COLD_PREFETCH_STREAM pushes
its value into the environment (where cold_prefetch.start() reads it after
RUN_START); a pre-set environment variable wins; no binding leaves the code
default (unset = off). The gin file ships no binding for it."""
import os
import unittest
from unittest import mock

from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

_GIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "gin", "yambda_5b.gin")


class StreamBindingTest(unittest.TestCase):
    def test_binding_sets_the_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_env_bootstrap(DL_COLD_PREFETCH_STREAM="auto")
            self.assertEqual(os.environ.get("DL_COLD_PREFETCH_STREAM"), "auto")

    def test_preset_env_wins(self) -> None:
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_STREAM": "0"}, clear=True):
            apply_env_bootstrap(DL_COLD_PREFETCH_STREAM="1")
            self.assertEqual(os.environ.get("DL_COLD_PREFETCH_STREAM"), "0")

    def test_no_binding_leaves_it_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_env_bootstrap()
            self.assertNotIn("DL_COLD_PREFETCH_STREAM", os.environ)

    def test_gin_binds_the_walk_on(self) -> None:
        # ship14 loader arm "walk" (gin_walk_on): exactly one live binding, "1".
        with open(_GIN) as f:
            lines = [ln.strip() for ln in f if not ln.lstrip().startswith("#")]
        live = [ln for ln in lines if "DL_COLD_PREFETCH_STREAM" in ln]
        self.assertEqual(live, ['apply_env_bootstrap.DL_COLD_PREFETCH_STREAM = "1"'])


if __name__ == "__main__":
    unittest.main()
