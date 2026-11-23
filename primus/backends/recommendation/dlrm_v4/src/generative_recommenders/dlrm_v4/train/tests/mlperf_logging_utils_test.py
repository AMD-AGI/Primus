# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# pyre-unsafe
"""Unit tests for the MLPerf Training 6.1 disclosure keys in MLPerfLogger.

Covers the nine keys training_6.1.0/common.yaml requires EXACTLY_ONE of
(lowest_numerical_precision_in_{linear,attn,comm}, {tensor,pipeline,context,
expert}_parallelism, micro_batch_size, config_filename), the FP8 switches the
precision resolver has to see (including mlu._FP8_LINEAR_PATHS) and the detected
submission_platform. CPU only: no GPU, no process group. torch and gin are
stubbed when they are not installed so this also runs on a login host with just
mlperf_logging; with the real packages present (training container) the stubs
are not used.

    python3 -m unittest generative_recommenders.dlrm_v4.train.tests.mlperf_logging_utils_test
"""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock


def _install_stubs() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = types.ModuleType("torch")
        torch.distributed = types.SimpleNamespace(
            is_available=lambda: False,
            is_initialized=lambda: False,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            barrier=lambda: None,
        )
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: 0,
            get_device_properties=lambda i: None,
        )
        sys.modules["torch"] = torch
    try:
        import gin  # noqa: F401
    except ImportError:
        gin = types.ModuleType("gin")

        def configurable(fn=None, **_kw):
            return fn if fn is not None else (lambda f: f)

        def query_parameter(name):
            raise ValueError(name)

        gin.configurable = configurable
        gin.query_parameter = query_parameter
        sys.modules["gin"] = gin


_install_stubs()
mlu = importlib.import_module(
    "generative_recommenders.dlrm_v4.train.mlperf_logging_utils"
)

NEW_KEYS = (
    "lowest_numerical_precision_in_linear",
    "lowest_numerical_precision_in_attn",
    "lowest_numerical_precision_in_comm",
    "tensor_parallelism",
    "pipeline_parallelism",
    "context_parallelism",
    "expert_parallelism",
    "micro_batch_size",
    "config_filename",
)
# training_6.1.0/common.yaml allow-list for the three precision keys.
ALLOWED_PRECISIONS = {
    "fp64", "fp32", "tf32", "fp16", "fp8", "mxfp6", "nvfp4", "mxfp4",
    "bfloat16", "Graphcore FLOAT 16.16", "int8", "uint8", "int4", "uint4",
}

# The FP8 env the gin default (apply_env_bootstrap.DLRM_HSTU_FP8_FWD=True)
# exports: the parts of _env_bootstrap._FP8_FORWARD_ARM the gating reads.
FP8_ON_ENV = {
    "DLRM_HSTU_FP8_FWD": "1",
    "DLRM_HSTU_FP8_GEMM": "1",
    "DLRM_HSTU_FP8_ATTN": "1",
    "DLRM_HSTU_FP8_FUSE_LN": "1",
    "DLRM_HSTU_FP8_FUSE_QKV": "1",
    "DLRM_HSTU_FP8_BWD": "1",
}

# The other FP8 linear switches yambda_5b.gin binds on (mlu._FP8_LINEAR_PATHS):
# E-M3a's preprocessor L2 GEMMs inside the E-M1 grouped tail, and the uvqk/vk
# projection backward. apply_env_bootstrap exports them whatever
# DLRM_HSTU_FP8_FWD/BWD say.
GIN_FP8_LINEAR_ENV = {
    "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL": "1",
    "DLRM_HSTU_FP8_PREPROC_L2": "1",
    "DLRM_HSTU_FP8_BWD_UVQK": "1",
}
# What the documented bf16 opt-out (DLRM_HSTU_FP8_FWD=0 DLRM_HSTU_FP8_BWD=0,
# rest from yambda_5b.gin) leaves for the gating to read.
FWD_BWD_OFF_ENV = dict(
    GIN_FP8_LINEAR_ENV, DLRM_HSTU_FP8_FWD="0", DLRM_HSTU_FP8_BWD="0"
)
# Every switch the resolver reads, all off. Patched over os.environ so a
# launcher-exported FP8 env cannot leak into a test.
ALL_OFF_ENV = {k: "0" for k in (*FP8_ON_ENV, *GIN_FP8_LINEAR_ENV)}

# gin values of the production yambda_5b.gin (env macros already resolved).
DEFAULT_GIN = {
    "make_model.bf16_training": True,
    "make_optimizer_and_shard.sparse_a2a_forward_precision": "fp16",
    "make_optimizer_and_shard.sparse_a2a_backward_precision": "fp16",
}


def _gin(values: Dict[str, Any]):
    def fake(name: str, default: Any) -> Any:
        return values.get(name, default)

    return mock.patch.object(mlu, "_gin_param", side_effect=fake)


# IX-FP8F (rccl): its _FP8_LINEAR_PATHS row gates on IX and on its own switch.
# Both are off in ALL_OFF_ENV, so an exported IX env cannot leak into a test.
IX_FP8F_ENV = {
    "DLRM_HSTU_FP8_EC_INDEXED_L1": "1",
    "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8": "1",
}
ALL_OFF_ENV.update({k: "0" for k in IX_FP8F_ENV})
_IX_MODULE = "generative_recommenders.ops.triton.triton_ec_indexed_l1"
_IX_FP8F_MODULE = "generative_recommenders.ops.triton.triton_ec_indexed_l1_fp8"
_SAVED_IX_ENV: Dict[str, Optional[str]] = {}


def setUpModule() -> None:
    # Tests that leave a switch unset must not see an exported IX env.
    for k in IX_FP8F_ENV:
        _SAVED_IX_ENV[k] = os.environ.pop(k, None)


def tearDownModule() -> None:
    for k, v in _SAVED_IX_ENV.items():
        if v is not None:
            os.environ[k] = v


class _Recorder:
    """Stands in for mllog's logger; records (kind, key, value) in order."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, Any]] = []

    def event(self, key, value=None, metadata=None):
        self.calls.append(("event", key, value))

    def start(self, key, value=None, metadata=None):
        self.calls.append(("start", key, value))

    def end(self, key, value=None, metadata=None):
        self.calls.append(("end", key, value))

    def keys(self) -> List[str]:
        return [k for _, k, _ in self.calls]

    def value(self, key: str) -> Any:
        vals = [v for _, k, v in self.calls if k == key]
        assert len(vals) == 1, (key, vals)
        return vals[0]


def _logger(rank: int = 0, platform: Optional[str] = "MI350X"):
    lg = mlu.MLPerfLogger(rank=rank, submission_platform=platform)
    rec = _Recorder()
    lg._logger = rec
    return lg, rec


@unittest.skipUnless(mlu._MLLOG_AVAILABLE, "mlperf_logging not installed")
class RunStartDisclosureTest(unittest.TestCase):
    def _run_start(self, env: Dict[str, str], gin_values: Dict[str, Any], **kw):
        lg, rec = _logger(**kw)
        with _gin(gin_values), mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(mlu, "sys", types.SimpleNamespace(modules={})):
            # sys.modules is hidden so the env path (not a live triton_fp8
            # from another test) decides; that path is tested separately.
            for k in (*FP8_ON_ENV, *GIN_FP8_LINEAR_ENV, "DLRM_HSTU_FP8_FWD_MXFP6"):
                if k not in env:
                    os.environ.pop(k, None)
            lg.log_run_start(
                global_batch_size=8192,
                seed=79275716,
                micro_batch_size=1024,
                config_filename="/x/gin/yambda_5b.gin",
            )
        return rec

    def test_default_stack_emits_each_key_once_before_run_start(self) -> None:
        rec = self._run_start(FP8_ON_ENV, DEFAULT_GIN)
        keys = rec.keys()
        run_start = keys.index("run_start")
        for key in NEW_KEYS:
            self.assertEqual(keys.count(key), 1, key)
            self.assertLess(keys.index(key), run_start, key)
            self.assertLess(keys.index(key), keys.index("init_stop"), key)
        self.assertEqual(rec.value("lowest_numerical_precision_in_linear"), "fp8")
        self.assertEqual(rec.value("lowest_numerical_precision_in_attn"), "fp8")
        self.assertEqual(rec.value("lowest_numerical_precision_in_comm"), "fp16")
        for key in NEW_KEYS[3:7]:
            v = rec.value(key)
            self.assertIsInstance(v, int)
            self.assertEqual(v, 1)
        self.assertEqual(rec.value("micro_batch_size"), 1024)
        self.assertEqual(rec.value("config_filename"), "yambda_5b.gin")
        self.assertEqual(rec.value("submission_platform"), "MI350X")
        self.assertEqual(keys.count("submission_platform"), 1)

    def test_mxfp6_trio_disclosed_at_run_start(self) -> None:
        rec = self._run_start(dict(FP8_ON_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1"), DEFAULT_GIN)
        keys = rec.keys()
        self.assertEqual(keys.count("lowest_numerical_precision_in_linear"), 1)
        self.assertEqual(rec.value("lowest_numerical_precision_in_linear"), "mxfp6")
        self.assertEqual(rec.value("lowest_numerical_precision_in_attn"), "fp8")
        self.assertIn(rec.value("lowest_numerical_precision_in_linear"), ALLOWED_PRECISIONS)

    def test_values_pass_checker_rules(self) -> None:
        for env in (FP8_ON_ENV, {}, dict(FP8_ON_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")):
            rec = self._run_start(env, DEFAULT_GIN)
            for key in NEW_KEYS[:3]:
                self.assertIn(rec.value(key), ALLOWED_PRECISIONS)
            for key in NEW_KEYS[3:8]:
                v = rec.value(key)
                self.assertTrue(isinstance(v, int) and v >= 1, key)
            self.assertNotEqual(rec.value("config_filename"), "")

    def test_fp8_off_logs_bfloat16(self) -> None:
        rec = self._run_start({"DLRM_HSTU_FP8_FWD": "0"}, DEFAULT_GIN)
        self.assertEqual(rec.value("lowest_numerical_precision_in_linear"), "bfloat16")
        self.assertEqual(rec.value("lowest_numerical_precision_in_attn"), "bfloat16")
        self.assertEqual(rec.value("lowest_numerical_precision_in_comm"), "fp16")

    def test_fwd_bwd_opt_out_with_gin_fp8_linear_logs_fp8_linear(self) -> None:
        # FWD/BWD off, E-M3a + uvqk backward still bound on by the gin.
        rec = self._run_start(FWD_BWD_OFF_ENV, DEFAULT_GIN)
        self.assertEqual(rec.value("lowest_numerical_precision_in_linear"), "fp8")
        self.assertEqual(rec.value("lowest_numerical_precision_in_attn"), "bfloat16")
        self.assertEqual(rec.value("lowest_numerical_precision_in_comm"), "fp16")

    def test_micro_batch_derived_when_not_passed(self) -> None:
        lg, rec = _logger()
        with _gin(DEFAULT_GIN):
            lg.log_run_start(global_batch_size=8192, seed=1, config_filename="a.gin")
        # No process group in the test: world size 1.
        self.assertEqual(rec.value("micro_batch_size"), 8192)

    def test_non_zero_rank_emits_nothing(self) -> None:
        lg, rec = _logger(rank=3)
        with _gin(DEFAULT_GIN):
            lg.log_run_start(
                global_batch_size=8192, seed=1, micro_batch_size=1024,
                config_filename="yambda_5b.gin",
            )
        self.assertEqual(rec.calls, [])

    def test_real_mllog_file_has_each_key_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlperf.log")
            lg = mlu.MLPerfLogger(rank=0, log_path=path, submission_platform=None)
            with _gin(DEFAULT_GIN), mock.patch.dict(os.environ, FP8_ON_ENV), \
                    mock.patch.object(mlu, "sys", types.SimpleNamespace(modules={})), \
                    mock.patch.object(
                        mlu, "detect_submission_platform", return_value="MI350X"
                    ):
                lg.log_run_start(
                    global_batch_size=8192, seed=1, micro_batch_size=1024,
                    config_filename="yambda_5b.gin",
                )
            # Detach mllog's handler for this file before the dir is removed.
            import logging
            ml = logging.getLogger("mllog_default")
            for h in list(ml.handlers):
                if getattr(h, "baseFilename", "") == path:
                    ml.removeHandler(h)
                    h.close()
            recs = []
            with open(path) as f:
                for line in f:
                    if ":::MLLOG" in line:
                        recs.append(json.loads(line.split(":::MLLOG", 1)[1]))
        keys = [r["key"] for r in recs]
        for key in NEW_KEYS:
            self.assertEqual(keys.count(key), 1, key)
            self.assertLess(keys.index(key), keys.index("run_start"))
        plat = [r["value"] for r in recs if r["key"] == "submission_platform"]
        self.assertEqual(plat, ["MI350X"])


class ResolvePrecisionTest(unittest.TestCase):
    def _resolve(self, env, gin_values=DEFAULT_GIN):
        with _gin(gin_values):
            return mlu.resolve_numerical_precisions(env=env)

    def test_default_arm(self) -> None:
        self.assertEqual(
            self._resolve(FP8_ON_ENV),
            {"linear": "fp8", "attn": "fp8", "comm": "fp16"},
        )

    def test_attention_opt_out_keeps_fp8_linear(self) -> None:
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_ATTN="0")
        p = self._resolve(env)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))

    def test_attention_needs_fused_qkv(self) -> None:
        # Mirrors triton_fp8: _HSTU_FP8_ATTENTION requires FUSE_LN and FUSE_QKV.
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_FUSE_QKV="0")
        self.assertEqual(self._resolve(env)["attn"], "bfloat16")

    def test_attention_without_gemm_is_not_fp8(self) -> None:
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_GEMM="0", DLRM_HSTU_FP8_BWD="0")
        p = self._resolve(env)
        self.assertEqual((p["linear"], p["attn"]), ("bfloat16", "bfloat16"))

    def test_fp8_backward_dgrad_alone_lowers_linear(self) -> None:
        # bwd_fp8.output_projection_dgrad is gated on DLRM_HSTU_FP8_BWD only,
        # so FWD=1 with a pre-set GEMM=0 still runs an e4m3 linear GEMM.
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_GEMM="0")
        p = self._resolve(env)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))

    def test_live_bwd_module_flag_read(self) -> None:
        live = types.SimpleNamespace(_HSTU_FP8_GEMM=False, _HSTU_FP8_ATTENTION=False)
        live_bwd = types.SimpleNamespace(_BWD_ENABLED=True)
        with _gin(DEFAULT_GIN), mock.patch.dict(
            sys.modules,
            {mlu._TRITON_FP8_MODULE: live, mlu._BWD_FP8_MODULE: live_bwd},
        ), mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_BWD": "0"}):
            p = mlu.resolve_numerical_precisions()
        self.assertEqual(p["linear"], "fp8")

    def test_fp32_model_and_comm(self) -> None:
        g = {
            "make_model.bf16_training": False,
            "make_optimizer_and_shard.sparse_a2a_forward_precision": "fp32",
            "make_optimizer_and_shard.sparse_a2a_backward_precision": "fp32",
        }
        self.assertEqual(
            self._resolve({}, g), {"linear": "fp32", "attn": "fp32", "comm": "fp32"}
        )

    def test_comm_lowest_of_fwd_bwd(self) -> None:
        g = dict(DEFAULT_GIN)
        g["make_optimizer_and_shard.sparse_a2a_backward_precision"] = "bf16"
        self.assertEqual(self._resolve({}, g)["comm"], "bfloat16")
        g["make_optimizer_and_shard.sparse_a2a_forward_precision"] = "fp32"
        g["make_optimizer_and_shard.sparse_a2a_backward_precision"] = "fp16"
        self.assertEqual(self._resolve({}, g)["comm"], "fp16")

    def test_unknown_comm_precision_raises(self) -> None:
        g = dict(DEFAULT_GIN)
        g["make_optimizer_and_shard.sparse_a2a_forward_precision"] = "int8"
        with self.assertRaises(ValueError):
            self._resolve({}, g)

    def test_live_module_flags_win_when_env_not_given(self) -> None:
        live = types.SimpleNamespace(_HSTU_FP8_GEMM=True, _HSTU_FP8_ATTENTION=False)
        with _gin(DEFAULT_GIN), mock.patch.dict(
            sys.modules, {mlu._TRITON_FP8_MODULE: live}
        ), mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_GEMM": "0"}):
            p = mlu.resolve_numerical_precisions()
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))

    # FP8 linear paths outside the HSTU projection arm (mlu._FP8_LINEAR_PATHS).

    def test_production_gin_env_unchanged(self) -> None:
        env = dict(FP8_ON_ENV, **GIN_FP8_LINEAR_ENV)
        self.assertEqual(
            self._resolve(env), {"linear": "fp8", "attn": "fp8", "comm": "fp16"}
        )

    def test_fwd_bwd_opt_out_keeps_gin_fp8_linear(self) -> None:
        # The bootstrap's bf16 opt-out leaves E-M3a and the uvqk backward on.
        p = self._resolve(FWD_BWD_OFF_ENV)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))
        self.assertEqual(
            mlu.armed_fp8_linear_paths(FWD_BWD_OFF_ENV), ("preproc_l2", "uvqk_bwd")
        )

    def test_preproc_l2_alone_lowers_linear(self) -> None:
        env = dict(
            ALL_OFF_ENV,
            DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL="1",
            DLRM_HSTU_FP8_PREPROC_L2="1",
        )
        p = self._resolve(env)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))
        self.assertEqual(mlu.armed_fp8_linear_paths(env), ("preproc_l2",))

    def test_preproc_l2_needs_grouped_tail(self) -> None:
        # E-M3a runs only inside the E-M1 grouped tail.
        env = dict(ALL_OFF_ENV, DLRM_HSTU_FP8_PREPROC_L2="1")
        self.assertEqual(self._resolve(env)["linear"], "bfloat16")
        self.assertEqual(mlu.armed_fp8_linear_paths(env), ())

    def test_uvqk_backward_alone_lowers_linear(self) -> None:
        env = dict(ALL_OFF_ENV, DLRM_HSTU_FP8_BWD_UVQK="1")
        p = self._resolve(env)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))
        self.assertEqual(mlu.armed_fp8_linear_paths(env), ("uvqk_bwd",))

    def test_all_fp8_off_logs_base(self) -> None:
        self.assertEqual(
            self._resolve(ALL_OFF_ENV),
            {"linear": "bfloat16", "attn": "bfloat16", "comm": "fp16"},
        )

    @staticmethod
    def _live(l2_enabled, grouped_tail=True, uvqk=False):
        """Every owning module imported, the HSTU projection arm off."""
        return {
            mlu._TRITON_FP8_MODULE: types.SimpleNamespace(
                _HSTU_FP8_GEMM=False, _HSTU_FP8_ATTENTION=False
            ),
            mlu._BWD_FP8_MODULE: types.SimpleNamespace(
                _BWD_ENABLED=False, uvqk_fp8_backward_enabled=lambda: uvqk
            ),
            "generative_recommenders.modules.preprocessors": types.SimpleNamespace(
                _PREPROC_GROUPED_TAIL=grouped_tail
            ),
            "generative_recommenders.ops.preproc_fp8_l2": types.SimpleNamespace(
                enabled=l2_enabled
            ),
        }

    def _resolve_live(self, modules, env, absent=()):
        with _gin(DEFAULT_GIN), mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, env):
            for name in absent:
                sys.modules.pop(name, None)
            return mlu.resolve_numerical_precisions()

    def test_live_preproc_l2_gate_wins_over_env(self) -> None:
        # The module's resolved gate decides: armed with the env switch "0" ...
        p = self._resolve_live(self._live(lambda: True), ALL_OFF_ENV)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))
        # ... and declined (e.g. not a HIP build) with every env switch "1".
        p = self._resolve_live(
            self._live(lambda: False), dict(ALL_OFF_ENV, **GIN_FP8_LINEAR_ENV)
        )
        self.assertEqual(p["linear"], "bfloat16")

    def test_live_grouped_tail_off_never_asks_preproc_l2(self) -> None:
        calls: List[int] = []

        def enabled() -> bool:
            calls.append(1)
            return True

        p = self._resolve_live(
            self._live(enabled, grouped_tail=False),
            dict(ALL_OFF_ENV, **GIN_FP8_LINEAR_ENV),
        )
        self.assertEqual(p["linear"], "bfloat16")
        self.assertEqual(calls, [])

    def test_live_uvqk_gate_read(self) -> None:
        p = self._resolve_live(self._live(lambda: False, uvqk=True), ALL_OFF_ENV)
        self.assertEqual((p["linear"], p["attn"]), ("fp8", "bfloat16"))

    def test_preproc_l2_env_fallback_when_not_imported(self) -> None:
        # preproc_fp8_l2 is imported lazily (first forward, or the pre-RUN_START
        # BLAS warm-up); until then the env switch decides.
        p = self._resolve_live(
            self._live(lambda: False),
            dict(ALL_OFF_ENV, DLRM_HSTU_FP8_PREPROC_L2="1"),
            absent=("generative_recommenders.ops.preproc_fp8_l2",),
        )
        self.assertEqual(p["linear"], "fp8")

    def test_ix_fp8f_row_needs_ix_and_its_switch(self) -> None:
        # IX-FP8F runs only inside IX: both switches must be "1" (env path).
        self.assertIn("ix_fp8f", [name for name, _ in mlu._FP8_LINEAR_PATHS])
        env = dict(ALL_OFF_ENV, **IX_FP8F_ENV)
        self.assertEqual(
            self._resolve(env), {"linear": "fp8", "attn": "bfloat16", "comm": "fp16"}
        )
        self.assertIn("ix_fp8f", mlu.armed_fp8_linear_paths(env))
        for k in IX_FP8F_ENV:  # either one off: IX alone is bf16
            self.assertEqual(
                self._resolve(dict(env, **{k: "0"}))["linear"], "bfloat16", k
            )
        # On top of the production FP8 forward it changes nothing.
        self.assertEqual(
            self._resolve(dict(FP8_ON_ENV, **IX_FP8F_ENV)),
            {"linear": "fp8", "attn": "fp8", "comm": "fp16"},
        )

    def test_ix_fp8f_live_gates_win(self) -> None:
        on_env = dict(ALL_OFF_ENV, **IX_FP8F_ENV)
        mods = dict(
            self._live(lambda: False),
            **{
                _IX_MODULE: types.SimpleNamespace(enabled=lambda: True),
                _IX_FP8F_MODULE: types.SimpleNamespace(fp8f_enabled=lambda: False),
            },
        )
        # FP8F declined at runtime: bf16 even with both switches "1".
        self.assertEqual(self._resolve_live(mods, on_env)["linear"], "bfloat16")
        # Armed: fp8 even with both switches "0".
        mods[_IX_FP8F_MODULE] = types.SimpleNamespace(fp8f_enabled=lambda: True)
        self.assertEqual(self._resolve_live(mods, ALL_OFF_ENV)["linear"], "fp8")
        # IX itself off: bf16.
        mods[_IX_MODULE] = types.SimpleNamespace(enabled=lambda: False)
        self.assertEqual(self._resolve_live(mods, on_env)["linear"], "bfloat16")

    def test_ix_fp8f_hook_row(self) -> None:
        # A future IX-FP8F switch needs only a row in _FP8_LINEAR_PATHS.
        row = (
            "ix_fp8f",
            (("DLRM_TEST_IX_FP8F", "test_only.ix_fp8f", "fp8f_enabled"),),
        )
        with mock.patch.object(
            mlu, "_FP8_LINEAR_PATHS", mlu._FP8_LINEAR_PATHS + (row,)
        ):
            env = dict(ALL_OFF_ENV, DLRM_TEST_IX_FP8F="1")
            self.assertEqual(self._resolve(env)["linear"], "fp8")
            self.assertEqual(mlu.armed_fp8_linear_paths(env), ("ix_fp8f",))
            self.assertEqual(self._resolve(ALL_OFF_ENV)["linear"], "bfloat16")
            # Imported: the module's gate wins over its env switch.
            mods = dict(
                self._live(lambda: False),
                **{"test_only.ix_fp8f": types.SimpleNamespace(fp8f_enabled=False)},
            )
            p = self._resolve_live(mods, env)
            self.assertEqual(p["linear"], "bfloat16")

    # MXFP6 forward K512 trio (ops/hip/mx6_fwd.py; DLRM_HSTU_FP8_FWD_MXFP6,
    # default off): the linear disclosure must say mxfp6, never fp8.
    def test_mxfp6_trio_lowers_linear(self) -> None:
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(
            self._resolve(env), {"linear": "mxfp6", "attn": "fp8", "comm": "fp16"}
        )

    def test_mxfp6_off_by_default(self) -> None:
        for v in ("0", "", "true"):
            env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_FWD_MXFP6=v)
            self.assertEqual(self._resolve(env)["linear"], "fp8", v)

    def test_mxfp6_needs_the_fp8_forward(self) -> None:
        # The trio replaces FP8 forward GEMMs only: with GEMM=0 it never runs, and
        # the e4m3 backward dgrad keeps the answer at fp8.
        env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_GEMM="0", DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(self._resolve(env)["linear"], "fp8")

    def test_live_mx6_module_read(self) -> None:
        live = types.SimpleNamespace(_HSTU_FP8_GEMM=True, _HSTU_FP8_ATTENTION=True)
        live_mx6 = types.SimpleNamespace(requested=lambda: True)
        with _gin(DEFAULT_GIN), mock.patch.dict(
            sys.modules, {mlu._TRITON_FP8_MODULE: live, mlu._MX6_FWD_MODULE: live_mx6}
        ), mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_FWD_MXFP6": "0"}):
            p = mlu.resolve_numerical_precisions()
        self.assertEqual((p["linear"], p["attn"]), ("mxfp6", "fp8"))
        live_off = types.SimpleNamespace(requested=lambda: False)
        with _gin(DEFAULT_GIN), mock.patch.dict(
            sys.modules, {mlu._TRITON_FP8_MODULE: live, mlu._MX6_FWD_MODULE: live_off}
        ), mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_FWD_MXFP6": "1"}):
            p = mlu.resolve_numerical_precisions()
        self.assertEqual(p["linear"], "fp8")

    def test_mxfp6_spelling_passes_checker_and_ranks_lowest(self) -> None:
        self.assertIn("mxfp6", ALLOWED_PRECISIONS)
        self.assertEqual(mlu._lowest("fp32", "mxfp6", "fp8", "bfloat16"), "mxfp6")
        self.assertEqual(mlu._lowest("fp16", "bfloat16"), "bfloat16")

    def test_mxfp6_with_the_gin_fp8_linear_paths(self) -> None:
        # r3 (ship14 resolver v2): the other FP8 linear paths keep linear at fp8
        # without the FP8 forward; with it, mxfp6 is still the lowest.
        env = dict(FP8_ON_ENV, **GIN_FP8_LINEAR_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(self._resolve(env)["linear"], "mxfp6")
        env = dict(FWD_BWD_OFF_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(self._resolve(env)["linear"], "fp8")
        env = dict(ALL_OFF_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(self._resolve(env)["linear"], "bfloat16")

    def test_mxfp6_below_ix_fp8f(self) -> None:
        # r4 (F14 resolver = v2 + P4r): IX-FP8F is an fp8 linear path; with the
        # FP8 forward and the MX6 knob mxfp6 is still the lowest; without the FP8
        # forward the trio cannot run and IX-FP8F keeps linear at fp8.
        env = dict(FP8_ON_ENV, **IX_FP8F_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(self._resolve(env)["linear"], "mxfp6")
        env = dict(ALL_OFF_ENV, **IX_FP8F_ENV, DLRM_HSTU_FP8_FWD_MXFP6="1")
        self.assertEqual(
            self._resolve(env), {"linear": "fp8", "attn": "bfloat16", "comm": "fp16"}
        )

    def test_mxfp6_env_parse_matches_the_gates(self) -> None:
        # r4 PARSE1: the env path reads the knob as the MX6 gates do (exactly "1");
        # " 1" / "1 " / "on" leave every MX6 site off, so linear stays fp8.
        for v in (" 1", "1 ", "on", "yes"):
            env = dict(FP8_ON_ENV, DLRM_HSTU_FP8_FWD_MXFP6=v)
            self.assertEqual(self._resolve(env)["linear"], "fp8", repr(v))


class Fp8LinearPathsContractTest(unittest.TestCase):
    """Every _FP8_LINEAR_PATHS gate names a real switch and module attribute.

    Reads the owning modules' source (find_spec + ast: nothing is imported or
    run), so a renamed gate fails here instead of silently falling back to the
    env switch.
    """

    def test_gates_exist_in_source(self) -> None:
        import ast
        import importlib.util

        self.assertTrue(mlu._FP8_LINEAR_PATHS)
        for name, gates in mlu._FP8_LINEAR_PATHS:
            self.assertTrue(gates, name)
            for switch, module, attr in gates:
                spec = importlib.util.find_spec(module)
                self.assertIsNotNone(spec, module)
                with open(spec.origin) as f:
                    src = f.read()
                self.assertIn(f'"{switch}"', src, (name, module))
                top = set()
                for node in ast.parse(src).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        top.add(node.name)
                    elif isinstance(node, ast.Assign):
                        top.update(
                            t.id for t in node.targets if isinstance(t, ast.Name)
                        )
                    elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name
                    ):
                        top.add(node.target.id)
                self.assertIn(attr, top, (name, module))


class PlatformTest(unittest.TestCase):
    def test_name_mapping(self) -> None:
        f = mlu._platform_from_device
        self.assertEqual(f("AMD Instinct MI350X"), "MI350X")
        self.assertEqual(f("AMD Instinct MI355X"), "MI355X")
        self.assertEqual(f("AMD Instinct MI300X"), "MI300X")
        self.assertEqual(f("NVIDIA B200"), "B200")
        self.assertEqual(f("", "gfx950:sramecc+:xnack-"), "AMD gfx950")
        self.assertEqual(f(""), "unknown")

    def test_detect_reads_current_device(self) -> None:
        props = types.SimpleNamespace(name="AMD Instinct MI350X", gcnArchName="gfx950")
        cuda = mlu.torch.cuda
        with mock.patch.object(cuda, "is_available", return_value=True), \
                mock.patch.object(cuda, "current_device", return_value=5), \
                mock.patch.object(
                    cuda, "get_device_properties", return_value=props
                ) as gdp:
            self.assertEqual(mlu.detect_submission_platform(), "MI350X")
        gdp.assert_called_once_with(5)

    def test_detect_never_empty_on_failure(self) -> None:
        cuda = mlu.torch.cuda
        with mock.patch.object(cuda, "is_available", side_effect=RuntimeError("x")):
            self.assertEqual(mlu.detect_submission_platform(), "unknown")

    @unittest.skipUnless(mlu._MLLOG_AVAILABLE, "mlperf_logging not installed")
    def test_override_order(self) -> None:
        with mock.patch.dict(os.environ, {"MLPERF_LOGGING": "1"}):
            os.environ.pop("MLPERF_SUBMISSION_PLATFORM", None)
            # Default: empty -> detected at submission_info().
            lg = mlu.get_mlperf_logger(rank=0)
            self.assertIsNone(lg.submission_platform)
            lg._logger = rec = _Recorder()
            with mock.patch.object(
                mlu, "detect_submission_platform", return_value="MI350X"
            ) as det:
                lg.submission_info()
            det.assert_called_once()
            self.assertEqual(rec.value("submission_platform"), "MI350X")
            # gin binding beats detection.
            lg = mlu.get_mlperf_logger(rank=0, submission_platform="MI355X")
            self.assertEqual(lg.submission_platform, "MI355X")
            # Env beats gin.
            os.environ["MLPERF_SUBMISSION_PLATFORM"] = "8xMI350X"
            lg = mlu.get_mlperf_logger(rank=0, submission_platform="MI355X")
            self.assertEqual(lg.submission_platform, "8xMI350X")


if __name__ == "__main__":
    unittest.main()
