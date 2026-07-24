###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ODC (On-Demand Communication) integration tests.

Covers the feature merged in PR #864 (``feat: odc adapt``):
  * ``primus/core/odc/runtime_config.py`` -- the config-driven runtime knobs the
    ODC library reads instead of the former ``ODC_*`` env vars.
  * the before_train patch wiring (``odc_torch_fsdp2_patches``,
    ``odc_lb_mini_patches``, ``distributed_init_patches``) -- registration and
    the ``enable_odc`` / ``use_torch_fsdp2`` / ``enable_odc_lb_mini`` gating.
  * the ODC training-config fixture (``test_odc_megatron_trainer.yaml``).
  * an end-to-end training smoke test on the ODC FSDP2 path.

Hardware handling (matches the neighbor trainer tests):
  * The pure config layer (``TestOdcRuntimeConfig``) is stdlib-only and runs
    CPU-only, always -- it needs neither torch nor a GPU.
  * The patch-wiring layer (``TestOdcPatchWiring``) needs ``torch`` importable
    (NOT a GPU -- the patch modules import torch at module scope but only touch
    CUDA at runtime); it skips cleanly where torch is absent.
  * The end-to-end run (``TestOdcMegatronTrainerE2E``) needs ROCm GPUs + a
    Primus-Turbo build with the rocSHMEM ops, so it skips with a clear reason
    when the hardware / turbo stack is unavailable -- the same GPU-gating idea as
    ``TestProjectionSimulate._require_2_gpus`` in test_megatron_trainer.py.
"""

import dataclasses
import importlib
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace

from tests.utils import PrimusUT, run_training_script

# Repo root: tests/trainer/<this file> -> repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The ``odc`` package lives at primus/core/odc/, imported as top-level ``odc``
# with primus/core on the path (exactly what run_odc.sh wires into PYTHONPATH).
_PRIMUS_CORE = os.path.join(_REPO_ROOT, "primus", "core")
if _PRIMUS_CORE not in sys.path:
    sys.path.insert(0, _PRIMUS_CORE)

_HAS_TORCH = importlib.util.find_spec("torch") is not None

_ODC_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_odc_megatron_trainer.yaml")


def _make_ctx(**params):
    """Build a minimal PatchContext whose get_args(ctx) returns the given params.

    Mirrors the real call site: get_args reads ctx.extra['module_config'].params.
    """
    from primus.core.patches import PatchContext

    module_config = SimpleNamespace(params=SimpleNamespace(**params))
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": module_config},
    )


class TestOdcRuntimeConfig(PrimusUT):
    """CPU-only unit tests for primus/core/odc/runtime_config.py.

    This is the config-driven layer that the ODC before_train patch populates
    (via ``odc.set_runtime_config(...)``) so the decoupled ODC primitives read
    their tuning knobs from config, not os.environ. Stdlib-only -> no torch/GPU.
    """

    def setUp(self):
        import odc

        # The runtime config is a process-wide singleton; reset it to defaults
        # before each test so ordering never leaks state.
        cfg = odc.get_runtime_config()
        for f in dataclasses.fields(odc.OdcRuntimeConfig):
            setattr(cfg, f.name, f.default)
        os.environ.pop("PRIMUS_TURBO_ODC_GDA_PIPE", None)

    def test_defaults_match_documented_env_defaults(self):
        # Defaults are byte-for-byte the previous ODC_* env defaults and match
        # the trainer_base.yaml odc_* defaults; assert the contract explicitly.
        import odc

        cfg = odc.get_runtime_config()
        self.assertEqual(cfg.p2p_backend, "mori")
        self.assertEqual(cfg.mori_init, "pg")
        self.assertEqual(cfg.max_buffer_size, 64 * 1024 * 1024)  # 67108864
        self.assertFalse(cfg.rocshmem_gda)
        self.assertIsNone(cfg.rocshmem_lib)
        self.assertEqual(cfg.gda_rs_blocks, 64)
        self.assertEqual(cfg.gda_pipe, 1)
        self.assertEqual(cfg.gda_defer_reduce, "auto")
        self.assertEqual(cfg.gda_warmup_mode, "strided")
        self.assertEqual(cfg.gda_stride_bytes, 65536)

    def test_set_config_applies_known_overrides(self):
        import odc

        returned = odc.set_runtime_config(
            p2p_backend="rocshmem",
            rocshmem_gda=True,
            gda_warmup_mode="hdp",
            gda_rs_blocks=128,
        )
        cfg = odc.get_runtime_config()
        # set_config returns the same singleton it mutates in place.
        self.assertIs(returned, cfg)
        self.assertEqual(cfg.p2p_backend, "rocshmem")
        self.assertTrue(cfg.rocshmem_gda)
        self.assertEqual(cfg.gda_warmup_mode, "hdp")
        self.assertEqual(cfg.gda_rs_blocks, 128)

    def test_set_config_ignores_none_and_unknown_keys(self):
        import odc

        odc.set_runtime_config(p2p_backend="rocshmem")
        # None must NOT clobber a previously-set value (keeps prior/default), and
        # unknown keys are silently ignored -- this is what lets the patch forward
        # every odc_* config item, unset ones as None, without wiping defaults.
        odc.set_runtime_config(p2p_backend=None, not_a_real_field="boom")
        cfg = odc.get_runtime_config()
        self.assertEqual(cfg.p2p_backend, "rocshmem")
        self.assertFalse(hasattr(cfg, "not_a_real_field"))

    def test_gda_pipe_bridged_to_turbo_env(self):
        # gda_pipe is the one knob that must stay visible to the Primus-Turbo C++
        # device kernel (read via getenv), so set_config bridges it back to the
        # PRIMUS_TURBO_ODC_GDA_PIPE env var.
        import odc

        odc.set_runtime_config(gda_pipe=4)
        self.assertEqual(os.environ.get("PRIMUS_TURBO_ODC_GDA_PIPE"), "4")

    def test_odc_package_reexports_runtime_config_api(self):
        # The before_train patch depends on `import odc` exposing the runtime
        # config API WITHOUT importing the heavy primitives (torch/triton/mori).
        import odc
        import odc.runtime_config as rc

        self.assertIs(odc.set_runtime_config, rc.set_config)
        self.assertIs(odc.get_runtime_config, rc.get_config)
        # Importing odc must not eagerly pull in the heavy primitives (whose
        # import-time backend selection must run AFTER set_runtime_config).
        self.assertNotIn("odc.primitives.scatter_accumulate", sys.modules)


@unittest.skipUnless(_HAS_TORCH, "ODC patch modules import torch at module scope")
class TestOdcPatchWiring(PrimusUT):
    """CPU-only patch-registration / gating tests.

    Imports the ODC before_train patch modules (which need torch importable, but
    NOT a GPU) and asserts that each patch is registered and that its condition
    gates exactly on the ODC config items. No training is run.
    """

    def _import_patches(self, dotted: str):
        try:
            return importlib.import_module(dotted)
        except Exception as e:  # noqa: BLE001 -- torch present but megatron stack absent
            self.skipTest(f"cannot import {dotted} ({type(e).__name__}: {e})")

    def test_odc_fsdp2_patch_registered_and_gated(self):
        self._import_patches("primus.backends.megatron.patches.odc_torch_fsdp2_patches")
        from primus.core.patches import PatchRegistry

        patch = PatchRegistry.get("megatron.fsdp.odc_torch_fsdp2")
        self.assertIsNotNone(patch, "ODC FSDP2 patch must be registered")
        self.assertEqual(patch.backend, "megatron")
        self.assertEqual(patch.phase, "before_train")

        # Off by default (no enable_odc).
        self.assertFalse(patch.applies_to(_make_ctx(use_torch_fsdp2=True)))
        # enable_odc alone is not enough -- ODC requires the torch-FSDP2 path.
        self.assertFalse(patch.applies_to(_make_ctx(enable_odc=True, use_torch_fsdp2=False)))
        # Both set -> ODC integrates.
        self.assertTrue(patch.applies_to(_make_ctx(enable_odc=True, use_torch_fsdp2=True)))

    def test_odc_lb_mini_patch_gated_independently_of_enable_odc(self):
        mod = self._import_patches("primus.backends.megatron.patches.odc_lb_mini_patches")
        from primus.core.patches import PatchRegistry

        patch = PatchRegistry.get("megatron.fsdp.odc_lb_mini")
        self.assertIsNotNone(patch, "ODC LB-Mini patch must be registered")

        # LB-Mini serving is gated by enable_odc_lb_mini + use_torch_fsdp2 ALONE,
        # orthogonal to the enable_odc comm switch.
        self.assertFalse(patch.applies_to(_make_ctx(use_torch_fsdp2=True)))
        self.assertFalse(patch.applies_to(_make_ctx(enable_odc_lb_mini=True, use_torch_fsdp2=False)))
        self.assertTrue(patch.applies_to(_make_ctx(enable_odc_lb_mini=True, use_torch_fsdp2=True)))
        # enable_odc is NOT required for LB-Mini to install.
        self.assertTrue(
            patch.applies_to(_make_ctx(enable_odc_lb_mini=True, use_torch_fsdp2=True, enable_odc=False))
        )

        # ...but enable_odc selects the micro-batch alignment MODE: OFF -> ALIGNED
        # (NCCL-safe same-count baseline), ON -> DECOUPLED (per-rank counts).
        self.assertTrue(mod._lb_mini_aligned(SimpleNamespace(enable_odc=False)))
        self.assertFalse(mod._lb_mini_aligned(SimpleNamespace(enable_odc=True)))

    def test_device_id_patch_skipped_under_odc(self):
        # #856's eager-RCCL device_id injection must be SKIPPED when ODC is on
        # (ODC drives gradient exchange over rocSHMEM P2P; the eager RCCL comms
        # would serialize its XGMI copy streams). Assert the condition flips.
        self._import_patches("primus.backends.megatron.patches.distributed_init_patches")
        from primus.core.patches import PatchRegistry

        patch = PatchRegistry.get("megatron.distributed.init_process_group_device_id")
        self.assertIsNotNone(patch, "device_id init patch must be registered")
        # FSDP2 without ODC -> device_id patch applies.
        self.assertTrue(patch.applies_to(_make_ctx(use_torch_fsdp2=True, enable_odc=False)))
        # FSDP2 with ODC -> device_id patch is skipped.
        self.assertFalse(patch.applies_to(_make_ctx(use_torch_fsdp2=True, enable_odc=True)))

    def test_populate_runtime_config_from_trainer_config(self):
        # The bridge that copies odc_* trainer-config items into the ODC library
        # runtime config at before_train. Exercises the real function end to end.
        import odc

        mod = self._import_patches("primus.backends.megatron.patches.odc_torch_fsdp2_patches")

        cfg = odc.get_runtime_config()
        for f in dataclasses.fields(odc.OdcRuntimeConfig):
            setattr(cfg, f.name, f.default)
        os.environ.pop("PRIMUS_TURBO_ODC_GDA_PIPE", None)

        ctx = _make_ctx(
            odc_p2p_backend="rocshmem",
            odc_rocshmem_gda=True,
            odc_gda_warmup_mode="hdp",
            odc_gda_pipe=3,
            odc_gda_defer_reduce=1,
        )
        mod._populate_odc_runtime_config(ctx)

        cfg = odc.get_runtime_config()
        self.assertEqual(cfg.p2p_backend, "rocshmem")
        self.assertTrue(cfg.rocshmem_gda)
        self.assertEqual(cfg.gda_warmup_mode, "hdp")
        self.assertEqual(cfg.gda_pipe, 3)
        # defer_reduce is stringified by the bridge (config int -> library str).
        self.assertEqual(cfg.gda_defer_reduce, "1")
        # ...and gda_pipe is mirrored to the turbo C++ env.
        self.assertEqual(os.environ.get("PRIMUS_TURBO_ODC_GDA_PIPE"), "3")


class TestOdcConfigFixture(PrimusUT):
    """CPU-only assertions on the ODC training-config fixture.

    Verifies the fixture actually declares the ODC path so a real-GPU run would
    exercise it (and documents the required co-settings), without needing the
    heavy Primus/megatron config stack.
    """

    def _load_overrides(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.assertTrue(os.path.exists(_ODC_FIXTURE), f"missing ODC fixture: {_ODC_FIXTURE}")
        with open(_ODC_FIXTURE) as f:
            cfg = yaml.safe_load(f)
        return cfg["modules"]["pre_trainer"]["overrides"]

    def test_fixture_enables_odc_and_lb_mini_on_fsdp2(self):
        ov = self._load_overrides()
        # ODC comm on, full wiring depth, LB-Mini on -- the feature under test.
        self.assertIs(ov["enable_odc"], True)
        self.assertEqual(ov["odc_phase"], 2)
        self.assertIs(ov["enable_odc_lb_mini"], True)
        # ODC + LB-Mini require the torch-FSDP2 path (the patch condition).
        self.assertIs(ov["use_torch_fsdp2"], True)
        # A valid symmetric-memory backend must be selected.
        self.assertIn(ov["odc_p2p_backend"], ("mori", "rocshmem"))

    def test_fixture_uses_odc_compatible_fsdp2_settings(self):
        ov = self._load_overrides()
        # ODC replaces the collective reduce-scatter, so these must be off (they
        # would otherwise conflict with / be redundant to ODC's transport).
        self.assertIs(ov["use_distributed_optimizer"], False)
        self.assertIs(ov["overlap_grad_reduce"], False)
        self.assertIs(ov["overlap_param_gather"], False)
        # Keep the model tiny so a real-GPU run is cheap.
        self.assertLessEqual(int(ov["train_iters"]), 5)


_GFX_TO_PLATFORM = {"gfx942": "MI300X", "gfx950": "MI355X"}


class TestOdcMegatronTrainerE2E(PrimusUT):
    """End-to-end ODC training smoke test on the torch-FSDP2 path.

    ODC needs ROCm GPUs + a Primus-Turbo build carrying the rocSHMEM ops, so this
    cannot run in plain CPU CI -- it skips with a clear reason when CUDA / turbo
    are unavailable (same GPU-gating pattern as the neighbor trainer tests).
    """

    def _require_odc_hardware(self):
        if not _HAS_TORCH:
            self.skipTest("torch not available")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            self.skipTest("ODC needs a ROCm GPU; none available")
        # ODC's rocSHMEM ops must be COMPILED INTO Primus-Turbo -- a stock/pip
        # primus_turbo is built with -DDISABLE_ROCSHMEM and carries NEITHER op, so a
        # bare find_spec("primus_turbo") is not enough (a "GPU present but turbo lacks
        # ODC ops" box would crash mid-run instead of skipping). Verify the op this
        # fixture actually needs is present. This single-node (TP/PP/EP=1) rocshmem
        # run uses the host/XGMI-IPC op (odc_rocshmem_host); a multi-node GDA run
        # would additionally require odc_rocshmem_gda. Build a host-capable
        # Primus-Turbo from source against rocSHMEM per the repro guide (§6.3) to run.
        try:
            import primus_turbo.pytorch._C as _turbo_c
        except Exception as e:  # noqa: BLE001 -- any import failure -> not runnable here
            self.skipTest(f"Primus-Turbo not importable ({type(e).__name__}: {e})")
        if not hasattr(_turbo_c, "odc_rocshmem_host"):
            self.skipTest(
                "Primus-Turbo lacks the ODC rocSHMEM host op (odc_rocshmem_host); build "
                "Primus-Turbo from source against rocSHMEM (repro guide §6.3)"
            )
        props = torch.cuda.get_device_properties(0)
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0].strip()
        platform = _GFX_TO_PLATFORM.get(arch)
        if platform is None:
            self.skipTest(f"unrecognized GPU arch {arch!r}; ODC validated on gfx942/gfx950")
        return platform

    def test_odc_fsdp2_pretrain_smoke(self):
        self._require_odc_hardware()

        env = os.environ.copy()
        env["EXP"] = _ODC_FIXTURE
        ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
        train_log_path = os.path.join(ut_log_path, "log.test_odc_megatron_trainer-fsdp2_smoke.txt")
        env["TRAIN_LOG"] = train_log_path
        # ODC's XGMI/rocSHMEM P2P bootstraps over the loopback iface on a single
        # node; mirror the launcher's infra env.
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")

        cmd = [
            "bash",
            "./runner/primus-cli",
            "direct",
            "--log_file",
            train_log_path,
            "--",
            "train",
            "pretrain",
            "--config",
            _ODC_FIXTURE,
            "--num_layers",
            "2",
            "--train_iters",
            "3",
        ]
        stdout, _ = run_training_script(
            tag="odc_fsdp2_smoke", cmd=cmd, train_log_path=train_log_path, env=env
        )
        self.assertIn("Training completed.", stdout)
        # ODC's before_train patch must have wired itself in.
        self.assertIn("[ODC.torch_fsdp2]", stdout)


if __name__ == "__main__":
    unittest.main(buffer=False)
