###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace
import os

from primus.backends.megatron.patches import args_patches  # noqa: F401
from primus.core.patches import PatchRegistry, PatchContext, run_patches
from tests.utils import PrimusUT


class TestMegatronArgsPatches(PrimusUT):
    def test_patches_registered_for_megatron_build_args(self):
        """All Megatron args_* patches should be registered for the build_args phase."""
        expected_ids = [
            "megatron.args.profile_tensorboard",
            "megatron.args.checkpoint_path",
            "megatron.args.tensorboard_path",
            "megatron.args.wandb_config",
            "megatron.args.logging_level",
            "megatron.args.data_path_split",
            "megatron.args.mock_data",
        ]

        for patch_id in expected_ids:
            patch = PatchRegistry.get(patch_id)
            self.assertEqual(patch.backend, "megatron")
            self.assertEqual(patch.phase, "build_args")

    def test_build_args_patches_apply_via_runner(self):
        """End-to-end check that build_args patches modify the Megatron args via run_patches()."""
        from primus.backends.megatron.patches import args_patches as _  # noqa: F401  # ensure import

        exp_root = os.path.abspath("ut_out/megatron_args_patches")
        config = {
            "primus_exp_root_path": exp_root,
            "primus_work_group": "wg",
            "primus_user_name": "user",
            "primus_exp_name": "exp",
        }

        os.environ["WANDB_API_KEY"] = "dummy"

        args = SimpleNamespace(
            profile=True,
            disable_tensorboard=True,
            save=None,
            tensorboard_dir=None,
            disable_wandb=False,
            wandb_project=None,
            wandb_exp_name=None,
            wandb_save_dir=None,
            stderr_sink_level="DEBUG",
            logging_level=None,
            data_path="data1 data2",
            train_data_path="train1 train2",
            valid_data_path=None,
            test_data_path="test1",
            mock_data=False,
        )

        applied = run_patches(
            backend="megatron",
            phase="build_args",
            backend_version="0.15.0rc5",
            primus_version=None,
            model_name="llama2_7B",
            extra={"args": args, "config": config},
        )

        self.assertGreater(applied, 0)

        # profile_tensorboard + tensorboard_path
        self.assertFalse(args.disable_tensorboard)
        self.assertIsNotNone(args.tensorboard_dir)
        self.assertTrue(args.tensorboard_dir.endswith("tensorboard"))

        # checkpoint_path
        self.assertIsNotNone(args.save)
        self.assertTrue(args.save.endswith("checkpoints"))

        # wandb_config
        self.assertEqual(args.wandb_save_dir, exp_root)
        self.assertIsNotNone(args.wandb_project)
        self.assertIsNotNone(args.wandb_exp_name)

        # logging_level
        self.assertEqual(args.logging_level, 10)  # DEBUG -> 10

        # data_path_split
        self.assertIsInstance(args.data_path, list)
        self.assertEqual(args.data_path, ["data1", "data2"])
        self.assertIsInstance(args.train_data_path, list)
        self.assertEqual(args.train_data_path, ["train1", "train2"])
        self.assertIsNone(args.valid_data_path)
        self.assertIsInstance(args.test_data_path, list)
        self.assertEqual(args.test_data_path, ["test1"])

    def test_mock_data_patch_disables_all_data_paths(self):
        """patch_mock_data should null out all data paths when mock_data=True."""
        from primus.backends.megatron.patches.args_patches import patch_mock_data

        args = SimpleNamespace(
            mock_data=True,
            data_path="data",
            train_data_path="train",
            valid_data_path="valid",
            test_data_path="test",
        )

        ctx = PatchContext(
            backend="megatron",
            phase="build_args",
            backend_version=None,
            primus_version=None,
            model_name=None,
            extra={"args": args},
        )

        patch_mock_data(ctx)

        self.assertIsNone(args.data_path)
        self.assertIsNone(args.train_data_path)
        self.assertIsNone(args.valid_data_path)
        self.assertIsNone(args.test_data_path)


