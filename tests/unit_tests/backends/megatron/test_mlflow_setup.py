###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from unittest.mock import MagicMock, patch

from primus.backends.megatron.training import mlflow_setup


class DummyArgs:
    def __init__(self, rank: int, world_size: int, mlflow_run_name=None):
        self.rank = rank
        self.world_size = world_size
        self.mlflow_run_name = mlflow_run_name


def _call_upload(generate_tracelens_report: bool, args: DummyArgs, mock_generate):
    with patch(f"{mlflow_setup.__name__}.get_mlflow_writer", return_value=None), patch(
        f"{mlflow_setup.__name__}.get_primus_args", return_value=args
    ), patch(f"{mlflow_setup.__name__}.generate_tracelens_reports_locally", mock_generate):
        mlflow_setup.upload_mlflow_artifacts(
            tensorboard_dir="/tmp/tb",
            exp_root_path="/tmp/exp",
            generate_tracelens_report=generate_tracelens_report,
        )


def test_local_generation_single_rank_no_mlflow():
    mock_generate = MagicMock()
    _call_upload(True, DummyArgs(rank=0, world_size=1, mlflow_run_name=None), mock_generate)
    mock_generate.assert_called_once()


def test_local_generation_skipped_when_mlflow_expected_distributed():
    mock_generate = MagicMock()
    _call_upload(True, DummyArgs(rank=0, world_size=8, mlflow_run_name="run"), mock_generate)
    mock_generate.assert_not_called()


def test_local_generation_distributed_without_mlflow():
    mock_generate = MagicMock()
    _call_upload(True, DummyArgs(rank=0, world_size=8, mlflow_run_name=None), mock_generate)
    mock_generate.assert_called_once()


def test_local_generation_disabled_when_flag_false():
    mock_generate = MagicMock()
    _call_upload(False, DummyArgs(rank=0, world_size=1, mlflow_run_name=None), mock_generate)
    mock_generate.assert_not_called()
