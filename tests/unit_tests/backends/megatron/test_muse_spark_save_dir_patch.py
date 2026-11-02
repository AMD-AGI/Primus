###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Muse Spark save-dir link (patches/args/muse_spark_save_dir_patches.py)."""

import os
from types import SimpleNamespace

from primus.backends.megatron.patches.args import muse_spark_save_dir_patches as patch


def test_link_is_created_and_points_to_the_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    ckpt, req = tmp_path / "exp" / "checkpoints", tmp_path / "out" / "save"
    assert patch.link_requested_save_dir(str(req), str(ckpt))
    assert req.is_symlink() and os.path.realpath(req) == os.path.realpath(ckpt) and ckpt.is_dir()
    assert patch.link_requested_save_dir(str(req), str(ckpt))  # again: the existing link is fine


def _logs(monkeypatch):
    logs = []
    monkeypatch.setattr(patch, "log_rank_0", logs.append)
    return logs


def test_an_existing_dir_is_left_alone_with_a_warning(tmp_path, monkeypatch):
    logs = _logs(monkeypatch)
    req = tmp_path / "save"
    req.mkdir()
    assert not patch.link_requested_save_dir(str(req), str(tmp_path / "checkpoints"))
    assert req.is_dir() and not req.is_symlink()
    assert len(logs) == 1 and "[WARN]" in logs[0] and "already exists" in logs[0]


def test_a_link_to_an_old_experiment_is_kept_with_a_warning(tmp_path, monkeypatch):
    # a --load <requested> would restore the old experiment: say so, do not replace it
    logs = _logs(monkeypatch)
    old, ckpt, req = tmp_path / "old", tmp_path / "exp" / "checkpoints", tmp_path / "save"
    old.mkdir()
    req.symlink_to(old, target_is_directory=True)
    assert not patch.link_requested_save_dir(str(req), str(ckpt))
    assert os.path.realpath(req) == os.path.realpath(old)
    assert len(logs) == 1 and "[WARN]" in logs[0] and str(old) in logs[0] and str(ckpt) in logs[0]


def test_a_rank_that_loses_the_race_accepts_the_winners_link(tmp_path, monkeypatch):
    # every rank may call it (LOCAL_RANK is not read): os.symlink is the one atomic step
    logs = _logs(monkeypatch)
    ckpt, req = tmp_path / "checkpoints", tmp_path / "save"
    real_symlink = os.symlink

    def racing(src, dst, target_is_directory=False):
        real_symlink(src, dst, target_is_directory=target_is_directory)  # the other rank
        raise FileExistsError(dst)

    monkeypatch.setattr(patch.os, "symlink", racing)
    assert patch.link_requested_save_dir(str(req), str(ckpt))
    assert not any("[WARN]" in m for m in logs)


def test_registered_before_checkpoint_path_for_muse_spark_only():
    from primus.backends.megatron.patches.args import (  # noqa: F401 - registers it
        checkpoint_path_patches,
    )
    from primus.core.patches import PatchRegistry

    mine = PatchRegistry.get("megatron.muse_spark.link_save_dir")
    other = PatchRegistry.get("megatron.args.checkpoint_path")
    assert mine.phase == "build_args" == other.phase
    assert mine.priority < other.priority  # it must see args.save before the rewrite

    def ctx(model_type):
        return SimpleNamespace(extra={"backend_args": SimpleNamespace(model_type=model_type)})

    assert mine.condition(ctx("muse_spark"))
    assert not mine.condition(ctx("gpt"))


def test_an_existing_link_is_compared_by_realpath(tmp_path, monkeypatch):
    ckpt, other = tmp_path / "checkpoints", tmp_path / "other"
    ckpt.mkdir()
    other.mkdir()
    (tmp_path / "alias").symlink_to(ckpt, target_is_directory=True)
    req = tmp_path / "save"
    req.symlink_to(tmp_path / "alias", target_is_directory=True)
    assert patch.link_requested_save_dir(str(req), str(ckpt))
    assert not patch.link_requested_save_dir(str(req), str(other))


def _ctx(save, exp_root):
    return SimpleNamespace(
        extra={
            "backend_args": SimpleNamespace(save=save),
            "primus_config": SimpleNamespace(exp_root_path=exp_root),
        }
    )


def test_patch_links_the_requested_save_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    req = tmp_path / "save"
    patch.patch_muse_spark_link_save_dir(_ctx(str(req), str(tmp_path / "exp")))
    assert os.path.realpath(req) == os.path.realpath(tmp_path / "exp" / "checkpoints")


def test_patch_is_a_no_op_when_save_is_the_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    seen = []
    monkeypatch.setattr(patch, "link_requested_save_dir", lambda *a: seen.append(a))
    exp = tmp_path / "exp"
    patch.patch_muse_spark_link_save_dir(_ctx(str(exp / "checkpoints"), str(exp)))
    patch.patch_muse_spark_link_save_dir(_ctx(None, str(exp)))  # no save: nothing to link
    assert seen == [] and not exp.exists()


def test_a_file_where_the_checkpoint_dir_should_be_is_reported_as_such(tmp_path, monkeypatch):
    # makedirs' FileExistsError is not "another rank won the race": report the real error
    logs = _logs(monkeypatch)
    ckpt, req = tmp_path / "exp" / "checkpoints", tmp_path / "req"
    ckpt.parent.mkdir()
    ckpt.write_text("not a dir")
    assert not patch.link_requested_save_dir(str(req), str(ckpt))
    assert not os.path.lexists(req)
    assert len(logs) == 1 and "cannot link" in logs[0] and "already exists" not in logs[0]
