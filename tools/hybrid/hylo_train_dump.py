#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Run HyLo's ``train_distill.py`` unchanged, recording the student's init and every training batch.

This is the recorder behind ``hylo_order_data_dir`` (``HyLoOrderDataset``). With ``HYLO_DUMP_DIR=<dir>``:

- ``<dir>/init/model.safetensors``: rank 0's student as ``init_distillation`` returns it, before training;
  ``convert_hylo_reference_to_megatron.py`` turns it into the Megatron checkpoint to start from.
- ``<dir>/batches_rank<r>.bin``: rank ``r``'s ``input_ids`` of every training step, int32, appended.
- ``<dir>/meta.json``: ``micro_batch``, ``seq_len``, ``world_size`` and ``steps``, written by rank 0
  only when training finishes. ``world_size`` is the data-parallel size, since the HyLo runs are pure
  data parallel.

Each training batch is one optimizer step, so the HyLo recipe must use ``gradient_accumulation_steps: 1``.

Batches must be packed (no padding). Run it where HyLo's ``train_distill.py`` would run, from
HyLo's directory (``HYLO_DIR``, default the current directory), in place of ``train_distill.py``::

    HYLO_DUMP_DIR=/data/dump/s2 accelerate launch --config_file configs/fsdp_GDN.yaml \\
        /path/to/Primus/tools/hybrid/hylo_train_dump.py <HyLo recipe.yaml>

Used against AMD-Hybrid-Models ``feat/HyLo`` at 84871cc with ``hylo_primus_parity.patch`` applied
(``git am``), which adds ``teacher_rope_from_student`` (the recordings set it to false) and
padded-batch Enhanced-ILD.
"""
import json
import os
import runpy
import sys

rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
# One Triton cache per rank: concurrent autotuning of the same kernel corrupts a shared cache.
base = os.environ.get("TRITON_HOME", os.path.expanduser("~/.triton"))
os.environ["TRITON_HOME"] = os.path.join(base, f"rank_{rank}")
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TRITON_HOME"], "cache")

HYLO = os.path.abspath(os.environ.get("HYLO_DIR", os.getcwd()))
os.chdir(HYLO)
sys.path.insert(0, HYLO)

import torch  # noqa: E402
from hybrid.hybrid_wrapper import HybridModelWrapper  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from trainer.kd_trainer import KDTrainer  # noqa: E402

DUMP = os.environ["HYLO_DUMP_DIR"]
os.makedirs(DUMP, exist_ok=True)
RANK = int(rank)
BATCHES = os.path.join(DUMP, f"batches_rank{RANK}.bin")
STATE = {"steps": 0, "shape": None}

_init_distillation = HybridModelWrapper.init_distillation


def init_distillation(*args, **kwargs):
    wrapper = _init_distillation(*args, **kwargs)
    if RANK == 0:
        sd = {k: v.detach().cpu().clone().contiguous() for k, v in wrapper.model.state_dict().items()}
        os.makedirs(os.path.join(DUMP, "init"), exist_ok=True)
        save_file(sd, os.path.join(DUMP, "init", "model.safetensors"), metadata={"format": "pt"})
        print(f"[hylo_train_dump] saved the init, {len(sd)} tensors", flush=True)
    return wrapper


HybridModelWrapper.init_distillation = staticmethod(init_distillation)

_compute_loss = KDTrainer.compute_loss


def compute_loss(self, model, inputs, *args, **kwargs):
    if model.training:
        if self.args.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "hylo_train_dump records one batch per optimizer step; set gradient_accumulation_steps to 1"
            )
        mask = inputs.get("attention_mask")
        if mask is not None and not bool(mask.all()):
            raise RuntimeError("hylo_train_dump expects packed batches without padding")
        arr = inputs["input_ids"].detach().to("cpu", torch.int32).numpy()
        if STATE["shape"] is None:
            STATE["shape"] = list(arr.shape)
        elif list(arr.shape) != STATE["shape"]:
            raise RuntimeError(f"batch shape changed from {STATE['shape']} to {list(arr.shape)}")
        with open(BATCHES, "ab") as f:
            arr.tofile(f)
        STATE["steps"] += 1
    return _compute_loss(self, model, inputs, *args, **kwargs)


KDTrainer.compute_loss = compute_loss

for stale in [BATCHES] + ([os.path.join(DUMP, "meta.json")] if RANK == 0 else []):
    if os.path.exists(stale):
        os.remove(stale)

sys.argv = [os.path.join(HYLO, "train_hybrid", "train_distill.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
if RANK == 0 and STATE["shape"] is not None:
    meta = {
        "micro_batch": STATE["shape"][0],
        "seq_len": STATE["shape"][1],
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "steps": STATE["steps"],
    }
    with open(os.path.join(DUMP, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
