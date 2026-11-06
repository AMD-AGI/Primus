###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json

import numpy as np
import pytest

from primus.backends.megatron.training.hylo_order_dataset import HyLoOrderDataset, unset

STEPS, MBS, SEQ, DP = 5, 3, 4, 2


def _record(tmp_path):
    batches = np.arange(DP * STEPS * MBS * SEQ, dtype=np.int32).reshape(DP, STEPS, MBS, SEQ)
    for r in range(DP):
        batches[r].tofile(tmp_path / f"batches_rank{r}.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps({"micro_batch": MBS, "seq_len": SEQ, "world_size": DP, "steps": STEPS})
    )
    return batches


def test_sequential_sampler_gives_each_rank_its_recorded_batch(tmp_path):
    batches = _record(tmp_path)
    ds = HyLoOrderDataset(str(tmp_path), seq_length=SEQ, micro_batch_size=MBS, data_parallel_size=DP)
    assert len(ds) == STEPS * MBS * DP

    # MegatronPretrainingSampler: each global batch of MBS * DP indices, rank r takes [r * MBS, (r + 1) * MBS).
    for step in range(STEPS):
        global_batch = list(range(step * MBS * DP, (step + 1) * MBS * DP))
        for rank in range(DP):
            served = np.stack(
                [ds[i]["tokens"].numpy() for i in global_batch[rank * MBS : (rank + 1) * MBS]]
            )
            np.testing.assert_array_equal(served, batches[rank, step])


def test_rejects_a_mismatched_layout(tmp_path):
    _record(tmp_path)
    with pytest.raises(ValueError, match="micro_batch_size"):
        HyLoOrderDataset(str(tmp_path), seq_length=SEQ, micro_batch_size=MBS + 1, data_parallel_size=DP)
    with pytest.raises(ValueError, match="data-parallel"):
        HyLoOrderDataset(str(tmp_path), seq_length=SEQ, micro_batch_size=MBS, data_parallel_size=DP * 2)


@pytest.mark.parametrize("value", [None, "", "null", "None", " NULL "])
def test_unset_values(value):
    assert unset(value)


@pytest.mark.parametrize("value", [32.0, "32", "/data/run"])
def test_set_values(value):
    assert not unset(value)
