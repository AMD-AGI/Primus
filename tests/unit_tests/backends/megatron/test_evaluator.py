###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect

from primus.backends.megatron.training.evaluator import primus_evaluate


def test_accepts_every_upstream_evaluate_argument():
    import megatron.training.training as megatron_training

    upstream = getattr(megatron_training.evaluate, "__wrapped__", megatron_training.evaluate)
    if upstream is primus_evaluate:
        return
    ours = inspect.signature(primus_evaluate).parameters
    missing = [name for name in inspect.signature(upstream).parameters if name not in ours]
    assert not missing, f"primus_evaluate does not accept {missing}"
