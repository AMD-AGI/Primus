"""reference_lr_warmup must reproduce the MLPerf Flux reference schedule step for step.

The reference (torchtitan linear_warmup_stable_decay through LambdaLR, decay_ratio 0) runs
optimizer step i (0-indexed) at peak * (i + 1) / W for i < W and at peak afterwards.
Megatron's scheduler is driven exactly as training drives it: the LR used by step i is the
one get_lr returns before scheduler.step(increment=GBS) advances num_steps.
"""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from primus.core.patches import PatchContext


def _ctx(**params):
    p = SimpleNamespace(**params)
    return PatchContext(backend="megatron", phase="before_train",
                        extra={"backend_args": p, "module_config": SimpleNamespace(params=p)})


def _reference_lr(step, peak, warmup):
    # torchtitan/components/lr_scheduler.py linear_warmup_stable_decay, decay_ratio 0
    if step < warmup:
        return peak * float(step + 1) / warmup
    return peak


@pytest.mark.parametrize("warmup,gbs,peak", [(800, 1024, 2.5e-4), (1600, 512, 2.0e-4), (3, 8, 1.0)])
def test_reference_warmup_matches_torchtitan(warmup, gbs, peak):
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    from primus.backends.megatron.patches import lr_schedule_patches as lsp

    orig = OptimizerParamScheduler.get_lr
    try:
        with mock.patch("megatron.training.get_args", return_value=SimpleNamespace(global_batch_size=gbs)):
            lsp.patch_reference_lr_warmup(_ctx(reference_lr_warmup=True))
            p = torch.nn.Parameter(torch.zeros(1))
            opt = torch.optim.AdamW([p], lr=peak)
            sched = OptimizerParamScheduler(
                opt, init_lr=0.0, max_lr=peak, min_lr=peak,
                lr_warmup_steps=warmup * gbs, lr_decay_steps=(warmup + 50) * gbs,
                lr_decay_style="constant", start_wd=0.1, end_wd=0.1, wd_incr_steps=1,
                wd_incr_style="constant")
            for step in range(warmup + 20):
                got = opt.param_groups[0]["lr"]
                assert got == pytest.approx(_reference_lr(step, peak, warmup), rel=1e-12, abs=0), step
                sched.step(increment=gbs)
    finally:
        OptimizerParamScheduler.get_lr = orig


def test_reference_and_nemo_warmup_are_exclusive():
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    from primus.backends.megatron.patches import lr_schedule_patches as lsp

    orig = OptimizerParamScheduler.get_lr
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            lsp.patch_reference_lr_warmup(_ctx(reference_lr_warmup=True, nemo_aligned_lr_warmup=True))
    finally:
        OptimizerParamScheduler.get_lr = orig
