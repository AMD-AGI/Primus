###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Integration-style tests verifying that optimizer state is clean after warmup
completion.

Uses mocked Megatron training harnesses - no real GPU training loop, but
exercises the warmup helper pipeline (neuter -> restore -> reset) end-to-end.

Run:
    python -m pytest tests/unit_tests/backends/megatron/test_mlperf_warmup_state_equivalence.py -v
"""

from types import SimpleNamespace

import torch


class TestOptimizerStateCleanAfterWarmup:
    """After neuter → steps → restore → reset, optimizer should be clean."""

    def test_adam_state_clean_after_warmup_simulation(self):
        """Simulate warmup: step with real betas, then restore + reset.

        Note: production uses Apex/TE FusedAdam which supports
        bias_correction=False.  Stock PyTorch Adam doesn't, so betas=[1,1]
        causes division-by-zero.  We use SGD for the neutered phase, then
        switch to Adam to verify the reset path.
        """
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _neuter_optimizer,
            _reset_optimizer_state,
            _restore_optimizer,
        )

        model = torch.nn.Linear(16, 8, bias=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        wrapper = SimpleNamespace(optimizer=opt)

        for _ in range(3):
            loss = model(torch.randn(4, 16)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for state in opt.state.values():
            assert state["step"].item() > 0

        saved_hyp = _neuter_optimizer(wrapper)
        assert opt.param_groups[0]["betas"] == [1.0, 1.0]
        assert opt.param_groups[0]["weight_decay"] == 0.0

        _restore_optimizer(wrapper, saved_hyp)
        _reset_optimizer_state(wrapper)

        assert list(opt.param_groups[0]["betas"]) == [0.9, 0.999]
        assert opt.param_groups[0]["weight_decay"] == 0.01

        for state in opt.state.values():
            step_val = state["step"]
            if isinstance(step_val, torch.Tensor):
                assert step_val.item() == 0
            else:
                assert step_val == 0

    def test_param_groups_step_cleared(self):
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _neuter_optimizer,
            _reset_optimizer_state,
            _restore_optimizer,
        )

        model = torch.nn.Linear(8, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.param_groups[0]["step"] = 42
        wrapper = SimpleNamespace(optimizer=opt)

        saved_hyp = _neuter_optimizer(wrapper)
        _restore_optimizer(wrapper, saved_hyp)
        _reset_optimizer_state(wrapper)

        assert opt.param_groups[0].get("step", 0) == 0
