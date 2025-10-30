###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.modules.trainer.torchtitan.patch_utils import (
    apply_patch_checkpoint_wrapper,
    patch_mock_hf_dataset,
)
from tests.utils import PrimusUT


class TestTorchtitanPatch(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mock_hf_dataset_patch(self):
        """
        Test that enable_mock_hf_dataset() successfully patches datasets.load_dataset
        and returns a fake HuggingFace Dataset.
        """
        # from primus.utils import mock_hf_dataset

        patch_mock_hf_dataset()

        # Reimport datasets and call load_dataset
        import datasets

        ds = datasets.load_dataset("allenai/c4", split="train")

        # Verify that this is an in-memory Dataset with expected content
        assert isinstance(ds, datasets.Dataset)
        assert "text" in ds.column_names
        assert len(ds) > 0
        sample = ds[0]
        assert isinstance(sample["text"], str)
        assert len(sample["text"].split()) > 0

    def test_patch_checkpoint_wrapper(self):
        """
        Verify Primus patch for torch.distributed.algorithms._checkpoint.checkpoint_wrapper
        correctly ignores unsupported kwargs (e.g., early_stop)
        without breaking checkpoint functionality.
        """
        import torch

        apply_patch_checkpoint_wrapper()

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        m = DummyModule()

        # Should NOT raise: TypeError: unexpected keyword argument 'early_stop'
        try:
            wrapped = checkpoint_wrapper(m, preserve_rng_state=False, early_stop=True)
        except TypeError as e:
            raise AssertionError(f"checkpoint_wrapper should ignore unsupported kwargs but raised: {e}")

        assert isinstance(wrapped, torch.nn.Module)

        # Verify normal forward/backward still works
        x = torch.tensor([2.0], requires_grad=True)
        y = wrapped(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([1.0]))
