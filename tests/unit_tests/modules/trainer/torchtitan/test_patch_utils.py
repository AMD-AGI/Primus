###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.modules.trainer.torchtitan.patch_utils import patch_mock_hf_dataset
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
