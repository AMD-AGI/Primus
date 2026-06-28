# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for fp4_utils.py MXFP4 recipe and context manager changes.

Tests get_fp4_recipe error handling for an unsupported recipe.
"""

from types import SimpleNamespace

import pytest

from tests.utils import PrimusUT


class TestGetFp4RecipeMXFP4(PrimusUT):
    """Verify get_fp4_recipe returns correct recipe objects for MXFP4."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def test_unsupported_recipe_produces_error(self):
        pytest.importorskip("transformer_engine")

        from primus.backends.megatron.core.fp4_utils import get_fp4_recipe

        config = SimpleNamespace(fp4_recipe="nonexistent_recipe")
        result = get_fp4_recipe(config)

        if isinstance(result, tuple):
            recipe, reason = result
            assert recipe is None, "Unsupported recipe should return None"
            assert (
                "Unsupported" in reason or "unsupported" in reason.lower()
            ), f"Expected 'Unsupported' in reason, got: {reason}"
        else:
            pytest.fail("HAVE_TE-only branch should raise ValueError for unsupported recipe")
