###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "hybrid" / "prepare_hylo_packed_data.py"


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prepare_hylo_packed_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_selection_is_the_double_shuffle_without_row_zero(prep):
    perm = np.random.default_rng(42).permutation(1000)
    expected = perm[perm[1:200]]
    np.testing.assert_array_equal(prep.select_rows(1000, "train", ratio=0.2, seed=42), expected)
    np.testing.assert_array_equal(prep.select_rows(1000, "test", ratio=0.2, seed=42), perm)


def test_train_selection_matches_datasets_shuffle(prep):
    datasets = pytest.importorskip("datasets")
    table = datasets.Dataset.from_dict({"row": list(range(500))})
    expected = table.shuffle(seed=42).shuffle(seed=42).select(range(1, 100))["row"]
    assert prep.select_rows(500, "train", ratio=0.2, seed=42).tolist() == expected


def test_constant_length_sequences_drop_each_buffer_remainder(prep):
    # A buffer closes at 4 * 1.0 * 2 = 8 characters: 5 + 4 = 9 make 9 // 4 = 2 sequences,
    # and the 3 tokens left over at the end make none.
    assert (
        prep.constant_length_sequences(
            [5, 4, 3], [5, 4, 3], seq_length=4, chars_per_token=1.0, num_sequences=2
        )
        == 2
    )
    assert (
        prep.constant_length_sequences([1, 1], [4, 4], seq_length=4, chars_per_token=1.0, num_sequences=2)
        == 2
    )


def test_decontamination_ignores_case_and_whitespace(prep):
    rows = [
        {
            "prompt": 'def f(x):\n    """Return the  Square of x."""\n',
            "canonical_solution": "    return x * x\n",
        },
        {"prompt": "def g():\n    '''Ignored helper.'''\n", "canonical_solution": "    return x + y\n"},
    ]
    strings = prep.contamination_strings(rows)
    assert strings == ["return the square of x.", "ignored helper.", "return x * x"]
    assert prep.is_contaminated("User: RETURN the square\n of X. please", strings)
    assert not prep.is_contaminated("return x + y", strings)


def test_docstring_extraction_follows_alignment_handbook(prep):
    assert prep.extract_docstring('"""first"""\ncode\n"""second"""') == "second"
    assert prep.extract_docstring("'''only'''") == "only"
    with pytest.raises(ValueError):
        prep.extract_docstring("no docstring here")
