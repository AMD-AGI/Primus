###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import numpy as np
import pytest
import torch

from primus.backends.megatron.checkpoint.philox_uniform import (
    grid_threads,
    offset_increment,
    philox4x32_10,
    uniform_like_torch,
    uniform_unit,
)

# Known-answer vectors for Philox4x32-10 from the Random123 distribution (kat_vectors).
_KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize("counter,key,expected", _KAT)
def test_philox_matches_random123_known_answers(counter, key, expected):
    words = philox4x32_10(tuple(np.array([c], dtype=np.uint32) for c in counter), key)
    assert tuple(int(w[0]) for w in words) == expected


def test_grid_and_offset_follow_the_kernel_launch():
    # 1000 elements fit in 4 blocks of 256 threads on any GPU that can launch 4 blocks.
    assert grid_threads(1000, compute_units=7) == 1024
    assert offset_increment(1000, compute_units=7) == 4
    # A large tensor fills compute_units * threads_per_cu threads and takes several draws each.
    assert grid_threads(1_000_000, compute_units=3) == 3 * 2048
    assert offset_increment(1_000_000, compute_units=3) == ((1_000_000 - 1) // (3 * 2048 * 4) + 1) * 4
    assert grid_threads(1_000_000, compute_units=3, threads_per_cu=256) == 3 * 256


def test_small_tensors_do_not_depend_on_the_compute_unit_count():
    index = np.arange(1000)
    a = uniform_unit(index, 1000, offset=8, compute_units=7, seed=11)
    b = uniform_unit(index, 1000, offset=8, compute_units=13, seed=11)
    np.testing.assert_array_equal(a, b)


def test_large_tensors_map_elements_by_the_compute_unit_count():
    numel = 200_000
    index = np.arange(numel)
    a = uniform_unit(index, numel, offset=0, compute_units=3, seed=11)
    b = uniform_unit(index, numel, offset=0, compute_units=5, seed=11)
    # Elements below the smaller grid take component 0 of the first draw on both.
    smaller = grid_threads(numel, 3)
    np.testing.assert_array_equal(a[:smaller], b[:smaller])
    assert np.mean(a[smaller:] == b[smaller:]) < 0.01


def test_sampled_elements_equal_the_full_draw():
    numel = 50_000
    full = uniform_unit(np.arange(numel), numel, offset=12, compute_units=2, seed=5)
    sample = np.array([0, 1, 4095, 4096, 17_001, numel - 1])
    np.testing.assert_array_equal(uniform_unit(sample, numel, 12, 2, 5), full[sample])


def test_unit_values_lie_in_the_half_open_unit_interval_for_both_flavors():
    index = np.arange(10_000)
    rocrand = uniform_unit(index, 10_000, 0, 4, seed=3, flavor="rocrand")
    curand = uniform_unit(index, 10_000, 0, 4, seed=3, flavor="curand")
    for values in (rocrand, curand):
        assert values.min() > 0.0 and values.max() <= 1.0
    # The conversions differ by 2**-33 before rounding to float32: at most one ulp after it.
    assert np.max(np.abs(rocrand - curand)) <= 2.0**-23
    with pytest.raises(ValueError, match="flavor"):
        uniform_unit(index, 10_000, 0, 4, seed=3, flavor="other")


def test_uniform_like_torch_stays_below_the_upper_bound():
    numel = 100_000
    values = uniform_like_torch(np.arange(numel), numel, 0, 6, -0.5, 0.5, torch.bfloat16, seed=1)
    assert values.dtype == torch.bfloat16
    assert values.min() >= -0.5 and values.max() < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_emulation_matches_this_gpu(dtype):
    props = torch.cuda.get_device_properties(0)
    compute_units = props.multi_processor_count
    threads = getattr(props, "max_threads_per_multi_processor", 2048)
    flavor = "rocrand" if torch.version.hip else "curand"
    first, second = 3_000_001, 70_003
    torch.cuda.manual_seed(1234)
    a = torch.empty(first, device="cuda", dtype=dtype).uniform_(-0.25, 0.75).cpu()
    b = torch.empty(second, device="cuda", dtype=dtype).uniform_(1.0, 16.0).cpu()
    emu_a = uniform_like_torch(
        np.arange(first), first, 0, compute_units, -0.25, 0.75, dtype, 1234, flavor, threads
    )
    offset = offset_increment(first, compute_units, threads)
    emu_b = uniform_like_torch(
        np.arange(second), second, offset, compute_units, 1.0, 16.0, dtype, 1234, flavor, threads
    )
    if dtype == torch.bfloat16:
        assert torch.equal(a, emu_a)
        assert torch.equal(b, emu_b)
    else:
        # The GPU may fuse the scale and shift into one rounding; the model rounds twice.
        torch.testing.assert_close(a, emu_a, rtol=2.0**-23, atol=2.0**-24)
        torch.testing.assert_close(b, emu_b, rtol=2.0**-23, atol=2.0**-24)
