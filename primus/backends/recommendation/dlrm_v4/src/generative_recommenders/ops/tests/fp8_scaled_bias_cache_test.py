#!/usr/bin/env python3
"""Scaled-bias cache of the FP8 forward (triton_fp8._scale_bias_for_fp8_output).

The cache key is (id, data_ptr, offset, _version) of the bias plus the host output-scale value. The targets-only
forward passes a TEMPORARY bias (torch.cat of the V/K slices of the uvqk bias Parameter): its _version is always 0,
and once it is freed a later step's temporary can get the same id and the same allocator block. Without the weakref
check such a step was served the scaled bias of an EARLIER step's bias values. These tests mirror that call structure
on CPU tensors (the cache logic is device independent)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from generative_recommenders.ops.triton import triton_fp8 as fp8

_U, _V, _Q, _K = 512, 512, 512, 512
_SCALE = 0.0083


def _adam_like_step(param: torch.Tensor, step: int) -> None:
    # R-C9 FusedDenseAdam writes through raw pointers, then bumps the version counter.
    with torch.no_grad():
        param.data.add_(0.01 * (step + 1))
    torch.autograd.graph.increment_version(param)


def _targets_only_site(uvqk_bias: torch.Tensor, output_scale: torch.Tensor) -> torch.Tensor:
    # Same structure as triton_hstu_preprocess_and_attention.py (targets-only forward): split first, cat later,
    # every local freed at return.
    b_u, b_v, b_q, b_k = uvqk_bias.split([_U, _V, _Q, _K], dim=0)
    b_vk = torch.cat((b_v, b_k), dim=0)
    return fp8._scale_bias_for_fp8_output(b_vk, output_scale, output_scale_value=_SCALE)


def _expected(uvqk_bias: torch.Tensor) -> torch.Tensor:
    b_vk = torch.cat((uvqk_bias[_U:_U + _V], uvqk_bias[_U + _V + _Q:]), dim=0)
    return (b_vk.detach().float() / _SCALE).to(torch.bfloat16)


class Fp8ScaledBiasCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        fp8.clear_fp8_scale_runtime_cache()
        self._strict = fp8._scaled_bias_strict
        fp8._scaled_bias_strict = True  # DLRM_HSTU_FP8_BIAS_CACHE_STRICT unset or 1, whatever this process's env

    def tearDown(self) -> None:
        fp8.clear_fp8_scale_runtime_cache()
        fp8._scaled_bias_strict = self._strict

    def test_temporary_bias_is_never_served_stale(self) -> None:
        param = torch.nn.Parameter(torch.zeros(_U + _V + _Q + _K))
        output_scale = torch.ones(1)
        stale = []
        for step in range(200):
            _adam_like_step(param, step)
            got = _targets_only_site(param, output_scale)
            if not torch.equal(got, _expected(param)):
                stale.append(step)
            del got
        self.assertEqual(stale, [], f"stale scaled bias returned at steps {stale[:10]}")

    def test_cache_stays_bounded(self) -> None:
        param = torch.nn.Parameter(torch.zeros(_U + _V + _Q + _K))
        output_scale = torch.ones(1)
        for step in range(300):
            _adam_like_step(param, step)
            view = param[_U:]  # the fused-QKV site's bias (a view: version follows the Parameter)
            fp8._scale_bias_for_fp8_output(view, output_scale, output_scale_value=_SCALE)
            del view
            fp8._scale_bias_for_fp8_output(param, output_scale, output_scale_value=_SCALE)
            _targets_only_site(param, output_scale)
        self.assertLessEqual(len(fp8._scaled_output_bias_cache), fp8._SCALED_BIAS_PURGE_AT + 3)
        self.assertEqual(set(fp8._scaled_output_bias_cache), set(fp8._scaled_output_bias_refs))

    def test_live_tensor_hit_is_kept(self) -> None:
        bias = torch.arange(64, dtype=torch.float32)
        output_scale = torch.ones(1)
        first = fp8._scale_bias_for_fp8_output(bias, output_scale, output_scale_value=0.5)
        second = fp8._scale_bias_for_fp8_output(bias, output_scale, output_scale_value=0.5)
        self.assertIs(first, second)
        torch.testing.assert_close(first.float(), (bias / 0.5).to(torch.bfloat16).float())

    def test_switch_is_read_lazily_from_the_env(self) -> None:
        for raw, want in ((None, True), ("1", True), ("0", False)):
            env = {k: v for k, v in os.environ.items() if k != "DLRM_HSTU_FP8_BIAS_CACHE_STRICT"}
            if raw is not None:
                env["DLRM_HSTU_FP8_BIAS_CACHE_STRICT"] = raw
            with mock.patch.dict(os.environ, env, clear=True):
                fp8._scaled_bias_strict = None
                self.assertIs(fp8._scaled_bias_cache_strict(), want)
                os.environ["DLRM_HSTU_FP8_BIAS_CACHE_STRICT"] = "0" if want else "1"
                self.assertIs(fp8._scaled_bias_cache_strict(), want)  # cached for the process

    def test_kill_switch_is_the_legacy_cache(self) -> None:
        # DLRM_HSTU_FP8_BIAS_CACHE_STRICT=0: no reference recorded, a key hit returned unchecked, nothing purged.
        fp8._scaled_bias_strict = False
        param = torch.nn.Parameter(torch.zeros(_U + _V + _Q + _K))
        output_scale = torch.ones(1)
        steps = fp8._SCALED_BIAS_PURGE_AT + 16
        for step in range(steps):
            _adam_like_step(param, step)
            got = fp8._scale_bias_for_fp8_output(param, output_scale, output_scale_value=_SCALE)
            self.assertTrue(torch.equal(got, (param.detach().float() / _SCALE).to(torch.bfloat16)))
        self.assertEqual(len(fp8._scaled_output_bias_refs), 0)
        self.assertEqual(len(fp8._scaled_output_bias_cache), steps)  # one entry per version, never purged
        bias = torch.arange(64, dtype=torch.float32)
        first = fp8._scale_bias_for_fp8_output(bias, output_scale, output_scale_value=0.5)
        self.assertIs(fp8._scale_bias_for_fp8_output(bias, output_scale, output_scale_value=0.5), first)

    def test_clear_also_clears_refs(self) -> None:
        bias = torch.ones(8)
        fp8._scale_bias_for_fp8_output(bias, torch.ones(1), output_scale_value=0.5)
        self.assertEqual(len(fp8._scaled_output_bias_refs), 1)
        fp8.clear_fp8_scale_runtime_cache()
        self.assertEqual(len(fp8._scaled_output_bias_cache), 0)
        self.assertEqual(len(fp8._scaled_output_bias_refs), 0)
        fp8._scale_bias_for_fp8_output(bias, torch.ones(1), output_scale_value=0.5)
        fp8.clear_fp8_weight_cache()
        self.assertEqual(len(fp8._scaled_output_bias_refs), 0)


if __name__ == "__main__":
    unittest.main()
