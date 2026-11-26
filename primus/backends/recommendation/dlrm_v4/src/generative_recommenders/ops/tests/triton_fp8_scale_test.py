#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

import torch

from generative_recommenders.ops.triton import triton_fp8 as fp8


class TritonFp8HostScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        fp8.clear_fp8_scale_runtime_cache()

    def tearDown(self) -> None:
        fp8.clear_fp8_scale_runtime_cache()

    def test_host_scale_contract_is_default_off(self) -> None:
        scale_a = torch.ones(1)
        output_scale = torch.ones(1)
        with patch.object(fp8, "_HSTU_FP8_HOST_SCALE_VALUES", False):
            self.assertIsNone(
                fp8._trusted_host_scale_values(
                    scale_a,
                    output_scale,
                    0.25,
                    0.5,
                )
            )

    def test_host_scale_contract_validates_or_falls_back(self) -> None:
        scale_a = torch.ones(1)
        output_scale = torch.ones(1)
        with (
            patch.object(fp8, "_HSTU_FP8_HOST_SCALE_VALUES", True),
            patch.object(fp8, "_HSTU_FP8_FAIL_CLOSED", True),
        ):
            self.assertEqual(
                fp8._trusted_host_scale_values(
                    scale_a,
                    output_scale,
                    0.25,
                    0.5,
                ),
                (0.25, 0.5),
            )
            for invalid in (None, 0.0, -1.0, math.inf, math.nan, True):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(RuntimeError):
                        fp8._trusted_host_scale_values(
                            scale_a,
                            output_scale,
                            invalid,
                            0.5,
                        )
        with (
            patch.object(fp8, "_HSTU_FP8_HOST_SCALE_VALUES", True),
            patch.object(fp8, "_HSTU_FP8_FAIL_CLOSED", False),
        ):
            self.assertIsNone(
                fp8._trusted_host_scale_values(
                    scale_a,
                    output_scale,
                    None,
                    0.5,
                )
            )

    def test_direct_host_values_bypass_scalar_reads_and_reuse_cache(self) -> None:
        scale_a = torch.tensor([0.25])
        output_scale = torch.tensor([0.5])
        bias = torch.arange(16, dtype=torch.float32)
        with patch.object(
            fp8,
            "activation_scale_value",
            side_effect=AssertionError("unexpected scalar read"),
        ):
            folded = fp8._fold_output_scale_into_input(
                scale_a.reshape(1, 1),
                output_scale,
                scale_a_value=0.25,
                output_scale_value=0.5,
            )
            repeated = fp8._fold_output_scale_into_input(
                scale_a.reshape(1, 1),
                output_scale.clone(),
                scale_a_value=0.25,
                output_scale_value=0.5,
            )
            scaled_bias = fp8._scale_bias_for_fp8_output(
                bias,
                output_scale,
                output_scale_value=0.5,
            )
        self.assertIs(folded, repeated)
        torch.testing.assert_close(folded, torch.tensor([[0.5]]))
        torch.testing.assert_close(
            scaled_bias.float(),
            (bias / 0.5).to(torch.bfloat16).float(),
        )

    def test_host_bias_cache_observes_bias_version(self) -> None:
        bias = torch.ones(16)
        output_scale = torch.ones(1)
        first = fp8._scale_bias_for_fp8_output(
            bias,
            output_scale,
            output_scale_value=0.5,
        )
        bias.add_(1.0)
        second = fp8._scale_bias_for_fp8_output(
            bias,
            output_scale,
            output_scale_value=0.5,
        )
        self.assertIsNot(first, second)
        torch.testing.assert_close(second, torch.full((16,), 4.0).bfloat16())

    def test_inverse_scale_uses_host_value(self) -> None:
        scale = torch.ones(1)
        with (
            patch.object(
                fp8,
                "activation_scale_value",
                side_effect=AssertionError("unexpected scalar read"),
            ),
            patch.object(fp8, "_fp8_output_scale_mode", return_value="multiply"),
        ):
            inverse = fp8.inverse_scale_tensor(scale, value=4.0)
            repeated = fp8.inverse_scale_tensor(scale.clone(), value=4.0)
        self.assertIs(inverse, repeated)
        torch.testing.assert_close(inverse, torch.tensor([[0.25]]))

    def test_public_fp8_output_path_never_reads_trusted_scales(self) -> None:
        dtype = fp8.fp8_e4m3_dtype()
        activation = torch.zeros((2, 16), dtype=dtype)
        weight = torch.zeros((16, 16), dtype=torch.bfloat16)
        weight_fp8 = torch.zeros((16, 16), dtype=dtype).t()
        scale_a = torch.full((1, 1), 0.25)
        scale_b = torch.ones((1, 1))
        output_scale = torch.full((1, 1), 0.5)
        bias = torch.arange(16, dtype=torch.float32)
        expected = torch.zeros((2, 16), dtype=dtype)
        with (
            patch.object(fp8, "_HSTU_FP8_HOST_SCALE_VALUES", True),
            patch.object(fp8, "_HSTU_FP8_FAIL_CLOSED", True),
            patch.object(fp8, "fp8_supported", return_value=True),
            patch.object(
                fp8,
                "_get_weight_shadow",
                return_value=(weight_fp8, scale_b),
            ),
            patch.object(
                fp8,
                "_fp8_output_scale_mode",
                return_value="input_scale",
            ),
            patch.object(fp8, "maybe_record_fp8_telemetry"),
            patch.object(
                fp8,
                "activation_scale_value",
                side_effect=AssertionError("unexpected scalar read"),
            ),
            patch.object(fp8, "_scaled_mm", return_value=expected) as scaled_mm,
        ):
            actual = fp8.scaled_mm_fp8_out_preq(
                activation_fp8=activation,
                scale_a=scale_a,
                weight=weight,
                bias=bias,
                output_scale=output_scale,
                scale_a_value=0.25,
                output_scale_value=0.5,
            )
        self.assertIs(actual, expected)
        args = scaled_mm.call_args.args
        kwargs = scaled_mm.call_args.kwargs
        torch.testing.assert_close(args[2], torch.tensor([[0.5]]))
        torch.testing.assert_close(
            kwargs["bias"].float(),
            (bias / 0.5).to(torch.bfloat16).float(),
        )


class RefuseFlydslFp8ForwardTest(unittest.TestCase):
    def test_fp8_fwd_plus_flydsl_is_silent(self) -> None:
        from generative_recommenders.ops.triton.triton_fp8 import (
            refuse_fp8_with_flydsl_attention_forward,
        )

        with patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_FWD": "1", "HSTU_ATTN_FWD": "flydsl"},
            clear=False,
        ):
            refuse_fp8_with_flydsl_attention_forward()

    def test_fp8_fwd_with_triton_is_silent(self) -> None:
        from generative_recommenders.ops.triton.triton_fp8 import (
            refuse_fp8_with_flydsl_attention_forward,
        )

        with patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_FWD": "1", "HSTU_ATTN_FWD": "triton"},
            clear=False,
        ):
            refuse_fp8_with_flydsl_attention_forward()

    def test_bf16_flydsl_is_silent(self) -> None:
        from generative_recommenders.ops.triton.triton_fp8 import (
            refuse_fp8_with_flydsl_attention_forward,
        )

        with patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_FWD": "0", "HSTU_ATTN_FWD": "flydsl"},
            clear=False,
        ):
            refuse_fp8_with_flydsl_attention_forward()

    def test_bootstrap_fp8_plus_flydsl_is_silent(self) -> None:
        from generative_recommenders.dlrm_v4.train._env_bootstrap import (
            apply_env_bootstrap,
        )

        with patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_FWD": "1", "HSTU_ATTN_FWD": "flydsl"},
            clear=False,
        ):
            apply_env_bootstrap()

    def test_fp8_plus_flydsl_selects_e4m3_fwd(self) -> None:
        from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
            flydsl_fp8_fwd_selected,
        )

        with patch.dict(
            os.environ,
            {
                "DLRM_HSTU_FP8_FWD": "1",
                "HSTU_ATTN_FWD": "flydsl",
                "DLRM_HSTU_FP8_FWD_FLYDSL": "0",
            },
            clear=False,
        ):
            self.assertTrue(flydsl_fp8_fwd_selected())

        with patch.dict(
            os.environ,
            {
                "DLRM_HSTU_FP8_FWD": "1",
                "HSTU_ATTN_FWD": "triton",
                "DLRM_HSTU_FP8_FWD_FLYDSL": "0",
            },
            clear=False,
        ):
            self.assertFalse(flydsl_fp8_fwd_selected())


class BwdQkMfmaKnobTest(unittest.TestCase):
    def _mode(self, **env):
        from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
            bwd_qk_mfma_mode,
        )

        with patch.dict(os.environ, env, clear=False):
            return bwd_qk_mfma_mode()

    def test_default_is_fp8(self) -> None:
        self.assertEqual(
            self._mode(DLRM_HSTU_FP8_BWD_QK_MFMA="", DLRM_HSTU_FP8_BWD_K128=""),
            "fp8",
        )

    def test_named_values(self) -> None:
        self.assertEqual(self._mode(DLRM_HSTU_FP8_BWD_QK_MFMA="fp8"), "fp8")
        self.assertEqual(
            self._mode(DLRM_HSTU_FP8_BWD_QK_MFMA="fp8_strict"), "fp8_strict"
        )
        self.assertEqual(
            self._mode(DLRM_HSTU_FP8_BWD_QK_MFMA="bf16"),
            "bf16",
        )

    def test_opt_out_and_strict_alias(self) -> None:
        self.assertEqual(
            self._mode(
                DLRM_HSTU_FP8_BWD_QK_MFMA="bf16",
                DLRM_HSTU_FP8_BWD_K128="auto",
            ),
            "bf16",
        )
        self.assertEqual(
            self._mode(
                DLRM_HSTU_FP8_BWD_QK_MFMA="fp8", DLRM_HSTU_FP8_BWD_K128="1"
            ),
            "fp8_strict",
        )
        # K128=0 is not an opt-out against the fp8 default; QK_MFMA=bf16 is.
        self.assertEqual(
            self._mode(DLRM_HSTU_FP8_BWD_QK_MFMA="fp8", DLRM_HSTU_FP8_BWD_K128="0"),
            "fp8",
        )


if __name__ == "__main__":
    unittest.main()
