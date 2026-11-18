# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import torch
from generative_recommenders.common import (
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from hypothesis import given, settings, strategies as st, Verbosity


class HSTUComputeTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=1000, max_value=1000),
        D=st.integers(min_value=128, max_value=128),
        L=st.integers(min_value=512, max_value=512),
        concat_u=st.booleans(),
        concat_x=st.booleans(),
        mul_u_activation_type=st.sampled_from(["silu", "sigmoid", "none"]),
        group_norm=st.booleans(),
        num_heads=st.sampled_from([4]),
        training=st.just(False),
        recompute_y_in_backward=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_compute_output(self, *args, **kwargs) -> None:
        self._test_compute_output(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            opt_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.just(1500000),
        D=st.just(512),
        L=st.just(512),
        concat_u=st.sampled_from([True]),
        concat_x=st.sampled_from([True]),
        mul_u_activation_type=st.sampled_from(["none"]),
        group_norm=st.sampled_from([False]),
        num_heads=st.sampled_from([4]),
        training=st.just(False),
        recompute_y_in_backward=st.sampled_from([False]),
        dtype=st.just(torch.bfloat16),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_long_sequences_compute_output(self, *args, **kwargs) -> None:
        self._test_compute_output(
            *args,
            **kwargs,
            test_backward=False,
            ref_kernel=HammerKernel.TRITON,
            opt_kernel=HammerKernel.TRITON,
            skip_comparisons=True,
        )

    def _test_compute_output(
        self,
        N: int,
        D: int,
        L: int,
        concat_u: bool,
        concat_x: bool,
        mul_u_activation_type: str,
        group_norm: bool,
        num_heads: int,
        training: bool,
        recompute_y_in_backward: bool,
        dtype: torch.dtype,
        test_backward: bool,
        ref_kernel: HammerKernel,
        opt_kernel: HammerKernel,
        skip_comparisons: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        from generative_recommenders.ops.hstu_compute import hstu_compute_output

        torch.manual_seed(0)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        dropout_ratio = 0.3 if training else 0.0
        attn = (
            torch.empty((N, L), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        u = (
            torch.empty((N, L), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_weight = (
            torch.empty(
                (L if not group_norm else num_heads,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_bias = (
            torch.empty(
                (L if not group_norm else num_heads,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_eps = 1e-6
        # When group_norm=True, only concat_ux = concat_u and concat_x is supported
        if group_norm:
            L_mult = 3 if (concat_u and concat_x) else 1
        else:
            L_mult = 1
            if concat_u:
                L_mult += 1
            if concat_x:
                L_mult += 1
        output_weight = (
            torch.empty((L * L_mult, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        # ref
        ref_out = hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            dropout_ratio=dropout_ratio,
            output_weight=output_weight,
            concat_u=concat_u,
            concat_x=concat_x,
            mul_u_activation_type=mul_u_activation_type,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=L // num_heads,
            training=training,
            recompute_y_in_backward=recompute_y_in_backward,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_out) * 0.1
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dattn, attn.grad = attn.grad.detach().clone(), None
        ref_du, u.grad = u.grad.detach().clone(), None
        ref_d_norm_w, norm_weight.grad = norm_weight.grad.detach().clone(), None
        ref_d_norm_b, norm_bias.grad = norm_bias.grad.detach().clone(), None
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_d_output_w, output_weight.grad = output_weight.grad.detach().clone(), None

        # opt
        attn = attn.detach().clone().requires_grad_()
        u = u.detach().clone().requires_grad_()
        norm_weight = norm_weight.detach().clone().requires_grad_()
        norm_bias = norm_bias.detach().clone().requires_grad_()
        output_weight = output_weight.detach().clone().requires_grad_()
        x = x.detach().clone().requires_grad_()
        opt_out = hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            dropout_ratio=dropout_ratio,
            output_weight=output_weight,
            concat_u=concat_u,
            concat_x=concat_x,
            mul_u_activation_type=mul_u_activation_type,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=L // num_heads,
            training=training,
            recompute_y_in_backward=recompute_y_in_backward,
            kernel=opt_kernel,
        )
        torch.testing.assert_close(
            ref_out,
            opt_out,
            atol=atol,
            rtol=rtol,
        )

        if test_backward:
            dout = dout.detach().clone()
            opt_out.backward(dout)
            opt_dattn, attn.grad = attn.grad.detach().clone(), None
            opt_du, u.grad = u.grad.detach().clone(), None
            opt_d_norm_w, norm_weight.grad = norm_weight.grad.detach().clone(), None
            opt_d_norm_b, norm_bias.grad = norm_bias.grad.detach().clone(), None
            opt_dx, x.grad = x.grad.detach().clone(), None
            opt_d_output_w, output_weight.grad = (
                output_weight.grad.detach().clone(),
                None,
            )
            torch.testing.assert_close(ref_du, opt_du)
            torch.testing.assert_close(ref_dattn, opt_dattn)
            torch.testing.assert_close(ref_d_norm_w, opt_d_norm_w)
            torch.testing.assert_close(ref_d_norm_b, opt_d_norm_b)
            torch.testing.assert_close(ref_dx, opt_dx)
            torch.testing.assert_close(ref_d_output_w, opt_d_output_w)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([100, 128, 256, 1300]),
        max_targets=st.sampled_from([20, 512]),
        embedding_dim=st.sampled_from([16, 32, 64]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contextual_seq_len=st.sampled_from([0]),
        has_max_attn_len=st.sampled_from([False, True]),
        sort_by_length=st.sampled_from([True, False]),
        recompute_uvqk_in_backward=st.sampled_from([True, False]),
        recompute_normed_x_in_backward=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=150,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_preprocess_and_attention(self, *args, **kwargs) -> None:
        self._test_hstu_preprocess_and_attention(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_hstu_preprocess_and_attention(
        self,
        batch_size: int,
        heads: int,
        max_uih_len: int,
        max_targets: int,
        embedding_dim: int,
        attn_dim: int,
        hidden_dim: int,
        causal: bool,
        has_multiple_targets: bool,
        has_max_attn_len: bool,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        test_backward: bool,
        contextual_seq_len: int,
        sort_by_length: bool,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sparsity: float = -1.0,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        set_dev_mode(True)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention,
        )

        alpha = 1.0 / (attn_dim**0.5)
        if sparsity > 0.0:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=max_uih_len,
                sparsity=sparsity,
                device=torch.device("cuda"),
            )
        else:
            lengths = torch.randint(
                max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
            )

        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = lengths + num_targets
        max_seq_len = max_uih_len + max_targets
        if has_max_attn_len:
            max_attn_len = random.randint(1, max_uih_len // 5)
        else:
            max_attn_len = 0
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)

        L = int(seq_offsets[-1].item())

        x = (
            torch.empty((L, embedding_dim), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_weight = (
            torch.empty((embedding_dim,), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_bias = (
            torch.empty(
                (embedding_dim,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_eps = 1e-6
        uvqk_weight = (
            torch.empty(
                (
                    embedding_dim,
                    (hidden_dim * 2 + attn_dim * 2) * heads,
                ),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        uvqk_bias = (
            torch.empty(
                (hidden_dim * 2 + attn_dim * 2) * heads,
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        # ref implementation
        ref_u, ref_attn_output, _, _ = hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=alpha,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
            kernel=ref_kernel,
        )
        ref_out = ref_u + ref_attn_output
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)

        # pyre-ignore
        ref_dx, x.grad = x.grad.clone(), None
        ref_d_norm_weight, norm_weight.grad = norm_weight.grad.clone(), None
        ref_d_norm_bias, norm_bias.grad = norm_bias.grad.clone(), None
        ref_d_uvqk_weight, uvqk_weight.grad = uvqk_weight.grad.clone(), None
        ref_d_uvqk_bias, uvqk_bias.grad = uvqk_bias.grad.clone(), None

        # real implementation
        x = x.detach().clone().requires_grad_()
        norm_weight = norm_weight.detach().clone().requires_grad_()
        norm_bias = norm_bias.detach().clone().requires_grad_()
        uvqk_weight = uvqk_weight.detach().clone().requires_grad_()
        uvqk_bias = uvqk_bias.detach().clone().requires_grad_()
        real_u, real_attn_output, _, _ = hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=alpha,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
            kernel=real_kernel,
        )
        real_out = real_u + real_attn_output
        torch.testing.assert_close(
            ref_u,
            real_u,
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            ref_attn_output,
            real_attn_output,
            atol=atol,
            rtol=rtol,
        )
        if test_backward:
            # real implementation
            dout = dout.detach().clone()
            real_out.backward(dout)
            (
                real_dx,
                real_d_norm_weight,
                real_d_norm_bias,
                real_d_uvqk_weight,
                real_d_uvqk_bias,
            ) = (
                x.grad.clone(),
                norm_weight.grad.clone(),
                norm_bias.grad.clone(),
                uvqk_weight.grad.clone(),
                uvqk_bias.grad.clone(),
            )
            torch.testing.assert_close(ref_dx, real_dx, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                ref_d_norm_weight, real_d_norm_weight, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_norm_bias, real_d_norm_bias, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_uvqk_weight, real_d_uvqk_weight, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_uvqk_bias, real_d_uvqk_bias, atol=atol, rtol=rtol
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_deferred_silu_end_to_end(self) -> None:
        from generative_recommenders.ops.hstu_compute import (
            hstu_compute_output,
            hstu_preprocess_and_attention,
        )

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float32
        heads = 2
        attn_dim = 16
        hidden_dim = 16
        embedding_dim = heads * hidden_dim
        lengths = torch.tensor([12, 9], dtype=torch.int64, device=device)
        seq_offsets = torch.zeros(3, dtype=torch.int64, device=device)
        seq_offsets[1:] = lengths.cumsum(0)
        rows = int(seq_offsets[-1].item())
        uvqk_dim = 2 * heads * (hidden_dim + attn_dim)
        output_dim = heads * hidden_dim

        tensors = {
            "x": torch.randn(rows, embedding_dim, dtype=dtype, device=device)
            * 0.02,
            "input_norm_weight": torch.randn(
                embedding_dim, dtype=dtype, device=device
            )
            * 0.02,
            "input_norm_bias": torch.randn(
                embedding_dim, dtype=dtype, device=device
            )
            * 0.02,
            "uvqk_weight": torch.randn(
                embedding_dim, uvqk_dim, dtype=dtype, device=device
            )
            * 0.02,
            "uvqk_bias": torch.randn(uvqk_dim, dtype=dtype, device=device)
            * 0.02,
            "output_norm_weight": torch.randn(
                output_dim, dtype=dtype, device=device
            )
            * 0.02,
            "output_norm_bias": torch.randn(
                output_dim, dtype=dtype, device=device
            )
            * 0.02,
            "output_weight": torch.randn(
                3 * output_dim, embedding_dim, dtype=dtype, device=device
            )
            * 0.02,
        }
        dout = torch.randn(rows, embedding_dim, dtype=dtype, device=device)

        def run(defer_silu: bool):
            values = {
                name: value.detach().clone().requires_grad_()
                for name, value in tensors.items()
            }
            u, attn, _, _ = hstu_preprocess_and_attention(
                x=values["x"],
                norm_weight=values["input_norm_weight"],
                norm_bias=values["input_norm_bias"],
                norm_eps=1e-6,
                num_heads=heads,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                uvqk_weight=values["uvqk_weight"],
                uvqk_bias=values["uvqk_bias"],
                max_seq_len=12,
                seq_offsets=seq_offsets,
                attn_alpha=1.0 / (attn_dim**0.5),
                causal=True,
                num_targets=None,
                max_attn_len=0,
                contextual_seq_len=0,
                recompute_uvqk_in_backward=True,
                recompute_normed_x_in_backward=True,
                sort_by_length=False,
                kernel=HammerKernel.TRITON,
                enable_tma=False,
                defer_silu=defer_silu,
            )
            output = hstu_compute_output(
                attn=attn,
                u=u,
                x=values["x"],
                norm_weight=values["output_norm_weight"],
                norm_bias=values["output_norm_bias"],
                norm_eps=1e-6,
                dropout_ratio=0.0,
                output_weight=values["output_weight"],
                group_norm=False,
                num_heads=heads,
                linear_dim=hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="silu" if defer_silu else "none",
                training=True,
                kernel=HammerKernel.TRITON,
                recompute_y_in_backward=True,
            )
            output.backward(dout)
            return output.detach(), {
                name: value.grad.detach().clone() for name, value in values.items()
            }

        ordinary_output, ordinary_grads = run(False)
        fused_output, fused_grads = run(True)
        torch.testing.assert_close(
            ordinary_output, fused_output, atol=2e-5, rtol=2e-5
        )
        for name in ordinary_grads:
            torch.testing.assert_close(
                ordinary_grads[name],
                fused_grads[name],
                atol=5e-4,
                rtol=5e-4,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_layer_norm_and_weight_refresh(self) -> None:
        if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
            self.skipTest("Plan 08 FP8 path requires ROCm scaled-mm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8
        from generative_recommenders.ops.triton.triton_layer_norm import (
            triton_weighted_layer_norm_fwd,
        )

        old_enabled = fp8._HSTU_FP8_GEMM
        old_telemetry_interval = fp8._HSTU_FP8_TELEMETRY_INTERVAL
        fp8._HSTU_FP8_GEMM = True
        try:
            device = torch.device("cuda")
            torch.manual_seed(20260824)
            x = torch.randn((4096, 512), device=device, dtype=torch.bfloat16)
            norm_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
            norm_bias = torch.zeros(512, device=device, dtype=torch.bfloat16)
            weight = torch.nn.Parameter(
                torch.randn(
                    (512, 2048), device=device, dtype=torch.float32
                )
                / 512**0.5
            )
            bias = torch.zeros(2048, device=device, dtype=torch.float32)

            normalized, _, _ = triton_weighted_layer_norm_fwd(
                x, norm_weight, norm_bias, 1.0e-6
            )
            scale = fp8.calibrated_scale(normalized)
            scale_value = float(scale.item())
            normalized_fp8, _, _ = triton_weighted_layer_norm_fwd(
                x,
                norm_weight,
                norm_bias,
                1.0e-6,
                output_fp8=True,
                inv_scale=1.0 / scale_value,
            )
            self.assertEqual(normalized_fp8.dtype, torch.float8_e4m3fn)

            fp8.clear_fp8_weight_cache()
            candidate = fp8.scaled_addmm_fp8_preq(
                normalized_fp8,
                scale,
                weight,
                bias,
            )
            reference = torch.addmm(
                bias.bfloat16(),
                normalized,
                weight.bfloat16(),
            )
            torch.testing.assert_close(
                candidate.float(),
                reference.float(),
                atol=0.35,
                rtol=0.12,
            )
            first = fp8.fp8_weight_cache_stats()
            fp8._HSTU_FP8_TELEMETRY_INTERVAL = 1
            fp8.scaled_addmm_fp8_preq(
                normalized_fp8,
                scale,
                weight,
                bias,
            )
            second = fp8.fp8_weight_cache_stats()
            self.assertEqual(first["misses"], 1)
            self.assertGreater(second["hits"], first["hits"])
            self.assertTrue(fp8.fp8_runtime_telemetry())
            with tempfile.TemporaryDirectory() as capability_dir:
                with mock.patch.dict(
                    os.environ,
                    {
                        "DLRM_HSTU_FP8_CAPABILITY_DIR": capability_dir,
                        "RANK": "3",
                        "LOCAL_RANK": "3",
                        "WORLD_SIZE": "8",
                    },
                ):
                    fp8.record_fp8_capability("phi1_input_ln_projection")
                    fp8.record_fp8_capability("phi2_qkv_attention")
                capability = json.loads(
                    (Path(capability_dir) / "rank3.json").read_text()
                )
                self.assertEqual(capability["rank"], 3)
                self.assertEqual(capability["world_size"], 8)
                self.assertEqual(
                    capability["sites"],
                    ["phi1_input_ln_projection", "phi2_qkv_attention"],
                )
            empty = fp8.scaled_addmm_fp8_preq(
                normalized_fp8[:0],
                scale,
                weight,
                bias,
            )
            self.assertEqual(tuple(empty.shape), (0, 2048))
            self.assertEqual(empty.dtype, torch.bfloat16)
            scale_state = torch.zeros(
                fp8.fp8_scale_state_size(),
                dtype=torch.float32,
                device=device,
            )

            def remote_amax(tensor: torch.Tensor, **_kwargs) -> None:
                tensor.fill_(2.0)

            with (
                mock.patch.object(
                    torch.distributed, "is_available", return_value=True
                ),
                mock.patch.object(
                    torch.distributed, "is_initialized", return_value=True
                ),
                mock.patch.object(
                    torch.distributed, "get_world_size", return_value=8
                ),
                mock.patch.object(
                    torch.distributed, "all_reduce", side_effect=remote_amax
                ) as all_reduce,
            ):
                fp8.calibrate_scale_buffer(
                    scale_state,
                    normalized[:0],
                    synchronize=True,
                )
            all_reduce.assert_called_once()
            self.assertEqual(float(scale_state[1]), 1.0)
            self.assertEqual(float(scale_state[2]), 2.0)

            cache_state = torch.ones(
                1, dtype=torch.float32, device=device
            )
            self.assertEqual(fp8.activation_scale_value(cache_state), 1.0)
            cache_state.copy_(
                torch.full_like(cache_state, 2.0)
            )
            self.assertEqual(fp8.activation_scale_value(cache_state), 2.0)

            with mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=mock.Mock(gcnArchName="gfx942:sramecc+"),
            ):
                self.assertFalse(fp8.fp8_supported(normalized, weight))

            optimizer = torch.optim.Adam([weight], lr=1.0e-3)
            weight.grad = torch.ones_like(weight)
            optimizer.step()
            self.assertTrue(
                all(
                    value.dtype == torch.float32
                    for value in optimizer.state[weight].values()
                    if isinstance(value, torch.Tensor)
                )
            )
            refreshed = fp8.scaled_addmm_fp8_preq(
                normalized_fp8,
                scale,
                weight,
                bias,
            )
            third = fp8.fp8_weight_cache_stats()
            self.assertEqual(third["misses"], 2)
            self.assertFalse(torch.equal(candidate, refreshed))
            with self.assertRaises(RuntimeError):
                fp8.scaled_addmm_fp8_preq(
                    normalized_fp8,
                    scale,
                    weight[:, :-1],
                    bias[:-1],
                )
        finally:
            fp8._HSTU_FP8_GEMM = old_enabled
            fp8._HSTU_FP8_TELEMETRY_INTERVAL = old_telemetry_interval
            fp8.clear_fp8_weight_cache()

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_output_projection_nonzero_bias_ordering(self) -> None:
        if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
            self.skipTest("Plan 08 FP8 path requires ROCm scaled-mm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8

        old_enabled = fp8._HSTU_FP8_GEMM
        fp8._HSTU_FP8_GEMM = True
        try:
            device = torch.device("cuda")
            torch.manual_seed(20260830)
            rows, input_dim, output_dim = 256, 64, 128
            activation = (
                torch.randn(
                    (rows, input_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.05
            )
            weight = torch.nn.Parameter(
                torch.randn(
                    (input_dim, output_dim),
                    dtype=torch.float32,
                    device=device,
                )
                / input_dim**0.5
            )
            bias_values = torch.tensor(
                [-1.5, -0.75, 0.75, 1.5],
                dtype=torch.float32,
                device=device,
            )
            bias = bias_values[
                torch.arange(output_dim, device=device) % bias_values.numel()
            ]

            input_scale = fp8.calibrated_scale(activation)
            activation_fp8 = fp8.quantize_with_scale(
                activation,
                input_scale,
            )
            reference = torch.addmm(
                bias.bfloat16(),
                activation,
                weight.bfloat16(),
            )
            output_scale = fp8.calibrated_scale(reference)

            fp8.clear_fp8_weight_cache()
            candidate_fp8 = fp8.scaled_mm_fp8_out_preq(
                activation_fp8=activation_fp8,
                scale_a=input_scale,
                weight=weight,
                bias=bias,
                output_scale=output_scale,
            )
            first = fp8.fp8_weight_cache_stats()
            repeated_fp8 = fp8.scaled_mm_fp8_out_preq(
                activation_fp8=activation_fp8,
                scale_a=input_scale,
                weight=weight,
                bias=bias,
                output_scale=output_scale,
            )
            second = fp8.fp8_weight_cache_stats()
            no_bias_fp8 = fp8.scaled_mm_fp8_out_preq(
                activation_fp8=activation_fp8,
                scale_a=input_scale,
                weight=weight,
                output_scale=output_scale,
            )

            self.assertEqual(candidate_fp8.dtype, torch.float8_e4m3fn)
            self.assertTrue(torch.equal(candidate_fp8, repeated_fp8))
            self.assertEqual(first["misses"], 1)
            self.assertEqual(second["misses"], first["misses"])
            self.assertGreater(second["hits"], first["hits"])

            candidate = candidate_fp8.float() * output_scale
            no_bias = no_bias_fp8.float() * output_scale
            torch.testing.assert_close(
                candidate,
                reference.float(),
                atol=0.35,
                rtol=0.12,
            )
            # Using the same quantized operands and output scale isolates the
            # FP8-output epilogue: the dequantized delta must be the BF16 bias,
            # not bias multiplied by (or added after) scale_result.
            torch.testing.assert_close(
                candidate - no_bias,
                bias.expand_as(candidate),
                atol=0.2,
                rtol=0.12,
            )
        finally:
            fp8._HSTU_FP8_GEMM = old_enabled
            fp8.clear_fp8_weight_cache()

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_output_ln_preserves_dropout_mask(self) -> None:
        if torch.version.hip is None:
            self.skipTest("Plan 08 FP8 path requires ROCm")

        from generative_recommenders.ops.triton.triton_fp8 import calibrated_scale
        from generative_recommenders.ops.triton.triton_hstu_linear import (
            triton_layer_norm_mul_dropout_fwd,
        )

        device = torch.device("cuda")
        torch.manual_seed(20260825)
        x = torch.randn((1024, 512), device=device, dtype=torch.bfloat16)
        u = torch.randn_like(x)
        weight = torch.ones(512, device=device, dtype=torch.bfloat16)
        bias = torch.zeros(512, device=device, dtype=torch.bfloat16)
        seed = 20260826

        reference = triton_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=1.0e-6,
            dropout_ratio=0.1,
            training=True,
            concat_u=True,
            concat_x=True,
            seed=seed,
        )
        scale = calibrated_scale(reference[0])
        candidate = triton_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=1.0e-6,
            dropout_ratio=0.1,
            training=True,
            concat_u=True,
            concat_x=True,
            seed=seed,
            output_fp8=True,
            inv_scale=1.0 / float(scale.item()),
        )
        dequantized = candidate[0].float() * scale
        torch.testing.assert_close(
            dequantized,
            reference[0].float(),
            atol=0.35,
            rtol=0.12,
        )
        if reference[6] is not None and candidate[6] is not None:
            self.assertTrue(torch.equal(reference[6], candidate[6]))

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_qkv_attention_forward_and_backward(self) -> None:
        if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
            self.skipTest("Plan 08 FP8 path requires ROCm scaled-mm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8
        import generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention as preprocess
        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention,
        )
        from generative_recommenders.ops.triton.triton_layer_norm import (
            triton_weighted_layer_norm_fwd,
        )

        flag_names = (
            "_HSTU_FP8_GEMM",
            "_HSTU_FP8_FUSE_LN",
            "_HSTU_FP8_FUSE_QKV",
            "_HSTU_FP8_ATTENTION",
            "_HSTU_FP8_FAIL_CLOSED",
        )
        old_flags = {name: getattr(fp8, name) for name in flag_names}
        original_fp8_qkv = preprocess.scaled_mm_fp8_out_preq
        fp8_qkv_outputs: list[torch.Tensor] = []
        fp8_qkv_no_bias_outputs: list[torch.Tensor] = []

        def capture_fp8_qkv(*args, **kwargs):
            output = original_fp8_qkv(*args, **kwargs)
            no_bias_kwargs = dict(kwargs)
            no_bias_kwargs["bias"] = None
            no_bias_output = original_fp8_qkv(*args, **no_bias_kwargs)
            fp8_qkv_outputs.append(output.detach().clone())
            fp8_qkv_no_bias_outputs.append(no_bias_output.detach().clone())
            return output

        try:
            device = torch.device("cuda")
            torch.manual_seed(20260829)
            lengths = torch.tensor([8, 16], dtype=torch.int64, device=device)
            offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    lengths.cumsum(0),
                )
            )
            rows = int(offsets[-1])
            x_control = (
                torch.randn((rows, 32), dtype=torch.bfloat16, device=device)
                * 0.05
            ).requires_grad_(True)
            x_candidate = x_control.detach().clone().requires_grad_(True)
            norm_weight_control = torch.ones(
                32, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            norm_bias_control = torch.zeros(
                32, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            norm_weight_candidate = (
                norm_weight_control.detach().clone().requires_grad_(True)
            )
            norm_bias_candidate = (
                norm_bias_control.detach().clone().requires_grad_(True)
            )
            uvqk_weight_control = (
                torch.randn((32, 128), dtype=torch.float32, device=device)
                / 32**0.5
            ).requires_grad_(True)
            uvqk_bias_control = torch.linspace(
                -1.5,
                1.5,
                128,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            uvqk_weight_candidate = (
                uvqk_weight_control.detach().clone().requires_grad_(True)
            )
            uvqk_bias_candidate = (
                uvqk_bias_control.detach().clone().requires_grad_(True)
            )
            kwargs = {
                "norm_eps": 1.0e-6,
                "num_heads": 2,
                "attn_dim": 16,
                "hidden_dim": 16,
                "max_seq_len": 16,
                "seq_offsets": offsets,
                "attn_alpha": 0.25,
                "causal": True,
                "num_targets": None,
                "max_attn_len": 0,
                "contextual_seq_len": 0,
                "recompute_uvqk_in_backward": True,
                "recompute_normed_x_in_backward": True,
                "sort_by_length": False,
                "kernel": HammerKernel.TRITON,
                "enable_tma": False,
            }

            for name in flag_names:
                setattr(fp8, name, False)
            control = hstu_preprocess_and_attention(
                x=x_control,
                norm_weight=norm_weight_control,
                norm_bias=norm_bias_control,
                uvqk_weight=uvqk_weight_control,
                uvqk_bias=uvqk_bias_control,
                **kwargs,
            )

            normalized, _, _ = triton_weighted_layer_norm_fwd(
                x_candidate,
                norm_weight_candidate,
                norm_bias_candidate,
                1.0e-6,
            )

            def frozen_state(value: torch.Tensor) -> torch.Tensor:
                state = torch.zeros(
                    fp8.fp8_scale_state_size(),
                    dtype=torch.float32,
                    device=device,
                )
                state[0] = fp8.calibrated_scale(value)
                state[1] = state.numel() - 2
                state[2:].fill_(float(value.detach().abs().max()))
                return state

            input_scale = frozen_state(normalized)
            vqk = torch.addmm(
                uvqk_bias_candidate[32:].to(normalized.dtype),
                normalized,
                uvqk_weight_candidate[:, 32:].to(normalized.dtype),
            )
            qkv_scale = frozen_state(vqk)

            for name in flag_names:
                setattr(fp8, name, True)
            preprocess.scaled_mm_fp8_out_preq = capture_fp8_qkv
            candidate = hstu_preprocess_and_attention(
                x=x_candidate,
                norm_weight=norm_weight_candidate,
                norm_bias=norm_bias_candidate,
                uvqk_weight=uvqk_weight_candidate,
                uvqk_bias=uvqk_bias_candidate,
                fp8_input_scale=input_scale,
                fp8_qkv_scale=qkv_scale,
                **kwargs,
            )
            self.assertEqual(len(fp8_qkv_outputs), 1)
            self.assertEqual(len(fp8_qkv_no_bias_outputs), 1)
            qkv_scale_value = fp8.fp8_scale_tensor(qkv_scale)
            torch.testing.assert_close(
                fp8_qkv_outputs[0].float() * qkv_scale_value,
                vqk.float(),
                atol=0.35,
                rtol=0.12,
            )
            torch.testing.assert_close(
                (
                    fp8_qkv_outputs[0].float()
                    - fp8_qkv_no_bias_outputs[0].float()
                )
                * qkv_scale_value,
                uvqk_bias_candidate[32:].detach().expand_as(
                    fp8_qkv_outputs[0]
                ),
                atol=0.25,
                rtol=0.15,
            )
            for got, expected in zip(candidate[:2], control[:2]):
                torch.testing.assert_close(
                    got.float(),
                    expected.float(),
                    atol=0.5,
                    rtol=0.2,
                )
            candidate_attention = candidate[1].float()
            control_attention = control[1].float()
            self.assertLessEqual(
                float(
                    torch.linalg.vector_norm(
                        candidate_attention - control_attention
                    )
                    / torch.linalg.vector_norm(control_attention).clamp_min(1.0e-12)
                ),
                0.35,
            )
            self.assertGreaterEqual(
                float(
                    torch.nn.functional.cosine_similarity(
                        candidate_attention.reshape(1, -1),
                        control_attention.reshape(1, -1),
                    )
                ),
                0.95,
            )

            sum(value.float().sum() for value in control[:2]).backward()
            sum(value.float().sum() for value in candidate[:2]).backward()
            self.assertEqual(len(fp8_qkv_outputs), 2)
            self.assertEqual(len(fp8_qkv_no_bias_outputs), 2)
            self.assertTrue(
                torch.equal(fp8_qkv_outputs[0], fp8_qkv_outputs[1])
            )
            self.assertTrue(
                torch.equal(
                    fp8_qkv_no_bias_outputs[0],
                    fp8_qkv_no_bias_outputs[1],
                )
            )
            assert x_candidate.grad is not None and x_control.grad is not None
            candidate_x_grad = x_candidate.grad.float()
            control_x_grad = x_control.grad.float()
            x_grad_delta = candidate_x_grad - control_x_grad
            self.assertTrue(torch.isfinite(candidate_x_grad).all())
            self.assertLessEqual(
                float(
                    torch.linalg.vector_norm(x_grad_delta)
                    / torch.linalg.vector_norm(control_x_grad).clamp_min(
                        1.0e-12
                    )
                ),
                0.05,
            )
            self.assertGreaterEqual(
                float(
                    torch.nn.functional.cosine_similarity(
                        candidate_x_grad.reshape(1, -1),
                        control_x_grad.reshape(1, -1),
                    )
                ),
                0.999,
            )
            for got, expected in (
                (norm_weight_candidate.grad, norm_weight_control.grad),
                (norm_bias_candidate.grad, norm_bias_control.grad),
                (uvqk_weight_candidate.grad, uvqk_weight_control.grad),
                (uvqk_bias_candidate.grad, uvqk_bias_control.grad),
            ):
                assert got is not None and expected is not None
                torch.testing.assert_close(
                    got.float(),
                    expected.float(),
                    atol=0.5,
                    rtol=0.2,
                )
        finally:
            preprocess.scaled_mm_fp8_out_preq = original_fp8_qkv
            for name, value in old_flags.items():
                setattr(fp8, name, value)
            fp8.clear_fp8_weight_cache()

    @unittest.skipIf(*gpu_unavailable)
    def test_saved_fp8_qkv_early_release_and_retained_graph(self) -> None:
        if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
            self.skipTest("Plan 09 FP8 path requires ROCm scaled-mm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8
        import generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention as preprocess
        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention,
        )
        from generative_recommenders.ops.triton.triton_layer_norm import (
            triton_weighted_layer_norm_fwd,
        )

        flag_names = (
            "_HSTU_FP8_GEMM",
            "_HSTU_FP8_FUSE_LN",
            "_HSTU_FP8_FUSE_QKV",
            "_HSTU_FP8_ATTENTION",
            "_HSTU_FP8_FAIL_CLOSED",
        )
        old_flags = {name: getattr(fp8, name) for name in flag_names}
        old_save = preprocess._SAVE_FP8_UVQK
        old_early_release = preprocess._EARLY_RELEASE_FP8_UVQK
        original_fp8_qkv = preprocess.scaled_mm_fp8_out_preq
        fp8_qkv_calls = 0

        def capture_fp8_qkv(*args, **kwargs):
            nonlocal fp8_qkv_calls
            fp8_qkv_calls += 1
            return original_fp8_qkv(*args, **kwargs)

        try:
            for name in flag_names:
                setattr(fp8, name, True)
            preprocess._SAVE_FP8_UVQK = True
            preprocess.scaled_mm_fp8_out_preq = capture_fp8_qkv

            device = torch.device("cuda")
            torch.manual_seed(20260901)
            lengths = torch.tensor([8, 16], dtype=torch.int64, device=device)
            offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    lengths.cumsum(0),
                )
            )
            rows = int(offsets[-1])
            base = {
                "x": torch.randn(
                    (rows, 32), dtype=torch.bfloat16, device=device
                )
                * 0.05,
                "norm_weight": torch.ones(
                    32, dtype=torch.bfloat16, device=device
                ),
                "norm_bias": torch.zeros(
                    32, dtype=torch.bfloat16, device=device
                ),
                "uvqk_weight": torch.randn(
                    (32, 128), dtype=torch.float32, device=device
                )
                / 32**0.5,
                "uvqk_bias": torch.linspace(
                    -0.5,
                    0.5,
                    128,
                    dtype=torch.float32,
                    device=device,
                ),
            }

            def frozen_state(value: torch.Tensor) -> torch.Tensor:
                state = torch.zeros(
                    fp8.fp8_scale_state_size(),
                    dtype=torch.float32,
                    device=device,
                )
                state[0] = fp8.calibrated_scale(value)
                state[1] = state.numel() - 2
                state[2:].fill_(float(value.detach().abs().max()))
                return state

            normalized, _, _ = triton_weighted_layer_norm_fwd(
                base["x"],
                base["norm_weight"],
                base["norm_bias"],
                1.0e-6,
            )
            input_scale = frozen_state(normalized)
            reference_vqk = torch.addmm(
                base["uvqk_bias"][32:].to(normalized.dtype),
                normalized,
                base["uvqk_weight"][:, 32:].to(normalized.dtype),
            )
            qkv_scale = frozen_state(reference_vqk)
            kwargs = {
                "norm_eps": 1.0e-6,
                "num_heads": 2,
                "attn_dim": 16,
                "hidden_dim": 16,
                "max_seq_len": 16,
                "seq_offsets": offsets,
                "attn_alpha": 0.25,
                "causal": True,
                "num_targets": None,
                "max_attn_len": 0,
                "contextual_seq_len": 0,
                "recompute_uvqk_in_backward": True,
                "recompute_normed_x_in_backward": True,
                "sort_by_length": False,
                "kernel": HammerKernel.TRITON,
                "enable_tma": False,
                "fp8_input_scale": input_scale,
                "fp8_qkv_scale": qkv_scale,
            }

            def fresh_inputs() -> dict[str, torch.Tensor]:
                return {
                    name: value.detach().clone().requires_grad_(True)
                    for name, value in base.items()
                }

            def run_forward(
                values: dict[str, torch.Tensor],
            ) -> tuple[torch.Tensor, ...]:
                return hstu_preprocess_and_attention(
                    x=values["x"],
                    norm_weight=values["norm_weight"],
                    norm_bias=values["norm_bias"],
                    uvqk_weight=values["uvqk_weight"],
                    uvqk_bias=values["uvqk_bias"],
                    **kwargs,
                )

            preprocess._EARLY_RELEASE_FP8_UVQK = False
            saved_values = fresh_inputs()
            saved_outputs = run_forward(saved_values)
            saved_context = saved_outputs[0].grad_fn
            assert saved_context is not None
            self.assertTrue(
                any(
                    tensor.dtype == torch.float8_e4m3fn
                    for tensor in saved_context.saved_tensors
                )
            )
            saved_call_count = fp8_qkv_calls
            sum(value.float().sum() for value in saved_outputs[:2]).backward()
            self.assertEqual(fp8_qkv_calls, saved_call_count)
            saved_gradients = {
                name: value.grad.detach().clone()
                for name, value in saved_values.items()
            }

            preprocess._EARLY_RELEASE_FP8_UVQK = True
            early_values = fresh_inputs()
            early_outputs = run_forward(early_values)
            for got, expected in zip(early_outputs[:2], saved_outputs[:2]):
                torch.testing.assert_close(got, expected)
            early_context = early_outputs[0].grad_fn
            assert early_context is not None
            self.assertFalse(
                any(
                    tensor.dtype == torch.float8_e4m3fn
                    for tensor in early_context.saved_tensors
                )
            )
            self.assertEqual(
                early_context._ephemeral_fp8_vqk.dtype,
                torch.float8_e4m3fn,
            )
            early_call_count = fp8_qkv_calls
            early_loss = sum(value.float().sum() for value in early_outputs[:2])
            early_loss.backward(retain_graph=True)
            self.assertEqual(fp8_qkv_calls, early_call_count)
            self.assertIsNone(early_context._ephemeral_fp8_vqk)
            for name, value in early_values.items():
                torch.testing.assert_close(
                    value.grad,
                    saved_gradients[name],
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )
                value.grad = None

            early_loss.backward()
            self.assertEqual(fp8_qkv_calls, early_call_count + 1)
            for name, value in early_values.items():
                torch.testing.assert_close(
                    value.grad,
                    saved_gradients[name],
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )
        finally:
            preprocess.scaled_mm_fp8_out_preq = original_fp8_qkv
            preprocess._SAVE_FP8_UVQK = old_save
            preprocess._EARLY_RELEASE_FP8_UVQK = old_early_release
            for name, value in old_flags.items():
                setattr(fp8, name, value)
            fp8.clear_fp8_weight_cache()

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_targets_only_forward_and_backward(self) -> None:
        if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
            self.skipTest("Plan 08 FP8 path requires ROCm scaled-mm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8
        import generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention as preprocess
        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention_targets_only,
        )
        from generative_recommenders.ops.triton.triton_layer_norm import (
            triton_weighted_layer_norm_fwd,
        )

        flag_names = (
            "_HSTU_FP8_GEMM",
            "_HSTU_FP8_FUSE_LN",
            "_HSTU_FP8_FUSE_QKV",
            "_HSTU_FP8_ATTENTION",
            "_HSTU_FP8_FAIL_CLOSED",
        )
        old_flags = {name: getattr(fp8, name) for name in flag_names}
        original_fp8_vk = preprocess.scaled_mm_fp8_out_preq
        fp8_vk_outputs: list[torch.Tensor] = []

        def capture_fp8_vk(*args, **kwargs):
            output = original_fp8_vk(*args, **kwargs)
            fp8_vk_outputs.append(output.detach().clone())
            return output

        try:
            device = torch.device("cuda")
            torch.manual_seed(20260828)
            lengths = torch.tensor(
                [8, 16, 24, 32], dtype=torch.int64, device=device
            )
            offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    lengths.cumsum(0),
                )
            )
            rows = int(offsets[-1])
            x_control = (
                torch.randn(
                    (rows, 32), dtype=torch.bfloat16, device=device
                )
                * 0.05
            ).requires_grad_(True)
            x_candidate = x_control.detach().clone().requires_grad_(True)
            norm_weight_control = torch.ones(
                32, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            norm_bias_control = torch.zeros(
                32, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            norm_weight_candidate = (
                norm_weight_control.detach().clone().requires_grad_(True)
            )
            norm_bias_candidate = (
                norm_bias_control.detach().clone().requires_grad_(True)
            )
            uvqk_weight_control = (
                torch.randn(
                    (32, 128), dtype=torch.float32, device=device
                )
                / 32**0.5
            ).requires_grad_(True)
            uvqk_bias_control = torch.zeros(
                128, dtype=torch.float32, device=device, requires_grad=True
            )
            uvqk_weight_candidate = (
                uvqk_weight_control.detach().clone().requires_grad_(True)
            )
            uvqk_bias_candidate = (
                uvqk_bias_control.detach().clone().requires_grad_(True)
            )
            num_targets = torch.ones(
                lengths.numel(), dtype=torch.int64, device=device
            )

            for name in flag_names:
                setattr(fp8, name, False)
            control = hstu_preprocess_and_attention_targets_only(
                x=x_control,
                norm_weight=norm_weight_control,
                norm_bias=norm_bias_control,
                norm_eps=1.0e-6,
                num_heads=2,
                attn_dim=16,
                hidden_dim=16,
                uvqk_weight=uvqk_weight_control,
                uvqk_bias=uvqk_bias_control,
                max_seq_len=32,
                seq_offsets=offsets,
                attn_alpha=0.25,
                num_targets=num_targets,
            )

            normalized, _, _ = triton_weighted_layer_norm_fwd(
                x_candidate,
                norm_weight_candidate,
                norm_bias_candidate,
                1.0e-6,
            )
            scale_state = torch.zeros(
                fp8.fp8_scale_state_size(),
                dtype=torch.float32,
                device=device,
            )
            scale_state[0] = fp8.calibrated_scale(normalized)
            scale_state[1] = scale_state.numel() - 2
            scale_state[2:].fill_(float(normalized.float().abs().max()))
            vk_weight = torch.cat(
                (
                    uvqk_weight_candidate[:, 32:64],
                    uvqk_weight_candidate[:, 96:128],
                ),
                dim=1,
            )
            projected_vk = torch.mm(normalized, vk_weight.to(normalized.dtype))
            qkv_scale_state = torch.zeros_like(scale_state)
            qkv_scale_state[0] = fp8.calibrated_scale(projected_vk)
            qkv_scale_state[1] = qkv_scale_state.numel() - 2
            qkv_scale_state[2:].fill_(
                float(projected_vk.float().abs().max())
            )

            for name in flag_names:
                setattr(fp8, name, True)
            preprocess.scaled_mm_fp8_out_preq = capture_fp8_vk
            candidate = hstu_preprocess_and_attention_targets_only(
                x=x_candidate,
                norm_weight=norm_weight_candidate,
                norm_bias=norm_bias_candidate,
                norm_eps=1.0e-6,
                num_heads=2,
                attn_dim=16,
                hidden_dim=16,
                uvqk_weight=uvqk_weight_candidate,
                uvqk_bias=uvqk_bias_candidate,
                max_seq_len=32,
                seq_offsets=offsets,
                attn_alpha=0.25,
                num_targets=num_targets,
                fp8_input_scale=scale_state,
                fp8_qkv_scale=qkv_scale_state,
            )
            self.assertEqual(len(fp8_vk_outputs), 1)
            for got, expected in zip(candidate, control):
                torch.testing.assert_close(
                    got.float(),
                    expected.float(),
                    atol=0.5,
                    rtol=0.2,
                )
            candidate_attention = candidate[1].float()
            control_attention = control[1].float()
            self.assertLessEqual(
                float(
                    torch.linalg.vector_norm(
                        candidate_attention - control_attention
                    )
                    / torch.linalg.vector_norm(control_attention).clamp_min(1.0e-12)
                ),
                0.35,
            )
            self.assertGreaterEqual(
                float(
                    torch.nn.functional.cosine_similarity(
                        candidate_attention.reshape(1, -1),
                        control_attention.reshape(1, -1),
                    )
                ),
                0.95,
            )

            sum(value.float().sum() for value in control).backward()
            sum(value.float().sum() for value in candidate).backward()
            self.assertEqual(len(fp8_vk_outputs), 2)
            self.assertTrue(
                torch.equal(fp8_vk_outputs[0], fp8_vk_outputs[1])
            )
            for got, expected in (
                (x_candidate.grad, x_control.grad),
                (norm_weight_candidate.grad, norm_weight_control.grad),
                (norm_bias_candidate.grad, norm_bias_control.grad),
                (uvqk_weight_candidate.grad, uvqk_weight_control.grad),
                (uvqk_bias_candidate.grad, uvqk_bias_control.grad),
            ):
                assert got is not None and expected is not None
                torch.testing.assert_close(
                    got.float(),
                    expected.float(),
                    atol=0.5,
                    rtol=0.2,
                )
        finally:
            preprocess.scaled_mm_fp8_out_preq = original_fp8_vk
            for name, value in old_flags.items():
                setattr(fp8, name, value)
            fp8.clear_fp8_weight_cache()

    @unittest.skipIf(*gpu_unavailable)
    def test_fp8_unsupported_shape_fallback_and_strict_mode(self) -> None:
        if torch.version.hip is None:
            self.skipTest("Plan 08 FP8 path requires ROCm")

        import generative_recommenders.ops.triton.triton_fp8 as fp8
        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention,
        )

        flag_names = (
            "_HSTU_FP8_GEMM",
            "_HSTU_FP8_FUSE_LN",
            "_HSTU_FP8_FUSE_QKV",
            "_HSTU_FP8_ATTENTION",
            "_HSTU_FP8_FAIL_CLOSED",
        )
        old_flags = {name: getattr(fp8, name) for name in flag_names}
        try:
            device = torch.device("cuda")
            x = torch.randn(
                (16, 30), dtype=torch.bfloat16, device=device
            )
            norm_weight = torch.ones(
                30, dtype=torch.bfloat16, device=device
            )
            norm_bias = torch.zeros(
                30, dtype=torch.bfloat16, device=device
            )
            uvqk_weight = torch.randn(
                (30, 128), dtype=torch.float32, device=device
            )
            uvqk_bias = torch.zeros(
                128, dtype=torch.float32, device=device
            )
            offsets = torch.tensor(
                [0, 8, 16], dtype=torch.int64, device=device
            )
            scale_state = torch.ones(
                fp8.fp8_scale_state_size(),
                dtype=torch.float32,
                device=device,
            )
            fp8._HSTU_FP8_GEMM = True
            fp8._HSTU_FP8_FUSE_LN = True
            fp8._HSTU_FP8_FUSE_QKV = False
            fp8._HSTU_FP8_ATTENTION = False
            fp8._HSTU_FP8_FAIL_CLOSED = False

            kwargs = dict(
                x=x,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                norm_eps=1.0e-6,
                num_heads=2,
                attn_dim=16,
                hidden_dim=16,
                uvqk_weight=uvqk_weight,
                uvqk_bias=uvqk_bias,
                max_seq_len=8,
                seq_offsets=offsets,
                attn_alpha=0.25,
                causal=True,
                num_targets=None,
                max_attn_len=0,
                contextual_seq_len=0,
                recompute_uvqk_in_backward=True,
                recompute_normed_x_in_backward=True,
                sort_by_length=False,
                kernel=HammerKernel.TRITON,
                enable_tma=False,
                fp8_input_scale=scale_state,
            )
            fallback = hstu_preprocess_and_attention(**kwargs)
            self.assertTrue(
                all(torch.isfinite(value).all() for value in fallback[:2])
            )

            fp8._HSTU_FP8_FAIL_CLOSED = True
            with self.assertRaisesRegex(RuntimeError, "unsupported contract"):
                hstu_preprocess_and_attention(**kwargs)
        finally:
            for name, value in old_flags.items():
                setattr(fp8, name, value)

    def test_saved_fp8_qkv_layer_mask(self) -> None:
        from generative_recommenders.modules.stu import (
            STULayer,
            STULayerConfig,
            STUStack,
        )

        config = STULayerConfig(
            embedding_dim=32,
            num_heads=2,
            hidden_dim=16,
            attention_dim=16,
        )
        layers = [STULayer(config) for _ in range(3)]
        STUStack(layers)
        self.assertEqual(
            [layer._stack_layer_index for layer in layers],
            [0, 1, 2],
        )
        with mock.patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_SAVE_UVQK_LAYERS": "0"},
        ):
            self.assertEqual(
                [layer._allow_saved_fp8_uvqk() for layer in layers],
                [True, False, False],
            )
        with mock.patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_SAVE_UVQK_LAYERS": "1"},
        ):
            self.assertEqual(
                [layer._allow_saved_fp8_uvqk() for layer in layers],
                [False, True, False],
            )
        with mock.patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_SAVE_UVQK_LAYERS": "bad"},
        ):
            with self.assertRaisesRegex(RuntimeError, "comma-separated"):
                layers[0]._allow_saved_fp8_uvqk()
        with mock.patch.dict(
            os.environ,
            {"DLRM_HSTU_FP8_SAVE_UVQK_LAYERS": "0"},
        ):
            with self.assertRaisesRegex(RuntimeError, "STUStack"):
                STULayer(config)._allow_saved_fp8_uvqk()

    def test_fp8_scale_checkpoint_and_legacy_load(self) -> None:
        from generative_recommenders.dlrm_v4.checkpoint import load_dense_state_dict
        from generative_recommenders.modules.stu import STULayer, STULayerConfig
        from generative_recommenders.ops.triton import triton_fp8 as fp8

        calibration = torch.zeros(fp8.fp8_scale_state_size(), dtype=torch.float32)
        empty_calibration = torch.zeros_like(calibration)
        self.assertEqual(
            fp8.calibrate_scale_buffer(
                empty_calibration,
                torch.empty(0),
                synchronize=False,
            ),
            0.0,
        )
        self.assertEqual(int(torch.count_nonzero(empty_calibration)), 0)
        for _ in range(calibration.numel() - 3):
            self.assertEqual(
                fp8.calibrate_scale_buffer(
                    calibration, torch.tensor([1.0]), synchronize=False
                ),
                0.0,
            )
        self.assertGreater(
            fp8.calibrate_scale_buffer(
                calibration, torch.tensor([2.0]), synchronize=False
            ),
            0.0,
        )
        self.assertEqual(int(calibration[1]), calibration.numel() - 2)
        self.assertGreater(float(calibration[2:].max()), 0.0)

        config = STULayerConfig(
            embedding_dim=32,
            num_heads=2,
            hidden_dim=16,
            attention_dim=16,
        )
        source = STULayer(config)
        source._fp8_input_scale.copy_(calibration)
        source._fp8_qkv_scale.copy_(calibration)
        source._fp8_output_scale.copy_(calibration)
        source._fp8_input_scale[0] = 0.125
        source._fp8_qkv_scale[0] = 0.1875
        source._fp8_output_scale[0] = 0.25
        checkpoint = {
            key: value.detach().clone() for key, value in source.state_dict().items()
        }

        restored = STULayer(config)
        with mock.patch.object(
            fp8, "clear_fp8_scale_runtime_cache"
        ) as clear_scales, mock.patch.object(
            fp8, "clear_fp8_weight_cache"
        ) as clear_weights:
            load_dense_state_dict(restored, checkpoint)
            clear_scales.assert_called_once()
            clear_weights.assert_called_once()
        self.assertEqual(float(restored._fp8_input_scale[0]), 0.125)
        self.assertEqual(float(restored._fp8_qkv_scale[0]), 0.1875)
        self.assertEqual(float(restored._fp8_output_scale[0]), 0.25)
        self.assertEqual(int(restored._fp8_input_scale[1]), int(calibration[1]))
        self.assertTrue(
            torch.equal(restored._fp8_input_scale[2:], calibration[2:])
        )
        self.assertTrue(
            torch.equal(restored._fp8_input_scale[1:], calibration[1:])
        )

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(checkpoint, handle.name)
            from_disk = torch.load(
                handle.name, map_location="cpu", weights_only=False
            )
        disk_restored = STULayer(config)
        load_dense_state_dict(disk_restored, from_disk)
        self.assertTrue(
            torch.equal(disk_restored._fp8_input_scale, restored._fp8_input_scale)
        )
        self.assertTrue(
            torch.equal(disk_restored._fp8_qkv_scale, restored._fp8_qkv_scale)
        )
        self.assertTrue(
            torch.equal(
                disk_restored._fp8_output_scale, restored._fp8_output_scale
            )
        )
        restored_state = restored.state_dict()
        fp8_keys = [
            key
            for key in checkpoint
            if "._fp8_" in key or key.startswith("_fp8_")
        ]
        self.assertTrue(fp8_keys)
        for key in fp8_keys:
            self.assertTrue(
                torch.equal(checkpoint[key], restored_state[key]),
                f"{key} did not round-trip through the checkpoint",
            )

        legacy = {
            key: value
            for key, value in checkpoint.items()
            if "._fp8_" not in key and not key.startswith("_fp8_")
        }
        restored._fp8_input_scale.fill_(1.0)
        restored._fp8_qkv_scale.fill_(1.0)
        restored._fp8_output_scale.fill_(1.0)
        load_dense_state_dict(restored, legacy)
        self.assertEqual(int(torch.count_nonzero(restored._fp8_input_scale)), 0)
        self.assertEqual(int(torch.count_nonzero(restored._fp8_qkv_scale)), 0)
        self.assertEqual(int(torch.count_nonzero(restored._fp8_output_scale)), 0)
        input_scale_key = next(
            key for key in checkpoint if key.endswith("_fp8_input_scale")
        )
        dirty_old_history = dict(checkpoint)
        dirty_old_history[input_scale_key] = checkpoint[input_scale_key][:-1]
        with self.assertRaisesRegex(
            RuntimeError, "calibration-history shape changed"
        ):
            load_dense_state_dict(STULayer(config), dirty_old_history)
        pristine_old_history = {
            **legacy,
            input_scale_key: torch.zeros(1, dtype=torch.float32),
        }
        migrated = STULayer(config)
        load_dense_state_dict(migrated, pristine_old_history)
        self.assertEqual(int(torch.count_nonzero(migrated._fp8_input_scale)), 0)
