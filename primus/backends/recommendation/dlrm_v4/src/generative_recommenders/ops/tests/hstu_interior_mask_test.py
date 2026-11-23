#!/usr/bin/env python3

# pyre-strict

import os
import unittest
from typing import Optional
from unittest.mock import patch

import fbgemm_gpu  # noqa: F401
import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha


class HSTUInteriorMaskTest(unittest.TestCase):
    def _run_case(
        self,
        lengths_list: list[int],
        *,
        targets_list: Optional[list[int]] = None,
        max_attn_len: int = 0,
        contextual_seq_len: int = 0,
    ) -> None:
        torch.manual_seed(20260820)
        device = torch.device("cuda")
        lengths = torch.tensor(lengths_list, dtype=torch.int64, device=device)
        offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        tokens = int(offsets[-1].item())
        heads = 2
        dim = 32
        max_seq_len = max(lengths_list)
        num_targets = (
            torch.tensor(targets_list, dtype=torch.int64, device=device)
            if targets_list is not None
            else None
        )
        q = (
            torch.empty(tokens, heads, dim, dtype=torch.bfloat16, device=device)
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        k = torch.empty_like(q).uniform_(-0.1, 0.1).requires_grad_()
        v = torch.empty_like(q).uniform_(-0.1, 0.1).requires_grad_()
        dout = torch.empty_like(q).uniform_(-0.1, 0.1)

        def run(
            kernel: HammerKernel,
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            return hstu_mha(
                max_seq_len=max_seq_len,
                alpha=dim**-0.5,
                q=q_in,
                k=k_in,
                v=v_in,
                seq_offsets=offsets,
                causal=True,
                dropout_pr=0.0,
                training=True,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                sort_by_length=True,
                kernel=kernel,
            )

        def run_triton(fast_interior: str) -> tuple[torch.Tensor, ...]:
            q_in = q.detach().clone().requires_grad_()
            k_in = k.detach().clone().requires_grad_()
            v_in = v.detach().clone().requires_grad_()
            # HSTU_ATTN_BWD defaults to the FlyDSL backward, which would own the
            # gradient here and leave the specialization unexercised. It declines
            # this dim today, but that is a shape accident, not a contract.
            with patch.dict(
                os.environ,
                {
                    "TRITON_HSTU_FAST_INTERIOR": fast_interior,
                    "HSTU_ATTN_BWD": "triton",
                },
            ):
                out = run(HammerKernel.TRITON, q_in, k_in, v_in)
                out.backward(dout)
            return out.detach(), q_in.grad, k_in.grad, v_in.grad

        expected = run(HammerKernel.PYTORCH, q, k, v)
        expected.backward(dout)
        expected_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

        baseline = run_triton("0")
        actual = run_triton("1")

        # The whole claim of the specialization: it removes work on tiles that
        # cannot contain a masked element, so it must not move a single bit.
        for name, unspecialized, specialized in zip(
            ("out", "dq", "dk", "dv"), baseline, actual
        ):
            self.assertTrue(
                torch.equal(specialized, unspecialized),
                f"{name} differs from the unspecialized kernel",
            )

        torch.testing.assert_close(actual[0], expected, atol=2e-2, rtol=2e-2)
        for actual_grad, expected_grad in zip(actual[1:], expected_grads):
            self.assertIsNotNone(actual_grad)
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-2, rtol=2e-2)

    @unittest.skipIf(*gpu_unavailable)
    def test_exact_tile_boundaries(self) -> None:
        cases = (
            {
                "lengths_list": [0, 1, 31, 32, 33, 127, 128, 129, 257],
            },
            {
                "lengths_list": [33, 64, 127, 128, 129, 255, 256, 257],
                "targets_list": [1, 2, 1, 2, 3, 1, 2, 3],
            },
            {
                "lengths_list": [129, 257, 513],
                "targets_list": [1, 2, 3],
                "max_attn_len": 64,
            },
            {
                "lengths_list": [20, 31, 32, 33, 129, 257],
                "contextual_seq_len": 10,
            },
        )
        for case in cases:
            with self.subTest(case=case):
                self._run_case(**case)


if __name__ == "__main__":
    unittest.main()
