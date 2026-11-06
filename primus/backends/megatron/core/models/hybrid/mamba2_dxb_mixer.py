###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Mamba-2 mixer with x and B on the key/value groups (MambaInLlama ``d_xb``).

Adapted from MambaInLlama's ``Mamba2`` (Apache-2.0) as used by Zebra-Llama in
AMD-Hybrid-Models, with ``repeat_kv_before_conv=False``. ``in_proj`` produces
``[z, x, B, C, dt]`` where x and B have ``d_xb`` channels, like GQA values and
keys. After the causal convolution, group ``g`` of x and B feeds heads
``g * r .. g * r + r - 1`` with ``r = d_inner // d_xb``. C has one group per
head, and the head width equals the SSM state width.

Training only, with tensor and context parallel size 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    RMSNormGated = None
    mamba_chunk_scan_combined = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@dataclass
class Mamba2DxbMixerSubmodules:
    """Module specs for the fused-norm input projection and the output projection."""

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


def repeat_groups(t: torch.Tensor, width: int, repeats: int) -> torch.Tensor:
    """``[b, l, g * width]`` to ``[b, l, g * repeats, width]``; head ``h`` reads group ``h // repeats``."""
    batch, length, channels = t.shape
    groups = channels // width
    t = t.view(batch, length, groups, 1, width).expand(batch, length, groups, repeats, width)
    return t.reshape(batch, length, groups * repeats, width)


class Mamba2DxbMixer(MegatronModule):
    """Grouped Mamba-2 token mixer; returns ``(output, bias)`` like ``MambaMixer``.

    Widths come from the config: ``mamba_num_groups`` heads of
    ``mamba_state_dim``, so ``d_inner = mamba_num_groups * mamba_state_dim``.
    ``d_xb`` defaults to the ``mamba_d_xb`` argument.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Mamba2DxbMixerSubmodules,
        d_model: int,
        d_xb: Optional[int] = None,
        d_conv: int = 4,
        A_init_range: tuple[float, float] = (1.0, 16.0),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        chunk_size: int = 256,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: int = 0,
    ):
        if mamba_chunk_scan_combined is None or RMSNormGated is None:
            raise ImportError("Mamba2DxbMixer needs mamba-ssm; install it with `pip install mamba-ssm`")
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for Mamba2DxbMixer"
        if pg_collection.tp.size() != 1 or pg_collection.cp.size() != 1:
            raise ValueError("Mamba2DxbMixer supports tensor and context parallel size 1 only")
        if d_xb is None:
            from megatron.training import get_args

            d_xb = get_args().mamba_d_xb
        if d_xb is None:
            raise ValueError("Mamba2DxbMixer needs mamba_d_xb")

        self.d_model = d_model
        self.d_state = config.mamba_state_dim
        self.ngroups = config.mamba_num_groups
        self.nheads = self.ngroups
        self.headdim = self.d_state
        self.d_inner = self.nheads * self.headdim
        self.d_xb = int(d_xb)
        if self.d_inner % self.d_xb or self.d_xb % self.d_state:
            raise ValueError(
                f"d_xb {self.d_xb} must divide d_inner {self.d_inner} and be a multiple of "
                f"mamba_state_dim {self.d_state}"
            )
        self.repeats = self.d_inner // self.d_xb
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.layer_number = layer_number
        self.pp_layer_offset = pp_layer_offset

        # nn.Linear's default init, as in the reference. The upcycler overwrites
        # the x, B, and C rows and out_proj; z and dt keep this init.
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            2 * self.d_inner + 2 * self.d_xb + self.nheads,
            config=self.config,
            init_method=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=pg_collection.tp,
        )

        conv_dim = self.d_inner + 2 * self.d_xb
        device = torch.cuda.current_device()
        with get_cuda_rng_tracker().fork():
            # nn.Conv1d's default init matches the reference.
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                bias=True,
                device=device,
                dtype=config.params_dtype,
            )
            dt = torch.exp(
                torch.rand(self.nheads, device=device) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus, so softplus(dt_bias) lies in [dt_min, dt_max].
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.norm = RMSNormGated(
            self.d_inner,
            eps=1e-5,
            group_size=self.d_inner // self.ngroups,
            norm_before_gate=False,
            device=device,
            dtype=config.params_dtype,
        )
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=pg_collection.tp,
        )

    def forward(
        self, hidden_states, inference_context=None, *, inference_params=None, packed_seq_params=None
    ):
        """``hidden_states`` is ``[s, b, h]``."""
        if inference_context is not None or inference_params is not None or packed_seq_params is not None:
            raise NotImplementedError("Mamba2DxbMixer supports unpacked training only")
        zxbcdt, _ = self.in_proj(hidden_states)
        zxbcdt = zxbcdt.transpose(0, 1).contiguous()
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_xb, self.nheads], dim=-1)
        x, B, C = torch.split(self._conv(xBC), [self.d_xb, self.d_xb, self.d_inner], dim=-1)
        y = mamba_chunk_scan_combined(
            repeat_groups(x, self.headdim, self.repeats),
            dt,
            -torch.exp(self.A_log.float()),
            repeat_groups(B, self.d_state, self.repeats),
            C.reshape(*C.shape[:2], self.ngroups, self.d_state),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
        )
        y = self.norm(y.flatten(2), z)
        return self.out_proj(y.transpose(0, 1).contiguous())

    def _conv(self, xBC: torch.Tensor) -> torch.Tensor:
        """Depthwise causal convolution plus SiLU over ``[b, l, channels]``."""
        if causal_conv1d_fn is not None:
            x = xBC.transpose(1, 2)
            if x.stride(0) % 8 or x.stride(2) % 8:
                # causal_conv1d's channel-last kernel needs both strides to be multiples of 8.
                x = x.contiguous()
            return causal_conv1d_fn(
                x,
                self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            ).transpose(1, 2)
        length = xBC.shape[1]
        return F.silu(self.conv1d(xBC.transpose(1, 2))[..., :length]).transpose(1, 2)
