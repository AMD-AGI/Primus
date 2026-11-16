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

import os
from typing import Optional

import torch
from generative_recommenders.ops.pytorch.pt_position import (
    pytorch_add_timestamp_positional_embeddings,
)

try:
    from hammer.ops.triton.cc.add_timestamp_position_embeddings.triton_cc_add_timestamp_position_embeddings import (
        triton_cc_add_timestamp_position_embeddings,
    )
except ImportError:
    triton_cc_add_timestamp_position_embeddings = None
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.eval_fwd_hygiene import eval_fwd_hygiene_enabled
from generative_recommenders.ops.triton.triton_position import (
    posenc_dropout_fuse_supported,
    triton_add_timestamp_positional_embeddings,
    triton_concat_add_timestamp_positional_embeddings,
    triton_concat_add_timestamp_positional_embeddings_dropout,
)

try:
    # @manual=//generative_recommenders/ops/triton_aot:triton_position
    from generative_recommenders.ops.triton_aot.triton_position import (  # pyre-ignore[21]
        aot_triton_kernel_wrapper_position,
    )
except ImportError:

    def aot_triton_kernel_wrapper_position(
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        raise ImportError("AOT-T is required for the TRITON_INFERENCE position kernel.")


torch.fx.wrap("triton_add_timestamp_positional_embeddings")
torch.fx.wrap("triton_concat_add_timestamp_positional_embeddings")
torch.fx.wrap("triton_concat_add_timestamp_positional_embeddings_dropout")

# DLRM_HSTU_FP8_POSENC_FAST (gin default on, yambda_5b.gin item (13); absent = off = the incumbent path).
# TRITON kernel with grad enabled only: fold `seq_embeddings *
# alpha` into the position kernel (forward bitwise identical; d_seq written by
# the backward's pos pass, also bitwise identical to MulBackward) and use the
# tiled pos/ts table-gradient reduction (fp32 in-tile accumulation instead of
# the incumbent's bf16 running sum; fp32 atomics as before).
_POSENC_FAST: bool = os.environ.get("DLRM_HSTU_FP8_POSENC_FAST", "0") != "0"

# DLRM_HSTU_FP8_POSENC_CONCAT_FOLD (gin default on, yambda_5b.gin item (20);
# absent or =0 = the incumbent path). On the POSENC_FAST path (TRITON, grad enabled) the ContextualPreprocessor
# no longer materializes concat_2D_jagged(contextual [B*L, D], uih [N_uih, D])
# (its only consumer is the positional encoder): the position kernel reads each
# row from left/right directly and its backward pos pass writes d_left/d_right
# directly, so the concat copy and its split_2D_jagged backward disappear.
# Values are bitwise those of concat + posenc.
_POSENC_CONCAT_FOLD: bool = (
    os.environ.get("DLRM_HSTU_FP8_POSENC_CONCAT_FOLD", "0") != "0"
)


def posenc_concat_fold_enabled(kernel: HammerKernel) -> bool:
    """Whether add_timestamp_positional_embeddings_concat may be used.

    DLRM_HSTU_FP8_EVAL_FWD_HYGIENE=1 (ops/eval_fwd_hygiene.py) lifts the grad
    gate: the fold is bitwise the concat + posenc it replaces, so no-grad
    (eval) forwards take it too.
    """
    return (
        _POSENC_CONCAT_FOLD
        and _POSENC_FAST
        and kernel == HammerKernel.TRITON
        and (torch.is_grad_enabled() or eval_fwd_hygiene_enabled())
    )


@torch.jit.unused
def add_timestamp_positional_embeddings_concat(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
) -> torch.Tensor:
    """== add_timestamp_positional_embeddings(seq_embeddings=concat_2D_jagged(
    values_left (dense, values_left.shape[0] // B rows per sequence),
    values_right (jagged; seq_offsets are the concatenated offsets)), ...)
    on the POSENC_FAST TRITON path. Callers check posenc_concat_fold_enabled."""
    assert time_bucket_fn in ["sqrt", "log"]
    return triton_concat_add_timestamp_positional_embeddings(
        values_left=values_left,
        values_right=values_right,
        seq_offsets=seq_offsets,
        pos_embeddings=position_embeddings_weight,
        ts_embeddings=timestamp_embeddings_weight,
        timestamps=timestamps,
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
        time_bucket_fn=time_bucket_fn,
        alpha=alpha,
    )


@torch.jit.unused
def add_timestamp_positional_embeddings_concat_dropout(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    dropout_p: float,
    attach_dropout_handoff: bool,
    time_bucket_fn: str = "sqrt",
) -> torch.Tensor:
    """DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: == input_dropout(
    add_timestamp_positional_embeddings_concat(...), dropout_p, training=True)
    bitwise (output, mask, CUDA generator state, gradients). Callers check
    posenc_concat_fold_enabled and posenc_dropout_fuse_supported."""
    assert time_bucket_fn in ["sqrt", "log"]
    return triton_concat_add_timestamp_positional_embeddings_dropout(
        values_left=values_left,
        values_right=values_right,
        seq_offsets=seq_offsets,
        pos_embeddings=position_embeddings_weight,
        ts_embeddings=timestamp_embeddings_weight,
        timestamps=timestamps,
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
        time_bucket_fn=time_bucket_fn,
        alpha=alpha,
        dropout_p=dropout_p,
        attach_dropout_handoff=attach_dropout_handoff,
    )


def add_timestamp_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # Script-mode fast path: bypass the HammerKernel ladder.
        seq_embeddings = seq_embeddings * alpha
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    assert time_bucket_fn in ["sqrt", "log"]
    # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE=1: no-grad (eval) forwards fold the alpha
    # too (same bf16 values as the separate `* alpha` pass below).
    if (
        _POSENC_FAST
        and kernel == HammerKernel.TRITON
        and (torch.is_grad_enabled() or eval_fwd_hygiene_enabled())
    ):
        return triton_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
            alpha=alpha,
        )
    seq_embeddings = seq_embeddings * alpha
    if kernel == HammerKernel.TRITON:
        return triton_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    elif kernel == HammerKernel.TRITON_INFERENCE:
        return aot_triton_kernel_wrapper_position(
            alpha=1.0,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            position_embeddings_weight=position_embeddings_weight.to(torch.float32),
            timestamp_embeddings_weight=timestamp_embeddings_weight.to(torch.float32),
            seq_offsets=seq_offsets,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            timestamps=timestamps,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    elif kernel == HammerKernel.TRITON_CC:
        if triton_cc_add_timestamp_position_embeddings is None:
            raise ImportError(
                "hammer is required for the TRITON_CC kernel in add_timestamp_positional_embeddings."
            )
        return triton_cc_add_timestamp_position_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    else:
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
