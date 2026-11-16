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

from typing import Dict, List, Optional, Tuple

import torch
from generative_recommenders.common import HammerKernel, HammerModule
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from torch.fx._symbolic_trace import is_fx_tracing


class ActionEncoder(HammerModule):
    def __init__(
        self,
        action_embedding_dim: int,
        action_feature_name: str,
        action_weights: List[int],
        watchtime_feature_name: str = "",
        watchtime_to_action_thresholds_and_weights: Optional[
            List[Tuple[int, int]]
        ] = None,
        embedding_init_std: float = 0.1,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._watchtime_feature_name: str = watchtime_feature_name
        self._action_feature_name: str = action_feature_name
        self._watchtime_to_action_thresholds_and_weights: List[Tuple[int, int]] = (
            watchtime_to_action_thresholds_and_weights
            if watchtime_to_action_thresholds_and_weights is not None
            else []
        )
        self.register_buffer(
            "_combined_action_weights",
            torch.tensor(
                action_weights
                + [x[1] for x in self._watchtime_to_action_thresholds_and_weights]
            ),
        )
        self._num_action_types: int = len(action_weights) + len(
            self._watchtime_to_action_thresholds_and_weights
        )
        self._action_embedding_dim = action_embedding_dim
        self._action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=embedding_init_std
            ),
        )
        self._target_action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((1, self._num_action_types * action_embedding_dim)).normal_(
                mean=0, std=embedding_init_std
            ),
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._action_embedding_dim * self._num_action_types

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        bf16_consumer: bool = False,
    ) -> torch.Tensor:
        """bf16_consumer: the caller's only use of the result is a consumer
        that casts it to bf16 under autocast (ContextualPreprocessor's action
        MLP). The fused path (DLRM_HSTU_FP8_ACTION_ENC_FUSE, ops/glue_fusion.py)
        then emits the bf16 rows directly -- the same RNE rounding of the same
        fp32 products the consumer's cast would apply -- and receives the bf16
        gradient the consumer then returns. The fused path arms ONLY in that
        case (bf16 consumer under bf16 autocast, the production path); an fp32
        result is always produced by the shipped chain below."""
        seq_actions = seq_payloads[self._action_feature_name]
        total_targets_f: int = seq_embeddings.size(0) - seq_actions.size(0)
        if (
            not torch.jit.is_scripting()
            and bf16_consumer
            and not is_fx_tracing()
            and len(self._watchtime_to_action_thresholds_and_weights) == 0
            and self.hammer_kernel() == HammerKernel.TRITON
        ):
            from generative_recommenders.ops import glue_fusion as _gf

            dev = seq_actions.device.type
            out_bf16 = torch.is_autocast_enabled(dev) and (
                torch.get_autocast_dtype(dev) == torch.bfloat16
            )
            if _gf.action_encode_fusable(
                seq_actions,
                self._combined_action_weights,
                self._action_embedding_table,
                self._target_action_embedding_table,
                uih_offsets,
                target_offsets,
                out_bf16,
            ):
                # table.unsqueeze(0) before the tile, as in the shipped graph:
                # keeps the AccumulateGrad (DDP grad-ready / bucket) order.
                table3 = self._action_embedding_table.unsqueeze(0)
                # Still an autograd input: RepeatBackward0 runs as shipped.
                right = self._target_action_embedding_table.tile(total_targets_f, 1)
                try:
                    out = _gf.action_encode_concat(
                        seq_actions,
                        self._combined_action_weights,
                        table3,
                        right,
                        uih_offsets,
                        target_offsets,
                        max_uih_len + max_targets,
                        True,
                    )
                except Exception as exc:  # fail closed to the shipped chain
                    _gf.disable("action_enc", exc)
                else:
                    _gf.note_armed(
                        "action_enc",
                        "one-pass action encoder fwd+bwd, bf16 rows "
                        f"[{tuple(out.shape)}] (kill switch "
                        "DLRM_HSTU_FP8_ACTION_ENC_FUSE=0)",
                    )
                    return out
        if len(self._watchtime_to_action_thresholds_and_weights) > 0:
            watchtimes = seq_payloads[self._watchtime_feature_name]
            for threshold, weight in self._watchtime_to_action_thresholds_and_weights:
                seq_actions = torch.bitwise_or(
                    seq_actions, (watchtimes >= threshold).to(torch.int64) * weight
                )
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)
            )
            > 0
        )
        action_embeddings = (
            exploded_actions.unsqueeze(-1) * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self._num_action_types * self._action_embedding_dim)
        total_targets: int = seq_embeddings.size(0) - action_embeddings.size(0)
        action_embeddings = concat_2D_jagged(
            max_seq_len=max_uih_len + max_targets,
            values_left=action_embeddings,
            values_right=self._target_action_embedding_table.tile(
                total_targets,
                1,
            ),
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.hammer_kernel(),
        )
        return action_embeddings
