###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Protocol

import torch


class Router(Protocol):
    """Backend-agnostic router interface."""

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          - probs/scores tensor
          - boolean routing map with shape [tokens, num_experts]
        """


class Dispatcher(Protocol):
    """Backend-agnostic token dispatcher interface."""

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def token_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def dispatch_postprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def combine_preprocess(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def token_combine(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def combine_postprocess(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class ExpertCompute(Protocol):
    """Backend-agnostic expert compute interface (grouped-MLP style)."""

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
