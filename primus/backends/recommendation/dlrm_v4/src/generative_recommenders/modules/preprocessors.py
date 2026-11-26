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

import abc
import os
import sys
from math import sqrt
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    HammerKernel,
    HammerModule,
    init_mlp_weights_optional_bias,
    jagged_to_padded_dense,
)
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.action_encoder import ActionEncoder
from torch.fx._symbolic_trace import is_fx_tracing
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.fused_linear_norm import FusedLinearNormMLP
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from generative_recommenders.ops.triton import preproc_fp8_snr

if TYPE_CHECKING:
    from generative_recommenders.ops.preproc_grouped_tail import GroupedTail


# E-M1 (ops/preproc_grouped_tail.py): run the three MLPs' final LayerNorm(512)s
# as ONE grouped launch instead of three chained ones. Bitwise identical to the
# shipped F1+F2 chain given the item (10) autotune pins (otherwise ulp-level;
# the armed line says which). Read once at import; absent or "0" (code default)
# = shipped; gin item (30) turns it on.
_PREPROC_GROUPED_TAIL: bool = (
    os.environ.get("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", "0") == "1"
)

# Payload key carrying the un-concatenated contextual rows when
# ContextualPreprocessor.forward(fold_contextual_concat=True)
# (DLRM_HSTU_FP8_POSENC_CONCAT_FOLD); consumed and removed by HSTUTransducer.
CONTEXTUAL_FOLD_KEY: str = "__dlrm_hstu_fp8_contextual_fold_left__"


class InputPreprocessor(HammerModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Args:
            max_uih_len: int
            max_targets: int
            total_uih_len: int
            total_targets: int
            seq_lengths: (B,)
            seq_embeddings: (L, D)
            seq_timestamps: (B, N)
            num_targets: (B,) Optional.
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            (max_seq_len, total_uih_len, total_targets, lengths, offsets, timestamps, embeddings, num_targets, payloads) updated based on input preprocessor.
        """
        pass

    def interleave_targets(self) -> bool:
        return False


def get_contextual_input_embeddings(
    seq_lengths: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
    contextual_feature_to_max_length: Dict[str, int],
    contextual_feature_to_min_uih_length: Dict[str, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        not torch.jit.is_scripting()
        and not is_fx_tracing()
        and all(
            contextual_feature_to_min_uih_length.get(key, 0) <= 0
            for key in contextual_feature_to_max_length
        )
    ):
        # DLRM_HSTU_FP8_CONTEXTUAL_FUSE (ops/glue_fusion.py): the casts, the
        # per-feature jagged_to_padded_dense and the cat in one Function --
        # same RNE casts, same zero padding, same column blocks; its backward
        # replaces the zeros + j2pd backward + ToCopyBackward0 per feature.
        from generative_recommenders.ops import glue_fusion as _gf

        _keys = list(contextual_feature_to_max_length.keys())
        _vals = [seq_payloads[k] for k in _keys]
        _offs = [seq_payloads[k + "_offsets"] for k in _keys]
        _lens = [int(contextual_feature_to_max_length[k]) for k in _keys]
        if _gf.contextual_pad_cat_fusable(_vals, _offs, _lens, dtype):
            try:
                _out = _gf.contextual_pad_cat(_vals, _offs, _lens)
            except Exception as exc:  # fail closed to the shipped ops below
                _gf.disable("contextual", exc)
            else:
                _gf.note_armed(
                    "contextual",
                    f"{len(_keys)} contextual features cast+pad+cat in one "
                    "Function, offsets in one row-wise scan (kill switch "
                    "DLRM_HSTU_FP8_CONTEXTUAL_FUSE=0)",
                )
                return _out
    padded_values: List[torch.Tensor] = []
    for key, max_len in contextual_feature_to_max_length.items():
        v = torch.flatten(
            jagged_to_padded_dense(
                values=seq_payloads[key].to(dtype),
                offsets=[seq_payloads[key + "_offsets"]],
                max_lengths=[max_len],
                padding_value=0.0,
            ),
            1,
            2,
        )
        min_uih_length = contextual_feature_to_min_uih_length.get(key, 0)
        if min_uih_length > 0:
            v = v * (seq_lengths.view(-1, 1) >= min_uih_length)
        padded_values.append(v)
    return torch.cat(padded_values, dim=1)


@torch.jit.ignore
def _catfused_base(parts: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """The [N, k*W] tensor whose column blocks are exactly `parts`, or None.

    Set by dlrm_v4/train/ec_bf16_output.py (DLRM_HSTU_FP8_EC_CAT_FUSE) on the
    column views of one interleaved expand; torch.cat(parts, dim=1) would
    rebuild that tensor byte for byte.
    """
    if is_fx_tracing() or len(parts) < 2:
        return None
    big = None
    for q, t in enumerate(parts):
        if isinstance(t, torch.fx.Proxy):
            return None
        m = getattr(t, "_dlrmv4_catfused", None)
        if m is None or m[1] != q or m[2] != len(parts):
            return None
        if big is None:
            big = m[0]
        elif m[0] is not big:
            return None
    return big


class ContextualPreprocessor(InputPreprocessor):
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        action_embedding_dim: int = 8,
        action_feature_name: str = "",
        action_weights: Optional[List[int]] = None,
        additional_embedding_features: List[str] = [],
        action_embedding_init_std: float = 0.1,
        is_inference: bool = True,
        fuse_linear_norm: bool = False,
        fuse_ln_residual: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        # F1: fold each Linear's bias gradient into the following norm's
        # backward, deleting the standalone aten::sum over dx. Off by default
        # until an e2e ladder has measured it. See ops/fused_linear_norm.py.
        self._fuse_linear_norm: bool = fuse_linear_norm
        # F2 rides INSIDE F1's container -- it is the same norm kernel gaining
        # an epilogue -- so it cannot be armed on its own. Asking for it
        # without F1 is a request this code cannot honour, and honouring it
        # silently at False would produce an arm that measures nothing while
        # the log says it is on. Raise instead.
        if fuse_ln_residual and not fuse_linear_norm:
            raise ValueError(
                "fuse_ln_residual=True requires fuse_linear_norm=True: the "
                "residual is folded into the fused container's norm epilogue, "
                "which only exists on the F1 path"
            )
        self._fuse_ln_residual: bool = fuse_ln_residual
        self._output_embedding_dim: int = output_embedding_dim
        self._input_embedding_dim: int = input_embedding_dim
        self._hidden_dim: int = hidden_dim
        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        if self._max_contextual_seq_len > 0:
            std = 1.0 * sqrt(
                2.0 / float(input_embedding_dim + self._output_embedding_dim)
            )
            self._batched_contextual_linear_weights: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (
                            self._max_contextual_seq_len,
                            input_embedding_dim,
                            self._output_embedding_dim,
                        )
                    ).normal_(0.0, std)
                )
            )
            self._batched_contextual_linear_bias: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (self._max_contextual_seq_len, self._output_embedding_dim)
                    ).fill_(0.0)
                )
            )
        self._content_embedding_mlp: torch.nn.Module = self._build_mlp(
            self._input_embedding_dim
        )
        self._additional_embedding_features: List[str] = additional_embedding_features
        self._additional_embedding_mlp: torch.nn.Module = self._build_mlp(
            self._input_embedding_dim * len(additional_embedding_features)
        )
        self._action_feature_name: str = action_feature_name
        self._action_weights: Optional[List[int]] = action_weights
        if self._action_weights is not None:
            self._action_encoder: ActionEncoder = ActionEncoder(
                action_feature_name=action_feature_name,
                action_weights=self._action_weights,
                action_embedding_dim=action_embedding_dim,
                embedding_init_std=action_embedding_init_std,
                is_inference=is_inference,
            )
            self._action_embedding_mlp: torch.nn.Module = self._build_mlp(
                self._action_encoder.output_embedding_dim
            )

    def _build_mlp(self, in_features: int) -> torch.nn.Module:
        """Linear -> SwishLayerNorm -> Linear -> LayerNorm.

        The two containers are interchangeable by construction:
        FusedLinearNormMLP is an nn.Sequential SUBCLASS holding the same
        children in the same order, so state_dict keys ("0.weight", "1.weight",
        ...) are identical either way and a checkpoint crosses the F1 flag in
        both directions. Only the backward differs -- see
        ops/fused_linear_norm.py.
        """
        cls = FusedLinearNormMLP if self._fuse_linear_norm else torch.nn.Sequential
        return cls(
            torch.nn.Linear(
                in_features=in_features,
                out_features=self._hidden_dim,
            ),
            SwishLayerNorm(self._hidden_dim),
            torch.nn.Linear(
                in_features=self._hidden_dim,
                out_features=self._output_embedding_dim,
            ),
            LayerNorm(self._output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def _mlp_add(
        self,
        mlp: torch.nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """`residual + mlp(x)`, folded into the MLP's last norm when F2 is on.

        The shipped expression is a full [M, 512] read-modify-write against
        HBM issued immediately after the MLP's final LayerNorm has just
        finished writing that same tensor -- 2.369 ms/rank/step across the two
        call sites, measured post-hoist so it is not double-counting item #3.
        Handing the residual to the norm makes it an epilogue on a tile the
        kernel already holds.

        The fused epilogue rounds to the output dtype before adding, which is
        what the shipped `+` does too (it adds a bf16 residual to an
        already-stored bf16 norm output), so the FUSION's arithmetic is exact.
        The container is not quite bit-identical: the residual makes a distinct
        Triton autotune key, so the kernel is benchmarked again as a second
        variant and picks a different num_warps, which moves rstd by an fp32
        ulp. Pinned to one config the two agree exactly; as autotuned they
        differ by ~2 bf16 ulp on ~2.7e-7 of elements, far below the
        nondeterminism this model's 7.1M-atomic scatter_add_ backward already
        has. No convergence gate of its own -- see _f2_module.py.
        """
        if self._fuse_ln_residual:
            return mlp(x, residual=residual)
        return residual + mlp(x)

    @torch.jit.unused
    def _grouped_tail_begin(
        self, seq_embeddings: torch.Tensor
    ) -> Optional["GroupedTail"]:
        """DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL: the chain object that defers each
        MLP's last (Linear, LayerNorm) pair to ONE grouped launch, or None when
        the preconditions fail (shipped path; one WARNING). See
        ops/preproc_grouped_tail.py."""
        from generative_recommenders.ops.preproc_grouped_tail import (
            begin_grouped_tail,
        )

        return begin_grouped_tail(
            mlp_add=self._mlp_add,
            content_mlp=self._content_embedding_mlp,
            content_x=seq_embeddings,
            fuse_linear_norm=self._fuse_linear_norm,
            fuse_ln_residual=self._fuse_ln_residual,
            has_additional=len(self._additional_embedding_features) > 0,
            output_dim=self._output_embedding_dim,
        )

    @torch.jit.unused
    def _preproc_dedup(
        self, seq_embeddings: torch.Tensor, seq_payloads: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Item (118) DLRM_HSTU_FP8_PREPROC_DEDUP (ops/preproc_dedup.py, default
        off): the content + additional MLPs (their F2 sum) on the unique (item,
        artist, album) EC row triples, gathered back to the tokens; None ->
        today's statements. Only a tensor the EC tagged engages (the EC reads
        the knob); untagged, nothing is imported, and a module that is already
        loaded (knob on) counts the per-token call (KS-K1: untagged=N)."""
        if is_fx_tracing() or isinstance(seq_embeddings, torch.fx.Proxy):
            return None
        # ops/preproc_dedup.PPDD_ATTR (literal)
        if getattr(seq_embeddings, "_dlrmv4_preproc_dedup", None) is None:
            loaded = sys.modules.get("generative_recommenders.ops.preproc_dedup")
            if loaded is not None:  # knob on: counted when armed (KS-K1)
                loaded.note_untagged(seq_embeddings)
            return None
        from generative_recommenders.ops import preproc_dedup

        parts = [seq_payloads.get(f) for f in self._additional_embedding_features]
        return preproc_dedup.run(self, seq_embeddings, _catfused_base(parts))

    @torch.jit.unused
    def prefill_action(
        self,
        max_uih_len: int,
        max_targets: int,
        total_targets: int,
        uih_seq_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> None:
        """Issue the action branch now (DLRM_HSTU_FP8_ACTION_PREFILL); see
        modules/action_prefill.py. forward() finishes it or falls back."""
        action_prefill.issue(
            self,
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_targets=total_targets,
            uih_seq_lengths=uih_seq_lengths,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

    @torch.jit.unused
    def _take_action_prefill(  # pyre-ignore[3]: ActionPrefill or None
        self,
        max_uih_len: int,
        max_targets: int,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ):
        """The prefill issued for this forward if its inputs are the ones this
        forward hands the action branch, else None (nothing issued, or a
        mismatch: FALLBACK, the original order runs)."""
        pre = action_prefill.take(self)
        if pre is None:
            return None
        reason = pre.mismatch(
            max_uih_len, max_targets, seq_embeddings, num_targets, seq_payloads
        )
        if reason is not None:
            action_prefill.fallback(reason)
            return None
        return pre

    @torch.jit.unused
    def _action_tail_from_prefill(
        self,
        pre,  # pyre-ignore[2]: modules/action_prefill.py ActionPrefill
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """_mlp_add(action MLP, ActionEncoder(...), residual), cut after the
        head the prefill ran (x = the attached head output)."""
        mlp = self._action_embedding_mlp
        # The three spellings of _mlp_add, cut after the head the prefill ran.
        if self._fuse_ln_residual:
            return mlp.forward_tail(x, pre.fused, residual=residual, prefilled=pre.l2)
        if isinstance(mlp, FusedLinearNormMLP):
            return residual + mlp.forward_tail(x, pre.fused, prefilled=pre.l2)
        for m in list(mlp)[-2:]:
            x = m(x)
        return residual + x

    def forward(  # noqa C901
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        fold_contextual_concat: bool = False,
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """fold_contextual_concat (DLRM_HSTU_FP8_POSENC_CONCAT_FOLD, set only by
        HSTUTransducer when the positional encoder can consume the two pieces):
        the contextual/UIH concat is NOT materialized; the returned
        seq_embeddings are the UIH rows only and the contextual rows
        [B * max_contextual_seq_len, D] are returned in a copy of seq_payloads
        under CONTEXTUAL_FOLD_KEY. Everything else is unchanged."""
        # The preprocessor MLP GEMM pins (ops/blas_solution_pins.py, preproc_*)
        # are keyed on this row count. _item_forward normally registers the same
        # count first; registering here too keeps the pins reachable without
        # depending on that. No-op after the first call per count. Under the
        # pipeline's fx trace the shape is a Proxy, hence the guard.
        num_rows = seq_embeddings.shape[0]
        if isinstance(num_rows, int):
            install_for_tokens(num_rows)
        if preproc_fp8_snr.ENABLED:
            # Default-off FP8 numerics probe (Arbor E-M3): count training
            # forwards only (grad mode reads False inside autograd.Function).
            preproc_fp8_snr.note_grad_mode(torch.is_grad_enabled())
        # E-M1 grouped tail (default off). When armed, each MLP call below runs
        # only its first (Linear, SwishLayerNorm) pair, in the same place, and
        # tail.finish() runs all the last pairs at the end of the MLP section.
        # The non-MLP work in between keeps its order.
        tail = None
        # Item (118) PREPROC_DEDUP (default off): c + a on the unique (item,
        # artist, album) row triples, gathered back to the tokens; None ->
        # today's statements. When it engages no grouped tail is begun, so the
        # action branch below runs its `tail is None` statements with this sum
        # as the residual.
        deduped: Optional[torch.Tensor] = None
        if not torch.jit.is_scripting():
            deduped = self._preproc_dedup(seq_embeddings, seq_payloads)
        if deduped is None and _PREPROC_GROUPED_TAIL and not torch.jit.is_scripting():
            tail = self._grouped_tail_begin(seq_embeddings)
        if deduped is not None:
            output_seq_embeddings = deduped
        elif tail is not None:
            tail.head(self._content_embedding_mlp, seq_embeddings)
        else:
            output_seq_embeddings = self._content_embedding_mlp(seq_embeddings)
        if deduped is None and len(self._additional_embedding_features) > 0:
            _parts = [
                seq_payloads[feature]
                for feature in self._additional_embedding_features
            ]
            _fused = None
            if not torch.jit.is_scripting():
                _fused = _catfused_base(_parts)
            if _fused is not None:
                additional_embeddings = _fused
            else:
                additional_embeddings = torch.cat(_parts, dim=1)
            if tail is not None:
                tail.head(self._additional_embedding_mlp, additional_embeddings)
            else:
                output_seq_embeddings = self._mlp_add(
                    self._additional_embedding_mlp,
                    additional_embeddings,
                    output_seq_embeddings,
                )
        max_seq_len = max_uih_len + max_targets
        # DLRM_HSTU_FP8_ACTION_PREFILL: DlrmHSTU.preprocess may already have
        # issued the offsets, ActionEncoder and action-MLP pair 1 (+ the last
        # pair's Linear) inside the forward all-to-all window. Finish it here,
        # at the branch's original graph position: into the grouped tail when
        # it is armed (its node reuses the prefilled Linear), else pair 2's
        # norm (+ F2 residual epilogue). None -> the original order below.
        pre = None
        if not torch.jit.is_scripting() and self._action_weights is not None:
            pre = self._take_action_prefill(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                seq_payloads=seq_payloads,
            )
        # (119) DLRM_HSTU_FP8_ACTION_DEDUP (ops/action_dedup.py, code default
        # off): the action MLP on its 2**T + 1 code rows (x2, hi/lo), issued by
        # the prefill above or planned here, and residual + table[code] added
        # after the MLP section; the per-token action branch below is skipped.
        # Not under torch.fx symbolic tracing (TorchRec's _rewrite_model traces
        # this forward once per rank): the trace records the shipped branch, as
        # it did before (119), and prints no DECLINED line (WBND r119_v0 M1).
        dd = None
        if (
            not torch.jit.is_scripting()
            and not is_fx_tracing()
            and self._action_weights is not None
            and os.environ.get("DLRM_HSTU_FP8_ACTION_DEDUP", "0").strip().lower()
            in ("1", "true", "on", "yes")
        ):
            from generative_recommenders.ops import action_dedup as _ad

            dd = _ad.take_or_plan(
                self, pre, seq_embeddings, seq_lengths, num_targets, seq_payloads
            )
        if dd is None and pre is not None and tail is not None:
            # A group that would decline AT the action MLP completes its queued
            # pairs before the action head; only the original order keeps the
            # backward order then (never the case for the production model).
            why = tail.prefilled_reason(self._action_embedding_mlp, pre.y, pre.fused)
            if why is not None:
                action_prefill.fallback("grouped tail declines the action MLP: " + why)
                pre = None
        if dd is None and pre is not None:
            x_prefilled = pre.attach()
            if tail is not None:
                tail.head_prefilled(
                    self._action_embedding_mlp, x_prefilled, pre.fused, pre.l2
                )
            else:
                output_seq_embeddings = self._action_tail_from_prefill(
                    pre, x_prefilled, output_seq_embeddings
                )
        elif dd is None:
            target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
            uih_offsets = seq_offsets - target_offsets
            if self._action_weights is not None:
                action_embeddings = self._action_encoder(
                    max_uih_len=max_uih_len,
                    max_targets=max_targets,
                    uih_offsets=uih_offsets,
                    target_offsets=target_offsets,
                    seq_embeddings=seq_embeddings,
                    seq_payloads=seq_payloads,
                    # The result's only consumer is the action MLP below, whose
                    # first Linear casts it to the autocast dtype (both the
                    # grouped-tail head and the _mlp_add path run
                    # _LinearNormFunction, which does x.to(amp_dtype)).
                    bf16_consumer=True,
                )
                if tail is not None:
                    tail.head(self._action_embedding_mlp, action_embeddings)
                else:
                    output_seq_embeddings = self._mlp_add(
                        self._action_embedding_mlp,
                        action_embeddings,
                        output_seq_embeddings,
                    )
        if tail is not None:
            # (119) x (118) guard (main's ruling (b)): finish only a chain that
            # an MLP went through. GroupedTail.finish() asserts on a chain that
            # holds no MLP (preproc_grouped_tail.py, `assert acc is not None`).
            # Unreachable as shipped: (118) never begins a tail when it engages
            # (its sum is then the residual) and the content head queues right
            # after begin; a variant that bypassed both members while keeping
            # the tail would keep its own residual here instead of crashing.
            if tail.declined or len(tail._mlps) > 0:  # pyre-ignore[16]
                output_seq_embeddings = tail.finish()
        if dd is not None:
            from generative_recommenders.ops import action_dedup as _ad

            output_seq_embeddings = _ad.finish(output_seq_embeddings, dd, tail)
        if pre is not None:
            action_prefill.armed(
                pre, grouped=None if tail is None else not tail.declined
            )

        output_max_seq_len = max_seq_len
        output_total_uih_len = total_uih_len
        output_total_targets = total_targets
        output_seq_lengths = seq_lengths
        output_num_targets = num_targets
        output_seq_timestamps = seq_timestamps
        output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            output_seq_lengths
        )
        # concat contextual embeddings
        if self._max_contextual_seq_len > 0:
            contextual_input_embeddings = get_contextual_input_embeddings(
                seq_lengths=seq_lengths,
                seq_payloads=seq_payloads,
                contextual_feature_to_max_length=self._contextual_feature_to_max_length,
                contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
                dtype=seq_embeddings.dtype,
            )
            contextual_embeddings = torch.baddbmm(
                self._batched_contextual_linear_bias.view(
                    -1, 1, self._output_embedding_dim
                ).to(contextual_input_embeddings.dtype),
                contextual_input_embeddings.view(
                    -1, self._max_contextual_seq_len, self._input_embedding_dim
                ).transpose(0, 1),
                self._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            ).transpose(0, 1)
            if fold_contextual_concat:
                seq_payloads = seq_payloads.copy()
                seq_payloads[CONTEXTUAL_FOLD_KEY] = fx_unwrap_optional_tensor(
                    contextual_embeddings
                ).reshape(-1, self._output_embedding_dim)
            else:
                output_seq_embeddings = concat_2D_jagged(
                    max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                    values_left=fx_unwrap_optional_tensor(
                        contextual_embeddings
                    ).reshape(-1, self._output_embedding_dim),
                    values_right=output_seq_embeddings,
                    max_len_left=self._max_contextual_seq_len,
                    max_len_right=output_max_seq_len,
                    offsets_left=None,
                    offsets_right=output_seq_offsets,
                    kernel=self.hammer_kernel(),
                )
            ctx_ts_fold: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
            if not torch.jit.is_scripting() and not is_fx_tracing():
                # DLRM_HSTU_FP8_CONCAT_FOLD (ops/glue_fusion.py section 4): the
                # zero-padded timestamp concat, lengths + L and their complete
                # cumsum in one integer kernel (bitwise). output_seq_offsets is
                # complete_cumsum(output_seq_lengths) here (computed above).
                ctx_ts_fold = self._ctx_ts_pad_fold(
                    output_seq_timestamps,
                    output_seq_lengths,
                    output_seq_offsets,
                    output_max_seq_len,
                )
            if ctx_ts_fold is not None:
                output_seq_timestamps, output_seq_lengths, output_seq_offsets = ctx_ts_fold
                output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            else:
                output_seq_timestamps = concat_2D_jagged(
                    max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                    values_left=torch.zeros(
                        (output_seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                        dtype=output_seq_timestamps.dtype,
                        device=output_seq_timestamps.device,
                    ),
                    values_right=output_seq_timestamps.unsqueeze(-1),
                    max_len_left=self._max_contextual_seq_len,
                    max_len_right=output_max_seq_len,
                    offsets_left=None,
                    offsets_right=output_seq_offsets,
                    kernel=self.hammer_kernel(),
                ).squeeze(-1)
                output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
                output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
                output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                    output_seq_lengths
                )
            output_total_uih_len = (
                output_total_uih_len
                + self._max_contextual_seq_len * output_seq_lengths.size(0)
            )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            seq_payloads,
        )

    @torch.jit.unused
    def _ctx_ts_pad_fold(
        self,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """DLRM_HSTU_FP8_CONCAT_FOLD[_CTX_TS]: (zero-padded timestamps,
        seq_lengths + L, complete_cumsum(seq_lengths + L)) from one integer
        kernel, or None for the shipped ops (switch off, not the TRITON kernel,
        unexpected dtype/layout, or a launch failure). seq_offsets must be
        complete_cumsum(seq_lengths)."""
        if self.hammer_kernel() != HammerKernel.TRITON:
            return None
        from generative_recommenders.ops import glue_fusion as _gf

        L = self._max_contextual_seq_len
        if not _gf.ctx_ts_pad_fusable(seq_timestamps, seq_lengths, seq_offsets, L):
            return None
        try:
            out = _gf.ctx_ts_pad(
                seq_timestamps, seq_lengths, seq_offsets, L, max_seq_len
            )
        except Exception as exc:  # fail closed to the shipped ops
            _gf.disable("concat_fold_ctx_ts", exc)
            return None
        _gf.note_armed_rank0(
            "concat_fold ctx_ts",
            "contextual timestamp zeros + concat_2D_jagged + lengths add + "
            "complete_cumsum as one integer kernel, _ctx_ts_pad_kernel (kill "
            "switch DLRM_HSTU_FP8_CONCAT_FOLD=0 or "
            "DLRM_HSTU_FP8_CONCAT_FOLD_CTX_TS=0)",
        )
        return out
