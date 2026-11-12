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


import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Final, List, NamedTuple, Optional, Tuple

import torch
import torchrec
from generative_recommenders.common import (
    fx_infer_max_len,
    fx_mark_length_features,
    HammerKernel,
    HammerModule,
    init_mlp_weights_optional_bias,
    set_static_max_seq_lens,
)
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.hstu_transducer import HSTUTransducer
from generative_recommenders.modules.multitask_module import (
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.eval_fwd_hygiene import eval_fwd_hygiene_enabled
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from torch.autograd.profiler import record_function
from torch.fx._symbolic_trace import is_fx_tracing
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

logger: logging.Logger = logging.getLogger(__name__)

# DLRM_HSTU_FP8_LAZY_FLOPS_STASH: default OFF in code (on only for an explicit
# 1/true/on/yes); the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_LAZY_FLOPS_STASH
# turns it on, =0 keeps the eager per-step stash. Read once at import.
_LAZY_FLOPS_STASH: bool = os.environ.get(
    "DLRM_HSTU_FP8_LAZY_FLOPS_STASH", ""
).strip().lower() in ("1", "true", "on", "yes")
_LAZY_FLOPS_STASH_LOGGED: bool = False


def _log_lazy_flops_stash_once() -> None:
    """The DLRM_HSTU_FP8_LAZY_FLOPS_STASH arming line, once per process, from
    the forward that took the lazy stash."""
    global _LAZY_FLOPS_STASH_LOGGED
    if not _LAZY_FLOPS_STASH_LOGGED:
        _LAZY_FLOPS_STASH_LOGGED = True
        logger.info(
            "DLRMV4 armed OK: lazy flops stash -- main_forward keeps the merged "
            "per-sample lengths; the jagged FLOPs/sample kernels run only when "
            "MetricsLogger reads them (kill switch DLRM_HSTU_FP8_LAZY_FLOPS_STASH=0)"
        )

# (tensor, _version, sum) for the batch's num_candidates, seeded by
# fx_length_stats so fx_total_targets needs no second device-to-host sync. The
# strong reference is deliberate (one int per row): it keeps the `is` test
# sound because the storage cannot be recycled under a live reference.
_total_targets_memo: Tuple[Optional[torch.Tensor], int, int] = (None, -1, 0)


@torch.jit.ignore
def _ec_premerged(embedding: torch.Tensor) -> bool:
    """True iff the EC expand already merged this uih+candidate pair.

    Set by dlrm_v4/train/ec_bf16_output.py (make_model.ec_merge_uih_candidate)
    as a Python attribute on the expand output. Under TrainPipelineSparseDist's
    fx trace the argument is a Proxy; the traced graph is only analysed for
    the EC inputs, so the trace takes the ordinary concat path.
    """
    if is_fx_tracing() or isinstance(embedding, torch.fx.Proxy):
        return False
    return bool(getattr(embedding, "_dlrmv4_uih_cand_premerged", False))


def fx_total_targets(num_candidates: torch.Tensor) -> int:
    """Sum a per-sample candidate-count tensor to a Python int.

    Wrapped with ``torch.fx.wrap`` so ``TrainPipelineSparseDist``'s symbolic
    trace treats it as an opaque leaf instead of recursing into the data-
    dependent ``int(Proxy.sum().item())`` (which raises during tracing).

    Returns the value fx_length_stats already read for this exact tensor (same
    object, same ``_version``) when there is one; otherwise syncs as before.
    """
    cached, version, value = _total_targets_memo
    if cached is num_candidates and version == num_candidates._version:
        return value
    return int(num_candidates.sum().item())


def _memoized_total_targets(num_candidates: torch.Tensor) -> Optional[int]:
    """fx_total_targets' memo hit, or None -- never a device-to-host read."""
    cached, version, value = _total_targets_memo
    if cached is num_candidates and version == num_candidates._version:
        return value
    return None


def fx_length_stats(
    num_candidates: torch.Tensor, uih_seq_lengths: torch.Tensor
) -> Tuple[int, int]:
    """``(max(num_candidates), max(uih_seq_lengths))`` in ONE device-to-host read.

    The same read also carries ``sum(num_candidates)`` and ``max(num_candidates)``
    into the two per-batch memos that would otherwise each pay their own
    ``.item()`` later in the forward: fx_total_targets (main_forward) and the
    FlyDSL attention's ``_max_num_targets`` support check. Every sync drains
    the launch queue, and the later two sit right in front of the first FP8
    GEMM of the step, whose host-side heuristic then runs against an empty
    GPU queue. The integers are identical to the separate reads (same
    reductions over the same tensors), so this changes no numerics -- only how
    many times the host waits for the GPU.
    """
    global _total_targets_memo
    if torch.compiler.is_compiling():
        return fx_infer_max_len(num_candidates), fx_infer_max_len(uih_seq_lengths)
    # Host-sync-free step: the same three integers, computed from the CPU
    # lengths before the H2D copy (dlrm_v4/host_sync_free.py). None -> the
    # device read below (untagged batch or knob off).
    from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

    host = _hsf.host_length_stats(num_candidates, uih_seq_lengths)
    if host is not None:
        max_c, max_u, total = host
    else:
        max_c, max_u, total = torch.stack(
            [
                num_candidates.max().to(torch.int64),
                uih_seq_lengths.max().to(torch.int64),
                num_candidates.sum().to(torch.int64),
            ]
        ).tolist()
    _total_targets_memo = (num_candidates, num_candidates._version, total)
    # Seed the FlyDSL memo only if that module is already loaded (it is
    # whenever FlyDSL attention is dispatched); never import it from here.
    mha = sys.modules.get("generative_recommenders.ops.flydsl.hstu_flydsl_mha")
    if mha is not None:
        seed = getattr(mha, "seed_max_num_targets", None)
        if seed is not None:
            seed(num_candidates, max_c)
    return max_c, max_u


torch.fx.wrap("fx_infer_max_len")
torch.fx.wrap("fx_total_targets")
torch.fx.wrap("fx_length_stats")
torch.fx.wrap("len")


class SequenceEmbedding(NamedTuple):
    lengths: torch.Tensor
    embedding: torch.Tensor


@dataclass
class DlrmHSTUConfig:
    max_seq_len: int = 16384
    max_num_candidates: int = 10
    max_num_candidates_inference: int = 5
    hstu_num_heads: int = 1
    hstu_attn_linear_dim: int = 256
    hstu_attn_qk_dim: int = 128
    hstu_attn_num_layers: int = 12
    hstu_embedding_table_dim: int = 192
    hstu_preprocessor_hidden_dim: int = 256
    hstu_transducer_embedding_dim: int = 0
    hstu_group_norm: bool = False
    hstu_input_dropout_ratio: float = 0.2
    hstu_linear_dropout_rate: float = 0.2
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    candidates_weight_feature_name: str = ""
    candidates_watchtime_feature_name: str = ""
    candidates_querytime_feature_name: str = ""
    causal_multitask_weights: float = 0.2
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    user_embedding_feature_names: List[str] = field(default_factory=list)
    item_embedding_feature_names: List[str] = field(default_factory=list)
    uih_post_id_feature_name: str = ""
    uih_action_time_feature_name: str = ""
    uih_weight_feature_name: str = ""
    hstu_uih_feature_names: List[str] = field(default_factory=list)
    hstu_candidate_feature_names: List[str] = field(default_factory=list)
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(
        default_factory=list
    )
    action_weights: Optional[List[int]] = None
    action_embedding_init_std: float = 0.1
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False


def _get_supervision_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    watchtime_sequence: torch.Tensor,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            supervision_labels[task.task_name] = watchtime_sequence.to(torch.float32)
        elif task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
            supervision_labels[task.task_name] = (
                torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
            ).to(torch.float32)
        else:
            raise RuntimeError("Unsupported MultitaskTaskType")
    return supervision_labels, supervision_weights


class DlrmHSTU(HammerModule):
    def __init__(  # noqa C901
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool,
        is_dense: bool = False,
        bf16_training: bool = True,
        use_gather_select: bool = False,
        transducer_bf16_dtype: bool = False,
        ec_bf16_output: bool = False,
        fuse_linear_norm: bool = False,
        fuse_ln_residual: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")
        # When True, forward() takes the whole `Samples` batch as its single
        # positional arg and reads the pre-merged sparse KJT off it. This keeps
        # the EmbeddingCollection's input a plain getattr on the batch placeholder
        # so TorchRec's TrainPipelineSparseDist can pipeline its input_dist. Set
        # by build_train_pipeline(); leave False for eager / inference / eval.
        self._pipeline_mode: bool = False
        self._hstu_configs = hstu_configs
        self._bf16_training: bool = bf16_training
        # dlrm_v4/train/ec_bf16_output.py hoists the EC-output bf16 cast above
        # TorchRec's dedup expand, so `seq_embeddings[...].embedding` for the
        # large UIH features arrives already in bf16. _user_forward must then
        # restore the transducer output to the EC's real storage dtype rather
        # than to "whatever the embedding happens to carry" -- see there.
        self._ec_bf16_output: bool = ec_bf16_output
        # Last batch's FLOPs/sample (0-d tensors on GPU), populated by
        # main_forward and read + .item()ed by MetricsLogger on each
        # compute_and_log, both against the dense yardstick from
        # get_num_flops_per_sample(). `jagged` charges every layer at full
        # length and yields `fill`; `executed` also discounts a targets-only
        # last layer and yields tflops_real/gpu, hfu and `exec`.
        self._last_jagged_flops_per_sample: Optional[torch.Tensor] = None
        self._last_executed_flops_per_sample: Optional[torch.Tensor] = None
        # DLRM_HSTU_FP8_LAZY_FLOPS_STASH (default OFF in code = the eager stash
        # below; the gin binding turns it on): main_forward keeps only the
        # merged per-sample lengths and
        # the targets-only flag; jagged_flops_per_sample() runs the ~33 0-d
        # kernels when MetricsLogger actually reads them (never, at
        # METRIC_LOG_FREQ=0). Same ops on the same tensor -> same values.
        self._flops_stash_pending: Optional[Tuple[torch.Tensor, bool]] = None
        set_static_max_seq_lens([self._hstu_configs.max_seq_len])

        if not is_dense:
            # use_gather_select picks how the index-dedup path expands unique rows
            # back to one row per token in TorchRec's construct_jagged_tensors:
            # torch.gather (True) instead of torch.index_select (False). The
            # forwards cost the same; the backwards do not. index_select's backward
            # is index_add_ -> ATen indexFuncLargeIndex, which torch 2.12 regressed
            # 2.4x, while gather's backward is scatter_add_, which it did not touch.
            # At this model's shape (69k unique rows, 512 dim, 7.1M tokens) the
            # backward is 45.8 ms vs 12.9 ms on torch 2.12. Both accumulate with
            # atomics in nondeterministic order and score identically against a
            # float64 reference, so this is a pure kernel-selection change. See
            # torch_212_index_scatter_regression.md (git history at 33610d2).
            ec_kwargs = {}
            if use_gather_select:
                if "use_gather_select" not in inspect.signature(
                    EmbeddingCollection.__init__
                ).parameters:
                    # Fail loudly rather than run with the flag silently inert.
                    raise RuntimeError(
                        "use_gather_select=True requires a torchrec that supports it "
                        f"(pinned: 1.7.0a0+bf55480); installed torchrec "
                        f"{getattr(torchrec, '__version__', '?')} does not. Either run "
                        "the pinned stack or set EC_GATHER_SELECT=0."
                    )
                ec_kwargs["use_gather_select"] = True
            self._embedding_collection: EmbeddingCollection = EmbeddingCollection(
                tables=list(embedding_tables.values()),
                need_indices=False,
                device=torch.device("meta"),
                **ec_kwargs,
            )

        # multitask configs must be sorted by task types
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
        self._multitask_module = DefaultMultitaskModule(
            task_configs=self._multitask_configs,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=512),
                SwishLayerNorm(512),
                torch.nn.Linear(in_features=512, out_features=num_tasks),
            ).apply(init_mlp_weights_optional_bias),
            causal_multitask_weights=hstu_configs.causal_multitask_weights,
            is_inference=self._is_inference,
        )
        self._additional_embedding_features: List[str] = [
            uih_feature_name
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self._hstu_configs.merge_uih_candidate_feature_mapping
            if (
                candidate_feature_name
                in self._hstu_configs.item_embedding_feature_names
            )
            and (uih_feature_name in self._hstu_configs.user_embedding_feature_names)
            and (uih_feature_name is not self._hstu_configs.uih_post_id_feature_name)
        ]

        # preprocessor setup
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            hidden_dim=hstu_configs.hstu_preprocessor_hidden_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            action_embedding_dim=8,
            action_feature_name=self._hstu_configs.uih_weight_feature_name,
            action_weights=self._hstu_configs.action_weights,
            action_embedding_init_std=self._hstu_configs.action_embedding_init_std,
            additional_embedding_features=self._additional_embedding_features,
            is_inference=is_inference,
            fuse_linear_norm=fuse_linear_norm,
            fuse_ln_residual=fuse_ln_residual,
        )

        # positional encoder
        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_seq_len=sum(
                dict(hstu_configs.contextual_feature_to_max_length).values()
            ),
            is_inference=self._is_inference,
        )

        if hstu_configs.enable_postprocessor:
            if hstu_configs.use_layer_norm_postprocessor:
                postprocessor = LayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
            else:
                postprocessor = TimestampLayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    time_duration_features=[
                        (60 * 60, 24),  # hour of day
                        (24 * 60 * 60, 7),  # day of week
                        # (24 * 60 * 60, 365), # time of year (approximate)
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

        # construct HSTU
        stu_module: STU = STUStack(
            stu_list=[
                STULayer(
                    config=STULayerConfig(
                        embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                        num_heads=hstu_configs.hstu_num_heads,
                        hidden_dim=hstu_configs.hstu_attn_linear_dim,
                        attention_dim=hstu_configs.hstu_attn_qk_dim,
                        output_dropout_ratio=hstu_configs.hstu_linear_dropout_rate,
                        use_group_norm=hstu_configs.hstu_group_norm,
                        causal=True,
                        target_aware=True,
                        max_attn_len=None,
                        attn_alpha=None,
                        recompute_normed_x=True,
                        recompute_uvqk=True,
                        recompute_y=True,
                        sort_by_length=True,
                        contextual_seq_len=0,
                    ),
                    is_inference=is_inference,
                )
                for _ in range(hstu_configs.hstu_attn_num_layers)
            ],
            is_inference=is_inference,
        )
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            stu_module=stu_module,
            input_preprocessor=preprocessor,
            output_postprocessor=postprocessor,
            input_dropout_ratio=hstu_configs.hstu_input_dropout_ratio,
            positional_encoder=positional_encoder,
            is_inference=self._is_inference,
            return_full_embeddings=False,
            listwise=False,
        )
        # HammerModule._training_dtype defaults to fp32 and nothing ever set it,
        # so the transducer's entry cast upcast the bf16 embedding _user_forward
        # hands it straight back to fp32 -- and the autocast right below then
        # cast it to bf16 again. That round trip is four full-size
        # [tokens, 512] copies per step (two forward, two in the backward),
        # 4.90 ms/step/rank by trace attribution, and it is the identity:
        # bf16 -> fp32 is exact and fp32 -> bf16 of an already-bf16 value is
        # exact, so dropping it is bit-for-bit equal.
        #
        # Scoped to the transducer on purpose. _multitask_module also gates an
        # autocast on _training_dtype, but it runs OUTSIDE the _user_forward
        # autocast, so flipping it there would be a real behaviour change; it
        # is not under this subtree and stays fp32.
        #
        # Gated separately from bf16_training so it can be rolled back on its
        # own: `make_model.transducer_bf16_dtype = False` in the gin restores
        # the fp32 round trip without giving up bf16 training. Default False
        # here and True in the gin, the same way use_gather_select is carried.
        if (not self._is_inference) and bf16_training and transducer_bf16_dtype:
            self._hstu_transducer.set_training_dtype(torch.bfloat16)

        # item embeddings
        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim
                * len(self._hstu_configs.item_embedding_feature_names),
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    # ========================= FLOPs estimation ==============================
    # An HSTU layer is three GEMM groups. With S = sequence positions,
    # D = transducer dim, H = heads, d_qk / d_v = per-head Q/K and V/U dims:
    #
    #   UVQK projection   S · D · H·(2·d_qk + 2·d_v)  MACs   (emits U, V, Q, K)
    #   attention         S²/2 · H·(d_qk + d_v)       MACs   (Q·K^T, then att·V)
    #   output projection S · (3·H·d_v) · D           MACs
    #
    # MULTIPLIERS. 2 FLOPs per MAC. Weight GEMMs ×3 for fwd+bwd (backward
    # computes both dX and dW). Attention ×3.5, not ×3: forward is 2 GEMMs
    # (Q·K^T, att·V) while backward needs ~5 to produce dQ/dK/dV, i.e. ~2.5×
    # forward. Attention additionally ×1/2 for causality — HSTU attention is
    # causal (STULayerConfig(causal=True)) and the kernel's key loop stops at
    # the diagonal (`high = start_m + BLOCK_M` in triton_hstu_attention), so the
    # upper triangle is never evaluated and counting full S² would overstate
    # required work 2× on the term that dominates at S=4096.
    #
    # Both choices matter: at the yambda-5b shape attention is ~57% of the
    # total, so charging ×3 instead of ×3.5 would move it by −8% and dropping
    # the causal ×1/2 by +57%. These are not cosmetic conventions.
    #
    # WHY THE ODD-LOOKING COEFFICIENTS ARE RIGHT (verified against real weight
    # shapes, not inferred):
    #   * 4·D output width on UVQK — `_uvqk_weight` is (D, (2·d_v+2·d_qk)·H)
    #     = (512, 2048) = (D, 4D), since H·d_qk == H·d_v == D here.
    #   * 3·H·d_v input width on the output projection — `_output_weight` is
    #     (3·H·d_v, D) = (1536, 512), because hstu_compute_output is called with
    #     concat_u=True, concat_x=True: the projection consumes a concatenation
    #     of the normed attention output, the gate U, and the residual X.
    #   * D is `hstu_transducer_embedding_dim`, NOT the embedding-table dim.
    #     Both are 512 in yambda-5b but only the former is the width the
    #     layer's GEMMs see, and the two are independently settable.
    #
    # RELATION TO THE PUBLISHED DLRMv3 FORMULA [1],
    #     FLOP_fwd = 2 · layers · (S²·D/2 + 4·S·D² + 3·S·D²)
    # whose three terms line up 1:1 with attn / uvqk / out above. Per-layer
    # forward MACs at S=4096:
    #
    #   term                 here     DLRMv3
    #   UVQK                4.29G      4.29G     agree
    #   output projection   3.22G      3.22G     agree
    #   attention           8.59G      4.29G     differ 2×
    #
    # The GEMM terms match exactly, including the non-obvious 3 on the output
    # projection, which independently corroborates the concat_u/concat_x reading
    # above. The attention terms do not: S²·D/2 is the cost of ONE causal
    # attention GEMM, but there are two (Q·K^T and att·V), each H·d = D wide.
    # It is not a config difference — the reference dlrm_v3 config sets
    # num_heads=4 and attn_qk_dim=attn_linear_dim=128, identical to here — and
    # the `/2` itself is a legitimate causal factor; the second GEMM simply
    # appears to have been dropped. Recomputed with both GEMMs that formula
    # gives ~395 GFLOP/sample forward rather than the published ~260.
    #
    # Note also that the 3× output projection is specific to this HSTU variant.
    # An implementation that skips the U/X concatenation projects H·d_v → D and
    # so carries that term at 1×, which is an architectural difference rather
    # than a disagreement about how to count.
    #
    # EXCLUDED (all of which make this estimate slightly conservative):
    #   * embedding lookups and the sparse all-to-all — memory/network bound,
    #     a large share of step time but not of FLOPs;
    #   * layer norms, SiLU gating, dropout — elementwise;
    #   * the preprocessor item MLP — runs per candidate (~9), not per position;
    #   * the causal ×1/2 is applied uniformly, but the leading contextual
    #     tokens actually attend bidirectionally (the kernel takes low=0,
    #     high=seq_len for start_m < contextual_seq_len). At 8 contextual tokens
    #     against S=4096 that understates attention by ~0.4%.
    #
    # There is deliberately no FFN term: HSTU collapses attention and FFN into
    # the single UVQK projection plus SiLU(U) ⊙ y elementwise gating, so unlike
    # a standard transformer layer there is no separate FFN block to charge.
    #
    # [1] https://mlcommons.org/2026/02/dlrmv3-inference-meta/
    #
    # Final[] so the values stay TorchScript-safe constants if this module is
    # ever scripted (main_forward guards the jagged call, but be defensive).
    _FLOPS_PER_MAC: Final[float] = 2.0
    _GEMM_FWD_BWD: Final[float] = 3.0  # backward computes dX and dW
    _ATTN_FWD_BWD: Final[float] = 3.5  # backward computes dQ/dK/dV, ~2.5× fwd

    def _hstu_layer_flops(
        self, n_tokens_linear: float, n_tokens_attn_sq: float
    ) -> float:
        """Per-layer FLOPs given linear-op token count and attention-token²
        count. Dense estimate uses ``S`` and ``S²``; jagged estimate
        substitutes ``mean(s_i)`` and ``mean(s_i²)``."""
        cfg = self._hstu_configs
        D = cfg.hstu_transducer_embedding_dim
        H = cfg.hstu_num_heads
        hd = cfg.hstu_attn_linear_dim  # V/U head dim
        qd = cfg.hstu_attn_qk_dim  # Q/K head dim
        gemm = self._FLOPS_PER_MAC * self._GEMM_FWD_BWD  # 6
        attn_mult = self._FLOPS_PER_MAC * self._ATTN_FWD_BWD  # 7
        uvqk = gemm * n_tokens_linear * D * (2 * hd + 2 * qd) * H
        attn = attn_mult * (n_tokens_attn_sq / 2) * H * (qd + hd)
        out = gemm * n_tokens_linear * (3 * H * hd) * D
        return uvqk + attn + out

    def _hstu_layer_flops_targets_only(self, n_tokens_linear: float) -> float:
        """Per-layer FLOPs for a layer running the targets-only path.

        With ``TRITON_HSTU_LAST_LAYER_TARGETS_ONLY`` the loss consumes one
        candidate row per sequence, so that layer computes U/Q for that row
        alone while V/K stay full length: the ``uvqk`` gemm keeps the V and K
        halves and drops all but one row of the U and Q halves, attention
        becomes one query row over the whole history rather than a causal
        triangle, and the output projection runs on that single row.

        Charging this instead of ``_hstu_layer_flops`` is what keeps HFU
        honest -- billing the layer at full length credits the model with
        ~28% of the 3-layer total that it never executes.
        """
        cfg = self._hstu_configs
        D = cfg.hstu_transducer_embedding_dim
        H = cfg.hstu_num_heads
        hd = cfg.hstu_attn_linear_dim
        qd = cfg.hstu_attn_qk_dim
        gemm = self._FLOPS_PER_MAC * self._GEMM_FWD_BWD
        attn_mult = self._FLOPS_PER_MAC * self._ATTN_FWD_BWD
        uvqk = gemm * (n_tokens_linear + 1.0) * D * (hd + qd) * H
        attn = attn_mult * n_tokens_linear * H * (qd + hd)
        out = gemm * (3 * H * hd) * D
        return uvqk + attn + out

    def get_num_flops_per_sample(self) -> float:
        """Dense-equivalent fwd+bwd FLOPs per sample at ``max_seq_len``.

        Used as the MFU yardstick (peak utilization the workload could
        theoretically reach if every sample's sequence were the full padded
        length). The actual ``tflops_real``/``hfu`` reported per step uses
        the jagged estimate stashed by ``main_forward``.

        Deliberately blind to ``TRITON_HSTU_LAST_LAYER_TARGETS_ONLY``: this is
        the workload the model *defines*, so a knob that removes work should
        not move it. That does mean MFU rises when the knob is on -- the
        jagged estimate, and therefore HFU, is the one that tracks what ran.
        """
        cfg = self._hstu_configs
        S = float(cfg.max_seq_len)
        n_layers = cfg.hstu_attn_num_layers
        flops = n_layers * self._hstu_layer_flops(
            n_tokens_linear=S, n_tokens_attn_sq=S * S
        )
        # Multitask head (Linear(D, n_tasks)) — negligible but cheap to add.
        n_tasks = len(self._multitask_configs)
        if n_tasks > 0:
            flops += (
                self._FLOPS_PER_MAC
                * self._GEMM_FWD_BWD
                * n_tasks
                * cfg.hstu_transducer_embedding_dim
            )
        return float(flops)

    def _compute_jagged_flops_per_sample(
        self,
        uih_seq_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        targets_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jagged fwd+bwd FLOPs per sample for THIS batch's actual lengths.

        Per-sample merged sequence length s_i = uih_seq_lengths[i] +
        num_candidates[i]. Returns ``(ragged, executed)`` as 0-d tensors on the
        batch's device; the caller should ``.item()`` them (one D→H sync per
        logging interval).

        Two numbers because two independent things shrink the work below the
        dense yardstick, and collapsing them would make either one unreadable:

        * ``ragged`` charges every layer at full length, so against the dense
          yardstick it isolates ragged sequences alone. This is ``fill``.
        * ``executed`` additionally discounts the last layer when it took the
          targets-only path, so it is the work the GPU really did. This drives
          ``tflops_real``/``hfu``, and against the yardstick it is ``exec``.

        The two are equal whenever the knob is off or the gate declined, so
        ``exec``/``fill`` isolates the work removal on its own.

        ``targets_only`` reports whether the last layer actually took the
        targets-only path on this batch, and must come from the transducer
        rather than the env flag: the gate declines per batch at eval, or when
        a batch is not exactly one target per sequence.
        """
        return self._jagged_flops_from_merged_lengths(
            uih_seq_lengths + num_candidates, targets_only
        )

    def _jagged_flops_from_merged_lengths(
        self, merged_lengths: torch.Tensor, targets_only: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = merged_lengths.float()
        cfg = self._hstu_configs
        # Same per-layer expression as the dense estimate, with mean(s_i) and
        # mean(s_i²) substituted for S and S²; sharing it keeps the two in step.
        # mean(s_i²) rather than mean(s_i)² is what the attention term needs:
        # cost is quadratic per sample, so it must be averaged after squaring
        # (Jensen — using the squared mean would understate a skewed batch).
        per_layer = self._hstu_layer_flops(
            n_tokens_linear=s.mean(), n_tokens_attn_sq=(s * s).mean()
        )
        ragged = cfg.hstu_attn_num_layers * per_layer
        if targets_only:
            executed = (
                cfg.hstu_attn_num_layers - 1
            ) * per_layer + self._hstu_layer_flops_targets_only(
                n_tokens_linear=s.mean()
            )
        else:
            executed = ragged
        n_tasks = len(self._multitask_configs)
        if n_tasks > 0:
            head = (
                self._FLOPS_PER_MAC
                * self._GEMM_FWD_BWD
                * n_tasks
                * cfg.hstu_transducer_embedding_dim
            )
            ragged = ragged + head
            executed = executed + head
        return ragged, executed

    @torch.jit.unused
    def jagged_flops_per_sample(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(jagged, executed) FLOPs/sample of the last forward batch."""
        pending = self._flops_stash_pending
        if pending is not None:
            self._flops_stash_pending = None
            (
                self._last_jagged_flops_per_sample,
                self._last_executed_flops_per_sample,
            ) = self._jagged_flops_from_merged_lengths(pending[0], pending[1])
        return (
            self._last_jagged_flops_per_sample,
            self._last_executed_flops_per_sample,
        )

    def _construct_payload(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> Dict[str, torch.Tensor]:
        if len(self._hstu_configs.contextual_feature_to_max_length) > 0:
            contextual_offsets: List[torch.Tensor] = []
            _ctx_lengths: List[torch.Tensor] = [
                seq_embeddings[x].lengths
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            ]
            _batched = False
            if not torch.jit.is_scripting() and not is_fx_tracing():
                # DLRM_HSTU_FP8_CONTEXTUAL_FUSE: one stack + one row-wise scan
                # instead of a (fill + scan) pair per feature; same integers.
                from generative_recommenders.ops import glue_fusion as _gf

                if _gf.complete_cumsum_many_fusable(_ctx_lengths):
                    try:
                        contextual_offsets = _gf.complete_cumsum_many(_ctx_lengths)
                        _batched = True
                    except Exception as exc:  # fail closed to fbgemm below
                        _gf.disable("contextual", exc)
                        contextual_offsets = []
            if not _batched:
                for _l in _ctx_lengths:
                    contextual_offsets.append(
                        torch.ops.fbgemm.asynchronous_complete_cumsum(_l)
                    )
        else:
            # Dummy, offsets are unused
            contextual_offsets = torch.empty((0, 0))
        if torch.jit.is_scripting():
            # Explicit loops are TS-clean (avoid the dict-merge / dict-comp
            # idioms below, which TorchScript cannot script).
            out: Dict[str, torch.Tensor] = {}
            for k, v in payload_features.items():
                out[k] = v
            for x in self._hstu_configs.contextual_feature_to_max_length.keys():
                out[x] = seq_embeddings[x].embedding
            i = 0
            for x in self._hstu_configs.contextual_feature_to_max_length.keys():
                # pyre-ignore[6]
                out[x + "_offsets"] = contextual_offsets[i]
                i += 1
            for x in self._additional_embedding_features:
                out[x] = seq_embeddings[x].embedding
            return out
        return {
            **payload_features,
            **{
                x: seq_embeddings[x].embedding
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            },
            **{
                x + "_offsets": contextual_offsets[i]
                for i, x in enumerate(
                    list(self._hstu_configs.contextual_feature_to_max_length.keys())
                )
            },
            **{
                x: seq_embeddings[x].embedding
                for x in self._additional_embedding_features
            },
        }

    @torch.jit.unused
    def _uih_ts_concat_fold(
        self,
        max_seq_len: int,
        payload_features: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """DLRM_HSTU_FP8_CONCAT_FOLD[_UIH_TS]: _user_forward's timestamp
        concat_2D_jagged (uih action times, then the candidate query time, per
        sample) in one 1-D kernel, or None for the shipped op (switch off, not
        the TRITON kernel, unexpected dtype/layout, or a launch failure)."""
        if self.hammer_kernel() != HammerKernel.TRITON:
            return None
        from generative_recommenders.ops import glue_fusion as _gf

        left = payload_features[self._hstu_configs.uih_action_time_feature_name]
        right = payload_features[
            self._hstu_configs.candidates_querytime_feature_name
        ]
        offsets_left = payload_features["uih_offsets"]
        offsets_right = payload_features["candidate_offsets"]
        if not _gf.uih_ts_concat_fusable(left, right, offsets_left, offsets_right):
            return None
        try:
            out = _gf.jagged_concat_1d(
                left, right, offsets_left, offsets_right, max_seq_len
            )
        except Exception as exc:  # fail closed to the shipped concat
            _gf.disable("concat_fold_uih_ts", exc)
            return None
        _gf.note_armed_rank0(
            "concat_fold uih_ts",
            "user_forward timestamp concat_2D_jagged as one 1-D jagged concat "
            "kernel, _jagged_concat_1d_kernel (kill switch "
            "DLRM_HSTU_FP8_CONCAT_FOLD=0 or DLRM_HSTU_FP8_CONCAT_FOLD_UIH_TS=0)",
        )
        return out

    def _user_forward(
        self,
        max_uih_len: int,
        max_candidates: int,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        num_candidates: torch.Tensor,
        total_uih_len: Optional[int] = None,
        total_targets: Optional[int] = None,
    ) -> torch.Tensor:
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths
        source_timestamps_fold: Optional[torch.Tensor] = None
        if not torch.jit.is_scripting() and not is_fx_tracing():
            # DLRM_HSTU_FP8_CONCAT_FOLD (ops/glue_fusion.py section 4): the
            # same 1-D int64 jagged concat in one kernel (bitwise).
            source_timestamps_fold = self._uih_ts_concat_fold(
                max_uih_len + max_candidates, payload_features
            )
        if source_timestamps_fold is not None:
            source_timestamps = source_timestamps_fold
        else:
            source_timestamps = concat_2D_jagged(
                max_seq_len=max_uih_len + max_candidates,
                max_len_left=max_uih_len,
                offsets_left=payload_features["uih_offsets"],
                values_left=payload_features[
                    self._hstu_configs.uih_action_time_feature_name
                ].unsqueeze(-1),
                max_len_right=max_candidates,
                offsets_right=payload_features["candidate_offsets"],
                values_right=payload_features[
                    self._hstu_configs.candidates_querytime_feature_name
                ].unsqueeze(-1),
                kernel=self.hammer_kernel(),
            ).squeeze(-1)
        if total_targets is None:
            total_targets = fx_total_targets(num_candidates)
        if total_uih_len is None:
            total_uih_len = source_timestamps.numel() - total_targets
        embedding = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].embedding
        # The dtype the transducer output is restored to at the end of this
        # method. With the EC-output hoist on, `embedding` is already bf16 and
        # reading its dtype here would silently turn that restore into a no-op
        # and hand the multitask head bf16 -- a real change, not the bitwise
        # one this arm claims. The hoist only ever engages on an fp32 EC
        # output (ec_bf16_output.py declines otherwise), so fp32 is the dtype
        # that was there before it.
        dtype = torch.float32 if self._ec_bf16_output else embedding.dtype
        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        if torch.jit.is_scripting():
            # TorchScript does not support ``with torch.autocast(...)``.
            # In script-mode inference the dense path is already in bf16
            # (move_sparse_output_to_device upcasts on the C++ side), so
            # autocast is a no-op for the path the predictor exercises.
            candidates_user_embeddings, _ = self._hstu_transducer(
                max_uih_len=max_uih_len,
                max_targets=max_candidates,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_embeddings=embedding,
                seq_lengths=source_lengths,
                seq_timestamps=source_timestamps,
                seq_payloads=self._construct_payload(
                    payload_features=payload_features,
                    seq_embeddings=seq_embeddings,
                ),
                num_targets=num_candidates,
            )
        else:
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=(not self.is_inference) and self._bf16_training,
            ):
                candidates_user_embeddings, _ = self._hstu_transducer(
                    max_uih_len=max_uih_len,
                    max_targets=max_candidates,
                    total_uih_len=total_uih_len,
                    total_targets=total_targets,
                    seq_embeddings=embedding,
                    seq_lengths=source_lengths,
                    seq_timestamps=source_timestamps,
                    seq_payloads=self._construct_payload(
                        payload_features=payload_features,
                        seq_embeddings=seq_embeddings,
                    ),
                    num_targets=num_candidates,
                )
        candidates_user_embeddings = candidates_user_embeddings.to(dtype)

        return candidates_user_embeddings

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:  # [L, D]
        all_embeddings = torch.cat(
            [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.item_embedding_feature_names
            ],
            dim=-1,
        )
        # The item MLP's row count is NOT the HSTU token count, so the install
        # in the HSTU forward (ops/hstu_compute.py) cannot cover this MLP's
        # GEMMs. Measured on 8 ranks x 5 steps of a profiled 200-step run:
        # num_tokens - all_embeddings.shape[0] == 8192 at every single point,
        # which is exactly global_batch_size -- preprocess() appends one token
        # per sample between here and the projection. A pin keyed on the HSTU
        # count therefore scores 0/40 lookups against the key this MLP actually
        # presents, and TunableOp silently runs the heuristic.
        #
        # That was not a hypothesis: the mlp_b_dgrad pin shipped "active on 8/8
        # ranks" and the traced arm still ran the DEFAULT kernel
        # (Cijk_..._MT256x256x32_...), byte-identical to the control, 1.5395 vs
        # 1.5875 ms. Installed is not the same as used.
        #
        # Registering at this count is what makes the MLP pins reachable at all.
        # It also writes cache lines for the six HSTU pins at this M; those keys
        # are never looked up (the projection never runs at this row count), so
        # they are inert padding, not a behaviour change. Under the pipeline's
        # fx trace the shape is a Proxy rather than an int, hence the guard --
        # same guard as the HSTU-side call site.
        num_item_tokens = all_embeddings.shape[0]
        if isinstance(num_item_tokens, int):
            install_for_tokens(num_item_tokens)
        item_embeddings = self._item_embedding_mlp(all_embeddings)
        return item_embeddings

    @torch.jit.unused
    def _issue_action_prefill(
        self,
        max_uih_len: int,
        max_num_candidates: int,
        uih_seq_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        payload_features: Dict[str, torch.Tensor],
    ) -> None:
        """DLRM_HSTU_FP8_ACTION_PREFILL (modules/action_prefill.py).

        Issues the preprocessor's action branch with exactly the inputs
        main_forward -> _user_forward will hand it (same payload tensors,
        num_targets=num_candidates, seq_lengths=uih_seq_lengths+num_candidates,
        total_targets from fx_length_stats' memo) under _user_forward's
        autocast. Training with grad only; the preprocessor re-checks the
        inputs and falls back to the original order on any mismatch.
        """
        action_prefill.clear()
        if not action_prefill.enabled():
            return
        if (
            self.is_inference
            or not self.training
            or not torch.is_grad_enabled()
            or is_fx_tracing()
            or torch.compiler.is_compiling()
        ):
            return
        pre = self._hstu_transducer._input_preprocessor
        if not isinstance(pre, ContextualPreprocessor) or pre._action_weights is None:
            return action_prefill.decline("no ContextualPreprocessor action branch")
        if self._hstu_configs.uih_post_id_feature_name not in [
            u for u, _ in self._hstu_configs.merge_uih_candidate_feature_mapping
        ]:
            return action_prefill.decline("uih_post_id feature is not merged")
        total_targets = _memoized_total_targets(num_candidates)
        if total_targets is None:
            return action_prefill.decline(
                "sum(num_candidates) is not on the host without a sync"
            )
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference) and self._bf16_training,
        ):
            pre.prefill_action(
                max_uih_len=max_uih_len,
                max_targets=max_num_candidates,
                total_targets=total_targets,
                uih_seq_lengths=uih_seq_lengths,
                num_targets=num_candidates,
                seq_payloads=payload_features,
            )

    def preprocess(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
        merged_sparse_features: Optional[KeyedJaggedTensor] = None,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        # Embedding lookup for uih + candidates. When the caller (the pipeline
        # path) supplies the pre-merged KJT from the batch, feed it straight to
        # the EmbeddingCollection: that keeps the lookup's input a plain getattr
        # off the batch so TorchRec's TrainPipelineSparseDist can hoist its
        # input_dist into the prefetch stage. Building it here (cat +
        # from_lengths_sync's .sync()) is an "input modification" that makes
        # TorchRec skip pipelining the embedding collection.
        if merged_sparse_features is None:
            merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=uih_features.keys() + candidates_features.keys(),
                values=torch.cat(
                    [uih_features.values(), candidates_features.values()],
                    dim=0,
                ),
                lengths=torch.cat(
                    [uih_features.lengths(), candidates_features.lengths()],
                    dim=0,
                ),
            )
        seq_embeddings_dict = self._embedding_collection(merged_sparse_features)
        num_candidates = fx_mark_length_features(
            candidates_features.lengths().view(len(candidates_features.keys()), -1)
        )[0]
        uih_seq_lengths = uih_features[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths()
        if torch.jit.is_scripting():
            max_num_candidates = fx_infer_max_len(num_candidates)
            max_uih_len = fx_infer_max_len(uih_seq_lengths)
        else:
            # One D2H read instead of four per step (two maxes here, the
            # target sum in main_forward, the target max in FlyDSL attention).
            max_num_candidates, max_uih_len = fx_length_stats(
                num_candidates, uih_seq_lengths
            )

        # prepare payload features
        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                candidate_feature_name
                not in self._hstu_configs.item_embedding_feature_names
                and uih_feature_name
                not in self._hstu_configs.user_embedding_feature_names
            ):
                values_left = uih_features[uih_feature_name].values()
                if self._is_inference and (
                    candidate_feature_name
                    == self._hstu_configs.candidates_weight_feature_name
                    or candidate_feature_name
                    == self._hstu_configs.candidates_watchtime_feature_name
                ):
                    total_candidates = torch.sum(num_candidates).item()
                    values_right = torch.zeros(
                        total_candidates,  # pyre-ignore
                        dtype=torch.int64,
                        device=values_left.device,
                    )
                else:
                    values_right = candidates_features[candidate_feature_name].values()
                payload_features[uih_feature_name] = values_left
                payload_features[candidate_feature_name] = values_right
        # DLRM_HSTU_FP8_GLUE_FOLD4_OFFS (ops/glue_fusion.py section 5): both
        # complete cumsums in one integer kernel. None = the shipped ops.
        _g4_offs = None
        if not torch.jit.is_scripting() and not is_fx_tracing():
            from generative_recommenders.ops import glue_fusion as _gf4

            _g4_offs = _gf4.g4_offsets(uih_seq_lengths, num_candidates, preproc=False)
        if _g4_offs is not None:
            payload_features["uih_offsets"] = _g4_offs[0]
            payload_features["candidate_offsets"] = _g4_offs[1]
        else:
            payload_features["uih_offsets"] = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(uih_seq_lengths)
            )
            payload_features["candidate_offsets"] = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(num_candidates)
            )

        # The comprehension below is the first touch of the EC output. With
        # the pipelined (sharded) EC that output is a LazyAwaitable, and this
        # touch is where the main stream starts waiting on the forward
        # embedding all-to-all (All2All_Seq_Req_Wait). DLRM_HSTU_FP8_ACTION_
        # PREFILL issues the embedding-independent action branch of the input
        # preprocessor before it, so it runs in that window.
        if not torch.jit.is_scripting():
            self._issue_action_prefill(
                max_uih_len=max_uih_len,
                max_num_candidates=max_num_candidates,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                payload_features=payload_features,
            )

        seq_embeddings = {
            k: SequenceEmbedding(
                lengths=seq_embeddings_dict[k].lengths(),
                embedding=seq_embeddings_dict[k].values(),
            )
            for k in self._hstu_configs.user_embedding_feature_names
            + self._hstu_configs.item_embedding_feature_names
        }

        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
        total_uih_len: Optional[int] = None,
        total_targets: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # M4 speed estimator (dlrm_v4/train/m4_speed.py): P_mf span start and
        # its end hook. No-op unless DLRM_HSTU_FP8_M4=1.
        if not torch.jit.is_scripting() and not is_fx_tracing():
            from generative_recommenders.dlrm_v4.train import m4_speed as _m4s

            _m4s.main_forward_begin(self, seq_embeddings)
        # merge uih and candidates embeddings
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if uih_feature_name in seq_embeddings:
                if not torch.jit.is_scripting() and _ec_premerged(
                    seq_embeddings[uih_feature_name].embedding
                ):
                    # The EC expand already wrote this pair in merged
                    # [uih_b..., cand_b...] order (ec_merge_uih_candidate);
                    # concatenating again would be wrong, not just slow.
                    seq_embeddings[uih_feature_name] = SequenceEmbedding(
                        lengths=uih_seq_lengths + num_candidates,
                        embedding=seq_embeddings[uih_feature_name].embedding,
                    )
                    continue
                seq_embeddings[uih_feature_name] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_seq_len=max_uih_len + max_num_candidates,
                        max_len_left=max_uih_len,
                        offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            uih_seq_lengths
                        ),
                        values_left=seq_embeddings[uih_feature_name].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            num_candidates
                        ),
                        values_right=seq_embeddings[candidate_feature_name].embedding,
                        kernel=self.hammer_kernel(),
                    ),
                )

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(
                seq_embeddings,
            )
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                num_candidates=num_candidates,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
            )

        # Stash this batch's jagged FLOPs/sample for MetricsLogger to read.
        # No D->H sync: the .item() happens once per metric_log_frequency in
        # the trainer, not on every step. Eval-mode batches also produce a
        # stash but the trainer only consumes it on train batches
        # (DLRM_HSTU_FP8_EVAL_FWD_HYGIENE=1: so eval batches skip it). Stashed
        # after the transducer ran so it can be priced against the path that
        # was actually taken, not the one the env flag asked for.
        if (
            not torch.jit.is_scripting()
            and not self.training
            and eval_fwd_hygiene_enabled()
        ):
            # Eval batch, hygiene on: no stash, lazy or eager (the trainer's
            # perf line reads only train batches). Training never reads the
            # switch (self.training short-circuits).
            pass
        elif not torch.jit.is_scripting() and _LAZY_FLOPS_STASH:
            # The add is kept eager so the stash owns a fresh tensor and never
            # aliases batch buffers a pipeline might recycle.
            self._flops_stash_pending = (
                uih_seq_lengths + num_candidates,
                bool(self._hstu_transducer._last_targets_only),
            )
            self._last_jagged_flops_per_sample = None
            self._last_executed_flops_per_sample = None
            _log_lazy_flops_stash_once()
        elif not torch.jit.is_scripting():
            (
                self._last_jagged_flops_per_sample,
                self._last_executed_flops_per_sample,
            ) = self._compute_jagged_flops_per_sample(
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                targets_only=self._hstu_transducer._last_targets_only,
            )

        with record_function("## multitask_module ##"):
            supervision_labels, supervision_weights = (
                _get_supervision_labels_and_weights(
                    supervision_bitmasks=payload_features[
                        self._hstu_configs.candidates_weight_feature_name
                    ],
                    watchtime_sequence=payload_features[
                        self._hstu_configs.candidates_watchtime_feature_name
                    ],
                    task_configs=self._multitask_configs,
                )
            )
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    supervision_labels=supervision_labels,
                    supervision_weights=supervision_weights,
                )
            )

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and self.training:
            for i, task in enumerate(self._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: Optional[KeyedJaggedTensor] = None,
        merged_sparse_features: Optional[KeyedJaggedTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # Pipeline mode: TorchRec fx-traces this forward (via DMP.module) and the
        # pipeline calls it with the single `Samples` batch. Unpacking the KJTs
        # here — rather than in the wrapper — makes the EmbeddingCollection's
        # input `batch.merged_sparse_features` a getattr off the batch placeholder,
        # which is what lets TrainPipelineSparseDist hoist the embedding input_dist
        # into the prefetch stage. Guarded from TorchScript (inference path).
        if not torch.jit.is_scripting() and self._pipeline_mode:
            batch = uih_features
            uih_features = batch.uih_features_kjt
            candidates_features = batch.candidates_features_kjt
            merged_sparse_features = batch.merged_sparse_features

        with record_function("## preprocess ##"):
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(
                uih_features=uih_features,
                candidates_features=candidates_features,
                merged_sparse_features=merged_sparse_features,
            )

        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
