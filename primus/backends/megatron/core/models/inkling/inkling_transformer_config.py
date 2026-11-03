###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer config for thinkingmachines/Inkling (text backbone, ``inkling_mm_model``).

Every ``inkling_*`` field is read off ``args`` by ``core_transformer_config_from_args``
(the experiment YAML / converter sets them). Megatron's own fields carry the shared
shapes: ``num_attention_heads`` / ``num_query_groups`` / ``kv_channels`` describe the
global (full-attention) layers, ``num_moe_experts`` / ``moe_router_topk`` /
``moe_ffn_hidden_size`` the routed experts. ``mtp_num_layers`` sets the number of
multi-token-prediction depths (HF ``mtp_config.num_nextn_predict_layers``).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.core.utils.module_utils import warning_rank_0

INKLING_DEFAULT_INIT_SEED = 1234
"""Default base seed of the layout-independent Inkling init (config default, converter default)."""


def _is_number(value):
    """A real int / float (bool excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class InklingTransformerConfig(TransformerConfig):
    # ---- attention ----
    inkling_d_rel: int = 16
    """Width of the per-head relative-position projection (``d_rel``)."""
    inkling_rel_extent: int = 1024
    """Relative-bias extent of the global layers (sliding layers use the window)."""
    inkling_sliding_window: int = 512
    """Window of the sliding (local) layers (``sliding_window_size``)."""
    inkling_local_layer_ids: Optional[List[int]] = field(default_factory=lambda: [0, 2, 4, 6])
    """0-based model-wide (absolute) layer indices that use sliding-window attention; all other
    layers are global (full attention)."""
    inkling_swa_num_attention_heads: int = 8
    """Attention heads of the sliding-window (local) layers; global layers use ``num_attention_heads``."""
    inkling_swa_num_query_groups: int = 4
    """K / V heads (query groups) of the sliding-window layers; global layers use ``num_query_groups``."""
    inkling_swa_kv_channels: int = 128
    """Head dim of the sliding-window layers; global layers use ``kv_channels``."""
    inkling_log_scaling_n_floor: Optional[float] = 128000.0
    """Global-layer query / bias log scaling: tau = 1 + alpha * log(max((pos+1)/n_floor, 1))."""
    inkling_log_scaling_alpha: float = 0.1
    """``alpha`` of the global-layer log scaling above (0 = no scaling); read only when
    ``inkling_log_scaling_n_floor`` is set."""
    inkling_sconv_kernel_size: int = 4
    """Kernel of the causal depthwise short convolutions (k, v, attn-out, mlp-out)."""
    inkling_sconv_impl: str = "auto"
    """Short-convolution kernel: ``causal_conv1d`` (Dao's fused ``causal_conv1d_fn``, must import),
    ``eager`` (fp32 ``F.conv1d``, the reference) or ``auto`` (fused when it imports, else eager)."""
    inkling_attention_impl: str = "eager"
    """Core attention: ``eager`` (query-blocked matmuls with the relative bias, the reference) or
    ``fused`` (a Triton flash-attention kernel that adds the banded relative bias in-kernel, with
    its backward; no ``[s, s]`` scores are stored). ``fused`` has no attention dropout."""

    # ---- MoE ----
    inkling_num_shared_experts: int = 2
    """Shared experts, weighted by their own router gammas (``shared_expert_sink``)."""
    inkling_route_scale: float = 8.0
    """Multiplies the normalized gate weights of the routed and the shared experts (HF ``route_scale``)."""
    inkling_dense_mlp_idx: int = 0
    """Layers below this index use a dense MLP (Inkling ships 0: every layer is MoE; others are refused)."""
    inkling_dense_intermediate_size: int = 3072
    """FFN width of the dense SwiGLU MLP (HF ``dense_intermediate_size``); only the MTP blocks use it."""
    inkling_dense_use_global_scale: bool = True
    """The dense MLP output is multiplied by a learned scalar ``global_scale`` (HF ``use_global_scale``)."""

    # ---- multi-token prediction (``mtp_num_layers`` depths, HF ``mtp_config``) ----
    inkling_mtp_local_layer_ids: Optional[List[int]] = field(default_factory=lambda: [1])
    """0-based MTP depths that use sliding-window attention (HF ``mtp_config.local_layer_ids``);
    the other depths are global (full attention, ``inkling_rel_extent``)."""
    inkling_mtp_local_extent: Optional[int] = None
    """Window (and relative-bias extent) of the sliding MTP depths (HF ``mtp_local_extent``);
    None = ``inkling_sliding_window``."""
    inkling_mtp_swa_num_attention_heads: Optional[int] = None
    """Attention heads of the sliding MTP depths; None = ``inkling_swa_num_attention_heads``."""
    inkling_mtp_chain_hidden_post_norm: bool = False
    """HF ``mtp_config.chain_hidden_post_norm`` (a shared norm on each depth output). Only False
    (the released checkpoints) is ported."""

    # ---- embedding / head ----
    inkling_use_embed_norm: bool = True
    """Apply an RMSNorm (``embed_norm``) to the token embeddings before the first layer."""
    inkling_logits_mup_width_multiplier: float = 3.0
    """The LM head input is divided by this (muP width multiplier)."""
    inkling_unpadded_vocab_size: Optional[int] = None
    """If set, logits at and beyond this id are masked (HF truncates them)."""
    inkling_ignored_label_id: Optional[int] = None
    """One label id ``>= inkling_unpadded_vocab_size`` that scores 0 even where ``loss_mask`` does
    not mask it (it stays in the caller's token count). Every other unmasked label past the HF
    vocab raises (a tokenizer / vocab mismatch). The builder sets it to the NullTokenizer EOD id
    (= ``vocab_size``) for mock data only: Megatron's ``eod_mask_loss`` masks the positions whose
    *input* is EOD, not those whose *label* is EOD."""
    inkling_init_seed: int = INKLING_DEFAULT_INIT_SEED
    """Base seed of the layout-independent init of every Inkling-owned parameter (attention
    relative table, router, routed and shared experts); per-tensor seeds are hashed from
    (base, layer, slot, expert) with SHA-256 (60 bits kept), so collisions are negligible."""

    def __post_init__(self):
        # Megatron's own checks run first; they default expert_tensor_parallel_size (ETP = TP
        # when unset), which the ETP check below relies on.
        super().__post_init__()
        errors = []
        tp = self.tensor_model_parallel_size
        local = list(self.inkling_local_layer_ids or [])
        # Ids >= num_layers are allowed: they name layers a truncated proxy (num_layers override)
        # does not build. The converter checks them against the full HF depth.
        bad_ids = [i for i in local if type(i) is not int or i < 0]
        if bad_ids:
            errors.append(f"inkling_local_layer_ids {bad_ids} must be non-negative ints")
        ids = [i for i in local if type(i) is int]
        beyond = sorted(i for i in ids if i >= self.num_layers)
        if beyond:
            warning_rank_0(
                f"Inkling: inkling_local_layer_ids {beyond} are >= num_layers={self.num_layers}; "
                "those layers are not built (truncated proxy). If the model is not a truncated "
                "proxy, set inkling_local_layer_ids to match num_layers"
            )
        if len(set(ids)) != len(ids):
            errors.append(f"inkling_local_layer_ids {local} has duplicates")
        for name in (
            "inkling_d_rel",
            "inkling_rel_extent",
            "inkling_sliding_window",
            "inkling_swa_num_attention_heads",
            "inkling_swa_num_query_groups",
            "inkling_swa_kv_channels",
            "inkling_sconv_kernel_size",
            "inkling_num_shared_experts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                errors.append(f"{name}={value!r} must be a positive int")
        swa_heads, swa_groups = self.inkling_swa_num_attention_heads, self.inkling_swa_num_query_groups
        swa_ok = all(type(v) is int and v > 0 for v in (swa_heads, swa_groups))
        # Checked whenever the model has sliding layers at all, also when a truncated proxy builds
        # none of them: the swa shapes must stay valid for the full-depth model.
        if swa_ok and ids:
            for name, value in (
                ("inkling_swa_num_attention_heads", swa_heads),
                ("inkling_swa_num_query_groups", swa_groups),
            ):
                if value % tp:
                    errors.append(f"{name}={value} is not divisible by TP={tp}")
            if swa_heads % swa_groups:
                errors.append(
                    f"inkling_swa_num_attention_heads={swa_heads} must be a multiple of "
                    f"inkling_swa_num_query_groups={swa_groups}"
                )
        for name in ("inkling_logits_mup_width_multiplier", "inkling_route_scale"):
            value = getattr(self, name)
            # a negative route scale would flip the sign of every routed and shared expert weight
            if not _is_number(value) or value <= 0:
                errors.append(f"{name}={value!r} must be a number > 0")
        alpha = self.inkling_log_scaling_alpha
        # alpha is read only with log scaling on (inkling_log_scaling_n_floor set)
        if self.inkling_log_scaling_n_floor is not None and (not _is_number(alpha) or alpha < 0):
            errors.append(f"inkling_log_scaling_alpha={alpha!r} must be a number >= 0 (0 = no log scaling)")
        if self.inkling_sconv_impl not in ("auto", "eager", "causal_conv1d"):
            errors.append(
                f"inkling_sconv_impl={self.inkling_sconv_impl!r} must be 'auto', 'eager' or 'causal_conv1d'"
            )
        if self.inkling_attention_impl not in ("eager", "fused"):
            errors.append(
                f"inkling_attention_impl={self.inkling_attention_impl!r} must be 'eager' or 'fused'"
            )
        elif self.inkling_attention_impl == "fused" and self.attention_dropout:
            errors.append(
                f"inkling_attention_impl=fused has no attention dropout (attention_dropout="
                f"{self.attention_dropout}); set attention_dropout: 0.0 or inkling_attention_impl: eager"
            )
        if self.inkling_dense_mlp_idx != 0:
            errors.append(
                f"inkling_dense_mlp_idx={self.inkling_dense_mlp_idx!r} must be 0 (dense MLP layers are "
                "not ported; Inkling ships 0)"
            )
        errors.extend(self._mtp_errors())
        if self.num_moe_experts is None or self.moe_router_topk is None:
            errors.append("Inkling is an MoE model: num_moe_experts and moe_router_topk are required")
        ignored, n_vocab = self.inkling_ignored_label_id, self.inkling_unpadded_vocab_size
        if ignored is not None and (type(ignored) is not int or n_vocab is None or ignored < n_vocab):
            errors.append(
                f"inkling_ignored_label_id={ignored!r} must be an int >= inkling_unpadded_vocab_size "
                f"({n_vocab}): only an id past the HF vocab can be ignored"
            )
        n_floor = self.inkling_log_scaling_n_floor
        if n_floor is not None and (not _is_number(n_floor) or n_floor <= 0):
            errors.append(
                f"inkling_log_scaling_n_floor={n_floor!r} must be a number > 0 (or None to disable log scaling)"
            )
        etp = self.expert_tensor_parallel_size
        # The MoE is replicated over TP: every TP rank holds the same experts and computes the
        # same expert gradients. Expert grads are reduced over the expert-data-parallel group
        # (world / (PP * ETP * EP) ranks); only ETP = TP keeps the TP replicas out of it. With
        # ETP < TP the replicated grads would be summed TP / ETP times too often.
        if etp != tp:
            errors.append(
                f"expert_tensor_parallel_size={etp} must equal tensor_model_parallel_size={tp}: "
                "Inkling's experts are replicated over TP (leave it unset to get ETP = TP)"
            )
        if self.fp8 or getattr(self, "fp4", None):
            errors.append("fp8 / fp4 training is not supported by the Inkling layers")
        if (
            getattr(self, "cuda_graph_impl", "none") not in (None, "none")
            or getattr(self, "enable_cuda_graph", False)
            or getattr(self, "external_cuda_graph", False)
        ):
            errors.append("CUDA graphs are not supported by the Inkling layers")
        # Both need Megatron's sub-module schedule (build_schedule_plan / split wgrad callables),
        # which InklingModel and its layers do not implement.
        for name in ("overlap_moe_expert_parallel_comm", "delay_wgrad_compute"):
            if getattr(self, name, False):
                errors.append(f"{name} is not supported by the Inkling layers")
        if getattr(self, "cpu_offloading", False):
            errors.append("activation CPU offloading is not supported by the Inkling layers")
        if errors:
            raise ValueError("InklingTransformerConfig: " + "; ".join(errors))

    # ---- MTP helpers ----
    @property
    def inkling_mtp_window(self):
        """Window / relative-bias extent of the sliding MTP depths."""
        if self.inkling_mtp_local_extent is None:
            return self.inkling_sliding_window
        return self.inkling_mtp_local_extent

    @property
    def inkling_mtp_swa_heads(self):
        """Attention heads of the sliding MTP depths."""
        if self.inkling_mtp_swa_num_attention_heads is None:
            return self.inkling_swa_num_attention_heads
        return self.inkling_mtp_swa_num_attention_heads

    def _mtp_errors(self):
        """Checks of the dense-MLP fields (always: the converter and FLOPs report read them) and of
        the MTP fields (only when ``mtp_num_layers`` is set)."""
        errors = []
        if type(self.inkling_dense_intermediate_size) is not int or self.inkling_dense_intermediate_size <= 0:
            errors.append(
                f"inkling_dense_intermediate_size={self.inkling_dense_intermediate_size!r} must be a positive int"
            )
        depths = self.mtp_num_layers
        if not depths:
            return errors
        if type(depths) is not int or depths < 1:
            return errors + [f"mtp_num_layers={depths!r} must be a positive int"]
        ids = list(self.inkling_mtp_local_layer_ids or [])
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            errors.append(f"inkling_mtp_local_layer_ids={ids} must be unique non-negative ints")
        # As inkling_local_layer_ids: ids >= mtp_num_layers name depths a run with fewer depths
        # (e.g. mtp_num_layers=1 with the release default [1]) does not build. The converter
        # checks the ids against the checkpoint's own depth count.
        beyond = sorted(i for i in ids if type(i) is int and i >= depths)
        if beyond:
            warning_rank_0(
                f"Inkling: inkling_mtp_local_layer_ids {beyond} are >= mtp_num_layers={depths}; "
                "those MTP depths are not built"
            )
        local = [i for i in ids if type(i) is int and 0 <= i < depths]
        window = self.inkling_mtp_window
        if type(window) is not int or window <= 0:
            source = (
                "inkling_mtp_local_extent"
                if self.inkling_mtp_local_extent is not None
                else "inkling_sliding_window (inkling_mtp_local_extent is unset)"
            )
            errors.append(f"the MTP sliding window {source}={window!r} must be a positive int")
        heads = self.inkling_mtp_swa_heads
        groups = self.inkling_swa_num_query_groups
        if type(heads) is not int or heads <= 0:
            errors.append(f"inkling_mtp_swa_num_attention_heads={heads!r} must be a positive int")
        elif local and type(groups) is int and groups > 0:
            # SGLang asserts the MTP sliding K / V heads and head dim equal the trunk's
            if heads % groups or heads % self.tensor_model_parallel_size:
                errors.append(
                    f"inkling_mtp_swa_num_attention_heads={heads} must be a multiple of "
                    f"inkling_swa_num_query_groups={groups} and of TP={self.tensor_model_parallel_size}"
                )
        if self.inkling_mtp_chain_hidden_post_norm:
            errors.append("inkling_mtp_chain_hidden_post_norm=True is not ported (the checkpoints use False)")
        if getattr(self, "mtp_use_repeated_layer", False):
            errors.append(
                "mtp_use_repeated_layer is not supported: every Inkling MTP depth has its own block"
            )
        if getattr(self, "mtp_hybrid_override_pattern", None):
            errors.append("mtp_hybrid_override_pattern is not supported by Inkling")
        if self.pipeline_model_parallel_layout is not None:
            errors.append(
                "a custom pipeline_model_parallel_layout is not supported with Inkling MTP: the MTP "
                "depths run on the last pipeline stage, after the last decoder layer"
            )
        return errors
