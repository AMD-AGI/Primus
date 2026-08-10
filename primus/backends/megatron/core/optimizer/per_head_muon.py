###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-Head Muon — Kimi K3 tech report §2.5, as an option on the existing Muon path.

What the report asks for
------------------------
    "instead of applying Newton-Schulz orthogonalization to the full Q, K, and V
    projection matrices, we partition their momentum matrices along the head
    dimension and orthogonalize each head's block separately. [...] per-head
    orthogonalization equalizes the update scale across heads. [...] It also
    slightly reduces optimizer overhead, as Newton-Schulz iterations on tall
    per-head blocks are cheaper than on the full projection matrix."

    -- report §2.5

Why this is a patch and not a new optimizer
-------------------------------------------
Upstream already has the exact hook the report needs:
:meth:`megatron.core.optimizer.muon.TensorParallelMuon.orthogonalize`
(``muon.py:110-162``), which upstream itself documents as *the* override point for
"splitting fused parameters" (``orthogonalized_optimizer.py:63-86``). It already
carries a ``split_qkv`` branch that splits a **fused** ``linear_qkv.weight`` into
Q/K/V and orthogonalizes each of the three *as a whole matrix*
(``muon.py:136-159``). Per-head blocking is the same idea one level finer, so it
belongs in that method rather than in a parallel optimizer that would have to
re-derive master weights, TP groups, param groups and checkpointing.

``third_party/`` is a clean submodule, so the override is installed by
monkeypatch from :mod:`primus.backends.megatron.patches.per_head_muon_patches`.

Why Kimi K3 needs this at all
-----------------------------
``muon.py:248-250`` sets ``param.is_qkv`` only for ``'linear_qkv.weight' in name``,
under a ``# TODO(deyuf): support MLA``. **Kimi K3 has no such parameter**: its full
attention layers are MLA (``linear_q_down_proj`` / ``linear_q_up_proj`` /
``linear_kv_down_proj`` / ``linear_kv_up_proj``) and its linear-attention layers are
KDA (separate ``q_proj`` / ``k_proj`` / ``v_proj``). So on K3 the upstream
``split_qkv`` flag is a silent no-op and *every* attention projection currently gets
whole-matrix orthogonalization.

Which parameters count as per-head Q/K/V
----------------------------------------
Selected by default (``head_axis=0``, i.e. heads along the output rows, which is also
the tensor-parallel partition dim for every one of them):

===============================  =========================================  ==================
leaf module                      logical weight shape                       per-head block
===============================  =========================================  ==================
``linear_q_up_proj`` (MLA)       ``[nh * q_head_dim, q_lora_rank]``          ``[q_head_dim, q_lora_rank]``
``linear_q_proj``   (MLA, no     ``[nh * q_head_dim, hidden]``               ``[q_head_dim, hidden]``
                     q-LoRA)
``linear_kv_up_proj`` (MLA)      ``[nh * (qk_head_dim + v_head_dim),``       ``[qk_head_dim, kv_lora_rank]``
                                 ``  kv_lora_rank]``                        and ``[v_head_dim, kv_lora_rank]``
``q_proj`` / ``k_proj`` (KDA)    ``[nkh * linear_key_head_dim, hidden]``     ``[linear_key_head_dim, hidden]``
``v_proj``            (KDA)      ``[nvh * linear_value_head_dim, hidden]``   ``[linear_value_head_dim, hidden]``
===============================  =========================================  ==================

``q_head_dim = qk_head_dim + qk_pos_emb_head_dim`` (``multi_latent_attention.py:115``).

Deliberately **not** selected:

* ``linear_q_down_proj`` / ``linear_kv_down_proj`` — these are the *latent*
  projections (``multi_latent_attention.py:421-438, 466-483``). They have no head
  axis at all: ``[q_lora_rank, hidden]`` and ``[kv_lora_rank + qk_pos_emb_head_dim,
  hidden]``. The report names only the Q/K/V projections.
* ``f_a_proj`` (KDA) — the *replicated* ``[key_head_dim, hidden]`` decay-gate latent
  (``kimi_delta_attention.py:315-327``); same argument.
* ``b_proj`` (KDA) — ``[num_heads, hidden]`` (``kimi_delta_attention.py:355``), one
  row per head. A per-head block would be ``[1, hidden]``, where Newton-Schulz
  degenerates to a sign/normalisation. Also not a Q/K/V projection.
* ``f_b_proj`` (KDA) — head-structured but it is the log-decay gate, not Q/K/V.
* embeddings, output layer, router, experts, MLP — untouched, exactly as before.

Opt-in only (see :class:`PerHeadMuonConfig`):

* ``linear_proj`` (MLA) / ``o_proj`` (KDA) — the *output* projections. They are
  head-structured, but along dim **1**: ``[hidden, nh * v_head_dim]``
  (``multi_latent_attention.py:171-183``, ``kimi_delta_attention.py:374-386``). The
  report says "Q, K and V", so this defaults to OFF.
* ``linear_o_gate`` (MLA) / ``g_proj`` (KDA) — the sigmoid output gates
  (``kimi_k3_mla_attention.py:306-320``, ``kimi_delta_attention.py:366``). Head
  structured on dim 0, but not Q/K/V. Defaults to OFF.

Fused ``linear_qkv.weight`` (DeepSeek-V4 / GPT / GQA) is **never** touched by this
module: it keeps going down upstream's unchanged ``split_qkv`` branch. Per-head
blocking of a fused GQA QKV is left as future work rather than silently changing a
path this work package cannot A/B.

Tensor parallelism
------------------
Every default-selected parameter is a ``ColumnParallelLinear`` weight partitioned on
dim 0, which *is* the head axis. Each rank therefore holds a whole number of
**complete** head blocks, so per-head orthogonalization is exactly local: no
all-gather (``newton_schulz_tp`` ``duplicated``) and no per-step all-reduce
(``distributed``) is needed, and the result is bit-identical to the unsharded
computation. This module never reads ``num_heads`` from the config at orthogonalize
time; it derives the local head count from ``grad.shape[head_axis] // rows_per_head``,
which makes the TP case correct by construction.

Concretely, for a ``head_axis=0`` parameter the per-block call passes
``partition_dim=None``, which sends ``newton_schulz_tp`` down its no-communication
fallback (``muon_utils.py:199-201``) and makes ``get_muon_scale_factor`` see the true
logical block shape (``muon.py:78-80`` would otherwise multiply the partitioned dim by
the TP size). For an opt-in ``head_axis=1`` parameter the blocks are *not* locally
complete, so the original ``partition_dim`` is passed straight through and upstream's
TP machinery handles it — supported in principle, **untested**: this work package's
runs are all TP=1.

Update scale
------------
This changes the update *magnitude*, not only its direction, and that is inherent to
the method rather than a choice made here. ``scaled_orthogonalize_fn`` derives the
Muon scale factor from the shape of whatever tensor it is handed (``muon.py:78-80,
89``), so a ``[D, N]`` block gets ``get_muon_scale_factor(D, N)`` instead of
``get_muon_scale_factor(H*D, N)``. Under the default ``spectral`` mode that is
``sqrt(max(D,N))`` instead of ``sqrt(max(H*D,N))``. Equalising the per-head update
scale is the report's stated goal, and upstream's ``split_qkv`` branch already
behaves this way for fused QKV (it re-enters ``scaled_orthogonalize_fn`` with the
split shapes), so this module follows the same convention rather than inventing a
compensating factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# transformer_engine SIGABRTs unless torch is imported first (node/README.md:125-136).
import torch

__all__ = [
    "PerHeadMuonConfig",
    "HeadBlockSpec",
    "PER_HEAD_SPEC_ATTR",
    "MIN_BLOCK_ROWS",
    "batched_newton_schulz",
    "orthogonalize_per_head",
    "head_block_spec_for",
    "tag_per_head_params",
    "propagate_specs_to_master_weights",
    "make_per_head_orthogonalize",
]

logger = logging.getLogger(__name__)

#: Attribute used to carry a :class:`HeadBlockSpec` on a parameter. Deliberately not
#: added to ``megatron.core.tensor_parallel.layers._MODEL_PARALLEL_ATTRIBUTE_DEFAULTS``
#: (which is what propagates ``is_qkv`` to master weights, ``layers.py:60-66,125-133``):
#: inserting a key there would also make ``set_tensor_model_parallel_attributes``'s
#: ``assert not hasattr(tensor, attribute)`` (``layers.py:106-107``) fire on any tensor
#: created after the insert. :func:`propagate_specs_to_master_weights` copies it across
#: the ``Float16OptimizerWithFloat16Params`` clone explicitly instead.
PER_HEAD_SPEC_ATTR = "_primus_per_head_spec"

#: Blocks thinner than this are left to the whole-matrix path. A ``[1, N]`` block makes
#: Newton-Schulz degenerate to a normalisation, which is not orthogonalization in any
#: useful sense; KDA's ``b_proj`` (``[num_heads, hidden]``) is the concrete case, and it
#: is already excluded by name.
MIN_BLOCK_ROWS = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a YAML/CLI value to bool without ``bool("false") is True``.

    Primus's YAML loader already maps the strings ``"true"``/``"false"`` to real bools
    when it substitutes ``${VAR:default}`` (``yaml_loader.py:15, 103, 111-112``), so on
    the normal path the value arrives as a ``bool``. A raw string can still reach here
    from a CLI override, and ``bool("false")`` being ``True`` would silently enable a
    flag someone explicitly turned off.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", "none", "null", ""):
            return False
        raise ValueError(f"cannot interpret {value!r} as a boolean")
    if value is None:
        return default
    return bool(value)


@dataclass
class PerHeadMuonConfig:
    """Per-Head Muon options, with the defaults this work package settled on.

    Every field is read off Megatron's ``args`` namespace with ``getattr(...,
    default)``, so setting any of them in *any* Primus YAML works and omitting all of
    them reproduces upstream behaviour exactly. They cannot live on
    :class:`~megatron.core.optimizer.OptimizerConfig` because
    ``get_megatron_optimizer_config`` only copies declared
    ``AdamOptimizerConfig`` fields (``training.py:1473-1476``) and that dataclass is
    in ``third_party/``. Primus YAML keys do reach ``get_args()`` regardless of
    Megatron's argparse, via ``train_runtime.py:442-443``.

    Attributes:
        enabled: master switch (``muon_per_head``). **Default False** — with it off
            this module changes nothing anywhere in the repo.
        split_kv: ``muon_per_head_split_kv``. MLA fuses K and V inside one head of
            ``linear_kv_up_proj``: the head's rows are ``[qk_head_dim | v_head_dim]``
            (``multi_latent_attention.py:737-741, 876``). The report is genuinely
            ambiguous here — "partition their momentum matrices along the head
            dimension" argues for one ``[qk+v, r]`` block per head, while "the full Q,
            **K**, and **V** projection matrices" treats K and V as distinct matrices.
            **Default True** (separate K and V blocks) for two reasons: leaving them
            fused preserves inside each head exactly the cross-matrix coupling the
            report objects to, and upstream's own ``split_qkv`` already separates K
            from V before orthogonalizing (``muon.py:145-159``).
        include_output_proj: ``muon_per_head_include_output_proj``. Also block
            ``linear_proj`` / ``o_proj`` (heads on dim 1). **Default False** — the
            report names only Q, K and V.
        include_gates: ``muon_per_head_include_gates``. Also block MLA
            ``linear_o_gate`` and KDA ``g_proj``. **Default False** — same reason.
        impl: ``muon_per_head_impl``, ``"loop"`` or ``"batched"``. ``loop`` re-enters
            ``TensorParallelMuon.scaled_orthogonalize_fn`` once per block and is the
            semantics-defining path (and the only TP-capable one). ``batched`` runs
            one 3-D Newton-Schulz over all heads at once via ``baddbmm``; it agrees
            with ``loop`` to fp32 round-off and is much faster, but falls back to
            ``loop`` whenever a block needs tensor-parallel communication.
            **Default "loop"**.
        strict: ``muon_per_head_strict``. Raise if the switch is on but no parameter
            was selected, instead of silently training whole-matrix Muon.
            **Default True** — a typo in a model name must not look like a successful
            per-head run.
    """

    enabled: bool = False
    split_kv: bool = True
    include_output_proj: bool = False
    include_gates: bool = False
    impl: str = "loop"
    strict: bool = True

    _VALID_IMPLS = ("loop", "batched")

    def __post_init__(self) -> None:
        if self.impl not in self._VALID_IMPLS:
            raise ValueError(f"muon_per_head_impl must be one of {self._VALID_IMPLS}, got {self.impl!r}")

    @classmethod
    def from_args(cls, args: Any) -> "PerHeadMuonConfig":
        """Build from Megatron's ``args`` namespace (or anything with attributes)."""
        return cls(
            enabled=_as_bool(getattr(args, "muon_per_head", False), False),
            split_kv=_as_bool(getattr(args, "muon_per_head_split_kv", True), True),
            include_output_proj=_as_bool(getattr(args, "muon_per_head_include_output_proj", False), False),
            include_gates=_as_bool(getattr(args, "muon_per_head_include_gates", False), False),
            impl=str(getattr(args, "muon_per_head_impl", "loop") or "loop"),
            strict=_as_bool(getattr(args, "muon_per_head_strict", True), True),
        )


# ---------------------------------------------------------------------------
# Head block description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadBlockSpec:
    """How to cut one parameter into per-head blocks.

    Attributes:
        rows: rows belonging to a single head, in order. ``(q_head_dim,)`` for
            ``linear_q_up_proj``; ``(qk_head_dim, v_head_dim)`` for a split
            ``linear_kv_up_proj``; ``(qk_head_dim + v_head_dim,)`` for an unsplit one.
            Each entry becomes its own orthogonalized block, so the tuple length is
            the number of *logical matrices* fused inside a head.
        head_axis: the dim the heads are laid out along. 0 for every default
            selection; 1 only for the opt-in output projections.
        rule: which selection rule matched, for logging and for tests to assert on.
    """

    rows: Tuple[int, ...]
    head_axis: int = 0
    rule: str = ""

    @property
    def rows_per_head(self) -> int:
        """Total extent along ``head_axis`` taken by one head."""
        return int(sum(self.rows))

    def num_heads(self, shape: Sequence[int]) -> int:
        """Local head count implied by ``shape``. TP-correct: it uses the local shape."""
        return int(shape[self.head_axis]) // self.rows_per_head

    def matches(self, shape: Sequence[int]) -> bool:
        """Whether ``shape`` is consistent with this spec and worth blocking.

        ``num_heads == 1`` is accepted on purpose. Under TP it means this rank owns
        exactly one head, and per-head semantics there are "orthogonalize the local
        shard on its own" — which is *not* what falling through to the whole-matrix
        path would do, since that path all-gathers the full matrix across the TP group
        first (``muon_utils.py:208-214``).
        """
        if len(shape) != 2:
            return False
        extent = int(shape[self.head_axis])
        rph = self.rows_per_head
        if rph <= 0 or extent < rph or extent % rph != 0:
            return False
        return min(self.rows) >= MIN_BLOCK_ROWS


# ---------------------------------------------------------------------------
# Batched Newton-Schulz
# ---------------------------------------------------------------------------


def batched_newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "quintic",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz on a stack of matrices, one ``baddbmm`` pair per step.

    A batched twin of ``emerging_optimizers...muon_utils.newton_schulz``
    (``muon_utils.py:67-161``), reusing that module's coefficient table so no numeric
    constant is duplicated here. It exists because ``newton_schulz_step`` is built on
    ``torch.addmm`` (``muon_utils.py:251-255``) and is therefore 2-D only, which forces
    the straightforward per-head implementation into a Python loop of ``H`` tiny GEMMs.

    Args:
        x: ``[H, D, N]`` fp32 stack. Each ``x[h]`` is orthogonalized independently.
        steps: Newton-Schulz iterations; must be a multiple of the coefficient-set
            length, exactly as upstream requires (``muon_utils.py:135-136``).
        coefficient_type: key into upstream's ``_COEFFICIENT_SETS``.
        eps: floor for the Frobenius pre-normalisation.

    Returns:
        ``[H, D, N]``, same dtype and layout convention as the input.
    """
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        _COEFFICIENT_SETS,
    )

    if x.ndim != 3:
        raise ValueError(f"batched_newton_schulz expects a 3-D stack, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"batched_newton_schulz expects float32, got {x.dtype}")

    try:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    except KeyError as exc:  # pragma: no cover - guarded by the config validator
        raise ValueError(f"Invalid coefficient type: {coefficient_type}") from exc
    if steps % len(coefficient_sets) != 0:
        raise ValueError(
            f"steps ({steps}) must be multiple of len(coefficient_sets) " f"({len(coefficient_sets)})."
        )

    # Same rule as upstream (muon_utils.py:115-118): whiten on the smaller dim. All
    # blocks share a shape, so the decision is uniform over the batch.
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    # Mirrors muon_utils.py:140-150: at "medium" precision the iteration runs in bf16.
    if torch.get_float32_matmul_precision() == "medium":
        X = X.to(torch.bfloat16)
    X = X.contiguous()

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    X = X.to(torch.float32)
    if transpose:
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# The blocking itself
# ---------------------------------------------------------------------------


def _split_into_components(grad: torch.Tensor, spec: HeadBlockSpec) -> Tuple[int, List[torch.Tensor]]:
    """``grad`` -> ``(num_heads, [stack_per_component])``, each stack ``[H, r_i, N]``.

    For ``head_axis == 1`` the component stacks are ``[H, M, r_i]`` instead; the
    caller keeps the orientation, because two of the three Muon scale modes are not
    symmetric in ``(size_out, size_in)`` (``muon.py:140-149``).
    """
    num_heads = spec.num_heads(grad.shape)
    rph = spec.rows_per_head
    if spec.head_axis == 0:
        view = grad.reshape(num_heads, rph, grad.shape[1])
        components = list(torch.split(view, list(spec.rows), dim=1))
    elif spec.head_axis == 1:
        # [M, H*rph] -> [M, H, rph] -> [H, M, rph]
        view = grad.reshape(grad.shape[0], num_heads, rph).transpose(0, 1)
        components = list(torch.split(view, list(spec.rows), dim=2))
    else:  # pragma: no cover - HeadBlockSpec is only constructed with 0 or 1
        raise ValueError(f"head_axis must be 0 or 1, got {spec.head_axis}")
    return num_heads, components


def _reassemble(
    components: Sequence[torch.Tensor], spec: HeadBlockSpec, out_shape: Sequence[int]
) -> torch.Tensor:
    """Inverse of :func:`_split_into_components`."""
    if spec.head_axis == 0:
        return torch.cat(list(components), dim=1).reshape(out_shape[0], out_shape[1])
    stacked = torch.cat(list(components), dim=2)  # [H, M, rph]
    return stacked.transpose(0, 1).reshape(out_shape[0], out_shape[1])


def orthogonalize_per_head(
    grad: torch.Tensor,
    spec: HeadBlockSpec,
    scaled_orthogonalize_fn: Callable[..., torch.Tensor],
    *,
    tp_group: Optional["torch.distributed.ProcessGroup"] = None,
    partition_dim: Optional[int] = None,
    impl: str = "loop",
    batched_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Orthogonalize ``grad`` one head block at a time and reassemble.

    ``impl="loop"`` defines the semantics: for every head ``h`` and every fused
    component ``i`` inside that head, call ``scaled_orthogonalize_fn`` on the
    ``[rows[i], N]`` block. ``impl="batched"`` must agree with it to fp32 round-off.

    Args:
        grad: the momentum, shaped like the (possibly TP-sharded) parameter.
        spec: how to cut it.
        scaled_orthogonalize_fn: ``TensorParallelMuon``'s closure, i.e. Newton-Schulz
            followed by the Muon scale factor for the shape it was handed
            (``muon.py:67-90``).
        tp_group: forwarded to ``scaled_orthogonalize_fn``.
        partition_dim: forwarded. For ``head_axis == 0`` the caller is expected to
            pass ``None``, because head blocks are locally complete along the
            partitioned dim; see the module docstring.
        impl: ``"loop"`` or ``"batched"``.
        batched_kwargs: for ``impl="batched"``: ``steps``, ``coefficient_type``,
            ``scale_mode``, ``extra_scale_factor``.

    Returns:
        A new tensor with ``grad``'s shape and dtype.
    """
    num_heads, components = _split_into_components(grad, spec)

    use_batched = impl == "batched" and partition_dim is None
    if impl == "batched" and partition_dim is not None:
        logger.debug(
            "per-head Muon: falling back to the loop implementation for a block that "
            "needs tensor-parallel communication (partition_dim=%s)",
            partition_dim,
        )

    out_components: List[torch.Tensor] = []
    for component in components:
        if use_batched:
            out_components.append(
                _orthogonalize_component_batched(component, spec, dict(batched_kwargs or {}))
            )
        else:
            blocks = [
                scaled_orthogonalize_fn(component[h], tp_group, partition_dim) for h in range(num_heads)
            ]
            out_components.append(torch.stack(blocks, dim=0))

    return _reassemble(out_components, spec, grad.shape)


def _orthogonalize_component_batched(
    component: torch.Tensor, spec: HeadBlockSpec, kwargs: Dict[str, Any]
) -> torch.Tensor:
    """One fused component of every head at once: batched NS plus the Muon scale."""
    from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor

    steps = int(kwargs.get("steps", 5))
    coefficient_type = str(kwargs.get("coefficient_type", "quintic"))
    scale_mode = str(kwargs.get("scale_mode", "spectral"))
    extra_scale_factor = float(kwargs.get("extra_scale_factor", 1.0))

    orth = batched_newton_schulz(component.contiguous(), steps=steps, coefficient_type=coefficient_type)
    # Identical shape bookkeeping to muon.py:78, 89 -- the scale factor sees the block.
    scale_factor = get_muon_scale_factor(component.shape[-2], component.shape[-1], mode=scale_mode)
    return orth * scale_factor * extra_scale_factor


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------


def _leaf_module_name(param_name: str) -> str:
    """``...layers.3.self_attention.linear_q_up_proj.weight`` -> ``linear_q_up_proj``.

    Matching the leaf *module* name rather than a substring matters: ``q_proj`` is a
    substring of ``linear_q_proj``, and ``f_a_proj`` / ``f_b_proj`` / ``b_proj`` all
    end in ``_proj``.
    """
    parts = param_name.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def _int_or_none(config: Any, name: str) -> Optional[int]:
    value = getattr(config, name, None)
    return int(value) if isinstance(value, int) and value > 0 else None


def head_block_spec_for(
    param_name: str, shape: Sequence[int], model_config: Any, config: PerHeadMuonConfig
) -> Optional[HeadBlockSpec]:
    """The per-head :class:`HeadBlockSpec` for ``param_name``, or ``None``.

    Pure and side-effect free so tests can enumerate the rule directly. Returns
    ``None`` for anything that is not a per-head Q/K/V projection under ``config``,
    for a non-weight tensor, and for any shape that contradicts the config geometry
    (which is reported as a warning rather than silently mis-blocked).
    """
    if not param_name.endswith(".weight") and param_name != "weight":
        return None
    if len(shape) != 2:
        return None

    leaf = _leaf_module_name(param_name)
    spec = _candidate_spec(leaf, model_config, config)
    if spec is None:
        return None
    if not spec.matches(shape):
        logger.warning(
            "[Primus:PerHeadMuon] %s matched rule %r but its shape %s is not a multiple "
            "of %d along dim %d; leaving it on the whole-matrix path.",
            param_name,
            spec.rule,
            tuple(shape),
            spec.rows_per_head,
            spec.head_axis,
        )
        return None
    return spec


def _candidate_spec(leaf: str, model_config: Any, config: PerHeadMuonConfig) -> Optional[HeadBlockSpec]:
    """Shape-independent part of the rule: leaf module name -> row structure."""
    # --- MLA (multi_latent_attention.py) ---------------------------------
    qk_head_dim = _int_or_none(model_config, "qk_head_dim")
    v_head_dim = _int_or_none(model_config, "v_head_dim")
    qk_pos_emb_head_dim = getattr(model_config, "qk_pos_emb_head_dim", 0) or 0
    q_head_dim = qk_head_dim + int(qk_pos_emb_head_dim) if qk_head_dim is not None else None

    if leaf in ("linear_q_up_proj", "linear_q_proj") and q_head_dim:
        return HeadBlockSpec(rows=(q_head_dim,), head_axis=0, rule=f"mla.{leaf}")

    if leaf == "linear_kv_up_proj" and qk_head_dim and v_head_dim:
        rows = (qk_head_dim, v_head_dim) if config.split_kv else (qk_head_dim + v_head_dim,)
        rule = "mla.linear_kv_up_proj" + (".split_kv" if config.split_kv else ".fused_kv")
        return HeadBlockSpec(rows=rows, head_axis=0, rule=rule)

    # --- KDA (kimi_delta_attention.py) -----------------------------------
    key_head_dim = _int_or_none(model_config, "linear_key_head_dim")
    value_head_dim = _int_or_none(model_config, "linear_value_head_dim")

    if leaf in ("q_proj", "k_proj") and key_head_dim:
        return HeadBlockSpec(rows=(key_head_dim,), head_axis=0, rule=f"kda.{leaf}")
    if leaf == "v_proj" and value_head_dim:
        return HeadBlockSpec(rows=(value_head_dim,), head_axis=0, rule="kda.v_proj")

    # --- opt-in: output gates (head_axis 0, but not Q/K/V) ----------------
    if config.include_gates:
        if leaf == "linear_o_gate" and v_head_dim:
            return HeadBlockSpec(rows=(v_head_dim,), head_axis=0, rule="mla.linear_o_gate")
        if leaf == "g_proj" and value_head_dim:
            return HeadBlockSpec(rows=(value_head_dim,), head_axis=0, rule="kda.g_proj")

    # --- opt-in: output projections (heads on dim 1) ----------------------
    if config.include_output_proj:
        if leaf == "linear_proj" and v_head_dim:
            return HeadBlockSpec(rows=(v_head_dim,), head_axis=1, rule="mla.linear_proj")
        if leaf == "o_proj" and value_head_dim:
            return HeadBlockSpec(rows=(value_head_dim,), head_axis=1, rule="kda.o_proj")

    return None


@dataclass
class TaggingSummary:
    """What :func:`tag_per_head_params` did, for logging and for tests."""

    selected: Dict[str, HeadBlockSpec] = field(default_factory=dict)
    skipped_head_structured: List[str] = field(default_factory=list)

    @property
    def num_selected(self) -> int:
        return len(self.selected)

    def by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for spec in self.selected.values():
            counts[spec.rule] = counts.get(spec.rule, 0) + 1
        return counts


#: Head-structured parameters that are deliberately left on the whole-matrix path when
#: their opt-in flag is off. Tracked only so the log can say so out loud.
_KNOWN_HEAD_STRUCTURED_OPT_INS = (
    "linear_proj",
    "o_proj",
    "linear_o_gate",
    "g_proj",
    "f_b_proj",
)


def tag_per_head_params(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    model_config: Any,
    config: PerHeadMuonConfig,
) -> TaggingSummary:
    """Attach a :class:`HeadBlockSpec` to every per-head Q/K/V parameter.

    This is the only function in the module that mutates anything, and it is where
    ``config.enabled`` is enforced: with the switch off it tags nothing and returns an
    empty summary, whatever :func:`head_block_spec_for` would have said. The rule
    itself stays pure and always answerable so tests can enumerate it directly.

    Args:
        named_parameters: ``(name, param)`` pairs, e.g. from
            ``model_chunk.named_parameters()``.
        model_config: the model's ``TransformerConfig`` (head dims are read from it).
        config: the per-head options.

    Returns:
        A :class:`TaggingSummary`.
    """
    summary = TaggingSummary()
    if not config.enabled:
        return summary
    for name, param in named_parameters:
        if not getattr(param, "requires_grad", False):
            continue
        spec = head_block_spec_for(name, tuple(param.shape), model_config, config)
        if spec is None:
            if _leaf_module_name(name) in _KNOWN_HEAD_STRUCTURED_OPT_INS:
                summary.skipped_head_structured.append(name)
            continue
        setattr(param, PER_HEAD_SPEC_ATTR, spec)
        summary.selected[name] = spec
    return summary


def propagate_specs_to_master_weights(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> int:
    """Copy the spec onto each param's fp32 master weight.

    ``Float16OptimizerWithFloat16Params`` clones every bf16 param into an fp32
    ``main_param`` and copies only ``shared`` plus the five keys in
    ``_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS`` (``optimizer.py:675-684``,
    ``tensor_parallel/layers.py:60-66``). Under bf16 — which is what
    ``get_megatron_muon_optimizer`` builds (``muon.py:292-303``) — ``orthogonalize``
    receives that clone, so without this step the spec would never be seen.

    Returns:
        How many master weights were tagged.
    """
    count = 0
    for _name, param in named_parameters:
        spec = getattr(param, PER_HEAD_SPEC_ATTR, None)
        if spec is None:
            continue
        main_param = getattr(param, "main_param", None)
        if main_param is not None and getattr(main_param, PER_HEAD_SPEC_ATTR, None) is None:
            setattr(main_param, PER_HEAD_SPEC_ATTR, spec)
            count += 1
    return count


# ---------------------------------------------------------------------------
# The patched method
# ---------------------------------------------------------------------------


def make_per_head_orthogonalize(
    original_orthogonalize: Callable[..., torch.Tensor], config: PerHeadMuonConfig
) -> Callable[..., torch.Tensor]:
    """Wrap ``TensorParallelMuon.orthogonalize`` with the per-head branch.

    The wrapper is a strict superset of the original: a parameter without a
    :class:`HeadBlockSpec` is handed straight to ``original_orthogonalize``, so the
    whole-matrix path and the fused-QKV ``split_qkv`` path (``muon.py:136-159``) are
    bit-identical to upstream.
    """

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        spec: Optional[HeadBlockSpec] = getattr(p, PER_HEAD_SPEC_ATTR, None)
        if spec is None or not spec.matches(grad.shape):
            return original_orthogonalize(self, p, grad, **kwargs)

        # Same tp_group selection as muon.py:123-130.
        pg_collection = getattr(self, "pg_collection", None)
        if pg_collection is not None:
            tp_group = pg_collection.expt_tp if getattr(p, "expert_tp", False) else pg_collection.tp
        else:
            tp_group = None

        if spec.head_axis == 0:
            # Heads are the partitioned dim, so each rank owns whole head blocks and
            # no communication is needed. Passing None also keeps
            # get_muon_scale_factor on the true logical block shape (muon.py:78-80).
            partition_dim: Optional[int] = None
        else:
            partition_dim = getattr(p, "partition_dim", None)
            if partition_dim == -1:
                partition_dim = None
            if getattr(self, "mode", None) == "blockwise":
                partition_dim = None

        return orthogonalize_per_head(
            grad,
            spec,
            self.scaled_orthogonalize_fn,
            tp_group=tp_group,
            partition_dim=partition_dim,
            impl=config.impl,
            batched_kwargs=getattr(self, "_primus_muon_ns_kwargs", None),
        )

    orthogonalize.__doc__ = (
        "Per-Head Muon orthogonalization (Kimi K3 report §2.5); falls through to "
        "upstream for any parameter without a HeadBlockSpec."
    )
    orthogonalize._primus_per_head_wrapped = True  # type: ignore[attr-defined]
    orthogonalize._primus_original = original_orthogonalize  # type: ignore[attr-defined]
    return orthogonalize
