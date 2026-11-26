###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo MegaUlysses Patches

Ulysses context parallelism with the all-to-alls fused into the attention projections
(``use_turbo_mega_ulysses: true``; CP > 1 within a node, TP = 1). Self-attention then runs

    x [s/cp, b, h] --ulysses_qkv_gemm_a2a--> q, k, v [s, b, h/cp, d] (local heads, full sequence)
      -> q/k layernorm -> RoPE over the full sequence -> flash_attn_func (local, causal)
      --ulysses_a2a_o_gemm--> [s/cp, b, h]

so the QKV / o_proj GEMMs overlap their all-to-alls (GemmA2A / A2AGemm) and the pack / unpack
copies disappear. The fused ops shard the sequence contiguously (rank r holds tokens
[r*s/cp, (r+1)*s/cp)), so the CP batch split is switched from Megatron's zigzag to contiguous
chunks; with Ulysses every rank attends over the whole sequence, so the zigzag load balancing buys
nothing.

With MLA, ``MLASelfAttention`` runs the latent formulation instead: the q / kv down-projections
and norms stay on the local tokens, k_pe is rotated at its own positions, and the latents
[q_c | c_kv | k_pe] (2112 elements a token for DeepSeek-V3) are all-gathered into the local-head
up-projections (``ulysses_mla_latent_up_proj``, AgGemm / GemmRS), q_pe is rotated over the full
sequence, and the exit is ``ulysses_a2a_o_gemm``.

``turbo_mega_ulysses_impl`` picks the implementation on this same attention path, for A/B runs:
``fused`` (the fused kernels), ``serial`` (the same GEMMs with RCCL collectives in series) and, for
MLA only, ``per_head`` (Megatron's per-head Ulysses: every head up-projected on the local tokens,
then one all-to-all of q, k, v).

``attn`` fuses the all-to-alls into the attention instead of the projections:

    x [s/cp, 1, h] --linear_qkv--> q, k, v [s/cp, 1, heads, d] (local tokens, every head)
      -> q/k layernorm -> RoPE at the tokens' sequence positions
      --ulysses_mega_attn--> o [s/cp, 1, heads, d] --linear_proj--> [s/cp, 1, h]

With MLA the up-projections run on the local tokens for every head (as ``per_head``), k_pe and q_pe
are rotated at the tokens' sequence positions, and ``ulysses_mega_attn`` runs the gfx950 MLA
attention (q / k 192, v 128) at the module's (YaRN) softmax scale.

``ulysses_mega_attn`` is one kernel each way (pull q / k / v, attention on the local heads, push O;
pull dO, the backward, push dq / dk / dv). Its sequence split is striped: chunk ``c`` of
``turbo_mega_ulysses_seq_chunk`` tokens lives on rank ``c % cp`` (0: contiguous), which lets the
attention start before the exchange ends; the CP batch split follows it. Micro-batch size 1 only,
and ``GPU_MAX_HW_QUEUES`` >= 4 (unset is HIP's 4; ``base_env.sh`` exports 2 unless it is set).

The split and the RoPE slicing are global, so no layer can fall back to the stock attention once
the patch is in: every unsupported setting known from the arguments keeps the whole run on the stock
path, and the ones only seen at call time (packed sequences, a non-causal mask, ...) raise.
"""

import importlib.util
import os

import torch
import torch.nn.functional as F

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH = "[Patch:megatron.turbo.mega_ulysses]"
_METADATA_KEYS = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "local_cp_size", "hybrid_cp_group")
_GQA_OPS = "primus_turbo.pytorch.ops.attention.ulysses_fused_a2a"
_MLA_OPS = "primus_turbo.pytorch.ops.attention.ulysses_mla_latent"
_ATTN_OPS = "primus_turbo.pytorch.ops.attention.ulysses_mega_attn"


def _impl(args) -> str:
    impl = getattr(args, "turbo_mega_ulysses_impl", None) or "fused"
    valid = (
        ("fused", "serial", "per_head", "attn")
        if getattr(args, "multi_latent_attention", False)
        else ("fused", "serial", "attn")
    )
    assert impl in valid, f"turbo_mega_ulysses_impl must be one of {valid}, got {impl!r}"
    return impl


def _mega_attn_heads_ok(hq, hkv, cp) -> bool:
    group = hq // hkv if hkv and hq % hkv == 0 else 0
    return hq % cp == 0 and hkv % cp == 0 and group >= 2 and group & (group - 1) == 0


# the head dims of the FlyDSL MLA attention (q / k nope + rope, v)
_MLA_ATTN_DIMS = (192, 128)
# the MLA forward's q block: a sequence chunk holds whole q blocks
_MLA_ATTN_BLOCK = 256
# ulysses_mega_attn's kernels wait on the peer ranks' kernels; with 2 hardware queues (base_env.sh's
# default) they share one with the overlapped grad-reduce / param-gather RCCL kernels and the ranks
# deadlock (the kernels' spin-wait watchdog traps). HIP's default is 4.
_ATTN_MIN_HW_QUEUES = 4


def _mla_attn_dims(args):
    dims = [getattr(args, name, None) for name in ("qk_head_dim", "qk_pos_emb_head_dim", "v_head_dim")]
    return None if None in dims else (dims[0] + dims[1], dims[2])


def _seq_chunk(args) -> int:
    """The ``attn`` implementation's striped-split chunk (0: contiguous)."""
    cs = getattr(args, "turbo_mega_ulysses_seq_chunk", None)
    return 256 if cs is None else int(cs)


_UNSUPPORTED_ARGS = (
    ("experimental_attention_variant", "experimental attention variants"),
    ("add_qkv_bias", "a QKV bias"),
    ("add_bias_linear", "linear biases"),
    ("attention_output_gate", "the attention output gate"),
    ("window_size", "sliding-window attention"),
    ("attention_dropout", "attention dropout"),
    (
        "use_turbo_fused_qk_rmsnorm_rope",
        "use_turbo_fused_qk_rmsnorm_rope (also replaces SelfAttention.forward)",
    ),
)


def _unsupported(args):
    """Why this run keeps the stock CP path, or None."""
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    # the fused kernels exchange over XGMI peer memory, so every CP group must sit inside one node;
    # with TP = 1 and Megatron's tp-cp-ep-dp-pp order a CP group is `cp` consecutive ranks
    if tp != 1 or cp == 1 or local_world % cp != 0:
        return f"needs TP=1 and CP>1 dividing {local_world} GPUs/node (got TP={tp}, CP={cp})"
    for name, what in _UNSUPPORTED_ARGS:
        if getattr(args, name, None):
            return f"{what} ({name}={getattr(args, name)}) is not supported"
    normalization = getattr(args, "normalization", "LayerNorm")
    if normalization not in ("RMSNorm", "LayerNorm"):
        return f"normalization={normalization} is not supported"
    if getattr(args, "multi_latent_attention", False):
        if getattr(args, "q_lora_rank", None) is None:
            return "MLA without a q LoRA (q_lora_rank=None) is not supported"
        if getattr(args, "cache_mla_latents", False):
            return "cache_mla_latents is not supported"
        if "mla_up_proj" in (getattr(args, "recompute_modules", None) or []):
            return "recomputing the MLA up-projection (recompute_modules: mla_up_proj) is not supported"
    if _impl(args) == "attn":
        mla = getattr(args, "multi_latent_attention", False)
        mbs = int(getattr(args, "micro_batch_size", 1) or 1)
        cs = _seq_chunk(args)
        seq = int(getattr(args, "seq_length", 0) or 0)
        hq = getattr(args, "num_attention_heads", None)
        hkv = getattr(args, "num_query_groups", None) if getattr(args, "group_query_attention", False) else hq
        block = _MLA_ATTN_BLOCK if mla else 128
        if mbs != 1:
            return f"turbo_mega_ulysses_impl=attn needs micro_batch_size=1 (got {mbs})"
        hw_queues = os.environ.get("GPU_MAX_HW_QUEUES")
        if hw_queues is not None and (not hw_queues.isdigit() or int(hw_queues) < _ATTN_MIN_HW_QUEUES):
            return (
                f"turbo_mega_ulysses_impl=attn needs GPU_MAX_HW_QUEUES>={_ATTN_MIN_HW_QUEUES} (got "
                f"{hw_queues}): with fewer hardware queues its kernels deadlock against overlapped RCCL"
            )
        if mla:
            dims = _mla_attn_dims(args)
            if dims is not None and dims != _MLA_ATTN_DIMS:
                return (
                    f"turbo_mega_ulysses_impl=attn with MLA needs q / k head dim {_MLA_ATTN_DIMS[0]} and v "
                    f"{_MLA_ATTN_DIMS[1]} (the FlyDSL MLA attention's), got {dims[0]} / {dims[1]}"
                )
            if hq is not None and hq % cp:
                return f"turbo_mega_ulysses_impl=attn needs the {hq} heads to split over CP={cp}"
        elif hq is not None and hkv is not None and not _mega_attn_heads_ok(hq, hkv, cp):
            return (
                f"turbo_mega_ulysses_impl=attn needs the heads to split over CP={cp} and a power-of-two "
                f"GQA group >= 2 (the FlyDSL backward's), got {hq} q / {hkv} kv heads"
            )
        if cs < 0 or cs % block or (seq and seq % ((cs or block) * cp)):
            return (
                f"turbo_mega_ulysses_seq_chunk={cs} must be 0 or a multiple of {block} whose CP stripe "
                f"(seq_chunk * CP, or {block} * CP when 0) divides seq_length={seq}"
            )
    return None


def _missing_ops(args):
    """The Primus-Turbo ops module this run needs and the installed Turbo lacks, or None."""
    if _impl(args) == "attn":
        needed = [_ATTN_OPS]
    else:
        needed = [_GQA_OPS] + ([_MLA_OPS] if getattr(args, "multi_latent_attention", False) else [])
    for name in needed:
        try:
            if importlib.util.find_spec(name) is None:
                return name
        except ModuleNotFoundError:
            return name
    return None


def _is_turbo_mega_ulysses_can_patch(ctx: PatchContext) -> bool:
    """
    Check if MegaUlysses is enabled and this run can take it.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True
      - use_turbo_mega_ulysses == True
      - TP == 1, 1 < CP within a node, and none of the settings in ``_unsupported``
      - the installed Primus-Turbo provides the Ulysses ops (MLA needs ``ulysses_mla_latent``)
    """
    args = get_args(ctx)
    if not bool(getattr(args, "use_turbo_mega_ulysses", False)):
        return False
    _impl(args)
    reason = _unsupported(args)
    if reason is not None:
        warning_rank_0(f"{_PATCH} {reason}, keeping the stock CP path")
        return False
    if not is_primus_turbo_can_patch(ctx):
        return False
    missing = _missing_ops(args)
    if missing is not None:
        warning_rank_0(f"{_PATCH} the installed Primus-Turbo has no {missing}, keeping the stock CP path")
        return False
    return True


def _contiguous_cp_split(batch, cp_group):
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    if cp_size == 1:
        return batch
    for key, val in batch.items():
        if key in _METADATA_KEYS or val is None:
            continue
        seq_dim = 2 if key == "attention_mask" else 1
        chunk = val.size(seq_dim) // cp_size
        batch[key] = val.narrow(seq_dim, cp_rank * chunk, chunk)
    return batch


def _striped_cp_split(seq_chunk):
    """The CP batch split of ``ulysses_mega_attn``: chunk ``c`` of ``seq_chunk`` tokens on rank ``c % cp``."""
    if seq_chunk == 0:
        return _contiguous_cp_split

    def split(batch, cp_group):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        if cp_size == 1:
            return batch
        for key, val in batch.items():
            if key in _METADATA_KEYS or val is None:
                continue
            seq_dim = 2 if key == "attention_mask" else 1
            stripes = val.size(seq_dim) // (seq_chunk * cp_size)
            striped = val.unflatten(seq_dim, (stripes, cp_size, seq_chunk))
            batch[key] = striped.select(seq_dim + 1, cp_rank).flatten(seq_dim, seq_dim + 1).contiguous()
        return batch

    return split


def _local_positions(s_local, cp_size, cp_rank, seq_chunk, device):
    """Sequence positions of this rank's tokens under ``_striped_cp_split``."""
    t = torch.arange(s_local, device=device)
    if seq_chunk == 0:
        return cp_rank * s_local + t
    return (t // seq_chunk * cp_size + cp_rank) * seq_chunk + t % seq_chunk


def _has_bias(linear) -> bool:
    # TE linears without a bias may still carry an empty `bias` tensor
    bias = getattr(linear, "bias", None)
    return isinstance(bias, torch.Tensor) and bias.numel() > 0


def _input_norm(linear, x):
    """The input norm TE fuses into ``linear_qkv`` (LayerNormLinear), applied as its forward would."""
    weight = getattr(linear, "layer_norm_weight", None)
    # fused_residual_rmsnorm hands over x already normalized
    if weight is None or getattr(linear, "_skip_fused_norm", False):
        return x
    if getattr(linear, "zero_centered_gamma", False):
        weight = weight + 1
    eps = linear.eps
    if linear.normalization == "RMSNorm":
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    if linear.normalization == "LayerNorm":
        return F.layer_norm(x, (x.shape[-1],), weight, linear.layer_norm_bias, eps)
    raise RuntimeError(f"{_PATCH} unsupported normalization {linear.normalization!r}")


def _run_forward_pre_hooks(module):
    # The fused path reads the module's weights without calling it, so run its forward pre-hooks the way
    # nn.Module.__call__ would; with overlap_param_gather, DDP waits there for the weights' all-gather.
    for hook_id, hook in module._forward_pre_hooks.items():
        if hook_id in module._forward_pre_hooks_with_kwargs:
            hook(module, (), {})
        else:
            hook(module, ())


def _raise_unsupported(unsupported, what):
    reasons = [name for name, hit in unsupported.items() if hit]
    if reasons:
        raise NotImplementedError(
            f"{_PATCH} {', '.join(reasons)} not supported{what}; the batch is already split contiguously "
            "for MegaUlysses, so the stock attention cannot run either. Set use_turbo_mega_ulysses: false"
        )


class _NoCPGroup:
    """RoPE's cp_group stand-in after the all-to-all: q and k already span the whole sequence."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _a2a_helper(b, seq, hq, hkv, d, n):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    return get_attention_cp_a2a_helper(b, seq, hq, hkv, d, d, 0, n)


def _a2a(t, group):
    from torch.distributed.nn.functional import all_to_all_single

    return all_to_all_single(torch.empty_like(t), t.contiguous(), group=group)


def _serial_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    """Unfused ``ulysses_qkv_gemm_a2a``: GEMM, pack, all-to-all, unpack in series."""
    s_local, b = x.shape[:2]
    helper = _a2a_helper(b, s_local * group.size(), hq, hkv, d, group.size())
    qkv = F.linear(x, w_qkv).view(s_local, b, hkv, hq // hkv + 2, d)
    q = qkv[..., : hq // hkv, :].reshape(s_local, b, hq, d)
    k, v = qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    return helper.splits_qkv_after_a2a(_a2a(helper.combine_qkv_before_a2a(q, k, v), group))


def _serial_a2a_o_gemm(o, w_o, residual, group, qkv_format):
    """Unfused ``ulysses_a2a_o_gemm``."""
    seq, b, heads, d = o.shape
    n = group.size()
    helper = _a2a_helper(b, seq, heads * n, heads * n, d, n)
    o_gathered = helper.reshape_o_after_a2a(_a2a(helper.reshape_o_before_a2a(o), group))
    return F.linear(o_gathered.reshape(*o_gathered.shape[:2], -1), w_o)


def _make_gqa_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType
    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
        ulysses_a2a_o_gemm,
        ulysses_qkv_gemm_a2a,
    )

    no_cp = _NoCPGroup()
    if _impl(args) == "serial":
        qkv_op, o_op = _serial_qkv_gemm_a2a, _serial_a2a_o_gemm
    else:
        qkv_op, o_op = ulysses_qkv_gemm_a2a, ulysses_a2a_o_gemm

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        _raise_unsupported(
            {
                "cross attention": key_value_states is not None,
                "inference": inference_context is not None or inference_params is not None,
                "packed sequences": packed_seq_params is not None,
                "an attention bias": attention_bias is not None,
                f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
                "projection biases": _has_bias(self.linear_qkv) or _has_bias(self.linear_proj),
            },
            "",
        )

        hq, hkv, d = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
        _run_forward_pre_hooks(self.linear_qkv)
        x = _input_norm(self.linear_qkv, hidden_states)
        q, k, v = qkv_op(x, self.linear_qkv.weight, hq, hkv, d, cp_group, "sbhd", "interleaved")
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
        if self.k_layernorm is not None:
            k = self.k_layernorm(k)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            q = apply_rotary_pos_emb(q, q_pos_emb, config=cfg, cp_group=no_cp)
            k = apply_rotary_pos_emb(k, k_pos_emb, config=cfg, cp_group=no_cp)
        # sbhd -> bshd views for flash-attn
        o = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            dropout_p=0.0,
            softmax_scale=cfg.softmax_scale,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        _run_forward_pre_hooks(self.linear_proj)
        y = o_op(o, self.linear_proj.weight, None, cp_group, "sbhd")
        return y, None

    return forward


def _make_mega_attn_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType
    from primus_turbo.pytorch.ops.attention.ulysses_mega_attn import ulysses_mega_attn

    seq_chunk = _seq_chunk(args)
    no_cp = _NoCPGroup()
    positions = {}

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        n = cp_group.size()
        s_local, b = hidden_states.shape[:2]
        hq, hkv, d = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
        _raise_unsupported(
            {
                "cross attention": key_value_states is not None,
                "inference": inference_context is not None or inference_params is not None,
                "packed sequences": packed_seq_params is not None,
                "an attention bias": attention_bias is not None,
                f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
                f"micro-batch size {b}": b != 1,
                f"softmax_scale={cfg.softmax_scale}": cfg.softmax_scale not in (None, d**-0.5),
                f"{hq} q / {hkv} kv heads on {n} ranks": not _mega_attn_heads_ok(hq, hkv, n),
            },
            " with turbo_mega_ulysses_impl=attn",
        )

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            at = (s_local, n, cp_group.rank(), q_pos_emb.device)
            if at not in positions:
                positions[at] = _local_positions(s_local, n, cp_group.rank(), seq_chunk, q_pos_emb.device)
            pos = positions[at]
            query = apply_rotary_pos_emb(query, q_pos_emb[pos], config=cfg, cp_group=no_cp)
            key = apply_rotary_pos_emb(key, k_pos_emb[pos], config=cfg, cp_group=no_cp)
        o = ulysses_mega_attn(query, key, value, cp_group, seq_chunk=seq_chunk)
        return self.linear_proj(o.view(s_local, b, hq * d))

    return forward


def _seq_to_heads(tensors, group):
    """``[s/n, b, h, d_i]`` tensors of this rank's tokens -> ``[s, b, h/n, d_i]`` of its heads, in one
    all-to-all (Megatron's per-head Ulysses)."""
    n = group.size()
    parts = [
        t.view(t.shape[0], t.shape[1], n, t.shape[2] // n, t.shape[3]).movedim(2, 0).reshape(n, -1)
        for t in tensors
    ]
    flat = _a2a(torch.cat(parts, dim=1), group)
    outs = flat.split([p.shape[1] for p in parts], dim=1)
    return [o.reshape(t.shape[0] * n, t.shape[1], t.shape[2] // n, t.shape[3]) for o, t in zip(outs, tensors)]


def _make_mla_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType

    impl = _impl(args)
    if impl == "attn":
        import inspect

        from primus_turbo.pytorch.ops.attention.ulysses_mega_attn import (
            ulysses_mega_attn,
        )

        if "softmax_scale" not in inspect.signature(ulysses_mega_attn).parameters:
            raise RuntimeError(
                f"{_PATCH} the installed Primus-Turbo's ulysses_mega_attn has no MLA support "
                "(no softmax_scale); turbo_mega_ulysses_impl=attn with MLA needs a newer Primus-Turbo"
            )
        seq_chunk = _seq_chunk(args)
        positions = {}
    else:
        from primus_turbo.pytorch.ops import flash_attn_func
        from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
            ulysses_a2a_o_gemm,
        )
        from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import (
            ulysses_mla_latent_up_proj,
        )

        o_op = ulysses_a2a_o_gemm if impl == "fused" else _serial_a2a_o_gemm
    no_cp = _NoCPGroup()

    def rope(t, emb, cfg, mscale):
        return apply_rotary_pos_emb(
            t, emb, config=cfg, mscale=mscale, cp_group=no_cp, mla_rotary_interleaved=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        _raise_unsupported(
            {
                "cross attention": key_value_states is not None,
                "inference": inference_context is not None or inference_params is not None,
                "packed sequences": packed_seq_params is not None,
                "an attention bias": attention_bias is not None,
                f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
                "projection biases": any(
                    _has_bias(m) for m in (self.linear_q_up_proj, self.linear_kv_up_proj, self.linear_proj)
                ),
            },
            " with MLA",
        )

        n, rank = cp_group.size(), cp_group.rank()
        s_local, b = hidden_states.shape[:2]
        heads, dn, dr, dv = cfg.num_attention_heads, cfg.qk_head_dim, cfg.qk_pos_emb_head_dim, cfg.v_head_dim
        if impl == "attn":
            _raise_unsupported(
                {
                    f"micro-batch size {b}": b != 1,
                    f"head dims {dn} + {dr} / {dv}": (dn + dr, dv) != _MLA_ATTN_DIMS,
                    f"{heads} heads on {n} ranks": heads % n != 0,
                },
                " with MLA and turbo_mega_ulysses_impl=attn",
            )

        q_c, kv_combined = self._qkv_down_projection(hidden_states)
        c_kv, k_pe = torch.split(kv_combined, [cfg.kv_lora_rank, dr], dim=-1)
        _run_forward_pre_hooks(self.linear_q_up_proj)
        _run_forward_pre_hooks(self.linear_kv_up_proj)
        q_c = _input_norm(self.linear_q_up_proj, self.q_layernorm(q_c))
        c_kv = _input_norm(self.linear_kv_up_proj, self.kv_layernorm(c_kv))
        w_q, w_kv = self.linear_q_up_proj.weight, self.linear_kv_up_proj.weight

        # full-sequence frequencies: a size-1 group keeps YaRN from slicing them zigzag
        emb = self.rotary_pos_emb(s_local * n, cp_group=no_cp)
        emb, mscale = emb if isinstance(emb, tuple) else (emb, 1.0)
        if impl == "attn":
            at = (s_local, n, rank, emb.device)
            if at not in positions:
                positions[at] = _local_positions(s_local, n, rank, seq_chunk, emb.device)
            emb_local = emb[positions[at]]
        else:
            emb_local = emb[rank * s_local : (rank + 1) * s_local]
        k_pe = rope(k_pe.unsqueeze(-2), emb_local, cfg, mscale).squeeze(-2)

        if impl in ("per_head", "attn"):
            q = F.linear(q_c, w_q).view(s_local, b, heads, dn + dr)
            k_nope, v = F.linear(c_kv, w_kv).view(s_local, b, heads, dn + dv).split([dn, dv], dim=-1)
            k = torch.cat((k_nope, k_pe.unsqueeze(-2).expand(s_local, b, heads, dr)), dim=-1)
            if impl == "attn":
                q_nope, q_pe = q.split([dn, dr], dim=-1)
                q = torch.cat((q_nope, rope(q_pe, emb_local, cfg, mscale)), dim=-1)
                o = ulysses_mega_attn(
                    q, k, v, cp_group, seq_chunk=seq_chunk, softmax_scale=self.softmax_scale
                )
                return self.linear_proj(o.view(s_local, b, heads * dv))
            q, k, v = _seq_to_heads([q, k, v], cp_group)
        else:
            kv_latent = torch.cat((c_kv, k_pe), dim=-1)
            q, k, v = ulysses_mla_latent_up_proj(
                q_c, kv_latent, w_q, w_kv, heads, cp_group, dn, dr, dv, "sbhd", fused=impl == "fused"
            )
        q_nope, q_pe = q.split([dn, dr], dim=-1)
        q = torch.cat((q_nope, rope(q_pe, emb, cfg, mscale)), dim=-1)

        o = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        _run_forward_pre_hooks(self.linear_proj)
        return o_op(o, self.linear_proj.weight, None, cp_group, "sbhd"), None

    return forward


_IMPL_DESCRIPTION = {
    (False, "fused"): "GemmA2A / A2AGemm",
    (False, "serial"): "GEMM + RCCL all-to-all in series",
    (False, "attn"): "local QKV / o_proj GEMMs, all-to-alls fused into the attention (ulysses_mega_attn)",
    (True, "fused"): "latent AgGemm / GemmRS + A2AGemm / GemmA2A",
    (True, "serial"): "latent RCCL all-gather + GEMM, o all-to-all + GEMM in series",
    (True, "per_head"): "per-head q/k/v all-to-all (Megatron Ulysses), o all-to-all + GEMM in series",
    (True, "attn"): "local up-projections / o_proj, all-to-alls fused into the attention (ulysses_mega_attn)",
}


@register_patch(
    "megatron.turbo.mega_ulysses",
    backend="megatron",
    phase="before_train",
    description="Ulysses CP with the all-to-alls fused into the attention projections (MegaUlysses)",
    condition=_is_turbo_mega_ulysses_can_patch,
)
def patch_mega_ulysses(ctx: PatchContext):
    import megatron.core.models.common.embeddings.rotary_pos_embedding as rope_mod
    import megatron.core.utils as mcore_utils

    args = get_args(ctx)
    mla = bool(getattr(args, "multi_latent_attention", False))
    impl = _impl(args)
    # builds the forward (importing the Turbo ops) first: a failure must come before anything global is switched
    if mla:
        forward = _make_mla_forward(args)
    elif impl == "attn":
        forward = _make_mega_attn_forward(args)
    else:
        forward = _make_gqa_forward(args)
    if impl == "attn":
        mcore_utils._get_batch_on_this_cp_rank_per_sequence_balancing = _striped_cp_split(_seq_chunk(args))
        split = (
            "contiguous" if _seq_chunk(args) == 0 else f"striped CP split ({_seq_chunk(args)}-token chunks)"
        )
    else:
        mcore_utils._get_batch_on_this_cp_rank_per_sequence_balancing = _contiguous_cp_split
        split = "contiguous CP split"
    # RoPE runs after the all-to-all over the full sequence, so keep the full-length freqs
    rope_mod.get_pos_emb_on_this_cp_rank = lambda emb, seq_dim, cp_group: emb
    if mla:
        from megatron.core.transformer.multi_latent_attention import (
            MLASelfAttention as attention_cls,
        )
    else:
        from megatron.core.transformer.attention import SelfAttention as attention_cls
    attention_cls.forward = forward
    log_rank_0(f"{_PATCH} {attention_cls.__name__} -> {_IMPL_DESCRIPTION[(mla, impl)]}, {split}")
