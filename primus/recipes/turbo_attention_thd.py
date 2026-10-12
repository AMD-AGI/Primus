"""THD input for Primus's ``PrimusTurboAttention`` (FlyDSL flash attention).

Megatron-Bridge hands core attention THD tensors (``[T, H, D]``, ``qkv_format="thd"``)
whenever the batch carries ``cu_seqlens``. ``PrimusTurboAttention`` only takes ``sbhd`` /
``bhsd``. This recipe packs one 8192-token sequence per row at MBS 1, so ``cu_seqlens``
has two entries and THD is exactly ``sbhd`` with B=1: ``[T, 1, H, D]``, causal. The
check is on ``cu_seqlens.numel()``, host metadata, so it never syncs.

A tail of padding inside that one row needs nothing special: causal attention keeps real
tokens from seeing it, and its loss mask zeroes the gradient it would send back.

More than one sequence per row goes to TE's THD kernel, which the module already is.

The backward is non-deterministic unless ``PRIMUS_TURBO_ATTN_DETERMINISTIC=1``:
FlyDSL's deterministic backward measured 8% slower than TE THD at the 70B shape.
"""

from __future__ import annotations

import os

_STATE = {"installed": False, "dense_calls": 0, "te_calls": 0}


def _log(msg: str) -> None:
    if os.environ.get("TURBO_DEBUG", "0") in ("1", "true", "True"):
        try:
            with open("/results/turbo_decide.log", "a") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass


def install() -> bool:
    if _STATE["installed"]:
        return True
    import transformer_engine as te
    from megatron.core.transformer.enums import AttnMaskType
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboAttention

    stock_forward = PrimusTurboAttention.forward
    deterministic = os.environ.get("PRIMUS_TURBO_ATTN_DETERMINISTIC", "0") in ("1", "true", "True")

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
    ):
        self.deterministic_mode = deterministic
        if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
            return stock_forward(
                self, query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params
            )

        cu_q = packed_seq_params.cu_seqlens_q_padded
        if cu_q is None:
            cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv_padded
        if cu_kv is None:
            cu_kv = packed_seq_params.cu_seqlens_kv

        if query.dim() == 3 and cu_q.numel() == 2 and cu_kv.numel() == 2:
            _STATE["dense_calls"] += 1
            if _STATE["dense_calls"] == 1:
                _log(
                    f"[TurboAttnTHD] one sequence per row: THD {tuple(query.shape)} -> sbhd B=1, "
                    f"deterministic={deterministic}"
                )
            out = stock_forward(
                self,
                query.unsqueeze(1),
                key.unsqueeze(1),
                value.unsqueeze(1),
                attention_mask,
                AttnMaskType.causal,
                attention_bias,
                None,
            )
            return out.view(out.shape[0], -1)

        _STATE["te_calls"] += 1
        if _STATE["te_calls"] == 1:
            _log(f"[TurboAttnTHD] {cu_q.numel() - 1} sequences per row: TE THD fallback")
        return te.pytorch.DotProductAttention.forward(
            self,
            query,
            key,
            value,
            attention_mask=None,
            qkv_format="thd",
            cu_seqlens_q=packed_seq_params.cu_seqlens_q,
            cu_seqlens_kv=packed_seq_params.cu_seqlens_kv,
            cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
            cu_seqlens_kv_padded=packed_seq_params.cu_seqlens_kv_padded,
            max_seqlen_q=packed_seq_params.max_seqlen_q,
            max_seqlen_kv=packed_seq_params.max_seqlen_kv,
            attn_mask_type="padding_causal",
        )

    PrimusTurboAttention.forward = forward

    # Without its own sharded_state_dict the module falls back to Megatron's default,
    # which asks the checkpoint for TE's core_attention._extra_state. A pretrained
    # checkpoint written with TEDotProductAttention has none, so the load fails.
    if "sharded_state_dict" not in PrimusTurboAttention.__dict__:
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            state_dict = {
                k: v
                for k, v in self.state_dict(prefix="", keep_vars=True).items()
                if not k.endswith("_extra_state")
            }
            return make_sharded_tensors_for_checkpoint(state_dict, prefix, {}, sharded_offsets)

        PrimusTurboAttention.sharded_state_dict = sharded_state_dict
        _log("[TurboAttnTHD] sharded_state_dict drops core_attention._extra_state")

    _STATE["installed"] = True
    return True
