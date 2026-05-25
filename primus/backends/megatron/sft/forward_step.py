###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Forward step function for Megatron SFT training.

This module contains the forward_step function used in supervised fine-tuning,
following the Megatron-Bridge pattern for loss computation while staying
compatible with newer Megatron-LM forward-step entrypoints.
"""

from functools import partial
from typing import Any, Callable, Iterator, Tuple

import torch


_PRE_FORWARD_CANARY_DONE = False


def _unwrap_to_base(m: Any) -> Any:
    """Strip DDP / Float16Module / ``.module`` wrappers down to the unwrapped
    GPT/Megatron model object so we can introspect ``decoder.layers[0]``.

    Stops as soon as no ``module`` attribute is exposed. This mirrors
    Megatron's ``unwrap_model`` semantics but avoids importing it here so
    forward_step has no extra dependency."""
    seen = set()
    cur = m
    while True:
        # Avoid infinite loops if some module proxies its own attribute.
        if id(cur) in seen:
            break
        seen.add(id(cur))
        nxt = getattr(cur, "module", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _pre_forward_canary(model: Any) -> None:
    """One-shot dump (rank-0 only) of base-model weight stats RIGHT BEFORE
    the very first ``model(...)`` call hits the GPU forward graph.

    The pre-wrap canary in ``megatron_sft_trainer._create_model_provider_with_lora``
    already proved (a) ``dist_checkpointing.load`` mutates the base params
    in place to real Llama-2 values and (b) LoRA wrap preserves both
    ``sum`` and ``data_ptr`` of every base param. Despite that, iter-1 LM
    loss is still 13.65 == random-init forward. The only remaining
    explanation is that something **after** the pre-wrap hook (``model.cuda()``
    / ``Float16Module`` wrap / DDP wrap / ``model.train()``) silently swaps
    the underlying tensor storage of the base params. This probe lets us
    see what the forward graph actually executes against."""
    global _PRE_FORWARD_CANARY_DONE
    if _PRE_FORWARD_CANARY_DONE:
        return
    _PRE_FORWARD_CANARY_DONE = True

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    try:
        base = _unwrap_to_base(model)
        layer0 = base.decoder.layers[0]

        def _stat(name: str, getter):
            try:
                t = getter()
                if t is None:
                    print(f"[PRE-FORWARD canary] {name}: None", flush=True)
                    return
                tf = t.detach().float()
                print(
                    f"[PRE-FORWARD canary] {name}: "
                    f"sum={tf.sum().item():.6f} "
                    f"abs_max={tf.abs().max().item():.6f} "
                    f"shape={tuple(t.shape)} dtype={t.dtype} "
                    f"device={t.device} ptr={t.data_ptr()}",
                    flush=True,
                )
            except Exception as e:
                print(f"[PRE-FORWARD canary] {name}: ERR {type(e).__name__}: {e}",
                      flush=True)

        # The pre-wrap canary stored these references on the unwrapped GPT
        # model. After LoRA wrap, the original linears live inside
        # ``LoRALinear.to_wrap``. Try the wrapped path first, then fall
        # back to the unwrapped path (covers the no-LoRA case too).
        def _qkv_w():
            attn = layer0.self_attention.linear_qkv
            return getattr(attn, "to_wrap", attn).weight

        def _qkv_lnw():
            attn = layer0.self_attention.linear_qkv
            tgt = getattr(attn, "to_wrap", attn)
            return getattr(tgt, "layer_norm_weight", None)

        def _proj_w():
            proj = layer0.self_attention.linear_proj
            return getattr(proj, "to_wrap", proj).weight

        def _fc1_w():
            fc1 = layer0.mlp.linear_fc1
            return getattr(fc1, "to_wrap", fc1).weight

        def _fc1_lnw():
            fc1 = layer0.mlp.linear_fc1
            tgt = getattr(fc1, "to_wrap", fc1)
            return getattr(tgt, "layer_norm_weight", None)

        def _fc2_w():
            fc2 = layer0.mlp.linear_fc2
            return getattr(fc2, "to_wrap", fc2).weight

        _stat("L0.attn.linear_qkv.weight", _qkv_w)
        _stat("L0.attn.linear_qkv.layer_norm_weight", _qkv_lnw)
        _stat("L0.attn.linear_proj.weight", _proj_w)
        _stat("L0.mlp.linear_fc1.weight", _fc1_w)
        _stat("L0.mlp.linear_fc1.layer_norm_weight", _fc1_lnw)
        _stat("L0.mlp.linear_fc2.weight", _fc2_w)
        _stat("final_layernorm.weight", lambda: base.decoder.final_layernorm.weight)
        _stat("embedding.word_embeddings.weight",
              lambda: base.embedding.word_embeddings.weight)
        _stat("output_layer.weight", lambda: base.output_layer.weight)

        # Also probe whether the same Parameter object is reachable via
        # DDP-wrapped path ``model.module....`` (which is what autograd
        # actually backprops through). If the canary above and this one
        # disagree on data_ptr / abs_max, that's the smoking gun: the
        # outer wrapper is forwarding through a *different* tensor.
        try:
            cur = model
            depth = 0
            chain = [f"{type(cur).__name__}"]
            while hasattr(cur, "module") and depth < 6:
                cur = cur.module
                depth += 1
                chain.append(f".module={type(cur).__name__}")
            print(f"[PRE-FORWARD canary] wrap-chain (depth={depth}): "
                  f"{''.join(chain)}", flush=True)
            wrapped_layer0 = cur.decoder.layers[0]
            wqkv = wrapped_layer0.self_attention.linear_qkv
            wt = getattr(wqkv, "to_wrap", wqkv).weight
            wtf = wt.detach().float()
            print(
                f"[PRE-FORWARD canary] (via outer wrapper) "
                f"L0.attn.linear_qkv.weight: "
                f"sum={wtf.sum().item():.6f} "
                f"abs_max={wtf.abs().max().item():.6f} "
                f"ptr={wt.data_ptr()}",
                flush=True,
            )
        except Exception as e:
            print(f"[PRE-FORWARD canary] outer-wrapper probe ERR "
                  f"{type(e).__name__}: {e}", flush=True)

    except Exception as e:
        print(f"[PRE-FORWARD canary] OUTER ERR {type(e).__name__}: {e}",
              flush=True)


def _move_to_runtime_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensors to CUDA when available, keep CPU for unit tests."""
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def _empty_loss_result(device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a no-op loss tuple with the expected Megatron shape."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return (
        torch.tensor(0.0, device=device),
        torch.tensor(0, device=device, dtype=torch.int),
        {},
    )


def _build_packed_seq_params(
    cu_seqlens: torch.Tensor,
    num_segments: torch.Tensor,
    max_sub_seqlen: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
):
    """Stitch per-sample cu_seqlens into a batch-global PackedSeqParams.

    ``cu_seqlens`` is shaped ``[batch, MAX_SEGMENTS+1]`` where each row encodes
    sub-sequence boundaries inside one packed sample (padded to a fixed length).
    Megatron's attention path expects a single 1-D ``cu_seqlens_q`` covering
    the whole batch, so we shift each sample's offsets by ``b * seq_len`` and
    concatenate the valid prefixes.
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens_cpu = cu_seqlens.detach().cpu()
    num_segments_cpu = num_segments.detach().cpu()
    batch_size = cu_seqlens_cpu.size(0)

    pieces = []
    pieces.append(torch.tensor([0], dtype=torch.int32))
    for b in range(batch_size):
        n = int(num_segments_cpu[b].item())
        if n <= 0:
            continue
        # Take cu_seqlens[1..n] (excluding leading 0) and shift by b*seq_len.
        valid = cu_seqlens_cpu[b, 1 : n + 1].to(torch.int32) + b * seq_len
        pieces.append(valid)

    global_cu = torch.cat(pieces).to(device=device, dtype=torch.int32)

    if max_sub_seqlen is not None:
        max_seqlen_q = int(max_sub_seqlen.detach().cpu().max().item())
    else:
        diffs = global_cu[1:] - global_cu[:-1]
        max_seqlen_q = int(diffs.max().item()) if diffs.numel() > 0 else seq_len

    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=global_cu,
        cu_seqlens_kv=global_cu,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_q,
    )


def create_sft_forward_step() -> Callable:
    """
    Create and return the forward_step function for SFT training.
    
    This follows the Megatron-Bridge pattern where:
    1. Model is called with labels and returns per-token losses
    2. Loss function applies masking to focus on response tokens
    3. Returns (loss, num_tokens, metrics_dict) for proper DP averaging
    
    Returns:
        forward_step function compatible with Megatron's pretrain loop
    """
    
    def forward_step(data_iterator: Iterator, model, return_schedule_plan: bool = False) -> Tuple:
        """
        Forward step for SFT training.
        
        Args:
            data_iterator: Iterator over training data batches
            model: Megatron GPT model
            return_schedule_plan: Whether to return a schedule plan for
                newer Megatron pipeline schedulers
            
        Returns:
            Tuple of (output_tensor, loss_function)
            - output_tensor: Per-token losses from model
            - loss_function: Lambda that computes final loss with masking
        """
        from megatron.training import get_args

        args = get_args()

        # Handle case where data_iterator is None (e.g., during eval without valid dataset)
        if data_iterator is None:
            return None, lambda output: _empty_loss_result()
        
        # Get batch from iterator
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Return None and a no-op loss function for iteration completion
            return None, lambda output: _empty_loss_result()
        
        # Extract tensors from batch
        tokens = _move_to_runtime_device(batch["input_ids"]).long()
        labels = _move_to_runtime_device(batch["labels"]).long()
        loss_mask = _move_to_runtime_device(batch["loss_mask"]).float()
        packed_seq_params = batch.get("packed_seq_params")

        # Ensure proper shapes [batch, seq]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(0)

        batch_size, seq_len = tokens.size()

        # Sequence-packing attention isolation:
        #   The "correct" way to keep attention from leaking across packed
        #   sub-samples is to forward `PackedSeqParams(qkv_format='thd', cu_seqlens=...)`
        #   to the model. TE then dispatches to its varlen / flash-attention thd
        #   kernel.
        #
        #   However on the current ROCm + aiter image this thd kernel hangs /
        #   SIGABRTs (rank-7 abort during the very first forward).
        #
        #   Default behaviour: DISABLE the thd path. Datasets still emit
        #   ``cu_seqlens`` (we just ignore them here) so the loss / metrics
        #   plumbing keeps working. Attention falls back to plain causal across
        #   the whole packed sequence -- this means a sub-sample CAN attend to
        #   tokens of an earlier sub-sample. ``loss_mask`` already restricts the
        #   loss to response tokens, so this is the same "implicit multi-turn"
        #   regime used by LLaMA-Factory / TRL packing: throughput +20x with a
        #   small accuracy hit.
        #
        #   Set ``use_packed_attention: true`` in the YAML to re-enable the thd
        #   path once the backend supports it.
        use_packed_attention = bool(getattr(args, "use_packed_attention", False))
        if (
            use_packed_attention
            and packed_seq_params is None
            and "cu_seqlens" in batch
        ):
            packed_seq_params = _build_packed_seq_params(
                batch["cu_seqlens"],
                batch["num_segments"],
                batch.get("max_sub_seqlen"),
                seq_len,
                tokens.device,
            )

        # position_ids selection:
        #   * use_packed_attention=True: dataset-provided per-segment position
        #     ids (each sub-segment restarts at 0) match the strict-isolation
        #     thd attention path -- RoPE computes relative distance inside each
        #     segment independently.
        #   * use_packed_attention=False: attention runs causal across the
        #     whole packed sequence, so we want a SINGLE continuous position
        #     stream (0..seq_length-1). Per-segment ids would confuse RoPE
        #     because tokens in different sub-samples could share the same
        #     position id, making relative-distance computation meaningless.
        if use_packed_attention and "position_ids" in batch:
            position_ids = _move_to_runtime_device(batch["position_ids"]).long()
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # attention_mask: None for causal mask (standard GPT autoregressive)
        attention_mask = None
        
        # Create loss function following Megatron-Bridge pattern
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None) -> Tuple:
            """
            Masked next-token loss function.
            
            This function applies the loss mask to focus training only on
            response tokens (where mask=1), ignoring instruction tokens (mask=0).
            
            Args:
                loss_mask: Binary mask [batch, seq] where 1=compute loss, 0=ignore
                output_tensor: Per-token losses from model [batch, seq] or [batch*seq]
                
            Returns:
                Tuple of (loss, num_tokens, metrics_dict):
                - loss: Summed loss for backpropagation
                - num_tokens: Number of non-masked tokens for proper averaging
                - metrics_dict: Dictionary with reporting metrics for logging
                
            Note:
                This follows Megatron's standard loss function signature.
                The training loop will use num_tokens to properly average loss
                across different micro-batches and data-parallel ranks.
            """
            # Model returns per-token losses, flatten for processing
            losses = output_tensor.view(-1).float()
            loss_mask = loss_mask.view(-1).float()
            
            # Apply mask: only compute loss on response tokens (mask=1)
            # Instruction tokens (mask=0) are ignored
            loss = torch.sum(losses * loss_mask)
            
            # Count number of non-masked tokens
            # This is crucial for proper loss averaging across micro-batches
            num_tokens = loss_mask.sum().clone().detach().to(torch.int)
            
            # Create reporting loss for logging
            # Format: [loss_value, num_tokens] concatenated
            # This allows Megatron to compute proper weighted average across DP ranks
            reporting_loss = torch.cat([
                loss.clone().detach().view(1),
                num_tokens.view(1)
            ])
            
            # Return standard Megatron loss function signature
            # (loss, num_tokens, metrics_dict)
            return (loss, num_tokens, {"lm loss": reporting_loss})

        _pre_forward_canary(model)

        if getattr(args, "use_legacy_models", False):
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            return output_tensor, partial(loss_func, loss_mask, model=model)

        if return_schedule_plan:
            if not hasattr(model, "build_schedule_plan"):
                raise AttributeError(
                    "Megatron SFT forward_step received return_schedule_plan=True, "
                    "but the model does not implement build_schedule_plan()."
                )

            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)

        model_kwargs = {
            "labels": labels,
            "loss_mask": loss_mask,
        }
        if packed_seq_params is not None:
            model_kwargs["packed_seq_params"] = packed_seq_params

        output_tensor = model(tokens, position_ids, attention_mask, **model_kwargs)

        return output_tensor, partial(loss_func, loss_mask, model=model)
    
    return forward_step
