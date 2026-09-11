###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Transformer per-layer recompute patch.

When ``config.recompute_layer_ids`` is set, ``TransformerBlock._checkpointed_forward``
recomputes exactly those layers (global indices) -- nothing more, nothing less.

Design:
    The patch is a thin wrapper around Megatron's ``_checkpointed_forward``:

    * When ``config.recompute_layer_ids is None`` the call is forwarded
      verbatim to the original function, so every existing code path
      (``recompute_method == 'uniform' | 'block'``, feature extraction
      via ``extract_layer_indices``, fp8 / fp4, te checkpoint, ...)
      stays owned by Megatron upstream.

    * When ``config.recompute_layer_ids`` is set, a small dedicated
      branch iterates the block's layers and checkpoints only those
      whose *global* index appears in the list.  The only Megatron
      internals duplicated here are the ``custom`` and
      ``checkpoint_handler`` closures -- they are inner functions of
      ``_checkpointed_forward`` and cannot be imported / reused.  We
      keep them byte-compatible with upstream so future fixes port
      mechanically.

    * Companion tests in
      ``tests/unit_tests/backends/megatron/test_recompute_layer_patches.py``
      pin the signature and source fingerprint of Megatron's
      ``_checkpointed_forward``.  Any upstream edit will fail the tests
      with a clear message, forcing the maintainer to re-validate.

MTP:
    ``MultiTokenPredictionLayer`` has its own ``_checkpointed_forward``, gated
    on ``recompute_granularity == 'full'`` alone and dispatching on
    ``recompute_method`` -- which ``recompute_layer_ids`` pins to ``None``.
    Upstream has no ``None`` branch, so MTP + ``recompute_layer_ids`` used to
    raise ``ValueError: Invalid activation recompute method.`` on the first
    forward.  It is wrapped here too, and the id space is extended past the
    decoder so MTP depth *d* is addressable as ``num_layers + d``.
"""

from contextlib import nullcontext

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _mtp_num_layers(args) -> int:
    """``args.mtp_num_layers`` as an int (0 when MTP is off)."""
    return int(getattr(args, "mtp_num_layers", 0) or 0)


def mtp_depth_to_layer_id(args, mtp_depth: int) -> int:
    """Global ``recompute_layer_ids`` index for MTP depth ``mtp_depth`` (0-based).

    Decoder layers own ``0 .. num_layers-1``; the MTP depths continue the same
    numbering, so depth 0 is ``num_layers``, depth 1 is ``num_layers + 1``, ...
    """
    return int(args.num_layers) + int(mtp_depth)


def validate_specified_recompute_layers(config, args):
    """Normalise and validate ``recompute_layer_ids`` on ``config``."""
    if config.recompute_layer_ids is None:
        return

    if isinstance(config.recompute_layer_ids, str):
        config.recompute_layer_ids = [
            int(x.strip()) for x in config.recompute_layer_ids.split(",") if x.strip()
        ]
    else:
        config.recompute_layer_ids = [int(x) for x in config.recompute_layer_ids]

    config.recompute_layer_ids = sorted(set(config.recompute_layer_ids))
    if len(config.recompute_layer_ids) == 0:
        raise ValueError("recompute_layer_ids must not be empty.")
    # The id space covers the decoder layers *and* the MTP depths appended
    # after them, so an MTP module can be checkpointed just like any other
    # layer (see ``mtp_depth_to_layer_id``).
    max_layer_id = int(args.num_layers) + _mtp_num_layers(args) - 1
    for layer_id in config.recompute_layer_ids:
        if layer_id < 0 or layer_id > max_layer_id:
            raise ValueError(
                f"recompute layer id must be between 0 and {max_layer_id} "
                f"(0..{args.num_layers - 1} are decoder layers"
                + (
                    f", {args.num_layers}..{max_layer_id} are the " f"{_mtp_num_layers(args)} MTP depths)"
                    if _mtp_num_layers(args) > 0
                    else ")"
                )
            )

    if args.recompute_granularity != "full":
        raise ValueError(
            f'When using recompute_layer_ids, recompute_granularity: {args.recompute_granularity} must be "full"'
        )

    if args.recompute_method is not None:
        raise ValueError(
            f"When using recompute_layer_ids, recompute_method: {args.recompute_method} must be None."
        )

    if args.distribute_saved_activations and args.sequence_parallel:
        raise ValueError(
            f"distribute_saved_activations: {args.distribute_saved_activations} must be "
            f"false when sequence parallel is enabled: {args.sequence_parallel}"
        )


def _make_checkpointed_forward_wrapper(original_fn):
    """Build the wrapper for ``TransformerBlock._checkpointed_forward``.

    When ``config.recompute_layer_ids is None`` the wrapper delegates to
    ``original_fn``. Otherwise it checkpoints exactly the layers whose
    *global* index appears in ``config.recompute_layer_ids``.
    """
    from megatron.core import tensor_parallel
    from megatron.core.fp4_utils import get_fp4_context
    from megatron.core.fp8_utils import get_fp8_context

    try:
        import transformer_engine.pytorch as _te  # noqa: F401
        from megatron.core.extensions.transformer_engine import te_checkpoint
    except ImportError:
        te_checkpoint = None

    def _checkpointed_forward(
        self,
        hidden_states,
        attention_mask,
        context,
        context_mask,
        rotary_pos_emb,
        attention_bias,
        packed_seq_params,
        use_inner_quantization_context,
        padding_mask=None,
        extract_layer_indices=None,
        layer_offset=0,
    ):
        recompute_layer_ids = getattr(self.config, "recompute_layer_ids", None)
        if recompute_layer_ids is None:
            return original_fn(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
                padding_mask=padding_mask,
                extract_layer_indices=extract_layer_indices,
                layer_offset=layer_offset,
            )

        if extract_layer_indices:
            raise NotImplementedError("recompute_layer_ids is incompatible with extract_layer_indices.")

        # The ``custom`` and ``checkpoint_handler`` closures below mirror the
        # closures of the same name inside Megatron's
        # ``TransformerBlock._checkpointed_forward``. Kept byte-compatible so
        # upstream fixes can be ported mechanically; see the fingerprint test.
        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask=None,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    padding_mask,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    padding_mask,
                )

        # ``layer_offset`` is already resolved by Megatron's ``forward`` and
        # passed in; reuse it so our block-local -> global mapping stays in
        # lockstep with upstream.
        recompute_ids = set(recompute_layer_ids)

        for block_layer_idx in range(self.num_layers_per_pipeline_rank):
            global_layer_idx = block_layer_idx + layer_offset
            # Skip checkpointing when either (a) this layer is not in the
            # user-requested set, or (b) we're in fp8/fp4 and the input does
            # not require grad (re-entrant autograd would have nothing to do).
            skip_recompute = global_layer_idx not in recompute_ids or (
                (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad
            )
            if skip_recompute:
                hidden_states, context = custom(block_layer_idx, block_layer_idx + 1)(
                    hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                )
            else:
                hidden_states, context = checkpoint_handler(custom(block_layer_idx, block_layer_idx + 1))

        return hidden_states

    _checkpointed_forward._primus_patched = True
    _checkpointed_forward._primus_original = original_fn
    return _checkpointed_forward


def _make_mtp_checkpointed_forward_wrapper(original_fn, args):
    """Build the wrapper for ``MultiTokenPredictionLayer._checkpointed_forward``.

    Megatron gates MTP recompute on ``recompute_granularity == 'full'`` alone
    and then dispatches on ``recompute_method``, which only knows ``'uniform'``
    and ``'block'``.  ``recompute_layer_ids`` requires ``recompute_method`` to be
    ``None``, so an MTP-enabled run would reach the dispatch's ``else`` branch
    and die with ``ValueError: Invalid activation recompute method.`` before the
    first step completes.

    With ``recompute_layer_ids`` set this wrapper takes over: MTP depth *d* is
    checkpointed iff ``num_layers + d`` is in the list, and skipped otherwise --
    the same "exactly these layers, nothing more" contract the decoder block
    already honors.
    """
    import torch
    from megatron.core import parallel_state, tensor_parallel

    def _checkpointed_forward(self, forward_func, *fwd_args, **fwd_kwargs):
        recompute_layer_ids = getattr(self.config, "recompute_layer_ids", None)
        if recompute_layer_ids is None:
            return original_fn(self, forward_func, *fwd_args, **fwd_kwargs)

        # ``layer_number`` is 1-based and already carries the pipeline layout's
        # MTP offset, so ``layer_number - 1`` is the global MTP depth.
        layer_id = mtp_depth_to_layer_id(args, int(getattr(self, "layer_number", 1)) - 1)
        if layer_id not in set(recompute_layer_ids):
            return forward_func(*fwd_args, **fwd_kwargs)

        # Pass the tensors through ``checkpoint`` so their activations are
        # dropped and recomputed, and close over everything else. Upstream
        # instead forwards ``*kwargs.values()`` positionally, which both relies
        # on the callee's parameter order matching dict insertion order and
        # feeds non-tensors to autograd; keying by name avoids both.
        tensor_keys = [k for k, v in fwd_kwargs.items() if torch.is_tensor(v)]
        tensor_values = [fwd_kwargs[k] for k in tensor_keys]
        static_kwargs = {k: v for k, v in fwd_kwargs.items() if not torch.is_tensor(v)}

        def _run(*tensors):
            merged = dict(zip(tensor_keys, tensors))
            merged.update(static_kwargs)
            return forward_func(*fwd_args, **merged)

        if self.config.fp8:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                _run,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                *tensor_values,
            )
        return tensor_parallel.checkpoint(_run, self.config.distribute_saved_activations, *tensor_values)

    _checkpointed_forward._primus_patched = True
    _checkpointed_forward._primus_original = original_fn
    return _checkpointed_forward


@register_patch(
    "megatron.transformer.custom_recompute_layer_ids",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch TransformerConfig, TransformerBlock._checkpointed_forward and "
        "MultiTokenPredictionLayer._checkpointed_forward to support Primus-provided "
        "recompute_layer_ids."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "recompute_layer_ids", None) is not None,
)
def patch_custom_recompute_layer_ids(ctx: PatchContext):
    """Install ``recompute_layer_ids`` support. Idempotent."""
    args = get_args(ctx)

    import megatron.core.transformer.multi_token_prediction as mtp_mod
    import megatron.core.transformer.transformer_block as transformer_block_mod
    import megatron.core.transformer.transformer_config as config_mod

    TransformerBlock = transformer_block_mod.TransformerBlock
    TransformerConfig = config_mod.TransformerConfig
    MultiTokenPredictionLayer = mtp_mod.MultiTokenPredictionLayer

    TransformerConfig.recompute_layer_ids = args.recompute_layer_ids

    # Wrap __post_init__ to bypass Megatron's "recompute_method must be set
    # when granularity='full'" check, then run our own validation.
    if not getattr(TransformerConfig.__post_init__, "_primus_patched", False):
        orig_post_init = TransformerConfig.__post_init__

        def new_post_init(self):
            tmp = getattr(self, "recompute_granularity", None)
            self.recompute_granularity = None
            orig_post_init(self)
            self.recompute_granularity = tmp
            validate_specified_recompute_layers(TransformerConfig, args)

        new_post_init._primus_patched = True
        new_post_init._primus_original = orig_post_init
        TransformerConfig.__post_init__ = new_post_init

    if not getattr(TransformerBlock._checkpointed_forward, "_primus_patched", False):
        original_fn = TransformerBlock._checkpointed_forward
        TransformerBlock._checkpointed_forward = _make_checkpointed_forward_wrapper(original_fn)
        log_rank_0(
            "[Patch:megatron.transformer.recompute_layer_ids] wrapped "
            "TransformerBlock._checkpointed_forward (delegates to upstream "
            "unless recompute_layer_ids is set)."
        )

    if not getattr(MultiTokenPredictionLayer._checkpointed_forward, "_primus_patched", False):
        mtp_original_fn = MultiTokenPredictionLayer._checkpointed_forward
        MultiTokenPredictionLayer._checkpointed_forward = _make_mtp_checkpointed_forward_wrapper(
            mtp_original_fn, args
        )
        log_rank_0(
            "[Patch:megatron.transformer.recompute_layer_ids] wrapped "
            "MultiTokenPredictionLayer._checkpointed_forward (MTP depth d maps to "
            f"layer id {mtp_depth_to_layer_id(args, 0)}+d)."
        )
