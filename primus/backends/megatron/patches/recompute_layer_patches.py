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
"""

from contextlib import nullcontext

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


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
    for layer_id in config.recompute_layer_ids:
        if layer_id < 0 or layer_id >= args.num_layers:
            raise ValueError(f"recompute layer id must be between 0 and {args.num_layers - 1}")

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


@register_patch(
    "megatron.transformer.custom_recompute_layer_ids",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch TransformerConfig and TransformerBlock._checkpointed_forward "
        "to support Primus-provided recompute_layer_ids."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "recompute_layer_ids", None) is not None,
)
def patch_custom_recompute_layer_ids(ctx: PatchContext):
    """Install ``recompute_layer_ids`` support. Idempotent."""
    args = get_args(ctx)

    import megatron.core.transformer.transformer_block as transformer_block_mod
    import megatron.core.transformer.transformer_config as config_mod

    TransformerBlock = transformer_block_mod.TransformerBlock
    TransformerConfig = config_mod.TransformerConfig

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
