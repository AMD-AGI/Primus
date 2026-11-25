###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Add the word-embedding gradient into ``main_grad`` for the rows the batch touches.

With ``embedding_rows_grad`` and one tensor-parallel rank, ``VocabParallelEmbedding`` computes
its weight gradient with ATen's own embedding backward over the batch's distinct ids and adds
those rows into ``main_grad`` (bit-identical to the dense ``[vocab, hidden]`` gradient + the DDP
hook's ``add_``; see ``core/fusions/embedding_rows_grad.py``), so no dense gradient is filled,
written, held or read per micro-batch. Megatron's ``grad_added_to_main_grad`` protocol tells the
DDP hook that the gradient is in the buffer already; the hook still runs, so the parameter is
registered for its bucket's reduction as before.

That protocol drops every other gradient of the parameter, so the path is taken only for
embeddings of a ``GPTModel`` built without ``share_embeddings_and_output_weights`` and without
multi-token prediction (whose layers look the embedding up again: autograd would sum those
lookups' gradients before the hook's single ``add_``, the rows would be added one by one). Tensor
parallelism, GTP rematerialisation, ``deterministic_mode``, ``reduce_scatter_embeddings``, module
hooks and every case ``embedding_rows_grad.eligible`` does not cover keep Megatron's forward;
the route may change between micro-batches of a step (the dense route clears the flag in its
backward, see ``_dense_forward``).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.embedding.rows_grad"
_UNTIED = "_primus_embedding_weight_untied"


def _has_hooks(module) -> bool:
    return bool(
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
    )


def _dense_forward(original_forward, module, input_):
    """Megatron's forward for a marked module. An earlier micro-batch of this step may have gone
    the row-wise way and left ``grad_added_to_main_grad`` set (Megatron resets it only when the
    buffer is zeroed); the DDP hook would then drop this micro-batch's dense gradient. A hook on
    the output clears the flag when this micro-batch's backward reaches the embedding, before
    the weight's gradient hook runs; it returns None, so the gradient is untouched."""
    output = original_forward(module, input_)
    weight = module.weight
    if output.requires_grad and hasattr(weight, "grad_added_to_main_grad"):

        def clear_flag(grad):
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = False
            return None

        output.register_hook(clear_flag)
    return output


def make_forward(original_forward):
    """``VocabParallelEmbedding.forward`` with the row-wise gradient where it applies."""
    logged = []
    fell_back = []

    def dense(module, input_, why):
        if why not in fell_back:
            fell_back.append(why)
            log_rank_0(f"[Patch:{_PATCH_KEY}] word embedding keeps Megatron's gradient: {why}")
        return _dense_forward(original_forward, module, input_)

    def forward(self, input_):
        # eligibility before any work; everything else is Megatron's forward
        if not getattr(self, _UNTIED, False):
            return original_forward(self, input_)
        if self.tp_group is not None and self.tp_group.size() > 1:  # None: one rank, as get_pg_size
            return dense(self, input_, "tensor parallelism")
        if getattr(self, "gtp_remat_size", 1) > 1:
            return dense(self, input_, "GTP rematerialisation")
        if self.deterministic_mode:
            return dense(self, input_, "deterministic_mode")
        if self.reduce_scatter_embeddings:
            return dense(self, input_, "reduce_scatter_embeddings")
        if _has_hooks(self):
            return dense(self, input_, "module hooks (such as overlap_param_gather's)")
        # Imported here: only a run with the switch on needs Triton.
        from primus.backends.megatron.core.fusions import embedding_rows_grad as erg

        if not erg.eligible(input_, self.weight):
            return dense(self, input_, "ids or weight the kernel does not cover")
        if not erg.checked(input_, self.weight):
            return _dense_forward(original_forward, self, input_)
        if not logged:
            logged.append(True)
            log_rank_0(
                f"[Patch:{_PATCH_KEY}] word-embedding gradient added into main_grad for the touched rows"
            )
        return erg.embedding(input_, self.weight)

    forward._primus_embedding_rows_grad = True
    return forward


def mark_untied(model) -> None:
    """Allow the path for ``model``'s word embedding when the lookup is its weight's only use: no
    output layer shares the weight and no multi-token-prediction layer looks it up again."""
    embedding = getattr(model, "embedding", None)
    word_embeddings = getattr(embedding, "word_embeddings", None)
    if word_embeddings is None or getattr(model, "share_embeddings_and_output_weights", True) is not False:
        return
    if getattr(model, "mtp_process", False) or getattr(
        getattr(model, "config", None), "mtp_num_layers", None
    ):
        return
    setattr(word_embeddings, _UNTIED, True)


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Word-embedding gradient added into main_grad for the touched rows only",
    condition=lambda ctx: bool(getattr(get_args(ctx), "embedding_rows_grad", False)),
)
def patch_embedding_rows_grad(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

    if getattr(VocabParallelEmbedding.forward, "_primus_embedding_rows_grad", False):
        return
    original_init = GPTModel.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        mark_untied(self)

    GPTModel.__init__ = init
    VocabParallelEmbedding.forward = make_forward(VocabParallelEmbedding.forward)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] enabled for untied word embeddings on one tensor-parallel rank "
        "(decided per call; checked against the dense gradient on first use)"
    )
