###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling language model (text backbone of thinkingmachines/Inkling)."""

from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.module import MegatronModule

from primus.backends.megatron.core.models.inkling.inkling_layers import (
    InklingDecoderLayer,
    InklingRMSNorm,
)
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)


class InklingEmbedding(MegatronModule):
    """Vocab-parallel word embedding followed by an RMSNorm (HF InklingNormedEmbedding)."""

    def __init__(self, config, vocab_size):
        super().__init__(config=config)
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, config.hidden_size, init_method=config.init_method, config=config)
        self.embed_norm = InklingRMSNorm(config, config.hidden_size)

    def forward(self, input_ids):
        x = self.word_embeddings(input_ids)  # [B, T, H]
        x = x.transpose(0, 1).contiguous()  # [T, B, H]
        return self.embed_norm(x)


class InklingDecoder(MegatronModule):
    def __init__(self, config):
        super().__init__(config=config)
        self.layers = torch.nn.ModuleList([InklingDecoderLayer(config, i) for i in range(config.num_layers)])
        self.final_layernorm = InklingRMSNorm(config, config.hidden_size)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_layernorm(hidden_states)


class InklingModel(LanguageModule):
    """Embedding -> Inkling decoder layers -> final norm -> / mup divisor -> output layer.

    Pipeline parallelism is not supported (single stage)."""

    def __init__(self, config: InklingTransformerConfig, vocab_size: int, max_sequence_length: int,
                 pre_process: bool = True, post_process: bool = True, fp16_lm_cross_entropy: bool = False,
                 parallel_output: bool = True, share_embeddings_and_output_weights: bool = False,
                 position_embedding_type: str = "none", pg_collection=None, vp_stage: Optional[int] = None):
        super().__init__(config=config, pg_collection=pg_collection)
        assert pre_process and post_process, "Inkling: pipeline parallelism is not supported"
        assert not share_embeddings_and_output_weights, "Inkling keeps embedding and output layer untied"
        self.vp_stage = vp_stage
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = False
        self.position_embedding_type = position_embedding_type  # unused: Inkling has no RoPE
        self.embedding = InklingEmbedding(config, vocab_size)
        self.decoder = InklingDecoder(config)
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size, vocab_size, config=config, init_method=config.init_method, bias=False,
            gather_output=not parallel_output)
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None, loss_mask=None,
                packed_seq_params=None, **kwargs):
        assert packed_seq_params is None, "Inkling: packed sequences are not supported"
        hidden = self.embedding(input_ids)
        hidden = self.decoder(hidden)
        hidden = hidden / self.config.inkling_logits_mup_divisor
        logits, _ = self.output_layer(hidden)  # [T, B, V(/tp)]
        unpadded = self.config.inkling_unpadded_vocab_size
        if unpadded is not None and unpadded < self.vocab_size:
            per_rank = logits.shape[-1]
            start = per_rank * (torch.distributed.get_rank(self.pg_collection.tp) if self.parallel_output else 0)
            ids = torch.arange(start, start + per_rank, device=logits.device)
            logits = logits.masked_fill(ids >= unpadded, torch.finfo(logits.dtype).min)
        if labels is None:
            return logits.transpose(0, 1).contiguous()  # [B, T, V]
        return self.compute_language_model_loss(labels, logits)
