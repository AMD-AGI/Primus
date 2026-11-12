###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_kv_rank_0, log_rank_0, warning_rank_0

DEFAULT_NULL_TOKENIZER_VOCAB_SIZE = 131072


def _resolve_model_vocab_size(tokenizer_model):
    """Vocab size from the HF config of ``tokenizer_model``, or None if it cannot be read."""
    if not tokenizer_model:
        return None
    try:
        if os.path.isdir(tokenizer_model):
            config_path = os.path.join(tokenizer_model, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(tokenizer_model, "config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        warning_rank_0(f"[Patch:megatron.args.mock_data] Could not read HF config of {tokenizer_model}: {e}")
        return None
    return config.get("vocab_size") or config.get("text_config", {}).get("vocab_size")


def _null_tokenizer_vocab_includes_eod():
    """Whether Megatron's NullTokenizer counts the EOD token inside ``vocab_size``.

    Older Megatron builds ``vocab_size + 1`` ids with EOD last; Megatron-core 0.19+ put
    EOD at ``vocab_size - 1``.
    """
    import inspect

    try:
        from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer
    except ImportError:
        return False
    return "eod_id" in inspect.signature(NullTokenizer.__init__).parameters


@register_patch(
    "megatron.args.mock_data",
    backend="megatron",
    phase="build_args",
    description="Disable data paths when using mock data",
)
def patch_mock_data(ctx: PatchContext):
    """
    Handle mock data mode.

    When mock_data=True, sets all data paths to None to prevent
    Megatron from trying to load actual data files.
    """
    args = ctx.extra.get("backend_args", {})
    module_config = ctx.extra.get("module_config", {})

    if not args or not module_config:
        return

    mock_data = getattr(module_config.params, "mock_data", False)
    if mock_data:
        args.data_path = None
        args.train_data_path = None
        args.valid_data_path = None
        args.test_data_path = None
        args.tokenizer_type = "NullTokenizer"

        # vocab_size below follows older Megatron's NullTokenizer: EOD comes on top of it.
        existing_vocab_size = getattr(args, "vocab_size", None)
        if existing_vocab_size is None:
            tokenizer_model = getattr(args, "tokenizer_model", None)
            model_vocab_size = _resolve_model_vocab_size(tokenizer_model)
            if model_vocab_size is not None:
                args.vocab_size = model_vocab_size - 1
                vocab_size_log = f"vocab_size={model_vocab_size} from {tokenizer_model} config"
            else:
                args.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
                vocab_size_log = f"defaulted vocab_size to {args.vocab_size}"
                warning_rank_0(
                    "[Patch:megatron.args.mock_data] Embedding/output layer sizes will not match the "
                    f"real model; set vocab_size explicitly (tokenizer_model={tokenizer_model})."
                )
        else:
            vocab_size_log = f"preserved vocab_size={existing_vocab_size}"
        # Newer NullTokenizers count EOD inside vocab_size; add it back so every version
        # builds the same ids, EOD id and padded embedding.
        if _null_tokenizer_vocab_includes_eod():
            args.vocab_size += 1
            vocab_size_log += " (+1 EOD for the newer NullTokenizer)"

        log_rank_0(
            "[Patch:megatron.args.mock_data] Mock data enabled; "
            "all data paths set to None, tokenizer_type forced to NullTokenizer, "
            f"{vocab_size_log}"
        )

    log_kv_rank_0(f"[Patch:megatron.args.mock_data] -mock_data", f"{mock_data}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -data_path", f"{args.data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -train_data_path", f"{args.train_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -valid_data_path", f"{args.valid_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -test_data_path", f"{args.test_data_path}")
    log_kv_rank_0(
        f"[Patch:megatron.args.mock_data]   -tokenizer_type",
        f"{getattr(args, 'tokenizer_type', None)}",
    )
    log_kv_rank_0(
        f"[Patch:megatron.args.mock_data]   -vocab_size",
        f"{getattr(args, 'vocab_size', None)}",
    )
