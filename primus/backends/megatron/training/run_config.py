###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect


def uses_cfg_container_api(pretrain_fn) -> bool:
    """Megatron-core 0.19+: ``pretrain(cfg_container, datasets_provider, model_type, ...)``."""
    try:
        params = list(inspect.signature(pretrain_fn).parameters)
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0] == "cfg_container"


def build_run_config():
    """Parse args and build the run config, as the upstream entry scripts do.

    The container's ``model`` stays None: otherwise Megatron builds the model from it and
    ignores the Primus ``model_provider``.
    """
    import megatron.training.argument_utils as argument_utils
    import megatron.training.arguments as megatron_args
    from megatron.training import global_vars

    args = megatron_args.parse_and_validate_args(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    run_config = argument_utils.pretrain_cfg_container_from_args(args, model_cfg=None)
    # Megatron-LM main moved these out of pretrain(); core 0.19 has no such helpers.
    if hasattr(global_vars, "set_run_config"):
        global_vars.set_run_config(run_config)
    if hasattr(global_vars, "initialize_runtime_services"):
        global_vars.initialize_runtime_services(args, training=True)
    if hasattr(argument_utils, "resolve_tokenizer_vocab_size"):
        argument_utils.resolve_tokenizer_vocab_size(run_config, args.padded_vocab_size)
    return run_config
