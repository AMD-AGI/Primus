###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Load Hugging Face encoders without the retired safetensors conversion space."""

from __future__ import annotations

import sys


def disable_safetensors_auto_conversion() -> None:
    """Stop Transformers from POSTing to ``safetensors-convert.hf.space``.

    Repos such as ``google/t5-v1_1-xxl`` publish ``pytorch_model.bin`` and no
    ``model.safetensors``. Transformers then starts ``Thread-auto_conversion``,
    the space answers HTTP 405, and the thread raises ``KeyError: 'event_id'``.
    The published PyTorch weights already load, so the conversion is unused.
    """
    try:
        import transformers.safetensors_conversion as conversion
    except Exception:
        return
    if getattr(conversion, "_primus_conversion_disabled", False):
        return

    original = conversion.auto_conversion

    def auto_conversion(*args, **kwargs):
        del args, kwargs
        # modeling_utils unpacks ``resolved_archive_file, revision, is_sharded``.
        # ``None`` raises ``TypeError: cannot unpack non-iterable NoneType``.
        return None, None, False

    conversion.auto_conversion = auto_conversion
    conversion._primus_conversion_disabled = True
    # from_pretrained may already have bound the original function.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "auto_conversion", None) is original:
            module.auto_conversion = auto_conversion
