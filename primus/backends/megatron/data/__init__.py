# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Megatron data loading infrastructure.

This module provides:
    - DatasetProvider abstraction for pluggable data pipelines
    - EnergonDatasetProvider for multimodal/diffusion (Energon)
    - SyntheticDatasetProvider for mock/synthetic data
    - MegatronDataloaderWrapper for generic dataloader compatibility
    - Synthetic datasets for testing and development
    - Diffusion-specific task encoders and preprocessing

Architecture:
    The strategy pattern allows different trainers to use different
    data sources while sharing the same training infrastructure.
"""

_LAZY_IMPORTS = {
    "DatasetProvider": ".dataset_provider",
    "EnergonDatasetProvider": ".energon_dataset_provider",
    "SyntheticDatasetProvider": ".synthetic_dataset_provider",
    "MegatronDataloaderWrapper": ".dataloader",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
