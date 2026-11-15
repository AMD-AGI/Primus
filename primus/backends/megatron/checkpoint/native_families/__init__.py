###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registrations of the native converter's model families, one module per family.

``native_registry.load_family_modules`` imports every module in this package
before the conversion hook looks a model up, so adding a family is adding a file
here (``<family>.py``) that calls ``register_native_family(NativeFamily(...))``;
no shared file is edited. See ``native_registry`` for the fields and an example.

A module here imports ``native_registry`` only: it names the converter and its
hooks as strings, imported only when that family converts. The built-in
families (Qwen3, DeepSeek, Llama) are registered in ``native_registry`` itself.
"""
