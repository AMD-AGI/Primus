# SPDX-License-Identifier: Apache-2.0

"""Minimal model configuration hooks retained from HY-WorldPlay."""


def is_double_block(name: str, module) -> bool:
    del module
    return "double" in name and str.isdigit(name.split(".")[-1])


def is_single_block(name: str, module) -> bool:
    del module
    return "single" in name and str.isdigit(name.split(".")[-1])


class HunyuanVideoConfig:
    _fsdp_shard_conditions = [is_double_block, is_single_block]
