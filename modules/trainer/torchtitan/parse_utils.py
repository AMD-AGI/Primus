###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from argparse import Namespace
from typing import List


def _parse_args(ignore_unknown_args: bool = False) -> tuple[Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Primus Arguments", allow_abbrev=False)
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Primus experiment yaml config file.",
    )
    return parser.parse_known_args() if ignore_unknown_args else (parser.parse_args(), [])


def _merge_args(args: Namespace, unknown_args: List[str]) -> Namespace:
    merged_dict = vars(args).copy()
    temp_parser = argparse.ArgumentParser()

    i = 0
    while i < len(unknown_args):
        key = unknown_args[i]
        if key.startswith("--"):
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                temp_parser.add_argument(key, type=str)
                i += 2
            else:
                temp_parser.add_argument(key, action="store_true")
                i += 1
        else:
            i += 1

    parsed_unknown, _ = temp_parser.parse_known_args(unknown_args)
    merged_dict.update(vars(parsed_unknown))
    return Namespace(**merged_dict)


def _convert_args_to_cli(args: Namespace) -> List[str]:
    cli_args = []

    if hasattr(args, "exp"):
        cli_args += ["--job.config_file", args.exp]

    for k, v in vars(args).items():
        if k in {"exp", "backend"}:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        else:
            cli_args += [f"--{k}", str(v)]

    return cli_args


def get_torchtitan_config_args() -> List[str]:
    args, unknown_args = _parse_args(ignore_unknown_args=True)
    merged = _merge_args(args, unknown_args)
    return _convert_args_to_cli(merged)
