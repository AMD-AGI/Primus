###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
import os
import re
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


def log_and_exit(message):
    logger.error(message)
    raise Exception("ABORT")


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Parse Primus training logs and generate checkpoint report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--primus-log-dir",
        type=str,
        required=True,
        help=("Directory containing Primus training log folders. " "The general structure is as follows: " """
            .
            ├── rank-0
            │ ├── debug.log
            │ ├── error.log
            │ ├── info.log
            │ └── warning.log
            ├── rank-1
            │ ├── debug.log
            │ ├── error.log
            │ ├── info.log
            │ └── warning.log
            ...
            """),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help=(
            "Optional, checkpoint directory, more metrics can be reported if given"
            "The general structure is as follows: "
            """
            .
            ├── iter_0000020
            │ ├── ...
            ├── iter_0000040
            │ ├── ...
            ├── iter_0000060
            │ ├── ...
            └── latest_checkpointed_iteration.txt
            ...
            """
        ),
    )
    args = parser.parse_args()
    return args


def remove_ansi_escape(text: str) -> str:
    ANSI_ESCAPE_PATTERN = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ANSI_ESCAPE_PATTERN.sub("", text)


def get_time_elapsed_in_sec(start_time, end_time):
    FMT = "%Y%m%d %H:%M:%S"
    start = datetime.strptime(start_time, FMT)
    end = datetime.strptime(end_time, FMT)
    return int((end - start).total_seconds())


def get_full_log(log_dir):
    PRIMUS_LOG_PATTERN = re.compile(r"\[(.*?)\].*?\.py:\d+\]:\s*(.*)")
    max_rank = max(
        [int(m.group(1)) for m in map(lambda name: re.match(r"rank-(\d+)$", name), os.listdir(log_dir)) if m]
    )
    logger.debug(f"max_rank for training log dir : {max_rank}")
    full_log = []
    # get debug && info message type from first && last rank
    for rank in set([0, max_rank]):
        for level in ["debug", "info"]:
            with open(os.path.join(log_dir, f"rank-{rank}", f"{level}.log"), "r") as f:
                logger.debug(f"get log from file {f.name} ...")
                for line in f:
                    match = re.search(PRIMUS_LOG_PATTERN, line)
                    if match:
                        full_log.append(
                            (
                                remove_ansi_escape(match.group(1).strip()),
                                remove_ansi_escape(match.group(2).strip()),
                            )
                        )

    return sorted(list(set(full_log)), key=lambda x: x[0])


def get_arguments_from_log(full_log):
    ARGUMENTS_PATTERN = re.compile(r"(\S+)\s\.{3,}\s(.+)")
    arguments = {}
    for _, line in full_log:
        match = ARGUMENTS_PATTERN.search(line)
        if match:
            arguments[match.group(1)] = match.group(2)
    return arguments


def get_statistics_from_log(full_log, arguments):
    async_save = True if arguments["async_save"] == "True" else False
    save_start_indice = [
        idx for (idx, log) in enumerate(full_log) if "saving checkpoint at iteration" in log[1]
    ]
    save_end_indice = [
        idx for (idx, log) in enumerate(full_log) if "successfully saved checkpoint from iteration" in log[1]
    ]

    if len(save_start_indice) == 0 or len(save_start_indice) != len(save_end_indice):
        log_and_exit("check save indice failed")
    save_total_time = get_time_elapsed_in_sec(
        full_log[save_start_indice[-1]][0], full_log[save_end_indice[-1]][0]
    )
    statistics = {
        "save_block_time": -1,
        "save_total_time": save_total_time,
        "accurate": True,
        "num_saved": len(save_start_indice),
    }
    if not async_save:
        statistics["save_block_time"] = save_total_time
    else:
        TARGET_STR = "Starting a checkpoint save before previous has finished"
        if any(TARGET_STR in log[1] for log in full_log):
            logger.warning(
                "The model is being saved too frequently, "
                "which may affect the accuracy of the benchmark_report data, "
                "consider increasing the checkpoint interval."
            )
            statistics["accurate"] = False
        async_save_start_indice = [
            idx for (idx, log) in enumerate(full_log) if "scheduled an async checkpoint save" in log[1]
        ]
        if len(async_save_start_indice) != len(save_start_indice):
            logger.error("check async indice failed")
        else:
            statistics["save_block_time"] = get_time_elapsed_in_sec(
                full_log[save_start_indice[-1]][0], full_log[async_save_start_indice[-1]][0]
            )

    load_start_index = [idx for (idx, log) in enumerate(full_log) if "loading checkpoint from" in log[1]]
    load_end_index = [
        idx for (idx, log) in enumerate(full_log) if "successfully loaded checkpoint from" in log[1]
    ]
    if len(load_start_index) == 0 or len(load_start_index) != len(load_end_index):
        logger.error("detect loading time from log failed")
    else:
        statistics["load_time"] = get_time_elapsed_in_sec(
            full_log[load_start_index[-1]][0], full_log[load_end_index[-1]][0]
        )
    return statistics


def get_iter_fold_sizes(ckpt_dir) -> int:
    iter_folder_sizes = []
    try:
        for name in os.listdir(ckpt_dir):
            if not name.startswith("iter_"):
                continue
            path = os.path.join(ckpt_dir, name)
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, files in os.walk(path)
                for f in files
            )
            iter_folder_sizes.append(total_size)
    except Exception as e:
        logger.error("fail to get checkpoint directory size")
        error_msg = traceback.format_exc()
        logger.error(f"{e} {error_msg}")
    finally:
        return iter_folder_sizes


def get_ckpt_report(log_dir, ckpt_dir):
    full_log = get_full_log(log_dir)
    arguments = get_arguments_from_log(full_log)
    statistics = get_statistics_from_log(full_log, arguments)
    iter_folder_sizes = get_iter_fold_sizes(ckpt_dir)
    logger.debug(f"iter_folder_sizes : {iter_folder_sizes}")
    ckpt_keys = [
        "world_size",
        "data_parallel_size",
        "ckpt_format",
        "ckpt_fully_parallel_save",
        "ckpt_fully_parallel_load",
        "async_save",
        "save",
        "save_interval",
        "optimizer",
        "use_distributed_optimizer",
        "params_dtype",
        "main_params_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
    ]
    report = {k: arguments[k] for k in ckpt_keys if k in ckpt_keys}
    report.update(statistics)
    if len(iter_folder_sizes) == statistics["num_saved"]:
        report["iter_folder_size"] = iter_folder_sizes[-1]
        report["save_bandwidth_in_mbps"] = iter_folder_sizes[-1] / (2**20) / report["save_total_time"]

    if "load_time" in report:
        report["load_bandwidth_in_mbps"] = iter_folder_sizes[-1] / (2**20) / report["load_time"]
    return report


def main(args):
    logger.debug(args)
    ckpt_report = get_ckpt_report(args.primus_log_dir, args.ckpt_dir)
    logger.info(ckpt_report)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    args = parse_cli_args()
    main(args)
