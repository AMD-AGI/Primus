###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import os
import subprocess
import sys

# { unit_test_path: nproc_per_node }
DISTRIBUTED_UNIT_TESTS = {}

UNIT_TEST_PASS = True


def get_all_unit_tests():
    global DISTRIBUTED_UNIT_TESTS

    cur_dir = "./tests"
    unit_tests = {}

    for root, dirs, files in os.walk(cur_dir):
        for file_name in files:
            if not file_name.endswith(".py") or not file_name.startswith("test_"):
                continue

            if file_name not in DISTRIBUTED_UNIT_TESTS:
                unit_tests[os.path.join(root, file_name)] = 1
            else:
                unit_tests[os.path.join(root, file_name)] = DISTRIBUTED_UNIT_TESTS[file_name]

    return unit_tests


def launch_unit_test(ut_path, nproc_per_node):
    global UNIT_TEST_PASS

    if nproc_per_node == 1:
        cmd = f"pytest {ut_path} -s"
    else:
        cmd = f"torchrun --nnodes 1 --nproc-per-node {nproc_per_node} {ut_path}"

    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

    if proc.returncode != 0:
        UNIT_TEST_PASS = False


def main():
    unit_tests = get_all_unit_tests()

    for ut_path, gpus in unit_tests.items():
        launch_unit_test(ut_path, gpus)

    if not UNIT_TEST_PASS:
        print("Unit Tests failed! More details please check log.")
        sys.exit(1)
    else:
        print("Unit Tests pass!")


if __name__ == "__main__":
    main()
