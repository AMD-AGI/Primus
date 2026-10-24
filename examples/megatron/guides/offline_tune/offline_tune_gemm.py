###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import copy
import csv
import os
import re
import shlex
import subprocess
import time
from multiprocessing import Process, Queue

HIPBLASLT_BENCH_BASE = (
    r"/opt/rocm/bin/hipblaslt-bench --api_method (?P<API_METHOD>\w+) "
    r"-m (?P<M>[\d ]+)"
    r"-n (?P<N>[\d ]+)"
    r"-k (?P<K>[\d ]+)"
    r"--lda (?P<LDA>[\d ]+)"
    r"--ldb (?P<LDB>[\d ]+)"
    r"--ldc (?P<LDC>[\d ]+)"
    r"--ldd (?P<LDD>[\d ]+)"
    r"--stride_a (?P<STRIDE_A>[\d ]+)"
    r"--stride_b (?P<STRIDE_B>[\d ]+)"
    r"--stride_c (?P<STRIDE_C>[\d ]+)"
    r"--stride_d (?P<STRIDE_D>[\d ]+)"
    r"--alpha (?P<ALPHA>[\d\. ]+)"
    r"--beta (?P<BETA>[\d\. ]+)"
    r"--transA (?P<TRANS_A>[\w ]+)"
    r"--transB (?P<TRANS_B>[\w ]+)"
    r"--batch_count (?P<BATCH_COUNT>[\d ]+)"
    r"--scaleA (?P<SCALE_A>[\d ]+)"
    r"--scaleB (?P<SCALE_B>[\d ]+)"
)

# Optional patterns for scale and bias
BIAS_PATTERN = r"--bias_vector --bias_source (?P<BIAS_SOURCE>[\w ]+)"

# Common ending pattern
TYPE_PATTERN = (
    r"--a_type (?P<A_TYPE>[\w ]+)"
    r"--b_type (?P<B_TYPE>[\w ]+)"
    r"--c_type (?P<C_TYPE>[\w ]+)"
    r"--d_type (?P<D_TYPE>[\w ]+)"
    r"--scale_type (?P<SCALE_TYPE>[\w ]+)"
    r"--bias_type (?P<BIAS_TYPE>[\w ]+)"
    r"--compute_type (?P<COMPUTE_TYPE>[\w ]+)"
    r"--activation_type (?P<ACTIVATION_TYPE>[\w ]+)"
)

KERNEL_NAME_PATTERN = r"--kernel name:\s+(?P<KERNEL_NAME>[A-Za-z0-9_]+)"


# Build the combined pattern with optional parts
def build_hipblaslt_command_pattern(has_bias=False):
    pattern = HIPBLASLT_BENCH_BASE
    if has_bias:
        pattern += BIAS_PATTERN
    pattern += TYPE_PATTERN
    return pattern


# Create the four variations
HIPBLASLT_BENCH_RE = build_hipblaslt_command_pattern()
HIPBLASLT_BENCH_RE_BIAS = build_hipblaslt_command_pattern(has_bias=True)


def match_hipblaslt_command_pattern(line):
    if "bias_vector" in line:
        match = re.search(HIPBLASLT_BENCH_RE_BIAS, line)
    else:
        match = re.search(HIPBLASLT_BENCH_RE, line)
    if match is None:
        print("WARNING: can't find match for", line)

    return match


def extract_problem_size(match):
    gdict = match.groupdict()

    m = int(gdict.get("M", 0).strip())
    n = int(gdict.get("N", 0).strip())
    k = int(gdict.get("K", 0).strip())
    batch_count = int(gdict.get("BATCH_COUNT", 0).strip())

    lda = int(gdict.get("LDA", 0).strip())
    ldb = int(gdict.get("LDB", 0).strip())
    ldc = int(gdict.get("LDC", 0).strip())
    ldd = int(gdict.get("LDD", 0).strip())

    alpha = float(gdict.get("ALPHA", 0).strip())
    beta = float(gdict.get("BETA", 0).strip())

    dtype_a = gdict.get("A_TYPE", "").strip()
    dtype_b = gdict.get("B_TYPE", "").strip()
    dtype_c = gdict.get("C_TYPE", "").strip()
    gdict.get("D_TYPE", "").strip()

    trans_a = gdict.get("TRANS_A", "").strip()
    trans_b = gdict.get("TRANS_B", "").strip()

    # TODO(ruibzhan): extract more options

    return {
        "m": m,
        "n": n,
        "k": k,
        "batch_count": batch_count,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "ldd": ldd,
        "alpha": alpha,
        "beta": beta,
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "trans_a": trans_a,
        "trans_b": trans_b,
    }


def is_hip():
    import torch

    if torch.version.hip is not None:
        return True
    return False


def worker(device_id, tune_gemm_results_file_path, task_queue, output_queue):
    env = os.environ.copy()
    if is_hip():
        env["HIP_VISIBLE_DEVICES"] = device_id
    else:
        env["CUDA_VISIBLE_DEVICES"] = device_id
    env["HIPBLASLT_TUNING_FILE"] = tune_gemm_results_file_path

    while True:
        script = task_queue.get()
        if script is None:
            break
        print(f"Device {device_id} processing: {script}")
        proc = subprocess.Popen(
            shlex.split(script), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        stdout, stderr = proc.communicate()

        # extract command option
        result = {}
        match = match_hipblaslt_command_pattern(script)
        if match is not None:
            result.update(extract_problem_size(match))

            if proc.returncode == 0:
                # extract outout
                output = stdout.strip().split("\n")

                tflops = float(output[-4].split(",")[-4]) / 1024  # Gflops -> Tflops
                kernel_name = re.search(KERNEL_NAME_PATTERN, output[-1]).group("KERNEL_NAME")

                result.update({"tflops": tflops, "kernel_name": kernel_name})
            else:
                result.update({"tflops": -1, "kernel_name": None})

            output_queue.put(result)


class OfflineTuneGemm:

    def __init__(self, dump_shape_path_or_file):
        self.HIPBLASLT_BENCH = "/opt/rocm/bin/hipblaslt-bench "
        self.ROTATING_BUFFER = 512
        self.RUN_NUMS = 20
        self.REQUESTED_SOLUTION = -1
        self.SKIP_LOW_SOLUTION = 0.7

        self.src_script_dict_list = []
        self.src_script_list = []
        self.tune_script_dict_list = []
        self.tune_script_list = []
        self.process_raw_dump(dump_shape_path_or_file)

    def collect_unique_lines(self, path):
        unique_lines = set()

        def process_file(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        unique_lines.add(line.strip())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if os.path.isfile(path):
            process_file(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    process_file(file_path)
        else:
            print(f"Error: {path} is neither a file nor a directory.")

        return list(unique_lines)

    def process_raw_dump(self, dump_shape_path_or_file):
        lines = self.collect_unique_lines(dump_shape_path_or_file)
        print(f"Total {len(lines)} shapes need to be tuned.", flush=True)

        for line in lines:
            line = line.strip().split(" ")
            line = [item for item in line if item.strip()]
            if line[0] == "hipblaslt-bench":
                src_script_dict = {}
                for item in line[1:]:
                    if item.startswith("--") or item.startswith("-"):
                        key = item
                    else:
                        src_script_dict[key] = item
                # src script
                src_script_dict["--rotating"] = self.ROTATING_BUFFER
                src_script_dict["--cold_iters"] = self.RUN_NUMS
                src_script_dict["--iters"] = self.RUN_NUMS
                src_script = self.HIPBLASLT_BENCH + " ".join(f"{k} {v}" for k, v in src_script_dict.items())
                self.src_script_dict_list.append(src_script_dict)
                self.src_script_list.append(src_script)
                # tune script
                tune_script_dict = copy.deepcopy(src_script_dict)
                del tune_script_dict["--algo_method"]
                del tune_script_dict["--solution_index"]
                tune_script_dict["--requested_solution"] = self.REQUESTED_SOLUTION
                tune_script_dict["--skip_slow_solution_ratio"] = self.SKIP_LOW_SOLUTION
                tune_script = self.HIPBLASLT_BENCH + " ".join(f"{k} {v}" for k, v in tune_script_dict.items())
                tune_script += " --print_kernel_info"  # enable print kernel info
                self.tune_script_dict_list.append(tune_script_dict)
                self.tune_script_list.append(tune_script)

    def tune(self, tune_gemm_results_file_path, reports_gemm_result_file_path=None, device_ids=["0"]):
        if reports_gemm_result_file_path is not None:
            assert reports_gemm_result_file_path.endswith("csv"), "--reports-result-path should be csv file."

        print(f"{tune_gemm_results_file_path=}", flush=True)
        task_queue = Queue()
        output_queue = Queue()
        for script in self.tune_script_list:
            task_queue.put(script)
        for _ in device_ids:
            task_queue.put(None)

        start_time = time.time()
        processes = []
        for device_id in device_ids:
            p = Process(
                target=worker, args=(device_id, tune_gemm_results_file_path, task_queue, output_queue)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Tune cases Nums: {len(self.tune_script_list)}. Elapsed Time: {elapsed_time:.2f} s",
        )

        if reports_gemm_result_file_path is not None:
            results = []
            while True:
                try:
                    result = output_queue.get(timeout=5)
                    results.append(result)
                except:
                    print("Queue is empty. Finish collecting results.")
                    break

            with open(reports_gemm_result_file_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-shape-path-or-file", type=str)
    parser.add_argument("--tune-result-path", type=str)
    parser.add_argument("--reports-result-path", type=str, default=None, required=False)
    parser.add_argument("--num-devices", type=int, default=1)
    args = parser.parse_args()
    device_ids = [str(i) for i in range(args.num_devices)]

    tuner = OfflineTuneGemm(args.dump_shape_path_or_file)
    tuner.tune(
        args.tune_result_path, reports_gemm_result_file_path=args.reports_result_path, device_ids=device_ids
    )
