###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import os
import time
from datetime import timedelta

import torch
from megatron.core.tensor_parallel import all_to_all

# [H, S(Seq length), TopK]
MODEL_PARAMS_TABLE = {
    "deepseek-v2-lite": (2048, 4096, 6),
    "deepseek-v2": (5120, 4096, 6),
    "deepseek-v3": (7168, 4096, 8),
    "mistral-8x7B": (4096, 4096, 2),
    "mistral-8x22B": (6144, 4096, 2),
    "gpt-oss-120B": (2880, 4096, 4),  # https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json
    # note: Qwen3 pretraining has two stages with different setting https://qwen.ai/blog?id=qwen3
    # "In the first stage (S1) ... context length of 4K tokens."
    # "In the final stage, .. 32K tokens."
    "Qwen3-235B-A22B-stage1": (
        4096,
        4096,
        8,
    ),  # https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
    "Qwen3-235B-A22B-stage2": (4096, 32768, 8),
    "llama4-maverick": (
        5120,
        1048576,
        1,
    ),  # claimed native 1M context length https://modelscope.cn/models/LLM-Research/Llama-4-Maverick-17B-128E-Instruct/file/view/master/config.json?status=1
}
MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


class A2ATest:
    def __init__(self):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.inited = False
        self.ep_size = self.world_size
        self.ep_group = torch.distributed.group.WORLD
        self.num_warmup = 5
        self.num_iters = 10

    def setup(self):
        if not torch.distributed.is_initialized() and self.rank >= 0:
            print(f"Initializing torch.distributed with rank: {self.rank}, world_size: {self.world_size}")
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.rank,
                timeout=timedelta(minutes=5),
            )
            torch.distributed.barrier()
            self.inited = True
        torch.manual_seed(42 + self.rank)

    def teardown(self):
        if self.inited:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            self.inited = False

    def pad_to_nearest_multiple(self, val, multiple):
        return val if val % multiple == 0 else val + (multiple - val % multiple)

    def test_a2a_performance_even(self, model_name, batch_size, dtype):
        hidden_size, seq_length, topk = MODEL_PARAMS_TABLE[model_name]
        buffer_size = self.pad_to_nearest_multiple(batch_size * seq_length * topk, self.ep_size)

        device = torch.device(f"cuda:{self.local_rank}")
        send_buf_list = [
            torch.randn((buffer_size, hidden_size), dtype=dtype, device=device) for _ in range(self.num_iters)
        ]
        recv_buf_list = [
            torch.empty((buffer_size, hidden_size), dtype=dtype, device=device) for _ in range(self.num_iters)
        ]

        output_split_sizes = [buffer_size // self.ep_size] * self.ep_size
        input_split_sizes = output_split_sizes[:]

        # warmup
        for _ in range(self.num_warmup):
            recv_buf_list[0] = all_to_all(
                self.ep_group, send_buf_list[0], output_split_sizes, input_split_sizes
            )
        torch.distributed.barrier()

        # benchmark
        start = time.time()
        for i in range(self.num_iters):
            recv_buf_list[i] = all_to_all(
                self.ep_group, send_buf_list[i], output_split_sizes, input_split_sizes
            )
        torch.distributed.barrier()
        end = time.time()

        avg_time = (end - start) / self.num_iters
        bandwidth = (
            send_buf_list[0].element_size()
            * send_buf_list[0].numel()
            * (self.ep_size - 1)
            / self.ep_size
            / avg_time
            / 1e9
        )

        if self.rank == 0:
            print(f"\n[Test All2All Performance(Even)] -- {model_name}")
            print(f"  > hidden_size={hidden_size}, seq_length={seq_length}, topk={topk}")
            print(f"  > ep_size={self.ep_size}, batch_size={batch_size}, dtype={dtype}")
            print(f"  > buffer_size={buffer_size}, output_split_sizes={output_split_sizes}")
            print(f"  > avg_time={avg_time * 1000:.3f} ms")
            print(f"  > Unidirectional bandwidth={bandwidth:.3f} GB/s")
            print(f"  > Bidirectional bandwidth={bandwidth * 2:.3f} GB/s")

        return avg_time, bandwidth


def main(output_csv_path):
    rank = int(os.environ["RANK"])

    test = A2ATest()
    test.setup()

    benchmark_results = []
    for model_name in MODEL_PARAMS_TABLE.keys():
        for batch_size in MBS_LIST:
            for dtype in [torch.bfloat16]:
                avg_time, unidirection_bandwidth = test.test_a2a_performance_even(
                    model_name, batch_size, dtype
                )
                if rank == 0:
                    result = {
                        "Model": model_name,
                        "MBS": batch_size,
                        "Seq": MODEL_PARAMS_TABLE[model_name][1],
                        "HiddenSize": MODEL_PARAMS_TABLE[model_name][0],
                        "DataType": dtype,
                        "EP-Size": int(os.environ["WORLD_SIZE"]),
                        "TopK": MODEL_PARAMS_TABLE[model_name][2],
                        "Time(s)": avg_time,
                        "Bandwidth(GB/s)": unidirection_bandwidth,
                    }
                    benchmark_results.append(result)

    test.teardown()

    if rank == 0:
        fieldnames = list(benchmark_results[0].keys())
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in benchmark_results:
                writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-csv-path", type=str)
    args = parser.parse_args()

    main(args.report_csv_path)
