# Large Model Checkpoint Benchmark
## 1. Overview
This directory provides a benchmark tool for checkpoint loading/saving when training large language models with Primus(megatron-lm backend).

It requires the user to specify a Primus YAML config file.

Since the performance of checkpoint loading/saving is related to the number of processes, data_parallel_size, and other parallel settings,
it also supports overwriting some of the parallel configurations in the YAML file during benchmarking.

In the current version, the final tool outputs the following checkpoint metrics.
```
{
    "world_size": "8",
    "data_parallel_size": "8",
    "ckpt_format": "torch",
    "ckpt_fully_parallel_save": "True",
    "ckpt_fully_parallel_load": "False",
    "async_save": "None",
    "save": "/apps/tas/limou/source/Primus/output/amd/root/exp-llama2_7B-pretrain/checkpoints",
    "save_interval": "10",
    "optimizer": "adam",
    "use_distributed_optimizer": "True",
    "params_dtype": "torch.bfloat16",
    "main_params_dtype": "torch.float32",
    "exp_avg_dtype": "torch.float32",
    "exp_avg_sq_dtype": "torch.float32",
    "save_block_time": 758, # time in seconds, which blocks main training process. (used when async_save = True)
    "save_total_time": 758,
    "accurate": true, # if the save_interval is too small, multiple checkpoints may be saved concurrently, which can lead to inaccurate results.
    "num_saved": 1,
    "load_time": 174,
    "iter_folder_size": 94338155448, # 87.86 GB
    "save_bandwidth_in_mbps": 118.69112916609228, # MB/s
    "load_bandwidth_in_mbps": 517.0567580913676 # MB/s
}
```

## 2. How to Run

The entry file is ckpt_launch.py, of course you can also run ckpt_report.py separately if needed.

example:
```
export DATA_PATH=/PATH/TO/DATA
python3 benchmark/megatron/checkpoint/ckpt_launch.py \
    --yaml-config-path examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-pretrain.yaml \
    --nnodes 1
```
If you need to benchmark multiple different models, parallel strategies, and checkpoint modes,
you can add a simple wrapper script around the tool to call it multiple times for statistics.

Due to permission issues, it is recommended to truncate or delete the leftover output directory at each startup.
