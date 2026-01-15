# Large Model Training Operator Benchmark

This benchmark focuses on evaluating the performance of key operators in large model training scenarios, including GEMM, Attention, and communication-related operators.


## How to run

### GEMM
GEMM benchmarks are now provided via the **Primus CLI benchmark suites** (see `docs/benchmark.md`):

```bash
# Single-shape GEMM
primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --dtype bf16 --duration 10

# Model-shape GEMM (dense / DeepSeek)
primus-cli direct -- benchmark gemm-dense --model Llama3.1_8B --dtype bf16
primus-cli direct -- benchmark gemm-deepseek --model Deepseek_V3 --dtype bf16
```

### Attention
Run the following command to generate benchmark data for Attention operators:
```
python3 benchmark_attention.py                         \
    --shapes-json-path /PATH/TO/attention_shapes.json  \
    --report-csv-path  /PATH/TO/attention_benchmark.csv
```


### RCCL
This benchmark evaluates the performance of commonly used communication primitives in large model training, including AllReduce, AllGather, ReduceScatter, Point-to-Point (P2P), and All2All operations.

To run it, simply configure the IP and PORT, then execute the script. Benchmark results will be automatically generated as multiple CSV files.
```
bash run_script.sh
```

#### FSDP
To run the fsdp profiling, you can use the following commands:
```
torchrun --master_addr "0.0.0.0"   \
        --master_port "1245"    \
        --nnodes=1                      \
        --node_rank=0                   \
        --nproc_per_node=8              \
    ./benchmark_fsdp.py            \
        --allgather-report-csv-path ./allgather_benchmark.csv \
        --reducescatter-report-csv-path ./reducescatter_benchmark.csv
```
To generate the rccl-test command without doing the real pytorch profile, for example
```
WORLD_SIZE=4 LOCAL_RANK=0 RANK=0 python ./benchmark_fsdp.py  -dry
```
The above python script will generate command for rccl test.

To launch single node rccl test, do the following:
```
# build and install rccl tests
git clone https://github.com/ROCm/rccl-tests.git
cd rccl-tests
make -j

# For multinode rccl-test, MPI is needed, please adjust the commands based on your env.
apt install -y openmpi-bin openmpi-common libopenmpi-dev
# We unset MPI HOME for building, as some docker env made the default env wrong
make clean && unset MPI_HOME &&  make MPI=1 -j

# example commands
HSA_NO_SCRATCH_RECLAIM=1 ./build/all_gather_perf -b  213909504  -e  213909504  -g 8 -d half
HSA_NO_SCRATCH_RECLAIM=1 ./build/reduce_scatter_perf -b  213909504  -e  213909504  -g 8 -d half
HSA_NO_SCRATCH_RECLAIM=1 ./build/reduce_scatter_perf -b  213909504  -e  213909504  -g 4 -d half

HSA_NO_SCRATCH_RECLAIM=1 mpirun --allow-run-as-root -np 4 --bind-to numa ./build/reduce_scatter_perf -b  213909504  -e  213909504  -g 1 -d half
```
To profile RCCL comm op in real workload, you can use the following setting:
```
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=COLL
```
Example output processing
```
$cat output/log_torchrun_pretrain_llama3_70B-pretrain.txt | grep opCount | grep att

```

Example trace for llama3 70B
```
cv350-zts-gtu-f15-18:86092:86620 [4] NCCL INFO ReduceScatter: opCount ce3 sendbuff 0x7f7a78400000 recvbuff 0x7f8429000000 acc (nil) count 106956800 datatype 9 op 4 root 0 c

cv350-zts-gtu-f15-18:86088:86640 [0] NCCL INFO AllGather: opCount c97 sendbuff 0x7ebe75600000 recvbuff 0x7ebe75600000 acc (nil) count 106956800 datatype 9 op 0 root 0 comm
```
##### Approximate model
The benchmark_fsdp.py is meant to be an approximate model, instead of an accurate model. For example, the real trace for llama3 70B shows element counts of 106956800 and the script shows 106954752, as 213909504/2. The difference is 0.002%.
