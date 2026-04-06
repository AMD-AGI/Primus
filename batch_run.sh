#!/bin/bash


export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN environment variable}"

configs=(
    "examples/megatron/configs/MI300X/mamba_370M-pretrain.yaml"
    "examples/megatron/configs/MI300X/zebra_llama_1B-pretrain.yaml"
    "examples/megatron/configs/MI300X/zebra_llama_3B-pretrain.yaml"
    "examples/megatron/configs/MI300X/zebra_llama_8B-pretrain.yaml"
)

results_dir="benchmark_results"
mkdir -p $results_dir

for config in "${configs[@]}"; do
    model_name=$(basename $config .yaml)
    for rep in {1..4}; do
        log_file="$results_dir/${model_name}_rep${rep}.log"
        start_time=$(date +%s)
        echo "Running ${model_name}_rep${rep} at $(date)" | tee $log_file
        echo "--------------------------------" | tee -a $log_file
        echo "Config: $config" | tee -a $log_file
        echo "--------------------------------" | tee -a $log_file
        cat $config | tee -a $log_file
        echo "--------------------------------" | tee -a $log_file
        echo "print environment variables:" | tee -a $log_file
        env | tee -a $log_file
        echo "--------------------------------" | tee -a $log_file
        echo "Running ./runner/primus-cli direct -- train pretrain --config $config 2>&1" | tee -a $log_file
        ./runner/primus-cli direct -- train pretrain --config $config 2>&1 | tee -a $log_file
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "End time: $(date)" | tee -a $log_file
        echo "Duration: ${duration}s" | tee -a $log_file
        echo "--------------------------------" | tee -a $log_file
    done
done