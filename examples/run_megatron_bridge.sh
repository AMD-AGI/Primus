#!/bin/bash
# Example script for running Megatron-Bridge post-training with Primus
# Megatron-Bridge is specialized for post-training tasks (SFT, instruction tuning, LoRA)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Example 1: Supervised Fine-Tuning (SFT) from Megatron checkpoint
example_sft_from_checkpoint() {
    echo "=== Example 1: Supervised Fine-Tuning from Megatron Checkpoint ==="

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/llama_sft_posttrain.yaml \
        --load /path/to/pretrained/checkpoint \
        --data_path /path/to/instruction/data
}

# Example 2: SFT from HuggingFace model
example_sft_from_huggingface() {
    echo "=== Example 2: SFT from HuggingFace Model ==="

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --convert_from_hf \
        --hf_model_name_or_path meta-llama/Llama-3-8B \
        --data_path /path/to/instruction/data \
        --micro_batch_size 1 \
        --global_batch_size 128 \
        --lr 5e-6 \
        --train_iters 5000
}

# Example 3: LoRA Fine-tuning
example_lora_finetuning() {
    echo "=== Example 3: LoRA Parameter-Efficient Fine-tuning ==="

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/llama_lora_posttrain.yaml \
        --use_lora \
        --lora_rank 16 \
        --lora_alpha 32
}

# Example 4: Instruction tuning with custom parallelism
example_instruction_tuning() {
    echo "=== Example 4: Instruction Tuning with Custom Parallelism ==="

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/llama_sft_posttrain.yaml \
        --tensor_model_parallel_size 2 \
        --pipeline_model_parallel_size 1 \
        --data_path /path/to/instruction/data \
        --dataset_format alpaca
}

# Example 5: SFT with prompt template and chat format
example_chat_finetuning() {
    echo "=== Example 5: Chat Fine-tuning with Prompt Template ==="

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/llama_sft_posttrain.yaml \
        --prompt_template chatml \
        --chat_format chatml \
        --data_path /path/to/chat/data
}

# Example 6: Export to HuggingFace format after training
example_export_to_huggingface() {
    echo "=== Example 6: Export Fine-tuned Model to HuggingFace ==="

    # This can be done during training or as a separate step
    python -m primus.cli.convert \
        --framework megatron_bridge \
        --load /path/to/finetuned/checkpoint \
        --hf_save_path /path/to/output/hf_model \
        --convert_to_hf
}

# Main menu
show_menu() {
    echo ""
    echo "Megatron-Bridge Post-Training Examples"
    echo "======================================="
    echo "Note: Megatron-Bridge is specialized for post-training (SFT, LoRA, instruction tuning)"
    echo ""
    echo "1. Supervised Fine-Tuning from Megatron checkpoint"
    echo "2. SFT from HuggingFace model"
    echo "3. LoRA Parameter-Efficient Fine-tuning"
    echo "4. Instruction tuning with custom parallelism"
    echo "5. Chat fine-tuning with prompt template"
    echo "6. Export to HuggingFace format"
    echo "q. Quit"
    echo ""
}

# Interactive mode
if [ $# -eq 0 ]; then
    while true; do
        show_menu
        read -p "Select an example to run (1-6, q to quit): " choice
        case $choice in
            1) example_sft_from_checkpoint ;;
            2) example_sft_from_huggingface ;;
            3) example_lora_finetuning ;;
            4) example_instruction_tuning ;;
            5) example_chat_finetuning ;;
            6) example_export_to_huggingface ;;
            q|Q) exit 0 ;;
            *) echo "Invalid option. Please try again." ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Command line mode
    case $1 in
        sft|sft-checkpoint) example_sft_from_checkpoint ;;
        hf|sft-hf) example_sft_from_huggingface ;;
        lora) example_lora_finetuning ;;
        instruction) example_instruction_tuning ;;
        chat) example_chat_finetuning ;;
        export|convert) example_export_to_huggingface ;;
        *)
            echo "Usage: $0 [sft|hf|lora|instruction|chat|export]"
            echo "Run without arguments for interactive mode"
            exit 1
            ;;
    esac
fi
