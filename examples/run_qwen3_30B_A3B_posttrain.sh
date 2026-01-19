#!/bin/bash
# Quick start script for Qwen3-30B-A3B post-training with Megatron-Bridge
# This script provides easy-to-use examples for fine-tuning Qwen3-30B-A3B

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Example 1: LoRA fine-tuning from HuggingFace (Recommended for beginners)
example_lora_from_hf() {
    print_header "Example 1: LoRA Fine-tuning from HuggingFace"
    print_info "This is the easiest way to get started"
    print_info "GPU requirement: 4x A100 40GB or better"

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/qwen3_30B_A3B_lora_posttrain.yaml \
        --data_path ${DATA_PATH:-/path/to/instruction/data}
}

# Example 2: Full fine-tuning from Megatron checkpoint
example_full_finetune() {
    print_header "Example 2: Full Fine-tuning from Megatron Checkpoint"
    print_info "Best performance but requires more resources"
    print_info "GPU requirement: 8x A100 80GB"

    if [ -z "${PRETRAINED_CKPT}" ]; then
        print_error "Please set PRETRAINED_CKPT environment variable"
        print_info "Example: export PRETRAINED_CKPT=/path/to/checkpoint"
        return 1
    fi

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=8 \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --config examples/configs/megatron_bridge/qwen3_30B_A3B_posttrain.yaml \
        --load $PRETRAINED_CKPT \
        --data_path ${DATA_PATH:-/path/to/instruction/data}
}

# Example 3: Quick test with mock data
example_quick_test() {
    print_header "Example 3: Quick Test with Mock Data"
    print_info "Test the setup without real data"
    print_info "GPU requirement: 4 GPUs (any size)"

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --convert_from_hf \
        --hf_model_name_or_path Qwen/Qwen3-30B-A3B \
        --use_lora \
        --lora_rank 16 \
        --lora_alpha 32 \
        --mock_data \
        --train_iters 10 \
        --micro_batch_size 1 \
        --global_batch_size 4 \
        --seq_length 512 \
        --save /tmp/qwen3_test
}

# Example 4: Low-memory LoRA training
example_low_memory_lora() {
    print_header "Example 4: Low-Memory LoRA Training"
    print_info "Optimized for limited GPU memory"
    print_info "GPU requirement: 4x A100 40GB (or similar)"

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --convert_from_hf \
        --hf_model_name_or_path Qwen/Qwen3-30B-A3B \
        --use_lora \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --micro_batch_size 1 \
        --global_batch_size 16 \
        --seq_length 1024 \
        --lr 2e-4 \
        --train_iters 2000 \
        --expert_model_parallel_size 4 \
        --recompute_activations \
        --data_path ${DATA_PATH:-/path/to/instruction/data} \
        --save ${SAVE_PATH:-/checkpoints/qwen3_lora_low_mem}
}

# Example 5: Multi-task LoRA training
example_multitask_lora() {
    print_header "Example 5: Multi-Task LoRA Training"
    print_info "Train different LoRA adapters for different tasks"

    if [ -z "${TASK_NAME}" ]; then
        print_error "Please set TASK_NAME environment variable"
        print_info "Example: export TASK_NAME=code_generation"
        return 1
    fi

    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        -m primus.cli.train \
        --framework megatron_bridge \
        --convert_from_hf \
        --hf_model_name_or_path Qwen/Qwen3-30B-A3B \
        --use_lora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --data_path ${DATA_PATH:-/path/to/${TASK_NAME}/data} \
        --save /checkpoints/qwen3_lora_${TASK_NAME} \
        --save_lora_only
}

# Example 6: Export to HuggingFace format
example_export_to_hf() {
    print_header "Example 6: Export to HuggingFace Format"
    print_info "Convert fine-tuned model to HuggingFace format"

    if [ -z "${CHECKPOINT_PATH}" ]; then
        print_error "Please set CHECKPOINT_PATH environment variable"
        print_info "Example: export CHECKPOINT_PATH=/checkpoints/qwen3_finetuned"
        return 1
    fi

    python -m primus.cli.convert \
        --framework megatron_bridge \
        --load $CHECKPOINT_PATH \
        --convert_to_hf \
        --hf_save_path ${HF_OUTPUT_PATH:-/output/qwen3_hf}
}

# Show system info
show_system_info() {
    print_header "System Information"
    echo "Number of nodes: $NNODES"
    echo "GPUs per node: $NPROC_PER_NODE"
    echo "Total GPUs: $((NNODES * NPROC_PER_NODE))"
    echo ""

    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    fi
    echo ""
}

# Main menu
show_menu() {
    echo ""
    print_header "Qwen3-30B-A3B Post-Training Examples"
    echo "Choose an example to run:"
    echo ""
    echo "1. ${GREEN}LoRA from HuggingFace${NC} (Recommended for beginners)"
    echo "   - Easiest setup, loads model from HF"
    echo "   - Requires: 4x A100 40GB"
    echo ""
    echo "2. ${YELLOW}Full fine-tuning from checkpoint${NC}"
    echo "   - Best performance"
    echo "   - Requires: 8x A100 80GB"
    echo ""
    echo "3. ${GREEN}Quick test with mock data${NC}"
    echo "   - Test your setup"
    echo "   - Requires: 4 GPUs (any size)"
    echo ""
    echo "4. ${GREEN}Low-memory LoRA${NC}"
    echo "   - Optimized for limited resources"
    echo "   - Requires: 4x A100 40GB"
    echo ""
    echo "5. ${YELLOW}Multi-task LoRA${NC}"
    echo "   - Train task-specific adapters"
    echo "   - Requires: 4x A100 40GB"
    echo ""
    echo "6. ${YELLOW}Export to HuggingFace${NC}"
    echo "   - Convert checkpoint to HF format"
    echo ""
    echo "i. Show system information"
    echo "q. Quit"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if Megatron-Bridge is available
    if [ ! -d "$PROJECT_ROOT/third_party/Megatron-Bridge" ]; then
        print_error "Megatron-Bridge not found in third_party/"
        print_info "Please run: git submodule update --init --recursive"
        return 1
    fi

    # Check if config files exist
    if [ ! -f "$PROJECT_ROOT/examples/configs/megatron_bridge/qwen3_30B_A3B_posttrain.yaml" ]; then
        print_error "Config file not found"
        return 1
    fi

    print_info "Prerequisites check passed"
    return 0
}

# Interactive mode
if [ $# -eq 0 ]; then
    check_prerequisites || exit 1

    while true; do
        show_menu
        read -p "Select an option: " choice
        case $choice in
            1) example_lora_from_hf ;;
            2) example_full_finetune ;;
            3) example_quick_test ;;
            4) example_low_memory_lora ;;
            5) example_multitask_lora ;;
            6) example_export_to_hf ;;
            i|I) show_system_info ;;
            q|Q) exit 0 ;;
            *) print_error "Invalid option. Please try again." ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Command line mode
    check_prerequisites || exit 1

    case $1 in
        lora|lora-hf) example_lora_from_hf ;;
        full|full-ft) example_full_finetune ;;
        test|quick-test) example_quick_test ;;
        low-mem|low-memory) example_low_memory_lora ;;
        multitask|multi) example_multitask_lora ;;
        export|convert) example_export_to_hf ;;
        info) show_system_info ;;
        *)
            echo "Usage: $0 [lora|full|test|low-mem|multitask|export|info]"
            echo ""
            echo "Examples:"
            echo "  $0 lora              # LoRA fine-tuning from HuggingFace"
            echo "  $0 test              # Quick test with mock data"
            echo "  $0 low-mem           # Low-memory LoRA training"
            echo ""
            echo "Environment variables:"
            echo "  DATA_PATH           Path to training data"
            echo "  PRETRAINED_CKPT     Path to pretrained checkpoint"
            echo "  SAVE_PATH           Path to save checkpoints"
            echo "  TASK_NAME           Task name for multi-task training"
            echo "  NPROC_PER_NODE      Number of GPUs per node (default: 4)"
            echo ""
            echo "Run without arguments for interactive mode"
            exit 1
            ;;
    esac
fi
