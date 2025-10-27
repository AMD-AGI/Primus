echo "Running Zebra-Llama in ${PWD}"
cd third_party/AMD-Hybrid-Models/Zebra-Llama
bash install.sh FLASH_ATTN=1 MAMBA=1
pip show huggingface-hub
pip show transformers
pip show deepspeed
pip show accelerate
pip show peft
pip show trl
pip show triton
pip show numpy
pip show datasets
pip show wandb
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_8MLA8M2_8bt_SFT.yaml 