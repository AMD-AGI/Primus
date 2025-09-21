
## How to run GPT-OSS model pretraining

```bash
export DOCKER_IMAGE=docker.io/rocm/pyt-megatron-lm-jax-nightly-private:pytorch_gfx950_20250908
export HF_TOKEN="yourhftoken"
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
EXP=examples/megatron/configs/gpt_oss_20B-pretrain.yaml bash ./examples/run_local_pretrain.sh

``` 

slurm run single node or mutlinode
```bash
bash run_gpt_pretrain.sh 
```