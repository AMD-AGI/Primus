
## How to run GPT-OSS model pretraining
```bash
export DOCKER_IMAGE=rocm/pyt-megatron-lm-jax-nightly-private:pytorch_gfx950_c9a8526_rocm_7.0.0.70000-3822.04_py_3.10.12_torch_2.9.0.dev20250821rocm7.0.0.lw.git125803b7_hblt_af95a726d6_te_2.
export HF_TOKEN="yourhftoken"
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
EXP=examples/megatron/configs/gpt_oss_20B-pretrain.yaml bash ./examples/run_local_pretrain.sh
``` 