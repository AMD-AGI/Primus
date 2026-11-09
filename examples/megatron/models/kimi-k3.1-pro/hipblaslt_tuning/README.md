# hipBLASLt tuning for the Kimi K3.1 Pro 4-layer proxy on MI455X

`kimi_k3.1_pro_4L-gfx1250-hipblaslt_01f6f9c1-rocm_10.2.0a20261013.txt` holds one tuned solution for each
of the 62 GEMM shapes of `examples/megatron/configs/MI455X/kimi_k3.1_pro_4L-BF16-pretrain.yaml`
(seq 4096, micro-batch 2, EP 4, BF16). The recipe's `env:` block points `HIPBLASLT_TUNING_OVERRIDE_FILE`
at it.

The file name gives the build it was tuned with: `hipblaslt-bench` of TheRock 10.2.0a20261013
(rocm-libraries `01f6f9c1`), the `Git Version` line of the file. Without it, hipBLASLt's heuristic
picks slower kernels for these shapes.

On another hipBLASLt build, the file's `Git Version` line does not match, and hipBLASLt uses each row
only while its kernel name still names a kernel of that build; the other shapes fall back to the
default. All 29 kernels of this file are in the hipBLASLt of `amdprimus/amdprimus:tas-20261019`
(rocm-libraries `e1e7fdcab9`, ROCm 10.2.0a20261011), where the file was measured, and of
`amdprimus/amdprimus:tas-20261109`, which has the same hipBLASLt. Tune again, and give
the new file the new build in its name, when the hipBLASLt build or the GEMM shapes change.

To remake it (`examples/megatron/guides/offline_tune/README.md`):

1. Log the GEMMs of a few training steps with `HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=<shapes>`.
2. Tune them with `examples/megatron/guides/offline_tune/offline_tune_gemm.py`, one worker per GPU.
   This file used `SKIP_LOW_SOLUTION = 0.0` and `RUN_NUMS = 50` (the script's defaults are 0.7 and 20),
   so `hipblaslt-bench` timed every solution with 50 cold and 50 timed iterations.
