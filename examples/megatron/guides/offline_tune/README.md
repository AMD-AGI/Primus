# Offline Tune


## 1. GEMM Tune

Use the `hipblaslt-bench` tool to perform GEMM tuning.

`hipblaslt-bench` is usually located under `/opt/rocm/bin`. However, if it's not available in some environments/docker, you'll need to reinstall hipblaslt.


### Install Hipblaslt (Optional)
You can reference: https://github.com/ROCm/hipBLASLt?tab=readme-ov-file#build-and-install

If only run MI300X, you can use the following command for a quick compilation, reducing the compilation time to under 2 hours.
```
./install.sh -idc --logic-yaml-filter gfx942/*/* -a gfx942 -j 256 --build_dir build
```

### 1.1 Hipblaslt-bench Tune

#### Step 1: Dump Shape
* Set the Hipblaslt ENV.
* Run Train code.
* Unset ENV.
* The gemm shape will be dumped into `dump_gemm_shapes.txt`.
* Note: If just to dump shape, in most cases, there's no need to train for many iters—just a few should be enough, as each step uses the same shape.
```
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE=dump_gemm_shapes.txt

./run_your_code

unset HIPBLASLT_LOG_MASK
unset HIPBLASLT_LOG_FILE
```
#### Step 2: Tuning
Run `offline_tune_gemm.py` and save tuned results in `tune_gemm_results.txt`
```
python3 offline_tune_gemm.py                                \
    --dump-shape-path-or-file /PATH/TO/dump_gemm_shapes.txt \
    --tune-result-path /PATH/TO/tune_gemm_results.txt       \
    --reports-result-path /PATH/TO/tune_gemm_reports.csv    \
    --num-devices 8
```

#### Step 3: Use tuned results to Train
* Set the results ENV.
* Start your tasks.
```
export HIPBLASLT_TUNING_OVERRIDE_FILE=tune_gemm_results.txt
./run_your_code
```

### 1.2 Tensile Tune

Tensile is a tool for creating benchmark-driven backend libraries for GEMMs on AMDGPU. If existing GEMM kernels' performance is not satisfied you can use tensile to generate a new GEMM kernel on your problem size.

#### Step 1: Clone and build hipblaslt from source code.

Tensile need build from hipblaslt source code and it was integrated on hipBLASLt.

```bash
git clone https://github.com/ROCm/hipBLASLt.git
cd hipBLASLt/
./install.sh -idc -a $(/opt/rocm/llvm/bin/offload-arch) -j 256 --build_dir build
```

#### Step2: Generate tensile config

Use `tensile_config_generator.py` to generate tensile config file. The XCC is the number of XCD on your device (e.g MI300X's XCD is 8).

```bash
cd hipBLASLt/tensile/Tensile/Utilities
XCC=<the number of XCD> python ./tensile_config_generator.py --hipblaslt_log PATH/TO/dump_gemm_shapes.txt --tensile_config ./tuning_template.yaml --iters 100
```

#### Step3: Generate new GEMM kernel

Use Tensile to generate new GEMM kernel base tensile config. Tensile can automatically generate GEMM kernel and pick the kernel with best performance.

```bash
source build/release/virtualenv/bin/activate
HIP_FORCE_DEV_KERNARG=1 ./tensilelite/Tensile/bin/Tensile PATH/TO/generated_yaml_path PATH/TO/tune_result_directory
```

#### Step4: Merge tune results and rebuild hipBLASLt

Merge tune results base GPU's architecture.

For MI300X:
```bash
python ./tensilelite/Tensile/Utilities/merge.py --no_eff library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/ <tune result directory>/3_LibraryLogic/ library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/
```

For MI308X:
```bash
python ./tensilelite/Tensile/Utilities/merge.py --no_eff library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942_80cu/Equality/ <tune result directory>/3_LibraryLogic/ library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942_80cu/Equality/
```

Rebuild hipBLASLt with the merged results:

```bash
./install.sh -idc -a $(/opt/rocm/llvm/bin/offload-arch) -j 256 --build_dir build
```

# Reference

https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html
https://github.com/ROCm/hipBLASLt/tree/develop/tensilelite/Tensile/Utilities/tensile_generator
