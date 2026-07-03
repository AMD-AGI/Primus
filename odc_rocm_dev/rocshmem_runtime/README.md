# rocshmem_runtime — 从源码构建的 rocSHMEM 后端运行时

本目录只提交**源码**（`.cpp` 绑定 + 构建脚本），运行所需的 `.so` / `.a` **不入库**，
首次使用前用 `build_rocshmem_backend.sh` 在基础镜像里从源码编出。这样**仅靠项目 +
基础镜像**（`tasimage/primus-odc:pr-722` 或 `tasimage/primus-odc:v26.2`，两者都是
ABI=1 / ROCm 7.2.0 基础）即可重建并运行 rocSHMEM 后端，无需依赖原开发容器
`odc_rocshmem_test` 的 `/root` 层，也无需在 git 里放任何二进制。

## 首次使用（必读）：从源码构建后端

在挂了本项目的基础镜像容器里跑一次（纯 CPU 交叉编译 gfx942，**不需要 GPU**）：

```bash
bash odc_rocm_dev/build_rocshmem_backend.sh          # 编全部：single + ro + gda + tensor_ipc
# 或只编单机后端（最常用）：
bash odc_rocm_dev/build_rocshmem_backend.sh single tensor_ipc
```

脚本幂等：会 ① git clone rocSHMEM @ pin 的 commit 并 cmake 编出各变体 `librocshmem.a`，
② hipcc 编出 3 个 ctypes 绑定 `.so`，③ `pip install -e .` 编 `tensor_ipc.so`。
全程日志落 `rocshmem_runtime/build.log`；产物（`.so`/`rocshmem_*/`）已被 `.gitignore`
忽略，不会误入库。要强制重编加 `--force`。

> 默认 P2P 后端仍是 `mori`，本目录只有在 `ODC_P2P_BACKEND=rocshmem` 时才被加载，
> 可随时回退。

## 目录结构

下面标 `[gen]` 的都是 `build_rocshmem_backend.sh` 生成、**不入库**（`.gitignore` 已忽略）；
其余为提交的源码。

```
rocshmem_runtime/
├── README.md                     ← 本文件（构建/使用说明）
├── host_bindings/                ← 单机 IPC 后端
│   ├── rs_host.cpp               ← 源码（入库）
│   └── librs_host5.so            ← [gen] host ctypes 绑定
├── ro_backend/                   ← 多机 RO 后端
│   ├── rs_host_ro.cpp            ← 源码（入库；含跨节点 put/get/int_p）
│   └── librs_host_ro.so          ← [gen]
├── gda_backend/                  ← 多机 GPU-direct (GDA) 后端
│   ├── rs_host_gda.cpp           ← 源码（入库；host + 设备 gather/reduce-scatter kernel）
│   └── librs_host_gda.so         ← [gen]
├── rocshmem_src/                 ← [gen] git clone 的 rocSHMEM 源码（pin 到下方 commit）
├── rocshmem_single/              ← [gen] 单机静态库安装树：USE_IPC + USE_SINGLE_NODE
│   └── lib/librocshmem.a, include/rocshmem/*.hpp
├── rocshmem_ro/                  ← [gen] RO 静态库：USE_IPC + USE_RO
├── rocshmem_gda/                 ← [gen] GDA 静态库：USE_GDA
├── build.log                     ← [gen] 构建日志
├── scripts/
│   ├── run_odc.sh                ← 可移植启动脚本（路径自动从脚本位置推导）
│   └── cleanup_rs.sh             ← 残留进程清理
└── backup/                       ← 适配代码与 smoke 配置的参考副本（非运行必需）
```

（构建脚本本体在上一级：`odc_rocm_dev/build_rocshmem_backend.sh`。）

## 关键事实

- **纯代码构建。** git 里不放任何 `.so`/`.a`；`build_rocshmem_backend.sh` 从源码把
  `librocshmem.a` 与 3 个绑定 `.so` 全部编出。`librs_host5.so` 把 `librocshmem.a`
  静态链进去，运行时只依赖基础镜像自带的 ROCm 运行时（`libamdhip64`、
  `libhsa-runtime64`）+ 系统 `libmpi`。
- **编译要点（踩坑记录，脚本已内置修法）**：
  1. 该版 `rocshmem.hpp` 无条件 `#include <mpi.h>` → 所有绑定编译都要加
     `-I /usr/lib/x86_64-linux-gnu/openmpi/include`，并链 `-lmpi`（连单机也要，
     因 `mpi_instance.cpp` 引用 MPI 符号）。
  2. `librocshmem.a` 需 `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` 编（否则链进
     `-shared` 报 `R_X86_64_32S ... recompile with -fPIC`）。
  3. `librocshmem.a` 含 `fgpu-rdc` 设备代码 → 绑定 `.so` 必须两阶段
     `hipcc -fgpu-rdc -c` + `hipcc -fgpu-rdc --hip-link -shared`（否则报
     `undefined hidden symbol: __hip_gpubin_handle_*`）。
  4. hipcc 的 `-x hip` 会把 `.a` 也当源码编（`source file is not valid UTF-8`）→
     归档前加 `-x none` 复位输入语言。
- **ABI / 工具链**：ROCm 7.2.0、rocSHMEM 3.2.0、`gfx942`（MI300/MI355 系）。
  `pr-722`、`v26.2` 两镜像都是同基础。
- 源码 commit：rocSHMEM `17ff985c026f9f97f85068647e863ab541dd5645`
  （"Update version to 3.2.0 for 7.2.0 rocm release"）。

## 适配代码如何找到 .so（已去写死路径）

`odc/primitives/_rocshmem_backend.py` 的 `_resolve_host_lib()` 按以下顺序解析（命中即用）：

1. `ODC_ROCSHMEM_LIB` —— 显式 .so 全路径（向后兼容，最高优先级）
2. `ODC_RS_HOST_LIB` —— 同上的别名
3. `ROCSHMEM_LIB_DIR` —— 存放 .so 的目录（自动取 `librs_host5.so` 或 RO 时 `librs_host_ro.so`）
4. **项目相对默认值** —— 本目录 `host_bindings/librs_host5.so`（`ODC_ROCSHMEM_RO=1`
   时取 `ro_backend/librs_host_ro.so`）

所以**什么 env 都不设**也能跑：默认走项目自带的单机 IPC `.so`。

## 单机快速使用

```bash
# 在挂了本项目的容器里：
bash odc_rocm_dev/rocshmem_runtime/scripts/run_odc.sh \
  rocshmem nopad \
  examples/megatron/configs/MI355X/deepseek1.5B-lbmini-cmp-fit.yaml \
  my_rocshmem_smoke
```

## 如何重新编译（就用构建脚本）

一律用 `build_rocshmem_backend.sh`，它把下面这些细节都封装好了：

```bash
# 全部重编：
bash odc_rocm_dev/build_rocshmem_backend.sh --force
# 只重编某个变体的 .so（静态库存在则复用）：
bash odc_rocm_dev/build_rocshmem_backend.sh single        # 或 ro / gda / tensor_ipc
```

各变体的 cmake / hipcc 命令（脚本内部实际执行的，供参考）：

- `librocshmem.a` cmake（都带 `-DGPU_TARGETS=gfx942 -DCMAKE_POSITION_INDEPENDENT_CODE=ON`）：
  - single：`-DUSE_IPC=ON -DUSE_SINGLE_NODE=ON`
  - ro：`-DUSE_IPC=ON -DUSE_RO=ON`
  - gda：`-DUSE_GDA=ON`
- 绑定 `.so`（两阶段，`<v>` ∈ single/ro/gda）：

```bash
hipcc -fgpu-rdc -x hip --offload-arch=gfx942:xnack- -fPIC -O2 -std=c++17 \
  -c <cpp> -o <obj> -I rocshmem_<v>/include -I /usr/lib/x86_64-linux-gnu/openmpi/include
hipcc -fgpu-rdc --hip-link -shared <obj> -o <out.so> --offload-arch=gfx942:xnack- \
  -x none rocshmem_<v>/lib/librocshmem.a -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 \
  -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi     # ro/gda 额外：-libverbs -lmlx5 -lnuma (gda)
```

## 未打包/废弃的产物（注明）

容器 `/root/odc_rs/` 内还有下列东西，**有意未抽取**：

- 早期废弃的设备 bitcode / 中间产物：`*.bc`（`full2.bc`、`odc_rs_device*.bc`、
  `slim*.bc`、`wrappers*.bc`）、`odc_rs_device.ll`、`*.o`、`wrappers.hip`
  —— 这是"把 rocSHMEM 设备 API 编进 Triton"的早期失败路线的残留；当前后端不用任何
  rocSHMEM 设备 bitcode（设备侧全是 Triton tl.load/store 走 XGMI），故不带。
- `librs_host.so`（被 `librs_host5.so` 取代的旧版）、`rs_host.o`/`rs_host5.o` 等中间物。
- 海量调参/对照日志 `console_*.log` / `runlog_*.log`、Triton 缓存 `tcache_*/`
  —— 体积大且非运行必需，不带。
- rocSHMEM 源码 `/root/rocshmem_src`（23M）—— 只记 commit，按上方命令可重获。
