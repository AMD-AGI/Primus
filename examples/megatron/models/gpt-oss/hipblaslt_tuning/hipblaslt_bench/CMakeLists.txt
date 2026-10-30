# Builds only hipblaslt-bench from the hipBLASLt sources, against the hipBLASLt already installed
# in the image (TheRock rocm-sdk wheels), so the bench enumerates the same kernel library and
# solution indices as training. Build steps: ../../README.md, "Building hipblaslt-bench".
cmake_minimum_required(VERSION 3.25)
project(hipblaslt_bench_standalone LANGUAGES CXX Fortran)

set(HIPBLASLT_SRC "" CACHE PATH "rocm-libraries/projects/hipblaslt at the image's TheRock commit")
if(NOT EXISTS "${HIPBLASLT_SRC}/clients/CMakeLists.txt")
  message(FATAL_ERROR "Set -DHIPBLASLT_SRC=<rocm-libraries>/projects/hipblaslt")
endif()

find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED CONFIG)
find_package(amd_smi REQUIRED)
find_package(OpenMP REQUIRED)
find_package(LAPACK REQUIRED)

set(HIPBLASLT_ENABLE_BLIS OFF)
set(HIPBLASLT_ENABLE_MXDATAGENERATOR OFF)
set(HIPBLASLT_ENABLE_ROCROLLER ON)
set(HIPBLASLT_ENABLE_COVERAGE OFF)
set(HIPBLASLT_ENABLE_ASAN OFF)
set(HIPBLASLT_ENABLE_GEMM_A2A_FUSION OFF)
set(HIPBLASLT_ENABLE_YAML OFF)
set(HIPBLASLT_ENABLE_OPENMP ON)
set(HIPBLASLT_ENABLE_SAMPLES OFF)
set(HIPBLASLT_BUILD_TESTING OFF)
set(BUILD_TESTING OFF)

# clients/CMakeLists.txt resolves library/ headers through PROJECT_SOURCE_DIR.
set(PROJECT_SOURCE_DIR "${HIPBLASLT_SRC}")
add_subdirectory("${HIPBLASLT_SRC}/clients" "${CMAKE_CURRENT_BINARY_DIR}/clients")

# The in-tree hipblaslt target carries these as BUILD_INTERFACE-only usage requirements, so the
# installed roc::hipblaslt does not export them. The source include dir adds headers that are not
# installed (e.g. hipblaslt_xfloat32.h).
target_compile_definitions(hipblaslt-clients-common PUBLIC ROCM_USE_FLOAT16 __HIP_HCC_COMPAT_MODE__=1)
target_include_directories(hipblaslt-clients-common AFTER PUBLIC "${HIPBLASLT_SRC}/library/include")
