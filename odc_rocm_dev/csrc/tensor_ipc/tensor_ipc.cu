#include <torch/extension.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(cmd) do {                                 \
  cudaError_t e = (cmd);                              \
  if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d CUDA error %d: %s\n",    \
              __FILE__, __LINE__, (int)e,             \
              cudaGetErrorString(e));                 \
      exit(1);                                        \
  }                                                   \
} while(0)

#define CU_CHECK(cmd) do {                                 \
  CUresult e = (cmd);                              \
  if (e != CUDA_SUCCESS) {                             \
      fprintf(stderr, "%s:%d CUDA Driver error %d\n",    \
              __FILE__, __LINE__, (int)e);                 \
      exit(1);                                        \
  }                                                   \
} while(0)

py::bytes get_ipc_handle(at::Tensor tensor) {
    void *ptr = (void*)tensor.data_ptr();

    CUdeviceptr base;
    size_t size;
    // Use cuMemGetAddressRange_v2 to avoid
    //     undefined symbol: cuMemGetAddressRange_v2
    // CU_CHECK(cuMemGetAddressRange(&base, &size, (CUdeviceptr)ptr));
    CU_CHECK(cuMemGetAddressRange_v2(&base, &size, (CUdeviceptr)ptr));
    // printf("ptr: %p, base ptr: %p, device: %d\n", ptr, base, attr.device);

    size_t pointer_offset = (size_t)ptr - (size_t)base;
    
    cudaIpcMemHandle_t mem_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle, (void*)base));
    std::string s;
    s.resize(sizeof(mem_handle) + sizeof(pointer_offset));
    memcpy(s.data(), &mem_handle, sizeof(mem_handle));
    memcpy(s.data() + sizeof(mem_handle), &pointer_offset, sizeof(pointer_offset));
    return py::bytes(s);
}

at::Tensor reconstruct_tensor(py::bytes handle, std::vector<int64_t> shape, torch::ScalarType dtype) {
  std::string s = handle.cast<std::string>();
  cudaIpcMemHandle_t mem_handle;
  size_t pointer_offset;
  memcpy(&mem_handle, s.data(), sizeof(mem_handle));
  memcpy(&pointer_offset, s.data() + sizeof(mem_handle), sizeof(pointer_offset));
  uint8_t *ptr = nullptr;
  CUDA_CHECK(cudaIpcOpenMemHandle((void**)&ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
  ptr += pointer_offset;

  return at::from_blob(ptr, shape, at::TensorOptions().dtype(dtype).device(at::kCUDA));
}
