#include <torch/extension.h>

// Declare the function from the CUDA file

py::bytes get_ipc_handle(at::Tensor tensor);
at::Tensor reconstruct_tensor(py::bytes handle, std::vector<int64_t> shape, torch::ScalarType dtype);

PYBIND11_MODULE(tensor_ipc, m) {
    m.def("get_ipc_handle", &get_ipc_handle, "Get IPC handle");
    m.def("reconstruct_tensor", &reconstruct_tensor, "Reconstruct tensor");
}
