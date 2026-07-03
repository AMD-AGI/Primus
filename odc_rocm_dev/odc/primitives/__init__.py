from .nvshmem_triton import (
    LIB_NVSHMEM_PATH,
    NVSHMEM_EXTERN_LIBS,
    __syncthreads,
    getmem_nbi_block,
    int_atomic_compare_swap,
    int_atomic_swap,
    int_g,
    int_p,
    int_p_remote,
    int_wait_until_equals,
    int_wait_until_equals_remote,
    putmem_nbi_block,
    quiet,
    tid,
)

# Try to import tensor_ipc as a proper extension module (for pip install)
try:
    # Need to import torch first to avoid:
    # ImportError: libc10.so: cannot open shared object file
    import torch  # noqa: F401

    from . import tensor_ipc  # pylint: disable=import-self

    get_ipc_handle = tensor_ipc.get_ipc_handle
    reconstruct_tensor = tensor_ipc.reconstruct_tensor
except ImportError as e:
    raise ImportError(
        f"Failed to import tensor_ipc. "
        f"Please install the package using 'pip install' or 'pip install -e .' to build the CUDA extensions: {e}"
    ) from e

__all__ = [
    # nvshmem_triton
    "int_atomic_compare_swap",
    "int_atomic_swap",
    "putmem_nbi_block",
    "getmem_nbi_block",
    "quiet",
    "int_p",
    "int_p_remote",
    "int_g",
    "int_wait_until_equals",
    "int_wait_until_equals_remote",
    "tid",
    "__syncthreads",
    "LIB_NVSHMEM_PATH",
    "NVSHMEM_EXTERN_LIBS",
    # tensor_ipc
    "get_ipc_handle",
    "reconstruct_tensor",
]
