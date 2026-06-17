// Reproduce hipBLASLt solution 332814 out-of-bounds write as a GPU fault.
// Problem (from training crash, PID 4680):
//   bf16, transA=T transB=N, m=1024 n=5407 k=2048, alpha=1 beta=0, in-place C==D,
//   workspace=256MB, computeType=COMPUTE_32F.
// Output D is backed by VMM: D_size mapped RW, followed by a READ-ONLY guard region.
// The buggy kernel writes far past D -> "Write access to a read-only page".
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_HIP(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                         \
    if (e != hipSuccess) {                                                      \
      printf("HIP error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__,           \
             hipGetErrorString(e));                                            \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CHECK_BLAS(x)                                                           \
  do {                                                                          \
    hipblasStatus_t s = (x);                                                    \
    if (s != HIPBLAS_STATUS_SUCCESS) {                                          \
      printf("hipBLASLt error %s at %s:%d -> %d\n", #x, __FILE__, __LINE__,     \
             (int)s);                                                          \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

static size_t roundup(size_t v, size_t g) { return ((v + g - 1) / g) * g; }

int main(int argc, char **argv) {
  int solIndex = (argc > 1) ? atoi(argv[1]) : 332814;
  const int m = 1024, n = 5407, k = 2048;
  const size_t lda = 2048, ldb = 2048, ldc = 1024, ldd = 1024;
  const size_t elem = 2; // bf16

  int dev = 0;
  CHECK_HIP(hipSetDevice(dev));

  // ---- input buffers (roomy hipMalloc) ----
  size_t sizeA = lda * 1024 * elem;  // A stored 2048x1024
  size_t sizeB = ldb * (size_t)n * elem; // B stored 2048x5407
  void *A = nullptr, *B = nullptr, *workspace = nullptr;
  CHECK_HIP(hipMalloc(&A, sizeA));
  CHECK_HIP(hipMalloc(&B, sizeB));
  CHECK_HIP(hipMemset(A, 1, sizeA));
  CHECK_HIP(hipMemset(B, 1, sizeB));
  size_t wsSize = 256ull * 1024 * 1024;
  CHECK_HIP(hipMalloc(&workspace, wsSize));

  // ---- output D via VMM with read-only guard ----
  size_t sizeD = ldd * (size_t)n * elem; // 1024*5407*2 = ~10.56 MB

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  size_t gran = 0;
  CHECK_HIP(hipMemGetAllocationGranularity(&gran, &prop,
                                           hipMemAllocationGranularityRecommended));
  size_t dMap = roundup(sizeD, gran);
  size_t guard = roundup(64ull * 1024 * 1024, gran);
  size_t total = dMap + guard;

  void *base = nullptr;
  CHECK_HIP(hipMemAddressReserve(&base, total, 0, 0, 0));

  // Map ONLY the D region (RW). The guard VA stays reserved but UNMAPPED,
  // so any write past D faults ("memory access fault" on a not-present page).
  hipMemGenericAllocationHandle_t hD;
  CHECK_HIP(hipMemCreate(&hD, dMap, &prop, 0));
  CHECK_HIP(hipMemMap(base, dMap, 0, hD, 0));

  hipMemAccessDesc adRW = {};
  adRW.location = prop.location;
  adRW.flags = hipMemAccessFlagsProtReadWrite;
  CHECK_HIP(hipMemSetAccess(base, dMap, &adRW, 1));

  void *D = base;
  void *C = base; // in-place C == D
  printf("D base        = %p\n", D);
  printf("D logical end = %p (sizeD=%zu, %.2f MB)\n", (char *)D + sizeD, sizeD,
         sizeD / 1048576.0);
  printf("RW mapped end = %p (dMap=%.2f MB)\n", (char *)D + dMap, dMap / 1048576.0);
  printf("UNMAPPED guard= [%p, %p)  (%.0f MB)\n", (char *)base + dMap,
         (char *)base + total, guard / 1048576.0);

  // ---- hipBLASLt setup ----
  hipblasLtHandle_t handle;
  CHECK_BLAS(hipblasLtCreate(&handle));

  hipblasLtMatmulDesc_t matmul;
  CHECK_BLAS(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
  CHECK_BLAS(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &opT, sizeof(opT)));
  CHECK_BLAS(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &opN, sizeof(opN)));

  hipblasLtMatrixLayout_t Ad, Bd, Cd, Dd;
  CHECK_BLAS(hipblasLtMatrixLayoutCreate(&Ad, HIP_R_16BF, 2048, 1024, lda));
  CHECK_BLAS(hipblasLtMatrixLayoutCreate(&Bd, HIP_R_16BF, 2048, n, ldb));
  CHECK_BLAS(hipblasLtMatrixLayoutCreate(&Cd, HIP_R_16BF, 1024, n, ldc));
  CHECK_BLAS(hipblasLtMatrixLayoutCreate(&Dd, HIP_R_16BF, 1024, n, ldd));

  std::vector<int> idx = {solIndex};
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  CHECK_BLAS(hipblaslt_ext::getAlgosFromIndex(handle, idx, results));
  if (results.empty()) {
    printf("No algo for solution index %d\n", solIndex);
    return 2;
  }
  printf("Got %zu algo(s) for solution index %d\n", results.size(), solIndex);

  float alpha = 1.0f, beta = 0.0f;
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  printf("Launching matmul with solution %d ...\n", solIndex);
  fflush(stdout);
  hipblasStatus_t st = hipblasLtMatmul(handle, matmul, &alpha, A, Ad, B, Bd,
                                       &beta, C, Cd, D, Dd, &results[0].algo,
                                       workspace, wsSize, stream);
  printf("hipblasLtMatmul returned %d\n", (int)st);
  fflush(stdout);
  CHECK_HIP(hipStreamSynchronize(stream));
  printf("Synchronized OK (no fault).\n");
  return 0;
}
