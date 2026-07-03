// Combined rocSHMEM GDA host binding + GPU-direct device kernels for ODC.
//
// ONE .so (rdc-linked with GDA librocshmem.a) holding BOTH host init/heap AND the
// device gather / reduce-scatter kernels, so the device default context set up by
// rocshmem_init is visible to the kernels (validated by the ctypes probe).
//
// Host ABI is a superset of the RO binding (rs_my_pe/n_pes/malloc/ptr/barrier/
// finalize/is_remote) so _rocshmem_backend.py can reuse it, plus:
//   rs_init_mpi()          -> rocshmem_init() (MPI bootstrap; mpirun + UCX-OSC)
//   gda_gather(...)        -> per cross-node peer device rocshmem_getmem_wg
//   gda_reduce_scatter_acc -> pull-based, on-chip fp32 accumulate (race-free)
//
// Build (inside container):
//   hipcc -fgpu-rdc -x hip --offload-arch=gfx942:xnack- -fPIC -c rs_host_gda.cpp -o x.o \
//     -I<gda_install>/include -I/usr/lib/x86_64-linux-gnu/openmpi/include
//   hipcc -fgpu-rdc --hip-link -shared x.o -o librs_host_gda.so --offload-arch=gfx942:xnack- \
//     <gda_install>/lib/librocshmem.a -L.../openmpi/lib -lmpi -L/opt/rocm/lib \
//     -lamdhip64 -lhsa-runtime64 -libverbs -lmlx5 -lnuma
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <rocshmem/rocshmem.hpp>
#include <cstdint>
#include <cstdio>
using namespace rocshmem;

// dtype codes (subset of torch c10::ScalarType used by ODC reduce-scatter)
#define DT_F32 6
#define DT_BF16 15
#define DT_F16 5

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------
// Gather: block p pulls cross-node peer peers[p]'s shard (at symmetric address
// `src`) into local `target + p*stride_bytes`. rocshmem_getmem_wg is blocking
// (data delivered on return), so no separate quiet is needed.
__global__ void gather_kernel(char *target, const char *src, size_t nbytes,
                              const int *peers, int n_peers, size_t stride_bytes) {
  int p = blockIdx.x;
  if (p < n_peers) {
    // write peer's shard into ITS GLOBAL-rank slot (target + peer*stride) so the
    // existing per-rank reassembly in gather.py reads the right slot.
    rocshmem_getmem_wg(target + (size_t)peers[p] * stride_bytes, (void *)src, nbytes, peers[p]);
  }
}

// Pull-based reduce-scatter accumulate: acc_fp32[i] += sum_pe input[my_seg + i]
// pulled from PE `pe` (input is symmetric; my shard's segment lives at the same
// symmetric byte offset `seg_off` on every PE). Each block owns a disjoint chunk
// of the shard -> no cross-block race; peers iterated sequentially within a block.
template <typename T>
__global__ void rs_acc_kernel(float *acc, const char *input_sym, size_t seg_off,
                              size_t shard_elems, int n_pes, char *scratch_sym,
                              size_t scratch_stride) {
  int b = blockIdx.x;
  int nb = gridDim.x;
  size_t chunk = (shard_elems + nb - 1) / nb;
  size_t c0 = (size_t)b * chunk;
  size_t c1 = c0 + chunk; if (c1 > shard_elems) c1 = shard_elems;
  if (c0 >= c1) return;
  size_t n = c1 - c0;
  T *scr = (T *)(scratch_sym + (size_t)b * scratch_stride);
  const char *src = input_sym + seg_off + c0 * sizeof(T);
  for (int pe = 0; pe < n_pes; ++pe) {
    // pull peer `pe`'s segment for my shard chunk into this block's scratch
    rocshmem_getmem_wg((void *)scr, (void *)src, n * sizeof(T), pe);
    __syncthreads();
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      acc[c0 + i] += (float)scr[i];
    }
    __syncthreads();
  }
}

// --- PIPELINED reduce-scatter (Deliverable 17): overlap per-peer RDMA latency ---------
// Original rs_acc pulls peers SEQUENTIALLY with blocking getmem_wg -> each block
// eats n_pes x (RTT+transfer) serially. For the small per-peer per-chunk segments
// (latency-bound; Deliverable 16 showed only LARGE msgs are BW-saturated) this serial RTT
// chain is the cross-node RDMA time we can shorten. Here each block issues a BATCH
// of `pipe_b` peers' getmem NON-BLOCKING (rocshmem_char_get_nbi_wg) into distinct
// scratch slots, then ONE rocshmem_quiet() waits the whole batch -> the batch's RTTs
// OVERLAP. Correctness identical (same data summed; quiet guarantees completion
// before accumulate; strided settle still primes visibility upstream). Scratch must
// be pipe_b slots/block (scratch_stride = pipe_b*chunk*sizeof(T), sized in python).
template <typename T>
__global__ void rs_acc_kernel_pipe(float *acc, const char *input_sym, size_t seg_off,
                                   size_t shard_elems, int n_pes, char *scratch_sym,
                                   size_t scratch_stride, int pipe_b) {
  int b = blockIdx.x;
  int nb = gridDim.x;
  size_t chunk = (shard_elems + nb - 1) / nb;
  size_t c0 = (size_t)b * chunk;
  size_t c1 = c0 + chunk; if (c1 > shard_elems) c1 = shard_elems;
  if (c0 >= c1) return;
  size_t n = c1 - c0;
  char *base = scratch_sym + (size_t)b * scratch_stride;  // pipe_b slots live here
  size_t slot = chunk * sizeof(T);                         // per-peer slot stride (bytes)
  const char *src = input_sym + seg_off + c0 * sizeof(T);
  for (int pe0 = 0; pe0 < n_pes; pe0 += pipe_b) {
    int pe1 = pe0 + pipe_b; if (pe1 > n_pes) pe1 = n_pes;
    // issue this batch of peers NON-BLOCKING (RTTs overlap)
    for (int pe = pe0; pe < pe1; ++pe) {
      rocshmem_char_get_nbi_wg(base + (size_t)(pe - pe0) * slot, src, n * sizeof(T), pe);
    }
    __syncthreads();         // all WG threads finished ISSUING before any quiets (race fix)
    rocshmem_quiet();        // wait the whole batch completes (data landed in scratch)
    __threadfence();         // make NIC-delivered scratch writes visible to accumulate reads
    __syncthreads();
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      float s = 0.0f;
      for (int pe = pe0; pe < pe1; ++pe) s += (float)(*(T *)(base + (size_t)(pe - pe0) * slot + i * sizeof(T)));
      acc[c0 + i] += s;
    }
    __syncthreads();
  }
}

// --- MULTI-QP (Deliverable 15) ctx-aware kernel variants ----------------------------
// The GDA backend allocates one QueuePair PER PE PER CONTEXT (queue_pair.cpp:
// hipMalloc(qps, sizeof(QueuePair)*num_pes)). So N host-created contexts give N
// QPs per peer-connection. These ctx variants round-robin blocks across a pool of
// `nctx` contexts (ctxs[blockIdx % nctx]) so concurrent cross-node getmem_wg use
// distinct QPs -> more NIC/RDMA concurrency. Selected only when ODC_GDA_NUM_QP>1;
// NUM_QP==1 keeps the original default-context path byte-for-byte.
__global__ void gather_kernel_ctx(char *target, const char *src, size_t nbytes,
                                  const int *peers, int n_peers, size_t stride_bytes,
                                  rocshmem_ctx_t *ctxs, int nctx) {
  int p = blockIdx.x;
  if (p < n_peers) {
    rocshmem_ctx_getmem_wg(ctxs[p % nctx], target + (size_t)peers[p] * stride_bytes,
                           (void *)src, nbytes, peers[p]);
  }
}

template <typename T>
__global__ void rs_acc_kernel_ctx(float *acc, const char *input_sym, size_t seg_off,
                                  size_t shard_elems, int n_pes, char *scratch_sym,
                                  size_t scratch_stride, rocshmem_ctx_t *ctxs, int nctx) {
  int b = blockIdx.x;
  int nb = gridDim.x;
  rocshmem_ctx_t ctx = ctxs[b % nctx];
  size_t chunk = (shard_elems + nb - 1) / nb;
  size_t c0 = (size_t)b * chunk;
  size_t c1 = c0 + chunk; if (c1 > shard_elems) c1 = shard_elems;
  if (c0 >= c1) return;
  size_t n = c1 - c0;
  T *scr = (T *)(scratch_sym + (size_t)b * scratch_stride);
  const char *src = input_sym + seg_off + c0 * sizeof(T);
  for (int pe = 0; pe < n_pes; ++pe) {
    rocshmem_ctx_getmem_wg(ctx, (void *)scr, (void *)src, n * sizeof(T), pe);
    __syncthreads();
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
      acc[c0 + i] += (float)scr[i];
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host / extern-C API
// ---------------------------------------------------------------------------
static int g_initialized = 0;

// --- geometry / multi-QP env knobs (cached) --------------------------------
static int g_env_block = -1;
static int env_block() {
  if (g_env_block < 0) {
    const char *e = getenv("ODC_GDA_BLOCK");
    g_env_block = e ? atoi(e) : 256;
    if (g_env_block < 32) g_env_block = 32;
    if (g_env_block > 1024) g_env_block = 1024;
  }
  return g_env_block;
}
static int g_env_pipe = -1;
static int env_pipe() {  // reduce-scatter peer-pipeline batch depth (1 = original serial)
  if (g_env_pipe < 0) {
    const char *e = getenv("ODC_GDA_PIPE");
    g_env_pipe = e ? atoi(e) : 1;
    if (g_env_pipe < 1) g_env_pipe = 1;
    if (g_env_pipe > 64) g_env_pipe = 64;
  }
  return g_env_pipe;
}
static int g_env_numqp = -1;
static int env_numqp() {
  if (g_env_numqp < 0) {
    const char *e = getenv("ODC_GDA_NUM_QP");
    g_env_numqp = e ? atoi(e) : 1;
    if (g_env_numqp < 1) g_env_numqp = 1;
    if (g_env_numqp > 256) g_env_numqp = 256;
  }
  return g_env_numqp;
}

// Lazily create a device-visible pool of `nqp` host contexts (each = own QP set).
// Returns 0 on success; negative on ctx_create / hipMalloc failure (caller aborts
// the multi-QP path so a backend that can't honor NUM_QP fails loud, not silent).
static rocshmem_ctx_t *g_dev_ctxs = nullptr;
static int g_dev_nctx = 0;
static int gda_ensure_ctxs(int nqp) {
  if (g_dev_ctxs && g_dev_nctx == nqp) return 0;
  rocshmem_ctx_t *host = (rocshmem_ctx_t *)malloc(sizeof(rocshmem_ctx_t) * nqp);
  if (!host) return -1;
  for (int i = 0; i < nqp; ++i) {
    int rc = rocshmem_ctx_create(0, &host[i]);  // 0 = default options, fresh QP set
    if (rc != 0) { free(host); return -100 - i; }
  }
  if (g_dev_ctxs) hipFree(g_dev_ctxs);
  if (hipMalloc(&g_dev_ctxs, sizeof(rocshmem_ctx_t) * nqp) != hipSuccess) { free(host); return -2; }
  hipMemcpy(g_dev_ctxs, host, sizeof(rocshmem_ctx_t) * nqp, hipMemcpyHostToDevice);
  g_dev_nctx = nqp;
  free(host);
  return 0;
}

extern "C" {

// RO-ABI-compatible surface (so _rocshmem_backend can reuse it) ----------------
int  rs_uid_bytes() { return (int)sizeof(rocshmem_uniqueid_t); }
void rs_get_uid(char *out) { rocshmem_uniqueid_t uid; rocshmem_get_uniqueid(&uid); memcpy(out, uid.data(), sizeof(uid)); }
void rs_init_uid(int rank, int nranks, const char *bytes) {
  rocshmem_uniqueid_t uid; memcpy(uid.data(), bytes, sizeof(uid));
  rocshmem_init_attr_t attr; rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  g_initialized = 1;
}
// GDA MPI bootstrap (the validated probe path): rocshmem_init() over MPI_COMM_WORLD.
void rs_init_mpi() { rocshmem_init(); g_initialized = 1; }
int  rs_my_pe() { return rocshmem_my_pe(); }
int  rs_n_pes() { return rocshmem_n_pes(); }
long long rs_malloc(size_t n) { return (long long)rocshmem_malloc(n); }
long long rs_ptr(long long p, int pe) { return (long long)rocshmem_ptr((void *)p, pe); }
int  rs_is_remote(long long base, int pe) { return rocshmem_ptr((void *)base, pe) == nullptr ? 1 : 0; }
void rs_barrier() { rocshmem_barrier_all(); }
void rs_finalize() { if (g_initialized) { rocshmem_finalize(); g_initialized = 0; } }

// host<->device copy + memset helpers (used by the standalone kernel numeric test)
void gda_h2d(long long dptr, const void *host, size_t nbytes) { hipMemcpy((void *)dptr, host, nbytes, hipMemcpyHostToDevice); hipDeviceSynchronize(); }
void gda_d2h(long long dptr, void *host, size_t nbytes) { hipMemcpy(host, (void *)dptr, nbytes, hipMemcpyDeviceToHost); hipDeviceSynchronize(); }
void gda_memset(long long dptr, int val, size_t nbytes) { hipMemset((void *)dptr, val, nbytes); hipDeviceSynchronize(); }

}  // (close extern "C" temporarily for the staging kernel below)

// Staging copy that ends in a SYSTEM-scope fence so the written symmetric buffer
// is visible to the NIC (and thus to a remote PE's device getmem) -- substitutes
// for the unavailable hipDeviceFlushGPUDirectRDMAWrites / rocSHMEM HDP-MR flush.
__global__ void stage_fence_kernel(char *dst, const char *src, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = src[i];
  }
  __threadfence_system();
}

// Strided page-touch warm-up: prime the cross-node read path for EVERY page of
// my shard's segment on EVERY PE with a tiny (touch_bytes) RDMA read -- the
// deterministic "read-triggered settle" of the full warm-up, but at minimal
// volume (one touch per `stride_bytes`, not the whole shard). Block index maps
// to a (pe, page) pair; a grid-stride loop covers all pairs. Each block uses its
// own throwaway scratch slot so concurrent blocks never collide.
__global__ void strided_touch_kernel(const char *input_sym, size_t seg_off,
                                     size_t seg_bytes, int n_pes,
                                     size_t stride_bytes, size_t touch_bytes,
                                     char *scratch_sym, size_t scratch_stride) {
  size_t npages = (seg_bytes + stride_bytes - 1) / stride_bytes;
  if (npages == 0) npages = 1;
  size_t total = (size_t)n_pes * npages;
  char *scr = scratch_sym + (size_t)blockIdx.x * scratch_stride;
  for (size_t idx = blockIdx.x; idx < total; idx += gridDim.x) {
    int pe = (int)(idx / npages);
    size_t off = (idx % npages) * stride_bytes;
    size_t n = touch_bytes;
    if (off + n > seg_bytes) n = seg_bytes - off;  // off < seg_bytes always
    rocshmem_getmem_wg((void *)scr, (void *)(input_sym + seg_off + off), n, pe);
  }
}

extern "C" {

// Strided page-touch warm-up launcher (see kernel above). Returns hipError_t.
int gda_strided_touch(long long input_sym, size_t seg_off_bytes, size_t seg_bytes,
                      int n_pes, size_t stride_bytes, size_t touch_bytes,
                      long long scratch_sym, size_t scratch_stride, int nblocks) {
  if (nblocks < 1) nblocks = 1;
  strided_touch_kernel<<<dim3((unsigned)nblocks), dim3(64)>>>(
      (const char *)input_sym, seg_off_bytes, seg_bytes, n_pes, stride_bytes,
      touch_bytes, (char *)scratch_sym, scratch_stride);
  return (int)hipDeviceSynchronize();
}

// Copy nbytes src->dst (both device) then system-fence; returns hipError_t.
int gda_stage_fence(long long dst, long long src, size_t nbytes) {
  int t = 256;
  long long b = (nbytes + t - 1) / t;
  if (b > 2048) b = 2048;
  if (b < 1) b = 1;
  stage_fence_kernel<<<dim3((unsigned)b), dim3(t)>>>((char *)dst, (const char *)src, nbytes);
  return (int)hipDeviceSynchronize();
}

// HDP (Host Data Path) flush register -- the PROPER GPUDirect-RDMA write-
// visibility primitive on AMD. Writing HDP_MEM_FLUSH_CNTL flushes this GPU's
// HDP cache so a remote NIC's RDMA read of our just-staged symmetric write sees
// fresh data, covering ALL pages/NICs at once in O(1) (vs the full-shard
// throwaway reduce-scatter "warm-up"). __threadfence_system alone is NOT
// sufficient on this mlx5/GDA path (verified: grad spikes return).
static volatile uint32_t *g_hdp_flush = nullptr;
static hsa_agent_t g_gpu_agents[16];
static int g_n_gpu = 0;

static hsa_status_t _collect_gpu_agents(hsa_agent_t agent, void *) {
  hsa_device_type_t t;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS &&
      t == HSA_DEVICE_TYPE_GPU && g_n_gpu < 16) {
    g_gpu_agents[g_n_gpu++] = agent;
  }
  return HSA_STATUS_SUCCESS;
}

// Resolve the HDP flush register for THIS rank's current HIP device. Must be
// called after hipSetDevice (so hipGetDevice returns the rank's GPU). HSA GPU
// agents enumerate in HIP-device order. Returns 0 on success, nonzero on error.
int gda_hdp_init() {
  if (g_hdp_flush) return 0;
  g_n_gpu = 0;
  hsa_iterate_agents(_collect_gpu_agents, nullptr);
  if (g_n_gpu == 0) return -1;
  int dev = 0;
  hipGetDevice(&dev);
  int idx = (dev >= 0 && dev < g_n_gpu) ? dev : 0;
  hsa_amd_hdp_flush_t hdp;
  hsa_status_t s = hsa_agent_get_info(
      g_gpu_agents[idx], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp);
  if (s != HSA_STATUS_SUCCESS) return (int)s;
  g_hdp_flush = (volatile uint32_t *)hdp.HDP_MEM_FLUSH_CNTL;
  return g_hdp_flush ? 0 : -2;
}

// Flush this GPU's HDP so prior symmetric writes are visible to the NIC.
// Returns 1 if the flush register is resolved (flush issued), 0 otherwise.
int gda_hdp_flush() {
  if (!g_hdp_flush) return 0;
  *g_hdp_flush = 1u;
  __sync_synchronize();
  return 1;
}

// GPU-direct gather: returns hipError_t (0 == success; nonzero/never-returns => hang)
int gda_gather(long long target, long long src, size_t nbytes,
               const int *peers_host, int n_peers, size_t stride_bytes) {
  if (n_peers <= 0) return 0;
  int *d_peers = nullptr;
  hipMalloc(&d_peers, n_peers * sizeof(int));
  hipMemcpy(d_peers, peers_host, n_peers * sizeof(int), hipMemcpyHostToDevice);
  int blk = env_block(), nqp = env_numqp();
  if (nqp > 1) {
    if (gda_ensure_ctxs(nqp) != 0) { hipFree(d_peers); return -999; }
    gather_kernel_ctx<<<dim3(n_peers), dim3(blk), 0, 0>>>((char *)target, (const char *)src,
        nbytes, d_peers, n_peers, stride_bytes, g_dev_ctxs, nqp);
  } else {
    gather_kernel<<<dim3(n_peers), dim3(blk), 0, 0>>>((char *)target, (const char *)src,
        nbytes, d_peers, n_peers, stride_bytes);
  }
  hipError_t e = hipDeviceSynchronize();
  hipFree(d_peers);
  return (int)e;
}

// GPU-direct pull-based reduce-scatter accumulate into fp32 acc.
int gda_reduce_scatter_acc(long long acc_fp32, long long input_sym, size_t seg_off_bytes,
                           size_t shard_elems, int n_pes, long long scratch_sym,
                           size_t scratch_stride_bytes, int dtype_code, int nblocks) {
  int blk = env_block(), nqp = env_numqp(), pipe = env_pipe();
  dim3 grid(nblocks), block(blk);
  if (pipe > 1) {  // Deliverable 17: peer-pipelined non-blocking getmem (scratch_stride sized pipe*chunk by python)
    if (dtype_code == DT_BF16) {
      rs_acc_kernel_pipe<__hip_bfloat16><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, pipe);
    } else if (dtype_code == DT_F16) {
      rs_acc_kernel_pipe<__half><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, pipe);
    } else {
      rs_acc_kernel_pipe<float><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, pipe);
    }
    return (int)hipDeviceSynchronize();
  }
  if (nqp > 1) {
    if (gda_ensure_ctxs(nqp) != 0) return -999;
    if (dtype_code == DT_BF16) {
      rs_acc_kernel_ctx<__hip_bfloat16><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, g_dev_ctxs, nqp);
    } else if (dtype_code == DT_F16) {
      rs_acc_kernel_ctx<__half><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, g_dev_ctxs, nqp);
    } else {
      rs_acc_kernel_ctx<float><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
          seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes, g_dev_ctxs, nqp);
    }
    return (int)hipDeviceSynchronize();
  }
  if (dtype_code == DT_BF16) {
    rs_acc_kernel<__hip_bfloat16><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
        seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes);
  } else if (dtype_code == DT_F16) {
    rs_acc_kernel<__half><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
        seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes);
  } else {  // fp32
    rs_acc_kernel<float><<<grid, block>>>((float *)acc_fp32, (const char *)input_sym,
        seg_off_bytes, shard_elems, n_pes, (char *)scratch_sym, scratch_stride_bytes);
  }
  return (int)hipDeviceSynchronize();
}

// --- Comm/compute OVERLAP variant -------------------------------------------
// Launch the reduce-scatter kernel on a persistent SIDE stream and return WITHOUT
// syncing, so the host can proceed to the next backward compute (which runs on the
// default/compute stream) while this RDMA-bound kernel runs concurrently. The
// caller MUST gda_rs_overlap_sync() before (a) re-staging input_sym for the next
// group and (b) the optimizer consuming acc -- correctness is guaranteed by those
// two waits (see scatter_accumulate.py ODC_GDA_OVERLAP path).
static hipStream_t g_rs_stream = nullptr;

int gda_reduce_scatter_acc_async(long long acc_fp32, long long input_sym, size_t seg_off_bytes,
                                 size_t shard_elems, int n_pes, long long scratch_sym,
                                 size_t scratch_stride_bytes, int dtype_code, int nblocks) {
  if (!g_rs_stream) {
    hipError_t ce = hipStreamCreateWithFlags(&g_rs_stream, hipStreamNonBlocking);
    if (ce != hipSuccess) return (int)ce;
  }
  dim3 grid(nblocks), block(256);
  if (dtype_code == DT_BF16) {
    rs_acc_kernel<__hip_bfloat16><<<grid, block, 0, g_rs_stream>>>((float *)acc_fp32,
        (const char *)input_sym, seg_off_bytes, shard_elems, n_pes,
        (char *)scratch_sym, scratch_stride_bytes);
  } else if (dtype_code == DT_F16) {
    rs_acc_kernel<__half><<<grid, block, 0, g_rs_stream>>>((float *)acc_fp32,
        (const char *)input_sym, seg_off_bytes, shard_elems, n_pes,
        (char *)scratch_sym, scratch_stride_bytes);
  } else {
    rs_acc_kernel<float><<<grid, block, 0, g_rs_stream>>>((float *)acc_fp32,
        (const char *)input_sym, seg_off_bytes, shard_elems, n_pes,
        (char *)scratch_sym, scratch_stride_bytes);
  }
  return (int)hipGetLastError();  // launch error only; NO sync (overlap)
}

// Wait for all pending overlapped reduce-scatter kernels on the side stream.
int gda_rs_overlap_sync() {
  if (!g_rs_stream) return 0;
  return (int)hipStreamSynchronize(g_rs_stream);
}

// --- Approach 4 feasibility microbench: device-side completion/visibility ---------
// Tests whether a device kernel's rocshmem_getmem (+ device rocshmem_quiet) makes
// the pulled cross-node data visible to the NEXT kernel on the same stream WITHOUT
// a host hipDeviceSynchronize. err==0 => device-side visibility works (gather-async
// failure was a buffer race, not visibility -> Approach 4 feasible). err>0 => stale
// (host sync is load-bearing for visibility -> Approach 4 blocked on this stack).
__global__ void mb_fill(int *buf, int val, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = val;
}
__global__ void mb_get(int *dst, const int *src_sym, int peer, int n, int do_quiet) {
  rocshmem_getmem_wg((void *)dst, (void *)src_sym, (size_t)n * sizeof(int), peer);
  if (do_quiet) rocshmem_quiet();
}
__global__ void mb_check(const int *dst, int expect, int n, int *err) {
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    if (dst[i] != expect) atomicAdd(err, 1);
}

extern "C" {
// Returns: <0 launch/alloc error; >=0 = mismatch count across `reps` get+check
// pairs with NO host sync between get and check (do_quiet: device rocshmem_quiet).
int gda_microbench(int n, int reps, int do_quiet) {
  int me = rocshmem_my_pe(), npes = rocshmem_n_pes();
  if (npes < 2) return -100;
  int peer = (me + 1) % npes;
  int *local = (int *)rocshmem_malloc((size_t)n * sizeof(int));
  int *scratch = (int *)rocshmem_malloc((size_t)n * sizeof(int));
  if (!local || !scratch) return -101;
  int *d_err = nullptr;
  if (hipMalloc(&d_err, sizeof(int)) != hipSuccess) return -102;
  hipMemset(d_err, 0, sizeof(int));
  int total_err = 0;
  for (int r = 0; r < reps; ++r) {
    // each rep writes a DIFFERENT value so a stale read (old rep's value) is caught
    int myval = me * 1000 + r;
    mb_fill<<<1, 256>>>(local, myval, n);
    hipDeviceSynchronize();
    rocshmem_barrier_all();
    hipMemset(d_err, 0, sizeof(int));
    // THE TEST: get peer's local (== peer*1000+r) into scratch, then check it in a
    // SEPARATE kernel on the same stream with NO host sync between them.
    mb_get<<<1, 256, 0, 0>>>(scratch, local, peer, n, do_quiet);
    mb_check<<<1, 256, 0, 0>>>(scratch, peer * 1000 + r, n, d_err);
    hipDeviceSynchronize();  // single sync only to READ the result, after both kernels
    int err = 0; hipMemcpy(&err, d_err, sizeof(int), hipMemcpyDeviceToHost);
    total_err += err;
    rocshmem_barrier_all();
  }
  hipFree(d_err);
  return total_err;
}
}  // extern "C"

// --- Async GATHER (Approach 1: gather prefetch/overlap) --------------------------
// Launch the all-gather kernel on the CALLER-PROVIDED stream WITHOUT syncing, so
// FSDP2's prefetch (issue layer L+1's all-gather during layer L compute) actually
// overlaps. Gather reads STABLE params (written a step earlier, read-only through
// fwd/bwd) -> no settle/barrier needed; correctness comes from stream ordering:
// the reassembly copies run on the SAME stream after this, and the consumer waits
// that stream (FSDP async_op event / wait_stream). Persistent d_peers (peers list
// is the constant [0..n_peers)) avoids per-call hipMalloc/Free+sync.
static int *g_gather_peers = nullptr;
static int g_gather_npeers = 0;

int gda_gather_async(long long target, long long src, size_t nbytes,
                     const int *peers_host, int n_peers, size_t stride_bytes,
                     long long stream) {
  if (n_peers <= 0) return 0;
  if (g_gather_npeers != n_peers) {
    if (g_gather_peers) hipFree(g_gather_peers);
    if (hipMalloc(&g_gather_peers, n_peers * sizeof(int)) != hipSuccess) return -1;
    g_gather_npeers = n_peers;
  }
  hipStream_t s = (hipStream_t)stream;
  hipMemcpyAsync(g_gather_peers, peers_host, n_peers * sizeof(int), hipMemcpyHostToDevice, s);
  gather_kernel<<<dim3(n_peers), dim3(256), 0, s>>>((char *)target, (const char *)src,
                                                    nbytes, g_gather_peers, n_peers, stride_bytes);
  return (int)hipGetLastError();  // NO sync (overlap); caller orders via stream
}

}  // extern "C"
