// Extended rocSHMEM host-API binding for the ODC P2P backend, RO-capable.
//
// Superset of rs_host.cpp (single-node IPC binding). Adds host-driven
// cross-node primitives backed by the rocSHMEM Reverse-Offload (RO) conduit:
//   rs_putmem / rs_getmem  -> rocshmem_putmem / rocshmem_getmem (host, MPI fwd)
//   rs_int_p               -> rocshmem_int_p   (host signalling)
//   rs_quiet               -> rocshmem_quiet   (completion ordering)
//   rs_fence               -> rocshmem_fence
// These host calls forward to the RO host runtime which drives the transport
// (UCX/MPI) from the CPU -- there is NO device-side completion polling, which
// is precisely what avoids MORI's GPU-initiated IBGDA hang on this fabric.
//
// rs_ptr() still returns a valid pointer for *intra-node* peers (IPC/XGMI) and
// 0 for *inter-node* peers; the Python layer uses that to pick XGMI vs RO.
//
// Build (inside container odc_rocshmem_test):
//   hipcc -shared -fPIC -O2 -std=c++17 \
//     -I/root/rocshmem_ro/include rs_host_ro.cpp \
//     -o /root/odc_rs/librs_host_ro.so \
//     /root/rocshmem_ro/lib/librocshmem.a \
//     -L/opt/rocm/lib -lhsa-runtime64 -lamdhip64 \
//     $(mpicc --showme:link)
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#include <cstring>
using namespace rocshmem;

extern "C" {

// ---- unchanged single-node surface (ABI-compatible with librs_host5.so) ----
int  rs_uid_bytes(){ return (int)sizeof(rocshmem_uniqueid_t); }
void rs_get_uid(char* out){
  rocshmem_uniqueid_t uid; rocshmem_get_uniqueid(&uid);
  memcpy(out, uid.data(), sizeof(rocshmem_uniqueid_t));
}
void rs_init_uid(int rank, int nranks, const char* bytes){
  rocshmem_uniqueid_t uid; memcpy(uid.data(), bytes, sizeof(rocshmem_uniqueid_t));
  rocshmem_init_attr_t attr; rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
}
void rs_get_ctx_fields(long long* a, long long* b){
  void* dctx = rocshmem_get_device_ctx();
  rocshmem_ctx_t h; hipMemcpy(&h, dctx, sizeof(rocshmem_ctx_t), hipMemcpyDeviceToHost);
  *a=(long long)h.ctx_opaque; *b=(long long)h.team_opaque;
}
int  rs_my_pe(){ return rocshmem_my_pe(); }
int  rs_n_pes(){ return rocshmem_n_pes(); }
long long rs_malloc(size_t n){ return (long long)rocshmem_malloc(n); }
long long rs_ptr(long long p, int pe){ return (long long)rocshmem_ptr((void*)p, pe); }
void rs_barrier(){ rocshmem_barrier_all(); }
void rs_finalize(){ rocshmem_finalize(); }

// ---- NEW: host-driven RO cross-node primitives -----------------------------
// 1 if peer `pe` is NOT directly addressable via IPC/XGMI (i.e. inter-node and
// must go through RO put/get); 0 if intra-node (rs_ptr works).
int rs_is_remote(long long base, int pe){
  return rocshmem_ptr((void*)base, pe) == nullptr ? 1 : 0;
}
// Blocking host put: copy `nbytes` from local symmetric `src` into the
// symmetric object `dest` on PE `pe`. Forwarded by the RO conduit over the
// transport from the host CPU (no device polling).
void rs_putmem(long long dest, long long src, size_t nbytes, int pe){
  rocshmem_putmem((void*)dest, (const void*)src, nbytes, pe);
}
// Blocking host get: pull `nbytes` from symmetric `src` on PE `pe` into local
// symmetric `dest`.
void rs_getmem(long long dest, long long src, size_t nbytes, int pe){
  rocshmem_getmem((void*)dest, (const void*)src, nbytes, pe);
}
// Host int store to a remote symmetric int (used for the scatter-accumulate
// request/ack handshake on the cross-node path).
void rs_int_p(long long dest, int value, int pe){
  rocshmem_int_p((int*)dest, value, pe);
}
void rs_quiet(){ rocshmem_quiet(); }
void rs_fence(){ rocshmem_fence(); }

}  // extern "C"
