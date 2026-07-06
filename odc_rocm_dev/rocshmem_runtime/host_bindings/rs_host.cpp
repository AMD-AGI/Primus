#include <hip/hip_runtime.h>

#include <cstring>
#include <rocshmem/rocshmem.hpp>
using namespace rocshmem;
extern "C" {
int rs_uid_bytes() { return (int)sizeof(rocshmem_uniqueid_t); }
void rs_get_uid(char* out) {
    rocshmem_uniqueid_t uid;
    rocshmem_get_uniqueid(&uid);
    memcpy(out, uid.data(), sizeof(rocshmem_uniqueid_t));
}
void rs_init_uid(int rank, int nranks, const char* bytes) {
    rocshmem_uniqueid_t uid;
    memcpy(uid.data(), bytes, sizeof(rocshmem_uniqueid_t));
    rocshmem_init_attr_t attr;
    rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
}
void rs_get_ctx_fields(long long* a, long long* b) {
    void* dctx = rocshmem_get_device_ctx();
    rocshmem_ctx_t h;
    hipMemcpy(&h, dctx, sizeof(rocshmem_ctx_t), hipMemcpyDeviceToHost);
    *a = (long long)h.ctx_opaque;
    *b = (long long)h.team_opaque;
}
int rs_my_pe() { return rocshmem_my_pe(); }
int rs_n_pes() { return rocshmem_n_pes(); }
long long rs_malloc(size_t n) { return (long long)rocshmem_malloc(n); }
long long rs_ptr(long long p, int pe) { return (long long)rocshmem_ptr((void*)p, pe); }
void rs_barrier() { rocshmem_barrier_all(); }
void rs_finalize() { rocshmem_finalize(); }
}
