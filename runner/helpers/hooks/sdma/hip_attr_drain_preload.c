/*
 * hip_attr_drain_preload.c -- user-space LD_PRELOAD workaround for the
 * "first kernel after ncclMemAlloc fails" bug on ROCm.
 *
 * Background:
 *   RCCL's allocator.cc calls
 *     (void) cuDeviceGetAttribute(&flag,
 *              CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
 *   inside its cuMem code path (NCCL_CUMEM_ENABLE=1). On ROCm 7.x that
 *   attribute id (128) is not supported -- the call returns
 *   hipErrorInvalidValue AND leaves the same error sitting in the HIP
 *   runtime's per-thread `last_error_` slot. RCCL discards the return
 *   value, so nothing drains the TLS. The next unrelated HIP API entry
 *   (e.g. a kernel launch) reads back that stale error and reports it
 *   as if its own operation failed.
 *   Lorri: Check JIRA ticket ROCM-24832 for more details.
 *
 *   On ROCm the macro CUPFN(x) in rocm-systems/projects/rccl/src/include/
 *   rocmwrap.h expands to a literal `x`, so the call site goes through
 *   the normal global symbol -- which makes it directly interposable
 *   via LD_PRELOAD. No RCCL rebuild required.
 *
 * What this interposer does:
 *   - Wraps  hipDeviceGetAttribute  AND  cuDeviceGetAttribute.
 *   - Forwards to the real implementation (resolved via RTLD_NEXT).
 *   - If the real call returns non-success, calls hipGetLastError() to
 *     drain the TLS error slot so it can't leak into the next HIP call.
 *   - Returns the original return value untouched -- callers see the
 *     same failure they would have, they just don't get the TLS pollution.
 *
 * Build (inside the ROCm container):
 *   gcc -O2 -fPIC -shared hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
 *
 * Use:
 *   LD_PRELOAD=/path/to/libhip_attr_drain.so  <your program>
 *
 * Always prints one line per drained call to stderr (intentional --
 * keeps the workaround visible in logs so it can be retired once the
 * underlying ROCm fix lands).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

/* Opaque ABI: hipError_t / CUresult are 32-bit ints, attribute enums are
 * ints, device handles are ints. Using bare int here keeps this .so free
 * of HIP/CUDA header and link dependencies -- it can be built in any
 * container with just gcc + libdl. */
typedef int (*get_attr_fn)(int *, int, int);
typedef int (*get_last_err_fn)(void);

static get_attr_fn      real_hipDeviceGetAttribute = NULL;
static get_attr_fn      real_cuDeviceGetAttribute  = NULL;
static get_last_err_fn  real_hipGetLastError       = NULL;
static void            *amdhip_handle              = NULL;

/* fwd-decl: defined below the resolver, used by maybe_drain */
static void *get_amdhip_handle(void);

__attribute__((constructor))
static void hip_attr_drain_init(void) {
    real_hipDeviceGetAttribute =
        (get_attr_fn)     dlsym(RTLD_NEXT, "hipDeviceGetAttribute");
    real_cuDeviceGetAttribute =
        (get_attr_fn)     dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    real_hipGetLastError =
        (get_last_err_fn) dlsym(RTLD_NEXT, "hipGetLastError");

    fprintf(stderr,
        "[hip_attr_drain] loaded. "
        "hipDeviceGetAttribute=%p cuDeviceGetAttribute=%p "
        "hipGetLastError=%p\n",
        (void *)real_hipDeviceGetAttribute,
        (void *)real_cuDeviceGetAttribute,
        (void *)real_hipGetLastError);
}

static inline void maybe_drain(const char *who, int attr, int rc) {
    if (rc == 0) return;   /* hipSuccess / CUDA_SUCCESS == 0 */
    if (!real_hipGetLastError) {
        /* Same handle-based resolution: bypasses our own symbol space,
         * avoids the RTLD_DEFAULT recursion hazard. */
        void *h = get_amdhip_handle();
        if (h) real_hipGetLastError =
                (get_last_err_fn) dlsym(h, "hipGetLastError");
        if (!real_hipGetLastError)
            real_hipGetLastError =
                (get_last_err_fn) dlsym(RTLD_NEXT, "hipGetLastError");
    }
    int drained = -1;
    if (real_hipGetLastError) drained = real_hipGetLastError();
    fprintf(stderr,
        "[hip_attr_drain] %s(attr=%d) -> %d ; drained TLS = %d%s\n",
        who, attr, rc, drained,
        real_hipGetLastError ? "" : "  (WARN: hipGetLastError unresolved)");
}

/* Lazy resolver. Symbol-interposing wrappers need to be careful:
 *   - dlsym(RTLD_NEXT, ...) is caller-context-sensitive (it returns the
 *     next definition after the caller in the link order). It works
 *     reliably from THIS .so as long as libamdhip64.so was loaded
 *     before us in the loader graph; it can return NULL otherwise.
 *   - dlsym(RTLD_DEFAULT, ...) searches the global symbol table, which
 *     contains OUR OWN wrapper symbol (LD_PRELOAD'd shared objects go
 *     into the global scope). Using it as a fallback would resolve to
 *     ourselves, giving infinite recursion -> stack overflow -> SEGV.
 *     PyTorch 2.12's cuda-bindings cython modules hit exactly that path.
 * So we explicitly dlopen("libamdhip64.so", ...NOLOAD) to get a handle
 * to the real lib (already resident at this point) and dlsym against
 * THAT handle -- guaranteed to bypass our own wrapper. */
static void *get_amdhip_handle(void) {
    if (amdhip_handle) return amdhip_handle;
    /* RTLD_NOLOAD: don't trigger a fresh load, just hand back the
     * existing in-memory handle if libamdhip64 has been loaded by
     * anyone (PyTorch's libtorch_hip.so depends on it). */
    amdhip_handle = dlopen("libamdhip64.so",   RTLD_LAZY | RTLD_NOLOAD);
    if (!amdhip_handle)
        amdhip_handle = dlopen("libamdhip64.so.7", RTLD_LAZY | RTLD_NOLOAD);
    if (!amdhip_handle)
        amdhip_handle = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_NOLOAD);
    /* Last-ditch: NOLOAD off, in case nobody has loaded it yet. */
    if (!amdhip_handle)
        amdhip_handle = dlopen("libamdhip64.so",   RTLD_LAZY);
    return amdhip_handle;
}

static get_attr_fn resolve_attr(get_attr_fn *cache, const char *name) {
    get_attr_fn fn = *cache;
    if (fn) return fn;
    /* Prefer dlsym(handle): bypasses our own wrapper symbol. */
    void *h = get_amdhip_handle();
    if (h) fn = (get_attr_fn) dlsym(h, name);
    /* Fallback to RTLD_NEXT (caller-relative, but at least also bypasses
     * us as long as our .so loaded before libamdhip64 in the chain). */
    if (!fn) fn = (get_attr_fn) dlsym(RTLD_NEXT, name);
    *cache = fn;
    return fn;
}

/* ------------------------------------------------------------------ */
/* hipDeviceGetAttribute -- caught directly by anyone in user code.   */
/* ------------------------------------------------------------------ */
int hipDeviceGetAttribute(int *value, int attrib, int device) {
    get_attr_fn fn = resolve_attr(&real_hipDeviceGetAttribute,
                                  "hipDeviceGetAttribute");
    if (!fn) {
        fprintf(stderr, "[hip_attr_drain] hipDeviceGetAttribute"
                " unresolved; faking hipErrorInvalidValue\n");
        if (value) *value = 0;
        return 1;   /* hipErrorInvalidValue -- safe, matches behavior for
                     * unsupported attributes; caller code that ignores
                     * the return (RCCL's allocator.cc) is unaffected. */
    }
    int rc = fn(value, attrib, device);
    maybe_drain("hipDeviceGetAttribute", attrib, rc);
    return rc;
}

/* ------------------------------------------------------------------ */
/* cuDeviceGetAttribute -- the symbol that RCCL's allocator.cc calls
 *    directly on ROCm (CUPFN(x) expands to `x`). This is the symbol we
 *    actually need to catch to fix the NCCL_CUMEM_ENABLE=1 bug.       */
/* ------------------------------------------------------------------ */
int cuDeviceGetAttribute(int *value, int attrib, int device) {
    get_attr_fn fn = resolve_attr(&real_cuDeviceGetAttribute,
                                  "cuDeviceGetAttribute");
    if (!fn) {
        fprintf(stderr, "[hip_attr_drain] cuDeviceGetAttribute"
                " unresolved; faking hipErrorInvalidValue\n");
        if (value) *value = 0;
        return 1;
    }
    int rc = fn(value, attrib, device);
    maybe_drain("cuDeviceGetAttribute", attrib, rc);
    return rc;
}
