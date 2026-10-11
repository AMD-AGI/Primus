// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// See LICENSE for license information.

// Exact BF16 output-head wgrad: C += dY.T @ X on the caller stream.
// Algorithm reconstruction only; no tuning, synchronization, or gradient reset.
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

namespace {
constexpr size_t kWorkspace = 32u * 1024u * 1024u;
thread_local char last_error[1024] = {};
int fail(int code, const char* where) {
    std::snprintf(last_error, sizeof(last_error), "%s (status=%d)", where, code);
    return code;
}
int blas(hipblasStatus_t s, const char* where) {
    return s == HIPBLAS_STATUS_SUCCESS ? 0 : fail(static_cast<int>(s), where);
}
template <class F> int boundary(F&& f) noexcept {
    last_error[0] = 0;
    try { return f(); }
    catch (const std::exception& e) { return fail(-4, e.what()); }
    catch (...) { return fail(-4, "unknown C++ exception"); }
}
#define LT_CHECK(expr) do { int rc = blas((expr), #expr); if (rc) return rc; } while (0)
struct Candidate {
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace = 0;
    int index = -1;
    int heuristic_rank = -1;  // -1 means reconstructed by index.
};
struct Context {
    int device = -1;
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    std::vector<Candidate> candidates;
    int release() noexcept {
        int first = 0;
        auto take = [&](hipblasStatus_t s) { if (!first && s != HIPBLAS_STATUS_SUCCESS) first = static_cast<int>(s); };
        if (operation) { take(hipblasLtMatmulDescDestroy(operation)); operation = nullptr; }
        if (d) { take(hipblasLtMatrixLayoutDestroy(d)); d = nullptr; }
        if (c) { take(hipblasLtMatrixLayoutDestroy(c)); c = nullptr; }
        if (b) { take(hipblasLtMatrixLayoutDestroy(b)); b = nullptr; }
        if (a) { take(hipblasLtMatrixLayoutDestroy(a)); a = nullptr; }
        if (handle) { take(hipblasLtDestroy(handle)); handle = nullptr; }
        return first;
    }
    ~Context() { release(); }
};
int check_device(Context* ctx) {
    if (!ctx) return fail(-1, "null context");
    int current = -1;
    const auto s = hipGetDevice(&current);
    if (s != hipSuccess) return fail(10000 + static_cast<int>(s), "hipGetDevice");
    return current == ctx->device ? 0 : fail(-2, "caller must select context device");
}
int support(Context* ctx, Candidate& candidate) {
    const float one = 1.0f;
    LT_CHECK(hipblaslt_ext::matmulIsAlgoSupported(ctx->handle, ctx->operation,
        &one, ctx->a, ctx->b, &one, ctx->c, ctx->d,
        candidate.algo, candidate.workspace));
    if (candidate.workspace > kWorkspace) return fail(-3, "candidate exceeds 32 MiB");
    candidate.index = hipblaslt_ext::getIndexFromAlgo(candidate.algo);
    return candidate.index >= 0 ? 0 : fail(-3, "invalid algorithm index");
}
Candidate* candidate_at(Context* ctx, int slot) {
    return ctx && slot >= 0 && static_cast<size_t>(slot) < ctx->candidates.size()
        ? &ctx->candidates[slot] : nullptr;
}
} // namespace

extern "C" {
// Error string belongs to the calling host thread; read before another ABI call.
const char* hw_last_error() noexcept { return last_error; }
uint64_t hw_workspace_limit() noexcept { return kWorkspace; }
// Appended ABI guard: prevents accidentally loading the older FP32-output library.
uint32_t hw_output_dtype_code() noexcept { return 1; } // 1 = BF16 C/D, compute FP32

// Does not change the current device. Caller must select device before creation.
int hw_create(int device, void** output) noexcept {
    return boundary([&]() -> int {
        if (!output) return fail(-1, "null output context pointer");
        *output = nullptr;
        auto ctx = std::make_unique<Context>();
        ctx->device = device;
        int rc = check_device(ctx.get()); if (rc) return rc;
        LT_CHECK(hipblasLtCreate(&ctx->handle));
        // Column-major views of contiguous row-major X, dY and main_grad.
        LT_CHECK(hipblasLtMatrixLayoutCreate(&ctx->a, HIP_R_16BF, 4096, 32768, 4096));
        LT_CHECK(hipblasLtMatrixLayoutCreate(&ctx->b, HIP_R_16BF, 128256, 32768, 128256));
        LT_CHECK(hipblasLtMatrixLayoutCreate(&ctx->c, HIP_R_16BF, 4096, 128256, 4096));
        LT_CHECK(hipblasLtMatrixLayoutCreate(&ctx->d, HIP_R_16BF, 4096, 128256, 4096));
        LT_CHECK(hipblasLtMatmulDescCreate(&ctx->operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t op_a = HIPBLAS_OP_N, op_b = HIPBLAS_OP_T;
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        LT_CHECK(hipblasLtMatmulDescSetAttribute(ctx->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(ctx->operation, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(ctx->operation, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        *output = ctx.release();
        return 0;
    });
}

// Reconstruct a saved library-version-specific index, support-check it for this
// exact descriptor and replace list with ONE candidate at slot 0.
int hw_load_index(void* context, int index) noexcept {
    return boundary([&]() -> int {
        auto* ctx = static_cast<Context*>(context);
        int rc = check_device(ctx); if (rc) return rc;
        if (index < 0) return fail(-1, "negative algorithm index");
        std::vector<int> indices{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> found;
        LT_CHECK(hipblaslt_ext::getAlgosFromIndex(ctx->handle, indices, found));
        if (found.size() != 1 || found[0].state != HIPBLAS_STATUS_SUCCESS) return fail(-3, "index did not reconstruct exactly one algorithm");
        Candidate c; c.algo = found[0].algo;
        rc = support(ctx, c); if (rc) return rc;
        if (c.index != index) return fail(-3, "reconstructed index mismatch");
        ctx->candidates.assign(1, c);
        return 0;
    });
}

int hw_candidate(void* context, int slot, int* index, uint64_t* workspace, int* heuristic_rank) noexcept {
    return boundary([&]() -> int {
        auto* c = candidate_at(static_cast<Context*>(context), slot);
        if (!c || !index || !workspace || !heuristic_rank) return fail(-1, "invalid candidate metadata arguments");
        *index = c->index; *workspace = c->workspace; *heuristic_rank = c->heuristic_rank;
        return 0;
    });
}

// which=0 solution, which=1 kernel. needed includes NUL; query with buffer=null,
// capacity=0, then allocate needed bytes. Short buffer returns -5, no truncation.
int hw_name(void* context, int slot, int which, char* buffer, uint64_t capacity, uint64_t* needed) noexcept {
    return boundary([&]() -> int {
        auto* ctx = static_cast<Context*>(context);
        auto* c = candidate_at(ctx, slot);
        if (!c || !needed || (which != 0 && which != 1)) return fail(-1, "invalid name arguments");
        int rc = check_device(ctx); if (rc) return rc;
        const auto name = which == 0 ? hipblaslt_ext::getSolutionNameFromAlgo(ctx->handle, c->algo)
                                     : hipblaslt_ext::getKernelNameFromAlgo(ctx->handle, c->algo);
        *needed = name.size() + 1;
        if (!buffer && capacity == 0) return 0;
        if (!buffer || capacity < *needed) return fail(-5, "name buffer too small");
        std::memcpy(buffer, name.c_str(), name.size() + 1);
        return 0;
    });
}

// Enqueues precisely one update: C += dY.T @ X. No synchronization or reset.
// workspace must be 16-byte aligned (256-byte alignment recommended). Buffers,
// stream and context must remain alive until caller synchronizes completion.
int hw_run(void* context, int slot, const void* x, const void* dy, void* cd,
           void* workspace, uint64_t workspace_bytes, void* stream) noexcept {
    return boundary([&]() -> int {
        auto* ctx = static_cast<Context*>(context);
        auto* c = candidate_at(ctx, slot);
        if (!c || !x || !dy || !cd) return fail(-1, "invalid execution arguments");
        int rc = check_device(ctx); if (rc) return rc;
        if (workspace_bytes < c->workspace || (c->workspace && !workspace)) return fail(-1, "insufficient workspace");
        if (workspace && reinterpret_cast<uintptr_t>(workspace) % 16) return fail(-1, "unaligned workspace");
        const float one = 1.0f;
        LT_CHECK(hipblasLtMatmul(ctx->handle, ctx->operation, &one,
            x, ctx->a, dy, ctx->b, &one, cd, ctx->c, cd, ctx->d,
            &c->algo, workspace, c->workspace, reinterpret_cast<hipStream_t>(stream)));
        return 0;
    });
}

// Caller must synchronize outstanding use first. On current-device mismatch,
// context remains alive so caller can select its device and retry destruction.
int hw_destroy(void* context) noexcept {
    return boundary([&]() -> int {
        if (!context) return 0;
        auto* ctx = static_cast<Context*>(context);
        int rc = check_device(ctx); if (rc) return rc;
        rc = ctx->release(); delete ctx;
        return rc ? fail(rc, "context cleanup") : 0;
    });
}
} // extern C
