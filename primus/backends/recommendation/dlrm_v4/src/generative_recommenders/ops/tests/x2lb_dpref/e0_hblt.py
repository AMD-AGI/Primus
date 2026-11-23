"""e0_hblt.py -- E0 lab kit: a SCRATCH COPY of ship14's ctypes hipBLASLt path (c68395d
generative_recommenders/ops/hipblaslt_fp8_nn.py, sha256 606aea41..., read only), generalized for measurement:

  * requestedAlgoCount N (production asks for ONE heuristic algo, hipblaslt_fp8_nn.py:215-218) and every returned
    solution kept: index (hipblaslt_ext::getIndexFromAlgo semantics = the int32 at algo.data[0:4]; cross-checked with
    the ext call when the shim is built), workspace, wavesCount, the 24-byte algo struct (serializable, hipblaslt.h)
  * the same enum values, structure layouts, descriptor calls and library resolution as production (the libhipblaslt
    torch loaded: same copy, same kernel database)
  * two problem builders:
      nnb  -- EXACTLY production's NN-b call (fp8_wgrad_nn(dh8, a1_tm, dh_scale, a1_scale)): m = 512, n = 256, k = M,
              A = dh8 e5m2 op N lda 512, B = a1_tm e4m3 op N ldb Mp, C = D fp32 ld 512 (one layout, one pointer),
              A scale = dh_scale, B scale = a1_scale, alpha 1, beta 0
      smm  -- what torch._scaled_mm(act, w_colmajor, scale_act, scale_w, bias, out bf16) does on ROCm, per ship14's
              ops/triton/hipblaslt_scaled_mm_bias.cpp ("reproduces, call for call, what torch._scaled_mm does"):
              D[N, M] = W^T (op T) x X (op N), A = weight [K, N] ld K, B = activation [K, M] ld K, C/D [N, M] ld N,
              A scale = weight scale, B scale = activation scale, optional bf16 bias epilogue, scalar scale modes
  * optional ext shim (e0_hblt_ext.cpp, built at run time with the host compiler): kernel / solution names, the full
    supported set (getAllAlgos + matmulIsAlgoSupported), getAlgosFromIndex, GemmTuning split-K / wgm variants
  * hipblaslt-bench probe (only a fallback for split-K when the shim cannot be built)
Nothing here changes model code; E0 is measurement only.
"""
import ctypes
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

# hipBLASLt / HIP enum values (hipblaslt.h, hipblas-common.h, library_types.h of ROCm 7.2; production's values plus the
# epilogue / bias / scale-mode attributes of hipblaslt_scaled_mm_bias.cpp)
OP_N, OP_T = 111, 112
COMPUTE_32F = 2
R_32F, R_16F, R_16BF, R_8F_E4M3, R_8F_E5M2 = 0, 2, 14, 28, 29
DESC_TRANSA, DESC_TRANSB, DESC_EPILOGUE, DESC_BIAS_POINTER, DESC_BIAS_DATA_TYPE = 0, 1, 2, 3, 4
DESC_A_SCALE_POINTER, DESC_B_SCALE_POINTER = 5, 6
DESC_A_SCALE_MODE, DESC_B_SCALE_MODE = 31, 32
EPILOGUE_DEFAULT, EPILOGUE_BIAS = 1, 4
PREF_MAX_WORKSPACE_BYTES = 1
MiB = 1 << 20
PROD_NNB_WS = 64 * MiB      # DLRM_HSTU_FP8_HBLT_NN_WORKSPACE_MB default (hipblaslt_fp8_nn.py:61-63)
TORCH_WS = 76 * MiB         # torch's process-wide hipBLASLt workspace on ROCm (triton_fp8.py:142-143 comment)
DP_WS = 32 * MiB            # B_fp8_gemm DESIGN section 2 / triton_fp8.py U_DP: < 63.75 MiB forces the data-parallel grid
STATUS = {0: "SUCCESS", 1: "NOT_INITIALIZED", 2: "ALLOC_FAILED", 3: "INVALID_VALUE", 4: "MAPPING_ERROR",
          5: "EXECUTION_FAILED", 6: "INTERNAL_ERROR", 7: "NOT_SUPPORTED", 8: "ARCH_MISMATCH", 9: "HANDLE_IS_NULLPTR",
          10: "INVALID_ENUM", 11: "UNKNOWN", -100: "ext: Gemm constructor threw", -101: "ext: workspace too small",
          -102: "ext: exception"}


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


assert ctypes.sizeof(_Algo) == 24 and ctypes.sizeof(_HeuristicResult) == 56


def algo_from_hex(h):
    a = _Algo()
    ctypes.memmove(ctypes.addressof(a), bytes.fromhex(h), ctypes.sizeof(a))
    return a


def algo_hex(a):
    return ctypes.string_at(ctypes.addressof(a), ctypes.sizeof(a)).hex()


def algo_hex_with_ws(h, ws_bytes):
    """The same 24-byte algo with max_workspace_bytes set to the workspace the call passes (review F7: a solution
    returned by the 256 MiB query and run with 64 MiB must look exactly like the 64 MiB query's copy, which is what a
    production pin would hold; whether hipblasLtMatmul honours the struct field or the passed size is [H])."""
    a = algo_from_hex(h)
    a.max_workspace_bytes = int(ws_bytes)
    return algo_hex(a)


def algo_index(a):
    """getIndexFromAlgo semantics: the int32 stored first in algo.data (-1 if negative)."""
    v = int.from_bytes(bytes(a.data[0:4]), "little", signed=True)
    return v if v >= 0 else -1


def rec_of(res, rank, source):
    a = res.algo
    return dict(index=algo_index(a), algo=algo_hex(a), ws=int(res.workspaceSize), waves=round(float(res.wavesCount), 4),
                state=int(res.state), rank=rank, source=source)


class HipBLASLt:
    """The hipBLASLt C API through ctypes on torch's library copy (production's _library())."""

    def __init__(self, torch):
        self.torch = torch
        self.lib = self._library()
        self._handles, self._ws = {}, {}
        self.ext = None
        self.ext_info = dict(built=False)

    # ------------------------------------------------------------------ library (as production)
    def _library(self):
        torch = self.torch
        path = None
        for line in open("/proc/self/maps"):
            p = line.split()[-1]
            if "/libhipblaslt.so" in p:
                path = p
                break
        if path is None:
            cand = os.path.join(os.path.dirname(torch.__file__), "lib", "libhipblaslt.so")
            path = cand if os.path.exists(cand) else "libhipblaslt.so"
        self.path = path
        lib = ctypes.CDLL(path)
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        lib.hipblasLtCreate.argtypes = [ctypes.POINTER(vp)]
        lib.hipblasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_uint64,
                                                    ctypes.c_uint64, ctypes.c_int64]
        lib.hipblasLtMatrixLayoutDestroy.argtypes = [vp]
        lib.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int]
        lib.hipblasLtMatmulDescDestroy.argtypes = [vp]
        lib.hipblasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
        lib.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(vp)]
        lib.hipblasLtMatmulPreferenceDestroy.argtypes = [vp]
        lib.hipblasLtMatmulPreferenceSetAttribute.argtypes = [vp, ctypes.c_int, vp, sz]
        lib.hipblasLtMatmulAlgoGetHeuristic.argtypes = [vp, vp, vp, vp, vp, vp, vp, ctypes.c_int,
                                                        ctypes.POINTER(_HeuristicResult), ctypes.POINTER(ctypes.c_int)]
        lib.hipblasLtMatmul.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.POINTER(_Algo), vp, sz, vp]
        try:
            lib.hipblasLtGetVersion.argtypes = [vp, ctypes.POINTER(ctypes.c_int)]
        except AttributeError:
            pass
        return lib

    def handle(self):
        torch = self.torch
        idx = torch.cuda.current_device()
        if idx not in self._handles:
            h = ctypes.c_void_p()
            _check(self.lib.hipblasLtCreate(ctypes.byref(h)), "Create")
            self._handles[idx] = h
        return self._handles[idx]

    def version(self):
        v = ctypes.c_int(0)
        try:
            st = self.lib.hipblasLtGetVersion(self.handle(), ctypes.byref(v))
            return dict(status=st, version=v.value, path=self.path)
        except Exception as ex:  # noqa
            return dict(error=repr(ex)[:200], path=self.path)

    def workspace(self, nbytes):
        """One uint8 device tensor per size (production allocates its 64 MiB once per device)."""
        if nbytes not in self._ws:
            self._ws[nbytes] = self.torch.empty(max(1, nbytes), dtype=self.torch.uint8, device="cuda")
        return self._ws[nbytes]

    def stream(self):
        return self.torch.cuda.current_stream().cuda_stream

    # ------------------------------------------------------------------ problems
    def nnb(self, dh8, a1_tm, dh_scale, a1_scale, out=None):
        """Production's NN-b (hipblaslt_fp8_nn.fp8_wgrad_nn(y=dh8, dout_t=a1_tm, scale_y=dh_scale, scale_d=a1_scale)):
        returns a Problem whose output is dW2^T fp32 [256, 512]."""
        torch = self.torch
        assert dh8.dim() == 2 and a1_tm.dim() == 2 and dh8.stride(1) == 1 and a1_tm.stride(1) == 1
        m_tok, c = dh8.shape
        n, mp = a1_tm.shape
        assert mp >= m_tok and dh8.stride(0) == c
        out = torch.empty((n, c), dtype=torch.float32, device=dh8.device) if out is None else out
        return Problem(self, "nnb", m=c, n=n, k=m_tok, types=(_hip(torch, dh8.dtype), _hip(torch, a1_tm.dtype), R_32F,
                                                              R_32F),
                       ops=(OP_N, OP_N), lds=(c, a1_tm.stride(0), c, c), A=dh8, B=a1_tm, D=out,
                       scale_a=dh_scale, scale_b=a1_scale, keep=(dh8, a1_tm, dh_scale, a1_scale), same_cd=True)

    def smm(self, act, w_col, scale_act, scale_w, bias=None, out=None, c_type=None):
        """torch._scaled_mm(act [M, K] row-major, w_col [K, N] column-major (stride (1, K)), scale_act, scale_w,
        bias bf16 [N] or None, out bf16 [M, N]) as torch issues it on ROCm (see the module docstring).
        c_type: the C layout's type (default bf16 = D's); the replica search in e0_l2fd.py also tries fp16."""
        torch = self.torch
        M, K = act.shape
        K2, N = w_col.shape
        assert K == K2 and act.is_contiguous() and w_col.stride() == (1, K), (act.shape, w_col.stride())
        out = torch.empty((M, N), dtype=torch.bfloat16, device=act.device) if out is None else out
        return Problem(self, "smm", m=N, n=M, k=K, types=(_hip(torch, w_col.dtype), _hip(torch, act.dtype),
                                                          R_16BF if c_type is None else c_type, R_16BF),
                       ops=(OP_T, OP_N), lds=(K, K, N, N), A=w_col, B=act, D=out, scale_a=scale_w, scale_b=scale_act,
                       bias=bias, keep=(act, w_col, scale_act, scale_w, bias), same_cd=False)

    # ------------------------------------------------------------------ heuristic / run
    def heuristic(self, prob, ws_bytes, count, source=None):
        """hipblasLtMatmulAlgoGetHeuristic(requestedAlgoCount=count) with MAX_WORKSPACE_BYTES = ws_bytes.
        Returns (status, [algo records in the library's order], n_bad_state)."""
        lib = self.lib
        pref = ctypes.c_void_p()
        _check(lib.hipblasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PrefCreate")
        wsb = ctypes.c_uint64(int(ws_bytes))
        _check(lib.hipblasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, ctypes.byref(wsb), 8),
               "PrefWs")
        res = (_HeuristicResult * int(count))()
        n = ctypes.c_int(0)
        st = lib.hipblasLtMatmulAlgoGetHeuristic(self.handle(), prob.desc, prob.la, prob.lb, prob.lc, prob.ld, pref,
                                                 int(count), res, ctypes.byref(n))
        lib.hipblasLtMatmulPreferenceDestroy(pref)
        src = source or f"heur{count}@{ws_bytes // MiB}MiB"
        recs = [rec_of(res[i], i, src) for i in range(max(0, n.value))]
        good = [r for r in recs if r["state"] == 0]
        return int(st), good, len(recs) - len(good)

    def run(self, prob, rec, ws_bytes, alpha=1.0, beta=0.0):
        """One hipblasLtMatmul with the record's algo on the current stream; returns the status (0 = launched)."""
        a = algo_from_hex(rec["algo"]) if isinstance(rec, dict) else rec
        ws = self.workspace(ws_bytes)
        al, be = ctypes.c_float(alpha), ctypes.c_float(beta)
        c_ptr = prob.D.data_ptr()
        return int(self.lib.hipblasLtMatmul(
            self.handle(), prob.desc, ctypes.byref(al), ctypes.c_void_p(prob.A.data_ptr()), prob.la,
            ctypes.c_void_p(prob.B.data_ptr()), prob.lb, ctypes.byref(be), ctypes.c_void_p(c_ptr), prob.lc,
            ctypes.c_void_p(prob.D.data_ptr()), prob.ld, ctypes.byref(a), ctypes.c_void_p(ws.data_ptr()),
            ctypes.c_size_t(int(ws_bytes)), ctypes.c_void_p(self.stream())))

    def caller(self, prob, rec, ws_bytes):
        """A no-argument callable for the timers (raises on a non-zero status)."""
        a = algo_from_hex(rec["algo"])
        ws = self.workspace(ws_bytes)
        lib, h = self.lib, self.handle()
        al, be = ctypes.c_float(1.0), ctypes.c_float(0.0)
        A, B, D = ctypes.c_void_p(prob.A.data_ptr()), ctypes.c_void_p(prob.B.data_ptr()), ctypes.c_void_p(
            prob.D.data_ptr())
        wsp, wsz = ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(int(ws_bytes))
        st_ptr = ctypes.c_void_p(self.stream())

        def call():
            st = lib.hipblasLtMatmul(h, prob.desc, ctypes.byref(al), A, prob.la, B, prob.lb, ctypes.byref(be), D,
                                     prob.lc, D, prob.ld, ctypes.byref(a), wsp, wsz, st_ptr)
            if st != 0:
                raise RuntimeError(f"hipblasLtMatmul status {st} ({STATUS.get(st)})")
        return call

    # ------------------------------------------------------------------ ext shim
    def build_ext(self, out_dir, timeout=600, include_dirs=("/opt/rocm/include",)):
        """Compile e0_hblt_ext.cpp (host only) and load it; links against the libhipblaslt torch loaded.
        include_dirs: the container's ROCm headers (the CPU dry run passes the host's ROCm 7.2.0 headers)."""
        src = os.path.join(HERE, "e0_hblt_ext.cpp")
        so = os.path.join(out_dir, "libe0hblt.so")
        os.makedirs(out_dir, exist_ok=True)
        tl = os.path.dirname(self.path)
        inc = [f"-I{p}" for p in include_dirs if os.path.isdir(p)]
        tries = []
        for cxx in (shutil.which("g++"), shutil.which("c++"), "/opt/rocm/llvm/bin/clang++",
                    "/opt/rocm/lib/llvm/bin/clang++"):
            if not cxx or not os.path.isfile(cxx):
                continue
            cmd = [cxx, "-std=c++17", "-O2", "-fPIC", "-shared", "-D__HIP_PLATFORM_AMD__=1", *inc, src, "-o", so,
                   f"-L{tl}", "-l:" + os.path.basename(self.path), f"-Wl,-rpath,{tl}"]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                tries.append(dict(cxx=cxx, rc=p.returncode, s=round(time.time() - t0, 1), log=(p.stderr or "")[-1500:]))
                if p.returncode == 0 and os.path.isfile(so):
                    break
            except Exception as ex:  # noqa
                tries.append(dict(cxx=cxx, error=repr(ex)[:300]))
        self.ext_info = dict(built=os.path.isfile(so), so=so, tries=tries)
        if not os.path.isfile(so):
            return self.ext_info
        return self.load_ext(so)

    def load_ext(self, so):
        """Load an already built shim (a child process never rebuilds: the parent has the file mapped)."""
        self.ext_info.setdefault("so", so)
        try:
            ext = ctypes.CDLL(so)
            sizes = (ctypes.c_int * 2)()
            ext.e0x_sizes(sizes)
            ok = ext.e0x_abi() == 1 and list(sizes) == [ctypes.sizeof(_Algo), ctypes.sizeof(_HeuristicResult)]
            self.ext_info.update(abi=ext.e0x_abi(), sizes=list(sizes), loaded=ok)
            if ok:
                vp, sz = ctypes.c_void_p, ctypes.c_size_t
                ext.e0x_index.argtypes = [vp]
                ext.e0x_names.argtypes = [vp, vp, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
                ext.e0x_all_algos.argtypes = [vp] + [ctypes.c_int] * 7 + [vp, ctypes.c_int,
                                                                        ctypes.POINTER(ctypes.c_int)]
                ext.e0x_is_supported.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.POINTER(sz)]
                ext.e0x_algos_from_index.argtypes = [vp, ctypes.POINTER(ctypes.c_int), ctypes.c_int, vp, ctypes.c_int,
                                                     ctypes.POINTER(ctypes.c_int)]
                ext.e0x_gemm_create.restype = vp
                ext.e0x_gemm_create.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.c_int, ctypes.c_int,
                                                vp, sz, ctypes.POINTER(sz), ctypes.POINTER(ctypes.c_int),
                                                ctypes.c_char_p, ctypes.c_int]
                ext.e0x_gemm_run.argtypes = [vp, vp]
                ext.e0x_gemm_destroy.argtypes = [vp]
                self.ext = ext
        except Exception as ex:  # noqa
            self.ext_info.update(loaded=False, load_error=repr(ex)[:300])
        return self.ext_info

    def ext_index(self, rec):
        if self.ext is None:
            return None
        a = algo_from_hex(rec["algo"])
        return int(self.ext.e0x_index(ctypes.byref(a)))

    def names(self, rec):
        """(kernel name, solution name) of an algo via the ext API, or (None, None)."""
        if self.ext is None:
            return None, None
        a = algo_from_hex(rec["algo"])
        kb, sb = ctypes.create_string_buffer(1024), ctypes.create_string_buffer(1024)
        st = self.ext.e0x_names(self.handle(), ctypes.byref(a), kb, 1024, sb, 1024)
        if st != 0:
            return None, None
        return kb.value.decode(errors="replace") or None, sb.value.decode(errors="replace") or None

    def all_supported(self, prob, max_n=20000):
        """(n_total, [records of every algo of this type combination the problem supports, with its workspace])."""
        if self.ext is None:
            return None, []
        arr = (_HeuristicResult * max_n)()
        n = ctypes.c_int(0)
        st = self.ext.e0x_all_algos(self.handle(), prob.ops[0], prob.ops[1], *prob.types, COMPUTE_32F, arr, max_n,
                                    ctypes.byref(n))
        if st != 0:
            return dict(status=st, n=n.value), []
        al, be = ctypes.c_float(1.0), ctypes.c_float(0.0)
        out = []
        for i in range(min(n.value, max_n)):
            a = _Algo()
            ctypes.memmove(ctypes.addressof(a), ctypes.addressof(arr[i].algo), ctypes.sizeof(a))
            w = ctypes.c_size_t(0)
            s = self.ext.e0x_is_supported(self.handle(), prob.desc, ctypes.byref(al), prob.la, prob.lb,
                                          ctypes.byref(be), prob.lc, prob.ld, ctypes.byref(a), ctypes.byref(w))
            if s == 0:
                out.append(dict(index=algo_index(a), algo=algo_hex(a), ws=int(w.value), waves=None, state=0, rank=i,
                                source="getAllAlgos+isSupported"))
        return dict(status=0, n=n.value, n_supported=len(out)), out

    def from_index(self, indices):
        if self.ext is None:
            return []
        arr = (ctypes.c_int * len(indices))(*indices)
        res = (_HeuristicResult * max(1, len(indices)))()
        n = ctypes.c_int(0)
        st = self.ext.e0x_algos_from_index(self.handle(), arr, len(indices), res, len(indices), ctypes.byref(n))
        return [rec_of(res[i], i, "getAlgosFromIndex") for i in range(n.value)] if st == 0 else []

    def tuned(self, prob, rec, splitk, wgm, ws_bytes):
        """A GemmTuning instance (split-K / wgm) of rec on prob: returns dict(ok, run (callable), kernel, ws_need,
        status, destroy) or dict(ok=False, status, ...)."""
        if self.ext is None:
            return dict(ok=False, status="no ext shim")
        a = algo_from_hex(rec["algo"])
        ws = self.workspace(ws_bytes)
        need, st = ctypes.c_size_t(0), ctypes.c_int(0)
        kb = ctypes.create_string_buffer(1024)
        h = self.ext.e0x_gemm_create(self.handle(), prob.desc, ctypes.c_void_p(prob.A.data_ptr()), prob.la,
                                     ctypes.c_void_p(prob.B.data_ptr()), prob.lb, ctypes.c_void_p(prob.D.data_ptr()),
                                     prob.lc, ctypes.c_void_p(prob.D.data_ptr()), prob.ld, ctypes.byref(a), int(splitk),
                                     int(wgm), ctypes.c_void_p(ws.data_ptr()), ctypes.c_size_t(int(ws_bytes)),
                                     ctypes.byref(need), ctypes.byref(st), kb, 1024)
        if not h:
            return dict(ok=False, status=int(st.value), status_name=STATUS.get(int(st.value)), ws_need=int(need.value))
        ext, sp = self.ext, ctypes.c_void_p(self.stream())
        hv = ctypes.c_void_p(h)

        def call():
            s = ext.e0x_gemm_run(hv, sp)
            if s != 0:
                raise RuntimeError(f"tuned run status {s}")
        return dict(ok=True, run=call, kernel=kb.value.decode(errors="replace"), ws_need=int(need.value), status=0,
                    destroy=lambda: ext.e0x_gemm_destroy(hv))


class Problem:
    """Matmul descriptor + layouts for one problem; kept alive for the run (destroyed by close())."""

    def __init__(self, hb, kind, m, n, k, types, ops, lds, A, B, D, scale_a, scale_b, bias=None, keep=(),
                 same_cd=True):
        self.hb, self.kind, self.m, self.n, self.k = hb, kind, m, n, k
        self.types, self.ops, self.lds = types, ops, lds
        self.A, self.B, self.D, self.keep = A, B, D, keep
        lib = hb.lib
        vp = ctypes.c_void_p
        self.desc, self.la, self.lb, self.ld = vp(), vp(), vp(), vp()
        _check(lib.hipblasLtMatmulDescCreate(ctypes.byref(self.desc), COMPUTE_32F, R_32F), "DescCreate")
        for attr, op in ((DESC_TRANSA, ops[0]), (DESC_TRANSB, ops[1])):
            o = ctypes.c_int(op)
            _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, attr, ctypes.byref(o), 4), "SetTrans")
        self._sa, self._sb = ctypes.c_void_p(scale_a.data_ptr()), ctypes.c_void_p(scale_b.data_ptr())
        _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_A_SCALE_POINTER, ctypes.byref(self._sa), 8),
               "SetScaleA")
        _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_B_SCALE_POINTER, ctypes.byref(self._sb), 8),
               "SetScaleB")
        self.attrs = dict(scale_mode=None, bias=bias is not None)
        if kind == "smm":
            # torch sets the tensor-wise scale mode explicitly when its hipBLASLt has the SCALE_MODE attributes
            # (hipblaslt_scaled_mm_bias.cpp HIPBLASLT_OUTER_VEC block); value 0 = scalar fp32 = the default
            z = ctypes.c_int(0)
            s1 = lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_A_SCALE_MODE, ctypes.byref(z), 4)
            s2 = lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_B_SCALE_MODE, ctypes.byref(z), 4)
            self.attrs["scale_mode"] = (int(s1), int(s2))
        if bias is not None:
            self._bias = ctypes.c_void_p(bias.data_ptr())
            _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_BIAS_POINTER, ctypes.byref(self._bias), 8),
                   "SetBias")
            ep = ctypes.c_uint32(EPILOGUE_BIAS)
            _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_EPILOGUE, ctypes.byref(ep), 4), "SetEpilogue")
            bt = ctypes.c_int(R_16BF)
            _check(lib.hipblasLtMatmulDescSetAttribute(self.desc, DESC_BIAS_DATA_TYPE, ctypes.byref(bt), 4),
                   "SetBiasType")
        ta, tb, tc, td = types
        lda, ldb, ldc, ldd = lds
        if kind == "nnb":
            rows_a, cols_a, rows_b, cols_b = m, k, k, n
        else:  # smm: A = weight [K, N] (op T), B = activation [K, M] (op N)
            rows_a, cols_a, rows_b, cols_b = k, m, k, n
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.la), ta, rows_a, cols_a, lda), "LayoutA")
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.lb), tb, rows_b, cols_b, ldb), "LayoutB")
        _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.ld), td, m, n, ldd), "LayoutD")
        if same_cd and tc == td:
            self.lc = self.ld                # production NN-b passes desc.ld for C and D
        else:
            self.lc = vp()
            _check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(self.lc), tc, m, n, ldc), "LayoutC")
        self.same_cd = self.lc is self.ld

    def describe(self):
        return dict(kind=self.kind, m=self.m, n=self.n, k=self.k, types=list(self.types), ops=list(self.ops),
                    lds=list(self.lds), attrs=self.attrs, same_cd_layout=self.same_cd)

    def close(self):
        lib = self.hb.lib
        lib.hipblasLtMatrixLayoutDestroy(self.la)
        lib.hipblasLtMatrixLayoutDestroy(self.lb)
        lib.hipblasLtMatrixLayoutDestroy(self.ld)
        if not self.same_cd:
            lib.hipblasLtMatrixLayoutDestroy(self.lc)
        lib.hipblasLtMatmulDescDestroy(self.desc)


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"hipBLASLt {what} failed with status {status} ({STATUS.get(status)})")


def _hip(torch, dt):
    return {torch.float8_e4m3fn: R_8F_E4M3, torch.float8_e5m2: R_8F_E5M2, torch.float32: R_32F,
            torch.bfloat16: R_16BF, torch.float16: R_16F}[dt]


def backend(torch):
    """The real ctypes backend, or e0_mock.MockHipBLASLt in the CPU dry run."""
    if os.environ.get("E0_MOCK") == "1":
        import e0_mock  # noqa: E402
        return e0_mock.MockHipBLASLt(torch)
    return HipBLASLt(torch)


# ================================================================================================ hipblaslt-bench
def bench_probe():
    """Path of hipblaslt-bench in the image and the options its --help lists (split-K fallback only)."""
    p = shutil.which("hipblaslt-bench") or ("/opt/rocm/bin/hipblaslt-bench"
                                           if os.path.isfile("/opt/rocm/bin/hipblaslt-bench") else None)
    if os.environ.get("E0_MOCK") == "1":
        p = None
    if not p:
        return dict(path=None)
    try:
        h = subprocess.run([p, "--help"], capture_output=True, text=True, timeout=60)
        txt = h.stdout + h.stderr
    except Exception as ex:  # noqa
        return dict(path=p, error=repr(ex)[:200])
    opts = sorted(set(re.findall(r"--([a-z_]+)", txt)))
    return dict(path=p, has=dict(algo_method="algo_method" in opts, splitk="splitk" in opts or "gsu" in opts,
                                 solution_index="solution_index" in opts, ldb="ldb" in opts,
                                 rotating="rotating" in opts, flush="flush" in opts), n_opts=len(opts))


def bench_nnb_cmd(path, M, Mp, extra=()):
    """hipblaslt-bench line for the NN-b at M (C API, scalar scales; OCP fp8: bf8_r = e5m2, f8_r = e4m3)."""
    return [path, "--api_method", "c", "-m", "512", "-n", "256", "-k", str(M), "--lda", "512", "--ldb", str(Mp),
            "--ldc", "512", "--ldd", "512", "--transA", "N", "--transB", "N", "--a_type", "bf8_r", "--b_type", "f8_r",
            "--c_type", "f32_r", "--d_type", "f32_r", "--compute_type", "f32_r", "--scaleA", "1", "--scaleB", "1",
            "--cold_iters", "5", "--iters", "20", "--rotating", "512", "--use_gpu_timer", *extra]
