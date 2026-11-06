// LD_PRELOAD diagnostic: log every code-object load of the HIP runtime with wall time and duration.
//   kpack_load_code_object (TheRock kpack: binary path + TU index), hsa_code_object_reader_create_from_memory
//   (pointer resolved with dladdr -> library + offset), hsa_executable_load_agent_code_object (load duration).
// Env COLOAD_LOG_DIR=<dir>: one file per process, <dir>/coload.<pid>.log. Unset: pass-through, no file.
// A 'reader' line also names the code object's first kernel (kd=<name>.kd, nkd=<count>) and its size / FNV-1a.
// Diagnostic only (never in production). Input of gen_hip_preload_table.py --loads (T12, docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md).
//   build (in the image):  gcc -O2 -fPIC -shared -Wall -o coload_shim.so tools/recommendation/dlrm_v4/coload_shim.c -ldl -lpthread
//   use:                   LD_PRELOAD=<dir>/coload_shim.so COLOAD_LOG_DIR=<log dir> <training command>
//   regenerate the table:  gen_hip_preload_table.py --lib ... --trace 1n=<traces> --loads '1n=<log dir>/coload.*.log'
//                          --run-start 1n=<that run's mlperf.log> --stack <label> --out <table>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>


#include <elf.h>
// first kernel descriptor symbol ("<name>.kd") of an in-memory ELF code object, and how many it has
static const char* first_kd(const unsigned char* p, size_t size, int* nkd) {
  *nkd = 0;
  if (!p || size < sizeof(Elf64_Ehdr) || memcmp(p, ELFMAG, SELFMAG) != 0 || p[EI_CLASS] != ELFCLASS64) return "-";
  const Elf64_Ehdr* eh = (const Elf64_Ehdr*)p;
  if (eh->e_shoff == 0 || eh->e_shentsize != sizeof(Elf64_Shdr) || eh->e_shoff + (size_t)eh->e_shnum * sizeof(Elf64_Shdr) > size) return "-";
  const Elf64_Shdr* sh = (const Elf64_Shdr*)(p + eh->e_shoff);
  const char* best = "-";
  for (int i = 0; i < eh->e_shnum; i++) {
    if (sh[i].sh_type != SHT_SYMTAB && sh[i].sh_type != SHT_DYNSYM) continue;
    if (sh[i].sh_link >= eh->e_shnum || sh[i].sh_entsize != sizeof(Elf64_Sym)) continue;
    const Elf64_Shdr* st = &sh[sh[i].sh_link];
    if (sh[i].sh_offset + sh[i].sh_size > size || st->sh_offset + st->sh_size > size) continue;
    const Elf64_Sym* sym = (const Elf64_Sym*)(p + sh[i].sh_offset);
    size_t n = sh[i].sh_size / sizeof(Elf64_Sym);
    for (size_t k = 0; k < n; k++) {
      if (sym[k].st_name >= st->sh_size) continue;
      const char* nm = (const char*)(p + st->sh_offset + sym[k].st_name);
      size_t L = strnlen(nm, st->sh_size - sym[k].st_name);
      if (L > 3 && memcmp(nm + L - 3, ".kd", 3) == 0) { if (*nkd == 0) best = nm; (*nkd)++; }
    }
    if (*nkd) break;
  }
  return best;
}

static int g_fd = -2;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double mono_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void logf_(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char* fmt, ...) {
  pthread_mutex_lock(&g_mu);
  if (g_fd == -2) {
    const char* d = getenv("COLOAD_LOG_DIR");
    g_fd = -1;
    if (d && *d) {
      char p[4096];
      snprintf(p, sizeof p, "%s/coload.%d.log", d, (int)getpid());
      g_fd = open(p, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    }
  }
  if (g_fd >= 0) {
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) { if (n > (int)sizeof buf - 1) n = sizeof buf - 1; (void)!write(g_fd, buf, n); }
  }
  pthread_mutex_unlock(&g_mu);
}

typedef int (*kpack_load_t)(void*, const void*, const char*, uint32_t, const char* const*, size_t, void**, size_t*);
int kpack_load_code_object(void* cache, const void* meta, const char* binary_path, uint32_t co_index,
                           const char* const* arch_list, size_t arch_count, void** out, size_t* out_size) {
  static kpack_load_t real = NULL;
  if (!real) real = (kpack_load_t)dlsym(RTLD_NEXT, "kpack_load_code_object");
  double w = now_s(), t0 = mono_s();
  int rc = real ? real(cache, meta, binary_path, co_index, arch_list, arch_count, out, out_size) : 8;
  double dt = mono_s() - t0;
  logf_("%.6f kpack dur_ms=%.3f rc=%d tid=%ld bin=%s idx=%u size=%zu\n", w, dt * 1e3, rc, (long)gettid(),
        binary_path ? binary_path : "?", co_index, (rc == 0 && out_size) ? *out_size : (size_t)0);
  return rc;
}

typedef int (*reader_t)(const void*, size_t, void*);
int hsa_code_object_reader_create_from_memory(const void* code_object, size_t size, void* reader) {
  static reader_t real = NULL;
  if (!real) real = (reader_t)dlsym(RTLD_NEXT, "hsa_code_object_reader_create_from_memory");
  double w = now_s(), t0 = mono_s();
  int rc = real ? real(code_object, size, reader) : 1;
  double dt = mono_s() - t0;
  Dl_info di; memset(&di, 0, sizeof di);
  const char* lib = "-"; unsigned long off = 0;
  if (dladdr(code_object, &di) && di.dli_fname) { lib = di.dli_fname; off = (unsigned long)((const char*)code_object - (const char*)di.dli_fbase); }
  // first 64 bytes of the ELF .note / header region are identical across objects; hash the whole image (FNV-1a 64)
  uint64_t h = 1469598103934665603ULL; const unsigned char* p = (const unsigned char*)code_object;
  if (p && size < (1u << 30)) for (size_t i = 0; i < size; i++) { h ^= p[i]; h *= 1099511628211ULL; }
  int nkd = 0;
  const char* kd = first_kd((const unsigned char*)code_object, size, &nkd);
  char mid[65] = "-";
  if (p && size >= 64) { for (int i = 0; i < 32; i++) snprintf(mid + 2 * i, 3, "%02x", p[size / 2 + i]); }
  logf_("%.6f reader dur_ms=%.3f rc=%d tid=%ld size=%zu ptr=%p lib=%s off=0x%lx fnv=%016llx mid=%s nkd=%d kd=%.300s\n",
        w, dt * 1e3, rc, (long)gettid(), size, code_object, lib, off, (unsigned long long)h, mid, nkd, kd);
  return rc;
}

typedef int (*load_t)(uint64_t, uint64_t, uint64_t, const char*, void*);
int hsa_executable_load_agent_code_object(uint64_t exe, uint64_t agent, uint64_t reader, const char* options, void* loaded) {
  static load_t real = NULL;
  if (!real) real = (load_t)dlsym(RTLD_NEXT, "hsa_executable_load_agent_code_object");
  double w = now_s(), t0 = mono_s();
  int rc = real ? real(exe, agent, reader, options, loaded) : 1;
  double dt = mono_s() - t0;
  logf_("%.6f load dur_ms=%.3f rc=%d tid=%ld agent=%lx\n", w, dt * 1e3, rc, (long)gettid(), (unsigned long)agent);
  return rc;
}

typedef int (*freeze_t)(uint64_t, const char*);
int hsa_executable_freeze(uint64_t exe, const char* options) {
  static freeze_t real = NULL;
  if (!real) real = (freeze_t)dlsym(RTLD_NEXT, "hsa_executable_freeze");
  double w = now_s(), t0 = mono_s();
  int rc = real ? real(exe, options) : 1;
  double dt = mono_s() - t0;
  logf_("%.6f freeze dur_ms=%.3f rc=%d tid=%ld\n", w, dt * 1e3, rc, (long)gettid());
  return rc;
}
