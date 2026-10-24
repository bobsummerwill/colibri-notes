/* bigalloc.so (2026-10-24, fix B probe; debug only): log every heap allocation of BIGALLOC_MIN_MB (default 32) MiB
 * or more, with its size, the thread and a backtrace, to the file BIGALLOC_LOG (default /tmp/bigalloc.<pid>.log).
 * Anonymous mmaps the program makes itself are logged as well. Frees (and munmaps) of logged pointers are logged too, so what is still live at a point can be reconstructed. Backtrace frames
 * are printed as module+offset (dladdr), for addr2line against an unstripped binary.
 *
 *   gcc -O2 -shared -fPIC -o bigalloc.so bigalloc.c -ldl -lpthread
 *   LD_PRELOAD=/path/bigalloc.so BIGALLOC_LOG=/path/big.log ./deepseek_v4 ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static void *(*real_malloc)(size_t), *(*real_calloc)(size_t, size_t), *(*real_realloc)(void *, size_t);
static int (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);
static void (*real_free)(void *);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);
static int g_fd = -1;
static size_t g_min = (size_t)32 << 20;
static __thread int in_hook;
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
#define LIVE 8192
static void *live[LIVE];
static size_t live_size[LIVE];

/* a static arena for dlsym's own calloc before real_calloc is known */
static char boot[65536];
static size_t boot_used;

static void init(void) {
    static int done;
    if (done) return;
    done = 1;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    const char *m = getenv("BIGALLOC_MIN_MB");
    if (m && *m) g_min = (size_t)atol(m) << 20;
    const char *p = getenv("BIGALLOC_LOG");
    char path[256];
    if (!p || !*p) { snprintf(path, sizeof path, "/tmp/bigalloc.%d.log", (int)getpid()); p = path; }
    g_fd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

static void out(const char *fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (g_fd >= 0 && n > 0) { ssize_t w = write(g_fd, b, (size_t)(n < (int)sizeof b ? n : (int)sizeof b - 1)); (void)w; }
}

static void record(const char *what, void *p, size_t size) {
    if (!p || size < g_min || in_hook) return;
    in_hook = 1;
    pthread_mutex_lock(&mx);
    for (int i = 0; i < LIVE; i++) if (!live[i]) { live[i] = p; live_size[i] = size; break; }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    out("ALLOC %s %p %zu MiB=%.1f tid=%ld t=%ld.%03ld\n", what, p, size, size / 1048576.0, (long)syscall(SYS_gettid),
        (long)ts.tv_sec, ts.tv_nsec / 1000000);
    void *bt[24];
    int n = backtrace(bt, 24);
    for (int i = 2; i < n; i++) {
        Dl_info di;
        if (dladdr(bt[i], &di) && di.dli_fname)
            out("  #%d %s+0x%lx %s\n", i, di.dli_fname, (unsigned long)((uintptr_t)bt[i] - (uintptr_t)di.dli_fbase),
                di.dli_sname ? di.dli_sname : "");
        else
            out("  #%d %p\n", i, bt[i]);
    }
    pthread_mutex_unlock(&mx);
    in_hook = 0;
}

static void forget(void *p) {
    if (!p) return;
    pthread_mutex_lock(&mx);
    for (int i = 0; i < LIVE; i++)
        if (live[i] == p) {
            out("FREE %p %zu MiB=%.1f\n", p, live_size[i], live_size[i] / 1048576.0);
            live[i] = NULL;
            break;
        }
    pthread_mutex_unlock(&mx);
}

void *malloc(size_t n) {
    if (!real_malloc) init();
    void *p = real_malloc(n);
    record("malloc", p, n);
    return p;
}
void *calloc(size_t a, size_t b) {
    if (!real_calloc) {
        if (!boot_used && !real_malloc) { /* dlsym inside init may calloc */ }
        size_t n = a * b;
        if (boot_used + n <= sizeof boot && !real_malloc) { void *p = boot + boot_used; boot_used += (n + 15) & ~(size_t)15; return p; }
        init();
    }
    void *p = real_calloc(a, b);
    record("calloc", p, a * b);
    return p;
}
void *realloc(void *q, size_t n) {
    if (!real_realloc) init();
    if (q) forget(q);
    void *p = real_realloc(q, n);
    record("realloc", p, n);
    return p;
}
int posix_memalign(void **pp, size_t al, size_t n) {
    if (!real_posix_memalign) init();
    int r = real_posix_memalign(pp, al, n);
    if (!r) record("posix_memalign", *pp, n);
    return r;
}
void *aligned_alloc(size_t al, size_t n) {
    if (!real_aligned_alloc) init();
    void *p = real_aligned_alloc(al, n);
    record("aligned_alloc", p, n);
    return p;
}
void free(void *p) {
    if ((char *)p >= boot && (char *)p < boot + sizeof boot) return;
    if (!real_free) init();
    if (p && !in_hook) forget(p);
    real_free(p);
}
/* anonymous mappings the program asks for itself (glibc malloc's own mmap calls are internal and not seen here) */
void *mmap(void *a, size_t n, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) init();
    void *p = real_mmap(a, n, prot, flags, fd, off);
    if (p != MAP_FAILED && (flags & MAP_ANONYMOUS)) record("mmap", p, n);
    return p;
}
int munmap(void *a, size_t n) {
    if (!real_munmap) init();
    if (n >= g_min) forget(a);
    return real_munmap(a, n);
}
