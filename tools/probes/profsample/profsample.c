/* profsample.c (2026-10-24): an unprivileged CPU sampler for the deepseek_v4 server (perf is off on shadowfax:
 * perf_event_paranoid=4, and ptrace_scope=1 keeps gdb off a process systemd started).
 *
 * LD_PRELOAD=profsample.so PROFSAMPLE_OUT=/dir [PROFSAMPLE_HZ=500] [PROFSAMPLE_EXE=deepseek_v4]
 * In the process whose executable's basename is PROFSAMPLE_EXE (the python launcher that execs it is left
 * alone), ITIMER_PROF fires every 1/HZ s of the process's CPU time, on whichever thread was running; the
 * handler writes one record (tid, depth, return addresses) with write(2) to /dir/samples.<pid>, and copies
 * /proc/self/maps to /dir/maps.<pid> at the 2nd sample and every 20,000th (libraries load late). Nothing
 * in the handler allocates: backtrace() is warmed up once in the constructor. profsample_report.py turns
 * the records into self / inclusive function counts.
 * Build: cc -O2 -fPIC -shared -o profsample.so profsample.c */
#define _GNU_SOURCE
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#define DEPTH 48
static int g_fd = -1;
static char g_maps[512];
static volatile unsigned long g_n;

static void copy_maps(void) {
    int in = open("/proc/self/maps", O_RDONLY), out;
    if (in < 0) return;
    out = open(g_maps, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out >= 0) {
        char buf[8192]; ssize_t r;
        while ((r = read(in, buf, sizeof buf)) > 0) if (write(out, buf, (size_t)r) != r) break;
        close(out);
    }
    close(in);
}

static void on_prof(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)si; (void)uc;
    int keep = errno;
    void *pc[DEPTH + 2];
    int n = backtrace(pc + 2, DEPTH);
    uint64_t rec[DEPTH + 2];
    rec[0] = (uint64_t)syscall(SYS_gettid);
    rec[1] = (uint64_t)n;
    for (int i = 0; i < n; i++) rec[2 + i] = (uint64_t)(uintptr_t)pc[2 + i];
    if (g_fd >= 0) { ssize_t w = write(g_fd, rec, (size_t)(2 + n) * sizeof rec[0]); (void)w; }
    unsigned long k = __atomic_add_fetch(&g_n, 1, __ATOMIC_RELAXED);
    if (k == 2 || k % 20000 == 0) copy_maps();
    errno = keep;
}

__attribute__((constructor)) static void profsample_init(void) {
    const char *dir = getenv("PROFSAMPLE_OUT"), *want = getenv("PROFSAMPLE_EXE");
    if (!dir || !*dir) return;
    if (!want || !*want) want = "deepseek_v4";
    char exe[512];
    ssize_t r = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (r <= 0) return;
    exe[r] = 0;
    const char *base = strrchr(exe, '/');
    base = base ? base + 1 : exe;
    if (strcmp(base, want)) return;
    char path[512];
    snprintf(path, sizeof path, "%s/samples.%d", dir, (int)getpid());
    g_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (g_fd < 0) return;
    snprintf(g_maps, sizeof g_maps, "%s/maps.%d", dir, (int)getpid());
    void *warm[4];
    backtrace(warm, 4);   /* loads libgcc_s now, not inside the handler */
    copy_maps();
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_prof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);
    const char *hz = getenv("PROFSAMPLE_HZ");
    long f = hz && *hz ? atol(hz) : 500;
    if (f < 1) f = 1;
    if (f > 10000) f = 10000;
    struct itimerval it = {{0, 1000000 / f}, {0, 1000000 / f}};
    setitimer(ITIMER_PROF, &it, NULL);
}
