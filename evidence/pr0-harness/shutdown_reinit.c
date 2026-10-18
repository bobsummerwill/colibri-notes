/* pr0 item 5 (F28): dsv4_cuda_shutdown, then dsv4_cuda_init again in the same
 * process (an engine closed and another opened).  The lazily grown work
 * buffers (dx/dy here, through buf()) must not survive the shutdown as freed
 * pointers with their old caps.  An f32 matvec (O = 32, I = 2^20: a 4 MiB
 * input buffer) before and after the re-init, checked against the CPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "backend_cuda_dsv4.h"
enum { O = 32, I = 1 << 20 };
static int matvec_ok(const float *w, const float *x, const char *when) {
    Dsv4CudaTensor *t = NULL;
    static float y[O];
    int ok = dsv4_cuda_upload_f32(&t, w, O, I, 0) && dsv4_cuda_matvec(t, y, x);
    double worst = 0;
    for (int o = 0; ok && o < O; o++) {
        double ref = 0;
        for (int i = 0; i < I; i++) ref += (double)w[(size_t)o * I + i] * x[i];
        double d = fabs(ref - y[o]) / (fabs(ref) + 1);
        if (d > worst) worst = d;
    }
    dsv4_cuda_tensor_free(t);
    int good = ok && worst < 1e-3;
    printf("%s: matvec %s (worst rel err %.2g)\n", when, good ? "ok" : "FAILED", worst);
    return good;
}
int main(void) {
    int dev = 0;
    float *w = malloc((size_t)O * I * sizeof(float)), *x = malloc((size_t)I * sizeof(float));
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = (float)((int)(i % 7) - 3) / 8.f;
    for (int i = 0; i < I; i++) x[i] = (float)((i % 5) - 2) / 4.f;
    if (!dsv4_cuda_init(&dev, 1)) { puts("SKIP init"); return 77; }
    int first = matvec_ok(w, x, "first init");
    dsv4_cuda_shutdown();
    if (!dsv4_cuda_init(&dev, 1)) { puts("re-init failed"); return 2; }
    int second = matvec_ok(w, x, "after shutdown + init");
    dsv4_cuda_shutdown();
    return first && second ? 0 : 1;
}
