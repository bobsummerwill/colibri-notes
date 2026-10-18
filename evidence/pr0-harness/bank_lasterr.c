/* pr0 item 2: a failed expert-bank allocation must not leave its OOM as the
 * CUDA runtime's last error.  The bank is asked for 1 TiB (count 1, H = I =
 * 2^20), so the first cudaMalloc fails without allocating anything; the
 * next call with a launch check (dsv4_cuda_rmsnorm: launch, then
 * cudaGetLastError) must still succeed.  A few MiB of device memory. */
#include <stdio.h>
#include <stdlib.h>
#include "backend_cuda_dsv4.h"
enum { N = 4096 };
int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("SKIP init"); return 77; }
    static float w[N], x[N];
    for (int i = 0; i < N; i++) { w[i] = 1.f; x[i] = (float)(i % 5); }
    Dsv4CudaTensor *weight = NULL;
    Dsv4CudaActivation *a = dsv4_cuda_activation_create(dev, N);
    if (!a || !dsv4_cuda_upload_f32(&weight, w, 1, N, dev) || !dsv4_cuda_activation_upload(a, x, N) ||
        !dsv4_cuda_rmsnorm(a, weight, 1e-6f, N) || !dsv4_cuda_activation_sync(a)) { puts("setup failed"); return 2; }
    Dsv4CudaExpertSet *bank = dsv4_cuda_expert_bank_create(1, 1 << 20, 1 << 20, dev, NULL, NULL, NULL);
    if (bank) { puts("a 1 TiB bank was allocated?"); return 2; }
    int launched = dsv4_cuda_rmsnorm(a, weight, 1e-6f, N) && dsv4_cuda_activation_sync(a);
    printf("after a failed bank allocation the next launch check %s\n",
           launched ? "passes: OK" : "FAILED (stale out-of-memory error)");
    dsv4_cuda_shutdown();
    return launched ? 0 : 1;
}
