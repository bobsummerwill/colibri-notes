/* pr0 item 1: are host->device mirror copies ordered against the kernels on
 * the device's compute stream?  Single GPU, ~60 MiB of device memory.
 *
 * DSV4_CUDA_PIN_HOST=0 (set below) forces the pageable refill path, as for
 * expert slabs past the 20 GiB page-lock cap.  T is a decode mirror (gate, up,
 * down of one expert).  Three scenarios, each N trials:
 *
 *  WAR     a fused MoE over T (content A) is queued on the compute stream
 *          behind a kernel backlog, then T is refilled in place with B
 *          (sync=1, the blocking call the engine makes): the queued MoE must
 *          still see A.
 *  RAW     T is refilled with B (sync=0, pageable) and the MoE is queued
 *          right after: it must see B.
 *  UPLOAD  a new mirror is uploaded with B and the MoE is queued right
 *          after: it must see B (dsv4_cuda_upload_fp4, the mirror-cache
 *          growth path).
 *
 * Prints the mismatch count per scenario; exit 1 when any trial mismatched. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend_cuda_dsv4.h"
enum { H = 4096, I = 2048, N = 1 << 22 };
static uint64_t g = 0x1234;
static uint32_t rnd(void) { g = g * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g >> 33); }
static void fillw(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rnd(); }
static void fills(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(119 + rnd() % 4); }
static void fill8(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) { uint8_t v = (uint8_t)rnd(); p[i] = (v & 0x7f) == 0x7f ? v ^ 1 : v; } }
static void *pa(size_t n) { void *p = NULL; if (posix_memalign(&p, 4096, n)) return NULL; return p; }
typedef struct { uint8_t *gw, *gs, *uw, *us, *dw, *ds; } Expert;
static size_t w1 = (size_t)I * H / 2, s1 = (size_t)I * H / 32, w2 = (size_t)H * I / 2, s2 = (size_t)H * I / 32;
static void expert_make(Expert *e) {
    e->gw = pa(w1); e->gs = pa(s1); e->uw = pa(w1); e->us = pa(s1); e->dw = pa(w2); e->ds = pa(s2);
    fillw(e->gw, w1); fills(e->gs, s1); fillw(e->uw, w1); fills(e->us, s1); fillw(e->dw, w2); fills(e->ds, s2);
}
static Dsv4CudaTensor *gT, *uT, *dT, *sg, *su, *sd, *W;
static Dsv4CudaActivation *X, *Y, *B;
static float wt = 0.7f;
static int refill(const Expert *e, int sync) {
    return dsv4_cuda_tensor_refill_fp4(gT, e->gw, e->gs, I, H, sync) &&
           dsv4_cuda_tensor_refill_fp4(uT, e->uw, e->us, I, H, sync) &&
           dsv4_cuda_tensor_refill_fp4(dT, e->dw, e->ds, H, I, sync);
}
static int moe_queue(Dsv4CudaTensor *g_, Dsv4CudaTensor *u_, Dsv4CudaTensor *d_) {
    Dsv4CudaTensor *gr[1] = {g_}, *ur[1] = {u_}, *dr[1] = {d_};
    return dsv4_cuda_moe_activation(gr, ur, dr, &wt, 1, sg, su, sd, 10.f, X, Y);
}
static int result(float *y) { return dsv4_cuda_activation_download(y, Y, H) && dsv4_cuda_activation_sync(Y); }
static int backlog(int k) { for (int i = 0; i < k; i++) if (!dsv4_cuda_rmsnorm(B, W, 1e-6f, N)) return 0; return 1; }
int main(int argc, char **argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 50, depth = argc > 2 ? atoi(argv[2]) : 20;
    setenv("DSV4_CUDA_PIN_HOST", "0", 1);
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("SKIP init"); return 77; }
    Expert A, Bx; expert_make(&A); expert_make(&Bx);
    if (!dsv4_cuda_upload_fp4(&gT, A.gw, A.gs, I, H, 0) || !dsv4_cuda_upload_fp4(&uT, A.uw, A.us, I, H, 0) ||
        !dsv4_cuda_upload_fp4(&dT, A.dw, A.ds, H, I, 0)) { puts("upload failed"); return 2; }
    {
        size_t wn = (size_t)I * H, sn = (size_t)((I + 127) / 128) * ((H + 127) / 128);
        uint8_t *w = malloc(wn), *s = malloc(sn);
        fill8(w, wn); fills(s, sn); if (!dsv4_cuda_upload_fp8(&sg, w, s, I, H, 0)) return 2;
        fill8(w, wn); if (!dsv4_cuda_upload_fp8(&su, w, s, I, H, 0)) return 2;
        fill8(w, wn); if (!dsv4_cuda_upload_fp8(&sd, w, s, H, I, 0)) return 2;
        free(w); free(s);
    }
    float *hw = malloc((size_t)N * 4), *hb = malloc((size_t)N * 4);
    for (int i = 0; i < N; i++) { hw[i] = 1.f; hb[i] = (float)(i % 7); }
    if (!dsv4_cuda_upload_f32(&W, hw, 1, N, 0)) return 2;
    X = dsv4_cuda_activation_create(0, H); Y = dsv4_cuda_activation_create(0, H); B = dsv4_cuda_activation_create(0, N);
    static float x[H], refA[H], refB[H], y[H];
    for (int i = 0; i < H; i++) x[i] = (float)((int)(rnd() % 2001) - 1000) / 1000.f;
    if (!X || !Y || !B || !dsv4_cuda_activation_upload(X, x, H) || !dsv4_cuda_activation_upload(B, hb, N) ||
        !dsv4_cuda_stream_drain(0)) return 2;
    if (!moe_queue(gT, uT, dT) || !result(refA)) { puts("ref A failed"); return 3; }
    if (!refill(&Bx, 1) || !dsv4_cuda_stream_drain(0) || !moe_queue(gT, uT, dT) || !result(refB)) { puts("ref B failed"); return 3; }
    if (!memcmp(refA, refB, sizeof refA)) { puts("A and B give the same output: no signal"); return 3; }
    int war = 0, raw = 0, up = 0;
    for (int t = 0; t < trials; t++) {
        if (!refill(&A, 1) || !dsv4_cuda_stream_drain(0)) return 4;
        if (!backlog(depth) || !moe_queue(gT, uT, dT)) return 4;
        if (!refill(&Bx, 1)) return 4;          /* overwrites T while the queued MoE has not run yet */
        if (!result(y)) return 4;
        if (memcmp(y, refA, sizeof y)) war++;
    }
    for (int t = 0; t < trials; t++) {
        if (!refill(&A, 1) || !dsv4_cuda_stream_drain(0)) return 5;
        if (!refill(&Bx, 0) || !moe_queue(gT, uT, dT) || !result(y)) return 5;
        if (memcmp(y, refB, sizeof y)) raw++;
    }
    for (int t = 0; t < trials; t++) {
        Dsv4CudaTensor *g2 = NULL, *u2 = NULL, *d2 = NULL;
        if (!dsv4_cuda_stream_drain(0)) return 6;
        if (!dsv4_cuda_upload_fp4(&g2, Bx.gw, Bx.gs, I, H, 0) || !dsv4_cuda_upload_fp4(&u2, Bx.uw, Bx.us, I, H, 0) ||
            !dsv4_cuda_upload_fp4(&d2, Bx.dw, Bx.ds, H, I, 0)) return 6;
        if (!moe_queue(g2, u2, d2) || !result(y)) return 6;
        if (memcmp(y, refB, sizeof y)) up++;
        dsv4_cuda_tensor_free(g2); dsv4_cuda_tensor_free(u2); dsv4_cuda_tensor_free(d2);
    }
    printf("h2d ordering (pageable, backlog %d): WAR %d/%d RAW %d/%d UPLOAD %d/%d mismatched\n",
           depth, war, trials, raw, trials, up, trials);
    dsv4_cuda_shutdown();
    return war || raw || up;
}
