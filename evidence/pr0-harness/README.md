# Harnesses for JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212): the three backend bugs, reproduced on upstream dev

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) and
> **PR #2** (two-GPU decode) are not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Small single-GPU programs (about 60 MiB of device memory) linked against `c/backend_cuda_dsv4.cu`
of a given commit. Each one reproduces one bug that [#2212](https://github.com/JustVugg/colibri/pull/2212) (branch `rebased/pr0a/dsv4-upload-ordering`) fixes.

| harness | bug (commit in #2212) | what it checks |
|---|---|---|
| `h2d_race.c` | host→device copies not ordered on the compute stream ("order host->device uploads on the compute stream") | `DSV4_CUDA_PIN_HOST=0` forces pageable refills. **WAR**: a fused MoE over a decode mirror is queued behind a kernel backlog, then the mirror is refilled in place (`sync=1`, the blocking call the engine makes); the queued MoE must still see the old content. **RAW**: refill, then MoE; must see the new content. **UPLOAD**: fresh mirror upload, then MoE. Prints mismatches per scenario; exit 1 on any. |
| `bank_lasterr.c` | a failed expert-bank allocation leaves its OOM as the CUDA last error ("clear the CUDA error a failed expert-bank allocation leaves") | asks for a 1 TiB bank (fails without allocating), then makes an unrelated launch-checked call, which must pass |
| `shutdown_reinit.c` | `dsv4_cuda_shutdown` leaves freed work buffers behind ("dsv4_cuda_shutdown zeroes each device slot it tears down", F28) | an f32 matvec, shutdown, init again in the same process, the same matvec |

## Build and run (any CUDA GPU; here an RTX 3090, CUDA 12.0)

```sh
cd <colibri checkout at the commit to test>/c
nvcc -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC \
     -arch=sm_86 -DCOLI_DSV4_NO_TC -c backend_cuda_dsv4.cu -o backend.o
for h in h2d_race bank_lasterr shutdown_reinit; do
  gcc -O2 -Wall -Wextra -I. path/to/$h.c backend.o -o $h -lcudart -lcuda -lstdc++ -lm
done
CUDA_VISIBLE_DEVICES=0 ./h2d_race 300 20     # 300 trials per scenario, kernel backlog 20
CUDA_VISIBLE_DEVICES=0 ./bank_lasterr
CUDA_VISIBLE_DEVICES=0 ./shutdown_reinit
```

`-DCOLI_DSV4_NO_TC` is only needed for CUDA toolkits older than the cuBLASLt block-scaling API (12.0 here).

## Results

2026-10-18, on the rebased series (`run.sh`, `run.log`; RTX 3090, nothing else on the GPU):

| | upstream dev 62eaa9dc | #2212 alone (ce2d4eff) | #2212–#2217 combined (760cb44a) |
|---|---|---|---|
| `h2d_race 300 20`: WAR / RAW / UPLOAD mismatched | **300/300** / 0/300 / 0/300 | 0 / 0 / 0 | 0 / 0 / 0 |
| `bank_lasterr` | **FAILED** (stale out-of-memory error) | OK | OK |
| `shutdown_reinit` | **FAILED** after shutdown + init | OK | OK |

2026-10-14, on the previous base 7b55a9b3 (`2026-10-14/`, 50 trials): dev WAR 50/50, RAW 0/50,
UPLOAD 0/50; the six fixes combined 0/50 in all three.

Only the WAR case (an in-place refill landing under a queued kernel) mismatches; RAW and UPLOAD
never did. In full-engine decode runs on dev the race did not change the output (17 stressed
runs, `measurements.md`), but on the batched-prefill path with the double bank it did (1 of 3
cold runs before the fix, 0 of 3 after; `manifests/final2/race/`).
