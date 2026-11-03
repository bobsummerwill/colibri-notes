# FINAL37: PR #1766 at `aabd99d` against dev `3b953308` (2026-11-03)

First exact-head GPU re-baseline of #1766 after Bob's `samebase/m1766-fixes` (`28b35786`) and dev
`3b953308` were merged into the PR. The earlier same-base numbers were measured at `ecc92586`
(pre-`f4` fix) and the C+ numbers at `e2933955`; this run is the first at the current head.

## Trees, machine, method

- **B (dev):** `3b953308f7de630f27357bfba42ab0ec5c630afb` in `cw22/f37-base`
- **M (#1766):** `aabd99d468c70afbfceaf7a5b767a97f9eb8d319` in `cw22/f37-head`
  (= `28b35786` + dev `3b953308`)
- Both built clean with `make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr
  NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8`.
- shadowfax: 2x RTX 3090 24 GB (NVLink), Ryzen 9 9950X3D, 62 GB, driver 595.99.02 / CUDA 13.2.
  Power limits 500 W (GPU 0) and 450 W (GPU 1); the earlier same-base runs were at 300 W, and
  final36 showed the PR #2 ratios change by less than 0.3% at 500/450 W.
- final30-derived harness (`~/colibri-bench/final37`): cold server per sample, `env -i` server
  environment, interleaved arms, 2 samples for the headline arms, `--ram 40 --ctx 8192`,
  `COLI_CUDA_MOE_BATCH=1` on every arm, `bench_decode.py` (REAP, 6 prompts x 300 tokens),
  `bench_fdec.py` (full model, 3 x 300 tokens), `bench_pr1.py` (full model, both long prompts).
- Every request text is SHA-256 compared to its group reference and to final27/final6 where a
  reference exists. All texts matched.
- llama (`llama-server-active`) was stopped for the GPU window and restored; the Hermes gateway
  was frozen with `SIGSTOP` and resumed. No OOM kills, no `bank_lost` in any measured arm.

## Results

| arm | model | GPUs | decode tok/s | vs dev 1 GPU |
|---|---|---:|---:|---:|
| B1 | REAP-150B | 1 | 2.9753 | — |
| M1 | REAP-150B | 1 | 3.6425 | **1.224x** |
| M2 | REAP-150B | 2 | 4.5877 | **1.542x** |
| DB1F | full V4 Flash, fdec | 1 | 2.6290 | — |
| DM1F | full V4 Flash, fdec | 1 | 3.1412 | **1.195x** |
| DM2F | full V4 Flash, fdec | 2 | 3.5838 | **~1.363x** |

- REAP 2 GPUs vs 1 GPU: 4.5877 / 3.6425 = **1.260x**.
- Full 2 GPUs vs 1 GPU: 3.5838 / 3.1412 = **1.141x**.
- Full 2 GPUs vs dev 1 GPU: 1.1948 x 1.1409 = **1.363x** (calculated from the two measured rows).

### TTFT / prefill

| arm | long_prefill TTFT | long_prefill_2 TTFT | decode |
|---|---:|---:|---:|
| AB1F (dev) | 85.32 s | 122.41 s | 1.828 tok/s |
| AM1F (head) | 76.38 s | 108.87 s | 2.092 tok/s |
| ratio | **0.895x** | **0.889x** | 1.145x |

The `f4` table fix is in the head and the long-prompt regression is gone: the head is now about
10% faster to first token than dev on both prompts.

### Reserve / residency variants (full model, 1 GPU, 1 sample each)

| arm | setting | decode tok/s | vs DM1F |
|---|---|---:|---:|
| DM1F | default (`auto`, reserve unset) | 3.1412 | — |
| DM1F-R2 | `DSV4_CUDA_VRAM_RESERVE_MB=2000` | 3.1953 | 1.017x |
| DM1F-R47 | `DSV4_CUDA_VRAM_RESERVE_MB=4700` | 3.1097 | 0.990x |
| DM1F-R0 | `DSV4_CUDA_RESIDENT_EXPERTS=0` | 3.1490 | 1.0025x |
| DM1F-R1 | `DSV4_CUDA_RESIDENT_EXPERTS=1` | refused | — |

`DM1F-R1` failed **cleanly**, as it should:

```text
v4_gpu resident-experts=cache-fallback (resident experts exceed device 0 budget (free=16596 MiB reserve=2800 MiB))
resident experts exceed device 0 budget (free=16596 MiB reserve=2800 MiB)
RuntimeError: colibri engine exited unexpectedly (exit status 1)
```

The default `auto` path logged the same budget refusal and fell back to the cache. So on 24 GB,
for both the full model and REAP, `auto` does **not** keep the resident experts:

```text
# full model
v4_gpu mirror-reserve=4681MiB (256-slot bank 3264MiB x1 + 600MiB headroom + 64 mirrors; DSV4_CUDA_VRAM_RESERVE_MB overrides)
v4_gpu resident-experts=cache-fallback (resident experts exceed device 0 budget (free=16596 MiB reserve=2800 MiB))

# REAP
v4_gpu mirror-reserve=3100MiB (132-slot bank 1683MiB x1 + 600MiB headroom + 64 mirrors; DSV4_CUDA_VRAM_RESERVE_MB overrides)
v4_gpu resident-experts=cache-fallback (resident experts exceed device 0 budget (free=16768 MiB reserve=2800 MiB))
```

This is the key interpretation point: the measured 1.22x/1.54x/1.19x/1.36x gains on 24 GB come
from the faster kernels (head, resident indexer, compressor, `wo_decode`), **not** from resident
experts. The bank-aware `mirror-reserve` line is present, but the preload path still uses the flat
2800 MiB `DSV4_CUDA_VRAM_RESERVE_MB` default. The PR's "automatic residency" headline engages on
larger cards (the author's 6x 5090 target), not on these 24 GB cards with these checkpoints.

## CUDA unit tests on 2x 3090

`DSV4_TEST_DEVICES=0,1`; all nine passed:

- `test_dsv4_head_cuda`: exact logits and compact argmax, batches 1/3/4/5/9/2, both devices.
- `test_dsv4_wo_decode_cuda`: exact 3/3, formats 8/9, widths 256/4096, both devices.
- `test_dsv4_compressor_cuda`: exact, prefill unchanged, both devices.
- `test_dsv4_compressor_state_cuda`: state/output/reset/partial-failure exact, both devices.
- `test_dsv4_indexer_prepare_cuda`: query/head-weights and score exact, both devices.
- `test_dsv4_multigpu_cuda`: ok; duplicate list refused, active contexts survive refusal.
- `test_dsv4_quant_cuda`: 35,769 cases, **0 mismatches**, both devices.
- `test_dsv4_resident_route`: exact parity, normal and hash routes, both devices.
- `test_dsv4_resident_indexer`: exact parity, both layouts, batches 1/3/33, both devices.

Raw log: [tests.log](tests.log).

## Comparison with the pre-head numbers

| | prior same-base (`ecc92586`) | head `aabd99d` |
|---|---:|---:|
| REAP 1 GPU | 1.26x | **1.224x** |
| REAP 2 GPUs vs dev 1 GPU | 1.60x | **1.542x** |
| REAP 2 GPUs vs 1 GPU | ~1.27x | 1.260x |
| full 1 GPU | 1.22x | **1.195x** |
| full 2 GPUs vs dev 1 GPU | 1.40x | **~1.363x** |
| full 2 GPUs vs 1 GPU | ~1.15x | 1.141x |
| long-prompt TTFT | ~0.91x | **0.895x / 0.889x** |

The head gains are 2-4 points lower than the pre-head same-base numbers. The current dev base is
newer and faster, and the head's absolute REAP result is ~3-4% below `ecc92586`. This is exactly
why the exact-head re-baseline was needed; the old 1.26/1.60/1.22/1.40 headline should not be
quoted for `aabd99d`.

## Caveats

- Only 2x 3090. No 5070 and no 6x 5090 at this head; the 5070 result in the notes (1.183x) is
  from the earlier tree.
- Two samples per headline arm (one for the reserve variants). CVs were small (mostly <2%), but
  this is not a multi-run distribution.
- 500/450 W, not the 300 W of the earlier same-base runs. final36 at 500/450 W reproduced the
  PR #2 ratios within 0.3%, but there is no equivalent #1766 power-repeat here.
- `DM1F-R1` is a negative control. The harness's `GATES: FAIL` is only that arm: it treats a
  server that never becomes ready as a failure, even when the refusal is the expected result.
  All other arms passed their premises; `gates.notes` is empty; no text mismatches.
- The full model and REAP did not fit resident experts on 24 GB at any tested reserve
  (2000/2800/4700). That is a capacity result, not a code regression.

## Review implications

The exact-head GPU-validation gap is closed. Performance and correctness are now supported at
`aabd99d`: byte-identical outputs, 9/9 CUDA tests, no OOM kills, no bank loss, and a better TTFT.
What remains is configuration/docs, not kernels:

1. **`auto` hard-failure semantics (P1).** `coli_v4_gpu_experts_init` still swallows every
   nonzero preload result under `auto`, including hard `-1` failures, while the docs say "Other
   loading failures still reject engine open." The explicit `=1` arm proves a clean refusal is
   possible. Either distinguish a budget refusal from a hard load failure, or correct the docs.
2. **`DSV4_CUDA_VRAM_RESERVE_MB` is overloaded (P1/P2).** #1766's preload path uses a flat 2800
   MiB default; dev/#2405 use the same variable as the bank-aware mirror-growth reserve
   (`mirror-reserve=4681MiB` here). The default `auto` falls back on 24 GB, so the docs should say
   so; splitting or renaming the variable would remove the collision.
3. **Docs/headline numbers (P2).** Quote 1.22x/1.54x/1.19x/1.36x at this head, not the old
   1.26/1.60/1.22/1.40. State that `auto` falls back on 24 GB and that the gain there is
   kernel-side. Add the new switch names to the env table.
4. **CUDA tests are still not in CI (P2).** All nine pass manually on 2x 3090, but
   `quant_cuda`, `resident_route` and `resident_indexer` are in no makefile, and none are in
   `dsv4-cuda-test` or CI. Add a target and a documented `DSV4_TEST_DEVICES` gate.
5. **P2P/NCCL gating (P2).** Peer access is attempted for every pair and `dsv4_cuda_init` aborts
   if enabling fails; layer split does not require P2P. Make it best-effort, and document that
   the 6-GPU path initializes NCCL. The 6x 5090 target remains independently unverified.
6. **Lower priority:** head upload is pinned to device 0, the GGUF dense path still targets one
   device, `v4_gpu_expert_find`/`coli_v4_gpu_resident_route` lack a couple of defensive NULL
   checks, and KV ownership relies on shutdown/reinit when the device list changes.

**Verdict:** request changes narrowed to the `auto`/reserve semantics and the docs. The
performance/correctness case is no longer a blocker; the remaining items affect configuration
behavior and every CUDA user's expectations, not the measured kernels.

## Artifacts

- [summary.txt](summary.txt): harness summary + JSON (75 KB)
- [run.log](run.log): full harness log (97 KB)
- [tests.log](tests.log): nine CUDA tests
- [DM1F-serve.log](DM1F-serve.log): default `auto` fallback and `mirror-reserve`
- [DM1F-R1-serve.log](DM1F-R1-serve.log): explicit `=1` refusal
- `sha256sum`:

```text
df35a1133934a8798742d1b5ce0f7774311584f036b9bba78f10ee8d9e823d73  DM1F-R1-serve.log
2fa13a519b892bf2f5d081e712e436b88aabfb1d1cd733e8078086d357d40acd  DM1F-serve.log
873e356b9035bfbb4d9f2d9793eded28707ab9091958cf90ef2aad7e80439c4d  run.log
a0921cd91be8d6d77a1228f3b02f991b89abb9350d9291d12f7b8439df974535  summary.txt
e0931e94649c4feb81be162107edfb930bb11933e1e202dddcc6c25bcac8e5b8  tests.log
```

Raw run directory on shadowfax: `~/colibri-bench/final37/bench/`; tarball
`~/colibri-bench/final37/bench.tgz` (sha256 `9cb6b473774b37f1e0c15714ed5cc818b0b091c386801fbf1354973d712f9788`).
