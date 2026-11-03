# Head-followups branch: GPU tests and head-placement A/B (2026-11-03)

Branch: [`fix/dsv4-head-followups`](https://github.com/bobsummerwill/colibri/tree/fix/dsv4-head-followups)
on `bobsummerwill/colibri`, based on PR #1766 head `aabd99d` (not rebased).

Commits:

```text
f4586dca feat(dsv4): allow an exact-head device override
7ad5319a test(dsv4): use tests/dsv4_test_devices.h in the resident rules
0956307c test(dsv4): fix the test-header dependency in the new rules
eaf63b1a docs(dsv4): document head numbers, reserve split and multi-GPU behavior
6b2efd31 test(dsv4): add dsv4-cuda-tests for all nine CUDA tests
9bcbc95b fix(dsv4): best-effort peer access and KV owner guard
07dc142a fix(dsv4): harden auto residency, reserve and head placement
```

## Build

shadowfax, `sm_86`, `NO_TC=1`, `make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1
CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8`:

- engine build: success
- all nine test binaries build through the makefile, including `quant`, `resident_route` and
  `resident_indexer` (which had no rule before)
- `test_v4_gpu_placement.py`, `test_v4_head_batch.py`, `test_v4_mirror_guard_source.py`: pass

## Nine CUDA tests (`dsv4-cuda-tests`, `DSV4_TEST_DEVICES=0,1`)

All nine passed (`TESTS_RC=0`); raw log: [tests.log](tests.log).

| test | result |
|---|---|
| `head_cuda` | exact logits and compact argmax, batches 1/3/4/5/9/2, both devices |
| `wo_decode_cuda` | exact 3/3, formats 8/9, widths 256/4096, both devices |
| `compressor_cuda` | exact, prefill unchanged, both devices |
| `compressor_state_cuda` | state/output/reset/partial-failure exact, both devices |
| `indexer_prepare_cuda` | query/head-weights and score exact, both devices |
| `multigpu_cuda` | ok; duplicate list refused, active contexts survive refusal |
| `quant_cuda` | 35,769 cases, **0 mismatches**, both devices |
| `resident_route` | exact parity, normal and hash routes, both devices |
| `resident_indexer` | exact parity, both layouts, batches 1/3/33, both devices |

The tests ran at `7ad5319a`. `f4586dca` adds only the `DSV4_CUDA_HEAD_DEVICE` override and its
docs; the A/B below exercises that code on both devices.

## Head-placement A/B (2x 3090, full V4 Flash, 2 GPUs)

`bench_fdec.py`, 3 x 300 tokens, cold server, 500/450 W, two samples per arm, interleaved
H0/H1/H1/H0. `DSV4_CUDA_HEAD_DEVICE=0` versus `=1`; both runs logged `v4_gpu head=resident
device=0` / `device=1` with no CPU fallback. All texts identical:
`3d0a5275`, `f0e57dde`, `d27222f1`.

| arm | sample means (tok/s) | average |
|---|---:|---:|
| head device 0 | 3.6017, 3.6057 | **3.6037** |
| head device 1 | 3.6120, 3.6093 | **3.6107** |

Device 1 is **+0.19%** over device 0, i.e. neutral within this two-sample noise. The owning/last
device default is safe; the override lets users pin the head when they want the old layout. No
regression was found. Raw log: [headab.log](headab.log).

## Services

`llama-server-active` was stopped for the GPU windows and restored (`{"status":"healthy"}`);
Hermes was frozen with `SIGSTOP` and resumed (`Ssl`). `colibri-headab` and `headab-serve` are
inactive; no GPU compute processes remain besides Tabby.

## Caveats

- The nine tests and A/B are on 2x 3090 only; no 5070 or 6x 5090 at this branch.
- The A/B is two samples per arm; the 0.19% difference is not a claim of a real gain, only
  evidence that the placement change does not regress.
- The nine tests ran at `7ad5319a`; `f4586dca` is the head-device override, exercised by the A/B.
