### Commit or version

77530b49 (dev, 2026-10-25); also seen on 61646cbd and 7df8c27c

### Environment

- Ubuntu 24.04, 62 GB RAM, 2× RTX 3090 24 GB (one used here), CUDA 12.0, gcc 13.3
- Also on an RTX 5070 12 GB (Ubuntu 24.04, 62 GB RAM), CUDA 12.8, DeepGEMM build (`CUDA_ARCH=sm_120a`)
- Model: full DeepSeek-V4-Flash-0731

### Reproduction steps

1. Build the CUDA engine and serve with the default configuration (no `COLI_CUDA_MOE_BATCH`, so no expert bank; `COLI_CUDA_ATTN_BATCH` on by default), `--ram 40 --ctx 8192`.
2. Send two long greedy prompts in a row (1,992 and 3,005 tokens here).
3. Count `[DSV4 CUDA] kv comp allocation: out of memory` in the server log, and compare the answers with a run that has none.

### Expected behavior

The decode mirror cache stops growing at its VRAM reserve (600 MiB without the bank, or `DSV4_CUDA_VRAM_RESERVE_MB`), so the compressed-KV buffer can still grow, attention stays on the GPU, and greedy answers don't depend on how full the card happened to get.

### Actual behavior and logs

The mirror cache grows past its reserve, up to the whole card:
- On dev 77530b49 the RTX 3090 peaked at 24,119 MiB.
- `dsv4_cuda_kv_comp_append` then can't double the compressed-KV buffer, and `coli_v4_gpu_kv_cache_sync` returns 0.
- That layer's attention runs on the CPU, which doesn't give CUDA's bits.

Measured (two prompts per run, a fresh server per run):
- **RTX 3090, dev 77530b49, 2 runs:** 1,139 handled `kv comp` failures in each. Answer 2 changed (`964826e1…` instead of the `8b3f437b…` that runs without fallbacks give).
- **RTX 3090, dev 61646cbd, 3 runs:** 420 kv comp failures plus 1 weight failure in each run. The texts happened to stay the same, but the 3,005-token time to first token was 271 s, against 256 s with the fix.
- **RTX 5070:** 0 to 1,066 failures per run in the same configuration. 2 of 7 runs changed answer 2.

```
[DSV4 CUDA] kv comp allocation: out of memory
```

**Cause:**
- The growth guard re-reads free VRAM only every 64 misses (`COLI_V4_MIRROR_PROBE_EVERY`) and reuses that answer in between.
- Against a stale answer the cache can grow up to 64 mirrors (816 MiB) past an explicit reserve, or past the 600 MiB default when there's no bank.
- The bank-sized reserve from #2264 already budgets those 64 mirrors, so it isn't affected.

**Workaround:** `DSV4_CUDA_VRAM_RESERVE_MB=2000` (0 failures in 4 runs, reference texts, normal TTFT).

A fix follows in a PR.

Evidence (raw logs): [final21](https://github.com/bobsummerwill/colibri-notes/blob/471769a/manifests/shadowfax-20261025/final21-run.log), [final10](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final10), [c4fa](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/c4fa)
