# 5070 (computer4) full-model A/B: dev `3b953308` vs head-followups `f4586dca` (2026-11-03)

- computer4: RTX 5070 12 GB, Ryzen host, CUDA 12.8 (`/usr/local/cuda`), generic sm_120,
  `NO_TC=1`, 62 GB RAM.
- Trees: dev `3b953308` in `Projects/colibri-dev-3b953308`; branch
  `fix/dsv4-head-followups` at `f4586dca` in `Projects/colibri`.
- Full V4 Flash, `--ram 40 --ctx 8192`, `COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1`,
  `DSV4_CUDA_VRAM_RESERVE_MB=2000` on both; the branch also gets
  `DSV4_CUDA_RESIDENT_RESERVE_MB=2000`.
- `bench_fdec.py` (3 prompts x 300 tokens), one cold sample per arm, dev then branch.

| arm | decode tok/s (mean of 3) | texts |
|---|---:|---|
| dev `3b953308` | 1.6223 | `3d0a5275`, `f0e57dde`, `d27222f1` |
| branch `f4586dca` | **1.9953** | `3d0a5275`, `f0e57dde`, `d27222f1` |

**1.230x** for the branch, all texts identical. The branch logged:

```text
v4_gpu resident-experts=cache-fallback (resident experts exceed device 0 budget (free=4369 MiB reserve=2000 MiB))
v4_gpu head=resident device=0 bytes=1059061760 arithmetic=separate-mul-add
```

So on 12 GB the full model still cannot keep resident experts; the gain is kernel-side, and
`auto` falls back correctly with no OOM. This is consistent with the 24 GB 3090 result and with the
earlier 5070 run (1.183x at `ca55947c`); the head is now 1.230x.

Caveats: one sample per arm (3 x 300 tokens), not the older 6 x 300 two-sample shape; treat as a
directional confirmation, not a distribution. Logs: [run.log](run.log), [dev/bench.jsonl](dev/bench.jsonl),
[new/bench.jsonl](new/bench.jsonl), serve logs in `dev/` and `new/`.
