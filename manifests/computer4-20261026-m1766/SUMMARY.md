# #1766 vs its dev base, one RTX 5070 12 GB (computer4, 2026-10-25/26)

#1766 ca55947c vs its dev base 88f25a60 (the dev it last merged), generic CUDA sm_120 builds (CUDA 12.8, no DeepGEMM), full
V4 Flash, `--ram 40 --ctx 8192`, the serve environment of every computer4 run (COLI_CUDA_ATTN_BATCH=1 etc.), bench_decode.py
(6 prompts x 300 tokens), interleaved, a fresh server each. The desktop holds 1.9 GB of the 12 GB at idle (voxtype 1.5 GB).

**Run 2 (bench2, `DSV4_CUDA_VRAM_RESERVE_MB=2000` for both), the clean one:**

| | decode tok/s, s1 / s2 | median TTFT | handled OOM lines | GPU peak |
|---|---|---|---|---|
| base 88f25a60 (MR) | 1.592 / 1.595 | 6.5 s | 0 | 10.2 GB |
| #1766 (M1R) | 1.885 / 1.884 | 6.4-6.6 s | 0 | 10.1 GB |

**#1766 / base: 1.183x decode, all 24 answers byte-identical.** #1766's residency: `cache-fallback (resident experts exceed
device 0 budget (free=2577 MiB reserve=2000 MiB))`; 21 layers' indexers resident.

**Run 1 (bench, the defaults, no reserve set):** #1766 logged 543,472 and 312,640 handled `[DSV4 CUDA] ... out of memory`
lines in its two runs (weight allocation, matvec launch, decode output launch, kv comp, scale allocation), and its texts differed
from the base's and between its own runs; the base logged 0 in one run and 166,658 in the other (its texts then changed too,
and answers stopped at 95-106 tokens). Decode 1.655 vs 1.533 tok/s (1.080x), not comparable. The base predates #2401's guard fix.
