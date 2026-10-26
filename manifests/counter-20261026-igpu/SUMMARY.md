# iGPU subgroup comparison for #2404 (counter, 2026-10-26 00:35-00:59)

Intel Meteor Lake iGPU (Mesa 26.2.5 ANV, sizes 8-32, reports 32), the #2404 tip d0765490 built with VK=1, OLMoE-1B-7B-0924-Instruct
Q4_K_M (allenai GGUF, sha256 in host.txt), `coli serve` with COLI_VULKAN=1 COLI_VK_CHAIN=1, bench_decode_anymodel.py (6 prompts x 300
tokens, temperature 0.7 with the server's per-request seed), 3 client runs per server, servers in the order 16, 32, 32, 16.

| server | pin | decode tok/s (runs 1-3) | median TTFT | first request TTFT | device lost |
|---|---|---|---|---|---|
| 1 | 16 (`COLI_VK_SUBGROUP=16`) | 28.53 / 28.44 / 28.50 | 4.0 s | 4.4 s | never (18 chain forwards) |
| 2 | 32 (unset: dev's pin) | 22.85 / 22.75 / 22.56 | 1.0 s | 22.3 s | on the first chain frame: CPU from then on |
| 3 | 32 | 10.58 / 14.88 / 22.74 | 20.1 / 20.1 / 1.1 s | 20.5 s | after 18 chain forwards, several times |
| 4 | 16 | 28.60 / 28.61 / 28.53 | 4.0 s | 4.4 s | never |

- At 32 the kernel logs `Fence expiration time out i915-0000:00:02.0:olmoe[...]` (kernel-i915.txt) and the engine logs
  `[VK] chain: fence wait failed (2): the device is lost` / `the CPU redoes the step ... and runs from here on`: the GPU hangs on
  the chain's work and decode continues on the CPU, so the 32 rows are mostly the CPU's speed (the docs' CPU figure for OLMoE is
  about this). At 16 the chain never lost the device; servers 1 and 4 gave the same texts for the same requests.
- 16's TTFT is worse on these short (~30-token) prompts: the chain's prompt step is slower than the CPU's here.
- The 38-minute compile seen in test_vk_chain does not happen in serve: at 32 the first request's TTFT is ~21 s.
