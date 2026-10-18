# PR #2 (rb/pr2/two-gpu-decode) in the final4 run: single-sample sanity check, no manifest

Arms G1 and X3 ran **one sample each** (bench.tgz `G1/s1`, `X3/s1`; sample seq 11 and 12, after the PR #1 arms).
`c/experiment_manifest.py` requires at least 3 samples per arm, so no experiment manifest is written for them;
these numbers are a sanity check that the PR #2 tip still builds, runs and reproduces the earlier texts, not a
measured speedup claim.

- Commit: bffe2aed04640c5871fbc9e6e4a1f570e8a515fa (tree 7ac4c4d2087d8f80bd1a7ee369b07ea79a8af1f3, meta/tips.txt),
  binary sha256 eb900feaa167b8456b812dfff18ad00f63154d43eca7a739bf14558ae8219100 (meta/trees.txt), clean tree.
  After the run the local branch was amended to 3a46d918 (docs/deepseek-v4.md only).
- Client: bench_decode.py, six prompts, 300-token greedy answers each, one freshly started server per arm,
  same serve flags as PR #1 (--ram 40 --ctx 8192), .coli_usage empty and unchanged.
- G1: one GPU (COLI_GPU=0 DSV4_CUDA_DEVICE=0, V4_SERVE_STATS_LOG=1). Per-prompt decode tok/s
  2.799, 2.892, 3.082, 2.801, 3.484, 3.008; **mean 3.011 tok/s**. GPU1 peak 85 MiB (unused).
- X3: G1 + V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 (serve.log: `v4_gpu peer-mirrors device=1
  reserve=1024MB mode=compute`). Per-prompt decode tok/s 3.92, 4.385, 4.353, 4.233, 4.767, 4.387;
  **mean 4.341 tok/s** (4.3408), X3/G1 = 1.4416.
- Correctness: all six answer texts of both arms (full-text sha256 recomputed from the stored text) are identical
  to the fixed 2026-10-15 G1 references (meta/final_tools.py REF_G1).
- For context only (different base and PR commits): the 2026-10-15 run (3 samples per arm) had G1 2.866,
  X3 4.0433, X3/G1 1.4108 (summary.txt).
