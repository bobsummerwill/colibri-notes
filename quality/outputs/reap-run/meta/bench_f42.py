#!/usr/bin/env python3
"""F42 client (run_f42.sh): decode right after a long prefill, then a short-prompt control.

Two streamed greedy requests to one cold server, one JSON line each (bench_common.stream: full
text, sha256, per-chunk arrival times as offsets from the request start, t0_epoch to align
with nvidia-smi and the serve log):
  long_200   bench_pr1.py's 1,992-token filler, byte for byte, + LONG_Q (bench_common), max_tokens
             200: the first request, so its prefill takes the GPU bank (COLI_CUDA_MOE_BATCH=1) or
             the CPU union (=0) and its answer is the decode right after that prefill
  short_200  bench_decode.py prompt 0, max_tokens 200: decode after a ~30-token prompt (below
             COLI_CUDA_MOE_BATCH_MIN, never the bank) in the same, now warm, server -- the control
usage: bench_f42.py OUT.jsonl
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc

NTOK = int(os.environ.get("F42_NTOK", "200"))


def main():
    reqs = [("long_200", bc.long_prompt(), NTOK), ("short_200", bc.load_short_prompt(), NTOK)]
    with open(sys.argv[1], "w") as f:
        for name, prompt, mt in reqs:
            bc.write(f, bc.stream(name, prompt, mt))


if __name__ == "__main__":
    main()
