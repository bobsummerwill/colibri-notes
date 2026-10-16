#!/usr/bin/env python3
"""Upload-race client (run_race.sh): one streamed greedy request per cold server.

  RACE_MODE=long   bench_pr1.py's 1,992-token filler + LONG_Q (bench_common): decode right after a
                   long first request (the documented trigger "from the first decode step after a
                   long first request")
  RACE_MODE=short  bench_decode.py prompt 0 (a ~30-token prompt)
  RACE_NTOK=N      max_tokens (both answers are long enough to reach it)
One JSON line (bench_common.stream: full text + sha256, per-chunk times). The per-token logits come
from the probe builds' serve.log (V4_LOGIT_LOG=1), not from here.
usage: RACE_MODE=long|short RACE_NTOK=N bench_race.py OUT.jsonl
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc


def main():
    mode = os.environ.get("RACE_MODE", "short")
    ntok = int(os.environ.get("RACE_NTOK", "120"))
    if mode == "long":
        prompt = bc.long_prompt()
    elif mode == "short":
        prompt = bc.load_short_prompt()
    else:
        raise SystemExit("RACE_MODE must be long or short, not %r" % mode)
    with open(sys.argv[1], "w") as f:
        bc.write(f, bc.stream("race_%s_%d" % (mode, ntok), prompt, ntok))


if __name__ == "__main__":
    main()
