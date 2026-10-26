#!/bin/bash
# M (#1766 merged onto dd03a9de) from the fork branch samebase/m1766: fetch, then CUDA sm_120 and CPU-only builds.
# usage: build25_m.sh [COMMIT]   (default: the fetched branch tip; pass the expected sha to pin it)
cd ~/Projects/colibri && git fetch -q origin samebase/m1766:refs/remotes/origin/samebase/m1766 || { echo "fetch failed"; exit 4; }
ref=${1:-$(git rev-parse origin/samebase/m1766)}
git cat-file -t "$ref" >/dev/null || { echo "no commit $ref"; exit 4; }
echo "M = $(git rev-parse $ref)"
~/colibri-bench/dg/build25.sh m:$ref:cuda mcpu:$ref:cpu
tail -3 ~/colibri-bench/dg/logs/build25.status
