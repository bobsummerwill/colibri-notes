#!/bin/bash
# same-base run (2026-10-26): CPU-only DeepSeek V4 tiny oracle (make -C c deepseek-v4-tiny-check, as CI's v4-tiny job)
# on trees given as NAME:REF; worktrees under this dir; pinned torch/transformers venv ~/colibri-bench/v4tiny-venv.
S=$(cd "$(dirname "$0")" && pwd)
for spec in "$@"; do
  IFS=: read n ref <<< "$spec"; t=$S/wt-$n
  [ -d $t ] || git -C ~/Projects/colibri worktree add -q --detach $t $ref || { echo "$n worktree failed" >> $S/tiny.status; continue; }
  ( cd $t/c && echo "tree=$n sha=$(git rev-parse HEAD) start=$(date -Is)" &&
    PATH=$HOME/colibri-bench/v4tiny-venv/bin:$PATH nice -n 10 make deepseek-v4-tiny-check PYTHON=$HOME/colibri-bench/v4tiny-venv/bin/python -j8;
    echo "EXIT=$? end=$(date -Is)" ) > $S/tiny-$n.log 2>&1
  echo "$n $(git -C $t rev-parse --short HEAD) $(tail -1 $S/tiny-$n.log) PASS=$(grep -c "PASS" $S/tiny-$n.log) FAIL=$(grep -ciE "\bFAIL" $S/tiny-$n.log)" >> $S/tiny.status
done; echo "done $*" >> $S/tiny.status
