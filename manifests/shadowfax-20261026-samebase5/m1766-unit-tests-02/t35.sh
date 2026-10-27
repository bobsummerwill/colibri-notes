#!/bin/bash
# t35.sh (2026-10-26): after cbench-f34, build fork samebase/m1766-fixes 28b35786 and samebase/cplus 66982895 (sm_86 NO_TC)
# and run #1766's CUDA unit tests with DSV4_TEST_DEVICES=0,1, plus dsv4-cuda-test on C+. Small GPU footprint; llama left up.
B=~/colibri-bench/t35; mkdir -p $B; L=$B/t35.log; W=~/colibri-bench/cw22
log() { echo "[$(date "+%F %T")] $*" >> $L; }
log "t35 start: waiting for cbench-f34"
while systemctl --user is-active -q cbench-f34; do sleep 60; done
while systemctl --user is-active -q colibri-final34; do sleep 60; done
log "cbench-f34 ended; GPUs: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
MK="make -f Makefile.deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1"
for spec in m1766fix:f35/m1766-fixes cplus:f35/cplus; do
  n=${spec%%:*}; ref=${spec#*:}; t=$W/f35-$n
  [ -d $t ] || git -C $W/repo worktree add -q --detach $t $ref
  cd $t/c || { log "$n: no tree"; continue; }
  log "$n at $(git rev-parse --short HEAD)"
  for u in head_cuda wo_decode_cuda compressor_cuda compressor_state_cuda indexer_prepare_cuda multigpu_cuda; do
    if $MK tests/test_dsv4_$u > $B/$n-build-$u.log 2>&1; then
      DSV4_TEST_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 timeout 900 ./tests/test_dsv4_$u > $B/$n-run-$u.log 2>&1; rc=$?
      log "$n $u: build ok, run rc=$rc | $(tail -1 $B/$n-run-$u.log)"
    else log "$n $u: BUILD FAILED | $(grep -m1 -E 'error' $B/$n-build-$u.log)"; fi
  done
  # no make target: built by hand
  if /usr/bin/nvcc -O2 -std=c++17 -arch=sm_86 tests/test_dsv4_quant_cuda.cu -o tests/test_dsv4_quant_cuda > $B/$n-build-quant.log 2>&1; then
    DSV4_TEST_DEVICES=0,1 timeout 600 ./tests/test_dsv4_quant_cuda > $B/$n-run-quant.log 2>&1; log "$n quant_cuda: run rc=$? | $(tail -1 $B/$n-run-quant.log)"
  else log "$n quant_cuda: BUILD FAILED | $(grep -m1 error $B/$n-build-quant.log)"; fi
  for u in resident_route resident_indexer; do
    if g++ -O2 -std=c++20 tests/test_dsv4_$u.cpp backend_cuda_dsv4.o -o tests/test_dsv4_$u -L/usr/lib64 -lcudart -lcublasLt -lcublas -lcuda -lstdc++ > $B/$n-build-$u.log 2>&1; then
      DSV4_TEST_DEVICES=0,1 timeout 600 ./tests/test_dsv4_$u > $B/$n-run-$u.log 2>&1; log "$n $u: run rc=$? | $(tail -1 $B/$n-run-$u.log)"
    else log "$n $u: BUILD FAILED (no make target) | $(grep -m1 -E 'error|undefined' $B/$n-build-$u.log)"; fi
  done
  DSV4_TEST_DEVICES=0,7 ./tests/test_dsv4_compressor_cuda > $B/$n-run-baddev.log 2>&1; log "$n bad pair 0,7: rc=$? (expect nonzero)"
  DSV4_TEST_DEVICES=0 ./tests/test_dsv4_compressor_cuda > $B/$n-run-onedev.log 2>&1; log "$n one device: rc=$? (expect 2) | $(tail -1 $B/$n-run-onedev.log)"
  if [ $n = cplus ]; then
    ( CUDA_VISIBLE_DEVICES=0,1 timeout 1500 make dsv4-cuda-test CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC" ) > $B/cplus-dsv4-cuda-test.log 2>&1
    log "cplus dsv4-cuda-test: rc=$? | $(grep -E 'peer init|peer MoE:' $B/cplus-dsv4-cuda-test.log | tr '\n' ' ')"
  fi
done
log "t35 done"
