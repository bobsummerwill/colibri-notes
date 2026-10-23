#!/bin/bash
# build_f18.sh (final7) -- the F18 probe tree's engine with -DCOLI_F18_PROBE (debug only, never a release):
# the generic sm_86 build of every final run plus the probe define. V4_DEEPGEMM_FLAGS is given whole on the
# command line, so NO_TC=1's "+= -DCOLI_DSV4_NO_TC" would be ignored: the define is in it instead.
exec make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=0 \
    "V4_DEEPGEMM_FLAGS=-DCOLI_DSV4_NO_TC -DCOLI_F18_PROBE" -j8
