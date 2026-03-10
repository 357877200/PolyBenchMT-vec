#!/bin/bash
# usage: ./analyze-vec.sh kernel_for_analysis.c
set -e
source make.conf   # 把 DEV_INCLUDE_FLAGS 拿过来

DEV_CC=${DEV_CC_BIN_PATH}/MT-3000-gcc

$DEV_CC -c -O3 -ftree-vectorize \
        -fopt-info-vec-missed \
        -fopt-info-vec-optimized \
        -fopt-info-loop-note \
        ${DEV_INCLUDE_FLAGS} \
        kernel_for_analysis.c \
        -o /dev/null 2>&1 | tee vector_report.log