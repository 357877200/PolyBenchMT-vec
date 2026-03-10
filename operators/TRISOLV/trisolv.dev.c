#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

//
// 设备端核函数：trisolv
// 解下三角矩阵 Lx = b ，前向代入
//
__global__ void trisolv_kernel(int n, double *L, double *x, double *b)
{
    int tid = get_thread_id();
    if (tid != 0) return;

    for (int i = 0; i < n; i++) {
        x[i] = b[i];
        for (int j = 0; j < i; j++) {
            x[i] -= L[i * n + j] * x[j];
        }
        x[i] = x[i] / L[i * n + i];
    }
}

#include "../TRISOLV/kernel_vec.h"       // 向量化优化文件
#include "../TRISOLV/kernel_cache_llm.h" // SM缓存优化文件