#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }
__global__ void seidel2d_kernel(int tsteps, int n, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    // 这里我们把二维 (i,j) 映射为一维元素索引
    int total_elements = (n - 2) * (n - 2);  // 有效计算范围 i=1..n-2, j=1..n-2
    if (total_elements <= 0) return;

    // 均匀分配给线程
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1)
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx   = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    if (start_idx >= end_idx) return;

    // 时间步循环，每步更新会立刻生效（原地更新）
    for (int t = 0; t < tsteps; t++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            // 映射回二维坐标
            int i = idx / (n - 2) + 1;
            int j = idx % (n - 2) + 1;

            A[i * n + j] =
                (A[(i - 1) * n + (j - 1)] + A[(i - 1) * n + j]     + A[(i - 1) * n + (j + 1)]
               + A[i * n + (j - 1)]       + A[i * n + j]           + A[i * n + (j + 1)]
               + A[(i + 1) * n + (j - 1)] + A[(i + 1) * n + j]     + A[(i + 1) * n + (j + 1)]) / 9.0;
        }
    }
}