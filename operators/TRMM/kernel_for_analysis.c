#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void trmm_kernel(int m, int n, double alpha, double *A, double *B)
{
    int tid         = get_thread_id();
    int num_threads = get_group_size();

    // 列并行分配
    int work_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * work_per_thread;
    int end_j   = min(start_j + work_per_thread, n);

    // 保持 i 循环顺序
    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j++) {
            for (int k = i + 1; k < m; k++) {
                B[i * n + j] += A[k * m + i] * B[k * n + j];
            }
            B[i * n + j] = alpha * B[i * n + j];
        }
    }
}