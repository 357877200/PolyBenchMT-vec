#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void convolution2D_kernel(int ni, int nj, double *A, double *B, uint64_t *before_hot_data, uint64_t *after_hot_data)
{
    int group_size = get_group_size();
    int thread_id  = get_thread_id();

    // 卷积核系数
    double c11 = +0.2, c21 = +0.5, c31 = -0.8;
    double c12 = -0.3, c22 = +0.6, c32 = -0.9;
    double c13 = +0.4, c23 = +0.7, c33 = +0.10;

    const int total_tasks = (ni - 2) * (nj - 2);
    if (total_tasks <= 0) return;

    const int base_tasks = total_tasks / group_size;
    const int remainder  = total_tasks % group_size;

    int start = (thread_id < remainder)
        ? thread_id * (base_tasks + 1)
        : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    // 单循环完成全部卷积计算
    for (int t = start; t < end; ++t)
    {
        const int i = 1 + t / (nj - 2); // i ∈ [1, ni-2]
        const int j = 1 + t % (nj - 2); // j ∈ [1, nj-2]

        double val =
            c11 * A[(i - 1) * nj + (j - 1)] +
            c21 * A[(i - 1) * nj + j]       +
            c31 * A[(i - 1) * nj + (j + 1)] +
            c12 * A[i * nj + (j - 1)]      +
            c22 * A[i * nj + j]            +
            c32 * A[i * nj + (j + 1)]      +
            c13 * A[(i + 1) * nj + (j - 1)] +
            c23 * A[(i + 1) * nj + j]       +
            c33 * A[(i + 1) * nj + (j + 1)];

        B[i * nj + j] = val;
    }
}