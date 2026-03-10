#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void durbin_kernel1(int k, int barrier_id, int n ,double *r, double *y, double *z, double *alpha, double *beta)
{
    // 只用一个线程执行所有操作
    int tid = get_thread_id();
    if (tid != 0) return; // 其他线程直接退出

    // 更新 beta
    *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);

    // 局部 sum（顺序计算）
    double sum_total = 0.0;
    for (int i = 0; i < k; i++) {
        sum_total += r[k - i - 1] * y[i];
    }

    // 更新 alpha
    *alpha = -(r[k] + sum_total) / (*beta);

    // 更新 z
    double alpha_val = *alpha;
    for (int i = 0; i < k; i++) {
        z[i] = y[i] + alpha_val * y[k - i - 1];
    }
}

__global__ void durbin_kernel2(int k, int barrier_id,double *y, double *z,double *alpha)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int chunk_size = (k + num_threads - 1) / num_threads;
    int start_i = tid * chunk_size;
    int end_i   = (start_i + chunk_size > k) ? k : start_i + chunk_size;

    // 阶段1：并行拷贝 z → y
    for (int i = start_i; i < end_i; i++) {
        y[i] = z[i];
    }

    // 阶段2：thread0 更新 y[k]
    if (tid == 0) {
        y[k] = *alpha;
    }
}