#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void jacobi1D_kernel1(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx;
    if (thread_id < remainder) {
        end_idx = start_idx + elements_per_thread;
    } else {
        end_idx = start_idx + elements_per_thread - 1;
    }

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);
    // 执行计算
    for (int i = start_idx; i <= end_idx; i++) {
        B[i] = 0.33333f * (A[i - 1] + A[i] + A[i + 1]);
    }
    return;
}

__global__ void jacobi1D_kernel2(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    // 执行计算
    for (int i = start_idx; i <= end_idx; i++) {
        A[i] = B[i];
    }

    return;
}