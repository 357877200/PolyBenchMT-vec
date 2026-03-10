#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void bicg_kernel1(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        s[j] = 0.0f;
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            s[j] += r[i] * A[i * ny + j];
        }
    }
}

__global__ void bicg_kernel2(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    for (int i = start_i; i < end_i; i++) {
        q[i] = 0.0f;

        for (int j = 0; j < ny; j++) {
            q[i] += A[i * ny + j] * p[j];
        }
    }
}