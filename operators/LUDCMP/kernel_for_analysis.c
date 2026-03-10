#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }
__global__ void ludcmp_kernel1(int n, int k, double *A)
{
    int tid = get_thread_id();
    if (tid != 0) return; // 只有 thread 0 执行

    double w;
    // 对列 j=0..k-1 顺序计算
    for (int j = 0; j < k; j++) {
        w = A[k * n + j];
        for (int p = 0; p < j; p++)
            w -= A[k * n + p] * A[p * n + j];
        A[k * n + j] = w / A[j * n + j];
    }
}


__global__ void ludcmp_kernel2(int n, int k, double *A)
{
    int tid = get_thread_id();
    if (tid != 0) return; // 只有 thread 0 执行

    double w;
    // 对列 j=k..n-1 顺序计算
    for (int j = k; j < n; j++) {
        w = A[k * n + j];
        for (int p = 0; p < k; p++)
            w -= A[k * n + p] * A[p * n + j];
        A[k * n + j] = w;
    }
}


__global__ void ludcmp_kernel3(int n, int i, double *A, double *b, double *y)
{
    int tid = get_thread_id();
    if (tid != 0) return; // 只有 thread 0 执行

    double w = b[i];
    for (int j = 0; j < i; j++) {
        w -= A[i * n + j] * y[j];
    }
    y[i] = w;
}


__global__ void ludcmp_kernel4(int n, int i, double *A, double *x, double *y)
{
    int tid = get_thread_id();
    if (tid != 0) return; // 只有 thread 0 执行

    double w = y[i];
    for (int j = i + 1; j < n; j++) {
        w -= A[i * n + j] * x[j];
    }
    x[i] = w / A[i * n + i];
}
