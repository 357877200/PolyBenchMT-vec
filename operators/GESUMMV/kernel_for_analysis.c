#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }
__global__ void gesummv_kernel(int n, double alpha, double beta, double *A, double *B, double *tmp,
                               double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);

    for (int i = start_i; i < end_i; i++) {
        tmp[i] = 0;
        y[i] = 0;

        for (int j = 0; j < n; j++) {
            tmp[i] += A[i * n + j] * x[j];
            y[i] += B[i * n + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}