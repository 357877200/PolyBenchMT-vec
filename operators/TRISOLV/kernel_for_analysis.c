#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

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