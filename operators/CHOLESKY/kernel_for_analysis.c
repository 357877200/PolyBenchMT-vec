#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void cholesky_kernel(int n, int barrier_id, double *A)
{
    int tid = get_thread_id();

    // Only one thread performs all computations
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double sum = A[i*n + j];
                for (int k = 0; k < j; k++) {
                    sum -= A[i*n + k] * A[j*n + k];
                }
                A[i*n + j] = sum / A[j*n + j];
            }

            double sum = A[i*n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[i*n + k] * A[i*n + k];
            }
            A[i*n + i] = sqrt(sum);
        }
    }
}