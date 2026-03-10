#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void syr2k_kernel(int ni, int nj, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;

        c[idx] *= beta;
        for (int k = 0; k < nj; k++) {
            c[idx] += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
        }
    }
}