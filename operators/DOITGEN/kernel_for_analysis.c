#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void doitgen_kernel1(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    for (int i = start_idx; i < end_idx; i++) {
        sum[r * (nq * np) + i] = (double)0.0;
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            sum[r * (nq * np) + i] += A[r * (nq * np) + q * np + s] * C4[s * np + p];
        }
    }
}

__global__ void doitgen_kernel2(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    for (int i = start_idx; i < end_idx; i++) {
        A[r * (nq * np) + i] = sum[r * (nq * np) + i];
    }
}