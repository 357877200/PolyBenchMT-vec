#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

__global__ void convolution3D_kernel(int ni, int nj, int nk, int i, double *A, double *B) {
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    int total_elements = (nj - 2) * (nk - 2); // 修正为有效区域
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 2) + 1;
        int k = idx % (nk - 2) + 1;
        int idx_B = i * (nk * nj) + j * nk + k;
        B[idx_B] = 
            c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + 
            c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] +
            c21 * A[(i - 1) * (nk * nj) + j * nk + (k - 1)] + 
            c23 * A[(i + 1) * (nk * nj) + j * nk + (k - 1)] +
            c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k - 1)] + 
            c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k - 1)] +
            c12 * A[i * (nk * nj) + (j - 1) * nk + k] + 
            c22 * A[i * (nk * nj) + j * nk + k] +
            c32 * A[i * (nk * nj) + (j + 1) * nk + k] + 
            c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)] +
            c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)] + 
            c21 * A[(i - 1) * (nk * nj) + j * nk + (k + 1)] +
            c23 * A[(i + 1) * (nk * nj) + j * nk + (k + 1)] + 
            c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)] +
            c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)];
    }
}
