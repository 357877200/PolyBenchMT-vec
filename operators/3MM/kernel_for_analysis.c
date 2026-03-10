#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }


__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        E[i * nj + j] = 0;
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                E[i * nj + j] += A[i * nk + k] * B[k * nj + j];
            }
        }
    }
}

__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        F[i * nl + j] = 0;
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                F[i * nl + j] += C[i * nm + k] * D[k * nl + j];
            }
        }
    }
}

__global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        G[i * nl + j] = 0;
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                G[i * nl + j] += E[i * nj + k] * F[k * nl + j];
            }
        }
    }
}
