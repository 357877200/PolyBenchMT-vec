#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

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

#ifdef MINI_DATASET
__global__ void mm3_kernel1_cache(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(E, double, E, 0, 14);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_B, tmp_E;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        CACHEs_WT(E, &E[i * nj + j], 0);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                CACHEs_RD(E, &E[i * nj + j], tmp_E);
                CACHEs_RD(A, &A[i * nk + k], tmp_A);
                CACHEs_RD(B, &B[k * nj + j], tmp_B);
                tmp_E += tmp_A * tmp_B;
                CACHEs_WT(E, &E[i * nj + j], tmp_E);
            }
        }
    }
    CACHEs_FLUSH(E);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm3_kernel2_cache(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(F, double, F, 0, 14);
    CACHEs_INIT(D, double, D, 0, 9);
    CACHEs_INIT(C, double, C, 0, 15);
    double tmp_C, tmp_D, tmp_F;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(F, &F[i * nl + j], 0);
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                CACHEs_RD(F, &F[i * nl + j], tmp_F);
                CACHEs_RD(C, &C[i * nm + k], tmp_C);
                CACHEs_RD(D, &D[k * nl + j], tmp_D);
                tmp_F += tmp_C * tmp_D;
                CACHEs_WT(F, &F[i * nl + j], tmp_F);
            }
        }
    }
    CACHEs_FLUSH(F);
    CACHEs_INVALID(C);
    CACHEs_INVALID(D);
}

__global__ void mm3_kernel3_cache(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(G, double, G, 0, 14);
    CACHEs_INIT(F, double, F, 0, 9);
    CACHEs_INIT(E, double, E, 0, 15);
    double tmp_E, tmp_F, tmp_G;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(G, &G[i * nl + j], 0);
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                CACHEs_RD(G, &G[i * nl + j], tmp_G);
                CACHEs_RD(E, &E[i * nj + k], tmp_E);
                CACHEs_RD(F, &F[k * nl + j], tmp_F);
                tmp_G += tmp_E * tmp_F;
                CACHEs_WT(G, &G[i * nl + j], tmp_G);
            }
        }
    }
    CACHEs_FLUSH(G);
    CACHEs_INVALID(E);
    CACHEs_INVALID(F);
}
#endif

#ifdef SMALL_DATASET
__global__ void mm3_kernel1_cache(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(E, double, E, 0, 14);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_B, tmp_E;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        CACHEs_WT(E, &E[i * nj + j], 0);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                CACHEs_RD(E, &E[i * nj + j], tmp_E);
                CACHEs_RD(A, &A[i * nk + k], tmp_A);
                CACHEs_RD(B, &B[k * nj + j], tmp_B);
                tmp_E += tmp_A * tmp_B;
                CACHEs_WT(E, &E[i * nj + j], tmp_E);
            }
        }
    }
    CACHEs_FLUSH(E);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm3_kernel2_cache(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(F, double, F, 0, 14);
    CACHEs_INIT(D, double, D, 0, 9);
    CACHEs_INIT(C, double, C, 0, 15);
    double tmp_C, tmp_D, tmp_F;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(F, &F[i * nl + j], 0);
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                CACHEs_RD(F, &F[i * nl + j], tmp_F);
                CACHEs_RD(C, &C[i * nm + k], tmp_C);
                CACHEs_RD(D, &D[k * nl + j], tmp_D);
                tmp_F += tmp_C * tmp_D;
                CACHEs_WT(F, &F[i * nl + j], tmp_F);
            }
        }
    }
    CACHEs_FLUSH(F);
    CACHEs_INVALID(C);
    CACHEs_INVALID(D);
}

__global__ void mm3_kernel3_cache(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(G, double, G, 0, 14);
    CACHEs_INIT(F, double, F, 0, 9);
    CACHEs_INIT(E, double, E, 0, 15);
    double tmp_E, tmp_F, tmp_G;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(G, &G[i * nl + j], 0);
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                CACHEs_RD(G, &G[i * nl + j], tmp_G);
                CACHEs_RD(E, &E[i * nj + k], tmp_E);
                CACHEs_RD(F, &F[k * nl + j], tmp_F);
                tmp_G += tmp_E * tmp_F;
                CACHEs_WT(G, &G[i * nl + j], tmp_G);
            }
        }
    }
    CACHEs_FLUSH(G);
    CACHEs_INVALID(E);
    CACHEs_INVALID(F);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void mm3_kernel1_cache(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(E, double, E, 0, 14);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_B, tmp_E;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        CACHEs_WT(E, &E[i * nj + j], 0);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                CACHEs_RD(E, &E[i * nj + j], tmp_E);
                CACHEs_RD(A, &A[i * nk + k], tmp_A);
                CACHEs_RD(B, &B[k * nj + j], tmp_B);
                tmp_E += tmp_A * tmp_B;
                CACHEs_WT(E, &E[i * nj + j], tmp_E);
            }
        }
    }
    CACHEs_FLUSH(E);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm3_kernel2_cache(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(F, double, F, 0, 14);
    CACHEs_INIT(D, double, D, 0, 9);
    CACHEs_INIT(C, double, C, 0, 15);
    double tmp_C, tmp_D, tmp_F;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(F, &F[i * nl + j], 0);
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                CACHEs_RD(F, &F[i * nl + j], tmp_F);
                CACHEs_RD(C, &C[i * nm + k], tmp_C);
                CACHEs_RD(D, &D[k * nl + j], tmp_D);
                tmp_F += tmp_C * tmp_D;
                CACHEs_WT(F, &F[i * nl + j], tmp_F);
            }
        }
    }
    CACHEs_FLUSH(F);
    CACHEs_INVALID(C);
    CACHEs_INVALID(D);
}

__global__ void mm3_kernel3_cache(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(G, double, G, 0, 14);
    CACHEs_INIT(F, double, F, 0, 9);
    CACHEs_INIT(E, double, E, 0, 15);
    double tmp_E, tmp_F, tmp_G;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(G, &G[i * nl + j], 0);
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                CACHEs_RD(G, &G[i * nl + j], tmp_G);
                CACHEs_RD(E, &E[i * nj + k], tmp_E);
                CACHEs_RD(F, &F[k * nl + j], tmp_F);
                tmp_G += tmp_E * tmp_F;
                CACHEs_WT(G, &G[i * nl + j], tmp_G);
            }
        }
    }
    CACHEs_FLUSH(G);
    CACHEs_INVALID(E);
    CACHEs_INVALID(F);
}
#endif

#ifdef LARGE_DATASET
__global__ void mm3_kernel1_cache(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(E, double, E, 0, 13);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_B, tmp_E;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        CACHEs_WT(E, &E[i * nj + j], 0);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                CACHEs_RD(E, &E[i * nj + j], tmp_E);
                CACHEs_RD(A, &A[i * nk + k], tmp_A);
                CACHEs_RD(B, &B[k * nj + j], tmp_B);
                tmp_E += tmp_A * tmp_B;
                CACHEs_WT(E, &E[i * nj + j], tmp_E);
            }
        }
    }
    CACHEs_FLUSH(E);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm3_kernel2_cache(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(F, double, F, 0, 13);
    CACHEs_INIT(D, double, D, 0, 9);
    CACHEs_INIT(C, double, C, 0, 15);
    double tmp_C, tmp_D, tmp_F;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(F, &F[i * nl + j], 0);
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                CACHEs_RD(F, &F[i * nl + j], tmp_F);
                CACHEs_RD(C, &C[i * nm + k], tmp_C);
                CACHEs_RD(D, &D[k * nl + j], tmp_D);
                tmp_F += tmp_C * tmp_D;
                CACHEs_WT(F, &F[i * nl + j], tmp_F);
            }
        }
    }
    CACHEs_FLUSH(F);
    CACHEs_INVALID(C);
    CACHEs_INVALID(D);
}

__global__ void mm3_kernel3_cache(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(G, double, G, 0, 13);
    CACHEs_INIT(F, double, F, 0, 9);
    CACHEs_INIT(E, double, E, 0, 15);
    double tmp_E, tmp_F, tmp_G;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(G, &G[i * nl + j], 0);
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                CACHEs_RD(G, &G[i * nl + j], tmp_G);
                CACHEs_RD(E, &E[i * nj + k], tmp_E);
                CACHEs_RD(F, &F[k * nl + j], tmp_F);
                tmp_G += tmp_E * tmp_F;
                CACHEs_WT(G, &G[i * nl + j], tmp_G);
            }
        }
    }
    CACHEs_FLUSH(G);
    CACHEs_INVALID(E);
    CACHEs_INVALID(F);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void mm3_kernel1_cache(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(E, double, E, 0, 14);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_B, tmp_E;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        CACHEs_WT(E, &E[i * nj + j], 0);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                CACHEs_RD(E, &E[i * nj + j], tmp_E);
                CACHEs_RD(A, &A[i * nk + k], tmp_A);
                CACHEs_RD(B, &B[k * nj + j], tmp_B);
                tmp_E += tmp_A * tmp_B;
                CACHEs_WT(E, &E[i * nj + j], tmp_E);
            }
        }
    }
    CACHEs_FLUSH(E);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm3_kernel2_cache(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(F, double, F, 0, 14);
    CACHEs_INIT(D, double, D, 0, 9);
    CACHEs_INIT(C, double, C, 0, 15);
    double tmp_C, tmp_D, tmp_F;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(F, &F[i * nl + j], 0);
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                CACHEs_RD(F, &F[i * nl + j], tmp_F);
                CACHEs_RD(C, &C[i * nm + k], tmp_C);
                CACHEs_RD(D, &D[k * nl + j], tmp_D);
                tmp_F += tmp_C * tmp_D;
                CACHEs_WT(F, &F[i * nl + j], tmp_F);
            }
        }
    }
    CACHEs_FLUSH(F);
    CACHEs_INVALID(C);
    CACHEs_INVALID(D);
}

__global__ void mm3_kernel3_cache(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    CACHEs_INIT(G, double, G, 0, 14);
    CACHEs_INIT(F, double, F, 0, 9);
    CACHEs_INIT(E, double, E, 0, 15);
    double tmp_E, tmp_F, tmp_G;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        CACHEs_WT(G, &G[i * nl + j], 0);
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                CACHEs_RD(G, &G[i * nl + j], tmp_G);
                CACHEs_RD(E, &E[i * nj + k], tmp_E);
                CACHEs_RD(F, &F[k * nl + j], tmp_F);
                tmp_G += tmp_E * tmp_F;
                CACHEs_WT(G, &G[i * nl + j], tmp_G);
            }
        }
    }
    CACHEs_FLUSH(G);
    CACHEs_INVALID(E);
    CACHEs_INVALID(F);
}
#endif

/* 一条向量 16 个 double */
#define SIMD_LEN 16
#define VEC_BYTES 128

/* 向量化版本的 3MM 内核1：计算 E = A * B */
__global__ void mm3_kernel1_vec(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    if (total_elements <= 0) return;

    const int base_tasks = total_elements / group_size;
    const int remainder = total_elements % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    /* 申请向量缓冲区，用于加载矩阵数据 */
    lvector double *buf_A = (lvector double *)vector_malloc(sizeof(lvector double));
    lvector double *buf_B = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 主循环：线性遍历所有任务 */
    for (int t = start; t < end; )
    {
        int i = t / nj;
        int j = t % nj;
        int remain_col = nj - j;

        if (remain_col >= SIMD_LEN && (end - t) >= SIMD_LEN)
        {
            /* 向量化处理 16 个元素 */
            lvector double res_vec = (lvector double)vec_svbcast(0.0); // 初始化结果向量为 0

            for (int k = 0; k < nk; ++k)
            {
                // 加载 A 的当前元素（广播到向量）
                double a_val = A[i * nk + k];
                lvector double vec_A = (lvector double)vec_svbcast(a_val);

                // 加载 B 的当前行 k，列 j 开始的 16 个元素
                size_t off_B = (size_t)k * nj + j;
                vector_load(B + off_B, buf_B, VEC_BYTES);

                // res_vec += A[i][k] * B[k][j:j+16]
                res_vec = vec_mula(*buf_B, vec_A, res_vec);
            }

            // 写回结果到 E
            vector_store(&res_vec, E + i * nj + j, VEC_BYTES);

            t += SIMD_LEN; // 一次性处理 16 个任务
        }
        else
        {
            /* 标量尾部处理 */
            double sum = 0.0;
            for (int k = 0; k < nk; ++k)
            {
                sum += A[i * nk + k] * B[k * nj + j];
            }
            E[i * nj + j] = sum;

            ++t; // 只处理 1 个任务
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
}

/* 向量化版本的 3MM 内核2：计算 F = C * D */
__global__ void mm3_kernel2_vec(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    if (total_elements <= 0) return;

    const int base_tasks = total_elements / group_size;
    const int remainder = total_elements % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    /* 申请向量缓冲区，用于加载矩阵数据 */
    lvector double *buf_C = (lvector double *)vector_malloc(sizeof(lvector double));
    lvector double *buf_D = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 主循环：线性遍历所有任务 */
    for (int t = start; t < end; )
    {
        int i = t / nl;
        int j = t % nl;
        int remain_col = nl - j;

        if (remain_col >= SIMD_LEN && (end - t) >= SIMD_LEN)
        {
            /* 向量化处理 16 个元素 */
            lvector double res_vec = (lvector double)vec_svbcast(0.0); // 初始化结果向量为 0

            for (int k = 0; k < nm; ++k)
            {
                // 加载 C 的当前元素（广播到向量）
                double c_val = C[i * nm + k];
                lvector double vec_C = (lvector double)vec_svbcast(c_val);

                // 加载 D 的当前行 k，列 j 开始的 16 个元素
                size_t off_D = (size_t)k * nl + j;
                vector_load(D + off_D, buf_D, VEC_BYTES);

                // res_vec += C[i][k] * D[k][j:j+16]
                res_vec = vec_mula(*buf_D, vec_C, res_vec);
            }

            // 写回结果到 F
            vector_store(&res_vec, F + i * nl + j, VEC_BYTES);

            t += SIMD_LEN; // 一次性处理 16 个任务
        }
        else
        {
            /* 标量尾部处理 */
            double sum = 0.0;
            for (int k = 0; k < nm; ++k)
            {
                sum += C[i * nm + k] * D[k * nl + j];
            }
            F[i * nl + j] = sum;

            ++t; // 只处理 1 个任务
        }
    }

    vector_free(buf_C);
    vector_free(buf_D);
}

/* 向量化版本的 3MM 内核3：计算 G = E * F */
__global__ void mm3_kernel3_vec(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    if (total_elements <= 0) return;

    const int base_tasks = total_elements / group_size;
    const int remainder = total_elements % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    /* 申请向量缓冲区，用于加载矩阵数据 */
    lvector double *buf_E = (lvector double *)vector_malloc(sizeof(lvector double));
    lvector double *buf_F = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 主循环：线性遍历所有任务 */
    for (int t = start; t < end; )
    {
        int i = t / nl;
        int j = t % nl;
        int remain_col = nl - j;

        if (remain_col >= SIMD_LEN && (end - t) >= SIMD_LEN)
        {
            /* 向量化处理 16 个元素 */
            lvector double res_vec = (lvector double)vec_svbcast(0.0); // 初始化结果向量为 0

            for (int k = 0; k < nj; ++k)
            {
                // 加载 E 的当前元素（广播到向量）
                double e_val = E[i * nj + k];
                lvector double vec_E = (lvector double)vec_svbcast(e_val);

                // 加载 F 的当前行 k，列 j 开始的 16 个元素
                size_t off_F = (size_t)k * nl + j;
                vector_load(F + off_F, buf_F, VEC_BYTES);

                // res_vec += E[i][k] * F[k][j:j+16]
                res_vec = vec_mula(*buf_F, vec_E, res_vec);
            }

            // 写回结果到 G
            vector_store(&res_vec, G + i * nl + j, VEC_BYTES);

            t += SIMD_LEN; // 一次性处理 16 个任务
        }
        else
        {
            /* 标量尾部处理 */
            double sum = 0.0;
            for (int k = 0; k < nj; ++k)
            {
                sum += E[i * nj + k] * F[k * nl + j];
            }
            G[i * nl + j] = sum;

            ++t; // 只处理 1 个任务
        }
    }

    vector_free(buf_E);
    vector_free(buf_F);
}
#include "../3MM/kernel_vec.h"//大模型生成的存储文件
#include "../3MM/kernel_cache_llm.h"//SM缓存优化文件